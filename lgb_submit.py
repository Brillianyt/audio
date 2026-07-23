"""LightGBM eval submission using best config."""
import sys, os, csv, io, zipfile, json, pickle
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS

# Load cached features
print('Loading cached features...')
c = np.load('stacker_cache.npz')
X_tr = np.column_stack([c['at_tr'], c['fu_tr'], c['aa_tr']])
X_de = np.column_stack([c['at_de'], c['fu_de'], c['aa_de']])
lb_tr, lb_de = c['lb_tr'], c['lb_de']

# Build features (same as stack_ensemble.py)
def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1-eps)
    return np.log(p / (1-p))

def make_features(at_p, fu_p, aa_p):
    max_p = np.maximum(np.maximum(at_p, fu_p), aa_p)
    min_p = np.minimum(np.minimum(at_p, fu_p), aa_p)
    median_p = np.median(np.stack([at_p, fu_p, aa_p]), axis=0)
    return np.stack([
        at_p, fu_p, aa_p,
        logit(at_p), logit(fu_p), logit(aa_p),
        at_p * fu_p, at_p * aa_p, fu_p * aa_p,
        np.abs(at_p - fu_p), np.abs(at_p - aa_p), np.abs(fu_p - aa_p),
        max_p, min_p, median_p, max_p - median_p,
    ], axis=1).astype(np.float32)

X_tr_feat = make_features(c['at_tr'], c['fu_tr'], c['aa_tr'])
X_de_feat = make_features(c['at_de'], c['fu_de'], c['aa_de'])

# Train LightGBM (best config)
X_s, X_v, y_s, y_v = train_test_split(X_tr_feat, lb_tr, test_size=0.2, random_state=42, stratify=lb_tr)
lgb_params = {
    'objective': 'binary', 'metric': 'auc',
    'max_depth': 3, 'num_leaves': 8,
    'learning_rate': 0.02, 'min_child_samples': 100,
    'l2_leaf_reg': 5.0,
    'subsample': 0.7, 'colsample_bytree': 0.7,
    'verbosity': -1, 'seed': 42,
}
model = lgb.train(lgb_params, lgb.Dataset(X_s, y_s), num_boost_round=1000,
    valid_sets=[lgb.Dataset(X_v, y_v)], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

dev_auc = roc_auc_score(lb_de, model.predict(X_de_feat))
print(f'Dev AUC: {dev_auc:.4f}')

# Save model
with open('lgb_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# ── Eval set inference ──
print('\nRunning eval set inference...')
import torch, torch.nn.functional as F
import soundfile as sf
import importlib.util
spec = importlib.util.spec_from_file_location('td', 'train_dual.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
AudioTextModel, DualConfig = td.AudioTextModel, td.Config
from train_fusion import AudioTextFusionModel
from baseline.train_whisper_v3 import MultiTaskWhisperKWSV3

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

at = AudioTextModel('', 256, unfreeze=0).to(device)
at.load_state_dict(torch.load('output/backup_final/at_v8_ep29_seen08462_unseen07889.pt', map_location=device)['model'], strict=False)
at.eval()

fu = AudioTextFusionModel(256, unfreeze=0).to(device)
fu.load_state_dict(torch.load('output/fusion_v2/best.pt', map_location=device)['model'], strict=False)
fu.eval()

aa = MultiTaskWhisperKWSV3('base', 256, unfreeze_layers=0).to(device)
aa.load_state_dict(torch.load('output/backup_final/aa_v3_best_seen07934.pt', map_location=device)['model'], strict=False)
aa.eval()

def read_and_pad(zf, pid, role):
    try: data = zf.read(f"wav/{pid}_{role}.wav")
    except: data = zf.read(f"wav/{pid}.wav")
    w, sr = sf.read(io.BytesIO(data), dtype='float32')
    if w.ndim > 1: w = w.mean(axis=1)
    if sr != 16000:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    w = torch.from_numpy(w).float()
    return F.pad(w, (8000, 8000))

def get_probs(zip_path, pairs):
    zf = zipfile.ZipFile(zip_path, 'r')
    at_ps, fu_ps, aa_ps = [], [], []
    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            eid = p.get("enroll_id", p['id'])
            qid = p.get("query_id", p['id'])
            e_w.append(read_and_pad(zf, eid, 'enroll'))
            q_w.append(read_and_pad(zf, qid, 'query'))
            txts.append(p['txt'])
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)
        with torch.no_grad():
            _, score, _, _ = at(eb, txts, qb)
            at_ps.extend(torch.sigmoid(score).detach().cpu().numpy().tolist())
            logit, _, _ = fu(eb, txts, qb)
            fu_ps.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
            logit_aa, _, _ = aa(eb, qb)
            aa_ps.extend(torch.sigmoid(logit_aa).detach().cpu().numpy().tolist())
        if (i + 128) % 10000 == 0 or (i + 128) >= len(pairs):
            print(f'  {min(i+128, len(pairs))}/{len(pairs)}')
    zf.close()
    return np.array(at_ps), np.array(fu_ps), np.array(aa_ps)

all_results = []
for prefix, csv_path, zip_path in [
    ('seen_pair', cfg.eval_seen_csv, cfg.eval_seen_zip),
    ('unseen_pair', cfg.eval_unseen_csv, cfg.eval_unseen_zip),
]:
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({"id": r["id"], "txt": r.get("enroll_txt", "").lower()})
    print(f'\n[{prefix}] {len(pairs)} pairs')
    
    at_p, fu_p, aa_p = get_probs(zip_path, pairs)
    feat = make_features(at_p, fu_p, aa_p)
    pred = model.predict(feat)
    
    for j, p in enumerate(pairs):
        all_results.append((f"{prefix}_{p['id'].replace('pair_', '')}", float(pred[j])))

# Save submission
out_path = 'submission_lgb.csv'
all_r = sorted(all_results, key=lambda x: x[0])
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id', 'posterior'])
    w.writerows(all_r)
ps = np.array([p for _, p in all_r])
print(f'\nSaved: {out_path}')
print(f'Total: {len(all_r)} rows')
print(f'Mean={ps.mean():.4f} Std={ps.std():.4f} >0.5={(ps>0.5).sum()}/{len(ps)}')

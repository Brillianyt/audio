"""Train stacking ensemble on TRAINING data, eval on dev."""
import sys, os, csv, io, zipfile, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS

import importlib.util
spec = importlib.util.spec_from_file_location('td', 'train_dual.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
AudioTextModel, DualConfig = td.AudioTextModel, td.Config
from train_fusion import AudioTextFusionModel
from baseline.train_whisper_v3 import MultiTaskWhisperKWSV3

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

# Load 3 models
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

def get_probs(zip_path, pairs, add_noise=False):
    zf = zipfile.ZipFile(zip_path, 'r')
    at_ps, fu_ps, aa_ps, labels = [], [], [], []
    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            eid = p.get("enroll_id", p['id'])
            qid = p.get("query_id", p['id'])
            e = read_and_pad(zf, eid, 'enroll')
            q = read_and_pad(zf, qid, 'query')
            # 训练集加多样噪声模拟测试分布
            if add_noise:
                for w in [e, q]:
                    kind = np.random.choice(
                        ['gauss_light','gauss_med','gauss_heavy','burst','lowpass','clip','tail','babble'],
                        p=[0.15,0.15,0.10,0.12,0.10,0.10,0.13,0.15])
                    T = w.shape[-1]
                    w_out = w.clone()
                    n_std = w.std().clamp(min=1e-6)
                    if kind == 'gauss_light':
                        snr = float(np.random.uniform(0, 5))
                        w_out = w + (10**(-snr/20)) * torch.randn_like(w) * n_std
                    elif kind == 'gauss_med':
                        snr = float(np.random.uniform(-5, 0))
                        w_out = w + (10**(-snr/20)) * torch.randn_like(w) * n_std
                    elif kind == 'gauss_heavy':
                        snr = float(np.random.uniform(-15, -5))
                        w_out = w + (10**(-snr/20)) * torch.randn_like(w) * n_std
                    elif kind == 'burst':
                        snr = float(np.random.uniform(-10, 0))
                        noise = (10**(-snr/20)) * torch.randn_like(w) * n_std
                        t_len = int(T * np.random.uniform(0.3, 0.8))
                        t_start = int(np.random.randint(0, max(1, T - t_len)))
                        w_out[t_start:t_start+t_len] += noise[t_start:t_start+t_len]
                    elif kind == 'lowpass':
                        cutoff = int(np.random.choice([1000, 2000, 3000, 4000]))
                        try:
                            ks = 16000 // cutoff
                            if 1 < ks < T:
                                kernel = torch.ones(1, 1, ks, device=w.device) / ks
                                filtered = F.conv1d(w_out.view(1,1,-1), kernel, padding=ks//2).view(-1)
                                w_out = filtered[:T] if filtered.shape[-1] >= T else F.pad(filtered, (0, T-filtered.shape[-1]))
                        except: pass
                    elif kind == 'clip':
                        th = float(np.random.uniform(0.1, 0.5))
                        w_out = w_out.clamp(-th, th)
                    elif kind == 'tail':
                        snr = float(np.random.uniform(-10, 5))
                        noise = (10**(-snr/20)) * torch.randn_like(w) * n_std
                        tr = float(np.random.uniform(0.2, 0.5))
                        w_out[int(T*(1-tr)):] += noise[int(T*(1-tr)):]
                    elif kind == 'babble':
                        if len(e_w) > 2:
                            other = e_w[np.random.randint(len(e_w))]
                            if other.shape[-1] > T:
                                start = np.random.randint(0, other.shape[-1] - T)
                                other = other[..., start:start+T]
                            else:
                                other = F.pad(other, (0, T-other.shape[-1]))
                            snr = float(np.random.uniform(-8, 3))
                            w_out += (10**(-snr/20)) * other / other.std().clamp(min=1e-6) * n_std
                        else:
                            t = torch.arange(T, device=w.device).float()
                            mod = 0.5 + 0.5 * torch.sin(2*np.pi*t/16000*np.random.uniform(3,8))
                            snr = float(np.random.uniform(-5, 5))
                            w_out += (10**(-snr/20)) * torch.randn_like(w) * n_std * mod
                    if w_out.dim() == 1:
                        w.copy_(w_out)
                    else:
                        w.copy_(w_out.squeeze())
            e_w.append(e)
            q_w.append(q)
            txts.append(p['enroll_txt'])
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
            labels.extend([p['label'] for p in batch])
        if (i + 128) % 5000 == 0 or (i + 128) >= len(pairs):
            print(f'  {min(i+128, len(pairs))}/{len(pairs)}')
    zf.close()
    return np.array(at_ps), np.array(fu_ps), np.array(aa_ps), np.array(labels)

# ── Load training data with hard negs ──
all_pairs = []
with open(cfg.train_csv) as f:
    for r in csv.DictReader(f):
        all_pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt','').lower()})

# Add hard negs
for hn in ["baseline/hard_neg_at_v8.json", "train/self_paired.json"]:
    hp = os.path.join(PATHS.root, hn)
    if os.path.isfile(hp):
        with open(hp) as f: all_pairs += json.load(f)

# Sample 100K for training
rng = np.random.default_rng(42)
pos = [p for p in all_pairs if p['label'] == 1]
neg = [p for p in all_pairs if p['label'] == 0]
n_pos = min(50000, len(pos))
n_neg = min(50000, len(neg))
train_pairs = rng.choice(pos, n_pos, replace=False).tolist() + rng.choice(neg, n_neg, replace=False).tolist()
rng.shuffle(train_pairs)
print(f'Train: {len(train_pairs)} pairs ({n_pos} pos, {n_neg} neg)')

# ── Dev set (always loaded) ──
dev_pairs = []
with open(cfg.dev_seen_csv) as f:
    for r in csv.DictReader(f):
        dev_pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt','').lower()})
print(f'Dev: {len(dev_pairs)} pairs')

cache_path = 'stacker_cache.npz'
if os.path.isfile(cache_path):
    print('Loading cached features...')
    c = np.load(cache_path)
    at_train, fu_train, aa_train, lb_train = c['at_tr'], c['fu_tr'], c['aa_tr'], c['lb_tr']
    at_dev, fu_dev, aa_dev, lb_dev = c['at_de'], c['fu_de'], c['aa_de'], c['lb_de']
else:
    print('\nGetting train probs (with noise)...')
    at_train, fu_train, aa_train, lb_train = get_probs(cfg.train_zip, train_pairs, add_noise=True)
    
    print('Getting dev probs...')
    at_dev, fu_dev, aa_dev, lb_dev = get_probs(cfg.dev_seen_zip, dev_pairs)
    
    np.savez(cache_path,
             at_tr=at_train, fu_tr=fu_train, aa_tr=aa_train, lb_tr=lb_train,
             at_de=at_dev, fu_de=fu_dev, aa_de=aa_dev, lb_de=lb_dev)
    print(f'Cached to {cache_path}')

# ── Dev set ──

# ── Features ──
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

X_tr = make_features(at_train, fu_train, aa_train)
X_de = make_features(at_dev, fu_dev, aa_dev)

# ── CatBoost + LightGBM Stacking ──
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

X_tr_s, X_val_s, y_tr_s, y_val_s = train_test_split(
    X_tr, lb_train, test_size=0.2, random_state=42, stratify=lb_train
)

# 1. CatBoost
print("Training CatBoost...")
cat_model = CatBoostClassifier(
    iterations=1000, depth=4, learning_rate=0.05,
    loss_function='Logloss', eval_metric='AUC',
    random_seed=42, l2_leaf_reg=3.0, verbose=50,
)
cat_model.fit(X_tr_s, y_tr_s, eval_set=(X_val_s, y_val_s),
              early_stopping_rounds=20, use_best_model=True)

# 2. LightGBM
print("\nTraining LightGBM...")
lgb_params = {
    'objective': 'binary', 'metric': 'auc', 'max_depth': 4,
    'num_leaves': 15, 'learning_rate': 0.05, 'min_child_samples': 20,
    'subsample': 0.8, 'colsample_bytree': 0.8, 'verbosity': -1, 'seed': 42,
}
train_data = lgb.Dataset(X_tr_s, label=y_tr_s)
val_data = lgb.Dataset(X_val_s, label=y_val_s, reference=train_data)
lgb_model = lgb.train(lgb_params, train_data, num_boost_round=1000,
    valid_sets=[val_data], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])

# 3. Fusion & Evaluation
cat_pred_de = cat_model.predict_proba(X_de)[:, 1]
lgb_pred_de = lgb_model.predict(X_de)
ens_pred_de = 0.5 * cat_pred_de + 0.5 * lgb_pred_de

print(f'\nAT alone dev:          {roc_auc_score(lb_dev, at_dev):.4f}')
print(f'CatBoost dev:          {roc_auc_score(lb_dev, cat_pred_de):.4f}')
print(f'LightGBM dev:          {roc_auc_score(lb_dev, lgb_pred_de):.4f}')
print(f'Cat+LGB ensemble dev:  {roc_auc_score(lb_dev, ens_pred_de):.4f}')

# Feature importance
feat_names = ['at_p','fu_p','aa_p','logit_at','logit_fu','logit_aa',
              'at_fu','at_aa','fu_aa','|at-fu|','|at-aa|','|fu-aa|',
              'max_p','min_p','median_p','max-median']
gain = lgb_model.feature_importance(importance_type='gain')
print("\nTop 10 Features (LightGBM):")
for n, g in sorted(zip(feat_names, gain), key=lambda x: -x[1])[:10]:
    print(f'  {n:>15}: {g:.1f}')

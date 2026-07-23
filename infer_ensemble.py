"""Learn ensemble weights on dev set, then apply to eval."""
import sys, os, csv, io, zipfile
import numpy as np
import torch, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS

import importlib.util
spec = importlib.util.spec_from_file_location('td', 'train_dual.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
AudioTextModel, DualConfig = td.AudioTextModel, td.Config

from train_fusion import AudioTextFusionModel

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

# Load both models
at = AudioTextModel('', 256, unfreeze=0).to(device)
at.load_state_dict(torch.load('output/backup_final/at_v8_ep29_seen08462_unseen07889.pt', map_location=device)['model'], strict=False)
at.eval()

fu = AudioTextFusionModel(256, unfreeze=0).to(device)
fu.load_state_dict(torch.load('output/fusion_v2/best.pt', map_location=device)['model'], strict=False)
fu.eval()

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

def run_model(name, zip_path, csv_path):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    zf = zipfile.ZipFile(zip_path, 'r')
    at_ps, fu_ps, labels = [], [], []

    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            e_w.append(read_and_pad(zf, p['id'], 'enroll'))
            q_w.append(read_and_pad(zf, p['id'], 'query'))
            txts.append(p['enroll_txt'])
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        with torch.no_grad():
            _, score, _, _ = at(eb, txts, qb)
            at_ps.extend(torch.sigmoid(score).detach().cpu().numpy().tolist())
            logit, _, _ = fu(eb, txts, qb)
            fu_ps.extend(torch.sigmoid(logit).detach().cpu().numpy().tolist())
            labels.extend([p['label'] for p in batch])

        if (i + 128) % 10000 == 0 or (i + 128) >= len(pairs):
            print(f'  [{name}] {min(i+128, len(pairs))}/{len(pairs)}')

    zf.close()
    return np.array(at_ps), np.array(fu_ps), np.array(labels)

# Run on dev set
print('Running on dev_seen...')
at_s, fu_s, lb_s = run_model('dev_seen', cfg.dev_seen_zip, cfg.dev_seen_csv)
print(f'AT seen AUC: {roc_auc_score(lb_s, at_s):.4f}')
print(f'Fusion seen AUC: {roc_auc_score(lb_s, fu_s):.4f}')

print('\nRunning on dev_unseen...')
at_u, fu_u, lb_u = run_model('dev_unseen', cfg.dev_unseen_zip, cfg.dev_unseen_csv)
print(f'AT unseen AUC: {roc_auc_score(lb_u, at_u):.4f}')
print(f'Fusion unseen AUC: {roc_auc_score(lb_u, fu_u):.4f}')

# ── Learn ensemble on dev_seen ──
print('\nLearning ensemble weights on dev_seen...')

# Method 1: Logistic regression on posteriors
X_s = np.stack([at_s, fu_s], axis=1)
clf = LogisticRegression()
clf.fit(X_s, lb_s)
w_at, w_fu = clf.coef_[0]
print(f'Logistic weights: AT={w_at:.4f}, Fusion={w_fu:.4f}, bias={clf.intercept_[0]:.4f}')

ens_s = clf.predict_proba(X_s)[:, 1]
ens_u = clf.predict_proba(np.stack([at_u, fu_u], axis=1))[:, 1]
print(f'Ensemble seen AUC: {roc_auc_score(lb_s, ens_s):.4f}')
print(f'Ensemble unseen AUC: {roc_auc_score(lb_u, ens_u):.4f}')

# Method 2: Simple weight grid search
print('\nGrid search:')
best_w, best_auc = 0, 0
for w in np.linspace(0, 1, 21):
    ens = w * at_s + (1 - w) * fu_s
    auc = roc_auc_score(lb_s, ens)
    if auc > best_auc:
        best_auc = auc
        best_w = w
ens_u_grid = best_w * at_u + (1 - best_w) * fu_u
print(f'Best weight: AT={best_w:.2f} Fusion={1-best_w:.2f}')
print(f'Seen AUC: {best_auc:.4f}')
print(f'Unseen AUC: {roc_auc_score(lb_u, ens_u_grid):.4f}')

# ── Apply best weights to eval set ──
print('\nRunning eval set with best ensemble...')

# Use logistic regression weights
final_w_at = w_at / (w_at + w_fu)  # normalize
final_w_fu = w_fu / (w_at + w_fu)

all_results = []

for prefix, csv_path, zip_path in [
    ('seen_pair', cfg.eval_seen_csv, cfg.eval_seen_zip),
    ('unseen_pair', cfg.eval_unseen_csv, cfg.eval_unseen_zip)
]:
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({"id": r["id"], "txt": r.get("enroll_txt", "").lower()})

    zf = zipfile.ZipFile(zip_path, 'r')


    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            e_w.append(read_and_pad(zf, p['id'], 'enroll'))
            q_w.append(read_and_pad(zf, p['id'], 'query'))
            txts.append(p['txt'])
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        with torch.no_grad():
            _, score, _, _ = at(eb, txts, qb)
            p_at = torch.sigmoid(score).detach().cpu().numpy()
            logit, _, _ = fu(eb, txts, qb)
            p_fu = torch.sigmoid(logit).detach().cpu().numpy()

        p_ens = final_w_at * p_at + final_w_fu * p_fu
        for j in range(len(batch)):
            all_results.append((f"{prefix}_{batch[j]['id'].replace('pair_', '')}", float(p_ens[j])))

        if (i + 128) % 10000 == 0 or (i + 128) >= len(pairs):
            print(f'  {min(i+128, len(pairs))}/{len(pairs)}')

    zf.close()

out_path = 'submission_ensemble.csv'
all_r = sorted(all_results, key=lambda x: x[0])
import csv as csv_mod
with open(out_path, 'w', newline='') as f:
    w = csv_mod.writer(f)
    w.writerow(['id', 'posterior'])
    w.writerows(all_r)

ps = np.array([p for _, p in all_r])
print(f'\nSaved: {out_path}')
print(f'Total: {len(all_r)} rows')
print(f'Mean={ps.mean():.4f} Std={ps.std():.4f} >0.5={(ps>0.5).sum()}/{len(ps)}')

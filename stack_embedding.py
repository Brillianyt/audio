"""Level 3: Embedding-level fusion. Extract 256d embeddings → MLP."""
import sys, os, csv, io, zipfile, json, copy
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

def get_embeds(zip_path, pairs, add_noise=False):
    """Extract embeddings + posteriors + labels."""
    zf = zipfile.ZipFile(zip_path, 'r')
    emb_list, prob_list, labels = [], [], []
    
    for i in range(0, len(pairs), 64):
        batch = pairs[i:i+64]
        e_w, q_w, txts = [], [], []
        for p in batch:
            eid = p.get("enroll_id", p['id'])
            qid = p.get("query_id", p['id'])
            e = read_and_pad(zf, eid, 'enroll')
            q = read_and_pad(zf, qid, 'query')
            # Noise augmentation
            if add_noise:
                for w in [e, q]:
                    if np.random.rand() < 0.5:
                        snr = float(np.random.uniform(-10, 10))
                        w += (10**(-snr/20)) * torch.randn_like(w) * w.std().clamp(min=1e-6)
            e_w.append(e); q_w.append(q)
            txts.append(p['enroll_txt'])
        
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)
        
        with torch.no_grad():
            # AT: ea (enroll audio), et (enroll text), posterior
            _, score_at, ea_at, et_at = at(eb, txts, qb)
            # Also get eq (query audio) from AT's encoder
            eq_at = at.encoder(qb)
            
            # Fusion: ea_emb, qa_emb, posterior
            logit_fu, ea_fu, qa_fu = fu(eb, txts, qb)
            
            # AA: e_emb, q_emb, posterior
            logit_aa, e_aa, q_aa = aa(eb, qb)
        
        # Concat 3 key embeddings: AT(et) + Fusion(qa) + AA(e) = 768d
        for j in range(len(batch)):
            emb = torch.cat([
                et_at[j],           # AT: text embedding (256)
                qa_fu[j],           # Fusion: query after cross-attn (256)
                e_aa[j],            # AA: enroll audio (256)
            ]).cpu().numpy()
            emb_list.append(emb)
            prob_list.append([
                torch.sigmoid(score_at[j]).item(),
                torch.sigmoid(logit_fu[j]).item(),
                torch.sigmoid(logit_aa[j]).item(),
            ])
            labels.append(batch[j]['label'])
        
        if (i + 64) % 5000 == 0 or (i + 64) >= len(pairs):
            print(f'  {min(i+64, len(pairs))}/{len(pairs)}')
    
    zf.close()
    return np.array(emb_list), np.array(prob_list), np.array(labels)


# ── Load data ──
all_pairs = []
with open(cfg.train_csv) as f:
    for r in csv.DictReader(f):
        all_pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt','').lower()})
for hn in ["baseline/hard_neg_at_v8.json", "train/self_paired.json"]:
    hp = os.path.join(PATHS.root, hn)
    if os.path.isfile(hp):
        with open(hp) as f: all_pairs += json.load(f)

rng = np.random.default_rng(42)
pos = [p for p in all_pairs if p['label'] == 1]
neg = [p for p in all_pairs if p['label'] == 0]
n_pos = min(50000, len(pos))
n_neg = min(50000, len(neg))
train_pairs = rng.choice(pos, n_pos, replace=False).tolist() + rng.choice(neg, n_neg, replace=False).tolist()
rng.shuffle(train_pairs)
print(f'Train: {len(train_pairs)} pairs')

dev_pairs = []
with open(cfg.dev_seen_csv) as f:
    for r in csv.DictReader(f):
        dev_pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt','').lower()})
print(f'Dev: {len(dev_pairs)} pairs')

# Cache
cache_path = 'embed_cache.npz'
if os.path.isfile(cache_path):
    print('Loading cached embeddings...')
    c = np.load(cache_path, allow_pickle=True)
    emb_tr, prob_tr, lb_tr = c['emb_tr'], c['prob_tr'], c['lb_tr']
    emb_de, prob_de, lb_de = c['emb_de'], c['prob_de'], c['lb_de']
else:
    print('Extracting train embeddings (with noise)...')
    emb_tr, prob_tr, lb_tr = get_embeds(cfg.train_zip, train_pairs, add_noise=True)
    print('Extracting dev embeddings...')
    emb_de, prob_de, lb_de = get_embeds(cfg.dev_seen_zip, dev_pairs)
    np.savez(cache_path, emb_tr=emb_tr, prob_tr=prob_tr, lb_tr=lb_tr,
             emb_de=emb_de, prob_de=prob_de, lb_de=lb_de)
    print('Cached!')

print(f'Embed dim: {emb_tr.shape[1]}, Train: {len(emb_tr)}, Dev: {len(emb_de)}')

# ── SuperStacker ──
class SuperStacker(nn.Module):
    def __init__(self, emb_dim=256, prob_dim=6):
        super().__init__()
        self.emb_branch = nn.Sequential(
            nn.Linear(emb_dim * 3, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
        )
        self.prob_branch = nn.Sequential(
            nn.Linear(prob_dim, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.2),
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(256 + 32, 128), nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, emb_at, emb_fu, emb_aa, p_at, p_fu, p_aa):
        emb_at = F.normalize(emb_at, p=2, dim=1)
        emb_fu = F.normalize(emb_fu, p=2, dim=1)
        emb_aa = F.normalize(emb_aa, p=2, dim=1)
        emb_concat = torch.cat([emb_at, emb_fu, emb_aa], dim=1)

        d_at_fu = torch.abs(p_at - p_fu)
        d_at_aa = torch.abs(p_at - p_aa)
        d_fu_aa = torch.abs(p_fu - p_aa)
        prob_concat = torch.cat([p_at, p_fu, p_aa, d_at_fu, d_at_aa, d_fu_aa], dim=1)

        emb_feat = self.emb_branch(emb_concat)
        prob_feat = self.prob_branch(prob_concat)
        return self.fusion_head(torch.cat([emb_feat, prob_feat], dim=1)).squeeze(-1)

model = SuperStacker().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

# Split embeddings into AT/FU/AA components
def split_emb(emb):
    return (emb[:, :256], emb[:, 256:512], emb[:, 512:])  # AT, Fusion, AA

X_tr_emb = torch.from_numpy(emb_tr).float().to(device)
X_tr_prob = torch.from_numpy(prob_tr).float().to(device)
X_de_emb = torch.from_numpy(emb_de).float().to(device)
X_de_prob = torch.from_numpy(prob_de).float().to(device)
y_t = torch.from_numpy(lb_tr.astype(np.float32)).to(device)

best_auc, best_state, patience = 0, None, 0
for ep in range(200):
    perm = torch.randperm(len(X_tr_emb))
    for i in range(0, len(X_tr_emb), 256):
        idx = perm[i:i+256]
        e_at, e_fu, e_aa = split_emb(X_tr_emb[idx])
        logit = model(e_at, e_fu, e_aa, X_tr_prob[idx,:1], X_tr_prob[idx,1:2], X_tr_prob[idx,2:])
        loss = F.binary_cross_entropy_with_logits(logit, y_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    
    with torch.no_grad():
        e_at_d, e_fu_d, e_aa_d = split_emb(X_de_emb)
        de_auc = roc_auc_score(lb_de, torch.sigmoid(model(e_at_d, e_fu_d, e_aa_d, X_de_prob[:,:1], X_de_prob[:,1:2], X_de_prob[:,2:])).cpu().numpy())
    
    if de_auc > best_auc:
        best_auc = de_auc
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        patience = 0
    else:
        patience += 1
    
    if ep % 10 == 0 or patience == 15:
        with torch.no_grad():
            e_at_t, e_fu_t, e_aa_t = split_emb(X_tr_emb)
            tr_auc = roc_auc_score(lb_tr, torch.sigmoid(model(e_at_t, e_fu_t, e_aa_t, X_tr_prob[:,:1], X_tr_prob[:,1:2], X_tr_prob[:,2:])).cpu().numpy())
            print(f'ep{ep:3d} tr={tr_auc:.4f} dev={de_auc:.4f} best={best_auc:.4f} pat={patience}')
    
    if patience >= 15:
        print(f'  Early stop at ep{ep}')
        break

model.load_state_dict(best_state)
with torch.no_grad():
    final = roc_auc_score(lb_de, torch.sigmoid(model(e_at_d, e_fu_d, e_aa_d, X_de_prob[:,:1], X_de_prob[:,1:2], X_de_prob[:,2:])).cpu().numpy())
    # Baseline: AT alone
    print(f'\nAT alone:        {roc_auc_score(lb_de, prob_de[:,0]):.4f}')
    print(f'Embed MLP dev:   {final:.4f} (best={best_auc:.4f})')

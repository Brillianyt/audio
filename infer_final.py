"""Ensemble inference: AA Hybrid v2 + AT backup + fusion heads → submission.csv"""
import torch, os, sys, csv, numpy as np, io, zipfile
import torch.nn as nn, torch.nn.functional as F
import soundfile as sf
sys.path.insert(0, 'baseline'); from config import PATHS

device = 'cuda'

# ── Models ──
import train_aa_hybrid
import train_at_orig

# AA Hybrid v2
aa = train_aa_hybrid.AAHybridModel(256, unfreeze=3).to(device)
ckpt_aa = torch.load('output/aa_hybrid/best.pt', map_location=device, weights_only=False)
aa.load_state_dict(ckpt_aa['model'], strict=False)
aa.eval()

# AT backup (need encoder + text_enc for cos(et, eq))
at = train_at_orig.AudioTextModel(256, unfreeze=4).to(device)
ckpt_at = torch.load('output/backup/at_best.pt', map_location=device, weights_only=False)
at.load_state_dict(ckpt_at['model'], strict=False)
at.eval()

# Fusion heads
class FusionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(5, 16), nn.ReLU(), nn.Linear(16, 1))
    def forward(self, p_aa, p_at):
        conf_aa = (p_aa - 0.5).abs() * 2
        conf_at = (p_at - 0.5).abs() * 2
        gap = p_aa - p_at
        return self.net(torch.stack([p_aa, p_at, conf_aa, conf_at, gap], dim=-1)).squeeze(-1)

ckpt_fusion = torch.load('output/fusion_heads_hybrid_v2.pt', map_location=device, weights_only=False)
fus_s = FusionHead().to(device); fus_s.load_state_dict(ckpt_fusion['model_s']); fus_s.eval()
fus_u = FusionHead().to(device); fus_u.load_state_dict(ckpt_fusion['model_u']); fus_u.eval()

print(f'AA seen={ckpt_aa.get("auc_seen",-1):.4f}, AT unseen={ckpt_at.get("auc_unseen",-1):.4f}')
print(f'Fusion seen={ckpt_fusion["auc_s"]:.4f}, unseen={ckpt_fusion["auc_u"]:.4f}')

# ── Inference ──
def read_wav(zf, pid, role):
    for name in [f'wav/{pid}_{role}.wav', f'wav/{pid}.wav']:
        try: data = zf.read(name); break
        except KeyError: continue
    else: raise KeyError(name)
    wav, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    ms = int(1.5 * 16000)
    if len(wav) > ms: wav = wav[:ms]
    return torch.from_numpy(wav).float()

def run_inference(csv_path, zip_path, prefix, fusion_head):
    results = []
    zf = zipfile.ZipFile(zip_path, 'r')
    
    # Load all rows
    rows = []
    with open(csv_path, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    
    # Batch process
    batch_size = 64
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start+batch_size]
        
        # Read all audio for this batch
        e_list, q_list, txt_list, pid_list = [], [], [], []
        for row in batch:
            pid = row['id']; txt = row['enroll_txt'].lower()
            e = read_wav(zf, pid, 'enroll')
            q = read_wav(zf, pid, 'query')
            e_list.append(e); q_list.append(q); txt_list.append(txt); pid_list.append(pid)
        
        # Pad batch
        ml = max(max(e.shape[-1], q.shape[-1]) for e,q in zip(e_list, q_list))
        e_batch = torch.stack([F.pad(e, (0, ml-e.shape[-1])) if e.shape[-1]<ml else e for e in e_list]).to(device)
        q_batch = torch.stack([F.pad(q, (0, ml-q.shape[-1])) if q.shape[-1]<ml else q for q in q_list]).to(device)
        
        with torch.no_grad():
            score_aa, *_ = aa(e_batch, q_batch)
            p_aa = torch.sigmoid(score_aa * 3.0).squeeze()
            
            _, _, et = at(e_batch, txt_list)
            eq_at = at.encoder(q_batch)
            cs_at = (et * eq_at).sum(-1)
            p_at = torch.sigmoid(cs_at * 8.0).squeeze()
            
            logit = fusion_head(p_aa.unsqueeze(-1) if p_aa.dim()==1 else p_aa, p_at.unsqueeze(-1) if p_at.dim()==1 else p_at)
            if logit.dim() > 0:
                posteriors = torch.sigmoid(logit).cpu().tolist()
            else:
                posteriors = [torch.sigmoid(logit).item()]
        
        for pid, post in zip(pid_list, posteriors if isinstance(posteriors, list) else [posteriors]):
            short_id = pid.replace('pair_', '')  # pair_000001 → 000001
            results.append((f'{prefix}_{short_id}', post))
        
        if (start // batch_size) % 10 == 0:
            print(f'  {prefix}: {start+len(batch)}/{len(rows)}')
    
    zf.close()
    return results

print('Inferring eval_seen...')
seen_results = run_inference(
    os.path.join(PATHS.root, 'evalcsv_without_label', 'eval_seen_without_label.csv'),
    os.path.join(PATHS.root, 'eval', 'eval_seen', 'wav.zip'),
    'seen_pair', fus_s)
print(f'  {len(seen_results)} samples')

print('Inferring eval_unseen...')
unseen_results = run_inference(
    os.path.join(PATHS.root, 'evalcsv_without_label', 'eval_unseen_without_label.csv'),
    os.path.join(PATHS.root, 'eval', 'eval_unseen', 'wav.zip'),
    'unseen_pair', fus_u)
print(f'  {len(unseen_results)} samples')

# Save
all_results = seen_results + unseen_results
out_path = os.path.join(PATHS.root, 'submission.csv')
with open(out_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'posterior'])
    for pid, post in all_results:
        if isinstance(post, (list, tuple)):
            post = post[0] if len(post) > 0 else 0.5
        writer.writerow([pid, f'{float(post):.6f}'])

print(f'Saved {out_path} ({len(all_results)} rows)')

# Stats
posteriors_flat = [float(p[0]) if isinstance(p, (list,tuple)) else float(p) for _, p in all_results]
print(f'Posterior stats: mean={np.mean(posteriors_flat):.4f} std={np.std(posteriors_flat):.4f} '
      f'min={np.min(posteriors_flat):.4f} max={np.max(posteriors_flat):.4f}')
print(f'Pos (>0.5): {sum(1 for p in posteriors_flat if p>0.5)}/{len(posteriors_flat)}')

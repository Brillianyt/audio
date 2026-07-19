cd /root/autodl-tmp/keyword_detect
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=baseline:. python3 -u -c "
import csv, io, os, torch, zipfile, soundfile as sf, numpy as np, sys
sys.path.insert(0,'baseline')
from train_dual import Config, PATHS
cfg = Config(); cfg.__post_init__()
device = 'cuda'

# Load AT v2 checkpoint
ckpt = torch.load('output/dual_at_v2_text/best.pt', map_location=device)

# Build model matching V2 architecture (CharBiGRU 64-dim)
import torch.nn as nn, torch.nn.functional as F
class CharBiGRU_64(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True, num_layers=1)
        self.proj = nn.Linear(128, dim)
    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0,256,device=device)
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1)

from train_dual import WhisperEncoder
model = nn.Module()
model.encoder = WhisperEncoder('base', 256, unfreeze=0).to(device)
model.text_enc = CharBiGRU_64(256).to(device)
model.log_var = nn.Parameter(torch.tensor(0.0))
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print(f'AT v2 loaded: unseen={ckpt.get(\"auc_unseen\",\"?\"):.4f}')

import re as _re
_cmu = None
def _cmudict():
    global _cmu
    if _cmu is None: import cmudict; _cmu = cmudict.dict()
    return _cmu
def w2p(word):
    cmu = _cmudict(); w = word.lower().strip(\"'s\\\"-.,!?;:\")
    if not w: return []
    pl = cmu.get(w)
    return [_re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []

def read_wav(zip_path, pid, role):
    with zipfile.ZipFile(zip_path,'r') as z: data = z.read(f'wav/{pid}_{role}.wav')
    wav, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
    if wav.ndim>1: wav=wav.mean(1)
    if sr!=16000: import torchaudio; wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),sr,16000).squeeze(0).numpy()
    return torch.from_numpy(wav.astype('float32')[:int(1.5*16000)]).float()

def predict(zip_path, csv_path, prefix):
    pairs = []
    with open(csv_path, encoding='utf-8') as f:
        for r in csv.DictReader(f): pairs.append(r)
    rows, bs = [], 128
    for i in range(0, len(pairs), bs):
        batch = pairs[i:i+bs]
        e_w, q_w, txts = [], [], []
        for bp in batch:
            e_w.append(read_wav(zip_path, bp['id'], 'enroll'))
            q_w.append(read_wav(zip_path, bp['id'], 'query'))
            txts.append(bp.get('enroll_txt',''))
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        ep, qp = torch.zeros(len(batch),ml), torch.zeros(len(batch),ml)
        for j,(ew,qw) in enumerate(zip(e_w,q_w)):
            ep[j,:ew.shape[0]]=ew; qp[j,:qw.shape[0]]=qw
        with torch.no_grad():
            cos_ae, ea, et = model.encoder(ep.to(device)), model.text_enc(txts), None
            cos_tq = (et * model.encoder(qp.to(device))).sum(-1)
            prob = torch.sigmoid(cos_tq).cpu().numpy()
        for j, bp in enumerate(batch):
            rows.append((f'{prefix}_{bp[\"id\"]}', float(prob[j])))
        if (i//bs)%20==0: print(f'  {prefix}: {min(i+bs,len(pairs))}/{len(pairs)}')
    return rows

rows = predict(cfg.eval_seen_zip, cfg.eval_seen_csv, 'seen') + predict(cfg.eval_unseen_zip, cfg.eval_unseen_csv, 'unseen')
print(f'Total: {len(rows)}')
with open('submission_at_v2.csv','w',newline='',encoding='utf-8') as f:
    w = csv.writer(f); w.writerow(['id','posterior']); w.writerows(rows)
print('Saved submission_at_v2.csv')
" > output/infer.log 2>&1 &
echo $!

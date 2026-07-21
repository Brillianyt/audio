"""Test AT + padding on dev seen."""
import torch, os, sys, csv, io, zipfile, numpy as np
import torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'baseline'); from config import PATHS
device='cuda'

from train_dual import WhisperEncoder, CharBiGRUEncoder

enc = WhisperEncoder('base', 256, unfreeze=4).to(device)
tenc = CharBiGRUEncoder(256).to(device)
ckpt = torch.load('output/backup/at_best.pt', map_location=device, weights_only=False)
enc.load_state_dict({k.replace('encoder.',''):v for k,v in ckpt['model'].items() if k.startswith('encoder.')}, strict=False)
tenc.load_state_dict({k.replace('text_enc.',''):v for k,v in ckpt['model'].items() if k.startswith('text_enc.')}, strict=False)
enc.eval(); tenc.eval()
print(f'AT loaded: unseen={ckpt.get("auc_unseen",-1):.4f}')

import random; random.seed(42)
rows=[]
with open(PATHS.dev_seen_csv) as f:
    for r in csv.DictReader(f): rows.append(r)
rows = random.sample(rows, 1024)

def test(pad_sec):
    scores, labels = [], []
    for r in rows:
        pid = r['id']; txt = r['enroll_txt'].lower()
        with zipfile.ZipFile(PATHS.dev_seen_zip, 'r') as zf:
            d = zf.read(f'wav/{pid}_query.wav')
        w, sr = sf.read(io.BytesIO(d), dtype='float32', always_2d=False)
        if w.ndim > 1: w = w.mean(axis=1)
        w = w[:int(2.5*16000)]
        w = torch.from_numpy(w).float()
        pad_len = int(pad_sec * 16000)
        if pad_len > 0:
            w = F.pad(w, (pad_len, pad_len))
        else:
            w = w[:int(1.5*16000)]
        w = w.unsqueeze(0).to(device)
        with torch.no_grad():
            et = tenc([txt])
            eq = enc(w)
            cs = torch.sigmoid((et * eq).sum(-1) * 8.0).item()
        scores.append(cs)
        labels.append(int(r['label']))
    auc = roc_auc_score(labels, scores)
    print(f'  pad={pad_sec}s: AUC={auc:.4f}')

for p in [0.0, 0.5]: test(p)

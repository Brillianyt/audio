"""Test AT + padding silence on dev."""
import torch, os, sys, csv, io, zipfile, numpy as np, re, cmudict
import torch.nn as nn, torch.nn.functional as F
import soundfile as sf, whisper
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'baseline'); from config import PATHS
device='cuda'

# ── Reconstruct AT backup model ──
_cmu=cmudict.dict()
def w2p(w):
    w=w.lower().strip("'s\"-.,!?;:"); pl=_cmu.get(w)
    return [re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []
PV=['AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH','UNK']
P2I={p:i for i,p in enumerate(PV)};_pc={}
def t2p(t):
    if t not in _pc: ph=w2p(t); _pc[t]=[P2I.get(p,39) for p in ph] if ph else [39]
    return _pc[t]

class PhBG(nn.Module):
    def __init__(self):
        super().__init__();self.emb=nn.Embedding(40,256)
        self.gru=nn.GRU(256,256,batch_first=True,bidirectional=True,num_layers=2,dropout=0.1)
        self.proj=nn.Linear(512,256)
    def forward(self,texts):
        dev=next(self.parameters()).device
        if not texts:return torch.zeros(0,256,device=dev)
        ph=[t2p(t) for t in texts];mx=max(len(p) for p in ph)
        idx=torch.zeros(len(texts),mx,dtype=torch.long,device=dev)
        for i,p in enumerate(ph):
            for j,pid in enumerate(p[:mx]):idx[i,j]=pid
        x=self.emb(idx);_,h=self.gru(x);h=torch.cat([h[-2],h[-1]],-1)
        return F.normalize(self.proj(h),dim=-1)

w=whisper.load_model('base');dim=w.dims.n_audio_state
class Enc(nn.Module):
    def __init__(self):
        super().__init__()
        self.whisper=w;self.whisper.eval();self.whisper.requires_grad_(False)
        self.attn_l=nn.Linear(dim,dim//4)
        self.attn_w=nn.Parameter(torch.randn(dim//4,1))
        self.proj=nn.Linear(dim*2,256)
    def forward(self,wav):
        wav=wav.to(device).float()
        import whisper.audio as wa
        mel=wa.log_mel_spectrogram(wav).to(device)
        x=self.whisper.encoder(mel)
        h=torch.tanh(self.attn_l(x));aw=F.softmax(h@self.attn_w,dim=1)
        mu=(x*aw).sum(1);sg=((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)),dim=-1)

enc=Enc().to(device);tenc=PhBG().to(device)
ckpt=torch.load('output/backup/at_best.pt',map_location=device,weights_only=False)
enc.load_state_dict({k.replace('encoder.',''):v for k,v in ckpt['model'].items() if k.startswith('encoder.')},strict=False)
tenc.load_state_dict({k.replace('text_enc.',''):v for k,v in ckpt['model'].items() if k.startswith('text_enc.')},strict=False)
enc.eval();tenc.eval();print(f'AT loaded: unseen={ckpt.get("auc_unseen",-1):.4f}')

import random;random.seed(42)
rows=[]
with open(PATHS.dev_seen_csv) as f:
    for r in csv.DictReader(f):rows.append(r)
rows=random.sample(rows,1024)

def eval_at(pad):
    s,ls=[],[]
    for r in rows:
        pid=r['id'];txt=r['enroll_txt'].lower()
        with zipfile.ZipFile(PATHS.dev_seen_zip,'r') as zf:
            d=zf.read(f'wav/{pid}_query.wav')
        w,_=sf.read(io.BytesIO(d),dtype='float32',always_2d=False)
        if w.ndim>1:w=w.mean(axis=1)
        ms=int(1.5*16000);w=w[:ms] if len(w)>ms else w
        w=torch.from_numpy(w).float()
        if pad>0:pad_s=int(pad*16000);w=F.pad(w,(pad_s,pad_s))
        w=w.unsqueeze(0).to(device)
        with torch.no_grad():
            et=tenc([txt]);eq=enc(w)
            cs=torch.sigmoid((et*eq).sum(-1)*8.0).item()
        s.append(cs);ls.append(int(r['label']))
    print(f'AT pad={pad}s seen: AUC={roc_auc_score(ls,s):.4f}')

for p in[0,0.5,0.75]:eval_at(p)

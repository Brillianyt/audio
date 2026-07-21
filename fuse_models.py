"""Train a fusion head to combine AA and AT predictions.
Freeze both models, train only the fusion MLP on dev set.
"""
import torch, os, sys, numpy as np
import torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
sys.path.insert(0,'baseline'); from config import PATHS
from train_aa import AudioAudioModel
from train_at_v3 import WhisperEncoder, ComparisonHead, PairDataset, load_pairs, collate_text
import re, cmudict
from torch.utils.data import DataLoader

device='cuda'

# ── AT model with 2-layer GRU ──
_cmu=cmudict.dict()
def w2p(w):
    w=w.lower().strip("'s\"-.,!?;:"); pl=_cmu.get(w)
    return [re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []
PV=['AA','AE','AH','AO','AW','AY','B','CH','D','DH','EH','ER','EY','F','G','HH','IH','IY','JH','K','L','M','N','NG','OW','OY','P','R','S','SH','T','TH','UH','UW','V','W','Y','Z','ZH','UNK']
P2I={p:i for i,p in enumerate(PV)}; _pc={}
def t2p(t):
    if t not in _pc: ph=w2p(t); _pc[t]=[P2I.get(p,39) for p in ph] if ph else [39]
    return _pc[t]
class PhBG(nn.Module):
    def __init__(self,d=256):
        super().__init__(); self.emb=nn.Embedding(40,d)
        self.gru=nn.GRU(d,d,batch_first=True,bidirectional=True,num_layers=2,dropout=0.1)
        self.proj=nn.Linear(d*2,d)
    def forward(self,texts):
        dev=next(self.parameters()).device
        if not texts: return torch.zeros(0,256,device=dev),None
        ph=[t2p(t) for t in texts]; mx=max(len(p) for p in ph)
        idx=torch.zeros(len(texts),mx,dtype=torch.long,device=dev)
        for i,p in enumerate(ph):
            for j,pid in enumerate(p[:mx]): idx[i,j]=pid
        x=self.emb(idx); _,h=self.gru(x); h=torch.cat([h[-2],h[-1]],-1)
        return F.normalize(self.proj(h),dim=-1),None
class ATM(nn.Module):
    def __init__(self,d=256,u=4):
        super().__init__(); self.encoder=WhisperEncoder('base',d,u); self.text_enc=PhBG(d); self.compare=ComparisonHead(d)
    def forward(self,e,txts,q=None):
        ea=self.encoder(e); et,_=self.text_enc(txts); logit=self.compare(ea,et).squeeze(-1)
        if q is not None: eq=self.encoder(q); return logit,ea,et,eq
        return logit,ea,et

# ── Fusion model ──
class FusionHead(nn.Module):
    """Learn to combine AA and AT predictions."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
    def forward(self, p_aa, p_at):
        conf_aa = (p_aa - 0.5).abs() * 2
        conf_at = (p_at - 0.5).abs() * 2
        gap = p_aa - p_at
        features = torch.stack([p_aa, p_at, conf_aa, conf_at, gap], dim=-1)
        return self.net(features).squeeze(-1)

cfg=type('C',(),{})(); cfg.max_audio_sec=1.5

# Load frozen models
aa=AudioAudioModel(256,unfreeze=2).to(device)
aa.load_state_dict(torch.load('output/aa_v5/best.pt',map_location=device,weights_only=False)['model'],strict=False)
for p in aa.parameters(): p.requires_grad=False; aa.eval()

at=ATM(256,4).to(device)
at.load_state_dict(torch.load('output/backup/at_best.pt',map_location=device,weights_only=False)['model'],strict=False)
for p in at.parameters(): p.requires_grad=False; at.eval()

# ── Extract predictions for both dev sets ──
def extract_preds(csv_f, zip_f):
    loader=DataLoader(PairDataset(load_pairs(csv_f),zip_f,cfg),batch_size=128,collate_fn=collate_text,shuffle=False,num_workers=0)
    ps_aa,ps_at,ls=[],[],[]
    with torch.no_grad():
        for e,q,y,txts,_ in loader:
            e,q=e.to(device),q.to(device)
            cs_aa,_,_=aa(e,q); p_aa=torch.sigmoid(cs_aa*8.0)
            _,_,et=at(e,txts); eq_at=at.encoder(q); cs_at=(et*eq_at).sum(-1); p_at=torch.sigmoid(cs_at*8.0)
            ps_aa.append(p_aa.cpu()); ps_at.append(p_at.cpu())
            ls.append(y)
    return torch.cat(ps_aa), torch.cat(ps_at), torch.cat(ls)

print("Extracting predictions...")
pa_s, pt_s, ls_s = extract_preds(PATHS.dev_seen_csv, PATHS.dev_seen_zip)
pa_u, pt_u, ls_u = extract_preds(PATHS.dev_unseen_csv, PATHS.dev_unseen_zip)

# Combine both sets for training
pa = torch.cat([pa_s, pa_u]); pt = torch.cat([pt_s, pt_u]); ls = torch.cat([ls_s, ls_u])
print(f"Total dev samples: {len(ls)}")

# ── Train fusion head ──
fusion = FusionHead().to(device)
opt = torch.optim.Adam(fusion.parameters(), lr=0.01, weight_decay=1e-4)
crit = nn.BCEWithLogitsLoss()

# Train
indices = torch.randperm(len(ls))
split = int(0.8 * len(ls))
tr_idx, val_idx = indices[:split], indices[split:]

best_auc = 0
for ep in range(100):
    fusion.train()
    idx = tr_idx[torch.randperm(len(tr_idx))]
    for i in range(0, len(idx), 256):
        b = idx[i:i+256]
        logit = fusion(pa[b].to(device), pt[b].to(device))
        loss = crit(logit, ls[b].float().to(device))
        opt.zero_grad(); loss.backward(); opt.step()

    # Eval
    fusion.eval()
    with torch.no_grad():
        logit_val = fusion(pa[val_idx].to(device), pt[val_idx].to(device))
        prob_val = torch.sigmoid(logit_val).cpu().numpy()
    auc_val = roc_auc_score(ls[val_idx].numpy(), prob_val)

    if auc_val > best_auc:
        best_auc = auc_val
        best_state = {k:v.cpu().clone() for k,v in fusion.state_dict().items()}

    if ep % 20 == 0:
        print(f"  ep{ep}: loss={loss.item():.4f} val_auc={auc_val:.4f}")

# ── Final eval ──
fusion.load_state_dict(best_state); fusion.eval()
for name, pa_sub, pt_sub, ls_sub in [
    ('seen', pa_s, pt_s, ls_s),
    ('unseen', pa_u, pt_u, ls_u),
    ('both', pa, pt, ls),
]:
    with torch.no_grad():
        prob = torch.sigmoid(fusion(pa_sub.to(device), pt_sub.to(device))).cpu().numpy()
    auc = roc_auc_score(ls_sub.numpy(), prob)
    print(f"  Fusion {name}: AUC={auc:.4f}")

# Save
torch.save({'model': best_state, 'auc': best_auc}, 'output/fusion_head.pt')
print(f"Saved fusion_head.pt (val_auc={best_auc:.4f})")

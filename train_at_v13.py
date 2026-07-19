"""AT v13 — text gate on both enroll and query, initialized to identity."""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

from train_at_v3 import WhisperEncoder, PhonemeBiGRUEncoder

class TextGateModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=12):
        super().__init__()
        self.audio_enc = WhisperEncoder("base", embed_dim, unfreeze)
        self.text_enc = PhonemeBiGRUEncoder(embed_dim)
        # Gate: starts at identity (output=0 so gate=1)
        self.gate_net = nn.Linear(embed_dim, embed_dim)
        nn.init.zeros_(self.gate_net.weight)
        nn.init.zeros_(self.gate_net.bias)
        # Learnable scale for gate effect
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, enroll_audio, texts, query_audio):
        ea = self.audio_enc(enroll_audio)
        et = self.text_enc(texts)[0]
        eq = self.audio_enc(query_audio)

        # Gate from text: 1 + alpha * gate_output (init: no effect)
        gate = 1.0 + self.alpha * torch.tanh(self.gate_net(et))
        ea_gated = ea * gate
        eq_gated = eq * gate

        score = (ea_gated * eq_gated).sum(-1)
        return score, ea_gated, eq_gated

def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows

def dedup(pairs, n=1):
    seen = {}; res = []
    for p in pairs:
        k = (p["enroll_txt"].lower(), p.get("query_txt",p["enroll_txt"]).lower())
        if k not in seen: seen[k] = 0
        if seen[k] < n: seen[k] += 1; res.append(p)
    return res

def load_all(cfg):
    fp = os.path.join(PATHS.root, "train", "cleaned_pairs.json")
    all_p = json.load(open(fp))
    pos = [p for p in all_p if p["label"]==1]; neg = [p for p in all_p if p["label"]==0]
    hard = [p for p in neg if any(k in p.get("id","") for k in {"hard_neg","hn_","phoneme","hnat_"})]
    hard_ids = {p["id"] for p in hard}
    easy = [p for p in neg if p["id"] not in hard_ids]
    print(f"  pos={len(pos)} hard={len(hard)} easy={len(easy)}")
    return pos, hard, easy

from train_at_v3 import PairDataset, collate_text, load_pairs as lp

class Config:
    embed_dim=256; sample_rate=16000; max_audio_sec=1.5
    unfreeze_at=12; epochs=50; lr=3e-4; batch_size=512
    pos_weight=2.0; num_workers=8; log_every=50
    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r,"train","wav.zip")
        self.train_csv = os.path.join(r,"train","train_label.csv")
        if os.path.isfile(os.path.join(r,"train_subset","wav.zip")):
            self.train_zip = os.path.join(r,"train_subset","wav.zip")
            self.train_csv = os.path.join(r,"train_subset","train_label.csv")
        for k in ("dev_seen","dev_unseen"):
            setattr(self,f"{k}_zip", os.path.join(r,"dev",k,"wav.zip"))
            setattr(self,f"{k}_csv", os.path.join(r,"dev",k,f"{k}_label.csv"))

def train(cfg, args):
    device = "cuda"; torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "at_v13")
    os.makedirs(out_dir, exist_ok=True)
    print("[v13] loading data..."); pos, hard, easy = load_all(cfg)
    model = TextGateModel(cfg.embed_dim, cfg.unfreeze_at).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded from {args.load_ckpt}")
    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir,"latest.pt"),os.path.join(out_dir,"best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen",-1); start_ep = ckpt.get("epoch",0)+1
                print(f"  resumed ep{start_ep}"); break
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {tr:,}")
    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params":wav_p,"lr":cfg.lr/5},{"params":oth_p,"lr":1e-3}], weight_decay=1e-4)
    total_steps = cfg.epochs * 360000 // cfg.batch_size
    warmup_steps = max(100, total_steps//20)
    def lr_lm(s):
        if s < warmup_steps: return s/warmup_steps
        return 0.5*(1+math.cos(math.pi*(s-warmup_steps)/(total_steps-warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lm)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    def dv_ld(z,c): return DataLoader(lp(c),batch_size=cfg.batch_size*2,num_workers=0,collate_fn=collate_text,shuffle=False)
    dv_s = dv_ld(cfg.dev_seen_zip,cfg.dev_seen_csv)
    dv_u = dv_ld(cfg.dev_unseen_zip,cfg.dev_unseen_csv)
    
    for ep in range(start_ep, cfg.epochs+1):
        np_ = min(60000,len(pos)); nh_ = min(300000,len(hard)*2)
        idx_p = np.random.permutation(len(pos))[:np_]
        idx_h = np.random.choice(len(hard), nh_, replace=True)
        subset = [pos[i] for i in idx_p] + [hard[i] for i in idx_h]
        np.random.shuffle(subset)
        loader = DataLoader(PairDataset(subset,cfg.train_zip,cfg), batch_size=cfg.batch_size,
                           shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_text,
                           pin_memory=True, drop_last=True)
        print(f"[v13 ep{ep}] train={len(subset)} pos={np_} hard={nh_}")
        model.train(); ts=time.time(); tl=0; nb=0; cp=0; cn=0; cn_=0
        for e,q,y,txts,_ in loader:
            e,q,y=e.to(device),q.to(device),y.to(device)
            for a in [e,q]:
                snr=float(np.random.choice(
                    [np.random.uniform(0,5),np.random.uniform(-5,0),np.random.uniform(-10,-5)],
                    p=[0.7,0.2,0.1]))
                a.add_((10**(-snr/20))*torch.randn_like(a)*a.std(-1,keepdim=True))
            with torch.cuda.amp.autocast():
                logit,_,_ = model(e,txts,q)
                loss = crit(logit,y)
            opt.zero_grad(); scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); scaler.step(opt); scaler.update(); sched.step()
            tl+=loss.item(); nb+=1; pm,nm=(y==1),(y==0)
            cp+=logit[pm].mean().item() if pm.any() else 0
            cn+=logit[nm].mean().item() if nm.any() else 0
            cn_+=1
            if cn_%cfg.log_every==0: print(f"  [ep{ep}] b{cn_} loss={tl/nb:.3f} cos+={cp/cn_:.3f} cos-={cn/cn_:.3f}")
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps,ls=[],[]
            for e,q,y,txts,_ in ld:
                e,q=e.to(device),q.to(device)
                logit,_,_ = model(e,txts,q)
                ps.append(torch.sigmoid(logit).cpu().numpy()); ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls),np.concatenate(ps))
        as_=ev(dv_s); au_=ev(dv_u)
        print(f"[v13 ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={tl/nb:.4f} ({time.time()-ts:.0f}s)")
        if au_>best:
            best=au_; torch.save({"model":model.state_dict(),"auc_unseen":au_,"auc_seen":as_,"epoch":ep}, os.path.join(out_dir,"best.pt"))
        torch.save({"model":model.state_dict(),"auc_unseen":au_,"auc_seen":as_,"epoch":ep}, os.path.join(out_dir,"latest.pt"))

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--load-ckpt",default=""); p.add_argument("--epochs",type=int,default=50)
    p.add_argument("--bs",type=int,default=512); p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    cfg=Config(); cfg.__post_init__()
    cfg.epochs=args.epochs; cfg.batch_size=args.bs
    train(cfg,args)

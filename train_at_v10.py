"""AT v10 — text-corrected enrollment audio, then cos(query, corrected_enroll)."""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

# ── Reuse AT components ──
from train_at_v3 import WhisperEncoder, PhonemeBiGRUEncoder, ComparisonHead

class TextCorrectedModel(nn.Module):
    """Text corrects enrollment audio, then cos with query audio."""
    def __init__(self, embed_dim=256, unfreeze=2):
        super().__init__()
        self.audio_enc = WhisperEncoder("base", embed_dim, unfreeze)
        self.text_enc = PhonemeBiGRUEncoder(embed_dim)
        # Text correction gate: et → correction vector (-1 to 1)
        self.corrector = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
            nn.Linear(embed_dim, embed_dim), nn.Tanh(),
        )
        # Learnable scale for how much correction to apply
        self.gamma = nn.Parameter(torch.tensor(0.05))

    def forward(self, enroll_audio, texts, query_audio):
        ea = self.audio_enc(enroll_audio)
        et = self.text_enc(texts)[0]  # (embedding, None) -> embedding
        eq = self.audio_enc(query_audio)

        # Text generates a correction mask
        correction = self.corrector(et)
        # Apply correction to enrollment audio embedding
        ea_corrected = ea + self.gamma * correction * ea

        # Corrected enrollment vs query
        logit = (ea_corrected * eq).sum(-1)
        return logit, ea_corrected, eq


# ── Data (same as AT v9) ──
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt","")})
    return rows

def dedup(pairs, n=1):
    seen = {}
    res = []
    for p in pairs:
        k = (p["enroll_txt"].lower(), p.get("query_txt",p["enroll_txt"]).lower())
        if k not in seen: seen[k] = 0
        if seen[k] < n: seen[k] += 1; res.append(p)
    return res

def load_all(cfg):
    fp = os.path.join(PATHS.root, "train", "cleaned_pairs.json")
    all_p = json.load(open(fp))
    pos = [p for p in all_p if p["label"]==1]
    neg = [p for p in all_p if p["label"]==0]
    hard = [p for p in neg if any(k in p.get("id","") for k in {"hard_neg","hn_","phoneme","hnat_"})]
    hard_ids = {p["id"] for p in hard}
    easy = [p for p in neg if p["id"] not in hard_ids]
    print(f"  pos={len(pos)} hard={len(hard)} easy={len(easy)}")
    return pos, hard, easy

class PairDataset(Dataset):
    def __init__(self, pairs, zp, cfg):
        self.pairs = pairs; self.zp = zp; self.cfg = cfg; self._zc = {}
    def _getz(self, p):
        z = p.get("zip","") or self.zp
        if z and not z.startswith("/"): z = os.path.join(PATHS.root, z)
        pid = os.getpid()
        if pid not in self._zc: self._zc[pid] = {}
        if z not in self._zc[pid]: self._zc[pid][z] = zipfile.ZipFile(z,"r")
        return self._zc[pid][z]
    def _rd(self, pid, role, zf):
        for n in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try: return zf.read(n)
            except: continue
        raise KeyError
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        p = self.pairs[i]; pid = p["id"]
        zf = self._getz(p)
        d = self._rd(p.get("enroll_id",pid), "enroll", zf)
        w,sr = sf.read(io.BytesIO(d), dtype="float32", always_2d=False)
        if w.ndim > 1: w = w.mean(1)
        if sr != 16000: import torchaudio; w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(1.5*16000)
        if len(w) > ms: w = w[:ms]
        e = torch.from_numpy(w).float()
        d = self._rd(p.get("query_id",pid), "query", zf)
        w,sr = sf.read(io.BytesIO(d), dtype="float32", always_2d=False)
        if w.ndim > 1: w = w.mean(1)
        if sr != 16000: w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        if len(w) > ms: w = w[:ms]
        q = torch.from_numpy(w).float()
        return e, q, float(p.get("label",0)), p.get("enroll_txt","").lower()

def collate(b):
    ml = max(max(x[0].shape[-1], x[1].shape[-1]) for x in b)
    es, qs, ls, ts = [], [], [], []
    for x in b:
        e,q = x[0],x[1]
        es.append(F.pad(e,(0,ml-e.shape[-1])) if e.shape[-1]<ml else e)
        qs.append(F.pad(q,(0,ml-q.shape[-1])) if q.shape[-1]<ml else q)
        ls.append(x[2]); ts.append(x[3])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls,dtype=torch.float32), ts

# ── Training ──
def train_v10(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "at_v10")
    os.makedirs(out_dir, exist_ok=True)

    print("[v10] loading data..."); pos, hard, easy = load_all(cfg)
    model = TextCorrectedModel(cfg.embed_dim, cfg.unfreeze_layers).to(device)

    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded from {args.load_ckpt}")

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0); start_ep = ckpt.get("epoch",0)+1
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

    from train_at_v3 import PairDataset as DevDS, collate_text as dv_coll, load_pairs as lp
    def dv_ld(z,c): return DataLoader(DevDS(lp(c),z,cfg), batch_size=cfg.batch_size*2, num_workers=0, collate_fn=dv_coll, shuffle=False)
    dv_s = dv_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dv_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs+1):
        np_ = min(60000, len(pos))
        nh_ = min(300000, len(hard)*10)
        idx_p = np.random.permutation(len(pos))[:np_]
        idx_h = np.random.choice(len(hard), nh_, replace=True)
        subset = [pos[i] for i in idx_p] + [hard[i] for i in idx_h]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg), batch_size=cfg.batch_size,
                            shuffle=True, num_workers=cfg.num_workers, collate_fn=collate,
                            pin_memory=True, drop_last=True)
        print(f"[v10 ep{ep}] train={len(subset)} pos={np_} hard={nh_}")

        model.train(); ts=time.time(); tl=0.0; nb=0
        cp, cn, cn_ = 0.0, 0.0, 0
        for e,q,y,txts in loader:
            e,q,y = e.to(device), q.to(device), y.to(device)
            for a in [e,q]:
                snr = np.random.uniform(-10,5)
                a.add_((10**(-snr/20))*torch.randn_like(a)*a.std(-1,keepdim=True))
            with torch.cuda.amp.autocast():
                logit,_,_ = model(e, txts, q)
                loss = crit(logit, y)
            opt.zero_grad(); scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            scaler.step(opt); scaler.update(); sched.step()
            tl += loss.item(); nb += 1
            pm, nm = (y==1), (y==0)
            cp += logit[pm].mean().item() if pm.any() else 0
            cn += logit[nm].mean().item() if nm.any() else 0
            cn_ += 1
            if cn_ % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cn_} loss={tl/nb:.3f} cos+={cp/cn_:.3f} cos-={cn/cn_:.3f}")

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e,q,y,txts,_ in ld:
                e,q = e.to(device), q.to(device)
                logit,_,_ = model(e, txts, q)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[v10 ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={tl/nb:.4f} ({time.time()-ts:.0f}s)")
        if au_ > best:
            best = au_
            torch.save({"model":model.state_dict(),"auc_unseen":au_,"auc_seen":as_,"epoch":ep}, os.path.join(out_dir,"best.pt"))
        torch.save({"model":model.state_dict(),"auc_unseen":au_,"auc_seen":as_,"epoch":ep}, os.path.join(out_dir,"latest.pt"))

class Config:
    embed_dim=256; sample_rate=16000; max_audio_sec=1.5; unfreeze_layers=4
    epochs=50; lr=3e-4; batch_size=512; pos_weight=5.0; num_workers=8; log_every=50
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.batch_size = args.bs
    train_v10(cfg, args)

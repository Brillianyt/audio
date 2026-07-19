"""End-to-end ensemble: AT + AA + fusion head, trained together."""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

# Reuse existing architectures
from train_at_v3 import AudioTextModel, WhisperEncoder as ATEncoder, PhonemeBiGRUEncoder, ComparisonHead
from train_aa_pk import AAPKModel


class EnsembleModel(nn.Module):
    """AT + AA + fusion head, all trainable."""
    def __init__(self, embed_dim=256, unfreeze_at=4, unfreeze_aa=6):
        super().__init__()
        self.at = AudioTextModel(embed_dim, unfreeze_at)
        self.aa = AAPKModel(embed_dim, unfreeze_aa)
        # Fusion head: 5 input features
        self.fusion = nn.Sequential(
            nn.Linear(5, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, enroll_audio, texts, query_audio):
        # AT forward
        _, ea_at, et = self.at(enroll_audio, texts)
        eq_at = self.at.encoder(query_audio)
        at_cos = (et * eq_at).sum(-1)
        at_align = (et * ea_at).sum(-1)

        # AA forward
        ea_aa, _ = self.aa(enroll_audio)
        eq_aa, _ = self.aa(query_audio)
        aa_cos = (ea_aa * eq_aa).sum(-1)

        # Build fusion features
        at_p = torch.sigmoid(at_cos * 8.0)
        aa_p = torch.sigmoid(aa_cos * 8.0)
        feats = torch.stack([
            at_p, aa_p,
            (at_p - 0.5).abs(),
            (aa_p - 0.5).abs(),
            torch.sigmoid(at_align * 8.0),
        ], dim=1)

        logit = self.fusion(feats).squeeze(-1)
        return logit, at_cos, aa_cos


# ═══════════ Data ═══════════
def load_cleaned():
    fp = os.path.join(PATHS.root, "train", "cleaned_pairs.json")
    return json.load(open(fp))

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.default_zip_path = zip_path; self.cfg = cfg; self._zc = {}
    def _get_zip(self, pair):
        z = pair.get("zip","") or self.default_zip_path
        if z and not z.startswith("/"): z = os.path.join(PATHS.root, z)
        pid = os.getpid()
        if pid not in self._zc: self._zc[pid] = {}
        if z not in self._zc[pid]: self._zc[pid][z] = zipfile.ZipFile(z, "r")
        return self._zc[pid][z]
    def _read(self, pid, role, zf):
        for n in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try: d = zf.read(n); break
            except KeyError: continue
        else: raise KeyError(f"wav not found")
        w, sr = sf.read(io.BytesIO(d), dtype="float32", always_2d=False)
        if w.ndim > 1: w = w.mean(1)
        if sr != 16000:
            import torchaudio
            w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(w) > ms: w = w[:ms]
        return torch.from_numpy(w).float()
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        zf = self._get_zip(p)
        return self._read(eid, "enroll", zf), self._read(qid, "query", zf), \
               float(p.get("label",0)), p.get("enroll_txt","").lower()

def collate(b):
    ml = max(max(x[0].shape[-1], x[1].shape[-1]) for x in b)
    es, qs, ls, txts = [], [], [], []
    for x in b:
        e, q = x[0], x[1]
        es.append(F.pad(e, (0, ml-e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml-q.shape[-1])) if q.shape[-1] < ml else q)
        ls.append(x[2]); txts.append(x[3])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls, dtype=torch.float32), txts


# ═══════════ Training ═══════════
def train_ensemble(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "ensemble_v1")
    os.makedirs(out_dir, exist_ok=True)

    print("[Ensemble] loading data...")
    all_pairs = load_cleaned()
    pos = [p for p in all_pairs if p["label"] == 1]
    hard_neg = [p for p in all_pairs if p["label"] == 0 and any(k in p.get("id","") for k in {"hard_neg","hn_","phoneme","hnat_"})]
    print(f"  pos={len(pos)} hard_neg={len(hard_neg)}")

    model = EnsembleModel(cfg.embed_dim, cfg.unfreeze_at, cfg.unfreeze_aa).to(device)

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
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed ep{start_ep}")
                break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    # Group LR
    at_wav = [p for n,p in model.named_parameters() if p.requires_grad and "at.encoder" in n and "whisper" in n]
    at_other = [p for n,p in model.named_parameters() if p.requires_grad and "at." in n and "at.encoder.whisper" not in n]
    aa_wav = [p for n,p in model.named_parameters() if p.requires_grad and "aa.encoder" in n and "whisper" in n]
    aa_other = [p for n,p in model.named_parameters() if p.requires_grad and "aa." in n and "aa.encoder.whisper" not in n]
    fusion_p = [p for n,p in model.named_parameters() if p.requires_grad and "fusion" in n]

    opt = torch.optim.AdamW([
        {"params": at_wav, "lr": cfg.lr/5},
        {"params": at_other, "lr": 1e-3},
        {"params": aa_wav, "lr": cfg.lr/5},
        {"params": aa_other, "lr": 1e-3},
        {"params": fusion_p, "lr": 1e-3},
    ], weight_decay=1e-4)

    total_steps = cfg.epochs * 360000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lambda(s):
        if s < warmup_steps: return s / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Dev loaders (from train_at_v3)
    from train_at_v3 import PairDataset as DevDS, collate_text as dev_collate, load_pairs as lp
    def dev_ld(z, c):
        return DataLoader(DevDS(lp(c), z, cfg), batch_size=cfg.batch_size*2, num_workers=0, collate_fn=dev_collate, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        n_p = min(60000, len(pos))
        n_h = min(300000, len(hard_neg) * 10)
        idx_p = np.random.permutation(len(pos))[:n_p]
        idx_h = np.random.choice(len(hard_neg), n_h, replace=True)
        subset = [pos[i] for i in idx_p] + [hard_neg[i] for i in idx_h]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate,
                            pin_memory=True, drop_last=True)
        print(f"[ep{ep}] train={len(subset)} pos={n_p} hard={n_h}")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        at_cos_pos, at_cos_neg, aa_cos_pos, aa_cos_neg, cn = 0.0, 0.0, 0.0, 0.0, 0

        for e, q, y, txts in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            # Noise
            for aud in [e, q]:
                snr = np.random.uniform(-10, 5)
                aud.add_((10**(-snr/20)) * torch.randn_like(aud) * aud.std(-1, keepdim=True))

            with torch.cuda.amp.autocast():
                logit, at_c, aa_c = model(e, txts, q)
                loss = crit(logit, y)

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()

            total_loss += loss.item(); n_batches += 1
            pos_m, neg_m = (y==1), (y==0)
            at_cos_pos += at_c[pos_m].mean().item() if pos_m.any() else 0
            at_cos_neg += at_c[neg_m].mean().item() if neg_m.any() else 0
            aa_cos_pos += aa_c[pos_m].mean().item() if pos_m.any() else 0
            aa_cos_neg += aa_c[neg_m].mean().item() if neg_m.any() else 0
            cn += 1
            if cn % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cn} loss={total_loss/n_batches:.3f} "
                      f"at_cos+={at_cos_pos/cn:.3f} at_cos-={at_cos_neg/cn:.3f} "
                      f"aa_cos+={aa_cos_pos/cn:.3f} aa_cos-={aa_cos_neg/cn:.3f}")

        # Eval
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                logit, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[Ensemble ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={total_loss/n_batches:.4f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))


class Config:
    embed_dim=256; sample_rate=16000; max_audio_sec=1.5
    unfreeze_at=4; unfreeze_aa=6; epochs=50; lr=3e-4; batch_size=512
    pos_weight=5.0; num_workers=8; log_every=50
    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")
        for k in ("dev_seen","dev_unseen"):
            setattr(self, f"{k}_zip", os.path.join(r, "dev", k, "wav.zip"))
            setattr(self, f"{k}_csv", os.path.join(r, "dev", k, f"{k}_label.csv"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    train_ensemble(cfg, args)

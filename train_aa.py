"""Audio-Audio Siamese training — shared Whisper encoder + cos similarity.
Strategy: use all pos pairs, all neg pairs, push seen to 0.9+ first.
"""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

# ═══════════ WhisperEncoder (same as AT) ═══════════
class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=2):
        super().__init__()
        import whisper
        self.whisper = whisper.load_model(model_name)
        self.whisper.eval()
        for p in self.whisper.parameters(): p.requires_grad = False
        total = len(self.whisper.encoder.blocks)
        self.frozen = max(0, total - unfreeze)
        for i in range(self.frozen, total):
            for p in self.whisper.encoder.blocks[i].parameters(): p.requires_grad = True
        self.whisper.encoder.ln_post.requires_grad_(True)
        self.register_buffer("hann", torch.hann_window(400), persistent=False)
        self.register_buffer("mel_filters", self._make_mel(80), persistent=False)
        self.attn_linear = nn.Linear(self.whisper.dims.n_audio_state, self.whisper.dims.n_audio_state//4)
        self.attn_w = nn.Parameter(torch.randn(self.whisper.dims.n_audio_state//4, 1))
        self.proj = nn.Linear(self.whisper.dims.n_audio_state*2, embed_dim)

    def _make_mel(self, n_mels, n_fft=400):
        import whisper.audio as wa
        filters = wa.mel_filters("cpu", n_mels)
        expected = n_fft // 2 + 1
        if filters.shape[-1] != expected:
            filters = filters[:, :expected] if filters.shape[-1] > expected else \
                      F.pad(filters, (0, expected - filters.shape[-1]))
        return filters

    def forward(self, wav):
        wav = wav.to(next(self.whisper.parameters()).device).float()
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mags = stft[..., :-1].abs() ** 2
        mel = self.mel_filters.to(mags.device).to(mags.dtype) @ mags
        mel = torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.maximum(mel, mel.max()-8.0)
        mel = (mel+4.0)/4.0
        enc = self.whisper.encoder
        n_valid = max(1, (wav.shape[-1]+319)//320)
        with torch.no_grad():
            x = F.gelu(enc.conv1(mel)); x = F.gelu(enc.conv2(x))
            x = x.permute(0,2,1)
            pe = enc.positional_embedding
            if pe.dim() == 2: pe = pe[:x.shape[1]].unsqueeze(0)
            else: pe = pe[:, :x.shape[1]]
            x = x + pe
            for blk in enc.blocks[:self.frozen]: x = blk(x)
            for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)[:, :n_valid, :]
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)

    # ── One-time noise injection on last unfrozen block ──
    def apply_noise(self, noise_ratio=0.3):
        """Replace noise_ratio of last block's params with noise. Save originals."""
        if self.frozen >= len(self.whisper.encoder.blocks):
            return
        blk = self.whisper.encoder.blocks[-1]
        self._saved_params = {}
        for name, p in blk.named_parameters():
            if p.requires_grad:
                self._saved_params[name] = p.data.clone()
                mask = torch.rand_like(p) < noise_ratio
                p.data[mask] = torch.randn_like(p.data[mask]) * p.data.std()
        print(f"  [noise] replaced {noise_ratio*100:.0f}% of last block's weights")

    def restore_weights(self):
        """Restore original weights saved before noise injection."""
        if not hasattr(self, '_saved_params') or not self._saved_params:
            return
        blk = self.whisper.encoder.blocks[-1]
        for name, p in blk.named_parameters():
            if name in self._saved_params:
                p.data = self._saved_params[name]
        print("  [noise] restored original weights")


# ═══════════ Siamese Audio-Audio Model ═══════════
class AudioAudioModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=2):
        super().__init__()
        self.encoder = WhisperEncoder("base", embed_dim, unfreeze)

    def forward(self, e, q):
        ea = self.encoder(e); eq = self.encoder(q)
        return (ea * eq).sum(-1), ea, eq


# ═══════════ Data ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows

def deduplicate_by_word_pair(pairs, max_per_pair=1):
    seen = {}
    deduped = []
    for p in pairs:
        key = (p["enroll_txt"].lower(), p["query_txt"].lower())
        if key not in seen:
            seen[key] = 0
        if seen[key] < max_per_pair:
            seen[key] += 1
            deduped.append(p)
    return deduped

def load_all_data(cfg):
    """AA: original train CSV only. No external data."""
    rng = np.random.default_rng(42)
    all_pairs = load_pairs(cfg.train_csv)
    print(f"  total: {len(all_pairs)} pairs (original CSV only)")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    pos_dedup = pos_pairs  # keep all pos for memorization
    hard_dedup = deduplicate_by_word_pair(neg_pairs, max_per_pair=3)
    rng.shuffle(pos_dedup); rng.shuffle(hard_dedup)
    print(f"  pos={len(pos_dedup)} neg={len(hard_dedup)}")
    return pos_dedup, hard_dedup

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs
        self.default_zip_path = zip_path
        self.cfg = cfg
        self._zc = {}

    def _get_zip_for(self, pair):
        zpath = pair.get("zip", "")
        if not zpath:
            zpath = self.default_zip_path
        else:
            zpath = os.path.join(PATHS.root, zpath)
        pid = os.getpid()
        if pid not in self._zc:
            self._zc[pid] = {}
        if zpath not in self._zc[pid]:
            self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]

    def _read_wav(self, pid, role, zf):
        for name in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try:
                data = zf.read(name)
                break
            except KeyError:
                continue
        else:
            raise KeyError(f"Neither 'wav/{pid}_{role}.wav' nor 'wav/{pid}.wav' found in zip")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pid = p["id"]
        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)
        zf = self._get_zip_for(p)
        e = self._read_wav(eid, "enroll", zf)
        q = self._read_wav(qid, "query", zf)
        return e, q, float(p.get("label", 0)), p.get("enroll_txt", "").lower(), pid

def collate_text(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, ls, txts, ids = [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, ml - e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml - q.shape[-1])) if q.shape[-1] < ml else q)
        ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls, dtype=torch.float32), txts, ids


# ═══════════ Training ═══════════
def train_aa(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "aa_v5")
    os.makedirs(out_dir, exist_ok=True)

    print("[AA] loading data...")
    pos_dedup, hard_dedup = load_all_data(cfg)

    model = AudioAudioModel(cfg.embed_dim, unfreeze=cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded ckpt from {args.load_ckpt}")

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed from {fp} epoch={start_ep}")
                break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    wav_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params": wav_p, "lr": cfg.lr/5}, {"params": oth_p, "lr": 1e-3}],
                            weight_decay=1e-4)
    total_steps = cfg.epochs * 350000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Dev loaders
    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        n_pos_ep = min(300000, len(pos_dedup))
        n_hard_ep = min(50000, len(hard_dedup) * 2)  # gentle hard neg oversample
        idx_pos = np.random.permutation(len(pos_dedup))[:n_pos_ep]
        idx_neg = np.random.choice(len(hard_dedup), n_hard_ep, replace=True)
        subset = [pos_dedup[i] for i in idx_pos] + [hard_dedup[i] for i in idx_neg]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[ep{ep}] train={len(subset)} pos={n_pos_ep} hard={n_hard_ep}")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        cos_pos_sum, cos_neg_sum, cos_n = 0.0, 0.0, 0
        all_cos_pos, all_cos_neg = [], []

        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            snr = float(np.random.choice(
                [np.random.uniform(0, 5), np.random.uniform(-5, 0), np.random.uniform(-10, -5)],
                p=[0.7, 0.2, 0.1]))
            e = e + (10**(-snr/20)) * torch.randn_like(e) * e.std(-1, keepdim=True)
            snr = float(np.random.choice(
                [np.random.uniform(0, 5), np.random.uniform(-5, 0), np.random.uniform(-10, -5)],
                p=[0.7, 0.2, 0.1]))
            q = q + (10**(-snr/20)) * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                logit, ea, eq = model(e, q)
                pos = (y == 1); neg = (y == 0)
                margin = 0.0
                if pos.any(): margin += F.relu(0.6 - logit[pos]).mean()
                if neg.any(): margin += F.relu(logit[neg] + 0.15).mean()
                loss = crit(logit * cfg.cos_scale, y) + 0.1 * margin

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item(); n_batches += 1
            _cp = logit[pos].detach() if pos.any() else torch.tensor([0.0])
            _cn = logit[neg].detach() if neg.any() else torch.tensor([0.0])
            cos_pos_sum += _cp.mean().item(); cos_neg_sum += _cn.mean().item(); cos_n += 1
            all_cos_pos.append(_cp.cpu()); all_cos_neg.append(_cn.cpu())
            if cos_n % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cos_n} loss={total_loss/n_batches:.3f} "
                      f"cos+={cos_pos_sum/cos_n:.3f} cos-={cos_neg_sum/cos_n:.3f} "
                      f"gap={cos_pos_sum/cos_n - cos_neg_sum/cos_n:.3f}")

        # Eval
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls, ids = [], [], []
            for e, q, y, txts, id_ in ld:
                e, q = e.to(device), q.to(device)
                cs, _, _ = model(e, q)
                ps.append(torch.sigmoid(cs * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy()); ids.extend(id_)
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps)), \
                   np.concatenate(ps), np.concatenate(ls), ids

        as_, _, _, _ = ev(dv_s)
        au_, _, _, _ = ev(dv_u)

        print(f"[AA ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={total_loss/n_batches:.4f} "
              f"cos+={cos_pos_sum/cos_n:.3f} cos-={cos_neg_sum/cos_n:.3f} "
              f"({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep}, os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_,
                    "auc_seen": as_, "epoch": ep}, os.path.join(out_dir, "latest.pt"))


# ═══════════ Config ═══════════
class Config:
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5
    unfreeze_layers: int = 6
    epochs: int = 50
    lr: float = 3e-4
    batch_size: int = 512
    cos_scale: float = 8.0
    num_workers: int = 8
    log_every: int = 50

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")
        for k in ("dev_seen","dev_unseen"):
            z = os.path.join(r, "dev", k, "wav.zip")
            c = os.path.join(r, "dev", k, f"{k}_label.csv")
            setattr(self, f"{k}_zip", z)
            setattr(self, f"{k}_csv", c)


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
    train_aa(cfg, args)

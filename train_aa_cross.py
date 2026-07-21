"""AA Cross-Attention — Frame-level alignment (learnable DTW).
Replaces simple cos(ea, eq) with cross-attention between query frames and enroll frames.
"""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS


# ═══════════ Whisper Frame Encoder (no pooling, outputs frames) ═══════════
class WhisperFrameEncoder(nn.Module):
    """Whisper encoder that outputs frame-level features instead of pooled embedding."""
    def __init__(self, model_name="base", embed_dim=256, unfreeze=6):
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
        self.hidden_dim = self.whisper.dims.n_audio_state  # 512 for base
        self.proj = nn.Linear(self.hidden_dim, embed_dim)  # project to embed_dim

    def _make_mel(self, n_mels, n_fft=400):
        import whisper.audio as wa
        filters = wa.mel_filters("cpu", n_mels)
        expected = n_fft // 2 + 1
        if filters.shape[-1] != expected:
            filters = filters[:, :expected] if filters.shape[-1] > expected else \
                      F.pad(filters, (0, expected - filters.shape[-1]))
        return filters

    def forward(self, wav):
        """Returns frame features: (B, T, embed_dim)"""
        wav = wav.to(next(self.whisper.parameters()).device).float()
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mags = stft[..., :-1].abs() ** 2
        mel = self.mel_filters.to(mags.device).to(mags.dtype) @ mags
        mel = torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.maximum(mel, mel.max()-8.0)
        mel = (mel+4.0)/4.0
        enc = self.whisper.encoder
        with torch.no_grad():
            x = F.gelu(enc.conv1(mel)); x = F.gelu(enc.conv2(x))
            x = x.permute(0,2,1)
            pe = enc.positional_embedding
            if pe.dim() == 2: pe = pe[:x.shape[1]].unsqueeze(0)
            else: pe = pe[:, :x.shape[1]]
            x = x + pe
            for blk in enc.blocks[:self.frozen]: x = blk(x)
        for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)
        return F.normalize(self.proj(x), dim=-1)  # (B, T, embed_dim)


# ═══════════ Frame Cross-Attention Matcher ═══════════
class FrameCrossAttention(nn.Module):
    """Cross-attention between query frames and enroll frames → alignment score."""
    def __init__(self, embed_dim=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query_frames, enroll_frames):
        """
        query_frames: (B, Tq, D)
        enroll_frames: (B, Te, D)
        Returns: alignment_score (B,), attended_query (B, D)
        """
        B, Tq, D = query_frames.shape
        _, Te, _ = enroll_frames.shape

        # Project to Q, K, V
        Q = self.q_proj(query_frames).view(B, Tq, self.n_heads, self.head_dim).transpose(1,2)  # (B,H,Tq,d)
        K = self.k_proj(enroll_frames).view(B, Te, self.n_heads, self.head_dim).transpose(1,2)  # (B,H,Te,d)
        V = self.v_proj(enroll_frames).view(B, Te, self.n_heads, self.head_dim).transpose(1,2)  # (B,H,Te,d)

        # Attention: (B, H, Tq, Te)
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        # Context: (B, H, Tq, d)
        context = torch.matmul(attn, V)
        context = context.transpose(1,2).contiguous().view(B, Tq, D)  # (B, Tq, D)
        context = self.out_proj(context)

        # Per-frame alignment score: cosine between query frame and its attended context
        frame_scores = (query_frames * context).sum(-1)  # (B, Tq)

        # Max pooling: focus on the best-aligned frame (keyword region)
        alignment_score = frame_scores.max(dim=-1).values  # (B,)

        # Also compute mean context representation for fusion
        attended_query = context.mean(dim=1)  # (B, D)

        return alignment_score, attended_query, frame_scores


# ═══════════ AA Cross-Attention Model ═══════════
class AACrossModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=6):
        super().__init__()
        self.encoder = WhisperFrameEncoder("base", embed_dim, unfreeze)
        self.cross_attn = FrameCrossAttention(embed_dim)

    def forward(self, enroll, query):
        """Returns alignment_score, attended_enroll_emb, attended_query_emb"""
        e_frames = self.encoder(enroll)     # (B, Te, D)
        q_frames = self.encoder(query)      # (B, Tq, D)
        score, q_attended, _ = self.cross_attn(q_frames, e_frames)
        e_attended = e_frames.mean(dim=1)   # simple mean for enroll (or could do self-attn)
        return score, e_attended, q_attended


# ═══════════ Data (same as train_aa.py) ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows

def deduplicate_by_word_pair(pairs, max_per_pair=1):
    seen = {}; deduped = []
    for p in pairs:
        key = (p["enroll_txt"].lower(), p["query_txt"].lower())
        if key not in seen: seen[key] = 0
        if seen[key] < max_per_pair: seen[key] += 1; deduped.append(p)
    return deduped

def load_all_data(cfg):
    """AA: original train CSV, only positive pairs."""
    rng = np.random.default_rng(42)
    all_pairs = load_pairs(cfg.train_csv)  # original competition data only
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    pos_dedup = deduplicate_by_word_pair(pos_pairs, max_per_pair=10)
    rng.shuffle(pos_dedup)
    print(f"  total pos={len(pos_dedup)} (original CSV only)")
    return pos_dedup

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.default_zip_path = zip_path; self.cfg = cfg; self._zc = {}
    def _get_zip_for(self, pair):
        zpath = pair.get("zip", "")
        if not zpath: zpath = self.default_zip_path
        else: zpath = os.path.join(PATHS.root, zpath)
        pid = os.getpid()
        if pid not in self._zc: self._zc[pid] = {}
        if zpath not in self._zc[pid]: self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]
    def _read_wav(self, pid, role, zf):
        for name in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try: data = zf.read(name); break
            except KeyError: continue
        else: raise KeyError(f"wav/{pid}_{role}.wav not found")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
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
def train_aa_cross(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "aa_cross")
    os.makedirs(out_dir, exist_ok=True)

    print("[AA-Cross] loading data...")
    pos_dedup = load_all_data(cfg)

    model = AACrossModel(cfg.embed_dim, unfreeze=cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        # Filter out incompatible keys (old model has ASP pooling, new model outputs frames)
        state = {k: v for k, v in ckpt["model"].items()
                 if not k.startswith("encoder.proj") and not k.startswith("encoder.attn_")}
        model.load_state_dict(state, strict=False)
        print(f"  loaded encoder from {args.load_ckpt} (skipped proj/attn)")

    best, start_ep = -1.0, 1
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    wav_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params": wav_p, "lr": cfg.lr/5}, {"params": oth_p, "lr": 1e-3}],
                            weight_decay=1e-4)
    total_steps = cfg.epochs * 300000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        n_pos_ep = min(80000, len(pos_dedup))
        idx_pos = np.random.permutation(len(pos_dedup))[:n_pos_ep]
        subset = [pos_dedup[i] for i in idx_pos]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[AA-Cross ep{ep}] train={len(subset)} (pos only)")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        for e, q, y, txts, _ in loader:
            e, q = e.to(device), q.to(device)
            snr = float(np.random.choice(
                [np.random.uniform(0,5), np.random.uniform(-5,0), np.random.uniform(-10,-5)],
                p=[0.7,0.2,0.1]))
            e = e + (10**(-snr/20)) * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + (10**(-snr/20)) * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                score, _, _ = model(e, q)
                # Simple: push cross-attn score > 0.6 for same-word pairs
                loss = F.relu(0.6 - score).mean()

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()

            total_loss += loss.item(); n_batches += 1
            if n_batches % cfg.log_every == 0:
                print(f"  [ep{ep}] b{n_batches} loss={total_loss/n_batches:.3f} "
                      f"score+={score.mean().item():.3f}")

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                score, _, _ = model(e, q)
                ps.append(torch.sigmoid(score * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[AA-Cross ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={total_loss/n_batches:.4f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep}, os.path.join(out_dir, "best.pt"))
            print(f"  saved best.pt (unseen={au_:.4f})")


class Config:
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5
    unfreeze_layers: int = 6
    epochs: int = 30
    lr: float = 3e-4
    batch_size: int = 256
    cos_scale: float = 3.0
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
            setattr(self, f"{k}_zip", z); setattr(self, f"{k}_csv", c)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=256)
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    train_aa_cross(cfg, args)

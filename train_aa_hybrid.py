"""AA Hybrid: cosine(ea,eq) + frame attention residual.
score = α * cos(ea, eq) + (1-α) * attn(q_frames, e_frames)
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


# ═══════════ Whisper Encoder: both pooled emb + frame features ═══════════
class WhisperHybridEncoder(nn.Module):
    """Outputs both pooled embedding (for cosine) and frame features (for attention)."""
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
        # Pooled output: ASP (same as original AA)
        self.hidden_dim = self.whisper.dims.n_audio_state
        self.attn_linear = nn.Linear(self.hidden_dim, self.hidden_dim//4)
        self.attn_w = nn.Parameter(torch.randn(self.hidden_dim//4, 1))
        self.proj = nn.Linear(self.hidden_dim*2, embed_dim)
        # Frame output: project hidden to embed_dim
        self.frame_proj = nn.Linear(self.hidden_dim, embed_dim)

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
        x = enc.ln_post(x)
        x_valid = x[:, :n_valid, :]
        # Pooled embedding (ASP)
        h = torch.tanh(self.attn_linear(x_valid))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x_valid*aw).sum(1)
        sg = ((x_valid**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        emb = F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)
        # Frame features
        frames = F.normalize(self.frame_proj(x_valid), dim=-1)
        return emb, frames


# ═══════════ Frame Cross-Attention ═══════════
class FrameCrossAttention(nn.Module):
    def __init__(self, embed_dim=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, q_frames, e_frames):
        B, Tq, D = q_frames.shape
        _, Te, _ = e_frames.shape
        Q = self.q_proj(q_frames).view(B, Tq, self.n_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(e_frames).view(B, Te, self.n_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(e_frames).view(B, Te, self.n_heads, self.head_dim).transpose(1,2)
        attn = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, Tq, D)
        frame_scores = (q_frames * context).sum(-1)  # (B, Tq)
        return frame_scores.max(dim=-1).values  # max alignment


# ═══════════ AA Hybrid Model ═══════════
class AAHybridModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=6):
        super().__init__()
        self.encoder = WhisperHybridEncoder("base", embed_dim, unfreeze)
        self.attn = FrameCrossAttention(embed_dim)
        self.alpha = nn.Parameter(torch.tensor(0.5))  # starts at 50/50

    def forward(self, enroll, query):
        ea_emb, ea_frames = self.encoder(enroll)
        eq_emb, eq_frames = self.encoder(query)

        # Cosine branch
        cos_score = (ea_emb * eq_emb).sum(-1)  # (B,)

        # Attention branch
        attn_score = self.attn(q_frames=eq_frames, e_frames=ea_frames)  # (B,)

        # Hybrid
        alpha_clamped = torch.sigmoid(self.alpha)  # keep α in [0,1]
        score = alpha_clamped * cos_score + (1 - alpha_clamped) * attn_score

        return score, ea_emb, eq_emb, cos_score, attn_score


# ═══════════ Data (same as train_aa.py, original CSV only) ═══════════
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
    rng = np.random.default_rng(42)
    all_pairs = load_pairs(cfg.train_csv)
    print(f"  total: {len(all_pairs)} pairs (original CSV only)")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    pos_dedup = pos_pairs
    hard_dedup = deduplicate_by_word_pair(neg_pairs, max_per_pair=3)
    rng.shuffle(pos_dedup); rng.shuffle(hard_dedup)
    print(f"  pos={len(pos_dedup)} neg={len(hard_dedup)}")
    return pos_dedup, hard_dedup

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
def train_aa_hybrid(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "aa_hybrid")
    os.makedirs(out_dir, exist_ok=True)

    print("[AA-Hybrid] loading data...")
    pos_dedup, hard_dedup = load_all_data(cfg)

    model = AAHybridModel(cfg.embed_dim, unfreeze=cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded encoder from {args.load_ckpt}")

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir,"latest.pt"), os.path.join(out_dir,"best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen",-1); start_ep = ckpt.get("epoch",0)+1
                print(f"  resumed ep{start_ep}"); break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params":wav_p,"lr":cfg.lr/5},{"params":oth_p,"lr":1e-3}], weight_decay=1e-4)
    total_steps = cfg.epochs * 150000 // cfg.batch_size
    warmup_steps = max(100, total_steps//20)
    def lr_lm(s):
        if s < warmup_steps: return s/warmup_steps
        return 0.5*(1+math.cos(math.pi*(s-warmup_steps)/(total_steps-warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lm)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    def dev_ld(z,c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        n_pos_ep = min(80000, len(pos_dedup))
        n_hard_ep = min(40000, len(hard_dedup) * 2)
        idx_pos = np.random.permutation(len(pos_dedup))[:n_pos_ep]
        idx_hard = np.random.choice(len(hard_dedup), n_hard_ep, replace=True)
        subset = [pos_dedup[i] for i in idx_pos] + \
                 [hard_dedup[i] for i in idx_hard]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[AA-Hybrid ep{ep}] train={len(subset)} pos={n_pos_ep} neg={n_hard_ep}")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            snr = float(np.random.choice(
                [np.random.uniform(0,5), np.random.uniform(-5,0), np.random.uniform(-10,-5)],
                p=[0.7,0.2,0.1]))
            e = e + (10**(-snr/20)) * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + (10**(-snr/20)) * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                score, ea, eq, cos_s, attn_s = model(e, q)
                pos = (y == 1); neg = (y == 0)
                margin = 0.0
                if pos.any(): margin += F.relu(0.6 - score[pos]).mean()
                if neg.any(): margin += F.relu(score[neg] + 0.15).mean()
                loss = crit(score * cfg.cos_scale, y) + 0.1 * margin

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()

            total_loss += loss.item(); n_batches += 1
            if n_batches % cfg.log_every == 0:
                cp = score[pos].mean().item() if pos.any() else 0
                cn = score[neg].mean().item() if neg.any() else 0
                print(f"  [ep{ep}] b{n_batches} loss={total_loss/n_batches:.3f} "
                      f"score+={cp:.3f} score-={cn:.3f} gap={cp-cn:.3f} "
                      f"cos={cos_s.mean().item():.3f} attn={attn_s.mean().item():.3f} α={torch.sigmoid(model.alpha).item():.3f}")

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                score, _, _, _, _ = model(e, q)
                ps.append(torch.sigmoid(score * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[AA-Hybrid ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
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
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    train_aa_hybrid(cfg, args)

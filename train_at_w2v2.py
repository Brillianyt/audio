"""AT with Wav2Vec2 encoder + CharBiGRU text encoder."""
import argparse, csv, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS


# ═══════════ Config ═══════════
class Config:
    embed_dim: int = 256
    max_audio_sec: float = 2.5
    unfreeze_layers: int = 2
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 128
    pos_weight: float = 2.0
    cos_scale: float = 8.0
    num_workers: int = 4
    seed: int = 42
    log_every: int = 50

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        for k in ("dev_seen","dev_unseen"):
            z = os.path.join(r, "dev", k, "wav.zip")
            c = os.path.join(r, "dev", k, f"{k}_label.csv")
            setattr(self, f"{k}_zip", z); setattr(self, f"{k}_csv", c)


# ═══════════ Wav2Vec2 Frame Encoder (no pooling) ═══════════
class W2V2FrameEncoder(nn.Module):
    """Outputs frame features (B, T, D) instead of pooled embedding."""
    def __init__(self, model_name="facebook/wav2vec2-base", embed_dim=256, unfreeze=2):
        super().__init__()
        self.w2v2 = Wav2Vec2Model.from_pretrained(model_name)
        self.w2v2.eval()
        self.dim = self.w2v2.config.hidden_size
        self.max_audio_sec = 2.5
        self.w2v2.feature_extractor.requires_grad_(False)
        total = len(self.w2v2.encoder.layers)
        self.frozen = max(0, total - unfreeze)
        for i, layer in enumerate(self.w2v2.encoder.layers):
            for p in layer.parameters():
                p.requires_grad = i >= self.frozen
        self.proj = nn.Linear(self.dim, embed_dim)

    def forward(self, wav):
        wav = wav.to(next(self.w2v2.parameters()).device)
        ms = int(self.max_audio_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        out = self.w2v2(wav, output_hidden_states=False)
        x = out.last_hidden_state  # (B, T, dim)
        return F.normalize(self.proj(x), dim=-1)  # (B, T, embed_dim)


# ═══════════ CharBiGRU64 Token Encoder (per-token, no pooling) ═══════════
class CharBiGRUTokens(nn.Module):
    """Outputs per-character features (B, L, D) with mask."""
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True, num_layers=1)
        self.proj = nn.Linear(128, dim)

    def forward(self, texts):
        dev = next(self.parameters()).device
        if not texts: return torch.zeros(0, 0, 256, device=dev), torch.zeros(0, 0, dtype=torch.bool, device=dev)
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=dev)
        mask = torch.zeros(len(texts), mx, dtype=torch.bool, device=dev)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i, j] = max(0, min(27, ord(c) - 97))
                mask[i, j] = True
        x = self.char_emb(idx)
        out, _ = self.gru(x)  # (B, L, 128)
        return F.normalize(self.proj(out), dim=-1), mask  # (B, L, D), (B, L)


# ═══════════ Cross-Attention Aligner ═══════════
class CrossAttnAligner(nn.Module):
    """Text tokens attend to audio frames.
    Score = cos(text_token, attended_context) - λ * attention_entropy"""
    def __init__(self, embed_dim=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, text_tokens, audio_frames, txt_mask=None):
        B, L, D = text_tokens.shape
        _, T, _ = audio_frames.shape
        
        Q = self.q_proj(text_tokens).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(audio_frames).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(audio_frames).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, L, T)
        
        # Position bias: sequential alignment
        t_pos = torch.arange(L, device=text_tokens.device).float() / max(L, 1)
        a_pos = torch.arange(T, device=audio_frames.device).float() / max(T, 1)
        pos_bias = -3.0 * (t_pos.view(1, 1, L, 1) - a_pos.view(1, 1, 1, T)).abs()
        attn = attn + pos_bias
        
        if txt_mask is not None:
            attn = attn.masked_fill(~txt_mask[:, None, :, None], -1e9)
        attn = F.softmax(attn, dim=-1)
        
        context = torch.matmul(attn, V).transpose(1, 2).contiguous().view(B, L, D)
        context = F.normalize(context, dim=-1)
        
        # Cosine between text token and its attended audio context (bounded [-1,1])
        cos_scores = (text_tokens * context).sum(-1)  # (B, L)
        if txt_mask is not None:
            cos_scores = cos_scores.masked_fill(~txt_mask, -1.0)
        
        # Attention entropy per token: low entropy = focused alignment
        attn_entropy = -(attn * (attn + 1e-8).log()).sum(-1)  # (B, H, L)
        attn_entropy = attn_entropy.mean(dim=1)  # (B, L) — mean over heads
        entropy_reg = attn_entropy.mean(dim=1)   # (B,) — mean over tokens
        
        # Final: mean cosine score - entropy regularization (penalizes unfocused attention)
        score = cos_scores.mean(dim=1) - 0.05 * entropy_reg  # (B,)
        return score


# ═══════════ Model ═══════════
class W2V2ATModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=2):
        super().__init__()
        self.encoder = W2V2FrameEncoder("facebook/wav2vec2-base", embed_dim, unfreeze)
        self.text_enc = CharBiGRUTokens(embed_dim)
        self.aligner = CrossAttnAligner(embed_dim)

    def forward(self, enroll, texts, query):
        ea_frames = self.encoder(enroll)      # (B, Te, D)
        et_tokens, mask = self.text_enc(texts) # (B, L, D)
        eq_frames = self.encoder(query)        # (B, Tq, D)
        # Both enroll and query audio align to the same text
        score_e = self.aligner(et_tokens, ea_frames, mask)
        score_q = self.aligner(et_tokens, eq_frames, mask)
        return (score_e + score_q) / 2, ea_frames, et_tokens, eq_frames


# ═══════════ Data ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt", ""),
                         "query_txt": r.get("query_txt", "")})
    return rows


class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.default_zip = zip_path; self.cfg = cfg; self._zc = {}

    def _get_zip(self, zpath=None):
        zpath = zpath or self.default_zip
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
        wav = torch.from_numpy(wav).float()
        # 前后补静音 0.5s
        pad = int(0.5 * 16000)
        wav = F.pad(wav, (pad, pad))
        return wav

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        zf = self._get_zip()
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
def train(cfg, args):
    device = "cuda"; torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    print(f"[{args.name}] loading data...")
    all_pairs = load_pairs(cfg.train_csv)
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    model = W2V2ATModel(cfg.embed_dim, cfg.unfreeze_layers).to(device)
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
                best = ckpt.get("auc_unseen", -1); start_ep = ckpt.get("epoch", 0)+1
                print(f"  resumed ep{start_ep}"); break

    print(f"  trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    wav_p = [p for n, p in model.named_parameters() if p.requires_grad and "w2v2" in n]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "w2v2" not in n]
    opt = torch.optim.AdamW([{"params": wav_p, "lr": cfg.lr/5}, {"params": oth_p, "lr": 1e-3}], weight_decay=1e-4)
    total_steps = cfg.epochs * 100000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lm(s):
        if s < warmup_steps: return s / warmup_steps
        return 0.5*(1+math.cos(math.pi*(s-warmup_steps)/(total_steps-warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lm)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg), batch_size=cfg.batch_size*2,
                          collate_fn=collate_text, shuffle=False, num_workers=0)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs+1):
        n_pos_ep = min(60000, len(pos_pairs))
        n_neg_ep = min(80000, len(neg_pairs))
        idx_pos = np.random.permutation(len(pos_pairs))[:n_pos_ep]
        idx_neg = np.random.choice(len(neg_pairs), n_neg_ep, replace=True)
        subset = [pos_pairs[i] for i in idx_pos] + [neg_pairs[i] for i in idx_neg]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg), batch_size=cfg.batch_size,
                            shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[ep{ep}] train={len(subset)} pos={n_pos_ep} neg={n_neg_ep}")

        model.train(); ts = time.time(); tl = 0; nb = 0; cp = 0; cn = 0; cn_ = 0
        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            snr = float(np.random.choice([np.random.uniform(0, 5), np.random.uniform(-5, 0), np.random.uniform(-10, -5)],p=[0.7, 0.2, 0.1]))
            e = e + (10**(-snr/20)) * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + (10**(-snr/20)) * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                score, ea_frames, et_tokens, eq_frames = model(e, txts, q)
                pos = (y == 1); neg = (y == 0)
                margin = 0.0
                if pos.any(): margin += F.relu(0.6 - score[pos]).mean()
                if neg.any(): margin += F.relu(score[neg] + 0.15).mean()
                bce = F.binary_cross_entropy_with_logits(score * cfg.cos_scale, y, pos_weight=torch.tensor(cfg.pos_weight, device=device))

                # SupCon: pull same-word pooled audio embeddings closer
                supcon = 0.0
                if pos.sum() >= 2:
                    # Mean pool frame features for sentence-level embedding
                    ea_pool = ea_frames.mean(dim=1)  # (B, D)
                    eq_pool = eq_frames.mean(dim=1)  # (B, D)
                    # Concatenate enroll and query into one embedding set
                    all_emb = torch.cat([ea_pool, eq_pool], dim=0)  # (2B, D)
                    all_emb = F.normalize(all_emb, dim=-1)
                    all_y = torch.cat([y, y], dim=0)                # (2B,)
                    # Positive pair cosine
                    cos_mat = all_emb @ all_emb.T  # (2B, 2B)
                    same = (all_y.unsqueeze(0) == all_y.unsqueeze(1)).float()
                    same.fill_diagonal_(0)
                    # SupCon: log(sum(negative exp)) - log(positive exp)
                    pos_sim = (cos_mat * same).sum(-1) / (same.sum(-1) + 1e-8)
                    neg_mask = 1.0 - same
                    neg_sim = (cos_mat.exp() * neg_mask).sum(-1).log()
                    supcon = F.relu(1.0 - (pos_sim - neg_sim)).mean() * 0.1

                loss = bce + 0.1 * margin + supcon

            opt.zero_grad(); scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()

            tl += loss.item(); nb += 1
            if pos.any(): cp += score[pos].mean().item()
            if neg.any(): cn += score[neg].mean().item()
            cn_ += 1
            if cn_ % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cn_} loss={tl/nb:.3f} cos+={cp/cn_:.3f} cos-={cn/cn_:.3f} gap={cp/cn_-cn/cn_:.3f}")

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                cos, _, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(cos * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={tl/nb:.4f} ({time.time()-ts:.0f}s)")
        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))


if __name__ == "__main__":
    cfg = Config(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="at_w2v2")
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--unfreeze", type=int, default=cfg.unfreeze_layers)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.batch_size = args.bs; cfg.unfreeze_layers = args.unfreeze
    train(cfg, args)

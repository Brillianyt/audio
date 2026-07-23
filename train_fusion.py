"""Audio-Text Deep Fusion: cross-attention text→enroll→query + fusion MLP.

Architecture:
  text ──→ attend enroll_frames ──→ text_guided_enroll
  text_guided_enroll ──→ attend query_frames ──→ alignment score
  5 embeddings + align_score → fusion MLP → logit
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


# ═══════════ Config ═══════════
class FusionConfig:
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 2.5
    unfreeze_layers: int = 4
    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 128
    pos_weight: float = 5.0
    num_workers: int = 8
    seed: int = 42
    log_every: int = 50

    train_zip: str = ""; train_csv: str = ""
    dev_seen_zip: str = ""; dev_seen_csv: str = ""
    dev_unseen_zip: str = ""; dev_unseen_csv: str = ""
    eval_seen_zip: str = ""; eval_seen_csv: str = ""
    eval_unseen_zip: str = ""; eval_unseen_csv: str = ""

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")
        db = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(db, "dev")): db = os.path.join(db, "dev")
        self.dev_seen_zip = os.path.join(db, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(db, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(db, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(db, "dev_unseen", "dev_unseen_label.csv")
        self.eval_seen_zip = os.path.join(r, "eval", "eval_seen", "wav.zip")
        self.eval_seen_csv = os.path.join(r, "evalcsv_without_label", "eval_seen_without_label.csv")
        self.eval_unseen_zip = os.path.join(r, "eval", "eval_unseen", "wav.zip")
        self.eval_unseen_csv = os.path.join(r, "evalcsv_without_label", "eval_unseen_without_label.csv")


# ═══════════ Whisper Encoder (returns both embedding + frames) ═══════════
class WhisperEncoderFrames(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=2, max_sec=1.5):
        super().__init__()
        import whisper, whisper.audio as wa
        self.whisper = whisper.load_model(model_name)
        self.whisper.eval()
        self.dim = self.whisper.dims.n_audio_state
        self.max_sec = max_sec
        if hasattr(self.whisper, 'decoder'): del self.whisper.decoder

        total = len(self.whisper.encoder.blocks)
        self.frozen = total - unfreeze
        for p in self.whisper.encoder.parameters(): p.requires_grad = False
        for i in range(self.frozen, total):
            for p in self.whisper.encoder.blocks[i].parameters(): p.requires_grad = True

        self.attn_linear = nn.Linear(self.dim, self.dim//4)
        self.attn_w = nn.Parameter(torch.randn(self.dim//4, 1))
        self.proj = nn.Linear(self.dim*2, embed_dim)
        # Frame projection: Whisper dim → embed_dim for cross-attention
        self.frame_proj = nn.Linear(self.dim, embed_dim)
        # Sinusoidal positional encoding (max 150 frames for 3s audio)
        self.register_buffer("hann", torch.hann_window(400))
        self.register_buffer("mel_filters", wa.mel_filters("cpu", 80))
        pe = torch.zeros(150, embed_dim)
        pos = torch.arange(150).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * (-np.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pos_enc", pe.unsqueeze(0))  # (1, 150, D)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for i in range(self.frozen): self.whisper.encoder.blocks[i].eval()
        return self

    def forward(self, wav):
        device = next(self.whisper.parameters()).device
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        wav = wav.to(device)
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mags = stft[..., :-1].abs()**2
        mel = self.mel_filters.to(mags.device).to(mags.dtype) @ mags
        mel = torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.maximum(mel, mel.max()-8.0)
        mel = (mel+4.0)/4.0

        # SpecAugment
        if self.training:
            if np.random.rand() < 0.5:
                f_max = min(10, mel.shape[1] // 4)
                f = np.random.randint(0, f_max + 1)
                f0 = np.random.randint(0, mel.shape[1] - f + 1)
                mel[:, f0:f0+f, :] = 0.0
            if np.random.rand() < 0.5:
                t_max = min(50, mel.shape[2] // 4)
                t = np.random.randint(0, t_max + 1)
                t0 = np.random.randint(0, mel.shape[2] - t + 1)
                mel[:, :, t0:t0+t] = 0.0

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
        x = enc.ln_post(x)[:, :n_valid, :]  # (B, T, D)

        # Pooled embedding (ASP)
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        emb = F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)

        # Frame features projected to embed_dim + positional encoding
        frames = F.normalize(self.frame_proj(x), dim=-1)  # (B, T, embed_dim)
        frames = frames + self.pos_enc[:, :frames.shape[1], :]

        return emb, frames


# ═══════════ CharBiGRU Text Encoder ═══════════
class CharBiGRUEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0, 256, device=device)
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i, j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1)


# ═══════════ Fusion Model ═══════════
class AudioTextFusionModel(nn.Module):
    """Deep fusion: text attends enroll frames → attends query frames → MLP."""
    def __init__(self, embed_dim=256, unfreeze=2):
        super().__init__()
        self.encoder = WhisperEncoderFrames("base", embed_dim, unfreeze)
        self.text_enc = CharBiGRUEncoder(embed_dim)

        # Cross-attention: text → enroll frames
        self.cross_enroll = nn.MultiheadAttention(embed_dim, num_heads=4,
                                                    batch_first=True, dropout=0.1)
        # Cross-attention: text_guided → query frames
        self.cross_query = nn.MultiheadAttention(embed_dim, num_heads=4,
                                                   batch_first=True, dropout=0.1)

        # Fusion MLP: [ea_emb, qa_emb, text_emb, tge, attn_q, align_score]
        fusion_in = embed_dim * 5 + 1
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, embed_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def forward(self, e, texts, q):
        # ── Encode ──
        ea_emb, ea_frames = self.encoder(e)        # (B, D), (B, T1, D)
        et = self.text_enc(texts)                   # (B, D)
        qa_emb, qa_frames = self.encoder(q)         # (B, D), (B, T2, D)

        # ── Cross-attention #1: text → enroll frames ──
        # "这个词发音应该长啥样？"
        tge, _ = self.cross_enroll(
            et.unsqueeze(1), ea_frames, ea_frames
        )  # (B, 1, D)
        tge = tge.squeeze(1)  # (B, D)

        # ── Cross-attention #2: text_guided → query frames ──
        # "query里有这个发音吗？"
        attn_q, attn_w = self.cross_query(
            tge.unsqueeze(1), qa_frames, qa_frames,
            need_weights=True, average_attn_weights=True
        )  # (B, 1, D), (B, T2)
        attn_q = attn_q.squeeze(1)  # (B, D)

        # Alignment score: how well does query match the guided template
        align_score = attn_w.max(dim=-1).values.squeeze(-1)  # (B,)
        align_score = align_score / align_score.detach().clamp(min=1e-6).mean()

        # ── Fusion ──
        feat = torch.cat([
            ea_emb, qa_emb, et, tge, attn_q,
            align_score.unsqueeze(-1),
        ], dim=-1)
        logit = self.fusion(feat).squeeze(-1)
        return logit, ea_emb, qa_emb


# ═══════════ Data ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows


class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.zip_path = zip_path; self.cfg = cfg
        self._zip_cache = {}

    def _get_zip(self):
        pid = os.getpid()
        if pid not in self._zip_cache:
            self._zip_cache[pid] = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_cache[pid]

    def __len__(self): return len(self.pairs)

    def _read(self, pid, role):
        data = self._get_zip().read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim>1: wav=wav.mean(axis=1)
        if sr!=16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),sr,16000).squeeze(0).numpy()
        wav=wav.astype(np.float32)
        ms=int(self.cfg.max_audio_sec*16000)
        if len(wav)>ms: wav=wav[:ms]
        wav=torch.from_numpy(wav).float()
        pad=int(0.5*16000); wav=F.pad(wav,(pad,pad))
        return wav

    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)
        e = self._read(eid, "enroll"); q = self._read(qid, "query")
        return e, q, float(p.get("label",0)), p.get("enroll_txt","").lower(), pid


def collate_fusion(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es,qs,ls,txts,ids = [],[],[],[],[]
    for b in batch:
        e,q=b[0],b[1]
        es.append(F.pad(e,(0,ml-e.shape[-1])) if e.shape[-1]<ml else e)
        qs.append(F.pad(q,(0,ml-q.shape[-1])) if q.shape[-1]<ml else q)
        ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls,dtype=torch.float32), txts, ids


# ═══════════ Training ═══════════
def train(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    # 原始CSV正负样本（label=0是正常的对比负样本）
    print(f"[fusion] {len(all_pairs)} pairs (original CSV only)")
    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    print(f"[fusion] {len(pos_pairs)} pos pairs")

    # 加载 AT v8 挖掘的难负样本
    hn_path = os.path.join(PATHS.root, "baseline/hard_neg_at_v8.json")
    if os.path.isfile(hn_path):
        with open(hn_path) as f:
            hard_negs = json.load(f)
        print(f"  loaded {len(hard_negs)} hard neg pairs from {hn_path}")
    else:
        hard_negs = []

    # 按词分组正样本，用于在线配对
    word_to_idx = {}
    for i, p in enumerate(pos_pairs):
        w = p["enroll_txt"].lower()
        word_to_idx.setdefault(w, []).append(i)
    pos_words = list(word_to_idx.keys())
    print(f"  unique words: {len(pos_words)}")

    def get_loader(ep):
        n_pos = min(10000, len(pos_pairs))
        subset = []
        idx = np.random.permutation(len(pos_pairs))[:n_pos]
        for i in idx:
            p = dict(pos_pairs[i])
            p["label"] = 1
            subset.append(p)
        # 难负样本 + 随机配对混合
        n_hard = min(n_pos // 2, len(hard_negs))
        hard_idx = rng.permutation(len(hard_negs))[:n_hard]
        for i in hard_idx:
            p = dict(hard_negs[i])
            p.setdefault("enroll_id", p["id"])
            p.setdefault("query_id", p["id"])
            subset.append(p)
        # 剩余用随机交叉配对补全
        n_easy = n_pos - n_hard
        for _ in range(n_easy):
            w1, w2 = rng.choice(pos_words, 2, replace=False)
            p1 = pos_pairs[rng.choice(word_to_idx[w1])]
            p2 = pos_pairs[rng.choice(word_to_idx[w2])]
            subset.append({"id": f"neg_{p1['id']}_x_{p2['id']}",
                           "enroll_id": p1["id"], "query_id": p2["id"],
                           "enroll_txt": p1["enroll_txt"], "query_txt": p2["query_txt"],
                           "label": 0})
        np.random.shuffle(subset)
        ds = PairDataset(subset, cfg.train_zip, cfg)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_fusion,
                          pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_fusion, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    model = AudioTextFusionModel(cfg.embed_dim, cfg.unfreeze_layers).to(device)
    best_at, start_ep = -1.0, 1

    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best_at = max(best_at, ckpt.get("auc_seen", -1.0))
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed from {fp} epoch={start_ep}")
                break

    if args.load_ckpt and os.path.isfile(args.load_ckpt):
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        own_state = model.state_dict()
        for k, v in ckpt["model"].items():
            if k in own_state and v.shape == own_state[k].shape:
                own_state[k] = v
        model.load_state_dict(own_state, strict=False)
        print(f"  loaded pretrained weights from {args.load_ckpt}")

    if args.freeze_encoders:
        for n, p in model.named_parameters():
            if (n.startswith("encoder.") or n.startswith("text_enc.")) and "frame_proj" not in n:
                p.requires_grad = False
        print("  frozen: encoder, text_enc (frame_proj trainable)")
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trainable after freeze: {trainable:,}")

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    te_p = [p for n,p in model.named_parameters() if p.requires_grad and "text_enc" in n]
    oth_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n and "text_enc" not in n]
    param_groups = []
    if wav_p:
        param_groups.append({"params": wav_p, "lr": cfg.lr / 5})
    if te_p:
        param_groups.append({"params": te_p, "lr": 3e-4})
    if oth_p:
        param_groups.append({"params": oth_p, "lr": 1e-3})
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4) if len(param_groups) > 1 else \
          torch.optim.AdamW(param_groups[0]["params"], lr=1e-3, weight_decay=1e-4) if param_groups else \
          torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))

    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls, n = 0.0, 0

        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)

            # ── Enhanced noise augmentation ──
            def _augment(wav, batch_ref=None):
                kind = np.random.choice(['gaussian_light', 'gaussian_med', 'gaussian_heavy',
                                         'burst', 'lowpass', 'clip', 'tail_noise', 'babble'],
                                        p=[0.15, 0.15, 0.10, 0.12, 0.10, 0.10, 0.13, 0.15])
                B, T = wav.shape
                wav_out = wav.clone()
                n_std = wav.std(-1, keepdim=True).clamp(min=1e-6)

                if kind == 'gaussian_light':
                    snr = float(np.random.uniform(0, 5))
                    wav_out = wav + (10**(-snr/20)) * torch.randn_like(wav) * n_std
                elif kind == 'gaussian_med':
                    snr = float(np.random.uniform(-5, 0))
                    wav_out = wav + (10**(-snr/20)) * torch.randn_like(wav) * n_std
                elif kind == 'gaussian_heavy':
                    snr = float(np.random.uniform(-15, -5))
                    wav_out = wav + (10**(-snr/20)) * torch.randn_like(wav) * n_std
                elif kind == 'burst':
                    snr = float(np.random.uniform(-10, 0))
                    noise = (10**(-snr/20)) * torch.randn_like(wav) * n_std
                    for b in range(B):
                        t_len = int(T * np.random.uniform(0.3, 0.8))
                        t_start = int(np.random.randint(0, max(1, T - t_len)))
                        wav_out[b, t_start:t_start+t_len] += noise[b, t_start:t_start+t_len]
                elif kind == 'tail_noise':
                    # 音频末尾噪声
                    snr = float(np.random.uniform(-10, 5))
                    noise = (10**(-snr/20)) * torch.randn_like(wav) * n_std
                    tail_ratio = float(np.random.uniform(0.2, 0.5))
                    t_start = int(T * (1 - tail_ratio))
                    wav_out[:, t_start:] += noise[:, t_start:]
                elif kind == 'babble':
                    # 人声喧哗：用batch里其他样本的音频做噪声
                    snr = float(np.random.uniform(-8, 3))
                    if batch_ref is not None and B > 1:
                        # 随机选取另一个样本
                        other_idx = torch.randperm(B, device=wav.device)
                        other = batch_ref[other_idx]
                        # 随机截取一段
                        other_T = other.shape[-1]
                        if other_T > T:
                            start = np.random.randint(0, other_T - T)
                            other = other[:, start:start+T]
                        else:
                            other = torch.nn.functional.pad(other, (0, T-other_T))
                        noise = (10**(-snr/20)) * other / other.std(-1, keepdim=True).clamp(min=1e-6)
                        wav_out = wav + noise * n_std
                    else:
                        # fallback: 调幅噪声模拟人声节奏
                        t = torch.arange(T, device=wav.device).float()
                        mod = 0.5 + 0.5 * torch.sin(2 * np.pi * t / 16000 * np.random.uniform(3, 8))
                        snr = float(np.random.uniform(-5, 5))
                        noise = (10**(-snr/20)) * torch.randn_like(wav) * n_std * mod.unsqueeze(0)
                        wav_out = wav + noise
                elif kind == 'lowpass':
                    cutoff = int(np.random.choice([1000, 2000, 3000, 4000]))
                    try:
                        import torchaudio
                        kernel_size = 16000 // cutoff
                        if kernel_size > 1 and kernel_size < T:
                            kernel = torch.ones(1, 1, kernel_size, device=wav.device) / kernel_size
                            wav_reshaped = wav_out.unsqueeze(1)
                            wav_filtered = F.conv1d(wav_reshaped, kernel, padding=kernel_size//2)
                            wav_out = wav_filtered.squeeze(1)
                    except Exception:
                        pass
                elif kind == 'clip':
                    threshold = float(np.random.uniform(0.1, 0.5))
                    wav_out = wav_out.clamp(-threshold, threshold)
                return wav_out

            e = _augment(e, batch_ref=q)  # 用query做babble参考
            q = _augment(q, batch_ref=e)  # 用enroll做babble参考

            logit, _, _ = model(e, txts, q)
            loss = crit(logit, y)

            # Online mining: focus on hard samples
            p = torch.sigmoid(logit.detach())
            hard = ((y == 0) & (p > 0.5)) | ((y == 1) & (p < 0.3))
            weight = torch.where(hard, 3.0, 0.3)
            loss = (F.binary_cross_entropy_with_logits(logit, y, reduction='none') * weight).mean()

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            ls += loss.item(); n += 1
            if n % cfg.log_every == 0:
                print(f"  [ep{ep}] b{n} loss={ls/n:.3f}")

        # ── Eval ──
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                logit, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s)
        au_ = ev(dv_u)
        print(f"[fusion ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={ls/n:.4f} ({time.time()-ts:.0f}s)")

        if as_ > best_at:
            best_at = as_
            torch.save({"model": model.state_dict(), "auc_seen": as_,
                        "auc_unseen": au_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_seen": as_,
                    "auc_unseen": au_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))


# ═══════════ Main ═══════════
def main():
    cfg = FusionConfig(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="fusion_v1")
    p.add_argument("--load-ckpt", default="", help="pretrained encoder weights")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--unfreeze", type=int, default=cfg.unfreeze_layers)
    p.add_argument("--freeze-encoders", action="store_true",
                   help="freeze encoder + text_enc, only train cross-attn + fusion")
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.lr = args.lr
    cfg.batch_size = args.bs; cfg.unfreeze_layers = args.unfreeze
    train(cfg, args)


if __name__ == "__main__":
    main()

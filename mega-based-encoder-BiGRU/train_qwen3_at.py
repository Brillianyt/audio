"""AT Training with Qwen3-ASR audio_tower encoder + CharBiGRU text encoder.

Replaces Whisper with Qwen3-ASR's 24-layer, 1024-dim audio encoder,
matching it with the proven CharBiGRU64 text encoder for keyword detection.
"""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from safetensors.torch import load_file

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from config import PATHS


# ═══════════ Config ═══════════

class Config:
    embed_dim: int = 256
    sample_rate: int = 16000
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
    grad_accum: int = 1

    train_zip: str = ""; train_csv: str = ""
    dev_seen_zip: str = ""; dev_seen_csv: str = ""
    dev_unseen_zip: str = ""; dev_unseen_csv: str = ""

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        db = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(db, "dev")): db = os.path.join(db, "dev")
        self.dev_seen_zip = os.path.join(db, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(db, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(db, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(db, "dev_unseen", "dev_unseen_label.csv")
        self.model_path = os.path.join(r, "mega", "Qwen3-ASR-1.7B")


# ═══════════ Qwen3-ASR Audio Tower ═══════════

class Qwen3ASTransformerLayer(nn.Module):
    """Single transformer layer matching thinkter.audio_tower.layers.N structure."""
    def __init__(self, dim=1024, ff_dim=4096):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, 16, batch_first=True)
        self.final_layer_norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, dim)

    def forward(self, x):
        residual = x
        x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(x, x, x, need_weights=False)
        x = residual + x
        residual = x
        x = self.final_layer_norm(x)
        x = F.gelu(self.fc1(x))
        x = self.fc2(x)
        x = residual + x
        return x


class Qwen3ASREncoder(nn.Module):
    """Qwen3-ASR audio_tower encoder: conv frontend + 24 transformer layers + projection.

    Architecture (Qwen3-ASR 1.7B, 16kHz input):
      conv2d1 (1→480, k=3, s=2) → GELU 
      conv2d2 (480→480, k=3, s=2) → GELU
      conv2d3 (480→480, k=3, s=2) → GELU 
      conv_out (flattened → 1024)
      → 24× TransformerLayer (dim=1024, ff=4096, 16 heads)
      → ln_post → proj1 (1024→1024) → GELU → proj2 (1024→2048)
      → ASP Pooling → Linear(4096→256) → L2-norm

    Total: ~317M params. Conv frontend always frozen.
    """
    def __init__(self, model_path, embed_dim=256, unfreeze=2, max_sec=2.5):
        super().__init__()
        self.max_sec = max_sec
        self.embed_dim = embed_dim
        self.dim = 1024
        self.ff_dim = 4096
        self.n_layers = 24
        self.n_heads = 16

        # ── Load weights ──
        s1 = load_file(os.path.join(model_path, "model-00001-of-00002.safetensors"))
        s2 = load_file(os.path.join(model_path, "model-00002-of-00002.safetensors"))
        sd = {}
        for k, v in {**s1, **s2}.items():
            if k.startswith("thinker.audio_tower."):
                sd[k.replace("thinker.audio_tower.", "")] = v

        # ── Conv frontend ──
        self.conv2d1 = nn.Conv2d(1, 480, 3, stride=2, padding=1, bias=True)
        self.conv2d2 = nn.Conv2d(480, 480, 3, stride=2, padding=1, bias=True)
        self.conv2d3 = nn.Conv2d(480, 480, 3, stride=2, padding=1, bias=True)
        self.conv_out = nn.Linear(7680, 1024, bias=False)
        
        self.conv2d1.load_state_dict({"weight": sd["conv2d1.weight"], "bias": sd["conv2d1.bias"]})
        self.conv2d2.load_state_dict({"weight": sd["conv2d2.weight"], "bias": sd["conv2d2.bias"]})
        self.conv2d3.load_state_dict({"weight": sd["conv2d3.weight"], "bias": sd["conv2d3.bias"]})
        self.conv_out.load_state_dict({"weight": sd["conv_out.weight"]})
        for p in list(self.conv2d1.parameters()) + list(self.conv2d2.parameters()) + \
                 list(self.conv2d3.parameters()) + list(self.conv_out.parameters()):
            p.requires_grad = False

        # ── Transformer layers ──
        self.layers = nn.ModuleList([Qwen3ASTransformerLayer(self.dim, self.ff_dim) for _ in range(self.n_layers)])
        self.frozen = self.n_layers - unfreeze
        for i, layer in enumerate(self.layers):
            prefix = f"layers.{i}."
            # Map MultiheadAttention internal weights
            sd_layer = {
                "self_attn.in_proj_weight": torch.cat([
                    sd[f"{prefix}self_attn.q_proj.weight"],
                    sd[f"{prefix}self_attn.k_proj.weight"],
                    sd[f"{prefix}self_attn.v_proj.weight"],
                ], dim=0),
                "self_attn.in_proj_bias": torch.cat([
                    sd[f"{prefix}self_attn.q_proj.bias"],
                    sd[f"{prefix}self_attn.k_proj.bias"],
                    sd[f"{prefix}self_attn.v_proj.bias"],
                ], dim=0),
                "self_attn.out_proj.weight": sd[f"{prefix}self_attn.out_proj.weight"],
                "self_attn.out_proj.bias": sd[f"{prefix}self_attn.out_proj.bias"],
                "self_attn_layer_norm.weight": sd[f"{prefix}self_attn_layer_norm.weight"],
                "self_attn_layer_norm.bias": sd[f"{prefix}self_attn_layer_norm.bias"],
                "final_layer_norm.weight": sd[f"{prefix}final_layer_norm.weight"],
                "final_layer_norm.bias": sd[f"{prefix}final_layer_norm.bias"],
                "fc1.weight": sd[f"{prefix}fc1.weight"],
                "fc1.bias": sd[f"{prefix}fc1.bias"],
                "fc2.weight": sd[f"{prefix}fc2.weight"],
                "fc2.bias": sd[f"{prefix}fc2.bias"],
            }
            layer.load_state_dict(sd_layer, strict=True)
            for p in layer.parameters():
                p.requires_grad = (i >= self.frozen)

        # ── Output projection ──
        self.ln_post = nn.LayerNorm(self.dim)
        self.proj1 = nn.Linear(self.dim, self.dim)
        self.proj2 = nn.Linear(self.dim, self.dim * 2)
        self.ln_post.load_state_dict({"weight": sd["ln_post.weight"], "bias": sd["ln_post.bias"]})
        self.proj1.load_state_dict({"weight": sd["proj1.weight"], "bias": sd["proj1.bias"]})
        self.proj2.load_state_dict({"weight": sd["proj2.weight"], "bias": sd["proj2.bias"]})

        # ── ASP Pooling ──
        self.attn_linear = nn.Linear(self.dim * 2, (self.dim * 2) // 4)
        self.attn_w = nn.Parameter(torch.randn((self.dim * 2) // 4, 1))
        self.proj = nn.Linear(self.dim * 4, embed_dim)

        # Free memory
        del sd, s1, s2

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.conv2d1.eval(); self.conv2d2.eval(); self.conv2d3.eval()
            self.conv_out.eval()
            for i in range(self.frozen):
                self.layers[i].eval()
        return self

    def forward(self, wav):
        device = next(self.parameters()).device
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms:
            wav = wav[:, :ms]
        # Pad to multiple of 128 (conv stride 8 × reshape factor 16)
        T_in = wav.shape[-1]
        pad_to = ((T_in + 127) // 128) * 128
        if pad_to > T_in:
            wav = F.pad(wav, (0, pad_to - T_in))
        wav = wav.to(device)

        # ── Conv frontend (always no_grad) ──
        with torch.no_grad():
            x = wav.unsqueeze(1).unsqueeze(-1)  # (B, 1, T, 1) — Conv2d needs 4D
            x = F.gelu(self.conv2d1(x))
            x = F.gelu(self.conv2d2(x))
            x = F.gelu(self.conv2d3(x))
            # x: (B, 480, T', 1), where T' = T // 8
            # Squeeze freq dim, then reshape for Linear(7680, 1024)
            x = x.squeeze(-1)  # (B, 480, T')
            B, C, Tf = x.shape
            x = x.reshape(B, C * 16, Tf // 16)  # (B, 7680, T_out)
            x = x.transpose(1, 2)  # (B, T_out, 7680) — Linear expects last dim as input
            x = self.conv_out(x)    # (B, T_out, 1024)

        # ── Transformer layers ──
        for i, layer in enumerate(self.layers):
            if i < self.frozen:
                with torch.no_grad():
                    x = layer(x)
            else:
                x = layer(x)

        # ── Output projection ──
        x = self.ln_post(x)
        x = F.gelu(self.proj1(x))
        x = self.proj2(x)  # (B, T, 2048)

        # ── ASP Pooling ──
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x * aw).sum(1)
        sg = ((x ** 2 * aw).sum(1) - mu ** 2).clamp(1e-5).sqrt()
        pooled = torch.cat([mu, sg], dim=-1)  # (B, 4096)
        return F.normalize(self.proj(pooled), dim=-1)  # (B, embed_dim)

    @property
    def sample_rate(self):
        return 16000


# ═══════════ CharBiGRU64 Text Encoder (from train_dual.py) ═══════════

class CharBiGRU64(nn.Module):
    """CharBiGRU_64: original AT v2 architecture (64d emb, 1-layer GRU)."""
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True, num_layers=1)
        self.proj = nn.Linear(128, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts:
            return torch.zeros(0, 256, device=device), None
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i, j] = max(0, min(27, ord(c) - 97))
        x = self.char_emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1), None


# ═══════════ AudioText Model ═══════════

class AudioTextModel(nn.Module):
    """Qwen3-ASR encoder + CharBiGRU64 with uncertainty-weighted fusion."""
    def __init__(self, model_path, embed_dim=256, unfreeze=2, max_sec=2.5):
        super().__init__()
        self.encoder = Qwen3ASREncoder(model_path, embed_dim, unfreeze, max_sec)
        self.text_enc = CharBiGRU64(embed_dim)
        self.log_var_a = nn.Parameter(torch.tensor(0.0))
        self.log_var_t = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, e, texts, q=None):
        ea = self.encoder(e)
        et, _ = self.text_enc(texts)
        cos_ae = (ea * et).sum(-1)
        if q is None:
            return cos_ae, cos_ae * 8.0, ea, et
        eq = self.encoder(q)
        sim_a = (ea * eq).sum(-1)
        sim_t = (et * eq).sum(-1)
        w_a = torch.exp(-self.log_var_a)
        w_t = torch.exp(-self.log_var_t)
        norm_diff = (ea.norm(dim=-1) - et.norm(dim=-1)).abs()
        score = w_a * sim_a + w_t * sim_t + 0.05 * (1.0 - norm_diff) + self.bias
        return cos_ae, score, ea, et


# ═══════════ Data ═══════════

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.zip_path = zip_path; self.cfg = cfg
        self._zip_cache = {}

    def _get_zip(self):
        pid = os.getpid()
        if pid not in self._zip_cache:
            self._zip_cache[pid] = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_cache[pid]

    def __len__(self):
        return len(self.pairs)

    def _read(self, pid, role):
        data = self._get_zip().read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        wav = torch.from_numpy(wav).float()
        pad = int(0.5 * 16000); wav = F.pad(wav, (pad, pad))
        return wav

    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        e = self._read(eid, "enroll"); q = self._read(qid, "query")
        label = p.get("label", -1)
        txt = p.get("enroll_txt", "").lower()
        return e, q, label, txt, pid


def collate_text(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, ls, txts, ids = [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, ml - e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml - q.shape[-1])) if q.shape[-1] < ml else q)
        ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls, dtype=torch.float32), txts, ids


def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt", ""),
                         "query_txt": r.get("query_txt", "")})
    return rows


# ═══════════ Training ═══════════

def train(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data ──
    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_iter1.json",
               "baseline/hard_neg_at_v8.json",
               "train/self_paired.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp):
            with open(hp) as f:
                all_pairs += json.load(f)
    rng.shuffle(all_pairs)
    print(f"[qwen3_at] {len(all_pairs)} total pairs")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    # ── Semi-dedup sampling ──
    pos_groups = {}
    for i, p in enumerate(pos_pairs):
        k = (p["enroll_txt"].lower(), p["query_txt"].lower())
        pos_groups.setdefault(k, []).append(i)
    neg_groups = {}
    for i, p in enumerate(neg_pairs):
        k = (p["enroll_txt"].lower(), p["query_txt"].lower())
        neg_groups.setdefault(k, []).append(i)
    pos_keys = list(pos_groups.keys())
    neg_keys = list(neg_groups.keys())
    rng.shuffle(pos_keys); rng.shuffle(neg_keys)
    print(f"  semidedup: {len(pos_keys)} pos keys, {len(neg_keys)} neg keys")

    def get_loader(ep):
        n_pos = min(30000, len(pos_keys) * 5)
        n_neg = min(150000, len(neg_keys))
        subset = []
        pp = (ep * n_pos) % len(pos_keys) if len(pos_keys) > 0 else 0
        for i in range(n_pos):
            k = pos_keys[(pp + i) % len(pos_keys)]
            idx = rng.choice(pos_groups[k])
            subset.append(pos_pairs[idx])
        np_start = (ep * n_neg) % len(neg_keys) if len(neg_keys) > 0 else 0
        for i in range(n_neg):
            k = neg_keys[(np_start + i) % len(neg_keys)]
            idx = rng.choice(neg_groups[k])
            subset.append(neg_pairs[idx])
        np.random.shuffle(subset)
        ds = PairDataset(subset, cfg.train_zip, cfg)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_text,
                          pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size * 2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)

    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    # ── Model ──
    model = AudioTextModel(cfg.model_path, cfg.embed_dim, cfg.unfreeze_layers, cfg.max_audio_sec).to(device)

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed from {os.path.basename(fp)} ep{start_ep}")
                break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    # ── Optimizer (encoder lr/10, text_enc + fusion lr=1e-3) ──
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and "encoder" in n and "proj" not in n]
    proj_p = [p for n, p in model.named_parameters() if p.requires_grad and "encoder.proj" in n]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "encoder" not in n]
    opt = torch.optim.AdamW([
        {"params": enc_p, "lr": cfg.lr / 10},
        {"params": proj_p, "lr": cfg.lr},
        {"params": oth_p, "lr": 1e-3},
    ], weight_decay=1e-4)

    total_steps = cfg.epochs * 100000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)

    def lr_lm(s):
        if s < warmup_steps:
            return s / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / (total_steps - warmup_steps)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lm)
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    stats_file = os.path.join(out_dir, "cos_dist.jsonl")

    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train()
        ts = time.time(); ls, n = 0.0, 0
        cs_pos, cs_neg, cs_n = 0.0, 0.0, 0
        cs_pos_std, cs_neg_std = 0.0, 0.0
        all_cos_pos, all_cos_neg = [], []
        opt.zero_grad()

        for bi, (e, q, y, txts, _) in enumerate(loader):
            e, q, y = e.to(device), q.to(device), y.to(device)

            # ── Noise augmentation (tiered) ──
            tier = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            if tier == 0: snr = np.random.uniform(0, 5)
            elif tier == 1: snr = np.random.uniform(-5, 0)
            else: snr = np.random.uniform(-10, -5)
            nl = 10 ** (-snr / 20)
            e = e + nl * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + nl * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                cos_ae, score, ea, et = model(e, txts, q)

                # ── BCE + online hard mining ──
                p = torch.sigmoid(score)
                hard = ((y == 0) & (p > 0.5)) | ((y == 1) & (p < 0.3))
                weight = torch.where(hard, 2.0, 0.3)
                loss = F.binary_cross_entropy_with_logits(score, y, reduction='none')
                loss = (loss * weight).mean()

                # ── FN booster ──
                pos = (y == 1)
                if pos.any():
                    hard_pos = cos_ae[pos] < 0.5
                    if hard_pos.any():
                        loss = loss + 0.05 * ((1.0 - cos_ae[pos][hard_pos]) ** 2).mean()

                # ── Monitor ──
                neg = (y == 0)
                _cp = cos_ae[pos]; _cn = cos_ae[neg]
                c_pv = _cp.mean().item() if pos.any() else 0.0
                c_nv = _cn.mean().item() if neg.any() else 0.0

            # ── Backward with grad accum ──
            loss = loss / cfg.grad_accum
            scaler.scale(loss).backward()

            if (bi + 1) % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt); scaler.update(); sched.step()
                opt.zero_grad()

            ls += loss.item() * cfg.grad_accum; n += 1
            cs_pos += c_pv; cs_neg += c_nv; cs_n += 1
            cs_pos_std += _cp.std().item() if pos.any() and _cp.numel() > 1 else 0
            cs_neg_std += _cn.std().item() if neg.any() and _cn.numel() > 1 else 0
            all_cos_pos.append(_cp.detach().cpu())
            all_cos_neg.append(_cn.detach().cpu())

            if cs_n % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cs_n} loss={ls/n:.3f} cos+={cs_pos/cs_n:.3f} cos-={cs_neg/cs_n:.3f} "
                      f"gap={cs_pos/cs_n - cs_neg/cs_n:.3f}")

            del cos_ae, score, ea, et, e, q, y, loss, _cp, _cn

        # ── Eval ──
        @torch.no_grad()
        def ev(ld):
            model.eval()
            ps, ls_list, ids, all_cs = [], [], [], []
            for e, q, y, txts, id_ in ld:
                e, q = e.to(device), q.to(device)
                cos, score, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(score).cpu().numpy())
                ls_list.append(y.numpy()); ids.extend(id_)
                all_cs.append(cos.cpu().numpy())
            return (roc_auc_score(np.concatenate(ls_list), np.concatenate(ps)),
                    np.concatenate(ps), np.concatenate(ls_list), ids,
                    np.concatenate(all_cs))

        as_, ps_s, ls_s, ids_s, css_s = ev(dv_s)
        au_, ps_u, ls_u, ids_u, css_u = ev(dv_u)

        # ── Cosine distribution diagnostics ──
        all_cp = torch.cat(all_cos_pos).numpy()
        all_cn = torch.cat(all_cos_neg).numpy()

        def _hist(arr, bins=20):
            if len(arr) == 0: return [0] * bins
            h, _ = np.histogram(arr, bins=bins, range=(-0.5, 1.0))
            return h.astype(int).tolist()

        def _pct(arr, lo, hi):
            if len(arr) == 0: return 0.0
            return float(((arr >= lo) & (arr < hi)).mean())

        dist = {
            "epoch": ep,
            "cos_pos": {"mean": float(all_cp.mean()), "std": float(all_cp.std()),
                        "p05": float(np.percentile(all_cp, 5)),
                        "p50": float(np.percentile(all_cp, 50)),
                        "p95": float(np.percentile(all_cp, 95)),
                        "n": len(all_cp), "hist": _hist(all_cp)},
            "cos_neg": {"mean": float(all_cn.mean()), "std": float(all_cn.std()),
                        "p05": float(np.percentile(all_cn, 5)),
                        "p50": float(np.percentile(all_cn, 50)),
                        "p95": float(np.percentile(all_cn, 95)),
                        "n": len(all_cn), "hist": _hist(all_cn)},
            "overlap_gt_0": _pct(all_cn, 0.0, 1.0),
            "overlap_gt_pos_med": _pct(all_cn, float(np.median(all_cp)), 1.0),
        }
        with open(stats_file, "a") as f:
            f.write(json.dumps(dist) + "\n")

        print(f"[qwen3_at ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={ls/n:.4f} cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} "
              f"cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} "
              f"overlap(>0)={dist['overlap_gt_0']:.3f} ({time.time() - ts:.0f}s)")
        print(f"  cos+ [p5={dist['cos_pos']['p05']:.3f} "
              f"p50={dist['cos_pos']['p50']:.3f} p95={dist['cos_pos']['p95']:.3f}]")
        print(f"  cos- [p5={dist['cos_neg']['p05']:.3f} "
              f"p50={dist['cos_neg']['p50']:.3f} p95={dist['cos_neg']['p95']:.3f}]")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_,
                    "auc_seen": as_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))


# ═══════════ Main ═══════════

def main():
    cfg = Config(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="qwen3_at_v1")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--unfreeze", type=int, default=cfg.unfreeze_layers)
    p.add_argument("--pos-weight", type=float, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--grad-accum", type=int, default=cfg.grad_accum)
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    cfg.unfreeze_layers = args.unfreeze; cfg.grad_accum = args.grad_accum
    if args.pos_weight is not None: cfg.pos_weight = args.pos_weight
    train(cfg, args)


if __name__ == "__main__":
    main()

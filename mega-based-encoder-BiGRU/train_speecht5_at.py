"""AT Training with SpeechT5 encoder + CharBiGRU text encoder."""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import SpeechT5ForSpeechToText

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "baseline"))
from config import PATHS


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
        self.model_path = os.path.join(r, "pretrained_models", "speecht5_asr")
        db = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(db, "dev")): db = os.path.join(db, "dev")
        self.dev_seen_zip = os.path.join(db, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(db, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(db, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(db, "dev_unseen", "dev_unseen_label.csv")


# ═══════════ SpeechT5 Encoder ═══════════

class SpeechT5Encoder(nn.Module):
    def __init__(self, model_path, embed_dim=256, unfreeze=2, max_sec=2.5):
        super().__init__()
        base = SpeechT5ForSpeechToText.from_pretrained(model_path)
        self.encoder = base.speecht5.encoder  # SpeechT5EncoderWithSpeechPrenet
        self.prenet = base.speecht5.encoder.prenet
        self.dim = base.config.hidden_size  # 768
        self.max_sec = max_sec
        del base

        # Freeze prenet, unfreeze last N wrapped_encoder layers
        for p in self.prenet.parameters():
            p.requires_grad = False
        self.wrapped = self.encoder.wrapped_encoder
        total = len(self.wrapped.layers)
        self.frozen = total - unfreeze
        for p in self.wrapped.parameters():
            p.requires_grad = False
        for i in range(self.frozen, total):
            for p in self.wrapped.layers[i].parameters():
                p.requires_grad = True

        # ASP Pooling
        self.attn_linear = nn.Linear(self.dim, self.dim // 4)
        self.attn_w = nn.Parameter(torch.randn(self.dim // 4, 1))
        self.proj = nn.Linear(self.dim * 2, embed_dim)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.prenet.eval()
            for i in range(self.frozen):
                self.wrapped.layers[i].eval()
        return self

    def forward(self, wav):
        device = next(self.parameters()).device
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms:
            wav = wav[:, :ms]
        wav = wav.to(device)

        # SpeechT5EncoderWithSpeechPrenet expects raw waveform
        out = self.encoder(wav)
        x = out.last_hidden_state  # (B, T, 768)

        # ASP Pooling
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x * aw).sum(1)
        sg = ((x ** 2 * aw).sum(1) - mu ** 2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu, sg], -1)), dim=-1)

    @property
    def sample_rate(self):
        return 16000


# ═══════════ CharBiGRU64 ═══════════

class CharBiGRU64(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True, num_layers=1)
        self.proj = nn.Linear(128, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0, 256, device=device), None
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
    def __init__(self, model_path, embed_dim=256, unfreeze=2, max_sec=2.5):
        super().__init__()
        self.encoder = SpeechT5Encoder(model_path, embed_dim, unfreeze, max_sec)
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

    def __len__(self): return len(self.pairs)

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
        txt = p.get("enroll_txt", "").lower()
        return e, q, p.get("label", -1), txt, pid


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
    device = "cuda"; torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_iter1.json", "baseline/hard_neg_at_v8.json", "train/self_paired.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp): all_pairs += json.load(open(hp))
    rng.shuffle(all_pairs)
    print(f"[speecht5_at] {len(all_pairs)} pairs")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    pos_groups = {}
    for i, p in enumerate(pos_pairs):
        k = (p["enroll_txt"].lower(), p["query_txt"].lower())
        pos_groups.setdefault(k, []).append(i)
    neg_groups = {}
    for i, p in enumerate(neg_pairs):
        k = (p["enroll_txt"].lower(), p["query_txt"].lower())
        neg_groups.setdefault(k, []).append(i)
    pos_keys = list(pos_groups.keys()); neg_keys = list(neg_groups.keys())
    rng.shuffle(pos_keys); rng.shuffle(neg_keys)
    print(f"  semidedup: {len(pos_keys)} pos, {len(neg_keys)} neg keys")

    def get_loader(ep):
        n_pos = min(30000, len(pos_keys) * 5)
        n_neg = min(150000, len(neg_keys))
        subset = []
        pp = (ep * n_pos) % len(pos_keys) if pos_keys else 0
        for i in range(n_pos):
            k = pos_keys[(pp + i) % len(pos_keys)]
            subset.append(pos_pairs[rng.choice(pos_groups[k])])
        np_start = (ep * n_neg) % len(neg_keys) if neg_keys else 0
        for i in range(n_neg):
            k = neg_keys[(np_start + i) % len(neg_keys)]
            subset.append(neg_pairs[rng.choice(neg_groups[k])])
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

    model = AudioTextModel(cfg.model_path, cfg.embed_dim, cfg.unfreeze_layers, cfg.max_audio_sec).to(device)
    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed ep{start_ep}"); break
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    # Optimizer: encoder lr/10, text_enc+fusion lr=1e-3
    enc_p = [p for n, p in model.named_parameters() if p.requires_grad and "encoder" in n and "proj" not in n and "attn" not in n]
    proj_p = [p for n, p in model.named_parameters() if p.requires_grad and ("encoder.proj" in n or "encoder.attn" in n)]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "encoder" not in n]
    opt = torch.optim.AdamW([
        {"params": enc_p, "lr": cfg.lr / 10},
        {"params": proj_p, "lr": cfg.lr},
        {"params": oth_p, "lr": 1e-3},
    ], weight_decay=1e-4)

    total_steps = cfg.epochs * 100000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lm(s):
        if s < warmup_steps: return s / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (s - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lm)
    scaler = torch.amp.GradScaler('cuda')

    stats_file = os.path.join(out_dir, "cos_dist.jsonl")
    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls, n = 0.0, 0
        cs_pos, cs_neg, cs_n = 0.0, 0.0, 0
        cs_pos_std, cs_neg_std = 0.0, 0.0
        all_cos_pos, all_cos_neg = [], []
        opt.zero_grad()

        for bi, (e, q, y, txts, _) in enumerate(loader):
            e, q, y = e.to(device), q.to(device), y.to(device)
            # Noise augmentation (tiered)
            tier = np.random.choice([0, 1, 2], p=[0.7, 0.2, 0.1])
            snr = np.random.uniform(*{0: (0, 5), 1: (-5, 0), 2: (-10, -5)}[tier])
            nl = 10 ** (-snr / 20)
            e = e + nl * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + nl * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.amp.autocast('cuda'):
                cos_ae, score, ea, et = model(e, txts, q)
                p = torch.sigmoid(score)
                hard = ((y == 0) & (p > 0.5)) | ((y == 1) & (p < 0.3))
                weight = torch.where(hard, 2.0, 0.3)
                loss = F.binary_cross_entropy_with_logits(score, y, reduction='none')
                loss = (loss * weight).mean()
                pos = (y == 1); neg = (y == 0)
                if pos.any():
                    hard_pos = cos_ae[pos] < 0.5
                    if hard_pos.any():
                        loss = loss + 0.05 * ((1.0 - cos_ae[pos][hard_pos]) ** 2).mean()
                _cp = cos_ae[pos]; _cn = cos_ae[neg]
                c_pv = _cp.mean().item() if pos.any() else 0.0
                c_nv = _cn.mean().item() if neg.any() else 0.0

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()
            opt.zero_grad()

            ls += loss.item(); n += 1
            cs_pos += c_pv; cs_neg += c_nv; cs_n += 1
            cs_pos_std += _cp.std().item() if pos.any() and _cp.numel() > 1 else 0
            cs_neg_std += _cn.std().item() if neg.any() and _cn.numel() > 1 else 0
            all_cos_pos.append(_cp.detach().cpu()); all_cos_neg.append(_cn.detach().cpu())

            if cs_n % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cs_n} loss={ls/n:.3f} cos+={cs_pos/cs_n:.3f} cos-={cs_neg/cs_n:.3f} gap={cs_pos/cs_n-cs_neg/cs_n:.3f}")
            del cos_ae, score, ea, et, e, q, y, loss, _cp, _cn

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls_list, all_cs = [], [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                cos, score, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(score).cpu().numpy())
                ls_list.append(y.numpy())
                all_cs.append(cos.cpu().numpy())
            return roc_auc_score(np.concatenate(ls_list), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        all_cp = torch.cat(all_cos_pos).numpy(); all_cn = torch.cat(all_cos_neg).numpy()
        dist = {"epoch": ep,
                "cos_pos": {"mean": float(all_cp.mean()), "std": float(all_cp.std()),
                            "p05": float(np.percentile(all_cp, 5)), "p50": float(np.percentile(all_cp, 50)),
                            "p95": float(np.percentile(all_cp, 95))},
                "cos_neg": {"mean": float(all_cn.mean()), "std": float(all_cn.std()),
                            "p05": float(np.percentile(all_cn, 5)), "p50": float(np.percentile(all_cn, 50)),
                            "p95": float(np.percentile(all_cn, 95))},
                "overlap": float((all_cn > 0).mean())}
        with open(stats_file, "a") as f: f.write(json.dumps(dist) + "\n")

        print(f"[speecht5 ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={ls/n:.4f} "
              f"cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} "
              f"overlap={dist['overlap']:.3f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))


def main():
    cfg = Config(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="speecht5_at_v1")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--unfreeze", type=int, default=cfg.unfreeze_layers)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.lr = args.lr
    cfg.batch_size = args.bs; cfg.unfreeze_layers = args.unfreeze
    train(cfg, args)


if __name__ == "__main__":
    main()

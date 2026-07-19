"""WavLM KWS — Minimal Viable Version.

与 whisper_v3 完全相同的架构路径:
  WavLM → ASP Pooling → cosine similarity + BCE → posterior

不做融合、不做 text 编码、不做音素辅助、不做 margin loss。
只验证一个假设: WavLM 能不能替代 Whisper 做这个任务。
"""
from __future__ import annotations

import argparse, csv, gc, json, math, os, time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import PATHS


# ═══════════════════════ Config ═══════════════════════

class Config:
    wavlm_model: str = "microsoft/wavlm-base-plus"
    wavlm_local: str = ""
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5   # KWS关键词<1s, 1.5s足够覆盖
    unfreeze_layers: int = 3
    use_weighted_sum: bool = True

    epochs: int = 20
    lr: float = 3e-4
    batch_size: int = 64
    grad_accum: int = 2
    pos_weight: float = 5.0
    num_workers: int = 4
    seed: int = 42
    log_every: int = 20
    subset: int = 500000

    pk_P: int = 48           # 更大batch → 更多样本/前向 → 更快
    pk_K: int = 4
    pk_batches_per_epoch: int = 300
    warmup_steps: int = 200

    train_zip: str = ""
    train_csv: str = ""
    dev_seen_zip: str = ""
    dev_seen_csv: str = ""
    dev_unseen_zip: str = ""
    dev_unseen_csv: str = ""
    eval_seen_zip: str = ""
    eval_seen_csv: str = ""
    eval_unseen_zip: str = ""
    eval_unseen_csv: str = ""

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        subset_zip = os.path.join(r, "train_subset", "wav.zip")
        subset_csv = os.path.join(r, "train_subset", "train_label.csv")
        if os.path.isfile(subset_zip):
            self.train_zip = subset_zip
            self.train_csv = subset_csv
        dev_base = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(dev_base, "dev")):
            dev_base = os.path.join(dev_base, "dev")
        self.dev_seen_zip = os.path.join(dev_base, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(dev_base, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(dev_base, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(dev_base, "dev_unseen", "dev_unseen_label.csv")
        self.eval_seen_zip = os.path.join(r, "eval", "eval_seen", "wav.zip")
        self.eval_seen_csv = os.path.join(r, "evalcsv_without_label", "eval_seen_without_label.csv")
        self.eval_unseen_zip = os.path.join(r, "eval", "eval_unseen", "wav.zip")
        self.eval_unseen_csv = os.path.join(r, "evalcsv_without_label", "eval_unseen_without_label.csv")
        hub_wavlm = os.path.join(r, "hub", "models--microsoft--wavlm-base-plus", "snapshots")
        if os.path.isdir(hub_wavlm):
            for snap in sorted(os.listdir(hub_wavlm), reverse=True):
                if os.path.isfile(os.path.join(hub_wavlm, snap, "config.json")):
                    self.wavlm_local = os.path.join(hub_wavlm, snap)
                    break


# ═══════════════════════ WavLM Encoder ═══════════════════════

class WavLMEncoder(nn.Module):
    def __init__(self, model_name_or_path, embed_dim=256, unfreeze_layers=3,
                 max_audio_sec=3.0, use_weighted_sum=True):
        super().__init__()
        from transformers import WavLMModel
        self.wavlm = WavLMModel.from_pretrained(model_name_or_path)
        self.wavlm.eval()
        self.hidden_size = self.wavlm.config.hidden_size
        self.num_layers = self.wavlm.config.num_hidden_layers
        self.max_audio_sec = max_audio_sec
        self.use_weighted_sum = use_weighted_sum

        for p in self.wavlm.feature_extractor.parameters():
            p.requires_grad = False
        total_blocks = len(self.wavlm.encoder.layers)
        self.frozen_blocks = total_blocks - unfreeze_layers
        for p in self.wavlm.encoder.parameters():
            p.requires_grad = False
        for i in range(self.frozen_blocks, total_blocks):
            for p in self.wavlm.encoder.layers[i].parameters():
                p.requires_grad = True

        tp = sum(p.requires_grad for p in self.wavlm.parameters())
        fp = sum(not p.requires_grad for p in self.wavlm.parameters())
        print(f"  [wavlm] frozen={fp:,} trainable={tp:,}")

        if use_weighted_sum:
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        self.output_norm = nn.LayerNorm(self.hidden_size)

        # ASP
        self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.attn_w = nn.Parameter(torch.randn(self.hidden_size // 4, 1))
        self.proj = nn.Linear(self.hidden_size * 2, embed_dim)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.wavlm.feature_extractor.eval()
            for i in range(self.frozen_blocks):
                self.wavlm.encoder.layers[i].eval()
        return self

    def forward(self, wav):
        device = next(self.wavlm.parameters()).device
        max_s = int(self.max_audio_sec * 16000)
        if wav.shape[-1] > max_s:
            wav = wav[:, :max_s]
        wav = wav.to(device)

        out = self.wavlm(wav, output_hidden_states=True, return_dict=True)
        all_hidden = out.hidden_states[1:]

        if self.use_weighted_sum and hasattr(self, 'layer_weights') and self.layer_weights is not None:
            stacked = torch.stack(all_hidden, dim=1)
            w = F.softmax(self.layer_weights, dim=0)
            x = (stacked * w.view(1, -1, 1, 1)).sum(dim=1)
        else:
            x = all_hidden[-1]

        x = self.output_norm(x)
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x * aw).sum(dim=1)
        sigma = ((x ** 2 * aw).sum(dim=1) - mu ** 2).clamp(min=1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu, sigma], dim=-1)), dim=-1)

    @property
    def sample_rate(self):
        return 16000


class KWSModel(nn.Module):
    def __init__(self, wavlm_path, embed_dim=256, unfreeze_layers=3,
                 max_audio_sec=3.0, use_weighted_sum=True):
        super().__init__()
        self.encoder = WavLMEncoder(wavlm_path, embed_dim, unfreeze_layers,
                                     max_audio_sec, use_weighted_sum)
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, enroll_wav, query_wav):
        e = self.encoder(enroll_wav)
        q = self.encoder(query_wav)
        sim = (e * q).sum(dim=-1)
        return self.scale * sim + self.bias, e, q


# ═══════════════════════ Data ═══════════════════════

def load_pairs(csv_path, with_label=True):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {"id": r["id"]}
            if with_label:
                item["label"] = int(r["label"])
            item["enroll_txt"] = r.get("enroll_txt", "")
            item["query_txt"] = r.get("query_txt", "")
            rows.append(item)
    return rows


class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg, inference=False):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference

    def __len__(self):
        return len(self.pairs)

    def _get_zip(self):
        import zipfile as zf
        pid = os.getpid()
        if not hasattr(self, '_zip_cache'):
            self._zip_cache = {}
        if pid not in self._zip_cache:
            self._zip_cache[pid] = zf.ZipFile(self.zip_path, "r")
        return self._zip_cache[pid]

    def _read(self, pid, role):
        import io, soundfile as sf
        data = self._get_zip().read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != self.cfg.sample_rate:
            import torchaudio
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, self.cfg.sample_rate
            ).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        max_s = int(self.cfg.max_audio_sec * self.cfg.sample_rate)
        if len(wav) > max_s:
            wav = wav[:max_s]
        return torch.from_numpy(wav).float()

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pid = p["id"]
        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)
        e = self._read(eid, "enroll")
        q = self._read(qid, "query")
        label = -1 if self.inference else p["label"]
        txt_e = p.get("enroll_txt", "").lower() if not self.inference else ""
        return e, q, label, pid, txt_e


def collate(batch):
    max_len = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, ls, ids, txts = [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, max_len - e.shape[-1])) if e.shape[-1] < max_len else e)
        qs.append(F.pad(q, (0, max_len - q.shape[-1])) if q.shape[-1] < max_len else q)
        ls.append(b[2])
        ids.append(b[3])
        txts.append(b[4])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls, dtype=torch.float32), ids, txts


# ═══════════════════════ PK Sampler ═══════════════════════

class PKSampler:
    def __init__(self, word_to_indices, P=32, K=4, batches_per_epoch=300):
        self.word_to_indices = word_to_indices
        self.P = P
        self.K = K
        self.batches_per_epoch = batches_per_epoch
        self.valid = [w for w, idxs in word_to_indices.items() if len(idxs) >= K]

    def __iter__(self):
        P = min(self.P, len(self.valid))
        if P == 0:
            return iter([])
        batches = []
        for _ in range(self.batches_per_epoch):
            words = list(np.random.choice(self.valid, P, replace=False))
            indices = []
            for w in words:
                indices.extend(np.random.choice(self.word_to_indices[w], self.K, replace=False).tolist())
            np.random.shuffle(indices)
            batches.append(indices)
        return iter(batches)

    def __len__(self):
        return self.batches_per_epoch


# ═══════════════════════ Angular Prototypical Loss ═══════════════════════

class AngularPrototypicalLoss(nn.Module):
    """Episodic training: support→prototype→query classification.
    Same as whisper_v3. Key mechanism for unseen word generalization."""
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embed, word_texts):
        device = embed.device
        unique = list(set(word_texts))
        P = len(unique)
        if P < 2:
            return embed.new_zeros(())
        w2i = {w: i for i, w in enumerate(unique)}
        labels = torch.tensor([w2i[w] for w in word_texts], device=device)
        qmask = torch.zeros(len(word_texts), dtype=torch.bool, device=device)
        for w in unique:
            idx = torch.tensor([i for i, t in enumerate(word_texts) if t==w], device=device)
            nq = max(1, len(idx)//3)
            qmask[idx[torch.randperm(len(idx), device=device)[:nq]]] = True
        smask = ~qmask
        proto = torch.zeros(P, embed.shape[-1], device=device)
        for i, w in enumerate(unique):
            idx = torch.tensor([j for j, t in enumerate(word_texts) if t==w], device=device)
            proto[i] = embed[idx][smask[idx]].mean(dim=0) if smask[idx].sum()>0 else embed[idx].mean(dim=0)
        proto = F.normalize(proto, dim=1)
        logits = self.w.clamp(min=1e-6) * torch.matmul(embed[qmask], proto.T) + self.b
        return F.cross_entropy(logits, labels[qmask])


# ═══════════════════════ Scheduler ═══════════════════════

class CosineWarmup:
    def __init__(self, opt, warmup, total, min_ratio=0.01):
        self.opt = opt
        self.warmup = warmup
        self.total = total
        self.min_ratio = min_ratio
        self.base_lrs = [pg["lr"] for pg in opt.param_groups]
        self.step_count = 0

    def step(self):
        self.step_count += 1
        if self.step_count <= self.warmup:
            p = self.step_count / max(1, self.warmup)
        else:
            p = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            p = self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * p))
        for i, pg in enumerate(self.opt.param_groups):
            pg["lr"] = self.base_lrs[i] * p

    def get_lr(self):
        return [pg["lr"] for pg in self.opt.param_groups]


# ═══════════════════════ EMA ═══════════════════════

class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[n] = p.data.clone()

    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[n] = p.data.clone()
                p.data.copy_(self.shadow[n])

    def restore(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[n])
        self.backup.clear()


# ═══════════════════════ Train ═══════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for e, q, y, _, _ in loader:
        e, q = e.to(device), q.to(device)
        logit, _, _ = model(e, q)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


def train(args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    all_pairs = load_pairs(cfg.train_csv)
    n = min(cfg.subset, len(all_pairs))
    rng = np.random.default_rng(cfg.seed)
    train_pairs = [all_pairs[i] for i in rng.permutation(len(all_pairs))[:n]]

    # Hard negatives
    hn_files = [
        os.path.join(PATHS.root, "baseline", "hard_neg_whisper.json"),
        os.path.join(PATHS.root, "baseline", "hard_neg_wavlm.json"),
    ]
    for hn_file in hn_files:
        if os.path.isfile(hn_file):
            with open(hn_file, encoding="utf-8") as f:
                hn = json.load(f)
            train_pairs = train_pairs + hn
            rng.shuffle(train_pairs)
            print(f"  + hard_neg: {len(hn)} pairs (total={len(train_pairs)})")
            break

    train_ds = PairDataset(train_pairs, cfg.train_zip, cfg)

    # PK sampler
    word_to_idx = defaultdict(list)
    for i, p in enumerate(train_pairs):
        w = p.get("enroll_txt", "").lower()
        if not w:
            continue
        word_to_idx[w].append(i)
    pos_words = {p["enroll_txt"].lower() for p in train_pairs
                 if p["label"] == 1 and p.get("enroll_txt")}
    word_to_idx = {w: idxs for w, idxs in word_to_idx.items()
                   if w in pos_words and len(idxs) >= cfg.pk_K}
    print(f"  pk words: {len(word_to_idx)} (≥{cfg.pk_K})")

    sampler = PKSampler(word_to_idx, cfg.pk_P, cfg.pk_K, cfg.pk_batches_per_epoch)
    train_loader = DataLoader(train_ds, batch_sampler=sampler,
                              num_workers=cfg.num_workers, collate_fn=collate,
                              pin_memory=True, prefetch_factor=2,
                              persistent_workers=True)

    opt_steps = cfg.pk_batches_per_epoch // cfg.grad_accum
    total_steps = args.epochs * opt_steps
    print(f"  batches/epoch={len(train_loader)}, opt_steps/epoch={opt_steps}, total={total_steps}")

    # Dev
    def dev_ld(z, c):
        ds = PairDataset(load_pairs(c), z, cfg)
        return DataLoader(ds, batch_size=cfg.batch_size, num_workers=0,
                          collate_fn=collate, shuffle=False, pin_memory=True)
    dev_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dev_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    # Model
    wp = cfg.wavlm_local or cfg.wavlm_model
    print(f"[model] {wp}")
    model = KWSModel(wp, cfg.embed_dim, cfg.unfreeze_layers,
                     cfg.max_audio_sec, cfg.use_weighted_sum).to(device)

    best, best_ep = -1.0, 0
    if args.resume:
        latest = os.path.join(out_dir, "latest.pt")
        if os.path.isfile(latest):
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            best = ckpt.get("best_unseen", ckpt.get("auc", -1.0))
            print(f"  [resume] loaded, best_unseen={best:.4f}")

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    # LR groups
    wavlm_p, other_p = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.wavlm" in n:
            wavlm_p.append(p)
        else:
            other_p.append(p)
    opt = torch.optim.AdamW([
        {"params": wavlm_p, "lr": cfg.lr / 5, "weight_decay": 1e-4},
        {"params": other_p, "lr": cfg.lr, "weight_decay": 1e-4},
    ])
    scheduler = CosineWarmup(opt, cfg.warmup_steps, total_steps)
    ema = EMA(model)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    crit_proto = AngularPrototypicalLoss()
    lambda_proto = 0.08

    if hasattr(model.encoder.wavlm, "gradient_checkpointing_enable"):
        model.encoder.wavlm.gradient_checkpointing_enable()

    # Train
    step = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()
        model.train()
        t_ep = time.time()
        loss_sum, n_batch = {"bce": 0.0, "proto": 0.0}, 0

        for it, (e, q, y, _, txts) in enumerate(train_loader, 1):
            e, q, y = e.to(device), q.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logit, e_emb, q_emb = model(e, q)
                loss_bce = crit(logit, y)
                emb_cat = torch.cat([e_emb, q_emb])
                txt_cat = txts + txts
                loss_proto = crit_proto(emb_cat, txt_cat)
                loss = (loss_bce + lambda_proto * loss_proto) / cfg.grad_accum

            scaler.scale(loss).backward()
            loss_sum["bce"] += loss_bce.item()
            loss_sum["proto"] += loss_proto.item()
            n_batch += 1

            if it % cfg.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                scheduler.step()
                ema.update()
                step += 1

            del logit, e_emb, q_emb, e, q, y, loss, loss_bce, loss_proto

            if it % 200 == 0:
                torch.save({"model": model.state_dict(), "best_unseen": best},
                           os.path.join(out_dir, "latest.pt"))

            if it % cfg.log_every == 0:
                lrs = scheduler.get_lr()
                print(f"  ep{ep} {it}/{len(train_loader)} "
                      f"bce={loss_sum['bce']/n_batch:.4f} "
                      f"proto={loss_sum['proto']/n_batch:.4f} "
                      f"lr={lrs[0]:.2e}/{lrs[1]:.2e}")

        # Eval
        ema.apply()
        auc_s = evaluate(model, dev_s, device)
        auc_u = evaluate(model, dev_u, device)
        ema.restore()
        mean = (auc_s + auc_u) / 2
        print(f"[ep {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
              f"mean={mean:.4f} steps={step} ({time.time()-t_ep:.0f}s)")

        torch.save({"model": model.state_dict(), "best_unseen": best},
                   os.path.join(out_dir, "latest.pt"))

        if auc_u > best:
            best, best_ep = auc_u, ep
            torch.save({"model": model.state_dict(), "best_unseen": auc_u, "auc_seen": auc_s},
                       os.path.join(out_dir, "best.pt"))
            print(f"  [best] unseen={auc_u:.4f} seen={auc_s:.4f}")

    print(f"\n[done] best_unseen={best:.4f} (ep{best_ep}), {time.time()-t0:.0f}s")
    with open(os.path.join(out_dir, "experiment.json"), "w") as f:
        json.dump({
            "name": args.name, "version": "wavlm_bare",
            "epochs": args.epochs, "lr": cfg.lr,
            "embed_dim": cfg.embed_dim, "wavlm": "base-plus",
            "unfreeze_layers": cfg.unfreeze_layers,
            "pooling": "asp", "weighted_sum": cfg.use_weighted_sum,
            "loss": "bce", "lambda_proto": 0.0,
            "pos_weight": cfg.pos_weight,
            "pk_P": cfg.pk_P, "pk_K": cfg.pk_K,
            "batches_per_epoch": cfg.pk_batches_per_epoch,
            "grad_accum": cfg.grad_accum,
            "batch_size": cfg.batch_size,
            "max_audio_sec": cfg.max_audio_sec,
            "warmup": cfg.warmup_steps, "scheduler": "cosine",
            "auc_seen": round(auc_s, 4),
            "auc_unseen": round(auc_u, 4),
            "best_unseen": round(best, 4),
            "best_epoch": best_ep,
            "steps": step, "duration": round(time.time()-t0, 1),
        }, f, indent=2)


# ═══════════════════════ Inference ═══════════════════════

@torch.no_grad()
def infer(args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = os.path.join(PATHS.root, "output", args.name)
    ckpt_path = args.ckpt or os.path.join(out_dir, "best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    wp = cfg.wavlm_local or cfg.wavlm_model
    model = KWSModel(wp, ckpt.get("embed_dim", 256), cfg.unfreeze_layers,
                     cfg.max_audio_sec, cfg.use_weighted_sum).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"[infer] AUC={ckpt.get('auc','?')}")

    def predict(zip_p, csv_p, prefix):
        ds = PairDataset(load_pairs(csv_p, with_label=False), zip_p, cfg, inference=True)
        loader = DataLoader(ds, batch_size=cfg.batch_size * 2, num_workers=cfg.num_workers,
                            collate_fn=collate, shuffle=False)
        rows = []
        for e, q, _, ids, _ in loader:
            e, q = e.to(device), q.to(device)
            logit, _, _ = model(e, q)
            prob = torch.sigmoid(logit).cpu().numpy()
            for pid, p in zip(ids, prob):
                rows.append((f"{prefix}_{pid}", float(p)))
        return rows

    rows = predict(cfg.eval_seen_zip, cfg.eval_seen_csv, "seen")
    rows += predict(cfg.eval_unseen_zip, cfg.eval_unseen_csv, "unseen")
    sub = os.path.join(out_dir, "submission.csv")
    with open(sub, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(rows)
    print(f"[sub] {sub} ({len(rows)} rows)")


def main():
    cfg = Config()
    cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="wavlm_bare")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--subset", type=int, default=cfg.subset)
    p.add_argument("--infer", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--ckpt", default="")
    args = p.parse_args()
    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.batch_size = args.bs
    cfg.subset = args.subset
    if args.infer:
        infer(args, cfg)
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()

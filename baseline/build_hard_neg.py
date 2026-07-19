"""Embedding-Based Hard Negative Mining — Generic.

支持两种 encoder:
  1. WavLM (frozen, 原始 hidden states mean pool) — 用于初步探测
  2. Whisper v3 checkpoint (已微调) — teacher model embedding → 高质量 hard negatives

用法:
  # Whisper checkpoint (推荐)
  python baseline/build_hard_neg.py \
      --csv train/train_label.csv --zip train/wav.zip \
      --whisper-ckpt output/whisper_v3/best.pt \
      --out baseline/hard_neg_whisper.json

  # WavLM frozen (fallback)
  python baseline/build_hard_neg.py \
      --csv train/train_label.csv --zip train/wav.zip \
      --out baseline/hard_neg_wavlm.json
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.config import PATHS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None)
    p.add_argument("--zip", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--max-pairs", type=int, default=80000)
    p.add_argument("--cos-threshold", type=float, default=0.85)
    p.add_argument("--topk", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--whisper-ckpt", default="",
                   help="path to whisper v3 checkpoint (recommended)")
    p.add_argument("--max-per-combo", type=int, default=20)
    p.add_argument("--merge", action="store_true",
                   help="merge with existing JSON instead of overwriting")
    return p.parse_args()


# ═════════════════════════════════════════════════════════
# Data loading
# ═════════════════════════════════════════════════════════

def load_csv(csv_path: str) -> List[dict]:
    import csv
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_word_index(rows: List[dict]) -> Dict[str, List[Tuple[str, str]]]:
    idx = defaultdict(list)
    for r in rows:
        if r["enroll_txt"] == r["query_txt"] and r["label"] == "1":
            w = r["enroll_txt"].lower()
            idx[w].append((r["id"], r["enroll_txt"]))
    return idx


def load_audio_batch(
    entries: List[Tuple[str, str]], zf: zipfile.ZipFile
) -> List[Tuple[str, str, np.ndarray]]:
    items = []
    for pair_id, word in entries:
        key = f"wav/{pair_id}_enroll.wav"
        try:
            data = zf.read(key)
            wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
            if sr != 16000:
                import torchaudio
                wav = torchaudio.functional.resample(
                    torch.from_numpy(wav).unsqueeze(0), sr, 16000
                ).squeeze(0).numpy()
            wav = wav.astype(np.float32)
            max_s = 3 * 16000
            if len(wav) > max_s:
                wav = wav[:max_s]
            items.append((word, pair_id, wav))
        except Exception:
            continue
    return items


# ═════════════════════════════════════════════════════════
# Encoder factory
# ═════════════════════════════════════════════════════════

def make_whisper_encoder(ckpt_path: str, device: str) -> nn.Module:
    """Load whisper v3 encoder from checkpoint."""
    from baseline.train_whisper_v3 import WhisperEncoderV3

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    whisper_model = ckpt.get("whisper_model", "base")
    embed_dim = ckpt.get("embed_dim", 256)

    class WhisperEmbedWrapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = WhisperEncoderV3(whisper_model, embed_dim,
                                            unfreeze_layers=2,
                                            max_audio_sec=3.0)
        def forward(self, wav):
            return self.encoder(wav)

    model = WhisperEmbedWrapper().to(device)
    # Load state dict (only encoder part)
    enc_state = {k.replace("encoder.", ""): v
                 for k, v in ckpt["model"].items()
                 if k.startswith("encoder.")}
    missing, unexpected = model.encoder.load_state_dict(enc_state, strict=False)
    if missing:
        print(f"  [whisper] missing keys: {len(missing)}")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def make_wavlm_raw_encoder(device: str) -> nn.Module:
    """Load frozen WavLM, use raw hidden states (768-dim) for embedding.

    Skip the randomly-initialized ASP + projection to get discriminative features.
    """
    from baseline.train_wavlm import WavLMConfig, WavLMEncoder
    cfg = WavLMConfig()
    cfg.__post_init__()
    path = cfg.wavlm_local or cfg.wavlm_model

    full_encoder = WavLMEncoder(path, embed_dim=256, unfreeze_layers=3,
                                max_audio_sec=3.0, use_weighted_sum=True).to(device)
    full_encoder.eval()
    for p in full_encoder.parameters():
        p.requires_grad = False

    class RawWavLMWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.wavlm = enc.wavlm
            self.layer_weights = enc.layer_weights
            self.use_weighted_sum = enc.use_weighted_sum
            self.max_audio_sec = enc.max_audio_sec
            self.output_norm = enc.output_norm
        def forward(self, wav):
            max_s = int(self.max_audio_sec * 16000)
            if wav.shape[-1] > max_s:
                wav = wav[:, :max_s]
            if self.use_weighted_sum and self.layer_weights is not None:
                out = self.wavlm(wav, output_hidden_states=True)
                hs = torch.stack(out.hidden_states[1:], dim=1)  # 12 layers
                w = F.softmax(self.layer_weights, dim=0)
                x = (hs * w.view(1, -1, 1, 1)).sum(dim=1).mean(dim=1)
            else:
                out = self.wavlm(wav, output_hidden_states=False)
                x = out.last_hidden_state.mean(dim=1)
            return F.normalize(x, dim=-1)

    return RawWavLMWrapper(full_encoder).to(device)


# ═════════════════════════════════════════════════════════
# Extraction
# ═════════════════════════════════════════════════════════

def extract_embeddings(
    model: nn.Module,
    word_index: Dict[str, List[Tuple[str, str]]],
    zip_path: str,
    device: str,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    zf = zipfile.ZipFile(zip_path, "r")

    all_entries = [(pid, w) for w, entries in word_index.items()
                   for pid, _ in entries]

    word_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)
    total = len(all_entries)
    t0 = time.time()

    for start in range(0, total, batch_size):
        batch_entries = all_entries[start : start + batch_size]
        items = load_audio_batch(batch_entries, zf)
        if not items:
            continue

        max_len = max(w.shape[0] for _, _, w in items)
        padded = torch.zeros(len(items), max_len)
        for i, (_, _, w) in enumerate(items):
            padded[i, : w.shape[0]] = torch.from_numpy(w)
        padded = padded.to(device)

        with torch.no_grad():
            embs = model(padded)

        for i, (word, pid, _) in enumerate(items):
            word_embs[word].append(embs[i].cpu())

        if (start // batch_size) % 100 == 0:
            elapsed = time.time() - t0
            done = start + len(batch_entries)
            rate = done / max(1, elapsed)
            eta = (total - done) / max(1, rate)
            print(f"  [{done}/{total}] {rate:.0f} audio/s, ETA {eta:.0f}s")

    zf.close()
    elapsed = time.time() - t0
    print(f"[miner] {total} audios in {elapsed:.1f}s")

    result = {}
    for w, embs in word_embs.items():
        if embs:
            result[w] = torch.stack(embs)
    return result


# ═════════════════════════════════════════════════════════
# Similarity search
# ═════════════════════════════════════════════════════════

def compute_centroids(word_embs: Dict[str, torch.Tensor]):
    words = sorted(word_embs.keys())
    centroids = torch.stack([
        F.normalize(word_embs[w].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        for w in words
    ])
    return words, centroids


def find_confusable_pairs(
    words: List[str],
    centroids: torch.Tensor,
    cos_threshold: float = 0.85,
    topk: int = 30,
) -> List[Tuple[str, str, float]]:
    centroids = F.normalize(centroids, dim=-1)
    W = len(words)
    device = centroids.device
    pairs = []
    chunk_size = 200

    for i in range(0, W, chunk_size):
        end_i = min(i + chunk_size, W)
        chunk = centroids[i:end_i]
        sim = torch.matmul(chunk, centroids.T)

        for ci in range(sim.shape[0]):
            gi = i + ci
            row = sim[ci]
            mask = torch.arange(W, device=device) > gi
            row_masked = row.clone()
            row_masked[~mask] = -1.0
            vals, idx = torch.topk(row_masked, min(topk, W - gi - 1))
            for j, s in zip(idx.tolist(), vals.tolist()):
                if s > cos_threshold:
                    pairs.append((words[gi], words[j], round(float(s), 4)))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def generate_hard_negatives(
    confusable: List[Tuple[str, str, float]],
    word_index: Dict[str, List[Tuple[str, str]]],
    max_pairs: int = 80000,
    max_per_combo: int = 20,
) -> List[dict]:
    result = []
    used = set()
    combo_count = defaultdict(int)
    rng = np.random.default_rng(42)

    for w1, w2, sim in confusable:
        if len(result) >= max_pairs:
            break
        pos1 = word_index.get(w1, [])
        pos2 = word_index.get(w2, [])
        if not pos1 or not pos2:
            continue
        ck = tuple(sorted([w1, w2]))
        if combo_count[ck] >= max_per_combo:
            continue
        n = min(max_per_combo - combo_count[ck], len(pos1), len(pos2))
        s1 = rng.choice(len(pos1), n, replace=False)
        s2 = rng.choice(len(pos2), n, replace=False)
        for si in s1:
            for sj in s2:
                id1, txt1 = pos1[si]
                id2, txt2 = pos2[sj]
                key = (id1, id2)
                if key in used:
                    continue
                used.add(key)
                result.append({
                    "id": f"{id1}_x_{id2}",
                    "enroll_id": id1,
                    "query_id": id2,
                    "enroll_txt": txt1,
                    "query_txt": txt2,
                    "label": 0,
                    "cos_sim": sim,
                })
                combo_count[ck] += 1
                if len(result) >= max_pairs:
                    break
            if len(result) >= max_pairs:
                break

    return result


# ═════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # Paths
    cfg_root = PATHS.root
    csv_path = args.csv or os.path.join(cfg_root, "train", "train_label.csv")
    zip_path = args.zip or os.path.join(cfg_root, "train", "wav.zip")
    subset_csv = os.path.join(cfg_root, "train_subset", "train_label.csv")
    subset_zip = os.path.join(cfg_root, "train_subset", "wav.zip")
    if os.path.isfile(subset_zip):
        csv_path = subset_csv
        zip_path = subset_zip
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "hard_neg_whisper.json"
    )

    print(f"[miner] csv={csv_path}")
    print(f"[miner] zip={zip_path}")
    print(f"[miner] out={out_path}")

    # ── Load model ──
    if args.whisper_ckpt:
        print(f"[miner] loading Whisper checkpoint: {args.whisper_ckpt}")
        model = make_whisper_encoder(args.whisper_ckpt, device)
        cos_thresh = 0.5  # 已微调的模型, embedding 分离度高
    else:
        print("[miner] loading frozen WavLM")
        model = make_wavlm_raw_encoder(device)
        cos_thresh = args.cos_threshold  # default 0.85

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[miner] model loaded ({n_params:,} params), cos_threshold={cos_thresh}")

    # ── Build index ──
    rows = load_csv(csv_path)
    word_index = build_word_index(rows)
    print(f"[miner] {len(word_index)} unique words with positive samples")

    # ── Extract ──
    word_embs = extract_embeddings(model, word_index, zip_path, device, args.batch_size)
    print(f"[miner] embeddings: {len(word_embs)} words")

    # ── Centroids ──
    words, centroids = compute_centroids(word_embs)
    centroids = centroids.to(device)
    print(f"[miner] centroids: {centroids.shape}")

    # ── Search ──
    t0 = time.time()
    confusable = find_confusable_pairs(words, centroids, cos_thresh, args.topk)
    print(f"[miner] confusable pairs: {len(confusable)} ({time.time()-t0:.1f}s)")

    if confusable:
        print("  top 15:")
        for w1, w2, s in confusable[:15]:
            n1 = len(word_index.get(w1, []))
            n2 = len(word_index.get(w2, []))
            print(f"    {w1:20s} <-> {w2:20s}  cos={s:.4f}  "
                  f"n1={n1:3d} n2={n2:3d}")

    # ── Generate ──
    hard_neg = generate_hard_negatives(
        confusable, word_index,
        max_pairs=args.max_pairs,
        max_per_combo=args.max_per_combo,
    )
    print(f"[miner] generated {len(hard_neg)} hard negative pairs")

    # ── Save ──
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if args.merge and os.path.isfile(out_path):
        with open(out_path, encoding="utf-8") as f:
            existing = json.load(f)
        existing_ids = {p["id"] for p in existing}
        new_only = [p for p in hard_neg if p["id"] not in existing_ids]
        hard_neg = existing + new_only
        print(f"[miner] merged: {len(existing)} existing + {len(new_only)} new "
              f"= {len(hard_neg)} total")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard_neg, f, ensure_ascii=False)
    print(f"[miner] saved to {out_path} "
          f"({os.path.getsize(out_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

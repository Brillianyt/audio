"""WavLM Embedding-Based Hard Negative Mining.

用 frozen WavLM 提取每条 enrollment 音频的 embedding，
按词聚合为中心向量，计算词间 cosine 距离，
距离最近的不同词对 → 真正的 acoustic confusable pairs → hard negatives。

原理:
  - 零手工特征: 不用音素表、不用调音特征、不限定 CMU 词典覆盖范围
  - 数据驱动: WavLM 自身的声学空间定义"相似"
  - 对齐模型: 训的 WavLM KWS 模型和挖掘用的WavLM是同一个encoder

用法:
  python baseline/build_hard_neg_wavlm.py \
      --csv dev/dev_seen/dev_seen_label.csv \
      --zip dev/dev_seen/wav.zip \
      --out baseline/hard_neg_wavlm.json \
      --max-pairs 80000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import time
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.config import PATHS
from baseline.train_wavlm import WavLMConfig, WavLMEncoder


def parse_args():
    p = argparse.ArgumentParser(description="WavLM embedding hard negative mining")
    p.add_argument("--csv", default=None, help="training CSV path")
    p.add_argument("--zip", default=None, help="training wav.zip path")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--max-pairs", type=int, default=80000)
    p.add_argument("--cos-threshold", type=float, default=0.60,
                   help="cosine similarity threshold, > this = confusable")
    p.add_argument("--topk", type=int, default=20,
                   help="top-K nearest neighbors per word")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-pairs-per-word-combo", type=int, default=15,
                   help="max cross-pairs per (word1, word2) combo")
    return p.parse_args()


def load_train_csv(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        import csv
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def build_word_index(rows: List[dict]) -> Dict[str, List[Tuple[str, str]]]:
    """word -> [(pair_id, word_text)]  for label=1 (positive) pairs."""
    idx = defaultdict(list)
    for r in rows:
        if r["enroll_txt"] == r["query_txt"] and r["label"] == "1":
            w = r["enroll_txt"].lower()
            idx[w].append((r["id"], r["enroll_txt"]))
    return idx


def extract_embeddings(
    encoder: WavLMEncoder,
    word_index: Dict[str, List[Tuple[str, str]]],
    zip_path: str,
    device: str,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """用 WavLM 原始 hidden states (768-dim, mean pooling) 提取 embedding。

    关键: 跳过随机初始化的 ASP + projection 层，直接用 WavLM 预训练特征。
    768-dim 空间中不同词的 embedding 天然分离，cosine 阈值 0.6 有效。
    """
    import io
    import zipfile
    import soundfile as sf

    zf = zipfile.ZipFile(zip_path, "r")
    encoder.eval()

    items: List[Tuple[str, str, np.ndarray]] = []
    for word, entries in word_index.items():
        for pair_id, _ in entries:
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
    zf.close()

    total = len(items)
    print(f"[miner] {total} enrollment audios ({batch_size}/batch)")
    t0 = time.time()

    word_embs: Dict[str, List[torch.Tensor]] = defaultdict(list)

    for start in range(0, total, batch_size):
        batch = items[start : start + batch_size]
        max_len = max(w.shape[0] for _, _, w in batch)
        padded = torch.zeros(len(batch), max_len)
        for i, (_, _, w) in enumerate(batch):
            padded[i, : w.shape[0]] = torch.from_numpy(w)
        padded = padded.to(device)

        with torch.no_grad():
            # 只用 WavLM 原始 hidden states (最后一层 mean pool)
            outputs = encoder.wavlm(padded, output_hidden_states=False)
            x = outputs.last_hidden_state  # (B, T, 768)
            x = x.mean(dim=1)              # (B, 768) mean pooling
            # 可选: 取所有层的加权平均
            if encoder.use_weighted_sum and encoder.layer_weights is not None:
                all_hidden = encoder.wavlm(padded, output_hidden_states=True).hidden_states[1:]
                stacked = torch.stack(all_hidden, dim=1)
                w = torch.softmax(encoder.layer_weights, dim=0)
                x = (stacked * w.view(1, -1, 1, 1)).sum(dim=1).mean(dim=1)
            embs = torch.nn.functional.normalize(x, dim=-1)

        for i, (word, pid, _) in enumerate(batch):
            word_embs[word].append(embs[i].cpu())

        if (start // batch_size) % 50 == 0:
            elapsed = time.time() - t0
            done = start + len(batch)
            eta = (elapsed / done) * (total - done) if done > 0 else 0
            print(f"  [{done}/{total}] {done / max(1, elapsed):.0f} audio/s, ETA {eta:.0f}s")

    elapsed = time.time() - t0
    print(f"[miner] {total} audios in {elapsed:.1f}s ({total / max(1, elapsed):.0f}/s)")

    result = {}
    for w, embs in word_embs.items():
        if embs:
            result[w] = torch.stack(embs)
    return result


def compute_centroids(
    word_embs: Dict[str, torch.Tensor]
) -> Tuple[List[str], torch.Tensor]:
    """每个词取 enrollment embedding 均值作为 centroid。"""
    import torch.nn.functional as F
    words = sorted(word_embs.keys())
    centroids = torch.stack([
        F.normalize(word_embs[w].mean(dim=0, keepdim=True), dim=-1).squeeze(0)
        for w in words
    ])  # (W, D)
    return words, centroids


def find_confusable_pairs(
    words: List[str],
    centroids: torch.Tensor,
    word_embs: Dict[str, torch.Tensor],
    cos_threshold: float = 0.60,
    topk: int = 20,
) -> List[Tuple[str, str, float]]:
    """找 cosine similarity 最高的不同词对。

    为避免重复计算只取上三角 + 只取 topk。
    对 8000+ 词，全对 pairwise ≈ 32M 对，在 GPU 上秒级完成。
    """
    import torch.nn.functional as F

    centroids = F.normalize(centroids, dim=-1)
    W = len(words)
    device = centroids.device

    pairs: List[Tuple[str, str, float]] = []

    # 分块计算，512-dim 以上用更小的 chunk
    chunk_size = 300 if centroids.shape[-1] <= 512 else 200
    for i in range(0, W, chunk_size):
        end_i = min(i + chunk_size, W)
        chunk_i = centroids[i:end_i]  # (C, D)

        # 与所有 centroids 的 cosine
        sim = torch.matmul(chunk_i, centroids.T)  # (C, W)

        # 只看上三角 (>i 的部分) + 排除自己
        for ci in range(sim.shape[0]):
            global_i = i + ci
            # 只看 j > global_i 且 sim > threshold
            row = sim[ci]  # (W,)
            mask = torch.zeros(W, dtype=torch.bool, device=device)
            mask[global_i + 1 :] = True  # 只看 j > i
            row_masked = row.clone()
            row_masked[~mask] = -1.0

            # topk
            if topk < W:
                top_vals, top_idx = torch.topk(row_masked, min(topk, W - global_i - 1))
            else:
                top_vals = row_masked[mask]
                top_idx = torch.where(mask)[0]
                if len(top_vals) > topk:
                    top_vals, perm = torch.topk(top_vals, topk)
                    top_idx = top_idx[perm]

            for j, s in zip(top_idx.tolist(), top_vals.tolist()):
                if s > cos_threshold:
                    pairs.append((words[global_i], words[j], round(float(s), 4)))

        # 定期清理
        if i % 2000 == 0:
            torch.cuda.empty_cache()

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def generate_hard_negatives(
    confusable_pairs: List[Tuple[str, str, float]],
    word_index: Dict[str, List[Tuple[str, str]]],
    max_pairs: int = 80000,
    max_per_combo: int = 15,
) -> List[dict]:
    """从 confusable word pairs 生成 hard negative 训练样本。"""
    result = []
    used_ids = set()
    combo_count = defaultdict(int)

    rng = np.random.default_rng(42)
    # 按 cosine 降序排列，最 confusable 的优先
    for w1, w2, sim in confusable_pairs:
        if len(result) >= max_pairs:
            break

        pos1 = word_index.get(w1, [])
        pos2 = word_index.get(w2, [])
        if not pos1 or not pos2:
            continue

        combo_key = tuple(sorted([w1, w2]))
        if combo_count[combo_key] >= max_per_combo:
            continue

        n = min(max_per_combo - combo_count[combo_key], len(pos1), len(pos2))
        s1 = rng.choice(len(pos1), n, replace=False)
        s2 = rng.choice(len(pos2), n, replace=False)

        for si in s1:
            for sj in s2:
                id1, txt1 = pos1[si]
                id2, txt2 = pos2[sj]
                key = (id1, id2)
                if key in used_ids:
                    continue
                used_ids.add(key)
                result.append({
                    "id": f"{id1}_x_{id2}",
                    "enroll_id": id1,
                    "query_id": id2,
                    "enroll_txt": txt1,
                    "query_txt": txt2,
                    "label": 0,
                    "wavlm_cos_sim": sim,
                })
                combo_count[combo_key] += 1
                if len(result) >= max_pairs:
                    break
            if len(result) >= max_pairs:
                break

    return result


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    # 路径
    cfg = WavLMConfig()
    cfg.__post_init__()
    csv_path = args.csv or cfg.train_csv
    zip_path = args.zip or cfg.train_zip
    out_path = args.out or os.path.join(
        os.path.dirname(__file__), "hard_neg_wavlm.json"
    )

    print(f"[miner] csv={csv_path}")
    print(f"[miner] zip={zip_path}")
    print(f"[miner] out={out_path}")
    print(f"[miner] cos_threshold={args.cos_threshold}, topk={args.topk}")

    # ── Step 1: Load WavLM encoder ──
    print("[miner] loading WavLM encoder ...")
    wavlm_path = cfg.wavlm_local or cfg.wavlm_model
    encoder = WavLMEncoder(
        wavlm_path,
        embed_dim=256,
        unfreeze_layers=3,
        max_audio_sec=3.0,
        use_weighted_sum=True,
    ).to(device)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[miner] encoder loaded ({sum(p.numel() for p in encoder.parameters()):,} params)")

    # ── Step 2: Build word index ──
    print("[miner] reading CSV ...")
    rows = load_train_csv(csv_path)
    word_index = build_word_index(rows)
    print(f"[miner] {len(word_index)} unique words with positive samples")

    # ── Step 3: Extract embeddings ──
    t0 = time.time()
    word_embs = extract_embeddings(encoder, word_index, zip_path, device, args.batch_size)
    print(f"[miner] extracted embeddings for {len(word_embs)} words ({time.time()-t0:.1f}s)")

    # ── Step 4: Compute centroids ──
    words, centroids = compute_centroids(word_embs)
    centroids = centroids.to(device)
    print(f"[miner] centroids: {centroids.shape}")

    # ── Step 5: Find confusable pairs ──
    t0 = time.time()
    confusable = find_confusable_pairs(
        words, centroids, word_embs, args.cos_threshold, args.topk
    )
    print(f"[miner] confusable pairs: {len(confusable)} ({time.time()-t0:.1f}s)")
    if confusable:
        print(f"  top10:")
        for w1, w2, s in confusable[:10]:
            n1 = len(word_index.get(w1, []))
            n2 = len(word_index.get(w2, []))
            print(f"    {w1:20s} <-> {w2:20s}  cos={s:.4f}  n1={n1} n2={n2}")

    # ── Step 6: Generate hard negative pairs ──
    hard_neg = generate_hard_negatives(
        confusable, word_index,
        max_pairs=args.max_pairs,
        max_per_combo=args.max_pairs_per_word_combo,
    )
    print(f"[miner] generated {len(hard_neg)} hard negative pairs")

    # ── Step 7: Save ──
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(hard_neg, f, ensure_ascii=False)
    print(f"[miner] saved to {out_path} ({os.path.getsize(out_path) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()

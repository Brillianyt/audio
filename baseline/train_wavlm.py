"""WavLM-based Keyword Spotting with Phoneme-Aware Hard Negative Mining.

核心架构:
  WavLM base-plus (weighted layer sum) → ASP Pooling → Projection → Cosine Similarity

关键改进 (vs whisper_v3):
  1. WavLM 编码器: 预训练 94k hours, 12 layers × 768 hidden, 全栈语音处理
     - 加权层求和 (learnable layer weights): 融合所有层的表示,
       保留低层声学特征和高层语义特征, 比只用最后一层好很多
  2. 音素级别难负样本挖掘: CMU Pronouncing Dictionary → ARPAbet 音素
     → 音素编辑距离 + 调音特征距离 → 捕捉真实发音相似性
  3. 增强正样本构造: 跨 utterance 的正样本配对, 多 enrollment 原型聚合
  4. 分层难负样本采样: 按音素距离分桶 (very_similar/similar/moderate), 保证多样性
  5. SupCon Loss (Supervised Contrastive): 显式拉近同类、推远异类
  6. Angular Prototypical Loss: episodic training 模拟 unseen 场景

用法:
    # 训练
    python baseline/train_wavlm.py --name wavlm_v1 --epochs 10 --hard-neg

    # 推理
    python baseline/train_wavlm.py --name wavlm_v1 --infer

依赖:
    pip install transformers torch torchaudio soundfile scikit-learn numpy
    (CMU Pronouncing Dictionary 自动下载)

参考:
  - WavLM: Large-Scale Self-Supervised Pre-Training (Chen et al., 2021)
  - Phoneme-Level Contrastive Learning for KWS (Li et al., 2024)
  - In Defence of Metric Learning for Speaker Recognition (Chung et al., 2020)
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import PATHS

# ── cmudict (pip install cmudict, pure offline) ──
_cmudict_cache = None
import re

def _get_cmudict():
    global _cmudict_cache
    if _cmudict_cache is None:
        import cmudict
        _cmudict_cache = cmudict.dict()
    return _cmudict_cache

def _word_to_phonemes(word: str) -> List[str]:
    """word -> ARPAbet phoneme list (stripped of stress markers)."""
    cmu = _get_cmudict()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w:
        return []
    plist = cmu.get(w)
    if plist:
        return [re.sub(r'[0-2]$', '', p) for p in plist[0]]
    return []


# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

class WavLMConfig:
    """WavLM 训练配置。"""

    # ── 模型 ──
    wavlm_model: str = "microsoft/wavlm-base-plus"
    wavlm_local: str = ""  # 本地路径（如有缓存）
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5   # KWS关键词<1s
    unfreeze_layers: int = 3  # 微调最后 N 层 transformer
    use_weighted_sum: bool = True  # 加权层求和 (WavLM 论文推荐)

    # ── 训练 ──
    epochs: int = 15
    lr: float = 3e-4           # 纯 BCE 可以更激进
    batch_size: int = 48  # 增大 batch, 更好利用 GPU
    grad_accum: int = 2   # 每 2 batch 更新一次, 降噪
    pos_weight: float = 5.0
    num_workers: int = 4           # 提速 I/O
    seed: int = 42
    log_every: int = 20
    subset: int = 500000

    # ── Loss weights ──
    lambda_proto: float = 0.0
    lambda_margin: float = 0.15
    lambda_phoneme: float = 0.2   # 音素辅助: 从 query_emb 预测 label (单 BCE)

    # ── Augmentation ──
    specaug_time: int = 50
    noise_aug: bool = True          # 加噪声 (对 -10~5dB 比赛场景关键)
    noise_snr_db: float = 5.0       # SNR for noise mixing

    # ── Model EMA ──
    ema_decay: float = 0.999

    # ── PK Sampler ──
    pk_P: int = 48
    pk_K: int = 4
    pk_batches_per_epoch: int = 400   # 更多 batch = 更多优化步 + 多样本

    # ── Phoneme Hard Negative ──
    hard_neg_phoneme_max_dist: float = 1.5
    hard_neg_num: int = 80000

    # ── Scheduler ──
    warmup_steps: int = 200
    flat_steps: int = 400

    # ── 路径（由 __post_init__ 设置）─
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
        # 训练数据: 优先 train_subset, 否则 train
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        subset_zip = os.path.join(r, "train_subset", "wav.zip")
        subset_csv = os.path.join(r, "train_subset", "train_label.csv")
        if os.path.isfile(subset_zip):
            self.train_zip = subset_zip
            self.train_csv = subset_csv

        # Dev
        dev_base = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(dev_base, "dev")):
            dev_base = os.path.join(dev_base, "dev")
        self.dev_seen_zip = os.path.join(dev_base, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(dev_base, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(dev_base, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(
            dev_base, "dev_unseen", "dev_unseen_label.csv"
        )

        # Eval
        self.eval_seen_zip = os.path.join(r, "eval", "eval_seen", "wav.zip")
        self.eval_seen_csv = os.path.join(
            r, "evalcsv_without_label", "eval_seen_without_label.csv"
        )
        self.eval_unseen_zip = os.path.join(r, "eval", "eval_unseen", "wav.zip")
        self.eval_unseen_csv = os.path.join(
            r, "evalcsv_without_label", "eval_unseen_without_label.csv"
        )

        # WavLM 本地路径: 找到包含 config.json 的完整 snapshot
        hub_wavlm = os.path.join(
            r, "hub", "models--microsoft--wavlm-base-plus", "snapshots"
        )
        if os.path.isdir(hub_wavlm):
            for snap in sorted(os.listdir(hub_wavlm), reverse=True):
                snap_path = os.path.join(hub_wavlm, snap)
                config = os.path.join(snap_path, "config.json")
                if os.path.isfile(config):
                    self.wavlm_local = snap_path
                    break


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_pairs_with_text(csv_path: str) -> List[dict]:
    """读取训练/开发 CSV，保留 enroll_txt / query_txt。"""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "id": r["id"],
                    "label": int(r["label"]),
                    "enroll_txt": r.get("enroll_txt", ""),
                    "query_txt": r.get("query_txt", ""),
                }
            )
    return rows


def load_pairs_no_label(csv_path: str) -> List[dict]:
    """读取评估 CSV (无标签)。"""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "enroll_txt": r.get("enroll_txt", "")})
    return rows


# ═══════════════════════════════════════════════════════════════════════
# WavLM Encoder
# ═══════════════════════════════════════════════════════════════════════

class WavLMEncoder(nn.Module):
    """WavLM 编码器: 加权层求和 + ASP Pooling + 投影。

    设计:
      1. WavLM CNN feature extractor (7 conv layers) - 完全冻结
      2. Transformer layers:
         - 前 (total - unfreeze_layers) 层: 冻结, no_grad 通过
         - 后 unfreeze_layers 层: 微调
      3. 加权层求和 (learnable weights): 融合所有 12 层输出
      4. ASP (Attentive Statistics Pooling): 加权 mean + std
      5. 投影层: 2×768 → embed_dim (256)

    参数量估算 (base-plus):
      - Total: ~360M
      - Frozen: ~310M (CNN + 前 9 层 transformer)
      - Trainable: ~50M (后 3 层 + ASP + projection)
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/wavlm-base-plus",
        embed_dim: int = 256,
        unfreeze_layers: int = 3,
        max_audio_sec: float = 1.5,   # KWS关键词<1s
        use_weighted_sum: bool = True,
    ):
        super().__init__()
        from transformers import WavLMModel

        # 加载模型: 优先本地路径, transformers 的缓存机制会自动发现本地文件
        # 如果本地有缓存 (snapshots), from_pretrained 会直接使用而不下载
        load_kwargs = {}
        if os.path.isdir(model_name_or_path):
            load_kwargs["local_files_only"] = False  # Allow HF to verify but use cache
        self.wavlm = WavLMModel.from_pretrained(model_name_or_path, **load_kwargs)
        self.wavlm.eval()
        self.hidden_size = self.wavlm.config.hidden_size  # 768
        self.num_layers = self.wavlm.config.num_hidden_layers  # 12
        self.max_audio_sec = max_audio_sec
        self.use_weighted_sum = use_weighted_sum

        # 释放不需要的组件
        if hasattr(self.wavlm, "quantizer"):
            del self.wavlm.quantizer
        if hasattr(self.wavlm, "project_q"):
            del self.wavlm.project_q
        if hasattr(self.wavlm, "project_hid"):
            del self.wavlm.project_hid

        # ── 冻结策略 ──
        # CNN 特征提取器 (7 conv): 完全冻结
        for p in self.wavlm.feature_extractor.parameters():
            p.requires_grad = False

        # Transformer: 冻结前 N 层, 微调后 M 层
        total_blocks = len(self.wavlm.encoder.layers)
        self.frozen_blocks = total_blocks - unfreeze_layers
        for p in self.wavlm.encoder.parameters():
            p.requires_grad = False
        for i in range(self.frozen_blocks, total_blocks):
            for p in self.wavlm.encoder.layers[i].parameters():
                p.requires_grad = True

        frozen_p = sum(
            not p.requires_grad for p in self.wavlm.parameters()
        )
        train_p = sum(p.requires_grad for p in self.wavlm.parameters())
        print(
            f"  [wavlm] total={frozen_p + train_p:,}, frozen={frozen_p:,}, "
            f"trainable={train_p:,} "
            f"(CNN frozen, layers 0-{self.frozen_blocks - 1} frozen, "
            f"{self.frozen_blocks}-{total_blocks - 1} trainable)"
        )

        # ── 加权层求和权重 ──
        if use_weighted_sum:
            self.layer_weights = nn.Parameter(torch.ones(self.num_layers) / self.num_layers)
        else:
            self.layer_weights = None

        # ── Layer Norm (WavLM 输出后可选) ──
        self.output_norm = nn.LayerNorm(self.hidden_size)

        # ── ASP Pooling ──
        self.attn_linear = nn.Linear(self.hidden_size, self.hidden_size // 4)
        self.attn_w = nn.Parameter(torch.randn(self.hidden_size // 4, 1))

        # ── 投影: 2 × hidden (mean+std) → embed_dim ──
        self.proj = nn.Linear(self.hidden_size * 2, embed_dim)

        # ── SpecAugment (time masking on hidden states) ──
        self.specaug_time_mask = 50
        self.specaug_n_time = 2

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # 保持冻结层为 eval
            self.wavlm.feature_extractor.eval()
            for i in range(self.frozen_blocks):
                self.wavlm.encoder.layers[i].eval()
        return self

    def forward(self, wav: torch.Tensor, return_frames: bool = False):
        """wav: (B, T) raw waveform; returns (B, embed_dim) L2-normalized.

        if return_frames=True: also returns pre-pooling frames (B, T_frame, D)
        """
        device = next(self.wavlm.parameters()).device

        max_samples = int(self.max_audio_sec * self.sample_rate)
        if wav.shape[-1] > max_samples:
            wav = wav[:, :max_samples]
        wav = wav.to(device)

        outputs = self.wavlm(wav, output_hidden_states=True, return_dict=True)
        all_hidden = outputs.hidden_states[1:]

        if self.use_weighted_sum and self.layer_weights is not None:
            stacked = torch.stack(all_hidden, dim=1)
            weights = F.softmax(self.layer_weights, dim=0)
            x = (stacked * weights.view(1, -1, 1, 1)).sum(dim=1)
        else:
            x = all_hidden[-1]

        x = self.output_norm(x)

        # SpecAugment
        if self.training and self.specaug_time_mask > 0:
            _, T, _ = x.shape
            for _ in range(self.specaug_n_time):
                t = min(self.specaug_time_mask, T - 1)
                if t > 0:
                    t0 = torch.randint(0, max(1, T - t), (1,)).item()
                    x[:, t0 : t0 + t, :] = 0.0

        # ASP Pooling
        h = torch.tanh(self.attn_linear(x))
        w = h @ self.attn_w
        w = F.softmax(w, dim=1)

        mu = (x * w).sum(dim=1)
        sigma = ((x ** 2 * w).sum(dim=1) - mu ** 2).clamp(min=1e-5).sqrt()

        if return_frames:
            return F.normalize(self.proj(torch.cat([mu, sigma], dim=-1)), dim=-1), x

        return F.normalize(self.proj(torch.cat([mu, sigma], dim=-1)), dim=-1)

    @property
    def sample_rate(self) -> int:
        return 16000


# ═══════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════

class WavLMDataset(Dataset):
    """WavLM 训练/评估数据集。

    返回 (enroll_wav, query_wav, label, enroll_txt, query_txt, pair_id)。

    支持:
      - 普通 pair: enroll 和 query 来自同一 pair_id
      - Hard negative: enroll_id ≠ query_id (跨 pair)
      - 推理模式: label = -1, text 为空
    """

    def __init__(
        self,
        pairs: List[dict],
        zip_path: str,
        cfg: WavLMConfig,
        inference: bool = False,
    ):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference

        # 延迟加载 zip
        import io
        import zipfile

        import soundfile as sf

        self._zip_cache = {}
        self._zip = None
        self._sf = sf
        self._io = io
        self._zipfile = zipfile

    @property
    def zip(self):
        import zipfile
        if self._zip is None:
            self._zip = zipfile.ZipFile(self.zip_path, "r")
        return self._zip

    def __len__(self):
        return len(self.pairs)

    def _read(self, pair_id: str, role: str) -> torch.Tensor:
        """读取 WAV 文件。"""
        import io
        import soundfile as sf

        key = f"wav/{pair_id}_{role}.wav"
        data = self.zip.read(key)
        wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)

        # 重采样 (WavLM 需要 16kHz)
        if file_sr != self.cfg.sample_rate:
            import torchaudio
            t = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), file_sr, self.cfg.sample_rate
            )
            wav = t.squeeze(0).numpy()
        wav = wav.astype(np.float32)

        # 截断
        max_samples = int(self.cfg.max_audio_sec * self.cfg.sample_rate)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        return torch.from_numpy(wav).float()

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pid = p["id"]

        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)

        e = self._read(eid, "enroll")
        q = self._read(qid, "query")

        label = -1 if self.inference else p["label"]

        if not self.inference:
            w_e = p.get("enroll_txt", "").lower()
            w_q = p.get("query_txt", "").lower()
            e_phon = _word_to_phonemes(w_e)
            q_phon = _word_to_phonemes(w_q)
        else:
            w_e = w_q = ""
            e_phon = []
            q_phon = []

        return e, q, label, w_e, w_q, pid, e_phon, q_phon


def collate_wavlm(batch):
    """Collate: pad waveforms + phoneme lists to max length in batch."""
    max_len = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, labels, w_e_list, w_q_list, ids, e_phons, q_phons = \
        [], [], [], [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        pad_e = max_len - e.shape[-1]
        pad_q = max_len - q.shape[-1]
        es.append(F.pad(e, (0, pad_e)) if pad_e > 0 else e)
        qs.append(F.pad(q, (0, pad_q)) if pad_q > 0 else q)
        labels.append(b[2])
        w_e_list.append(b[3])
        w_q_list.append(b[4])
        ids.append(b[5])
        e_phons.append(b[6])
        q_phons.append(b[7])
    return (
        torch.stack(es), torch.stack(qs),
        torch.tensor(labels, dtype=torch.float32),
        w_e_list, w_q_list, ids,
        e_phons, q_phons,
    )


# ═══════════════════════════════════════════════════════════════════════
# PK Sampler (Phoneme-Aware)
# ═══════════════════════════════════════════════════════════════════════

class PhonemeAwarePKSampler:
    """音素感知 PK 采样器。

    增强点:
      - 优先采样音素相似的词对作为同一 batch 内的负样本
      - 每个 batch: P 个词 × K 个样本/词
      - 正负样本自然混合 (同一词内可能有正负 pair)
      - 每 epoch 固定 batch 数
    """

    def __init__(
        self,
        word_to_indices: Dict[str, List[int]],
        P: int = 32,
        K: int = 4,
        batches_per_epoch: int = 250,
        phoneme_miner=None,
    ):
        self.word_to_indices = word_to_indices
        self.P = P
        self.K = K
        self.batches_per_epoch = batches_per_epoch
        self.phoneme_miner = phoneme_miner

        # 过滤样本数 ≥ K 的词
        self.valid_words = sorted(
            [w for w, idxs in word_to_indices.items() if len(idxs) >= K]
        )

    def __iter__(self):
        P = min(self.P, len(self.valid_words))
        if P == 0:
            return iter([])

        indices_batches = []
        for _ in range(self.batches_per_epoch):
            # 选择 P 个词来构造 batch
            if self.phoneme_miner is not None and len(self.valid_words) > P * 2:
                # 音素感知采样：选一个锚点词，再选 P-1 个发音相似的词
                anchor = np.random.choice(self.valid_words)
                words = [anchor]

                # 计算锚点词与其他词的距离
                candidates = [w for w in self.valid_words if w != anchor]
                np.random.shuffle(candidates)
                distances = []
                for w in candidates[: min(500, len(candidates))]:
                    d = self.phoneme_miner.get_phoneme_distance(anchor, w)
                    distances.append((w, d if d is not None else 2.0))
                distances.sort(key=lambda x: x[1])

                # 取最近的 P-1 个 + 随机补充
                n_similar = min(len(distances), P - 1)
                words += [w for w, _ in distances[:n_similar]]
                if len(words) < P:
                    remaining = [w for w in candidates if w not in words]
                    words += list(
                        np.random.choice(
                            remaining, min(P - len(words), len(remaining)), replace=False
                        )
                    )
            else:
                words = list(np.random.choice(self.valid_words, P, replace=False))

            # 每个词取 K 个样本
            batch_indices = []
            for w in words:
                chosen = list(
                    np.random.choice(self.word_to_indices[w], self.K, replace=False)
                )
                batch_indices.extend(chosen)
            np.random.shuffle(batch_indices)
            indices_batches.append(batch_indices)

        return iter(indices_batches)

    def __len__(self):
        return self.batches_per_epoch


# ═══════════════════════════════════════════════════════════════════════
# Loss Functions
# ═══════════════════════════════════════════════════════════════════════

class AngularPrototypicalLoss(nn.Module):
    """Angular Prototypical Loss - episodic training for open-set KWS.

    原理:
      1. 每个 batch 的 embedding 按词分组
      2. 每组随机分为 support (算 prototype) 和 query (被分类)
      3. query 与所有 prototype 计算 cosine similarity
      4. learnable scale + bias → CE loss
      5. 模拟推理时的 enrollment→verification → 泛化到 unseen 词

    参考: Chung et al., "In Defence of Metric Learning", Interspeech 2020
          WeSpeaker / VoxCeleb SOTA
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(
        self, embed: torch.Tensor, word_texts: List[str], labels: torch.Tensor
    ) -> torch.Tensor:
        """embed: (N, D) L2-normalized embeddings
           word_texts: [N] word texts for each embedding
           labels: [N] ground truth labels (1=match, 0=no match)
        """
        device = embed.device
        unique_words = list(set(word_texts))
        P = len(unique_words)
        if P < 2:
            return embed.new_zeros(())

        word_to_idx = {w: i for i, w in enumerate(unique_words)}
        gt_labels = torch.tensor([word_to_idx[w] for w in word_texts], device=device)

        # 每组随机分 support 和 query
        query_mask = torch.zeros(len(word_texts), dtype=torch.bool, device=device)
        for w in unique_words:
            indices = torch.tensor(
                [i for i, t in enumerate(word_texts) if t == w], device=device
            )
            n_q = max(1, len(indices) // 3)  # ~1/3 做 query
            perm = torch.randperm(len(indices), device=device)
            query_mask[indices[perm[:n_q]]] = True

        support_mask = ~query_mask

        # Prototype = mean of support embeddings
        prototypes = torch.zeros(P, embed.shape[-1], device=device)
        for i, w in enumerate(unique_words):
            idx_w = torch.tensor(
                [j for j, t in enumerate(word_texts) if t == w], device=device
            )
            mask_s = support_mask[idx_w]
            if mask_s.sum() > 0:
                prototypes[i] = embed[idx_w][mask_s].mean(dim=0)
            else:
                prototypes[i] = embed[idx_w].mean(dim=0)
        prototypes = F.normalize(prototypes, dim=1)

        # Query vs prototypes
        query_emb = embed[query_mask]
        query_labels = gt_labels[query_mask]
        cosine = torch.matmul(query_emb, prototypes.T)
        logits = self.w.clamp(min=1e-6) * cosine + self.b

        return F.cross_entropy(logits, query_labels)


# ═══════════════════════════════════════════════════════════════════════
# Phoneme Text Encoder
# ═══════════════════════════════════════════════════════════════════════

# 40 ARPAbet phonemes
PHONEME_VOCAB = [
    "AA","AE","AH","AO","AW","AY","B","CH","D","DH",
    "EH","ER","EY","F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P","R","S","SH",
    "T","TH","UH","UW","V","W","Y","Z","ZH","UNK",
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}


class PhonemeTextEncoder(nn.Module):
    """音素序列 → 固定维 text embedding + 音素多标签分布。

    用于: (1) enrollment text embedding 参与多模态融合,
         (2) 生成 phoneme target 用于辅助损失。
    """
    def __init__(self, vocab_size: int = 40, embed_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1),
        )

    def forward(self, phoneme_batch: List[List[str]]):
        """
        Returns:
            pooled: (B, D) L2-normalized text embedding
            multi_hot: (B, 40) multi-hot phoneme presence vector
        """
        device = next(self.parameters()).device
        B = len(phoneme_batch)
        if B == 0:
            return (torch.zeros(0, self.embedding.weight.shape[1], device=device),
                    torch.zeros(0, 40, device=device))

        max_len = max((len(p) for p in phoneme_batch), default=0)
        indices = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        multi_hot = torch.zeros(B, 40, device=device)

        for i, phons in enumerate(phoneme_batch):
            n = len(phons)
            for j, p in enumerate(phons):
                idx = PHONEME_TO_IDX.get(p, PHONEME_TO_IDX["UNK"])
                indices[i, j] = idx
                if idx < 39:  # skip UNK
                    multi_hot[i, idx] = 1.0
            mask[i, :n] = True

        emb = self.embedding(indices)                    # (B, T, D)
        attn_w = self.attn(emb).squeeze(-1)              # (B, T)
        attn_w = attn_w.masked_fill(~mask, -1e4)  # fp16-safe
        attn_w = torch.softmax(attn_w, dim=1)            # (B, T)
        pooled = (emb * attn_w.unsqueeze(-1)).sum(dim=1) # (B, D)
        return F.normalize(pooled, dim=-1), multi_hot


# ═══════════════════════════════════════════════════════════════════════
# Model: MM-KWS inspired multi-modal fusion
# ═══════════════════════════════════════════════════════════════════════

class WavLMKWS(nn.Module):
    def __init__(self, wavlm_path, embed_dim=256, unfreeze_layers=3,
                 max_audio_sec=3.0, use_weighted_sum=True, bare=False):
        super().__init__()
        self.bare = bare
        self.encoder = WavLMEncoder(wavlm_path, embed_dim, unfreeze_layers,
                                     max_audio_sec, use_weighted_sum)
        if bare:
            # 纯 siamese: 和 whisper_v3 完全一致的 cos+scale+bias
            self.scale = nn.Parameter(torch.tensor(4.0))
            self.bias = nn.Parameter(torch.tensor(0.0))
        else:
            self.text_encoder = PhonemeTextEncoder(len(PHONEME_VOCAB), embed_dim)
            fusion_in = embed_dim * 6
            self.fusion = nn.Sequential(
                nn.Linear(fusion_in, embed_dim), nn.ReLU(),
                nn.BatchNorm1d(embed_dim),
                nn.Linear(embed_dim, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            self.phoneme_head = nn.Linear(embed_dim, 40)
            self.cross_attn_proj = nn.Linear(768, embed_dim)
            nn.init.normal_(self.fusion[-1].weight, std=1e-4)
            nn.init.zeros_(self.fusion[-1].bias)
            nn.init.normal_(self.phoneme_head.weight, std=1e-4)
            nn.init.constant_(self.phoneme_head.bias, -1.0)

    def forward(self, enroll_wav, query_wav, enroll_phonemes=None):
        if self.bare:
            e = self.encoder(enroll_wav)
            q = self.encoder(query_wav)
            sim = (e * q).sum(dim=-1)
            return self.scale * sim + self.bias, e, q

        import math
        q = self.encoder(query_wav)
        e_audio_pooled, e_audio_frames = self.encoder(enroll_wav, return_frames=True)
        e_text, _ = self.text_encoder(enroll_phonemes or [])

        if self.training:
            frames_proj = self.cross_attn_proj(e_audio_frames)
            attn = torch.matmul(frames_proj, e_text.unsqueeze(-1))
            attn = torch.softmax(attn.squeeze(-1) / math.sqrt(256), dim=1)
            e_audio_text = (frames_proj * attn.unsqueeze(-1)).sum(dim=1)
            e_audio = F.normalize(0.7 * e_audio_pooled + 0.3 * e_audio_text, dim=-1)
        else:
            e_audio = e_audio_pooled

        feat = torch.cat([
            q, e_audio, e_text,
            q * e_audio, q * e_text, (q - e_audio).abs(),
        ], dim=-1)
        utt_logit = self.fusion(feat).squeeze(-1)
        phoneme_logits = self.phoneme_head(q.detach())
        return utt_logit, q, e_audio, e_text, phoneme_logits


# ═══════════════════════════════════════════════════════════════════════
# Scheduler
# ═══════════════════════════════════════════════════════════════════════

# 使用 PyTorch 内置 CosineAnnealingWarmRestarts 或自定义 Cosine
# 比 linear decay 更平滑, 在 KWS/SV 任务中通常好 0.5-2%

class CosineWarmupScheduler:
    """Warmup + Cosine Annealing (行业标准)."""
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            # Linear warmup
            progress = self.current_step / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = 0.5 * (1.0 + math.cos(math.pi * progress))
            progress = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * progress

        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.base_lrs[i] * progress

    def get_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# ═══════════════════════════════════════════════════════════════════════
# Model EMA
# ═══════════════════════════════════════════════════════════════════════

class ModelEMA:
    """Exponential Moving Average of model weights.

    SSL fine-tuning 标准做法: 用 EMA 模型做验证, 结果更稳定平滑。
    """
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self._register()

    def _register(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self):
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.shadow[name].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """将 EMA 权重应用到模型上 (验证前调用)."""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self):
        """恢复原始权重 (验证后调用)."""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                p.data.copy_(self.backup[name])
        self.backup.clear()


# ═══════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for batch in loader:
        e, q, y = batch[0], batch[1], batch[2]
        phonemes = batch[6]  # enroll phonemes
        e, q = e.to(device), q.to(device)
        utt_logit, *_ = model(e, q, phonemes)
        probs.append(torch.sigmoid(utt_logit).cpu().numpy())
        labels.append(y.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


# ═══════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════

def train(args, cfg: WavLMConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(output_dir, exist_ok=True)

    # ── 数据 ──
    print(f"[data] train_zip={cfg.train_zip}")
    all_pairs = load_pairs_with_text(cfg.train_csv)
    n = min(cfg.subset, len(all_pairs))
    rng = np.random.default_rng(cfg.seed)
    train_pairs = [all_pairs[i] for i in rng.permutation(len(all_pairs))[:n]]

    # ── Hard negative pairs ──
    phoneme_miner = None
    if args.hard_neg:
        # 自动发现预计算的 hard negative JSON
        candidates = [
            args.hard_neg_file,
            os.path.join(PATHS.root, "baseline", "hard_neg_whisper.json"),
            os.path.join(PATHS.root, "baseline", "hard_neg_wavlm.json"),
        ]
        hard_neg_file = next((f for f in candidates if f and os.path.isfile(f)), "")

        if hard_neg_file:
            import json
            with open(hard_neg_file, encoding="utf-8") as f:
                hard_neg_pairs = json.load(f)
            train_pairs = train_pairs + hard_neg_pairs
            rng.shuffle(train_pairs)
            print(f"  + hard_neg pairs: {len(hard_neg_pairs)} "
                  f"(from {os.path.basename(hard_neg_file)}) "
                  f"(total={len(train_pairs)})")
        else:
            from hard_neg_phoneme import PhonemeHardNegativeMiner
            print(f"[hard_neg] no precomputed file found, "
                  f"falling back to phoneme-based ...")
            miner = PhonemeHardNegativeMiner(
                cfg.train_csv, max_phoneme_dist=cfg.hard_neg_phoneme_max_dist
            )
            phoneme_miner = miner
            hard_neg_pairs = miner.generate(
                max_pairs=cfg.hard_neg_num, balance_by_distance=True
            )
            train_pairs = train_pairs + hard_neg_pairs
            rng.shuffle(train_pairs)
            print(f"  + hard_neg pairs (phoneme): {len(hard_neg_pairs)} "
                  f"(total={len(train_pairs)})")
    else:
        print(f"  pairs={len(train_pairs)}")

    train_ds = WavLMDataset(train_pairs, cfg.train_zip, cfg)

    # PK 采样器构建
    word_to_indices = defaultdict(list)
    for idx, p in enumerate(train_pairs):
        w = p.get("enroll_txt", "").lower()
        if not w:
            continue
        word_to_indices[w].append(idx)

    # 过滤样本数 ≥ K 的词
    pos_words = {
        p["enroll_txt"].lower()
        for p in train_pairs
        if p["label"] == 1 and p.get("enroll_txt")
    }
    word_to_indices = {
        w: idxs
        for w, idxs in word_to_indices.items()
        if w in pos_words and len(idxs) >= cfg.pk_K
    }

    print(
        f"  valid words for PK sampler: {len(word_to_indices)} "
        f"(≥{cfg.pk_K} samples)"
    )

    sampler = PhonemeAwarePKSampler(
        word_to_indices,
        P=cfg.pk_P,
        K=cfg.pk_K,
        batches_per_epoch=cfg.pk_batches_per_epoch,
        phoneme_miner=phoneme_miner if args.hard_neg else None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        collate_fn=collate_wavlm,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    opt_steps_per_epoch = cfg.pk_batches_per_epoch // cfg.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    print(
        f"  batches/epoch={len(train_loader)} "
        f"(PK: {cfg.pk_P}w × {cfg.pk_K}s = {cfg.pk_P * cfg.pk_K}/batch)"
    )
    print(
        f"  opt_steps/epoch={opt_steps_per_epoch}, "
        f"total_opt_steps={total_steps}"
    )

    # ── Dev loaders ──
    def dev_loader(zip_p, csv_p):
        ds = WavLMDataset(load_pairs_with_text(csv_p), zip_p, cfg)
        return DataLoader(
            ds,
            batch_size=cfg.batch_size,
            num_workers=0,
            collate_fn=collate_wavlm,
            shuffle=False,
            pin_memory=True,
        )

    dev_seen = dev_loader(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dev_unseen = dev_loader(cfg.dev_unseen_zip, cfg.dev_unseen_csv)
    print(
        f"  dev_seen={len(dev_seen.dataset)}, "
        f"dev_unseen={len(dev_unseen.dataset)}"
    )

    # ── 模型 ──
    t0 = time.time()
    wavlm_path = cfg.wavlm_local or cfg.wavlm_model
    print(f"[model] loading WavLM from {wavlm_path} ...")

    if args.resume and os.path.isfile(os.path.join(output_dir, "latest.pt")):
        ckpt = torch.load(
            os.path.join(output_dir, "latest.pt"), map_location="cpu", weights_only=False
        )
        model = WavLMKWS(
            wavlm_path,
            embed_dim=ckpt.get("embed_dim", cfg.embed_dim),
            unfreeze_layers=cfg.unfreeze_layers,
            max_audio_sec=cfg.max_audio_sec,
            use_weighted_sum=cfg.use_weighted_sum,
        ).to(device)
        missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
        if missing:
            print(f"  [resume] missing keys: {len(missing)} (new layers)")
        print(f"  [resume] loaded from latest.pt")
    else:
        model = WavLMKWS(
            wavlm_path,
            embed_dim=cfg.embed_dim,
            unfreeze_layers=cfg.unfreeze_layers,
            max_audio_sec=cfg.max_audio_sec,
            use_weighted_sum=cfg.use_weighted_sum,
        ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] {time.time() - t0:.1f}s, trainable={trainable:,} / total={total:,}")

    # ── 分组 LR ──
    wavlm_params = []
    proj_params = []       # encoder projection + text_encoder + phoneme_head
    fusion_params = []     # fusion MLP (随机初始化, 需要更大 LR)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.wavlm" in name:
            wavlm_params.append(p)
        elif "fusion" in name:
            fusion_params.append(p)
        else:
            proj_params.append(p)

    print(f"  [opt] wavlm={len(wavlm_params)}, proj={len(proj_params)}, "
          f"fusion={len(fusion_params)}")

    opt = torch.optim.AdamW(
        [
            {"params": wavlm_params, "lr": cfg.lr / 5, "weight_decay": 1e-4},
            {"params": proj_params,  "lr": cfg.lr,     "weight_decay": 1e-4},
            {"params": fusion_params,"lr": cfg.lr * 2,  "weight_decay": 1e-5},
            # fusion 层随机初始化, 需要更快收敛, weight_decay 放轻避免压死
        ],
    )

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Losses ──
    crit_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.pos_weight, device=device)
    )
    crit_proto = AngularPrototypicalLoss()

    # ── Gradient checkpointing (省显存) ──
    if hasattr(model.encoder.wavlm, "gradient_checkpointing_enable"):
        model.encoder.wavlm.gradient_checkpointing_enable()

    # ── EMA ──
    ema = ModelEMA(model, decay=cfg.ema_decay) if cfg.ema_decay > 0 else None

    # ── Scheduler ──
    opt_steps_per_epoch = cfg.pk_batches_per_epoch // cfg.grad_accum
    total_sched_steps = args.epochs * opt_steps_per_epoch
    scheduler = CosineWarmupScheduler(
        opt, warmup_steps=cfg.warmup_steps, total_steps=total_sched_steps
    )

    # ── 训练循环 ──
    t_start = time.time()
    global_step = 0
    best, best_ep = -1.0, 0

    for ep in range(1, args.epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        t_ep = time.time()
        losses = {"bce": 0.0, "margin": 0.0, "phoneme": 0.0}
        n_batches = 0

        for it, batch in enumerate(train_loader, 1):
            e, q, y = batch[0], batch[1], batch[2]
            w_e_list, w_q_list = batch[3], batch[4]
            e_phonemes = batch[6]   # List[List[str]]
            q_phonemes = batch[7]   # List[List[str]]
            e, q, y = e.to(device), q.to(device), y.to(device)

            # 噪声增强: 直接在波形上加高斯噪声, 模拟 -10~5dB 场景
            if cfg.noise_aug and (it % 3) != 0:  # 2/3 概率加噪
                noise_level = 10 ** (-cfg.noise_snr_db / 20)
                e = e + noise_level * torch.randn_like(e) * e.std(dim=-1, keepdim=True)
                q = q + noise_level * torch.randn_like(q) * q.std(dim=-1, keepdim=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                utt_logit, e_emb, q_emb, t_emb, phoneme_logits = \
                    model(e, q, e_phonemes)
                loss_bce = crit_bce(utt_logit, y)

                # Cosine margin on audio embeddings
                loss_margin = torch.tensor(0.0, device=device)
                if cfg.lambda_margin > 0:
                    cos_sim = (e_emb * q_emb).sum(dim=-1)
                    mask_pos = (y == 1)
                    mask_neg = (y == 0)
                    if mask_pos.any():
                        loss_margin += F.relu(0.6 - cos_sim[mask_pos]).mean()
                    if mask_neg.any():
                        loss_margin += F.relu(cos_sim[mask_neg] - 0.4).mean()

                # Phoneme auxiliary: predict which ENROLL phonemes appear in QUERY
                # Target = enroll_phoneme_set ∩ query_phoneme_set (40-dim multi-hot)
                loss_phoneme = torch.tensor(0.0, device=device)
                if cfg.lambda_phoneme > 0:
                    phoneme_targets = torch.zeros(
                        len(e_phonemes), 40, device=device
                    )
                    for i, (e_p, q_p) in enumerate(zip(e_phonemes, q_phonemes)):
                        e_set = set(e_p)
                        q_set = set(q_p)
                        overlap = e_set & q_set
                        for p in overlap:
                            idx = PHONEME_TO_IDX.get(p, 39)
                            if idx < 39:
                                phoneme_targets[i, idx] = 1.0
                    # 正例加 pos_weight（稀疏: ~3/40=7.5% 是 1）
                    loss_phoneme = F.binary_cross_entropy_with_logits(
                        phoneme_logits, phoneme_targets,
                        pos_weight=torch.tensor(5.0, device=device)
                    )

            loss = (loss_bce + cfg.lambda_margin * loss_margin
                    + cfg.lambda_phoneme * loss_phoneme) / cfg.grad_accum

            scaler.scale(loss).backward()

            losses["bce"] += loss_bce.item()
            losses["margin"] += loss_margin.item()
            losses["phoneme"] += loss_phoneme.item()
            n_batches += 1

            if (it % cfg.grad_accum) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=5.0,
                )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                scheduler.step()
                if ema is not None:
                    ema.update()
                global_step += 1

            del utt_logit, e_emb, q_emb, t_emb, phoneme_logits
            del e, q, y, loss_bce, loss_margin, loss_phoneme, loss

            if it % 200 == 0:
                torch.save(
                    {
                        "model": model.state_dict(),
                        "embed_dim": cfg.embed_dim,
                        "step": (ep, it),
                        "auc": best,
                    },
                    os.path.join(output_dir, "latest.pt"),
                )
                torch.cuda.empty_cache()

            if it % cfg.log_every == 0:
                avg = {k: v / it for k, v in losses.items()}
                lrs = scheduler.get_lr()
                print(
                    f"  ep{ep} {it}/{len(train_loader)} "
                    f"bce={avg['bce']:.4f} "
                    f"margin={avg['margin']:.4f} "
                    f"phon={avg['phoneme']:.4f} "
                    f"lr={lrs[0]:.2e}/{lrs[1]:.2e}/{lrs[2]:.2e}"
                )

        # ── Epoch end ──
        torch.save(
            {
                "model": model.state_dict(),
                "embed_dim": cfg.embed_dim,
                "auc": best,
            },
            os.path.join(output_dir, "latest.pt"),
        )

        gc.collect()
        torch.cuda.empty_cache()

        # 验证 (使用 EMA 模型)
        if ema is not None:
            ema.apply_shadow()
        try:
            auc_s = evaluate(model, dev_seen, device)
        except Exception as ex:
            print(f"  [warn] dev_seen eval failed: {ex}")
            auc_s = 0.5

        try:
            auc_u = evaluate(model, dev_unseen, device)
        except Exception as ex:
            print(f"  [warn] dev_unseen eval failed: {ex}")
            auc_u = 0.5
        if ema is not None:
            ema.restore()

        mean_auc = (auc_s + auc_u) / 2
        print(
            f"[epoch {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
            f"mean={mean_auc:.4f} steps={global_step} "
            f"({time.time() - t_ep:.0f}s)"
        )

        if mean_auc > best:
            best = mean_auc
            best_ep = ep
            torch.save(
                {
                    "model": model.state_dict(),
                    "embed_dim": cfg.embed_dim,
                    "auc": best,
                },
                os.path.join(output_dir, "best.pt"),
            )
            print(f"  [ckpt] saved best (AUC={best:.4f})")

    t_total = time.time() - t_start
    print(f"\n[done] best AUC={best:.4f} (epoch {best_ep}), {t_total:.0f}s")

    # 保存实验记录
    records = {
        "name": args.name,
        "version": "wavlm_v1",
        "epochs": args.epochs,
        "lr": cfg.lr,
        "embed_dim": cfg.embed_dim,
        "wavlm": cfg.wavlm_model,
        "unfreeze_layers": cfg.unfreeze_layers,
        "weighted_sum": cfg.use_weighted_sum,
        "pooling": "asp",
        "loss": "bce+margin+phoneme",
        "lambda_proto": cfg.lambda_proto,
        "lambda_margin": cfg.lambda_margin,
        "lambda_phoneme": cfg.lambda_phoneme,
        "scheduler": "cosine_warmup",
        "ema_decay": cfg.ema_decay,
        "noise_aug": cfg.noise_aug,
        "noise_snr_db": cfg.noise_snr_db,
        "hard_neg": args.hard_neg,
        "gradient_checkpointing": True,
        "pk_P": cfg.pk_P,
        "pk_K": cfg.pk_K,
        "batches_per_epoch": cfg.pk_batches_per_epoch,
        "batch_size": cfg.batch_size,
        "grad_accum": cfg.grad_accum,
        "max_audio_sec": cfg.max_audio_sec,
        "warmup": cfg.warmup_steps,
        "total_opt_steps": global_step,
        "auc_seen": round(auc_s, 4),
        "auc_unseen": round(auc_u, 4),
        "best_unseen": round(auc_u, 4),
        "auc_mean": round(best, 4),
        "best_epoch": best_ep,
        "duration": round(t_total, 1),
    }
    with open(os.path.join(output_dir, "experiment.json"), "w") as f:
        json.dump(records, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer(args, cfg: WavLMConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(PATHS.root, "output", args.name)

    ckpt_path = args.ckpt or os.path.join(output_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[error] checkpoint not found: {ckpt_path}")
        return

    wavlm_path = cfg.wavlm_local or cfg.wavlm_model
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = WavLMKWS(
        wavlm_path,
        embed_dim=ckpt.get("embed_dim", cfg.embed_dim),
        unfreeze_layers=cfg.unfreeze_layers,
        max_audio_sec=cfg.max_audio_sec,
        use_weighted_sum=cfg.use_weighted_sum,
    ).to(device)

    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        new_layers = [
            k
            for k in missing
            if any(x in k for x in ("attn_linear", "attn_w", "proj", "layer_weights"))
        ]
        if new_layers:
            print(f"  [infer] new layers (random init): {new_layers}")
    model.eval()
    print(f"[infer] loaded ckpt (AUC={ckpt.get('auc', '?')})")

    def _predict(zip_path, csv_path, prefix):
        pairs = load_pairs_no_label(csv_path)
        ds = WavLMDataset(pairs, zip_path, cfg, inference=True)
        loader = DataLoader(
            ds,
            batch_size=cfg.batch_size * 2,
            num_workers=cfg.num_workers,
            collate_fn=collate_wavlm,
            shuffle=False,
        )
        rows = []
        total = len(loader)
        t_s = time.time()
        print(f"  [{prefix}] {total} batches, {len(ds)} pairs")
        for idx, batch in enumerate(loader, 1):
            e, q = batch[0], batch[1]
            phonemes = batch[6]
            ids = batch[5]
            e, q = e.to(device), q.to(device)
            utt_logit, *_ = model(e, q, phonemes)
            prob = torch.sigmoid(utt_logit).cpu().numpy()
            for pid, p in zip(ids, prob):
                rows.append((f"{prefix}_{pid}", float(p)))
            if idx % 10 == 0 or idx == 1 or idx == total:
                print(f"    [{prefix}] {idx}/{total} ({time.time() - t_s:.0f}s)")
        return rows

    rows = _predict(cfg.eval_seen_zip, cfg.eval_seen_csv, "seen")
    rows += _predict(cfg.eval_unseen_zip, cfg.eval_unseen_csv, "unseen")

    sub_path = os.path.join(output_dir, "submission.csv")
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(rows)
    print(f"[submission] {sub_path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════════

def main():
    cfg = WavLMConfig()
    cfg.__post_init__()

    parser = argparse.ArgumentParser(
        description="WavLM Keyword Spotting Training"
    )
    parser.add_argument("--name", default="wavlm_v1", help="experiment name")
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--bs", type=int, default=cfg.batch_size)
    parser.add_argument("--subset", type=int, default=cfg.subset)
    parser.add_argument("--infer", action="store_true", help="inference mode")
    parser.add_argument("--resume", action="store_true", help="resume from latest.pt")
    parser.add_argument(
        "--hard-neg", action="store_true", help="enable phoneme hard negative mining"
    )
    parser.add_argument(
        "--hard-neg-dist",
        type=float,
        default=cfg.hard_neg_phoneme_max_dist,
        help="max phoneme distance for hard negatives",
    )
    parser.add_argument(
        "--hard-neg-num",
        type=int,
        default=cfg.hard_neg_num,
        help="max hard negative pairs",
    )
    parser.add_argument("--ckpt", default="", help="checkpoint path for inference")
    parser.add_argument(
        "--hard-neg-file",
        default="",
        help="pre-computed hard negative JSON (from build_hard_neg_wavlm.py)",
    )
    parser.add_argument(
        "--unfreeze",
        type=int,
        default=cfg.unfreeze_layers,
        help="number of WavLM layers to unfreeze",
    )
    parser.add_argument(
        "--no-weighted-sum",
        action="store_true",
        help="disable weighted layer sum (use last layer only)",
    )
    args = parser.parse_args()

    cfg.epochs = args.epochs
    cfg.lr = args.lr
    cfg.batch_size = args.bs
    cfg.subset = args.subset
    cfg.unfreeze_layers = args.unfreeze
    cfg.use_weighted_sum = not args.no_weighted_sum
    cfg.hard_neg_phoneme_max_dist = args.hard_neg_dist
    cfg.hard_neg_num = args.hard_neg_num

    if args.infer:
        infer(args, cfg)
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()

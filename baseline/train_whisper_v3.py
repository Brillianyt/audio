"""Whisper 预训练编码器 + Angular Prototypical + ASP Pooling (v3)。

核心设计（参考 WeSpeaker / VoxCeleb SOTA / Chung et al. Interspeech 2020）:
  - BCE loss: pairwise verification 决策边界
  - Angular Prototypical loss: episodic training (support→prototype→query CE)
    VoxCeleb benchmark: EER 2.31, 优于 ArcFace 2.40 / GE2E 2.53
  - ASP pooling: 加权 mean+std (tanh activation, 与 WeSpeaker 一致)
  - PK sampler + 固定 batch 数
  - Warmup + Flat + Linear Decay scheduler

v2 → v3 改动:
  1. TripletLoss(word2idx) → GE2E Loss (centroid + leave-one-out + softmax)
     参考 Wan et al. 2018 (speaker verification) + Zhu et al. 2024 (KWS)
     正确标注 enroll/query 各自的词文本，不再错误映射
  2. Attention Pooling → ASP (Attentive Statistics Pooling)
     学习帧权重 + 加权 mean/std
  3. PK Sampler: 每 epoch 固定 batch 数
     原: 34 batches/epoch  →  现: 200 batches/epoch
  4. Scheduler: warmup + flat + linear decay
  5. Hard Negative Mining: 挖掘发音相似词对 (hi↔haier) 增强区分能力

用法:
    python baseline/train_whisper_v3.py --name whisper_v3 --epochs 10
    python baseline/train_whisper_v3.py --infer --name whisper_v3

依赖:
    pip install openai-whisper
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
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import soundfile as sf

from config import AUDIO, PATHS, TRAIN
from data import read_wav, add_noise


# ═══════════════════════════════════════════════════════════════════
# Collate
# ═══════════════════════════════════════════════════════════════════

def collate_whisper_v3(batch):
    """collate: (wav, wav, label, w_e, w_q, id) → padded batch。

    返回两组文本：w_e_list (enroll 词), w_q_list (query 词)。
    """
    max_len = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, labels, w_e_list, w_q_list, ids = [], [], [], [], [], []
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
    return (torch.stack(es), torch.stack(qs),
            torch.tensor(labels, dtype=torch.float32),
            w_e_list, w_q_list, ids)


# ═══════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════

class WhisperConfigV3:
    """Whisper v3 配置。"""
    whisper_model: str = "base"
    embed_dim: int = 256
    sample_rate: int = 16000
    epochs: int = 10              # ↓ (v3: 更多 step/epoch，更少 epoch)
    lr: float = 3e-4
    batch_size: int = 64
    grad_accum: int = 4
    pos_weight: float = 5.0
    num_workers: int = 2
    seed: int = 42
    log_every: int = 10            # ↓ (v3: batch 更多，log 频率降低)
    subset: int = 500000
    lambda_proto: float = 0.0        # 先关掉，验证 BCE-only 能否泛化
    specaug_freq: int = 10          # 保守: 验证任务不可破坏共振峰
    specaug_time: int = 50
    noise_aug: bool = False
    unfreeze_layers: int = 2        # 少改 Whisper → 保留预训练通用特征
    max_audio_sec: float = 3.0

    # PK sampler
    pk_P: int = 32
    pk_K: int = 4
    pk_batches_per_epoch: int = 200

    # Scheduler
    warmup_steps: int = 100
    flat_steps: int = 200

    # 路径
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
        if not os.path.isfile(self.train_zip):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")

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


# ═══════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════

def load_pairs_with_text(csv_path: str) -> List[dict]:
    """读取 CSV，保留 enroll_txt / query_txt。"""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({
                "id": r["id"],
                "label": int(r["label"]),
                "enroll_txt": r.get("enroll_txt", ""),
                "query_txt": r.get("query_txt", ""),
            })
    return rows


def load_pairs_no_label(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"]})
    return rows


class WhisperPairDatasetV3(Dataset):
    """返回 (enroll_wav, query_wav, label, w_e, w_q, id)。

    v3 关键变更:
      - word_text 代替 word_label —— 不做 word2idx 映射
      - 支持 hard negative pairs（来自 cross-pair 挖掘，enroll_id ≠ query_id）
    """

    def __init__(self, pairs: List[dict], zip_path: str, cfg: WhisperConfigV3,
                 inference: bool = False):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference

    def __len__(self):
        return len(self.pairs)

    def _read(self, pid: str, role: str) -> torch.Tensor:
        wav = read_wav(self.zip_path, f"wav/{pid}_{role}.wav", self.cfg.sample_rate)
        max_samples = int(self.cfg.max_audio_sec * self.cfg.sample_rate)
        if len(wav) > max_samples:
            wav = wav[:max_samples]
        return torch.from_numpy(wav).float()

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pid = p["id"]
        # 支持 hard negative: enroll 和 query 来自不同原始 pair
        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)
        e = self._read(eid, "enroll")
        q = self._read(qid, "query")
        label = -1 if self.inference else p["label"]
        if not self.inference:
            w_e = p.get("enroll_txt", "").lower()
            w_q = p.get("query_txt", "").lower()
        else:
            w_e = w_q = ""
        return e, q, label, w_e, w_q, pid


# ═══════════════════════════════════════════════════════════════════
# PK Sampler (改进版)
# ═══════════════════════════════════════════════════════════════════

class ImprovedPKSampler:
    """每 epoch 生成固定数量 batch 的 PK 采样器。

    v3 改动: 每 epoch 固定 N 个 batch（而非扫一遍所有 word），保证足够的 optimizer step。
    正负样本自然混合在 PK batch 中（与 v2 一致）。
    """

    def __init__(self, word_to_indices: Dict[str, List[int]],
                 P: int = 32, K: int = 4, batches_per_epoch: int = 200):
        self.word_to_indices = word_to_indices
        self.P = P
        self.K = K
        self.batches_per_epoch = batches_per_epoch
        self.valid_words = [w for w, idxs in word_to_indices.items()
                            if len(idxs) >= K]

    def __iter__(self):
        P = min(self.P, len(self.valid_words))
        if P == 0:
            return iter([])

        indices = []
        for _ in range(self.batches_per_epoch):
            words = list(np.random.choice(self.valid_words, P, replace=False))
            batch_indices = []
            for w in words:
                chosen = list(np.random.choice(self.word_to_indices[w],
                                               self.K, replace=False))
                batch_indices.extend(chosen)
            np.random.shuffle(batch_indices)
            indices.append(batch_indices)
        return iter(indices)

    def __len__(self):
        return self.batches_per_epoch


# ═══════════════════════════════════════════════════════════════════
# SpecAugment (保持不变)
# ═══════════════════════════════════════════════════════════════════

class SpecAugment:
    def __init__(self, freq_mask=12, time_mask=20, n_freq=2, n_time=2):
        self.freq_mask = freq_mask
        self.time_mask = time_mask
        self.n_freq = n_freq
        self.n_time = n_time

    def __call__(self, mel: torch.Tensor) -> torch.Tensor:
        B, n_mels, T = mel.shape
        mel = mel.clone()
        for _ in range(self.n_freq):
            f = int(torch.randint(0, min(self.freq_mask, n_mels - 1) + 1, (1,)).item())
            f0 = int(torch.randint(0, max(1, n_mels - f), (1,)).item())
            if f > 0:
                mel[:, f0:f0 + f, :] = mel.mean()
        for _ in range(self.n_time):
            t = int(torch.randint(0, min(self.time_mask, T - 1) + 1, (1,)).item())
            t0 = int(torch.randint(0, max(1, T - t), (1,)).item())
            if t > 0:
                mel[:, :, t0:t0 + t] = mel.mean()
        return mel


# ═══════════════════════════════════════════════════════════════════
# Angular Prototypical Loss — open-set 泛化最优 (VoxCeleb benchmark)
# 参考: Chung et al., "In Defence of Metric Learning", Interspeech 2020
#       WeSpeaker toolkit, VoxCeleb SOTA systems
# ═══════════════════════════════════════════════════════════════════

class AngularPrototypicalLoss(nn.Module):
    """Angular Prototypical Loss for open-set keyword verification。

    VoxCeleb benchmark 中 EER 2.31（优于 ArcFace 2.40 / GE2E 2.53）。

    原理:
      1. 每个关键词的 embedding 随机分为 support 和 query
      2. support 算 prototype (centroid)，query 与所有 prototype 比 cos_sim
      3. learnable scale + bias → P-way CE loss
      4. 模拟推理时的 enrollment→verification 流程

    为什么泛化好:
      - episodic training 天然匹配 unseen 场景
      - 不需要全局类别 → batch 内 P 个词即时定义
      - learnable scale/bias 自适应校准（而非固定 margin）
    """

    def __init__(self, init_w: float = 10.0, init_b: float = -5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embed: torch.Tensor, word_texts: List[str]) -> torch.Tensor:
        """embed: (N, D) L2 归一化
           word_texts: [N] — 每个 embedding 正确标注的词文本
        """
        device = embed.device

        unique_words = list(set(word_texts))
        P = len(unique_words)
        if P < 2:
            return embed.new_zeros(())

        word_to_idx = {w: i for i, w in enumerate(unique_words)}
        labels = torch.tensor([word_to_idx[w] for w in word_texts], device=device)

        # 每组随机分 support (算 prototype) 和 query (被分类)
        query_mask = torch.zeros(len(word_texts), dtype=torch.bool, device=device)
        for w in unique_words:
            indices = torch.tensor([i for i, t in enumerate(word_texts) if t == w],
                                   device=device)
            n_q = max(1, len(indices) // 3)  # ~1/3 做 query, 2/3 做 support
            perm = torch.randperm(len(indices), device=device)
            query_mask[indices[perm[:n_q]]] = True

        support_mask = ~query_mask

        # prototype = mean of support embeddings
        prototypes = torch.zeros(P, embed.shape[-1], device=device)
        for i, w in enumerate(unique_words):
            mask_w = torch.tensor([t == w for t in word_texts], device=device)
            mask_s = mask_w & support_mask
            if mask_s.sum() > 0:
                prototypes[i] = embed[mask_s].mean(dim=0)
            else:
                prototypes[i] = embed[mask_w].mean(dim=0)  # fallback
        prototypes = F.normalize(prototypes, dim=1)

        # query vs all prototypes
        query_emb = embed[query_mask]
        query_labels = labels[query_mask]
        cosine = torch.matmul(query_emb, prototypes.T)         # (N_q, P)

        # learnable scale + bias
        logits = self.w.clamp(min=1e-6) * cosine + self.b

        return F.cross_entropy(logits, query_labels)


# ═══════════════════════════════════════════════════════════════════
# Whisper Encoder (核心改动 #2: Statistics Pooling)
# ═══════════════════════════════════════════════════════════════════

class WhisperEncoderV3(nn.Module):
    """Whisper 编码器 + Attentive Statistics Pooling (ASP) + 投影。

    v3 改动:
      - Attention Pooling (v2) → ASP (v3)
      - ASP = 学习帧权重 + 加权 mean/std，语音验证行业标准
      - 比纯 mean/std 多了 ~66K 参数，但能自动聚焦关键词帧、抑制 silence
      - 比 v2 的纯 attention pooling 多了 std 信息（spread → 泛化更好）
    """

    def __init__(self, model_name: str = "base", embed_dim: int = 256,
                 unfreeze_layers: int = 4, max_audio_sec: float = 3.0):
        super().__init__()
        import whisper
        import whisper.audio as wa

        self.whisper = whisper.load_model(model_name)
        self.whisper_dim = self.whisper.dims.n_audio_state          # base: 512
        self.whisper.eval()
        self.max_audio_sec = max_audio_sec

        # 释放未使用的 decoder（省 ~37M params × 16 bytes ≈ 600 MB 显存）
        if hasattr(self.whisper, 'decoder'):
            del self.whisper.decoder

        # 冻结策略
        total_blocks = len(self.whisper.encoder.blocks)
        self.frozen_blocks = total_blocks - unfreeze_layers
        for p in self.whisper.encoder.parameters():
            p.requires_grad = False
        for i in range(self.frozen_blocks, total_blocks):
            for p in self.whisper.encoder.blocks[i].parameters():
                p.requires_grad = True

        frozen_p = sum(not p.requires_grad for p in self.whisper.encoder.parameters())
        train_p = sum(p.requires_grad for p in self.whisper.encoder.parameters())
        print(f"  [whisper] frozen={frozen_p:,}, trainable={train_p:,} "
              f"(blocks 0-{self.frozen_blocks-1} frozen, "
              f"{self.frozen_blocks}-{total_blocks-1} train)")

        # v3: Attentive Statistics Pooling (ASP)
        # 学习每个帧的重要性权重 → 加权 mean + 加权 std
        self.attn_linear = nn.Linear(self.whisper_dim, self.whisper_dim // 4)  # 512 → 128
        self.attn_w = nn.Parameter(torch.randn(self.whisper_dim // 4, 1))       # 128 → 1

        # 投影: 2× Whisper dim → embed_dim（单层，最小容量→防 seen 过拟合）
        self.proj = nn.Linear(self.whisper_dim * 2, embed_dim)

        self.register_buffer("hann", torch.hann_window(400))
        self.register_buffer("mel_filters", wa.mel_filters("cpu", 80))
        self.specaug = SpecAugment(freq_mask=10, time_mask=50)   # 保守: 保留共振峰结构

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for i in range(self.frozen_blocks):
                self.whisper.encoder.blocks[i].eval()
        return self

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        """wav: (B, T); 返回 (B, embed_dim) L2 归一化。"""
        T_orig = wav.shape[-1]
        device = next(self.whisper.parameters()).device
        wav_gpu = wav.to(device)

        # v3: 截断过长音频（KWS 关键词 ≤1s，3s 足够覆盖前后 context）
        max_samples = int(self.max_audio_sec * 16000)
        if wav_gpu.shape[-1] > max_samples:
            wav_gpu = wav_gpu[:, :max_samples]
            T_orig = min(T_orig, max_samples)

        # Whisper 前端: STFT → Mel → log → normalize
        stft = torch.stft(wav_gpu, 400, 160, 400, self.hann, return_complex=True)
        mags = stft[..., :-1].abs() ** 2
        mel = self.mel_filters.to(mags.device).to(mags.dtype) @ mags
        mel = torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0

        if self.training and self.specaug is not None:
            mel = self.specaug(mel)

        enc = self.whisper.encoder
        n_valid = max(1, (T_orig + 319) // 320)

        # 冻结层 no_grad
        with torch.no_grad():
            x = F.gelu(enc.conv1(mel))
            x = F.gelu(enc.conv2(x))
            x = x.permute(0, 2, 1)
            x = x + enc.positional_embedding[:x.shape[1]]
            for block in enc.blocks[:self.frozen_blocks]:
                x = block(x)

        for block in enc.blocks[self.frozen_blocks:]:
            x = block(x)

        x = enc.ln_post(x)                                      # (B, T_frame, D)
        x = x[:, :n_valid, :]                                   # 截取有效帧

        # v3: Attentive Statistics Pooling (ASP)
        #   学习帧权重 → 加权 mean + 加权 std
        #   自动聚焦关键词帧，抑制 silence/padding
        h = torch.tanh(self.attn_linear(x))                      # (B, T, D//4)
        w = h @ self.attn_w                                      # (B, T, 1)
        w = F.softmax(w, dim=1)                                  # (B, T, 1)

        mu = (x * w).sum(dim=1)                                  # (B, D) 加权均值
        sigma = ((x ** 2 * w).sum(dim=1) - mu ** 2) \
            .clamp(min=1e-5).sqrt()                              # (B, D) 加权标准差
        x = torch.cat([mu, sigma], dim=-1)                       # (B, 2D)

        return F.normalize(self.proj(x), dim=-1)


# ═══════════════════════════════════════════════════════════════════
# Multi-Task Model (BCE + Contrastive)
# ═══════════════════════════════════════════════════════════════════

class MultiTaskWhisperKWSV3(nn.Module):
    """Whisper 孪生网络 (BCE matching + Pairwise Contrastive)。"""

    def __init__(self, whisper_model: str = "base", embed_dim: int = 256,
                 unfreeze_layers: int = 4, max_audio_sec: float = 3.0):
        super().__init__()
        self.encoder = WhisperEncoderV3(whisper_model, embed_dim,
                                        unfreeze_layers=unfreeze_layers,
                                        max_audio_sec=max_audio_sec)
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, enroll_wav, query_wav):
        e = self.encoder(enroll_wav)
        q = self.encoder(query_wav)
        sim = (e * q).sum(dim=-1)                                # cosine sim
        return self.scale * sim + self.bias, e, q


# ═══════════════════════════════════════════════════════════════════
# Text-Enhanced Model (Whisper frozen + Phoneme Text branch)
# ═══════════════════════════════════════════════════════════════════

PHONEME_VOCAB = [
    "AA","AE","AH","AO","AW","AY","B","CH","D","DH",
    "EH","ER","EY","F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P","R","S","SH",
    "T","TH","UH","UW","V","W","Y","Z","ZH","UNK",
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}

import re as _re
_cmudict_cache = None
def _get_cmudict():
    global _cmudict_cache
    if _cmudict_cache is None:
        import cmudict
        _cmudict_cache = cmudict.dict()
    return _cmudict_cache

def _word_to_phonemes(word: str) -> List[str]:
    cmu = _get_cmudict()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w: return []
    plist = cmu.get(w)
    if plist: return [_re.sub(r'[0-2]$', '', p) for p in plist[0]]
    return []


# ═══════════════════════════════════════════════════════════════════
# RoPE (Rotary Position Embedding)
# ═══════════════════════════════════════════════════════════════════

class RoPE(nn.Module):
    """Rotary Position Embedding. 标准实现, 参考 LLaMA / GPT-NeoX."""
    def __init__(self, dim, max_len=32, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_len = max_len
        self._cache = None

    def _build_cache(self, device, dtype):
        t = torch.arange(self.max_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cache = (emb.cos().to(dtype), emb.sin().to(dtype))

    def forward(self, x, offset=0):
        if self._cache is None or self._cache[0].device != x.device:
            self._build_cache(x.device, x.dtype)
        cos, sin = self._cache
        T = x.shape[1]
        return cos[offset:offset+T, :], sin[offset:offset+T, :]


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply RoPE to query and key. q,k: (B, H, T, D)"""
    cos = cos[:q.shape[2], :].unsqueeze(0).unsqueeze(0)  # (1,1,T,D)
    sin = sin[:q.shape[2], :].unsqueeze(0).unsqueeze(0)
    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin
    return q_embed, k_embed


# ═══════════════════════════════════════════════════════════════════
# RoPE Transformer Phoneme Encoder
# ═══════════════════════════════════════════════════════════════════

class RoPETransformerEncoder(nn.Module):
    """2-layer Transformer with RoPE, CLS pooling. ~300K params."""
    def __init__(self, vocab_size=40, embed_dim=256, n_heads=4, n_layers=2, dropout=0.05):
        super().__init__()
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.rope = RoPE(embed_dim // n_heads, max_len=32)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "ln_attn": nn.LayerNorm(embed_dim),
                "qkv": nn.Linear(embed_dim, 3 * embed_dim, bias=False),
                "out_proj": nn.Linear(embed_dim, embed_dim),
                "ln_ffn": nn.LayerNorm(embed_dim),
                "ffn": nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim * 4, embed_dim),
                    nn.Dropout(dropout),
                ),
            }) for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(embed_dim)

    def forward(self, indices, mask):
        """indices: (B, T), mask: (B, T)"""
        B, T = indices.shape
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, self.embedding(indices)], dim=1)  # (B, 1+T, D)
        attn_mask = torch.cat([
            torch.ones(B, 1, dtype=torch.bool, device=indices.device), mask
        ], dim=1)  # CLS always attends
        causal_mask = torch.zeros(1+T, 1+T, dtype=torch.bool, device=indices.device)

        cos, sin = self.rope(x)

        for layer in self.layers:
            residual = x
            x = layer["ln_attn"](x)
            qkv = layer["qkv"](x).reshape(B, 1+T, 3, self.n_heads, self.head_dim)
            q, k, v = qkv.unbind(2)
            q, k = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3)  # (B,H,T,D)
            v = v.permute(0, 2, 1, 3)
            q, k = apply_rope(q, k, cos, sin)

            # Scaled dot-product attention
            scale = self.head_dim ** -0.5
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale
            attn = attn.masked_fill(~attn_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
            attn = torch.softmax(attn, dim=-1)
            x = torch.matmul(attn, v)
            x = x.permute(0, 2, 1, 3).reshape(B, 1+T, -1)
            x = layer["out_proj"](x)
            x = residual + x

            residual = x
            x = layer["ln_ffn"](x)
            x = layer["ffn"](x)
            x = residual + x

        x = self.ln_final(x[:, 0])  # CLS token
        return F.normalize(x, dim=-1)


class PhonemeTextEncoder(nn.Module):
    """Phoneme Text Encoder: 兼容旧接口."""
    def __init__(self, vocab_size=40, embed_dim=256):
        super().__init__()
        self.encoder = RoPETransformerEncoder(vocab_size, embed_dim=128, n_heads=4, n_layers=2)
        self.proj = nn.Linear(128, embed_dim) if embed_dim != 128 else nn.Identity()

    def forward(self, phoneme_batch: List[List[str]]):
        device = next(self.parameters()).device
        B = len(phoneme_batch)
        if B == 0:
            return torch.zeros(0, 256, device=device), None

        max_len = max(len(p) for p in phoneme_batch)
        indices = torch.zeros(B, max_len, dtype=torch.long, device=device)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=device)
        for i, phons in enumerate(phoneme_batch):
            for j, p in enumerate(phons):
                indices[i, j] = PHONEME_TO_IDX.get(p, 39)
            mask[i, :len(phons)] = True

        emb = self.encoder(indices, mask)
        emb = self.proj(emb)
        return F.normalize(emb, dim=-1), None


class WhisperTextKWS(nn.Module):
    """Whisper encoder + Phoneme text branch → cosine scores."""

    def __init__(self, whisper_ckpt_path, embed_dim=256, unfreeze_whisper=0):
        super().__init__()
        ckpt = torch.load(whisper_ckpt_path, map_location="cpu", weights_only=False)
        wm = ckpt.get("whisper_model", "base")
        self.unfreeze_whisper = unfreeze_whisper
        self.encoder = WhisperEncoderV3(wm, embed_dim, unfreeze_layers=unfreeze_whisper,
                                         max_audio_sec=3.0)
        missing, _ = self.encoder.load_state_dict(
            {k.replace("encoder.",""): v for k,v in ckpt["model"].items()
             if k.startswith("encoder.")}, strict=False)
        print(f"  [text] whisper encoder loaded (missing={len(missing)}, "
              f"unfreeze={unfreeze_whisper})")
        if unfreeze_whisper == 0:
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.encoder.eval()
        self.text_encoder = PhonemeTextEncoder(40, embed_dim)
        # From checkpoint: inherit scale/bias, lock audio
        self.log_var_a = nn.Parameter(torch.tensor(0.0))   # 音频不确定性
        self.log_var_t = nn.Parameter(torch.tensor(0.0))   # 文本不确定性
        self.bias = nn.Parameter(torch.tensor(0.0))
        # score = exp(-σ_a)*cos_audio + exp(-σ_t)*cos_text + bias

    def forward(self, enroll_wav, query_wav, enroll_phonemes, query_phonemes=None):
        if self.unfreeze_whisper == 0:
            with torch.no_grad(), torch.amp.autocast("cuda", enabled=enroll_wav.is_cuda):
                e_audio = self.encoder(enroll_wav)
                q = self.encoder(query_wav)
        else:
            e_audio = self.encoder(enroll_wav)
            q = self.encoder(query_wav)
        e_text, _ = self.text_encoder(enroll_phonemes)
        q_text = None
        if query_phonemes:
            q_text, _ = self.text_encoder(query_phonemes)
        sim_a = (e_audio * q).sum(dim=-1)
        sim_t = (e_text * q).sum(dim=-1)
        w_a = torch.exp(-self.log_var_a)
        w_t = torch.exp(-self.log_var_t)
        return w_a * sim_a + w_t * sim_t + self.bias, e_audio, q, e_text, q_text


# ═══════════════════════════════════════════════════════════════════
# Scheduler (核心改动 #4)
# ═══════════════════════════════════════════════════════════════════

def get_lr(step: int, warmup: int, flat: int, total: int, base_lr: float) -> float:
    """Warmup → Flat → Linear Decay。

    避免 cosine 在步数少时无效的问题。
    - step < warmup:         线性增加
    - warmup ≤ step < w+f:   保持 base_lr
    - step ≥ w+f:            线性衰减到 0
    """
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    elif step < warmup + flat:
        return base_lr
    else:
        decay_steps = max(1, total - warmup - flat)
        progress = (step - warmup - flat) / decay_steps
        return base_lr * max(0.0, 1.0 - progress)


# ═══════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for e, q, y, *__ in loader:
        e, q = e.to(device), q.to(device)
        logit, _, _ = model(e, q)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


# ═══════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════

def train(args, cfg: WhisperConfigV3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(PATHS.ckpt_dir, exist_ok=True)
    latest_path = os.path.join(output_dir, "latest.pt")

    # ── 数据 ──
    print(f"[data] train_zip={cfg.train_zip}")
    all_pairs = load_pairs_with_text(cfg.train_csv)
    n = min(cfg.subset, len(all_pairs))
    rng = np.random.default_rng(cfg.seed)
    train_pairs = [all_pairs[i] for i in rng.permutation(len(all_pairs))[:n]]

    # 难负样本: 优先使用预计算的 JSON, 否则字符编辑距离
    if args.hard_neg:
        import json as _json
        hn_files = [
            args.hard_neg_file,
            os.path.join(PATHS.root, "baseline", "hard_neg_whisper.json"),
            os.path.join(PATHS.root, "baseline", "hard_neg_wavlm.json"),
        ]
        hn_file = next((f for f in hn_files if f and os.path.isfile(f)), "")
        if hn_file:
            with open(hn_file, encoding="utf-8") as f:
                hard_neg_pairs = _json.load(f)
            print(f"  + hard_neg pairs: {len(hard_neg_pairs)} (from {os.path.basename(hn_file)})")
        else:
            from hard_neg import HardNegativeMiner
            miner = HardNegativeMiner(cfg.train_csv, max_dist=args.hard_neg_dist)
            hard_neg_pairs = miner.generate(max_pairs=args.hard_neg_num)
            print(f"  + hard_neg pairs: {len(hard_neg_pairs)} (from HardNegativeMiner)")
        train_pairs = train_pairs + hard_neg_pairs
        rng.shuffle(train_pairs)
        print(f"  total={len(train_pairs)}")

    train_ds = WhisperPairDatasetV3(train_pairs, cfg.train_zip, cfg)

    # v3: 构建 word → indices 映射（正负样本混合，与 v2 一致）
    # 只用 label=1 的词来定义 PK 词表，但每个词包含它所有的 pair（正+负）
    word_to_indices = defaultdict(list)
    for idx, p in enumerate(train_pairs):
        w = p.get("enroll_txt", "").lower()
        if not w:
            continue
        # 只收录正样本涉及的词（v2 用 Counter on label=1）
        word_to_indices[w].append(idx)

    # 过滤样本数 ≥ K 的词，且至少有一个正样本
    pos_words = {p["enroll_txt"].lower() for p in train_pairs
                 if p["label"] == 1 and p.get("enroll_txt")}
    word_to_indices = {w: idxs for w, idxs in word_to_indices.items()
                       if w in pos_words and len(idxs) >= cfg.pk_K}

    # 过滤样本数 ≥ K 的词
    valid_words = {w for w, idxs in word_to_indices.items()
                   if len(idxs) >= cfg.pk_K}
    word_to_indices = {w: idxs for w, idxs in word_to_indices.items()
                       if w in valid_words}
    print(f"  valid words for PK sampler: {len(word_to_indices)} "
          f"(≥{cfg.pk_K} samples)")

    sampler = ImprovedPKSampler(
        word_to_indices,
        P=cfg.pk_P, K=cfg.pk_K,
        batches_per_epoch=cfg.pk_batches_per_epoch,
    )
    train_loader = DataLoader(
        train_ds, batch_sampler=sampler,
        num_workers=cfg.num_workers, collate_fn=collate_whisper_v3,
        pin_memory=True,
    )
    opt_steps_per_epoch = cfg.pk_batches_per_epoch // cfg.grad_accum
    total_steps = args.epochs * opt_steps_per_epoch
    print(f"  batches/epoch={len(train_loader)} "
          f"(PK: {cfg.pk_P}w×{cfg.pk_K}s={cfg.pk_P * cfg.pk_K}/batch)")
    print(f"  opt_steps/epoch={opt_steps_per_epoch}, "
          f"total_opt_steps={total_steps}")

    # Dev loaders
    def dev_loader(zip_p, csv_p):
        ds = WhisperPairDatasetV3(load_pairs_with_text(csv_p), zip_p, cfg)
        return DataLoader(ds, batch_size=128, num_workers=0,
                          collate_fn=collate_whisper_v3, shuffle=False)

    dev_seen = dev_loader(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dev_unseen = dev_loader(cfg.dev_unseen_zip, cfg.dev_unseen_csv)
    print(f"  dev_seen={len(dev_seen.dataset)}, "
          f"dev_unseen={len(dev_unseen.dataset)}")

    # ── 模型 ──
    t0 = time.time()
    resume_ep = 0
    best, best_ep = -1.0, 0

    if args.resume and os.path.isfile(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        model = MultiTaskWhisperKWSV3(
            ckpt.get("whisper_model", cfg.whisper_model),
            ckpt.get("embed_dim", cfg.embed_dim),
            unfreeze_layers=cfg.unfreeze_layers,
            max_audio_sec=cfg.max_audio_sec,
        ).to(device)
        model.load_state_dict(ckpt["model"], strict=False)
        resume_ep = ckpt.get("step", (0, 0))[0]
        best = ckpt.get("auc", 0.0)
        print(f"[model] resumed from {latest_path} (epoch {resume_ep}, AUC={best:.4f})")
    else:
        model = MultiTaskWhisperKWSV3(
            cfg.whisper_model, cfg.embed_dim,
            unfreeze_layers=cfg.unfreeze_layers,
            max_audio_sec=cfg.max_audio_sec,
        ).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[model] {time.time() - t0:.1f}s, "
          f"trainable={trainable:,} / total={total:,}")

    # 分组 LR
    whisper_params, proj_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "encoder.whisper" in name:
            whisper_params.append(p)
        else:
            proj_params.append(p)

    opt = torch.optim.AdamW([
        {"params": whisper_params, "lr": args.lr / 10},
        {"params": proj_params,    "lr": args.lr},
    ], weight_decay=1e-4)
    print(f"  [opt] whisper_params={len(whisper_params)}, "
          f"proj_params={len(proj_params)}")

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    # ── Losses ──
    crit_bce = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(cfg.pos_weight, device=device))
    crit_embed = AngularPrototypicalLoss()

    # ── 训练循环 ──
    t_start = time.time()
    global_step = 0

    for ep in range(resume_ep + 1, args.epochs + 1):
        torch.cuda.empty_cache()
        gc.collect()

        model.train()
        t_ep = time.time()
        losses = {"bce": 0.0, "embed": 0.0}

        for it, (e, q, y, w_e_list, w_q_list, _) in enumerate(train_loader, 1):
            e, q, y = e.to(device), q.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logit, e_emb, q_emb = model(e, q)
                loss_bce = crit_bce(logit, y)

                # ArcFace: 在 embedding 空间做 P-way 分类（angular margin）
                emb_cat = torch.cat([e_emb, q_emb])
                texts_cat = w_e_list + w_q_list
                loss_embed = crit_embed(emb_cat, texts_cat)

            loss = (loss_bce + cfg.lambda_proto * loss_embed) / cfg.grad_accum
            scaler.scale(loss).backward()

            losses["bce"] += loss_bce.item()
            losses["embed"] += loss_embed.item()

            if (it % cfg.grad_accum) == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=5.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()

                # v3: Warmup → Flat → Linear Decay
                global_step += 1
                lr_whisper = get_lr(global_step, cfg.warmup_steps, cfg.flat_steps,
                                    total_steps, args.lr / 10)
                lr_proj = get_lr(global_step, cfg.warmup_steps, cfg.flat_steps,
                                 total_steps, args.lr)
                for pg, lr in zip(opt.param_groups, [lr_whisper, lr_proj]):
                    pg["lr"] = lr

            del logit, e_emb, q_emb, e, q, y, loss_embed, loss_bce, loss

            # 每 200 batch 保存
            if it % 200 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "embed_dim": cfg.embed_dim,
                    "whisper_model": cfg.whisper_model,
                    "step": (ep, it), "auc": best,
                }, os.path.join(output_dir, "latest.pt"))
                torch.cuda.empty_cache()

            if it % cfg.log_every == 0:
                print(f"  ep{ep} {it}/{len(train_loader)} "
                      f"bce={losses['bce'] / it:.4f} "
                      f"proto={losses['embed'] / it:.4f} "
                      f"lr_w={opt.param_groups[0]['lr']:.2e} "
                      f"lr_p={opt.param_groups[1]['lr']:.2e}")

        # ── Epoch end ──
        torch.save({
            "model": model.state_dict(),
            "embed_dim": cfg.embed_dim,
            "whisper_model": cfg.whisper_model,
            "auc": best,
        }, os.path.join(output_dir, "latest.pt"))

        gc.collect()
        torch.cuda.empty_cache()

        # 验证
        try:
            auc_s = evaluate(model, dev_seen, device)
        except Exception:
            auc_s = 0.5
        try:
            auc_u = evaluate(model, dev_unseen, device)
        except Exception:
            auc_u = 0.5

        mean_auc = (auc_s + auc_u) / 2
        n_opt = global_step
        print(f"[epoch {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
              f"mean={mean_auc:.4f} steps={n_opt} "
              f"({time.time() - t_ep:.0f}s)")

        if mean_auc > best:
            best = mean_auc
            best_ep = ep
            torch.save({
                "model": model.state_dict(),
                "embed_dim": cfg.embed_dim,
                "whisper_model": cfg.whisper_model,
                "auc": best,
            }, os.path.join(output_dir, "best.pt"))
            print(f"  [ckpt] saved (AUC={best:.4f})")

    t_total = time.time() - t_start
    print(f"\n[done] best AUC={best:.4f} (epoch {best_ep}), {t_total:.0f}s")

    # 保存实验记录
    records = {
        "name": args.name, "version": "v3",
        "epochs": args.epochs, "lr": args.lr,
        "embed_dim": cfg.embed_dim,
        "whisper": cfg.whisper_model,
        "unfreeze_layers": cfg.unfreeze_layers,
        "pooling": "asp",
        "loss": "bce+angular_prototypical",
        "hard_neg": args.hard_neg,
        "lambda_proto": cfg.lambda_proto,
        "pk_P": cfg.pk_P, "pk_K": cfg.pk_K,
        "batches_per_epoch": cfg.pk_batches_per_epoch,
        "total_opt_steps": global_step,
        "auc_seen": round(auc_s, 4),
        "auc_unseen": round(auc_u, 4),
        "auc_mean": round(best, 4),
        "best_epoch": best_ep,
        "duration": round(t_total, 1),
    }
    with open(os.path.join(output_dir, "experiment.json"), "w") as f:
        json.dump(records, f, indent=2)


# ═══════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def infer(args, cfg: WhisperConfigV3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = os.path.join(PATHS.root, "output", args.name)
    ckpt_path = args.ckpt or os.path.join(output_dir, "best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[error] checkpoint not found: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = MultiTaskWhisperKWSV3(
        ckpt.get("whisper_model", "base"),
        ckpt.get("embed_dim", 256),
        unfreeze_layers=cfg.unfreeze_layers,
        max_audio_sec=cfg.max_audio_sec,
    ).to(device)
    # strict=False: 兼容旧 ckpt（缺 ASP 层 / 不同 unfreeze_layers 等）
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        new_layers = [k for k in missing
                      if any(x in k for x in ("attn_linear", "attn_w", "proj"))]
        if new_layers:
            print(f"  [infer] new layers (random init): {new_layers}")
    model.eval()
    print(f"[infer] loaded (AUC={ckpt.get('auc', '?'):.4f})")

    def _predict(zip_path, csv_path, prefix):
        pairs = load_pairs_no_label(csv_path)
        ds = WhisperPairDatasetV3(pairs, zip_path, cfg, inference=True)
        loader = DataLoader(ds, batch_size=128, num_workers=cfg.num_workers,
                            collate_fn=collate_whisper_v3, shuffle=False)
        rows = []
        total = len(loader)
        t_s = time.time()
        print(f"  [{prefix}] {total} batches, {len(ds)} pairs")
        for idx, batch in enumerate(loader, 1):
            e, q = batch[0], batch[1]
            ids = batch[-1]
            e, q = e.to(device), q.to(device)
            logit, _, _ = model(e, q)
            prob = torch.sigmoid(logit).cpu().numpy()
            for pid, p in zip(ids, prob):
                rows.append((f"{prefix}_{pid}", float(p)))
            if idx % 10 == 0 or idx == 1 or idx == total:
                print(f"    [{prefix}] {idx}/{total} ({time.time() - t_s:.0f}s)")
        return rows

    rows = _predict(cfg.eval_seen_zip, cfg.eval_seen_csv, "seen")
    rows += _predict(cfg.eval_unseen_zip, cfg.eval_unseen_csv, "unseen")
    sub_path = os.path.join(output_dir, "submission.csv")
    os.makedirs(os.path.dirname(sub_path), exist_ok=True)
    with open(sub_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(rows)
    print(f"[submission] {sub_path} ({len(rows)} rows)")


# ═══════════════════════════════════════════════════════════════════
# Text-Enhanced Training
# ═══════════════════════════════════════════════════════════════════

class TextPairDataset(Dataset):
    """Same as WhisperPairDatasetV3 but also returns phonemes."""
    def __init__(self, pairs, zip_path, cfg, inference=False):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference

    def __len__(self): return len(self.pairs)

    def _read(self, pid, role):
        wav = read_wav(self.zip_path, f"wav/{pid}_{role}.wav", self.cfg.sample_rate)
        max_s = int(self.cfg.max_audio_sec * self.cfg.sample_rate)
        if len(wav) > max_s: wav = wav[:max_s]
        return torch.from_numpy(wav).float()

    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        e = self._read(eid, "enroll"); q = self._read(qid, "query")
        label = -1 if self.inference else p["label"]
        e_phon = _word_to_phonemes(p.get("enroll_txt","").lower()) if not self.inference else []
        q_phon = _word_to_phonemes(p.get("query_txt","").lower()) if not self.inference else []
        return e, q, label, e_phon, q_phon, pid


def collate_text(batch):
    max_len = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, ls, epns, qpns, ids = [], [], [], [], [], []
    for b in batch:
        e,q = b[0],b[1]
        es.append(F.pad(e,(0,max_len-e.shape[-1])) if e.shape[-1]<max_len else e)
        qs.append(F.pad(q,(0,max_len-q.shape[-1])) if q.shape[-1]<max_len else q)
        ls.append(b[2]); epns.append(b[3]); qpns.append(b[4]); ids.append(b[5])
    return (torch.stack(es), torch.stack(qs), torch.tensor(ls,dtype=torch.float32),
            epns, qpns, ids)


def train_text_mode(args, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)

    # 找到最佳 whisper checkpoint
    text_ckpt = args.text_ckpt
    if not text_ckpt:
        candidates = [
            os.path.join(PATHS.root, "output", d, "best.pt")
            for d in os.listdir(os.path.join(PATHS.root, "output"))
            if os.path.isdir(os.path.join(PATHS.root, "output", d))
            and "whisper" in d
        ]
        candidates = [c for c in candidates if os.path.isfile(c)]
        if candidates:
            text_ckpt = max(candidates, key=lambda c: torch.load(
                c, map_location="cpu", weights_only=False).get("auc", 0))
    if not text_ckpt or not os.path.isfile(text_ckpt):
        print(f"[text] no whisper checkpoint found!"); return
    ckpt_auc = torch.load(text_ckpt,map_location="cpu",weights_only=False).get("auc",0)
    print(f"[text] using checkpoint: {text_ckpt} (AUC={ckpt_auc})")

    out_dir = os.path.join(PATHS.root, "output", f"text_{args.name}")
    os.makedirs(out_dir, exist_ok=True)

    # Data
    all_pairs = load_pairs_with_text(cfg.train_csv)
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
            import json as _json
            with open(hn_file, encoding="utf-8") as f:
                hn = _json.load(f)
            train_pairs = train_pairs + hn
            rng.shuffle(train_pairs)
            print(f"  + hard_neg: {len(hn)} (from {os.path.basename(hn_file)})")
    # Also load iter1/iter2/phoneme if exists
    for iname in ["hard_neg_iter1.json", "hard_neg_iter2.json", "hard_neg_phoneme.json"]:
        ipath = os.path.join(PATHS.root, "baseline", iname)
        if os.path.isfile(ipath):
            with open(ipath, encoding="utf-8") as f:
                hn = json.load(f)
            train_pairs = train_pairs + hn
            rng.shuffle(train_pairs)
            print(f"  + hard_neg: {len(hn)} (from {iname})")
    # Self-paired data
    sp = os.path.join(PATHS.root, "train", "self_paired.json")
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            sp_pairs = json.load(f)
        train_pairs = train_pairs + sp_pairs
        rng.shuffle(train_pairs)
        print(f"  + self_paired: {len(sp_pairs)} pairs")
    print(f"  total={len(train_pairs)}")

    # Keep full pairs, re-sample each epoch to prevent overfitting
    full_pairs = train_pairs
    print(f"  total pairs: {len(full_pairs)}")
    train_loader = None  # will be recreated each epoch

    def get_loader(ep):
        """Fresh 200K sample each epoch."""
        if len(full_pairs) > 200000:
            rng = np.random.default_rng(cfg.seed + ep)
            subset = rng.choice(full_pairs, 200000, replace=False).tolist()
        else:
            subset = full_pairs
        ds = TextPairDataset(subset, cfg.train_zip, cfg)
        return DataLoader(ds, batch_size=256, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_text,
                          pin_memory=True, drop_last=True)

    def dev_ld(zip_p, csv_p):
        ds = TextPairDataset(load_pairs_with_text(csv_p), zip_p, cfg)
        return DataLoader(ds, batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dev_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dev_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)
    print(f"  dev_seen={len(dev_s.dataset)}, dev_unseen={len(dev_u.dataset)}")

    # Model
    model = WhisperTextKWS(text_ckpt, cfg.embed_dim,
                           unfreeze_whisper=args.unfreeze_whisper).to(device)
    best, best_ep = -1.0, 0

    # Load text checkpoint as starting point
    if args.load_text and os.path.isfile(args.load_text):
        ckpt = torch.load(args.load_text, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        best = ckpt.get("auc_unseen", -1.0)
        print(f"  [load] text checkpoint: best_unseen={best:.4f}")

    if args.resume:
        latest = os.path.join(out_dir, "latest.pt")
        if os.path.isfile(latest):
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            best = ckpt.get("auc_unseen", -1.0)
            print(f"  [resume] loaded, best_unseen={best:.4f}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [text] trainable={trainable:,} (whisper frozen)")

    # Optimizer: whisper part gets lower LR
    if args.unfreeze_whisper > 0:
        wavlm_p = [p for n, p in model.named_parameters()
                   if p.requires_grad and "encoder.whisper" in n]
        other_p = [p for n, p in model.named_parameters()
                   if p.requires_grad and "encoder.whisper" not in n]
        opt = torch.optim.AdamW([
            {"params": wavlm_p, "lr": 3e-5, "weight_decay": 1e-4},
            {"params": other_p, "lr": 3e-4, "weight_decay": 1e-4},
        ])
    else:
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        loader = get_loader(ep)
        model.train(); t_ep = time.time(); loss_sum, n = 0.0, 0
        total_batches = len(loader)
        for i, (e, q, y, epns, qpns, _) in enumerate(loader):
            e, q, y = e.to(device), q.to(device), y.to(device)
            # Noise aug: random SNR [-10, 5] dB, independent per audio
            snr_e = np.random.uniform(-10, 5)
            snr_q = np.random.uniform(-10, 5)
            e = e + (10**(-snr_e/20)) * torch.randn_like(e) * e.std(dim=-1,keepdim=True)
            q = q + (10**(-snr_q/20)) * torch.randn_like(q) * q.std(dim=-1,keepdim=True)
            logit, e_audio, q_emb, e_text, q_text = model(e, q, epns, qpns)
            loss = crit(logit, y)
            # Multi-way alignment + margin losses
            cos_ea_et = (e_audio * e_text).sum(dim=-1)
            cos_ea_q = (e_audio * q_emb).sum(dim=-1)
            cos_et_qt = (e_text * q_text).sum(dim=-1) if q_text is not None else None
            cos_ea_qt = (e_audio * q_text).sum(dim=-1) if q_text is not None else None

            loss_align = torch.tensor(0.0, device=device)
            # 1. enroll_audio ↔ enroll_text: 恒正, push cos>0.8
            loss_align += F.relu(0.8 - cos_ea_et).mean()
            # 2. enroll_audio ↔ query_audio: label决定
            pos_mask = (y == 1); neg_mask = (y == 0)
            if pos_mask.any():
                loss_align += F.relu(0.8 - cos_ea_q[pos_mask]).mean()
            if neg_mask.any():
                loss_align += F.relu(cos_ea_q[neg_mask] - 0.2).mean()
            # 3. text ↔ text: label决定
            if cos_et_qt is not None:
                if pos_mask.any():
                    loss_align += F.relu(0.8 - cos_et_qt[pos_mask]).mean()
                if neg_mask.any():
                    loss_align += F.relu(cos_et_qt[neg_mask] - 0.2).mean()
            # 4. enroll_audio ↔ query_text: label决定
            if cos_ea_qt is not None:
                if pos_mask.any():
                    loss_align += F.relu(0.8 - cos_ea_qt[pos_mask]).mean()
                if neg_mask.any():
                    loss_align += F.relu(cos_ea_qt[neg_mask] - 0.2).mean()
            loss = loss + 0.05 * loss_align + 0.01 * (model.log_var_a**2 + model.log_var_t**2)
            opt.zero_grad(); loss.backward(); opt.step()
            loss_sum += loss.item(); n += 1
            if i == 0 or (i+1) % 100 == 0 or i == total_batches-1:
                wa = torch.exp(-model.log_var_a).item()
                wt = torch.exp(-model.log_var_t).item()
                print(f"  ep{ep} [{i+1}/{total_batches}] loss={loss_sum/n:.4f} "
                      f"lr={opt.param_groups[0]['lr']:.2e} "
                      f"wa={wa:.2f} wt={wt:.2f} "
                      f"({time.time()-t_ep:.0f}s)")
            del logit, e_audio, q_emb, e_text, q_text, e, q, y, loss, loss_align
        scheduler.step()
        torch.save({"model": model.state_dict(), "auc_unseen": best},
                   os.path.join(out_dir, "latest.pt"))

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e,q,y,epns,qpns,_ in ld:
                e,q=e.to(device),q.to(device)
                lp,_,_,_,_ = model(e,q,epns,qpns)
                ps.append(torch.sigmoid(lp).cpu().numpy()); ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        auc_s = ev(dev_s); auc_u = ev(dev_u)
        print(f"[ep {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
              f"mean={(auc_s+auc_u)/2:.4f} ({time.time()-t_ep:.0f}s)")
        if auc_u > best:
            best, best_ep = auc_u, ep
            torch.save({"model": model.state_dict(), "auc_seen": auc_s,
                         "auc_unseen": auc_u, "whisper_ckpt": text_ckpt},
                       os.path.join(out_dir, "best.pt"))
            print(f"  [best] unseen={auc_u:.4f}")

    print(f"\n[done] best_unseen={best:.4f} (ep{best_ep}), {time.time()-t0:.0f}s")


# ═══════════════════════════════════════════════════════════════════
# Entry
# ═══════════════════════════════════════════════════════════════════

def main():
    cfg = WhisperConfigV3()
    cfg.__post_init__()

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="whisper_v3")
    parser.add_argument("--whisper", default=cfg.whisper_model,
                        choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--epochs", type=int, default=cfg.epochs)
    parser.add_argument("--lr", type=float, default=cfg.lr)
    parser.add_argument("--bs", type=int, default=cfg.batch_size)
    parser.add_argument("--subset", type=int, default=cfg.subset)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--resume", action="store_true", help="resume from latest.pt")
    parser.add_argument("--hard-neg", action="store_true", help="enable hard negative mining")
    parser.add_argument("--hard-neg-dist", type=int, default=2, help="max edit distance for similar words")
    parser.add_argument("--hard-neg-num", type=int, default=50000, help="max hard negative pairs")
    parser.add_argument("--hard-neg-file", default="", help="pre-computed hard neg JSON")
    parser.add_argument("--ckpt", default="")
    parser.add_argument("--text-mode", action="store_true",
                        help="freeze whisper + add phoneme text branch")
    parser.add_argument("--text-ckpt", default="",
                        help="whisper best.pt for text mode (overrides default)")
    parser.add_argument("--unfreeze-whisper", type=int, default=0,
                        help="unfreeze last N whisper layers")
    parser.add_argument("--load-text", default="",
                        help="load text model checkpoint as starting point")
    args = parser.parse_args()

    cfg.whisper_model = args.whisper

    if args.infer:
        infer(args, cfg)
    elif args.text_mode:
        train_text_mode(args, cfg)
    else:
        train(args, cfg)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import io
import math
import os
import zipfile
from typing import List

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from config import AudioConfig

_ZIP_CACHE: dict = {}


def _get_zip(path: str) -> zipfile.ZipFile:
    key = (os.getpid(), path)
    if key not in _ZIP_CACHE:
        _ZIP_CACHE[key] = zipfile.ZipFile(path, "r")
    return _ZIP_CACHE[key]


def read_wav(zip_path: str, name: str, sr: int) -> np.ndarray:
    data = _get_zip(zip_path).read(name)
    wav, file_sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if file_sr != sr:
        t = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), file_sr, sr)
        wav = t.squeeze(0).numpy()
    return wav.astype(np.float32)


def load_pairs(csv_path: str, with_label: bool) -> List[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            item = {"id": r["id"]}
            if with_label:
                item["label"] = int(r["label"])
            rows.append(item)
    return rows


def add_noise(wav: np.ndarray, snr_range: tuple, rng: np.random.Generator) -> np.ndarray:
    """添加高斯噪声，SNR (dB) 从 snr_range 中随机采样。"""
    snr_db = rng.uniform(*snr_range)
    sig_power = float(np.mean(wav ** 2)) + 1e-12
    noise = rng.standard_normal(len(wav)).astype(np.float32)
    noise_power = float(np.mean(noise ** 2)) + 1e-12
    target_noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = noise * math.sqrt(target_noise_power / noise_power)
    return wav + noise


class LogMel:
    def __init__(self, cfg: AudioConfig):
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            power=2.0,
        )

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        return torch.log(self.mel(wav) + 1e-6)


def pad_spec(spec: torch.Tensor, max_frames: int) -> torch.Tensor:
    """(n_mels, T) -> (1, n_mels, max_frames)"""
    T = spec.shape[-1]
    if T < max_frames:
        spec = torch.nn.functional.pad(spec, (0, max_frames - T))
    else:
        spec = spec[:, :max_frames]
    return spec.unsqueeze(0)


class PairDataset(Dataset):
    def __init__(self, pairs: List[dict], zip_path: str, cfg: AudioConfig,
                 inference: bool = False,
                 noise_aug: bool = False,
                 noise_augmentor=None,
                 feat_type: str = "logmel",
                 specaug=None):
        self.pairs = pairs
        self.zip_path = zip_path
        self.cfg = cfg
        self.inference = inference
        self.noise_aug = noise_aug
        self.noise_augmentor = noise_augmentor
        self.feat_type = feat_type  # "logmel" | "raw"
        self.specaug = specaug      # SpecAugment transform or None
        self.logmel = LogMel(cfg) if feat_type == "logmel" else None

    def __len__(self):
        return len(self.pairs)

    def _feat(self, wav_name: str) -> torch.Tensor:
        wav = read_wav(self.zip_path, wav_name, self.cfg.sample_rate)

        # 训练时添加噪声增强（仅对 query 音频）
        if self.noise_aug and not self.inference and "query" in wav_name:
            if self.noise_augmentor is not None:
                wav = self.noise_augmentor.mix(wav)
            else:
                rng = np.random.default_rng(2026)
                wav = add_noise(wav, (-10, 5), rng)

        if self.feat_type == "raw":
            return torch.from_numpy(wav).unsqueeze(0).float()

        # log-mel 谱图
        spec = self.logmel(torch.from_numpy(wav))               # (n_mels, T)
        spec = pad_spec(spec, self.cfg.max_frames)               # (1, n_mels, max_frames)

        # SpecAugment (仅训练时)
        if self.specaug is not None and not self.inference:
            spec = self.specaug(spec)

        return spec

    def __getitem__(self, idx: int):
        p = self.pairs[idx]
        pid = p["id"]
        # 支持 hard negative: enroll 和 query 来自不同原始 pair
        enroll_id = p.get("enroll_id", pid)
        query_id = p.get("query_id", pid)
        e = self._feat(f"wav/{enroll_id}_enroll.wav")
        q = self._feat(f"wav/{query_id}_query.wav")
        label = -1 if self.inference else p["label"]
        return e, q, label, pid


def collate(batch):
    es = torch.stack([b[0] for b in batch])
    qs = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    ids = [b[3] for b in batch]
    return es, qs, labels, ids


def collate_raw_wav(batch):
    """用于原始波形（变长）的 collate 函数。

    将 batch 内的 wav pad 到相同长度，返回 (B, T_max) 张量。
    """
    max_len = 0
    for b in batch:
        max_len = max(max_len, b[0].shape[-1], b[1].shape[-1])

    es, qs = [], []
    for b in batch:
        e, q = b[0], b[1]
        if e.shape[-1] < max_len:
            e = torch.nn.functional.pad(e, (0, max_len - e.shape[-1]))
        if q.shape[-1] < max_len:
            q = torch.nn.functional.pad(q, (0, max_len - q.shape[-1]))
        es.append(e)
        qs.append(q)

    es = torch.stack(es)
    qs = torch.stack(qs)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    ids = [b[3] for b in batch]
    return es, qs, labels, ids

"""Mel-based audio quality assessment for routing inference.

Computes a damage_score ∈ [0, 1] from mel spectrogram features:

  1. snr_estimate:  语音频带能量 vs 高频噪声频带能量比，低 SNR → 高 damage
  2. band_dropout:  最高频带能量相对中频的衰减，严重压缩/带限 → 高 damage
  3. flatness:      谱平坦度（保留，区分度好）

Usage:
    from audio_quality import compute_damage_score_batch, classify_damage
    scores = compute_damage_score_batch(list_of_waveforms)  # [0, 1]
"""
import torch
import torch.nn.functional as F
import numpy as np
import whisper.audio as wa

_MEL_FILTERS_80 = wa.mel_filters("cpu", 80)


def _mel_spec_batch(wav_batch, n_mels=80, n_fft=400, hop=160):
    """Compute log-mel spectrograms for a batch.

    Returns (log_mel, lin_energy):
      - log_mel: (B, n_mels, T) 归一化到 [0,1]
      - lin_energy: (B, n_mels, T) 线性域能量 (≥1e-10)
    """
    device = wav_batch.device
    hann = torch.hann_window(n_fft, device=device)
    stft = torch.stft(wav_batch, n_fft, hop, n_fft, hann, return_complex=True)
    mags = stft[..., :-1].abs() ** 2
    filters = _MEL_FILTERS_80.to(device=device, dtype=mags.dtype)
    mel = filters @ mags  # linear energy in mel bands
    lin_energy = mel.clamp(min=1e-10)

    log_mel = torch.log10(lin_energy)
    log_mel = torch.maximum(log_mel, log_mel.max(dim=-1, keepdim=True).values - 8.0)
    log_mel = (log_mel + 4.0) / 4.0

    return log_mel, lin_energy


def _snr_estimate(lin_energy):
    """SNR 估计：语音频带能量 vs 高频噪声频带能量比。

    mel bands (16kHz audio):
      - 0-32  (0~3.2kHz):  语音主体
      - 48-80 (4.8~8kHz):  高频噪声/细节

    用 log10(noise/speech) 映射:
      clean (SNR>20dB):  noise/speech ≈ 0.01 → log10 ≈ -2.0 → score≈0.17
      moderate (SNR~5dB): noise/speech ≈ 0.3 → log10 ≈ -0.5 → score≈0.57
      severe (SNR<0dB):   noise/speech ≈ 1.0 → log10 ≈ 0   → score≈0.71
    """
    speech = lin_energy[:, :32, :].sum(dim=(1, 2))  # (B,)
    noise_band = lin_energy[:, 48:, :].sum(dim=(1, 2))  # (B,)
    ratio = noise_band / speech.clamp(min=1e-10)
    log_r = torch.log10(ratio.clamp(min=1e-10))
    # log_r range: clean ≈ -2.5~-1.5, noisy ≈ -1.0~-0.3
    # 映射: (-log_r + 0.5) / 3.0, 即 log_r = -3 → 0.17, log_r = 0 → 0.83
    score = (3.0 - log_r) / 6.0  # log_r=-3 → 1.0, log_r=0 → 0.5
    score = 1.0 - score  # 翻转: 低 SNR → 高 score
    return score.clamp(0, 1)


def _band_dropout(lin_energy):
    """最高频带衰减：mel bins 60-80 vs 20-40 的能量比。

    严重压缩/带限的音频高频几乎为零，比值极小。
    干净语音的高频也有一定能量（摩擦音等），比值较大。

    返回: (B,) ∈ [0, 1], 越大表示音损越大
    """
    mid = lin_energy[:, 20:40, :].sum(dim=(1, 2))  # (B,) 中频参考
    high = lin_energy[:, 60:, :].sum(dim=(1, 2))   # (B,) 极高频
    ratio = high / mid.clamp(min=1e-10)
    # 干净语音 ratio ~0.01-0.1, 带限音频 ratio ~0.0001-0.001
    # 用 log 压缩后映射到 [0, 1]
    log_ratio = torch.log10(ratio.clamp(min=1e-10))
    # log_ratio 范围大约 [-4, -1], 映射到 [0, 1]
    score = 1.0 - (log_ratio + 4.0) / 3.0  # [-4, -1] → [0, 1]
    return score.clamp(0, 1)


def _spectral_flatness(lin_energy):
    """谱平坦度：几何平均 / 算术平均。

    平坦度 ≈ 1 → 类噪声；≈ 0 → 有清晰谐波结构。
    使用语音主体频带 (0-3.2kHz) 的平坦度。
    """
    speech = lin_energy[:, :32, :]  # (B, 32, T) 只取语音频段
    log_e = torch.log(speech)  # safe because speech ≥ 1e-10
    geo = torch.exp(log_e.mean(dim=1))  # (B, T)
    arith = speech.mean(dim=1)  # (B, T)
    flat = geo / arith.clamp(min=1e-10)
    flat = flat.mean(dim=-1)  # (B,)
    return flat.clamp(0, 1)


def compute_damage_score_batch(wavs):
    """Compute damage scores for a batch of waveforms.

    Args:
        wavs: list of (T,) float32 tensors, 16kHz

    Returns:
        scores: list of float, damage_score ∈ [0, 1]
    """
    if not wavs:
        return []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pad to same length
    max_len = max(w.shape[0] for w in wavs)
    max_len = min(max_len, int(3.0 * 16000))
    batch = torch.zeros(len(wavs), max(max_len, 400), device=device)
    for i, w in enumerate(wavs):
        w = w.to(device)
        if w.shape[0] > max_len:
            w = w[:max_len]
        batch[i, :w.shape[0]] = w

    _, lin_energy = _mel_spec_batch(batch)

    snr = _snr_estimate(lin_energy)
    dropout = _band_dropout(lin_energy)
    flat = _spectral_flatness(lin_energy)

    # 加权融合
    scores = 0.4 * snr + 0.4 * dropout + 0.2 * flat
    return scores.cpu().tolist()


def compute_damage_score(wav):
    """Single waveform convenience wrapper."""
    return compute_damage_score_batch([wav])[0]


def classify_damage(score):
    """三档分类: 'clean', 'moderate', 'severe'
    
    基于 dev 数据分布设定的阈值:
      - < 0.30: 干净, AT 主导 (~70% 数据)
      - 0.30~0.42: 中度, AT+AA ensemble (~25%)
      - > 0.42: 严重, seen→AA, unseen→降级 (~5%)
    """
    if score < 0.30:
        return 'clean'
    elif score < 0.42:
        return 'moderate'
    else:
        return 'severe'

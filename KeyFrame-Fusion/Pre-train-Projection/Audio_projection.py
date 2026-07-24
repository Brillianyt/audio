import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T


# ═══════════ 1. 正弦位置编码 ═══════════
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T_a, D]
        return self.pe[:, :x.size(1), :]


# ═══════════ 2. 工业级音频特征提取前端 (Mel-Spectrogram) ═══════════
class IndustrialAudioFrontend(nn.Module):
    """
    音频前端预处理模块：
    1. 动态重采样至目标采样率 (默认 16kHz)
    2. 多通道自动归一化为单通道 (Mono)
    3. 幅度归一化 (防止爆音和音量差异过大)
    4. 提取工业级 Log-Mel 声学特征
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,        # 25ms 帧长
        hop_length: int = 160    # 10ms 帧移
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0
        )

    def forward(self, waveforms: list[torch.Tensor], orig_srs: list[int]) -> torch.Tensor:
        """
        Args:
            waveforms: 音频波形 Tensor 列表，形状不一，如 [C, Length]
            orig_srs: 对应的原始采样率列表
        Returns:
            mel_features: 经过 Padding 后的 Log-Mel 特征 [B, n_mels, T_a]
        """
        processed_mels = []

        for wave, sr in zip(waveforms, orig_srs):
            # A. 多通道转单通道 (均值融合)
            if wave.ndim > 1 and wave.shape[0] > 1:
                wave = torch.mean(wave, dim=0, keepdim=True)
            elif wave.ndim == 1:
                wave = wave.unsqueeze(0)

            # B. 动态重采样 (如果不符合 16kHz)
            if sr != self.sample_rate:
                resampler = T.Resample(orig_freq=sr, new_freq=self.sample_rate).to(wave.device)
                wave = resampler(wave)

            # C. 波形幅值归一化
            max_val = torch.abs(wave).max()
            if max_val > 1e-6:
                wave = wave / max_val

            # D. 提取 Mel 谱图并转 Log 域
            mel = self.mel_transform(wave)  # [1, n_mels, T_frame]
            log_mel = torch.log(torch.clamp(mel, min=1e-5))
            processed_mels.append(log_mel.squeeze(0))  # [n_mels, T_frame]

        # E. 沿时间维度动态 Padding 对齐 Batch
        max_time = max(m.size(1) for m in processed_mels)
        padded_mels = []
        for m in processed_mels:
            pad_len = max_time - m.size(1)
            if pad_len > 0:
                m = F.pad(m, (0, pad_len), value=-11.5129)  # 填入 ln(1e-5) 背景噪声底噪值
            padded_mels.append(m)

        return torch.stack(padded_mels, dim=0)  # [B, n_mels, T_a]


# ═══════════ 3. 完整 Audio Encoder + Proj 模块 ═══════════
class AudioEncoderModule(nn.Module):
    """
    符合 KFC-KWS 论文标准的 Audio 分支：
    1. 工业级声学特征提取 (Log-Mel)
    2. Subsampling Conv1D Downsampling (下采样降维与上下文聚合)
    3. Projection Layer 投射到统一维度 D
    4. + Positional Encoding + Modality Embedding
    5. L2 Normalization 保证模态对齐
    
    输出：E_a^s [B, T_a, D]
    """
    def __init__(
        self,
        dim: int = 128,
        n_mels: int = 80,
        sample_rate: int = 16000,
        max_len: int = 2000
    ):
        super().__init__()
        self.dim = dim

        # 1. 音频预处理前端
        self.frontend = IndustrialAudioFrontend(sample_rate=sample_rate, n_mels=n_mels)

        # 2. 1D 卷积下采样编码器 (通常使用两层步长为2的1D卷积实现时域4倍下采样，提升帧率计算效率)
        self.encoder_conv = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Conv1d(256, 512, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # 3. Audio Projection 映射层 (从 512 维投射到统一维度 D=128)
        self.audio_proj = nn.Linear(512, dim)

        # 4. 位置编码与音频模态向量
        self.pos_encoder = SinusoidalPositionalEncoding(dim=dim, max_len=max_len)
        self.modality_embed = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

    def forward(
        self,
        audio_inputs: list[torch.Tensor] | torch.Tensor,
        sample_rates: list[int] | None = None
    ) -> torch.Tensor:
        """
        支持两种输入形态：
        1. 原始波形列表：list[Tensor]，自动完成重采样、Mono转换与Padding
        2. 已提好的 Mel-Spectrogram Tensor：[B, n_mels, T_raw]
        """
        # Step A: 预处理得到 Log-Mel 特征 [B, n_mels, T_raw]
        if isinstance(audio_inputs, list):
            if sample_rates is None:
                raise ValueError("传入原始波形列表时，必须提供 sample_rates 参数")
            x = self.frontend(audio_inputs, sample_rates)
        else:
            x = audio_inputs

        # Step B: 卷积下采样编码 -> [B, 512, T_a]
        conv_feat = self.encoder_conv(x)

        # Step C: 转换维度送入 Proj 层 -> [B, T_a, 512] -> [B, T_a, D]
        conv_feat = conv_feat.transpose(1, 2)
        projected = self.audio_proj(conv_feat)

        # Step D: 叠加位置编码与音频模态编码 (Summation)
        pos_embed = self.pos_encoder(projected)
        E_out = projected + pos_embed + self.modality_embed

        # Step E: L2 Normalization 归一化，保证与文本/音素向量在同一空间
        return F.normalize(E_out, p=2, dim=-1)


# ═══════════ 4. 单元运行与测试 ═══════════
if __name__ == "__main__":
    audio_module = AudioEncoderModule(dim=128, sample_rate=16000)

    # 模拟真实工程场景：两个采样率不同、长度不一的音频波形 Tensor
    # 音频 1: 44.1kHz 采样率, 3 秒 (多通道 2 channel)
    fake_waveform_1 = torch.randn(2, 44100 * 3)
    # 音频 2: 16kHz 采样率, 2 秒 (单通道 1 channel)
    fake_waveform_2 = torch.randn(1, 16000 * 2)

    waveforms = [fake_waveform_1, fake_waveform_2]
    sample_rates = [44100, 16000]

    # 前向传播，自动完成特征对齐与归一化
    E_out = audio_module(waveforms, sample_rates)

    print("输入音频数量:", len(waveforms))
    print("输出音频特征 E_out 维度:", E_out.shape)  # [2, T_a, 128]
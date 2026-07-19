"""孪生网络：共享 CNN 编码器 + 余弦相似度打分。

流程：
  注册音频、测试音频 -> 各自 log-mel -> 共享 CNN -> L2 归一化向量
  余弦相似度 -> 可学习 scale/bias -> sigmoid 得 posterior
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """两层卷积 + 自适应全局池化 -> embedding。"""

    def __init__(self, n_mels: int, embed_dim: int = 64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, embed_dim)

    def forward(self, x):
        h = self.pool(self.cnn(x)).flatten(1)   # (B, 64)
        emb = self.fc(h)
        return F.normalize(emb, dim=-1)


class SiameseKWS(nn.Module):
    def __init__(self, n_mels: int, embed_dim: int = 64):
        super().__init__()
        self.encoder = Encoder(n_mels, embed_dim)
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, enroll, query):
        e = self.encoder(enroll)
        q = self.encoder(query)
        sim = (e * q).sum(dim=-1)       # 余弦相似度（已 L2 归一化）
        return self.scale * sim + self.bias

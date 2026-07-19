"""Dual-Encoder Training — audio-audio + audio-text independently.

Model A: frozen Whisper → attn-pool → comparison head → PURE pairwise BCE
         (no contrastive word-identity loss — it caused vocab memorization
          and destroyed unseen-word AUC; hard negatives ≥50% of negs)
Model B: Whisper + TextEnc → BCE + alignment

Trains separately, combines at inference.
No fusion MLP, no multi-task entanglement.
"""
import argparse, csv, gc, json, math, os, time, io, zipfile, re as _re
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from config import PATHS


# ═══════════ Config ═══════════

class Config:
    whisper_model: str = "base"
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5
    unfreeze_layers: int = 4

    epochs: int = 15
    lr: float = 3e-4
    batch_size: int = 256
    pos_weight: float = 5.0
    cos_scale: float = 8.0         # moderate scale — high enough to get sigmoid >0.9 for cos>0.3,
                                   # low enough NOT to kill pos gradient at cos=0.2
    num_workers: int = 4
    seed: int = 42
    log_every: int = 50

    train_zip: str = ""; train_csv: str = ""
    dev_seen_zip: str = ""; dev_seen_csv: str = ""
    dev_unseen_zip: str = ""; dev_unseen_csv: str = ""
    eval_seen_zip: str = ""; eval_seen_csv: str = ""
    eval_unseen_zip: str = ""; eval_unseen_csv: str = ""

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")
        db = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(db, "dev")): db = os.path.join(db, "dev")
        self.dev_seen_zip = os.path.join(db, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(db, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(db, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(db, "dev_unseen", "dev_unseen_label.csv")
        self.eval_seen_zip = os.path.join(r, "eval", "eval_seen", "wav.zip")
        self.eval_seen_csv = os.path.join(r, "evalcsv_without_label", "eval_seen_without_label.csv")
        self.eval_unseen_zip = os.path.join(r, "eval", "eval_unseen", "wav.zip")
        self.eval_unseen_csv = os.path.join(r, "evalcsv_without_label", "eval_unseen_without_label.csv")


# ═══════════ Whisper Encoder ═══════════

class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=2, max_sec=1.5):
        super().__init__()
        import whisper, whisper.audio as wa
        self.whisper = whisper.load_model(model_name)
        self.whisper.eval()
        self.dim = self.whisper.dims.n_audio_state
        self.max_sec = max_sec
        if hasattr(self.whisper, 'decoder'): del self.whisper.decoder

        total = len(self.whisper.encoder.blocks)
        self.frozen = total - unfreeze
        for p in self.whisper.encoder.parameters(): p.requires_grad = False
        for i in range(self.frozen, total):
            for p in self.whisper.encoder.blocks[i].parameters(): p.requires_grad = True

        self.attn_linear = nn.Linear(self.dim, self.dim//4)
        self.attn_w = nn.Parameter(torch.randn(self.dim//4, 1))
        self.proj = nn.Linear(self.dim*2, embed_dim)
        self.register_buffer("hann", torch.hann_window(400))
        self.register_buffer("mel_filters", wa.mel_filters("cpu", 80))

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for i in range(self.frozen): self.whisper.encoder.blocks[i].eval()
        return self

    def forward(self, wav):
        device = next(self.whisper.parameters()).device
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        wav = wav.to(device)
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mags = stft[..., :-1].abs()**2
        mel = self.mel_filters.to(mags.device).to(mags.dtype) @ mags
        mel = torch.log10(torch.clamp(mel, min=1e-10))
        mel = torch.maximum(mel, mel.max()-8.0)
        mel = (mel+4.0)/4.0
        enc = self.whisper.encoder
        n_valid = max(1, (wav.shape[-1]+319)//320)
        with torch.no_grad():
            x = F.gelu(enc.conv1(mel)); x = F.gelu(enc.conv2(x))
            x = x.permute(0,2,1)
            pe = enc.positional_embedding
            if pe.dim() == 2: pe = pe[:x.shape[1]].unsqueeze(0)
            else: pe = pe[:, :x.shape[1]]
            x = x + pe
            for blk in enc.blocks[:self.frozen]: x = blk(x)
        for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)[:, :n_valid, :]
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)

    @property
    def sample_rate(self): return 16000


# ═══════════ Text Encoder (simple GRU) ═══════════

PHONEMES = ["AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY","F","G",
            "HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P","R","S","SH",
            "T","TH","UH","UW","V","W","Y","Z","ZH","UNK"]
P2I = {p:i for i,p in enumerate(PHONEMES)}

_cmu = None
def _load_cmu():
    global _cmu
    if _cmu is None: import cmudict; _cmu = cmudict.dict()
    return _cmu
def w2p(w):
    cmu = _load_cmu(); w = w.lower().strip("'s\"-.,!?;:")
    if not w: return []
    pl = cmu.get(w)
    return [_re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []

class CharBiGRUEncoder(nn.Module):
    """Character-level BiGRU with moderate capacity and dropout."""
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.2)
        self.proj = nn.Linear(dim*2, dim)
        self.dropout = nn.Dropout(0.2)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0,self.proj.out_features,device=device),None
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx)
        x = self.dropout(x)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1), None


class PhonemeEncoder(nn.Module):
    """音素级文本编码器: w2p() → embedding → Transformer.

    相比 CharBiGRU:
      - 直接建模发音单元（ARPAbet 音素），对 unseen 词泛化更好
      - Transformer 自注意力捕捉音素间长程依赖
      - cmudict 缺词时 fallback 到 char-level
    """
    def __init__(self, dim=256, nhead=4, num_layers=2):
        super().__init__()
        self.dim = dim
        self.emb = nn.Embedding(len(PHONEMES), dim)
        self.pos_enc = nn.Parameter(torch.randn(1, 64, dim) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead,
                                                dim_feedforward=dim*2,
                                                dropout=0.1, activation='gelu',
                                                batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.proj = nn.Linear(dim, dim)

    def _text_to_phonemes(self, texts):
        """Convert text list to phoneme index sequences, padded to max len."""
        device = next(self.parameters()).device
        batch_size = len(texts)
        seqs = []
        for t in texts:
            t = t.lower().strip("'s\"-.,!?;:")
            words = t.split()
            p_list = []
            for w in words:
                ps = w2p(w)
                if ps:
                    p_list.extend(ps)
                else:
                    # fallback to char-level approximation
                    p_list.extend([f"U{c}" for c in w])
            # map to indices
            idx = [P2I.get(p, P2I["UNK"]) for p in p_list]
            if not idx:
                idx = [P2I["UNK"]]
            seqs.append(idx[:60])  # cap at 60 phonemes

        max_len = max(len(s) for s in seqs) if seqs else 1
        x = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        mask = torch.ones(batch_size, max_len, dtype=torch.bool, device=device)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, device=device)
            mask[i, :len(s)] = False
        return x, mask

    def forward(self, texts):
        if not texts:
            device = next(self.parameters()).device
            return torch.zeros(0, self.dim, device=device), None
        x, mask = self._text_to_phonemes(texts)
        emb = self.emb(x)  # (B, L, D)
        if emb.shape[1] <= self.pos_enc.shape[1]:
            emb = emb + self.pos_enc[:, :emb.shape[1], :]
        h = self.transformer(emb, src_key_padding_mask=mask)
        # Mean pooling over valid (unmasked) positions
        lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        pooled = h.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / lengths
        return F.normalize(self.proj(pooled), dim=-1), None


class WavLMEncoder(nn.Module):
    """WavLM base-plus encoder. Set unfreeze=0 for full freeze + no_grad."""
    def __init__(self, embed_dim=256, unfreeze=2, max_sec=1.5):
        super().__init__()
        from transformers import WavLMModel
        path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wavlm"))
        self.wavlm = WavLMModel.from_pretrained(path, local_files_only=True)
        self.wavlm.eval()
        self.dim = self.wavlm.config.hidden_size
        self.max_sec = max_sec
        for p in self.wavlm.feature_extractor.parameters(): p.requires_grad = False
        total = len(self.wavlm.encoder.layers)
        self.frozen = total - unfreeze
        self.frozen_backbone = (unfreeze == 0)
        for p in self.wavlm.encoder.parameters(): p.requires_grad = False
        for i in range(self.frozen, total):
            for p in self.wavlm.encoder.layers[i].parameters(): p.requires_grad = True
        self.attn_linear = nn.Linear(self.dim, self.dim//4)
        self.attn_w = nn.Parameter(torch.randn(self.dim//4, 1))
        self.proj = nn.Linear(self.dim*2, embed_dim)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.wavlm.feature_extractor.eval()
            for i in range(self.frozen): self.wavlm.encoder.layers[i].eval()
        return self

    def forward(self, wav):
        device = next(self.wavlm.parameters()).device
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        wav = wav.to(device)
        ctx = torch.no_grad() if self.frozen_backbone else torch.enable_grad()
        with ctx:
            out = self.wavlm(wav, output_hidden_states=False, return_dict=True)
            x = out.last_hidden_state.clone()
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)

    @property
    def sample_rate(self): return 16000


# ═══════════ Model A: Audio-Audio (improved) ═══════════

class SimpleCnnEncoder(nn.Module):
    """From-scratch CNN encoder for audio comparison.

    No pretrained components — learns comparison from scratch.
    Input: raw waveform (B, T) → internal log-mel → CNN → L2-normed embedding.
    """
    def __init__(self, embed_dim=128, n_mels=80, max_sec=1.5):
        super().__init__()
        self.max_sec = max_sec
        self.n_mels = n_mels
        self.register_buffer("hann", torch.hann_window(400))
        import torchaudio
        _fb = torchaudio.functional.melscale_fbanks(201, 0, 8000, n_mels, 16000)  # (201, n_mels)
        self.register_buffer("mel_fb", _fb)
        # CNN: 4 conv blocks, each: Conv2d + BN + ReLU + MaxPool
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, embed_dim)

    def forward(self, wav):
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        # STFT → mag → mel → log
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mag = stft[..., :-1].abs().pow(2)  # (B, F, T)
        # mel: (B, T, F) @ (F, n_mels) → (B, T, n_mels) → permute
        mel = (mag.permute(0, 2, 1) @ self.mel_fb.to(mag.device)).permute(0, 2, 1)
        mel = mel.clamp(min=1e-10).log().unsqueeze(1)  # (B, 1, n_mels, T)
        h = self.cnn(mel)                     # (B, 256, H, W)
        h = self.pool(h).flatten(1)           # (B, 256)
        emb = self.fc(h)                      # (B, embed_dim)
        return F.normalize(emb, dim=-1)

    @property
    def sample_rate(self):
        return 16000


class AudioAudioModel(nn.Module):
    """Audio-Audio siamese with deep comparison head.

    Supports encoders: whisper, wavlm, simple.
    'simple' = from-scratch CNN (no pretraining) — best for unseen generalization.
    """
    def __init__(self, ckpt_path, embed_dim=256, unfreeze=2, encoder="whisper"):
        super().__init__()
        if encoder == "simple":
            self.encoder = SimpleCnnEncoder(embed_dim=min(128, embed_dim))
            embed_dim = min(128, embed_dim)  # simple encoder uses 128 max
        elif encoder == "wavlm":
            self.encoder = WavLMEncoder(embed_dim, unfreeze)
        else:
            self.encoder = WhisperEncoder("base", embed_dim, unfreeze)
            if unfreeze == 0:
                if ckpt_path:
                    print("  [audio] unfreeze=0 → skip ckpt, use pristine whisper")
            elif ckpt_path and os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                enc_state = {k.replace("encoder.",""): v for k,v in ckpt["model"].items()
                             if k.startswith("encoder.")}
                self.encoder.load_state_dict(enc_state, strict=False)
                print(f"  [audio] loaded whisper weights")
        d = embed_dim
        self.head = nn.Sequential(
            nn.Linear(d * 4, d * 2),
            nn.BatchNorm1d(d * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(d * 2, d),
            nn.BatchNorm1d(d),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.05),
            nn.Linear(d, d // 2),
            nn.BatchNorm1d(d // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(d // 2, 1),
        )

    def forward(self, e, q):
        ea = self.encoder(e); qa = self.encoder(q)
        feat = torch.cat([ea, qa, ea * qa, (ea - qa).abs()], dim=-1)
        logit = self.head(feat).squeeze(-1)
        cos = (ea * qa).sum(-1)
        return cos, logit, ea, qa


# ═══════════ Model C: Whisper Comparison (frame-level) ═══════════

class WhisperComparisonModel(nn.Module):
    """Frame-level comparison using frozen Whisper features.

    不再用Siamese分别编码再比，而是：
    1. 冻结Whisper提取帧级特征 (phonetic posterior-like)
    2. 显式计算 frame-level 差异图 [e, q, |e-q|, e*q]
    3. 1D CNN 学习比较音素序列

    冻结的 Whisper 提供通用语音理解能力，差异图让比较一目了然。
    """
    def __init__(self, embed_dim=256, unfreeze=0, max_sec=1.5):
        super().__init__()
        import whisper, whisper.audio as wa
        self.whisper = whisper.load_model("base")
        self.whisper.eval()
        self.dim = self.whisper.dims.n_audio_state  # 512
        self.max_sec = max_sec
        if hasattr(self.whisper, 'decoder'): del self.whisper.decoder
        for p in self.whisper.encoder.parameters(): p.requires_grad = False
        # Register whisper mel filterbank as buffer
        _mf = wa.mel_filters("cpu", 80)
        self.register_buffer("mel_filters", _mf)
        self.register_buffer("hann", torch.hann_window(400))
        # Comparison: 1D conv on frame-level diff features
        self.compare = nn.Sequential(
            nn.Conv1d(self.dim * 4, 256, 3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(64, 1)

    def _feat(self, wav):
        """Extract whisper frame-level features (frozen)."""
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        device = wav.device
        stft = torch.stft(wav, 400, 160, 400, self.hann.to(device), return_complex=True)
        mags = stft[..., :-1].abs().pow(2)
        mel = (self.mel_filters.to(device) @ mags).clamp(min=1e-10).log10()
        mel = torch.maximum(mel, mel.max() - 8.0)
        mel = (mel + 4.0) / 4.0
        # Ensure whisper is on same device
        self.whisper = self.whisper.to(device)
        x = self.whisper.encoder.conv1(mel)
        x = F.gelu(x)
        x = self.whisper.encoder.conv2(x)
        x = F.gelu(x)
        x = x.permute(0, 2, 1)  # (B, T, 512)
        pe = self.whisper.encoder.positional_embedding
        x = x + pe[:x.shape[1]].unsqueeze(0)
        for blk in self.whisper.encoder.blocks:
            x = blk(x)
        x = self.whisper.encoder.ln_post(x)
        return x  # (B, T, 512)

    def forward(self, e, q):
        ef = self._feat(e)  # (B, Te, 512)
        qf = self._feat(q)  # (B, Tq, 512)
        # Align to same length, center-crop
        T = min(ef.shape[1], qf.shape[1])
        def center(x, T):
            t = x.shape[1]
            s = (t - T) // 2
            return x[:, s:s+T, :]
        ef = center(ef, T)
        qf = center(qf, T)
        # Frame-level difference features
        feat = torch.cat([ef, qf, (ef - qf).abs(), ef * qf], dim=-1)  # (B, T, 2048)
        feat = feat.permute(0, 2, 1)  # (B, 2048, T)
        h = self.compare(feat)  # (B, 64, T)
        h = self.pool(h).flatten(1)
        logit = self.head(h).squeeze(-1)
        return logit, logit, None, None


# ═══════════ Model D: HuBERT Comparison (frame-level) ═══════════

class WavLMSimilarityModel(nn.Module):
    """Similarity matrix comparison using frozen WavLM.

    把 enroll 和 query 的帧级特征投影到低维，计算帧间相似度矩阵 (T1×T2)，
    用 2D CNN 分类"这个匹配模式是同词还是异词"。
    """
    def __init__(self, max_sec=1.5):
        super().__init__()
        from transformers import WavLMModel
        wlm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wavlm"))
        self.wavlm = WavLMModel.from_pretrained(wlm_path, local_files_only=True)
        self.wavlm.eval()
        self.dim = self.wavlm.config.hidden_size  # 768
        self.max_sec = max_sec
        for p in self.wavlm.parameters(): p.requires_grad = False
        # Project noisy 768-dim features to compact 128-dim
        self.proj = nn.Linear(self.dim, 128)
        # 2D CNN on similarity matrix
        self.sim_cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(64, 1)

    def _feat(self, wav):
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        self.wavlm = self.wavlm.to(wav.device)
        with torch.no_grad():
            out = self.wavlm(wav, output_hidden_states=False, return_dict=True)
        return out.last_hidden_state  # (B, T, 768)

    def forward(self, e, q):
        ef = self.proj(self._feat(e))  # (B, T1, 128)
        qf = self.proj(self._feat(q))  # (B, T2, 128)
        # Similarity matrix
        sim = torch.bmm(ef, qf.transpose(1, 2))  # (B, T1, T2)
        sim = sim.unsqueeze(1)  # (B, 1, T1, T2)
        h = self.sim_cnn(sim)
        h = self.pool(h).flatten(1)
        logit = self.head(h).squeeze(-1)
        return logit, logit, None, None


class DirectCompareModel(nn.Module):
    """2-channel spectrogram direct comparison CNN.

    不再用 Siamese 分别编码再比较，而是把 enroll 和 query 的
    log-mel 叠成 2 个通道，让 CNN 从第一层就学习差异模式。
    类似量化交易的"直接建模价差"思路。
    """
    def __init__(self, n_mels=80, max_sec=1.5):
        super().__init__()
        self.max_sec = max_sec
        self.n_mels = n_mels
        self.register_buffer("hann", torch.hann_window(400))
        import torchaudio as _ta
        _fb = _ta.functional.melscale_fbanks(201, 0, 8000, n_mels, 16000)
        self.register_buffer("mel_fb", _fb)

        self.cnn = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1),
        )

    def _preprocess(self, wav):
        """Clean waveform: energy-center crop → RMS norm. Fully vectorized."""
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        B, T = wav.shape
        # 1) Energy-weighted center of mass → crop to ~8000 samples (0.5s)
        frame_len = 160
        n_frames = T // frame_len
        energy = wav[:, :n_frames * frame_len].view(B, -1, frame_len).pow(2).mean(dim=-1)  # (B, n_frames)
        # Find crop window centered on energy centroid
        weights = torch.arange(n_frames, device=wav.device).float() + 0.5
        centroid = (energy * weights).sum(dim=1) / energy.sum(dim=1).clamp(min=1e-8)
        centroid = centroid.long().clamp(0, n_frames - 1)
        crop_frames = min(50, n_frames)  # 0.5s
        half = crop_frames // 2
        start = (centroid - half).clamp(0, n_frames - crop_frames)
        # Gather crop windows
        idx = start.unsqueeze(1) + torch.arange(crop_frames, device=wav.device).unsqueeze(0)  # (B, crop_frames)
        idx = idx * frame_len
        offset = torch.arange(frame_len, device=wav.device).unsqueeze(0).unsqueeze(0)  # (1, 1, FL)
        gather_idx = (idx.unsqueeze(-1) + offset).clamp(0, T - 1)  # (B, crop_frames, FL)
        cropped = wav.gather(1, gather_idx.view(B, -1))  # (B, crop_frames * FL)
        # 2) RMS normalize to 0.1
        rms = cropped.pow(2).mean(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
        return cropped / rms * 0.1

    def _spec(self, wav):
        """Clean log-mel spectrogram with per-file normalization."""
        wav = self._preprocess(wav)
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mag = stft[..., :-1].abs().pow(2)
        mel = (mag.permute(0, 2, 1) @ self.mel_fb.to(mag.device)).permute(0, 2, 1)
        mel = mel.clamp(min=1e-10).log()
        # Per-file mean/std normalization
        mel = (mel - mel.mean(dim=(1,2), keepdim=True)) / mel.std(dim=(1,2), keepdim=True).clamp(min=1e-6)
        return mel.unsqueeze(1)  # (B, 1, n_mels, T)

    def forward(self, e, q):
        me = self._spec(e)                         # (B, 1, n_mels, T_e)
        mq = self._spec(q)                         # (B, 1, n_mels, T_q)
        # Center both around the middle, take the central overlap
        T = min(me.shape[-1], mq.shape[-1])
        # Center-crop both to length T
        def center_crop(x, T):
            _, _, _, t = x.shape
            start = (t - T) // 2
            return x[:, :, :, start:start+T]
        me = center_crop(me, T)
        mq = center_crop(mq, T)
        x = torch.cat([me, mq], dim=1)  # (B, 2, n_mels, T)
        h = self.cnn(x)
        h = self.pool(h).flatten(1)
        logit = self.head(h).squeeze(-1)
        return logit, logit, None, None  # compatible unpacking


# ═══════════ Model B: Audio-Text ═══════════

class AudioTextModel(nn.Module):
    def __init__(self, ckpt_path, embed_dim=256, unfreeze=2,
                 text_encoder="phoneme"):
        super().__init__()
        self.encoder = WhisperEncoder("base", embed_dim, unfreeze)
        if text_encoder == "phoneme":
            self.text_enc = PhonemeEncoder(embed_dim)
        else:
            self.text_enc = CharBiGRUEncoder(embed_dim)
        self.log_var = nn.Parameter(torch.tensor(0.0))
        self.compare = nn.Sequential(
            nn.Linear(embed_dim * 4, embed_dim), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(embed_dim, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            enc_state = {k.replace("encoder.",""): v for k,v in ckpt["model"].items()
                         if k.startswith("encoder.")}
            self.encoder.load_state_dict(enc_state, strict=False)

    def forward(self, e, texts):
        ea = self.encoder(e)
        et, _ = self.text_enc(texts)
        cos = (ea * et).sum(-1)
        feat = torch.cat([ea, et, ea * et, (ea - et).abs()], dim=-1)
        logit = self.compare(feat).squeeze(-1)
        return cos, ea, et, logit


# ═══════════ Model E: Audio-Text V2 (stronger) ═══════════

class AudioTextModelV3(nn.Module):
    """Cross-attention AT: frame-level audio ↔ text alignment."""
    def __init__(self, ckpt_path, embed_dim=256, unfreeze=2):
        super().__init__()
        from transformers import WavLMModel
        wlm_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "wavlm"))
        self.wavlm = WavLMModel.from_pretrained(wlm_path, local_files_only=True)
        self.wavlm.eval()
        self.dim = self.wavlm.config.hidden_size  # 768
        for p in self.wavlm.parameters(): p.requires_grad = False
        self.audio_proj = nn.Linear(self.dim, embed_dim)
        # Text: char emb + positional + transformer
        self.char_emb = nn.Embedding(28, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.1)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*2, dropout=0.1, activation='gelu', batch_first=True)
        self.text_enc = nn.TransformerEncoder(enc_layer, num_layers=2)
        # Cross-attention: text queries attend to audio keys/values
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.logit_scale = nn.Parameter(torch.tensor(8.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))
        if ckpt_path and os.path.isfile(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.load_state_dict(ckpt["model"], strict=False)

    def _audio_feats(self, wav):
        ms = int(1.5 * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        self.wavlm = self.wavlm.to(wav.device)
        with torch.no_grad():
            out = self.wavlm(wav, output_hidden_states=False, return_dict=True)
        return self.audio_proj(out.last_hidden_state)

    def _text_feats(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0, 1, 256, device=device)
        mx = min(max(len(t) for t in texts), 64)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        mask = torch.ones(len(texts), mx, dtype=torch.bool, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = min(27, max(0, ord(c)-97)); mask[i,j] = False
        x = self.char_emb(idx) + self.pos_emb[:, :mx, :]
        return self.text_enc(x, src_key_padding_mask=mask), mask

    def forward(self, e, texts):
        af = self._audio_feats(e)       # (B, Ta, D)
        tf, mask = self._text_feats(texts)  # (B, Tt, D)
        # Cross-attend: text queries attend to audio
        attn_out, _ = self.cross_attn(tf, af, af)  # (B, Tt, D)
        # Mean pool over valid text tokens
        lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
        et = attn_out.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / lengths
        # Compare with pooled audio
        ea = af.mean(dim=1)
        ea = F.normalize(ea, dim=-1); et = F.normalize(et, dim=-1)
        logit = (ea * et).sum(-1)
        return logit, logit, ea, et


# ═══════════ Data ═══════════# ═══════════ Data ═══════════# ═══════════ Data ═══════════# ═══════════ Data ═══════════

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg, mode="audio"):
        self.pairs = pairs; self.zip_path = zip_path; self.cfg = cfg; self.mode = mode
        import zipfile as zf
        self._zip_cache = {}
        self._zip_path = zip_path

    def _get_zip(self, path=None):
        path = path or self._zip_path
        key = (os.getpid(), path)
        if key not in self._zip_cache:
            import zipfile as zf
            self._zip_cache[key] = zf.ZipFile(path, "r")
        return self._zip_cache[key]

    def __len__(self): return len(self.pairs)

    def _read(self, pid, role, zip_path=None):
        z = self._get_zip(zip_path)
        try:
            data = z.read(f"wav/{pid}_{role}.wav")
        except KeyError:
            data = z.read(f"wav/{pid}.wav")   # external single-file utterances
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim>1: wav=wav.mean(axis=1)
        if sr!=16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),sr,16000).squeeze(0).numpy()
        wav=wav.astype(np.float32)
        ms=int(self.cfg.max_audio_sec*16000)
        if len(wav)>ms: wav=wav[:ms]
        return torch.from_numpy(wav).float()

    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        # per-pair zip routing (external corpora live in their own zip)
        zp = os.path.join(PATHS.root, p["zip"]) if p.get("zip") else None
        e = self._read(eid, "enroll", zp); q = self._read(qid, "query", zp)
        label = p.get("label", -1)
        if self.mode == "text":
            txt = p.get("enroll_txt","").lower()
            return e, q, label, txt, pid
        elif self.mode == "audio":
            e_txt = p.get("enroll_txt","").lower()
            q_txt = p.get("query_txt","").lower()
            is_hard = p.get("is_hard", False)
            return e, q, label, e_txt, q_txt, pid, is_hard
        return e, q, label, pid


def collate_audio(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es,qs,ls,e_txts,q_txts,ids,is_hards = [],[],[],[],[],[],[]
    for b in batch:
        e,q=b[0],b[1]
        es.append(F.pad(e,(0,ml-e.shape[-1])) if e.shape[-1]<ml else e)
        qs.append(F.pad(q,(0,ml-q.shape[-1])) if q.shape[-1]<ml else q)
        ls.append(b[2]); e_txts.append(b[3]); q_txts.append(b[4]); ids.append(b[5]); is_hards.append(b[6])
    return (torch.stack(es), torch.stack(qs),
            torch.tensor(ls,dtype=torch.float32), e_txts, q_txts, ids,
            torch.tensor(is_hards, dtype=torch.bool))

def collate_text(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es,qs,ls,txts,ids = [],[],[],[],[]
    for b in batch:
        e,q=b[0],b[1]
        es.append(F.pad(e,(0,ml-e.shape[-1])) if e.shape[-1]<ml else e)
        qs.append(F.pad(q,(0,ml-q.shape[-1])) if q.shape[-1]<ml else q)
        ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return torch.stack(es),torch.stack(qs),torch.tensor(ls,dtype=torch.float32),txts,ids


def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows


# ═══════════ Noise bank (real interference noise, gaussian fallback) ═══════════

_noise_bank = None  # 1-D float32 tensor, all noise files concatenated

def _load_noise_bank():
    """Scan candidate locations for noise wavs.
    Returns a single concatenated 16kHz float32 tensor, or None → gaussian fallback."""
    global _noise_bank
    if _noise_bank is not None:
        return _noise_bank if len(_noise_bank) else None
    r = PATHS.root
    chunks = []

    # 0) Pre-built noise bank from training speech
    prebuilt = os.path.join(r, "train_noise_bank.pt")
    if os.path.isfile(prebuilt):
        try:
            _noise_bank = torch.load(prebuilt, map_location="cpu", weights_only=True)
            print(f"  [noise] loaded pre-built noise bank: {len(_noise_bank)/16000:.0f}s")
            return _noise_bank
        except Exception:
            pass

    # 1) plain dirs of wavs
    import glob as _glob
    for d in ["train/noise", "noise", "noises", "train/noises", "datasets/noise"]:
        for fp in sorted(_glob.glob(os.path.join(r, d, "*.wav")))[:500]:
            try:
                w, sr = sf.read(fp, dtype="float32", always_2d=False)
                if w.ndim > 1: w = w.mean(axis=1)
                if sr != 16000:
                    import torchaudio
                    w = torchaudio.functional.resample(
                        torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
                chunks.append(torch.from_numpy(w.astype(np.float32)))
            except Exception:
                pass
    # 2) noise zip candidates
    for zp in ["train/noise.zip", "noise.zip"]:
        zpath = os.path.join(r, zp)
        if not os.path.isfile(zpath): continue
        try:
            with zipfile.ZipFile(zpath) as z:
                for n in z.namelist()[:500]:
                    if not n.lower().endswith(".wav"): continue
                    w, sr = sf.read(io.BytesIO(z.read(n)), dtype="float32", always_2d=False)
                    if w.ndim > 1: w = w.mean(axis=1)
                    if sr != 16000:
                        import torchaudio
                        w = torchaudio.functional.resample(
                            torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
                    chunks.append(torch.from_numpy(w.astype(np.float32)))
        except Exception:
            pass
    if chunks:
        _noise_bank = torch.cat(chunks)
        print(f"  [noise] real noise bank: {len(chunks)} files, "
              f"{len(_noise_bank)/16000:.0f}s total")
    else:
        _noise_bank = torch.zeros(0)
        print("  [noise] no noise wavs found → gaussian fallback")
    return _noise_bank if len(_noise_bank) else None


def mix_noise_batch(wav, bank, p_clean=0.5, snr_lo=-10.0, snr_hi=5.0):
    """Per-sample noise mixing, fully vectorized on wav's device.

    Each sample: with prob p_clean stays clean; else mixed with a random
    slice of the noise bank (or gaussian noise) at SNR ~ U(snr_lo, snr_hi).
    wav: (B, L) float tensor. Returns (B, L).
    """
    B, L = wav.shape
    device = wav.device
    use = torch.rand(B, device=device) >= p_clean          # True = add noise
    if not use.any():
        return wav
    if bank is not None:
        nb = bank.to(device)
        if len(nb) <= L:                                   # bank too short: tile
            nb = nb.repeat((L // max(1, len(nb))) + 1)
        off = torch.randint(0, len(nb) - L, (B,), device=device)
        idx = off.unsqueeze(1) + torch.arange(L, device=device).unsqueeze(0)
        noise = nb[idx]
    else:
        noise = torch.randn_like(wav)
    snr = torch.rand(B, 1, device=device) * (snr_hi - snr_lo) + snr_lo
    sig_p = wav.pow(2).mean(-1, keepdim=True).clamp(min=1e-8)
    noi_p = noise.pow(2).mean(-1, keepdim=True).clamp(min=1e-8)
    scale = (sig_p / (noi_p * 10 ** (snr / 10.0))).sqrt()
    return torch.where(use.unsqueeze(1), wav + noise * scale, wav)


# ═══════════ Training: Audio-Audio (SupCon) ═══════════

def train_audio(cfg, args):
    """Model A: audio-audio — PURE pairwise BCE on the comparison head.

    Design (post-mortem of the SupCon experiment):
    - NO word-identity contrastive loss: it makes the encoder memorise the
      8357 train words instead of learning to *compare*, killing unseen AUC.
    - Recommend --unfreeze 0 (frozen pristine whisper): pretrained features
      already carry phonetic information; only pool+head are trained.
    - Hard negatives are ≥50% of every epoch's negatives.
    - Per-sample noise, asymmetric: query 50% clean / SNR~U(-10,5) (matches
      eval), enroll 75% clean / light U(0,5) (registration is usually clean).
      Real interference noise, gaussian fallback if no noise files found.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", f"dual_{args.name}_audio")
    os.makedirs(out_dir, exist_ok=True)

    # ── pair pools: base csv + extra jsons, split by role ──
    base_pairs = load_pairs(cfg.train_csv)
    pos_pairs = [p for p in base_pairs if p["label"] == 1]
    easy_neg = [p for p in base_pairs if p["label"] == 0]
    hard_neg = []
    for hn in ["baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
               "baseline/hard_neg_iter2.json", "train/self_paired.json",
               "train/external_pairs.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp):
            with open(hp) as f:
                extra = json.load(f)
            pos_pairs += [p for p in extra if p["label"] == 1]
            for p in extra:
                if p["label"] == 0:
                    p["is_hard"] = True
                    hard_neg.append(p)
            print(f"  + {os.path.basename(hn)}: {len(extra)} pairs")
    # Deduplicate hard negatives: same word combo → 1 sample
    _seen_combo = set()
    hard_neg_deduped = []
    for p in hard_neg:
        key = (p.get("enroll_txt","").lower(), p.get("query_txt","").lower())
        if key not in _seen_combo:
            _seen_combo.add(key)
            hard_neg_deduped.append(p)
    hard_neg = hard_neg_deduped
    rng = np.random.default_rng(cfg.seed)
    print(f"[audio] pos={len(pos_pairs)} easy_neg={len(easy_neg)} hard_neg={len(hard_neg)} (deduped)")

    # ── Unique word-pair pools (no word pair repeats across epochs) ──
    def _group_by_word(pairs):
        groups = {}
        for p in pairs:
            key = (p.get("enroll_txt","").lower(), p.get("query_txt","").lower())
            groups.setdefault(key, []).append(p)
        return groups

    pos_groups = _group_by_word(pos_pairs)
    hard_neg_groups = _group_by_word(hard_neg)
    easy_neg_groups = _group_by_word(easy_neg)
    pos_keys = list(pos_groups.keys())
    hard_neg_keys = list(hard_neg_groups.keys())
    easy_neg_keys = list(easy_neg_groups.keys())
    rng.shuffle(pos_keys)
    rng.shuffle(hard_neg_keys)
    rng.shuffle(easy_neg_keys)
    # Per-epoch pointers (pop from list, refill when exhausted)
    _pp, _hp, _ep = 0, 0, 0
    print(f"[audio] unique word pairs: pos={len(pos_keys)} hard_neg={len(hard_neg_keys)} easy_neg={len(easy_neg_keys)}")

    def get_loader(ep):
        nonlocal _pp, _hp, _ep
        n_pos = min(100000, len(pos_keys))
        n_neg = min(400000, len(hard_neg_keys) + len(easy_neg_keys))
        n_hard = min(n_neg // 2, len(hard_neg_keys))
        n_easy = n_neg - n_hard

        def _sample(groups, keys, n, ptr):
            """Pick n unique word-pair keys, cycling through shuffled list."""
            if len(keys) == 0:
                return [], ptr
            chosen = []
            for _ in range(n):
                key = keys[ptr % len(keys)]
                ptr += 1
                pool = groups[key]
                chosen.append(rng.choice(pool))  # pick random audio pair for this word
            return chosen, ptr

        subset, _pp = _sample(pos_groups, pos_keys, n_pos, _pp)
        hn, _hp = _sample(hard_neg_groups, hard_neg_keys, n_hard, _hp)
        en, _ep = _sample(easy_neg_groups, easy_neg_keys, n_easy, _ep)
        subset += hn + en
        rng.shuffle(subset)
        ds = PairDataset(subset, cfg.train_zip, cfg, "audio")
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_audio,
                          pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg, "audio"),
                          batch_size=cfg.batch_size * 2, num_workers=0,
                          collate_fn=collate_audio, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    if args.encoder == "direct":
        model = DirectCompareModel().to(device)
        is_direct = True
    elif args.encoder == "whisper_compare":
        model = WhisperComparisonModel(unfreeze=cfg.unfreeze_layers).to(device)
        is_direct = True
    elif args.encoder in ("hubert", "wavlm_sim"):
        model = WavLMSimilarityModel().to(device)
        is_direct = True
    else:
        model = AudioAudioModel(args.load_ckpt, cfg.embed_dim, cfg.unfreeze_layers,
                                encoder=args.encoder).to(device)
        is_direct = False
    best, start_ep = -1.0, 1
    if args.resume:
        latest = os.path.join(out_dir, "latest.pt")
        best_pt = os.path.join(out_dir, "best.pt")
        ckpt_path = latest if os.path.isfile(latest) else (best_pt if os.path.isfile(best_pt) else None)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            best = ckpt.get("auc_unseen", -1.0)
            start_ep = ckpt.get("epoch", 0)
            if start_ep == 0: start_ep = 1  # old checkpoints lack epoch
            start_ep += 1
            print(f"  [resume] from {os.path.basename(ckpt_path)} epoch={start_ep} best_unseen={best:.4f}")
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    # with frozen encoder only pool+head train → plain AdamW is fine
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=1e-4)

    # class imbalance 1:4 → pos_weight so the head sees a balanced signal
    # With unfrozen encoder, use gentler pos_weight to avoid overfitting to seen words
    pp = 3.0 if cfg.unfreeze_layers > 0 else 4.0
    pos_weight = torch.tensor([pp], device=device)

    noise_bank = _load_noise_bank()
    stats_file = os.path.join(out_dir, "cos_dist.jsonl")

    # Gradient clipping to stabilise unfrozen Whisper training
    max_grad_norm = 1.0 if cfg.unfreeze_layers > 0 else 5.0

    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls, n = 0.0, 0
        cs_pos, cs_neg, cs_n = 0.0, 0.0, 0
        ls_pos, ls_neg = 0.0, 0.0
        cs_pos_std, cs_neg_std = 0.0, 0.0
        all_cos_pos, all_cos_neg = [], []

        for e, q, y, e_txts, q_txts, _, is_hard in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            is_hard = is_hard.to(device)

            # ── per-sample noise, ASYMMETRIC (matches eval scenario):
            #    query  = test audio in the wild → 50% clean, SNR~U(-10,5)
            #    enroll = pre-recorded registration → 75% clean, light U(0,5)
            e = mix_noise_batch(e, noise_bank, p_clean=0.75, snr_lo=0.0, snr_hi=5.0)
            q = mix_noise_batch(q, noise_bank, p_clean=0.50, snr_lo=-10.0, snr_hi=5.0)

            # ── forward ──
            cos_sim, logit, ea, qa = model(e, q)               # cos:(B,), logit:(B,)

            # ── BCE loss with hard-negative focussing ──
            bce = F.binary_cross_entropy_with_logits(logit, y, reduction='none', pos_weight=pos_weight)
            with torch.no_grad():
                pt = torch.sigmoid(logit)
                hardness = 1.0 - (pt * y + (1 - pt) * (1 - y))
                hardness = hardness ** 0.5
            loss_bce = (bce * (1.0 + hardness)).mean()

            loss = loss_bce
            if not is_direct and ea is not None:
                # ── Siamese contrastive loss (pos + hard neg only) ──
                cos_emb = (ea * qa).sum(-1)
                contr_mask = y.bool() | is_hard
                if contr_mask.any():
                    yc = y[contr_mask].float() * 2.0 - 1.0
                    loss_contrast = (1.0 - yc * cos_emb[contr_mask]).mean()
                    loss = loss_bce + 0.3 * loss_contrast

            # ── monitoring ──
            pos = (y == 1); neg = (y == 0)
            if is_direct:
                # Direct model: monitor logit-based metrics
                prob = torch.sigmoid(logit)
                cp_val = prob[pos].mean().item() if pos.any() else 0.0
                cn_val = prob[neg].mean().item() if neg.any() else 0.0
                cs_n += 1
            else:
                _cp = cos_sim[pos]; _cn = cos_sim[neg]
                cp_val = _cp.mean().item() if pos.any() else 0.0
                cn_val = _cn.mean().item() if neg.any() else 0.0
                cs_pos_std += _cp.std().item() if pos.any() else 0.0
                cs_neg_std += _cn.std().item() if neg.any() else 0.0
                all_cos_pos.append(_cp.detach().cpu())
                all_cos_neg.append(_cn.detach().cpu())
            lp_val = logit[pos].mean().item() if pos.any() else 0.0
            ln_val = logit[neg].mean().item() if neg.any() else 0.0
            cs_pos += cp_val; cs_neg += cn_val
            ls_pos += lp_val; ls_neg += ln_val

            if cs_n % 50 == 0:
                tag = "prob" if is_direct else "cos"
                print(f"  [ep{ep}] b{cs_n} loss={ls/n:.3f} "
                      f"{tag}+={cs_pos/cs_n:.3f} {tag}-={cs_neg/cs_n:.3f} "
                      f"head+={ls_pos/cs_n:+.2f} head-={ls_neg/cs_n:+.2f}")

            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            ls += loss.item(); n += 1
            if is_direct:
                del cos_sim, logit, loss, bce
            else:
                del cos_sim, logit, ea, qa, loss, bce, _cp, _cn

        # ── eval (head logit → sigmoid → AUC; same as inference) ──
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls_list, ids, all_cs = [], [], [], []
            for batch in ld:
                e, q = batch[0].to(device), batch[1].to(device)
                cs, logit, _, _ = model(e, q)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls_list.append(batch[2].numpy()); ids.extend(batch[5])
                all_cs.append(cs.cpu().numpy())
            return (roc_auc_score(np.concatenate(ls_list), np.concatenate(ps)),
                    np.concatenate(ps), np.concatenate(ls_list), ids,
                    np.concatenate(all_cs))

        as_, ps_s, ls_s, ids_s, css_s = ev(dv_s)
        au_, ps_u, ls_u, ids_u, css_u = ev(dv_u)

        # ── epoch-level diagnostic ──
        if is_direct:
            # Direct model: use prob-based monitoring
            print(f"[audio ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
                  f"loss={ls/n:.4f} prob+={cs_pos/cs_n:.3f} prob-={cs_neg/cs_n:.3f} "
                  f"({time.time() - ts:.0f}s)")
        else:
            all_cp = torch.cat(all_cos_pos).numpy()
            all_cn = torch.cat(all_cos_neg).numpy()

            def _hist(arr, bins=20):
                if len(arr) == 0: return [0] * bins
                h, _ = np.histogram(arr, bins=bins, range=(-0.5, 1.0))
                return h.astype(int).tolist()

            def _pct(arr, lo, hi):
                if len(arr) == 0: return 0.0
                return float(((arr >= lo) & (arr < hi)).mean())

            dist = {
                "epoch": ep,
                "cos_pos": {"mean": float(all_cp.mean()), "std": float(all_cp.std()),
                            "p05": float(np.percentile(all_cp, 5)),
                            "p50": float(np.percentile(all_cp, 50)),
                            "p95": float(np.percentile(all_cp, 95)),
                            "n": len(all_cp), "hist": _hist(all_cp)},
                "cos_neg": {"mean": float(all_cn.mean()), "std": float(all_cn.std()),
                            "p05": float(np.percentile(all_cn, 5)),
                            "p50": float(np.percentile(all_cn, 50)),
                            "p95": float(np.percentile(all_cn, 95)),
                            "n": len(all_cn), "hist": _hist(all_cn)},
                "overlap_gt_0": _pct(all_cn, 0.0, 1.0),
                "overlap_gt_pos_med": _pct(all_cn, float(np.median(all_cp)), 1.0),
            }
            with open(stats_file, "a") as f:
                f.write(json.dumps(dist) + "\n")

            print(f"[audio ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
                  f"loss={ls/n:.4f} cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} "
                  f"cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} "
                  f"overlap(>0)={dist['overlap_gt_0']:.3f} ({time.time() - ts:.0f}s)")
            print(f"  cos+ [p5={dist['cos_pos']['p05']:.3f} "
                  f"p50={dist['cos_pos']['p50']:.3f} p95={dist['cos_pos']['p95']:.3f}]")
            print(f"  cos- [p5={dist['cos_neg']['p05']:.3f} "
                  f"p50={dist['cos_neg']['p50']:.3f} p95={dist['cos_neg']['p95']:.3f}]")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_,
                    "auc_seen": as_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))


# ═══════════ Training: Audio-Text (BCE + alignment) ═══════════

def train_text(cfg, args):
    """Model B: audio-text cosine matching."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", f"dual_{args.name}_text")
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
               "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
               "train/self_paired.json", "train/external_pairs.json",
               "train/external_pairs_v2.json", "train/self_paired_xl.json", "train/fill_pos_pairs.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp): all_pairs += json.load(open(hp))
    rng.shuffle(all_pairs)
    print(f"[text] {len(all_pairs)} pairs")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    rng.shuffle(pos_pairs); rng.shuffle(neg_pairs)
    print(f"[text] total={len(all_pairs)} pos={len(pos_pairs)} neg={len(neg_pairs)}")

    def get_loader(ep):
        n_pos = min(100000, len(pos_pairs))
        n_neg = min(400000, len(neg_pairs))
        subset = rng.choice(pos_pairs, n_pos, replace=False).tolist()
        subset += rng.choice(neg_pairs, n_neg, replace=False).tolist()
        rng.shuffle(subset)
        ds = PairDataset(subset, cfg.train_zip, cfg, "text")
        return DataLoader(ds, batch_size=512, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_text,
                          pin_memory=True, drop_last=True)

    def dev_ld(z,c):
        return DataLoader(PairDataset(load_pairs(c),z,cfg,"text"),
                          batch_size=cfg.batch_size*2,num_workers=0,
                          collate_fn=collate_text,shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    if args.model_version == 2:
        model = AudioTextModelV2(args.load_ckpt, cfg.embed_dim,
                                  text_encoder=args.text_encoder).to(device)
    else:
        model = AudioTextModel(args.load_ckpt, cfg.embed_dim,
                                text_encoder=args.text_encoder).to(device)
    best, start_ep = -1.0, 1
    if args.resume:
        latest = os.path.join(out_dir, "latest.pt")
        best_pt = os.path.join(out_dir, "best.pt")
        # Always prefer latest.pt — SupCon training AUC is a poor metric,
        # latest.pt has the most recent weights regardless of dev AUC.
        ckpt_path = latest if os.path.isfile(latest) else (best_pt if os.path.isfile(best_pt) else None)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            # If latest.pt AUC is much worse than best.pt, warn but still use latest
            if ckpt_path == latest and os.path.isfile(best_pt):
                ckpt_best = torch.load(best_pt, map_location=device, weights_only=False)
                if ckpt_best.get("auc_unseen", -1) > ckpt.get("auc_unseen", -1):
                    print(f"  [resume] note: best.pt AUC={ckpt_best['auc_unseen']:.4f} > latest.pt AUC={ckpt['auc_unseen']:.4f}")
            model.load_state_dict(ckpt["model"], strict=False)
            best = ckpt.get("auc_unseen", -1.0)
            start_ep = ckpt.get("epoch", 0)
            if start_ep == 0: start_ep = 1  # old checkpoints lack epoch
            start_ep += 1
            print(f"  [resume] from {os.path.basename(ckpt_path)} epoch={start_ep} best_unseen={best:.4f}")
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params":wav_p,"lr":cfg.lr/5},{"params":oth_p,"lr":1e-3}], weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    scaler = torch.amp.GradScaler('cuda')  # mixed precision ~2x speed

    stats_file = os.path.join(out_dir, "cos_dist.jsonl")
    for ep in range(start_ep, cfg.epochs+1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls,n=0.0,0
        cs_pos, cs_neg, cs_n = 0.0, 0.0, 0
        cs_pos_std, cs_neg_std = 0.0, 0.0
        all_cos_pos, all_cos_neg = [], []
        for e,q,y,txts,_ in loader:
            e,q,y = e.to(device),q.to(device),y.to(device)
            opt.zero_grad()
            with torch.amp.autocast('cuda'):
                snr = np.random.uniform(-10,5); nl=10**(-snr/20)
                e = e+nl*torch.randn_like(e)*e.std(-1,keepdim=True)
                snr = np.random.uniform(-10,5); nl=10**(-snr/20)
                q = q+nl*torch.randn_like(q)*q.std(-1,keepdim=True)

                cos_ae, ea, et, logit = model(e, txts)
                cos_tq = (et * model.encoder(q)).sum(-1)
                # Combined loss: comparison head BCE + cosine alignment
                loss = crit(logit, y)
                # Cosine margin loss on emb space
                margin = torch.tensor(0.0, device=device)
                pos = (y==1); neg = (y==0)
                _cp = cos_tq[pos]; _cn = cos_tq[neg]
                if pos.any(): margin += F.relu(0.6 - _cp).mean()
                if neg.any(): margin += F.relu(_cn + 0.15).mean()
                margin += F.relu(0.6 - cos_ae).mean()
                loss = loss + 0.05 * margin + 0.1 * F.mse_loss(et, ea)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            cos_pos_val = _cp.mean().item() if pos.any() else 0.0
            cos_neg_val = _cn.mean().item() if neg.any() else 0.0
            ls+=loss.item(); n+=1
            cs_pos += cos_pos_val; cs_neg += cos_neg_val; cs_n += 1
            cs_pos_std += _cp.std().item() if pos.any() else 0
            cs_neg_std += _cn.std().item() if neg.any() else 0
            all_cos_pos.append(_cp.detach().cpu()); all_cos_neg.append(_cn.detach().cpu())
            if cs_n % 50 == 0:
                print(f"  [ep{ep}] b{cs_n} loss={ls/n:.3f} cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} "
                      f"cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} gap={cs_pos/cs_n-cs_neg/cs_n:.3f}")
            del cos_ae,cos_tq,ea,et,e,q,y,loss,margin,_cp,_cn

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps,ls,ids,all_cs=[],[],[],[]
            for e,q,y,txts,id_ in ld:
                e,q=e.to(device),q.to(device)
                _, _, et, logit = model(e, txts)
                cs = (et * model.encoder(q)).sum(-1)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls.append(y.numpy()); ids.extend(id_)
                all_cs.append(cs.cpu().numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps)), np.concatenate(ps), np.concatenate(ls), ids, np.concatenate(all_cs)

        as_, ps_s, ls_s, ids_s, css_s = ev(dv_s)
        au_, ps_u, ls_u, ids_u, css_u = ev(dv_u)

        # ── epoch-level cosine distribution diagnostic ──
        all_cp = torch.cat(all_cos_pos).numpy(); all_cn = torch.cat(all_cos_neg).numpy()
        def _hist(arr, bins=20):
            if len(arr)==0: return [0]*bins
            h,_ = np.histogram(arr, bins=bins, range=(-0.5,1.0))
            return h.astype(int).tolist()
        def _pct(arr, lo, hi):
            if len(arr)==0: return 0.0
            return float(((arr>=lo)&(arr<hi)).mean())
        dist = {
            "epoch": ep,
            "cos_pos": {"mean": float(all_cp.mean()), "std": float(all_cp.std()),
                        "p05": float(np.percentile(all_cp,5)), "p50": float(np.percentile(all_cp,50)),
                        "p95": float(np.percentile(all_cp,95)), "n": len(all_cp),
                        "hist": _hist(all_cp)},
            "cos_neg": {"mean": float(all_cn.mean()), "std": float(all_cn.std()),
                        "p05": float(np.percentile(all_cn,5)), "p50": float(np.percentile(all_cn,50)),
                        "p95": float(np.percentile(all_cn,95)), "n": len(all_cn),
                        "hist": _hist(all_cn)},
            "overlap_gt_0": _pct(all_cn, 0.0, 1.0),
            "overlap_gt_pos_med": _pct(all_cn, float(np.median(all_cp)), 1.0),
        }
        with open(stats_file, "a") as f: f.write(json.dumps(dist)+"\n")

        print(f"[text ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={ls/n:.4f} cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} "
              f"cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} "
              f"overlap(>0)={dist['overlap_gt_0']:.3f} ({time.time()-ts:.0f}s)")
        print(f"  cos+ [p5={dist['cos_pos']['p05']:.3f} p50={dist['cos_pos']['p50']:.3f} p95={dist['cos_pos']['p95']:.3f}]")
        print(f"  cos- [p5={dist['cos_neg']['p05']:.3f} p50={dist['cos_neg']['p50']:.3f} p95={dist['cos_neg']['p95']:.3f}]")

        if au_>best:
            best=au_
            torch.save({"model":model.state_dict(),"auc_unseen":au_,"auc_seen":as_,"epoch":ep},
                       os.path.join(out_dir,"best.pt"))
        torch.save({"model":model.state_dict(),"auc_unseen":au_,"auc_seen":as_,"epoch":ep},
                   os.path.join(out_dir,"latest.pt"))


# ═══════════ Main ═══════════

def main():
    cfg = Config(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="default")
    p.add_argument("--mode", default="audio", choices=["audio","text","both"])
    p.add_argument("--load-ckpt", default="", help="whisper checkpoint to init encoder")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--encoder", default="whisper", choices=["whisper","wavlm","simple","direct","whisper_compare","hubert","wavlm_sim"])
    p.add_argument("--unfreeze", type=int, default=None, help="override unfreeze_layers (0=full freeze)")
    p.add_argument("--text-encoder", default="char", choices=["char", "phoneme"],
                   help="text encoder type for AudioTextModel")
    p.add_argument("--model-version", type=int, default=1, choices=[1, 2],
                   help="1=original AudioTextModel, 2=AudioTextModelV2 (multi-layer+fusion)")
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    if args.unfreeze is not None: cfg.unfreeze_layers = args.unfreeze

    if args.mode in ("audio","both"): train_audio(cfg, args)
    if args.mode in ("text","both"): train_text(cfg, args)


if __name__ == "__main__":
    main()
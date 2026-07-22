"""Dual-Encoder Training — audio-audio + audio-text independently.

Model A: Whisper → SupCon (supervised contrastive) on all batch embeddings
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

import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS


# ═══════════ Config ═══════════

class Config:
    whisper_model: str = "base"
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 2.5  # padded audio: ~0.8s + 1s silence
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
    """Character-level BiGRU: 前后依赖 + 轻量 (~100K params)."""
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0,256,device=device),None
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)  # last fwd + last bwd
        return F.normalize(self.proj(h), dim=-1), None


class CharBiGRU64(nn.Module):
    """CharBiGRU_64: original AT v2 architecture (64d emb, 1-layer GRU)."""
    def __init__(self, dim=256):
        super().__init__()
        self.char_emb = nn.Embedding(28, 64)
        self.gru = nn.GRU(64, 64, batch_first=True, bidirectional=True, num_layers=1)
        self.proj = nn.Linear(128, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0,256,device=device),None
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1), None


class WavLMEncoder(nn.Module):
    """WavLM base-plus encoder (same arch as whisper_v3)."""
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
        out = self.wavlm(wav, output_hidden_states=False, return_dict=True)
        x = out.last_hidden_state
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)

    @property
    def sample_rate(self): return 16000


# ═══════════ Model A: Audio-Audio ═══════════

class AudioAudioModel(nn.Module):
    def __init__(self, ckpt_path, embed_dim=256, unfreeze=2, encoder="whisper"):
        super().__init__()
        if encoder == "wavlm":
            self.encoder = WavLMEncoder(embed_dim, unfreeze)
        else:
            self.encoder = WhisperEncoder("base", embed_dim, unfreeze)
            if ckpt_path and os.path.isfile(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                enc_state = {k.replace("encoder.",""): v for k,v in ckpt["model"].items()
                             if k.startswith("encoder.")}
                self.encoder.load_state_dict(enc_state, strict=False)
                print(f"  [audio] loaded whisper weights")
        # Comparison head: [ea, qa, ea*qa, |ea-qa|] → logit
        d = embed_dim
        self.head = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(0.1),
            nn.Linear(d, 64),
            nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

    def forward(self, e, q):
        ea = self.encoder(e); qa = self.encoder(q)
        feat = torch.cat([ea, qa, ea * qa, (ea - qa).abs()], dim=-1)
        logit = self.head(feat).squeeze(-1)
        cos = (ea * qa).sum(-1)
        return cos, logit, ea, qa


# ═══════════ Model B: Audio-Text ═══════════

class AudioTextModel(nn.Module):
    def __init__(self, ckpt_path, embed_dim=256, unfreeze=2, small_te=False, use_phoneme=False):
        super().__init__()
        self.encoder = WhisperEncoder("base", embed_dim, unfreeze)
        if use_phoneme:
            self.text_enc = PhonemeBiGRUEncoder(embed_dim)
        elif small_te:
            self.text_enc = CharBiGRU64(embed_dim)
        else:
            self.text_enc = CharBiGRUEncoder(embed_dim)
        # Comparison head: [ea, et, ea*et, |ea-et|] → logit (same as AA)
        d = embed_dim
        self.head = nn.Sequential(
            nn.Linear(d * 4, d),
            nn.ReLU(), nn.BatchNorm1d(d), nn.Dropout(0.1),
            nn.Linear(d, 64),
            nn.ReLU(), nn.BatchNorm1d(64),
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
        feat = torch.cat([ea, et, ea * et, (ea - et).abs()], dim=-1)
        logit = self.head(feat).squeeze(-1)
        cos = (ea * et).sum(-1)
        return cos, logit, ea, et


# ═══════════ PhonemeBiGRU ═══════════

class PhonemeBiGRUEncoder(nn.Module):
    """CMU phoneme → BiGRU text encoder."""
    def __init__(self, dim=256):
        super().__init__()
        self.emb = nn.Embedding(40, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        import re, cmudict as _cd
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0, 256, device=device), None
        cmu = _cd.dict()
        def w2p(w):
            w = w.lower().strip("'s\"-.,!?;:")
            if not w: return []
            pl = cmu.get(w)
            return [re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []
        PV = ["AA","AE","AH","AO","AW","AY","B","CH","D","DH","EH","ER","EY","F","G",
              "HH","IH","IY","JH","K","L","M","N","NG","OW","OY","P","R","S","SH",
              "T","TH","UH","UW","V","W","Y","Z","ZH","UNK"]
        P2I = {p:i for i,p in enumerate(PV)}
        ph_lists = [[P2I.get(p,39) for p in w2p(t)] or [39] for t in texts]
        mx = max(len(p) for p in ph_lists)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, ph in enumerate(ph_lists):
            for j, pid in enumerate(ph[:mx]): idx[i,j] = pid
        x = self.emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1), None


# ═══════════ SupCon Loss ═══════════

def supcon_loss(embeddings, labels, temperature=0.07):
    """
    Supervised contrastive loss (Khosla et al., NeurIPS 2020).
    For each anchor, pulls same-class embeddings closer and pushes different-class
    embeddings apart — all within a single batch.

    Gradient is naturally higher for hard negatives (those with high cosine),
    so no manual margin or pos_weight is needed.

    embeddings: (N, D) L2-normalized
    labels:     (N,) integer class IDs
    Returns scalar loss.
    """
    N = embeddings.shape[0]
    cos = (embeddings @ embeddings.T) / temperature           # (N, N)

    same = (labels.unsqueeze(0) == labels.unsqueeze(1))       # (N, N)
    same.fill_diagonal_(False)                                 # exclude self

    exp_cos = cos.exp()                                        # (N, N)
    num = (exp_cos * same.float()).sum(dim=1)                  # (N,)
    denom = exp_cos.sum(dim=1) - exp_cos.diag()               # (N,)

    has_pos = same.sum(dim=1) > 0
    if has_pos.sum() == 0:
        return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    return -torch.log((num[has_pos] + 1e-8) / (denom[has_pos] + 1e-8)).mean()


# ═══════════ Data ═══════════

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg, mode="audio"):
        self.pairs = pairs; self.zip_path = zip_path; self.cfg = cfg; self.mode = mode
        import zipfile as zf
        self._zip_cache = {}
        self._zip_path = zip_path

    def _get_zip(self):
        pid = os.getpid()
        if pid not in self._zip_cache:
            import zipfile as zf
            self._zip_cache[pid] = zf.ZipFile(self._zip_path, "r")
        return self._zip_cache[pid]

    def __len__(self): return len(self.pairs)

    def _read(self, pid, role):
        data = self._get_zip().read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim>1: wav=wav.mean(axis=1)
        if sr!=16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),sr,16000).squeeze(0).numpy()
        wav=wav.astype(np.float32)
        ms=int(self.cfg.max_audio_sec*16000)
        if len(wav)>ms: wav=wav[:ms]
        wav=torch.from_numpy(wav).float()
        pad=int(0.5*16000); wav=F.pad(wav,(pad,pad))
        return wav

    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        e = self._read(eid, "enroll"); q = self._read(qid, "query")
        label = p.get("label", -1)
        if self.mode == "text":
            txt = p.get("enroll_txt","").lower()
            return e, q, label, txt, pid
        elif self.mode == "audio":
            e_txt = p.get("enroll_txt","").lower()
            q_txt = p.get("query_txt","").lower()
            return e, q, label, e_txt, q_txt, pid
        return e, q, label, pid


def collate_audio(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es,qs,ls,e_txts,q_txts,ids = [],[],[],[],[],[]
    for b in batch:
        e,q=b[0],b[1]
        es.append(F.pad(e,(0,ml-e.shape[-1])) if e.shape[-1]<ml else e)
        qs.append(F.pad(q,(0,ml-q.shape[-1])) if q.shape[-1]<ml else q)
        ls.append(b[2]); e_txts.append(b[3]); q_txts.append(b[4]); ids.append(b[5])
    return torch.stack(es),torch.stack(qs),torch.tensor(ls,dtype=torch.float32),e_txts,q_txts,ids

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


# ═══════════ Training: Audio-Audio (SupCon) ═══════════

def train_audio(cfg, args):
    """Model A: audio-audio — SupCon loss on full batch embedding matrix."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", f"dual_{args.name}_audio")
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
               "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
               "train/self_paired.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp):
            with open(hp) as f: all_pairs += json.load(f)
    rng.shuffle(all_pairs)
    print(f"[audio] {len(all_pairs)} total pairs")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    temperature = 0.25   # SupCon temperature — high enough to give negs meaningful gradient
                          # cos+≈0.83→exp(3.32)=27.7, cos-≈0.06→exp(0.24)=1.27, ratio≈22x

    def get_loader(ep):
        n_pos = min(100000, len(pos_pairs))
        n_neg = min(400000, len(neg_pairs))
        subset = rng.choice(pos_pairs, n_pos, replace=False).tolist()
        subset += rng.choice(neg_pairs, n_neg, replace=False).tolist()
        np.random.shuffle(subset)
        ds = PairDataset(subset, cfg.train_zip, cfg, "audio")
        return DataLoader(ds, batch_size=512, shuffle=True,
                          num_workers=cfg.num_workers, collate_fn=collate_audio,
                          pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg, "audio"),
                          batch_size=cfg.batch_size * 2, num_workers=0,
                          collate_fn=collate_audio, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    model = AudioAudioModel(args.load_ckpt, cfg.embed_dim, cfg.unfreeze_layers,
                            encoder=args.encoder).to(device)
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

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    eval_scale = cfg.cos_scale   # for eval sigmoid

    stats_file = os.path.join(out_dir, "cos_dist.jsonl")
    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls, n = 0.0, 0
        cs_pos, cs_neg, cs_n = 0.0, 0.0, 0
        cs_pos_std, cs_neg_std = 0.0, 0.0
        all_cos_pos, all_cos_neg = [], []

        for e, q, y, e_txts, q_txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)

            # ── noise augmentation ──
            snr = np.random.uniform(-10, 5)
            nl = 10 ** (-snr / 20)
            e = e + nl * torch.randn_like(e) * e.std(-1, keepdim=True)
            snr = np.random.uniform(-10, 5)
            nl = 10 ** (-snr / 20)
            q = q + nl * torch.randn_like(q) * q.std(-1, keepdim=True)

            # ── encode ──
            ea = model.encoder(e)                            # (B, D)
            qa = model.encoder(q)                            # (B, D)

            # ── forward: comparison head + BCE (pure pair classification) ──
            cos_sim, logit, ea, qa = model(e, q)
            pos_weight = torch.tensor([4.0], device=device)
            loss = F.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)

            # ── monitor ──
            pos = (y == 1); neg = (y == 0)
            _cp = cos_sim[pos]; _cn = cos_sim[neg]

            # ── centering: push pairwise neg cos below 0 ──
            # SupCon only cares about ranking (pos > neg), not absolute position.
            # But inference uses sigmoid(cos * scale) → need neg cos < 0.
            if neg.any():
                loss = loss + 0.5 * F.relu(_cn.mean() + 0.05)  # penalize mean neg cos > -0.05
            cp_val = _cp.mean().item() if pos.any() else 0.0
            cn_val = _cn.mean().item() if neg.any() else 0.0
            cs_pos += cp_val; cs_neg += cn_val; cs_n += 1
            cs_pos_std += _cp.std().item() if pos.any() and _cp.numel() > 1 else 0.0
            cs_neg_std += _cn.std().item() if neg.any() and _cn.numel() > 1 else 0.0
            all_cos_pos.append(_cp.detach().cpu())
            all_cos_neg.append(_cn.detach().cpu())

            if cs_n % 50 == 0:
                print(f"  [ep{ep}] b{cs_n} loss={ls/n:.3f} "
                      f"cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} "
                      f"cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} "
                      f"gap={cs_pos/cs_n - cs_neg/cs_n:.3f}")

            opt.zero_grad(); loss.backward(); opt.step()
            ls += loss.item(); n += 1
            del cos_sim, logit, ea, qa, loss, _cp, _cn

        # ── eval (pairwise cosine → sigmoid → AUC) ──
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls_list, ids, all_cs = [], [], [], []
            for e, q, y, _, _, id_ in ld:
                e, q = e.to(device), q.to(device)
                _, logit, _, _ = model(e, q)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls_list.append(y.numpy()); ids.extend(id_)
                all_cs.append(logit.cpu().numpy())
            return (roc_auc_score(np.concatenate(ls_list), np.concatenate(ps)),
                    np.concatenate(ps), np.concatenate(ls_list), ids,
                    np.concatenate(all_cs))

        as_, ps_s, ls_s, ids_s, css_s = ev(dv_s)
        au_, ps_u, ls_u, ids_u, css_u = ev(dv_u)

        # ── epoch-level cosine distribution diagnostic ──
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
               "train/self_paired.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp): all_pairs += json.load(open(hp))
    rng.shuffle(all_pairs)
    print(f"[text] {len(all_pairs)} pairs")
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    if args.small_te and args.pk:
        from collections import defaultdict
        word_to_idx = defaultdict(list)
        for i, p in enumerate(pos_pairs):
            word_to_idx[p["enroll_txt"].lower()].append(i)
        valid_words = [w for w, idxs in word_to_idx.items() if len(idxs) >= 2]
        print(f"  PK: {len(valid_words)} words with ≥2 pos pairs")

        def get_loader(ep):
            P, K, B = 32, 4, 256
            np.random.seed(cfg.seed + ep)
            subset = []
            # 50 PK batches = 6400 PK pos pairs
            for _ in range(50):
                words = list(np.random.choice(valid_words, P, replace=False))
                for w in words:
                    chosen = list(np.random.choice(word_to_idx[w], K, replace=False))
                    subset += [pos_pairs[i] for i in chosen]
            n_neg = max(len(subset), 128)
            neg_idx = np.random.choice(len(neg_pairs), n_neg, replace=True)
            subset += [neg_pairs[i] for i in neg_idx]
            np.random.shuffle(subset)
            ds = PairDataset(subset, cfg.train_zip, cfg, "text")
            return DataLoader(ds, batch_size=B, shuffle=True, num_workers=0,
                              collate_fn=collate_text, pin_memory=True, drop_last=True)
    else:
        # Dedup: group by (enroll_txt, query_txt), pick 1 random per key
        pos_groups = {}
        for i, p in enumerate(pos_pairs):
            k = (p["enroll_txt"].lower(), p["query_txt"].lower())
            pos_groups.setdefault(k, []).append(i)
        neg_groups = {}
        for i, p in enumerate(neg_pairs):
            k = (p["enroll_txt"].lower(), p["query_txt"].lower())
            neg_groups.setdefault(k, []).append(i)
        pos_keys = list(pos_groups.keys())
        neg_keys = list(neg_groups.keys())
        rng.shuffle(pos_keys); rng.shuffle(neg_keys)
        print(f"  AT dedup: {len(pos_keys)} pos keys, {len(neg_keys)} neg keys")

        def get_loader(ep):
            n_pos = min(100000, len(pos_keys))
            n_neg = min(200000, len(neg_keys))
            subset = []
            # Cycle through shuffled keys, pick 1 random audio per key
            pp = (ep * n_pos) % len(pos_keys) if len(pos_keys) > 0 else 0
            for i in range(n_pos):
                k = pos_keys[(pp + i) % len(pos_keys)]
                idx = rng.choice(pos_groups[k])
                subset.append(pos_pairs[idx])
            np_start = (ep * n_neg) % len(neg_keys) if len(neg_keys) > 0 else 0
            for i in range(n_neg):
                k = neg_keys[(np_start + i) % len(neg_keys)]
                idx = rng.choice(neg_groups[k])
                subset.append(neg_pairs[idx])
            np.random.shuffle(subset)
            ds = PairDataset(subset, cfg.train_zip, cfg, "text")
            return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=cfg.num_workers, collate_fn=collate_text,
                              pin_memory=True, drop_last=True)

    def dev_ld(z,c):
        return DataLoader(PairDataset(load_pairs(c),z,cfg,"text"),
                          batch_size=cfg.batch_size*2,num_workers=0,
                          collate_fn=collate_text,shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    uf = 0 if args.small_te else cfg.unfreeze_layers
    model = AudioTextModel(args.load_ckpt, cfg.embed_dim, unfreeze=uf, small_te=args.small_te, use_phoneme=args.phoneme).to(device)
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

    stats_file = os.path.join(out_dir, "cos_dist.jsonl")
    for ep in range(start_ep, cfg.epochs+1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls,n=0.0,0
        cs_pos, cs_neg, cs_n = 0.0, 0.0, 0
        cs_pos_std, cs_neg_std = 0.0, 0.0
        all_cos_pos, all_cos_neg = [], []
        for e,q,y,txts,_ in loader:
            e,q,y = e.to(device),q.to(device),y.to(device)
            snr = np.random.uniform(-10,5); nl=10**(-snr/20)
            e = e+nl*torch.randn_like(e)*e.std(-1,keepdim=True)
            snr = np.random.uniform(-10,5); nl=10**(-snr/20)
            q = q+nl*torch.randn_like(q)*q.std(-1,keepdim=True)

            # Comparison head: [ea, et, ea*et, |ea-et|] → logit
            cos_ae, logit, ea, et = model(e, txts)
            loss = F.binary_cross_entropy_with_logits(logit, y,
                       pos_weight=torch.tensor(cfg.pos_weight, device=device))

            pos = (y==1); neg = (y==0)
            _cp = cos_ae[pos]; _cn = cos_ae[neg]
            c_pv = _cp.mean().item() if pos.any() else 0.0
            c_nv = _cn.mean().item() if neg.any() else 0.0
            opt.zero_grad(); loss.backward(); opt.step()
            ls+=loss.item(); n+=1
            cs_pos += c_pv; cs_neg += c_nv; cs_n += 1
            cs_pos_std += _cp.std().item() if pos.any() and _cp.numel() > 1 else 0
            cs_neg_std += _cn.std().item() if neg.any() and _cn.numel() > 1 else 0
            all_cos_pos.append(_cp.detach().cpu()); all_cos_neg.append(_cn.detach().cpu())
            if cs_n % 50 == 0:
                print(f"  [ep{ep}] b{cs_n} loss={ls/n:.3f} cos+={cs_pos/cs_n:.3f}±{cs_pos_std/cs_n:.2f} "
                      f"cos-={cs_neg/cs_n:.3f}±{cs_neg_std/cs_n:.2f} gap={cs_pos/cs_n-cs_neg/cs_n:.3f}")
            del cos_ae,logit,ea,et,e,q,y,loss,_cp,_cn

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps,ls,ids,all_cs=[],[],[],[]
            for e,q,y,txts,id_ in ld:
                e,q=e.to(device),q.to(device)
                _, logit, _, _ = model(e, txts)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls.append(y.numpy()); ids.extend(id_)
                all_cs.append(logit.cpu().numpy())
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
    p.add_argument("--encoder", default="whisper", choices=["whisper","wavlm"])
    p.add_argument("--small-te", action="store_true", help="use 64d CharBiGRU")
    p.add_argument("--pk", action="store_true", help="use PK sampling (32×4 pos + 128 neg per batch)")
    p.add_argument("--unfreeze", type=int, default=cfg.unfreeze_layers, help="unfreeze layers")
    p.add_argument("--phoneme", action="store_true", help="use PhonemeBiGRU (CMU dict)")
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs; cfg.unfreeze_layers = args.unfreeze

    if args.mode in ("audio","both"): train_audio(cfg, args)
    if args.mode in ("text","both"): train_text(cfg, args)


if __name__ == "__main__":
    main()
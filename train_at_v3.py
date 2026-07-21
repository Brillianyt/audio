"""AT v3 Training — deduplicated data, one-word-once sampling."""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile, re as _re
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
    max_audio_sec: float = 1.5
    unfreeze_layers: int = 12
    lora_r: int = 0

    epochs: int = 100
    lr: float = 3e-4
    batch_size: int = 256
    pos_weight: float = 5.0
    cos_scale: float = 2.0
    num_workers: int = 8
    seed: int = 42
    log_every: int = 50

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")
        for k in ("dev_seen","dev_unseen"):
            z = os.path.join(r, "dev", k, "wav.zip")
            c = os.path.join(r, "dev", k, f"{k}_label.csv")
            setattr(self, f"{k}_zip", z)
            setattr(self, f"{k}_csv", c)


# ═══════════ Whisper Encoder (same as AT v2) ═══════════

class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=2, lora_r=0):
        super().__init__()
        import whisper
        self.whisper = whisper.load_model(model_name)
        self.whisper.eval()
        
        # Freeze all Whisper weights
        for p in self.whisper.parameters():
            p.requires_grad = False
            
        # Apply LoRA to attention layers if requested
        self.lora_r = lora_r
        if lora_r > 0:
            self._apply_lora(lora_r)
        else:
            # Original behavior: unfreeze specific layers
            total = len(self.whisper.encoder.blocks)
            self.frozen = max(0, total - unfreeze)
            for i in range(self.frozen, total):
                for p in self.whisper.encoder.blocks[i].parameters():
                    p.requires_grad = True
            self.whisper.encoder.ln_post.requires_grad_(True)
        self.register_buffer("hann", torch.hann_window(400), persistent=False)
        self.register_buffer("mel_filters", self._make_mel(80), persistent=False)
        self.attn_linear = nn.Linear(self.whisper.dims.n_audio_state, self.whisper.dims.n_audio_state//4)
        self.attn_w = nn.Parameter(torch.randn(self.whisper.dims.n_audio_state//4, 1))
        self.proj = nn.Linear(self.whisper.dims.n_audio_state*2, embed_dim)

    def _apply_lora(self, r=8, last_n=6):
        """Replace Linear layers in last N blocks with LoRA-wrapped versions."""
        class LoraLinear(nn.Module):
            def __init__(self, original, r):
                super().__init__()
                self.original = original
                for p in self.original.parameters(): p.requires_grad = False
                self.lora_A = nn.Linear(original.in_features, r, bias=False)
                self.lora_B = nn.Linear(r, original.out_features, bias=False)
                nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
                nn.init.zeros_(self.lora_B.weight)
            def forward(self, x):
                return self.original(x) + self.lora_B(self.lora_A(x))

        self.lora_r = r
        count = 0
        for blk in self.whisper.encoder.blocks[-last_n:]:
            for name, mod in list(blk.named_modules()):
                if isinstance(mod, nn.Linear) and not hasattr(mod, 'lora_A'):
                    parent = blk
                    for part in name.split('.')[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, name.split('.')[-1], LoraLinear(mod, r))
                    count += 1
        self.frozen = len(self.whisper.encoder.blocks)
        print(f"  [LoRA] r={r} on last {last_n} blocks ({count} layers)")

    def _make_mel(self, n_mels, n_fft=400):
        import whisper.audio as wa
        filters = wa.mel_filters("cpu", n_mels)
        # filters shape: [n_mels, n_fft//2 + 1]
        expected = n_fft // 2 + 1
        if filters.shape[-1] != expected:
            filters = filters[:, :expected] if filters.shape[-1] > expected else \
                      F.pad(filters, (0, expected - filters.shape[-1]))
        return filters

    def forward(self, wav):
        wav = wav.to(next(self.whisper.parameters()).device).float()
        stft = torch.stft(wav, 400, 160, 400, self.hann, return_complex=True)
        mags = stft[..., :-1].abs() ** 2
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
            if self.lora_r > 0:
                # With LoRA: all blocks need gradients
                for blk in enc.blocks: x = blk(x)
            else:
                for blk in enc.blocks[:self.frozen]: x = blk(x)
        for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)[:, :n_valid, :]
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)

    # ── One-time noise injection on last unfrozen block ──
    def apply_noise(self, noise_ratio=0.3):
        if self.frozen >= len(self.whisper.encoder.blocks):
            return
        blk = self.whisper.encoder.blocks[-1]
        self._saved_params = {}
        for name, p in blk.named_parameters():
            if p.requires_grad:
                self._saved_params[name] = p.data.clone()
                mask = torch.rand_like(p) < noise_ratio
                p.data[mask] = torch.randn_like(p.data[mask]) * p.data.std()
        print(f"  [noise] AT: replaced {noise_ratio*100:.0f}% of last block's weights")

    def restore_weights(self):
        if not hasattr(self, '_saved_params') or not self._saved_params:
            return
        blk = self.whisper.encoder.blocks[-1]
        for name, p in blk.named_parameters():
            if name in self._saved_params:
                p.data = self._saved_params[name]
        print("  [noise] AT: restored original weights")


# ═══════════ CMU Phoneme utilities ═══════════
import re as _re
_cmudict_cache = None

def _get_cmudict():
    global _cmudict_cache
    if _cmudict_cache is None:
        import cmudict
        _cmudict_cache = cmudict.dict()
    return _cmudict_cache

def _word_to_phonemes(word: str):
    cmu = _get_cmudict()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w:
        return []
    plist = cmu.get(w)
    if plist:
        return [_re.sub(r'[0-2]$', '', p) for p in plist[0]]
    return []

PHONEME_VOCAB = [
    "AA","AE","AH","AO","AW","AY","B","CH","D","DH",
    "EH","ER","EY","F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P","R","S","SH",
    "T","TH","UH","UW","V","W","Y","Z","ZH","UNK",
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}
_p2i_cache = {}


def _text_to_phoneme_ids(text: str):
    """word -> phoneme id list. Cached per word."""
    if text not in _p2i_cache:
        phons = _word_to_phonemes(text)
        _p2i_cache[text] = [PHONEME_TO_IDX.get(p, 39) for p in phons] if phons else [39]
    return _p2i_cache[text]


# ═══════════ Phoneme BiGRU Text Encoder ═══════════

class PhonemeBiGRUEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.emb = nn.Embedding(40, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=4, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts:
            return torch.zeros(0, 256, device=device), None
        ph_lists = [_text_to_phoneme_ids(t) for t in texts]
        mx = max(len(p) for p in ph_lists)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, ph in enumerate(ph_lists):
            for j, pid in enumerate(ph[:mx]):
                idx[i, j] = pid
        x = self.emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1), None


# ═══════════ CharTransformer Text Encoder (备选) ═══════════

class CharTransformerEncoder(nn.Module):
    def __init__(self, dim=256, n_layers=4, n_heads=4, max_len=20):
        super().__init__()
        self.char_emb = nn.Embedding(28, dim)
        self.pos_enc = nn.Parameter(torch.randn(1, max_len, dim) * 0.1)
        enc_layer = nn.TransformerEncoderLayer(dim, n_heads, dim*4, dropout=0.1, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, n_layers)
        self.proj = nn.Linear(dim, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0,256,device=device),None
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx) + self.pos_enc[:, :mx]
        x = self.transformer(x)
        x = x.mean(dim=1)  # mean pooling over sequence
        return F.normalize(self.proj(x), dim=-1), None


# ═══════════ Comparison Head ═══════════

class ComparisonHead(nn.Module):
    """MLP comparison head: concat(ea, et, ea-et, ea*et) → score."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim*4, embed_dim), nn.ReLU(), nn.BatchNorm1d(embed_dim),
            nn.Linear(embed_dim, 64), nn.ReLU(), nn.BatchNorm1d(64),
            nn.Linear(64, 1),
        )

    def forward(self, ea, et):
        return self.net(torch.cat([ea, et, ea-et, ea*et], -1))


# ═══════════ Audio-Text Model ═══════════

class AudioTextModel(nn.Module):
    """Audio-Text model: cosine(et, eq) for text↔query matching."""
    def __init__(self, embed_dim=256, unfreeze=2, text_encoder="phoneme_bigru", lora_r=0):
        super().__init__()
        self.encoder = WhisperEncoder("base", embed_dim, unfreeze, lora_r)
        if text_encoder == "char":
            self.text_enc = CharTransformerEncoder(embed_dim)
        else:
            self.text_enc = PhonemeBiGRUEncoder(embed_dim)

    def forward(self, e, texts):
        """Return ae_cos(enroll_audio,enroll_text), ea, et for margin only."""
        ea = self.encoder(e)
        et, _ = self.text_enc(texts)
        return (et * ea).sum(-1), ea, et

    def score(self, et, eq):
        """Cosine(enroll_text, query_audio)."""
        return (et * eq).sum(-1)


# ═══════════ Deduplicated Data Loading ═══════════

def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows


def deduplicate_by_word_pair(pairs, max_per_pair=1):
    """Keep at most max_per_pair examples per unique (enroll_txt, query_txt) pair."""
    seen = {}
    deduped = []
    for p in pairs:
        key = (p["enroll_txt"].lower(), p["query_txt"].lower())
        if key not in seen:
            seen[key] = 0
        if seen[key] < max_per_pair:
            seen[key] += 1
            deduped.append(p)
    return deduped


def load_all_data(cfg):
    """Load all data sources and deduplicate."""
    rng = np.random.default_rng(cfg.seed)

    # 1. Original competition CSV
    all_pairs = load_pairs(cfg.train_csv)

    # 2. Additional pair files
    extra_files = [
        "baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
        "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
        "train/self_paired.json", "train/self_paired_xl.json",
        "train/fill_pos_pairs.json", "train/mega_pairs.json",
        "baseline/hard_neg_at_ohem.json",  # True OHEM: model error-driven
    ]
    for fn in extra_files:
        fp = os.path.join(PATHS.root, fn)
        if os.path.isfile(fp):
            data = json.load(open(fp))
            # Filter pairs whose zip file actually exists
            valid = []
            for p in data:
                zp = p.get("zip", "")
                if zp and not os.path.isfile(os.path.join(PATHS.root, zp)):
                    continue  # skip pairs referencing deleted zips
                valid.append(p)
            all_pairs += valid
            if len(valid) != len(data):
                print(f"  loaded {fn}: {len(valid)}/{len(data)} pairs (filtered missing zips)")
            else:
                print(f"  loaded {fn}: {len(data)} pairs")

    print(f"  total before dedup: {len(all_pairs)} pairs")

    # 3. Deduplicate
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]

    # Keep all pos pairs (let model memorize seen words first)
    pos_dedup = deduplicate_by_word_pair(pos_pairs, max_per_pair=30)
    # For hard negs: use set for O(1) membership
    hard_neg_ids = {"hard_neg", "hn_", "phoneme"}
    hard_neg = [p for p in neg_pairs if any(k in p.get("id","") for k in hard_neg_ids)]
    hard_id_set = {p["id"] for p in hard_neg}
    easy_neg = [p for p in neg_pairs if p["id"] not in hard_id_set]
    hard_dedup = deduplicate_by_word_pair(hard_neg, max_per_pair=10)
    easy_dedup = deduplicate_by_word_pair(easy_neg, max_per_pair=1)

    rng.shuffle(pos_dedup); rng.shuffle(hard_dedup); rng.shuffle(easy_dedup)

    print(f"  pos: {len(pos_pairs)} -> {len(pos_dedup)} (kept all)")
    print(f"  hard neg: {len(hard_neg)} -> {len(hard_dedup)} (deduped)")
    print(f"  easy neg: {len(easy_neg)} -> {len(easy_dedup)} (deduped)")

    return pos_dedup, hard_dedup, easy_dedup


# ═══════════ Dataset ═══════════

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs
        self.default_zip_path = zip_path
        self.cfg = cfg
        self._zc = {}  # pid -> dict of zip_path -> ZipFile

    def _get_zip_for(self, pair):
        """Get zip for a specific pair, respecting per-pair 'zip' field."""
        zpath = pair.get("zip", "")
        if not zpath:
            zpath = self.default_zip_path
        else:
            zpath = os.path.join(PATHS.root, zpath)
        pid = os.getpid()
        if pid not in self._zc:
            self._zc[pid] = {}
        if zpath not in self._zc[pid]:
            self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]

    def _read_wav(self, pid, role, zf):
        # Try both naming conventions: pid_role.wav (original) and pid.wav (external)
        for name in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try:
                data = zf.read(name)
                break
            except KeyError:
                continue
        else:
            raise KeyError(f"Neither 'wav/{pid}_{role}.wav' nor 'wav/{pid}.wav' found in zip")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pid = p["id"]
        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)
        # OHEM pairs: query_aid has real audio ID, enroll_txt ← query_txt (the confused word)
        is_ohem = p.get("type","") == "at_ohem_fp"
        if is_ohem:
            real_aid = p.get("query_aid", qid)
            eid = real_aid  # enroll audio = same as query (only used for margin)
            qid = real_aid  # query audio = the real audio
            txt = p.get("query_txt", "").lower()  # confused word as text target
        else:
            txt = p.get("enroll_txt", "").lower()
        zf = self._get_zip_for(p)
        e = self._read_wav(eid, "enroll", zf)
        q = self._read_wav(qid, "query", zf)
        return e, q, float(p.get("label", 0)), txt, pid


def collate_text(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, ls, txts, ids = [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, ml - e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml - q.shape[-1])) if q.shape[-1] < ml else q)
        ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls, dtype=torch.float32), txts, ids


# ═══════════ Training ═══════════

def train_v3(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", f"at_v6")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load and deduplicate data ──
    print("[v3] loading data...")
    pos_dedup, hard_dedup, easy_dedup = load_all_data(cfg)
    print(f"[v3] pos={len(pos_dedup)} hard_neg={len(hard_dedup)} easy_neg={len(easy_dedup)}")

    # ── Model ──
    model = AudioTextModel(cfg.embed_dim, cfg.unfreeze_layers, lora_r=cfg.lora_r).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded encoder from {args.load_ckpt}")
        if "auc_unseen" in ckpt:
            print(f"    prev unseen={ckpt['auc_unseen']:.4f}")

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed from {fp} epoch={start_ep}")
                break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    # ── Optimizer (grouped LR) with Cosine Schedule + Warmup ──
    wav_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params": wav_p, "lr": cfg.lr/5}, {"params": oth_p, "lr": 1e-3}],
                            weight_decay=1e-4)
    total_steps = cfg.epochs * 600000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # ── Dev loaders ──
    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        # ── Build epoch-specific balanced sample ──
        # Strategy: lots of pos + hard_neg ×10 oversampling
        n_pos_ep = min(60000, len(pos_dedup))
        n_hard_ep = min(300000, len(hard_dedup) * 2)  # gentle oversample, avoid memorization

        # Fresh shuffle each epoch
        idx_pos = np.random.permutation(len(pos_dedup))[:n_pos_ep]
        # Hard neg oversample with replacement
        idx_hard = np.random.choice(len(hard_dedup), n_hard_ep, replace=True)

        subset = [pos_dedup[i] for i in idx_pos]
        subset += [hard_dedup[i] for i in idx_hard]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[ep{ep}] train={len(subset)} pos={n_pos_ep} hard={n_hard_ep} "
              f"words={len(set(p['enroll_txt'].lower() for p in subset))}")

        # ── Train ──
        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        cos_pos_sum, cos_neg_sum, cos_n = 0.0, 0.0, 0
        all_cos_pos, all_cos_neg = [], []

        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)

            # Tiered noise: mostly mild, occasionally heavy (matches test distribution)
            snr = float(np.random.choice(
                [np.random.uniform(0, 5), np.random.uniform(-5, 0), np.random.uniform(-10, -5)],
                p=[0.7, 0.2, 0.1]))
            nl = 10 ** (-snr / 20)
            e = e + nl * torch.randn_like(e) * e.std(-1, keepdim=True)
            snr = float(np.random.choice(
                [np.random.uniform(0, 5), np.random.uniform(-5, 0), np.random.uniform(-10, -5)],
                p=[0.7, 0.2, 0.1]))
            nl = 10 ** (-snr / 20)
            q = q + nl * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                ae_cos, ea, et = model(e, txts)
                scale = cfg.cos_scale
                eq = model.encoder(q)
                match_cos = model.score(et, eq)  # cosine(enroll_text, query_audio)

                pos = (y == 1)
                neg = (y == 0)
                margin = 0.0
                if pos.any(): margin += F.relu(0.6 - match_cos[pos]).mean()
                if neg.any(): margin += F.relu(match_cos[neg] + 0.15).mean()
                margin += F.relu(0.6 - ae_cos).mean()

                loss = crit(match_cos * scale, y) + 0.1 * margin

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item(); n_batches += 1
            _cp = match_cos[pos].detach() if pos.any() else torch.tensor([0.0])
            _cn = match_cos[neg].detach() if neg.any() else torch.tensor([0.0])
            cos_pos_sum += _cp.mean().item(); cos_neg_sum += _cn.mean().item(); cos_n += 1
            all_cos_pos.append(_cp.cpu()); all_cos_neg.append(_cn.cpu())

            if cos_n % max(1, cfg.log_every) == 0:
                print(f"  [ep{ep}] b{cos_n} loss={total_loss/n_batches:.3f} "
                      f"cos+={cos_pos_sum/cos_n:.3f} cos-={cos_neg_sum/cos_n:.3f} "
                      f"gap={cos_pos_sum/cos_n - cos_neg_sum/cos_n:.3f}")

        # ── Eval (cosine(enroll_text, query_audio), same as training) ──
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls, ids, all_cs = [], [], [], []
            for e, q, y, txts, id_ in ld:
                e, q = e.to(device), q.to(device)
                _, _, et = model(e, txts)
                eq = model.encoder(q)
                cs = model.score(et, eq)  # cosine [-1,1]
                ps.append(torch.sigmoid(cs * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy()); ids.extend(id_)
                all_cs.append(cs.cpu().numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps)), \
                   np.concatenate(ps), np.concatenate(ls), ids, np.concatenate(all_cs)

        as_, ps_s, ls_s, ids_s, css_s = ev(dv_s)
        au_, ps_u, ls_u, ids_u, css_u = ev(dv_u)

        # ── Diagnostics ──
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
                        "p95": float(np.percentile(all_cp, 95))},
            "cos_neg": {"mean": float(all_cn.mean()), "std": float(all_cn.std()),
                        "p05": float(np.percentile(all_cn, 5)),
                        "p50": float(np.percentile(all_cn, 50)),
                        "p95": float(np.percentile(all_cn, 95))},
            "overlap_gt_0": _pct(all_cn, 0.0, 1.0),
        }
        with open(os.path.join(out_dir, "cos_dist.jsonl"), "a") as f:
            f.write(json.dumps(dist) + "\n")

        print(f"[v3 ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={total_loss/n_batches:.4f} "
              f"cos+={cos_pos_sum/cos_n:.3f} cos-={cos_neg_sum/cos_n:.3f} "
              f"overlap>0={dist['overlap_gt_0']:.3f} ({time.time()-ts:.0f}s)")
        print(f"  cos+ [p5={dist['cos_pos']['p05']:.3f} p50={dist['cos_pos']['p50']:.3f} p95={dist['cos_pos']['p95']:.3f}]")
        print(f"  cos- [p5={dist['cos_neg']['p05']:.3f} p50={dist['cos_neg']['p50']:.3f} p95={dist['cos_neg']['p95']:.3f}]")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
            print(f"  saved best.pt (unseen={au_:.4f})")


# ═══════════ Main ═══════════

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="", help="pretrained encoder to init")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    cfg.unfreeze_layers = args.unfreeze
    train_v3(cfg, args)

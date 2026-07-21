"""AT v4 — Frame-level cross-attention between Whisper frames and phoneme tokens.
Key changes from v3:
1. No ASP pooling — keep Whisper frame features
2. PhonemeBiGRU outputs per-token features, not single vector
3. Cross-attention: audio frames attend to phoneme tokens → alignment score
4. No MSE(audio, text), no ae_cos margin
"""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS


# ═══════════ Whisper Frame Encoder (no ASP) ═══════════
class WhisperFrameEncoder(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=4):
        super().__init__()
        import whisper
        self.whisper = whisper.load_model(model_name)
        self.whisper.eval()
        for p in self.whisper.parameters(): p.requires_grad = False
        total = len(self.whisper.encoder.blocks)
        self.frozen = max(0, total - unfreeze)
        for i in range(self.frozen, total):
            for p in self.whisper.encoder.blocks[i].parameters(): p.requires_grad = True
        self.whisper.encoder.ln_post.requires_grad_(True)
        self.register_buffer("hann", torch.hann_window(400), persistent=False)
        self.register_buffer("mel_filters", self._make_mel(80), persistent=False)
        self.hidden_dim = self.whisper.dims.n_audio_state  # 512
        self.proj = nn.Linear(self.hidden_dim, embed_dim)

    def _make_mel(self, n_mels, n_fft=400):
        import whisper.audio as wa
        filters = wa.mel_filters("cpu", n_mels)
        expected = n_fft // 2 + 1
        if filters.shape[-1] != expected:
            filters = filters[:, :expected] if filters.shape[-1] > expected else \
                      F.pad(filters, (0, expected - filters.shape[-1]))
        return filters

    def forward(self, wav):
        """Returns frame features (B, T_frames, embed_dim)."""
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
            for blk in enc.blocks[:self.frozen]: x = blk(x)
        for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)[:, :n_valid, :]
        return F.normalize(self.proj(x), dim=-1)


# ═══════════ CMU Phoneme utilities ═══════════
import re as _re
_cmudict_cache = None
def _get_cmudict():
    global _cmudict_cache
    if _cmudict_cache is None:
        import cmudict; _cmudict_cache = cmudict.dict()
    return _cmudict_cache
def _word_to_phonemes(word: str):
    cmu = _get_cmudict(); w = word.lower().strip("'s\"-.,!?;:")
    if not w: return []
    plist = cmu.get(w)
    return [_re.sub(r'[0-2]$', '', p) for p in plist[0]] if plist else []
PHONEME_VOCAB = [
    "AA","AE","AH","AO","AW","AY","B","CH","D","DH",
    "EH","ER","EY","F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P","R","S","SH",
    "T","TH","UH","UW","V","W","Y","Z","ZH","UNK",
]
PHONEME_TO_IDX = {p: i for i, p in enumerate(PHONEME_VOCAB)}
_p2i_cache = {}
def _text_to_phoneme_ids(text: str):
    if text not in _p2i_cache:
        phons = _word_to_phonemes(text)
        _p2i_cache[text] = [PHONEME_TO_IDX.get(p, 39) for p in phons] if phons else [39]
    return _p2i_cache[text]


# ═══════════ Phoneme Token Encoder (per-token, no final pooling) ═══════════
class PhonemeTokenEncoder(nn.Module):
    """Outputs per-phoneme features (B, L, D) instead of single vector."""
    def __init__(self, dim=256):
        super().__init__()
        self.emb = nn.Embedding(40, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts:
            return torch.zeros(0, 256, device=device)
        ph_lists = [_text_to_phoneme_ids(t) for t in texts]
        mx = max(len(p) for p in ph_lists)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        mask = torch.zeros(len(texts), mx, dtype=torch.bool, device=device)
        for i, ph in enumerate(ph_lists):
            for j, pid in enumerate(ph[:mx]):
                idx[i,j] = pid; mask[i,j] = True
        x = self.emb(idx)
        out, _ = self.gru(x)  # (B, L, D*2)
        return F.normalize(self.proj(out), dim=-1), mask  # (B, L, D), (B, L)


# ═══════════ Audio-Phoneme Cross Attention ═══════════
class AudioPhonemeCrossAttention(nn.Module):
    """Audio frames attend to phoneme tokens → per-phoneme alignment → top-k mean."""
    def __init__(self, embed_dim=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, audio_frames, phoneme_tokens, ph_mask=None):
        B, Ta, D = audio_frames.shape
        _, Lp, _ = phoneme_tokens.shape
        Q = self.q_proj(audio_frames).view(B, Ta, self.n_heads, self.head_dim).transpose(1,2)
        K = self.k_proj(phoneme_tokens).view(B, Lp, self.n_heads, self.head_dim).transpose(1,2)
        V = self.v_proj(phoneme_tokens).view(B, Lp, self.n_heads, self.head_dim).transpose(1,2)
        attn = torch.matmul(Q, K.transpose(-2,-1)) * self.scale
        if ph_mask is not None:
            attn = attn.masked_fill(~ph_mask.unsqueeze(1).unsqueeze(2), -1e9)
        attn = F.softmax(attn, dim=-1)
        context = torch.matmul(attn, V).transpose(1,2).contiguous().view(B, Ta, D)
        context = F.normalize(context, dim=-1)
        frame_scores = (audio_frames * context).sum(-1)  # (B, Ta)
        # Top-k mean: requires multiple frames to agree
        k = max(2, Ta // 4)
        return frame_scores.topk(k, dim=-1).values.mean(-1)  # (B,)


# ═══════════ AT v4 Model ═══════════
class ATv4Model(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=4):
        super().__init__()
        self.audio_enc = WhisperFrameEncoder("base", embed_dim, unfreeze)
        self.text_enc = PhonemeTokenEncoder(embed_dim)
        self.aligner = AudioPhonemeCrossAttention(embed_dim)

    def forward(self, enroll_audio, texts, query_audio):
        ea_frames = self.audio_enc(enroll_audio)   # (B, Te, D)
        et_tokens, _ = self.text_enc(texts)          # (B, Lp, D)
        eq_frames = self.audio_enc(query_audio)     # (B, Tq, D)
        # Both enroll and query audio align to the same phoneme sequence
        score_e = self.aligner(ea_frames, et_tokens)  # enroll→phoneme
        score_q = self.aligner(eq_frames, et_tokens)   # query→phoneme
        score = (score_e + score_q) / 2               # balanced
        return score, ea_frames, et_tokens, eq_frames


# ═══════════ Data (simplified from v3) ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows

def dedup(pairs, max_per_pair=1):
    seen = {}; res = []
    for p in pairs:
        k = (p["enroll_txt"].lower(), p.get("query_txt",p["enroll_txt"]).lower())
        if k not in seen: seen[k] = 0
        if seen[k] < max_per_pair: seen[k] += 1; res.append(p)
    return res

def load_all_data(cfg):
    rng = np.random.default_rng(42)
    all_pairs = load_pairs(cfg.train_csv)
    extra_files = [
        "baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
        "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
        "train/self_paired.json", "train/self_paired_xl.json",
        "train/fill_pos_pairs.json", "train/mega_pairs.json",
        "baseline/hard_neg_at_ohem.json",
    ]
    for fn in extra_files:
        fp = os.path.join(PATHS.root, fn)
        if os.path.isfile(fp):
            data = json.load(open(fp))
            valid = []
            for p in data:
                zp = p.get("zip", "")
                if zp and not os.path.isfile(os.path.join(PATHS.root, zp)):
                    continue
                valid.append(p)
            all_pairs += valid
            print(f"  loaded {fn}: {len(valid)} pairs")
    print(f"  total: {len(all_pairs)} pairs")
    pos = [p for p in all_pairs if p["label"] == 1]
    neg = [p for p in all_pairs if p["label"] == 0]
    hard_ids = {"hard_neg", "hn_", "phoneme"}
    hard = [p for p in neg if any(k in p.get("id","") for k in hard_ids)]
    easy = [p for p in neg if p["id"] not in {x["id"] for x in hard}]
    pos_d = dedup(pos, max_per_pair=10)
    hard_d = dedup(hard, max_per_pair=5)
    easy_d = dedup(easy, max_per_pair=2)
    rng.shuffle(pos_d); rng.shuffle(hard_d); rng.shuffle(easy_d)
    print(f"  pos={len(pos_d)} hard_neg={len(hard_d)} easy_neg={len(easy_d)}")
    return pos_d, hard_d, easy_d


class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.default_zip_path = zip_path; self.cfg = cfg; self._zc = {}
    def _get_zip_for(self, pair):
        zpath = pair.get("zip", "")
        if not zpath: zpath = self.default_zip_path
        else: zpath = os.path.join(PATHS.root, zpath)
        pid = os.getpid()
        if pid not in self._zc: self._zc[pid] = {}
        if zpath not in self._zc[pid]: self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]
    def _read_wav(self, pid, role, zf):
        for name in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try: data = zf.read(name); break
            except KeyError: continue
        else: raise KeyError(f"wav/{pid}_{role}.wav not found")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        # OHEM pairs: use real audio ID from query_aid, swap text to confused word
        is_ohem = p.get("type","") == "at_ohem_fp"
        if is_ohem:
            real_aid = p.get("query_aid", qid)
            eid = real_aid; qid = real_aid
            txt = p.get("query_txt", "").lower()
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
def train_at_v4(cfg, args):
    device = "cuda"; torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "at_v4")
    os.makedirs(out_dir, exist_ok=True)

    print("[ATv4] loading data...")
    pos_dedup, hard_dedup, easy_dedup = load_all_data(cfg)

    model = ATv4Model(cfg.embed_dim, cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded encoder from {args.load_ckpt}")

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir,"latest.pt"), os.path.join(out_dir,"best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen",-1); start_ep = ckpt.get("epoch",0)+1
                print(f"  resumed ep{start_ep}"); break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params":wav_p,"lr":cfg.lr/5},{"params":oth_p,"lr":1e-3}], weight_decay=1e-4)
    total_steps = cfg.epochs * 200000 // cfg.batch_size
    warmup_steps = max(100, total_steps//20)
    def lr_lm(s):
        if s < warmup_steps: return s/warmup_steps
        return 0.5*(1+math.cos(math.pi*(s-warmup_steps)/(total_steps-warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lm)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(2.0, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    def dev_ld(z,c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        n_pos_ep = min(60000, len(pos_dedup))
        n_hard_ep = min(100000, len(hard_dedup) * 2)
        n_easy_ep = min(40000, len(easy_dedup))
        idx_p = np.random.permutation(len(pos_dedup))[:n_pos_ep]
        idx_h = np.random.choice(len(hard_dedup), n_hard_ep, replace=True)
        idx_e = np.random.choice(len(easy_dedup), n_easy_ep, replace=True)
        subset = [pos_dedup[i] for i in idx_p] + \
                 [hard_dedup[i] for i in idx_h] + \
                 [easy_dedup[i] for i in idx_e]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[ATv4 ep{ep}] train={len(subset)} pos={n_pos_ep} hard={n_hard_ep} easy={n_easy_ep}")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            snr = float(np.random.choice(
                [np.random.uniform(0,5), np.random.uniform(-5,0), np.random.uniform(-10,-5)],
                p=[0.7,0.2,0.1]))
            e = e + (10**(-snr/20)) * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + (10**(-snr/20)) * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                score, _, _, _ = model(e, txts, q)
                pos = (y == 1); neg = (y == 0)
                margin = 0.0
                if pos.any(): margin += F.relu(0.6 - score[pos]).mean()
                if neg.any(): margin += F.relu(score[neg] + 0.15).mean()
                loss = crit(score * cfg.cos_scale, y) + 0.1 * margin

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()

            total_loss += loss.item(); n_batches += 1
            if n_batches % cfg.log_every == 0:
                cp = score[pos].mean().item() if pos.any() else 0
                cn = score[neg].mean().item() if neg.any() else 0
                print(f"  [ep{ep}] b{n_batches} loss={total_loss/n_batches:.3f} "
                      f"align+={cp:.3f} align-={cn:.3f} gap={cp-cn:.3f}")

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                score, _, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(score * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[ATv4 ep{ep}] seen={as_:.4f} unseen={au_:.4f} "
              f"loss={total_loss/n_batches:.4f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep}, os.path.join(out_dir, "best.pt"))
            print(f"  saved best.pt (unseen={au_:.4f})")


class Config:
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5
    unfreeze_layers: int = 4
    epochs: int = 30
    lr: float = 3e-4
    batch_size: int = 256
    cos_scale: float = 3.0
    num_workers: int = 8
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
            setattr(self, f"{k}_zip", z); setattr(self, f"{k}_csv", c)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=256)
    p.add_argument("--unfreeze", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    cfg.unfreeze_layers = args.unfreeze
    train_at_v4(cfg, args)

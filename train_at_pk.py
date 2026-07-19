"""AT PKSampler + AngularPrototypicalLoss — cross-modal (audio↔text) PK training."""
import argparse, csv, gc, json, math, os, sys, time, io, zipfile
from collections import defaultdict
from typing import Dict, List
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS


# ═══════════ WhisperEncoder (same as AT v5) ═══════════
class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=2):
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
        self.attn_linear = nn.Linear(self.whisper.dims.n_audio_state, self.whisper.dims.n_audio_state//4)
        self.attn_w = nn.Parameter(torch.randn(self.whisper.dims.n_audio_state//4, 1))
        self.proj = nn.Linear(self.whisper.dims.n_audio_state*2, embed_dim)

    def _make_mel(self, n_mels, n_fft=400):
        import whisper.audio as wa
        filters = wa.mel_filters("cpu", n_mels)
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
            for blk in enc.blocks[:self.frozen]: x = blk(x)
            for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)[:, :n_valid, :]
        h = torch.tanh(self.attn_linear(x))
        aw = F.softmax(h @ self.attn_w, dim=1)
        mu = (x*aw).sum(1); sg = ((x**2*aw).sum(1)-mu**2).clamp(1e-5).sqrt()
        return F.normalize(self.proj(torch.cat([mu,sg],-1)), dim=-1)


# ═══════════ CharTransformer Text Encoder ═══════════
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
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.char_emb(idx) + self.pos_enc[:, :mx]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return F.normalize(self.proj(x), dim=-1)


# ═══════════ AT PK Model ═══════════
class ATPKModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=2):
        super().__init__()
        self.audio_enc = WhisperEncoder("base", embed_dim, unfreeze)
        self.text_enc = CharTransformerEncoder(embed_dim)

    def forward_audio(self, x):
        return self.audio_enc(x)

    def forward_text(self, texts):
        return self.text_enc(texts)


# ═══════════ PK Loss (adapted for cross-modal) ═══════════
class CrossModalPrototypicalLoss(nn.Module):
    """Audio embeddings vs text prototypes — cross-entropy with learnable scale/bias."""
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, audio_emb, text_protos, word_labels):
        """
        audio_emb: (N, D) L2-normalized audio embeddings
        text_protos: (P, D) L2-normalized text prototypes (one per word)
        word_labels: (N,) word index for each audio
        """
        # Cosine between each audio and all text prototypes
        cosine = torch.matmul(audio_emb, text_protos.T)  # (N, P)
        logits = self.w.clamp(min=1e-6) * cosine + self.b
        return F.cross_entropy(logits, word_labels)


# ═══════════ PKSampler ═══════════
class ATPKSampler:
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


# ═══════════ Dataset ═══════════
class ATWordDataset(Dataset):
    """Returns (audio_waveform, word_string) for PK training."""
    def __init__(self, entries, default_zip, cfg):
        self.entries = entries
        self.default_zip_path = default_zip
        self.cfg = cfg
        self._zc = {}

    def _get_zip(self, zpath):
        pid = os.getpid()
        if pid not in self._zc:
            self._zc[pid] = {}
        if zpath not in self._zc[pid]:
            self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]

    def _read_wav(self, pid, zf):
        for name in [f"wav/{pid}_enroll.wav", f"wav/{pid}_query.wav", f"wav/{pid}.wav"]:
            try:
                data = zf.read(name)
                break
            except KeyError:
                continue
        else:
            raise KeyError(f"Cannot find wav/{pid}_*.wav in zip")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        e = self.entries[idx]
        zpath = e.get("zip", self.default_zip_path)
        if zpath != self.default_zip_path:
            zpath = os.path.join(PATHS.root, zpath)
        zf = self._get_zip(zpath)
        wav = self._read_wav(e["id"], zf)
        return wav, e["word"].lower()


def collate_audio(batch):
    ml = max(b[0].shape[-1] for b in batch)
    audios, words = [], []
    for wav, word in batch:
        if wav.shape[-1] < ml:
            wav = F.pad(wav, (0, ml - wav.shape[-1]))
        audios.append(wav)
        words.append(word)
    return torch.stack(audios), words


# ═══════════ Data loading ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows

def build_word_audio_index(cfg):
    """Build word → audio_ids mapping from all data sources."""
    all_pairs = load_pairs(cfg.train_csv)
    extra_files = [
        "baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
        "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
        "train/self_paired.json", "train/self_paired_xl.json",
        "train/fill_pos_pairs.json", "train/mega_pairs.json",
        "train/speech_commands_pairs.json",
        "baseline/hard_neg_atv2.json",
        "train/librispeech_pairs.json",
    ]
    for fn in extra_files:
        fp = os.path.join(PATHS.root, fn)
        if os.path.isfile(fp):
            data = json.load(open(fp))
            all_pairs += data
    print(f"  total pairs: {len(all_pairs)}")

    word_to_ids = defaultdict(set)
    seen_ids = set()
    entries = []
    for p in all_pairs:
        if p["label"] == 1:
            word = p["enroll_txt"].lower()
            eid = p.get("enroll_id", p["id"])
            qid = p.get("query_id", p["id"])
            for aid in [eid, qid]:
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    word_to_ids[word].add(aid)
                    entries.append({"id": aid, "word": word,
                                    "zip": p.get("zip", cfg.train_zip)})

    K = 4
    valid_words = {w for w, ids in word_to_ids.items() if len(ids) >= K}
    entries = [e for e in entries if e["word"] in valid_words]
    word_to_idx = {w: [i for i, e in enumerate(entries) if e["word"] == w]
                   for w in valid_words}
    print(f"  unique words (≥{K} audios): {len(valid_words)}")
    print(f"  total audio samples: {len(entries)}")
    return word_to_idx, entries, sorted(valid_words)


# ═══════════ Pair-based eval (same as before) ═══════════
class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs
        self.default_zip_path = zip_path
        self.cfg = cfg
        self._zc = {}

    def _get_zip_for(self, pair):
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
        for name in [f"wav/{pid}_{role}.wav", f"wav/{pid}.wav"]:
            try:
                data = zf.read(name)
                break
            except KeyError:
                continue
        else:
            raise KeyError(f"Neither 'wav/{pid}_{role}.wav' nor 'wav/{pid}.wav' found")
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
        zf = self._get_zip_for(p)
        e = self._read_wav(eid, "enroll", zf)
        q = self._read_wav(qid, "query", zf)
        return e, q, float(p.get("label", 0)), p.get("enroll_txt", "").lower(), pid

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
def train_at_pk(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "at_pk")
    os.makedirs(out_dir, exist_ok=True)

    print("[AT-PK] building word index...")
    word_to_idx, entries, all_words = build_word_audio_index(cfg)
    dataset = ATWordDataset(entries, cfg.train_zip, cfg)
    sampler = ATPKSampler(word_to_idx, P=32, K=4, batches_per_epoch=300)

    model = ATPKModel(cfg.embed_dim, cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        # Load audio encoder and text encoder separately
        audio_state = {k.replace("audio_enc.", ""): v for k, v in ckpt["model"].items()
                       if k.startswith("audio_enc.")}
        text_state = {k.replace("text_enc.", ""): v for k, v in ckpt["model"].items()
                      if k.startswith("text_enc.")}
        if audio_state:
            model.audio_enc.load_state_dict(audio_state, strict=False)
            print(f"  loaded audio_enc from {args.load_ckpt}")
        if text_state:
            model.text_enc.load_state_dict(text_state, strict=False)
            print(f"  loaded text_enc from {args.load_ckpt}")

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

    wav_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n, p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    opt = torch.optim.AdamW([{"params": wav_p, "lr": cfg.lr/5}, {"params": oth_p, "lr": 1e-3}],
                            weight_decay=1e-4)
    total_steps = cfg.epochs * len(sampler)
    warmup_steps = max(100, total_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    proto_loss = CrossModalPrototypicalLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Dev loaders (pair-based)
    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=cfg.num_workers, collate_fn=collate_audio,
                            pin_memory=True)
        print(f"[AT-PK ep{ep}] PK batches={len(sampler)} P=32 K=4")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        for audios, words in loader:
            audios = audios.to(device)
            # Noise augmentation
            snr = np.random.uniform(-10, 5)
            audios = audios + (10**(-snr/20)) * torch.randn_like(audios) * audios.std(-1, keepdim=True)

            # Get unique words in this batch (P words)
            unique_words = list(set(words))
            word_to_label = {w: i for i, w in enumerate(unique_words)}
            labels = torch.tensor([word_to_label[w] for w in words], device=device)

            with torch.cuda.amp.autocast():
                # Audio embeddings
                ae = model.forward_audio(audios)  # (P*K, D)
                # Text prototypes (one per unique word)
                te = model.forward_text(unique_words)  # (P, D)
                loss = proto_loss(ae, te, labels)

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item(); n_batches += 1
            if n_batches % cfg.log_every == 0:
                print(f"  [ep{ep}] b{n_batches} loss={total_loss/n_batches:.3f}")

        # Eval (pair-based: text vs query audio cosine)
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls, ids = [], [], []
            for e, q, y, txts, id_ in ld:
                e, q = e.to(device), q.to(device)
                et = model.forward_text(txts)
                eq = model.forward_audio(q)
                cs = (et * eq).sum(-1)
                ps.append(torch.sigmoid(cs * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy()); ids.extend(id_)
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps)), \
                   np.concatenate(ps), np.concatenate(ls), ids

        as_, _, _, _ = ev(dv_s)
        au_, _, _, _ = ev(dv_u)

        print(f"[AT-PK ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={total_loss/n_batches:.4f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_,
                        "auc_seen": as_, "epoch": ep}, os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_,
                    "auc_seen": as_, "epoch": ep}, os.path.join(out_dir, "latest.pt"))


# ═══════════ Config ═══════════
class Config:
    embed_dim: int = 256
    sample_rate: int = 16000
    max_audio_sec: float = 1.5
    unfreeze_layers: int = 2
    epochs: int = 50
    lr: float = 3e-4
    batch_size: int = 128  # P*K = 32*4
    cos_scale: float = 8.0
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
            setattr(self, f"{k}_zip", z)
            setattr(self, f"{k}_csv", c)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr
    train_at_pk(cfg, args)

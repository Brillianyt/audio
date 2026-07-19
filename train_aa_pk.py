"""AA PKSampler + AngularPrototypicalLoss — target seen 0.9+."""
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


# ═══════════ WhisperEncoder (same as before) ═══════════
class WhisperEncoder(nn.Module):
    def __init__(self, model_name="base", embed_dim=256, unfreeze=6):
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


# ═══════════ PK Model — single encoder, outputs embeddings ═══════════
class AAPKModel(nn.Module):
    def __init__(self, embed_dim=256, unfreeze=8):
        super().__init__()
        self.encoder = WhisperEncoder("base", embed_dim, unfreeze)
        self.phoneme_head = nn.Linear(embed_dim, 40)
        self.ph_proj = nn.Linear(embed_dim + 40, embed_dim)
        # Comparison head for pair evaluation
        self.compare = nn.Sequential(
            nn.Linear(embed_dim * 3, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        emb = self.encoder(x)
        ph_logits = self.phoneme_head(emb)
        ph_probs = torch.sigmoid(ph_logits)
        aug = torch.cat([emb, ph_probs], dim=-1)
        fused = F.normalize(self.ph_proj(aug), dim=-1)
        return fused, ph_logits

    def pair_score(self, e, q):
        """Comparison head for (enroll_audio, query_audio) pairs."""
        ee, _ = self.forward(e)
        qq, _ = self.forward(q)
        feat = torch.cat([ee, qq, ee * qq], dim=-1)
        return self.compare(feat).squeeze(-1)


# ═══════════ Angular Prototypical Loss ═══════════
class AngularPrototypicalLoss(nn.Module):
    def __init__(self, init_w=10.0, init_b=-5.0):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(init_w))
        self.b = nn.Parameter(torch.tensor(init_b))

    def forward(self, embed, word_texts):
        device = embed.device
        unique_words = list(set(word_texts))
        P = len(unique_words)
        if P < 2:
            return embed.new_zeros(())

        word_to_idx = {w: i for i, w in enumerate(unique_words)}
        labels = torch.tensor([word_to_idx[w] for w in word_texts], device=device)

        # Split each word's audios into support (2/3) and query (1/3)
        query_mask = torch.zeros(len(word_texts), dtype=torch.bool, device=device)
        for w in unique_words:
            indices = torch.tensor([i for i, t in enumerate(word_texts) if t == w], device=device)
            n_q = max(1, len(indices) // 3)
            perm = torch.randperm(len(indices), device=device)
            query_mask[indices[perm[:n_q]]] = True

        support_mask = ~query_mask

        # Prototypes from support embeddings
        prototypes = torch.zeros(P, embed.shape[-1], device=device)
        for i, w in enumerate(unique_words):
            mask_w = torch.tensor([t == w for t in word_texts], device=device)
            mask_s = mask_w & support_mask
            if mask_s.sum() > 0:
                prototypes[i] = embed[mask_s].mean(dim=0)
            else:
                prototypes[i] = embed[mask_w].mean(dim=0)
        prototypes = F.normalize(prototypes, dim=1)

        # Query vs all prototypes
        query_emb = embed[query_mask]
        query_labels = labels[query_mask]
        cosine = torch.matmul(query_emb, prototypes.T)
        logits = self.w.clamp(min=1e-6) * cosine + self.b
        return F.cross_entropy(logits, query_labels)


# ═══════════ Phoneme utilities ═══════════
PHONEME_VOCAB = [
    "AA","AE","AH","AO","AW","AY","B","CH","D","DH",
    "EH","ER","EY","F","G","HH","IH","IY","JH","K",
    "L","M","N","NG","OW","OY","P","R","S","SH",
    "T","TH","UH","UW","V","W","Y","Z","ZH","UNK",
]
import re as _re
_cmu_cache = None
def _get_cmu():
    global _cmu_cache
    if _cmu_cache is None:
        import cmudict
        _cmu_cache = cmudict.dict()
    return _cmu_cache

def word_to_ph(word):
    cmu = _get_cmu()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w:
        return []
    plist = cmu.get(w)
    if plist:
        return [_re.sub(r'[0-2]$', '', p) for p in plist[0]]
    return []

def get_phoneme_multihot(word):
    """Return 40-dim multi-hot vector for a word's phonemes."""
    ph = word_to_ph(word)
    vec = [0.0]*40
    p2i = {"AA":0,"AE":1,"AH":2,"AO":3,"AW":4,"AY":5,"B":6,"CH":7,"D":8,"DH":9,
           "EH":10,"ER":11,"EY":12,"F":13,"G":14,"HH":15,"IH":16,"IY":17,"JH":18,"K":19,
           "L":20,"M":21,"N":22,"NG":23,"OW":24,"OY":25,"P":26,"R":27,"S":28,"SH":29,
           "T":30,"TH":31,"UH":32,"UW":33,"V":34,"W":35,"Y":36,"Z":37,"ZH":38,"UNK":39}
    for p in ph:
        idx = p2i.get(p, 39)
        if idx < 39:
            vec[idx] = 1.0
    return vec

# ═══════════ PKSampler ═══════════
class PKSampler:
    def __init__(self, word_to_indices: Dict[str, List[int]],
                 P: int = 32, K: int = 4, batches_per_epoch: int = 200,
                 hard_neg_map=None):
        """
        hard_neg_map: {word: [hard_neg_word1, hard_neg_word2, ...]}
        """
        self.word_to_indices = word_to_indices
        self.P = P
        self.K = K
        self.batches_per_epoch = batches_per_epoch
        self.valid_words = [w for w, idxs in word_to_indices.items()
                            if len(idxs) >= K]
        self.hard_neg_map = hard_neg_map or {}

    def __iter__(self):
        P = min(self.P, len(self.valid_words))
        if P == 0:
            return iter([])
        indices = []
        for _ in range(self.batches_per_epoch):
            # Pick P words, then inject hard neg counterparts
            words = set(np.random.choice(self.valid_words, P, replace=False))

            # Replace some words with their hard negatives
            inject_words = set()
            for w in list(words):
                hards = self.hard_neg_map.get(w, [])
                valid_hards = [h for h in hards if h in self.word_to_indices and h not in words]
                if valid_hards and np.random.random() < 0.4:
                    inject_words.add(np.random.choice(valid_hards))
                    words.remove(w)

            words |= inject_words
            words = list(words)
            np.random.shuffle(words)

            batch_indices = []
            for w in words:
                chosen = list(np.random.choice(self.word_to_indices[w],
                                               min(self.K, len(self.word_to_indices[w])), replace=False))
                batch_indices.extend(chosen)
            np.random.shuffle(batch_indices)
            indices.append(batch_indices)
        return iter(indices)

    def __len__(self):
        return self.batches_per_epoch


# ═══════════ Audio Dataset for PK ═══════════
class AudioWordDataset(Dataset):
    """Returns (audio_waveform, word_string) for a list of audio entries."""
    def __init__(self, entries, default_zip, cfg):
        """
        entries: list of {"id": audio_id, "word": word, "zip": optional_zip_path}
        """
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
    """Pad audio to same length, stack, return words."""
    ml = max(b[0].shape[-1] for b in batch)
    audios, words = [], []
    for wav, word in batch:
        if wav.shape[-1] < ml:
            wav = F.pad(wav, (0, ml - wav.shape[-1]))
        audios.append(wav)
        words.append(word)
    return torch.stack(audios), words


# ═══════════ Data loading — build word_to_indices ═══════════
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows


def build_word_audio_index(cfg):
    """Build word → [audio_id, ...] mapping from all data sources.
    Also builds entries list for AudioWordDataset.
    """
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

    # Build word→audio_ids from ALL positive pairs
    word_to_ids = defaultdict(set)
    seen_ids = set()
    entries = []
    for p in all_pairs:
        if p["label"] == 1:
            word = p["enroll_txt"].lower()
            eid = p.get("enroll_id", p["id"])
            qid = p.get("query_id", p["id"])
            # Add both enroll and query audio for this word
            for aid in [eid, qid]:
                if aid not in seen_ids:
                    seen_ids.add(aid)
                    word_to_ids[word].add(aid)
                    entries.append({"id": aid, "word": word,
                                    "zip": p.get("zip", cfg.train_zip)})

    # Filter words with at least K audios
    K = 4
    valid_words = {w for w, ids in word_to_ids.items() if len(ids) >= K}
    entries = [e for e in entries if e["word"] in valid_words]
    word_to_idx = {w: [i for i, e in enumerate(entries) if e["word"] == w]
                   for w in valid_words}

    print(f"  unique words: {len(valid_words)} (with ≥{K} audios)")
    print(f"  total audio samples: {len(entries)}")
    return word_to_idx, entries


# ═══════════ Training ═══════════
def train_aa_pk(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "aa_pk")
    os.makedirs(out_dir, exist_ok=True)

    print("[AA-PK] building word index...")
    word_to_idx, entries = build_word_audio_index(cfg)
    dataset = AudioWordDataset(entries, cfg.train_zip, cfg)

    # Build hard negative map from AA-mined pairs
    hn_path = os.path.join(PATHS.root, "baseline", "hard_neg_aa_final.json")
    hard_neg_map = {}
    if os.path.isfile(hn_path):
        hn_data = json.load(open(hn_path))
        for p in hn_data:
            w1, w2 = p["enroll_txt"].lower(), p["query_txt"].lower()
            hard_neg_map.setdefault(w1, set()).add(w2)
            hard_neg_map.setdefault(w2, set()).add(w1)
        hard_neg_map = {k: list(v) for k, v in hard_neg_map.items()}
        print(f"  loaded {len(hn_data)} hard neg pairs for PK sampling")

    sampler = PKSampler(word_to_idx, P=32, K=4, batches_per_epoch=300,
                        hard_neg_map=hard_neg_map)

    model = AAPKModel(cfg.embed_dim, cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        print(f"  loaded ckpt from {args.load_ckpt}")

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
    ang_loss_fn = AngularPrototypicalLoss()
    ph_crit = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Text encoder for anchoring (frozen, from AT model)
    text_enc = None
    if args.text_ckpt:
        print(f"[AA-PK] loading text encoder from {args.text_ckpt}...")
        ckpt_t = torch.load(args.text_ckpt, map_location=device, weights_only=False)
        from train_at_v3 import PhonemeBiGRUEncoder
        text_enc = PhonemeBiGRUEncoder(cfg.embed_dim).to(device)
        text_state = {k.replace("text_enc.", ""): v for k, v in ckpt_t["model"].items()
                      if k.startswith("text_enc.")}
        text_enc.load_state_dict(text_state, strict=False)
        text_enc.eval()
        for p in text_enc.parameters():
            p.requires_grad = False
        print(f"  text encoder loaded (frozen)")

    # Dev loaders (same pair-based eval as before)
    from train_aa import PairDataset, collate_text, load_pairs as lp
    def dev_ld(z, c):
        return DataLoader(PairDataset(lp(c), z, cfg),
                          batch_size=cfg.batch_size*2, num_workers=0,
                          collate_fn=collate_text, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        loader = DataLoader(dataset, batch_sampler=sampler,
                            num_workers=cfg.num_workers, collate_fn=collate_audio,
                            pin_memory=True)
        print(f"[ep{ep}] PK batches={len(sampler)} P=32 K=4")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        for audios, words in loader:
            audios = audios.to(device)
            # Noise augmentation
            snr = np.random.uniform(-10, 5)
            audios = audios + (10**(-snr/20)) * torch.randn_like(audios) * audios.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                embs, ph_logits = model(audios)
                # 1. Angular Prototypical Loss
                loss = ang_loss_fn(embs, words)

                # 2. Phoneme auxiliary loss (multi-label BCE)
                ph_targets = torch.tensor([get_phoneme_multihot(w) for w in words],
                                          device=device, dtype=torch.float32)
                ph_loss = ph_crit(ph_logits, ph_targets)
                loss = loss + 0.1 * ph_loss

                # 3. Text anchoring (pull audio emb toward text emb)
                if text_enc is not None:
                    unique_words = list(set(words))
                    with torch.no_grad():
                        te = text_enc(unique_words)[0]  # (P, D) embedding
                    word_to_t = {w: te[i] for i, w in enumerate(unique_words)}
                    t_targets = torch.stack([word_to_t[w] for w in words])
                    align_loss = 1 - (embs * t_targets).sum(-1).mean()
                    loss = loss + 0.2 * align_loss

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            total_loss += loss.item(); n_batches += 1
            if n_batches % cfg.log_every == 0:
                print(f"  [ep{ep}] b{n_batches} loss={total_loss/n_batches:.3f}")

        # Eval (same pair-based AUC)
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls, ids = [], [], []
            for e, q, y, txts, id_ in ld:
                e, q = e.to(device), q.to(device)
                cs = model.pair_score(e, q)
                ps.append(torch.sigmoid(cs * cfg.cos_scale).cpu().numpy())
                ls.append(y.numpy()); ids.extend(id_)
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps)), \
                   np.concatenate(ps), np.concatenate(ls), ids

        as_, _, _, _ = ev(dv_s)
        au_, _, _, _ = ev(dv_u)

        print(f"[AA-PK ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={total_loss/n_batches:.4f} ({time.time()-ts:.0f}s)")

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
    unfreeze_layers: int = 8
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
    p.add_argument("--text-ckpt", default="", help="AT checkpoint for frozen text encoder")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr
    train_aa_pk(cfg, args)

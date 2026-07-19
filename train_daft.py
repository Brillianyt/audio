"""Double-Audio Fused Text — unified model: fuses enrollment audio, text, and query audio."""
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


# ═══════════ WhisperEncoder ═══════════
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


# ═══════════ Phoneme BiGRU Text Encoder ═══════════
import re as _re
_cmu_cache = None
def _get_cmu():
    global _cmu_cache
    if _cmu_cache is None:
        import cmudict
        _cmu_cache = cmudict.dict()
    return _cmu_cache

PHONEME_TO_IDX = {"AA":0,"AE":1,"AH":2,"AO":3,"AW":4,"AY":5,"B":6,"CH":7,"D":8,"DH":9,
                  "EH":10,"ER":11,"EY":12,"F":13,"G":14,"HH":15,"IH":16,"IY":17,"JH":18,
                  "K":19,"L":20,"M":21,"N":22,"NG":23,"OW":24,"OY":25,"P":26,"R":27,
                  "S":28,"SH":29,"T":30,"TH":31,"UH":32,"UW":33,"V":34,"W":35,"Y":36,
                  "Z":37,"ZH":38,"UNK":39}
_p2i_cache = {}

def _text_to_phoneme_ids(text):
    if text not in _p2i_cache:
        cmu = _get_cmu()
        w = text.lower().strip("'s\"-.,!?;:")
        ph = []
        if w:
            plist = cmu.get(w)
            if plist:
                ph = [_re.sub(r'[0-2]$', '', p) for p in plist[0]]
        _p2i_cache[text] = [PHONEME_TO_IDX.get(p, 39) for p in ph] if ph else [39]
    return _p2i_cache[text]


class PhonemeBiGRUEncoder(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.emb = nn.Embedding(40, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0, 256, device=device)
        ph_lists = [_text_to_phoneme_ids(t) for t in texts]
        mx = max(len(p) for p in ph_lists)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, ph in enumerate(ph_lists):
            for j, pid in enumerate(ph[:mx]):
                idx[i, j] = pid
        x = self.emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1)


# ═══════════ Deep Fusion Model ═══════════
class DoubleAudioFusedText(nn.Module):
    """Deep fusion: text gates audio, then compare modulated features."""
    def __init__(self, embed_dim=256, unfreeze=2):
        super().__init__()
        self.audio_enc = WhisperEncoder("base", embed_dim, unfreeze)
        self.text_enc = PhonemeBiGRUEncoder(embed_dim)
        # Simple text gate
        self.text_gate = nn.Linear(embed_dim, embed_dim)
        # Shallow fusion: concat(ea_gated, eq_gated, ea_gated*eq_gated)
        self.fusion = nn.Linear(embed_dim * 3, 1)
        # Very small init so residual dominates
        nn.init.normal_(self.fusion.weight, std=0.0001)
        nn.init.zeros_(self.fusion.bias)

    def forward(self, enroll_audio, texts, query_audio):
        ea = self.audio_enc(enroll_audio)
        et = self.text_enc(texts)
        eq = self.audio_enc(query_audio)

        # Text gate modulates both audio embeddings
        gate = torch.sigmoid(self.text_gate(et))
        ea_gated = ea * gate
        eq_gated = eq * gate

        # Residual: raw cosine similarity
        cos_raw = (et * eq).sum(-1, keepdim=True)

        # Fuse gated features
        fused = torch.cat([ea_gated, eq_gated, ea_gated * eq_gated], dim=-1)
        # cos(et,eq) dominates, fusion is tiny correction on top
        logit = cos_raw * 5.0 + 0.1 * self.fusion(fused)
        return logit.squeeze(-1), ea, et, eq


# ═══════════ Data ═══════════
from train_at_v3 import PairDataset as DevDataset, collate_text as dev_collate, load_pairs as lp

def load_pairs(csv_path):
    return lp(csv_path)

def deduplicate_by_word_pair(pairs, max_per_pair=1):
    seen = {}
    deduped = []
    for p in pairs:
        key = (p["enroll_txt"].lower(), p["query_txt"].lower())
        if key not in seen: seen[key] = 0
        if seen[key] < max_per_pair:
            seen[key] += 1
            deduped.append(p)
    return deduped

def load_all_data(cfg):
    """Load pre-cleaned pairs."""
    fp = os.path.join(PATHS.root, "train", "cleaned_pairs.json")
    all_pairs = json.load(open(fp))
    print(f"  loaded {len(all_pairs)} cleaned pairs")

    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]

    # No further dedup needed — already cleaned
    hard_neg = [p for p in neg_pairs if any(k in p.get("id","") for k in {"hard_neg","hn_","phoneme","hnat_"})]
    hard_ids = {p["id"] for p in hard_neg}
    easy_neg = [p for p in neg_pairs if p["id"] not in hard_ids]

    print(f"  pos={len(pos_pairs)} hard_neg={len(hard_neg)} easy_neg={len(easy_neg)}")
    return pos_pairs, hard_neg, easy_neg

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path, cfg):
        self.pairs = pairs; self.default_zip_path = zip_path; self.cfg = cfg; self._zc = {}
    def _get_zip_for(self, pair):
        zpath = pair.get("zip","") or self.default_zip_path
        if zpath and not zpath.startswith("/"): zpath = os.path.join(PATHS.root, zpath)
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
        wav = wav.astype(np.float32)
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()
    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        zf = self._get_zip_for(p)
        return self._read_wav(eid, "enroll", zf), self._read_wav(qid, "query", zf), \
               float(p.get("label",0)), p.get("enroll_txt","").lower(), pid

def collate_text(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, ls, txts, ids = [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, ml-e.shape[-1])) if e.shape[-1]<ml else e)
        qs.append(F.pad(q, (0, ml-q.shape[-1])) if q.shape[-1]<ml else q)
        ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return torch.stack(es), torch.stack(qs), torch.tensor(ls, dtype=torch.float32), txts, ids


# ═══════════ Training ═══════════
def train_daft(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42); np.random.seed(42)
    out_dir = os.path.join(PATHS.root, "output", "daft_v1")
    os.makedirs(out_dir, exist_ok=True)

    print("[DAFT] loading data...")
    pos_dedup, hard_dedup, easy_dedup = load_all_data(cfg)

    model = DoubleAudioFusedText(cfg.embed_dim, cfg.unfreeze_layers).to(device)
    if args.load_ckpt:
        ckpt = torch.load(args.load_ckpt, map_location=device, weights_only=False)
        # Map AT checkpoint keys (encoder.*) to DAFT keys (audio_enc.*)
        mapped = {}
        for k, v in ckpt["model"].items():
            if k.startswith("encoder."):
                mapped["audio_enc." + k[len("encoder."):]] = v
            elif k.startswith("text_enc."):
                mapped[k] = v
        model.load_state_dict(mapped, strict=False)
        print(f"  loaded encoder+text from {args.load_ckpt}")
        if "auc_unseen" in ckpt:
            print(f"    AT best unseen={ckpt['auc_unseen']:.4f}")

    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed ep{start_ep}")
                break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable params: {tr:,}")

    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    text_p = [p for n,p in model.named_parameters() if p.requires_grad and "text_enc" in n]
    fusion_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n and "text_enc" not in n]
    opt = torch.optim.AdamW([
        {"params":wav_p, "lr":cfg.lr/5},
        {"params":text_p, "lr":1e-3},
        {"params":fusion_p, "lr":1e-3},  # fusion head
    ], weight_decay=1e-4)
    total_steps = cfg.epochs * 360000 // cfg.batch_size
    warmup_steps = max(100, total_steps // 20)
    def lr_lambda(step):
        if step < warmup_steps: return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Dev loaders
    def dev_ld(z, c):
        return DataLoader(DevDataset(load_pairs(c), z, cfg), batch_size=cfg.batch_size*2, num_workers=0, collate_fn=dev_collate, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    for ep in range(start_ep, cfg.epochs + 1):
        n_pos = min(60000, len(pos_dedup))
        n_hard = min(300000, len(hard_dedup) * 10)
        idx_p = np.random.permutation(len(pos_dedup))[:n_pos]
        idx_h = np.random.choice(len(hard_dedup), n_hard, replace=True)
        subset = [pos_dedup[i] for i in idx_p] + [hard_dedup[i] for i in idx_h]
        np.random.shuffle(subset)

        loader = DataLoader(PairDataset(subset, cfg.train_zip, cfg),
                            batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, collate_fn=collate_text,
                            pin_memory=True, drop_last=True)
        print(f"[ep{ep}] train={len(subset)} pos={n_pos} hard={n_hard}")

        model.train(); ts = time.time(); total_loss = 0.0; n_batches = 0
        cos_pos, cos_neg, cos_n = 0.0, 0.0, 0

        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            snr = np.random.uniform(-10, 5)
            e = e + (10**(-snr/20)) * torch.randn_like(e) * e.std(-1, keepdim=True)
            q = q + (10**(-snr/20)) * torch.randn_like(q) * q.std(-1, keepdim=True)

            with torch.cuda.amp.autocast():
                logit, ea, et, eq = model(e, txts, q)
                loss = crit(logit, y)

            opt.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt); scaler.update(); sched.step()

            total_loss += loss.item(); n_batches += 1
            cs = (et * eq).sum(-1).detach()
            pos_m = (y==1); neg_m = (y==0)
            cos_pos += cs[pos_m].mean().item() if pos_m.any() else 0
            cos_neg += cs[neg_m].mean().item() if neg_m.any() else 0
            cos_n += 1
            if cos_n % cfg.log_every == 0:
                print(f"  [ep{ep}] b{cos_n} loss={total_loss/n_batches:.3f} "
                      f"cos+={cos_pos/cos_n:.3f} cos-={cos_neg/cos_n:.3f}")

        # Eval
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, ls = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                logit, _, _, _ = model(e, txts, q)
                ps.append(torch.sigmoid(logit).cpu().numpy())
                ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[DAFT ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={total_loss/n_batches:.4f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))


# ═══════════ Config ═══════════
class Config:
    embed_dim: int = 256; sample_rate: int = 16000; max_audio_sec: float = 1.5
    unfreeze_layers: int = 2; epochs: int = 50; lr: float = 3e-4; batch_size: int = 512
    pos_weight: float = 5.0; num_workers: int = 8; log_every: int = 50
    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")
        for k in ("dev_seen","dev_unseen"):
            setattr(self, f"{k}_zip", os.path.join(r, "dev", k, "wav.zip"))
            setattr(self, f"{k}_csv", os.path.join(r, "dev", k, f"{k}_label.csv"))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--load-ckpt", default="")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bs", type=int, default=512)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg = Config(); cfg.__post_init__()
    cfg.epochs = args.epochs; cfg.lr = args.lr; cfg.batch_size = args.bs
    train_daft(cfg, args)

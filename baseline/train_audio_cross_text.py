"""Audio-Cross-Text: frame-level cross-attention for query-by-example KWS.

Enroll audio → WavLM (frozen) → frame features
Enroll text → CharEmb + Transformer → token features
→ Cross-attention (text attends to audio)
→ Pooling → cosine similarity → posterior

Usage:
    python baseline/train_audio_cross_text.py --epochs 15 --bs 128
"""
import argparse, csv, gc, json, math, os, time, io, zipfile, re as _re
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ── Config ──
class Config:
    embed_dim: int = 256
    sample_rate: int = 16000
    max_sec: float = 1.5
    epochs: int = 15
    lr: float = 3e-4
    batch_size: int = 128
    pos_weight: float = 5.0
    num_workers: int = 4
    seed: int = 42
    unfreeze_wavlm: int = 0  # 0 = fully frozen

    def __post_init__(self):
        r = ROOT
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
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


# ── Model: Audio-Cross-Text ──
class AudioCrossText(nn.Module):
    """Frame-level cross-attention between audio and text."""
    def __init__(self, embed_dim=256, unfreeze=0, max_sec=1.5):
        super().__init__()
        self.max_sec = max_sec
        from transformers import WavLMModel
        wlm_path = os.path.join(ROOT, "wavlm")
        self.wavlm = WavLMModel.from_pretrained(wlm_path, local_files_only=True)
        self.wavlm.eval()
        for p in self.wavlm.parameters(): p.requires_grad = False
        if unfreeze > 0:
            total = len(self.wavlm.encoder.layers)
            for i in range(total - unfreeze, total):
                for p in self.wavlm.encoder.layers[i].parameters():
                    p.requires_grad = True
        self.audio_proj = nn.Linear(self.wavlm.config.hidden_size, embed_dim)

        # Text encoder
        self.char_emb = nn.Embedding(28, embed_dim)
        self.pos_emb = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.1)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim*2,
            dropout=0.1, activation='gelu', batch_first=True)
        self.text_enc = nn.TransformerEncoder(enc_layer, num_layers=2)

        # Cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)

        # Logit scale
        self.scale = nn.Parameter(torch.tensor(8.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def _audio_feats(self, wav):
        ms = int(self.max_sec * 16000)
        if wav.shape[-1] > ms: wav = wav[:, :ms]
        self.wavlm = self.wavlm.to(wav.device)
        with torch.no_grad():
            out = self.wavlm(wav, output_hidden_states=False, return_dict=True)
        return self.audio_proj(out.last_hidden_state)  # (B, T_a, D)

    def _text_feats(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0, 1, 256, device=device), None
        mx = min(max(len(t) for t in texts), 64)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        mask = torch.ones(len(texts), mx, dtype=torch.bool, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
                mask[i,j] = False
        x = self.char_emb(idx) + self.pos_emb[:, :mx, :]
        return self.text_enc(x, src_key_padding_mask=mask), mask

    def forward(self, e, texts):
        af = self._audio_feats(e)        # (B, T_a, D)
        tf, mask = self._text_feats(texts)  # (B, T_t, D)
        # Cross-attend: text queries → audio keys/values
        attn_out, _ = self.cross_attn(tf, af, af)  # (B, T_t, D)
        # Mean pool attended text features (valid tokens only)
        if mask is not None:
            lengths = (~mask).sum(dim=1, keepdim=True).float().clamp(min=1)
            et = attn_out.masked_fill(mask.unsqueeze(-1), 0).sum(dim=1) / lengths
        else:
            et = attn_out.mean(dim=1)
        # Pool audio to single vector
        ea = af.mean(dim=1)
        # L2 normalize
        ea = F.normalize(ea, dim=-1)
        et = F.normalize(et, dim=-1)
        logit = (ea * et).sum(-1) * self.scale + self.bias
        return logit, ea, et


# ── Data ──
_load_pairs_cache = None
_zip_cache = {}

def _get_zip(path):
    key = (os.getpid(), path)
    if key not in _zip_cache:
        _zip_cache[key] = zipfile.ZipFile(path, "r")
    return _zip_cache[key]

def read_wav(zip_path, pid, role):
    z = _get_zip(zip_path)
    try:
        data = z.read(f"wav/{pid}_{role}.wav")
    except KeyError:
        data = z.read(f"wav/{pid}.wav")  # fallback for flat utterance files
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio as _ta
        wav = _ta.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    wav = wav.astype(np.float32)[:int(1.5*16000)]
    return torch.from_numpy(wav).float()

def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows

class PairDataset(Dataset):
    def __init__(self, pairs, zip_path):
        self.pairs = pairs
        self.zip_path = zip_path

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        pid = p["id"]
        zip_path = os.path.join(ROOT, p["zip"]) if p.get("zip") else self.zip_path
        eid = p.get("enroll_id", pid)
        qid = p.get("query_id", pid)
        e = read_wav(zip_path, eid, "enroll")
        q = read_wav(zip_path, qid, "query")
        txt = p.get("enroll_txt", "").lower()
        return e, q, p["label"], txt, pid

def collate(batch):
    ml = max(max(b[0].shape[0], b[1].shape[0]) for b in batch)
    es, qs, ls, txts, ids = [], [], [], [], []
    for b in batch:
        e = F.pad(b[0], (0, ml - b[0].shape[0])) if b[0].shape[0] < ml else b[0]
        q = F.pad(b[1], (0, ml - b[1].shape[0])) if b[1].shape[0] < ml else b[1]
        es.append(e); qs.append(q); ls.append(b[2]); txts.append(b[3]); ids.append(b[4])
    return (torch.stack(es), torch.stack(qs),
            torch.tensor(ls, dtype=torch.float32), txts, ids)


# ── Noise ──
_noise_bank = None
def _load_noise_bank():
    global _noise_bank
    if _noise_bank is not None: return _noise_bank if len(_noise_bank) else None
    pt = os.path.join(ROOT, "train_noise_bank.pt")
    if os.path.isfile(pt):
        _noise_bank = torch.load(pt, map_location="cpu", weights_only=True)
        print(f"  [noise] bank: {len(_noise_bank)/16000:.0f}s")
        return _noise_bank
    _noise_bank = torch.zeros(0)
    return None

def mix_noise_batch(wav, bank, p_clean=0.5, snr_lo=-10.0, snr_hi=5.0):
    B, L = wav.shape
    use = torch.rand(B, device=wav.device) >= p_clean
    if not use.any(): return wav
    if bank is not None:
        nb = bank.to(wav.device)
        if len(nb) <= L: nb = nb.repeat(L // max(1, len(nb)) + 1)
        offset = torch.randint(0, len(nb) - L, (B,), device=wav.device)
        noise = nb[offset.unsqueeze(1) + torch.arange(L, device=wav.device).unsqueeze(0)]
    else:
        noise = torch.randn_like(wav)
    snr = torch.rand(B, 1, device=wav.device) * (snr_hi - snr_lo) + snr_lo
    sig_p = wav.pow(2).mean(-1, keepdim=True).clamp(min=1e-8)
    noi_p = noise.pow(2).mean(-1, keepdim=True).clamp(min=1e-8)
    scale = (sig_p / (noi_p * 10 ** (snr / 10.0))).sqrt()
    return torch.where(use.unsqueeze(1), wav + noise * scale, wav)


# ── Training ──
def train(cfg, args=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(ROOT, "output", "audio_cross_text")
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
               "baseline/hard_neg_iter2.json", "train/self_paired.json",
               "train/external_pairs.json", "train/external_pairs_v2.json",
               "train/self_paired_xl.json", "train/fill_pos_pairs.json",
               "train/mega_pairs.json"]:
        hp = os.path.join(ROOT, hn)
        if os.path.isfile(hp):
            all_pairs += json.load(open(hp))
    rng.shuffle(all_pairs)
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"[data] {len(all_pairs)} pairs ({len(pos_pairs)} pos, {len(neg_pairs)} neg)")

    def get_loader(ep):
        n_pos = min(100000, len(pos_pairs))
        n_neg = min(400000, len(neg_pairs))
        subset = rng.choice(pos_pairs, n_pos, replace=False).tolist()
        subset += rng.choice(neg_pairs, n_neg, replace=False).tolist()
        rng.shuffle(subset)
        return DataLoader(PairDataset(subset, cfg.train_zip), batch_size=cfg.batch_size,
                          shuffle=True, num_workers=cfg.num_workers,
                          collate_fn=collate, pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        return DataLoader(PairDataset(load_pairs(c), z), batch_size=cfg.batch_size*2,
                          num_workers=0, collate_fn=collate, shuffle=False)
    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    model = AudioCrossText(cfg.embed_dim, cfg.unfreeze_wavlm).to(device)
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")
    
    best, start_ep = -1.0, 1
    if args is not None and args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model"], strict=False)
                best = ckpt.get("auc_unseen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed from {fp} epoch={start_ep} best_unseen={best:.4f}")
                break

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))

    noise_bank = _load_noise_bank()
    best = -1.0

    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train()
        ts = time.time()
        ls, n = 0.0, 0
        cs_pos, cs_neg = 0.0, 0.0

        for e, q, y, txts, _ in loader:
            e, q, y = e.to(device), q.to(device), y.to(device)
            # Noise on query
            q = mix_noise_batch(q, noise_bank, p_clean=0.5, snr_lo=-10.0, snr_hi=5.0)

            logit, ea, et = model(e, txts)
            cos_tq = (et * model._audio_feats(q).mean(1)).sum(-1)

            loss = crit(logit, y)
            # Margin loss
            pos = (y == 1); neg = (y == 0)
            margin = torch.tensor(0.0, device=device)
            if pos.any(): margin += F.relu(0.6 - cos_tq[pos]).mean()
            if neg.any(): margin += F.relu(cos_tq[neg] + 0.15).mean()
            loss += 0.05 * margin

            opt.zero_grad(); loss.backward(); opt.step()
            ls += loss.item(); n += 1

            cp = cos_tq[pos].mean().item() if pos.any() else 0.0
            cn = cos_tq[neg].mean().item() if neg.any() else 0.0
            cs_pos += cp; cs_neg += cn

            if n % 100 == 0:
                print(f"  [ep{ep}] b{n} loss={ls/n:.3f} cos+={cs_pos/n:.3f} cos-={cs_neg/n:.3f} gap={(cs_pos-cs_neg)/n:.3f}")

        # Eval
        @torch.no_grad()
        def ev(ld):
            model.eval()
            ps, ls_list = [], []
            for e, q, y, txts, _ in ld:
                e, q = e.to(device), q.to(device)
                logit, _, et = model(e, txts)
                cos_tq = (et * model._audio_feats(q).mean(1)).sum(-1)
                ps.append(torch.sigmoid(cos_tq * model.scale + model.bias).cpu().numpy())
                ls_list.append(y.numpy())
            return roc_auc_score(np.concatenate(ls_list), np.concatenate(ps))

        as_ = ev(dv_s); au_ = ev(dv_u)
        print(f"[ep{ep}] seen={as_:.4f} unseen={au_:.4f} loss={ls/n:.4f} ({time.time()-ts:.0f}s)")

        if au_ > best:
            best = au_
            torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                       os.path.join(out_dir, "best.pt"))
        torch.save({"model": model.state_dict(), "auc_unseen": au_, "auc_seen": as_, "epoch": ep},
                   os.path.join(out_dir, "latest.pt"))

    print(f"done. best unseen AUC = {best:.4f}")


# ── Main ──
def main():
    cfg = Config(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--unfreeze", type=int, default=cfg.unfreeze_wavlm)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg.epochs, cfg.lr, cfg.batch_size, cfg.unfreeze_wavlm = args.epochs, args.lr, args.bs, args.unfreeze
    train(cfg, args)

if __name__ == "__main__":
    main()

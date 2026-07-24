"""QbyOmni: Multi-modal query-by-example KWS with global + keyframe branches.

Architecture:
  Inputs: Query Audio, Enroll Audio, Enroll Text, Enroll Phoneme
  → Frozen pretrained encoders (XLS-R, DistilBERT, G2P)
  → Positional Encoding + Modality Embedding
  → Global Context Branch (Self-Attention + GRU)
  → Keyframe Branch (CTC Selector + Cosine Similarity + Attention Pooling)
  → Mean Pooling + Fusion + Classification

Loss: BCE + frame-BCE(phoneme) + frame-BCE(text) + 0.2*CTC
"""
import argparse, csv, io, json, math, os, sys, time, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS


# ═══════════ Config ═══════════
class QbyOmniConfig:
    dim: int = 128
    sample_rate: int = 16000
    max_audio_sec: float = 3.0
    epochs: int = 50
    lr: float = 1e-3
    batch_size: int = 16
    grad_accum: int = 32  # effective batch = 512
    num_workers: int = 0
    seed: int = 42
    log_every: int = 5
    pos_weight: float = 5.0
    ctc_weight: float = 0.2
    modality_dropout: float = 0.5

    train_zip: str = ""; train_csv: str = ""
    dev_seen_zip: str = ""; dev_seen_csv: str = ""
    dev_unseen_zip: str = ""; dev_unseen_csv: str = ""
    eval_seen_zip: str = ""; eval_seen_csv: str = ""
    eval_unseen_zip: str = ""; eval_unseen_csv: str = ""

    def __post_init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        db = os.path.join(r, "dev")
        if os.path.isdir(os.path.join(db, "dev")): db = os.path.join(db, "dev")
        self.dev_seen_zip = os.path.join(db, "dev_seen", "wav.zip")
        self.dev_seen_csv = os.path.join(db, "dev_seen", "dev_seen_label.csv")
        self.dev_unseen_zip = os.path.join(db, "dev_unseen", "wav.zip")
        self.dev_unseen_csv = os.path.join(db, "dev_unseen", "dev_unseen_label.csv")


# ═══════════ Sinusoidal Positional Encoding ═══════════
class PositionalEncoding(nn.Module):
    def __init__(self, dim=128, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.shape[1], :]


# ═══════════ Pretrained Encoders (Frozen) ═══════════
def _find_hf_cache(repo):
    base = os.path.expanduser("~/.cache/huggingface/hub")
    model_dir = os.path.join(base, f"models--{repo.replace('/', '--')}", "snapshots")
    if os.path.isdir(model_dir):
        snapshots = [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]
        if snapshots:
            return os.path.join(model_dir, snapshots[0])
    return repo


class XLSREncoder(nn.Module):
    """Frozen XLS-R encoder → audio projection to dim."""
    def __init__(self, dim=128):
        super().__init__()
        from transformers import Wav2Vec2Model
        path = _find_hf_cache("facebook/wav2vec2-xls-r-300m")
        self.xlsr = Wav2Vec2Model.from_pretrained(path, local_files_only=(path != "facebook/wav2vec2-xls-r-300m"))
        self.xlsr.eval()
        for p in self.xlsr.parameters(): p.requires_grad = False
        self.dim = self.xlsr.config.hidden_size  # 1024
        self.proj = nn.Linear(self.dim, dim)

    def forward(self, wav):
        with torch.no_grad():
            out = self.xlsr(wav, output_hidden_states=False, return_dict=True)
        return F.normalize(self.proj(out.last_hidden_state), dim=-1)


class DistilBERTEncoder(nn.Module):
    """Frozen DistilBERT → text projection to dim."""
    def __init__(self, dim=128):
        super().__init__()
        from transformers import DistilBertModel
        path = _find_hf_cache("distilbert-base-uncased")
        self.bert = DistilBertModel.from_pretrained(path, local_files_only=(path != "distilbert-base-uncased"))
        self.bert.eval()
        for p in self.bert.parameters(): p.requires_grad = False
        self.dim = self.bert.config.hidden_size  # 768
        self.proj = nn.Linear(self.dim, dim)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return F.normalize(self.proj(out.last_hidden_state), dim=-1)


PHONEME_MAP = {
    'AA':'AA','AE':'AE','AH':'AH','AO':'AO','AW':'AW','AY':'AY',
    'B':'B','CH':'CH','D':'D','DH':'DH','EH':'EH','ER':'ER','EY':'EY',
    'F':'F','G':'G','HH':'HH','IH':'IH','IY':'IY','JH':'JH','K':'K',
    'L':'L','M':'M','N':'N','NG':'NG','OW':'OW','OY':'OY','P':'P',
    'R':'R','S':'S','SH':'SH','T':'T','TH':'TH','UH':'UH','UW':'UW',
    'V':'V','W':'W','Y':'Y','Z':'Z','ZH':'ZH',
}
PHONEMES = sorted(set(PHONEME_MAP.values()))
PHONE_TO_ID = {p: i for i, p in enumerate(PHONEMES)}
PHONE_VOCAB_SIZE = len(PHONEMES) + 1  # +1 for blank


class PhonemeEncoder(nn.Module):
    """G2P embedding: input=pre-computed phone_ids tensor [B, Lp], ID=0 reserved for CTC blank."""
    def __init__(self, vocab_size=65, emb_dim=64, out_dim=128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)  # ID=0 is blank/padding
        self.proj = nn.Linear(emb_dim, out_dim)

    def forward(self, phone_ids):
        return F.normalize(self.proj(self.emb(phone_ids)), dim=-1)


# G2P utility (call in DataLoader preprocessing, NOT in model forward)
def text_to_phoneme_ids(text, g2p):
    """Convert text to phoneme ID list. ID=0 reserved for CTC blank."""
    phones = g2p(text)
    clean = [p for p in phones if p.strip() and p not in "',+-.!?;:"]
    ids = [PHONE_TO_ID.get(p, 1) for p in clean]  # 1=AA fallback, skip ID=0
    return [i for i in ids if i > 0]  # filter blank

PHONEME_VOCAB_SIZE_CTC = len(PHONEMES) + 1  # +1 for blank (ID=0)


# ═══════════ QbyOmni Model ═══════════
class QbyOmniModel(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        self.dim = dim
        # Encoders
        self.audio_enc = XLSREncoder(dim)
        self.text_enc = DistilBERTEncoder(dim)
        self.phone_enc = PhonemeEncoder(emb_dim=64, out_dim=dim)

        # Positional encoding
        self.pos_enc = PositionalEncoding(dim)
        # Modality embeddings (learnable)
        self.mod_emb = nn.Embedding(3, dim)  # 0=audio, 1=text, 2=phone

        # Self-Attention (global context, paper QbyOmni branch)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=4, dim_feedforward=512,
                                                 dropout=0.1, activation='gelu', batch_first=True)
        self.sa = nn.TransformerEncoder(enc_layer, num_layers=2)
        # GRU+FC for frame-level losses (paper: F^c_m = FC(GRU(Ē_sm)))
        self.gru = nn.GRU(dim, 64, batch_first=True, bidirectional=False)
        self.gru_fc = nn.Linear(64, dim)

        # CTC Selector
        self.ctc_proj = nn.Linear(dim, PHONEME_VOCAB_SIZE_CTC)

        # Frame-level classifiers for aux loss
        self.frame_cls_phone = nn.Linear(dim, 1)
        self.frame_cls_text = nn.Linear(dim, 1)

        # Final classifier (3 modalities concat: audio + phone + text)
        self.classifier = nn.Sequential(
            nn.Linear(dim * 3, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def _extract_keyframes(self, raw_feat, ctc_logits):
        """CTC-guided keyframe extraction (paper Section 2.1 QbyKeyframe).
        Args:
            raw_feat: (B, T, D) — raw audio features BEFORE PE/ME
            ctc_logits: (B, T, V+1) — CTC posteriors
        Returns:
            K: (B, Tp, D) — padded keyframe features
        """
        B, T, D = raw_feat.shape
        keyframes = []
        for b in range(B):
            preds = ctc_logits[b].argmax(dim=-1)
            seen = set()
            peaks = []
            for t in range(T):
                p = preds[t].item()
                if p != 0 and p not in seen:
                    peaks.append(t)
                    seen.add(p)
            if peaks:
                kf = []
                for t in peaks:
                    s, e = max(0, t - 2), min(T, t + 3)
                    kf.append(raw_feat[b, s:e].mean(dim=0))
                keyframes.append(torch.stack(kf))
            else:
                keyframes.append(raw_feat[b, :1])
        max_kf = max(k.shape[0] for k in keyframes)
        return torch.stack([F.pad(k, (0, 0, 0, max_kf - k.shape[0])) for k in keyframes])

    def forward(self, q_wav, s_wav, text_ids, text_mask, phone_ids, modality_dropout=0.0):
        # ── 1. Raw feature extraction (frozen encoders) ──
        Eq_raw = self.audio_enc(q_wav)  # (B, Tq, D)
        Es_raw = self.audio_enc(s_wav)  # (B, Ts, D)
        Et_raw = self.text_enc(text_ids, text_mask)  # (B, Lt, D)
        Ep_raw = self.phone_enc(phone_ids)  # (B, Lp, D)

        # ── 2. CTC Selector on RAW features (paper Eq.9, both audio streams) ──
        ctc_logits_q = self.ctc_proj(Eq_raw)
        ctc_logits_s = self.ctc_proj(Es_raw)

        # ── 3. Position + Modality Encoding ──
        mod = self.mod_emb(torch.tensor([0, 1, 2], device=Eq_raw.device))
        Eq = self.pos_enc(Eq_raw + mod[0])
        Es = self.pos_enc(Es_raw + mod[0])
        Et = self.pos_enc(Et_raw + mod[1])
        Ep = self.pos_enc(Ep_raw + mod[2])

        # ── 4. Modality Dropout ──
        if self.training and modality_dropout > 0:
            if torch.rand(1).item() < modality_dropout: Es = 0.0 * Es
            if torch.rand(1).item() < modality_dropout: Et = 0.0 * Et
            if torch.rand(1).item() < modality_dropout: Ep = 0.0 * Ep

        B, Tq, D = Eq.shape
        Ts, Lp, Lt = Es.shape[1], Ep.shape[1], Et.shape[1]

        # ═══════════ QbyOmni Branch (Global Context) ═══════════
        # ── 5a. Self-Attention → E_bar (goes UP to CA as K/V) ──
        cat_a = torch.cat([Eq, Es], dim=1)
        cat_p = torch.cat([Eq, Ep], dim=1)
        cat_t = torch.cat([Eq, Et], dim=1)

        E_bar_a = self.sa(cat_a)  # (B, Tq+Ts, D)
        E_bar_p = self.sa(cat_p)  # (B, Tq+Lp, D)
        E_bar_t = self.sa(cat_t)  # (B, Tq+Lt, D)

        # Slice to enroll-only portion for CA
        E_bar_a_s = E_bar_a[:, -Ts:, :]    # (B, Ts, D)
        E_bar_p_s = E_bar_p[:, -Lp:, :]    # (B, Lp, D)
        E_bar_t_s = E_bar_t[:, -Lt:, :]    # (B, Lt, D)

        # ── 5b. GRU+FC → F^c (goes DOWN to frame losses) ──
        # Only phone and text branches have frame-level supervision (paper Eq.8)
        Fp = self.gru_fc(self.gru(E_bar_p)[0][:, -Lp:, :])  # (B, Lp, D)
        Ft = self.gru_fc(self.gru(E_bar_t)[0][:, -Lt:, :])  # (B, Lt, D)

        # ═══════════ QbyKeyframe Branch ═══════════
        # ── 6. Keyframe extraction from QUERY audio ──
        Kq = self._extract_keyframes(Eq_raw, ctc_logits_q)  # (B, Tp, D)

        # ── 7. Cosine Similarity (paper Eq.5) ──
        def cos_sim(A, B):
            return torch.bmm(F.normalize(A, dim=-1), F.normalize(B, dim=-1).transpose(1, 2))

        # All three use E_bar (SA output) as the comparison target
        M_aa = cos_sim(Kq, E_bar_a_s)   # (B, Tp, Ts) query keyframes vs enroll SA feats
        M_ap = cos_sim(Kq, E_bar_p_s)   # (B, Tp, Lp)  query keyframes vs phone SA feats
        M_at = cos_sim(Kq, E_bar_t_s)   # (B, Tp, Lt)  query keyframes vs text SA feats

        # ── 8. Cross-Attention: softmax(M) @ E_bar (paper Eq.6) ──
        scale = 10.0
        Out_a = torch.bmm(F.softmax(M_aa * scale, dim=-1), E_bar_a_s)  # (B, Tp, D)
        Out_p = torch.bmm(F.softmax(M_ap * scale, dim=-1), E_bar_p_s)  # (B, Tp, D)
        Out_t = torch.bmm(F.softmax(M_at * scale, dim=-1), E_bar_t_s)  # (B, Tp, D)

        # ── 9. Fusion: concat modality-specific pooling → classifier ──
        pooled = torch.cat([
            Out_a.mean(dim=1),   # (B, D)
            Out_p.mean(dim=1),   # (B, D)
            Out_t.mean(dim=1),   # (B, D)
        ], dim=-1)                # (B, 3*D)
        score = self.classifier(pooled).squeeze(-1)

        return {
            "score": score,
            "ctc_logits_q": ctc_logits_q,
            "ctc_logits_s": ctc_logits_s,
            "frame_phone": self.frame_cls_phone(Fp).squeeze(-1),
            "frame_text": self.frame_cls_text(Ft).squeeze(-1),
        }

# Module-level G2P cache (persists across epochs)
_G2P_CACHE = {}

class QbyOmniDataset(Dataset):
    """Same as PairDataset from train_dual.py. Uses module-level G2P cache."""
    def __init__(self, pairs, zip_path, cfg, g2p=None, phone_vocab=None):
        self.pairs = pairs; self.zip_path = zip_path; self.cfg = cfg
        self._zip_cache = {}
        if g2p is not None and phone_vocab is not None:
            # Precompute BOTH enroll_txt and query_txt phonemes
            for p in pairs:
                for key in ("enroll_txt", "query_txt"):
                    txt = p.get(key, "").lower()
                    if txt and txt not in _G2P_CACHE:
                        phones = g2p(txt)
                        clean = [ph for ph in phones if ph.strip() and ph not in "',+-.!?;:"]
                        ids = [phone_vocab.get(ph, 1) for ph in clean]
                        ids = [i for i in ids if i > 0]
                        _G2P_CACHE[txt] = ids if ids else [1]
    def _get_zip(self):
        pid = os.getpid()
        if pid not in self._zip_cache:
            self._zip_cache[pid] = zipfile.ZipFile(self.zip_path, "r")
        return self._zip_cache[pid]
    def __len__(self): return len(self.pairs)
    def _read(self, pid, role):
        data = self._get_zip().read(f"wav/{pid}_{role}.wav")
        w, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if w.ndim > 1: w = w.mean(axis=1)
        if sr != 16000:
            import torchaudio
            w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        w = w.astype(np.float32)
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(w) > ms: w = w[:ms]
        w = torch.from_numpy(w).float()
        return w
    def __getitem__(self, idx):
        p = self.pairs[idx]; pid = p["id"]
        eid = p.get("enroll_id", pid); qid = p.get("query_id", pid)
        e = self._read(eid, "enroll"); q = self._read(qid, "query")
        e_txt = p.get("enroll_txt", "").lower()
        q_txt = p.get("query_txt", "").lower()
        e_pids = _G2P_CACHE.get(e_txt, [1])
        q_pids = _G2P_CACHE.get(q_txt, [1])
        return e, q, float(p.get("label", 0)), e_txt, e_pids, q_pids, pid


def train(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Tokenizer + G2P
    from transformers import DistilBertTokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", local_files_only=True)
    from g2p_en import G2p
    g2p = G2p()

    # Load pairs (same as train_text)
    all_pairs = []
    with open(cfg.train_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            all_pairs.append({"id": r["id"], "label": int(r["label"]),
                          "enroll_txt": r.get("enroll_txt", "").lower(),
                          "query_txt": r.get("query_txt", "").lower()})
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_at_v8.json", "train/self_paired.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp):
            with open(hp) as f: all_pairs += json.load(f)
    rng.shuffle(all_pairs)
    print(f"[qbyomni] {len(all_pairs)} total pairs")

    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"  pos={len(pos_pairs)} neg={len(neg_pairs)}")

    # Semi-dedup
    pos_groups = {}
    for i, p in enumerate(pos_pairs):
        k = (p["enroll_txt"].lower(), p.get("query_txt", "").lower())
        pos_groups.setdefault(k, []).append(i)
    neg_groups = {}
    for i, p in enumerate(neg_pairs):
        k = (p["enroll_txt"].lower(), p.get("query_txt", "").lower())
        neg_groups.setdefault(k, []).append(i)
    pos_keys = list(pos_groups.keys()); neg_keys = list(neg_groups.keys())
    rng.shuffle(pos_keys); rng.shuffle(neg_keys)
    print(f"  semidedup: {len(pos_keys)} pos keys, {len(neg_keys)} neg keys")

    def get_loader(ep):
        n_pos = min(1500, len(pos_keys) * 3)
        n_neg = min(15000, len(neg_keys))
        subset = []
        pp = (ep * n_pos) % len(pos_keys) if len(pos_keys) > 0 else 0
        for i in range(n_pos):
            k = pos_keys[(pp + i) % len(pos_keys)]
            subset.append(pos_pairs[rng.choice(pos_groups[k])])
        np_start = (ep * n_neg) % len(neg_keys) if len(neg_keys) > 0 else 0
        for i in range(n_neg):
            k = neg_keys[(np_start + i) % len(neg_keys)]
            subset.append(neg_pairs[rng.choice(neg_groups[k])])
        np.random.shuffle(subset)
        ds = QbyOmniDataset(subset, cfg.train_zip, cfg, g2p=g2p, phone_vocab=PHONE_TO_ID)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=0,
                          collate_fn=lambda b: collate_qbyomni(b, tokenizer),
                          pin_memory=False, drop_last=True)

    def collate_qbyomni(batch, tok):
        ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
        es, qs, ls, txts, e_phones, e_lens, q_phones, q_lens = [], [], [], [], [], [], [], []
        for b in batch:
            e, q = b[0], b[1]
            es.append(F.pad(e, (0, ml - e.shape[-1])) if e.shape[-1] < ml else e)
            qs.append(F.pad(q, (0, ml - q.shape[-1])) if q.shape[-1] < ml else q)
            ls.append(b[2])
            txts.append(str(b[3]).lower())
            e_pids, q_pids = b[4], b[5]
            e_phones.append(e_pids); e_lens.append(len(e_pids))
            q_phones.append(q_pids); q_lens.append(len(q_pids))
        eb = torch.stack(es)
        qb = torch.stack(qs)
        tok_out = tok(txts, return_tensors="pt", padding=True)
        def pad_phones(phones, lens):
            max_pl = max(lens) if lens else 1
            t = torch.zeros(len(phones), max_pl, dtype=torch.long)
            for i, pids in enumerate(phones):
                pids_t = torch.tensor(pids, dtype=torch.long)[:max_pl]
                t[i, :len(pids_t)] = pids_t
            return t
        y = torch.tensor(ls, dtype=torch.float32)
        return eb, qb, tok_out.input_ids, tok_out.attention_mask, \
               pad_phones(e_phones, e_lens), torch.tensor(e_lens, dtype=torch.long), \
               pad_phones(q_phones, q_lens), torch.tensor(q_lens, dtype=torch.long), y

    def dev_loader(zp, cp):
        pairs = []
        with open(cp) as f:
            for r in csv.DictReader(f):
                pairs.append({"id": r["id"], "label": int(r["label"]),
                              "enroll_txt": r.get("enroll_txt", "").lower()})
        ds = QbyOmniDataset(pairs, zp, cfg, g2p=g2p, phone_vocab=PHONE_TO_ID)
        return DataLoader(ds, batch_size=32, num_workers=0,
                          collate_fn=lambda b: collate_qbyomni(b, tokenizer),
                          shuffle=False)

    dv_s = dev_loader(cfg.dev_seen_zip, cfg.dev_seen_csv)

    # Model
    model = QbyOmniModel().to(device)
    best, start_ep = -1.0, 1
    if args.resume:
        for fp in [os.path.join(out_dir, "latest.pt"), os.path.join(out_dir, "best.pt")]:
            if os.path.isfile(fp):
                ckpt = torch.load(fp, map_location=device, weights_only=False)
                # Only load trainable params; frozen encoders match init
                cur = model.state_dict()
                cur.update({k: v for k, v in ckpt["model"].items() if k in cur})
                model.load_state_dict(cur, strict=False)
                best = ckpt.get("auc_seen", -1.0)
                start_ep = ckpt.get("epoch", 0) + 1
                print(f"  resumed ep={start_ep} best={best:.4f}")
                break

    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  trainable={tr:,}")

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg.lr, weight_decay=1e-4)
    bce_crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    ctc_crit = nn.CTCLoss(blank=0, zero_infinity=True)

    for ep in range(start_ep, cfg.epochs + 1):
        loader = get_loader(ep)
        model.train(); ts = time.time(); ls, n = 0.0, 0; opt.zero_grad()

        for bi, batch_data in enumerate(loader, 1):
            eb, qb = batch_data[0].to(device), batch_data[1].to(device)
            tids, tmask = batch_data[2].to(device), batch_data[3].to(device)
            e_pids, e_plens = batch_data[4].to(device), batch_data[5].to(device)
            q_pids, q_plens = batch_data[6].to(device), batch_data[7].to(device)
            y = batch_data[8].to(device)
            out = model(qb, eb, tids, tmask, e_pids, modality_dropout=cfg.modality_dropout)

            # Utterance-level BCE (Eq. 7)
            loss = bce_crit(out["score"], y)

            # Frame-level BCE (Eq. 8) — all samples, per-frame, broadcast label
            yf = y.unsqueeze(1)
            loss = loss + 0.05 * F.binary_cross_entropy_with_logits(
                out["frame_phone"], yf.expand_as(out["frame_phone"]))
            loss = loss + 0.05 * F.binary_cross_entropy_with_logits(
                out["frame_text"], yf.expand_as(out["frame_text"]))

            # CTC loss on QUERY audio (Eq. 9): CTC(z^q, p^q)
            B_q = out["ctc_logits_q"].shape[0]
            log_probs_q = out["ctc_logits_q"].log_softmax(dim=-1).transpose(0, 1)
            loss = loss + cfg.ctc_weight * ctc_crit(
                log_probs_q, q_pids,
                torch.full((B_q,), out["ctc_logits_q"].shape[1], dtype=torch.long, device=device),
                q_plens)

            # CTC loss on ENROLL audio (Eq. 9): CTC(z^s, p^s)
            B_s = out["ctc_logits_s"].shape[0]
            log_probs_s = out["ctc_logits_s"].log_softmax(dim=-1).transpose(0, 1)
            loss = loss + cfg.ctc_weight * ctc_crit(
                log_probs_s, e_pids,
                torch.full((B_s,), out["ctc_logits_s"].shape[1], dtype=torch.long, device=device),
                e_plens)

            loss = loss / cfg.grad_accum
            loss.backward()
            ls += loss.item() * cfg.grad_accum; n += 1

            if bi % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()

            if n % cfg.log_every == 0:
                print(f"  [ep{ep}] b{n} loss={ls/n:.3f}")
            del out, loss

        # Eval
        @torch.no_grad()
        def ev(ld):
            model.eval(); ps, lbs = [], []
            for batch_data in ld:
                eb, qb = batch_data[0].to(device), batch_data[1].to(device)
                tids, tmask = batch_data[2].to(device), batch_data[3].to(device)
                e_pids = batch_data[4].to(device)
                out = model(qb, eb, tids, tmask, e_pids)
                ps.append(torch.sigmoid(out["score"]).cpu().numpy())
                lbs.append(batch_data[8].cpu().numpy())
            return roc_auc_score(np.concatenate(lbs), np.concatenate(ps))

        as_ = ev(dv_s)
        print(f"[qbyomni ep{ep}] seen={as_:.4f} loss={ls/n:.4f} ({time.time()-ts:.0f}s)")
        ckpt = {
            "model": {k: v.cpu() for k, v in model.state_dict().items()
                      if not k.startswith(("audio_enc.xlsr.", "text_enc.bert."))},
            "auc_seen": as_, "epoch": ep,
        }
        if as_ > best:
            best = as_
            torch.save(ckpt, os.path.join(out_dir, "best.pt"))
        torch.save(ckpt, os.path.join(out_dir, "latest.pt"))


if __name__ == "__main__":
    cfg = QbyOmniConfig(); cfg.__post_init__()
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="qbyomni_v1")
    p.add_argument("--epochs", type=int, default=cfg.epochs)
    p.add_argument("--bs", type=int, default=cfg.batch_size)
    p.add_argument("--lr", type=float, default=cfg.lr)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    cfg.epochs = args.epochs; cfg.batch_size = args.bs; cfg.lr = args.lr
    train(cfg, args)

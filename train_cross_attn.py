"""Cross-modal Frame Attention: Whisper frames + CharBiGRU text → CrossAttn → matching.

Instead of pooling frames into a single vector, keep per-frame features [T, 512]
and let the text embedding attend to relevant frames via cross-attention.

Architecture:
  enroll_text → CharBiGRU → et [256]
  enroll_wav  → WhisperNoPool → ea [T, 512] → Linear(512→256) → ea_proj [T, 256]
  query_wav   → WhisperNoPool → eq [T, 512] → Linear(512→256) → eq_proj [T, 256]
  
  enroll_rep = CrossAttn(et, ea_proj)  → [256]  (text knows which frames to attend)
  query_rep  = CrossAttn(et, eq_proj)  → [256]
  
  score = cos(enroll_rep, query_rep) * scale + bias → logit
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, sys, time, json, argparse, csv, io, zipfile
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
from config import PATHS


# ── Whisper Frame Encoder (no pooling) ──

class WhisperFrameEncoder(nn.Module):
    def __init__(self, model_name="base", unfreeze=0, max_sec=1.5):
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

        self.register_buffer("hann", torch.hann_window(400))
        self.register_buffer("mel_filters", wa.mel_filters("cpu", 80))
        
        # Project 512-dim whisper features to match text embedding dim
        self.proj = nn.Linear(self.dim, 256)

    def train(self, mode=True):
        super().train(mode)
        if mode:
            for i in range(self.frozen): self.whisper.encoder.blocks[i].eval()
        return self

    def forward(self, wav):
        """Returns [B, T, 256] frame-level features (NO pooling), L2-normalized per frame."""
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
            x = x + enc.positional_embedding[:x.shape[1]]
            for blk in enc.blocks[:self.frozen]: x = blk(x)
        for blk in enc.blocks[self.frozen:]: x = blk(x)
        x = enc.ln_post(x)[:, :n_valid, :]
        return F.normalize(self.proj(x), dim=-1)  # [B, T, 256]


# ── Text Encoder (CharBiGRU) ──

class CharBiGRU(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.emb = nn.Embedding(28, dim)
        self.gru = nn.GRU(dim, dim, batch_first=True, bidirectional=True, num_layers=2, dropout=0.1)
        self.proj = nn.Linear(dim*2, dim)

    def forward(self, texts):
        device = next(self.parameters()).device
        if not texts: return torch.zeros(0,256,device=device)
        mx = max(len(t) for t in texts)
        idx = torch.zeros(len(texts), mx, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            for j, c in enumerate(t[:mx]):
                idx[i,j] = max(0, min(27, ord(c)-97))
        x = self.emb(idx)
        _, h = self.gru(x)
        h = torch.cat([h[-2], h[-1]], -1)
        return F.normalize(self.proj(h), dim=-1)


# ── Cross-Attention Model ──

class CrossAttnModel(nn.Module):
    """
    et [256] → Query
    ea [T, 256] → Key/Value
    
    CrossAttn(et, ea) → text-aware audio representation [256]
    """
    def __init__(self, whisper_model="base", unfreeze=0, max_sec=1.5, load_ckpt=""):
        super().__init__()
        self.audio_enc = WhisperFrameEncoder(whisper_model, unfreeze, max_sec)
        self.text_enc = CharBiGRU(256)
        
        # Load pretrained weights from AT checkpoint
        if load_ckpt and os.path.isfile(load_ckpt):
            ckpt = torch.load(load_ckpt, map_location="cpu", weights_only=False)
            # Load audio encoder (skip proj since shape differs)
            ae_state = {k.replace("encoder.", ""): v for k, v in ckpt["model"].items()
                        if k.startswith("encoder.") and "proj" not in k}
            self.audio_enc.load_state_dict(ae_state, strict=False)
            # Load text encoder
            te_state = {k.replace("text_enc.", ""): v for k, v in ckpt["model"].items()
                        if k.startswith("text_enc.")}
            self.text_enc.load_state_dict(te_state, strict=False)
            print(f"  [load] AT weights from {os.path.basename(load_ckpt)}")
        
        # Cross-attention: text → frames
        self.cross_attn = nn.MultiheadAttention(256, num_heads=8, batch_first=True, dropout=0.1)
        
        # Learnable fusion of distribution comparison features
        self.feat_head = nn.Sequential(
            nn.Linear(6, 16), nn.ReLU(),
            nn.Linear(16, 1),
        )
        # Zero-init output layer for stable start (output ≈ 0 = sigmoid≈0.5)
        self.feat_head[-1].weight.data.zero_()
        self.feat_head[-1].bias.data.zero_()

    def forward(self, enroll_wav, enroll_text, query_wav):
        # Encode
        ea = self.audio_enc(enroll_wav)  # [B, T, 256]
        et = self.text_enc(enroll_text)   # [B, 256]
        eq = self.audio_enc(query_wav)    # [B, T, 256]
        
        # Cross-attention: text attends to audio frames, GET ATTENTION WEIGHTS
        et_1 = et.unsqueeze(1)  # [B, 1, 256]
        
        _, attn_e = self.cross_attn(et_1, ea, ea)  # attn_e: [B, 1, T_e]
        _, attn_q = self.cross_attn(et_1, eq, eq)   # attn_q: [B, 1, T_q]
        
        # ── Distribution comparison ──
        T_fixed = 128
        w_e = F.interpolate(attn_e, size=T_fixed, mode='linear', align_corners=False).squeeze(1)  # [B, T]
        w_q = F.interpolate(attn_q, size=T_fixed, mode='linear', align_corners=False).squeeze(1)  # [B, T]
        
        # Per-frame features
        idx = torch.linspace(0, 1, T_fixed, device=w_e.device).unsqueeze(0)  # [1, T]
        mu_e = (w_e * idx).sum(dim=-1, keepdim=True)   # [B, 1]
        mu_q = (w_q * idx).sum(dim=-1, keepdim=True)   # [B, 1]
        var_e = ((w_e * (idx - mu_e)**2).sum(dim=-1, keepdim=True)).clamp(min=1e-6)  # [B, 1]
        var_q = ((w_q * (idx - mu_q)**2).sum(dim=-1, keepdim=True)).clamp(min=1e-6)  # [B, 1]
        
        # Per-sample KL divergence
        log_e = w_e.log_softmax(dim=-1)
        log_q = w_q.log_softmax(dim=-1)
        soft_e = w_e.softmax(dim=-1)
        soft_q = w_q.softmax(dim=-1)
        kl_e2q = (soft_e * (log_e - log_q)).sum(dim=-1, keepdim=True)  # [B, 1]
        kl_q2e = (soft_q * (log_q - log_e)).sum(dim=-1, keepdim=True)  # [B, 1]
        
        # Concatenate all features → MLP learns to weight them
        feat = torch.cat([kl_e2q, kl_q2e, mu_e - mu_q, var_e - var_q,
                         (mu_e - mu_q).abs(), (var_e - var_q).abs()], dim=-1)  # [B, 6]
        
        return self.feat_head(feat).squeeze(-1)  # [B]


# ═══════════ Data ── reuse from root train_dual.py ═══════════

import importlib.util as _iu
_spec = _iu.spec_from_file_location("rtd", os.path.join(os.path.dirname(__file__), "train_dual.py"))
_rtd = _iu.module_from_spec(_spec); _spec.loader.exec_module(_rtd)
PairDataset = _rtd.PairDataset
collate_text = _rtd.collate_text
load_pairs = _rtd.load_pairs
Config = _rtd.Config


# ═══════════ Training ═══════════

def train_crossattn(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    for hn in ["baseline/hard_neg_at_v8.json", "train/self_paired.json"]:
        hp = os.path.join(PATHS.root, hn)
        if os.path.isfile(hp): all_pairs += json.load(open(hp))
    rng.shuffle(all_pairs)
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    print(f"[data] pos={len(pos_pairs)} neg={len(neg_pairs)}")

    # Semi-dedup
    pos_groups, neg_groups = {}, {}
    for i,p in enumerate(pos_pairs):
        k=(p["enroll_txt"].lower(),p["query_txt"].lower()); pos_groups.setdefault(k,[]).append(i)
    for i,p in enumerate(neg_pairs):
        k=(p["enroll_txt"].lower(),p["query_txt"].lower()); neg_groups.setdefault(k,[]).append(i)
    pos_keys=list(pos_groups.keys()); neg_keys=list(neg_groups.keys())
    rng.shuffle(pos_keys); rng.shuffle(neg_keys)

    def get_loader(ep):
        n_pos = min(30000, len(pos_keys)*3)
        n_neg = min(150000, len(neg_keys))
        subset = []
        pp = (ep*n_pos)%len(pos_keys) if pos_keys else 0
        for i in range(n_pos):
            k=pos_keys[(pp+i)%len(pos_keys)]; idx=rng.choice(pos_groups[k]); subset.append(pos_pairs[idx])
        np_start = (ep*n_neg)%len(neg_keys) if neg_keys else 0
        for i in range(n_neg):
            k=neg_keys[(np_start+i)%len(neg_keys)]; idx=rng.choice(neg_groups[k]); subset.append(neg_pairs[idx])
        np.random.shuffle(subset)
        from torch.utils.data import DataLoader
        return DataLoader(PairDataset(subset, cfg.train_zip, cfg, "text"),
                         batch_size=cfg.batch_size, shuffle=True, num_workers=8,
                         collate_fn=collate_text, pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        from torch.utils.data import DataLoader
        return DataLoader(PairDataset(load_pairs(c), z, cfg, "text"),
                         batch_size=cfg.batch_size*2, num_workers=8,
                         collate_fn=collate_text, shuffle=False)

    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    model = CrossAttnModel(unfreeze=args.unfreeze, load_ckpt=args.load_ckpt).to(device)
    
    if args.freeze_whisper:
        for p in model.audio_enc.parameters(): p.requires_grad = False
        print(f"  [freeze] whisper frozen. Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    wav_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" in n]
    oth_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n]
    # Discriminator head (feat_head + cross_attn) → higher LR
    head_p = [p for n,p in model.named_parameters() if p.requires_grad and "feat_head" in n]
    rest_p = [p for n,p in model.named_parameters() if p.requires_grad and "whisper" not in n and "feat_head" not in n]
    opt = torch.optim.AdamW([
        {"params":rest_p, "lr":3e-4},
        {"params":head_p, "lr":1e-2},
    ], weight_decay=1e-4)
    warmup_steps = 500
    def lr_fn(s): return min(1.0, s/max(1, warmup_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    
    from sklearn.metrics import roc_auc_score
    best_macro, gs = -1.0, 0

    for ep in range(1, args.epochs+1):
        model.train(); ts = time.time(); ls, n = 0.0, 0
        for e,q,y,txts,_ in get_loader(ep):
            e,q,y=e.to(device),q.to(device),y.to(device)
            logit = model(e, txts, q)
            loss = crit(logit, y.float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step(); gs += 1
            ls += loss.item(); n += 1
            if n%50==0: print(f"  [ep{ep}] b{n} loss={ls/n:.4f}", flush=True)

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps,ls=[],[]
            for e,q,y,txts,_ in ld:
                e,q=e.to(device),q.to(device)
                logit = model(e, txts, q)
                ps.append(torch.sigmoid(logit).cpu().numpy()); ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_=ev(dv_s); au_=ev(dv_u); macro=(as_+au_)/2
        print(f"[epoch {ep}] seen={as_:.4f} unseen={au_:.4f} macro={macro:.4f} loss={ls/n:.4f} ({time.time()-ts:.0f}s)")
        if macro>best_macro:
            best_macro=macro
            torch.save({"model":model.state_dict(),"auc_seen":as_,"auc_unseen":au_,"epoch":ep}, os.path.join(out_dir,"best.pt"))
        torch.save({"model":model.state_dict(),"auc_seen":as_,"auc_unseen":au_,"epoch":ep}, os.path.join(out_dir,"latest.pt"))
    print(f"\n[done] best macro={best_macro:.4f}")


# ═══════════ Main ═══════════

def main():
    cfg = Config(); cfg.__post_init__()
    cfg.batch_size = 256
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="cross_attn")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--unfreeze", type=int, default=0)
    p.add_argument("--load-ckpt", default="output/dual_at_v8_text/best.pt")
    p.add_argument("--freeze-whisper", action="store_true")
    args = p.parse_args()
    train_crossattn(cfg, args)

if __name__ == "__main__":
    main()

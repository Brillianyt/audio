"""Conv1D on frames: replace ASP pooling with CNN + global max pooling.

Instead of ASP (which averages frames) or CrossAttn (which filters through text),
use Conv1D over frame sequences to capture temporal patterns, then global max pool
to keep the most discriminative features.

Architecture:
  enroll_wav → WhisperFrameEncoder(frozen) → ea [T, 512]
  query_wav  → WhisperFrameEncoder(frozen) → eq [T, 512]
  
  ea → Conv1D(512→256, k=3) → ReLU → global_max_pool → [256] → L2Norm
  eq → Conv1D(512→256, k=3) → ReLU → global_max_pool → [256] → L2Norm
  
  score = cos(ea_conv, eq_conv) * scale + bias → logit
"""
import torch, torch.nn as nn, torch.nn.functional as F, numpy as np, os, sys, time, json, argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
from config import PATHS


# ── Whisper Frame Encoder (from cross_attn) ──

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
            x = x + enc.positional_embedding[:x.shape[1]]
            for blk in enc.blocks[:self.frozen]: x = blk(x)
        for blk in enc.blocks[self.frozen:]: x = blk(x)
        return enc.ln_post(x)[:, :n_valid, :]  # [B, T, 512]


class ConvPoolModel(nn.Module):
    """Conv1D on frames + global max pool → cosine matching."""
    def __init__(self, load_ckpt="", mode="audio"):
        """
        mode="audio": audio-audio matching (Conv1D + max pool → cos)
        mode="text":  text-audio matching (text → et, audio → Conv1D → max pool → cos)
        """
        super().__init__()
        self.mode = mode
        self.audio_enc = WhisperFrameEncoder(unfreeze=0)
        
        # Conv1D over frame dimension
        self.conv = nn.Sequential(
            nn.Conv1d(512, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.BatchNorm1d(256), nn.ReLU(),
        )
        
        if mode == "text":
            # CharBiGRU text encoder
            from train_cross_attn import CharBiGRU
            self.text_enc = CharBiGRU(256)
        
        # Load pretrained whisper weights
        if load_ckpt and os.path.isfile(load_ckpt):
            ckpt = torch.load(load_ckpt, map_location="cpu", weights_only=False)
            ae_state = {k.replace("encoder.", ""): v for k, v in ckpt["model"].items()
                        if k.startswith("encoder.") and "proj" not in k}
            self.audio_enc.load_state_dict(ae_state, strict=False)
            if mode == "text":
                te_state = {k.replace("text_enc.", ""): v for k, v in ckpt["model"].items()
                            if k.startswith("text_enc.")}
                self.text_enc.load_state_dict(te_state, strict=False)
            print(f"  [load] whisper{' + text' if mode=='text' else ''} from {os.path.basename(load_ckpt)}")
        
        self.scale = nn.Parameter(torch.tensor(4.0))
        self.bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, wav1, wav2, texts=None):
        e1 = self.audio_enc(wav1 if self.mode == 'audio' else wav2)  # [B, T, 512]
        e2 = self.audio_enc(wav2) if self.mode == 'audio' else None
        
        if self.mode == 'text' and texts is not None:
            et = F.normalize(self.text_enc(texts), dim=-1)  # [B, 256]
            eq = self.conv(e1.permute(0, 2, 1))
            eq = F.normalize(eq.max(dim=-1).values, dim=-1)
            sim = (et * eq).sum(-1)
        else:
            e1 = self.conv(e1.permute(0, 2, 1))
            e2 = self.conv(e2.permute(0, 2, 1))
            e1 = F.normalize(e1.max(dim=-1).values, dim=-1)
            e2 = F.normalize(e2.max(dim=-1).values, dim=-1)
            sim = (e1 * e2).sum(-1)
        return self.scale * sim + self.bias


# ═══════════ Data ═══════════

import importlib.util as _iu
_spec = _iu.spec_from_file_location("rtd", os.path.join(os.path.dirname(__file__), "train_dual.py"))
_rtd = _iu.module_from_spec(_spec); _spec.loader.exec_module(_rtd)

from output.train_dual import PairDataset, collate_audio, collate_text, load_pairs, Config


# ═══════════ Training ═══════════

def train_convpool(cfg, args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    out_dir = os.path.join(PATHS.root, "output", args.name)
    os.makedirs(out_dir, exist_ok=True)

    # Data
    data_mode = "text" if args.mode == "text" else "audio"
    all_pairs = load_pairs(cfg.train_csv)
    rng = np.random.default_rng(cfg.seed)
    rng.shuffle(all_pairs)
    pos_pairs = [p for p in all_pairs if p["label"] == 1]
    neg_pairs = [p for p in all_pairs if p["label"] == 0]
    collate_fn = collate_text if data_mode == "text" else collate_audio
    print(f"[data] mode={args.mode} pos={len(pos_pairs)} neg={len(neg_pairs)}")

    def get_loader(ep):
        n_pos = min(50000, len(pos_pairs))
        n_neg = min(150000, len(neg_pairs))
        subset = rng.choice(pos_pairs, n_pos, replace=False).tolist()
        subset += rng.choice(neg_pairs, n_neg, replace=False).tolist()
        rng.shuffle(subset)
        from torch.utils.data import DataLoader
        return DataLoader(PairDataset(subset, cfg.train_zip, cfg, data_mode),
                         batch_size=cfg.batch_size, shuffle=True, num_workers=8,
                         collate_fn=collate_fn, pin_memory=True, drop_last=True)

    def dev_ld(z, c):
        from torch.utils.data import DataLoader
        return DataLoader(PairDataset(load_pairs(c), z, cfg, data_mode),
                         batch_size=cfg.batch_size*2, num_workers=8,
                         collate_fn=collate_fn, shuffle=False)

    dv_s = dev_ld(cfg.dev_seen_zip, cfg.dev_seen_csv)
    dv_u = dev_ld(cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    model = ConvPoolModel(load_ckpt=args.load_ckpt, mode=args.mode).to(device)
    if args.freeze_whisper:
        for p in model.audio_enc.parameters(): p.requires_grad = False
        if args.mode == "text":
            for p in model.text_enc.parameters(): p.requires_grad = False
        print(f"  [freeze] whisper{' + text' if args.mode=='text' else ''} frozen. Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(cfg.pos_weight, device=device))
    
    from sklearn.metrics import roc_auc_score
    best_macro = -1.0

    for ep in range(1, args.epochs+1):
        model.train(); ts = time.time(); ls, n = 0.0, 0
        for batch in get_loader(ep):
            if args.mode == "text":
                e,q,y,txts,_ = batch
                e,q,y = e.to(device), q.to(device), y.to(device)
                logit = model(None, q, texts=list(txts))
            else:
                e,q,y,*_ = batch
                e,q,y = e.to(device), q.to(device), y.to(device)
                logit = model(e, q)
            loss = crit(logit, y.float())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ls += loss.item(); n += 1
            if n%50==0: print(f"  [ep{ep}] b{n} loss={ls/n:.4f}", flush=True)

        @torch.no_grad()
        def ev(ld):
            model.eval(); ps,ls=[],[]
            for batch in ld:
                if args.mode == "text":
                    e,q,y,txts,_ = batch
                    e,q = e.to(device), q.to(device)
                    logit = model(None, q, texts=list(txts))
                else:
                    e,q,y,*_ = batch
                    e,q = e.to(device), q.to(device)
                    logit = model(e,q)
                ps.append(torch.sigmoid(logit).cpu().numpy()); ls.append(y.numpy())
            return roc_auc_score(np.concatenate(ls), np.concatenate(ps))

        as_=ev(dv_s); au_=ev(dv_u); macro=(as_+au_)/2
        print(f"[epoch {ep}] seen={as_:.4f} unseen={au_:.4f} macro={macro:.4f} loss={ls/n:.4f} ({time.time()-ts:.0f}s)")
        if macro>best_macro:
            best_macro=macro
            torch.save({"model":model.state_dict(),"auc_seen":as_,"auc_unseen":au_,"epoch":ep}, os.path.join(out_dir,"best.pt"))
        torch.save({"model":model.state_dict(),"auc_seen":as_,"auc_unseen":au_,"epoch":ep}, os.path.join(out_dir,"latest.pt"))
    print(f"\n[done] best macro={best_macro:.4f}")


def main():
    cfg = Config(); cfg.__post_init__()
    cfg.batch_size = 256
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="conv_pool")
    p.add_argument("--mode", default="audio", choices=["audio", "text"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--load-ckpt", default="output/dual_at_v8_text/best.pt")
    p.add_argument("--freeze-whisper", action="store_true")
    args = p.parse_args()
    train_convpool(cfg, args)

if __name__ == "__main__":
    main()

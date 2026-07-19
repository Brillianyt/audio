"""Ensemble inference: Audio-Audio + Audio-Text → weighted posterior.

Loads both trained models, runs inference on eval_seen and eval_unseen,
then produces a weighted ensemble posterior. The weights are calibrated
on the dev set for optimal (seen + unseen) AUC.

Usage:
    python baseline/ensemble_infer.py \
        --aa-ckpt output/dual_aa_v1_audio/best.pt \
        --at-ckpt output/dual_at_v1_text/best.pt \
        --aa-weight 0.4 --at-weight 0.6 \
        --out submission.csv
"""
import argparse, csv, io, json, os, sys, time, zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from baseline.train_dual import (
    AudioAudioModel, AudioTextModel, WhisperEncoder, PhonemeEncoder,
    Config, PATHS, mix_noise_batch, _load_noise_bank
)


def load_pairs_no_label(csv_path):
    """Load eval CSV (id, enroll_txt only)."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "enroll_txt": r.get("enroll_txt", "")})
    return rows


def load_pairs_with_label(csv_path):
    """Load dev CSV with labels."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt", "")})
    return rows


def read_wav(zip_path, pid, role):
    """Read a single wav file from zip, return float32 tensor."""
    with zipfile.ZipFile(zip_path, "r") as z:
        data = z.read(f"wav/{pid}_{role}.wav")
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    wav = wav.astype(np.float32)
    max_s = 3 * 16000
    if len(wav) > max_s:
        wav = wav[:max_s]
    return torch.from_numpy(wav).float()


def pad_batch(wavs):
    """Pad list of (T,) tensors to (B, max_T)."""
    max_len = max(w.shape[0] for w in wavs)
    batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch[i, :w.shape[0]] = w
    return batch


@torch.no_grad()
def predict_aa(model, zip_path, csv_path, prefix, device, batch_size=128):
    """Audio-Audio inference."""
    model.eval()
    pairs = load_pairs_no_label(csv_path)
    rows = []
    total = len(pairs)
    for i in range(0, total, batch_size):
        batch = pairs[i:i+batch_size]
        e_wavs, q_wavs = [], []
        for bp in batch:
            e_wavs.append(read_wav(zip_path, bp["id"], "enroll"))
            q_wavs.append(read_wav(zip_path, bp["id"], "query"))
        e = pad_batch(e_wavs).to(device)
        q = pad_batch(q_wavs).to(device)
        _, logit, _, _ = model(e, q)
        prob = torch.sigmoid(logit).cpu().numpy()
        for j, bp in enumerate(batch):
            rows.append((f"{prefix}_{bp['id']}", float(prob[j])))
        if (i + 1) % 2000 == 0 or (i + batch_size) >= total:
            print(f"  [aa {prefix}] {min(i+batch_size, total)}/{total}")
    return rows


@torch.no_grad()
def predict_at(model, zip_path, csv_path, prefix, device, batch_size=128):
    """Audio-Text inference: enroll text + enroll audio vs query audio."""
    model.eval()
    pairs = load_pairs_no_label(csv_path)
    rows = []
    total = len(pairs)
    for i in range(0, total, batch_size):
        batch = pairs[i:i+batch_size]
        e_wavs, q_wavs, txts = [], [], []
        for bp in batch:
            e_wavs.append(read_wav(zip_path, bp["id"], "enroll"))
            q_wavs.append(read_wav(zip_path, bp["id"], "query"))
            txts.append(bp.get("enroll_txt", ""))
        e = pad_batch(e_wavs).to(device)
        q = pad_batch(q_wavs).to(device)
        # Audio-Text matching: enroll text vs query audio
        cos_ae, ea, et = model(e, txts)
        cos_tq = (et * model.encoder(q)).sum(-1)
        # Combine: posterior from both enroll-text and text-query similarity
        prob = torch.sigmoid(cos_tq).cpu().numpy()
        for j, bp in enumerate(batch):
            rows.append((f"{prefix}_{bp['id']}", float(prob[j])))
        if (i + 1) % 2000 == 0 or (i + batch_size) >= total:
            print(f"  [at {prefix}] {min(i+batch_size, total)}/{total}")
    return rows


def calibrate_weights(device, batch_size=256):
    """Run AA and AT on dev set to find optimal ensemble weights."""
    cfg = Config(); cfg.__post_init__()
    print("=" * 50)
    print("Calibrating ensemble weights on dev set...")
    print("=" * 50)

    # Check for trained models
    aa_ckpt_candidates = [
        "output/dual_aa_v1_audio/best.pt",
        "output/dual_aa_frozen_audio/best.pt",
        "output/dual_r2_init_wavlm_audio/best.pt",
        "output/dual_r2_init_audio/best.pt",
    ]
    at_ckpt_candidates = [
        "output/dual_at_v1_text/best.pt",
        "output/dual_r2_init_text/best.pt",
        "output/text_from_proto_r5/best.pt",
    ]

    aa_ckpt = None
    for c in aa_ckpt_candidates:
        p = os.path.join(PATHS.root, c)
        if os.path.isfile(p):
            aa_ckpt = p
            break

    at_ckpt = None
    for c in at_ckpt_candidates:
        p = os.path.join(PATHS.root, c)
        if os.path.isfile(p):
            at_ckpt = p
            break

    if not aa_ckpt or not at_ckpt:
        print("  [calibrate] checkpoint(s) not found, using default weights")
        return {"seen": 0.35, "unseen": 0.55}

    # Load models
    from collections import OrderedDict

    # AA model
    aa_state = torch.load(aa_ckpt, map_location="cpu", weights_only=False)
    aa_model = AudioAudioModel("", aa_state.get("embed_dim", 256),
                                unfreeze=0, encoder="whisper").to(device)
    # Handle potential key mismatch
    try:
        aa_model.load_state_dict(aa_state["model"], strict=False)
    except Exception as e:
        print(f"  [aa] load warning: {e}")
    aa_model.eval()
    print(f"  [aa] loaded {aa_ckpt} (unseen={aa_state.get('auc_unseen','?'):.4f})")

    # AT model
    at_state = torch.load(at_ckpt, map_location="cpu", weights_only=False)
    at_model = AudioTextModel("", at_state.get("embed_dim", 256),
                               text_encoder="phoneme").to(device)
    try:
        at_model.load_state_dict(at_state["model"], strict=False)
    except Exception as e:
        print(f"  [at] load warning: {e}")
    at_model.eval()
    print(f"  [at] loaded {at_ckpt} (unseen={at_state.get('auc_unseen','?'):.4f})")

    from sklearn.metrics import roc_auc_score

    def eval_subset(zip_p, csv_p, prefix):
        pairs = load_pairs_with_label(csv_p)
        total = len(pairs)
        aa_probs, at_probs, labels = [], [], []
        for i in range(0, total, batch_size):
            batch = pairs[i:i+batch_size]
            e_w, q_w, txts = [], [], []
            for bp in batch:
                e_w.append(read_wav(zip_p, bp["id"], "enroll"))
                q_w.append(read_wav(zip_p, bp["id"], "query"))
                txts.append(bp.get("enroll_txt", ""))
            e = pad_batch(e_w).to(device)
            q = pad_batch(q_w).to(device)

            # AA
            _, logit, _, _ = aa_model(e, q)
            aa_probs.append(torch.sigmoid(logit).cpu().numpy())

            # AT
            cos_ae, ea, et = at_model(e, txts)
            cos_tq = (et * at_model.encoder(q)).sum(-1)
            at_probs.append(torch.sigmoid(cos_tq).cpu().numpy())

            labels.append(np.array([bp["label"] for bp in batch]))

            if (i + 1) % 2000 == 0:
                print(f"  [calibrate {prefix}] {min(i+batch_size, total)}/{total}")

        aa_p = np.concatenate(aa_probs)
        at_p = np.concatenate(at_probs)
        lb = np.concatenate(labels)

        # Score each model alone
        aa_auc = roc_auc_score(lb, aa_p)
        at_auc = roc_auc_score(lb, at_p)
        print(f"  {prefix}: AA AUC={aa_auc:.4f}, AT AUC={at_auc:.4f}")

        # Grid search for best ensemble weight (AA weight from 0 to 1)
        best_w = 0.5
        best_auc = 0
        for w in np.linspace(0, 1, 21):
            ens = w * aa_p + (1 - w) * at_p
            auc = roc_auc_score(lb, ens)
            if auc > best_auc:
                best_auc = auc
                best_w = w

        ens_auc = roc_auc_score(lb, best_w * aa_p + (1 - best_w) * at_p)
        print(f"  {prefix}: best AA weight={best_w:.2f}, ensemble AUC={ens_auc:.4f}")
        return best_w, best_auc

    w_seen, auc_seen = eval_subset(cfg.dev_seen_zip, cfg.dev_seen_csv, "seen")
    w_unseen, auc_unseen = eval_subset(cfg.dev_unseen_zip, cfg.dev_unseen_csv, "unseen")

    weights = {"seen": w_seen, "unseen": w_unseen}
    print(f"\n  Optimal weights: seen={w_seen:.2f}, unseen={w_unseen:.2f}")
    print(f"  Ensemble AUC: seen={auc_seen:.4f}, unseen={auc_unseen:.4f}")
    return weights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--aa-ckpt", default="",
                   help="Audio-Audio checkpoint")
    p.add_argument("--at-ckpt", default="",
                   help="Audio-Text checkpoint")
    p.add_argument("--aa-weight", type=float, default=None,
                   help="AA weight in ensemble (0-1). If None, grid-search on dev")
    p.add_argument("--at-weight", type=float, default=None,
                   help="AT weight. If None, 1 - aa_weight")
    p.add_argument("--out", default=os.path.join(PATHS.root, "submission.csv"))
    p.add_argument("--bs", type=int, default=128)
    p.add_argument("--calibrate", action="store_true",
                   help="Run calibration on dev set before inferring eval")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Config(); cfg.__post_init__()

    # ── Calibration (or manual weights) ──
    if args.calibrate and args.aa_weight is None:
        weights = calibrate_weights(device, args.bs)
        w_seen = weights["seen"]
        w_unseen = weights["unseen"]
    else:
        w_seen = args.aa_weight if args.aa_weight is not None else 0.35
        w_unseen = args.aa_weight if args.aa_weight is not None else 0.55

    if args.at_weight is not None:
        w_seen = 1.0 - args.at_weight if args.at_weight <= 1.0 else 1.0 - args.at_weight
        w_unseen = 1.0 - args.at_weight

    print(f"Ensemble weights: seen AA={w_seen:.2f} AT={1-w_seen:.2f}, "
          f"unseen AA={w_unseen:.2f} AT={1-w_unseen:.2f}")

    # ── Resolve checkpoints ──
    aa_ckpt = args.aa_ckpt or os.path.join(PATHS.root, "output/dual_aa_v1_audio/best.pt")
    at_ckpt = args.at_ckpt or os.path.join(PATHS.root, "output/dual_at_v1_text/best.pt")

    for label, path in [("AA", aa_ckpt), ("AT", at_ckpt)]:
        if not os.path.isfile(path):
            print(f"[error] {label} checkpoint not found: {path}")
            return
        print(f"[{label}] loading {path}")

    # ── Load AA model ──
    aa_state = torch.load(aa_ckpt, map_location=device, weights_only=False)
    aa_model = AudioAudioModel("", aa_state.get("embed_dim", 256),
                                unfreeze=0, encoder="whisper").to(device)
    aa_model.load_state_dict(aa_state["model"], strict=False)
    aa_model.eval()
    print(f"  AA: seen={aa_state.get('auc_seen','?'):.4f} unseen={aa_state.get('auc_unseen','?'):.4f}")

    # ── Load AT model ──
    at_state = torch.load(at_ckpt, map_location=device, weights_only=False)
    # Detect text_encoder type from checkpoint keys
    has_phoneme = any("transformer" in k for k in at_state["model"].keys())
    text_enc = "phoneme" if has_phoneme else "char"
    at_model = AudioTextModel("", at_state.get("embed_dim", 256),
                               text_encoder=text_enc).to(device)
    at_model.load_state_dict(at_state["model"], strict=False)
    at_model.eval()
    print(f"  AT: seen={at_state.get('auc_seen','?'):.4f} unseen={at_state.get('auc_unseen','?'):.4f}")

    # ── Run inference on eval sets ──
    seen_rows_aa = predict_aa(aa_model, cfg.eval_seen_zip, cfg.eval_seen_csv,
                               "seen", device, args.bs)
    unseen_rows_aa = predict_aa(aa_model, cfg.eval_unseen_zip, cfg.eval_unseen_csv,
                                 "unseen", device, args.bs)

    seen_rows_at = predict_at(at_model, cfg.eval_seen_zip, cfg.eval_seen_csv,
                               "seen", device, args.bs)
    unseen_rows_at = predict_at(at_model, cfg.eval_unseen_zip, cfg.eval_unseen_csv,
                                 "unseen", device, args.bs)

    # ── Ensemble ──
    seen_map = {}
    for prefix, aa_list, at_list in [
        ("seen", seen_rows_aa, seen_rows_at),
        ("unseen", unseen_rows_aa, unseen_rows_at),
    ]:
        aa_dict = {r[0]: r[1] for r in aa_list}
        at_dict = {r[0]: r[1] for r in at_list}
        w = w_seen if prefix == "seen" else w_unseen
        for rid in aa_dict:
            p_aa = aa_dict[rid]
            p_at = at_dict.get(rid, 0.5)
            seen_map[rid] = w * p_aa + (1 - w) * p_at

    # ── Write submission ──
    rows = sorted(seen_map.items(), key=lambda x: x[0])
    print(f"\nTotal: {len(rows)} rows")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(rows)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()

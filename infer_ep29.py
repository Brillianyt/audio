"""Simple AT v8 inference for submission.
Loads the ep29 checkpoint and runs eval set inference."""
import argparse, csv, io, os, sys, zipfile
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATHS

import importlib.util
_td_spec = importlib.util.spec_from_file_location(
    "train_dual_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dual.py"))
_td = importlib.util.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td)
AudioTextModel = _td.AudioTextModel
DualConfig = _td.Config

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

cfg = DualConfig()
cfg.__post_init__()

# ── Load model ──
ckpt_path = "output/backup_final/at_v8_ep29_seen08462_unseen07889.pt"
model = AudioTextModel("", 256, unfreeze=0).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"Loaded: seen={ckpt['auc_seen']:.4f} unseen={ckpt['auc_unseen']:.4f}")


def read_wav(zf, pid, role):
    try:
        data = zf.read(f"wav/{pid}_{role}.wav")
    except KeyError:
        data = zf.read(f"wav/{pid}.wav")
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    wav = torch.from_numpy(wav).float()
    pad = int(0.5 * 16000)
    wav = F.pad(wav, (pad, pad))
    return wav


def pad_batch(wavs):
    max_len = max(w.shape[0] for w in wavs)
    batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch[i, :w.shape[0]] = w
    return batch


def process(prefix, csv_path, zip_path):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({"id": r["id"], "enroll_txt": r.get("enroll_txt", "").lower()})

    zf = zipfile.ZipFile(zip_path, "r")
    results = []
    total = len(pairs)
    print(f"[{prefix}] {total} pairs")

    for i in range(0, total, 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for bp in batch:
            e_w.append(read_wav(zf, bp["id"], "enroll"))
            q_w.append(read_wav(zf, bp["id"], "query"))
            txts.append(bp["enroll_txt"])

        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        e_b = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        q_b = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        with torch.no_grad():
            _, score, _, _ = model(e_b, txts, q_b)
            p = torch.sigmoid(score).detach().cpu().numpy()

        for j, bp in enumerate(batch):
            results.append((f"{prefix}_{bp['id'].replace('pair_', '')}", float(p[j])))

        if (i + 128) % 10000 == 0 or (i + 128) >= total:
            print(f"  {min(i+128, total)}/{total}")

    zf.close()
    return results


seen = process("seen_pair", cfg.eval_seen_csv, cfg.eval_seen_zip)
unseen = process("unseen_pair", cfg.eval_unseen_csv, cfg.eval_unseen_zip)

all_r = sorted(seen + unseen, key=lambda x: x[0])
out_path = os.path.join(PATHS.root, "submission_ep29.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "posterior"])
    w.writerows(all_r)

ps = np.array([p for _, p in all_r])
print(f"\nSaved: {out_path}")
print(f"Total: {len(all_r)} rows")
print(f"Mean={ps.mean():.4f} Std={ps.std():.4f} >0.5={(ps>0.5).sum()}/{len(ps)}")

"""Routing inference: AT v8 + AA v3 with mel-based audio quality routing.

Process:
  1. 对每个 query audio 计算 damage_score (mel 特征)
  2. 根据 damage 程度路由:
     - clean (damage < 0.3): AT alone
     - moderate (0.3~0.6): AT + AA confidence-weighted ensemble
     - severe (damage > 0.6):
       * seen 词 → AA alone
       * unseen 词 → AT 保守降级

Usage:
    python infer_routed.py
"""
import argparse, csv, io, os, sys, time, zipfile, json
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "baseline"))
sys.path.insert(0, _ROOT)

from config import PATHS
from audio_quality import compute_damage_score_batch, classify_damage

# ── import model classes ──
# AT v8 用的是 root/train_dual.py (CharBiGRUEncoder 架构)
# baseline/train_dual.py 有同名类但架构不同，用 importlib 显式区分
import importlib.util
_td_spec = importlib.util.spec_from_file_location(
    "train_dual_root", os.path.join(_ROOT, "train_dual.py"))
_td = importlib.util.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td)
AudioTextModel = _td.AudioTextModel
DualConfig = _td.Config

# AA 用的是 baseline/train_whisper_v3.py 的 MultiTaskWhisperKWSV3
from baseline.train_whisper_v3 import MultiTaskWhisperKWSV3


def load_train_vocab():
    """Load training vocabulary for seen/unseen detection."""
    csv_path = os.path.join(PATHS.root, "train", "train_label.csv")
    words = set()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            words.add(r.get("enroll_txt", "").lower().strip())
    return words


def load_eval_pairs(csv_path):
    """Load eval CSV (id, enroll_txt)."""
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "enroll_txt": r.get("enroll_txt", "").lower()})
    return rows


def read_wav(zf, pid, role, max_sec=3.0):
    """Read a single wav from ZipFile, return (T,) float32 tensor."""
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
    wav = wav.astype(np.float32)
    ms = int(max_sec * 16000)
    if len(wav) > ms:
        wav = wav[:ms]
    return torch.from_numpy(wav).float()


def pad_batch(wavs):
    """Pad list of (T,) tensors to (B, max_T)."""
    max_len = max(w.shape[0] for w in wavs)
    batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch[i, :w.shape[0]] = w
    return batch


@torch.no_grad()
def run_inference(args):
    device = args.device
    seen_vocab = load_train_vocab()
    print(f"[vocab] {len(seen_vocab)} train words")

    # ── Load AT v8 ──
    cfg = DualConfig()
    cfg.__post_init__()
    at_model = AudioTextModel("", 256, unfreeze=0).to(device)
    at_ckpt = torch.load(args.at_ckpt, map_location=device, weights_only=False)
    at_model.load_state_dict(at_ckpt["model"], strict=False)
    at_model.eval()
    print(f"[AT] loaded {args.at_ckpt} (seen={at_ckpt.get('auc_seen','?'):.4f} unseen={at_ckpt.get('auc_unseen','?'):.4f})")

    # ── Load AA v3 ──
    aa_model = MultiTaskWhisperKWSV3("base", 256, unfreeze_layers=0).to(device)
    aa_ckpt = torch.load(args.aa_ckpt, map_location=device, weights_only=False)
    aa_model.load_state_dict(aa_ckpt["model"], strict=False)
    aa_model.eval()
    aa_auc = aa_ckpt.get('auc', '?')
    print(f"[AA] loaded {args.aa_ckpt} (auc={aa_auc})")

    # ── Statistics ──
    stats = {"clean": 0, "moderate": 0, "severe_seen": 0, "severe_unseen": 0,
             "total": 0, "by_score": []}

    def process_set(prefix, csv_path, zip_path):
        pairs = load_eval_pairs(csv_path)
        results = []
        total = len(pairs)
        print(f"\n[{prefix}] {total} pairs")
        zf = zipfile.ZipFile(zip_path, "r")
        last_print = 0

        for i in range(0, total, args.batch_size):
            batch = pairs[i:i + args.batch_size]
            e_wavs, q_wavs, txts = [], [], []
            for bp in batch:
                e_wavs.append(read_wav(zf, bp["id"], "enroll"))
                q_wavs.append(read_wav(zf, bp["id"], "query"))
                txts.append(bp["enroll_txt"])

            e_batch = pad_batch(e_wavs).to(device)
            q_batch = pad_batch(q_wavs).to(device)

            # ── Damage assessment (batched on GPU) ──
            damage_scores = compute_damage_score_batch(q_wavs)

            # ── AT forward (entire batch) ──
            cos_ae, score_at, ea_at, et_at = at_model(e_batch, txts, q_batch)
            p_at = torch.sigmoid(score_at)  # (B,)

            # ── AA forward (entire batch) ──
            logit_aa, _, _ = aa_model(e_batch, q_batch)
            p_aa = torch.sigmoid(logit_aa)  # (B,)

            # ── Per-sample routing ──
            for j in range(len(batch)):
                pid = batch[j]["id"]
                txt = txts[j]
                ds = damage_scores[j]
                p_at_j = p_at[j].item()
                p_aa_j = p_aa[j].item()

                level = classify_damage(ds)

                if level == "clean":
                    posterior = p_at_j
                    stats["clean"] += 1

                elif level == "moderate":
                    # confidence-weighted ensemble
                    conf_at = abs(p_at_j - 0.5) * 2
                    conf_aa = abs(p_aa_j - 0.5) * 2
                    if conf_at + conf_aa > 0:
                        posterior = (conf_at * p_at_j + conf_aa * p_aa_j) / (conf_at + conf_aa)
                    else:
                        posterior = (p_at_j + p_aa_j) / 2
                    stats["moderate"] += 1

                else:  # severe
                    if txt in seen_vocab:
                        posterior = p_aa_j
                        stats["severe_seen"] += 1
                    else:
                        # unseen + severe → AT 降级, 降低置信度
                        posterior = p_at_j * 0.8
                        stats["severe_unseen"] += 1

                stats["by_score"].append(ds)
                results.append((f"{prefix}_{pid.replace('pair_', '')}", posterior))

            if (i + args.batch_size) - last_print >= 5000 or (i + args.batch_size) >= total:
                print(f"  [{prefix}] {min(i + args.batch_size, total)}/{total}")
                last_print = i + args.batch_size

        zf.close()
        return results

    seen_results = process_set("seen_pair",
        cfg.eval_seen_csv, cfg.eval_seen_zip)
    unseen_results = process_set("unseen_pair",
        cfg.eval_unseen_csv, cfg.eval_unseen_zip)

    # ── Write submission ──
    all_results = seen_results + unseen_results
    all_results.sort(key=lambda x: x[0])

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "posterior"])
        w.writerows(all_results)

    # ── Stats ──
    total = len(all_results)
    scores = np.array(stats["by_score"])
    print(f"\n{'='*50}")
    print(f"Routing stats (total={total}):")
    print(f"  clean         : {stats['clean']:>6d} ({stats['clean']/total*100:.1f}%)")
    print(f"  moderate      : {stats['moderate']:>6d} ({stats['moderate']/total*100:.1f}%)")
    print(f"  severe+seen   : {stats['severe_seen']:>6d} ({stats['severe_seen']/total*100:.1f}%)")
    print(f"  severe+unseen : {stats['severe_unseen']:>6d} ({stats['severe_unseen']/total*100:.1f}%)")
    print(f"Damage score: mean={scores.mean():.3f} std={scores.std():.3f} "
          f"p50={np.median(scores):.3f} p95={np.percentile(scores,95):.3f}")
    print(f"Saved: {args.out}")


def main():
    p = argparse.ArgumentParser(description="Routing inference: AT v8 + AA v3")
    p.add_argument("--at-ckpt", default=os.path.join(PATHS.root,
                    "output/dual_at_v8_text/best.pt"))
    p.add_argument("--aa-ckpt", default=os.path.join(PATHS.root,
                    "output/backup_final/aa_v3_best_seen07934.pt"))
    p.add_argument("--out", default=os.path.join(PATHS.root, "submission_routed.csv"))
    p.add_argument("--batch-size", type=int, default=64, dest="batch_size")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    run_inference(args)


if __name__ == "__main__":
    main()

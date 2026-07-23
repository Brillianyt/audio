"""OHEM for AA: find actual model mistakes on training data.

AA model (MultiTaskWhisperKWSV3) evaluates:
- FP: different words scored as same (label=0, score high)
- FN: same words scored as different (label=1, score low)
"""
import argparse, csv, json, os, sys, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
from config import PATHS
from train_whisper_v3 import MultiTaskWhisperKWSV3, WhisperPairDatasetV3, WhisperConfigV3, load_pairs_with_text, collate_whisper_v3

import torch
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/aa_v3/best.pt")
    p.add_argument("--fp-thresh", type=float, default=0.4)
    p.add_argument("--fn-thresh", type=float, default=0.3)
    p.add_argument("--max-pairs", type=int, default=200000)
    p.add_argument("--out", default="baseline/hard_neg_aa_ohem.json")
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load AA model
    print(f"Loading AA from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = MultiTaskWhisperKWSV3("base", embed_dim=256, unfreeze_layers=2).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"  loaded (mean_auc={ckpt.get('auc', '?'):.4f})")

    # Load training pairs
    csv_path = os.path.join(PATHS.root, "train", "train_label.csv")
    zip_path = os.path.join(PATHS.root, "train", "wav.zip")
    all_pairs = load_pairs_with_text(csv_path)
    print(f"Total pairs: {len(all_pairs)}")

    # Sample
    rng = np.random.default_rng(42)
    if len(all_pairs) > args.max_pairs:
        all_pairs = rng.choice(all_pairs, args.max_pairs, replace=False).tolist()
        print(f"Sampled: {len(all_pairs)}")

    # Run AA inference
    cfg = WhisperConfigV3()
    cfg.max_audio_sec = 3.0; cfg.sample_rate = 16000; cfg.num_workers = 0
    ds = WhisperPairDatasetV3(all_pairs, zip_path, cfg)
    ld = DataLoader(ds, args.bs, shuffle=False, collate_fn=collate_whisper_v3, num_workers=0)

    fps, fns = [], []
    with torch.no_grad():
        for i, (e, q, y, wt, wq, ids) in enumerate(ld):
            e, q = e.to(device), q.to(device)
            logit, _, _ = model(e, q)
            prob = torch.sigmoid(logit).cpu().numpy()
            y_np = y.numpy()

            for j in range(len(ids)):
                idx = i * args.bs + j
                if idx >= len(all_pairs): break
                orig = all_pairs[idx]

                if y_np[j] == 0 and prob[j] > args.fp_thresh:
                    fps.append({
                        "id": f"aa_ohem_{ids[j]}",
                        "enroll_id": ids[j], "query_id": ids[j],
                        "enroll_txt": orig.get("enroll_txt", ""),
                        "query_txt": orig.get("query_txt", ""),
                        "label": 0, "score": float(prob[j]), "type": "FP",
                    })
                elif y_np[j] == 1 and prob[j] < args.fn_thresh:
                    fns.append({
                        "id": f"aa_ohem_{ids[j]}",
                        "enroll_id": ids[j], "query_id": ids[j],
                        "enroll_txt": orig.get("enroll_txt", ""),
                        "query_txt": orig.get("query_txt", ""),
                        "label": 1, "score": float(prob[j]), "type": "FN",
                    })

            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}: FP={len(fps)} FN={len(fns)}", flush=True)

    print(f"\nResults: FP={len(fps)} FN={len(fns)}")

    # Deduplicate by word pair
    seen = set()
    deduped = []
    for p in fps + fns:
        key = (p["enroll_txt"].lower(), p["query_txt"].lower())
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    print(f"Unique word pairs: {len(deduped)}")

    # Show samples
    print("\nSample FPs:")
    for p in fps[:10]:
        print(f"  {p['enroll_txt']:18s} <-> {p['query_txt']:18s}  score={p['score']:.3f}")
    print("\nSample FNs:")
    for p in fns[:10]:
        print(f"  {p['enroll_txt']:18s} <-> {p['query_txt']:18s}  score={p['score']:.3f}")

    # Save
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(deduped, f, ensure_ascii=False)
    print(f"\nSaved {len(deduped)} pairs → {args.out}")


if __name__ == "__main__":
    main()

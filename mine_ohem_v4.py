"""OHEM for AT: find actual model mistakes (FP/FN) on training data.

Uses the correct AT pipeline (root train_dual.py AudioTextModel).
Collects hard negatives that the model actually confuses.
"""
import argparse, csv, json, os, sys, numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
from config import PATHS

# Import root train_dual.py explicitly
import importlib.util as _iu
spec = _iu.spec_from_file_location("root_td", 
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dual.py"))
rtd = _iu.module_from_spec(spec)
spec.loader.exec_module(rtd)
AudioTextModel = rtd.AudioTextModel
PairDataset = rtd.PairDataset
collate_text = rtd.collate_text
Config = rtd.Config

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/dual_at_v2_text/best.pt",
                   help="AT model checkpoint")
    p.add_argument("--fp-thresh", type=float, default=0.4,
                   help="False positive threshold: label=0, score>thresh")
    p.add_argument("--fn-thresh", type=float, default=0.3,
                   help="False negative threshold: label=1, score<thresh")
    p.add_argument("--max-pairs", type=int, default=200000,
                   help="Max training pairs to evaluate")
    p.add_argument("--out", default="baseline/hard_neg_at_ohem.json")
    p.add_argument("--bs", type=int, default=256)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model
    print(f"Loading AT from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = AudioTextModel("", embed_dim=256).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"  seen={ckpt.get('auc_seen','?'):.4f} unseen={ckpt.get('auc_unseen','?'):.4f}")

    # Load training pairs
    csv_path = os.path.join(PATHS.root, "train", "train_label.csv")
    zip_path = os.path.join(PATHS.root, "train", "wav.zip")
    all_pairs = rtd.load_pairs(csv_path)
    print(f"Original pairs: {len(all_pairs)}")

    # Sample for speed
    rng = np.random.default_rng(42)
    if len(all_pairs) > args.max_pairs:
        all_pairs = rng.choice(all_pairs, args.max_pairs, replace=False).tolist()
        print(f"Sampled: {len(all_pairs)}")

    # Run inference
    cfg = Config()
    cfg.batch_size = args.bs
    ds = PairDataset(all_pairs, zip_path, cfg, "text")
    ld = DataLoader(ds, args.bs, shuffle=False, collate_fn=collate_text, num_workers=4)

    fps, fns = [], []
    with torch.no_grad():
        for i, (e, q, y, txts, ids) in enumerate(ld):
            e, q = e.to(device), q.to(device)
            _, score, _, _ = model(e, txts, q)
            prob = torch.sigmoid(score).cpu().numpy()
            y_np = y.numpy()
            
            for j in range(len(ids)):
                pair_idx = i * args.bs + j
                if pair_idx >= len(all_pairs): break
                orig = all_pairs[pair_idx]
                
                if y_np[j] == 0 and prob[j] > args.fp_thresh:
                    fps.append({
                        "id": f"ohem_{ids[j]}_x_{ids[j]}",
                        "enroll_id": ids[j],
                        "query_id": ids[j],
                        "enroll_txt": orig.get("enroll_txt", ""),
                        "query_txt": orig.get("query_txt", ""),
                        "label": 0,
                        "score": float(prob[j]),
                        "type": "FP",
                    })
                elif y_np[j] == 1 and prob[j] < args.fn_thresh:
                    fns.append({
                        "id": f"ohem_{ids[j]}_x_{ids[j]}",
                        "enroll_id": ids[j],
                        "query_id": ids[j],
                        "enroll_txt": orig.get("enroll_txt", ""),
                        "query_txt": orig.get("query_txt", ""),
                        "label": 1,
                        "score": float(prob[j]),
                        "type": "FN",
                    })
            
            if (i + 1) % 50 == 0:
                print(f"  batch {i+1}: FP={len(fps)} FN={len(fns)}", flush=True)

    print(f"\nResults:")
    print(f"  False positives (label=0, score>{args.fp_thresh}): {len(fps)}")
    print(f"  False negatives (label=1, score<{args.fn_thresh}): {len(fns)}")

    # Deduplicate by word pair
    seen_words = set()
    deduped = []
    for p in fps + fns:
        key = (p["enroll_txt"].lower(), p["query_txt"].lower())
        if key not in seen_words:
            seen_words.add(key)
            deduped.append(p)
    
    print(f"  Unique word pairs: {len(deduped)}")

    # Show examples
    print("\nSample FPs (model thinks match, but they don't):")
    for p in fps[:10]:
        print(f"  {p['enroll_txt']:18s} <-> {p['query_txt']:18s}  score={p['score']:.3f}")

    print("\nSample FNs (model thinks don't match, but they do):")
    for p in fns[:10]:
        print(f"  {p['enroll_txt']:18s} <-> {p['query_txt']:18s}  score={p['score']:.3f}")

    # Save
    out = [{"id":p["id"],"enroll_id":p["enroll_id"],"query_id":p["query_id"],
            "enroll_txt":p["enroll_txt"],"query_txt":p["query_txt"],
            "label":p["label"],"score":p["score"],"type":p["type"]}
           for p in deduped]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"\nSaved {len(out)} pairs → {args.out}")


if __name__ == "__main__":
    main()

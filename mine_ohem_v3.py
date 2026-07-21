"""OHEM v3 — mine on actual training pairs, not word-level cross-similarity."""
import torch, os, sys, numpy as np, json, argparse
import torch.nn.functional as F
sys.path.insert(0,'baseline'); from config import PATHS
import train_at_orig

device='cuda'

p = argparse.ArgumentParser()
p.add_argument('--ckpt', default='output/backup/at_best.pt')
p.add_argument('--fp-thresh', type=float, default=0.4)
p.add_argument('--fn-thresh', type=float, default=0.3)
p.add_argument('--max-pairs', type=int, default=200000)
args = p.parse_args()

# Load model
model = train_at_orig.AudioTextModel(256, unfreeze=4).to(device)
ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
print(f'AT loaded: unseen={ckpt.get("auc_unseen",-1):.4f}')

# Load original training pairs
from train_at_orig import load_pairs, PairDataset, collate_text
from torch.utils.data import DataLoader

cfg = type('C',(),{})(); cfg.max_audio_sec=1.5
csv_path = os.path.join(PATHS.root, 'train', 'train_label.csv')
zip_path = os.path.join(PATHS.root, 'train', 'wav.zip')
all_pairs = load_pairs(csv_path)
print(f'Total original pairs: {len(all_pairs)}')

# Sample for speed
rng = np.random.default_rng(42)
if len(all_pairs) > args.max_pairs:
    all_pairs = rng.choice(all_pairs, args.max_pairs, replace=False).tolist()
    print(f'Sampled: {len(all_pairs)}')

loader = DataLoader(PairDataset(all_pairs, zip_path, cfg),
                   batch_size=256, collate_fn=collate_text, shuffle=False, num_workers=4)

false_positives = []
false_negatives = []

with torch.no_grad():
    for e, q, y, txts, ids in loader:
        e, q = e.to(device), q.to(device)
        # AT model: cos(text(enroll_txt), audio(query_audio))
        _, _, et = model(e, txts)
        eq = model.encoder(q)
        scores = (et * eq).sum(-1)  # (B,)

        for i in range(len(y)):
            label = float(y[i])
            score = float(scores[i])
            tid = ids[i]
            e_txt = txts[i]

            if label == 0 and score > args.fp_thresh:
                false_positives.append({
                    'id': tid, 'enroll_txt': e_txt,
                    'score': score, 'label': 0, 'type': 'ohem_fp'
                })
            elif label == 1 and score < args.fn_thresh:
                false_negatives.append({
                    'id': tid, 'enroll_txt': e_txt,
                    'score': score, 'label': 1, 'type': 'ohem_fn'
                })

false_positives.sort(key=lambda x: -x['score'])
false_negatives.sort(key=lambda x: x['score'])

print(f'\nFalse Positives (label=0, score>{args.fp_thresh}): {len(false_positives)}')
print(f'False Negatives (label=1, score<{args.fn_thresh}): {len(false_negatives)}')

if false_positives:
    top = false_positives[:10]
    for p in top:
        print(f'  FP: {p["id"]} {p["enroll_txt"]} score={p["score"]:.4f}')
if false_negatives:
    top = false_negatives[:10]
    for p in top:
        print(f'  FN: {p["id"]} {p["enroll_txt"]} score={p["score"]:.4f}')

json.dump(false_positives, open('baseline/hard_neg_at_ohem_v3.json','w'), indent=2)
json.dump(false_negatives, open('baseline/hard_pos_at_ohem_v3.json','w'), indent=2)
print(f'Saved hard_neg_at_ohem_v3.json ({len(false_positives)})')
print(f'Saved hard_pos_at_ohem_v3.json ({len(false_negatives)})')

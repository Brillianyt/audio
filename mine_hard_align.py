"""Mine hard alignment examples: find pairs where cos(et, ea) is low."""
import torch, sys, os, csv, json, numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'baseline')
import importlib.util as iu
spec = iu.spec_from_file_location("rtd", "train_dual.py")
rtd = iu.module_from_spec(spec); spec.loader.exec_module(rtd)
from config import PATHS
from torch.utils.data import DataLoader
import torch.nn.functional as F

device = 'cuda'
model = rtd.AudioTextModel("", 256).to(device)
ckpt = torch.load("output/dual_at_v8_text/best.pt", map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# Load all training pairs, sample 200K
pairs = rtd.load_pairs(os.path.join(PATHS.root, "train", "train_label.csv"))
rng = np.random.default_rng(42)
sample = rng.choice(pairs, 200000, replace=False).tolist()
print(f"Sample: {len(sample)} pairs")

cfg = rtd.Config(); cfg.__post_init__()
ld = DataLoader(rtd.PairDataset(sample, cfg.train_zip, cfg, "text"),
                512, shuffle=False, collate_fn=rtd.collate_text, num_workers=4)

results = []
with torch.no_grad():
    for i, (e, q, y, txts, ids) in enumerate(ld):
        e, q = e.to(device), q.to(device)
        txts = [t.lower() for t in txts]
        ea = F.normalize(model.encoder(e), dim=-1)
        eq = F.normalize(model.encoder(q), dim=-1)
        et = F.normalize(model.text_enc(txts)[0], dim=-1)
        
        cos_e = (ea * et).sum(-1)
        cos_q = (eq * et).sum(-1)
        
        for j in range(len(ids)):
            idx = i * 512 + j
            if idx >= len(sample): break
            p = sample[idx]
            results.append({
                'word': p['enroll_txt'], 'label': p['label'],
                'cos_enroll': float(cos_e[j]), 'cos_query': float(cos_q[j]),
                'id': p['id']
            })
        if (i+1) % 50 == 0:
            cs_e = [r['cos_enroll'] for r in results]
            cs_q = [r['cos_query'] for r in results]
            print(f"  batch {i+1}: mean_cos_e={np.mean(cs_e):.4f} mean_cos_q={np.mean(cs_q):.4f}")

# Save
min_cos = min(r['cos_enroll'] for r in results)
print(f"\nTotal: {len(results)} pairs, min_cos_e={min_cos:.4f}")

# Sort by cos_enroll (worst first)
results.sort(key=lambda r: r['cos_enroll'])

# Save top hard examples
hard = [r for r in results if r['cos_enroll'] < 0.3]
print(f"cos(et,ea) < 0.3: {len(hard)} pairs")
print(f"\nBottom 20:")
for r in results[:20]:
    print(f"  {r['word']:20s} cos_e={r['cos_enroll']:.4f} cos_q={r['cos_query']:.4f}")

with open("baseline/hard_align.json", "w") as f:
    json.dump(results[:5000], f, indent=2)
print("\nSaved top 5000 hard alignment pairs to baseline/hard_align.json")

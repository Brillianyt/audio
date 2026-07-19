"""Clean up training data — dedup, filter, consolidate."""
import json, os, sys, csv
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

root = PATHS.root

# ── 1. Load and merge all data ──
all_pairs = []
with open(os.path.join(root, "train", "train_label.csv"), encoding="utf-8") as f:
    all_pairs += [r for r in csv.DictReader(f)]

extra_files = [
    "baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
    "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
    "train/self_paired.json", "train/self_paired_xl.json",
    "train/fill_pos_pairs.json", "train/mega_pairs.json",
    "train/speech_commands_pairs.json", "baseline/hard_neg_atv2.json",
    "train/librispeech_pairs.json",
    "baseline/hard_neg_at_final.json", "baseline/hard_neg_aa_final.json",
]
for fn in extra_files:
    fp = os.path.join(root, fn)
    if os.path.isfile(fp):
        data = json.load(open(fp))
        all_pairs += data
        print(f"  loaded {fn}: {len(data)}")

print(f"\nTotal before cleaning: {len(all_pairs)} pairs")

# ── 2. Separate pos and neg ──
pos = [p for p in all_pairs if str(p.get("label",0)) == "1" or p.get("label") == 1]
neg = [p for p in all_pairs if str(p.get("label",0)) == "0" or p.get("label") == 0]

print(f"  pos: {len(pos)}, neg: {len(neg)}")

# ── 3. Clean pos: max 10 per word (keep diverse audio) ──
word_counter = defaultdict(int)
pos_clean = []
for p in pos:
    word = p.get("enroll_txt", "").lower()
    if word_counter[word] < 10:
        word_counter[word] += 1
        pos_clean.append(p)

print(f"  pos after dedup: {len(pos_clean)} (max 10/word)")

# ── 4. Clean neg: dedup word pairs, keep unique ──
seen_pairs = set()
neg_clean = []
for p in neg:
    w1 = p.get("enroll_txt", "").lower()
    w2 = p.get("query_txt", "").lower()
    key = tuple(sorted([w1, w2]))
    if key not in seen_pairs:
        seen_pairs.add(key)
        neg_clean.append(p)

print(f"  neg after dedup: {len(neg_clean)} (unique word pairs)")

# ── 5. Filter neg by similarity if available ──
neg_filtered = []
for p in neg_clean:
    sim = p.get("similarity", 0)
    # Keep if similarity exists and is >= 0.3, or if no similarity field (keep old hard_neg)
    if "similarity" not in p or sim >= 0.3:
        neg_filtered.append(p)

print(f"  neg after sim filter: {len(neg_filtered)}")

# ── 6. Save cleaned data ──
cleaned = pos_clean + neg_filtered
out_path = os.path.join(root, "train", "cleaned_pairs.json")
with open(out_path, "w") as f:
    json.dump(cleaned, f)
print(f"\nSaved {len(cleaned)} cleaned pairs to {out_path}")

# Stats
words = set()
for p in pos_clean:
    words.add(p.get("enroll_txt","").lower())
print(f"  unique words: {len(words)}")

hard_neg_count = sum(1 for p in neg_filtered if any(k in p.get("id","") for k in {"hard_neg","hn_","phoneme","hnat_"}))
print(f"  hard neg: {hard_neg_count}")
print(f"  easy neg: {len(neg_filtered) - hard_neg_count}")

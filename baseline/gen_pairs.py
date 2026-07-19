"""Self-paired training data generator.

从现有训练数据中自行配对生成更多训练样本对。

原理(赛题允许):
  正样本(label=1): 同词的不同 enrollment 音频互相配对
    例: cat_enroll_A.wav + cat_enroll_B.wav → label=1
  负样本(label=0): 不同词的 enrollment 音频交叉配对
    例: cat_enroll_A.wav + dog_enroll_B.wav → label=0

用法:
  python baseline/gen_pairs.py \
      --csv train/train_label.csv \
      --out train/self_paired.json \
      --n-pos 100000 --n-neg 200000
"""
import argparse, csv, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="原始训练 CSV")
    p.add_argument("--out", required=True, help="输出 JSON")
    p.add_argument("--n-pos", type=int, default=100000, help="正样本对数")
    p.add_argument("--n-neg", type=int, default=200000, help="负样本对数")
    p.add_argument("--pos-per-word-combo", type=int, default=10)
    p.add_argument("--neg-per-word-combo", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Read original pairs, collect enrollment audios per word
    word_to_enrolls = defaultdict(list)  # word -> [(pair_id, word_text)]
    with open(args.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["enroll_txt"] == r["query_txt"] and r["label"] == "1":
                w = r["enroll_txt"].lower()
                word_to_enrolls[w].append((r["id"], r["enroll_txt"]))

    words = sorted(word_to_enrolls.keys())
    total_enrolls = sum(len(v) for v in word_to_enrolls.values())
    print(f"[gen] {len(words)} words, {total_enrolls} enrollment audios")

    # Filter words with >=2 enrollments (needed for positive pairs)
    valid_words = [w for w in words if len(word_to_enrolls[w]) >= 2]
    print(f"[gen] {len(valid_words)} words with >=2 enrollments")

    pairs = []
    used_ids = set()

    # ── Positive pairs: same word, different audios ──
    print(f"[gen] generating {args.n_pos} positive pairs...")
    n_pos = 0
    for w in valid_words:
        enrolls = word_to_enrolls[w]
        n_enroll = len(enrolls)
        # How many unique pairs can we make?
        max_pairs_this_word = min(
            args.n_pos // max(1, len(valid_words)) + 5,
            n_enroll * (n_enroll - 1) // 2
        )
        for _ in range(min(max_pairs_this_word, args.pos_per_word_combo * n_enroll)):
            i, j = rng.choice(n_enroll, 2, replace=False)
            id1, txt1 = enrolls[i]
            id2, txt2 = enrolls[j]
            key = (id1, id2)
            if key in used_ids:
                continue
            used_ids.add(key)
            pairs.append({
                "id": f"pos_{id1}_x_{id2}",
                "enroll_id": id1,
                "query_id": id2,
                "enroll_txt": txt1,
                "query_txt": txt2,
                "label": 1,
                "source": "self_paired",
            })
            n_pos += 1
            if n_pos >= args.n_pos:
                break
        if n_pos >= args.n_pos:
            break
    print(f"  generated {n_pos} positive pairs")

    # ── Negative pairs: different words ──
    print(f"[gen] generating {args.n_neg} negative pairs...")
    n_neg = 0
    # Sort by enrollment count for balanced sampling
    weighted_words = []
    for w in valid_words:
        weighted_words.extend([w] * len(word_to_enrolls[w]))
    rng.shuffle(weighted_words)

    combo_count = defaultdict(int)
    for _ in range(args.n_neg * 3):  # 3x attempts to fill quota
        w1 = weighted_words[rng.integers(0, len(weighted_words))]
        w2 = weighted_words[rng.integers(0, len(weighted_words))]
        if w1 >= w2:  # ensure w1 < w2 for dedup
            continue

        ck = (w1, w2)
        if combo_count[ck] >= args.neg_per_word_combo:
            continue

        e1 = word_to_enrolls[w1]
        e2 = word_to_enrolls[w2]
        i = rng.integers(0, len(e1))
        j = rng.integers(0, len(e2))
        id1, txt1 = e1[i]
        id2, txt2 = e2[j]
        key = (id1, id2)
        if key in used_ids:
            continue
        used_ids.add(key)
        pairs.append({
            "id": f"neg_{id1}_x_{id2}",
            "enroll_id": id1,
            "query_id": id2,
            "enroll_txt": txt1,
            "query_txt": txt2,
            "label": 0,
            "source": "self_paired",
        })
        combo_count[ck] += 1
        n_neg += 1
        if n_neg >= args.n_neg:
            break

    print(f"  generated {n_neg} negative pairs")
    print(f"[gen] total: {len(pairs)} pairs")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"[gen] saved to {args.out} ({os.path.getsize(args.out)/1024:.0f} KB)")


if __name__ == "__main__":
    main()

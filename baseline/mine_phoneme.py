"""Broader Hard Negative Mining — Phoneme Edit Distance + Embedding Combined.

覆盖训练词表中所有发音相似词对（不限于已有正样本的词）。

1. 用 CMU dict 将全部训练词表转为 ARPAbet 音素
2. 对所有词对计算纯音素编辑距离
3. abs_edit_distance <= 2 的词对 → hard negative candidates
4. 从每个词的 enrollment 音频中交叉配对

与 iter_hard_neg.py 的区别:
  - 不依赖模型 embedding，纯语音学距离
  - 覆盖所有词对，不限于有正样本的词
  - 缺样本的词也能参与（只要 CMU dict 里有）

用法:
  python baseline/mine_phoneme.py \
      --csv train/train_label.csv \
      --out baseline/hard_neg_phoneme.json \
      --max-abs-dist 2 --max-pairs 100000
"""
import argparse, csv, io, json, os, re as _re, sys, zipfile
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.config import PATHS

# ── CMU dict ──
_cmudict = None
def _get_cmudict():
    global _cmudict
    if _cmudict is None:
        import cmudict
        _cmudict = cmudict.dict()
        print(f"[cmudict] {len(_cmudict):,} words")
    return _cmudict

def word_to_arpabet(word):
    cmu = _get_cmudict()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w: return []
    pl = cmu.get(w)
    return [_re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []

# ── Phoneme edit distance (pure) ──
def phoneme_edit_distance(seq1, seq2):
    m, n = len(seq1), len(seq2)
    if m==0 and n==0: return 0
    if m==0 or n==0: return max(m,n)
    dp = np.zeros((m+1,n+1), dtype=np.int32)
    for i in range(m+1): dp[i,0]=i
    for j in range(n+1): dp[0,j]=j
    for i in range(1,m+1):
        for j in range(1,n+1):
            dp[i,j] = min(dp[i-1,j]+1, dp[i,j-1]+1,
                          dp[i-1,j-1] + (0 if seq1[i-1]==seq2[j-1] else 1))
    return int(dp[m,n])

# ── Main ──
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--out", default="")
    p.add_argument("--max-abs-dist", type=int, default=2)
    p.add_argument("--max-pairs", type=int, default=100000)
    p.add_argument("--max-per-combo", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    # Collect all enrollment audios per word (use ALL pairs, not just positive)
    word_to_audios = defaultdict(list)
    all_words_set = set()
    with open(args.csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            w = r["enroll_txt"].lower()
            word_to_audios[w].append((r["id"], r["enroll_txt"]))
            all_words_set.add(w)
            all_words_set.add(r["query_txt"].lower())

    all_words = sorted(all_words_set)
    print(f"[mine] {len(all_words)} unique words, "
          f"{sum(len(v) for v in word_to_audios.values())} audios")

    # Convert to ARPAbet
    word_to_phon = {}
    for w in all_words:
        ph = word_to_arpabet(w)
        if ph: word_to_phon[w] = ph
    print(f"[mine] {len(word_to_phon)} words with phonemes")

    # Find phonemically similar pairs
    wlist = sorted(word_to_phon.keys(), key=lambda w: len(word_to_phon[w]))
    similar = []
    for i in range(len(wlist)):
        w1 = wlist[i]; p1 = word_to_phon[w1]; l1 = len(p1)
        j = i+1
        while j < len(wlist):
            w2 = wlist[j]; p2 = word_to_phon[w2]
            if len(p2) - l1 > args.max_abs_dist + 1: break
            if abs(len(p2)-l1) <= args.max_abs_dist:
                d = phoneme_edit_distance(p1, p2)
                if 0 < d <= args.max_abs_dist:
                    similar.append((w1, w2, d, round(d/max(l1,len(p2)),4)))
            j += 1

    similar.sort(key=lambda x: x[2])  # sort by absolute distance
    print(f"[mine] {len(similar)} phonemically similar pairs")
    if similar:
        print("  top 20:")
        for w1,w2,d,nd in similar[:20]:
            p1=' '.join(word_to_phon.get(w1,[]))
            p2=' '.join(word_to_phon.get(w2,[]))
            n1=len(word_to_audios.get(w1,[]))
            n2=len(word_to_audios.get(w2,[]))
            print(f"    {w1:15s} [{p1:30s}]")
            print(f"    {w2:15s} [{p2:30s}]  d={d} n1={n1} n2={n2}")
            print()

    # Generate hard negative pairs
    extra = []; used = set()
    combo_count = defaultdict(int)
    max_per = args.max_per_combo

    for w1, w2, d, nd in similar:
        a1 = word_to_audios.get(w1, []); a2 = word_to_audios.get(w2, [])
        if not a1 or not a2: continue
        ck = tuple(sorted([w1,w2]))
        if combo_count[ck] >= max_per: continue
        n = min(max_per - combo_count[ck], len(a1), len(a2))
        s1 = rng.choice(len(a1), n, replace=False)
        s2 = rng.choice(len(a2), n, replace=False)
        for si in s1:
            for sj in s2:
                id1,t1 = a1[si]; id2,t2 = a2[sj]
                if (id1,id2) in used: continue
                used.add((id1,id2))
                extra.append({"id":f"ph_{id1}_x_{id2}","enroll_id":id1,"query_id":id2,
                              "enroll_txt":t1,"query_txt":t2,"label":0,
                              "phoneme_dist":d,"phoneme_norm":nd})
                combo_count[ck] += 1
                if len(extra) >= args.max_pairs: break
            if len(extra) >= args.max_pairs: break
        if len(extra) >= args.max_pairs: break

    out_path = args.out or os.path.join(os.path.dirname(__file__), "hard_neg_phoneme.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False)
    print(f"[mine] {len(extra)} pairs → {out_path} "
          f"({len(combo_count)} word-combos, max_per={max_per})")

if __name__ == "__main__":
    main()

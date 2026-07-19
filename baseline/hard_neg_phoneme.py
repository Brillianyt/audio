"""Phoneme-Level Hard Negative Mining for Keyword Spotting.

基于 CMU Pronouncing Dictionary + 纯 ARPAbet 编辑距离的难负样本挖掘。

设计原则 (参考 PLCL / MALEFA 等 SOTA 工作):
  1. 纯音素符号编辑距离: compare ARPAbet labels, NOT articulatory features
     - "IH" vs "AY" → cost=1 (全替换), not 0.39 (同为元音)
     - 只有完全相同音素才 cost=0
  2. 允许长度差异 ±2: 不强制等长，捕获 hi↔hire 这类关键 confusable pairs
  3. 阈值收紧到 0.6: 只保留真正发音相似的词对

用法:
    from hard_neg_phoneme import PhonemeHardNegativeMiner
    miner = PhonemeHardNegativeMiner(csv_path, max_phoneme_dist=0.6)
    extra_pairs = miner.generate(max_pairs=50000)
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

# ── cmudict (pip install cmudict, 纯离线) ──
_cmudict = None

def _get_cmudict() -> Dict[str, List[List[str]]]:
    global _cmudict
    if _cmudict is None:
        import cmudict
        _cmudict = cmudict.dict()
        print(f"[cmudict] loaded {len(_cmudict):,} words (offline)")
    return _cmudict

def word_to_arpabet(word: str) -> List[str]:
    """单词 -> ARPAbet 音素序列 (去stress)，OOV 返回空。"""
    cmu = _get_cmudict()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w:
        return []
    plist = cmu.get(w)
    if plist:
        return [re.sub(r'[0-2]$', '', p) for p in plist[0]]
    return []


def phoneme_edit_distance(seq1: List[str], seq2: List[str]) -> float:
    """纯 ARPAbet 符号 Levenshtein 编辑距离，归一化到 [0,1]。

    只有 phoneme label 完全相同才 cost=0  (如 "L"=="L"),
    不同 label 一律 cost=1 (如 "IH"!="AY")。
    """
    m, n = len(seq1), len(seq2)
    if m == 0 and n == 0:
        return 0.0
    if m == 0 or n == 0:
        return 1.0

    dp = np.full((m + 1, n + 1), 0, dtype=np.int32)
    for i in range(m + 1):
        dp[i, 0] = i
    for j in range(n + 1):
        dp[0, j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost,
            )

    return float(dp[m, n]) / max(m, n)


# ═══════════════════════════════════════════════════════════════════════
# Phoneme Hard Negative Miner
# ═══════════════════════════════════════════════════════════════════════

class PhonemeHardNegativeMiner:
    """基于纯音素编辑距离的难负样本挖掘器。

    核心逻辑:
      1. 查 CMU 词典 -> ARPAbet 音素序列
      2. 对所有词对计算纯音素编辑距离 (允许长度差 ±2)
      3. 筛选 distance <= max_dist 的不同词对
      4. 按距离分桶采样 + 每词对采样上限 -> 生成训练 pair
    """

    def __init__(
        self,
        csv_path: str,
        max_phoneme_dist: float = 0.6,
        min_phoneme_dist: float = 0.05,
    ):
        self.max_dist = max_phoneme_dist
        self.min_dist = min_phoneme_dist

        _get_cmudict()

        # 读 CSV
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append(r)

        # word -> [(id, word)]
        self.word_to_pos: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for r in self.rows:
            if r["enroll_txt"] == r["query_txt"] and r["label"] == "1":
                self.word_to_pos[r["enroll_txt"].lower()].append(
                    (r["id"], r["enroll_txt"])
                )

        # 收集所有词 + 转 ARPAbet
        all_words: Set[str] = set()
        for r in self.rows:
            all_words.add(r["enroll_txt"].lower())
            all_words.add(r["query_txt"].lower())

        self.word_to_phon: Dict[str, List[str]] = {}
        for w in sorted(all_words):
            phons = word_to_arpabet(w)
            if phons:
                self.word_to_phon[w] = phons

        # 查找相似词对
        self.similar_pairs = self._find_similar(all_words)

        n_has_phon = len(self.word_to_phon)
        n_has_pos = len(self.word_to_pos)
        print(
            f"[PhonemeMiner] words={len(all_words)} "
            f"phon_mapped={n_has_phon} has_pos={n_has_pos} "
            f"similar_pairs={len(self.similar_pairs)}"
        )

    def _find_similar(
        self, words: Set[str]
    ) -> List[Tuple[str, str, float]]:
        """用绝对音素编辑距离 ≤ max_abs_dist 找 confusable pairs。

        只用 <= 2 个音素不同的词对 (如 hi↔hire 差1个, cat↔cut 差1个)。
        这天然保证只有真正发音相似的词对入选，不受词长度影响。
        """
        max_abs_dist = 1  # 只差1个音素: hi↔hire, cat↔cut

        word_list = sorted(
            [w for w in words if w in self.word_to_phon],
            key=lambda w: len(self.word_to_phon[w]),
        )

        similar: List[Tuple[str, str, float]] = []

        for i in range(len(word_list)):
            w1 = word_list[i]
            p1 = self.word_to_phon[w1]
            l1 = len(p1)

            j = i + 1
            while j < len(word_list):
                w2 = word_list[j]
                p2 = self.word_to_phon[w2]
                l2 = len(p2)

                if l2 - l1 > 3:
                    break

                if abs(l2 - l1) <= 2 and w1 != w2:
                    # 绝对编辑距离
                    abs_dist = int(phoneme_edit_distance(p1, p2) * max(l1, l2))
                    if abs_dist <= max_abs_dist:
                        # 存归一化距离用于排序
                        norm = round(float(abs_dist) / max(l1, l2), 4)
                        similar.append((w1, w2, norm))

                j += 1

        similar.sort(key=lambda x: x[2])
        return similar

    def generate(
        self, max_pairs: int = 50000, balance_by_distance: bool = True
    ) -> List[dict]:
        """生成 hard negative 配对。"""
        if not self.similar_pairs:
            print("[PhonemeMiner] 无相似词对")
            return []

        extra: List[dict] = []
        used: Set[Tuple[str, str]] = set()

        # 距离分桶
        if balance_by_distance and len(self.similar_pairs) > 100:
            dists = [p[2] for p in self.similar_pairs]
            q33, q66 = np.percentile(dists, [33, 66])
            groups = {"near": [], "mid": [], "far": []}
            for w1, w2, d in self.similar_pairs:
                if d <= q33:
                    groups["near"].append((w1, w2, d))
                elif d <= q66:
                    groups["mid"].append((w1, w2, d))
                else:
                    groups["far"].append((w1, w2, d))
            n_near = int(max_pairs * 0.50)
            n_mid = int(max_pairs * 0.30)
            n_far = max_pairs - n_near - n_mid

            for g in groups.values():
                np.random.shuffle(g)

            candidates = []
            for g_name, n in [("near", n_near), ("mid", n_mid), ("far", n_far)]:
                candidates.extend(groups[g_name][:n])
            np.random.shuffle(candidates)
        else:
            candidates = list(self.similar_pairs)
            np.random.shuffle(candidates)

        # 每词对最多贡献 max_per_pair 个样本
        max_per_pair = max(1, max_pairs // max(1, len(candidates))) + 5
        pair_used: Dict[Tuple[str, str], int] = defaultdict(int)

        for w1, w2, dist in candidates:
            pk = tuple(sorted([w1, w2]))
            if pair_used[pk] >= max_per_pair:
                continue

            pos1 = self.word_to_pos.get(w1, [])
            pos2 = self.word_to_pos.get(w2, [])
            if not pos1 or not pos2:
                continue

            n_samples = min(10, len(pos1), len(pos2),
                            max_per_pair - pair_used[pk])
            s1 = list(np.random.choice(len(pos1), n_samples, replace=False))
            s2 = list(np.random.choice(len(pos2), n_samples, replace=False))

            for si in s1:
                for sj in s2:
                    id1, txt1 = pos1[si]
                    id2, txt2 = pos2[sj]
                    key = (id1, id2)
                    if key in used:
                        continue
                    used.add(key)
                    extra.append({
                        "id": f"{id1}_x_{id2}",
                        "enroll_id": id1,
                        "query_id": id2,
                        "enroll_txt": txt1,
                        "query_txt": txt2,
                        "label": 0,
                        "phoneme_dist": dist,
                    })
                    pair_used[pk] += 1
                    if len(extra) >= max_pairs:
                        break
                if len(extra) >= max_pairs:
                    break
            if len(extra) >= max_pairs:
                break

        print(
            f"[PhonemeMiner] generated {len(extra)} hard neg pairs "
            f"from {len(pair_used)} word-pairs (max={max_pairs})"
        )
        return extra

    def get_phoneme_distance(self, word1: str, word2: str) -> Optional[float]:
        w1, w2 = word1.lower(), word2.lower()
        if w1 == w2:
            return 0.0
        p1 = self.word_to_phon.get(w1)
        p2 = self.word_to_phon.get(w2)
        if not p1 or not p2:
            return None
        return phoneme_edit_distance(p1, p2)


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else None
    if csv_path is None:
        print("Usage: python hard_neg_phoneme.py <train_csv>")
        sys.exit(1)

    miner = PhonemeHardNegativeMiner(csv_path, max_phoneme_dist=0.6)
    pairs = miner.generate(max_pairs=50, balance_by_distance=True)

    print(f"\n--- Hard negatives (top 20) ---")
    for p in pairs[:20]:
        d = p.get("phoneme_dist", "?")
        print(f"  {p['enroll_txt']:18s} <-> {p['query_txt']:18s}  dist={d}")

    # Demo: 绝对音素编辑距离
    print("\n--- Absolute phoneme edit distance demo ---")
    demos = [
        ("hi", "hire"), ("hi", "high"), ("cat", "cut"),
        ("bat", "pat"), ("lincoln", "lifetime"), ("myself", "music"),
        ("minutes", "indicate"), ("bringing", "creating"),
    ]
    for w1, w2 in demos:
        p1 = miner.word_to_phon.get(w1, [])
        p2 = miner.word_to_phon.get(w2, [])
        if p1 and p2:
            norm = phoneme_edit_distance(p1, p2)
            abs_d = int(norm * max(len(p1), len(p2)))
            mark = "*** CONFUSABLE ***" if abs_d <= 2 else "OK"
            print(f"  {w1:15s} {' '.join(p1):35s}")
            print(f"  {w2:15s} {' '.join(p2):35s}")
            print(f"  -> abs_ed={abs_d} norm={norm:.3f}  {mark}")
            print()

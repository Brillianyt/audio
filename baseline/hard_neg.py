"""难负样本挖掘。

在训练数据中找出发音/拼写相似的词对（如 hi <-> haier），
利用已有的音频文件构造 hard negative 训练对，
提高模型对相似词的区分能力。

用法：
    from hard_neg import HardNegativeMiner

    miner = HardNegativeMiner(csv_path, max_dist=2)
    extra_pairs = miner.generate(max_pairs=50000)
    # extra_pairs 可以和原始 pairs 合并训练
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from typing import List, Optional

import Levenshtein


def find_similar_words(words: set, max_dist: int = 2) -> List[tuple]:
    """在词表中找出拼写相似的不同词对（编辑距离 ≤ max_dist）。"""
    buckets = defaultdict(list)
    for w in words:
        w_lower = w.lower().strip("'s")
        buckets[(w_lower[0] if w_lower else "", len(w_lower))].append(w)

    similar = set()
    for key, group in buckets.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                w1, w2 = group[i].lower(), group[j].lower()
                if w1 == w2:
                    continue
                d = Levenshtein.distance(w1, w2)
                if 0 < d <= max_dist:
                    pair = tuple(sorted([group[i], group[j]]))
                    similar.add(pair)
    return sorted(similar)


class HardNegativeMiner:
    """难负样本挖掘器。

    从训练 CSV 中提取相似词对，利用已有音频构造 hard negative 配对。
    """

    def __init__(self, csv_path: str, max_dist: int = 2):
        self.max_dist = max_dist
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append(r)

        # 索引: word -> [(pair_id, word)]
        self.word_to_pos = defaultdict(list)
        for r in self.rows:
            if r["enroll_txt"] == r["query_txt"] and r["label"] == "1":
                self.word_to_pos[r["enroll_txt"].lower()].append(
                    (r["id"], r["enroll_txt"])
                )

        # 找到所有相似词对
        all_words = set()
        for r in self.rows:
            all_words.add(r["enroll_txt"].lower())
            all_words.add(r["query_txt"].lower())
        self.similar_pairs = find_similar_words(all_words, max_dist)

        print(f"[HardNegativeMiner] 词表大小: {len(all_words)}")
        print(f"[HardNegativeMiner] 相似词对: {len(self.similar_pairs)}")
        print(f"[HardNegativeMiner] 有正样本的词数: {len(self.word_to_pos)}")

    def generate(self, max_pairs: int = 50000) -> List[dict]:
        """生成 hard negative 配对。

        返回的 dict 格式:
            {"id": "<enroll_id>_x_<query_id>",
             "enroll_id": "<实际音频 id>",
             "query_id": "<实际音频 id>",
             "enroll_txt": "...",
             "query_txt": "...",
             "label": 0}
        """
        extra = []
        used = set()

        for w1, w2 in self.similar_pairs:
            pos1 = self.word_to_pos.get(w1.lower(), [])
            pos2 = self.word_to_pos.get(w2.lower(), [])
            if not pos1 or not pos2:
                continue

            for id1, txt1 in pos1:
                for id2, txt2 in pos2:
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
                    })
                    if len(extra) >= max_pairs:
                        break
                if len(extra) >= max_pairs:
                    break
            if len(extra) >= max_pairs:
                break

        print(f"[HardNegativeMiner] 生成了 {len(extra)} 个 hard negative 对")
        return extra

"""External word-audio → pair-json generator (Speech Commands / MSWC / any
word-folder corpus).

Scans  ROOT/<word>/*.wav  (any nesting depth below <word>), then builds:

  1. a zip of 16kHz mono utterances          → train/external_wav.zip
  2. a pair json in the same schema as the   → train/external_pairs.json
     other hard_neg/self_paired files, with a per-pair "zip" field that
     PairDataset (train_dual.py) routes automatically.

Pair logic:
  positive (label=1): same word, DIFFERENT speaker (fallback: different file)
  hard neg (label=0): word pairs whose CMU-dict phoneme edit distance ≤ D
                      (falls back to character edit distance if cmudict
                      is unavailable)
  easy neg (label=0): random different-word pairs (default 0 — the main
                      train csv already has plenty)

Usage (on the server, after downloading Speech Commands):
  python baseline/gen_pairs_external.py \
      --root ~/autodl-tmp/keyword_detect/speech_commands \
      --zip-out train/external_wav.zip \
      --out train/external_pairs.json \
      --n-pos 30000 --n-hard 60000
"""
import argparse, io, json, os, re as _re, sys, zipfile
from collections import defaultdict
import numpy as np
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.config import PATHS

AUDIO_EXT = (".wav", ".flac", ".mp3", ".ogg", ".opus")


# ── phoneme distance (same approach as mine_phoneme.py) ──

_cmudict = None
def _get_cmudict():
    global _cmudict
    if _cmudict is None:
        try:
            import cmudict
            _cmudict = cmudict.dict()
            print(f"[cmudict] {len(_cmudict):,} words")
        except Exception:
            _cmudict = {}
            print("[cmudict] unavailable → character edit distance fallback")
    return _cmudict

def word_to_phonemes(word):
    cmu = _get_cmudict()
    w = word.lower().strip("'s\"-.,!?;:")
    if not w or not cmu:
        return list(w) if w else []          # char fallback
    pl = cmu.get(w)
    return [_re.sub(r'[0-2]$', '', p) for p in pl[0]] if pl else list(w)

def edit_distance(a, b):
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return max(m, n)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1,
                                     prev + (0 if a[i - 1] == b[j - 1] else 1))
    return dp[n]


# ── scan ──

def scan_words(root, max_utts, rng):
    """word → [(speaker, abs_path)]; skips dirs starting with '_' (e.g.
    Speech Commands' _background_noise_ — copy that to train/noise/ instead)."""
    word_to_utts = defaultdict(list)
    for w in sorted(os.listdir(root)):
        wd = os.path.join(root, w)
        if not os.path.isdir(wd) or w.startswith("_"):
            continue
        word = w.lower()
        for dp, _, fns in os.walk(wd):
            for fn in fns:
                if fn.lower().endswith(AUDIO_EXT):
                    speaker = os.path.splitext(fn)[0].split("_")[0]
                    word_to_utts[word].append((speaker, os.path.join(dp, fn)))
    for w in word_to_utts:
        utts = word_to_utts[w]
        rng.shuffle(utts)
        word_to_utts[w] = utts[:max_utts]
    return word_to_utts


# ── audio → zip ──

class UtteranceStore:
    """Writes each used utterance once into the zip, returns its uid."""
    def __init__(self, zip_path):
        os.makedirs(os.path.dirname(zip_path), exist_ok=True)
        self.zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)
        self.path2uid = {}
        self.n_bad = 0

    def add(self, src_path):
        if src_path in self.path2uid:
            return self.path2uid[src_path]
        try:
            w, sr = sf.read(src_path, dtype="float32", always_2d=False)
            if w.ndim > 1:
                w = w.mean(axis=1)
            if sr != 16000:
                import torch, torchaudio
                w = torchaudio.functional.resample(
                    torch.from_numpy(w).unsqueeze(0), sr, 16000
                ).squeeze(0).numpy()
            buf = io.BytesIO()
            sf.write(buf, w.astype(np.float32), 16000, format="WAV",
                     subtype="FLOAT")
            uid = f"ext_{len(self.path2uid):06d}"
            self.zf.writestr(f"wav/{uid}.wav", buf.getvalue())
            self.path2uid[src_path] = uid
            return uid
        except Exception:
            self.n_bad += 1
            return None

    def close(self):
        self.zf.close()


# ── pair builders ──

def make_pos_pairs(word_to_utts, n_want, store, rng, max_per_word):
    pairs, used = [], set()
    words = [w for w, v in word_to_utts.items() if len(v) >= 2]
    quota = max(2, n_want // max(1, len(words)))
    for w in words:
        utts = word_to_utts[w]
        got = 0
        for _ in range(min(quota, max_per_word) * 4):
            if got >= min(quota, max_per_word):
                break
            i, j = rng.choice(len(utts), 2, replace=False)
            sp_i, sp_j = utts[i][0], utts[j][0]
            if sp_i == sp_j and len({u[0] for u in utts}) > 1:
                continue                       # prefer different speakers
            key = tuple(sorted((utts[i][1], utts[j][1])))
            if key in used:
                continue
            used.add(key)
            uid1, uid2 = store.add(utts[i][1]), store.add(utts[j][1])
            if uid1 and uid2:
                pairs.append((uid1, uid2, w, w, 1))
                got += 1
    return pairs


def make_hard_neg_pairs(word_to_utts, n_want, store, rng, max_abs_dist):
    words = sorted(word_to_utts.keys())
    phon = {w: word_to_phonemes(w) for w in words}
    similar = []
    for i, w1 in enumerate(words):
        p1 = phon[w1]
        for w2 in words[i + 1:]:
            p2 = phon[w2]
            if abs(len(p2) - len(p1)) > max_abs_dist:
                continue
            d = edit_distance(p1, p2)
            if 0 < d <= max_abs_dist:
                similar.append((w1, w2, d))
    similar.sort(key=lambda x: x[2])
    print(f"[ext] {len(similar)} phonemically similar word pairs "
          f"(dist<={max_abs_dist})")
    for w1, w2, d in similar[:15]:
        print(f"    {w1} <-> {w2}  d={d}")
    pairs, used = [], set()
    if not similar:
        return pairs
    quota = max(1, n_want // len(similar)) + 1
    for w1, w2, _ in similar:
        for _ in range(quota):
            if len(pairs) >= n_want:
                break
            u1 = word_to_utts[w1][rng.integers(len(word_to_utts[w1]))]
            u2 = word_to_utts[w2][rng.integers(len(word_to_utts[w2]))]
            key = tuple(sorted((u1[1], u2[1])))
            if key in used:
                continue
            used.add(key)
            uid1, uid2 = store.add(u1[1]), store.add(u2[1])
            if uid1 and uid2:
                pairs.append((uid1, uid2, w1, w2, 0))
    return pairs


def make_easy_neg_pairs(word_to_utts, n_want, store, rng):
    words = [w for w, v in word_to_utts.items() if v]
    pairs, used = [], set()
    while len(pairs) < n_want and len(words) >= 2:
        w1, w2 = rng.choice(words, 2, replace=False)
        u1 = word_to_utts[w1][rng.integers(len(word_to_utts[w1]))]
        u2 = word_to_utts[w2][rng.integers(len(word_to_utts[w2]))]
        key = tuple(sorted((u1[1], u2[1])))
        if key in used:
            continue
        used.add(key)
        uid1, uid2 = store.add(u1[1]), store.add(u2[1])
        if uid1 and uid2:
            pairs.append((uid1, uid2, w1, w2, 0))
    return pairs


# ── main ──

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True,
                   help="corpus dir with <word>/*.wav layout")
    p.add_argument("--zip-out", default="train/external_wav.zip")
    p.add_argument("--out", default="train/external_pairs.json")
    p.add_argument("--n-pos", type=int, default=30000)
    p.add_argument("--n-hard", type=int, default=60000)
    p.add_argument("--n-easy", type=int, default=0)
    p.add_argument("--max-utts-per-word", type=int, default=400)
    p.add_argument("--max-per-word-pos", type=int, default=2000)
    p.add_argument("--max-abs-dist", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    word_to_utts = scan_words(args.root, args.max_utts_per_word, rng)
    n_utts = sum(len(v) for v in word_to_utts.values())
    print(f"[ext] {len(word_to_utts)} words, {n_utts} utterances")

    zip_abs = os.path.join(PATHS.root, args.zip_out) \
        if not os.path.isabs(args.zip_out) else args.zip_out
    store = UtteranceStore(zip_abs)

    pairs = []
    print(f"[ext] positives (target {args.n_pos}) ...")
    pairs += make_pos_pairs(word_to_utts, args.n_pos, store, rng,
                            args.max_per_word_pos)
    print(f"  → {len(pairs)}")
    n0 = len(pairs)
    print(f"[ext] hard negatives (target {args.n_hard}) ...")
    pairs += make_hard_neg_pairs(word_to_utts, args.n_hard, store, rng,
                                 args.max_abs_dist)
    print(f"  → {len(pairs) - n0}")
    n0 = len(pairs)
    if args.n_easy > 0:
        print(f"[ext] easy negatives (target {args.n_easy}) ...")
        pairs += make_easy_neg_pairs(word_to_utts, args.n_easy, store, rng)
        print(f"  → {len(pairs) - n0}")
    store.close()
    if store.n_bad:
        print(f"[ext] WARNING: {store.n_bad} unreadable audio files skipped")

    rng.shuffle(pairs)
    out = []
    for k, (uid1, uid2, w1, w2, label) in enumerate(pairs):
        out.append({
            "id": f"extp_{k:06d}",
            "enroll_id": uid1, "query_id": uid2,
            "enroll_txt": w1, "query_txt": w2,
            "label": label,
            "zip": args.zip_out,
        })
    out_abs = os.path.join(PATHS.root, args.out) \
        if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    n_pos = sum(1 for p in out if p["label"] == 1)
    print(f"[ext] wrote {len(out)} pairs ({n_pos} pos / {len(out)-n_pos} neg), "
          f"{len(store.path2uid)} utterances")
    print(f"  pairs → {out_abs}")
    print(f"  wavs  → {zip_abs} ({os.path.getsize(zip_abs)/1e6:.0f} MB)")
    print(f"[ext] done. train_dual.py picks up '{args.out}' automatically.")


if __name__ == "__main__":
    main()

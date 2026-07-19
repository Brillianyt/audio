"""Multithreaded LibriSpeech pair generator."""
import argparse, io, json, os, re, zipfile, time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import soundfile as sf
import torchaudio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def scan(root_dir):
    w2u = defaultdict(list)
    for spk in sorted(os.listdir(root_dir)):
        sd = os.path.join(root_dir, spk)
        if not os.path.isdir(sd): continue
        for ch in sorted(os.listdir(sd)):
            cd = os.path.join(sd, ch)
            if not os.path.isdir(cd): continue
            tp = os.path.join(cd, f"{spk}-{ch}.trans.txt")
            if not os.path.isfile(tp): continue
            with open(tp, encoding="utf-8") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln: continue
                    pts = ln.split(" ", 1)
                    if len(pts) != 2: continue
                    uid, text = pts
                    fp = os.path.join(cd, f"{uid}.flac")
                    if not os.path.isfile(fp): continue
                    for w in set(re.findall(r"[a-z']+", text.lower())):
                        if len(w) >= 2:
                            w2u[w].append((uid, spk, fp))
    print(f"[LS] {sum(len(v) for v in w2u.values())} utts, {len(w2u)} words", flush=True)
    return w2u

def convert_flac(path, sr=16000, max_s=3):
    """Convert flac to wav bytes."""
    wav, fs = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(1)
    if fs != sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), fs, sr).squeeze(0).numpy()
    wav = wav.astype("float32")[:int(max_s * sr)]
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--libri", default="librispeech_data/LibriSpeech/train-clean-100")
    p.add_argument("--zip", default="train/librispeech_wav.zip")
    p.add_argument("--json", default="train/librispeech_pairs.json")
    p.add_argument("--n-pos", type=int, default=15000)
    p.add_argument("--n-neg", type=int, default=30000)
    p.add_argument("--min", type=int, default=3)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    t0 = time.time()
    libri_dir = os.path.join(ROOT, args.libri)
    w2u = scan(libri_dir)
    print(f"[LS] scan: {time.time()-t0:.0f}s", flush=True)

    # Filter qualified words
    valid = {}
    for w, utts in w2u.items():
        spks = set(u[1] for u in utts)
        if len(utts) >= args.min and len(spks) >= 2:
            valid[w] = utts
    words = list(valid.keys())
    print(f"[LS] {len(words)} qualified words, {args.workers} workers", flush=True)

    # Determine which flac files we need (no conversion yet)
    rng = np.random.default_rng(42)
    pos_entries = []  # [(w, flac1, flac2)]
    used_uids = set()

    for w in words:
        if len(pos_entries) >= args.n_pos: break
        utts = valid[w]
        by_spk = defaultdict(list)
        for uid, spk, fp in utts:
            by_spk[spk].append(fp)
        if len(by_spk) < 2: continue
        s1, s2 = rng.choice(list(by_spk.keys()), 2, replace=False)
        fp1 = by_spk[s1][rng.integers(len(by_spk[s1]))]
        fp2 = by_spk[s2][rng.integers(len(by_spk[s2]))]
        pos_entries.append((w, fp1, fp2))
    print(f"[LS] {len(pos_entries)} pos entries planned", flush=True)

    # Negative entries
    neg_entries = []  # [(w1, w2, flac1, flac2)]
    used_pairs = set()
    for _ in range(args.n_neg * 5):
        if len(neg_entries) >= args.n_neg: break
        w1, w2 = words[rng.integers(0, len(words))], words[rng.integers(0, len(words))]
        if w1 == w2: continue
        u1, u2 = valid[w1][rng.integers(len(valid[w1]))], valid[w2][rng.integers(len(valid[w2]))]
        if u1[1] == u2[1]: continue
        pk = tuple(sorted([u1[0], u2[0]]))
        if pk in used_pairs: continue
        used_pairs.add(pk)
        neg_entries.append((w1, w2, u1[2], u2[2]))
    print(f"[LS] {len(neg_entries)} neg entries planned", flush=True)

    all_paths = set()
    for _, fp1, fp2 in pos_entries:
        all_paths.add(fp1); all_paths.add(fp2)
    for _, _, fp1, fp2 in neg_entries:
        all_paths.add(fp1); all_paths.add(fp2)
    total_unique = len(all_paths)
    print(f"[LS] {total_unique} unique flac files to convert", flush=True)

    # Convert all flac files in parallel
    path_list = list(all_paths)
    wav_data = {}  # path -> wav bytes
    conv_t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        fut_to_path = {exe.submit(convert_flac, p): p for p in path_list}
        done = 0
        for fut in as_completed(fut_to_path):
            p = fut_to_path[fut]
            wav_data[p] = fut.result()
            done += 1
            if done % 1000 == 0:
                print(f"  convert {done}/{total_unique} ({time.time()-conv_t0:.0f}s)", flush=True)
    print(f"[LS] all converted: {time.time()-conv_t0:.0f}s", flush=True)

    # Write to zip (sequential)
    zf = zipfile.ZipFile(os.path.join(ROOT, args.zip), "w", zipfile.ZIP_STORED)
    uid_cnt = [0]
    def add_wav(data):
        uid = f"ls_{uid_cnt[0]:06d}"
        zf.writestr(f"wav/{uid}.wav", data)
        uid_cnt[0] += 1
        return uid

    pairs = []
    for w, fp1, fp2 in pos_entries:
        uid1, uid2 = add_wav(wav_data[fp1]), add_wav(wav_data[fp2])
        pairs.append({"id": f"lspos_{len(pairs):06d}", "enroll_id": uid1, "query_id": uid2,
                      "enroll_txt": w, "query_txt": w, "label": 1, "zip": args.zip})
    print(f"[LS] {len(pairs)} pos written", flush=True)

    for w1, w2, fp1, fp2 in neg_entries:
        uid1, uid2 = add_wav(wav_data[fp1]), add_wav(wav_data[fp2])
        pairs.append({"id": f"lsneg_{len(pairs)-len(pos_entries):06d}", "enroll_id": uid1, "query_id": uid2,
                      "enroll_txt": w1, "query_txt": w2, "label": 0, "zip": args.zip})

    zf.close()
    t = time.time()
    print(f"[LS] zip: {os.path.getsize(os.path.join(ROOT, args.zip))/1e9:.1f}GB", flush=True)
    with open(os.path.join(ROOT, args.json), "w") as f:
        json.dump(pairs, f)
    print(f"[LS] {len(pairs)} pairs → {args.json}  total: {t-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()

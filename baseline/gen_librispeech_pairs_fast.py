"""Fast LibriSpeech pair generator — skip O(n²) similarity, just pos + easy neg."""
import argparse, csv, io, json, os, re, zipfile, time
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def scan_librispeech(root_dir):
    word_to_utts = defaultdict(list)
    total = 0
    for spk in sorted(os.listdir(root_dir)):
        spk_dir = os.path.join(root_dir, spk)
        if not os.path.isdir(spk_dir): continue
        for ch in sorted(os.listdir(spk_dir)):
            ch_dir = os.path.join(spk_dir, ch)
            if not os.path.isdir(ch_dir): continue
            trans_path = os.path.join(ch_dir, f"{spk}-{ch}.trans.txt")
            if not os.path.isfile(trans_path): continue
            with open(trans_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(" ", 1)
                    if len(parts) != 2: continue
                    uid, text = parts
                    flac_path = os.path.join(ch_dir, f"{uid}.flac")
                    if not os.path.isfile(flac_path): continue
                    words = set(re.findall(r"[a-z']+", text.lower()))
                    for w in words:
                        if len(w) >= 2:
                            word_to_utts[w].append((uid, spk, flac_path))
                    total += 1
    print(f"[LS] {total} utts, {len(word_to_utts)} unique words")
    return word_to_utts

def convert_flac(flac_path, target_sr=16000, max_sec=3):
    wav, sr = sf.read(flac_path, dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = torchaudio.functional.resample(
            torch.from_numpy(wav).unsqueeze(0), sr, target_sr
        ).squeeze(0).numpy()
    wav = wav.astype(np.float32)
    max_len = max_sec * target_sr
    if len(wav) > max_len: wav = wav[:max_len]
    return wav

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--libri-dir", default="librispeech_data/LibriSpeech/train-clean-100")
    p.add_argument("--zip-out", default="train/librispeech_wav.zip")
    p.add_argument("--json-out", default="train/librispeech_pairs.json")
    p.add_argument("--n-pos", type=int, default=30000)
    p.add_argument("--n-easy", type=int, default=30000)
    p.add_argument("--min-occur", type=int, default=3)
    args = p.parse_args()

    t0 = time.time()
    libri_dir = os.path.join(ROOT, args.libri_dir)
    word_to_utts = scan_librispeech(libri_dir)
    print(f"[LS] scan: {time.time()-t0:.0f}s", flush=True)

    # Filter words with enough occurrences AND multiple speakers
    valid = {}
    for w, utts in word_to_utts.items():
        speakers = set(u[1] for u in utts)
        if len(utts) >= args.min_occur and len(speakers) >= 2:
            valid[w] = utts
    words = sorted(valid.keys())
    print(f"[LS] {len(words)} words with >= {args.min_occur} utts and >= 2 speakers", flush=True)

    rng = np.random.default_rng(42)
    zip_path = os.path.join(ROOT, args.zip_out)
    json_path = os.path.join(ROOT, args.json_out)
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    uid_counter = [0]
    def add_audio(wav):
        buf = io.BytesIO()
        sf.write(buf, wav, 16000, format="WAV", subtype="FLOAT")
        uid = f"ls_{uid_counter[0]:06d}"
        zf.writestr(f"wav/{uid}.wav", buf.getvalue())
        uid_counter[0] += 1
        return uid

    pairs = []
    used_keys = set()
    conv_time = 0.0

    # ── Positive pairs: same word, different speakers ──
    print(f"[LS] generating {args.n_pos} positive pairs...", flush=True)
    words_shuffled = list(words)
    rng.shuffle(words_shuffled)
    n_pos = 0
    for w in words_shuffled:
        if n_pos >= args.n_pos: break
        utts = valid[w]
        by_spk = defaultdict(list)
        for uid, spk, path in utts:
            by_spk[spk].append((uid, path))
        spk_list = list(by_spk.keys())
        if len(spk_list) < 2: continue
        s1, s2 = rng.choice(spk_list, 2, replace=False)
        u1 = by_spk[s1][rng.integers(len(by_spk[s1]))]
        u2 = by_spk[s2][rng.integers(len(by_spk[s2]))]
        key = tuple(sorted([u1[0], u2[0]]))
        if key in used_keys: continue
        used_keys.add(key)

        tc = time.time()
        wav1 = convert_flac(u1[1])  # u1 = (uid, path) → index 1 is path
        wav2 = convert_flac(u2[1])  # same
        conv_time += time.time() - tc
        uid1, uid2 = add_audio(wav1), add_audio(wav2)
        pairs.append({
            "id": f"lspos_{n_pos:06d}",
            "enroll_id": uid1, "query_id": uid2,
            "enroll_txt": w, "query_txt": w,
            "label": 1, "zip": args.zip_out,
        })
        n_pos += 1
        if n_pos % 5000 == 0:
            print(f"  pos {n_pos}/{args.n_pos} ({conv_time:.0f}s)", flush=True)
    print(f"  done: {n_pos} pos", flush=True)

    # ── Easy negative pairs: different words, different speakers ──
    print(f"[LS] generating {args.n_easy} easy negative pairs...", flush=True)
    n_easy = 0
    words_arr = list(words)
    for _ in range(args.n_easy * 3):
        if n_easy >= args.n_easy: break
        w1, w2 = rng.choice(words_arr, 2, replace=False)
        if w1 == w2: continue
        u1 = valid[w1][rng.integers(len(valid[w1]))]
        u2 = valid[w2][rng.integers(len(valid[w2]))]
        # u = (uid, speaker, flac_path) → index 2 is path
        if u1[1] == u2[1]: continue
        key = tuple(sorted([u1[0], u2[0]]))
        if key in used_keys: continue
        used_keys.add(key)

        tc = time.time()
        wav1 = convert_flac(u1[2])
        wav2 = convert_flac(u2[2])
        conv_time += time.time() - tc
        uid1, uid2 = add_audio(wav1), add_audio(wav2)
        pairs.append({
            "id": f"lseasy_{n_easy:06d}",
            "enroll_id": uid1, "query_id": uid2,
            "enroll_txt": w1, "query_txt": w2,
            "label": 0, "zip": args.zip_out,
        })
        n_easy += 1
        if n_easy % 5000 == 0:
            print(f"  easy {n_easy}/{args.n_easy} ({conv_time:.0f}s)", flush=True)
    print(f"  done: {n_easy} easy", flush=True)

    zf.close()
    print(f"[LS] conversion total: {conv_time:.0f}s", flush=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    print(f"[LS] {len(pairs)} pairs → {json_path}", flush=True)
    print(f"[LS] zip: {zip_path} ({os.path.getsize(zip_path)/1e6:.0f} MB)", flush=True)
    print(f"[LS] total time: {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()

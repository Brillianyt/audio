"""Generate pairs from LibriSpeech train-clean-100.

LibriSpeech has structure:
  LibriSpeech/train-clean-100/{speaker}/{chapter}/{speaker}-{chapter}-{utterance}.flac
  LibriSpeech/train-clean-100/{speaker}/{chapter}/{speaker}-{chapter}.trans.txt

trans.txt format: SPEAKER-CHAPTER-UTTERANCE TEXT
We extract individual words from transcriptions and create pairs.
"""
import argparse, io, json, os, re, zipfile
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torchaudio

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def parse_transcript(trans_path):
    """Parse a .trans.txt file, return list of (utterance_id, text)."""
    utts = []
    with open(trans_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(" ", 1)
            if len(parts) == 2:
                uid, text = parts
                text = text.lower().strip(".,!?;:\"'")
                # Simple word tokenization
                words = re.findall(r"[a-z']+", text)
                if words:
                    utts.append((uid, words, text))
    return utts

def scan_librispeech(root_dir, min_word_occurrences=3):
    """Scan LibriSpeech, group utterances by word."""
    word_to_utts = defaultdict(list)  # word -> [(uid, speaker, flac_path)]
    speaker_dirs = sorted(os.listdir(root_dir))
    total_utts = 0
    
    for spk in speaker_dirs:
        spk_dir = os.path.join(root_dir, spk)
        if not os.path.isdir(spk_dir): continue
        for ch in sorted(os.listdir(spk_dir)):
            ch_dir = os.path.join(spk_dir, ch)
            if not os.path.isdir(ch_dir): continue
            
            # Find trans file
            trans_path = os.path.join(ch_dir, f"{spk}-{ch}.trans.txt")
            if not os.path.isfile(trans_path):
                continue
            
            utts = parse_transcript(trans_path)
            for uid, words, text in utts:
                flac_path = os.path.join(ch_dir, f"{uid}.flac")
                if not os.path.isfile(flac_path):
                    continue
                for w in set(words):  # Use set to avoid double-counting same word in one utterance
                    if len(w) >= 2:  # Skip single chars
                        word_to_utts[w].append((uid, spk, flac_path))
                total_utts += 1
    
    # Filter words with enough occurrences
    filtered = {w: utts for w, utts in word_to_utts.items() 
                if len(utts) >= min_word_occurrences}
    print(f"[LibriSpeech] {total_utts} utterances, "
          f"{len(word_to_utts)} unique words, "
          f"{len(filtered)} words with >= {min_word_occurrences} occurrences")
    return filtered

def convert_flac_to_wav(flac_path, target_sr=16000, max_sec=3):
    """Read flac, resample to 16kHz, return tensor."""
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

def generate_pairs(word_to_utts, out_zip, out_json, n_pos=50000, n_hard=50000, seed=42):
    """Generate positive and hard negative pairs."""
    rng = np.random.default_rng(seed)
    words = sorted(word_to_utts.keys())
    
    # Build phoneme distances for hard negatives
    # Use character edit distance as a proxy (CMU dict might not cover all LibriSpeech words)
    def char_ed(a, b):
        m, n = len(a), len(b)
        dp = list(range(n+1))
        for i in range(1, m+1):
            prev, dp[0] = dp[0], i
            for j in range(1, n+1):
                prev, dp[j] = dp[j], min(dp[j]+1, dp[j-1]+1, prev + (0 if a[i-1]==b[j-1] else 1))
        return dp[n] / max(m, n)
    
    # Find similar word pairs (edit distance <= 0.4)
    similar = []
    for i, w1 in enumerate(words):
        for w2 in words[i+1:]:
            d = char_ed(w1, w2)
            if 0 < d <= 0.4:
                similar.append((w1, w2, d))
    similar.sort(key=lambda x: x[2])
    print(f"  {len(similar)} phonetically similar word pairs")
    for w1, w2, d in similar[:10]:
        print(f"    {w1:20s} <-> {w2:20s}  d={d:.3f}")
    
    # Open zip for writing
    os.makedirs(os.path.dirname(out_zip), exist_ok=True)
    zf = zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED)
    
    uid_counter = [0]
    def add_audio(wav):
        """Add audio to zip, return uid."""
        buf = io.BytesIO()
        sf.write(buf, wav, 16000, format="WAV", subtype="FLOAT")
        uid = f"ls_{uid_counter[0]:06d}"
        zf.writestr(f"wav/{uid}.wav", buf.getvalue())
        uid_counter[0] += 1
        return uid
    
    pairs = []
    used = set()
    
    # Positive pairs: same word, different speakers
    pos_candidates = [(w, utts) for w, utts in word_to_utts.items() 
                      if len(set(u[1] for u in utts)) >= 2]  # >=2 speakers
    rng.shuffle(pos_candidates)
    
    n_pos_gen = 0
    for w, utts in pos_candidates:
        if n_pos_gen >= n_pos:
            break
        # Group by speaker
        by_spk = defaultdict(list)
        for uid, spk, path in utts:
            by_spk[spk].append((uid, path))
        speakers = list(by_spk.keys())
        if len(speakers) < 2:
            continue
        # Pick 2 different speakers
        s1, s2 = rng.choice(speakers, 2, replace=False)
        u1 = by_spk[s1][rng.integers(len(by_spk[s1]))]
        u2 = by_spk[s2][rng.integers(len(by_spk[s2]))]
        key = tuple(sorted([u1[0], u2[0]]))
        if key in used: continue
        used.add(key)
        
        wav1 = convert_flac_to_wav(u1[1])
        wav2 = convert_flac_to_wav(u2[1])
        uid1, uid2 = add_audio(wav1), add_audio(wav2)
        pairs.append({
            "id": f"lspos_{n_pos_gen:06d}",
            "enroll_id": uid1, "query_id": uid2,
            "enroll_txt": w, "query_txt": w,
            "label": 1, "zip": out_zip,
        })
        n_pos_gen += 1
    print(f"  positive: {n_pos_gen}")
    
    # Hard negative pairs: phonetically similar, different speakers
    n_hard_gen = 0
    for w1, w2, d in similar:
        if n_hard_gen >= n_hard:
            break
        u1_list = word_to_utts.get(w1, [])
        u2_list = word_to_utts.get(w2, [])
        if not u1_list or not u2_list: continue
        
        # Prefer different speakers
        for _ in range(20):
            u1 = u1_list[rng.integers(len(u1_list))]
            u2 = u2_list[rng.integers(len(u2_list))]
            if u1[1] != u2[1]:  # Different speakers
                break
        
        key = tuple(sorted([u1[0], u2[0]]))
        if key in used: continue
        used.add(key)
        
        wav1 = convert_flac_to_wav(u1[1])
        wav2 = convert_flac_to_wav(u2[1])
        uid1, uid2 = add_audio(wav1), add_audio(wav2)
        pairs.append({
            "id": f"lshard_{n_hard_gen:06d}",
            "enroll_id": uid1, "query_id": uid2,
            "enroll_txt": w1, "query_txt": w2,
            "label": 0, "phoneme_dist": round(d, 4),
            "zip": out_zip,
        })
        n_hard_gen += 1
    print(f"  hard neg: {n_hard_gen}")
    
    zf.close()
    
    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False)
    
    print(f"[LibriSpeech] {len(pairs)} pairs → {out_json}")
    print(f"  zip: {out_zip} ({os.path.getsize(out_zip)/1e6:.0f} MB)")
    return pairs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--libri-dir", default="librispeech_data/LibriSpeech/train-clean-100")
    p.add_argument("--zip-out", default="train/librispeech_wav.zip")
    p.add_argument("--json-out", default="train/librispeech_pairs.json")
    p.add_argument("--n-pos", type=int, default=50000)
    p.add_argument("--n-hard", type=int, default=50000)
    p.add_argument("--min-occur", type=int, default=3)
    args = p.parse_args()
    
    libri_dir = os.path.join(ROOT, args.libri_dir)
    word_to_utts = scan_librispeech(libri_dir, args.min_occur)
    
    zip_out = os.path.join(ROOT, args.zip_out)
    json_out = os.path.join(ROOT, args.json_out)
    generate_pairs(word_to_utts, zip_out, json_out, args.n_pos, args.n_hard)

"""True OHEM — run model on actual query↔candidate pairs, find false positives/negatives.
For AT: cos(audio_emb(query), text_emb(candidate)) — the model's actual decision.
For AA: cos(audio_emb(query), audio_emb(enroll)) — the model's actual decision.
"""
import argparse, csv, gc, json, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_at_model(ckpt_path):
    from train_at_v3 import AudioTextModel
    model = AudioTextModel(256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"  AT loaded: unseen={ckpt.get('auc_unseen',-1):.4f}")
    return model

def load_aa_model(ckpt_path):
    from train_aa import AudioAudioModel
    model = AudioAudioModel(256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"  AA loaded: seen={ckpt.get('auc_seen',-1):.4f}")
    return model

def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f): rows.append(r)
    return rows

class AudioReader:
    def __init__(self, cfg): self.cfg = cfg; self._zc = {}
    def _get_zip(self, zpath):
        pid = os.getpid()
        if pid not in self._zc: self._zc[pid] = {}
        if zpath not in self._zc[pid]: self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]
    def read(self, pid, zpath):
        zf = self._get_zip(zpath)
        for name in [f"wav/{pid}_enroll.wav", f"wav/{pid}_query.wav", f"wav/{pid}.wav"]:
            try: data = zf.read(name); break
            except KeyError: continue
        else: return None
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()

def build_word_audio_map(cfg):
    all_pairs = load_pairs(cfg.train_csv)
    extra_dirs = ["baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
                  "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
                  "train/self_paired.json", "train/self_paired_xl.json",
                  "train/fill_pos_pairs.json", "train/mega_pairs.json",
                  "train/speech_commands_pairs.json", "baseline/hard_neg_atv2.json",
                  "train/librispeech_pairs.json"]
    for fn in extra_dirs:
        fp = os.path.join(PATHS.root, fn)
        if os.path.isfile(fp):
            data = json.load(open(fp)); all_pairs += data
    word_audio = defaultdict(set)
    for p in all_pairs:
        if p["label"] == 1:
            word = p["enroll_txt"].lower()
            for aid in [p.get("enroll_id", p["id"]), p.get("query_id", p["id"])]:
                word_audio[word].add(aid)
    return {w: list(aids) for w, aids in word_audio.items()}


def mine_at_ohem(at_model, word_audio_map, reader, cfg, samples_per_word=3, top_k=20):
    """
    True OHEM for AT:
    For each word, take samples_per_word query audios.
    Run model: cos(query_audio_emb, text_emb(candidate)) for ALL candidate words.
    Collect: false positives (high score, wrong word), false negatives (low score, correct word).
    """
    print("[mine-AT-OHEM] encoding text for all words...")
    words = sorted(word_audio_map.keys())
    n_words = len(words)

    # 1. Encode text for all words (done once)
    T = np.zeros((n_words, 256), dtype=np.float32)
    with torch.no_grad():
        bs = 256
        for i in range(0, n_words, bs):
            batch = words[i:i+bs]
            et, _ = at_model.text_enc(batch)
            T[i:i+len(batch)] = et.cpu().numpy()
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    print(f"  text: {n_words} words encoded")

    # 2. Per word: encode audio samples, compute scores against all text
    false_positives = []  # (query_word, candidate_word, score) — high score, wrong word
    false_negatives = []  # (word, score) — low score, correct word
    seen_pairs = set()

    print(f"[mine-AT-OHEM] running model on {n_words} words × {samples_per_word} queries...")
    with torch.no_grad():
        for wi, w in enumerate(words):
            aids = word_audio_map[w]
            if len(aids) < 1: continue
            sample_aids = np.random.choice(aids, min(samples_per_word, len(aids)), replace=False)

            for aid in sample_aids:
                wav = reader.read(aid, cfg.train_zip)
                if wav is None: continue
                ms = int(cfg.max_audio_sec * 16000)
                wav = F.pad(wav, (0, max(0, ms - len(wav))))[:ms].unsqueeze(0).to(device)
                q_emb = at_model.encoder(wav).cpu().numpy().flatten()
                q_emb = q_emb / np.linalg.norm(q_emb)

                # Model's actual decision: cos(query_audio_emb, text_emb(candidate))
                scores = q_emb @ T.T  # (n_words,)

                # False negative: same word, low score
                self_score = scores[wi]
                if self_score < 0.3:
                    false_negatives.append({"word": w, "score": float(self_score), "aid": aid})

                # False positives: different words, high score
                fp_idx = np.argsort(-scores)
                for j in fp_idx:
                    if j == wi: continue
                    if scores[j] < 0.4:  # threshold: model thinks it's a match
                        break
                    other = words[j]
                    pair_key = tuple(sorted([w, other]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        false_positives.append({
                            "id": f"hnat_{len(false_positives):06d}",
                            "enroll_txt": w, "query_txt": other,
                            "label": 0, "score": float(scores[j]),
                            "type": "at_ohem_fp", "query_aid": aid
                        })
                        if len([fp for fp in false_positives if fp["enroll_txt"] == w]) >= top_k:
                            break

            if (wi+1) % 1000 == 0:
                print(f"  AT-OHEM {wi+1}/{n_words}")

    false_positives.sort(key=lambda x: -x["score"])
    false_negatives.sort(key=lambda x: x["score"])
    print(f"[mine-AT-OHEM] {len(false_positives)} hard neg (score≥0.4)")
    print(f"[mine-AT-OHEM] {len(false_negatives)} hard pos (self-score<0.3)")
    return false_positives, false_negatives


def mine_aa_ohem(aa_model, word_audio_map, reader, cfg, samples_per_word=3, top_k=20):
    """
    True OHEM for AA:
    For each word, take samples_per_word query audios.
    Run model: cos(query_audio_emb, enroll_audio_emb(candidate)) for ALL candidates.
    """
    print("[mine-AA-OHEM] encoding audio centroids for all words...")
    words = sorted(word_audio_map.keys())
    n_words = len(words)

    # Encode centroid per word (use mean of up to 5 audios)
    A = np.zeros((n_words, 256), dtype=np.float32)
    with torch.no_grad():
        for wi, w in enumerate(words):
            aids = word_audio_map[w][:5]
            embs = []
            for aid in aids:
                wav = reader.read(aid, cfg.train_zip)
                if wav is None: continue
                ms = int(cfg.max_audio_sec * 16000)
                wav = F.pad(wav, (0, max(0, ms - len(wav))))[:ms].unsqueeze(0).to(device)
                emb = aa_model.encoder(wav).cpu().numpy().flatten()
                embs.append(emb)
            if embs: A[wi] = np.mean(embs, axis=0)
            if (wi+1) % 2000 == 0:
                print(f"  AA centroids {wi+1}/{n_words}")
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    print(f"  audio centroids: {n_words} words encoded")

    # Per word: encode query samples, compute against all centroids
    false_positives = []
    false_negatives = []
    seen_pairs = set()

    print(f"[mine-AA-OHEM] running model on {n_words} words × {samples_per_word} queries...")
    with torch.no_grad():
        for wi, w in enumerate(words):
            aids = word_audio_map[w]
            if len(aids) < 2: continue
            sample_aids = np.random.choice(aids, min(samples_per_word, len(aids)), replace=False)

            for aid in sample_aids:
                wav = reader.read(aid, cfg.train_zip)
                if wav is None: continue
                ms = int(cfg.max_audio_sec * 16000)
                wav = F.pad(wav, (0, max(0, ms - len(wav))))[:ms].unsqueeze(0).to(device)
                q_emb = aa_model.encoder(wav).cpu().numpy().flatten()
                q_emb = q_emb / np.linalg.norm(q_emb)

                scores = q_emb @ A.T  # (n_words,)

                # False negative
                self_score = scores[wi]
                if self_score < 0.3:
                    false_negatives.append({"word": w, "score": float(self_score), "aid": aid})

                # False positives
                fp_idx = np.argsort(-scores)
                for j in fp_idx:
                    if j == wi: continue
                    if scores[j] < 0.5:
                        break
                    other = words[j]
                    pair_key = tuple(sorted([w, other]))
                    if pair_key not in seen_pairs:
                        seen_pairs.add(pair_key)
                        false_positives.append({
                            "id": f"hnaa_{len(false_positives):06d}",
                            "enroll_txt": w, "query_txt": other,
                            "label": 0, "score": float(scores[j]),
                            "type": "aa_ohem_fp", "query_aid": aid
                        })
                        if len([fp for fp in false_positives if fp["enroll_txt"] == w]) >= top_k:
                            break

            if (wi+1) % 1000 == 0:
                print(f"  AA-OHEM {wi+1}/{n_words}")

    false_positives.sort(key=lambda x: -x["score"])
    false_negatives.sort(key=lambda x: x["score"])
    print(f"[mine-AA-OHEM] {len(false_positives)} hard neg (score≥0.5)")
    print(f"[mine-AA-OHEM] {len(false_negatives)} hard pos (self-score<0.3)")
    return false_positives, false_negatives


class Config:
    sample_rate: int = 16000
    max_audio_sec: float = 1.5
    def __init__(self):
        r = PATHS.root
        self.train_zip = os.path.join(r, "train", "wav.zip")
        self.train_csv = os.path.join(r, "train", "train_label.csv")
        if os.path.isfile(os.path.join(r, "train_subset", "wav.zip")):
            self.train_zip = os.path.join(r, "train_subset", "wav.zip")
            self.train_csv = os.path.join(r, "train_subset", "train_label.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--at-ckpt", default="output/backup/at_best.pt")
    p.add_argument("--aa-ckpt", default="output/aa_pk/aa_final.pt")
    p.add_argument("--out-dir", default="baseline")
    p.add_argument("--samples", type=int, default=3, help="query samples per word")
    p.add_argument("--at-only", action="store_true", help="only mine AT, skip AA")
    args = p.parse_args()

    cfg = Config()
    print("Loading models...")
    at_model = load_at_model(args.at_ckpt)
    aa_model = load_aa_model(args.aa_ckpt)
    reader = AudioReader(cfg)

    print("Building word-audio map...")
    word_audio_map = build_word_audio_map(cfg)
    print(f"  {len(word_audio_map)} words")

    # True OHEM — model scores every pair
    at_fp, at_fn = mine_at_ohem(at_model, word_audio_map, reader, cfg, samples_per_word=args.samples)

    out_dir = os.path.join(PATHS.root, args.out_dir)
    # Save AT results immediately (don't wait for AA)
    json.dump(at_fp, open(os.path.join(out_dir, "hard_neg_at_ohem.json"), "w"), indent=2)
    json.dump(at_fn, open(os.path.join(out_dir, "hard_pos_at_ohem.json"), "w"), indent=2)
    print(f"  saved hard_neg_at_ohem.json ({len(at_fp)} pairs)")
    print(f"  saved hard_pos_at_ohem.json ({len(at_fn)} words)")

    if not args.at_only:
        aa_fp, aa_fn = mine_aa_ohem(aa_model, word_audio_map, reader, cfg, samples_per_word=args.samples)
        json.dump(aa_fp, open(os.path.join(out_dir, "hard_neg_aa_ohem.json"), "w"), indent=2)
        json.dump(aa_fn, open(os.path.join(out_dir, "hard_pos_aa_ohem.json"), "w"), indent=2)

    print(f"\n=== True OHEM Results ===")
    print(f"  AT false positives: {len(at_fp)}")
    print(f"  AT false negatives: {len(at_fn)}")
    if at_fp:
        top5 = [(p["enroll_txt"], p["query_txt"], f"{p['score']:.3f}") for p in at_fp[:5]]
        print(f"  AT top-5 FP: {top5}")

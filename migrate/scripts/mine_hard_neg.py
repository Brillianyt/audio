"""Hard negative mining using trained AT and AA models.
For each word, find the top-K most confusable different words (highest cosine similarity).
"""
import argparse, csv, gc, json, os, sys, time, io, zipfile
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

device = "cuda" if torch.cuda.is_available() else "cpu"

# ── Load models ──
def load_at_model(ckpt_path):
    from train_at_v3 import AudioTextModel, WhisperEncoder, PhonemeBiGRUEncoder, ComparisonHead
    model = AudioTextModel(256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"  AT loaded: unseen={ckpt.get('auc_unseen',-1):.4f}")
    return model

def load_aa_model(ckpt_path):
    from train_aa_pk import AAPKModel, WhisperEncoder
    model = AAPKModel(256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"  AA loaded: seen={ckpt.get('auc_seen',-1):.4f}")
    return model

# ── Data loading ──
def load_pairs(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt",""),
                         "query_txt": r.get("query_txt","")})
    return rows

class AudioReader:
    def __init__(self, cfg):
        self.cfg = cfg
        self._zc = {}

    def _get_zip(self, zpath):
        pid = os.getpid()
        if pid not in self._zc:
            self._zc[pid] = {}
        if zpath not in self._zc[pid]:
            self._zc[pid][zpath] = zipfile.ZipFile(zpath, "r")
        return self._zc[pid][zpath]

    def read(self, pid, zpath):
        zf = self._get_zip(zpath)
        for name in [f"wav/{pid}_enroll.wav", f"wav/{pid}_query.wav", f"wav/{pid}.wav"]:
            try:
                data = zf.read(name)
                break
            except KeyError:
                continue
        else:
            return None
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(self.cfg.max_audio_sec * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()

def build_word_audio_map(cfg):
    """{word: [audio_id, ...], ...} from all data."""
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
            data = json.load(open(fp))
            all_pairs += data

    word_audio = defaultdict(set)
    for p in all_pairs:
        if p["label"] == 1:
            word = p["enroll_txt"].lower()
            for aid in [p.get("enroll_id", p["id"]), p.get("query_id", p["id"])]:
                word_audio[word].add(aid)
    return {w: list(aids) for w, aids in word_audio.items()}


def mine_at(at_model, word_audio_map, reader, cfg, top_k=50):
    """Mine hard negatives using AT model: text_emb vs audio_emb cross-modal."""
    print("[mine-AT] encoding words...")
    words = sorted(word_audio_map.keys())
    word_embs = {}  # word → text_embedding

    with torch.no_grad():
        for i, w in enumerate(words):
            et = at_model.text_enc([w])[0].cpu().numpy().flatten()
            word_embs[w] = et
            if (i+1) % 1000 == 0:
                print(f"  text {i+1}/{len(words)}")

    # Compute word-level similarity matrix in chunks
    word_list = list(word_embs.keys())
    W = np.stack([word_embs[w] for w in word_list])  # (N, 256)
    W = W / np.linalg.norm(W, axis=1, keepdims=True)
    sim = W @ W.T  # (N, N) cosine

    # For each word, find top-K most similar different words
    hard_pairs = []
    seen_pairs = set()
    for i, w in enumerate(word_list):
        scores = sim[i]
        scores[i] = -2  # exclude self
        top_idx = np.argsort(-scores)[:top_k]
        for j in top_idx:
            if scores[j] < 0.2:
                break
            other = word_list[j]
            pair_key = tuple(sorted([w, other]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                hard_pairs.append({
                    "id": f"hn_at_{len(hard_pairs):06d}",
                    "enroll_txt": w, "query_txt": other,
                    "label": 0, "similarity": float(scores[j])
                })

    # Sort by similarity descending
    hard_pairs.sort(key=lambda x: -x["similarity"])
    print(f"[mine-AT] {len(hard_pairs)} hard neg pairs (cos≥0.2)")
    return hard_pairs


def mine_aa(aa_model, word_audio_map, reader, cfg, top_k=50):
    """Mine hard negatives using AA model: audio centroid similarity."""
    print("[mine-AA] encoding audio centroids...")
    words = sorted(word_audio_map.keys())
    word_centroids = {}

    with torch.no_grad():
        for i, w in enumerate(words):
            aids = word_audio_map[w][:5]  # use up to 5 audios per word
            embs = []
            for aid in aids:
                zpath = cfg.train_zip
                wav = reader.read(aid, zpath)
                if wav is None:
                    continue
                wav = wav.unsqueeze(0).to(device)
                emb = aa_model(wav)[0].cpu().numpy().flatten()
                embs.append(emb)
            if embs:
                word_centroids[w] = np.mean(embs, axis=0)
            if (i+1) % 1000 == 0:
                print(f"  audio {i+1}/{len(words)}")

    word_list = sorted(word_centroids.keys())
    C = np.stack([word_centroids[w] for w in word_list])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    sim = C @ C.T

    hard_pairs = []
    seen_pairs = set()
    for i, w in enumerate(word_list):
        scores = sim[i]
        scores[i] = -2
        top_idx = np.argsort(-scores)[:top_k]
        for j in top_idx:
            if scores[j] < 0.2:
                break
            other = word_list[j]
            pair_key = tuple(sorted([w, other]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                hard_pairs.append({
                    "id": f"hn_aa_{len(hard_pairs):06d}",
                    "enroll_txt": w, "query_txt": other,
                    "label": 0, "similarity": float(scores[j])
                })

    hard_pairs.sort(key=lambda x: -x["similarity"])
    print(f"[mine-AA] {len(hard_pairs)} hard neg pairs (cos≥0.2)")
    return hard_pairs


# ── Main ──
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

cfg = Config()

print("Loading models...")
at_model = load_at_model("output/at_v5/at_final.pt")
aa_model = load_aa_model("output/aa_pk/aa_final.pt")
reader = AudioReader(cfg)

print("Building word-audio map...")
word_audio_map = build_word_audio_map(cfg)
print(f"  {len(word_audio_map)} words")

# Mine
at_hard = mine_at(at_model, word_audio_map, reader, cfg, top_k=50)
aa_hard = mine_aa(aa_model, word_audio_map, reader, cfg, top_k=50)

# Save
out_dir = os.path.join(PATHS.root, "baseline")
with open(os.path.join(out_dir, "hard_neg_at_final.json"), "w") as f:
    json.dump(at_hard, f, indent=2)
print(f"  saved hard_neg_at_final.json ({len(at_hard)})")

with open(os.path.join(out_dir, "hard_neg_aa_final.json"), "w") as f:
    json.dump(aa_hard, f, indent=2)
print(f"  saved hard_neg_aa_final.json ({len(aa_hard)})")

# Stats
all_pairs = at_hard + aa_hard
unique_words = set()
for p in all_pairs:
    unique_words.add(p["enroll_txt"])
    unique_words.add(p["query_txt"])
print(f"  total unique words in hard neg: {len(unique_words)}")
avg_sim = np.mean([p["similarity"] for p in all_pairs])
print(f"  avg cosine similarity: {avg_sim:.4f}")

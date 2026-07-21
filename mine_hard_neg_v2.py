"""Hard negative mining v2 — model error-driven (OHEM).
Finds where the model actually makes mistakes, not just static embedding neighbors.

Level 1: AT cross-modal — text_emb(word) vs audio_centroid(query) → high-score wrong pairs
Level 2: AA audio-audio — audio centroid similarity → high-score wrong pairs
Level 3: Ensemble disagreement — AT high, AA low (or vice versa)
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
    """Load AT model (cosine-based)."""
    from train_at_v3 import AudioTextModel
    model = AudioTextModel(256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"  AT loaded: unseen={ckpt.get('auc_unseen',-1):.4f}")
    return model

def load_aa_model(ckpt_path):
    """Load AA model."""
    from train_aa import AudioAudioModel
    model = AudioAudioModel(256)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    print(f"  AA loaded: seen={ckpt.get('auc_seen',-1):.4f}")
    return model


# ── Data ──
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
    """{word: [audio_id, ...], ...} from all data sources."""
    all_pairs = load_pairs(cfg.train_csv)
    extra_dirs = [
        "baseline/hard_neg_whisper.json", "baseline/hard_neg_iter1.json",
        "baseline/hard_neg_iter2.json", "baseline/hard_neg_phoneme.json",
        "train/self_paired.json", "train/self_paired_xl.json",
        "train/fill_pos_pairs.json", "train/mega_pairs.json",
        "train/speech_commands_pairs.json", "baseline/hard_neg_atv2.json",
        "train/librispeech_pairs.json",
    ]
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


# ═══════════════════════════════════════════
# Level 1+2: Model error-driven mining
# ═══════════════════════════════════════════

def mine_at_cross_modal(at_model, word_audio_map, reader, cfg, top_k=30):
    """
    AT cross-modal mining:
    For each word, compute text_emb(word).
    For each other word, compute audio centroid.
    Find pairs where cos(text_emb, audio_centroid) is high but words differ.
    → These are the FALSE POSITIVES the model would make.
    """
    print("[mine-AT-cross] encoding words...")
    words = sorted(word_audio_map.keys())
    n_words = len(words)

    # 1. Encode text for all words
    text_embs = {}
    with torch.no_grad():
        bs = 256
        for i in range(0, n_words, bs):
            batch = words[i:i+bs]
            et, _ = at_model.text_enc(batch)
            for j, w in enumerate(batch):
                text_embs[w] = et[j].cpu().numpy()
            if (i+bs) % 2000 == 0:
                print(f"  text {min(i+bs, n_words)}/{n_words}")

    # 2. Encode audio centroids for all words
    print("[mine-AT-cross] encoding audio centroids...")
    audio_centroids = {}
    with torch.no_grad():
        for i, w in enumerate(words):
            aids = word_audio_map[w][:10]  # up to 10 audios
            embs = []
            for aid in aids:
                wav = reader.read(aid, cfg.train_zip)
                if wav is None: continue
                wav_batch = F.pad(wav.unsqueeze(0), (0, max(0, int(cfg.max_audio_sec*16000)-len(wav))))[:,:int(cfg.max_audio_sec*16000)]
                emb = at_model.encoder(wav_batch.to(device)).cpu().numpy().flatten()
                embs.append(emb)
            if embs:
                audio_centroids[w] = np.mean(embs, axis=0)
            if (i+1) % 1000 == 0:
                print(f"  audio {i+1}/{n_words}")

    # 3. Cross-modal similarity: text_emb(word_i) vs audio_centroid(word_j)
    common_words = sorted(set(text_embs.keys()) & set(audio_centroids.keys()))
    print(f"[mine-AT-cross] {len(common_words)} words with both text and audio")
    
    T = np.stack([text_embs[w] for w in common_words])   # (N, D)
    A = np.stack([audio_centroids[w] for w in common_words])  # (N, D)
    T = T / np.linalg.norm(T, axis=1, keepdims=True)
    A = A / np.linalg.norm(A, axis=1, keepdims=True)

    # Cross similarity: text(word_i) vs audio(word_j) — the actual model decision space
    cross_sim = T @ A.T  # (N, N)

    # 4. Collect hard negatives: high cross_sim[i,j] but i != j
    #    Also collect hard positives: low cross_sim[i,i] (same word, low score)
    hard_neg_pairs = []
    hard_pos_words = []
    seen_pairs = set()

    for i, w in enumerate(common_words):
        # Hard positives: same word, low cross-modal score
        self_score = cross_sim[i, i]
        if self_score < 0.3:  # model fails on its own word
            hard_pos_words.append({"word": w, "score": float(self_score)})

        # Hard negatives: different word, high cross-modal score
        scores = cross_sim[i].copy()
        scores[i] = -2  # exclude self
        top_idx = np.argsort(-scores)[:top_k]
        for j in top_idx:
            if scores[j] < 0.15:  # threshold lower than before — model-level, more selective
                break
            other = common_words[j]
            pair_key = tuple(sorted([w, other]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                hard_neg_pairs.append({
                    "id": f"hnat_{len(hard_neg_pairs):06d}",
                    "enroll_txt": w, "query_txt": other,
                    "label": 0,
                    "score": float(scores[j]),
                    "type": "at_cross_fp"
                })

    hard_neg_pairs.sort(key=lambda x: -x["score"])
    hard_pos_words.sort(key=lambda x: x["score"])
    print(f"[mine-AT-cross] {len(hard_neg_pairs)} hard neg (score≥0.15)")
    print(f"[mine-AT-cross] {len(hard_pos_words)} hard pos (self-score<0.3)")
    return hard_neg_pairs, hard_pos_words


def mine_aa_error(aa_model, word_audio_map, reader, cfg, top_k=30):
    """
    AA error-driven mining:
    Compute audio centroids, find cross-word pairs with high cosine similarity.
    This is still embedding-based but now uses actual model encoding.
    Also finds hard positives (same word, low self-similarity across speakers).
    """
    print("[mine-AA] encoding audio centroids...")
    words = sorted(word_audio_map.keys())

    word_centroids = {}
    word_all_embs = {}  # keep individual embeddings for self-variance check
    with torch.no_grad():
        for i, w in enumerate(words):
            aids = word_audio_map[w][:10]
            embs = []
            for aid in aids:
                wav = reader.read(aid, cfg.train_zip)
                if wav is None: continue
                wav_batch = F.pad(wav.unsqueeze(0), (0, max(0, int(cfg.max_audio_sec*16000)-len(wav))))[:,:int(cfg.max_audio_sec*16000)]
                emb = aa_model.encoder(wav_batch.to(device)).cpu().numpy().flatten()
                embs.append(emb)
            if embs:
                word_centroids[w] = np.mean(embs, axis=0)
                word_all_embs[w] = embs
            if (i+1) % 1000 == 0:
                print(f"  audio {i+1}/{len(words)}")

    word_list = sorted(word_centroids.keys())
    C = np.stack([word_centroids[w] for w in word_list])
    C = C / np.linalg.norm(C, axis=1, keepdims=True)
    sim = C @ C.T

    hard_neg_pairs = []
    hard_pos_words = []
    seen_pairs = set()

    for i, w in enumerate(word_list):
        # Hard positives: same word, low cross-speaker similarity
        if w in word_all_embs and len(word_all_embs[w]) >= 2:
            e1 = np.array(word_all_embs[w][0])
            e2 = np.array(word_all_embs[w][1])
            e1 = e1 / np.linalg.norm(e1)
            e2 = e2 / np.linalg.norm(e2)
            cross_spk = float(e1 @ e2)
            if cross_spk < 0.3:
                hard_pos_words.append({"word": w, "cross_speaker_cos": cross_spk})

        # Hard negatives
        scores = sim[i].copy()
        scores[i] = -2
        top_idx = np.argsort(-scores)[:top_k]
        for j in top_idx:
            if scores[j] < 0.2:
                break
            other = word_list[j]
            pair_key = tuple(sorted([w, other]))
            if pair_key not in seen_pairs:
                seen_pairs.add(pair_key)
                hard_neg_pairs.append({
                    "id": f"hnaa_{len(hard_neg_pairs):06d}",
                    "enroll_txt": w, "query_txt": other,
                    "label": 0,
                    "score": float(scores[j]),
                    "type": "aa_fp"
                })

    hard_neg_pairs.sort(key=lambda x: -x["score"])
    hard_pos_words.sort(key=lambda x: x.get("cross_speaker_cos", 0))
    print(f"[mine-AA] {len(hard_neg_pairs)} hard neg (cos≥0.2)")
    print(f"[mine-AA] {len(hard_pos_words)} hard pos (cross-spk<0.3)")
    return hard_neg_pairs, hard_pos_words


# ═══════════════════════════════════════════
# Level 3: Ensemble disagreement mining
# ═══════════════════════════════════════════

def mine_ensemble_disagreement(at_model, aa_model, word_audio_map, reader, cfg, top_k=20):
    """
    Find pairs where AT and AA strongly disagree.
    AT high + AA low → possible unseen word or text-acoustic mismatch
    AT low + AA high → possible seen word with speaker variation
    """
    print("[mine-ensemble] computing dual scores...")
    words = sorted(word_audio_map.keys())
    common_words = []
    at_scores = {}
    aa_scores = {}

    with torch.no_grad():
        for i, w in enumerate(words):
            aids = word_audio_map[w][:3]
            if len(aids) < 2: continue
            # AT score: text(w) vs audio(w) — self-consistency
            et, _ = at_model.text_enc([w])
            # AA score: audio centroid self-similarity
            embs = []
            for aid in aids[:5]:
                wav = reader.read(aid, cfg.train_zip)
                if wav is None: continue
                wav_batch = F.pad(wav.unsqueeze(0), (0, max(0, int(cfg.max_audio_sec*16000)-len(wav))))[:,:int(cfg.max_audio_sec*16000)]
                emb = at_model.encoder(wav_batch.to(device)).cpu().numpy().flatten()
                embs.append(emb)
            if len(embs) < 2: continue
            centroid = np.mean(embs, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            at_score = float((et.cpu().numpy().flatten() / np.linalg.norm(et.cpu().numpy().flatten())) @ centroid)
            aa_embs = [np.array(e) / np.linalg.norm(e) for e in embs]
            aa_score = float(np.mean([aa_embs[0] @ aa_embs[j] for j in range(1, len(aa_embs))]))
            common_words.append(w)
            at_scores[w] = at_score
            aa_scores[w] = aa_score
            if (i+1) % 1000 == 0:
                print(f"  ensemble {i+1}/{len(words)}")

    # Find disagreement cases
    # AT-Agreement > AA-Agreement → unseen-like (text matches but audio varies)
    # AA-Agreement > AT-Agreement → seen-like (audio matches but text unsure)
    disagreements = []
    for w in common_words:
        diff = at_scores[w] - aa_scores[w]
        disagreements.append({"word": w, "at": at_scores[w], "aa": aa_scores[w], "diff": diff})
    disagreements.sort(key=lambda x: -abs(x["diff"]))

    # Top disagreements
    top_unseen_like = [d for d in disagreements if d["diff"] > 0.2][:top_k]
    top_seen_like = [d for d in disagreements if d["diff"] < -0.2][:top_k]

    print(f"[mine-ensemble] {len(top_unseen_like)} unseen-like (AT>AA)")
    print(f"[mine-ensemble] {len(top_seen_like)} seen-like (AA>AT)")
    return top_unseen_like, top_seen_like


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

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
    p.add_argument("--at-ckpt", default="output/at_v5/at_final.pt")
    p.add_argument("--aa-ckpt", default="output/aa_pk/aa_final.pt")
    p.add_argument("--out-dir", default="baseline")
    args = p.parse_args()

    cfg = Config()

    print("Loading models...")
    at_model = load_at_model(args.at_ckpt)
    aa_model = load_aa_model(args.aa_ckpt)
    reader = AudioReader(cfg)

    print("Building word-audio map...")
    word_audio_map = build_word_audio_map(cfg)
    print(f"  {len(word_audio_map)} words")

    # Level 1+2: Model error-driven mining
    at_hard_neg, at_hard_pos = mine_at_cross_modal(at_model, word_audio_map, reader, cfg)
    aa_hard_neg, aa_hard_pos = mine_aa_error(aa_model, word_audio_map, reader, cfg)

    # Level 3: Ensemble disagreement
    unseen_like, seen_like = mine_ensemble_disagreement(at_model, aa_model, word_audio_map, reader, cfg)

    # Save
    out_dir = os.path.join(PATHS.root, args.out_dir)

    # AT hard negatives (cross-modal false positives)
    with open(os.path.join(out_dir, "hard_neg_at_v2.json"), "w") as f:
        json.dump(at_hard_neg, f, indent=2)
    print(f"  saved hard_neg_at_v2.json ({len(at_hard_neg)} pairs)")

    # AA hard negatives
    with open(os.path.join(out_dir, "hard_neg_aa_v2.json"), "w") as f:
        json.dump(aa_hard_neg, f, indent=2)
    print(f"  saved hard_neg_aa_v2.json ({len(aa_hard_neg)} pairs)")

    # Hard positives (model fails on its own words)
    with open(os.path.join(out_dir, "hard_pos_at_v2.json"), "w") as f:
        json.dump(at_hard_pos, f, indent=2)
    print(f"  saved hard_pos_at_v2.json ({len(at_hard_pos)} words)")

    with open(os.path.join(out_dir, "hard_pos_aa_v2.json"), "w") as f:
        json.dump(aa_hard_pos, f, indent=2)
    print(f"  saved hard_pos_aa_v2.json ({len(aa_hard_pos)} words)")

    # Ensemble disagreement
    with open(os.path.join(out_dir, "ensemble_disagreement.json"), "w") as f:
        json.dump({"unseen_like": unseen_like, "seen_like": seen_like}, f, indent=2)
    print(f"  saved ensemble_disagreement.json")

    # Summary
    all_hard_neg = at_hard_neg + aa_hard_neg
    unique_words = set()
    for p in all_hard_neg:
        unique_words.add(p["enroll_txt"])
        unique_words.add(p["query_txt"])
    print(f"\n=== Summary ===")
    print(f"  AT hard neg: {len(at_hard_neg)}")
    print(f"  AA hard neg: {len(aa_hard_neg)}")
    print(f"  Total hard neg: {len(all_hard_neg)}")
    print(f"  AT hard pos: {len(at_hard_pos)}")
    print(f"  AA hard pos: {len(aa_hard_pos)}")
    print(f"  Unseen-like: {len(unseen_like)}, Seen-like: {len(seen_like)}")
    print(f"  Unique words in hard neg: {len(unique_words)}")
    if all_hard_neg:
        top5_at = [(p["enroll_txt"], p["query_txt"], f"{p['score']:.3f}") for p in at_hard_neg[:5]]
        top5_aa = [(p["enroll_txt"], p["query_txt"], f"{p['score']:.3f}") for p in aa_hard_neg[:5]]
        print(f"  AT top-5: {top5_at}")
        print(f"  AA top-5: {top5_aa}")
        print(f"  AT hard pos: {at_hard_pos[:5]}")
        print(f"  AA hard pos: {aa_hard_pos[:5]}")

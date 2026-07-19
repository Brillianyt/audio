"""AT v2 模型级难负例挖掘 — 基于 AudioTextModel 的 embedding 相似度."""
import argparse, csv, io, json, os, sys, time, zipfile
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import PATHS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="output/dual_at_v2_text/best.pt")
    p.add_argument("--csv", default="train/train_label.csv")
    p.add_argument("--zip", default="train/wav.zip")
    p.add_argument("--out", default="baseline/hard_neg_atv2.json")
    p.add_argument("--max-pairs", type=int, default=80000)
    p.add_argument("--topk", type=int, default=15)
    p.add_argument("--cos-thresh", type=float, default=0.2,
                   help="only keep pairs with cosine >= thresh")
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = os.path.join(PATHS.root, args.ckpt)
    csv_path = os.path.join(PATHS.root, args.csv)
    zip_path = os.path.join(PATHS.root, args.zip)
    out_path = os.path.join(PATHS.root, args.out)

    # ── Load model (only encoder — text_enc dim mismatch handled by strict=False) ──
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from train_dual import AudioTextModel, Config
    cfg = Config(); cfg.__post_init__()
    model = AudioTextModel("", cfg.embed_dim).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Only load encoder weights (text_enc dims changed between v1 and v2)
    enc_state = {k.replace("encoder.", ""): v for k, v in ckpt["model"].items()
                 if k.startswith("encoder.")}
    model.encoder.load_state_dict(enc_state, strict=False)
    model.eval()
    print(f"[miner] loaded encoder from {ckpt_path}")

    # ── Build word index ──
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    word_to_pos = defaultdict(list)
    for r in rows:
        if r["enroll_txt"] == r["query_txt"] and r["label"] == "1":
            word_to_pos[r["enroll_txt"].lower()].append((r["id"], r["enroll_txt"]))
    words = sorted(word_to_pos.keys())
    print(f"[miner] {len(words)} words, {sum(len(v) for v in word_to_pos.values())} samples")

    # ── Encode up to 20 audios per word ──
    zf = zipfile.ZipFile(zip_path, "r")
    def read_wav(pid, role):
        data = zf.read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        if len(wav) > 3 * 16000: wav = wav[:3 * 16000]
        return torch.from_numpy(wav).float()

    word_emb = {}
    total = sum(min(20, len(v)) for v in word_to_pos.values())
    done = 0; t0 = time.time()
    for w in words:
        entries = word_to_pos[w][:20]
        embs = []
        for pid, _ in entries:
            wav = read_wav(pid, "enroll")
            with torch.no_grad():
                ea = model.encoder(wav.unsqueeze(0).to(device))
                embs.append(ea.squeeze(0).cpu())
        if embs:
            word_emb[w] = F.normalize(torch.stack(embs).mean(0, keepdim=True), dim=-1).squeeze(0)
        done += len(entries)
        if done % 2000 == 0:
            print(f"  [{done}/{total}] {time.time()-t0:.0f}s")
    zf.close()
    print(f"  encode done {time.time()-t0:.0f}s")

    # ── Pairwise search ──
    words_list = sorted(word_emb.keys())
    W = len(words_list)
    emb_mat = torch.stack([word_emb[w] for w in words_list]).to(device)
    K = min(args.topk, W-1)

    confusable = []
    seen = set()
    chunk = 200
    for i in range(0, W, chunk):
        end = min(i+chunk, W)
        sim = torch.matmul(emb_mat[i:end], emb_mat.T)
        for ci in range(sim.shape[0]):
            gi = i+ci
            row = sim[ci]
            row[gi] = -1.0  # exclude self
            vals, idx = torch.topk(row, K)
            for j, s in zip(idx.tolist(), vals.tolist()):
                if s >= args.cos_thresh:
                    pk = tuple(sorted([words_list[gi], words_list[j]]))
                    if pk not in seen:
                        seen.add(pk)
                        confusable.append((words_list[gi], words_list[j], round(float(s), 4)))

    confusable.sort(key=lambda x: x[2], reverse=True)
    print(f"[miner] {len(confusable)} unique confusable pairs")
    for w1, w2, s in confusable[:20]:
        print(f"  {w1:20s} <-> {w2:20s}  cos={s:.4f}")

    # ── Generate hard negatives ──
    extra = []; used = set()
    rng = np.random.default_rng(42)
    max_per = max(1, args.max_pairs // max(1, len(confusable))) + 5
    combo_used = defaultdict(int)
    for w1, w2, score in confusable:
        if len(extra) >= args.max_pairs:
            break
        pk = tuple(sorted([w1, w2]))
        if combo_used[pk] >= max_per:
            continue
        p1 = word_to_pos.get(w1, []); p2 = word_to_pos.get(w2, [])
        if not p1 or not p2:
            continue
        n = min(5, len(p1), len(p2), max_per - combo_used[pk])
        s1 = rng.choice(len(p1), n, replace=False)
        s2 = rng.choice(len(p2), n, replace=False)
        for si in s1:
            for sj in s2:
                id1, t1 = p1[si]; id2, t2 = p2[sj]
                if (id1, id2) in used:
                    continue
                used.add((id1, id2))
                extra.append({
                    "id": f"hn_atv2_{id1}_x_{id2}",
                    "enroll_id": id1, "query_id": id2,
                    "enroll_txt": t1, "query_txt": t2,
                    "label": 0,
                    "cos_sim": score,
                })
                combo_used[pk] += 1
                if len(extra) >= args.max_pairs:
                    break
            if len(extra) >= args.max_pairs:
                break

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(extra, f, ensure_ascii=False)
    print(f"[miner] {len(extra)} pairs → {out_path}")

if __name__ == "__main__":
    main()

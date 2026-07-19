"""Iterative Hard Negative Mining — per-word top-K nearest neighbors."""
import argparse, csv, io, json, os, sys, time, zipfile
from collections import defaultdict
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.train_whisper_v3 import (
    WhisperTextKWS, WhisperConfigV3, _word_to_phonemes, load_pairs_with_text
)
from baseline.config import PATHS

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--text-ckpt", required=True)
    p.add_argument("--whisper-ckpt", default="")
    p.add_argument("--csv", default=""); p.add_argument("--zip", default="")
    p.add_argument("--out", default="")
    p.add_argument("--max-pairs", type=int, default=80000)
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    cfg = WhisperConfigV3(); cfg.__post_init__()
    device = args.device if torch.cuda.is_available() else "cpu"
    csv_path = args.csv or cfg.train_csv
    zip_path = args.zip or cfg.train_zip
    out_path = args.out or os.path.join(os.path.dirname(__file__), "hard_neg_iter1.json")

    # Load model
    print(f"[miner] loading {args.text_ckpt}")
    ckpt = torch.load(args.text_ckpt, map_location="cpu", weights_only=False)
    wp = args.whisper_ckpt or ckpt.get("whisper_ckpt", "")
    if not wp or not os.path.isfile(wp):
        for d in sorted(os.listdir(os.path.join(PATHS.root, "output"))):
            pth = os.path.join(PATHS.root, "output", d, "best.pt")
            if os.path.isfile(pth) and "whisper" in d and "text_" not in d:
                wp = pth; break
    model = WhisperTextKWS(wp, cfg.embed_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    # Word index
    rows = load_pairs_with_text(csv_path)
    word_to_pos = defaultdict(list)
    for r in rows:
        if r["enroll_txt"] == r["query_txt"] and r["label"] == 1:
            word_to_pos[r["enroll_txt"].lower()].append((r["id"], r["enroll_txt"]))
    words = sorted(word_to_pos.keys())
    print(f"[miner] {len(words)} words, {sum(len(v) for v in word_to_pos.values())} samples")

    # Encode all enrollment audios
    zf = zipfile.ZipFile(zip_path, "r")
    def read_wav(pid, role):
        data = zf.read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim>1: wav=wav.mean(axis=1)
        if sr!=16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),sr,16000).squeeze(0).numpy()
        wav=wav.astype(np.float32)
        if len(wav)>3*16000: wav=wav[:3*16000]
        return torch.from_numpy(wav).float()

    centroids_a, centroids_t = {}, {}
    total = sum(min(20, len(v)) for v in word_to_pos.values())
    done = 0; t0 = time.time()
    for w in words:
        entries = word_to_pos[w][:20]
        ea_list, et_list = [], []
        for pid, _ in entries:
            wav = read_wav(pid, "enroll")
            phons = _word_to_phonemes(w)
            with torch.no_grad():
                ea = model.encoder(wav.unsqueeze(0).to(device)).cpu()
                et = model.text_encoder([phons])[0].cpu() if phons else torch.zeros(1, 256)
                ea_list.append(ea.squeeze(0)); et_list.append(et.squeeze(0))
        centroids_a[w] = F.normalize(torch.stack(ea_list).mean(0,keepdim=True),dim=-1).squeeze(0)
        centroids_t[w] = F.normalize(torch.stack(et_list).mean(0,keepdim=True),dim=-1).squeeze(0) if et_list else torch.zeros(256)
        done += len(entries)
        if done % 1000 == 0: print(f"  [{done}/{total}]")
    zf.close()
    print(f"  done {time.time()-t0:.0f}s")

    # Pairwise scores: per-word top-K nearest neighbors
    a_mat = torch.stack([centroids_a[w] for w in words])
    t_mat = torch.stack([centroids_t[w] for w in words])
    sa = model.scale_a.item(); st = model.scale_t.item(); sb = model.bias.item()
    W = len(words); K = min(args.topk, W-1)

    confusable = []
    seen = set()
    chunk = 200
    for i in range(0, W, chunk):
        end = min(i+chunk, W)
        ca = torch.matmul(a_mat[i:end], a_mat.T); ct = torch.matmul(t_mat[i:end], t_mat.T)
        scores = sa*ca + st*ct + sb
        for ci in range(scores.shape[0]):
            gi = i+ci; row = scores[ci]; row[gi] = -1e9
            vals, idx = torch.topk(row, K)
            for j, s in zip(idx.tolist(), vals.tolist()):
                pk = tuple(sorted([words[gi], words[j]]))
                if pk not in seen:
                    seen.add(pk)
                    confusable.append((words[gi], words[j], round(float(s),4)))

    confusable.sort(key=lambda x: x[2], reverse=True)
    print(f"[miner] {len(confusable)} unique pairs")
    for w1,w2,s in confusable[:15]:
        print(f"  {w1:20s} <-> {w2:20s}  s={s:.4f}")

    # Generate hard negatives
    extra = []; used = set()
    rng = np.random.default_rng(42)
    max_per = max(1, args.max_pairs//max(1,len(confusable)))+5
    pc = defaultdict(int)
    for w1,w2,score in confusable:
        pk = tuple(sorted([w1,w2]))
        if pc[pk]>=max_per: continue
        p1=word_to_pos.get(w1,[]); p2=word_to_pos.get(w2,[])
        if not p1 or not p2: continue
        n = min(10, len(p1), len(p2), max_per-pc[pk])
        s1=rng.choice(len(p1),n,replace=False); s2=rng.choice(len(p2),n,replace=False)
        for si in s1:
            for sj in s2:
                id1,t1=p1[si]; id2,t2=p2[sj]
                if (id1,id2) in used: continue
                used.add((id1,id2))
                extra.append({"id":f"{id1}_x_{id2}","enroll_id":id1,"query_id":id2,
                              "enroll_txt":t1,"query_txt":t2,"label":0,"score":score})
                pc[pk]+=1
                if len(extra)>=args.max_pairs: break
            if len(extra)>=args.max_pairs: break
        if len(extra)>=args.max_pairs: break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path,"w",encoding="utf-8") as f: json.dump(extra,f,ensure_ascii=False)
    print(f"[miner] {len(extra)} pairs → {out_path}")

if __name__=="__main__": main()

"""Pre-compute Whisper embeddings for text-mode training.

一次性用 frozen whisper 编码所有音频 → 存盘。
训练时直接读缓存，省去每个 epoch 重复编码的开销。

用法:
  python baseline/precompute_whisper_emb.py \
      --whisper-ckpt output/whisper_v3_proto/best.pt \
      --csv train/train_label.csv --zip train/wav.zip \
      --out baseline/whisper_emb_cache.pt
"""
import argparse, os, sys, time, io, zipfile
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.train_whisper_v3 import WhisperEncoderV3, load_pairs_with_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--whisper-ckpt", required=True)
    p.add_argument("--csv", required=True)
    p.add_argument("--zip", required=True)
    p.add_argument("--out", default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-pairs", type=int, default=0)
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    out_path = args.out or os.path.join(os.path.dirname(__file__), "whisper_emb_cache.pt")

    # Load whisper encoder
    ckpt = torch.load(args.whisper_ckpt, map_location="cpu", weights_only=False)
    wm = ckpt.get("whisper_model", "base")
    ed = ckpt.get("embed_dim", 256)
    encoder = WhisperEncoderV3(wm, ed, unfreeze_layers=0, max_audio_sec=3.0).to(device)
    enc_state = {k.replace("encoder.", ""): v for k, v in ckpt["model"].items()
                 if k.startswith("encoder.")}
    encoder.load_state_dict(enc_state, strict=False)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    print(f"[precompute] whisper encoder loaded")

    # Load pairs
    pairs = load_pairs_with_text(args.csv)
    if args.max_pairs > 0:
        pairs = pairs[:args.max_pairs]
    total = len(pairs)
    print(f"[precompute] {total} pairs")

    # Also load hard negatives
    import json as _json
    hn_files = [
        os.path.join(os.path.dirname(__file__), "hard_neg_whisper.json"),
        os.path.join(os.path.dirname(__file__), "hard_neg_wavlm.json"),
    ]
    for hn_file in hn_files:
        if os.path.isfile(hn_file):
            with open(hn_file) as f:
                hn = _json.load(f)
            pairs = pairs + hn
            print(f"  + hard_neg: {len(hn)} (total={len(pairs)})")
            break

    zf = zipfile.ZipFile(args.zip, "r")
    e_embs, q_embs, labels, phonemes = [], [], [], []

    def read_wav(pid, role):
        data = zf.read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        wav = wav.astype(np.float32)
        max_s = 3 * 16000
        if len(wav) > max_s: wav = wav[:max_s]
        return torch.from_numpy(wav).float()

    import re as _re
    _cmudict_cache = None
    def _get_cmudict():
        nonlocal _cmudict_cache
        if _cmudict_cache is None:
            import cmudict
            _cmudict_cache = cmudict.dict()
        return _cmudict_cache

    def word_to_phon(word):
        cmu = _get_cmudict()
        w = word.lower().strip("'s\"-.,!?;:")
        if not w: return []
        plist = cmu.get(w)
        if plist: return [_re.sub(r'[0-2]$', '', p) for p in plist[0]]
        return []

    t0 = time.time()
    bs = args.batch_size
    for start in range(0, total, bs):
        batch = pairs[start:start+bs]
        e_wavs, q_wavs = [], []
        for p in batch:
            pid = p["id"]
            eid = p.get("enroll_id", pid)
            qid = p.get("query_id", pid)
            e_wavs.append(read_wav(eid, "enroll"))
            q_wavs.append(read_wav(qid, "query"))
            labels.append(p.get("label", 0))
            phonemes.append(word_to_phon(p.get("enroll_txt", "")))

        max_len = max(max(w.shape[0] for w in e_wavs), max(w.shape[0] for w in q_wavs))
        e_pad = torch.zeros(len(batch), max_len)
        q_pad = torch.zeros(len(batch), max_len)
        for i, (ew, qw) in enumerate(zip(e_wavs, q_wavs)):
            e_pad[i, :ew.shape[0]] = ew; q_pad[i, :qw.shape[0]] = qw

        with torch.no_grad():
            e_emb = encoder(e_pad.to(device)).cpu()
            q_emb = encoder(q_pad.to(device)).cpu()
        e_embs.append(e_emb); q_embs.append(q_emb)

        done = min(start + bs, total)
        elapsed = time.time() - t0
        rate = done / max(1, elapsed)
        eta = (total - done) / max(1, rate)
        if start % (bs * 10) == 0 or done == total:
            print(f"  [{done}/{total}] {rate:.0f} pairs/s, ETA {eta:.0f}s")

    zf.close()
    e_embs = torch.cat(e_embs); q_embs = torch.cat(q_embs)
    labels = torch.tensor(labels, dtype=torch.long)
    torch.save({"e": e_embs, "q": q_embs, "y": labels, "pn": phonemes}, out_path)
    sz = os.path.getsize(out_path) / 1024**2
    print(f"[precompute] saved {out_path} ({sz:.0f} MB), {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

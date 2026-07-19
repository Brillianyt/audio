"""Inference for Whisper + Text mode model."""
import argparse, csv, io, os, sys, time, zipfile
import numpy as np
import torch
import soundfile as sf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from baseline.train_whisper_v3 import WhisperTextKWS, WhisperConfigV3, load_pairs_no_label
from baseline.config import PATHS


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="from_proto")
    p.add_argument("--ckpt", default="")
    p.add_argument("--bs", type=int, default=128)
    args = p.parse_args()

    cfg = WhisperConfigV3(); cfg.__post_init__()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt_path = args.ckpt or os.path.join(PATHS.root, "output", f"text_{args.name}", "best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[error] {ckpt_path} not found")
        return

    print(f"[infer] loading {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    whisper_path = ckpt.get("whisper_ckpt", "")
    if not whisper_path or not os.path.isfile(whisper_path):
        # Fallback: scan for whisper checkpoints
        for d in sorted(os.listdir(os.path.join(PATHS.root, "output"))):
            p = os.path.join(PATHS.root, "output", d, "best.pt")
            if os.path.isfile(p) and "whisper" in d and "text_" not in d:
                whisper_path = p; break
    print(f"  whisper ckpt: {whisper_path}")
    model = WhisperTextKWS(whisper_path, cfg.embed_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"[infer] seen={ckpt.get('auc_seen','?')} unseen={ckpt.get('auc_unseen','?')}")

    import re as _re
    _cmu = None
    def _cmudict():
        nonlocal _cmu
        if _cmu is None:
            import cmudict; _cmu = cmudict.dict()
        return _cmu
    def w2p(word):
        cmu = _cmudict()
        w = word.lower().strip("'s\"-.,!?;:")
        if not w: return []
        pl = cmu.get(w)
        return [_re.sub(r'[0-2]$','',p) for p in pl[0]] if pl else []

    def pred(zip_p, csv_p, prefix):
        zf = zipfile.ZipFile(zip_p, "r")
        pairs = load_pairs_no_label(csv_p)
        rows = []
        total = len(pairs)
        for i in range(0, total, args.bs):
            batch = pairs[i:i+args.bs]
            e_w, q_w, pns = [], [], []
            for bp in batch:
                pid = bp["id"]
                e_w.append(read_wav(zf, pid, "enroll"))
                q_w.append(read_wav(zf, pid, "query"))
                pns.append(w2p(bp.get("enroll_txt","")))
            max_l = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
            ep = torch.zeros(len(batch), max_l); qp = torch.zeros(len(batch), max_l)
            for j, (ew, qw) in enumerate(zip(e_w, q_w)):
                ep[j,:ew.shape[0]]=ew; qp[j,:qw.shape[0]]=qw
            with torch.no_grad():
                logit, _, _, _ = model(ep.to(device), qp.to(device), pns)
            prob = torch.sigmoid(logit).cpu().numpy()
            for j, bp in enumerate(batch):
                rows.append((f"{prefix}_{bp['id']}", float(prob[j])))
            if (i+1) % 1000 == 0 or (i+args.bs) >= total:
                print(f"  [{prefix}] {min(i+args.bs,total)}/{total}")
        zf.close(); return rows

    def read_wav(zf, pid, role):
        data = zf.read(f"wav/{pid}_{role}.wav")
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim>1: wav=wav.mean(axis=1)
        if sr!=16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0),sr,16000).squeeze(0).numpy()
        wav=wav.astype(np.float32);max_s=3*16000
        if len(wav)>max_s: wav=wav[:max_s]
        return torch.from_numpy(wav).float()

    rows = pred(cfg.eval_seen_zip, cfg.eval_seen_csv, "seen")
    rows += pred(cfg.eval_unseen_zip, cfg.eval_unseen_csv, "unseen")

    out_dir = os.path.join(PATHS.root, "output", f"text_{args.name}")
    out = os.path.join(out_dir, "submission.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["id","posterior"]); w.writerows(rows)
    print(f"[done] {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()

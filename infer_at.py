"""Inference with AT model — produce submission.csv."""
import argparse, csv, json, os, sys, time, io, zipfile
import numpy as np
import torch, torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

device = "cuda" if torch.cuda.is_available() else "cpu"
from train_at_v3 import AudioTextModel

# ── Load model ──
ckpt_path = "output/at_v5/best.pt"
model = AudioTextModel(256).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()
print(f"Loaded: seen={ckpt.get('auc_seen',-1):.4f} unseen={ckpt.get('auc_unseen',-1):.4f}")

scale = 8.0

# ── Eval dataset (reads pairs from CSV, audio from zip) ──
class EvalDataset(Dataset):
    def __init__(self, csv_path, zip_path):
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        self.zip = zipfile.ZipFile(zip_path, "r")
        self.zip_path = zip_path

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        pid = r["id"]
        txt = r.get("enroll_txt", "").lower()
        # Read enroll audio
        e_wav = self._read(f"wav/{pid}_enroll.wav")
        q_wav = self._read(f"wav/{pid}_query.wav")
        if e_wav is None:
            e_wav = self._read(f"wav/{pid}.wav")
        if q_wav is None:
            q_wav = self._read(f"wav/{pid}.wav")
        return e_wav, q_wav, txt, pid

    def _read(self, name):
        try:
            data = self.zip.read(name)
        except KeyError:
            return None
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(1.5 * 16000)
        if len(wav) > ms:
            wav = wav[:ms]
        return torch.from_numpy(wav).float()

def collate(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, txts, ids = [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, ml - e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml - q.shape[-1])) if q.shape[-1] < ml else q)
        txts.append(b[2]); ids.append(b[3])
    return torch.stack(es), torch.stack(qs), txts, ids

@torch.no_grad()
def infer(subset_name):
    csv_path = getattr(PATHS, f"eval_{subset_name}_csv")
    zip_path = getattr(PATHS, f"eval_{subset_name}_zip")
    ds = EvalDataset(csv_path, zip_path)
    ld = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=collate)
    results = []
    for e, q, txts, ids in ld:
        e, q = e.to(device), q.to(device)
        _, _, et = model(e, txts)
        eq = model.encoder(q)
        cs = (et * eq).sum(-1)
        probs = torch.sigmoid(cs * scale).cpu().numpy()
        for pid, p in zip(ids, probs):
            results.append({"id": pid, "posterior": float(p)})
    print(f"  {subset_name}: {len(results)} samples")
    return results

print("\nInferring...")
seen = infer("seen")
unseen = infer("unseen")
all_results = seen + unseen

out = "submission.csv"
with open(out, "w") as f:
    f.write("id,posterior\n")
    for r in all_results:
        f.write(f"{r['id']},{r['posterior']:.6f}\n")
print(f"Saved {out} ({len(all_results)} rows)")

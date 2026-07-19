"""Final submission with AT+AA fusion head."""
import csv, json, os, sys, io, zipfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

device = "cuda" if torch.cuda.is_available() else "cpu"
scale = 8.0

# Load frozen models
from train_at_v3 import AudioTextModel as AT
from train_aa_pk import AAPKModel as AA

at = AT(256).to(device)
ckpt = torch.load("output/at_v5/best.pt", map_location=device, weights_only=False)
at.load_state_dict(ckpt["model"], strict=False); at.eval()
for p in at.parameters(): p.requires_grad_(False)

aa = AA(256).to(device)
aa.load_state_dict(torch.load("output/aa_pk/latest.pt", map_location=device, weights_only=False)["model"], strict=False)
aa.eval()
for p in aa.parameters(): p.requires_grad_(False)

# Fusion head
class FH(nn.Module):
    def __init__(self): super().__init__(); self.net = nn.Sequential(nn.Linear(5,8), nn.ReLU(), nn.Linear(8,1))
    def forward(self, x): return self.net(x).squeeze(-1)
fh = FH().to(device)
fh.load_state_dict(torch.load("output/fusion_head.pt", weights_only=True))
fh.eval()

# Load normalizer from training
fusion_mu = np.load("output/fusion_mu.npy")
fusion_std = np.load("output/fusion_std.npy")

# Dataset
class EvalDS(Dataset):
    def __init__(self, csv_path, zip_path):
        self.rows = [r for r in csv.DictReader(open(csv_path))]
        self.zip = zipfile.ZipFile(zip_path, "r")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]; pid = r["id"]; txt = r.get("enroll_txt","").lower()
        e = self._read(f"wav/{pid}_enroll.wav")
        if e is None: e = self._read(f"wav/{pid}.wav")
        q = self._read(f"wav/{pid}_query.wav")
        if q is None: q = self._read(f"wav/{pid}.wav")
        return e, q, txt, pid
    def _read(self, n):
        try: d = self.zip.read(n)
        except: return None
        w,sr = sf.read(io.BytesIO(d), dtype="float32", always_2d=False)
        if w.ndim > 1: w = w.mean(1)
        if sr != 16000:
            import torchaudio
            w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(1.5*16000)
        if len(w) > ms: w = w[:ms]
        return torch.from_numpy(w).float()

def collate(b):
    ml = max(max(x[0].shape[-1], x[1].shape[-1]) for x in b)
    es, qs, txts, ids = [], [], [], []
    for x in b:
        e, q = x[0], x[1]
        es.append(F.pad(e, (0, ml-e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml-q.shape[-1])) if q.shape[-1] < ml else q)
        txts.append(x[2]); ids.append(x[3])
    return torch.stack(es), torch.stack(qs), txts, ids

@torch.no_grad()
def infer(sub):
    csv_p = getattr(PATHS, f"eval_{sub}_csv")
    zip_p = getattr(PATHS, f"eval_{sub}_zip")
    dl = DataLoader(EvalDS(csv_p, zip_p), batch_size=256, num_workers=0, collate_fn=collate)
    results = []
    for e, q, txts, ids in dl:
        e, q = e.to(device), q.to(device)
        _, _, et = at(e, txts)
        at_p = torch.sigmoid((et * at.encoder(q)).sum(-1) * scale)
        ae = aa(e)[0]; qe = aa(q)[0]
        aa_p = torch.sigmoid((ae * qe).sum(-1) * scale)
        at_a = torch.sigmoid((et * at.encoder(e)).sum(-1) * scale)

        feats = torch.stack([at_p, aa_p, (at_p-0.5).abs(), (aa_p-0.5).abs(), at_a], dim=1)
        # Normalize
        feats_np = feats.cpu().numpy()
        feats_norm = (feats_np - fusion_mu) / fusion_std
        fusion_p = torch.sigmoid(fh(torch.from_numpy(feats_norm).float().to(device)))
        for pid, p in zip(ids, fusion_p.cpu().numpy()):
            results.append({"id": pid, "posterior": float(p)})
    print(f"  {sub}: {len(results)}")
    return results

print("Inferring with fusion head...")
seen = infer("seen")
unseen = infer("unseen")
all_r = seen + unseen

with open("submission_fusion.csv", "w") as f:
    f.write("id,posterior\n")
    for r in all_r:
        f.write(f"{r['id']},{r['posterior']:.6f}\n")
print(f"Saved submission_fusion.csv ({len(all_r)} rows)")

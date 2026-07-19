"""Train small fusion head on frozen AT+AA — 5 features, tiny MLP."""
import csv, json, os, sys, io, zipfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

device = "cuda" if torch.cuda.is_available() else "cpu"
scale = 8.0

from train_at_v3 import AudioTextModel as AT
at = AT(256).to(device)
ckpt = torch.load("output/backup/at_best.pt", map_location=device, weights_only=False)
at.load_state_dict(ckpt["model"], strict=False)
at.eval()
for p in at.parameters(): p.requires_grad = False
print(f"AT frozen: unseen={ckpt.get('auc_unseen',-1):.4f}")

from train_aa_pk import AAPKModel as AA
aa = AA(256).to(device)
aa_ckpt = torch.load("output/aa_pk/aa_final.pt", map_location=device, weights_only=False)
aa.load_state_dict(aa_ckpt["model"], strict=False)
aa.eval()
for p in aa.parameters(): p.requires_grad = False
print(f"AA frozen: seen={aa_ckpt.get('auc_seen',-1):.4f}")

class FeatDataset(Dataset):
    def __init__(self, csv_path, zip_path):
        self.rows = [r for r in csv.DictReader(open(csv_path))]
        self.zip = zipfile.ZipFile(zip_path, "r")
    def __len__(self): return len(self.rows)
    def __getitem__(self, idx):
        r = self.rows[idx]
        pid = r["id"]; txt = r.get("enroll_txt","").lower(); label = float(r.get("label",0))
        e = self._read(f"wav/{pid}_enroll.wav")
        if e is None: e = self._read(f"wav/{pid}.wav")
        q = self._read(f"wav/{pid}_query.wav")
        if q is None: q = self._read(f"wav/{pid}.wav")
        return e, q, txt, label, pid
    def _read(self, name):
        try: data = self.zip.read(name)
        except: return None
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(1)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
        ms = int(1.5 * 16000)
        if len(wav) > ms: wav = wav[:ms]
        return torch.from_numpy(wav).float()

def collate(batch):
    ml = max(max(b[0].shape[-1], b[1].shape[-1]) for b in batch)
    es, qs, txts, ls, ids = [], [], [], [], []
    for b in batch:
        e, q = b[0], b[1]
        es.append(F.pad(e, (0, ml - e.shape[-1])) if e.shape[-1] < ml else e)
        qs.append(F.pad(q, (0, ml - q.shape[-1])) if q.shape[-1] < ml else q)
        txts.append(b[2]); ls.append(b[3]); ids.append(b[4])
    return torch.stack(es), torch.stack(qs), txts, torch.tensor(ls, dtype=torch.float32), ids

@torch.no_grad()
def extract(ld):
    feats, labels = [], []
    for e, q, txts, y, _ in ld:
        e, q = e.to(device), q.to(device)
        _, _, et = at(e, txts)
        at_prob = torch.sigmoid((et * at.encoder(q)).sum(-1) * scale)
        ae = aa(e)[0]; qe = aa(q)[0]
        aa_prob = torch.sigmoid((ae * qe).sum(-1) * scale)
        at_align = torch.sigmoid((et * at.encoder(e)).sum(-1) * scale)
        for i in range(len(y)):
            feats.append([at_prob[i].item(), aa_prob[i].item(),
                          abs(at_prob[i].item()-0.5), abs(aa_prob[i].item()-0.5),
                          at_align[i].item()])
            labels.append(y[i].item())
    return np.array(feats), np.array(labels)

class FusionHead(nn.Module):
    def __init__(self, dim=5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 8), nn.ReLU(), nn.Linear(8, 1))
    def forward(self, x): return self.net(x).squeeze(-1)

print("\nExtracting features...")
all_X, all_y = [], []
for subset in ["seen", "unseen"]:
    csv_p = getattr(PATHS, f"dev_{subset}_csv")
    zip_p = getattr(PATHS, f"dev_{subset}_zip")
    ds = FeatDataset(csv_p, zip_p)
    dl = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=collate)
    X, y = extract(dl)
    all_X.append(X); all_y.append(y)
    print(f"  {subset}: {len(X)}")

X = np.concatenate(all_X); y = np.concatenate(all_y)
idx = np.random.RandomState(42).permutation(len(X))
sp = int(0.8 * len(X))
Xt, yt = X[idx[:sp]], y[idx[:sp]]
Xv, yv = X[idx[sp:]], y[idx[sp:]]

fh = FusionHead().to(device)
opt = torch.optim.Adam(fh.parameters(), lr=1e-2)
best_auc = 0.0

print(f"\nTraining fusion head ({len(Xt)} train, {len(Xv)} val)...")
for ep in range(30):
    perm = np.random.permutation(len(Xt))
    for i in range(0, len(perm), 256):
        b = perm[i:i+256]
        x = torch.from_numpy(Xt[b]).float().to(device)
        logit = fh(x)
        loss = F.binary_cross_entropy_with_logits(logit, torch.from_numpy(yt[b]).float().to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        preds = torch.sigmoid(fh(torch.from_numpy(Xv).float().to(device))).cpu().numpy()
        auc = roc_auc_score(yv, preds)
        if auc > best_auc:
            best_auc = auc; torch.save(fh.state_dict(), "output/fusion_head.pt")
    if (ep+1)%10==0: print(f"  ep{ep+1}: val AUC={auc:.4f} (best={best_auc:.4f})")

fh.load_state_dict(torch.load("output/fusion_head.pt"))
fh.eval()
with torch.no_grad():
    all_preds = torch.sigmoid(fh(torch.from_numpy(X).float().to(device))).cpu().numpy()
print(f"\nFusion dev AUC: {roc_auc_score(y, all_preds):.4f}")
print(f"AT only dev AUC: {roc_auc_score(y, X[:,0]):.4f}")
print(f"AA only dev AUC: {roc_auc_score(y, X[:,1]):.4f}")

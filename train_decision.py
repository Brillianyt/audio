"""Train a small decision network to fuse AT and AA predictions."""
import argparse, csv, json, os, sys, time, io, zipfile
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "baseline"))
from config import PATHS

device = "cuda" if torch.cuda.is_available() else "cpu"
scale = 8.0

# ── Load frozen models ──
from train_at_v3 import AudioTextModel as ATModel
at_model = ATModel(256).to(device)
ckpt = torch.load("output/at_v5/at_final.pt", map_location=device, weights_only=False)
at_model.load_state_dict(ckpt["model"], strict=False)
at_model.eval()
for p in at_model.parameters(): p.requires_grad = False
print(f"AT frozen: unseen={ckpt.get('auc_unseen',-1):.4f}")

from train_aa_pk import AAPKModel as AAModel
aa_model = AAModel(256).to(device)
aa_ckpt = torch.load("output/aa_pk/aa_final.pt", map_location=device, weights_only=False)
aa_model.load_state_dict(aa_ckpt["model"], strict=False)
aa_model.eval()
for p in aa_model.parameters(): p.requires_grad = False
print(f"AA frozen: seen={aa_ckpt.get('auc_seen',-1):.4f}")

# ── Feature extraction dataset ──
class FeatureDataset(Dataset):
    def __init__(self, csv_path, zip_path):
        self.rows = []
        with open(csv_path, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        self.zip = zipfile.ZipFile(zip_path, "r")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        pid = r["id"]
        txt = r.get("enroll_txt", "").lower()
        label = float(r.get("label", 0))
        e_wav = self._read(f"wav/{pid}_enroll.wav") or self._read(f"wav/{pid}.wav")
        q_wav = self._read(f"wav/{pid}_query.wav") or self._read(f"wav/{pid}.wav")
        return e_wav, q_wav, txt, label, pid

    def _read(self, name):
        try:
            data = self.zip.read(name)
        except KeyError:
            return None
        wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if wav.ndim > 1: wav = wav.mean(axis=1)
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
def extract_features(dl):
    """Extract features from AT and AA for each sample."""
    feats, labels = [], []
    for e, q, txts, y, _ in dl:
        e, q = e.to(device), q.to(device)

        # AT features
        _, _, et = at_model(e, txts)
        eq = at_model.encoder(q)
        ea = at_model.encoder(e)
        at_cos = (et * eq).sum(-1)           # text↔query
        at_align = (et * ea).sum(-1)           # text↔enroll alignment
        at_prob = torch.sigmoid(at_cos * scale)

        # AA features
        ae, _ = aa_model(e)
        qe, _ = aa_model(q)
        aa_cos = (ae * qe).sum(-1)
        aa_prob = torch.sigmoid(aa_cos * scale)

        for i in range(len(y)):
            feats.append([
                at_prob[i].item(),             # 0: AT probability
                aa_prob[i].item(),             # 1: AA probability
                abs(at_prob[i].item() - 0.5),  # 2: AT confidence
                abs(aa_prob[i].item() - 0.5),  # 3: AA confidence
                at_cos[i].item(),              # 4: AT raw cosine
                aa_cos[i].item(),              # 5: AA raw cosine
                at_align[i].item(),            # 6: AT text↔enroll alignment
            ])
            labels.append(y[i].item())
    return np.array(feats, dtype=np.float32), np.array(labels)


# ═══════════ Decision Network ═══════════
class DecisionNet(nn.Module):
    def __init__(self, input_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 8), nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ── Load dev data and extract features ──
print("\nExtracting features from dev sets...")
def load_and_extract(subset):
    csv_path = getattr(PATHS, f"dev_{subset}_csv")
    zip_path = getattr(PATHS, f"dev_{subset}_zip")
    ds = FeatureDataset(csv_path, zip_path)
    dl = DataLoader(ds, batch_size=256, num_workers=0, collate_fn=collate)
    X, y = extract_features(dl)
    print(f"  {subset}: {len(X)} samples")
    return X, y

X_s, y_s = load_and_extract("seen")
X_u, y_u = load_and_extract("unseen")

# Combine seen+unseen for training
X_all = np.concatenate([X_s, X_u])
y_all = np.concatenate([y_s, y_u])

# Train/val split (80/20)
np.random.seed(42)
idx = np.random.permutation(len(X_all))
split = int(0.8 * len(X_all))
X_train, y_train = X_all[idx[:split]], y_all[idx[:split]]
X_val, y_val = X_all[idx[split:]], y_all[idx[split:]]

# Normalize features
mu, std = X_train.mean(0), X_train.std(0) + 1e-8
X_train_n = (X_train - mu) / std
X_val_n = (X_val - mu) / std

# Train decision network
dn = DecisionNet(7).to(device)
opt = torch.optim.Adam(dn.parameters(), lr=1e-3)
crit = nn.BCEWithLogitsLoss()
best_auc = 0.0

print(f"\nTraining decision network on {len(X_train)} samples...")
for ep in range(50):
    perm = np.random.permutation(len(X_train_n))
    for i in range(0, len(perm), 256):
        batch = perm[i:i+256]
        x = torch.from_numpy(X_train_n[batch]).to(device)
        y = torch.from_numpy(y_train[batch]).to(device)
        logit = dn(x)
        loss = crit(logit, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    # Val
    with torch.no_grad():
        xv = torch.from_numpy(X_val_n).to(device)
        preds = torch.sigmoid(dn(xv)).cpu().numpy()
        auc = roc_auc_score(y_val, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save(dn.state_dict(), "output/decision_net.pt")
    if (ep+1) % 10 == 0:
        print(f"  ep{ep+1}: val AUC={auc:.4f} (best={best_auc:.4f})")

print(f"\nBest val AUC: {best_auc:.4f}")

# Evaluate on dev sets
dn.load_state_dict(torch.load("output/decision_net.pt", weights_only=True))
dn.eval()
for name, X, y in [("dev_seen", X_s, y_s), ("dev_unseen", X_u, y_u), ("dev_all", X_all, y_all)]:
    Xn = (X - mu) / std
    with torch.no_grad():
        preds = torch.sigmoid(dn(torch.from_numpy(Xn).to(device))).cpu().numpy()
        auc = roc_auc_score(y, preds)
        print(f"  {name}: AUC={auc:.4f}")

# Compare with AT-only, AA-only, and naive fusion
at_only = roc_auc_score(y_all, X_all[:, 0])
aa_only = roc_auc_score(y_all, X_all[:, 1])
# Confidence-weighted fusion
at_conf = np.abs(X_all[:, 0] - 0.5) * 2
aa_conf = np.abs(X_all[:, 1] - 0.5) * 2
naive_fusion = (X_all[:, 0] * at_conf + X_all[:, 1] * aa_conf) / (at_conf + aa_conf + 1e-8)
naive_auc = roc_auc_score(y_all, naive_fusion)

print(f"\nComparison on dev_all:")
print(f"  AT only:       {at_only:.4f}")
print(f"  AA only:       {aa_only:.4f}")
print(f"  Naive fusion:  {naive_auc:.4f}")
print(f"  Decision Net:  {best_auc:.4f}")

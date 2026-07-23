"""Extract AT v8 failure cases for manual listening."""
import sys, os, csv, io, zipfile
import torch, torch.nn.functional as F
import soundfile as sf
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS

import importlib.util
spec = importlib.util.spec_from_file_location('td', 'train_dual.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
AudioTextModel = td.AudioTextModel; DualConfig = td.Config

device = 'cuda'; cfg = DualConfig(); cfg.__post_init__()

model = AudioTextModel('', 256, unfreeze=0).to(device)
ckpt = torch.load('output/backup_final/at_v8_ep29_seen08462_unseen07889.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()

out_dir = 'Debug/failures'
os.makedirs(out_dir, exist_ok=True)

def find_failures(name, zip_path, csv_path):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    pos = [p for p in pairs if p['label'] == 1]
    print(f'\n[{name}] {len(pos)} positive pairs')

    results = []
    zf = zipfile.ZipFile(zip_path, 'r')
    for i in range(0, len(pos), 64):
        batch = pos[i:i+64]
        e_w, q_w, txts = [], [], []
        for p in batch:
            for role, lst in [('enroll', e_w), ('query', q_w)]:
                data = zf.read(f"wav/{p['id']}_{role}.wav")
                w, sr = sf.read(io.BytesIO(data), dtype='float32')
                if w.ndim > 1: w = w.mean(axis=1)
                if sr != 16000:
                    import torchaudio
                    w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
                w = torch.from_numpy(w).float()
                lst.append(F.pad(w, (8000, 8000)))
            txts.append(p['enroll_txt'])
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)
        with torch.no_grad():
            _, score, _, _ = model(eb, txts, qb)
            p = torch.sigmoid(score).detach().cpu().numpy()
        for j, bp in enumerate(batch):
            results.append((p[j].item(), bp['id'], bp['enroll_txt']))
    zf.close()

    # Worst failures: label=1 but AT gives lowest posterior
    results.sort(key=lambda r: r[0])
    return results[:10]

for name, zp, cp in [('seen', cfg.dev_seen_zip, cfg.dev_seen_csv),
                       ('unseen', cfg.dev_unseen_zip, cfg.dev_unseen_csv)]:
    fails = find_failures(name, zp, cp)
    print(f'\nTop 10 {name} failures:')
    for i, (p, pid, txt) in enumerate(fails):
        # Extract audio files
        with zipfile.ZipFile(zp, 'r') as zf:
            for role in ['enroll', 'query']:
                data = zf.read(f"wav/{pid}_{role}.wav")
                fname = f"fail_{name}_{i+1}_{pid}_{role}.wav"
                with open(os.path.join(out_dir, fname), 'wb') as f:
                    f.write(data)
        print(f'  {i+1}. p={p:.4f} id={pid} txt="{txt}"  → {fname.replace(".wav","")}')

print(f'\nSaved to {out_dir}/')

"""Compare AT v8 ep29 vs Fusion v2 ep9 on failure cases."""
import sys, os, csv, io, zipfile
import torch, torch.nn.functional as F
import soundfile as sf
import numpy as np
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS

import importlib.util
spec = importlib.util.spec_from_file_location('td', 'train_dual.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
AudioTextModel, DualConfig = td.AudioTextModel, td.Config

from train_fusion import AudioTextFusionModel, FusionConfig

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

# AT v8 ep29
at = AudioTextModel('', 256, unfreeze=0).to(device)
ckpt = torch.load('output/backup_final/at_v8_ep29_seen08462_unseen07889.pt', map_location=device)
at.load_state_dict(ckpt['model'], strict=False)
at.eval()

# Fusion v2 ep9 (best)
fusion = AudioTextFusionModel(256, unfreeze=0).to(device)
ckpt2 = torch.load('output/fusion_v2/best.pt', map_location=device)
fusion.load_state_dict(ckpt2['model'], strict=False)
fusion.eval()

seen_vocab = set()
train_csv = os.path.join(PATHS.root, "train", "train_label.csv")
if not os.path.isfile(train_csv):
    train_csv = os.path.join(PATHS.root, "train_subset", "train_label.csv")
with open(train_csv) as f:
    for r in csv.DictReader(f):
        seen_vocab.add(r['enroll_txt'].lower().strip())

def read_wav(zf, pid, role):
    try: data = zf.read(f"wav/{pid}_{role}.wav")
    except: data = zf.read(f"wav/{pid}.wav")
    w, sr = sf.read(io.BytesIO(data), dtype='float32')
    if w.ndim > 1: w = w.mean(axis=1)
    if sr != 16000:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    return torch.from_numpy(w).float()

for name, zp, cp in [('dev_seen', cfg.dev_seen_zip, cfg.dev_seen_csv)]:
    pairs = []
    with open(cp) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    pos = [p for p in pairs if p['label'] == 1]
    zf = zipfile.ZipFile(zp, 'r')
    results = []

    for i in range(0, len(pos), 128):
        batch = pos[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            e_w.append(read_wav(zf, p['id'], 'enroll'))
            q_w.append(read_wav(zf, p['id'], 'query'))
            txts.append(p['enroll_txt'])
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        eb_pad = F.pad(eb, (8000, 8000))
        qb_pad = F.pad(qb, (8000, 8000))

        with torch.no_grad():
            _, score_at, _, _ = at(eb_pad, txts, qb_pad)
            p_at = torch.sigmoid(score_at).detach().cpu().numpy()
            logit_fu, _, _ = fusion(eb, txts, qb)
            p_fu = torch.sigmoid(logit_fu).detach().cpu().numpy()

        for j in range(len(batch)):
            results.append((p_at[j].item(), p_fu[j].item(), batch[j]['id'], batch[j]['enroll_txt']))

    zf.close()

    # AT false negatives (label=1, p_at < 0.3)
    fn = [r for r in results if r[0] < 0.3]
    fn.sort(key=lambda r: r[0])

    print(f'\n{name}: {len(fn)} AT false negatives (p_at < 0.3)\n')
    print(f'{"AT_p":>8} {"Fu_p":>8} {"AT_err":>8} {"Fu_err":>8} {"word":>15} {"id":>12}')
    print('-' * 55)

    at_wrong = 0
    fu_fixed = 0
    for p_at, p_fu, pid, txt in fn[:50]:
        at_err = p_at < 0.5  # AT wrong (should be >0.5 since label=1)
        fu_err = p_fu < 0.5
        if at_err: at_wrong += 1
        if not fu_err and at_err: fu_fixed += 1
        fix = '✓' if (not fu_err and at_err) else ''
        print(f'{p_at:>8.4f} {p_fu:>8.4f} {str(at_err):>8} {str(fu_err):>8} {txt:>15} {pid[-10:]:>12} {fix}')

    print(f'\nAT wrong: {at_wrong}/{len(fn)}')
    print(f'Fusion fixed: {fu_fixed}/{at_wrong} ({fu_fixed/at_wrong*100:.1f}% of AT errors)')

    # Also check: on ALL positive pairs, how often does fusion beat AT?
    all_pos = np.array([r[0] for r in results])
    all_fu = np.array([r[1] for r in results])
    at_correct = (all_pos > 0.5).mean()
    fu_correct = (all_fu > 0.5).mean()
    print(f'\nOverall on positives: AT acc={at_correct*100:.1f}% Fusion acc={fu_correct*100:.1f}%')

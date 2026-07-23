"""Find cases where only AA gets it right."""
import sys, os, csv, io, zipfile
import numpy as np
import torch, torch.nn.functional as F
import soundfile as sf
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS

import importlib.util
spec = importlib.util.spec_from_file_location('td', 'train_dual.py')
td = importlib.util.module_from_spec(spec)
spec.loader.exec_module(td)
AudioTextModel, DualConfig = td.AudioTextModel, td.Config
from train_fusion import AudioTextFusionModel
from baseline.train_whisper_v3 import MultiTaskWhisperKWSV3

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

at = AudioTextModel('', 256, unfreeze=0).to(device)
at.load_state_dict(torch.load('output/backup_final/at_v8_ep29_seen08462_unseen07889.pt', map_location=device)['model'], strict=False)
at.eval()

fu = AudioTextFusionModel(256, unfreeze=0).to(device)
fu.load_state_dict(torch.load('output/fusion_v2/best.pt', map_location=device)['model'], strict=False)
fu.eval()

aa = MultiTaskWhisperKWSV3('base', 256, unfreeze_layers=0).to(device)
aa.load_state_dict(torch.load('output/backup_final/aa_v3_best_seen07934.pt', map_location=device)['model'], strict=False)
aa.eval()

def read_and_pad(zf, pid, role):
    try: data = zf.read(f"wav/{pid}_{role}.wav")
    except: data = zf.read(f"wav/{pid}.wav")
    w, sr = sf.read(io.BytesIO(data), dtype='float32')
    if w.ndim > 1: w = w.mean(axis=1)
    if sr != 16000:
        import torchaudio
        w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    w = torch.from_numpy(w).float()
    return F.pad(w, (8000, 8000))

for name, zp, cp in [('dev_seen', cfg.dev_seen_zip, cfg.dev_seen_csv)]:
    pairs = []
    with open(cp) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt','').lower()})

    zf = zipfile.ZipFile(zp, 'r')
    results = []
    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            e_w.append(read_and_pad(zf, p['id'], 'enroll'))
            q_w.append(read_and_pad(zf, p['id'], 'query'))
            txts.append(p['enroll_txt'])
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)
        with torch.no_grad():
            _, score, _, _ = at(eb, txts, qb)
            p_at = torch.sigmoid(score).detach().cpu().numpy()
            logit, _, _ = fu(eb, txts, qb)
            p_fu = torch.sigmoid(logit).detach().cpu().numpy()
            logit_aa, _, _ = aa(eb, qb)
            p_aa = torch.sigmoid(logit_aa).detach().cpu().numpy()

        for j in range(len(batch)):
            results.append((p_at[j].item(), p_fu[j].item(), p_aa[j].item(),
                           batch[j]['label'], batch[j]['id'], batch[j]['enroll_txt']))
    zf.close()

    # Find: AT wrong + Fusion wrong, but AA right
    # label=1: p<0.5 = wrong, label=0: p>0.5 = wrong
    aa_only_right = []
    for p_at, p_fu, p_aa, label, pid, txt in results:
        at_wrong = (label == 1 and p_at < 0.5) or (label == 0 and p_at > 0.5)
        fu_wrong = (label == 1 and p_fu < 0.5) or (label == 0 and p_fu > 0.5)
        aa_right = (label == 1 and p_aa >= 0.5) or (label == 0 and p_aa < 0.5)
        if at_wrong and fu_wrong and aa_right:
            aa_only_right.append((p_at, p_fu, p_aa, label, pid, txt))

    aa_only_right.sort(key=lambda r: abs(r[2]-0.5), reverse=True)
    print(f'\n{name}: {len(results)} pairs')
    print(f'AT+Fusion both wrong, AA right: {len(aa_only_right)} cases\n')

    print(f'{"AT":>8} {"Fusion":>8} {"AA":>8} {"label":>6} {"word":>15} {"id":>12}')
    print('-' * 60)
    for p_at, p_fu, p_aa, label, pid, txt in aa_only_right[:20]:
        print(f'{p_at:>8.4f} {p_fu:>8.4f} {p_aa:>8.4f} {label:>6} {txt:>15} {pid[-10:]:>12}')

    # Also check: how many total errors for each model
    at_err = sum(1 for r in results if (r[3]==1 and r[0]<0.5) or (r[3]==0 and r[0]>=0.5))
    fu_err = sum(1 for r in results if (r[3]==1 and r[1]<0.5) or (r[3]==0 and r[1]>=0.5))
    aa_err = sum(1 for r in results if (r[3]==1 and r[2]<0.5) or (r[3]==0 and r[2]>=0.5))
    print(f'\nTotal errors: AT={at_err}, Fusion={fu_err}, AA={aa_err}')

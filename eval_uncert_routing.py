"""Test AT uncertainty-zone routing.
- p_at close to 0 or 1 → AT is confident, keep
- p_at in 0.2~0.6 → uncertain zone, use AA (for seen words)
"""
import sys, os, csv, io, zipfile
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import PATHS

import importlib.util
_td_spec = importlib.util.spec_from_file_location(
    "train_dual_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dual.py"))
_td = importlib.util.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td)
AudioTextModel = _td.AudioTextModel
DualConfig = _td.Config
from baseline.train_whisper_v3 import MultiTaskWhisperKWSV3

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

# ── Load AT ──
at_model = AudioTextModel('', 256, unfreeze=0).to(device)
at_ckpt = torch.load('output/dual_at_v8_text/best.pt', map_location=device, weights_only=False)
at_model.load_state_dict(at_ckpt['model'], strict=False)
at_model.eval()

# ── Load AA ──
aa_model = MultiTaskWhisperKWSV3('base', 256, unfreeze_layers=0).to(device)
aa_ckpt = torch.load('output/backup_final/aa_v3_best_seen07934.pt', map_location=device, weights_only=False)
aa_model.load_state_dict(aa_ckpt['model'], strict=False)
aa_model.eval()

seen_vocab = set()
with open(os.path.join(PATHS.root, 'train', 'train_label.csv')) as f:
    for r in csv.DictReader(f):
        seen_vocab.add(r.get('enroll_txt', '').lower().strip())


def read_wav(zf, pid, role):
    try:
        data = zf.read(f"wav/{pid}_{role}.wav")
    except KeyError:
        data = zf.read(f"wav/{pid}.wav")
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    wav = torch.from_numpy(wav).float()
    pad = int(0.5 * 16000)
    wav = F.pad(wav, (pad, pad))
    return wav


def evaluate(name, zip_path, csv_path, uncert_lo=0.2, uncert_hi=0.6):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    print(f'\n[{name}] {len(pairs)} pairs')
    zf = zipfile.ZipFile(zip_path, 'r')

    # Collect all AT/AA predictions first
    all_p_at, all_p_aa, all_labels, all_txts = [], [], [], []
    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w = [read_wav(zf, p['id'], 'enroll') for p in batch]
        q_w = [read_wav(zf, p['id'], 'query') for p in batch]
        txts = [p['enroll_txt'] for p in batch]
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        with torch.no_grad():
            _, score_at, _, _ = at_model(eb, txts, qb)
            p_at = torch.sigmoid(score_at).detach().cpu().numpy()
            logit_aa, _, _ = aa_model(eb, qb)
            p_aa = torch.sigmoid(logit_aa).detach().cpu().numpy()

        all_p_at.extend(p_at.tolist())
        all_p_aa.extend(p_aa.tolist())
        all_labels.extend([p['label'] for p in batch])
        all_txts.extend(txts)

    zf.close()

    p_at_arr = np.array(all_p_at)
    p_aa_arr = np.array(all_p_aa)
    labels = np.array(all_labels)

    # Analysis by AT confidence zone
    print(f'\n  Zone analysis:')
    print(f'  {"Zone":>12} {"Count":>8} {"%pos":>8} {"AT AUC":>8} {"AA AUC":>8} {"AA better?":>10}')
    print(f'  {"-"*12} {"-"*8} {"-"*8} {"-"*8} {"-"*8} {"-"*10}')

    for lo, hi, label in [(0, 0.2, 'confident-'),
                           (0.2, 0.6, 'uncertain '),
                           (0.6, 1.0, 'confident+')]:
        mask = (p_at_arr >= lo) & (p_at_arr < hi)
        if mask.sum() == 0:
            continue
        at_auc = roc_auc_score(labels[mask], p_at_arr[mask]) if len(np.unique(labels[mask])) > 1 else 0
        aa_auc = roc_auc_score(labels[mask], p_aa_arr[mask]) if len(np.unique(labels[mask])) > 1 else 0
        pos_pct = labels[mask].mean() * 100
        better = '✓' if aa_auc > at_auc else '✗'
        print(f'  {label:>12} {mask.sum():>8} {pos_pct:>7.1f}% {at_auc:>8.4f} {aa_auc:>8.4f} {better:>10}')

    # Try different uncertainty thresholds
    print(f'\n  Threshold sweep:')
    print(f'  {"low":>5} {"high":>5} {"n_uncert":>10} {"AT AUC":>8} {"Routed AUC":>8} {"Delta":>8}')
    print(f'  {"-"*5} {"-"*5} {"-"*10} {"-"*8} {"-"*8} {"-"*8}')

    best_delta = -1
    best_config = None
    for lo in [0.1, 0.15, 0.2, 0.25, 0.3]:
        for hi in [0.5, 0.6, 0.7, 0.8]:
            if lo >= hi: continue
            mask = (p_at_arr >= lo) & (p_at_arr < hi)
            txt_arr = np.array(all_txts)

            # Route: uncertain + seen → use AA
            routed = p_at_arr.copy()
            for j in range(len(routed)):
                if mask[j] and txt_arr[j] in seen_vocab:
                    routed[j] = p_aa_arr[j]

            at_auc = roc_auc_score(labels, p_at_arr)
            rt_auc = roc_auc_score(labels, routed)
            delta = rt_auc - at_auc

            n_uncert = mask.sum()
            changed = (p_at_arr != routed).sum()
            print(f'  {lo:.2f} {hi:.2f} {n_uncert:>10} {at_auc:>8.4f} {rt_auc:>8.4f} {delta:>+8.4f}')

            if delta > best_delta:
                best_delta = delta
                best_config = (lo, hi, rt_auc)

    if best_config:
        print(f'\n  Best: lo={best_config[0]:.2f} hi={best_config[1]:.2f} AUC={best_config[2]:.4f} delta={best_delta:+.4f}')


evaluate('dev_seen', cfg.dev_seen_zip, cfg.dev_seen_csv)
evaluate('dev_unseen', cfg.dev_unseen_zip, cfg.dev_unseen_csv)

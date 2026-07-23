"""Check if routing helps on AT v8's worst failure cases.
    
Approach:
  1. Run AT v8 on dev set, find the positive pairs with lowest posterior (worst FNs)
  2. These are the severely damaged cases the user heard
  3. Check: does routing to AA (for seen) fix these cases?
"""
import sys, os, csv, io, zipfile, json
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATHS
from audio_quality import compute_damage_score_batch, classify_damage

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

# Load models
at_model = AudioTextModel('', 256, unfreeze=0).to(device)
at_ckpt = torch.load('output/dual_at_v8_text/best.pt', map_location=device, weights_only=False)
at_model.load_state_dict(at_ckpt['model'], strict=False)
at_model.eval()

aa_model = MultiTaskWhisperKWSV3('base', 256, unfreeze_layers=0).to(device)
aa_ckpt = torch.load('output/backup_final/aa_v3_best_seen07934.pt', map_location=device, weights_only=False)
aa_model.load_state_dict(aa_ckpt['model'], strict=False)
aa_model.eval()

seen_vocab = set()
with open(os.path.join(PATHS.root, 'train', 'train_label.csv')) as f:
    for r in csv.DictReader(f):
        seen_vocab.add(r.get('enroll_txt', '').lower().strip())


def analyze_subset(name, zip_path, csv_path):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    pos_pairs = [p for p in pairs if p['label'] == 1]
    print(f'\n{"="*60}')
    print(f'[{name}] {len(pairs)} total, {len(pos_pairs)} positive')

    zf = zipfile.ZipFile(zip_path, 'r')
    all_results = []

    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w = [read_wav(zf, p['id'], 'enroll') for p in batch]
        q_w = [read_wav(zf, p['id'], 'query') for p in batch]
        txts = [p['enroll_txt'] for p in batch]
        labels = [p['label'] for p in batch]

        e_b = pad_batch(e_w).to(device)
        q_b = pad_batch(q_w).to(device)

        _, score_at, _, _ = at_model(e_b, txts, q_b)
        p_at = torch.sigmoid(score_at).detach().cpu().numpy()

        logit_aa, _, _ = aa_model(e_b, q_b)
        p_aa = torch.sigmoid(logit_aa).detach().cpu().numpy()

        ds = compute_damage_score_batch(q_w)

        for j in range(len(batch)):
            all_results.append({
                'id': batch[j]['id'],
                'label': labels[j],
                'txt': txts[j],
                'p_at': float(p_at[j]),
                'p_aa': float(p_aa[j]),
                'damage': ds[j],
                'seen': txts[j] in seen_vocab,
            })

    zf.close()

    # ── Worst false negatives: label=1 but p_at is lowest ──
    pos_results = [r for r in all_results if r['label'] == 1]
    pos_results.sort(key=lambda r: r['p_at'])

    print(f'\nTop 30 worst AT failures (label=1, lowest p_at):')
    print(f'{"idx":>3} {"id":>12} {"p_at":>7} {"p_aa":>7} {"damage":>7} {"seen":>5} {"txt":>12}')
    print('-' * 55)

    for k, r in enumerate(pos_results[:30]):
        routed = r['p_aa'] if r['seen'] else r['p_at'] * 0.8
        fix = '✓' if routed > r['p_at'] + 0.1 else ''
        print(f'{k+1:>3} {r["id"][-10:]:>12} {r["p_at"]:>7.4f} {r["p_aa"]:>7.4f} '
              f'{r["damage"]:>7.3f} {"seen" if r["seen"] else "unseen":>5} {r["txt"][:12]:>12} {fix}')

    # ── Stats on worst failures ──
    worst_pos = pos_results[:max(50, len(pos_results)//100)]
    at_avg = np.mean([r['p_at'] for r in worst_pos])
    aa_avg = np.mean([r['p_aa'] for r in worst_pos])

    # Routing outcome on worst cases
    routed_improved = 0
    for r in worst_pos:
        posterior_at = r['p_at']
        posterior_aa = r['p_aa']
        damage = r['damage']
        level = classify_damage(damage)
        if level == 'severe' and r['seen']:
            routed = posterior_aa
        elif level == 'severe' and not r['seen']:
            routed = posterior_at * 0.8
        elif level == 'moderate':
            conf_at = abs(posterior_at - 0.5) * 2
            conf_aa = abs(posterior_aa - 0.5) * 2
            if conf_at + conf_aa > 0:
                routed = (conf_at * posterior_at + conf_aa * posterior_aa) / (conf_at + conf_aa)
            else:
                routed = (posterior_at + posterior_aa) / 2
        else:
            routed = posterior_at

        if routed > posterior_at + 0.1:
            routed_improved += 1

    print(f'\nWorst {len(worst_pos)} failures (lowest p_at):')
    print(f'  AT avg: {at_avg:.4f}')
    print(f'  AA avg: {aa_avg:.4f}')
    print(f'  Routing improves >0.1: {routed_improved}/{len(worst_pos)}')

    # ── Check: does routing help the LOWEST p_at cases? ──
    # For each failure case, compare p_at vs routed
    improvements = []
    for r in pos_results:
        ds = r['damage']
        level = classify_damage(ds)
        if level == 'severe' and r['seen']:
            routed = r['p_aa']
        elif level == 'severe' and not r['seen']:
            routed = r['p_at'] * 0.8
        elif level == 'moderate':
            conf_at = abs(r['p_at'] - 0.5) * 2
            conf_aa = abs(r['p_aa'] - 0.5) * 2
            if conf_at + conf_aa > 0:
                routed = (conf_at * r['p_at'] + conf_aa * r['p_aa']) / (conf_at + conf_aa)
            else:
                routed = (r['p_at'] + r['p_aa']) / 2
        else:
            routed = r['p_at']
        improvements.append(routed - r['p_at'])

    print(f'\nRouting impact on ALL positive samples:')
    impr = np.array(improvements)
    print(f'  mean delta: {impr.mean():+.5f}')
    print(f'  max improvement: {impr.max():+.4f}')
    print(f'  max regression: {impr.min():+.4f}')
    print(f'  % improved:  {(impr > 0.01).mean()*100:.1f}%')
    print(f'  % regressed: {(impr < -0.01).mean()*100:.1f}%')
    print(f'  % neutral:   {((impr >= -0.01) & (impr <= 0.01)).mean()*100:.1f}%')

    # ── AUCS ──
    labels = np.array([r['label'] for r in all_results])
    at_only = np.array([r['p_at'] for r in all_results])

    # Compute routing AUC
    routed_ps = []
    for r in all_results:
        ds = r['damage']
        level = classify_damage(ds)
        if level == 'severe' and r['seen']:
            routed_ps.append(r['p_aa'])
        elif level == 'severe' and not r['seen']:
            routed_ps.append(r['p_at'] * 0.8)
        elif level == 'moderate':
            conf_at = abs(r['p_at'] - 0.5) * 2
            conf_aa = abs(r['p_aa'] - 0.5) * 2
            if conf_at + conf_aa > 0:
                routed_ps.append((conf_at * r['p_at'] + conf_aa * r['p_aa']) / (conf_at + conf_aa))
            else:
                routed_ps.append((r['p_at'] + r['p_aa']) / 2)
        else:
            routed_ps.append(r['p_at'])

    print(f'\n  AT AUC:         {roc_auc_score(labels, at_only):.4f}')
    print(f'  Routing AUC:    {roc_auc_score(labels, routed_ps):.4f}')


def read_wav(zf, pid, role):
    try:
        data = zf.read(f"wav/{pid}_{role}.wav")
    except KeyError:
        data = zf.read(f"wav/{pid}.wav")
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    return torch.from_numpy(wav).float()


def pad_batch(wavs):
    max_len = max(w.shape[0] for w in wavs)
    batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch[i, :w.shape[0]] = w
    return batch


analyze_subset("dev_seen", cfg.dev_seen_zip, cfg.dev_seen_csv)
analyze_subset("dev_unseen", cfg.dev_unseen_zip, cfg.dev_unseen_csv)

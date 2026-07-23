"""Confidence-based routing: if AT posterior < threshold → fallback to AA.
Only applies to seen words (AA is random on unseen).

Evaluates on dev set with proper silence padding (matching training).
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

# ── Load models ──
at_model = AudioTextModel('', 256, unfreeze=0).to(device)
at_ckpt = torch.load('output/dual_at_v8_text/best.pt', map_location=device, weights_only=False)
at_model.load_state_dict(at_ckpt['model'], strict=False)
at_model.eval()
print(f'AT v8: seen={at_ckpt["auc_seen"]:.4f} unseen={at_ckpt["auc_unseen"]:.4f}')

aa_model = MultiTaskWhisperKWSV3('base', 256, unfreeze_layers=0).to(device)
aa_ckpt = torch.load('output/backup_final/aa_v3_best_seen07934.pt', map_location=device, weights_only=False)
aa_model.load_state_dict(aa_ckpt['model'], strict=False)
aa_model.eval()

# Train vocab
seen_vocab = set()
with open(os.path.join(PATHS.root, 'train', 'train_label.csv')) as f:
    for r in csv.DictReader(f):
        seen_vocab.add(r.get('enroll_txt', '').lower().strip())


def read_wav(zf, pid, role):
    """Read wav AND add 0.5s silence padding (matching training)."""
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
    wav = torch.from_numpy(wav).float()
    # Silence padding: 0.5s front + 0.5s back (matching training)
    pad = int(0.5 * 16000)
    wav = F.pad(wav, (pad, pad))
    return wav


def pad_batch(wavs):
    max_len = max(w.shape[0] for w in wavs)
    batch = torch.zeros(len(wavs), max_len)
    for i, w in enumerate(wavs):
        batch[i, :w.shape[0]] = w
    return batch


def evaluate(name, zip_path, csv_path):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    pos_pairs = [p for p in pairs if p['label'] == 1]
    print(f'\n[{name}] {len(pairs)} pairs, {len(pos_pairs)} positive')

    zf = zipfile.ZipFile(zip_path, 'r')

    at_all, aa_all, routed_all, labels_all = [], [], [], []

    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w = [read_wav(zf, p['id'], 'enroll') for p in batch]
        q_w = [read_wav(zf, p['id'], 'query') for p in batch]
        txts = [p['enroll_txt'] for p in batch]

        e_b = pad_batch(e_w).to(device)
        q_b = pad_batch(q_w).to(device)

        with torch.no_grad():
            _, score_at, _, _ = at_model(e_b, txts, q_b)
            p_at = torch.sigmoid(score_at).detach().cpu().numpy()

            logit_aa, _, _ = aa_model(e_b, q_b)
            p_aa = torch.sigmoid(logit_aa).detach().cpu().numpy()

        # ── Confidence routing ──
        # 核心原则: 只在 AT 与 AA 强烈不一致时才路由
        # AT 低分 + AA 低分 → 两者都认为是负样本 → 不动
        # AT 低分 + AA 高分 → AT 可能漏报 → 用 AA
        for j in range(len(batch)):
            txt = txts[j]
            is_seen = txt in seen_vocab
            p_at_j = p_at[j].item()
            p_aa_j = p_aa[j].item()

            if is_seen and p_at_j < 0.01 and p_aa_j > 0.5:
                # AT 确信为负, AA 确信为正 → AT 错了 → 用 AA
                routed = p_aa_j
            elif is_seen and p_at_j < 0.05 and p_aa_j > 0.5:
                # 中等分歧 → 置信度加权 ensemble
                conf_at = abs(p_at_j - 0.5) * 2
                conf_aa = abs(p_aa_j - 0.5) * 2
                if conf_at + conf_aa > 0:
                    routed = (conf_at * p_at_j + conf_aa * p_aa_j) / (conf_at + conf_aa)
                else:
                    routed = (p_at_j + p_aa_j) / 2
            else:
                # 一致或 AA 也无信心 → 保持 AT
                routed = p_at_j

            at_all.append(p_at_j)
            aa_all.append(p_aa_j)
            routed_all.append(routed)
            labels_all.append(batch[j]['label'])

    zf.close()

    labels = np.array(labels_all)
    at_auc = roc_auc_score(labels, at_all)
    aa_auc = roc_auc_score(labels, aa_all)
    routed_auc = roc_auc_score(labels, routed_all)

    print(f'  AT alone:       {at_auc:.4f}')
    print(f'  AA alone:       {aa_auc:.4f}')
    print(f'  Routing(conf):  {routed_auc:.4f}')
    print(f'  Delta:          {routed_auc - at_auc:+.4f}')

    # Stats on which samples got routed
    at_arr = np.array(at_all)
    routed_arr = np.array(routed_all)
    changed = ~np.isclose(at_arr, routed_arr)
    improved = (routed_arr > at_arr) & changed
    regressed = (routed_arr < at_arr) & changed
    print(f'  Changed:  {changed.sum()}/{len(changed)} ({changed.mean()*100:.1f}%)')
    print(f'  Improved: {improved.sum()}/{len(changed)} ({improved.mean()*100:.1f}%)')
    print(f'  Regressed:{regressed.sum()}/{len(changed)} ({regressed.mean()*100:.1f}%)')

    return at_auc, routed_auc


print('=' * 60)
seen_at, seen_rt = evaluate('dev_seen', cfg.dev_seen_zip, cfg.dev_seen_csv)
unseen_at, unseen_rt = evaluate('dev_unseen', cfg.dev_unseen_zip, cfg.dev_unseen_csv)

print(f'\n{"="*60}')
print(f'SUMMARY')
print(f'  {"Method":<25} {"Seen AUC":>10} {"Unseen AUC":>12}')
print(f'  {"-"*25} {"-"*10} {"-"*12}')
print(f'  {"AT alone":<25} {seen_at:>10.4f} {unseen_at:>12.4f}')
print(f'  {"Confidence routing":<25} {seen_rt:>10.4f} {unseen_rt:>12.4f}')
print(f'  {"Delta":<25} {seen_rt-seen_at:>+10.4f} {unseen_rt-unseen_at:>+12.4f}')

"""Test: in AT uncertainty zone, use AT's OWN sim_a (audio-audio) instead of sim_t (text-query).
No separate AA model needed — AT already computes both.
"""
import sys, os, csv, io, zipfile
import numpy as np
import torch, torch.nn.functional as F
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

device = 'cuda'
cfg = DualConfig(); cfg.__post_init__()

at_model = AudioTextModel('', 256, unfreeze=0).to(device)
at_ckpt = torch.load('output/dual_at_v8_text/best.pt', map_location=device, weights_only=False)
at_model.load_state_dict(at_ckpt['model'], strict=False)
at_model.eval()

print(f'AT v8: seen={at_ckpt["auc_seen"]:.4f} unseen={at_ckpt["auc_unseen"]:.4f}')

seen_vocab = set()
with open(os.path.join(PATHS.root, 'train', 'train_label.csv')) as f:
    for r in csv.DictReader(f):
        seen_vocab.add(r.get('enroll_txt', '').lower().strip())


def read_wav(zf, pid, role):
    try: data = zf.read(f"wav/{pid}_{role}.wav")
    except KeyError: data = zf.read(f"wav/{pid}.wav")
    wav, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    wav = torch.from_numpy(wav).float()
    pad = int(0.5 * 16000)
    wav = F.pad(wav, (pad, pad))
    return wav


def evaluate(name, zip_path, csv_path):
    pairs = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            pairs.append({'id': r['id'], 'label': int(r['label']),
                          'enroll_txt': r.get('enroll_txt', '').lower()})

    print(f'\n[{name}] {len(pairs)} pairs')
    zf = zipfile.ZipFile(zip_path, 'r')

    all_sim_t, all_sim_a, all_labels, all_txts = [], [], [], []

    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w = [read_wav(zf, p['id'], 'enroll') for p in batch]
        q_w = [read_wav(zf, p['id'], 'query') for p in batch]
        txts = [p['enroll_txt'] for p in batch]
        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        with torch.no_grad():
            cos_ae, score_at, ea_at, et_at = at_model(eb, txts, qb)
            # Forward again to get sim_a
            ea = at_model.encoder(eb)
            eq = at_model.encoder(qb)
            sim_a = (ea * eq).sum(-1)  # audio-audio cosine within AT encoder
            p_t = torch.sigmoid(score_at).detach().cpu().numpy()
            sim_a_np = sim_a.detach().cpu().numpy()

        all_sim_t.extend(p_t.tolist())
        all_sim_a.extend(sim_a_np.tolist())
        all_labels.extend([p['label'] for p in batch])
        all_txts.extend(txts)

    zf.close()

    st = np.array(all_sim_t)
    sa = np.array(all_sim_a)
    lb = np.array(all_labels)

    # ── Zone analysis: compare sim_t vs sim_a in uncertainty zone ──
    print(f'\n  Zone: AT posterior in [0.2, 0.6] (uncertain)')
    mask = (st >= 0.2) & (st < 0.6)
    if mask.sum() > 0 and len(np.unique(lb[mask])) > 1:
        auc_t = roc_auc_score(lb[mask], st[mask])
        auc_a = roc_auc_score(lb[mask], sa[mask])
        print(f'  Samples: {mask.sum()}')
        print(f'  AT sim_t AUC:  {auc_t:.4f}')
        print(f'  AT sim_a AUC:  {auc_a:.4f}  (audio-audio from same encoder)')

    # ── Threshold sweep: blend sim_a into sim_t for uncertainty zone ──
    print(f'\n  Sweep: in uncertainty zone, replace p_t with sigmoid(sim_a * scale)')
    print(f'  {"low":>5} {"high":>5} {"n":>6} {"AT AUC":>8} {"Routed AUC":>8} {"Delta":>8}')
    print(f'  {"-"*5} {"-"*5} {"-"*6} {"-"*8} {"-"*8} {"-"*8}')

    for lo in [0.15, 0.2, 0.25, 0.3]:
        for hi in [0.5, 0.6, 0.7]:
            if lo >= hi: continue
            mask = (st >= lo) & (st < hi)
            routed = st.copy()
            # Calibrate sim_a: map to [0,1] range using its own stats
            # (not the same as sigmoid(sim_a*8) since sim_a distribution differs from sim_t)
            sa_calib = (sa - sa.min()) / (sa.max() - sa.min() + 1e-10)
            for j in range(len(routed)):
                if mask[j]:
                    routed[j] = sa_calib[j]

            auc_t = roc_auc_score(lb, st)
            auc_r = roc_auc_score(lb, routed)
            print(f'  {lo:.2f} {hi:.2f} {mask.sum():>6} {auc_t:>8.4f} {auc_r:>8.4f} {auc_r-auc_t:>+8.4f}')

    # ── Better: use sim_a directly for all seen words (calibrated) ──
    txt_arr = np.array(all_txts)
    sa_for_seen = sa.copy()
    # Calibrate with simple scaling
    import sklearn.linear_model as lm
    # Use logit of sim_t as target, learn from confident region
    confident = (st < 0.2) | (st > 0.8)
    if confident.sum() > 100:
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        # Use sim_a to predict whether sim_t says positive (>0.5)
        clf.fit(sa[confident].reshape(-1, 1), (st[confident] > 0.5).astype(int))
        sa_prob = clf.predict_proba(sa.reshape(-1, 1))[:, 1]
        routed2 = st.copy()
        for j in range(len(routed2)):
            if mask[j]:  # uncertainty zone
                routed2[j] = sa_prob[j]
        auc_r2 = roc_auc_score(lb, routed2)
        print(f'\n  Logistic regression (sim_a → predict AT label, use in uncertainty zone):')
        print(f'    Routed AUC: {auc_r2:.4f} (delta={auc_r2-auc_t:+.4f})')


evaluate('dev_seen', cfg.dev_seen_zip, cfg.dev_seen_csv)
evaluate('dev_unseen', cfg.dev_unseen_zip, cfg.dev_unseen_csv)

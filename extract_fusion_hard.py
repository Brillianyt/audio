"""Extract fusion v1 hard negatives for manual inspection."""
import sys, os, csv, io, zipfile
import torch, torch.nn.functional as F
import soundfile as sf
import numpy as np
sys.path.insert(0, 'baseline'); sys.path.insert(0, '.')
from config import PATHS
from train_fusion import AudioTextFusionModel, FusionConfig, PairDataset, collate_fusion, load_pairs

device = 'cuda'
cfg = FusionConfig(); cfg.__post_init__()

model = AudioTextFusionModel(256, unfreeze=0).to(device)
ckpt = torch.load('output/fusion_v1/latest.pt', map_location=device, weights_only=False)
model.load_state_dict(ckpt['model'], strict=False)
model.eval()
ep = ckpt.get("epoch", "?")
print(f'Loaded fusion v1: epoch={ep}')

out_dir = 'Debug/fusion_hard'
os.makedirs(out_dir, exist_ok=True)

for name, zp, cp in [('seen', cfg.dev_seen_zip, cfg.dev_seen_csv),
                       ('unseen', cfg.dev_unseen_zip, cfg.dev_unseen_csv)]:
    pairs = load_pairs(cp)
    pos = [p for p in pairs if p['label'] == 1]
    neg = [p for p in pairs if p['label'] == 0]
    print(f'\n[{name}] {len(pos)} pos, {len(neg)} neg')

    results = []
    zf = zipfile.ZipFile(zp, 'r')

    for i in range(0, len(pairs), 128):
        batch = pairs[i:i+128]
        e_w, q_w, txts = [], [], []
        for p in batch:
            eid = p.get('enroll_id', p['id'])
            qid = p.get('query_id', p['id'])
            for role, pid, lst in [('enroll', eid, e_w), ('query', qid, q_w)]:
                data = zf.read(f'wav/{pid}_{role}.wav')
                w, sr = sf.read(io.BytesIO(data), dtype='float32')
                if w.ndim > 1: w = w.mean(axis=1)
                if sr != 16000:
                    import torchaudio
                    w = torchaudio.functional.resample(torch.from_numpy(w).unsqueeze(0), sr, 16000).squeeze(0).numpy()
                lst.append(torch.from_numpy(w).float())
            txts.append(p['enroll_txt'].lower())

        ml = max(max(w.shape[0] for w in e_w), max(w.shape[0] for w in q_w))
        eb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in e_w]).to(device)
        qb = torch.stack([F.pad(w, (0, ml-w.shape[0])) for w in q_w]).to(device)

        with torch.no_grad():
            logit, _, _ = model(eb, txts, qb)
            p = torch.sigmoid(logit).detach().cpu().numpy()

        for j in range(len(batch)):
            results.append((p[j].item(), batch[j]['id'], batch[j]['enroll_txt'], batch[j]['label']))

    zf.close()

    # Hard negatives: FN (label=1, p<0.3) + FP (label=0, p>0.5)
    hard = [r for r in results if (r[3]==1 and r[0]<0.3) or (r[3]==0 and r[0]>0.5)]
    hard.sort(key=lambda r: abs(r[0]-0.5), reverse=True)
    print(f'  Hard samples: {len(hard)}')

    for k, (p_val, pid, txt, label) in enumerate(hard[:10]):
        tag = 'FN' if label==1 else 'FP'
        with zipfile.ZipFile(zp, 'r') as zf:
            for role in ['enroll', 'query']:
                try:
                    data = zf.read(f'wav/{pid}_{role}.wav')
                except:
                    continue
                fname = f'{name}_{tag}_{k+1}_{pid}_{role}.wav'
                with open(os.path.join(out_dir, fname), 'wb') as f:
                    f.write(data)
        print(f'  {k+1}. {tag} p={p_val:.4f} id={pid} txt=\"{txt}\"')

print(f'\nSaved to {out_dir}/')

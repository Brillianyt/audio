"""Simple inference: AA Hybrid v2 + AT backup → submission.csv"""
import torch, os, sys, csv, io, zipfile, numpy as np
import torch.nn.functional as F
import soundfile as sf
sys.path.insert(0, 'baseline'); from config import PATHS
import train_aa_hybrid, train_at_orig

device = 'cuda'

# Load AA
aa = train_aa_hybrid.AAHybridModel(256, unfreeze=3).to(device)
aa.load_state_dict(torch.load('output/aa_hybrid/best.pt', map_location=device, weights_only=False)['model'], strict=False)
aa.eval()

# Load AT
at = train_at_orig.AudioTextModel(256, unfreeze=4).to(device)
at.load_state_dict(torch.load('output/backup/at_best.pt', map_location=device, weights_only=False)['model'], strict=False)
at.eval()

def read_wav(zf, pid, role):
    for name in [f'wav/{pid}_{role}.wav', f'wav/{pid}.wav']:
        try: data = zf.read(name); break
        except KeyError: continue
    wav, sr = sf.read(io.BytesIO(data), dtype='float32', always_2d=False)
    if wav.ndim > 1: wav = wav.mean(axis=1)
    if sr != 16000:
        import torchaudio
        wav = torchaudio.functional.resample(torch.from_numpy(wav).unsqueeze(0), sr, 16000).squeeze(0).numpy()
    ms = int(1.5 * 16000)
    if len(wav) > ms: wav = wav[:ms]
    wav = torch.from_numpy(wav).float()
    # 补静音：前后各加 0.5 秒
    pad = int(0.5 * 16000)
    wav = F.pad(wav, (pad, pad))
    return wav

def process(prefix, csv_path, zip_path):
    zf = zipfile.ZipFile(zip_path, 'r')
    results = []
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    
    for i, row in enumerate(rows):
        pid = row['id']; txt = row['enroll_txt'].lower()
        e = read_wav(zf, pid, 'enroll').unsqueeze(0).to(device)
        q = read_wav(zf, pid, 'query').unsqueeze(0).to(device)
        ml = max(e.shape[-1], q.shape[-1])
        if e.shape[-1] < ml: e = F.pad(e, (0, ml - e.shape[-1]))
        if q.shape[-1] < ml: q = F.pad(q, (0, ml - q.shape[-1]))
        
        with torch.no_grad():
            score_aa, *_ = aa(e, q)
            p_aa = torch.sigmoid(score_aa * 3.0).item()
            
            _, _, et = at(e, [txt])
            eq = at.encoder(q)
            cs = (et * eq).sum(-1)
            p_at = torch.sigmoid(cs * 8.0).item()
            
            # Simple ensemble: confidence-weighted
            conf_aa = abs(p_aa - 0.5) * 2
            conf_at = abs(p_at - 0.5) * 2
            if conf_aa + conf_at > 0:
                posterior = (conf_aa * p_aa + conf_at * p_at) / (conf_aa + conf_at)
            else:
                posterior = (p_aa + p_at) / 2
        
        results.append((f'{prefix}_{pid.replace("pair_","")}', posterior))
        
        if (i+1) % 5000 == 0:
            print(f'  {prefix}: {i+1}/{len(rows)}')
    
    zf.close()
    return results

print('Processing seen...')
seen = process('seen_pair',
    os.path.join(PATHS.root, 'evalcsv_without_label', 'eval_seen_without_label.csv'),
    os.path.join(PATHS.root, 'eval', 'eval_seen', 'wav.zip'))

print('Processing unseen...')
unseen = process('unseen_pair',
    os.path.join(PATHS.root, 'evalcsv_without_label', 'eval_unseen_without_label.csv'),
    os.path.join(PATHS.root, 'eval', 'eval_unseen', 'wav.zip'))

all_results = seen + unseen

with open('submission.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'posterior'])
    for pid, post in all_results:
        writer.writerow([pid, f'{post:.6f}'])

ps = np.array([p for _, p in all_results])
print(f'Saved {len(all_results)} rows')
print(f'Mean={ps.mean():.4f} Std={ps.std():.4f} >0.5={(ps>0.5).sum()}/{len(ps)}')

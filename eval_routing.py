"""Compare AT alone vs routing on dev set (has labels)."""
import argparse, csv, io, os, sys, time, zipfile
import numpy as np
import torch
import torch.nn.functional as F
import soundfile as sf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PATHS
from audio_quality import compute_damage_score_batch, classify_damage

# Import root train_dual for AT
import importlib.util
_td_spec = importlib.util.spec_from_file_location(
    "train_dual_root", os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dual.py"))
_td = importlib.util.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td)
AudioTextModel = _td.AudioTextModel
DualConfig = _td.Config

from baseline.train_whisper_v3 import MultiTaskWhisperKWSV3


def load_pairs(csv_path):
    rows = []
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({"id": r["id"], "label": int(r["label"]),
                         "enroll_txt": r.get("enroll_txt", "").lower()})
    return rows


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


def load_train_vocab():
    csv_path = os.path.join(PATHS.root, "train", "train_label.csv")
    words = set()
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            words.add(r.get("enroll_txt", "").lower().strip())
    return words


@torch.no_grad()
def evaluate(device, batch_size=128):
    seen_vocab = load_train_vocab()
    print(f"[vocab] {len(seen_vocab)} train words")

    # Load models
    cfg = DualConfig()
    cfg.__post_init__()

    at_model = AudioTextModel("", 256, unfreeze=0).to(device)
    at_ckpt = torch.load("output/dual_at_v8_text/best.pt", map_location=device, weights_only=False)
    at_model.load_state_dict(at_ckpt["model"], strict=False)
    at_model.eval()
    print(f"[AT] loaded (unseen={at_ckpt.get('auc_unseen','?'):.4f})")

    aa_model = MultiTaskWhisperKWSV3("base", 256, unfreeze_layers=0).to(device)
    aa_ckpt = torch.load("output/backup_final/aa_v3_best_seen07934.pt", map_location=device, weights_only=False)
    aa_model.load_state_dict(aa_ckpt["model"], strict=False)
    aa_model.eval()
    print(f"[AA] loaded")

    def eval_subset(name, zip_path, csv_path):
        pairs = load_pairs(csv_path)
        print(f"\n[{name}] {len(pairs)} pairs")

        all_labels = []
        all_at = []       # AT alone
        all_aa = []       # AA alone
        all_ensemble = []  # confidence-weighted ensemble
        all_routed = []    # full routing
        all_damage = []

        zf = zipfile.ZipFile(zip_path, "r")

        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            e_wavs, q_wavs, txts, labels = [], [], [], []
            for bp in batch:
                e_wavs.append(read_wav(zf, bp["id"], "enroll"))
                q_wavs.append(read_wav(zf, bp["id"], "query"))
                txts.append(bp["enroll_txt"])
                labels.append(bp["label"])

            e_batch = pad_batch(e_wavs).to(device)
            q_batch = pad_batch(q_wavs).to(device)

            # Damage scores
            damage_scores = compute_damage_score_batch(q_wavs)

            # AT forward
            _, score_at, _, _ = at_model(e_batch, txts, q_batch)
            p_at = torch.sigmoid(score_at).cpu().numpy()

            # AA forward
            logit_aa, _, _ = aa_model(e_batch, q_batch)
            p_aa = torch.sigmoid(logit_aa).cpu().numpy()

            for j in range(len(batch)):
                p_at_j = p_at[j].item()
                p_aa_j = p_aa[j].item()
                ds = damage_scores[j]
                txt = txts[j]

                # ── AT alone ──
                all_at.append(p_at_j)

                # ── AA alone ──
                all_aa.append(p_aa_j)

                # ── Ensemble (always) ──
                conf_at = abs(p_at_j - 0.5) * 2
                conf_aa = abs(p_aa_j - 0.5) * 2
                if conf_at + conf_aa > 0:
                    ens = (conf_at * p_at_j + conf_aa * p_aa_j) / (conf_at + conf_aa)
                else:
                    ens = (p_at_j + p_aa_j) / 2
                all_ensemble.append(ens)

                # ── Routing ──
                level = classify_damage(ds)
                if level == "clean":
                    routed = p_at_j
                elif level == "moderate":
                    routed = ens
                else:  # severe
                    if txt in seen_vocab:
                        routed = p_aa_j
                    else:
                        routed = p_at_j * 0.8
                all_routed.append(routed)

                all_labels.append(labels[j])
                all_damage.append(ds)

            if (i + batch_size) % 10000 == 0 or (i + batch_size) >= len(pairs):
                print(f"  {min(i + batch_size, len(pairs))}/{len(pairs)} "
                      f"damage mean={np.mean(all_damage[-5000:]):.3f}")

        zf.close()

        labels = np.array(all_labels)
        at_auc = roc_auc_score(labels, all_at)
        aa_auc = roc_auc_score(labels, all_aa)
        ens_auc = roc_auc_score(labels, all_ensemble)
        routed_auc = roc_auc_score(labels, all_routed)

        damage = np.array(all_damage)
        print(f"\n  [{name}] AUC comparison:")
        print(f"    AT alone:         {at_auc:.4f}")
        print(f"    AA alone:         {aa_auc:.4f}")
        print(f"    AT+AA ensemble:   {ens_auc:.4f}")
        print(f"    Routing (ours):   {routed_auc:.4f}")
        print(f"  Damage: mean={damage.mean():.3f} std={damage.std():.3f} "
              f"p50={np.median(damage):.3f} p95={np.percentile(damage,95):.3f}")
        print(f"  Routing breakdown:")
        for level, thresh in [("clean", 0.30), ("moderate", 0.42), ("severe", 1.0)]:
            cnt = sum(1 for d in damage if (d < thresh if level == "clean" else
                                            (d >= 0.30 and d < 0.42) if level == "moderate" else d >= 0.42))
            print(f"    {level}: {cnt}/{len(damage)} ({cnt/len(damage)*100:.1f}%)")

        return at_auc, aa_auc, ens_auc, routed_auc

    # Dev seen
    seen = eval_subset("dev_seen", cfg.dev_seen_zip, cfg.dev_seen_csv)
    # Dev unseen
    unseen = eval_subset("dev_unseen", cfg.dev_unseen_zip, cfg.dev_unseen_csv)

    print(f"\n{'='*60}")
    print(f"SUMMARY:")
    print(f"  {'Method':<25} {'Seen AUC':>10} {'Unseen AUC':>12}")
    print(f"  {'-'*25} {'-'*10} {'-'*12}")
    print(f"  {'AT alone':<25} {seen[0]:>10.4f} {unseen[0]:>12.4f}")
    print(f"  {'AA alone':<25} {seen[1]:>10.4f} {unseen[1]:>12.4f}")
    print(f"  {'AT+AA ensemble':<25} {seen[2]:>10.4f} {unseen[2]:>12.4f}")
    print(f"  {'Routing (ours)':<25} {seen[3]:>10.4f} {unseen[3]:>12.4f}")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate(device, batch_size=128)

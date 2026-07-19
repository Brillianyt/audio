from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Subset

from config import AUDIO, PATHS, TRAIN
from data import PairDataset, collate, load_pairs
from model import SiameseKWS


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=TRAIN.epochs)
    ap.add_argument("--bs", type=int, default=TRAIN.batch_size)
    ap.add_argument("--lr", type=float, default=TRAIN.lr)
    ap.add_argument("--subset", type=int, default=TRAIN.train_subset,
                    help="训练子集大小，越小分数通常越低")
    ap.add_argument("--workers", type=int, default=TRAIN.num_workers)
    ap.add_argument("--out", type=str, default=os.path.join(PATHS.ckpt_dir, "best.pt"))
    return ap.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    probs, labels = [], []
    for e, q, y, _ in loader:
        e, q = e.to(device), q.to(device)
        logit = model(e, q)
        probs.append(torch.sigmoid(logit).cpu().numpy())
        labels.append(y.numpy())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


def main():
    args = parse_args()
    torch.manual_seed(TRAIN.seed)
    np.random.seed(TRAIN.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(PATHS.ckpt_dir, exist_ok=True)

    all_pairs = load_pairs(PATHS.train_csv, with_label=True)
    n = min(args.subset, len(all_pairs))
    idx = np.random.default_rng(TRAIN.seed).permutation(len(all_pairs))[:n]
    train_pairs = [all_pairs[i] for i in idx]
    print(f"train: {n} / {len(all_pairs)} pairs")

    train_ds = PairDataset(train_pairs, PATHS.train_zip, AUDIO)
    train_loader = DataLoader(train_ds, batch_size=args.bs, shuffle=True,
                              num_workers=args.workers, collate_fn=collate,
                              pin_memory=True, drop_last=True)

    def dev_loader(zip_p, csv_p):
        ds = PairDataset(load_pairs(csv_p, True), zip_p, AUDIO)
        return DataLoader(ds, batch_size=args.bs, shuffle=False,
                          num_workers=args.workers, collate_fn=collate)

    dev_seen = dev_loader(PATHS.dev_seen_zip, PATHS.dev_seen_csv)
    dev_unseen = dev_loader(PATHS.dev_unseen_zip, PATHS.dev_unseen_csv)

    model = SiameseKWS(AUDIO.n_mels, TRAIN.embed_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,} ({n_params/1e6:.2f}M)")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(TRAIN.pos_weight, device=device))

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        for it, (e, q, y, _) in enumerate(train_loader, 1):
            e, q, y = e.to(device), q.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(e, q), y)
            loss.backward()
            opt.step()
            loss_sum += loss.item()
            if it % TRAIN.log_every == 0:
                print(f"  ep{ep} {it}/{len(train_loader)} loss={loss_sum/it:.4f}")

        auc_s = evaluate(model, dev_seen, device)
        auc_u = evaluate(model, dev_unseen, device)
        mean = (auc_s + auc_u) / 2
        print(f"[epoch {ep}] seen={auc_s:.4f} unseen={auc_u:.4f} "
              f"mean={mean:.4f} ({time.time()-t0:.0f}s)")

        if mean > best:
            best = mean
            torch.save({"model": model.state_dict(),
                        "embed_dim": TRAIN.embed_dim,
                        "auc": mean}, args.out)
            print(f"  saved -> {args.out}")

    print(f"done. best dev mean AUC = {best:.4f}")


if __name__ == "__main__":
    main()

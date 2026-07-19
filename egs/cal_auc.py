"""
Calculate AUC metric for wake word detection from a CSV file.

Usage:
    python cal_auc.py --csv <csv_path>

CSV format requirements:
    - Must contain columns: id, posterior, label
    - id: sample identifier (e.g., 000001)
    - posterior: wake probability, float in [0, 1] (e.g., 0.5905、0.590493、0.59049325)
    - label: ground truth, 0 (non-wake) or 1 (wake)

Example:
    python cal_auc.py --csv example.csv
"""

import csv
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from tabulate import tabulate


def main():
    parser = argparse.ArgumentParser(description='Calculate AUC metric for wake word detection')
    parser.add_argument('--csv', type=str, default='example.csv', help='CSV file path')
    args = parser.parse_args()

    posteriors = []
    labels = []
    with open(args.csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            posteriors.append(float(row['posterior']))
            labels.append(int(row['label']))

    posteriors = np.array(posteriors)
    labels = np.array(labels)

    total = len(labels)
    num_positive = int(np.sum(labels == 1))
    num_negative = total - num_positive
    auc = round(roc_auc_score(labels, posteriors), 4)

    table = [
        ["Total Samples", total],
        ["Positive (Wake)", num_positive],
        ["Negative (Non-wake)", num_negative],
        ["Posterior Mean", f"{np.mean(posteriors):.4f}"],
        ["Posterior Std", f"{np.std(posteriors):.4f}"],
        ["AUC", auc],
    ]
    print("\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="grid") + "\n")


if __name__ == '__main__':
    main()

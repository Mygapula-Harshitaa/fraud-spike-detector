"""
Stage A: unsupervised anomaly scorer on merchant-minute aggregates.

Fits an IsolationForest on per-merchant, per-minute traffic aggregates
(txn_count, decline_rate, unique_devices, unique_card_bins, avg_amount,
txn_count_zscore_60min) WITHOUT using is_attack as an input. This stage is meant
to catch spike/anomaly patterns without needing labeled data — useful for
flagging novel attack shapes that Stage B (the supervised classifier) was never
trained on.

is_attack / attack_txn_count are used ONLY for evaluation after scoring, never
as model inputs.

Saves:
  - the trained IsolationForest model (joblib)
  - a metrics report (txt) evaluating anomaly flags against true attack minutes
  - a per-merchant example plot: anomaly score over time with true attack
    windows shaded, for the merchant with the most attack activity in test
  - a precision/recall-style summary at minute level and at attack-window level

Usage:
    python train_stage_a.py --train out/features_train_merchant_minute.csv \
                             --test out/features_test_merchant_minute.csv \
                             --out out/stage_a
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

# Columns fed to the IsolationForest. Deliberately excludes attack_txn_count
# (label, eval-only) and identifiers.
FEATURE_COLS = [
    "txn_count",
    "decline_rate",
    "unique_devices",
    "unique_card_bins",
    "avg_amount",
    "txn_count_zscore_60min",
]


def load_xy(path: str):
    df = pd.read_csv(path)
    df["minute"] = pd.to_datetime(df["minute"])
    X = df[FEATURE_COLS].fillna(0)
    # ground truth for eval only: was this merchant-minute touched by an attack?
    y = (df["attack_txn_count"] > 0).astype(int)
    return df, X, y


def evaluate_at_contamination(y_true, anomaly_flag):
    tp = int(((anomaly_flag == 1) & (y_true == 1)).sum())
    fp = int(((anomaly_flag == 1) & (y_true == 0)).sum())
    fn = int(((anomaly_flag == 0) & (y_true == 1)).sum())
    tn = int(((anomaly_flag == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to features_train_merchant_minute.csv")
    ap.add_argument("--test", required=True, help="path to features_test_merchant_minute.csv")
    ap.add_argument("--out", required=True, help="output prefix/dir for model + plots + metrics")
    ap.add_argument(
        "--contamination",
        type=float,
        default=None,
        help="expected proportion of anomalous minutes in training data; "
        "defaults to the true positive rate observed in the train set",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    train_df, X_train, y_train = load_xy(args.train)
    test_df, X_test, y_test = load_xy(args.test)

    print(f"Train: {X_train.shape}, attack-minute rate {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, attack-minute rate {y_test.mean():.4f}")

    # IsolationForest's `contamination` sets the internal score threshold for
    # what counts as "anomalous". We default it to roughly the true attack-minute
    # rate seen in training (label used ONLY to pick this hyperparameter, not fed
    # to the model as a feature).
    contamination = args.contamination if args.contamination is not None else max(min(y_train.mean(), 0.5), 1e-4)
    print(f"Using contamination = {contamination:.5f}")

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )

    print("Training IsolationForest...")
    model.fit(X_train)

    print("Scoring test set...")
    # decision_function: higher = more normal, lower = more anomalous
    raw_scores = model.decision_function(X_test)
    # anomaly_score: flip sign so higher = more anomalous (more intuitive for reporting)
    anomaly_score = -raw_scores
    pred = model.predict(X_test)  # -1 = anomaly, 1 = normal
    anomaly_flag = (pred == -1).astype(int)

    minute_metrics = evaluate_at_contamination(y_test.to_numpy(), anomaly_flag)

    # ---- Attack-window-level detection: for each true attack window (contiguous
    # attack minutes per merchant), did we flag at least one minute in it? ----
    eval_df = test_df.copy()
    eval_df["anomaly_score"] = anomaly_score
    eval_df["anomaly_flag"] = anomaly_flag
    eval_df["is_attack_minute"] = y_test.to_numpy()

    # Build contiguous attack-minute groups per merchant for window-level eval
    window_detection = None
    windows_total = 0
    windows_detected = 0
    for merchant_id, g in eval_df.sort_values("minute").groupby("merchant_id"):
        is_attack = g["is_attack_minute"].to_numpy()
        flags = g["anomaly_flag"].to_numpy()
        # find contiguous runs of is_attack == 1
        i = 0
        n = len(is_attack)
        while i < n:
            if is_attack[i] == 1:
                j = i
                while j < n and is_attack[j] == 1:
                    j += 1
                windows_total += 1
                if flags[i:j].sum() > 0:
                    windows_detected += 1
                i = j
            else:
                i += 1

    if windows_total > 0:
        window_detection = {
            "total_attack_windows": windows_total,
            "windows_with_at_least_one_alert": windows_detected,
            "window_detection_rate": windows_detected / windows_total,
        }

    metrics = {
        "contamination_used": float(contamination),
        "minute_level": minute_metrics,
        "attack_window_level": window_detection,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_attack_minute_rate": float(y_train.mean()),
        "test_attack_minute_rate": float(y_test.mean()),
    }

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    metrics_txt_path = os.path.join(args.out, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("=== Stage A — IsolationForest Anomaly Scorer (merchant-minute) ===\n\n")
        f.write(f"Contamination used: {contamination:.5f}\n\n")
        f.write("Minute-level detection (each merchant-minute treated independently):\n")
        f.write(f"  Precision: {minute_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {minute_metrics['recall']:.4f}\n")
        f.write(f"  F1:        {minute_metrics['f1']:.4f}\n")
        f.write(f"  TP={minute_metrics['tp']} FP={minute_metrics['fp']} "
                 f"FN={minute_metrics['fn']} TN={minute_metrics['tn']}\n\n")
        if window_detection:
            f.write("Attack-window-level detection (contiguous attack minutes per merchant):\n")
            f.write(f"  Total attack windows in test set: {window_detection['total_attack_windows']}\n")
            f.write(f"  Windows with >=1 flagged minute:   {window_detection['windows_with_at_least_one_alert']}\n")
            f.write(f"  Window detection rate:             {window_detection['window_detection_rate']:.4f}\n")

    print(f"Saved metrics to {metrics_path} and {metrics_txt_path}")

    # ---- Save model ----
    model_path = os.path.join(args.out, "stage_a_isoforest_model.joblib")
    joblib.dump({"model": model, "feature_cols": FEATURE_COLS, "contamination": contamination}, model_path)
    print(f"Saved model to {model_path}")

    # ---- Plot: anomaly score distribution, normal vs attack minutes ----
    plt.figure(figsize=(7, 5))
    plt.hist(anomaly_score[y_test.to_numpy() == 0], bins=60, alpha=0.6, label="Normal minutes", density=True)
    plt.hist(anomaly_score[y_test.to_numpy() == 1], bins=60, alpha=0.6, label="Attack minutes", density=True)
    plt.xlabel("Anomaly score (higher = more anomalous)")
    plt.ylabel("Density")
    plt.title("Stage A — Anomaly Score Distribution")
    plt.legend()
    plt.tight_layout()
    dist_plot_path = os.path.join(args.out, "score_distribution.png")
    plt.savefig(dist_plot_path, dpi=150)
    plt.close()
    print(f"Saved score distribution plot to {dist_plot_path}")

    # ---- Plot: example timeline for the merchant with the most attack activity ----
    attack_counts_by_merchant = eval_df[eval_df["is_attack_minute"] == 1]["merchant_id"].value_counts()
    if len(attack_counts_by_merchant) > 0:
        top_merchant = attack_counts_by_merchant.index[0]
        mdf = eval_df[eval_df["merchant_id"] == top_merchant].sort_values("minute")

        plt.figure(figsize=(12, 5))
        plt.plot(mdf["minute"], mdf["anomaly_score"], color="steelblue", linewidth=0.8, label="Anomaly score")

        # shade true attack windows
        attack_mask = mdf["is_attack_minute"].to_numpy()
        minutes = mdf["minute"].to_numpy()
        in_window = False
        start = None
        for k in range(len(attack_mask)):
            if attack_mask[k] == 1 and not in_window:
                in_window = True
                start = minutes[k]
            if attack_mask[k] == 0 and in_window:
                in_window = False
                plt.axvspan(start, minutes[k], color="red", alpha=0.25)
        if in_window:
            plt.axvspan(start, minutes[-1], color="red", alpha=0.25)

        plt.xlabel("Time")
        plt.ylabel("Anomaly score")
        plt.title(f"Stage A — Anomaly Score Over Time ({top_merchant}), red = true attack windows")
        plt.legend()
        plt.xticks(rotation=30)
        plt.tight_layout()
        timeline_plot_path = os.path.join(args.out, "example_merchant_timeline.png")
        plt.savefig(timeline_plot_path, dpi=150)
        plt.close()
        print(f"Saved example merchant timeline plot to {timeline_plot_path}")

    print("\nDone. Summary:")
    print(f"  Minute-level: precision={minute_metrics['precision']:.4f} "
          f"recall={minute_metrics['recall']:.4f} f1={minute_metrics['f1']:.4f}")
    if window_detection:
        print(f"  Attack-window detection rate: {window_detection['window_detection_rate']:.4f} "
              f"({window_detection['windows_with_at_least_one_alert']}/{window_detection['total_attack_windows']})")


if __name__ == "__main__":
    main()
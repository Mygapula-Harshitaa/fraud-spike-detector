"""
Combined Stage A + Stage B fraud-spike detection pipeline.

Loads the trained Stage A (IsolationForest, merchant-minute) and Stage B
(XGBoost, transaction-level) models and evaluates several ways of combining
them on the test set:

  1. stage_b_only   - alert if Stage B probability >= its own tuned threshold.
  2. stage_a_only    - alert if the transaction's merchant-minute was flagged
                        anomalous by Stage A.
  3. combined_or      - alert if EITHER stage fires (maximizes recall).
  4. combined_and     - alert if BOTH stages fire (maximizes precision).
  5. cascade          - the recommended production design: Stage A is a cheap,
                        label-free first pass over every merchant-minute.
                        Transactions inside a minute Stage A already flagged
                        get a LOWERED Stage B bar (more sensitive, since Stage A
                        already raised suspicion); transactions elsewhere still
                        need to clear Stage B's normal, higher bar. This keeps
                        precision reasonable while catching more of what Stage A
                        alone would flag, without needing every transaction to
                        clear the same strict threshold.

Metrics are reported per-transaction (precision/recall/F1/false-positive count)
and per-attack-window (did we alert at least once inside each true attack
window) for every strategy, plus a comparison chart.

Usage:
    python combine_stages.py \
        --txn_test out/features_test_transaction.csv \
        --merchant_minute_test out/features_test_merchant_minute.csv \
        --stage_a_model out/stage_a/stage_a_isoforest_model.joblib \
        --stage_b_model out/stage_b/stage_b_xgb_model.joblib \
        --out out/combined
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

NON_FEATURE_COLS = ["payment_id", "merchant_id", "created_at", "is_attack", "attack_window_id", "minute"]


def window_level_detection_rate(df: pd.DataFrame, alert_col: str) -> dict:
    """For each true attack window (attack_window_id), did we fire >=1 alert in it?"""
    attack_rows = df[df["is_attack"] == 1]
    windows = attack_rows["attack_window_id"].dropna().unique()
    if len(windows) == 0:
        return {"total_attack_windows": 0, "windows_with_at_least_one_alert": 0, "window_detection_rate": None}
    detected = 0
    for w in windows:
        wdf = attack_rows[attack_rows["attack_window_id"] == w]
        if wdf[alert_col].sum() > 0:
            detected += 1
    return {
        "total_attack_windows": int(len(windows)),
        "windows_with_at_least_one_alert": int(detected),
        "window_detection_rate": float(detected / len(windows)),
    }


def txn_level_metrics(y_true, y_pred) -> dict:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1), "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--txn_test", required=True, help="path to features_test_transaction.csv")
    ap.add_argument("--merchant_minute_test", required=True, help="path to features_test_merchant_minute.csv")
    ap.add_argument("--stage_a_model", required=True, help="path to stage_a_isoforest_model.joblib")
    ap.add_argument("--stage_b_model", required=True, help="path to stage_b_xgb_model.joblib")
    ap.add_argument("--out", required=True, help="output dir for metrics + plots")
    ap.add_argument(
        "--cascade_ratio",
        type=float,
        default=0.4,
        help="fraction of Stage B's own threshold used as the lowered bar inside "
        "Stage-A-flagged minutes (default 0.4 = 40%% of the normal threshold)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading models...")
    stage_a_bundle = joblib.load(args.stage_a_model)
    stage_b_bundle = joblib.load(args.stage_b_model)
    a_model = stage_a_bundle["model"]
    a_feature_cols = stage_a_bundle["feature_cols"]
    b_model = stage_b_bundle["model"]
    b_feature_cols = stage_b_bundle["feature_cols"]
    b_threshold = stage_b_bundle["threshold"]

    print("Loading test data...")
    txn_df = pd.read_csv(args.txn_test)
    txn_df["created_at"] = pd.to_datetime(txn_df["created_at"])
    txn_df["minute"] = txn_df["created_at"].dt.floor("min")

    mm_df = pd.read_csv(args.merchant_minute_test)
    mm_df["minute"] = pd.to_datetime(mm_df["minute"])

    print("Scoring Stage A (merchant-minute anomaly flags)...")
    a_X = mm_df[a_feature_cols].fillna(0)
    mm_df["stage_a_flag"] = (a_model.predict(a_X) == -1).astype(int)

    print("Scoring Stage B (per-transaction probability)...")
    b_X = txn_df[b_feature_cols]
    txn_df["stage_b_prob"] = b_model.predict_proba(b_X)[:, 1]

    print("Merging Stage A flags onto transactions by (merchant_id, minute)...")
    merged = txn_df.merge(
        mm_df[["merchant_id", "minute", "stage_a_flag"]],
        on=["merchant_id", "minute"],
        how="left",
    )
    merged["stage_a_flag"] = merged["stage_a_flag"].fillna(0).astype(int)

    y_true = merged["is_attack"].astype(int).to_numpy()

    # ---- Strategy 1: Stage B only ----
    merged["alert_stage_b_only"] = (merged["stage_b_prob"] >= b_threshold).astype(int)

    # ---- Strategy 2: Stage A only ----
    merged["alert_stage_a_only"] = merged["stage_a_flag"]

    # ---- Strategy 3: combined OR ----
    merged["alert_combined_or"] = (
        (merged["alert_stage_b_only"] == 1) | (merged["alert_stage_a_only"] == 1)
    ).astype(int)

    # ---- Strategy 4: combined AND ----
    merged["alert_combined_and"] = (
        (merged["alert_stage_b_only"] == 1) & (merged["alert_stage_a_only"] == 1)
    ).astype(int)

    # ---- Strategy 5: cascade (recommended design) ----
    cascade_threshold = b_threshold * args.cascade_ratio
    merged["alert_cascade"] = np.where(
        merged["stage_a_flag"] == 1,
        (merged["stage_b_prob"] >= cascade_threshold).astype(int),
        (merged["stage_b_prob"] >= b_threshold).astype(int),
    )

    strategies = {
        "stage_a_only": "alert_stage_a_only",
        "stage_b_only": "alert_stage_b_only",
        "combined_or": "alert_combined_or",
        "combined_and": "alert_combined_and",
        "cascade": "alert_cascade",
    }

    results = {}
    for name, col in strategies.items():
        y_pred = merged[col].to_numpy()
        txn_metrics = txn_level_metrics(y_true, y_pred)
        window_metrics = window_level_detection_rate(merged, col)
        results[name] = {"transaction_level": txn_metrics, "attack_window_level": window_metrics}

    metrics = {
        "stage_b_threshold": float(b_threshold),
        "cascade_threshold": float(cascade_threshold),
        "cascade_ratio": float(args.cascade_ratio),
        "n_test_transactions": int(len(merged)),
        "n_test_attack_transactions": int(y_true.sum()),
        "strategies": results,
    }

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    metrics_txt_path = os.path.join(args.out, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("=== Combined Stage A + Stage B Pipeline ===\n\n")
        f.write(f"Stage B threshold: {b_threshold:.4f}\n")
        f.write(f"Cascade threshold (inside Stage-A-flagged minutes): {cascade_threshold:.4f} "
                 f"({args.cascade_ratio:.0%} of Stage B's normal threshold)\n\n")
        for name in strategies:
            r = results[name]
            t = r["transaction_level"]
            w = r["attack_window_level"]
            f.write(f"--- {name} ---\n")
            f.write(f"  Transaction-level: precision={t['precision']:.4f} recall={t['recall']:.4f} "
                     f"f1={t['f1']:.4f} (TP={t['tp']} FP={t['fp']} FN={t['fn']})\n")
            if w["window_detection_rate"] is not None:
                f.write(f"  Attack-window detection rate: {w['window_detection_rate']:.4f} "
                         f"({w['windows_with_at_least_one_alert']}/{w['total_attack_windows']})\n")
            f.write("\n")

    print(f"Saved metrics to {metrics_path} and {metrics_txt_path}")

    # ---- Comparison plot: precision / recall / f1 per strategy ----
    names = list(strategies.keys())
    precisions = [results[n]["transaction_level"]["precision"] for n in names]
    recalls = [results[n]["transaction_level"]["recall"] for n in names]
    f1s = [results[n]["transaction_level"]["f1"] for n in names]

    x = np.arange(len(names))
    width = 0.25
    plt.figure(figsize=(10, 6))
    plt.bar(x - width, precisions, width, label="Precision")
    plt.bar(x, recalls, width, label="Recall")
    plt.bar(x + width, f1s, width, label="F1")
    plt.xticks(x, names, rotation=20)
    plt.ylim(0, 1.05)
    plt.ylabel("Score")
    plt.title("Transaction-level metrics by strategy")
    plt.legend()
    plt.tight_layout()
    cmp_path = os.path.join(args.out, "strategy_comparison.png")
    plt.savefig(cmp_path, dpi=150)
    plt.close()
    print(f"Saved strategy comparison plot to {cmp_path}")

    # ---- Attack-window detection rate comparison ----
    window_rates = [
        results[n]["attack_window_level"]["window_detection_rate"] or 0.0 for n in names
    ]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, window_rates, color="teal")
    plt.ylim(0, 1.05)
    plt.ylabel("Attack-window detection rate")
    plt.title("Attack-window detection rate by strategy")
    plt.xticks(rotation=20)
    for b, r in zip(bars, window_rates):
        plt.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center")
    plt.tight_layout()
    window_cmp_path = os.path.join(args.out, "window_detection_comparison.png")
    plt.savefig(window_cmp_path, dpi=150)
    plt.close()
    print(f"Saved window detection comparison plot to {window_cmp_path}")

    print("\nDone. Summary:")
    for name in names:
        t = results[name]["transaction_level"]
        w = results[name]["attack_window_level"]
        wr = w["window_detection_rate"]
        wr_str = f"{wr:.4f}" if wr is not None else "n/a"
        print(f"  {name:16s} precision={t['precision']:.3f} recall={t['recall']:.3f} "
              f"f1={t['f1']:.3f} | window_detect={wr_str}")


if __name__ == "__main__":
    main()
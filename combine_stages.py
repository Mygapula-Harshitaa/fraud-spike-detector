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


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
# Precision/recall treat every false positive and every false negative as
# equally bad, which isn't true for a payments business. This assigns a
# rupee cost to each outcome so strategies can be compared on money, not
# just on an abstract score. The numbers below are documented ASSUMPTIONS,
# not measured — they are exposed as CLI flags / dashboard sliders so anyone
# reviewing this can substitute their own numbers and see how the ranking
# changes.
#
#   cost_fp        - a legitimate transaction gets held/declined. Cost =
#                     manual review effort + customer friction + the real
#                     chance the customer abandons the purchase.
#   cost_fn        - a card-testing transaction is missed. The transaction
#                     itself is usually tiny (that's the point of card
#                     testing), but letting it through is what lets the
#                     attacker validate a stolen card, which enables a much
#                     larger fraudulent purchase downstream, plus the
#                     merchant/bank eventually eats a chargeback fee. We
#                     price the miss at the downstream risk, not the ticket
#                     size of the probe transaction itself.
#   cost_tp_review - even a correct catch isn't free: someone (a rule
#                     engine or an analyst) has to review/action the alert.
DEFAULT_COST_FP = 40.0          # INR, per false positive
DEFAULT_COST_FN = 500.0         # INR, per missed attack transaction
DEFAULT_COST_TP_REVIEW = 5.0    # INR, per correctly caught transaction


def cost_metrics(
    tp: int,
    fp: int,
    fn: int,
    cost_fp: float = DEFAULT_COST_FP,
    cost_fn: float = DEFAULT_COST_FN,
    cost_tp_review: float = DEFAULT_COST_TP_REVIEW,
) -> dict:
    """Turn a confusion matrix into a rupee cost, given per-outcome cost assumptions."""
    total_cost = fp * cost_fp + fn * cost_fn + tp * cost_tp_review
    n_flagged = tp + fp
    return {
        "cost_fp_assumed": cost_fp,
        "cost_fn_assumed": cost_fn,
        "cost_tp_review_assumed": cost_tp_review,
        "cost_from_false_positives": fp * cost_fp,
        "cost_from_false_negatives": fn * cost_fn,
        "cost_from_review": tp * cost_tp_review,
        "total_cost": total_cost,
        "n_flagged_for_review": n_flagged,
    }


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
    ap.add_argument("--cost_fp", type=float, default=DEFAULT_COST_FP,
                     help=f"INR cost per false positive (default {DEFAULT_COST_FP})")
    ap.add_argument("--cost_fn", type=float, default=DEFAULT_COST_FN,
                     help=f"INR cost per missed attack transaction (default {DEFAULT_COST_FN})")
    ap.add_argument("--cost_tp_review", type=float, default=DEFAULT_COST_TP_REVIEW,
                     help=f"INR cost to review/action a correct catch (default {DEFAULT_COST_TP_REVIEW})")
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
        cost = cost_metrics(
            txn_metrics["tp"], txn_metrics["fp"], txn_metrics["fn"],
            cost_fp=args.cost_fp, cost_fn=args.cost_fn, cost_tp_review=args.cost_tp_review,
        )
        results[name] = {
            "transaction_level": txn_metrics,
            "attack_window_level": window_metrics,
            "cost": cost,
        }

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
            c = r["cost"]
            f.write(f"  Estimated cost: Rs.{c['total_cost']:,.0f} total "
                     f"(FP cost Rs.{c['cost_from_false_positives']:,.0f} + "
                     f"FN cost Rs.{c['cost_from_false_negatives']:,.0f} + "
                     f"review cost Rs.{c['cost_from_review']:,.0f}) "
                     f"[assumes Rs.{c['cost_fp_assumed']:.0f}/FP, Rs.{c['cost_fn_assumed']:.0f}/FN, "
                     f"Rs.{c['cost_tp_review_assumed']:.0f}/review]\n")
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

    # ---- Cost comparison plot ----
    costs = [results[n]["cost"]["total_cost"] for n in names]
    plt.figure(figsize=(8, 5))
    bars = plt.bar(names, costs, color="#e76f51")
    plt.ylabel("Estimated total cost (Rs.)")
    plt.title(
        f"Estimated cost by strategy (Rs.{args.cost_fp:.0f}/FP, Rs.{args.cost_fn:.0f}/FN, "
        f"Rs.{args.cost_tp_review:.0f}/review)"
    )
    plt.xticks(rotation=20)
    for b, c in zip(bars, costs):
        plt.text(b.get_x() + b.get_width() / 2, c, f"Rs.{c:,.0f}", ha="center", va="bottom")
    plt.tight_layout()
    cost_cmp_path = os.path.join(args.out, "cost_comparison.png")
    plt.savefig(cost_cmp_path, dpi=150)
    plt.close()
    print(f"Saved cost comparison plot to {cost_cmp_path}")

    print("\nDone. Summary:")
    for name in names:
        t = results[name]["transaction_level"]
        w = results[name]["attack_window_level"]
        c = results[name]["cost"]
        wr = w["window_detection_rate"]
        wr_str = f"{wr:.4f}" if wr is not None else "n/a"
        print(f"  {name:16s} precision={t['precision']:.3f} recall={t['recall']:.3f} "
              f"f1={t['f1']:.3f} | window_detect={wr_str} | cost=Rs.{c['total_cost']:,.0f}")


if __name__ == "__main__":
    main()
  

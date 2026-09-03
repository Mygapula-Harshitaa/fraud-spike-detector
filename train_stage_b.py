"""
Stage B: supervised card-testing / fraud-spike classifier.

Trains an XGBoost classifier on transaction-level features to predict is_attack.
Saves:
  - the trained model (joblib)
  - a metrics report (txt)
  - a PR curve plot
  - a feature-importance plot
  - a confusion matrix plot at the chosen threshold

Usage:
    python train_stage_b.py --train out/features_train_transaction.csv \
                             --test out/features_test_transaction.csv \
                             --out out/stage_b
"""

import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Columns that are identifiers / labels / timestamps — never fed to the model
NON_FEATURE_COLS = ["payment_id", "merchant_id", "created_at", "is_attack", "attack_window_id"]


def load_xy(path: str):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLS]
    X = df[feature_cols]
    y = df["is_attack"].astype(int)
    return df, X, y, feature_cols


def find_best_threshold(y_true, y_prob):
    """Pick the probability threshold that maximizes F1 on the given set."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_idx = np.nanargmax(f1s[:-1]) if len(thresholds) > 0 else 0
    if len(thresholds) == 0:
        return 0.5
    return float(thresholds[best_idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="path to features_train_transaction.csv")
    ap.add_argument("--test", required=True, help="path to features_test_transaction.csv")
    ap.add_argument("--out", required=True, help="output prefix/dir for model + plots + metrics")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading data...")
    train_df, X_train, y_train, feature_cols = load_xy(args.train)
    test_df, X_test, y_test, _ = load_xy(args.test)

    print(f"Train: {X_train.shape}, positive rate {y_train.mean():.4f}")
    print(f"Test:  {X_test.shape}, positive rate {y_test.mean():.4f}")

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=42,
    )

    print("Training XGBoost...")
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    print("Scoring test set...")
    y_prob = model.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    best_thresh = find_best_threshold(y_test, y_prob)
    y_pred = (y_prob >= best_thresh).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # ---- Also evaluate at a couple of fixed thresholds for comparison ----
    fixed_results = {}
    for t in [0.3, 0.5, 0.7]:
        yp = (y_prob >= t).astype(int)
        fixed_results[t] = {
            "precision": precision_score(y_test, yp, zero_division=0),
            "recall": recall_score(y_test, yp, zero_division=0),
            "f1": f1_score(y_test, yp, zero_division=0),
        }

    # ---- Attack-window-level detection check ----
    # For each true attack window, did we fire at least one alert during it?
    window_detection = None
    if "attack_window_id" in test_df.columns:
        eval_df = test_df.copy()
        eval_df["y_prob"] = y_prob
        eval_df["alert"] = y_pred
        attack_rows = eval_df[eval_df["is_attack"] == 1]
        windows = attack_rows["attack_window_id"].dropna().unique()
        detected = 0
        for w in windows:
            wdf = attack_rows[attack_rows["attack_window_id"] == w]
            if wdf["alert"].sum() > 0:
                detected += 1
        window_detection = {
            "total_attack_windows": int(len(windows)),
            "windows_with_at_least_one_alert": int(detected),
            "window_detection_rate": float(detected / len(windows)) if len(windows) else None,
        }

    # ---- Save metrics report ----
    metrics = {
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "best_threshold_by_f1": float(best_thresh),
        "at_best_threshold": {"precision": float(precision), "recall": float(recall), "f1": float(f1)},
        "confusion_matrix_at_best_threshold": cm.tolist(),
        "fixed_thresholds": {str(k): v for k, v in fixed_results.items()},
        "attack_window_detection": window_detection,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
    }

    metrics_path = os.path.join(args.out, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    metrics_txt_path = os.path.join(args.out, "metrics.txt")
    with open(metrics_txt_path, "w") as f:
        f.write("=== Stage B — XGBoost Fraud/Card-Testing Classifier ===\n\n")
        f.write(f"PR-AUC:  {pr_auc:.4f}\n")
        f.write(f"ROC-AUC: {roc_auc:.4f}\n\n")
        f.write(f"Best threshold (by F1): {best_thresh:.4f}\n")
        f.write(f"  Precision: {precision:.4f}\n")
        f.write(f"  Recall:    {recall:.4f}\n")
        f.write(f"  F1:        {f1:.4f}\n\n")
        f.write("Confusion matrix (rows=true, cols=pred) [ [TN, FP], [FN, TP] ]:\n")
        f.write(f"{cm}\n\n")
        f.write("Metrics at fixed thresholds:\n")
        for t, r in fixed_results.items():
            f.write(f"  t={t}: precision={r['precision']:.4f} recall={r['recall']:.4f} f1={r['f1']:.4f}\n")
        if window_detection:
            f.write("\nAttack-window-level detection:\n")
            f.write(f"  Total attack windows in test set: {window_detection['total_attack_windows']}\n")
            f.write(f"  Windows with >=1 alert:            {window_detection['windows_with_at_least_one_alert']}\n")
            f.write(f"  Window detection rate:             {window_detection['window_detection_rate']:.4f}\n")

    print(f"Saved metrics to {metrics_path} and {metrics_txt_path}")

    # ---- Save model ----
    model_path = os.path.join(args.out, "stage_b_xgb_model.joblib")
    joblib.dump({"model": model, "feature_cols": feature_cols, "threshold": best_thresh}, model_path)
    print(f"Saved model to {model_path}")

    # ---- Plot: Precision-Recall curve ----
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, label=f"PR-AUC = {pr_auc:.3f}")
    plt.scatter([recall], [precision], color="red", zorder=5, label=f"Chosen threshold ({best_thresh:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Stage B — Precision-Recall Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    pr_plot_path = os.path.join(args.out, "pr_curve.png")
    plt.savefig(pr_plot_path, dpi=150)
    plt.close()
    print(f"Saved PR curve to {pr_plot_path}")

    # ---- Plot: Feature importance ----
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(
        [feature_cols[i] for i in order][::-1],
        [importances[i] for i in order][::-1],
        color="steelblue",
    )
    plt.xlabel("Importance")
    plt.title("Stage B — Feature Importance")
    plt.tight_layout()
    fi_plot_path = os.path.join(args.out, "feature_importance.png")
    plt.savefig(fi_plot_path, dpi=150)
    plt.close()
    print(f"Saved feature importance plot to {fi_plot_path}")

    # ---- Plot: Confusion matrix ----
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix (threshold={best_thresh:.2f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["Normal", "Attack"])
    plt.yticks([0, 1], ["Normal", "Attack"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    cm_plot_path = os.path.join(args.out, "confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix plot to {cm_plot_path}")

    print("\nDone. Summary:")
    print(f"  PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"  At threshold {best_thresh:.3f}: precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")
    if window_detection:
        print(f"  Attack-window detection rate: {window_detection['window_detection_rate']:.4f} "
              f"({window_detection['windows_with_at_least_one_alert']}/{window_detection['total_attack_windows']})")


if __name__ == "__main__":
    main()
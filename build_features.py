"""
Feature engineering for the fraud-spike-detector.

Produces two tables:
  1. transaction_features: per-transaction features for the supervised classifier
     (Stage B in the architecture).
  2. merchant_minute_features: per-merchant, per-minute rolling aggregates for the
     unsupervised anomaly scorer (Stage A).

CRITICAL: every feature is computed causally — using only transactions at or before
the current row's timestamp. No feature may peek at future rows, or evaluation
metrics will be meaningless. `is_attack` / `attack_window_id` are carried through
untouched as labels only; they are never used to compute a feature value.

Usage:
    python build_features.py --in ../out/transactions_train.csv --out ../out/features_train
"""

import argparse

import numpy as np
import pandas as pd


def _rolling_unique_count(times: np.ndarray, keys: np.ndarray, window_seconds: float) -> np.ndarray:
    """
    For each i, count the number of DISTINCT `keys` values in the trailing window
    (t_i - window_seconds, t_i], inclusive of the current row, using a two-pointer
    sweep. `times` must be sorted ascending (seconds since epoch, float).
    """
    n = len(times)
    result = np.empty(n, dtype=np.int32)
    left = 0
    from collections import Counter

    counts: "Counter" = Counter()
    for right in range(n):
        counts[keys[right]] += 1
        while times[right] - times[left] > window_seconds:
            counts[keys[left]] -= 1
            if counts[keys[left]] == 0:
                del counts[keys[left]]
            left += 1
        result[right] = len(counts)
    return result


def add_merchant_rolling_features(df: pd.DataFrame, window_minutes: int = 5) -> pd.DataFrame:
    """Causal rolling features computed within each merchant's own transaction stream."""
    df = df.sort_values(["merchant_id", "created_at"]).reset_index(drop=True)
    window_s = window_minutes * 60

    out_parts = []
    for merchant_id, g in df.groupby("merchant_id", sort=False):
        g = g.sort_values("created_at").reset_index(drop=True)
        t = g["created_at"].astype("int64").to_numpy() / 1e9  # seconds since epoch

        # Rolling txn count in trailing window (time-indexed rolling, causal by construction)
        s = pd.Series(1, index=pd.DatetimeIndex(g["created_at"]))
        g["merchant_txn_count_5min"] = s.rolling(f"{window_minutes}min").sum().to_numpy()

        # Rolling decline rate in trailing window
        failed = pd.Series((g["status"] == "failed").astype(int).to_numpy(), index=pd.DatetimeIndex(g["created_at"]))
        rolling_failed = failed.rolling(f"{window_minutes}min").sum().to_numpy()
        g["merchant_decline_rate_5min"] = rolling_failed / g["merchant_txn_count_5min"].clip(lower=1)

        # Rolling distinct devices / card BINs / IPs in trailing window (card-testing signature)
        device_keys = g["device_id"].fillna("none").to_numpy()
        ip_keys = g["ip_address"].fillna("none").to_numpy()
        bin_keys = g["card_bin"].fillna(-1).to_numpy()
        g["merchant_unique_devices_5min"] = _rolling_unique_count(t, device_keys, window_s)
        g["merchant_unique_ips_5min"] = _rolling_unique_count(t, ip_keys, window_s)
        g["merchant_unique_card_bins_5min"] = _rolling_unique_count(t, bin_keys, window_s)

        # Causal expanding baseline for amount z-score: mean/std of all PRIOR transactions only.
        amt = g["amount"].to_numpy()
        prior_mean = pd.Series(amt).shift(1).expanding().mean().to_numpy()
        prior_std = pd.Series(amt).shift(1).expanding().std().to_numpy()
        prior_std = np.where((prior_std == 0) | np.isnan(prior_std), 1.0, prior_std)
        prior_mean = np.where(np.isnan(prior_mean), amt.mean(), prior_mean)
        g["amount_zscore_vs_merchant_history"] = (amt - prior_mean) / prior_std

        out_parts.append(g)

    return pd.concat(out_parts, ignore_index=True)


def add_entity_recency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Time-since-last-transaction for the same device and same IP (card-testing reuses these fast)."""
    df = df.sort_values(["device_id", "created_at"]).reset_index(drop=True)
    df["seconds_since_last_txn_device"] = (
        df.groupby("device_id")["created_at"].diff().dt.total_seconds()
    )
    df = df.sort_values(["ip_address", "created_at"]).reset_index(drop=True)
    df["seconds_since_last_txn_ip"] = (
        df.groupby("ip_address")["created_at"].diff().dt.total_seconds()
    )
    # Missing = first-ever transaction from that entity; fill with a large sentinel
    df["seconds_since_last_txn_device"] = df["seconds_since_last_txn_device"].fillna(86400)
    df["seconds_since_last_txn_ip"] = df["seconds_since_last_txn_ip"].fillna(86400)
    return df


def build_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    df["created_at"] = pd.to_datetime(df["created_at"])
    df = add_merchant_rolling_features(df)
    df = add_entity_recency_features(df)

    df["hour_of_day"] = df["created_at"].dt.hour
    df["is_card"] = (df["method"] == "card").astype(int)
    df["is_new_customer"] = df["is_new_customer"].astype(int)
    df["is_low_ticket"] = (df["amount"] < 150).astype(int)

    feature_cols = [
        "payment_id", "merchant_id", "created_at",
        "merchant_txn_count_5min", "merchant_decline_rate_5min",
        "merchant_unique_devices_5min", "merchant_unique_ips_5min", "merchant_unique_card_bins_5min",
        "amount_zscore_vs_merchant_history", "seconds_since_last_txn_device", "seconds_since_last_txn_ip",
        "hour_of_day", "is_card", "is_new_customer", "is_low_ticket",
        # labels — keep separate from X at training time, never feed to the model
        "is_attack", "attack_window_id",
    ]
    return df[feature_cols].sort_values("created_at").reset_index(drop=True)


def build_merchant_minute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-merchant, per-minute aggregates for the unsupervised anomaly scorer (Stage A).
    A rolling z-score of txn_count vs a trailing 60-minute baseline is included as a
    ready-to-threshold anomaly signal; swap in IsolationForest on these columns later.
    """
    df["created_at"] = pd.to_datetime(df["created_at"])
    df["minute"] = df["created_at"].dt.floor("min")

    agg = (
        df.groupby(["merchant_id", "minute"])
        .agg(
            txn_count=("payment_id", "count"),
            decline_rate=("status", lambda s: (s == "failed").mean()),
            unique_devices=("device_id", "nunique"),
            unique_card_bins=("card_bin", "nunique"),
            avg_amount=("amount", "mean"),
            attack_txn_count=("is_attack", "sum"),  # label for eval only
        )
        .reset_index()
        .sort_values(["merchant_id", "minute"])
    )

    parts = []
    for merchant_id, g in agg.groupby("merchant_id", sort=False):
        g = g.drop(columns=["merchant_id"]).set_index("minute").sort_index()
        g = g.asfreq("min", fill_value=0)  # dense minute grid, no gaps
        roll_mean = g["txn_count"].shift(1).rolling("60min").mean()
        roll_std = g["txn_count"].shift(1).rolling("60min").std().replace(0, np.nan)
        g["txn_count_zscore_60min"] = ((g["txn_count"] - roll_mean) / roll_std).fillna(0)
        g["merchant_id"] = merchant_id
        parts.append(g.reset_index())

    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="infile", required=True)
    ap.add_argument("--out", dest="outprefix", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.infile)
    txn_features = build_transaction_features(df.copy())
    merchant_features = build_merchant_minute_features(df.copy())

    txn_features.to_csv(f"{args.outprefix}_transaction.csv", index=False)
    merchant_features.to_csv(f"{args.outprefix}_merchant_minute.csv", index=False)

    print(f"Transaction features: {txn_features.shape}")
    print(f"Merchant-minute features: {merchant_features.shape}")
    print(f"Positive rate (is_attack): {txn_features['is_attack'].mean():.4f}")


if __name__ == "__main__":
    main()

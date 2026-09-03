"""
Synthetic transaction data generator for the AI Risk Manager / fraud-spike-detector
buildathon project.

Simulates a Razorpay-like payments stream for N merchants over T days, with normal
diurnal traffic, and injects labeled "card-testing" attack windows: short bursts of
low-value, high-decline-rate transactions replaying many distinct card numbers
through a small pool of devices/IPs.

v2 changes (vs original):
  - Attack amounts now OVERLAP with normal traffic amounts instead of always being
    a trivially separable uniform(5, 100). Most probe amounts are still small, but a
    meaningful fraction are drawn from the merchant's normal ticket-size distribution,
    so `amount` / `is_low_ticket` alone can no longer perfectly separate the classes.
  - Normal traffic now includes genuine micro-transactions (e.g. food-delivery /
    gaming top-ups), so small amounts are no longer an attack-only signal.
  - `is_new_customer` on attack rows is no longer always True (real card-testing
    sometimes reuses/spoofs existing-looking customer identifiers).
  - Added occasional legitimate traffic bursts (flash-sale-like) with many
    transactions in a short window from a MODERATE number of distinct
    devices/IPs, so raw transaction-count / device-count spikes alone are not
    a giveaway either — the model has to combine several weaker signals.

Ground-truth labels (is_attack, attack_window_id) are kept ONLY for evaluation.
Downstream feature/model code must not use them as inputs.

Usage:
    python generate_data.py --out ../out --merchants 25 --days 30 --seed 42
"""

import argparse
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

METHODS = ["card", "upi", "netbanking", "wallet"]
METHOD_WEIGHTS = [0.55, 0.30, 0.10, 0.05]

# Categories where genuine small-ticket transactions are common (micro-transactions,
# top-ups, snack orders) — this keeps "low amount" from being an attack-only signal.
MICRO_TXN_CATEGORIES = {"food_delivery", "gaming"}


def diurnal_rate_multiplier(hour: int) -> float:
    """Rough day/night traffic curve: low at night, peaks late morning + evening."""
    return 0.15 + 0.85 * (
        0.5 * np.exp(-((hour - 11) ** 2) / 18) + 0.5 * np.exp(-((hour - 20) ** 2) / 10)
    )


def make_merchants(n_merchants: int, rng: np.random.Generator) -> pd.DataFrame:
    categories = ["ecommerce", "food_delivery", "travel", "saas", "gaming", "edtech"]
    merchants = pd.DataFrame(
        {
            "merchant_id": [f"merch_{i:04d}" for i in range(n_merchants)],
            "category": rng.choice(categories, size=n_merchants),
            "avg_daily_txns": rng.integers(200, 4000, size=n_merchants),
            "avg_ticket_size": rng.uniform(150, 6000, size=n_merchants).round(2),
            "baseline_decline_rate": rng.uniform(0.02, 0.08, size=n_merchants).round(3),
        }
    )
    return merchants


def gen_normal_traffic(
    merchant: pd.Series, start: datetime, days: int, rng: np.random.Generator
) -> pd.DataFrame:
    """Non-homogeneous Poisson stream of normal transactions for one merchant."""
    rows = []
    base_hourly_rate = merchant["avg_daily_txns"] / 24.0
    t = start
    end = start + timedelta(days=days)
    device_pool = [f"dev_{merchant['merchant_id']}_{i}" for i in range(int(merchant["avg_daily_txns"] * 0.6))]
    ip_pool = [f"ip_{merchant['merchant_id']}_{i}" for i in range(int(merchant["avg_daily_txns"] * 0.5))]
    is_micro_category = merchant["category"] in MICRO_TXN_CATEGORIES

    while t < end:
        rate = base_hourly_rate * diurnal_rate_multiplier(t.hour) / 60.0  # per-minute
        n_this_minute = rng.poisson(max(rate, 0.001))
        for _ in range(n_this_minute):
            is_card = rng.random() < 0.55
            method = rng.choice(METHODS, p=METHOD_WEIGHTS)

            # Most normal transactions follow the merchant's usual ticket size, but
            # a genuine slice are small legitimate purchases (more so for
            # food-delivery/gaming-style merchants) — this keeps "small amount"
            # from being a pure attack tell.
            micro_prob = 0.18 if is_micro_category else 0.06
            if rng.random() < micro_prob:
                amount = max(5.0, rng.uniform(20, 180))
            else:
                amount = max(10.0, rng.lognormal(np.log(merchant["avg_ticket_size"]), 0.6))

            declined = rng.random() < merchant["baseline_decline_rate"]
            rows.append(
                {
                    "merchant_id": merchant["merchant_id"],
                    "created_at": t + timedelta(seconds=int(rng.integers(0, 60))),
                    "amount": round(amount, 2),
                    "method": method,
                    "card_bin": rng.integers(400000, 499999) if is_card else None,
                    "device_id": rng.choice(device_pool),
                    "ip_address": rng.choice(ip_pool),
                    "customer_id": f"cust_{uuid.uuid4().hex[:10]}",
                    "is_new_customer": bool(rng.random() < 0.25),
                    "status": "failed" if declined else "captured",
                    "is_attack": 0,
                    "attack_window_id": None,
                }
            )
        t += timedelta(minutes=1)
    return pd.DataFrame(rows)


def gen_legit_traffic_burst(
    merchant: pd.Series, window_start: datetime, rng: np.random.Generator
) -> pd.DataFrame:
    """
    A legitimate but bursty traffic spike (e.g. a flash sale, a viral social post,
    a payday rush) — many transactions in a short window, but from a MODERATE,
    more realistic spread of devices/IPs and normal-ish amounts/decline rates.
    Not labeled as an attack. Exists so that raw txn-count / device-count spikes
    alone don't trivially separate attacks from benign activity.
    """
    duration_min = int(rng.integers(5, 20))
    n_txns = int(rng.integers(15, 80))
    # A real flash sale draws many distinct genuine customers/devices, unlike a
    # card-testing attack which is confined to a tiny device/IP pool.
    n_devices = int(rng.integers(8, 30))
    n_ips = int(rng.integers(6, 25))
    devices = [f"dev_{merchant['merchant_id']}_burst_{uuid.uuid4().hex[:6]}" for _ in range(n_devices)]
    ips = [f"ip_{merchant['merchant_id']}_burst_{uuid.uuid4().hex[:6]}" for _ in range(n_ips)]

    rows = []
    for _ in range(n_txns):
        offset_sec = int(rng.uniform(0, duration_min * 60))
        is_card = rng.random() < 0.55
        method = rng.choice(METHODS, p=METHOD_WEIGHTS)
        amount = max(10.0, rng.lognormal(np.log(merchant["avg_ticket_size"]), 0.5))
        declined = rng.random() < merchant["baseline_decline_rate"]
        rows.append(
            {
                "merchant_id": merchant["merchant_id"],
                "created_at": window_start + timedelta(seconds=offset_sec),
                "amount": round(amount, 2),
                "method": method,
                "card_bin": rng.integers(400000, 499999) if is_card else None,
                "device_id": rng.choice(devices),
                "ip_address": rng.choice(ips),
                "customer_id": f"cust_{uuid.uuid4().hex[:10]}",
                "is_new_customer": bool(rng.random() < 0.4),
                "status": "failed" if declined else "captured",
                "is_attack": 0,
                "attack_window_id": None,
            }
        )
    return pd.DataFrame(rows).sort_values("created_at")


def inject_card_testing_attack(
    merchant_id: str, merchant_avg_ticket: float, window_start: datetime, rng: np.random.Generator
) -> pd.DataFrame:
    """
    Card-testing burst: many small, rapid transactions against a merchant, replaying
    sequential/random card numbers through a small pool of devices/IPs, with an
    elevated decline rate (most stolen/guessed cards get rejected).

    Amounts are MOSTLY small probe amounts, but a meaningful fraction are drawn to
    overlap with the merchant's normal ticket-size range (attackers testing larger
    "does this card work for a real purchase" amounts too), so the amount feature
    alone can't perfectly separate attacks from normal traffic.
    """
    duration_min = int(rng.integers(2, 15))
    n_txns = int(rng.integers(20, 150))
    n_devices = int(rng.integers(1, 4))
    n_ips = int(rng.integers(1, 3))
    devices = [f"attackdev_{uuid.uuid4().hex[:6]}" for _ in range(n_devices)]
    ips = [f"attackip_{uuid.uuid4().hex[:6]}" for _ in range(n_ips)]
    window_id = f"atk_{uuid.uuid4().hex[:8]}"
    base_bin = int(rng.integers(400000, 499999))

    rows = []
    for i in range(n_txns):
        offset_sec = int(rng.uniform(0, duration_min * 60))
        declined = rng.random() < 0.75  # most probe attempts fail

        # ~70% classic tiny probe amounts, ~30% overlap with normal merchant range
        if rng.random() < 0.7:
            amount = float(rng.uniform(5, 100))
        else:
            amount = max(10.0, float(rng.lognormal(np.log(max(merchant_avg_ticket, 50)), 0.5)))

        rows.append(
            {
                "merchant_id": merchant_id,
                "created_at": window_start + timedelta(seconds=offset_sec),
                "amount": round(amount, 2),
                "method": "card",
                "card_bin": base_bin + i % 40,  # sequential-ish card range
                "device_id": rng.choice(devices),
                "ip_address": rng.choice(ips),
                "customer_id": f"cust_{uuid.uuid4().hex[:10]}",
                # real card-testing sometimes reuses/spoofs an existing-looking
                # customer id rather than always presenting as brand new
                "is_new_customer": bool(rng.random() < 0.85),
                "status": "failed" if declined else "captured",
                "is_attack": 1,
                "attack_window_id": window_id,
            }
        )
    return pd.DataFrame(rows).sort_values("created_at")


def generate(
    n_merchants: int,
    days: int,
    seed: int,
    attack_prob_per_merchant_day: float = 0.08,
    legit_burst_prob_per_merchant_day: float = 0.10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    start = datetime(2026, 1, 1)
    merchants = make_merchants(n_merchants, rng)

    all_txns = []
    for _, merchant in merchants.iterrows():
        normal = gen_normal_traffic(merchant, start, days, rng)
        all_txns.append(normal)

        for day in range(days):
            day_start = start + timedelta(days=day)

            # Inject a handful of attack windows per merchant over the period
            if rng.random() < attack_prob_per_merchant_day:
                # attacks skew toward low-traffic hours, when spikes stand out most
                hour = int(rng.choice(range(0, 6), size=1)[0]) if rng.random() < 0.6 else int(rng.integers(0, 24))
                window_start = day_start + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                attack = inject_card_testing_attack(
                    merchant["merchant_id"], merchant["avg_ticket_size"], window_start, rng
                )
                all_txns.append(attack)

            # Occasionally inject a legitimate bursty spike (flash sale, viral
            # moment) — a hard negative that looks superficially similar to an
            # attack (lots of txns in a short window) but isn't one.
            if rng.random() < legit_burst_prob_per_merchant_day:
                hour = int(rng.integers(8, 23))  # bursts of real interest happen in waking hours
                window_start = day_start + timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                burst = gen_legit_traffic_burst(merchant, window_start, rng)
                all_txns.append(burst)

    txns = pd.concat(all_txns, ignore_index=True)
    txns = txns.sort_values(["created_at"]).reset_index(drop=True)
    txns.insert(0, "payment_id", [f"pay_{uuid.uuid4().hex[:12]}" for _ in range(len(txns))])
    return merchants, txns


def time_based_split(txns: pd.DataFrame, test_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by time, NOT randomly — the test set is always the most recent slice."""
    cutoff = txns["created_at"].quantile(1 - test_frac)
    train = txns[txns["created_at"] < cutoff].reset_index(drop=True)
    test = txns[txns["created_at"] >= cutoff].reset_index(drop=True)
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../out")
    ap.add_argument("--merchants", type=int, default=25)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import os

    os.makedirs(args.out, exist_ok=True)

    merchants, txns = generate(args.merchants, args.days, args.seed)
    train, test = time_based_split(txns)

    merchants.to_csv(f"{args.out}/merchants.csv", index=False)
    txns.to_csv(f"{args.out}/transactions_full.csv", index=False)
    train.to_csv(f"{args.out}/transactions_train.csv", index=False)
    test.to_csv(f"{args.out}/transactions_test.csv", index=False)

    n_attack_windows = txns["attack_window_id"].nunique()
    print(f"Merchants: {len(merchants)}")
    print(f"Transactions: {len(txns):,}  ({txns['is_attack'].sum():,} attack rows, "
          f"{n_attack_windows} attack windows)")
    print(f"Train: {len(train):,}  ({train['created_at'].min()} to {train['created_at'].max()})")
    print(f"Test:  {len(test):,}  ({test['created_at'].min()} to {test['created_at'].max()})")
    print(f"Wrote CSVs to {args.out}/")


if __name__ == "__main__":
    main()
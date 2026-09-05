"""
Backfills the `cost` block into an existing combine_stages.py metrics.json,
using the TP/FP/FN counts that are already there. Lets you add/update the
cost analysis without re-running the full training pipeline.

Usage:
    python add_cost_to_metrics.py \
        --metrics fraud_dashboard/data/metrics.json \
        --cost_fp 40 --cost_fn 500 --cost_tp_review 5
"""

import argparse
import json

from combine_stages import cost_metrics, DEFAULT_COST_FP, DEFAULT_COST_FN, DEFAULT_COST_TP_REVIEW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", required=True, help="path to metrics.json to update in place")
    ap.add_argument("--cost_fp", type=float, default=DEFAULT_COST_FP)
    ap.add_argument("--cost_fn", type=float, default=DEFAULT_COST_FN)
    ap.add_argument("--cost_tp_review", type=float, default=DEFAULT_COST_TP_REVIEW)
    args = ap.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)

    for name, s in metrics["strategies"].items():
        t = s["transaction_level"]
        s["cost"] = cost_metrics(
            t["tp"], t["fp"], t["fn"],
            cost_fp=args.cost_fp, cost_fn=args.cost_fn, cost_tp_review=args.cost_tp_review,
        )

    with open(args.metrics, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Updated {args.metrics} with cost analysis "
          f"(Rs.{args.cost_fp}/FP, Rs.{args.cost_fn}/FN, Rs.{args.cost_tp_review}/review).")
    for name, s in metrics["strategies"].items():
        print(f"  {name:16s} total_cost=Rs.{s['cost']['total_cost']:,.0f}")


if __name__ == "__main__":
    main()

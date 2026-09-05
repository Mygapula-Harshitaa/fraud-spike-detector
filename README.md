# Fraud Spike Detector — Two-Stage Card-Testing Defense

**Track:** Razorpay AI Buildathon — AI Risk Manager (fraud / chargeback / abuse-ring defense)
**Scope:** Strictly defense-only. Detects and reports suspected card-testing activity; takes no blocking or enforcement action on live transactions.

[Live dashboard](https://fraud-spike-dashboard.streamlit.app/) · [Repo](https://github.com/Mygapula-Harshitaa/fraud-spike-detector)

---

## What this is

A two-stage anomaly detection pipeline for **card-testing attacks** — bursts of small, high-decline-rate transactions used to validate stolen card numbers before they're used for real fraud. Card-testing splits naturally into two learnable signals, so the pipeline is built as two stages rather than one model:

| Stage | Type | Model | Input | Catches |
|---|---|---|---|---|
| **Stage A** | Unsupervised | `IsolationForest` | Merchant × minute aggregates (txn count, decline rate, unique devices/card bins, 60-min rolling txn-count z-score) | Sudden bursts/velocity anomalies — works on attack shapes it's never seen, since it's never trained on the `is_attack` label |
| **Stage B** | Supervised | `XGBoost` classifier | Per-transaction features (amount z-score vs. merchant history, time-since-last-txn per device/IP, decline rate, hour, method) | Learned fraud signatures — high precision, but blind to genuinely novel patterns |
| **Cascade** | Combined | — | Stage A's flagged candidates get a *lowered* Stage B threshold (`cascade_ratio × stage_b_threshold`); everything else still needs Stage B's normal, higher bar | Stage A's broad recall + Stage B's precision, without running the expensive model on 100% of traffic |

## Architecture walkthrough

```
                         incoming transaction
                                 │
                                 ▼
                 ┌───────────────────────────────┐
                 │  build_features.py             │
                 │  causal feature engineering    │
                 │  (rolling windows, no lookahead)│
                 └───────────────┬─────────────────┘
                                 │
              ┌──────────────────┴───────────────────┐
              ▼                                       ▼
  ┌────────────────────────────┐        ┌───────────────────────────┐
  │ STAGE A — IsolationForest   │        │ (per-transaction features  │
  │ merchant × minute aggregates│        │  held for Stage B)         │
  │ unsupervised, no labels     │        └─────────────┬───────────────┘
  └──────────────┬───────────────┘                      │
                 │                                       │
        flagged? ┴── yes ──────────┐                     │
                 │                 ▼                     │
                 no          lowered Stage B         normal Stage B
                 │            threshold               threshold
                 │        (cascade_ratio × t)             (t)
                 │                 │                       │
                 └─────────────────┴───────────┬───────────┘
                                                ▼
                                  ┌───────────────────────────┐
                                  │ STAGE B — XGBoost           │
                                  │ transaction-level classifier│
                                  │ supervised, precision-tuned │
                                  └─────────────┬─────────────┘
                                                ▼
                                     alert / no alert
                                (flag for human review only —
                                 no automatic blocking action)
```

Stage A runs on **100% of traffic** cheaply; Stage B only scores the fraction Stage A already flagged as suspicious, at a lowered bar — everything else still has to clear Stage B's normal, stricter threshold on its own. This is what keeps the cascade's precision close to Stage B alone while lifting recall toward Stage A's ceiling, without paying Stage B's cost on every transaction. See `combine_stages.py` for the exact threshold logic.

## Scope

Strictly defense-only. This system flags and reports suspected card-testing activity for human/downstream review; it has no capability to block, execute, or otherwise act on live transactions.

## Project layout

```
fraud-spike-detector/
├── generate_data.py          # synthetic card-testing scenario generator (v2 — see below)
├── build_features.py         # causal feature engineering for both stages
├── train_stage_a.py          # IsolationForest training (merchant-minute anomaly scorer)
├── train_stage_b.py          # XGBoost training (transaction-level classifier)
├── combine_stages.py         # evaluates 5 combination strategies, incl. the cascade
├── README.md
├── LICENSE
└── fraud_dashboard/
    ├── app.py                # Streamlit dashboard (Overview / Synthetic Benchmark /
    │                          #   Real-World Validation / Methodology & Caveats)
    ├── requirements.txt
    └── data/
        ├── metrics.json                          # precomputed synthetic benchmark results
        ├── features_train_transaction.csv        # Stage B training data
        ├── features_test_transaction.csv         # Stage B test data
        ├── features_train_merchant_minute.csv    # Stage A training data
        └── features_test_merchant_minute.csv     # Stage A test data
```

## Why two stages, and why this combination logic specifically

`combine_stages.py` evaluates five ways of merging the two stages on the same held-out test set, not just the cascade:

- `stage_a_only` — Stage A's flag, alone
- `stage_b_only` — Stage B's flag, alone
- `combined_or` — either stage fires (maximizes recall)
- `combined_and` — both stages must fire (maximizes precision)
- `cascade` — the recommended design (see table above)

Results on the held-out synthetic test set (129,251 transactions, 677 attack transactions across 9 attack windows):

**The split is temporal, not random:** `generate_data.py`'s `time_based_split` cuts by time — the test set is always the most recent slice of traffic, never a random shuffle. For a fraud/temporal problem this matters: a random split would let the model see transactions from *after* an attack window in training and evaluate on transactions from *before* it, which quietly leaks future information into training and inflates every metric below. The numbers here reflect what the model would actually see in production — trained on the past, evaluated on data strictly after it.

| Strategy | Precision | Recall | F1 | TP | FP | FN | Attack windows caught |
|---|---|---|---|---|---|---|---|
| Stage A only | 0.167 | 0.907 | 0.283 | 614 | 3,055 | 63 | 9/9 |
| Stage B only | 0.999 | 0.985 | 0.992 | 667 | 1 | 10 | 9/9 |
| Combined (OR) | 0.181 | 0.996 | 0.306 | 674 | 3,055 | 3 | 9/9 |
| Combined (AND) | 0.998 | 0.897 | 0.945 | 607 | 1 | 70 | 9/9 |
| **Cascade** | **0.997** | **0.988** | **0.993** | **669** | **2** | **8** | **9/9** |

**Reading this honestly:** Stage A alone catches every attack window but at a heavy false-positive cost (3,055 false alerts — it has no label signal, so it flags anything bursty, including legitimate spikes). Stage B alone is nearly perfect on precision but leaves 10 attack transactions unflagged. The cascade is the actual production recommendation: it holds Stage B's precision (0.997) while closing most of the recall gap (8 missed vs. 10), and — the part that matters operationally — it only ever runs the expensive classifier on the fraction of traffic Stage A already flagged as a candidate, rather than scoring every transaction at Stage B's cost.

**False-positive cost, concretely:** at the cascade's operating point, 2 out of 129,251 held-out transactions were false alarms. If you attach a real per-review cost (analyst time, customer friction on a held transaction) to that count, this is the number to plug in — I haven't invented a dollar figure here since it depends on your actual review-cost assumption, but the false-positive count itself is real, not estimated.

## Why synthetic data, and what changed the second time

Labeled real card-testing data isn't obtainable for a buildathon (proprietary, privacy-sensitive, rare). `generate_data.py` builds a synthetic merchant-traffic generator instead — but the first version of it had a problem worth stating plainly:

**What broke:** the first generator drew attack amounts from `uniform(5, 100)` while normal traffic followed each merchant's own ticket-size distribution. That's trivially separable — a model could hit near-perfect scores by learning "small amount = attack," a synthetic-generator artifact rather than real fraud shape.

**What we changed (v2):**
- Attack amounts now partly overlap with the merchant's normal ticket-size range (~30%), instead of always being a tiny, distinct band.
- Normal traffic now includes genuine micro-transactions (food-delivery/gaming-style small purchases), so small amount is no longer an attack-only signal.
- `is_new_customer` on attack rows is no longer always `True` — real card-testing sometimes reuses or spoofs existing-looking identifiers.
- Legitimate bursty traffic (flash-sale-like spikes) is now injected as a hard negative: many transactions in a short window, but from a realistically broad pool of devices/IPs, so raw transaction-count or device-count spikes alone don't separate attacks from genuine demand either.

The result is a generator the model can't shortcut on a single feature — it has to combine several weaker signals, which is what `build_features.py`'s feature set (rolling device/IP/card-bin counts, causal amount z-scores, decline rate) is designed to let it do.

**One more thing enforced by design, not just intention:** every feature in `build_features.py` is computed causally — using only transactions at or before the current row's timestamp (rolling windows, expanding means shifted by one). `is_attack` and `attack_window_id` are carried through as labels only and never enter the feature set. This isn't a claim in the README; it's checkable directly in the rolling-window and `.shift(1)` calls in the feature code.

## AI judgment: where ML was used, and why nothing here is an LLM

The entire pipeline — Stage A, Stage B, feature engineering, and the combination logic — is classical, non-LLM machine learning: `IsolationForest`, `XGBoost`, rolling/causal pandas feature engineering. That's a deliberate fit to the problem, not a default: card-testing detection is a structured, tabular, temporal-signal problem, and an LLM would add latency and unpredictability without adding detection power here. Where the dashboard uses a second model family (`RandomForestClassifier` for the Kaggle validation tab), that's also classical ML, chosen to match what a from-scratch pipeline on an unfamiliar schema (PCA'd features, no merchant/device fields) can reasonably use live in a browser session.

## Real-world validation (Kaggle) — what's comparable, and what isn't

The synthetic benchmark proves the pipeline works on the generator it was built against — it doesn't rule out having memorized that generator's quirks. The dashboard's Real-World Validation tab (see below) re-runs an *analogous* Stage A / Stage B / cascade pipeline, trained live, on the public [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 real anonymized European transactions, September 2013, 492 confirmed frauds — 0.17%).

This is explicitly **not** a like-for-like re-run — the Kaggle schema has no merchant IDs, device/IP fields, or discrete attack-window structure, only PCA components (`V1`–`V28`), `Time`, and `Amount`. What's preserved is the *shape* of the approach:

| | Synthetic pipeline | Kaggle analog |
|---|---|---|
| Stage A | Unsupervised `IsolationForest` on merchant-minute aggregates | Unsupervised `IsolationForest` on PCA'd transaction features |
| Stage B | Supervised `XGBoost` on transaction features | Supervised `RandomForestClassifier` on the same features, trained with labels |
| Cascade | B scores only A's flagged candidates, at a lowered threshold | Same logic, all thresholds adjustable live via sliders |
| Window-level recall | Discrete attack windows, all caught | Not computed — no window/session structure in the Kaggle data |

Kaggle's 0.17% fraud rate is even more imbalanced than the synthetic set, so a precision drop there is expected and isn't by itself evidence the approach failed — treat the real-world numbers as a directional generalization check, not a strict benchmark comparison.

## Quick start

```bash
# 1. Generate synthetic data
python generate_data.py --out ./out --merchants 25 --days 30 --seed 42

# 2. Build features (run once for train, once for test)
python build_features.py --in ./out/transactions_train.csv --out ./out/features_train
python build_features.py --in ./out/transactions_test.csv --out ./out/features_test

# 3. Train both stages
python train_stage_a.py --train ./out/features_train_merchant_minute.csv \
                         --test ./out/features_test_merchant_minute.csv \
                         --out ./out/stage_a

python train_stage_b.py --train ./out/features_train_transaction.csv \
                         --test ./out/features_test_transaction.csv \
                         --out ./out/stage_b

# 4. Evaluate all five combination strategies
python combine_stages.py --txn_test ./out/features_test_transaction.csv \
                          --merchant_minute_test ./out/features_test_merchant_minute.csv \
                          --stage_a_model ./out/stage_a/stage_a_isoforest_model.joblib \
                          --stage_b_model ./out/stage_b/stage_b_xgb_model.joblib \
                          --out ./out/combined

# 5. Explore results interactively
cd fraud_dashboard
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` and needs no regeneration to run — `fraud_dashboard/data/` already ships with precomputed synthetic results and features (~16.6 MB).

## Dashboard pages

**Overview** — the two-stage/cascade architecture at a glance, plus headline test-set numbers.

**Synthetic Benchmark** — full results for all five combination strategies from `data/metrics.json`: precision/recall/F1 bar chart, attack-window detection rate, a confusion-count breakdown per strategy, and a feature-distribution explorer comparing attack vs. normal transactions (e.g. `amount_zscore_vs_merchant_history`, `merchant_txn_count_5min`).

**Real-World Validation (Kaggle)** — upload `creditcard.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (free account required). The app trains a live, analogous Stage A + Stage B pipeline on it — Stage A contamination, held-out test fraction, Stage B threshold, and cascade ratio all adjustable via sliders — and compares the resulting cascade precision/recall/F1 directly against the synthetic benchmark's numbers.

**Methodology & Caveats** — states plainly what is and isn't comparable between the synthetic and Kaggle pipelines (no merchant IDs, device/IP fields, or attack-window structure in the Kaggle data), and calls out that Kaggle's more extreme class imbalance (0.17% fraud) makes a precision drop there expected rather than a sign of failure.

Uploaded Kaggle files are never written to disk or sent anywhere — they stay in memory for the session, and the pipeline retrains fresh every run.

## License

MIT

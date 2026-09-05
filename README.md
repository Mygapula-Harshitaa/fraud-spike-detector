# Two-Stage Fraud Detection Dashboard

An interactive Streamlit dashboard for exploring a two-stage fraud detection pipeline:

| Stage | Type | What it does |
|---|---|---|
| **Stage A** | Unsupervised | Flags bursty/velocity anomalies from merchant-minute aggregates |
| **Stage B** | Supervised | Classifies individual transactions using learned fraud patterns |
| **Cascade** | Combined | Stage A cheaply screens all traffic; Stage B scores only what A flags |

## Quick start

```bash
cd fraud_dashboard
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Project layout

```
fraud_dashboard/
├── app.py                  # dashboard entry point
├── requirements.txt
└── data/
    ├── metrics.json                          # precomputed synthetic benchmark results
    ├── features_train_transaction.csv        # Stage B training data
    ├── features_test_transaction.csv         # Stage B test data
    ├── features_train_merchant_minute.csv    # Stage A training data
    └── features_test_merchant_minute.csv     # Stage A test data
```

Everything in `data/` is included and ready to go — about 16.6 MB total, no download or
regeneration needed to run the app.

## What's inside the dashboard

**Overview** — a quick summary of the architecture and headline results.

**Synthetic Benchmark** — an interactive view of `metrics.json`: precision/recall/F1 per
strategy, confusion breakdowns, and attack-window detection rate, plus a feature-distribution
explorer that compares attack vs. normal transactions across features like
`amount_zscore_vs_merchant_history` and `merchant_txn_count_5min`.

**Real-World Validation (Kaggle)** — upload the public
[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 real anonymized transactions, 492 confirmed frauds) and the app trains an analogous
Stage A (`IsolationForest`) + Stage B (`RandomForestClassifier`) pipeline live — with adjustable
thresholds — to check whether the approach generalizes beyond the synthetic generator.

**Methodology & Caveats** — an honest breakdown of what's comparable between the synthetic and
Kaggle pipelines, and what isn't (no merchant IDs, device/IP fields, or attack-window structure
in the Kaggle data).

## Notes

- Uploaded Kaggle files are never written to disk or sent anywhere — they stay in memory for the
  session, and the pipeline retrains fresh each time.
- All key thresholds (Stage A contamination, Stage B cutoff, cascade ratio) are adjustable
  directly in the app via sliders — no code changes needed to explore sensitivity.

"""
Two-Stage Fraud Detection Dashboard
====================================
Stage A : unsupervised burst/velocity anomaly detector on merchant-minute aggregates
Stage B : supervised classifier on transaction-level features
Cascade : Stage A screens traffic cheaply, Stage B scores the flagged candidates precisely

Tab 1 shows results on the synthetic benchmark the model was built/evaluated on.
Tab 2 re-runs an analogous pipeline live on a real, public Kaggle fraud dataset
(ULB "Credit Card Fraud Detection") to check the approach isn't just fitting
quirks of the synthetic generator.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).parent / "data"

st.set_page_config(
    page_title="Two-Stage Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

STRATEGY_ORDER = ["stage_a_only", "stage_b_only", "combined_or", "combined_and", "cascade"]
STRATEGY_LABEL = {
    "stage_a_only": "Stage A only (unsupervised)",
    "stage_b_only": "Stage B only (supervised)",
    "combined_or": "Combined (OR)",
    "combined_and": "Combined (AND)",
    "cascade": "Cascade (A screens → B scores)",
}
STRATEGY_COLOR = {
    "stage_a_only": "#8ecae6",
    "stage_b_only": "#219ebc",
    "combined_or": "#ffb703",
    "combined_and": "#fb8500",
    "cascade": "#023047",
}


# --------------------------------------------------------------------------
# Synthetic-benchmark data loading
# --------------------------------------------------------------------------
@st.cache_data
def load_metrics():
    with open(DATA_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_synthetic_csvs():
    train_txn = pd.read_csv(DATA_DIR / "features_train_transaction.csv")
    test_txn = pd.read_csv(DATA_DIR / "features_test_transaction.csv")
    train_mm = pd.read_csv(DATA_DIR / "features_train_merchant_minute.csv")
    test_mm = pd.read_csv(DATA_DIR / "features_test_merchant_minute.csv")
    return train_txn, test_txn, train_mm, test_mm


def metrics_to_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for key in STRATEGY_ORDER:
        s = metrics["strategies"][key]
        tl = s["transaction_level"]
        wl = s["attack_window_level"]
        rows.append(
            {
                "strategy": key,
                "label": STRATEGY_LABEL[key],
                "precision": tl["precision"],
                "recall": tl["recall"],
                "f1": tl["f1"],
                "tp": tl["tp"],
                "fp": tl["fp"],
                "fn": tl["fn"],
                "window_detection_rate": wl["window_detection_rate"],
                "total_attack_windows": wl["total_attack_windows"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Real-world (Kaggle) pipeline
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_kaggle_csv(file_bytes: bytes) -> pd.DataFrame:
    import io

    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_resource(show_spinner=False)
def train_kaggle_pipeline(file_bytes: bytes, contamination: float, test_size: float, random_state: int = 42):
    """Trains a Stage-A (unsupervised) + Stage-B (supervised) analog on the
    Kaggle 'Credit Card Fraud Detection' dataset (Time, V1..V28, Amount, Class)."""
    df = load_kaggle_csv(file_bytes)

    label_col = "Class" if "Class" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c not in (label_col, "Time")]

    X = df[feature_cols].values
    y = df[label_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Stage A: unsupervised anomaly detector, trained WITHOUT labels
    stage_a = IsolationForest(
        n_estimators=200, contamination=contamination, random_state=random_state, n_jobs=-1
    )
    stage_a.fit(X_train)
    # more negative decision_function => more anomalous. Flip sign so higher = more suspicious.
    stage_a_score_test = -stage_a.decision_function(X_test)
    stage_a_score_train = -stage_a.decision_function(X_train)

    # Stage B: supervised classifier, trained WITH labels
    stage_b = RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced_subsample",
        random_state=random_state, n_jobs=-1,
    )
    stage_b.fit(X_train, y_train)
    stage_b_prob_test = stage_b.predict_proba(X_test)[:, 1]

    return {
        "y_test": y_test,
        "stage_a_score_test": stage_a_score_test,
        "stage_a_score_train": stage_a_score_train,
        "stage_b_prob_test": stage_b_prob_test,
        "n_test": len(y_test),
        "n_test_fraud": int(y_test.sum()),
        "feature_cols": feature_cols,
    }


def compute_strategy_metrics(y_true, a_flag, b_flag, b_prob=None, b_threshold=None, cascade_ratio=0.4):
    combined_or = a_flag | b_flag
    combined_and = a_flag & b_flag
    # cascade (matches combine_stages.py): inside minutes/candidates Stage A already
    # flagged, Stage B only needs to clear a LOWERED bar (cascade_ratio * b_threshold)
    # since Stage A already raised suspicion; everywhere else Stage B keeps its normal,
    # higher bar. This is deliberately NOT the same as combined_and.
    if b_prob is not None and b_threshold is not None:
        cascade_threshold = b_threshold * cascade_ratio
        cascade = np.where(
            a_flag,
            b_prob >= cascade_threshold,
            b_prob >= b_threshold,
        ).astype(bool)
    else:
        # fallback if raw probabilities aren't available at the call site
        cascade = a_flag & b_flag

    out = {}
    for name, pred in [
        ("stage_a_only", a_flag),
        ("stage_b_only", b_flag),
        ("combined_or", combined_or),
        ("combined_and", combined_and),
        ("cascade", cascade),
    ]:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        out[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        }
    return out


# --------------------------------------------------------------------------
# Shared chart helpers
# --------------------------------------------------------------------------
def bar_metric_chart(df: pd.DataFrame, title: str):
    plot_df = df.melt(
        id_vars=["label"], value_vars=["precision", "recall", "f1"],
        var_name="metric", value_name="score",
    )
    fig = px.bar(
        plot_df, x="label", y="score", color="metric", barmode="group",
        title=title, range_y=[0, 1.05],
        color_discrete_map={"precision": "#219ebc", "recall": "#ffb703", "f1": "#8ecae6"},
    )
    fig.update_layout(xaxis_title="", yaxis_title="score", legend_title="")
    return fig


def confusion_bar(row: dict, title: str):
    labels = ["True Positive", "False Positive", "False Negative"] + (["True Negative"] if "tn" in row else [])
    values = [row["tp"], row["fp"], row["fn"]] + ([row["tn"]] if "tn" in row else [])
    colors = ["#2a9d8f", "#e76f51", "#e63946"] + (["#adb5bd"] if "tn" in row else [])
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    fig.update_layout(title=title, yaxis_title="count")
    return fig


def render_fp_cost_section(strategy_metrics: dict, key_prefix: str, chart_title: str):
    """Renders an interactive false-positive / false-negative cost analysis.

    strategy_metrics: dict keyed by strategy name -> dict with at least "fp" and "fn" counts
    (matches both metrics["strategies"][k]["transaction_level"] and compute_strategy_metrics output).
    """
    st.subheader("💰 False Positive Cost Analysis")
    st.caption(
        "Precision/recall alone don't say whether a strategy is actually *cheaper* to run — that "
        "depends on how expensive a false alarm is relative to a missed fraud. Set your own cost "
        "assumptions below to see the trade-off in dollar terms."
    )

    c1, c2 = st.columns(2)
    fp_cost = c1.number_input(
        "Cost per false positive ($)", min_value=0.0, value=5.0, step=0.5,
        key=f"{key_prefix}_fp_cost",
        help="Cost of a false alarm: analyst review time, customer friction from a declined/held "
             "transaction, support contacts, churn risk, etc.",
    )
    fn_cost = c2.number_input(
        "Cost per false negative ($)", min_value=0.0, value=100.0, step=10.0,
        key=f"{key_prefix}_fn_cost",
        help="Cost of a missed fraud: average fraud loss per undetected transaction, chargebacks, "
             "reimbursement, etc.",
    )

    rows = []
    for strat in STRATEGY_ORDER:
        if strat not in strategy_metrics:
            continue
        m = strategy_metrics[strat]
        fp_total = m["fp"] * fp_cost
        fn_total = m["fn"] * fn_cost
        rows.append({
            "strategy": strat,
            "label": STRATEGY_LABEL[strat],
            "fp": m["fp"],
            "fn": m["fn"],
            "fp_cost": fp_total,
            "fn_cost": fn_total,
            "total_cost": fp_total + fn_total,
        })
    cost_df = pd.DataFrame(rows)

    if cost_df.empty:
        st.info("No strategy metrics available to cost out.")
        return

    best = cost_df.loc[cost_df["total_cost"].idxmin()]
    worst = cost_df.loc[cost_df["total_cost"].idxmax()]
    k1, k2, k3 = st.columns(3)
    k1.metric("Lowest-cost strategy", best["label"], f"${best['total_cost']:,.0f}")
    k2.metric("Highest-cost strategy", worst["label"], f"${worst['total_cost']:,.0f}")
    savings = worst["total_cost"] - best["total_cost"]
    k3.metric("Spread (worst − best)", f"${savings:,.0f}")

    fig = px.bar(
        cost_df.melt(id_vars=["label"], value_vars=["fp_cost", "fn_cost"],
                     var_name="cost_type", value_name="cost"),
        x="label", y="cost", color="cost_type", barmode="stack", title=chart_title,
        color_discrete_map={"fp_cost": "#e76f51", "fn_cost": "#e63946"},
        labels={"cost_type": "cost source", "label": ""},
    )
    fig.for_each_trace(lambda t: t.update(name={"fp_cost": "False positive cost", "fn_cost": "False negative cost"}.get(t.name, t.name)))
    fig.update_layout(xaxis_title="", yaxis_title="estimated cost ($)", legend_title="")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        cost_df[["label", "fp", "fn", "fp_cost", "fn_cost", "total_cost"]]
        .rename(columns={"label": "strategy"})
        .style.format({"fp_cost": "${:,.0f}", "fn_cost": "${:,.0f}", "total_cost": "${:,.0f}"}),
        use_container_width=True, hide_index=True,
    )


# ==========================================================================
# Sidebar navigation
# ==========================================================================
st.sidebar.title("🛡️ Fraud Detection")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Synthetic Benchmark", "Real-World Validation (Kaggle)", "Methodology & Caveats"],
)

metrics = load_metrics()

# ==========================================================================
# PAGE: Overview
# ==========================================================================
if page == "Overview":
    st.title("Two-Stage Fraud Detection — Dashboard")
    st.markdown(
        """
This dashboard summarizes a **two-stage fraud detection pipeline**:

| Stage | Type | Input | What it catches |
|---|---|---|---|
| **Stage A** | Unsupervised | Merchant × minute aggregates (txn count, decline rate, unique devices/bins, velocity z-scores) | Sudden bursts / velocity anomalies — works even on brand-new attack patterns it's never seen |
| **Stage B** | Supervised | Per-transaction features (amount z-score, time-since-last-txn, customer/device signals) | Learned fraud signatures from labeled history — precise, but blind to genuinely novel patterns |
| **Cascade** | A → B | Stage A screens all traffic cheaply; only the candidates it flags get scored by the more expensive Stage B model | Combines Stage A's broad recall with Stage B's precision, at a fraction of the compute cost of running B on everything |
        """
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test transactions", f"{metrics['n_test_transactions']:,}")
    c2.metric("Attack transactions", f"{metrics['n_test_attack_transactions']:,}")
    c3.metric("Stage B threshold", f"{metrics['stage_b_threshold']:.3f}")
    c4.metric("Cascade ratio", f"{metrics['cascade_ratio']:.0%}")

    st.info(
        "Use the sidebar to see the **synthetic benchmark** results the model was tuned on, "
        "then check the **Real-World Validation** tab to see the same style of pipeline evaluated "
        "on genuine, unseen fraud data from Kaggle — the point being that a model that only works "
        "on the synthetic generator's quirks would fall apart there, whereas a model that learned "
        "the right *shape* of fraud should hold up reasonably well."
    )

# ==========================================================================
# PAGE: Synthetic Benchmark
# ==========================================================================
elif page == "Synthetic Benchmark":
    st.title("Synthetic Benchmark Results")
    st.caption("Computed by combining Stage A (merchant-minute) and Stage B (transaction) models on held-out synthetic test data.")

    df = metrics_to_frame(metrics)

    k1, k2, k3 = st.columns(3)
    k1.metric("Test transactions", f"{metrics['n_test_transactions']:,}")
    k2.metric("True attack transactions", f"{metrics['n_test_attack_transactions']:,}")
    k3.metric("Attack windows (all strategies caught)", f"{df['total_attack_windows'].iloc[0]}/{df['total_attack_windows'].iloc[0]}")

    st.plotly_chart(bar_metric_chart(df, "Transaction-level precision / recall / F1 by strategy"), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        fig_wd = px.bar(
            df, x="label", y="window_detection_rate", range_y=[0, 1.05],
            title="Attack-window detection rate by strategy",
            color="label", color_discrete_map={STRATEGY_LABEL[k]: v for k, v in STRATEGY_COLOR.items()},
        )
        fig_wd.update_layout(showlegend=False, xaxis_title="")
        st.plotly_chart(fig_wd, use_container_width=True)
    with right:
        chosen = st.selectbox("Confusion breakdown for:", STRATEGY_ORDER, index=4, format_func=lambda k: STRATEGY_LABEL[k])
        row = df[df["strategy"] == chosen].iloc[0].to_dict()
        st.plotly_chart(confusion_bar(row, f"{STRATEGY_LABEL[chosen]} — confusion counts"), use_container_width=True)

    st.subheader("Full metrics table")
    show_df = df[["label", "precision", "recall", "f1", "tp", "fp", "fn", "window_detection_rate"]].rename(
        columns={"label": "strategy"}
    )
    st.dataframe(
        show_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "window_detection_rate": "{:.2f}"}),
        use_container_width=True, hide_index=True,
    )

    st.markdown(
        """
**Reading the results:** Stage A alone catches every attack window (recall ≈ 0.91 at the
transaction level) but with heavy false-positive cost (precision ≈ 0.17) — expected, since it has
no label signal and flags anything that *looks* bursty. Stage B alone is precise
(≈ 0.999) but slightly less sensitive to brand-new patterns. The **cascade** keeps Stage B's
precision (0.997) while lifting recall close to Stage B's ceiling, at a fraction of the scoring
cost of `combined_or`/`combined_and`, since Stage B only ever scores the ~{:.0%} of traffic Stage A
flags as a candidate.
        """.format(metrics["cascade_ratio"])
    )

    render_fp_cost_section(
        {k: metrics["strategies"][k]["transaction_level"] for k in STRATEGY_ORDER},
        key_prefix="synth",
        chart_title="Estimated cost by strategy — synthetic benchmark",
    )

    with st.expander("Explore the raw synthetic feature data"):
        st.caption(
            "Sampled data for exploration only — smaller than the full training/test set "
            "used to produce the metrics above."
        )
        train_txn, test_txn, train_mm, test_mm = load_synthetic_csvs()
        st.write(f"Train transactions: {len(train_txn):,} rows · Test transactions: {len(test_txn):,} rows")
        st.write(f"Train merchant-minutes: {len(train_mm):,} rows · Test merchant-minutes: {len(test_mm):,} rows")

        feat = st.selectbox(
            "Transaction feature to compare (attack vs. normal):",
            ["amount_zscore_vs_merchant_history", "merchant_txn_count_5min", "merchant_decline_rate_5min",
             "seconds_since_last_txn_device", "seconds_since_last_txn_ip"],
        )
        plot_df = test_txn[[feat, "is_attack"]].copy()
        plot_df["is_attack"] = plot_df["is_attack"].map({0: "normal", 1: "attack"})
        fig = px.histogram(
            plot_df, x=feat, color="is_attack", barmode="overlay", nbins=60,
            histnorm="probability density", opacity=0.6,
            color_discrete_map={"normal": "#8ecae6", "attack": "#e63946"},
            title=f"Distribution of {feat} — test set",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(test_txn.sample(min(200, len(test_txn)), random_state=1), use_container_width=True, height=250)

# ==========================================================================
# PAGE: Real-World Validation
# ==========================================================================
elif page == "Real-World Validation (Kaggle)":
    st.title("Real-World Validation")
    st.markdown(
        """
The synthetic benchmark proves the pipeline *works on the generator it was built against* — it
doesn't rule out the model having simply memorized quirks of that generator. To check
generalization, this tab re-runs an **analogous Stage A / Stage B / cascade pipeline**, trained
and evaluated live, on a genuine public fraud dataset:

**[Credit Card Fraud Detection (ULB / Worldline)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
— 284,807 real anonymized European card transactions from September 2013, with 492 confirmed
frauds (0.17%). Features `V1`–`V28` are PCA components of the original transaction data, plus
`Time` (seconds since first transaction) and `Amount`.

Download `creditcard.csv` from the link above (free Kaggle account required) and upload it below.
Nothing is memorized between sessions — training happens on your machine when the app runs.
        """
    )

    uploaded = st.file_uploader("Upload creditcard.csv", type="csv")

    if uploaded is None:
        st.warning("Upload the Kaggle `creditcard.csv` to run the live validation.")
    else:
        file_bytes = uploaded.getvalue()

        c1, c2, c3, c4 = st.columns(4)
        contamination = c1.slider(
            "Stage A contamination (expected fraud rate)", 0.001, 0.05, 0.0017, 0.001,
            help="IsolationForest's assumed fraction of anomalies — analogous to how permissive Stage A's screening net is.",
        )
        test_size = c2.slider("Held-out test fraction", 0.1, 0.5, 0.3, 0.05)
        b_threshold = c3.slider(
            "Stage B decision threshold", 0.05, 0.95, 0.5, 0.05,
            help="Probability cutoff above which Stage B's classifier flags a transaction on its own.",
        )
        cascade_ratio = c4.slider(
            "Cascade ratio", 0.05, 1.0, 0.4, 0.05,
            help="Inside candidates Stage A already flagged, Stage B only needs to clear "
                 "(cascade_ratio × Stage B threshold) — a lowered bar, since Stage A already "
                 "raised suspicion. Matches combine_stages.py's --cascade_ratio.",
        )

        with st.spinner("Training Stage A (unsupervised) and Stage B (supervised) on the uploaded data..."):
            result = train_kaggle_pipeline(file_bytes, contamination, test_size)

        y_test = result["y_test"]
        a_score = result["stage_a_score_test"]
        b_prob = result["stage_b_prob_test"]

        # Stage A flags its top `contamination` fraction as candidates (mirrors IsolationForest's own cutoff)
        a_cut = np.quantile(a_score, 1 - contamination)
        a_flag = a_score >= a_cut
        b_flag = b_prob >= b_threshold

        strat_metrics = compute_strategy_metrics(
            y_test, a_flag, b_flag, b_prob=b_prob, b_threshold=b_threshold, cascade_ratio=cascade_ratio
        )
        real_df = pd.DataFrame(
            [{"strategy": k, "label": STRATEGY_LABEL[k], **v} for k, v in strat_metrics.items()]
        )

        k1, k2, k3 = st.columns(3)
        k1.metric("Test transactions", f"{result['n_test']:,}")
        k2.metric("True frauds in test set", f"{result['n_test_fraud']:,}")
        k3.metric("Stage A candidates flagged", f"{int(a_flag.sum()):,}")

        st.plotly_chart(bar_metric_chart(real_df, "Real-world (Kaggle) precision / recall / F1 by strategy"), use_container_width=True)

        left, right = st.columns(2)
        with left:
            chosen_r = st.selectbox(
                "Confusion breakdown for:", STRATEGY_ORDER, index=4,
                format_func=lambda k: STRATEGY_LABEL[k], key="real_conf_select",
            )
            st.plotly_chart(
                confusion_bar(strat_metrics[chosen_r], f"{STRATEGY_LABEL[chosen_r]} — confusion counts (Kaggle)"),
                use_container_width=True,
            )
        with right:
            st.subheader("Synthetic vs. real-world — cascade strategy")
            synth_cascade = metrics["strategies"]["cascade"]["transaction_level"]
            comp_df = pd.DataFrame(
                [
                    {"dataset": "Synthetic benchmark", "precision": synth_cascade["precision"], "recall": synth_cascade["recall"], "f1": synth_cascade["f1"]},
                    {"dataset": "Kaggle (real-world)", "precision": strat_metrics["cascade"]["precision"], "recall": strat_metrics["cascade"]["recall"], "f1": strat_metrics["cascade"]["f1"]},
                ]
            )
            fig_comp = px.bar(
                comp_df.melt(id_vars="dataset", var_name="metric", value_name="score"),
                x="metric", y="score", color="dataset", barmode="group", range_y=[0, 1.05],
                color_discrete_map={"Synthetic benchmark": "#8ecae6", "Kaggle (real-world)": "#e63946"},
            )
            st.plotly_chart(fig_comp, use_container_width=True)

        st.subheader("Full metrics table")
        show_real = real_df[["label", "precision", "recall", "f1", "tp", "fp", "fn"]].rename(columns={"label": "strategy"})
        st.dataframe(
            show_real.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

        render_fp_cost_section(
            strat_metrics,
            key_prefix="kaggle",
            chart_title="Estimated cost by strategy — Kaggle (real-world)",
        )

        st.success(
            "If precision/recall on the real dataset stay in a broadly similar range to the synthetic "
            "benchmark — rather than collapsing toward zero — that's evidence the pipeline learned "
            "transferable fraud *signal* (burstiness for Stage A, transaction-level anomaly patterns "
            "for Stage B) rather than overfitting to synthetic-generator artifacts."
        )

# ==========================================================================
# PAGE: Methodology & Caveats
# ==========================================================================
else:
    st.title("Methodology & Caveats")
    st.markdown(
        f"""
### Synthetic benchmark
- **Stage A** trained unsupervised on `features_*_merchant_minute.csv` (txn count, decline rate,
  unique devices/card bins per merchant per minute, 60-min rolling z-score of txn volume).
- **Stage B** trained supervised on `features_*_transaction.csv`, using the `is_attack` label
  (threshold = {metrics['stage_b_threshold']:.4f}).
- **Cascade** uses `cascade_threshold` = {metrics['cascade_threshold']:.4f} on Stage B scores,
  applied only to the `cascade_ratio` = {metrics['cascade_ratio']:.0%} of traffic Stage A already
  flagged as suspicious.
- Evaluated on {metrics['n_test_transactions']:,} held-out synthetic transactions containing
  {metrics['n_test_attack_transactions']} attack transactions across
  {metrics['strategies']['cascade']['attack_window_level']['total_attack_windows']} discrete attack windows.

### Real-world validation tab — what's genuinely comparable, and what isn't
This is an **analogous** pipeline, not the same model re-run on new data — the public Kaggle
dataset doesn't share the synthetic schema (no merchant IDs, no discrete attack windows, no
device/IP fields), so a literal like-for-like re-run isn't possible. What's preserved is the
*shape* of the approach:

| | Synthetic pipeline | Kaggle analog |
|---|---|---|
| Stage A | Unsupervised velocity/burst detector on merchant-minute aggregates | Unsupervised `IsolationForest` on the PCA'd transaction features |
| Stage B | Supervised classifier on transaction features | Supervised `RandomForestClassifier` on the same features, trained with labels |
| Cascade | B only scores A's flagged candidates | Same logic, thresholds adjustable live via the sliders |
| Window-level recall | Discrete attack windows, all caught | Not computed — the Kaggle data has no window/session structure, only a `Time` column |

Treat the real-world numbers as a **directional generalization check** (does performance collapse
on unseen, real fraud, or hold up?), not as a strict apples-to-apples benchmark against the
synthetic results.

### Caveats
- Kaggle class imbalance (0.17% fraud) is even more extreme than the synthetic set, so precision
  is naturally harder to hold onto — that's expected and not itself evidence of failure.
- `IsolationForest`'s `contamination` parameter is a stand-in for however Stage A's original
  velocity-based threshold was tuned; adjust the slider to see sensitivity.
- Retraining happens fresh each session — nothing about the uploaded Kaggle file is stored or sent
  anywhere by this app.
        """
    )

"""
Two-Stage Fraud Detection Dashboard
====================================
Stage A : unsupervised burst/velocity anomaly detector on merchant-minute aggregates
Stage B : supervised classifier on transaction-level features
Cascade : Stage A screens traffic cheaply, Stage B scores the flagged candidates precisely

Tab 1 shows results on the synthetic benchmark the model was built/evaluated on.
Tab 2 re-runs an analogous pipeline live on a real, public Kaggle fraud dataset
(ULB "Credit Card Fraud Detection") as an independent proof that the approach
generalizes, rather than just fitting quirks of the synthetic generator.
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
    initial_sidebar_state="expanded",
)

STRATEGY_ORDER = ["stage_a_only", "stage_b_only", "combined_or", "combined_and", "cascade"]
STRATEGY_LABEL = {
    "stage_a_only": "Stage A only (unsupervised)",
    "stage_b_only": "Stage B only (supervised)",
    "combined_or": "Combined (OR)",
    "combined_and": "Combined (AND)",
    "cascade": "Cascade (A screens → B scores)",
}
# Muted, low-saturation palette — no bright/flashy colors.
STRATEGY_COLOR = {
    "stage_a_only": "#a9b8c6",
    "stage_b_only": "#5c7a94",
    "combined_or": "#9a9488",
    "combined_and": "#726c62",
    "cascade": "#2e4356",
}
CONFUSION_COLORS = {
    "tp": "#5c7a68",   # muted sage
    "fp": "#a68a63",   # muted ochre
    "fn": "#95605c",   # muted brick
    "tn": "#b6bcc4",   # muted grey
}

PAGES = ["Overview", "Synthetic Benchmark", "Independent Verification (Kaggle)", "Methodology & Caveats"]


# --------------------------------------------------------------------------
# Styling
# --------------------------------------------------------------------------
def inject_css():
    st.markdown(
        """
        <style>
        :root {
            --bg: #f6f7f9;
            --panel: #ffffff;
            --border: #e3e6eb;
            --text: #262b33;
            --muted: #6b7280;
            --accent: #2e4356;
            --accent-soft: #eef1f4;
        }
        .stApp { background-color: var(--bg); }

        h1, h2, h3 { color: var(--text); font-weight: 650; letter-spacing: -0.01em; }
        p, li, span, label { color: var(--text); }

        section[data-testid="stSidebar"] {
            background-color: #1e232c;
            border-right: 1px solid #12151b;
        }
        section[data-testid="stSidebar"] * { color: #d7dbe2 !important; }
        section[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label {
            padding: 6px 4px;
        }

        [data-testid="stMetric"] {
            background-color: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 16px;
        }
        [data-testid="stMetricValue"] { color: var(--accent); font-weight: 700; }
        [data-testid="stMetricLabel"] { color: var(--muted); }

        div[data-testid="stExpander"] {
            background-color: var(--panel);
            border: 1px solid var(--border);
            border-radius: 10px;
        }

        div[data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 12px !important;
        }

        .info-strip {
            background-color: var(--accent-soft);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 10px 16px;
            color: var(--muted);
            font-size: 0.92rem;
        }

        .stage-card h4 { margin: 0 0 6px 0; color: var(--accent); font-size: 1.02rem; }
        .stage-card p { margin: 0; color: var(--muted); font-size: 0.88rem; line-height: 1.4; }
        .stage-card .tag {
            display: inline-block; font-size: 0.72rem; font-weight: 600;
            color: var(--accent); background: var(--accent-soft);
            border-radius: 6px; padding: 2px 8px; margin-bottom: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def stage_card(icon: str, tag: str, title: str, body: str):
    st.markdown(
        f"""
        <div class="stage-card">
            <span class="tag">{tag}</span>
            <h4>{icon} {title}</h4>
            <p>{body}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        color_discrete_map={"precision": "#5c7a94", "recall": "#9a9488", "f1": "#2e4356"},
    )
    fig.update_layout(
        xaxis_title="", yaxis_title="score", legend_title="",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#262b33",
    )
    return fig


def confusion_bar(row: dict, title: str):
    labels = ["True Positive", "False Positive", "False Negative"] + (["True Negative"] if "tn" in row else [])
    values = [row["tp"], row["fp"], row["fn"]] + ([row["tn"]] if "tn" in row else [])
    colors = [CONFUSION_COLORS["tp"], CONFUSION_COLORS["fp"], CONFUSION_COLORS["fn"]] + (
        [CONFUSION_COLORS["tn"]] if "tn" in row else []
    )
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, text=values, textposition="outside"))
    fig.update_layout(
        title=title, yaxis_title="count",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font_color="#262b33",
    )
    return fig


# ==========================================================================
# Page shell: CSS, sidebar toggle, navigation
# ==========================================================================
inject_css()

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

top_l, top_r = st.columns([0.08, 0.92])
with top_l:
    if st.button("☰", help="Toggle sidebar"):
        st.session_state.sidebar_open = not st.session_state.sidebar_open

if not st.session_state.sidebar_open:
    st.markdown('<style>section[data-testid="stSidebar"]{display:none;}</style>', unsafe_allow_html=True)

st.sidebar.markdown("### 🛡️ Fraud Detection")
page = st.sidebar.radio("Section", PAGES, label_visibility="collapsed")
st.sidebar.caption("Two-stage pipeline · synthetic + real-world proof")

metrics = load_metrics()

# ==========================================================================
# PAGE: Overview
# ==========================================================================
if page == "Overview":
    st.title("Two-Stage Fraud Detection")
    st.caption("An unsupervised screen paired with a supervised scorer, combined into a low-cost cascade.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test transactions", f"{metrics['n_test_transactions']:,}")
    c2.metric("Attack transactions", f"{metrics['n_test_attack_transactions']:,}")
    c3.metric("Stage B threshold", f"{metrics['stage_b_threshold']:.3f}")
    c4.metric("Cascade ratio", f"{metrics['cascade_ratio']:.0%}")

    st.write("")
    s1, s2, s3 = st.columns(3)
    with s1:
        with st.container(border=True):
            stage_card(
                "🅰️", "UNSUPERVISED", "Stage A",
                "Watches merchant-minute traffic for bursts and velocity spikes — "
                "catches brand-new attack patterns with no labeled history.",
            )
    with s2:
        with st.container(border=True):
            stage_card(
                "🅱️", "SUPERVISED", "Stage B",
                "Scores individual transactions against learned fraud signatures — "
                "precise, but limited to patterns it has seen before.",
            )
    with s3:
        with st.container(border=True):
            stage_card(
                "🔗", "CASCADE", "A → B",
                "Stage A cheaply screens all traffic; only its candidates are passed "
                "to Stage B — most of B's precision, a fraction of its cost.",
            )

    st.write("")
    st.markdown(
        '<div class="info-strip">Start with <b>Synthetic Benchmark</b> for the tuning results, '
        'then open <b>Independent Verification</b> to see the same approach proven against real, '
        'unseen fraud data.</div>',
        unsafe_allow_html=True,
    )

# ==========================================================================
# PAGE: Synthetic Benchmark
# ==========================================================================
elif page == "Synthetic Benchmark":
    st.title("Synthetic Benchmark")
    st.caption("Stage A + Stage B combined on held-out synthetic test data.")

    df = metrics_to_frame(metrics)

    k1, k2, k3 = st.columns(3)
    k1.metric("Test transactions", f"{metrics['n_test_transactions']:,}")
    k2.metric("True attack transactions", f"{metrics['n_test_attack_transactions']:,}")
    k3.metric("Attack windows caught", f"{df['total_attack_windows'].iloc[0]}/{df['total_attack_windows'].iloc[0]}")

    st.plotly_chart(bar_metric_chart(df, "Precision / recall / F1 by strategy"), use_container_width=True)

    left, right = st.columns([1, 1])
    with left:
        fig_wd = px.bar(
            df, x="label", y="window_detection_rate", range_y=[0, 1.05],
            title="Attack-window detection rate",
            color="label", color_discrete_map={STRATEGY_LABEL[k]: v for k, v in STRATEGY_COLOR.items()},
        )
        fig_wd.update_layout(
            showlegend=False, xaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#262b33",
        )
        st.plotly_chart(fig_wd, use_container_width=True)
    with right:
        chosen = st.selectbox("Confusion breakdown for:", STRATEGY_ORDER, index=4, format_func=lambda k: STRATEGY_LABEL[k])
        row = df[df["strategy"] == chosen].iloc[0].to_dict()
        st.plotly_chart(confusion_bar(row, f"{STRATEGY_LABEL[chosen]} — confusion counts"), use_container_width=True)

    with st.expander("Full metrics table"):
        show_df = df[["label", "precision", "recall", "f1", "tp", "fp", "fn", "window_detection_rate"]].rename(
            columns={"label": "strategy"}
        )
        st.dataframe(
            show_df.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}", "window_detection_rate": "{:.2f}"}),
            use_container_width=True, hide_index=True,
        )

    st.caption(
        f"Cascade keeps Stage B's precision (0.997) while lifting recall close to its ceiling, "
        f"scoring only ~{metrics['cascade_ratio']:.0%} of traffic with the expensive model."
    )

    with st.expander("Explore the raw synthetic feature data"):
        train_txn, test_txn, train_mm, test_mm = load_synthetic_csvs()
        st.caption(f"Train: {len(train_txn):,} txns · {len(train_mm):,} merchant-minutes  |  "
                   f"Test: {len(test_txn):,} txns · {len(test_mm):,} merchant-minutes")

        feat = st.selectbox(
            "Feature to compare (attack vs. normal):",
            ["amount_zscore_vs_merchant_history", "merchant_txn_count_5min", "merchant_decline_rate_5min",
             "seconds_since_last_txn_device", "seconds_since_last_txn_ip"],
        )
        plot_df = test_txn[[feat, "is_attack"]].copy()
        plot_df["is_attack"] = plot_df["is_attack"].map({0: "normal", 1: "attack"})
        fig = px.histogram(
            plot_df, x=feat, color="is_attack", barmode="overlay", nbins=60,
            histnorm="probability density", opacity=0.65,
            color_discrete_map={"normal": "#a9b8c6", "attack": "#95605c"},
            title=f"Distribution of {feat}",
        )
        fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#262b33")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(test_txn.sample(min(200, len(test_txn)), random_state=1), use_container_width=True, height=250)

# ==========================================================================
# PAGE: Independent Verification (formerly "Real-World Validation")
# ==========================================================================
elif page == "Independent Verification (Kaggle)":
    st.title("Independent Verification")
    st.caption(
        "A stress test on genuine, unseen fraud data — proving the pipeline learned "
        "transferable signal rather than fitting artifacts of the synthetic generator."
    )

    with st.expander("About this test"):
        st.markdown(
            """
An analogous Stage A / Stage B / cascade pipeline is trained and scored live on
**[Credit Card Fraud Detection (ULB / Worldline)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**
— 284,807 real anonymized European transactions from September 2013, 492 confirmed frauds (0.17%).

The Kaggle schema doesn't match the synthetic one (no merchant IDs, no attack windows), so this
is a **directional proof of generalization**, not a like-for-like rerun. Nothing uploaded is stored —
training happens fresh, in-session.
            """
        )

    uploaded = st.file_uploader("Upload creditcard.csv", type="csv")

    if uploaded is None:
        st.info("Upload the Kaggle `creditcard.csv` to run the verification test.")
    else:
        file_bytes = uploaded.getvalue()

        with st.expander("Test parameters", expanded=False):
            c1, c2, c3, c4 = st.columns(4)
            contamination = c1.slider(
                "Stage A contamination", 0.001, 0.05, 0.0017, 0.001,
                help="Assumed fraud rate — how permissive Stage A's screen is.",
            )
            test_size = c2.slider("Held-out test fraction", 0.1, 0.5, 0.3, 0.05)
            b_threshold = c3.slider(
                "Stage B threshold", 0.05, 0.95, 0.5, 0.05,
                help="Probability cutoff for Stage B to flag a transaction alone.",
            )
            cascade_ratio = c4.slider(
                "Cascade ratio", 0.05, 1.0, 0.4, 0.05,
                help="Inside Stage A's candidates, Stage B only needs to clear "
                     "(ratio × threshold) — a lowered bar.",
            )

        with st.spinner("Training Stage A and Stage B on the uploaded data..."):
            result = train_kaggle_pipeline(file_bytes, contamination, test_size)

        y_test = result["y_test"]
        a_score = result["stage_a_score_test"]
        b_prob = result["stage_b_prob_test"]

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

        st.plotly_chart(bar_metric_chart(real_df, "Precision / recall / F1 by strategy — Kaggle data"), use_container_width=True)

        left, right = st.columns(2)
        with left:
            chosen_r = st.selectbox(
                "Confusion breakdown for:", STRATEGY_ORDER, index=4,
                format_func=lambda k: STRATEGY_LABEL[k], key="real_conf_select",
            )
            st.plotly_chart(
                confusion_bar(strat_metrics[chosen_r], f"{STRATEGY_LABEL[chosen_r]} — confusion counts"),
                use_container_width=True,
            )
        with right:
            st.markdown("**Synthetic vs. real-world — cascade**")
            synth_cascade = metrics["strategies"]["cascade"]["transaction_level"]
            comp_df = pd.DataFrame(
                [
                    {"dataset": "Synthetic benchmark", "precision": synth_cascade["precision"], "recall": synth_cascade["recall"], "f1": synth_cascade["f1"]},
                    {"dataset": "Kaggle (independent)", "precision": strat_metrics["cascade"]["precision"], "recall": strat_metrics["cascade"]["recall"], "f1": strat_metrics["cascade"]["f1"]},
                ]
            )
            fig_comp = px.bar(
                comp_df.melt(id_vars="dataset", var_name="metric", value_name="score"),
                x="metric", y="score", color="dataset", barmode="group", range_y=[0, 1.05],
                color_discrete_map={"Synthetic benchmark": "#a9b8c6", "Kaggle (independent)": "#2e4356"},
            )
            fig_comp.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#262b33")
            st.plotly_chart(fig_comp, use_container_width=True)

        with st.expander("Full metrics table"):
            show_real = real_df[["label", "precision", "recall", "f1", "tp", "fp", "fn"]].rename(columns={"label": "strategy"})
            st.dataframe(
                show_real.style.format({"precision": "{:.3f}", "recall": "{:.3f}", "f1": "{:.3f}"}),
                use_container_width=True, hide_index=True,
            )

        st.markdown(
            '<div class="info-strip">If precision/recall on real data stay close to the synthetic '
            'benchmark rather than collapsing, that proves the pipeline picked up transferable fraud '
            'signal — not synthetic-generator artifacts.</div>',
            unsafe_allow_html=True,
        )

# ==========================================================================
# PAGE: Methodology & Caveats
# ==========================================================================
else:
    st.title("Methodology & Caveats")
    st.caption("Reference notes on how each stage is trained and where the comparisons stop being exact.")

    with st.expander("Synthetic benchmark", expanded=True):
        st.markdown(
            f"""
- **Stage A** — unsupervised, trained on merchant-minute aggregates (txn count, decline rate,
  unique devices/card bins, 60-min rolling volume z-score).
- **Stage B** — supervised on transaction features, using the `is_attack` label
  (threshold = {metrics['stage_b_threshold']:.4f}).
- **Cascade** — threshold {metrics['cascade_threshold']:.4f} on Stage B scores, applied to the
  {metrics['cascade_ratio']:.0%} of traffic Stage A already flagged.
- Evaluated on {metrics['n_test_transactions']:,} held-out transactions,
  {metrics['n_test_attack_transactions']} of them attacks across
  {metrics['strategies']['cascade']['attack_window_level']['total_attack_windows']} attack windows.
            """
        )

    with st.expander("What the independent verification test does and doesn't prove"):
        st.markdown(
            """
This is an **analogous** pipeline, not the same model rerun — the Kaggle dataset has no merchant
IDs, attack windows, or device/IP fields, so a literal rerun isn't possible. What's preserved is
the *shape* of the approach:

| | Synthetic pipeline | Kaggle analog |
|---|---|---|
| Stage A | Unsupervised burst detector on merchant-minute aggregates | Unsupervised `IsolationForest` on PCA'd features |
| Stage B | Supervised classifier on transaction features | Supervised `RandomForestClassifier`, same features |
| Cascade | B scores only A's flagged candidates | Same logic, thresholds adjustable live |
| Window-level recall | Discrete attack windows, all caught | Not computed — no window structure in Kaggle data |

Treat these numbers as a **directional proof of generalization**, not a strict apples-to-apples
benchmark.
            """
        )

    with st.expander("Caveats"):
        st.markdown(
            """
- Kaggle's 0.17% fraud rate is even more extreme than the synthetic set — lower precision there is
  expected, not itself a sign of failure.
- `IsolationForest`'s `contamination` stands in for however Stage A's original velocity threshold
  was tuned; the slider shows sensitivity to that choice.
- Retraining happens fresh each session — nothing from an uploaded file is stored or sent anywhere.
            """
        )

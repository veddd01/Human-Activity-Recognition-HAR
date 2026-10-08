"""
Human Activity Recognition — Portfolio Dashboard
A stunning, modern Streamlit app for exploring HAR model predictions and analytics.
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import os
import base64

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="HAR · Human Activity Recognition",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CSS INJECTION — Premium dark‑theme styling
# ══════════════════════════════════════════════════════════════════════════════

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Reset & base ────────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0e1117;
    font-family: 'Inter', sans-serif;
    color: #e0e0e0;
}

/* Hide Streamlit chrome */
#MainMenu, header[data-testid="stHeader"], footer,
div[data-testid="stToolbar"] { display: none !important; }

/* ── Scrollbar ───────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0e1117; }
::-webkit-scrollbar-thumb { background: #333; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #555; }

/* ── Sidebar ─────────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141c 0%, #0e1117 100%);
    border-right: 1px solid rgba(102,126,234,0.15);
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li { color: #b0b8c8; font-size: 0.92rem; }

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(255,255,255,0.03);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 10px 22px;
    font-weight: 600;
    font-size: 0.88rem;
    color: #9e9e9e;
    background: transparent;
    transition: all 0.3s ease;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(118,75,162,0.25));
    color: #fff !important;
    border: none;
    box-shadow: 0 2px 12px rgba(102,126,234,0.15);
}

/* ── Glass card ──────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 28px;
    margin-bottom: 16px;
    transition: transform 0.35s ease, box-shadow 0.35s ease;
}
.glass-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 32px rgba(102,126,234,0.12);
}

/* ── KPI / Metric card ───────────────────────────────────────────────────── */
.kpi-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    position: relative;
    overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 16px 16px 0 0;
}
.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
}
.kpi-card .kpi-value {
    font-size: 2.2rem;
    font-weight: 800;
    margin: 8px 0 4px;
}
.kpi-card .kpi-label {
    font-size: 0.82rem;
    font-weight: 500;
    color: #9e9e9e;
    text-transform: uppercase;
    letter-spacing: 1.2px;
}

/* Border-top color helpers */
.kpi-purple::before { background: linear-gradient(90deg,#667eea,#764ba2); }
.kpi-teal::before   { background: linear-gradient(90deg,#00d4aa,#00b894); }
.kpi-amber::before  { background: linear-gradient(90deg,#ffa726,#ff7043); }
.kpi-blue::before   { background: linear-gradient(90deg,#42a5f5,#478ed1); }

.kpi-purple .kpi-value { color: #667eea; }
.kpi-teal   .kpi-value { color: #00d4aa; }
.kpi-amber  .kpi-value { color: #ffa726; }
.kpi-blue   .kpi-value { color: #42a5f5; }

/* ── Activity badge card ─────────────────────────────────────────────────── */
.activity-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}
.activity-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 24px rgba(0,0,0,0.25);
}
.activity-card .emoji { font-size: 2.2rem; margin-bottom: 8px; }
.activity-card .name  { font-weight: 600; font-size: 0.92rem; color: #e0e0e0; }

/* ── Prediction result card ──────────────────────────────────────────────── */
.pred-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(10px);
    border-radius: 16px;
    padding: 22px 18px;
    text-align: center;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    border: 2px solid transparent;
}
.pred-card:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.3);
}
.pred-card.correct  { border-color: #00d4aa; }
.pred-card.wrong    { border-color: #ef5350; }

.pred-card .model-name {
    font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: #9e9e9e; margin-bottom: 10px;
}
.pred-card .pred-activity {
    font-size: 1.15rem; font-weight: 700; margin-bottom: 6px;
}
.pred-card .pred-emoji { font-size: 1.8rem; margin-bottom: 6px; }

/* progress bar inside pred card */
.conf-bar-bg {
    background: rgba(255,255,255,0.08);
    border-radius: 6px;
    height: 8px;
    width: 100%;
    margin-top: 10px;
    overflow: hidden;
}
.conf-bar {
    height: 100%;
    border-radius: 6px;
    transition: width 0.6s ease;
}
.conf-bar.correct { background: linear-gradient(90deg,#00d4aa,#00b894); }
.conf-bar.wrong   { background: linear-gradient(90deg,#ef5350,#e57373); }

/* ── True-label hero ─────────────────────────────────────────────────────── */
.true-label-card {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 18px;
    padding: 30px;
    text-align: center;
    margin-bottom: 24px;
}
.true-label-card .label { font-size: 0.82rem; color: #9e9e9e; text-transform: uppercase; letter-spacing: 1.2px; }
.true-label-card .emoji { font-size: 3rem; margin: 10px 0; }
.true-label-card .activity { font-size: 1.6rem; font-weight: 800; color: #fff; }

/* ── Gradient heading ────────────────────────────────────────────────────── */
.gradient-text {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #42a5f5 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 900;
}

.hero-title {
    font-size: 2.8rem;
    line-height: 1.15;
    margin-bottom: 6px;
}

.hero-sub {
    color: #9e9e9e;
    font-size: 1.05rem;
    font-weight: 400;
    margin-bottom: 32px;
}

/* ── Badges (About tab) ──────────────────────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 4px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.06);
    color: #c0c8d8;
}

/* ── Section divider ─────────────────────────────────────────────────────── */
.section-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.3), transparent);
    margin: 32px 0;
}

/* ── Streamlit metric overrides ──────────────────────────────────────────── */
[data-testid="stMetricValue"] { font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; text-transform: uppercase; letter-spacing: 1px; }

/* ── Data-frame styling ──────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    overflow: hidden;
}

/* ── Plotly chart container ──────────────────────────────────────────────── */
.stPlotlyChart {
    border-radius: 14px;
    overflow: hidden;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    font-weight: 600;
    font-size: 0.92rem;
    color: #b0b8c8;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

_APP_ROOT = Path(__file__).resolve().parent
DATA_DIR = _APP_ROOT / "data" / "UCI_HAR_Dataset" / "UCI HAR Dataset"
MODELS_DIR = _APP_ROOT / "models"

ACTIVITY_EMOJI = {
    "WALKING": "🚶",
    "WALKING_UPSTAIRS": "⬆️",
    "WALKING_DOWNSTAIRS": "⬇️",
    "SITTING": "🪑",
    "STANDING": "🧍",
    "LAYING": "🛏️",
}

ACTIVITY_COLORS = {
    "WALKING": "#667eea",
    "WALKING_UPSTAIRS": "#764ba2",
    "WALKING_DOWNSTAIRS": "#42a5f5",
    "SITTING": "#ffa726",
    "STANDING": "#00d4aa",
    "LAYING": "#ef5350",
}

MODEL_FILES = {
    "Logistic Regression":  ("logistic_regression.pkl",  "sklearn"),
    "Decision Tree":        ("decision_tree.pkl",        "sklearn"),
    "SVM":                  ("svm.pkl",                  "sklearn"),
    "Random Forest":        ("random_forest.pkl",        "sklearn"),
    "Gradient Boosting":    ("gradient_boosting.pkl",    "sklearn"),
    "XGBoost":              ("xgboost.pkl",              "sklearn"),
    "LightGBM":             ("lightgbm.pkl",             "sklearn"),
    "Ensemble":             ("ensemble.pkl",             "sklearn"),
    "Neural Network":       ("nn_model.keras",           "keras"),
    "CNN-LSTM":             ("cnn_lstm.keras",           "keras"),
}


PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#c0c8d8"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)

# ══════════════════════════════════════════════════════════════════════════════
# DATA & MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_labels_and_test_data():
    """Load test features, labels, and activity‑label mapping from UCI dataset."""
    if not DATA_DIR.exists():
        return None, None, None

    features_file = DATA_DIR / "features.txt"
    features = pd.read_csv(features_file, sep=r"\s+", header=None, usecols=[1])
    feature_names = features[1].tolist()

    # De-duplicate feature names
    s = pd.Series(feature_names)
    dup_mask = s.duplicated(keep=False)
    s[dup_mask] = s[dup_mask] + "_" + s.groupby(s).cumcount().astype(str)
    feature_names = s.tolist()

    activity_labels = pd.read_csv(
        DATA_DIR / "activity_labels.txt", sep=r"\s+", header=None,
        names=["label_id", "activity_name"],
    )

    X_test = pd.read_csv(
        DATA_DIR / "test" / "X_test.txt", sep=r"\s+", header=None, names=feature_names,
    )
    y_test = pd.read_csv(
        DATA_DIR / "test" / "y_test.txt", sep=r"\s+", header=None, names=["label_id"],
    )
    y_test["label"] = y_test["label_id"] - 1

    return X_test, y_test["label"], activity_labels


@st.cache_resource(show_spinner=False)
def load_all_models():
    """Load every available model from disk."""
    models: dict = {}
    for display_name, (filename, kind) in MODEL_FILES.items():
        path = MODELS_DIR / filename
        if not path.exists():
            continue
        try:
            if kind == "keras":
                import tensorflow as tf
                from tensorflow import keras
                models[display_name] = keras.models.load_model(path)
            else:
                models[display_name] = joblib.load(path)
        except Exception:
            pass  # Silently skip corrupt / incompatible files
    return models


@st.cache_data(show_spinner=False)
def load_json(path: Path):
    """Load a JSON file; return None on failure.

    If the JSON root is a list of objects with a ``model_name`` key, convert it
    to a dict keyed by ``model_name`` so callers can use ``.items()`` uniformly.
    Also normalises common field names used across the codebase.
    """
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            result = {}
            for entry in data:
                key = entry.get("model_name", entry.get("name", str(len(result))))
                # Normalise field names so downstream code finds them
                if "f1_macro" in entry and "f1" not in entry:
                    entry["f1"] = entry["f1_macro"]
                if "inference_time_per_sample_ms" in entry and "inference_speed" not in entry:
                    # Store in seconds (downstream multiplies by 1000)
                    entry["inference_speed"] = entry["inference_time_per_sample_ms"] / 1000
                if "training_time_s" in entry and "training_time" not in entry:
                    entry["training_time"] = entry["training_time_s"]
                result[key] = entry
            return result
        return data
    except Exception:
        return None


def predict_single(model, model_name: str, sample: pd.DataFrame):
    """Run a single prediction; return (predicted_label_index, confidence_pct)."""
    try:
        if model_name == "CNN-LSTM":
            # CNN-LSTM requires raw (128, 9) temporal sequences — cannot
            # predict from 561 flat features.  Skip gracefully.
            return None, None
        elif model_name == "Neural Network":
            val = sample.values if isinstance(sample, pd.DataFrame) else sample
            probs = model.predict(val, verbose=0)
            idx = int(np.argmax(probs[0]))
            conf = float(np.max(probs[0])) * 100
        else:
            idx = int(model.predict(sample)[0])
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(sample)[0]
                conf = float(np.max(probs)) * 100
            elif hasattr(model, "decision_function"):
                dec = model.decision_function(sample)[0]
                exp = np.exp(dec - np.max(dec))
                conf = float(np.max(exp / exp.sum())) * 100
            else:
                conf = 100.0
        return idx, conf
    except Exception as e:
        return None, None



def label_name(label_idx, activity_labels):
    """Map a 0‑based label index → activity name string."""
    row = activity_labels[activity_labels["label_id"] == label_idx + 1]
    if row.empty:
        return "UNKNOWN"
    return row["activity_name"].values[0]


# ── Load data ────────────────────────────────────────────────────────────────
X_test, y_test, activity_labels = load_labels_and_test_data()
models = load_all_models() if X_test is not None else {}
training_meta = load_json(MODELS_DIR / "training_metadata.json")
eval_results  = load_json(MODELS_DIR / "evaluation_results.json")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; padding:18px 0 8px;">
            <div style="font-size:2.4rem;">🏃</div>
            <div class="gradient-text" style="font-size:1.35rem; font-weight:800; line-height:1.25; margin-top:4px;">
                Human Activity<br>Recognition
            </div>
            <div style="color:#666; font-size:0.75rem; margin-top:6px; letter-spacing:1px; text-transform:uppercase;">
                Portfolio Dashboard
            </div>
        </div>
        <div class="section-divider"></div>
        """,
        unsafe_allow_html=True,
    )

    # Project stats
    st.markdown("##### 📌 Quick Stats")
    if X_test is not None:
        st.markdown(f"- **Test Samples**: {len(X_test):,}")
    st.markdown(f"- **Models Loaded**: {len(models)}")
    st.markdown(f"- **Activities**: 6")
    st.markdown(f"- **Features**: 561")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Model filter for predict tab
    if models:
        selected_models = st.multiselect(
            "🔧 Models to Show",
            options=list(models.keys()),
            default=list(models.keys()),
            help="Filter which models appear on the Predict tab.",
        )
    else:
        selected_models = []

    # Theme switcher
    theme = st.radio("🎨 Theme Mode", ["Dark", "Light"], horizontal=True)
    if theme == "Light":
        st.markdown(
            """
            <style>
            .stApp { background: #f8f9fa !important; color: #212529 !important; }
            section[data-testid="stSidebar"] { background: #ffffff !important; }
            .glass-card, .kpi-card, .activity-card, .pred-card { background: #ffffff !important; color: #212529 !important; border: 1px solid #e9ecef !important; }
            .hero-sub, .kpi-label { color: #6c757d !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div style="text-align:center; padding:12px 0;">
            <span style="color:#444; font-size:0.72rem;">
                Built with Streamlit · scikit‑learn · TensorFlow
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# GUARD — dataset not found
# ══════════════════════════════════════════════════════════════════════════════

if X_test is None:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; margin-top:60px;">
            <div style="font-size:3rem; margin-bottom:12px;">⚠️</div>
            <div style="font-size:1.3rem; font-weight:700; color:#ffa726;">Dataset Not Found</div>
            <p style="color:#9e9e9e; margin-top:8px;">
                Run <code>python src/data.py</code> to download and prepare the UCI HAR Dataset.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

if not models:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; margin-top:60px;">
            <div style="font-size:3rem; margin-bottom:12px;">🤖</div>
            <div style="font-size:1.3rem; font-weight:700; color:#ffa726;">No Models Found</div>
            <p style="color:#9e9e9e; margin-top:8px;">
                Run <code>python src/train.py</code> to train and save models, then refresh.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_dashboard, tab_live, tab_predict, tab_compare, tab_analytics, tab_health, tab_about = st.tabs(
    ["🏠 Dashboard", "🔴 Live Monitor", "🔮 Predict", "📊 Model Comparison", "📈 Analytics", "❤️ Health Metrics", "ℹ️ About"]
)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 1 — DASHBOARD                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab_dashboard:
    # Hero
    st.markdown(
        """
        <div class="hero-title gradient-text">Human Activity Recognition System</div>
        <p class="hero-sub">
            Real-time classification of human activities from smartphone accelerometer &amp; gyroscope data
            using state-of-the-art machine learning models.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI cards ────────────────────────────────────────────────────────────
    best_acc = "—"
    best_model_name = ""
    if eval_results:
        try:
            best_model_name = max(eval_results, key=lambda k: eval_results[k].get("accuracy", 0))
            best_acc = f"{eval_results[best_model_name]['accuracy'] * 100:.1f}%"
        except Exception:
            pass

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="kpi-card kpi-purple">
                <div class="kpi-label">Test Samples</div>
                <div class="kpi-value">{len(X_test):,}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="kpi-card kpi-teal">
                <div class="kpi-label">Models Loaded</div>
                <div class="kpi-value">{len(models)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="kpi-card kpi-amber">
                <div class="kpi-label">Best Accuracy</div>
                <div class="kpi-value">{best_acc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Activities</div>
                <div class="kpi-value">6</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Activity overview grid (3 × 2) ──────────────────────────────────────
    st.markdown("### Activity Classes")
    activities = list(ACTIVITY_EMOJI.items())
    for row_start in range(0, 6, 3):
        cols = st.columns(3)
        for i, col in enumerate(cols):
            idx = row_start + i
            if idx < len(activities):
                name, emoji = activities[idx]
                color = ACTIVITY_COLORS.get(name, "#667eea")
                with col:
                    st.markdown(
                        f"""
                        <div class="activity-card" style="border-left:3px solid {color};">
                            <div class="emoji">{emoji}</div>
                            <div class="name">{name.replace("_"," ").title()}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Quick accuracy comparison (Plotly) ───────────────────────────────────
    st.markdown("### Model Accuracy Overview")

    # Try to get accuracies from eval_results or compute on the fly
    acc_data = {}
    if eval_results:
        for m, info in eval_results.items():
            if "accuracy" in info:
                acc_data[m] = info["accuracy"] * 100
    else:
        # Compute on‑the‑fly for each model
        for m_name, m_obj in models.items():
            try:
                if m_name == "Neural Network":
                    preds = np.argmax(m_obj.predict(X_test.values, verbose=0), axis=1)
                else:
                    preds = m_obj.predict(X_test)
                acc_data[m_name] = float((preds == y_test.values).mean()) * 100
            except Exception:
                pass

    if acc_data:
        import plotly.graph_objects as go

        sorted_items = sorted(acc_data.items(), key=lambda x: x[1])
        names = [x[0] for x in sorted_items]
        accs  = [x[1] for x in sorted_items]
        colors = [
            f"rgba(102,126,234,{0.5 + 0.5 * i / max(len(accs)-1,1)})"
            for i in range(len(accs))
        ]

        fig = go.Figure(go.Bar(
            x=accs,
            y=names,
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(width=0),
                cornerradius=6,
            ),
            text=[f"{a:.1f}%" for a in accs],
            textposition="outside",
            textfont=dict(size=13, color="#e0e0e0"),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=50 + 55 * len(names),
            xaxis=dict(range=[0, 105], showgrid=False, zeroline=False, title=""),
            yaxis=dict(showgrid=False, title=""),
            title=None,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Train models and run evaluation to see accuracy comparison.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB — LIVE MONITOR                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab_live:
    import plotly.graph_objects as go
    from src.sensor_simulator import generate_sensor_window, extract_features_from_window
    from src.prediction_smoother import PredictionSmoother

    st.markdown('<div class="hero-title gradient-text" style="font-size:1.8rem;">🔴 Real-Time Sensor Stream</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="font-size:0.92rem;">Simulate live smartphone sensor streaming (50 Hz), continuous prediction, and rolling window smoothing.</p>', unsafe_allow_html=True)

    c_sim1, c_sim2, c_sim3 = st.columns([2, 2, 1])
    with c_sim1:
        sim_activity = st.selectbox(
            "🏃 Activity Profile to Simulate",
            ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"],
        )
    with c_sim2:
        sim_model = st.selectbox(
            "🤖 Model for Real-Time Inference",
            options=list(models.keys()),
        )
    with c_sim3:
        window_size = st.slider("🌊 Smoother Window", 1, 10, 5)

    if "live_history" not in st.session_state:
        st.session_state["live_history"] = []
    if "live_smoother" not in st.session_state:
        st.session_state["live_smoother"] = PredictionSmoother(window_size=window_size)
    else:
        st.session_state["live_smoother"].window_size = window_size

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⚡ Generate Live Sensor Sample", use_container_width=True):
            win = generate_sensor_window(sim_activity)
            feats = extract_features_from_window(win)
            feats_df = pd.DataFrame([feats], columns=X_test.columns)

            pred_idx, conf = predict_single(models[sim_model], sim_model, feats_df)
            if pred_idx is not None:
                raw_name = label_name(pred_idx, activity_labels)
                sm_res = st.session_state["live_smoother"].add(pred_idx, (conf or 100) / 100)
                sm_name = label_name(sm_res.smoothed_label, activity_labels)

                st.session_state["live_history"].append({
                    "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
                    "simulated_activity": sim_activity,
                    "raw_predicted": raw_name,
                    "smoothed_predicted": sm_name,
                    "confidence_pct": round(conf or 100, 1),
                    "smoothed_confidence_pct": round(sm_res.smoothed_confidence * 100, 1),
                    "channels": win.channels,
                })
    with col_btn2:
        if st.button("🗑️ Clear Live History", use_container_width=True):
            st.session_state["live_history"] = []
            st.session_state["live_smoother"].reset()
            st.rerun()

    if st.session_state["live_history"]:
        latest = st.session_state["live_history"][-1]

        # Live prediction status card
        kc1, kc2, kc3 = st.columns(3)
        with kc1:
            emoji = ACTIVITY_EMOJI.get(latest["raw_predicted"], "❓")
            st.markdown(
                f"""
                <div class="kpi-card kpi-purple">
                    <div class="kpi-label">Raw Prediction</div>
                    <div class="kpi-value">{emoji} {latest['raw_predicted']}</div>
                    <div style="font-size:0.8rem; color:#9e9e9e; margin-top:4px;">Conf: {latest['confidence_pct']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kc2:
            emoji_sm = ACTIVITY_EMOJI.get(latest["smoothed_predicted"], "❓")
            st.markdown(
                f"""
                <div class="kpi-card kpi-teal">
                    <div class="kpi-label">Smoothed Prediction</div>
                    <div class="kpi-value">{emoji_sm} {latest['smoothed_predicted']}</div>
                    <div style="font-size:0.8rem; color:#9e9e9e; margin-top:4px;">Conf: {latest['smoothed_confidence_pct']}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with kc3:
            st.markdown(
                f"""
                <div class="kpi-card kpi-amber">
                    <div class="kpi-label">Simulated Signal</div>
                    <div class="kpi-value">{ACTIVITY_EMOJI.get(sim_activity, '❓')} {sim_activity}</div>
                    <div style="font-size:0.8rem; color:#9e9e9e; margin-top:4px;">128 Samples @ 50 Hz</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        # Real-time Waveform Chart (Plotly)
        st.markdown("### 📡 Live Sensor Waveform (Body Accelerometer & Gyroscope)")
        ch_data = latest["channels"]
        t_axis = np.arange(128) / 50.0

        fig_sig = go.Figure()
        fig_sig.add_trace(go.Scatter(x=t_axis, y=ch_data["body_acc_x"], name="Acc X", line=dict(color="#667eea", width=1.5)))
        fig_sig.add_trace(go.Scatter(x=t_axis, y=ch_data["body_acc_y"], name="Acc Y", line=dict(color="#764ba2", width=1.5)))
        fig_sig.add_trace(go.Scatter(x=t_axis, y=ch_data["body_acc_z"], name="Acc Z", line=dict(color="#42a5f5", width=1.5)))
        fig_sig.add_trace(go.Scatter(x=t_axis, y=ch_data["body_gyro_x"], name="Gyro X", line=dict(color="#00d4aa", width=1.5, dash="dash")))
        fig_sig.add_trace(go.Scatter(x=t_axis, y=ch_data["body_gyro_y"], name="Gyro Y", line=dict(color="#ffa726", width=1.5, dash="dash")))
        fig_sig.add_trace(go.Scatter(x=t_axis, y=ch_data["body_gyro_z"], name="Gyro Z", line=dict(color="#ef5350", width=1.5, dash="dash")))

        fig_sig.update_layout(
            **PLOTLY_LAYOUT, height=350,
            xaxis=dict(title="Time (seconds)", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Signal Amplitude (g / rad/s)", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
        )
        st.plotly_chart(fig_sig, use_container_width=True)

        # Prediction History Table
        st.markdown("### 📜 Streamed Predictions Timeline")
        df_hist = pd.DataFrame(st.session_state["live_history"])[["timestamp", "simulated_activity", "raw_predicted", "smoothed_predicted", "confidence_pct", "smoothed_confidence_pct"]]
        st.dataframe(df_hist.iloc[::-1], use_container_width=True)
    else:
        st.info("Click **'Generate Live Sensor Sample'** above to start live streaming predictions!")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 2 — PREDICT                                                         ║
# ╚════════════════════════════════════════════════════════════════════════════╝


with tab_predict:
    st.markdown('<div class="hero-title gradient-text" style="font-size:1.8rem;">🔮 Activity Prediction</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="font-size:0.92rem;">Select a test sample or upload your own data to see how each model classifies the activity.</p>', unsafe_allow_html=True)

    mode = st.radio("Input Mode", ["Test Sample", "Upload CSV"], horizontal=True, label_visibility="collapsed")

    sample_features = None
    true_label_idx = None

    if mode == "Test Sample":
        col_slider, col_btn = st.columns([4, 1])
        with col_slider:
            sample_idx = st.slider("Select test sample index", 0, len(X_test) - 1, 0, key="pred_slider")
        with col_btn:
            st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
            if st.button("🎲 Random", use_container_width=True):
                st.session_state["pred_slider"] = int(np.random.randint(0, len(X_test)))
                st.rerun()

        sample_idx = st.session_state.get("pred_slider", 0)
        sample_features = X_test.iloc[[sample_idx]]
        true_label_idx = int(y_test.iloc[sample_idx])
    else:
        uploaded = st.file_uploader("Upload a CSV with 561 features (one row)", type=["csv"])
        if uploaded is not None:
            try:
                df_up = pd.read_csv(uploaded, header=None if True else 0)
                if df_up.shape[1] == 561:
                    sample_features = df_up.iloc[[0]]
                    sample_features.columns = X_test.columns
                    st.success(f"✅ Loaded {df_up.shape[0]} row(s) — using the first row for prediction.")
                else:
                    st.error(f"Expected 561 features but got {df_up.shape[1]}.")
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

    if sample_features is not None:
        # True label card (only for test samples)
        if true_label_idx is not None:
            true_name = label_name(true_label_idx, activity_labels)
            true_emoji = ACTIVITY_EMOJI.get(true_name, "❓")
            st.markdown(
                f"""
                <div class="true-label-card">
                    <div class="label">Ground Truth</div>
                    <div class="emoji">{true_emoji}</div>
                    <div class="activity">{true_name.replace("_"," ").title()}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Predictions grid
        show_models = {k: v for k, v in models.items() if k in selected_models} if selected_models else models
        if not show_models:
            st.warning("No models selected — use the sidebar to pick at least one model.")
        else:
            cols_per_row = min(len(show_models), 3)
            cols = st.columns(cols_per_row)
            for i, (m_name, m_obj) in enumerate(show_models.items()):
                with cols[i % cols_per_row]:
                    pred_idx, conf = predict_single(m_obj, m_name, sample_features)
                    if pred_idx is None:
                        st.markdown(
                            f"""
                            <div class="pred-card">
                                <div class="model-name">{m_name}</div>
                                <div style="color:#ef5350;">Prediction Error</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        continue

                    pred_name = label_name(pred_idx, activity_labels)
                    pred_emoji = ACTIVITY_EMOJI.get(pred_name, "❓")

                    if true_label_idx is not None:
                        correct = pred_idx == true_label_idx
                        cls = "correct" if correct else "wrong"
                        badge = "✅" if correct else "❌"
                    else:
                        cls = "correct"
                        badge = ""

                    conf_display = f"{conf:.1f}" if conf is not None else "—"
                    bar_width = f"{conf:.0f}" if conf else "0"

                    st.markdown(
                        f"""
                        <div class="pred-card {cls}">
                            <div class="model-name">{m_name}</div>
                            <div class="pred-emoji">{pred_emoji}</div>
                            <div class="pred-activity">{pred_name.replace("_"," ").title()} {badge}</div>
                            <div style="color:#9e9e9e; font-size:0.82rem; margin-top:4px;">
                                Confidence: {conf_display}%
                            </div>
                            <div class="conf-bar-bg">
                                <div class="conf-bar {cls}" style="width:{bar_width}%;"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        # Raw data expander
        with st.expander("📋 Raw Feature Data"):
            st.dataframe(sample_features, use_container_width=True, height=220)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 3 — MODEL COMPARISON                                                ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab_compare:
    import plotly.graph_objects as go
    import plotly.express as px

    st.markdown('<div class="hero-title gradient-text" style="font-size:1.8rem;">📊 Model Comparison</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="font-size:0.92rem;">Side-by-side analysis of all trained models — accuracy, speed, and confusion matrices.</p>', unsafe_allow_html=True)

    # ── Accuracy chart ───────────────────────────────────────────────────────
    st.markdown("### Accuracy Comparison")

    if acc_data:
        sorted_items = sorted(acc_data.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_items]
        accs  = [x[1] for x in sorted_items]

        gradient_colors = ["#667eea", "#764ba2", "#42a5f5", "#00d4aa", "#ffa726", "#ef5350"]
        bar_colors = [gradient_colors[i % len(gradient_colors)] for i in range(len(names))]

        fig = go.Figure(go.Bar(
            x=names,
            y=accs,
            marker=dict(color=bar_colors, cornerradius=8),
            text=[f"{a:.2f}%" for a in accs],
            textposition="outside",
            textfont=dict(size=13, color="#e0e0e0"),
        ))
        fig.update_layout(
            **PLOTLY_LAYOUT,
            height=420,
            yaxis=dict(range=[0, 105], showgrid=True, gridcolor="rgba(255,255,255,0.04)", title="Accuracy (%)"),
            xaxis=dict(showgrid=False, title=""),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No accuracy data available yet. Run evaluation first.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Evaluation results table & inference speed ───────────────────────────
    if eval_results:
        st.markdown("### Detailed Metrics")

        metric_rows = []
        for m_name, info in eval_results.items():
            row = {"Model": m_name}
            for key in ["accuracy", "precision", "recall", "f1", "inference_speed"]:
                val = info.get(key)
                if val is not None:
                    if key == "inference_speed":
                        row["Inference (ms/sample)"] = f"{val * 1000:.2f}" if isinstance(val, (int, float)) else val
                    else:
                        row[key.capitalize()] = f"{val * 100:.2f}%" if isinstance(val, (int, float)) and val <= 1 else val
            metric_rows.append(row)

        if metric_rows:
            st.dataframe(pd.DataFrame(metric_rows).set_index("Model"), use_container_width=True)

        # Inference speed chart
        speed_data = {}
        for m_name, info in eval_results.items():
            sp = info.get("inference_speed")
            if sp is not None:
                speed_data[m_name] = sp * 1000  # → ms

        if speed_data:
            st.markdown("### Inference Speed")
            sd = sorted(speed_data.items(), key=lambda x: x[1])
            fig_sp = go.Figure(go.Bar(
                x=[x[0] for x in sd],
                y=[x[1] for x in sd],
                marker=dict(color="#42a5f5", cornerradius=8),
                text=[f"{x[1]:.2f} ms" for x in sd],
                textposition="outside",
                textfont=dict(size=12, color="#e0e0e0"),
            ))
            fig_sp.update_layout(
                **PLOTLY_LAYOUT,
                height=380,
                yaxis=dict(title="ms / sample", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
                xaxis=dict(title=""),
            )
            st.plotly_chart(fig_sp, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Training metadata ────────────────────────────────────────────────────
    if training_meta:
        st.markdown("### Training Time")
        train_rows = []
        for m_name, info in training_meta.items():
            t = info.get("training_time") or info.get("time")
            a = info.get("accuracy")
            if t is not None:
                train_rows.append({"Model": m_name, "Time (s)": round(t, 2), "Train Acc": f"{a*100:.1f}%" if a else "—"})
        if train_rows:
            df_train = pd.DataFrame(train_rows)
            fig_t = go.Figure(go.Bar(
                x=df_train["Model"],
                y=df_train["Time (s)"],
                marker=dict(color="#ffa726", cornerradius=8),
                text=[f"{t:.1f}s" for t in df_train["Time (s)"]],
                textposition="outside",
                textfont=dict(size=12, color="#e0e0e0"),
            ))
            fig_t.update_layout(**PLOTLY_LAYOUT, height=380,
                                yaxis=dict(title="Seconds", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
                                xaxis=dict(title=""))
            st.plotly_chart(fig_t, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Confusion matrices ───────────────────────────────────────────────────
    st.markdown("### Confusion Matrices")

    cm_files = sorted(MODELS_DIR.glob("*_cm.png"))
    if cm_files:
        cm_cols = st.columns(min(len(cm_files), 3))
        for i, cm_path in enumerate(cm_files):
            with cm_cols[i % 3]:
                nice_name = cm_path.stem.replace("_cm", "").replace("_", " ").title()
                st.markdown(
                    f"<div style='text-align:center; font-weight:600; color:#b0b8c8; margin-bottom:6px;'>{nice_name}</div>",
                    unsafe_allow_html=True,
                )
                st.image(str(cm_path), use_container_width=True)
    else:
        st.info("No confusion matrix images found in the models folder.")


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 4 — ANALYTICS                                                       ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab_analytics:
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<div class="hero-title gradient-text" style="font-size:1.8rem;">📈 Data Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="font-size:0.92rem;">Deep dive into the test dataset — activity distributions, feature insights, and per‑class performance.</p>', unsafe_allow_html=True)

    # ── Activity distribution ────────────────────────────────────────────────
    st.markdown("### Activity Distribution in Test Set")

    label_counts = y_test.value_counts().sort_index()
    act_names = [label_name(int(l), activity_labels) for l in label_counts.index]
    act_emojis = [ACTIVITY_EMOJI.get(n, "") + " " + n.replace("_", " ").title() for n in act_names]
    act_colors_list = [ACTIVITY_COLORS.get(n, "#667eea") for n in act_names]

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie = go.Figure(go.Pie(
            labels=act_emojis,
            values=label_counts.values,
            hole=0.45,
            marker=dict(colors=act_colors_list, line=dict(color="#0e1117", width=2)),
            textinfo="percent+label",
            textfont=dict(size=11),
        ))
        fig_pie.update_layout(**PLOTLY_LAYOUT, height=420, showlegend=False)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_bar:
        fig_bar = go.Figure(go.Bar(
            x=act_emojis,
            y=label_counts.values,
            marker=dict(color=act_colors_list, cornerradius=8),
            text=label_counts.values,
            textposition="outside",
            textfont=dict(size=12, color="#e0e0e0"),
        ))
        fig_bar.update_layout(
            **PLOTLY_LAYOUT, height=420,
            yaxis=dict(title="Count", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
            xaxis=dict(title="", tickangle=-25),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Per-activity accuracy breakdown ──────────────────────────────────────
    st.markdown("### Per-Activity Accuracy (Best Model)")

    # Pick the first sklearn model for quick per-activity calc (or best)
    ref_model_name = best_model_name if best_model_name and best_model_name in models else list(models.keys())[0]
    ref_model = models[ref_model_name]

    try:
        if ref_model_name == "Neural Network":
            preds_all = np.argmax(ref_model.predict(X_test.values, verbose=0), axis=1)
        else:
            preds_all = ref_model.predict(X_test)

        per_act_acc = {}
        for lbl in sorted(y_test.unique()):
            mask = y_test.values == lbl
            if mask.sum() > 0:
                name = label_name(int(lbl), activity_labels)
                per_act_acc[name] = float((preds_all[mask] == y_test.values[mask]).mean()) * 100

        if per_act_acc:
            pa_names = list(per_act_acc.keys())
            pa_accs  = list(per_act_acc.values())
            pa_colors = [ACTIVITY_COLORS.get(n, "#667eea") for n in pa_names]
            pa_labels = [ACTIVITY_EMOJI.get(n, "") + " " + n.replace("_", " ").title() for n in pa_names]

            fig_pa = go.Figure(go.Bar(
                x=pa_labels,
                y=pa_accs,
                marker=dict(color=pa_colors, cornerradius=8),
                text=[f"{a:.1f}%" for a in pa_accs],
                textposition="outside",
                textfont=dict(size=12, color="#e0e0e0"),
            ))
            fig_pa.update_layout(
                **PLOTLY_LAYOUT, height=420,
                yaxis=dict(range=[0, 105], title="Accuracy (%)", showgrid=True, gridcolor="rgba(255,255,255,0.04)"),
                xaxis=dict(title="", tickangle=-25),
                title=dict(text=f"Model: {ref_model_name}", font=dict(size=13, color="#9e9e9e")),
            )
            st.plotly_chart(fig_pa, use_container_width=True)
    except Exception:
        st.info("Could not compute per-activity accuracy.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Feature correlation heatmap (top 20) ─────────────────────────────────
    st.markdown("### Feature Correlation Heatmap (Top 20 Features)")

    try:
        # Pick top 20 features by variance
        variances = X_test.var().sort_values(ascending=False)
        top_features = variances.head(20).index.tolist()
        corr = X_test[top_features].corr()

        # Shorten names for display
        short = [f[:22] + "…" if len(f) > 24 else f for f in top_features]

        fig_hm = go.Figure(go.Heatmap(
            z=corr.values,
            x=short,
            y=short,
            colorscale=[[0, "#0e1117"], [0.5, "#667eea"], [1, "#ef5350"]],
            zmin=-1, zmax=1,
            colorbar=dict(title="r", tickfont=dict(color="#9e9e9e")),
        ))
        fig_hm.update_layout(
            **PLOTLY_LAYOUT,
            height=560,
            xaxis=dict(tickangle=-45, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_hm, use_container_width=True)
    except Exception:
        st.info("Could not generate correlation heatmap.")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Feature statistics ───────────────────────────────────────────────────
    st.markdown("### Feature Statistics Summary")
    with st.expander("🔢 Show Full Statistics Table"):
        st.dataframe(X_test.describe().T.style.format("{:.4f}"), use_container_width=True, height=400)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB — HEALTH METRICS                                                    ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab_health:
    import plotly.express as px
    import plotly.graph_objects as go
    from src.health_metrics import HealthMetrics

    st.markdown('<div class="hero-title gradient-text" style="font-size:1.8rem;">❤️ Health & Fitness Analytics</div>', unsafe_allow_html=True)
    st.markdown('<p class="hero-sub" style="font-size:0.92rem;">Real-time step counting, MET-based calorie burn estimation, active time vs. sedentary time tracking.</p>', unsafe_allow_html=True)

    user_weight = st.number_input("🏋️ Enter Body Weight (kg)", min_value=30.0, max_value=200.0, value=70.0, step=1.0)

    # Compute health metrics from test dataset or session history
    hm = HealthMetrics(weight_kg=user_weight, prediction_interval_s=2.56)

    # Simulate predictions over sample dataset for realistic health snapshot
    for idx in range(min(500, len(y_test))):
        act_str = label_name(int(y_test.iloc[idx]), activity_labels)
        hm.update(act_str)

    snap = hm.get_snapshot()

    hc1, hc2, hc3, hc4, hc5 = st.columns(5)
    with hc1:
        st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-label">Steps Counted</div><div class="kpi-value">👟 {snap.total_steps:,}</div></div>', unsafe_allow_html=True)
    with hc2:
        st.markdown(f'<div class="kpi-card kpi-teal"><div class="kpi-label">Calories Burned</div><div class="kpi-value">🔥 {snap.calories_burned} kcal</div></div>', unsafe_allow_html=True)
    with hc3:
        st.markdown(f'<div class="kpi-card kpi-amber"><div class="kpi-label">Distance</div><div class="kpi-value">📏 {snap.distance_km} km</div></div>', unsafe_allow_html=True)
    with hc4:
        st.markdown(f'<div class="kpi-card kpi-blue"><div class="kpi-label">Active Time</div><div class="kpi-value">⏱️ {snap.active_time_min} min</div></div>', unsafe_allow_html=True)
    with hc5:
        st.markdown(f'<div class="kpi-card kpi-purple"><div class="kpi-label">Sedentary Time</div><div class="kpi-value">🪑 {snap.sedentary_time_min} min</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    c_pie1, c_pie2 = st.columns(2)
    with c_pie1:
        st.markdown("### Active vs. Sedentary Breakdown")
        fig_as = go.Figure(go.Pie(
            labels=["Active Time", "Sedentary Time"],
            values=[snap.active_time_min, snap.sedentary_time_min],
            hole=0.5,
            marker=dict(colors=["#00d4aa", "#ffa726"]),
        ))
        fig_as.update_layout(**PLOTLY_LAYOUT, height=350)
        st.plotly_chart(fig_as, use_container_width=True)

    with c_pie2:
        st.markdown("### Estimated Calories Burned per Activity")
        act_cals = snap.activity_calories
        if act_cals:
            fig_cal = go.Figure(go.Bar(
                x=[ACTIVITY_EMOJI.get(k, '') + ' ' + k.replace('_', ' ').title() for k in act_cals.keys()],
                y=list(act_cals.values()),
                marker=dict(color="#ef5350", cornerradius=8),
                text=[f"{v:.1f} kcal" for v in act_cals.values()],
                textposition="outside",
            ))
            fig_cal.update_layout(**PLOTLY_LAYOUT, height=350, yaxis=dict(title="kcal"))
            st.plotly_chart(fig_cal, use_container_width=True)


# ╔════════════════════════════════════════════════════════════════════════════╗
# ║  TAB 5 — ABOUT                                                           ║
# ╚════════════════════════════════════════════════════════════════════════════╝

with tab_about:

    st.markdown('<div class="hero-title gradient-text" style="font-size:1.8rem;">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#e0e0e0; margin-top:0;">Project Overview</h4>
            <p style="color:#b0b8c8; line-height:1.75;">
                This <strong>Human Activity Recognition (HAR)</strong> system classifies six daily activities
                — Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, and Laying —
                using tri-axial accelerometer and gyroscope signals captured from a Samsung Galaxy S II
                smartphone worn on the waist.
            </p>
            <p style="color:#b0b8c8; line-height:1.75;">
                The dataset originates from the
                <a href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
                   target="_blank" style="color:#667eea; text-decoration:none; font-weight:600;">
                   UCI Machine Learning Repository</a>.
                Each sample contains <strong>561 features</strong> derived from time‑domain and frequency‑domain
                signal processing applied to raw sensor readings sampled at 50 Hz.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#e0e0e0; margin-top:0;">🏗️ Architecture</h4>
            <p style="color:#b0b8c8; line-height:1.75;">
                <strong>Data Pipeline</strong>&ensp;→&ensp;Download &amp; extract UCI HAR zip
                → Parse fixed‑width sensor files → De-duplicate feature names → Train / Test split (provided by UCI)
            </p>
            <p style="color:#b0b8c8; line-height:1.75;">
                <strong>Training</strong>&ensp;→&ensp;scikit‑learn classifiers (Logistic Regression, Decision Tree,
                SVM, Random Forest, Gradient Boosting) + Keras Sequential neural network → Serialised to
                <code>.pkl</code> / <code>.keras</code>
            </p>
            <p style="color:#b0b8c8; line-height:1.75;">
                <strong>Evaluation</strong>&ensp;→&ensp;Accuracy, Precision, Recall, F1, Confusion Matrices,
                Inference Speed benchmarking → persisted as JSON + PNG
            </p>
            <p style="color:#b0b8c8; line-height:1.75;">
                <strong>Dashboard</strong>&ensp;→&ensp;Streamlit multi‑tab interface with Plotly interactive charts,
                real‑time predictions, glassmorphism UI
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tech stack badges
    st.markdown("#### Tech Stack")
    badges = [
        "Python 3.10+", "Streamlit", "scikit‑learn", "TensorFlow / Keras",
        "Plotly", "Pandas", "NumPy", "Joblib", "Seaborn", "Matplotlib",
    ]
    badge_html = " ".join(f'<span class="badge">{b}</span>' for b in badges)
    st.markdown(f"<div style='margin-bottom:24px;'>{badge_html}</div>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="glass-card">
            <h4 style="color:#e0e0e0; margin-top:0;">📂 Project Structure</h4>
            <pre style="color:#9e9e9e; font-size:0.82rem; line-height:1.6; margin:0;">
HAR/
├── app.py                  ← This dashboard
├── src/
│   ├── data.py             ← Data download & preparation
│   ├── train.py            ← Model training pipeline
│   └── evaluate.py         ← Evaluation & metric generation
├── models/
│   ├── *.pkl / *.keras     ← Serialised models
│   ├── *_cm.png            ← Confusion matrix plots
│   ├── activity_labels.csv
│   ├── training_metadata.json
│   └── evaluation_results.json
├── data/
│   └── UCI_HAR_Dataset/    ← Raw dataset
├── requirements.txt
└── README.md
            </pre>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="glass-card" style="text-align:center;">
            <h4 style="color:#e0e0e0; margin-top:0;">Credits &amp; License</h4>
            <p style="color:#9e9e9e;">
                Dataset:
                <a href="https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones"
                   target="_blank" style="color:#667eea; text-decoration:none;">
                   UCI HAR Dataset</a> — Anguita et al. (2013)
            </p>
            <p style="color:#666; font-size:0.78rem; margin-top:16px;">
                Built with ❤️ using Streamlit · Plotly · scikit‑learn · TensorFlow
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

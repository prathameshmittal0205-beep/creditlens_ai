"""
CreditLens AI — app.py  (Main Entry Point)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Hero · Model loading sequence · KPI strip · Model registry · Platform notes
"""

import streamlit as st
import pandas as pd
import time

from utils.ui_components import (
    set_page_config,
    render_kpi,
    render_status_pill,
    render_section,
    COLOR_EMERALD,
    COLOR_AMBER,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_INDIGO,
)
from utils.data_gen import generate_synthetic_data
from utils.preprocessing import (
    engineer_features,
    preprocess_for_segmentation,
    preprocess_for_classification,
    prepare_lifetimes_data,
)
from models.segmentation import train_segmentation_model, apply_pca, assign_segment_labels
from models.clv import train_clv_models, predict_clv_metrics
from models.credit_scoring import train_credit_model, get_feature_importance
from models.explainability import generate_shap_values

# ═══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG + GLOBAL CSS
# ═══════════════════════════════════════════════════════════════════════════════
set_page_config("CreditLens AI")

# ── Additional home-page-only overrides ────────────────────────────────────────
st.markdown(
    """
    <style>
      /* ── Hero title ── */
      .cl-hero-title {
        font-family: 'Fraunces', serif;
        font-size: clamp(2rem, 5vw, 3.2rem);
        font-weight: 300;
        letter-spacing: -1.5px;
        line-height: 1.1;
        color: #e8eaf0;
        margin: 0 0 0.25rem;
      }
      .cl-hero-title span { color: #10b981; }

      /* ── Hero subtitle ── */
      .cl-hero-sub {
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 1.25rem;
      }

      /* ── Hero body ── */
      .cl-hero-body {
        font-family: 'DM Mono', monospace;
        font-size: 13px;
        color: #9ca3af;
        line-height: 1.8;
        max-width: 680px;
        border-left: 2px solid #1e2430;
        padding-left: 1rem;
        margin-bottom: 2rem;
      }

      /* ── Progress bar ── */
      .stProgress > div > div > div {
        background: linear-gradient(90deg, #10b981, #3b82f6) !important;
        border-radius: 2px !important;
      }
      .stProgress > div > div {
        background: #1e2430 !important;
        border-radius: 2px !important;
      }

      /* ── Loading label ── */
      .cl-load-label {
        font-family: 'DM Mono', monospace;
        font-size: 11px;
        color: #6b7280;
        letter-spacing: 0.08em;
        margin: 0 0 6px;
      }
      .cl-load-label span { color: #10b981; }

      /* ── Model registry cards ── */
      .model-card {
        background: #161b22;
        border: 1px solid #1e2430;
        border-radius: 6px;
        padding: 1rem 1.1rem;
        height: 100%;
      }
      .model-card-cat {
        font-size: 9px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 4px;
      }
      .model-card-name {
        font-family: 'Fraunces', serif;
        font-size: 16px;
        font-weight: 300;
        color: #e8eaf0;
        margin-bottom: 2px;
      }
      .model-card-detail {
        font-size: 10px;
        color: #6b7280;
        margin-bottom: 10px;
        font-family: 'DM Mono', monospace;
      }

      /* ── About info box ── */
      .cl-about {
        background: #111318;
        border: 1px solid #1e2430;
        border-radius: 6px;
        padding: 1.25rem 1.5rem;
        font-family: 'DM Mono', monospace;
        font-size: 12px;
        color: #9ca3af;
        line-height: 1.9;
      }
      .cl-about strong { color: #e8eaf0; font-weight: 500; }
      .cl-about .cl-bullet { color: #10b981; margin-right: 6px; }

      /* ── Success box override ── */
      [data-testid="stAlertContainer"] [data-baseweb="notification"] {
        background: rgba(16,185,129,0.08) !important;
        border: 1px solid rgba(16,185,129,0.25) !important;
        border-radius: 4px !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  HERO
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
    <div class="cl-hero-sub">Behavioral Credit Intelligence Platform</div>
    <div class="cl-hero-title">Credit<span>Lens</span> AI</div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="cl-hero-body">
      Evaluates creditworthiness through transaction behaviour — enabling fair,
      explainable decisions for gig workers, freelancers, and informal earners
      who lack traditional credit histories. Powered by KMeans segmentation,
      BG/NBD lifetime value modelling, and XGBoost credit scoring with SHAP
      explainability built in at every layer.
    </div>
    """,
    unsafe_allow_html=True,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data
def load_data():
    return generate_synthetic_data(num_customers=500, months=12)


@st.cache_resource
def build_models(df):
    features = engineer_features(df)

    # ── Segmentation ──────────────────────────────────────────────────────
    scaled_data, scaler, num_cols = preprocess_for_segmentation(features)
    kmeans, clusters, silhouette  = train_segmentation_model(scaled_data, num_clusters=3)
    pca, pca_result               = apply_pca(scaled_data)
    features_segmented            = assign_segment_labels(features, clusters)

    # ── CLV ───────────────────────────────────────────────────────────────
    summary_df        = prepare_lifetimes_data(df)
    bgf, ggf          = train_clv_models(summary_df)
    clv_metrics       = predict_clv_metrics(bgf, ggf, summary_df)

    # ── Credit Scoring ────────────────────────────────────────────────────
    X, y, feature_names                            = preprocess_for_classification(features_segmented)
    xgb_model, metrics, X_train, X_test, y_train, y_test = train_credit_model(X, y)

    # ── Explainability ────────────────────────────────────────────────────
    explainer, shap_values = generate_shap_values(xgb_model, X)

    return {
        "raw_data":               df,
        "features":               features_segmented,
        "pca_result":             pca_result,
        "segmentation_model":     kmeans,
        "silhouette_score":       silhouette,
        "clv_metrics":            clv_metrics,
        "xgb_model":              xgb_model,
        "classification_metrics": metrics,
        "X_ml":                   X,
        "y_ml":                   y,
        "ml_feature_names":       feature_names,
        "shap_explainer":         explainer,
        "shap_values":            shap_values,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  LOADING SEQUENCE
# ═══════════════════════════════════════════════════════════════════════════════
LOAD_STEPS = [
    ("Generating synthetic customer profiles",    12),
    ("Engineering behavioural features",          28),
    ("Training KMeans segmentation + PCA",        45),
    ("Fitting BG/NBD & Gamma-Gamma CLV models",   62),
    ("Training XGBoost credit scorer (SMOTE)",    80),
    ("Generating SHAP explainability values",     93),
    ("Finalising model registry",                100),
]

if "raw_data" not in st.session_state:
    bar_slot   = st.empty()
    label_slot = st.empty()

    with st.spinner(""):
        for label, pct in LOAD_STEPS:
            label_slot.markdown(
                f'<p class="cl-load-label"><span>▸</span> {label}…</p>',
                unsafe_allow_html=True,
            )
            bar_slot.progress(pct / 100)
            time.sleep(0.04)

        df           = load_data()
        system_state = build_models(df)

        bar_slot.empty()
        label_slot.empty()

    for k, v in system_state.items():
        st.session_state[k] = v

    st.success("✦  All models initialised — navigate using the sidebar.")

else:
    st.success("✦  Platform ready — models loaded from cache.")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SYSTEM KPI STRIP
# ═══════════════════════════════════════════════════════════════════════════════
render_section("System Overview")

c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    n = len(st.session_state.get("features", [])) if "features" in st.session_state else 500
    render_kpi("Customers", f"{n:,}", delta="Synthetic profiles", delta_positive=True)

with c2:
    render_kpi("Time Horizon", "12 mo", delta="Transaction history", delta_positive=True)

with c3:
    render_kpi("Active Models", "5", delta="KMeans · BG/NBD · XGB", delta_positive=True)

with c4:
    roc = st.session_state.get("classification_metrics", {}).get("roc_auc", 0)
    render_kpi(
        "ROC-AUC",
        f"{roc:.4f}" if roc else "—",
        delta="XGBoost scorer",
        delta_positive=True,
    )

with c5:
    render_kpi("Fairness Audit", "ON", delta="SHAP + segment parity", delta_positive=True)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Model Registry")

registry = [
    {
        "category": "Segmentation",
        "name":     "KMeans",
        "detail":   "k=3 clusters · PCA 3D",
        "status":   "active",
        "color":    COLOR_EMERALD,
        "icon":     "atom",
    },
    {
        "category": "Lifetime Value",
        "name":     "BG / NBD",
        "detail":   "Pareto/NBD variant",
        "status":   "active",
        "color":    COLOR_BLUE,
        "icon":     "trending-up",
    },
    {
        "category": "Spend Model",
        "name":     "Gamma-Gamma",
        "detail":   "Monetary prediction",
        "status":   "active",
        "color":    COLOR_BLUE,
        "icon":     "currency-dollar",
    },
    {
        "category": "Credit Score",
        "name":     "XGBoost",
        "detail":   "Binary classifier · SMOTE",
        "status":   "active",
        "color":    COLOR_INDIGO,
        "icon":     "cpu",
    },
    {
        "category": "Explainability",
        "name":     "SHAP",
        "detail":   "TreeExplainer · Waterfall",
        "status":   "warning",
        "color":    COLOR_AMBER,
        "icon":     "eye",
    },
]

cols = st.columns(5)
for col, card in zip(cols, registry):
    with col:
        pill  = render_status_pill(
            label="LIVE" if card["status"] == "active" else "READY",
            status=card["status"],
        )
        st.markdown(
            f"""
            <div class="model-card">
              <div class="model-card-cat">{card["category"]}</div>
              <div class="model-card-name">{card["name"]}</div>
              <div class="model-card-detail">{card["detail"]}</div>
              {pill}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE METRICS SNAPSHOT (only when models are ready)
# ═══════════════════════════════════════════════════════════════════════════════
if "features" in st.session_state:
    render_section("Live Portfolio Snapshot")

    features    = st.session_state["features"]
    clv_metrics = st.session_state["clv_metrics"]
    metrics     = st.session_state["classification_metrics"]

    m1, m2, m3, m4, m5, m6 = st.columns(6)

    with m1:
        render_kpi(
            "Avg Savings Ratio",
            f"{features['savings_ratio'].mean() * 100:.1f}%",
        )
    with m2:
        render_kpi(
            "Default Rate",
            f"{features['loan_default'].mean() * 100:.1f}%",
            delta="portfolio-wide",
            delta_positive=False,
        )
    with m3:
        render_kpi(
            "Avg CLV",
            f"${clv_metrics['clv_score'].mean():.2f}",
        )
    with m4:
        render_kpi(
            "Avg Retention",
            f"{clv_metrics['retention_probability'].mean():.1%}",
            delta="6-month horizon",
            delta_positive=True,
        )
    with m5:
        render_kpi(
            "ROC-AUC",
            f"{metrics['roc_auc']:.3f}",
            delta="credit model",
            delta_positive=True,
        )
    with m6:
        silhouette = st.session_state.get("silhouette_score", 0)
        render_kpi(
            "Silhouette Score",
            f"{silhouette:.3f}" if silhouette else "—",
            delta="cluster quality",
            delta_positive=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  ABOUT / PLATFORM NOTES
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Platform Notes")

about_items = [
    ("Data Source",      "500 synthetic customers · 12-month transaction histories"),
    ("Segmentation",     "KMeans (k=3) with PCA dimensionality reduction for 3D cluster visualisation"),
    ("Lifetime Value",   "BG/NBD for purchase frequency + Gamma-Gamma for monetary value (6-month horizon)"),
    ("Credit Scoring",   "XGBoost binary classifier trained on engineered behavioural features, balanced with SMOTE"),
    ("Explainability",   "SHAP TreeExplainer — every decision is interpretable via waterfall + global summary plots"),
    ("Fairness",         "Approval-rate and default-rate parity checks across gender, employment type, and age group"),
]

rows_html = "".join(
    f'<div style="padding:6px 0;border-bottom:1px solid #1e2430;">'
    f'<span class="cl-bullet">▸</span>'
    f'<strong>{key}</strong> — {val}'
    f'</div>'
    for key, val in about_items
)

st.markdown(
    f'<div class="cl-about">{rows_html}</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar footer ─────────────────────────────────────────────────────────────
st.sidebar.markdown(
    """
    <div style="
      position: fixed; bottom: 1.5rem; left: 0; width: 200px;
      padding: 0 1.25rem;
      font-family: 'DM Mono', monospace;
      font-size: 10px;
      color: #4b5563;
      line-height: 1.7;
    ">
      <div style="border-top:1px solid #1e2430;padding-top:10px;">
        CreditLens AI v2.4.1<br>
        Model trained 2d ago<br>
        500 customers · 12 months
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
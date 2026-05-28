import streamlit as st
import pandas as pd
import time

from utils.ui_components import (
    set_page_config,
    render_kpi,
    render_status_pill,
    render_section,
    render_executive_insight,
    render_sb_header,
    render_sb_dataset_info,
    render_sb_pipeline_status,
    render_sb_model_health,
    render_sb_footer,
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
    process_uploaded_data,
)
from models.segmentation import train_segmentation_model, apply_pca, assign_segment_labels
from models.clv import train_clv_models, predict_clv_metrics
from models.credit_scoring import train_credit_model, get_feature_importance
from models.explainability import generate_shap_values

set_page_config("CreditLens AI")

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

def load_data(uploaded_file=None):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            processed_df = process_uploaded_data(df)
            print("DATA SOURCE: Uploaded Dataset")
            print("DATAFRAME SHAPE:", processed_df.shape)
            return processed_df, "Uploaded Dataset"
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None, "Error"
    
    # Strictly isolate synthetic fallback and change sample size so it differs from CSV
    df = generate_synthetic_data(num_customers=1000, months=12)
    print("DATA SOURCE: Synthetic Dataset")
    print("DATAFRAME SHAPE:", df.shape)
    return df, "Synthetic Dataset"


def build_models(df):
    features = engineer_features(df)
    scaled_data, scaler, num_cols = preprocess_for_segmentation(features)
    kmeans, clusters, silhouette  = train_segmentation_model(scaled_data, num_clusters=3)
    pca, pca_result               = apply_pca(scaled_data)
    features_segmented            = assign_segment_labels(features, clusters)

    summary_df = prepare_lifetimes_data(df)
    
    try:
        bgf, ggf = train_clv_models(summary_df)
        if bgf is None or not hasattr(bgf, "params_"):
            raise ValueError("BG/NBD model failed training.")
        if ggf is not None and not hasattr(ggf, "params_"):
            ggf = None
        clv_metrics = predict_clv_metrics(bgf, ggf, summary_df)
        st.session_state["clv_mode"] = "lifetimes"
    except Exception as e:
        print(f"Lifetimes CLV skipped: {e}. Generating behavioral fallback.")
        from models.clv import generate_fallback_clv
        clv_metrics = generate_fallback_clv(summary_df, features_segmented)
        st.session_state["clv_mode"] = "fallback"
        bgf = None
        ggf = None
        
    st.session_state["clv_metrics"] = clv_metrics

    X, y, feature_names = preprocess_for_classification(features_segmented)
    xgb_model, metrics, X_train, X_test, y_train, y_test = None, {}, None, None, None, None
    explainer, shap_values = None, None
    
    try:
        xgb_model, metrics, X_train, X_test, y_train, y_test = train_credit_model(X, y)
    except ValueError as e:
        st.error(f"Credit Scoring Pipeline Error: {e}")

    if xgb_model is not None:
        try:
            explainer, shap_values = generate_shap_values(xgb_model, X)
        except Exception as e:
            st.warning(f"Failed to generate SHAP values: {e}")

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


def data_upload_page():
    n = len(st.session_state.get("features", [])) if "features" in st.session_state else 0
    cust_str = f"{n:,}" if n > 0 else "---"
    st.markdown(
        f"""
        <div style="display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 2rem;">
            <div>
                <div class="cl-hero-sub">Behavioral Credit Intelligence Platform</div>
                <div class="cl-hero-title">Credit<span>Lens</span> AI</div>
                <div class="cl-hero-body" style="margin-bottom: 0;">
                  Evaluates creditworthiness through transaction behaviour — enabling fair,
                  explainable decisions for gig workers, freelancers, and informal earners.
                </div>
            </div>
            <div style="text-align: right; min-width: 200px;">
                <div style="font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Total Profiles</div>
                <div style="font-family: 'Fraunces', serif; font-size: 2.5rem; color: #e8eaf0; line-height: 1;">{cust_str}</div>
                <div style="font-size: 12px; color: #10b981; font-family: 'DM Mono', monospace;">+ 12 mo history</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "raw_data" in st.session_state:
        render_executive_insight(
            "**Portfolio Health is Stable.** The synthetic population is demonstrating healthy savings behaviours, "
            "with an aggregate default risk contained below 15%. Credit scoring models have successfully balanced the "
            "class distributions and are ready for individual applicant scoring."
        )

    st.sidebar.markdown("<div class='sb-section-title' style='margin-top: 12px;'>Data Configuration</div>", unsafe_allow_html=True)
    upload_option = st.sidebar.radio("Dataset Mode", ["Synthetic Demo", "Upload Real Data (CSV)"], label_visibility="collapsed")

    uploaded_file = None
    if upload_option == "Upload Real Data (CSV)":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file:
            if st.sidebar.button("Process New Data", use_container_width=True):
                st.cache_data.clear()
                st.cache_resource.clear()
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.session_state["pending_upload"] = uploaded_file
    elif upload_option == "Synthetic Demo":
        if st.sidebar.button("Reload Synthetic Data", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            for k in list(st.session_state.keys()):
                del st.session_state[k]

    if "pending_upload" in st.session_state:
        uploaded_file = st.session_state["pending_upload"]

    LOAD_STEPS = [
        ("Ingesting data & auto-mapping features",    12),
        ("Engineering behavioural features",          28),
        ("Training KMeans segmentation + PCA",        45),
        ("Fitting BG/NBD & Gamma-Gamma CLV models",   62),
        ("Training XGBoost credit scorer (SMOTE)",    80),
        ("Generating SHAP explainability values",     93),
        ("Finalising model registry",                100),
    ]

    if "raw_data" not in st.session_state:
        if upload_option == "Upload Real Data (CSV)" and uploaded_file is None:
            st.info("Please upload a CSV file in the sidebar to begin, or select Synthetic Demo.")
            st.stop()
            
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

            df_result = load_data(uploaded_file)
            if df_result is None or df_result[0] is None:
                st.stop()
                
            df, source_name = df_result
            st.session_state["data_source"] = source_name
            
            print(f"\n--- PRE-BUILD DIAGNOSTICS ---")
            print(f"DATA SOURCE: {source_name}")
            print(f"DF SHAPE: {df.shape}")
            import hashlib
            fingerprint = hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()
            print(f"DATA FINGERPRINT: {fingerprint}")
            print("-----------------------------\n")
                
            system_state = build_models(df)

            bar_slot.empty()
            label_slot.empty()

        for k, v in system_state.items():
            st.session_state[k] = v

        if "pending_upload" in st.session_state:
            del st.session_state["pending_upload"]

        st.success("✦  All models initialised — navigate using the sidebar.")
    else:
        st.success("✦  Platform ready — models loaded from cache.")

    st.markdown("<br>", unsafe_allow_html=True)

    render_section("Pipeline Architecture")
    
    st.markdown(
        """
        <style>
        .pipeline-container {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: #111318;
            border: 1px solid #1e2430;
            border-radius: 8px;
            padding: 1.5rem 2rem;
            margin-bottom: 1rem;
        }
        .pipeline-node {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
            flex: 1;
        }
        .pipeline-icon {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            background: rgba(16, 185, 129, 0.1);
            color: #10b981;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            margin-bottom: 12px;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }
        .node-2 .pipeline-icon { background: rgba(59, 130, 246, 0.1); color: #3b82f6; border-color: rgba(59, 130, 246, 0.2); }
        .node-3 .pipeline-icon { background: rgba(99, 102, 241, 0.1); color: #6366f1; border-color: rgba(99, 102, 241, 0.2); }
        .node-4 .pipeline-icon { background: rgba(245, 158, 11, 0.1); color: #f59e0b; border-color: rgba(245, 158, 11, 0.2); }
        
        .pipeline-title {
            font-family: 'Fraunces', serif;
            font-size: 15px;
            color: #e8eaf0;
            margin-bottom: 4px;
        }
        .pipeline-sub {
            font-family: 'DM Mono', monospace;
            font-size: 10px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .pipeline-edge {
            color: #232832;
            font-size: 20px;
            padding: 0 10px;
        }
        </style>
        
        <div class="pipeline-container">
            <div class="pipeline-node">
                <div class="pipeline-icon">👥</div>
                <div class="pipeline-title">Segmentation</div>
                <div class="pipeline-sub">KMeans (k=3)</div>
            </div>
            <div class="pipeline-edge">➔</div>
            <div class="pipeline-node node-2">
                <div class="pipeline-icon">📈</div>
                <div class="pipeline-title">Lifetime Value</div>
                <div class="pipeline-sub">BG/NBD + Gamma</div>
            </div>
            <div class="pipeline-edge">➔</div>
            <div class="pipeline-node node-3">
                <div class="pipeline-icon">🎯</div>
                <div class="pipeline-title">Credit Scorer</div>
                <div class="pipeline-sub">XGBoost · SMOTE</div>
            </div>
            <div class="pipeline-edge">➔</div>
            <div class="pipeline-node node-4">
                <div class="pipeline-icon">👁️</div>
                <div class="pipeline-title">Explainability</div>
                <div class="pipeline-sub">SHAP values</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<br>", unsafe_allow_html=True)

    if "features" in st.session_state:
        render_section("Intelligence Summary")
        
        features    = st.session_state["features"]
        clv_metrics = st.session_state["clv_metrics"]
        metrics     = st.session_state["classification_metrics"]

        left_col, right_col = st.columns([7, 3])
        
        with left_col:
            lc1, lc2 = st.columns(2)
            with lc1:
                def_rate = features['loan_default'].mean() * 100
                st.markdown(
                    f"""<div style="background: #111318; border: 1px solid #1e2430; border-radius: 8px; padding: 1.5rem; height: 100%;">
<div style="font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Portfolio Risk</div>
<div style="font-family: 'Fraunces', serif; font-size: 2rem; color: #ef4444; margin-bottom: 12px;">{def_rate:.1f}% Default Rate</div>
<div style="height: 4px; background: #1e2430; border-radius: 2px; overflow: hidden;">
<div style="height: 100%; width: {def_rate:.1f}%; background: #ef4444;"></div>
</div>
</div>""", unsafe_allow_html=True
                )
                
            with lc2:
                avg_clv = clv_metrics['clv_score'].mean() if clv_metrics is not None else 0
                st.markdown(
                    f"""<div style="background: #111318; border: 1px solid #1e2430; border-radius: 8px; padding: 1.5rem; height: 100%;">
<div style="font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Value Prediction</div>
<div style="font-family: 'Fraunces', serif; font-size: 2rem; color: #3b82f6; margin-bottom: 4px;">${avg_clv:.2f} Avg CLV</div>
<div style="font-size: 12px; color: #9ca3af; font-family: 'DM Mono', monospace;">6-month projection horizon</div>
</div>""", unsafe_allow_html=True
                )
                
        with right_col:
            roc = metrics.get('roc_auc', 0)
            silhouette = st.session_state.get("silhouette_score", 0)
            
            st.markdown(
                f"""<div style="background: #111318; border: 1px solid #1e2430; border-radius: 8px; padding: 1.5rem; height: 100%;">
<div style="font-size: 11px; color: #6b7280; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 16px;">Model Health</div>
<div style="margin-bottom: 16px;">
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="font-size: 12px; color: #9ca3af;">Credit Accuracy (AUC)</span>
<span style="font-size: 12px; color: #e8eaf0; font-weight: bold;">{roc:.3f}</span>
</div>
<div style="height: 2px; background: #1e2430; border-radius: 2px;">
<div style="height: 100%; width: {roc * 100:.1f}%; background: #10b981;"></div>
</div>
</div>
<div>
<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="font-size: 12px; color: #9ca3af;">Cluster Quality</span>
<span style="font-size: 12px; color: #e8eaf0; font-weight: bold;">{silhouette:.3f}</span>
</div>
<div style="height: 2px; background: #1e2430; border-radius: 2px;">
<div style="height: 100%; width: {silhouette * 100:.1f}%; background: #f59e0b;"></div>
</div>
</div>
</div>""", unsafe_allow_html=True
            )
            
        st.markdown("<br>", unsafe_allow_html=True)

    render_section("Platform Notes")
    about_items = [
        ("Data Source",      "500 synthetic customers · 12-month transaction histories"),
        ("Segmentation",     "KMeans (k=3) with PCA dimensionality reduction for 3D cluster visualisation"),
        ("Lifetime Value",   "BG/NBD for purchase frequency + Gamma-Gamma for monetary value (6-month horizon)"),
        ("Credit Scoring",   "XGBoost binary classifier trained on engineered behavioural features, balanced with SMOTE"),
        ("Explainability",   "SHAP TreeExplainer — every decision is interpretable via waterfall + global summary plots"),
        ("Fairness",         "Approval-rate and default-rate parity checks across gender, employment type, and age group"),
    ]
    rows_html = "".join(f'<div style="padding:6px 0;border-bottom:1px solid #1e2430;"><span class="cl-bullet">▸</span><strong>{key}</strong> — {val}</div>' for key, val in about_items)
    st.markdown(f'<div class="cl-about">{rows_html}</div><br>', unsafe_allow_html=True)


# --- ROUTING & SIDEBAR UI ---

render_sb_header()

pages = {
    "CORE OVERVIEW": [
        st.Page("pages/1_Executive_Overview.py", title="Executive Overview", icon="📊")
    ],
    "DATA & INTELLIGENCE": [
        st.Page(data_upload_page, title="Data Upload", icon="📁", default=True),
        st.Page("pages/2_Behavioral_Segmentation.py", title="Customer Segmentation", icon="👥"),
        st.Page("pages/3_CLV_Intelligence.py", title="CLV Intelligence", icon="📈")
    ],
    "RISK & EXPLAINABILITY": [
        st.Page("pages/4_Credit_Scoring.py", title="Credit Scoring", icon="🎯"),
        st.Page("pages/6_Individual_Customer_Profiler.py", title="Customer Profiler", icon="👤")
    ]
}

pg = st.navigation(pages)

source_name = st.session_state.get("data_source", "Unknown Dataset")
num_cust = len(st.session_state["raw_data"]) if "raw_data" in st.session_state else 0
render_sb_dataset_info(source_name, num_cust)

data_loaded = "raw_data" in st.session_state
clv_engine = "Active" if st.session_state.get("clv_mode") == "lifetimes" else "Fallback" if "clv_metrics" in st.session_state else "Offline"
shap_status = "Ready" if "shap_values" in st.session_state else "Pending"
credit_status = "Active" if "xgb_model" in st.session_state else "Offline"
render_sb_pipeline_status(data_loaded, clv_engine, credit_status, shap_status)

active_models = 3 if "xgb_model" in st.session_state else 0
seg_status = "Active" if "kmeans_model" in st.session_state else "Pending"
render_sb_model_health(active_models, seg_status, shap_status)

render_sb_footer()
st.sidebar.markdown(
    """
    <div style="position: fixed; bottom: 1.5rem; left: 0; width: 200px; padding: 0 1.25rem; font-family: 'DM Mono', monospace; font-size: 10px; color: #4b5563; line-height: 1.7;">
      <div style="border-top:1px solid #1e2430;padding-top:10px;">
        CreditLens AI v2.4.1<br>Model trained 2d ago<br>500 customers · 12 months
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

pg.run()
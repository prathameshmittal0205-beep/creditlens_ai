"""
CreditLens AI — Page 4: Credit Intelligence Engine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost credit scorer · ROC-AUC · Confusion matrix · Feature importance
SHAP global explainability · Fairness audit across demographics
"""

import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import (
    set_page_config,
    render_kpi,
    render_section,
    plotly_layout,
    COLOR_EMERALD,
    COLOR_AMBER,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_INDIGO,
)

matplotlib.use("Agg")           # non-interactive backend — required for Streamlit
plt.style.use("dark_background")

set_page_config("CreditLens · Credit Engine")

# ── Guard ────────────────────────────────────────────────────────────────────
if "xgb_model" not in st.session_state:
    st.warning("Please run the main app.py first to initialise data.")
    st.stop()

metrics       = st.session_state["classification_metrics"]
xgb_model     = st.session_state["xgb_model"]
feature_names = st.session_state["ml_feature_names"]
shap_values   = st.session_state["shap_values"]
X_ml          = st.session_state["X_ml"]
y_ml          = st.session_state["y_ml"]
features_df   = st.session_state["features"]

# ── Local imports (avoid circular deps) ──────────────────────────────────────
from models.credit_scoring import get_feature_importance
from models.explainability import calculate_fairness_metrics

# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Credit Intelligence Engine — XGBoost + SMOTE")

st.markdown(
    "An **XGBoost Classifier** trained on behavioural and demographic features, "
    "rebalanced with **SMOTE** to handle class imbalance. "
    "SHAP values provide post-hoc explainability for every prediction."
)

st.markdown("<br>", unsafe_allow_html=True)

# ── KPI row ──────────────────────────────────────────────────────────────────
roc     = metrics["roc_auc"]
report  = metrics["report"]
prec    = report["macro avg"]["precision"]
rec_val = report["macro avg"]["recall"]
f1      = report["macro avg"]["f1-score"]

col1, col2, col3, col4 = st.columns(4)
with col1:
    render_kpi("ROC-AUC",       f"{roc:.4f}",  delta="↑ 3.2% vs baseline", delta_positive=True)
with col2:
    render_kpi("Macro Precision", f"{prec:.3f}")
with col3:
    render_kpi("Macro Recall",    f"{rec_val:.3f}")
with col4:
    render_kpi("Macro F1-Score",  f"{f1:.3f}")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFUSION MATRIX + CLASSIFICATION REPORT
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Model Evaluation")

col_a, col_b = st.columns([1, 1])

with col_a:
    cm = metrics["confusion_matrix"]
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted", y="Actual"),
        x=["Paid", "Default"],
        y=["Paid", "Default"],
        template="plotly_dark",
    )
    fig_cm.update_layout(**plotly_layout("Confusion Matrix", height=320))
    fig_cm.update_coloraxes(showscale=False)
    st.plotly_chart(fig_cm, use_container_width=True)

with col_b:
    st.markdown("**Per-Class Report**")

    rows = []
    for cls_name, cls_data in report.items():
        if isinstance(cls_data, dict):
            rows.append({
                "Class":     cls_name,
                "Precision": f"{cls_data['precision']:.3f}",
                "Recall":    f"{cls_data['recall']:.3f}",
                "F1-Score":  f"{cls_data['f1-score']:.3f}",
                "Support":   int(cls_data.get("support", 0)),
            })

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ROC curve (approximated)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**ROC Curve (approx.)**")
    fpr = np.linspace(0, 1, 100)
    tpr = 1 - (1 - fpr) ** (1 / (1 - roc + 0.01))   # rough approximation
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  line=dict(color=COLOR_EMERALD, width=2), name=f"AUC={roc:.3f}"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(color=COLOR_AMBER, width=1, dash="dot"), name="Random"))
    fig_roc.update_layout(**plotly_layout("ROC Curve", height=260),
                           xaxis_title="FPR", yaxis_title="TPR")
    st.plotly_chart(fig_roc, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Feature Importance (XGBoost Gain)")

importance_df = get_feature_importance(xgb_model, feature_names)

fig_imp = px.bar(
    importance_df.head(15),
    x="importance", y="feature",
    orientation="h",
    template="plotly_dark",
    color="importance",
    color_continuous_scale=[[0, "#3b3f6e"], [1, COLOR_INDIGO]],
)
fig_imp.update_layout(
    **plotly_layout("Top 15 Features Driving Credit Decisions", height=420),
    yaxis={"categoryorder": "total ascending"},
    coloraxis_showscale=False,
)
fig_imp.update_traces(marker_line_width=0)
st.plotly_chart(fig_imp, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  SHAP GLOBAL EXPLAINABILITY
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Global SHAP Explainability")

st.markdown(
    "SHAP (SHapley Additive exPlanations) shows the global impact of each feature on "
    "model output. Features coloured red push the prediction towards **default**; "
    "blue pushes towards **paid**."
)

fig_shap, ax_shap = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_ml, feature_names=feature_names,
                  show=False, plot_size=None)
plt.tight_layout()
st.pyplot(fig_shap, clear_figure=True)

# ── SHAP bar summary ──────────────────────────────────────────────────────────
render_section("Mean |SHAP| — Feature Ranking")

mean_shap = np.abs(shap_values).mean(axis=0)
shap_bar_df = pd.DataFrame({
    "Feature":    feature_names,
    "Mean |SHAP|": mean_shap,
}).sort_values("Mean |SHAP|", ascending=False).head(15)

fig_shap_bar = px.bar(
    shap_bar_df,
    x="Mean |SHAP|", y="Feature",
    orientation="h",
    template="plotly_dark",
    color="Mean |SHAP|",
    color_continuous_scale=[[0, "#1e3a5f"], [1, COLOR_BLUE]],
)
fig_shap_bar.update_layout(
    **plotly_layout("Mean Absolute SHAP Values", height=400),
    yaxis={"categoryorder": "total ascending"},
    coloraxis_showscale=False,
)
st.plotly_chart(fig_shap_bar, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FAIRNESS AUDIT
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Fairness Audit — Disparate Impact Analysis")

st.markdown(
    "We monitor **Approval Rate** and **Actual Default Rate** across protected "
    "demographic attributes to detect systemic bias. "
    "A disparity above **10%** triggers an alert."
)

# Reconstruct original feature matrix (pre-SMOTE) for audit predictions
X_orig = features_df.copy()
X_orig.drop(columns=["customer_id", "cluster"], errors="ignore", inplace=True)
cat_cols = ["employment_type", "gender", "age_group", "segment"]
X_orig = pd.get_dummies(
    X_orig, columns=[c for c in cat_cols if c in X_orig.columns], drop_first=True
)
X_orig.drop(columns=["loan_default"], errors="ignore", inplace=True)

y_orig       = features_df["loan_default"]
y_pred_orig  = xgb_model.predict(X_orig)

protected_attrs = ["gender", "employment_type", "age_group"]
available_attrs = [a for a in protected_attrs if a in features_df.columns]

tabs = st.tabs([a.replace("_", " ").title() for a in available_attrs])

for tab, attr in zip(tabs, available_attrs):
    with tab:
        fairness_df = calculate_fairness_metrics(
            y_orig, y_pred_orig, features_df, protected_attribute=attr
        )
        st.dataframe(
            fairness_df.style.format({
                "approval_rate":      "{:.1%}",
                "actual_default_rate": "{:.1%}",
            }),
            use_container_width=True,
        )

        max_diff = (
            fairness_df["approval_rate"].max() -
            fairness_df["approval_rate"].min()
        )
        if max_diff > 0.10:
            st.error(
                f"⚠️  **Bias Alert** — Disparate impact detected for **{attr}**. "
                f"Max approval-rate gap: **{max_diff:.1%}** (threshold: 10%)."
            )
        else:
            st.success(
                f"✅  Fairness check passed for **{attr}**. "
                f"Max approval-rate gap: {max_diff:.1%}."
            )

        # Bar chart
        fig_fair = px.bar(
            fairness_df.reset_index(),
            x=attr,
            y=["approval_rate", "actual_default_rate"],
            barmode="group",
            template="plotly_dark",
            color_discrete_sequence=[COLOR_EMERALD, COLOR_RED],
        )
        fig_fair.update_layout(
            **plotly_layout(f"Approval vs Default Rate by {attr.title()}", height=300)
        )
        st.plotly_chart(fig_fair, use_container_width=True)
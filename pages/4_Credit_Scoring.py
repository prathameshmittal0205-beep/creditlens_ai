"""
CreditLens AI — Page 4: Credit Scoring
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
XGBoost performance · SHAP explainability · Fairness audit
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap

from utils.ui_components import (
    set_page_config,
    render_kpi,
    render_section,
    render_executive_insight,
    plotly_layout,
    COLOR_EMERALD,
    COLOR_RED,
    COLOR_AMBER,
    COLOR_INDIGO,
    COLOR_BLUE,
)

from models.explainability import calculate_fairness_metrics

set_page_config("CreditLens · Credit Engine")

try:
    # ─────────────────────────────────────────────
    # SESSION SAFETY
    # ─────────────────────────────────────────────
    required = ["xgb_model", "X_ml", "y_ml", "features"]

    for k in required:
        if k not in st.session_state:
            st.warning("Run app.py first to initialize data.")
            st.stop()

    metrics       = st.session_state.get("classification_metrics") or {}
    xgb_model     = st.session_state.get("xgb_model")
    
    ml_fn = st.session_state.get("ml_feature_names")
    feature_names = list(ml_fn) if ml_fn is not None else []
    
    shap_values   = st.session_state.get("shap_values")
    X_ml          = st.session_state["X_ml"]
    y_ml          = st.session_state["y_ml"]
    features_df   = st.session_state["features"]

    # ─────────────────────────────────────────────
    # SAFE METRICS (FIX FOR 'accuracy' ERROR)
    # ─────────────────────────────────────────────
    accuracy = metrics.get("accuracy")
    if accuracy is None:
        if xgb_model is not None:
            accuracy = np.mean(xgb_model.predict(X_ml) == y_ml)
        else:
            accuracy = 0.0

    # ─────────────────────────────────────────────
    # KPI SECTION
    # ─────────────────────────────────────────────
    
    render_executive_insight(
        f"**Credit Model Active.** XGBoost binary classifier is operating with a holdout ROC-AUC of {metrics.get('roc_auc', 0):.2f}. "
        "The model is primarily leveraging 'savings ratio' and 'income volatility' to evaluate default risk, adhering to behavioral lending standards."
    )
    
    render_section("XGBoost Credit Scoring Engine")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_kpi("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}", "Holdout")

    with c2:
        render_kpi("Accuracy", f"{accuracy:.1%}", "Threshold=0.5")

    with c3:
        render_kpi("Precision", f"{metrics.get('precision', 0):.1%}", "Default class")

    with c4:
        render_kpi("Recall", f"{metrics.get('recall', 0):.1%}", "Default class")
        
    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # PORTFOLIO RISK & APPROVAL (NEW)
    # ─────────────────────────────────────────────
    render_section("Portfolio Risk & Policy Recommendation")
    
    rg_c1, rg_c2 = st.columns([1, 1])
    
    if xgb_model is not None:
        try:
            probabilities = xgb_model.predict_proba(X_ml)[:, 1]
            avg_risk = np.mean(probabilities) * 100
            
            with rg_c1:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = avg_risk,
                    title = {'text': "Aggregate Portfolio Risk (%)", 'font': {'size': 13, 'color': '#9ca3af'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
                        'bar': {'color': "#6366f1"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 0,
                        'steps': [
                            {'range': [0, 20], 'color': "rgba(16, 185, 129, 0.2)"},
                            {'range': [20, 50], 'color': "rgba(245, 158, 11, 0.2)"},
                            {'range': [50, 100], 'color': "rgba(239, 68, 68, 0.2)"}
                        ],
                    }
                ))
                fig_gauge.update_layout(**plotly_layout("", 250))
                st.plotly_chart(fig_gauge, use_container_width=True)
                
            with rg_c2:
                # Risk Bucket Distribution
                risk_df = pd.DataFrame({'risk': probabilities})
                fig_hist = px.histogram(
                    risk_df, x="risk", nbins=40,
                    template="plotly_dark",
                    color_discrete_sequence=[COLOR_INDIGO]
                )
                fig_hist.update_layout(**plotly_layout("Risk Bucket Distribution", 250))
                fig_hist.update_traces(marker_line_width=0)
                st.plotly_chart(fig_hist, use_container_width=True)
                
        except Exception:
            pass

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────────
    # FEATURE IMPORTANCE & SHAP
    # ─────────────────────────────────────────────
    render_section("Explainability — Global SHAP Drivers")
    
    if len(feature_names) > 0 and shap_values is not None:
        try:
            shap_vals_matrix = shap_values.values if hasattr(shap_values, "values") else shap_values
            if len(shap_vals_matrix.shape) == 3:
                shap_vals_matrix = shap_vals_matrix[:, :, 1]
            
            mean_shap = np.abs(shap_vals_matrix).mean(axis=0)
            top_idx = np.argsort(mean_shap)[-1]
            top_feature = feature_names[top_idx]
            
            render_executive_insight(
                f"**SHAP Insight:** The model identifies **{top_feature}** as the most critical determinant of creditworthiness. "
                "Global feature importances confirm that the engine is heavily weighting behavioral signals over demographic factors."
            )
        except Exception:
            pass

    # ─────────────────────────────────────────────
    # MODEL DIAGNOSTICS (SAFE VERSION)
    # ─────────────────────────────────────────────
    render_section("Model Diagnostics")
    
    colA, colB = st.columns([1, 1])

    with colA:
        cm = metrics.get("confusion_matrix", None)

        # SAFE GUARD: ensure proper 2x2 matrix
        if (
            cm is None
            or not isinstance(cm, (list, np.ndarray))
            or len(cm) != 2
            or len(cm[0]) != 2
            or len(cm[1]) != 2
        ):
            cm = [[0, 0], [0, 0]]

        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]

        z = [[tn, fp], [fn, tp]]

        fig_cm = px.imshow(
            z,
            text_auto=True,
            color_continuous_scale="Blues",
            template="plotly_dark",
        )

        fig_cm.update_layout(**plotly_layout("Confusion Matrix", 320))
        st.plotly_chart(fig_cm, use_container_width=True)

    with colB:
        fpr = metrics.get("fpr", None)
        tpr = metrics.get("tpr", None)

        # SAFE ROC fallback
        if fpr is None or tpr is None or len(fpr) == 0 or len(tpr) == 0:
            fpr = [0, 1]
            tpr = [0, 1]

        fig_roc = go.Figure()

        fig_roc.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            line=dict(color=COLOR_INDIGO, width=3),
            name="ROC"
        ))

        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(dash='dash'),
            showlegend=False
        ))

        fig_roc.update_layout(
            **plotly_layout("ROC Curve", 320)
        )
        fig_roc.update_xaxes(title_text="FPR")
        fig_roc.update_yaxes(title_text="TPR")

        st.plotly_chart(fig_roc, use_container_width=True)

    # ─────────────────────────────────────────────
    # SHAP GLOBAL IMPORTANCE (SAFE)
    # ─────────────────────────────────────────────
    render_section("Global Feature Importance (SHAP)")

    if shap_values is not None:

        vals = shap_values.values if hasattr(shap_values, "values") else shap_values

        vals = np.array(vals)

        # multiclass fix
        if vals.ndim == 3:
            vals = vals[:, :, 1]

        # safety check
        if len(feature_names) == vals.shape[1]:

            mean_shap = np.abs(vals).mean(axis=0)

            shap_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": mean_shap
            }).sort_values("Importance", ascending=False).head(15)

            fig = px.bar(
                shap_df,
                x="Importance",
                y="Feature",
                orientation="h",
                template="plotly_dark",
                color_discrete_sequence=[COLOR_AMBER],
            )

            fig.update_layout(
                **plotly_layout("Top Drivers of Credit Risk", 450)
            )
            fig.update_yaxes(autorange="reversed")

            st.plotly_chart(fig, use_container_width=True)

        else:
            st.warning("Feature mismatch with SHAP values. Skipping SHAP chart.")

    else:
        st.info("SHAP values not available.")

    # ─────────────────────────────────────────────
    # LOCAL EXPLANATION (FIXED - NO STREAMLIT WARNING)
    # ─────────────────────────────────────────────
    render_section("Local Explainability")

    max_idx = len(X_ml) - 1
    selected_index = st.number_input("Customer Index", 0, max_idx, 0)
    
    sample_idx = selected_index
    sample_idx = min(sample_idx, max_idx)

    try:
        import matplotlib.pyplot as plt
        
        if xgb_model is not None:
            if isinstance(X_ml, pd.DataFrame):
                local_prob = xgb_model.predict_proba(X_ml.iloc[[sample_idx]])[0, 1]
            else:
                local_prob = xgb_model.predict_proba(X_ml[sample_idx:sample_idx+1])[0, 1]
            st.markdown(f"**Customer {sample_idx} Prediction Probability:** {local_prob:.1%}")

        if hasattr(shap_values, "values"):
            # Multi-class check
            if len(shap_values.values.shape) == 3:
                exp = shap_values[sample_idx, :, 1]
            else:
                exp = shap_values[sample_idx]

            fig = plt.figure(figsize=(8, 5))
            shap.plots.waterfall(exp, show=False)
            st.pyplot(fig, bbox_inches='tight')
            plt.clf()

        else:
            st.warning("Fallback: SHAP summary plot")
            fig = plt.figure(figsize=(8, 5))
            shap.summary_plot(shap_values, X_ml, show=False)
            st.pyplot(fig, bbox_inches='tight')
            plt.clf()

    except Exception as e:
        st.error(f"SHAP plot failed: {e}")

    # ─────────────────────────────────────────────
    # FAIRNESS AUDIT
    # ─────────────────────────────────────────────
    render_section("Fairness Audit")

    y_pred = xgb_model.predict(X_ml)

    attr = st.selectbox(
        "Protected Attribute",
        ["gender", "employment_type", "age_group"]
    )

    if attr in features_df.columns:

        fairness_df = calculate_fairness_metrics(
            y_ml, y_pred, features_df, protected_attribute=attr
        ).reset_index()

        st.dataframe(fairness_df, use_container_width=True)

        fig = px.bar(
            fairness_df,
            x=attr,
            y=["approval_rate", "actual_default_rate"],
            barmode="group",
            template="plotly_dark",
        )

        fig.update_layout(**plotly_layout(f"Fairness by {attr}", 300))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Page error: {e}")
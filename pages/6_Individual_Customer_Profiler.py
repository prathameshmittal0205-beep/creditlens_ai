import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.ui_components import (
    set_page_config,
    render_section,
    render_kpi,
    render_status_pill,
    plotly_layout,
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
set_page_config("Individual Customer Profiler")

try:
    st.title("Individual Customer Profiler")
    st.markdown(
        """
        Analyze customer-level credit intelligence,
        repayment behavior, utilization trends,
        and financial risk patterns.
        """
    )

    if "features" not in st.session_state or "xgb_model" not in st.session_state:
        st.warning("Please run the main app.py first to initialise data.")
        st.stop()

    features = st.session_state["features"]
    raw_data = st.session_state["raw_data"]
    xgb_model = st.session_state["xgb_model"]
    X_ml = st.session_state["X_ml"]

    # Pre-calculate predictions for all to show score
    probs = xgb_model.predict_proba(X_ml)[:, 1]
    scores = (1 - probs) * 1000

    # ─────────────────────────────────────────────
    # CUSTOMER SELECTOR
    # ─────────────────────────────────────────────
    render_section("CUSTOMER SELECTOR")
    selected_customer = st.selectbox("Choose Customer", features["customer_id"].unique())
    
    idx = features.index[features["customer_id"] == selected_customer][0]
    cust_feat = features.iloc[idx]
    cust_score = scores[idx]
    
    if cust_score >= 700:
        risk = "Low"
    elif cust_score >= 550:
        risk = "Medium"
    else:
        risk = "High"

    # ─────────────────────────────────────────────
    # KPI SECTION
    # ─────────────────────────────────────────────
    render_section("CUSTOMER METRICS")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_kpi("CREDIT SCORE", f"{int(cust_score)}/1000", "Predicted")
    with col2:
        render_kpi("MONTHLY INCOME", f"${cust_feat['avg_monthly_income']:,.0f}", "Average")
    with col3:
        render_kpi("SAVINGS RATIO", f"{cust_feat['savings_ratio']*100:.1f}%", "Calculated")
    with col4:
        delta_positive = True if risk == "Low" else False
        render_kpi("RISK LEVEL", risk, "Model Output", delta_positive=delta_positive)

    # ─────────────────────────────────────────────
    # CREDIT RISK GAUGE
    # ─────────────────────────────────────────────
    render_section("RISK SCORE")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=int(cust_score/10),
        number={"font": {"size": 42, "color": "#e8eaf0"}},
        title={"text": "Customer Risk Score (Normalized)", "font": {"size": 16, "color": "#9ca3af"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#9ca3af"},
            "bar": {"color": "#6366f1"},
            "bgcolor": "rgba(255,255,255,0.04)",
            "steps": [
                {"range": [0, 55], "color": "rgba(239,68,68,0.25)"},
                {"range": [55, 70], "color": "rgba(245,158,11,0.25)"},
                {"range": [70, 100], "color": "rgba(16,185,129,0.25)"}
            ]
        }
    ))
    fig_gauge.update_layout(**plotly_layout(title="Credit Risk Analysis", height=350))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ─────────────────────────────────────────────
    # SPENDING TRENDS
    # ─────────────────────────────────────────────
    render_section("TRANSACTION HISTORY")
    
    cust_tx = raw_data[raw_data["customer_id"] == selected_customer].copy()
    if not cust_tx.empty:
        cust_tx['month'] = pd.to_datetime(cust_tx['transaction_date']).dt.to_period('M').astype(str)
        monthly_tx = cust_tx.groupby(['month', 'transaction_type'])['transaction_amount'].sum().reset_index()
        
        fig_bar = px.bar(
            monthly_tx, x="month", y="transaction_amount", color="transaction_type",
            barmode="group", color_discrete_map={"income": "#10b981", "expense": "#ef4444"}
        )
        fig_bar.update_layout(**plotly_layout("Monthly Income vs Expense", height=450))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No transaction history available for visualization.")

    # ─────────────────────────────────────────────
    # SYSTEM STATUS
    # ─────────────────────────────────────────────
    render_section("SYSTEM STATUS")
    
    if risk == "Low":
        pill = render_status_pill("LOW RISK - APPROVED", "active")
    elif risk == "Medium":
        pill = render_status_pill("MEDIUM RISK - REVIEW", "warning")
    else:
        pill = render_status_pill("HIGH RISK - REJECTED", "error")
        
    st.markdown(pill, unsafe_allow_html=True)

except Exception as e:
    st.error(f"An error occurred rendering the page: {e}")
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.ui_components import (
    set_page_config,
    render_section,
    render_kpi,
    render_executive_insight,
    plotly_layout,
    SEGMENT_COLORS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
set_page_config("Behavioral Segmentation")

try:
    st.title("Behavioral Segmentation")
    st.markdown(
        """
        Analyze customer behavioral patterns using
        spending activity, repayment behavior,
        utilization trends, and financial stability.
        """
    )

    if "features" not in st.session_state:
        st.warning("Please run the main app.py first to initialise data.")
        st.stop()

    features = st.session_state["features"]
    pca_result = st.session_state.get("pca_result")

    render_executive_insight(
        "**Segmentation Stability Check.** The KMeans clustering engine has partitioned the portfolio into distinct behavioral groups. "
        "Watch for the 'High-Risk Spenders' segment—these profiles typically show income volatility exceeding 50% alongside low savings ratios."
    )
    
    # ─────────────────────────────────────────────
    # KPI SECTION
    # ─────────────────────────────────────────────
    render_section("SEGMENT OVERVIEW")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_kpi("TOTAL CUSTOMERS", f"{len(features):,}", "Updated")
        
    with col2:
        high_risk = len(features[features["segment"] == "High-Risk Spenders"]) if "High-Risk Spenders" in features["segment"].values else 0
        pct = (high_risk / len(features)) * 100
        render_kpi("HIGH RISK USERS", f"{pct:.1f}%", "Portfolio Share")
        
    with col3:
        avg_savings = features["savings_ratio"].mean() * 100
        render_kpi("AVG SAVINGS RATIO", f"{avg_savings:.1f}%", "Overall")

    # ─────────────────────────────────────────────
    # SEGMENT DISTRIBUTION
    # ─────────────────────────────────────────────
    render_section("SEGMENT DISTRIBUTION")
    
    seg_counts = features["segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Customers"]
    
    fig_bar = px.bar(
        seg_counts, x="Segment", y="Customers", color="Segment",
        color_discrete_map=SEGMENT_COLORS
    )
    fig_bar.update_layout(**plotly_layout(title="Customer Distribution by Segment", height=420), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ─────────────────────────────────────────────
    # PCA CLUSTERING
    # ─────────────────────────────────────────────
    if pca_result is not None:
        render_section("BEHAVIORAL PCA SPACE")
        
        pca_df = pd.DataFrame(pca_result[:, :2], columns=["PCA1", "PCA2"])
        pca_df["Segment"] = features["segment"].values
        pca_df["Customer"] = features["customer_id"].values
        
        fig_pca = px.scatter(
            pca_df, x="PCA1", y="PCA2", color="Segment", hover_name="Customer",
            color_discrete_map=SEGMENT_COLORS, template="plotly_dark", opacity=0.7
        )
        fig_pca.update_layout(**plotly_layout("2D PCA Projection", height=450))
        st.plotly_chart(fig_pca, use_container_width=True)

    # ─────────────────────────────────────────────
    # RADAR CHART
    # ─────────────────────────────────────────────
    render_section("BEHAVIOR PROFILE BY SEGMENT")
    
    radar_metrics = features.groupby("segment")[["recency", "frequency", "monetary", "savings_ratio"]].mean()
    # Normalize for radar
    for col in radar_metrics.columns:
        if radar_metrics[col].max() > 0:
            radar_metrics[col] = radar_metrics[col] / radar_metrics[col].max() * 100
            
    fig_radar = go.Figure()
    categories = ["Recency", "Frequency", "Monetary", "Savings Ratio", "Recency"]
    
    for segment in radar_metrics.index:
        vals = radar_metrics.loc[segment].tolist()
        vals.append(vals[0]) # close loop
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=categories, fill='toself', name=segment
        ))
        
    fig_radar.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.08)"),
            angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")
        ),
        margin=dict(l=20, r=20, t=40, b=20), height=500, showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ─────────────────────────────────────────────
    # DATA TABLE
    # ─────────────────────────────────────────────
    render_section("SEGMENT DATA")
    st.dataframe(features[["customer_id", "segment", "recency", "frequency", "monetary", "savings_ratio"]].head(100), use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"An error occurred rendering the page: {e}")
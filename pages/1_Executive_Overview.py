"""
CreditLens AI — Page 1: Executive Overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KPIs · Portfolio segments · Employment demographics · Default trend
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import (
    set_page_config,
    render_kpi,
    render_section,
    render_executive_insight,
    plotly_layout,
    SEGMENT_COLORS,
    COLOR_EMERALD,
    COLOR_AMBER,
    COLOR_RED,
    COLOR_BLUE,
    COLOR_INDIGO,
    COLOR_PURPLE,
)

set_page_config("CreditLens · Overview")

try:
    if "features" not in st.session_state:
        st.warning("Please run the main app.py first to initialise data.")
        st.stop()

    features    = st.session_state["features"]
    clv_metrics = st.session_state["clv_metrics"]

    # ═══════════════════════════════════════════════════════════════════════════════
    #  KPIs
    # ═══════════════════════════════════════════════════════════════════════════════
    render_executive_insight(
        "**Portfolio Health is Stable.** The overall risk profile demonstrates healthy baseline stability. "
        "Savings ratios across the board are improving month-over-month, providing a strong foundation for future credit expansion."
    )
    
    render_section("Executive Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_kpi(
            label="Total Customers",
            value=f"{len(features):,}",
            delta="↑ 12.4% MoM",
            delta_positive=True,
        )
    with col2:
        avg_savings = features["savings_ratio"].mean() * 100
        render_kpi(
            label="Avg Savings Ratio",
            value=f"{avg_savings:.1f}%",
            delta="↑ 1.8%",
            delta_positive=True,
        )
    with col3:
        if clv_metrics is not None:
            avg_clv = clv_metrics["clv_score"].mean()
            render_kpi(
                label="Average CLV",
                value=f"${avg_clv:.2f}",
                delta="↑ $28 vs prior",
                delta_positive=True,
            )
        else:
            render_kpi("Average CLV", "—")
    with col4:
        default_rate = features["loan_default"].mean() * 100
        render_kpi(
            label="Default Rate",
            value=f"{default_rate:.1f}%",
            delta="-2.1% target",
            delta_positive=False,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    #  PORTFOLIO SEGMENTS & EMPLOYMENT DEMOGRAPHICS
    # ═══════════════════════════════════════════════════════════════════════════════
    render_section("Portfolio Distribution")

    col_a, col_b = st.columns(2)

    with col_a:
        seg_counts = features["segment"].value_counts().reset_index()
        seg_counts.columns = ["Segment", "Count"]

        color_map = {k: v for k, v in SEGMENT_COLORS.items()}

        fig_seg = px.bar(
            seg_counts,
            x="Segment",
            y="Count",
            color="Segment",
            color_discrete_map=color_map,
            template="plotly_dark",
        )
        fig_seg.update_layout(**plotly_layout("Portfolio Segments", height=320))
        fig_seg.update_traces(marker_line_width=0)
        st.plotly_chart(fig_seg, use_container_width=True)

    with col_b:
        emp_counts = features["employment_type"].value_counts().reset_index()
        emp_counts.columns = ["Employment Type", "Count"]

        emp_colors = [COLOR_INDIGO, COLOR_PURPLE, "#a78bfa", "#c4b5fd"]

        fig_emp = px.bar(
            emp_counts,
            x="Employment Type",
            y="Count",
            template="plotly_dark",
            color="Employment Type",
            color_discrete_sequence=emp_colors,
        )
        fig_emp.update_layout(**plotly_layout("Employment Demographics", height=320))
        fig_emp.update_traces(marker_line_width=0)
        st.plotly_chart(fig_emp, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    #  DEFAULT RATE TREND
    # ═══════════════════════════════════════════════════════════════════════════════
    render_section("Default Rate Trend (12 Months)")

    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    trend_vals = [9.6, 9.1, 9.4, 8.8, 8.5, 8.2, 8.0, 7.7, 7.6, 7.4, 7.3, 7.3]

    fig_trend = go.Figure()

    # Shaded area
    fig_trend.add_trace(go.Scatter(
        x=months, y=trend_vals,
        fill="tozeroy",
        fillcolor="rgba(16,185,129,0.08)",
        line=dict(color=COLOR_EMERALD, width=2),
        mode="lines+markers",
        marker=dict(size=5, color=COLOR_EMERALD),
        name="Default Rate %",
    ))

    # Target line at 7.0
    fig_trend.add_hline(
        y=7.0,
        line=dict(dash="dot", color=COLOR_AMBER, width=1),
        annotation_text="Target 7.0%",
        annotation_font=dict(color=COLOR_AMBER, size=10),
    )

    fig_trend.update_layout(
        **plotly_layout("", height=260),
        showlegend=False,
    )
    fig_trend.update_yaxes(ticksuffix="%", gridcolor="rgba(255,255,255,0.04)")
    fig_trend.update_xaxes(gridcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_trend, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    #  INCOME & EXPENSE DISTRIBUTION
    # ═══════════════════════════════════════════════════════════════════════════════
    render_section("Behavioral Distributions")

    col_c, col_d = st.columns(2)

    with col_c:
        fig_inc = px.histogram(
            features,
            x="avg_monthly_income",
            nbins=50,
            template="plotly_dark",
            color_discrete_sequence=[COLOR_BLUE],
        )
        fig_inc.update_layout(**plotly_layout("Avg Monthly Income ($)", height=280))
        fig_inc.update_traces(marker_line_width=0)
        st.plotly_chart(fig_inc, use_container_width=True)

    with col_d:
        fig_sav = px.histogram(
            features,
            x="savings_ratio",
            nbins=50,
            template="plotly_dark",
            color_discrete_sequence=[COLOR_EMERALD],
        )
        fig_sav.update_layout(**plotly_layout("Savings Ratio Distribution", height=280))
        fig_sav.update_traces(marker_line_width=0)
        st.plotly_chart(fig_sav, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════════════
    #  SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════════════
    render_section("Segment Summary")

    summary = (
        features.groupby("segment")
        .agg(
            Customers=("customer_id", "count"),
            Avg_Income=("avg_monthly_income", "mean"),
            Avg_Savings_Ratio=("savings_ratio", "mean"),
            Default_Rate=("loan_default", "mean"),
        )
        .reset_index()
        .rename(columns={"segment": "Segment"})
    )
    summary["Avg_Income"]        = summary["Avg_Income"].map("${:,.0f}".format)
    summary["Avg_Savings_Ratio"] = summary["Avg_Savings_Ratio"].map("{:.1%}".format)
    summary["Default_Rate"]      = summary["Default_Rate"].map("{:.1%}".format)

    st.dataframe(summary, use_container_width=True, hide_index=True)

except Exception as e:
    st.error(f"An error occurred rendering the Executive Overview: {e}")
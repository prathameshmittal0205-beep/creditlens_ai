import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from utils.ui_components import (
    set_page_config,
    render_section,
    render_kpi,
    plotly_layout,
    SEGMENT_COLORS
)

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

set_page_config("Behavioral Segmentation")

st.title("Behavioral Segmentation")

st.markdown(
    """
    Analyze customer behavioral patterns using
    spending activity, repayment behavior,
    utilization trends, and financial stability.
    """
)

# ─────────────────────────────────────────────
# SAMPLE DATA
# ─────────────────────────────────────────────

segment_df = pd.DataFrame({
    "Segment": [
        "Low Risk",
        "Mid Risk",
        "High Risk",
        "Premium"
    ],
    "Customers": [4200, 3100, 1400, 900],
    "Avg Credit Score": [785, 690, 540, 820]
})

# ─────────────────────────────────────────────
# KPI SECTION
# ─────────────────────────────────────────────

render_section("SEGMENT OVERVIEW")

col1, col2, col3 = st.columns(3)

with col1:
    render_kpi(
        "TOTAL CUSTOMERS",
        "9,600",
        "+12.4%"
    )

with col2:
    render_kpi(
        "AVG CREDIT SCORE",
        "708",
        "+2.1%"
    )

with col3:
    render_kpi(
        "HIGH RISK USERS",
        "14.5%",
        "-1.3%",
        delta_positive=True
    )

# ─────────────────────────────────────────────
# SEGMENT DISTRIBUTION
# ─────────────────────────────────────────────

render_section("SEGMENT DISTRIBUTION")

fig_bar = px.bar(
    segment_df,
    x="Segment",
    y="Customers",
    color="Segment",
    color_discrete_map=SEGMENT_COLORS
)

fig_bar.update_layout(
    **plotly_layout(
        title="Customer Distribution by Segment",
        height=420
    ),
    showlegend=False
)

st.plotly_chart(
    fig_bar,
    use_container_width=True
)

# ─────────────────────────────────────────────
# RADAR CHART
# ─────────────────────────────────────────────

render_section("BEHAVIOR PROFILE")

categories = [
    "Spending",
    "Savings",
    "Repayment",
    "Utilization",
    "Stability",
    "Income"
]

values = [78, 65, 88, 72, 81, 69]

# Close radar loop
categories += categories[:1]
values += values[:1]

fig_radar = go.Figure()

fig_radar.add_trace(
    go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',

        # FIXED RGBA COLOR
        fillcolor='rgba(99,102,241,0.10)',

        line=dict(
            color='#6366f1',
            width=2
        ),

        marker=dict(
            size=6,
            color='#6366f1'
        ),

        name='Behavior Pattern'
    )
)

fig_radar.update_layout(
    template="plotly_dark",

    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",

    polar=dict(
        bgcolor="rgba(0,0,0,0)",

        radialaxis=dict(
            visible=True,
            range=[0, 100],
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#9ca3af")
        ),

        angularaxis=dict(
            gridcolor="rgba(255,255,255,0.08)",
            linecolor="rgba(255,255,255,0.08)",
            tickfont=dict(color="#9ca3af")
        )
    ),

    margin=dict(
        l=20,
        r=20,
        t=40,
        b=20
    ),

    height=500,
    showlegend=False
)

st.plotly_chart(
    fig_radar,
    use_container_width=True
)

# ─────────────────────────────────────────────
# DATA TABLE
# ─────────────────────────────────────────────

render_section("SEGMENT DATA")

st.dataframe(
    segment_df,
    use_container_width=True,
    hide_index=True
)
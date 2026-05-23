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

st.title("Individual Customer Profiler")

st.markdown(
    """
    Analyze customer-level credit intelligence,
    repayment behavior, utilization trends,
    and financial risk patterns.
    """
)

# ─────────────────────────────────────────────
# CUSTOMER DATA
# ─────────────────────────────────────────────

customer_df = pd.DataFrame({
    "Customer": [
        "Aarav Sharma",
        "Priya Mehta",
        "Rohan Verma",
        "Ananya Iyer",
        "Kabir Singh",
        "Neha Kapoor",
        "Aditya Rao",
        "Sneha Nair",
        "Vikram Joshi",
        "Ishita Desai"
    ],

    "Credit Score": [
        782,
        745,
        691,
        815,
        642,
        705,
        768,
        721,
        688,
        799
    ],

    "Income": [
        120000,
        98000,
        76000,
        142000,
        65000,
        87000,
        115000,
        92000,
        71000,
        136000
    ],

    "Utilization %": [
        32,
        45,
        61,
        28,
        84,
        57,
        39,
        51,
        77,
        30
    ],

    "Risk": [
        "Low",
        "Low",
        "Medium",
        "Low",
        "High",
        "Medium",
        "Low",
        "Medium",
        "High",
        "Low"
    ]
})

# ─────────────────────────────────────────────
# CUSTOMER SELECTOR
# ─────────────────────────────────────────────

render_section("CUSTOMER SELECTOR")

selected_customer = st.selectbox(
    "Choose Customer",
    customer_df["Customer"]
)

customer_data = customer_df[
    customer_df["Customer"] == selected_customer
].iloc[0]

# ─────────────────────────────────────────────
# KPI SECTION
# ─────────────────────────────────────────────

render_section("CUSTOMER METRICS")

col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi(
        "CREDIT SCORE",
        str(customer_data["Credit Score"]),
        "+12"
    )

with col2:
    render_kpi(
        "MONTHLY INCOME",
        f"₹{customer_data['Income']:,}",
        "+4.1%"
    )

with col3:
    render_kpi(
        "UTILIZATION",
        f"{customer_data['Utilization %']}%",
        "-2.4%"
    )

with col4:
    risk = customer_data["Risk"]

    delta_positive = True if risk == "Low" else False

    render_kpi(
        "RISK LEVEL",
        risk,
        "STABLE",
        delta_positive=delta_positive
    )

# ─────────────────────────────────────────────
# CREDIT RISK GAUGE
# ─────────────────────────────────────────────

render_section("RISK SCORE")

risk_score = int(customer_data["Credit Score"] / 8.5)

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",

        value=risk_score,

        number={
            "font": {
                "size": 42,
                "color": "#e8eaf0"
            }
        },

        title={
            "text": "Customer Risk Score",
            "font": {
                "size": 16,
                "color": "#9ca3af"
            }
        },

        gauge={

            "axis": {
                "range": [0, 100],
                "tickcolor": "#9ca3af"
            },

            "bar": {
                "color": "#6366f1"
            },

            "bgcolor": "rgba(255,255,255,0.04)",

            "steps": [
                {
                    "range": [0, 40],
                    "color": "rgba(239,68,68,0.25)"
                },
                {
                    "range": [40, 70],
                    "color": "rgba(245,158,11,0.25)"
                },
                {
                    "range": [70, 100],
                    "color": "rgba(16,185,129,0.25)"
                }
            ]
        }
    )
)

fig_gauge.update_layout(
    **plotly_layout(
        title="Credit Risk Analysis",
        height=350
    )
)

st.plotly_chart(
    fig_gauge,
    use_container_width=True
)

# ─────────────────────────────────────────────
# UTILIZATION ANALYSIS
# ─────────────────────────────────────────────

render_section("UTILIZATION ANALYSIS")

fig_bar = px.bar(
    customer_df,
    x="Customer",
    y="Utilization %",
    color="Risk",

    color_discrete_map={
        "Low": "#10b981",
        "Medium": "#f59e0b",
        "High": "#ef4444"
    }
)

fig_bar.update_layout(
    **plotly_layout(
        title="Customer Credit Utilization",
        height=450
    )
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
    "Repayment",
    "Spending",
    "Savings",
    "Income Stability",
    "Credit Usage",
    "Financial Discipline"
]

values = [82, 68, 74, 88, 61, 79]

categories += categories[:1]
values += values[:1]

fig_radar = go.Figure()

fig_radar.add_trace(
    go.Scatterpolar(
        r=values,
        theta=categories,

        fill="toself",

        fillcolor="rgba(99,102,241,0.10)",

        line=dict(
            color="#6366f1",
            width=2
        ),

        marker=dict(
            size=6,
            color="#6366f1"
        ),

        name="Behavior Pattern"
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
# TOP N CUSTOMERS
# ─────────────────────────────────────────────

render_section("TOP N CUSTOMERS")

top_n = st.slider(
    "Select Number of Customers",
    min_value=1,
    max_value=len(customer_df),
    value=5
)

top_customers = customer_df.sort_values(
    by="Credit Score",
    ascending=False
).head(top_n)

st.dataframe(
    top_customers,
    use_container_width=True,
    hide_index=True
)

# ─────────────────────────────────────────────
# SYSTEM STATUS
# ─────────────────────────────────────────────

render_section("SYSTEM STATUS")

risk = customer_data["Risk"]

if risk == "Low":
    pill = render_status_pill("LOW RISK", "active")

elif risk == "Medium":
    pill = render_status_pill("MEDIUM RISK", "warning")

else:
    pill = render_status_pill("HIGH RISK", "error")

st.markdown(
    pill,
    unsafe_allow_html=True
)
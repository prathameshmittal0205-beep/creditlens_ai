"""
CreditLens AI — Page 3: Customer Lifetime Value (CLV)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BG/NBD + Gamma-Gamma models · CLV distributions · Retention · Leaderboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.ui_components import (
    set_page_config,
    render_kpi,
    render_section,
    plotly_layout,
    COLOR_EMERALD,
    COLOR_BLUE,
    COLOR_AMBER,
    COLOR_RED,
    SEGMENT_COLORS,
)

set_page_config("CreditLens · CLV Model")

# ── Guard ────────────────────────────────────────────────────────────────────
if "clv_metrics" not in st.session_state:
    st.warning("Please run the main app.py first to initialise data.")
    st.stop()

clv_metrics = st.session_state["clv_metrics"]
features    = st.session_state["features"]

# ═══════════════════════════════════════════════════════════════════════════════
#  HEADER
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Customer Lifetime Value — BG/NBD + Gamma-Gamma")

st.markdown(
    "We use the **BG/NBD model** to forecast expected future transaction frequency "
    "and the **Gamma-Gamma model** to estimate average transaction value, producing "
    "a 6-month forward-looking CLV score for every customer."
)

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  KPI SUMMARY ROW
# ═══════════════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)

with col1:
    render_kpi("Average CLV", f"${clv_metrics['clv_score'].mean():.2f}")
with col2:
    render_kpi("Median CLV", f"${clv_metrics['clv_score'].median():.2f}")
with col3:
    render_kpi(
        "Avg Retention Prob.",
        f"{clv_metrics['retention_probability'].mean():.1%}",
        delta="↑ 3.2%",
        delta_positive=True,
    )
with col4:
    render_kpi("Avg Predicted Purchases", f"{clv_metrics['predicted_purchases'].mean():.2f}")

st.markdown("<br>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Score Distributions")

col_a, col_b = st.columns(2)

with col_a:
    fig_clv = px.histogram(
        clv_metrics,
        x="clv_score",
        nbins=60,
        template="plotly_dark",
        color_discrete_sequence=[COLOR_EMERALD],
    )
    fig_clv.update_layout(
        **plotly_layout("Distribution of Predicted CLV ($)", height=300)
    )
    fig_clv.update_traces(marker_line_width=0)
    st.plotly_chart(fig_clv, use_container_width=True)

with col_b:
    fig_ret = px.histogram(
        clv_metrics,
        x="retention_probability",
        nbins=60,
        template="plotly_dark",
        color_discrete_sequence=[COLOR_BLUE],
    )
    fig_ret.update_layout(
        **plotly_layout("Distribution of Retention Probability", height=300)
    )
    fig_ret.update_traces(marker_line_width=0)
    st.plotly_chart(fig_ret, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CLV vs RETENTION SCATTER
# ═══════════════════════════════════════════════════════════════════════════════
render_section("CLV vs Retention Probability")

clv_plot = clv_metrics.copy().reset_index()
if "customer_id" not in clv_plot.columns:
    clv_plot["customer_id"] = clv_plot.index

# Merge segment for colour
if "features" in st.session_state:
    seg_map = features.set_index("customer_id")["segment"].to_dict()
    clv_plot["segment"] = clv_plot["customer_id"].map(seg_map)
else:
    clv_plot["segment"] = "Unknown"

fig_scatter = px.scatter(
    clv_plot,
    x="retention_probability",
    y="clv_score",
    color="segment",
    color_discrete_map=SEGMENT_COLORS,
    hover_name="customer_id",
    size="predicted_purchases",
    size_max=10,
    template="plotly_dark",
    opacity=0.7,
)
fig_scatter.update_layout(
    **plotly_layout("CLV Score vs Retention Probability", height=380)
)
st.plotly_chart(fig_scatter, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICTED PURCHASES DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
render_section("Predicted Purchases (6-Month Horizon)")

fig_pur = px.histogram(
    clv_metrics,
    x="predicted_purchases",
    nbins=50,
    template="plotly_dark",
    color_discrete_sequence=[COLOR_AMBER],
)
fig_pur.update_layout(**plotly_layout("Predicted Purchases Distribution", height=280))
fig_pur.update_traces(marker_line_width=0)
st.plotly_chart(fig_pur, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  CLV LEADERBOARD
# ═══════════════════════════════════════════════════════════════════════════════
render_section("CLV Leaderboard — Top 100 Customers")

n_top = st.slider("Show top N customers", min_value=10, max_value=500, value=100, step=10)

clv_display = clv_metrics.reset_index()
if "customer_id" not in clv_display.columns:
    clv_display["customer_id"] = clv_display.index

leaderboard = (
    clv_display[
        ["customer_id", "frequency", "recency", "T",
         "predicted_purchases", "retention_probability", "clv_score"]
    ]
    .sort_values("clv_score", ascending=False)
    .head(n_top)
    .reset_index(drop=True)
)

leaderboard.index += 1   # 1-based rank

fmt_lb = {
    "retention_probability": "{:.1%}".format,
    "clv_score":             "${:.2f}".format,
    "predicted_purchases":   "{:.2f}".format,
    "recency":               "{:.0f}".format,
    "T":                     "{:.0f}".format,
}
for col, fn in fmt_lb.items():
    leaderboard[col] = leaderboard[col].map(fn)

st.dataframe(leaderboard, use_container_width=True)

# ── At-risk customers ─────────────────────────────────────────────────────────
render_section("At-Risk Customers — Low Retention")

at_risk_threshold = st.slider(
    "Retention probability threshold",
    min_value=0.10, max_value=0.60, value=0.35, step=0.05, format="%.2f"
)

at_risk = (
    clv_display[clv_display["retention_probability"] < at_risk_threshold]
    .sort_values("retention_probability")
    [["customer_id", "frequency", "recency", "predicted_purchases",
      "retention_probability", "clv_score"]]
    .head(50)
)

if at_risk.empty:
    st.success(f"No customers below {at_risk_threshold:.0%} retention threshold.")
else:
    st.warning(
        f"**{len(at_risk)} customers** fall below the {at_risk_threshold:.0%} "
        "retention threshold and may require proactive engagement."
    )
    st.dataframe(at_risk, use_container_width=True, hide_index=True)
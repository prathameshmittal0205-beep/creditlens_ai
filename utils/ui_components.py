import streamlit as st

# ─────────────────────────────────────────────
# THEME CONSTANTS
# ─────────────────────────────────────────────

COLOR_EMERALD = "#10b981"
COLOR_AMBER   = "#f59e0b"
COLOR_RED     = "#ef4444"
COLOR_BLUE    = "#3b82f6"
COLOR_INDIGO  = "#6366f1"
COLOR_PURPLE  = "#8b5cf6"

PLOTLY_TEMPLATE = "plotly_dark"

PLOTLY_LAYOUT_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(
        family="DM Mono, monospace",
        color="#9ca3af",
        size=11
    ),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        bordercolor="rgba(255,255,255,0.05)"
    ),
)

SEGMENT_COLORS = {
    "Low Risk":  COLOR_EMERALD,
    "Mid Risk":  COLOR_AMBER,
    "High Risk": COLOR_RED,
    "Premium":   COLOR_BLUE,
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

def set_page_config(page_title: str = "CreditLens AI"):
    st.set_page_config(
        page_title=page_title,
        page_icon="💳",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_global_css()


# ─────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────

def _inject_global_css():
    st.markdown(
        """
        <link href="https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,600;1,9..144,300&display=swap" rel="stylesheet"/>

        <style>
        :root {
            --cl-bg: #0a0c10;
            --cl-surface: #111318;
            --cl-card: #161b22;
            --cl-border: #1e2430;

            --cl-emerald: #10b981;
            --cl-amber: #f59e0b;
            --cl-red: #ef4444;
            --cl-blue: #3b82f6;

            --cl-text: #e8eaf0;
            --cl-muted: #6b7280;
            --cl-label: #9ca3af;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--cl-bg) !important;
            color: var(--cl-text) !important;
            font-family: 'DM Mono', monospace !important;
        }

        [data-testid="stSidebar"] {
            background: var(--cl-surface) !important;
            border-right: 1px solid var(--cl-border) !important;
        }

        [data-testid="stSidebarNav"] a {
            font-family: 'DM Mono', monospace !important;
            font-size: 12px !important;
            color: var(--cl-muted) !important;
        }

        [data-testid="stSidebarNav"] a[aria-current="page"] {
            color: var(--cl-emerald) !important;
            border-left: 2px solid var(--cl-emerald);
            padding-left: 6px;
        }

        .sec-lbl {
            font-size: 10px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--cl-muted);
            border-bottom: 1px solid var(--cl-border);
            padding-bottom: 8px;
            margin: 1.5rem 0;
        }

        .kpi-card {
            background: var(--cl-card);
            border: 1px solid var(--cl-border);
            border-radius: 6px;
            padding: 1.1rem 1.25rem;
        }

        .kpi-label {
            font-size: 10px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--cl-muted);
        }

        .kpi-value {
            font-size: 24px;
            font-weight: 500;
            color: var(--cl-text);
        }

        .kpi-delta {
            font-size: 11px;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────

def render_kpi(label: str, value: str, delta: str = "", delta_positive: bool = True):
    delta_color = COLOR_EMERALD if delta_positive else COLOR_RED

    delta_html = (
        f'<div class="kpi-delta" style="color:{delta_color}">{delta}</div>'
        if delta else ""
    )

    st.markdown(
        f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label: str, status: str) -> str:
    cls_map = {
        "active": "pill-active",
        "warning": "pill-warning",
        "error": "pill-error",
        "neutral": "pill-neutral",
    }

    cls = cls_map.get(status, "pill-neutral")
    return f'<span class="status-pill {cls}">{label}</span>'


def render_section(title: str):
    st.markdown(f'<div class="sec-lbl">{title}</div>', unsafe_allow_html=True)


def plotly_layout(title: str = "", height: int = 360, **kwargs) -> dict:
    layout = {
        **PLOTLY_LAYOUT_BASE,
        "height": height
    }

    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=13, color="#9ca3af"),
            x=0,
        )

    layout.update(kwargs)

    return layout
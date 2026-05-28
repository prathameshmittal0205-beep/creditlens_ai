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
        page_icon="assets/logo.png",
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
            --cl-bg: #0a0b0e;
            --cl-surface: #11141a;
            --cl-card: #181b22;
            --cl-card-hover: #1f232a;
            --cl-border: #232832;

            --cl-emerald: #10b981;
            --cl-amber: #f59e0b;
            --cl-red: #ef4444;
            --cl-blue: #3b82f6;

            --cl-text: #e8eaf0;
            --cl-muted: #8b949e;
            --cl-label: #a3b3cc;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background: var(--cl-bg) !important;
            color: var(--cl-text) !important;
            font-family: 'DM Mono', monospace !important;
        }

        [data-testid="stSidebar"] {
            background: var(--cl-surface) !important;
            border-right: 1px solid var(--cl-border) !important;
            min-width: 320px !important;
            max-width: 380px !important;
        }

        @keyframes pulse-glow {
            0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.4); }
            70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
            100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
        }
        .status-dot-active {
            width: 8px; height: 8px; border-radius: 50%; background-color: var(--cl-emerald);
            animation: pulse-glow 2s infinite;
        }

        /* ── Fintech Sidebar Console ── */
        [data-testid="stSidebar"] {
            background: #0F1629 !important; /* Deep navy */
            border-right: 1px solid #1e293b !important;
        }
        .sb-header {
            margin-bottom: 24px;
        }
        .sb-title {
            font-size: 20px;
            font-weight: 700;
            color: #F1F5F9;
            letter-spacing: 0.5px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .sb-subtitle {
            font-size: 11px;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }
        .sb-section {
            background: rgba(30, 41, 59, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            backdrop-filter: blur(8px);
        }
        .sb-section-title {
            font-size: 11px;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 12px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 8px;
        }
        .sb-pill-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }
        .sb-pill-row:last-child {
            margin-bottom: 0;
        }
        .sb-pill-label {
            font-size: 12px;
            color: #cbd5e1;
            font-weight: 500;
        }
        .sb-pill-badge {
            font-size: 9px;
            font-weight: 700;
            padding: 4px 8px;
            border-radius: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .sb-badge-green { background: rgba(16,185,129,0.1); color: #10B981; border: 1px solid rgba(16,185,129,0.2); }
        .sb-badge-yellow { background: rgba(245,158,11,0.1); color: #F59E0B; border: 1px solid rgba(245,158,11,0.2); }
        .sb-badge-red { background: rgba(239,68,68,0.1); color: #EF4444; border: 1px solid rgba(239,68,68,0.2); }
        .sb-badge-gray { background: rgba(148,163,184,0.1); color: #94A3B8; border: 1px solid rgba(148,163,184,0.2); }
        
        .sb-health-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
        .sb-health-kpi { background: rgba(0,0,0,0.2); border-radius: 8px; padding: 12px; text-align: center; border: 1px solid rgba(255, 255, 255, 0.02); }
        .sb-health-val { font-size: 18px; font-weight: 700; color: #F1F5F9; margin-bottom: 2px; }
        .sb-health-lbl { font-size: 9px; color: #94A3B8; text-transform: uppercase; letter-spacing: 0.5px; }
        /* ── End Fintech Sidebar Console ── */

        .sidebar-header-box {
            display: flex;
            align-items: center;
            gap: 10px;
            padding-bottom: 1.5rem;
            border-bottom: 1px solid var(--cl-border);
            margin-bottom: 1.5rem;
        }
        
        .dataset-status-card {
            background: var(--cl-card);
            border: 1px solid var(--cl-border);
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.02);
        }
        
        .dataset-meta-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.75rem;
            margin-top: 0.75rem;
        }
        
        .mini-kpi-val {
            font-size: 14px;
            font-weight: 600;
            color: var(--cl-text);
        }
        .mini-kpi-lbl {
            font-size: 10px;
            color: var(--cl-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 0.5rem;
        }
        .telemetry-pill {
            background: rgba(255,255,255,0.03);
            border: 1px solid var(--cl-border);
            border-radius: 6px;
            padding: 0.5rem;
            font-size: 11px;
            display: flex;
            flex-direction: column;
            gap: 2px;
        }
        
        .sidebar-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid var(--cl-border);
            font-size: 10px;
            color: var(--cl-muted);
            text-align: center;
        }

        [data-testid="stSidebarNav"] a {
            font-family: 'DM Mono', monospace !important;
            font-size: 13px !important;
            color: var(--cl-muted) !important;
            transition: color 0.2s ease;
        }
        
        [data-testid="stSidebarNav"] a:hover {
            color: var(--cl-text) !important;
            background-color: transparent !important;
        }

        [data-testid="stSidebarNav"] a[aria-current="page"] {
            color: var(--cl-emerald) !important;
            border-left: 2px solid var(--cl-emerald);
            padding-left: 6px;
            font-weight: 500;
        }

        .sec-lbl {
            font-size: 11px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--cl-muted);
            border-bottom: 1px solid var(--cl-border);
            padding-bottom: 8px;
            margin: 2rem 0 1rem 0;
            display: flex;
            align-items: center;
        }

        .kpi-card {
            background: var(--cl-card);
            border: 1px solid var(--cl-border);
            border-radius: 8px;
            padding: 1.25rem 1.25rem;
            transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }
        
        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
            background: var(--cl-card-hover);
            border-color: rgba(16, 185, 129, 0.2);
        }

        .kpi-label {
            font-size: 10px;
            letter-spacing: 1.5px;
            text-transform: uppercase;
            color: var(--cl-muted);
            margin-bottom: 0.5rem;
        }

        .kpi-value {
            font-size: 26px;
            font-weight: 500;
            color: var(--cl-text);
            font-family: 'Fraunces', serif;
            margin-bottom: 0.25rem;
        }

        .kpi-delta {
            font-size: 11px;
            font-weight: 500;
        }
        
        /* ── Executive Insight Card ── */
        .exec-insight-card {
            background: linear-gradient(145deg, rgba(24,27,34,0.7) 0%, rgba(17,20,26,0.9) 100%);
            border: 1px solid rgba(16, 185, 129, 0.15);
            border-left: 3px solid var(--cl-emerald);
            border-radius: 8px;
            padding: 1.25rem 1.5rem;
            margin: 1rem 0 1.5rem 0;
            box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        }
        .exec-insight-header {
            font-size: 11px;
            letter-spacing: 2px;
            text-transform: uppercase;
            color: var(--cl-emerald);
            margin-bottom: 0.75rem;
            display: flex;
            align-items: center;
            gap: 6px;
            font-weight: 600;
        }
        .exec-insight-body {
            font-family: 'DM Mono', monospace;
            font-size: 13px;
            line-height: 1.6;
            color: #d1d5db;
        }
        .exec-insight-body strong {
            color: #fff;
            font-weight: 600;
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


def render_executive_insight(content: str):
    """Renders a sleek, fintech-grade insight card."""
    st.markdown(
        f"""
        <div class="exec-insight-card">
            <div class="exec-insight-header">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                AI Executive Insight
            </div>
            <div class="exec-insight-body">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_section(title: str):
    st.markdown(f'<div class="sec-lbl">{title}</div>', unsafe_allow_html=True)


def plotly_layout(title: str = "", height: int = 360, **kwargs) -> dict:
    layout = {
        **PLOTLY_LAYOUT_BASE,
        "height": height,
        "hovermode": "closest",
    }

    if title:
        layout["title"] = dict(
            text=title,
            font=dict(size=14, color="#e8eaf0", family="Fraunces, serif"),
            x=0.01,
            y=0.95
        )
    
    # Improve axis styling
    layout["xaxis"] = dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
    )
    layout["yaxis"] = dict(
        showgrid=True,
        gridcolor="rgba(255,255,255,0.05)",
        zeroline=False,
        showline=True,
        linecolor="rgba(255,255,255,0.1)",
    )

    layout.update(kwargs)

    return layout

def render_sb_header():
    st.sidebar.markdown(
        """
        <div class="sb-header">
            <div class="sb-title">
                CreditLens AI
                <div class="status-dot-active"></div>
            </div>
            <div class="sb-subtitle">Behavioral Credit Intelligence</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_sb_dataset_info(source: str = None, customers: int = 0):
    source = source or "Unknown Dataset"
    is_synth = source == "Synthetic Dataset"
    color_class = "sb-badge-gray" if is_synth else "sb-badge-green"
    
    st.sidebar.markdown(
        f"""
        <div class="sb-section" style="margin-top: -12px;">
            <div class="sb-pill-row">
                <span class="sb-pill-label">Rows Processed</span>
                <span class="sb-pill-badge {color_class}">{customers:,}</span>
            </div>
            <div class="sb-pill-row">
                <span class="sb-pill-label">Data Mode</span>
                <span class="sb-pill-badge {color_class}">{source}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_sb_pipeline_status(data_loaded: bool, clv_status: str, credit_status: str, shap_status: str):
    def get_badge(status):
        if status == "Active" or status == "Ready" or status == True:
            return "sb-badge-green", "ACTIVE"
        elif status == "Fallback":
            return "sb-badge-yellow", "FALLBACK"
        elif status == "Pending" or status == "Offline" or status == False:
            return "sb-badge-red", "OFFLINE"
        return "sb-badge-gray", str(status).upper()

    d_cls, d_lbl = get_badge(data_loaded)
    clv_cls, clv_lbl = get_badge(clv_status)
    c_cls, c_lbl = get_badge(credit_status)
    s_cls, s_lbl = get_badge(shap_status)

    st.sidebar.markdown(
        f"""
        <div class="sb-section">
            <div class="sb-section-title">Pipeline Status</div>
            <div class="sb-pill-row">
                <span class="sb-pill-label">Data Loaded</span>
                <span class="sb-pill-badge {d_cls}">{d_lbl}</span>
            </div>
            <div class="sb-pill-row">
                <span class="sb-pill-label">CLV Engine</span>
                <span class="sb-pill-badge {clv_cls}">{clv_lbl}</span>
            </div>
            <div class="sb-pill-row">
                <span class="sb-pill-label">Credit Model</span>
                <span class="sb-pill-badge {c_cls}">{c_lbl}</span>
            </div>
            <div class="sb-pill-row">
                <span class="sb-pill-label">SHAP Explainer</span>
                <span class="sb-pill-badge {s_cls}">{s_lbl}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_sb_model_health(active_models: int, seg_status: str, exp_status: str):
    st.sidebar.markdown(
        f"""
        <div class="sb-section">
            <div class="sb-section-title">Model Health</div>
            <div class="sb-health-grid">
                <div class="sb-health-kpi">
                    <div class="sb-health-val">{active_models}</div>
                    <div class="sb-health-lbl">Models Active</div>
                </div>
                <div class="sb-health-kpi">
                    <div class="sb-health-val">{seg_status}</div>
                    <div class="sb-health-lbl">Segmentation</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_sb_footer():
    st.sidebar.markdown(
        """
        <div style="font-size: 10px; color: #64748b; text-align: center; margin-top: 24px;">
            v2.4.0 • Built with Streamlit
        </div>
        """,
        unsafe_allow_html=True
    )
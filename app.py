"""
app.py — Invest Infinity Dashboard
====================================
Run: streamlit run app.py

Architecture:
  - Step 1  : Fetch live data (yfinance)
  - Step 2  : Train LSTM + GRU for each selected company
  - Step 3  : Evaluate on strict time-based test split
  - Step 4  : User selects best model per company
  - Step 5  : Portfolio optimisation + future forecast

Bug fixes vs previous version:
  - BUG FIX: add_vline crash — replaced with add_shape + add_annotation (works all Plotly versions)
  - BUG FIX: Strict time-based split (no random shuffle)
  - BUG FIX: Scaler fitted only on training data
  - BUG FIX: Test metrics computed only on out-of-sample test period
"""

import os, sys, warnings, pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from utils.data_fetcher import (
    fetch_stock_data, time_split, STOCK_UNIVERSE, SORTED_TICKERS
)
from model.trainer import train_model, evaluate_model, forecast_future
from model.portfolio_optimizer import (
    build_price_matrix, optimise, backtest
)

MODEL_DIR    = os.path.join(BASE, "models")
FALLBACK_CSV = os.path.join(BASE, "data", "stocks_data.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Translations
# ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "app_title":     "Invest Infinity",
        "tagline":       "AI-powered portfolio optimization using LSTM & GRU",
        "fetch":         "Fetch Data",
        "train":         "Train Models",
        "run":           "Run Analysis",
        "tab_home":      "Home",
        "tab_model_sel": "Model Selection",
        "tab_forecast":  "Forecast",
        "tab_portfolio": "Portfolio",
        "tab_perf":      "Performance",
        "tab_about":     "About",
        "period_opts":   ["1y","2y","3y","5y","10y"],
        "login":         "Login / Sign Up",
    },
    "हिन्दी": {
        "app_title":     "Invest Infinity",
        "tagline":       "LSTM और GRU द्वारा AI-संचालित पोर्टफोलियो अनुकूलन",
        "fetch":         "डेटा प्राप्त करें",
        "train":         "मॉडल प्रशिक्षित करें",
        "run":           "विश्लेषण चलाएं",
        "tab_home":      "होम",
        "tab_model_sel": "मॉडल चयन",
        "tab_forecast":  "पूर्वानुमान",
        "tab_portfolio": "पोर्टफोलियो",
        "tab_perf":      "प्रदर्शन",
        "tab_about":     "जानकारी",
        "period_opts":   ["1y","2y","3y","5y","10y"],
        "login":         "लॉगिन / साइन अप",
    },
    "Français": {
        "app_title":     "Invest Infinity",
        "tagline":       "Optimisation de portefeuille par IA (LSTM & GRU)",
        "fetch":         "Récupérer les données",
        "train":         "Entraîner les modèles",
        "run":           "Lancer l'analyse",
        "tab_home":      "Accueil",
        "tab_model_sel": "Sélection du modèle",
        "tab_forecast":  "Prévision",
        "tab_portfolio": "Portefeuille",
        "tab_perf":      "Performance",
        "tab_about":     "À propos",
        "period_opts":   ["1y","2y","3y","5y","10y"],
        "login":         "Connexion / Inscription",
    },
}

def T(key):
    return TRANSLATIONS.get(
        st.session_state.get("language","English"),
        TRANSLATIONS["English"]
    ).get(key, key)

# ──────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Invest Infinity",
    page_icon="assets/logo.png" if os.path.exists("assets/logo.png") else "📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── White background professional CSS ─────────────────────────
st.markdown("""
<style>
/* ── Global white background, black text ── */
html, body, [data-testid="stApp"] {
    background-color: #ffffff !important;
    color: #111111 !important;
}
[data-testid="stHeader"] { background-color: #ffffff !important; }
[data-testid="stSidebar"] { display: none; }

/* ── All standard text elements black on white ── */
p, span, div, label, h1, h2, h3, h4, h5, h6, li, td, th, caption {
    color: #111111 !important;
}

/* ── Streamlit native text widgets ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong,
[data-testid="stMarkdownContainer"] em,
[data-testid="stMarkdownContainer"] code,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] h4 {
    color: #111111 !important;
}

/* ── Metric widget ── */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: #111111 !important;
}

/* ── Dataframe / table text ── */
[data-testid="stDataFrame"] td,
[data-testid="stDataFrame"] th,
.stDataFrame td, .stDataFrame th {
    color: #111111 !important;
}

/* ── Caption / info / warning text ── */
[data-testid="stCaptionContainer"],
[data-testid="stCaption"],
small { color: #555555 !important; }

/* ── Radio buttons ── */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span,
[data-testid="stRadio"] div { color: #111111 !important; }

/* ── Selectbox / multiselect text ── */
[data-testid="stSelectbox"] label,
[data-testid="stMultiSelect"] label,
[data-testid="stSelectbox"] span,
[data-testid="stMultiSelect"] span { color: #111111 !important; }

/* ── Slider label + value ── */
[data-testid="stSlider"] label,
[data-testid="stSlider"] span,
[data-testid="stSlider"] p { color: #111111 !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] label,
[data-testid="stNumberInput"] span { color: #111111 !important; }

/* ── Expander header ── */
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span { color: #111111 !important; }

/* ── Info / success / warning / error boxes ── */
[data-testid="stAlert"] p,
[data-testid="stAlert"] span { color: #111111 !important; }

/* ── Progress bar label ── */
[data-testid="stProgress"] p { color: #111111 !important; }

/* ── Top nav bar (dark bg → white text) ── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #111111;
    padding: 0 32px;
    height: 56px;
    position: sticky;
    top: 0;
    z-index: 999;
    border-radius: 0 0 12px 12px;
}
.nav-logo {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff !important;
    letter-spacing: .03em;
}
.nav-links { display: flex; gap: 6px; }
.nav-btn {
    background: transparent;
    color: #cccccc !important;
    border: none;
    padding: 6px 18px;
    font-size: 14px;
    cursor: pointer;
    border-radius: 6px;
    transition: background .15s;
}
.nav-btn:hover, .nav-btn.active {
    background: #333333;
    color: #ffffff !important;
}
.nav-login {
    background: #ffffff;
    color: #111111 !important;
    border: none;
    padding: 6px 18px;
    font-size: 13px;
    border-radius: 6px;
    font-weight: 600;
    cursor: pointer;
}

/* ── Streamlit tab override (dark bg → white/grey text) ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #111111 !important;
    border-radius: 10px !important;
    padding: 4px 8px !important;
    gap: 4px !important;
    justify-content: space-evenly !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: #aaaaaa !important;
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    flex: 1 !important;
    min-width: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] p,
[data-testid="stTabs"] button[role="tab"] span {
    color: inherit !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: #333333 !important;
    color: #ffffff !important;
}
[data-testid="stTabs"] [data-testid="stTabContent"] {
    padding-top: 24px !important;
}

/* ── Black action buttons (dark bg → white text) ── */
div.stButton > button {
    background-color: #111111 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    width: 100% !important;
    letter-spacing: .02em !important;
    transition: background .15s !important;
}
div.stButton > button:hover { background-color: #333333 !important; }
div.stButton > button[kind="primary"] { background-color: #111111 !important; }
div.stButton > button p,
div.stButton > button span { color: #ffffff !important; }

/* ── Metric cards (light bg → black text) ── */
.metric-card {
    background: #f5f5f5;
    border: 1px solid #e0e0e0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .mc-label { font-size: 11px; color: #666666 !important;
    text-transform: uppercase; letter-spacing: .06em; margin-bottom: 4px; }
.metric-card .mc-value { font-size: 22px; font-weight: 700; color: #111111 !important; }
.metric-card.good .mc-value { color: #1a7a4a !important; }
.metric-card.warn .mc-value { color: #b34000 !important; }
.metric-card.bad  .mc-value { color: #c0392b !important; }

/* ── Recommended badge (green bg → white text) ── */
.rec-badge {
    display: inline-block;
    background: #1a7a4a;
    color: #ffffff !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: .06em;
    border-radius: 4px;
    padding: 2px 8px;
    margin-left: 8px;
    vertical-align: middle;
    text-transform: uppercase;
}
.better-badge {
    display: inline-block;
    background: #b34000;
    color: #ffffff !important;
    font-size: 11px;
    border-radius: 4px;
    padding: 2px 8px;
    margin-left: 8px;
    vertical-align: middle;
}

/* ── Section header (white bg → black text) ── */
.sec-h {
    font-size: 16px;
    font-weight: 700;
    color: #111111 !important;
    border-bottom: 2px solid #111111;
    padding-bottom: 6px;
    margin: 24px 0 14px 0;
}

/* ── Right panel (light bg → black text) ── */
.right-panel {
    background: #f9f9f9;
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 24px;
    color: #111111 !important;
}
.right-panel p, .right-panel span, .right-panel label,
.right-panel h1, .right-panel h2, .right-panel h3,
.right-panel h4, .right-panel div { color: #111111 !important; }

/* ── Stock badge (dark bg → white text) ── */
.sbadge {
    display: inline-block;
    background: #111111;
    color: #ffffff !important;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
}

/* ── Section divider ── */
hr { border: none; border-top: 1px solid #e5e5e5; margin: 20px 0; }

/* ── Remove Streamlit watermark ── */
#MainMenu, footer { visibility: hidden; }

/* ── Input / select styling ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: #ffffff !important;
    border: 1px solid #cccccc !important;
    border-radius: 6px !important;
    color: #111111 !important;
}

/* ── Dropdown options text ── */
[data-testid="stSelectbox"] option,
[data-baseweb="select"] [data-testid="stText"],
[data-baseweb="select"] span { color: #111111 !important; }

/* ── Multiselect tags ── */
[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    background-color: #111111 !important;
}
[data-testid="stMultiSelect"] [data-baseweb="tag"] span {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session state
# ──────────────────────────────────────────────────────────────
_defaults = {
    "language":         "English",
    "selected_stocks":  ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "data_period":      "5y",
    "investment":       100_000.0,
    "risk_free_rate":   0.05,
    "stock_data":       None,
    "data_source":      None,
    "train_cut":        None,
    "test_start":       None,
    "train_data":       {},     # {company: train_df}
    "test_data":        {},     # {company: test_df}
    "train_results":    {},     # {company: {LSTM: {...}, GRU: {...}}}
    "eval_results":     {},     # {company: {LSTM: {...}, GRU: {...}}}
    "selected_models":  {},     # {company: "LSTM" or "GRU"}
    "forecast_results": {},     # {company: forecast dict}
    "opt_results":      None,
    "price_matrix":     None,
    "logged_in":        False,
    "username":         "",
    "saved_analyses":   [],
    "active_tab":       0,
    "forecast_horizon": 30,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def ts_to_str(ts) -> str:
    """Convert any Timestamp/date to ISO string — fixes add_vline bug."""
    if ts is None:
        return ""
    if isinstance(ts, str):
        return ts
    return pd.to_datetime(ts).strftime("%Y-%m-%d")


def metric_card(label, value, kind=""):
    return (
        f'<div class="metric-card {kind}">'
        f'<div class="mc-label">{label}</div>'
        f'<div class="mc-value">{value}</div>'
        f'</div>'
    )


def recommend(eval_lstm, eval_gru) -> str:
    """Return 'LSTM' or 'GRU' based on lower RMSE (test period)."""
    return "LSTM" if eval_lstm["rmse"] <= eval_gru["rmse"] else "GRU"


# ──────────────────────────────────────────────────────────────
# MAIN TABS
# ──────────────────────────────────────────────────────────────
tab_labels = [
    T("tab_home"),
    T("tab_model_sel"),
    T("tab_forecast"),
    T("tab_portfolio"),
    T("tab_perf"),
    T("tab_about"),
]
tabs = st.tabs(tab_labels)

# ════════════════════════════════════════════════════════════════
# TAB 0 — HOME
# ════════════════════════════════════════════════════════════════
with tabs[0]:
    # ── Header row ──
    hdr_left, hdr_right = st.columns([2, 1])
    with hdr_left:
        st.markdown(
            "<h1 style='font-size:32px;font-weight:800;color:#111;margin:0'>"
            "Invest Infinity</h1>"
            "<p style='color:#555;font-size:15px;margin:4px 0 0 0'>"
            "AI-powered portfolio optimization using LSTM &amp; GRU</p>",
            unsafe_allow_html=True
        )
    with hdr_right:
        lang = st.selectbox(
            "Language",
            list(TRANSLATIONS.keys()),
            index=list(TRANSLATIONS.keys()).index(
                st.session_state.get("language","English")
            ),
            label_visibility="collapsed"
        )
        st.session_state.language = lang
        if st.session_state.logged_in:
            st.success(f"Logged in: {st.session_state.username}")
        else:
            if st.button("Login / Sign Up", key="login_top"):
                st.session_state.show_login = True

    if st.session_state.get("show_login"):
        with st.expander("Account", expanded=True):
            c1, c2, c3 = st.columns([1,1,1])
            with c1:
                uname = st.text_input("Username")
            with c2:
                pwd   = st.text_input("Password", type="password")
            with c3:
                st.write("")
                st.write("")
                if st.button("Login"):
                    if uname and pwd:
                        st.session_state.logged_in = True
                        st.session_state.username  = uname
                        st.session_state.show_login = False
                        st.success(f"Welcome, {uname}!")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Two-column layout: How it works (left) | Control panel (right) ──
    left_col, right_col = st.columns([1.2, 1])

    with left_col:
        st.markdown('<div class="sec-h">How it works</div>', unsafe_allow_html=True)
        st.markdown("""
**Step 1 — Select stocks** from the panel on the right (5–10 recommended).  
**Step 2 — Fetch Data** — live OHLCV prices downloaded from Yahoo Finance.  
**Step 3 — Train Models** — one LSTM and one GRU trained per company,  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
using data up to the training cutoff date.  
**Step 4 — Model Selection tab** — compare LSTM vs GRU metrics for each  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
company, pick the best (or accept the recommendation).  
**Step 5 — Run Analysis** — forecast + portfolio optimisation using your  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
chosen models.
        """)

        if st.session_state.stock_data is not None:
            df = st.session_state.stock_data
            st.markdown('<div class="sec-h">Data loaded</div>', unsafe_allow_html=True)
            src = st.session_state.data_source
            tc  = ts_to_str(st.session_state.train_cut)
            ts  = ts_to_str(st.session_state.test_start)
            st.markdown(
                f"Source: **{src}** &nbsp;|&nbsp; "
                f"{len(df):,} rows &nbsp;|&nbsp; "
                f"{df['Date'].min().date()} → {df['Date'].max().date()}  \n"
                f"Train cutoff: **{tc}** &nbsp;|&nbsp; Test from: **{ts}**"
            )
            st.dataframe(
                df.sort_values(["Company","Date"]).tail(20),
                use_container_width=True, hide_index=True
            )

    with right_col:
        st.markdown('<div class="right-panel">', unsafe_allow_html=True)
        st.markdown('<div class="sec-h">Control Panel</div>', unsafe_allow_html=True)

        # Stock picker — trending first with tag
        label_to_ticker = {}
        for tk in SORTED_TICKERS:
            info  = STOCK_UNIVERSE[tk]
            lbl   = f"{tk} {'[Trending]' if info['trending'] else ''} — {info['name']}"
            label_to_ticker[lbl] = tk

        default_labels = [
            l for l, tk in label_to_ticker.items()
            if tk in st.session_state.selected_stocks
        ]
        chosen = st.multiselect(
            "Select stocks (5–10 recommended)",
            options=list(label_to_ticker.keys()),
            default=default_labels
        )
        st.session_state.selected_stocks = [label_to_ticker[l] for l in chosen]
        if len(st.session_state.selected_stocks) < 2:
            st.warning("Select at least 2 stocks.")

        # Period + investment
        period_opts = T("period_opts")
        p_idx = period_opts.index(st.session_state.data_period) \
                if st.session_state.data_period in period_opts else 3
        st.session_state.data_period = st.selectbox(
            "Historical data period", period_opts, index=p_idx
        )
        st.session_state.investment = st.number_input(
            "Investment amount ($)",
            min_value=1_000.0, max_value=10_000_000.0,
            value=st.session_state.investment, step=1_000.0, format="%.0f"
        )
        rfr = st.slider(
            "Risk-free rate (%)", 0.0, 10.0,
            value=st.session_state.risk_free_rate * 100, step=0.25
        )
        st.session_state.risk_free_rate = rfr / 100.0

        st.markdown("---")

        # ── Step 1: Fetch ──
        if st.button(T("fetch"), key="btn_fetch"):
            if not st.session_state.selected_stocks:
                st.error("Select stocks first.")
            else:
                bar = st.progress(0, text="Connecting to Yahoo Finance…")
                try:
                    df, src, train_cut, test_start = fetch_stock_data(
                        st.session_state.selected_stocks,
                        period=st.session_state.data_period,
                        fallback_csv=FALLBACK_CSV
                    )
                    bar.progress(50, text="Processing data…")
                    st.session_state.stock_data  = df
                    st.session_state.data_source = src
                    st.session_state.train_cut   = train_cut
                    st.session_state.test_start  = test_start

                    # Pre-split per company
                    train_data, test_data = {}, {}
                    for c in df["Company"].unique():
                        df_c = df[df["Company"] == c].copy()
                        tr, te = time_split(df_c, train_cut, test_start)
                        if len(tr) > 80 and len(te) > 10:
                            train_data[c] = tr
                            test_data[c]  = te
                    st.session_state.train_data = train_data
                    st.session_state.test_data  = test_data
                    bar.progress(100, text="Done!")
                    st.success(
                        f"Fetched via **{src}** — "
                        f"{len(df):,} rows · {df['Company'].nunique()} tickers  \n"
                        f"Train: up to {ts_to_str(train_cut)} · "
                        f"Test: from {ts_to_str(test_start)}"
                    )
                except Exception as e:
                    bar.empty()
                    st.error(f"Fetch error: {e}")

        # ── Step 2: Train ──
        if st.button(T("train"), key="btn_train"):
            if not st.session_state.train_data:
                st.error("Fetch data first.")
            else:
                train_data = st.session_state.train_data
                companies  = list(train_data.keys())
                total      = len(companies) * 2   # LSTM + GRU each
                done       = 0
                bar        = st.progress(0, text="Starting training…")
                train_results = {}

                for company in companies:
                    train_results[company] = {}
                    for mtype in ["LSTM", "GRU"]:
                        bar.progress(done / total,
                                     text=f"Training {company} / {mtype}…")
                        try:
                            res = train_model(
                                train_data[company], company, mtype,
                                model_dir=MODEL_DIR, verbose=0
                            )
                            train_results[company][mtype] = res
                            st.success(
                                f"{company} / {mtype} — "
                                f"val_loss: {res['val_loss']:.5f} | "
                                f"{res['epochs_run']} epochs"
                            )
                        except Exception as e:
                            st.error(f"{company}/{mtype}: {e}")
                        done += 1

                bar.progress(1.0, text="Training complete!")
                st.session_state.train_results = train_results

                # Auto-evaluate on test set
                bar2 = st.progress(0, text="Evaluating on test data…")
                eval_results = {}
                test_data    = st.session_state.test_data
                for i, company in enumerate(companies):
                    eval_results[company] = {}
                    for mtype in ["LSTM", "GRU"]:
                        if mtype in train_results.get(company, {}):
                            try:
                                er = evaluate_model(
                                    test_data[company],
                                    train_data[company],
                                    company, mtype,
                                    model_dir=MODEL_DIR
                                )
                                eval_results[company][mtype] = er
                            except Exception as e:
                                st.error(f"Eval {company}/{mtype}: {e}")
                    bar2.progress((i + 1) / len(companies))

                bar2.progress(1.0, text="Evaluation complete!")
                st.session_state.eval_results = eval_results

                # Auto-assign recommended models
                sel = {}
                for c, models in eval_results.items():
                    if "LSTM" in models and "GRU" in models:
                        sel[c] = recommend(models["LSTM"], models["GRU"])
                    elif models:
                        sel[c] = list(models.keys())[0]
                st.session_state.selected_models = sel
                st.info("Go to the **Model Selection** tab to review and confirm model choices.")

        # ── Step 3: Run Analysis ──
        if st.button(T("run"), key="btn_run", type="primary"):
            if not st.session_state.selected_models:
                st.error("Train models and select one per company first.")
            elif not st.session_state.stock_data is not None:
                st.error("Fetch data first.")
            else:
                df         = st.session_state.stock_data
                sel_models = st.session_state.selected_models
                train_data = st.session_state.train_data
                companies  = list(sel_models.keys())
                bar        = st.progress(0, text="Forecasting…")

                forecast_results = {}
                for i, company in enumerate(companies):
                    mtype  = sel_models[company]
                    df_c   = df[df["Company"] == company].copy()
                    bar.progress(i / len(companies), text=f"Forecasting {company}…")
                    try:
                        fr = forecast_future(
                            df_c, company, mtype,
                            model_dir=MODEL_DIR,
                            n_days=st.session_state.forecast_horizon
                        )
                        forecast_results[company] = fr
                    except Exception as e:
                        st.error(f"{company}: {e}")

                bar.progress(0.8, text="Optimising portfolio…")
                if forecast_results:
                    st.session_state.forecast_results = forecast_results
                    pm = build_price_matrix(df, list(forecast_results.keys()))
                    st.session_state.price_matrix = pm
                    try:
                        opt = optimise(
                            forecast_results, pm,
                            investment_amount=st.session_state.investment,
                            risk_free_rate=st.session_state.risk_free_rate
                        )
                        st.session_state.opt_results = opt
                        bar.progress(1.0, text="Analysis complete!")
                        st.success("Analysis complete! See Forecast and Portfolio tabs.")
                        if opt.get("rfr_was_adjusted"):
                            eff = opt["effective_rfr"] * 100
                            st.warning(
                                f"\u26a0\ufe0f **Risk-free rate auto-adjusted to {eff:.2f}%** — "
                                f"forecast returns were all below your set rate "
                                f"({st.session_state.risk_free_rate*100:.2f}%). "
                                f"Portfolio is still optimised. Consider lowering the "
                                f"risk-free rate slider."
                            )
                    except Exception as e:
                        st.error(f"Optimisation failed: {e}")

        st.markdown('</div>', unsafe_allow_html=True)

        # Forecast horizon slider
        st.session_state.forecast_horizon = st.slider(
            "Future forecast horizon (days)", 7, 90,
            value=st.session_state.forecast_horizon, step=1
        )


# ════════════════════════════════════════════════════════════════
# TAB 1 — MODEL SELECTION
# ════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<h2 style="color:#111">Model Selection</h2>', unsafe_allow_html=True)
    st.markdown(
        "Review LSTM vs GRU performance for each company. "
        "The dashboard recommends the better model, but you can override."
    )

    er = st.session_state.eval_results
    if not er:
        st.info("Train models first (Home tab → Train Models).")
    else:
        for company, models in er.items():
            if not models:
                continue

            lstm_e = models.get("LSTM")
            gru_e  = models.get("GRU")

            # Determine recommended
            rec = None
            if lstm_e and gru_e:
                rec = recommend(lstm_e, gru_e)
            elif lstm_e:
                rec = "LSTM"
            elif gru_e:
                rec = "GRU"

            info = STOCK_UNIVERSE.get(company, {})
            with st.expander(
                f"{company} — {info.get('name', company)}",
                expanded=True
            ):
                # ── Side-by-side metric comparison ──
                col_lstm, col_gru = st.columns(2)

                for col, mtype, e in [
                    (col_lstm, "LSTM", lstm_e),
                    (col_gru,  "GRU",  gru_e)
                ]:
                    with col:
                        rec_html = (
                            '<span class="rec-badge">Recommended</span>'
                            if mtype == rec else ""
                        )
                        st.markdown(
                            f"<h4 style='color:#111;margin:0'>{mtype} {rec_html}</h4>",
                            unsafe_allow_html=True
                        )
                        if e is None:
                            st.warning("Not trained.")
                            continue

                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("RMSE", f"{e['rmse']:.3f}")
                        m2.metric("MSE",  f"{e['mse']:.3f}")
                        m3.metric("MAE",  f"{e['mae']:.3f}")
                        m4.metric("MAPE", f"{e['mape']:.2f}%")

                        # ── Actual vs Predicted chart ──
                        # BUG FIX: convert dates to strings before plotting
                        # (avoids Plotly Timestamp arithmetic crash)
                        dates_str  = [ts_to_str(d) for d in e["actual_dates"]]

                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=dates_str,
                            y=e["actual_prices"],
                            name="Actual Prices",
                            line=dict(color="#1565C0", width=1.8)
                        ))
                        fig.add_trace(go.Scatter(
                            x=dates_str,
                            y=e["predicted_prices"],
                            name="Predicted Prices",
                            line=dict(color="#C62828", width=1.8, dash="dot")
                        ))
                        fig.update_layout(
                            title=dict(
                                text=f"{company} — {mtype} Stock Price Prediction",
                                font=dict(size=13, color="#111")
                            ),
                            template="plotly_white",
                            height=300,
                            margin=dict(l=0, r=0, t=36, b=0),
                            xaxis_title="Date",
                            yaxis_title="Stock Price ($)",
                            legend=dict(orientation="h", y=-0.3),
                            paper_bgcolor="#ffffff",
                            plot_bgcolor="#ffffff",
                            font=dict(color="#111")
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # ── Model selector for this company ──
                current = st.session_state.selected_models.get(company, rec)
                choices = [m for m in ["LSTM", "GRU"] if m in models]

                chosen_model = st.radio(
                    f"Use for {company}:",
                    choices,
                    index=choices.index(current) if current in choices else 0,
                    horizontal=True,
                    key=f"sel_{company}"
                )
                st.session_state.selected_models[company] = chosen_model
                st.caption(
                    f"Selected: **{chosen_model}** for {company}  "
                    f"{'(matches recommendation)' if chosen_model == rec else '(manual override)'}"
                )


# ════════════════════════════════════════════════════════════════
# TAB 2 — FORECAST
# ════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<h2 style="color:#111">Future Price Forecast</h2>',
                unsafe_allow_html=True)

    fr = st.session_state.forecast_results
    er = st.session_state.eval_results
    sel = st.session_state.selected_models

    if not fr:
        st.info("Run Analysis first (Home tab).")
    else:
        for company, res in fr.items():
            mtype = sel.get(company, "LSTM")
            info  = STOCK_UNIVERSE.get(company, {})
            with st.expander(
                f"{company} — {info.get('name', company)} | Model: {mtype}",
                expanded=True
            ):
                # KPI row
                pct = (res["forecast_prices"][-1] - res["last_known_price"]) \
                      / res["last_known_price"] * 100
                c1, c2, c3 = st.columns(3)
                c1.metric("Last known price",   f"${res['last_known_price']:.2f}")
                c2.metric(f"Day-{len(res['forecast_prices'])} forecast",
                          f"${res['forecast_prices'][-1]:.2f}")
                c3.metric("Expected change", f"{pct:+.1f}%", delta_color="normal")

                # BUG FIX: convert all dates to plain strings
                last_date_str   = ts_to_str(res["last_known_date"])
                forecast_dates  = [ts_to_str(d) for d in res["forecast_dates"]]

                # Include eval test-period actual+pred if available
                fig = go.Figure()
                e = er.get(company, {}).get(mtype)
                if e:
                    test_dates_str = [ts_to_str(d) for d in e["actual_dates"]]
                    fig.add_trace(go.Scatter(
                        x=test_dates_str,
                        y=e["actual_prices"],
                        name="Actual (test period)",
                        line=dict(color="#1565C0", width=1.8)
                    ))
                    fig.add_trace(go.Scatter(
                        x=test_dates_str,
                        y=e["predicted_prices"],
                        name=f"{mtype} on test data",
                        line=dict(color="#C62828", width=1.5, dash="dot")
                    ))

                # Forecast band ±2%
                fig.add_trace(go.Scatter(
                    x=forecast_dates + forecast_dates[::-1],
                    y=[p * 1.02 for p in res["forecast_prices"]] +
                      [p * 0.98 for p in res["forecast_prices"][::-1]],
                    fill="toself",
                    fillcolor="rgba(39,174,96,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="±2% confidence band"
                ))
                fig.add_trace(go.Scatter(
                    x=forecast_dates,
                    y=res["forecast_prices"],
                    name="Future forecast",
                    line=dict(color="#27ae60", width=2.2)
                ))

                # FIX: add_vline crashes across Plotly versions when x is a
                # date string (TypeError: sum([str])). Use add_shape +
                # add_annotation instead — both accept plain ISO strings safely.
                if forecast_dates:
                    fig.add_shape(
                        type="line",
                        x0=forecast_dates[0], x1=forecast_dates[0],
                        y0=0, y1=1,
                        xref="x", yref="paper",
                        line=dict(dash="dash", color="rgba(0,0,0,0.30)", width=1.5)
                    )
                    fig.add_annotation(
                        x=forecast_dates[0], y=1,
                        xref="x", yref="paper",
                        text="Forecast start",
                        showarrow=False,
                        xanchor="left",
                        yanchor="bottom",
                        font=dict(size=11, color="#111111"),
                        bgcolor="rgba(255,255,255,0.7)",
                        borderpad=3
                    )

                fig.update_layout(
                    title=dict(
                        text=f"{company} Stock Price Prediction",
                        font=dict(size=14, color="#111")
                    ),
                    template="plotly_white",
                    height=400,
                    margin=dict(l=0, r=0, t=36, b=0),
                    xaxis_title="Date",
                    yaxis_title="Stock Price ($)",
                    legend=dict(orientation="h", y=-0.25),
                    paper_bgcolor="#ffffff",
                    plot_bgcolor="#ffffff",
                    font=dict(color="#111")
                )
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<h2 style="color:#111">Portfolio Optimisation</h2>',
                unsafe_allow_html=True)

    opt = st.session_state.opt_results
    if opt is None:
        st.info("Run Analysis first.")
    else:
        if opt.get("rfr_was_adjusted"):
            eff = opt["effective_rfr"] * 100
            st.warning(
                f"\u26a0\ufe0f **Note:** Risk-free rate was auto-adjusted to **{eff:.2f}%** "
                f"because the model's forecast returns were all below your configured rate. "
                f"Sharpe ratios below are computed at the adjusted rate."
            )
        strategy = st.radio(
            "Strategy",
            ["Max Sharpe", "Min Volatility", "Equal Weight"],
            horizontal=True
        )
        key_map = {
            "Max Sharpe":    ("max_sharpe_weights", "max_sharpe"),
            "Min Volatility":("min_vol_weights",    "min_vol"),
            "Equal Weight":  ("equal_weights",      "equal"),
        }
        w_key, p_key = key_map[strategy]
        weights = opt[w_key]
        perf    = opt["performance"][p_key]
        risk    = opt["risk_metrics"]

        # KPI strip
        kpi_cols = st.columns(6)
        kpis = [
            ("Sharpe Ratio",     f"{perf['sharpe_ratio']:.3f}",         "good" if perf["sharpe_ratio"] > 1 else ""),
            ("Annual Return",    f"{perf['expected_return']*100:.1f}%",  "good" if perf["expected_return"] > 0 else "bad"),
            ("Annual Volatility",f"{perf['volatility']*100:.1f}%",       ""),
            ("VaR 95%",          f"{risk['var_95']*100:.2f}%",           "warn"),
            ("CVaR 95%",         f"{risk['cvar_95']*100:.2f}%",          "bad"),
            ("Max Drawdown",     f"{risk['max_drawdown']*100:.1f}%",     "bad"),
        ]
        for col, (lbl, val, kind) in zip(kpi_cols, kpis):
            col.markdown(metric_card(lbl, val, kind), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        col_donut, col_table = st.columns([1.1, 1])

        with col_donut:
            st.markdown('<div class="sec-h">Optimal Allocation</div>',
                        unsafe_allow_html=True)
            labels  = [k for k, v in weights.items() if v > 0.001]
            values  = [weights[k] for k in labels]
            amounts = [v * st.session_state.investment for v in values]
            fig_d   = go.Figure(go.Pie(
                labels=labels, values=values, hole=0.55,
                marker=dict(colors=px.colors.qualitative.Dark2[:len(labels)]),
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Weight: %{percent}<br>"
                              "Amount: $%{customdata:,.0f}<extra></extra>",
                customdata=amounts
            ))
            fig_d.add_annotation(
                text=f"${st.session_state.investment/1000:.0f}K",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=22, color="#111")
            )
            fig_d.update_layout(
                template="plotly_white", height=320,
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#ffffff",
                font=dict(color="#111")
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with col_table:
            st.markdown('<div class="sec-h">Shares to Buy</div>',
                        unsafe_allow_html=True)
            pm = st.session_state.price_matrix
            if w_key == "max_sharpe_weights" and opt.get("discrete_allocation") and pm is not None:
                alloc = opt["discrete_allocation"]
                rows  = []
                for c, n_sh in alloc.items():
                    price = float(pm[c].iloc[-1]) if c in pm.columns else 0
                    rows.append({
                        "Ticker": c,
                        "Weight": f"{weights.get(c,0)*100:.1f}%",
                        "Shares": n_sh,
                        "Price":  f"${price:.2f}",
                        "Amount": f"${n_sh*price:,.0f}"
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
                st.metric("Leftover cash", f"${opt['leftover']:.2f}")
            else:
                rows = [
                    {"Ticker": c, "Weight": f"{w*100:.1f}%",
                     "Amount": f"${w*st.session_state.investment:,.0f}"}
                    for c, w in weights.items() if w > 0.001
                ]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Correlation heatmap
        st.markdown('<div class="sec-h">Return Correlation Matrix</div>',
                    unsafe_allow_html=True)
        corr = opt["correlation_matrix"]
        fig_h = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, text_auto=".2f",
            template="plotly_white"
        )
        fig_h.update_layout(
            height=320, margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="#ffffff", font=dict(color="#111")
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ════════════════════════════════════════════════════════════════
# TAB 4 — PERFORMANCE
# ════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<h2 style="color:#111">Portfolio Performance</h2>',
                unsafe_allow_html=True)

    opt = st.session_state.opt_results
    pm  = st.session_state.price_matrix

    if opt is None or pm is None:
        st.info("Run Analysis first.")
    else:
        col_ef, col_bt = st.columns(2)

        with col_ef:
            st.markdown('<div class="sec-h">Efficient Frontier</div>',
                        unsafe_allow_html=True)
            fd   = opt["frontier_data"]
            ms_p = opt["performance"]["max_sharpe"]
            mv_p = opt["performance"]["min_vol"]
            fig_ef = go.Figure()
            if fd["volatilities"]:
                fig_ef.add_trace(go.Scatter(
                    x=[v*100 for v in fd["volatilities"]],
                    y=[r*100 for r in fd["returns"]],
                    mode="lines", name="Efficient Frontier",
                    line=dict(color="#1565C0", width=2)
                ))
            fig_ef.add_trace(go.Scatter(
                x=[ms_p["volatility"]*100], y=[ms_p["expected_return"]*100],
                mode="markers+text", text=["Max Sharpe"],
                textposition="top right",
                marker=dict(size=14, color="#e65100", symbol="star"),
                name="Max Sharpe"
            ))
            fig_ef.add_trace(go.Scatter(
                x=[mv_p["volatility"]*100], y=[mv_p["expected_return"]*100],
                mode="markers+text", text=["Min Vol"],
                textposition="top right",
                marker=dict(size=12, color="#1a7a4a", symbol="diamond"),
                name="Min Volatility"
            ))
            fig_ef.update_layout(
                template="plotly_white", height=360,
                xaxis_title="Annual Volatility (%)",
                yaxis_title="Annual Return (%)",
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#ffffff", font=dict(color="#111"),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        with col_bt:
            st.markdown('<div class="sec-h">Historical Backtest</div>',
                        unsafe_allow_html=True)
            bt = backtest(pm, opt["max_sharpe_weights"])
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt["Date"].astype(str), y=bt["Portfolio"],
                name="Optimised Portfolio",
                line=dict(color="#1565C0", width=2),
                fill="tozeroy", fillcolor="rgba(21,101,192,0.05)"
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt["Date"].astype(str), y=bt["EqualWeight"],
                name="Equal Weight",
                line=dict(color="#e65100", width=1.5, dash="dash")
            ))
            fig_bt.update_layout(
                template="plotly_white", height=360,
                yaxis_title="Value (normalised to 100)",
                xaxis_title="Date",
                margin=dict(l=0, r=0, t=10, b=0),
                paper_bgcolor="#ffffff", font=dict(color="#111"),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        # Strategy comparison table
        st.markdown('<div class="sec-h">Strategy Comparison</div>',
                    unsafe_allow_html=True)
        p = opt["performance"]; r = opt["risk_metrics"]
        table = pd.DataFrame({
            "Metric": ["Annual Return","Annual Volatility","Sharpe Ratio",
                       "VaR 95%","CVaR 95%","Max Drawdown"],
            "Max Sharpe": [
                f"{p['max_sharpe']['expected_return']*100:.1f}%",
                f"{p['max_sharpe']['volatility']*100:.1f}%",
                f"{p['max_sharpe']['sharpe_ratio']:.3f}",
                f"{r['var_95']*100:.2f}%", f"{r['cvar_95']*100:.2f}%",
                f"{r['max_drawdown']*100:.1f}%"
            ],
            "Min Volatility": [
                f"{p['min_vol']['expected_return']*100:.1f}%",
                f"{p['min_vol']['volatility']*100:.1f}%",
                f"{p['min_vol']['sharpe_ratio']:.3f}",
                "—", "—", "—"
            ],
            "Equal Weight": [
                f"{p['equal']['expected_return']*100:.1f}%",
                f"{p['equal']['volatility']*100:.1f}%",
                f"{p['equal']['sharpe_ratio']:.3f}",
                "—", "—", "—"
            ]
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        # Per-company weight bar chart
        st.markdown('<div class="sec-h">Weight Breakdown by Strategy</div>',
                    unsafe_allow_html=True)
        bar_rows = []
        for strat, wts in [("Max Sharpe", opt["max_sharpe_weights"]),
                           ("Min Vol",    opt["min_vol_weights"]),
                           ("Equal",      opt["equal_weights"])]:
            for tk, w in wts.items():
                bar_rows.append({"Strategy": strat, "Ticker": tk, "Weight %": w*100})
        fig_bar = px.bar(
            pd.DataFrame(bar_rows),
            x="Ticker", y="Weight %", color="Strategy",
            barmode="group", template="plotly_white",
            color_discrete_sequence=["#1565C0","#1a7a4a","#e65100"]
        )
        fig_bar.update_layout(
            height=300, margin=dict(l=0,r=0,t=10,b=0),
            paper_bgcolor="#ffffff", font=dict(color="#111")
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Saved analyses
        if st.session_state.logged_in and st.session_state.saved_analyses:
            st.markdown('<div class="sec-h">Saved Analyses</div>',
                        unsafe_allow_html=True)
            for i, a in enumerate(st.session_state.saved_analyses, 1):
                st.markdown(
                    f"**{i}.** `{', '.join(a['stocks'])}` — "
                    f"Sharpe: **{a['sharpe']:.3f}** | Period: {a['period']}"
                )


# ════════════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<h2 style="color:#111">About & Methodology</h2>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
### Data pipeline
1. **yfinance** downloads live OHLCV (auto-adjusted for splits/dividends)
2. **Feature engineering** adds 4 technical indicators:
   - RSI-14 (momentum)
   - MACD 12-26 (trend)
   - Bollinger %B (mean-reversion)
   - EMA ratio = Close / EMA50 (trend strength)
3. **Strict time split** — scaler fitted on training data only
4. **60-timestep sliding windows** × 9 features per sample

### Time-based split
| Period | Dates |
|--------|-------|
| Training | Start → 2023-12-31 |
| Testing  | 2024-01-01 → present |

Training and test sets are **never shuffled**. No future data leakage.
        """)
    with c2:
        st.markdown("""
### Models
| Parameter   | LSTM / GRU |
|-------------|------------|
| Layer 1     | 128 units, return_seq=True |
| Dropout     | 0.30       |
| Layer 2     | 64 units, return_seq=False |
| Dropout     | 0.30       |
| Dense       | 32 → 1 (linear) |
| Optimiser   | Adam lr=0.001 |
| Loss        | MSE        |
| EarlyStopping | patience=15 |
| Forecast    | 30-day iterative |

### Portfolio optimisation
- **Expected returns**: 60% LSTM/GRU + 40% historical mean
- **Covariance**: Ledoit-Wolf shrinkage
- **Strategies**: Max Sharpe · Min Volatility · Equal Weight
- **Risk metrics**: VaR 95%, CVaR 95%, Max Drawdown

### Tech stack
`TensorFlow 2.x` · `PyPortfolioOpt` · `yfinance` · `Streamlit` · `Plotly`

### Authors
**Saisha Verma** (18101012024) · **Ritisha Sood** (17201012024)

> *Educational purposes only. Not financial advice.*
        """)
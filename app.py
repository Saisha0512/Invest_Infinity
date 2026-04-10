"""
app.py — Portfolio Optimization Dashboard
==========================================
Run:  streamlit run app.py

Pages
─────
🏠 Home        — stock picker, investment input, quick KPIs
📊 Forecast    — per-stock LSTM price charts + accuracy metrics
💼 Portfolio   — allocation donut, shares-to-buy, correlation heatmap
📈 Performance — efficient frontier, historical backtest, strategy table
ℹ️  About       — methodology, hyperparameter table, authors

Languages: English · हिन्दी · Français  (extend via TRANSLATIONS dict)
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── path setup ────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

from utils.data_fetcher import (
    fetch_stock_data, STOCK_UNIVERSE, SORTED_TICKERS,
    get_trending_tickers, get_sector_map
)
from model.lstm_model import train, forecast, TIMESTEPS, FORECAST_DAYS
from model.portfolio_optimizer import (
    build_price_matrix, optimise, backtest
)

MODEL_DIR   = os.path.join(BASE, "models")
FALLBACK_CSV = os.path.join(BASE, "data", "stocks_data_1.csv")
os.makedirs(MODEL_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Translations
# ──────────────────────────────────────────────────────────────
TRANSLATIONS = {
    "English": {
        "app_title":        "Portfolio Optimizer — LSTM & ML/DL",
        "tagline":          "AI-powered stock forecasting & portfolio construction",
        "lang_label":       "🌐 Language",
        "sidebar_stocks":   "Select stocks (5–10 recommended)",
        "sidebar_period":   "Historical data period",
        "sidebar_invest":   "Total investment amount ($)",
        "sidebar_rfr":      "Risk-free rate (%)",
        "btn_fetch":        "🔄 Fetch Data",
        "btn_train":        "🧠 Train Models",
        "btn_run":          "🚀 Run Analysis",
        "tab_home":         "🏠 Home",
        "tab_forecast":     "📊 Forecast",
        "tab_portfolio":    "💼 Portfolio",
        "tab_perf":         "📈 Performance",
        "tab_about":        "ℹ️ About",
        "strategy":         "Optimisation strategy",
        "max_sharpe":       "Max Sharpe",
        "min_vol":          "Min Volatility",
        "equal_w":          "Equal Weight",
        "sharpe":           "Sharpe Ratio",
        "ann_ret":          "Annual Return",
        "ann_vol":          "Annual Volatility",
        "var95":            "VaR (95%)",
        "cvar95":           "CVaR (95%)",
        "max_dd":           "Max Drawdown",
        "allocation":       "Optimal Allocation",
        "shares_to_buy":    "Shares to buy",
        "leftover":         "Leftover cash",
        "corr_title":       "Return Correlation Matrix",
        "frontier_title":   "Efficient Frontier",
        "backtest_title":   "Historical Backtest (normalised to 100)",
        "compare_title":    "Strategy Comparison",
        "forecast_title":   "30-Day Price Forecast",
        "login_header":     "👤 Account",
        "login_user":       "Username",
        "login_pwd":        "Password",
        "login_btn":        "Login",
        "save_btn":         "💾 Save Analysis",
        "saved_header":     "📁 Saved Analyses",
        "no_data_msg":      "Fetch data first using the sidebar.",
        "no_model_msg":     "Train models first.",
        "no_result_msg":    "Run Analysis first.",
        "trending_badge":   "🔥 Trending",
        "sector_label":     "Sector",
        "rmse_label":       "RMSE",
        "mape_label":       "MAPE",
        "last_price":       "Last price",
        "day30_forecast":   "Day-30 forecast",
        "exp_change":       "Expected change",
        "data_preview":     "Data preview",
        "fetch_success":    "Data fetched successfully!",
        "train_success":    "All models trained!",
        "run_success":      "Analysis complete!",
        "period_options":   ["1y", "2y", "3y", "5y", "10y"],
    },
    "हिन्दी": {
        "app_title":        "पोर्टफोलियो ऑप्टिमाइज़र — LSTM & ML/DL",
        "tagline":          "AI-आधारित स्टॉक पूर्वानुमान और पोर्टफोलियो निर्माण",
        "lang_label":       "🌐 भाषा",
        "sidebar_stocks":   "स्टॉक चुनें (5–10 अनुशंसित)",
        "sidebar_period":   "ऐतिहासिक डेटा अवधि",
        "sidebar_invest":   "कुल निवेश राशि ($)",
        "sidebar_rfr":      "जोखिम-मुक्त दर (%)",
        "btn_fetch":        "🔄 डेटा प्राप्त करें",
        "btn_train":        "🧠 मॉडल प्रशिक्षित करें",
        "btn_run":          "🚀 विश्लेषण चलाएं",
        "tab_home":         "🏠 होम",
        "tab_forecast":     "📊 पूर्वानुमान",
        "tab_portfolio":    "💼 पोर्टफोलियो",
        "tab_perf":         "📈 प्रदर्शन",
        "tab_about":        "ℹ️ जानकारी",
        "strategy":         "अनुकूलन रणनीति",
        "max_sharpe":       "अधिकतम शार्प",
        "min_vol":          "न्यूनतम अस्थिरता",
        "equal_w":          "समान भार",
        "sharpe":           "शार्प अनुपात",
        "ann_ret":          "वार्षिक रिटर्न",
        "ann_vol":          "वार्षिक अस्थिरता",
        "var95":            "VaR (95%)",
        "cvar95":           "CVaR (95%)",
        "max_dd":           "अधिकतम गिरावट",
        "allocation":       "इष्टतम आवंटन",
        "shares_to_buy":    "खरीदने के शेयर",
        "leftover":         "शेष नकद",
        "corr_title":       "सहसंबंध मैट्रिक्स",
        "frontier_title":   "कुशल सीमा",
        "backtest_title":   "ऐतिहासिक बैकटेस्ट (100 से सामान्यीकृत)",
        "compare_title":    "रणनीति तुलना",
        "forecast_title":   "30-दिन का मूल्य पूर्वानुमान",
        "login_header":     "👤 खाता",
        "login_user":       "उपयोगकर्ता नाम",
        "login_pwd":        "पासवर्ड",
        "login_btn":        "लॉगिन",
        "save_btn":         "💾 विश्लेषण सहेजें",
        "saved_header":     "📁 सहेजे गए विश्लेषण",
        "no_data_msg":      "पहले साइडबार से डेटा प्राप्त करें।",
        "no_model_msg":     "पहले मॉडल प्रशिक्षित करें।",
        "no_result_msg":    "पहले विश्लेषण चलाएं।",
        "trending_badge":   "🔥 ट्रेंडिंग",
        "sector_label":     "क्षेत्र",
        "rmse_label":       "RMSE",
        "mape_label":       "MAPE",
        "last_price":       "अंतिम मूल्य",
        "day30_forecast":   "30वें दिन का पूर्वानुमान",
        "exp_change":       "अपेक्षित परिवर्तन",
        "data_preview":     "डेटा पूर्वावलोकन",
        "fetch_success":    "डेटा सफलतापूर्वक प्राप्त!",
        "train_success":    "सभी मॉडल प्रशिक्षित!",
        "run_success":      "विश्लेषण पूर्ण!",
        "period_options":   ["1y", "2y", "3y", "5y", "10y"],
    },
    "Français": {
        "app_title":        "Optimiseur de Portefeuille — LSTM & ML/DL",
        "tagline":          "Prévision boursière et construction de portefeuille par IA",
        "lang_label":       "🌐 Langue",
        "sidebar_stocks":   "Choisir des actions (5–10 recommandées)",
        "sidebar_period":   "Période de données historiques",
        "sidebar_invest":   "Montant total d'investissement ($)",
        "sidebar_rfr":      "Taux sans risque (%)",
        "btn_fetch":        "🔄 Récupérer les données",
        "btn_train":        "🧠 Entraîner les modèles",
        "btn_run":          "🚀 Lancer l'analyse",
        "tab_home":         "🏠 Accueil",
        "tab_forecast":     "📊 Prévision",
        "tab_portfolio":    "💼 Portefeuille",
        "tab_perf":         "📈 Performance",
        "tab_about":        "ℹ️ À propos",
        "strategy":         "Stratégie d'optimisation",
        "max_sharpe":       "Sharpe max",
        "min_vol":          "Volatilité min",
        "equal_w":          "Équipondéré",
        "sharpe":           "Ratio de Sharpe",
        "ann_ret":          "Rendement annuel",
        "ann_vol":          "Volatilité annuelle",
        "var95":            "VaR (95%)",
        "cvar95":           "CVaR (95%)",
        "max_dd":           "Drawdown max",
        "allocation":       "Allocation optimale",
        "shares_to_buy":    "Actions à acheter",
        "leftover":         "Liquidités restantes",
        "corr_title":       "Matrice de corrélation",
        "frontier_title":   "Frontière efficiente",
        "backtest_title":   "Backtest historique (normalisé à 100)",
        "compare_title":    "Comparaison des stratégies",
        "forecast_title":   "Prévision des prix sur 30 jours",
        "login_header":     "👤 Compte",
        "login_user":       "Nom d'utilisateur",
        "login_pwd":        "Mot de passe",
        "login_btn":        "Connexion",
        "save_btn":         "💾 Sauvegarder l'analyse",
        "saved_header":     "📁 Analyses sauvegardées",
        "no_data_msg":      "Récupérez d'abord les données via la barre latérale.",
        "no_model_msg":     "Entraînez d'abord les modèles.",
        "no_result_msg":    "Lancez d'abord l'analyse.",
        "trending_badge":   "🔥 Tendance",
        "sector_label":     "Secteur",
        "rmse_label":       "RMSE",
        "mape_label":       "MAPE",
        "last_price":       "Dernier prix",
        "day30_forecast":   "Prévision J+30",
        "exp_change":       "Variation attendue",
        "data_preview":     "Aperçu des données",
        "fetch_success":    "Données récupérées avec succès !",
        "train_success":    "Tous les modèles entraînés !",
        "run_success":      "Analyse terminée !",
        "period_options":   ["1y", "2y", "3y", "5y", "10y"],
    }
}

# ──────────────────────────────────────────────────────────────
# Page config & CSS
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Portfolio Optimizer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
/* Metric cards */
.kpi-card {
    background: #0d1b2a;
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    margin-bottom: 8px;
}
.kpi-card .kpi-label { font-size: 11px; color: #7ec8e3; text-transform: uppercase;
                        letter-spacing: .06em; margin-bottom: 4px; }
.kpi-card .kpi-value { font-size: 24px; font-weight: 700; color: #ffffff; }
.kpi-card .kpi-sub   { font-size: 11px; color: #aaaaaa; margin-top: 2px; }

/* Positive / negative colouring */
.pos { color: #4caf93 !important; }
.neg { color: #e57373 !important; }

/* Stock badge pill */
.sbadge {
    display: inline-block;
    background: #1e3a5f;
    color: #7ec8e3;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 12px;
    margin: 2px 4px 2px 0;
}
.tbadge {
    background: #b34000;
    color: #fff;
    font-size: 10px;
    border-radius: 4px;
    padding: 1px 5px;
    margin-left: 4px;
    vertical-align: middle;
}
/* Section divider */
.sec-header {
    font-size: 15px;
    font-weight: 600;
    color: #7ec8e3;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 4px;
    margin: 16px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Session state defaults
# ──────────────────────────────────────────────────────────────
_defaults = {
    "language":         "English",
    "selected_stocks":  ["AAPL", "MSFT", "GOOGL", "AMZN"],
    "data_period":      "5y",
    "investment":       100_000.0,
    "risk_free_rate":   0.05,
    "stock_data":       None,
    "forecast_results": {},
    "opt_results":      None,
    "price_matrix":     None,
    "logged_in":        False,
    "username":         "",
    "saved_analyses":   [],
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def T(key: str) -> str:
    lang = st.session_state.language
    return TRANSLATIONS.get(lang, TRANSLATIONS["English"]).get(key, key)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # ── Language ──────────────────────────────────────────────
    lang_choice = st.selectbox(
        T("lang_label"),
        list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.language)
    )
    st.session_state.language = lang_choice
    st.divider()

    # ── Stock picker ──────────────────────────────────────────
    st.markdown(f"**{T('sidebar_stocks')}**")

    sector_map   = get_sector_map()
    trend_tickers = get_trending_tickers()

    # Build display labels: trending ones get 🔥
    label_to_ticker = {}
    for ticker in SORTED_TICKERS:
        info  = STOCK_UNIVERSE[ticker]
        label = f"{ticker} {'🔥' if info['trending'] else '  '} — {info['name']}"
        label_to_ticker[label] = ticker

    default_labels = [
        lbl for lbl, tk in label_to_ticker.items()
        if tk in st.session_state.selected_stocks
    ]

    chosen_labels = st.multiselect(
        "",
        options=list(label_to_ticker.keys()),
        default=default_labels,
        label_visibility="collapsed"
    )
    st.session_state.selected_stocks = [label_to_ticker[l] for l in chosen_labels]

    if len(st.session_state.selected_stocks) < 2:
        st.warning("⚠ Select at least 2 stocks.")

    st.divider()

    # ── Data period ───────────────────────────────────────────
    period_opts = T("period_options")
    period_idx  = period_opts.index(st.session_state.data_period) \
                  if st.session_state.data_period in period_opts else 3
    st.session_state.data_period = st.selectbox(
        T("sidebar_period"), period_opts, index=period_idx
    )

    # ── Investment & risk-free rate ───────────────────────────
    st.session_state.investment = st.number_input(
        T("sidebar_invest"),
        min_value=1_000.0,
        max_value=10_000_000.0,
        value=st.session_state.investment,
        step=1_000.0,
        format="%.0f"
    )
    rfr_pct = st.slider(
        T("sidebar_rfr"),
        min_value=0.0, max_value=10.0,
        value=st.session_state.risk_free_rate * 100,
        step=0.25
    )
    st.session_state.risk_free_rate = rfr_pct / 100.0
    st.divider()

    # ── STEP 1: Fetch data ─────────────────────────────────────
    if st.button(T("btn_fetch"), use_container_width=True):
        if not st.session_state.selected_stocks:
            st.error("Select stocks first.")
        else:
            with st.spinner("Fetching data from yfinance…"):
                try:
                    df = fetch_stock_data(
                        st.session_state.selected_stocks,
                        period=st.session_state.data_period,
                        fallback_csv=FALLBACK_CSV
                    )
                    st.session_state.stock_data = df
                    st.success(T("fetch_success"))
                    st.caption(
                        f"{len(df)} rows · "
                        f"{df['Company'].nunique()} companies · "
                        f"{df['Date'].min().date()} → {df['Date'].max().date()}"
                    )
                except Exception as e:
                    st.error(f"Fetch failed: {e}")

    # ── STEP 2: Train models ──────────────────────────────────
    if st.button(T("btn_train"), use_container_width=True):
        if st.session_state.stock_data is None:
            st.error(T("no_data_msg"))
        else:
            df    = st.session_state.stock_data
            total = df["Company"].nunique()
            prog  = st.progress(0)
            log   = st.empty()
            for i, company in enumerate(df["Company"].unique()):
                log.text(f"Training {company} ({i+1}/{total})…")
                df_c = df[df["Company"] == company].copy()
                try:
                    res = train(df_c, company, model_dir=MODEL_DIR)
                    st.success(
                        f"✅ **{company}** — val_loss: {res['val_loss']:.5f} "
                        f"| RMSE≈{res['val_mae']:.4f} | {res['epochs_run']} epochs"
                    )
                except Exception as e:
                    st.error(f"❌ {company}: {e}")
                prog.progress((i + 1) / total)
            log.text(T("train_success"))

    # ── STEP 3: Run analysis ──────────────────────────────────
    if st.button(T("btn_run"), use_container_width=True, type="primary"):
        if st.session_state.stock_data is None:
            st.error(T("no_data_msg"))
        else:
            df = st.session_state.stock_data
            forecast_results = {}
            prog = st.progress(0)
            companies = df["Company"].unique().tolist()
            for i, company in enumerate(companies):
                df_c = df[df["Company"] == company].copy()
                try:
                    res = forecast(df_c, company, model_dir=MODEL_DIR)
                    forecast_results[company] = res
                except FileNotFoundError:
                    st.warning(f"⚠ {company}: {T('no_model_msg')}")
                except Exception as e:
                    st.error(f"❌ {company}: {e}")
                prog.progress((i + 1) / len(companies))

            if forecast_results:
                pm = build_price_matrix(df, list(forecast_results.keys()))
                st.session_state.forecast_results = forecast_results
                st.session_state.price_matrix     = pm
                try:
                    st.session_state.opt_results = optimise(
                        forecast_results, pm,
                        investment_amount=st.session_state.investment,
                        risk_free_rate=st.session_state.risk_free_rate
                    )
                    st.success(T("run_success"))
                except Exception as e:
                    st.error(f"Optimisation failed: {e}")

    st.divider()

    # ── Optional login ─────────────────────────────────────────
    with st.expander(T("login_header")):
        uname = st.text_input(T("login_user"), key="uname_input")
        pwd   = st.text_input(T("login_pwd"), type="password", key="pwd_input")
        if st.button(T("login_btn")):
            if uname and pwd:
                st.session_state.logged_in = True
                st.session_state.username  = uname
                st.success(f"Welcome, {uname}!")
        if st.session_state.logged_in:
            st.info(f"Logged in: {st.session_state.username}")
            if st.button(T("save_btn")):
                if st.session_state.opt_results:
                    entry = {
                        "user":    st.session_state.username,
                        "stocks":  st.session_state.selected_stocks[:],
                        "period":  st.session_state.data_period,
                        "weights": st.session_state.opt_results["max_sharpe_weights"],
                        "sharpe":  st.session_state.opt_results["performance"]["max_sharpe"]["sharpe_ratio"]
                    }
                    st.session_state.saved_analyses.append(entry)
                    st.success("Saved!")
                else:
                    st.warning(T("no_result_msg"))

# ──────────────────────────────────────────────────────────────
# MAIN — TABS
# ──────────────────────────────────────────────────────────────
tab_home, tab_fc, tab_port, tab_perf, tab_about = st.tabs([
    T("tab_home"), T("tab_forecast"), T("tab_portfolio"),
    T("tab_perf"), T("tab_about")
])


# ════════════════════════════════════════════
# TAB 1 — HOME
# ════════════════════════════════════════════
with tab_home:
    st.title(f"📈 {T('app_title')}")
    st.caption(T("tagline"))
    st.divider()

    col_steps, col_stocks, col_kpi = st.columns([1.2, 1, 1])

    with col_steps:
        st.markdown('<div class="sec-header">How it works</div>', unsafe_allow_html=True)
        st.markdown("""
**1.** Select stocks in the sidebar (trending shown with 🔥)  
**2.** Click **Fetch Data** — downloads live prices from Yahoo Finance  
**3.** Click **Train Models** — fits one LSTM per stock  
**4.** Click **Run Analysis** — forecasts prices + optimises portfolio  
**5.** Explore Forecast · Portfolio · Performance tabs
        """)

    with col_stocks:
        st.markdown('<div class="sec-header">Selected stocks</div>', unsafe_allow_html=True)
        if st.session_state.selected_stocks:
            for c in st.session_state.selected_stocks:
                info = STOCK_UNIVERSE.get(c, {})
                badge = f'<span class="tbadge">🔥 Trending</span>' if info.get("trending") else ""
                st.markdown(
                    f'<span class="sbadge">{c}</span> {info.get("name","")}{badge}',
                    unsafe_allow_html=True
                )
        else:
            st.info("No stocks selected.")

    with col_kpi:
        st.markdown('<div class="sec-header">Quick summary</div>', unsafe_allow_html=True)
        st.metric("Stocks selected", len(st.session_state.selected_stocks))
        st.metric("Investment", f"${st.session_state.investment:,.0f}")
        if st.session_state.opt_results:
            ms = st.session_state.opt_results["performance"]["max_sharpe"]
            st.metric(T("sharpe"), f"{ms['sharpe_ratio']:.3f}")
            st.metric(T("ann_ret"), f"{ms['expected_return']*100:.1f}%")

    # Data preview
    if st.session_state.stock_data is not None:
        st.divider()
        st.markdown(f'<div class="sec-header">{T("data_preview")}</div>',
                    unsafe_allow_html=True)
        df_prev = st.session_state.stock_data
        st.caption(
            f"Source: **yfinance** · {len(df_prev):,} rows · "
            f"{df_prev['Company'].nunique()} tickers · "
            f"{df_prev['Date'].min().date()} → {df_prev['Date'].max().date()}"
        )
        st.dataframe(
            df_prev.sort_values(["Company","Date"]).tail(30),
            use_container_width=True, hide_index=True
        )


# ════════════════════════════════════════════
# TAB 2 — FORECAST
# ════════════════════════════════════════════
with tab_fc:
    st.header(f"📊 {T('forecast_title')}")

    fr = st.session_state.forecast_results
    if not fr:
        st.info(f"⬅️ {T('no_result_msg')}")
    else:
        for company, res in fr.items():
            info = STOCK_UNIVERSE.get(company, {"name": company, "sector": "—"})
            with st.expander(
                f"**{company}** — {info['name']}  |  {info['sector']}",
                expanded=True
            ):
                # ── KPI row ──
                c1, c2, c3, c4, c5 = st.columns(5)
                pct_change = (
                    (res["forecast_prices"][-1] - res["last_known_price"])
                    / res["last_known_price"] * 100
                )
                delta_cls = "pos" if pct_change >= 0 else "neg"
                c1.metric(T("last_price"),     f"${res['last_known_price']:.2f}")
                c2.metric(T("day30_forecast"),  f"${res['forecast_prices'][-1]:.2f}")
                c3.metric(T("exp_change"),      f"{pct_change:+.1f}%",
                          delta_color="normal")
                c4.metric(T("rmse_label"),      f"{res['rmse']:.3f}")
                c5.metric(T("mape_label"),      f"{res['mape']:.2f}%")

                # ── Chart ──
                fig = go.Figure()

                # Actual (test period)
                fig.add_trace(go.Scatter(
                    x=res["actual_dates"],
                    y=res["actual_prices"],
                    name="Actual (test)",
                    line=dict(color="#4fc3f7", width=2)
                ))
                # Model predictions on test period
                fig.add_trace(go.Scatter(
                    x=res["actual_dates"],
                    y=res["predicted_prices"],
                    name="LSTM predicted",
                    line=dict(color="#ffb300", width=1.5, dash="dot")
                ))
                # Future forecast band
                fig.add_trace(go.Scatter(
                    x=res["forecast_dates"] + res["forecast_dates"][::-1],
                    y=[p * 1.02 for p in res["forecast_prices"]] +
                      [p * 0.98 for p in res["forecast_prices"][::-1]],
                    fill="toself",
                    fillcolor="rgba(129,199,132,0.10)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="±2% band",
                    showlegend=True
                ))
                # Future forecast line
                fig.add_trace(go.Scatter(
                    x=res["forecast_dates"],
                    y=res["forecast_prices"],
                    name="Forecast (30d)",
                    line=dict(color="#81c784", width=2)
                ))
                # Vertical separator
                fig.add_vline(
                    x=pd.to_datetime(res["forecast_dates"][0]),
                    line_dash="dash", line_color="rgba(255,255,255,0.3)",
                    annotation_text="Forecast →",
                    annotation_position="top right"
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=380,
                    margin=dict(l=0, r=0, t=30, b=0),
                    legend=dict(orientation="h", y=-0.22),
                    yaxis_title="Price ($)",
                    xaxis_title="Date",
                    hovermode="x unified"
                )
                st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — PORTFOLIO
# ════════════════════════════════════════════
with tab_port:
    st.header(f"💼 {T('allocation')}")

    opt = st.session_state.opt_results
    if opt is None:
        st.info(f"⬅️ {T('no_result_msg')}")
    else:
        # Strategy radio
        strategy = st.radio(
            T("strategy"),
            [T("max_sharpe"), T("min_vol"), T("equal_w")],
            horizontal=True
        )
        key_map = {
            T("max_sharpe"): ("max_sharpe_weights", "max_sharpe"),
            T("min_vol"):    ("min_vol_weights",    "min_vol"),
            T("equal_w"):    ("equal_weights",      "equal"),
        }
        w_key, p_key = key_map[strategy]
        weights = opt[w_key]
        perf    = opt["performance"][p_key]
        risk    = opt["risk_metrics"]

        # ── Top KPI strip ──
        kpis = [
            (T("sharpe"),    f"{perf['sharpe_ratio']:.3f}",   ""),
            (T("ann_ret"),   f"{perf['expected_return']*100:.1f}%",
             "pos" if perf["expected_return"] > 0 else "neg"),
            (T("ann_vol"),   f"{perf['volatility']*100:.1f}%", ""),
            (T("var95"),     f"{risk['var_95']*100:.2f}%",     "neg"),
            (T("cvar95"),    f"{risk['cvar_95']*100:.2f}%",    "neg"),
            (T("max_dd"),    f"{risk['max_drawdown']*100:.1f}%","neg"),
        ]
        cols = st.columns(len(kpis))
        for col, (lbl, val, cls) in zip(cols, kpis):
            col.markdown(
                f'<div class="kpi-card">'
                f'<div class="kpi-label">{lbl}</div>'
                f'<div class="kpi-value {cls}">{val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.divider()
        col_donut, col_table = st.columns([1, 1])

        with col_donut:
            st.markdown(f'<div class="sec-header">{T("allocation")} — donut</div>',
                        unsafe_allow_html=True)
            labels  = [k for k, v in weights.items() if v > 0.001]
            values  = [weights[k] for k in labels]
            amounts = [v * st.session_state.investment for v in values]
            colors  = px.colors.qualitative.Bold[:len(labels)]

            fig_d = go.Figure(go.Pie(
                labels=labels,
                values=values,
                hole=0.55,
                marker=dict(colors=colors),
                textinfo="label+percent",
                hovertemplate=(
                    "<b>%{label}</b><br>"
                    "Weight: %{percent}<br>"
                    "Amount: $%{customdata:,.0f}"
                    "<extra></extra>"
                ),
                customdata=amounts
            ))
            fig_d.add_annotation(
                text=f"${st.session_state.investment/1000:.0f}K",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20, color="white")
            )
            fig_d.update_layout(
                template="plotly_dark", height=340,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=True,
                legend=dict(orientation="v", x=1.02, y=0.5)
            )
            st.plotly_chart(fig_d, use_container_width=True)

        with col_table:
            st.markdown(
                f'<div class="sec-header">{T("shares_to_buy")}</div>',
                unsafe_allow_html=True
            )
            pm = st.session_state.price_matrix
            if w_key == "max_sharpe_weights" and opt.get("discrete_allocation") and pm is not None:
                alloc = opt["discrete_allocation"]
                rows = []
                for c, n_shares in alloc.items():
                    price = float(pm[c].iloc[-1]) if c in pm.columns else 0
                    rows.append({
                        "Ticker":   c,
                        "Name":     STOCK_UNIVERSE.get(c, {}).get("name", c),
                        "Weight":   f"{weights.get(c,0)*100:.1f}%",
                        "Shares":   n_shares,
                        "Price":    f"${price:.2f}",
                        "Amount":   f"${n_shares * price:,.0f}"
                    })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True, hide_index=True
                )
                st.metric(T("leftover"), f"${opt['leftover']:.2f}")
            else:
                rows = []
                for c, w in weights.items():
                    if w > 0.001:
                        rows.append({
                            "Ticker": c,
                            "Weight": f"{w*100:.1f}%",
                            "Amount": f"${w*st.session_state.investment:,.0f}"
                        })
                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True, hide_index=True
                )

        # ── Correlation heatmap ──
        st.divider()
        st.markdown(
            f'<div class="sec-header">{T("corr_title")}</div>',
            unsafe_allow_html=True
        )
        corr = opt["correlation_matrix"]
        fig_h = px.imshow(
            corr, color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1, text_auto=".2f",
            template="plotly_dark"
        )
        fig_h.update_layout(
            height=360, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_h, use_container_width=True)


# ════════════════════════════════════════════
# TAB 4 — PERFORMANCE
# ════════════════════════════════════════════
with tab_perf:
    st.header(f"📈 {T('perf_title') if 'perf_title' in TRANSLATIONS['English'] else 'Portfolio Performance'}")

    opt = st.session_state.opt_results
    pm  = st.session_state.price_matrix

    if opt is None or pm is None:
        st.info(f"⬅️ {T('no_result_msg')}")
    else:
        col_ef, col_bt = st.columns([1, 1])

        # ── Efficient frontier ──
        with col_ef:
            st.markdown(
                f'<div class="sec-header">{T("frontier_title")}</div>',
                unsafe_allow_html=True
            )
            fd   = opt["frontier_data"]
            ms_p = opt["performance"]["max_sharpe"]
            mv_p = opt["performance"]["min_vol"]

            fig_ef = go.Figure()
            if fd["volatilities"]:
                fig_ef.add_trace(go.Scatter(
                    x=[v * 100 for v in fd["volatilities"]],
                    y=[r * 100 for r in fd["returns"]],
                    mode="lines", name="Efficient Frontier",
                    line=dict(color="#4fc3f7", width=2.5)
                ))
            fig_ef.add_trace(go.Scatter(
                x=[ms_p["volatility"] * 100],
                y=[ms_p["expected_return"] * 100],
                mode="markers+text",
                text=["★ Max Sharpe"],
                textposition="top right",
                marker=dict(size=14, color="#ffb300", symbol="star"),
                name="Max Sharpe"
            ))
            fig_ef.add_trace(go.Scatter(
                x=[mv_p["volatility"] * 100],
                y=[mv_p["expected_return"] * 100],
                mode="markers+text",
                text=["◆ Min Vol"],
                textposition="top right",
                marker=dict(size=12, color="#81c784", symbol="diamond"),
                name="Min Volatility"
            ))
            fig_ef.update_layout(
                template="plotly_dark", height=350,
                xaxis_title="Annual Volatility (%)",
                yaxis_title="Annual Return (%)",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_ef, use_container_width=True)

        # ── Historical backtest ──
        with col_bt:
            st.markdown(
                f'<div class="sec-header">{T("backtest_title")}</div>',
                unsafe_allow_html=True
            )
            bt = backtest(pm, opt["max_sharpe_weights"])
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(
                x=bt["Date"], y=bt["Portfolio"],
                name="Optimised Portfolio",
                line=dict(color="#4fc3f7", width=2),
                fill="tozeroy", fillcolor="rgba(79,195,247,0.06)"
            ))
            fig_bt.add_trace(go.Scatter(
                x=bt["Date"], y=bt["EqualWeight"],
                name="Equal Weight",
                line=dict(color="#ff7043", width=1.5, dash="dash")
            ))
            fig_bt.update_layout(
                template="plotly_dark", height=350,
                yaxis_title="Value (start=100)",
                xaxis_title="Date",
                margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", y=-0.25)
            )
            st.plotly_chart(fig_bt, use_container_width=True)

        # ── Strategy comparison table ──
        st.divider()
        st.markdown(
            f'<div class="sec-header">{T("compare_title")}</div>',
            unsafe_allow_html=True
        )
        p  = opt["performance"]
        r  = opt["risk_metrics"]
        table = pd.DataFrame({
            "Metric": [T("ann_ret"), T("ann_vol"), T("sharpe"),
                       T("var95"), T("cvar95"), T("max_dd")],
            T("max_sharpe"): [
                f"{p['max_sharpe']['expected_return']*100:.1f}%",
                f"{p['max_sharpe']['volatility']*100:.1f}%",
                f"{p['max_sharpe']['sharpe_ratio']:.3f}",
                f"{r['var_95']*100:.2f}%",
                f"{r['cvar_95']*100:.2f}%",
                f"{r['max_drawdown']*100:.1f}%"
            ],
            T("min_vol"): [
                f"{p['min_vol']['expected_return']*100:.1f}%",
                f"{p['min_vol']['volatility']*100:.1f}%",
                f"{p['min_vol']['sharpe_ratio']:.3f}",
                "—", "—", "—"
            ],
            T("equal_w"): [
                f"{p['equal']['expected_return']*100:.1f}%",
                f"{p['equal']['volatility']*100:.1f}%",
                f"{p['equal']['sharpe_ratio']:.3f}",
                "—", "—", "—"
            ]
        })
        st.dataframe(table, use_container_width=True, hide_index=True)

        # ── Per-stock weight bar chart ──
        st.divider()
        st.markdown('<div class="sec-header">Weight breakdown by strategy</div>',
                    unsafe_allow_html=True)
        strategies_for_bar = {
            T("max_sharpe"): opt["max_sharpe_weights"],
            T("min_vol"):    opt["min_vol_weights"],
            T("equal_w"):    opt["equal_weights"],
        }
        bar_data = []
        for strat, wts in strategies_for_bar.items():
            for ticker, w in wts.items():
                bar_data.append({"Strategy": strat, "Ticker": ticker, "Weight": w * 100})
        fig_bar = px.bar(
            pd.DataFrame(bar_data),
            x="Ticker", y="Weight", color="Strategy",
            barmode="group", template="plotly_dark",
            labels={"Weight": "Weight (%)"},
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_bar.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_bar, use_container_width=True)

        # ── Saved analyses ──
        if st.session_state.logged_in and st.session_state.saved_analyses:
            st.divider()
            st.markdown(
                f'<div class="sec-header">{T("saved_header")}</div>',
                unsafe_allow_html=True
            )
            for i, a in enumerate(st.session_state.saved_analyses, 1):
                st.markdown(
                    f"**{i}.** `{', '.join(a['stocks'])}` — "
                    f"Sharpe: **{a['sharpe']:.3f}** | "
                    f"Period: {a['period']}"
                )


# ════════════════════════════════════════════
# TAB 5 — ABOUT
# ════════════════════════════════════════════
with tab_about:
    st.header("ℹ️ Methodology & About")

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("""
### Data pipeline
1. **yfinance** downloads OHLCV prices for any ticker (auto-adjusted for splits & dividends)
2. **Feature engineering** adds 4 technical indicators per stock:
   - RSI-14 (momentum oscillator)
   - MACD 12-26 (trend / momentum)
   - Bollinger %B (mean-reversion position)
   - EMA ratio = Close / EMA50 (trend strength)
3. **MinMaxScaler** fitted on training data only (no leakage)
4. **Sliding windows** of 60 timesteps × 9 features → supervised sequences

### LSTM architecture
| Layer    | Config                          |
|----------|---------------------------------|
| Input    | (60, 9)                         |
| LSTM-1   | 128 units, return_sequences=True|
| Dropout  | 0.30                            |
| LSTM-2   | 64 units, return_sequences=False|
| Dropout  | 0.30                            |
| Dense-1  | 32 units, ReLU                  |
| Dense-2  | 1 unit, Linear (regression)     |
| Optimiser| Adam lr=0.001                   |
| Loss     | MSE                             |
""")

    with col_r:
        st.markdown("""
### Portfolio optimisation
1. **Expected returns** = 60% LSTM 30-day forecast (annualised) + 40% historical mean
2. **Covariance** = Ledoit-Wolf shrinkage estimator
3. **Efficient Frontier** via PyPortfolioOpt:
   - Max Sharpe portfolio
   - Min Volatility portfolio
   - Equal-weight baseline
4. **Discrete allocation** = greedy algorithm for whole-share counts
5. **Risk metrics**:
   - VaR 95% (Historical simulation)
   - CVaR 95% (Expected Shortfall)
   - Maximum Drawdown

### Hyperparameters
| Parameter     | Value  |
|---------------|--------|
| Timesteps     | 60     |
| Features      | 9      |
| LSTM units    | 128→64 |
| Dense units   | 32→1   |
| Dropout       | 0.30   |
| Batch size    | 32     |
| Max epochs    | 150    |
| Early stop    | 15 ep  |
| Forecast days | 30     |
""")

    st.divider()
    st.markdown("""
### Tech stack
`TensorFlow 2.x` · `PyPortfolioOpt` · `yfinance` · `Streamlit` · `Plotly` · `scikit-learn` · `Pandas` · `NumPy`

### Authors
**Saisha Verma** (18101012024) · **Ritisha Sood** (17201012024)

> *This dashboard is for educational purposes only and does not constitute financial advice.*
    """)
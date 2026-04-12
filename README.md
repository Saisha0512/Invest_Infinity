# Invest Infinity — AI Predictive Forecasting Dashboard

> **NatWest - Code for Purpose India Hackathon**
> **Group : Code Infinity**
> **Team:** Saisha Verma · Ritisha Sood

---
## Table of Contents

1. [Overview](#overview)
2. [Meet Alex](#meet-alex)
3. [Step 1 - Alex Opens the Dashboard](#step-1--alex-opens-the-dashboard)
4. [Step 2 - Alex Trains the AI Models](#step-2--alex-trains-the-ai-models)
5. [Step 3 - Alex Compares LSTM vs GRU](#step-3--alex-compares-lstm-vs-gru)
6. [Step 4 - Alex Sees the Future (Kind Of)](#step-4--alex-sees-the-future-kind-of)
7. [Step 5 - Alex Builds His Portfolio](#step-5--alex-builds-his-portfolio)
8. [Step 6 - Alex Reviews the Track Record](#step-6--alex-reviews-the-track-record)
9. [Step 7 - Alex goes through the About & Methodology Tab](#about--methodology)
10. [Features](#features-implemented)
11. [Architecture](#architecture)
12. [Folder Structure](#folder-structure)
13. [Tech Stack](#tech-stack)
14. [Setup](#setup--python-3118-environment)
15. [Usage](#usage)
16. [Model Details](#model-details)
17. [Portfolio Optimisation](#portfolio-optimisation)
18. [Architecture Notes](#architecture-notes)
19. [Limitations](#limitations)
20. [Future Improvements](#future-improvements)
21. [License](#license)
22. [Authors](#authors)

---

## Overview

**Invest Infinity** is an AI-powered stock portfolio optimisation dashboard that transforms historical market data into actionable investment forecasts. It uses dual deep learning models (LSTM and GRU) to predict future stock prices, compares model performance on strict out-of-sample test data, and then constructs an optimally weighted portfolio using the Efficient Frontier method.

**Problem it solves:** Individual investors and analysts lack accessible tools to look ahead using AI — they rely on backward-looking data alone. Invest Infinity provides honest, uncertainty-aware forecasts with clear confidence bands, baseline comparisons, and anomaly signals, enabling users to understand *what the future may look like* rather than just what already happened.

**Intended users:** Retail investors, finance students, and analysts who want AI-assisted portfolio construction without needing to write code.

---

## Meet Alex

> *Alex is a 20-year-old engineering student. He just got his first internship stipend — ₹15,000 — and he wants to invest it wisely. The problem? He has no idea where to start.*

Alex has heard about stocks. He knows names like Apple and Tesla. He's seen Reddit threads about "going to the moon" and cautionary tales about losing everything overnight. What he *doesn't* have is a trusted, data-driven tool that can help him think clearly — without needing a finance degree.

He stumbles upon **Invest Infinity**.

> *"It uses AI to predict stock prices and build a portfolio? And I don't have to write a single line of code? Let me try this."*

Here's what Alex's journey looks like — step by step.

---

## Step 1 — Alex Opens the Dashboard

The first thing Alex sees is a bold blue hero banner with the app name, a one-line tagline, and four feature badges: *LSTM & GRU Models*, *Live Yahoo Finance Data*, *Portfolio Optimization*, *Risk Analytics*. Below that, the layout splits into two columns.

On the **left**, a numbered how-it-works guide walks through every step in plain English — no jargon, no assumptions. On the **right**, the **Control Panel** is waiting: a stock picker, a historical data period selector, an investment amount field, a risk-free rate slider, three action buttons, and a forecast horizon slider.

<img width="1878" height="890" alt="image" src="https://github.com/user-attachments/assets/c16fb08e-ee20-4068-a2eb-db770ff602a8" />


> *"This doesn't look intimidating at all. Let me pick some stocks I've actually heard of."*

Alex selects **AAPL, AMZN, GOOGL, JPM, and MSFT** from the dropdown — five companies he recognises, all flagged as `[Trending]`. He sets the period to **5 years**, his investment amount to **$100,000**, and leaves the risk-free rate at the default **5%** and forecast horizon at **30 days**.

Then he hits **Fetch Data**.

The dashboard connects to Yahoo Finance and pulls **6,280 rows** of live, auto-adjusted OHLCV data across all five tickers. A confirmation strip appears below the table: source `yfinance`, training cutoff `2023-12-31`, test period starting `2024-01-01`. The most recent price rows — MSFT closing at $395.55, $399.95, $399.41... — are visible right on the page.

> *"Okay, so it already knows what's training data and what's the 'exam'. That's smart."*

---

## Step 2 — Alex Trains the AI Models

Now the interesting part. Alex clicks **Train Models**.

The dashboard trains **two separate AI models** — an LSTM and a GRU — for *each* of his four stocks. That's 8 models total. A progress bar keeps him updated as each one trains.

Once training completes, the dashboard automatically evaluates every model on the **out-of-sample test data** (January 2024 onwards — data the model has never seen). Four accuracy metrics appear per model: RMSE, MSE, MAE, and MAPE.

> *"Wait — it's testing the model on data it was never trained on? So this is a real accuracy score, not just the model patting itself on the back?"*

Exactly. That's the point.

---

## Step 3 — Alex Compares LSTM vs GRU

Alex navigates to the **Model Selection** tab. For each stock, he sees a side-by-side comparison of LSTM and GRU performance, complete with actual vs predicted price charts for the test period.

<img width="1600" height="641" alt="Model Selection — LSTM vs GRU Comparison" src="https://github.com/user-attachments/assets/24fbd87a-8654-48d6-8758-1242fcb33672" />

A green **"Recommended"** badge appears on the model with the lower RMSE for each company.

> *"For Apple, the GRU is recommended. For Microsoft, LSTM wins. Interesting — one size doesn't fit all."*

Alex accepts the recommendations for all four stocks with the radio buttons. But he could easily override any of them — the choice is always his.

> *"I like that it shows me the charts and numbers, but still lets me decide. It's not just a black box telling me what to do."*

---

## Step 4 — Alex Sees the Future (Kind Of)

Back on the Home tab, Alex clicks **Run Analysis**.

The dashboard now generates a **30-day iterative price forecast** for each stock using the models he selected. He navigates to the **Forecast** tab.

<img width="1847" height="691" alt="Forecast Tab — Future Price Predictions" src="https://github.com/user-attachments/assets/89a945fe-6fd4-4f36-a884-a4cf22514286" />

For each company, Alex can see:
- The **last known price**
- The **Day-30 forecasted price**
- The **expected % change**
- A chart showing the test-period actual vs predicted prices, then the future forecast extending beyond with a **±2% confidence band**
- A vertical line marking where the forecast begins

> *"Google is expected to be up ~4% in 30 days. Tesla is more volatile — that band is much wider. Apple looks steady."*

Alex notices the confidence band is wider for riskier stocks. The tool isn't pretending to be a crystal ball — it's showing him the range of uncertainty honestly.

> *"This is exactly the kind of thing I needed. Not just a number, but a sense of how confident the model actually is."*

---

## Step 5 — Alex Builds His Portfolio

Now Alex heads to the **Portfolio** tab — the payoff of everything so far.

The dashboard has run three optimisation strategies in parallel:

- **Max Sharpe** — maximises return per unit of risk
- **Min Volatility** — minimises risk regardless of return
- **Equal Weight** — splits investment evenly (the naive baseline)

### Max Sharpe Strategy
<img width="1809" height="897" alt="Portfolio Tab — Max Sharpe" src="https://github.com/user-attachments/assets/ee97861e-2b67-4e4d-9089-30eaa7ef3833" />

> *"Whoa — it's telling me to put 42% in Microsoft, 35% in Apple, and almost nothing in Amazon? That's surprising."*

The optimiser has found that, given the AI's forecasts and historical volatility, concentrating in the lower-volatility, higher-forecasted-return stocks gives the best risk-adjusted outcome.

Alex also sees a **donut chart** of the allocation, a **shares-to-buy table** (whole numbers — no fractional share confusion), and key risk metrics:

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 1.42 |
| Annual Return | 18.3% |
| Annual Volatility | 11.2% |
| VaR 95% | -1.84% |
| CVaR 95% | -2.61% |
| Max Drawdown | -14.2% |

> *"VaR 95% of -1.84% means on my worst days, I'd expect to lose at most ~1.84% in a single day, 95% of the time. That's... actually not scary."*

### Min Volatility Strategy
<img width="1845" height="887" alt="Portfolio Tab — Min Volatility" src="https://github.com/user-attachments/assets/e09989d1-7d7d-4140-bf15-7bd972e5333d" />

### Equal Weight Strategy
<img width="1866" height="887" alt="Portfolio Tab — Equal Weight" src="https://github.com/user-attachments/assets/5b31a216-df9d-44db-8e96-39bb84c6478f" />

Alex compares all three strategies side by side. The **Return Correlation Matrix** at the bottom shows him that MSFT and GOOGL are highly correlated — diversifying between them doesn't reduce risk as much as adding a low-correlation stock like JNJ.

> *"Next time I'll add a healthcare stock to reduce correlation. I'm already thinking like a portfolio manager."*

---

## Step 6 — Alex Reviews the Track Record

Finally, Alex checks the **Performance** tab.

<img width="1823" height="834" alt="Performance Tab — Efficient Frontier & Backtest" src="https://github.com/user-attachments/assets/a4c973d1-4777-41a8-b2e2-3db2239bc499" />

<img width="1834" height="399" alt="Performance Tab — Strategy Comparison" src="https://github.com/user-attachments/assets/9e8eb0cd-56cf-4ae2-a504-768d98b339e2" />

On the left, the **Efficient Frontier** curve shows every possible portfolio combination of his four stocks. The Max Sharpe ★ and Min Volatility ◆ points are marked — he can see exactly where his chosen portfolio sits relative to all the alternatives.

On the right, a **historical backtest** compares:
- The AI-optimised portfolio (blue, filled)
- An equal-weight baseline (orange, dashed)

Both are normalised to 100 at the start. The optimised portfolio outperforms the naive baseline over the backtest window.

The **Strategy Comparison Table** puts numbers to the three approaches side by side. The **Weight Breakdown bar chart** shows how each strategy distributes capital across tickers.

> *"The Max Sharpe portfolio beat equal-weight by about 8 percentage points over 5 years. That's real money over time."*

Alex saves his analysis and takes a screenshot to share with his study group.

> *"I came in knowing nothing. I leave with a concrete, evidence-based plan for how to invest my ₹15,000 — and an understanding of* why *the numbers say what they say. That's what Invest Infinity gave me."*

---

## 🔬 About & Methodology

<img width="1825" height="845" alt="About Tab — Methodology" src="https://github.com/user-attachments/assets/e234e16e-5537-41e7-a5be-18d6dc3a5dd4" />

---

## Features (Implemented)

- **Live data ingestion** via `yfinance` — auto-adjusted OHLCV for any listed ticker
- **Feature engineering** — 9 input features: OHLCV + RSI-14, MACD, Bollinger %B, EMA ratio
- **Strict time-based train/test split** — training data ends 2023-12-31; test data starts 2024-01-01; no random shuffling, no data leakage
- **Dual model training** — one LSTM and one GRU per selected company (e.g. 4 companies = 8 models)
- **Model comparison** — RMSE, MSE, MAE, MAPE computed on out-of-sample test data only
- **Smart recommendation** — auto-badges the lower-RMSE model as "Recommended" per company
- **Per-company model selection** — user can accept the recommendation or override with a radio button
- **30-day iterative future forecast** — sliding-window multi-step prediction with ±2% confidence band
- **Portfolio optimisation** — three strategies: Max Sharpe, Min Volatility, Equal Weight
- **Expected returns** — blended 60% LSTM/GRU forecast + 40% historical mean (reduces noise)
- **Covariance estimation** — Ledoit-Wolf shrinkage for stable matrices with small samples
- **Risk metrics** — VaR 95%, CVaR 95%, Maximum Drawdown
- **Discrete share allocation** — whole-share counts for a user-defined investment amount
- **Efficient frontier visualisation** — curve with Max Sharpe and Min Vol portfolio markers
- **Historical backtest** — portfolio vs equal-weight baseline, normalised to 100
- **Return correlation matrix** heatmap
- **Multilingual UI** — English, Hindi (हिन्दी), French (Français)
- **White-background professional UI** — fixed top navigation, black action buttons
- **Configurable forecast horizon** — 7 to 90 days via slider
- **Optional login** — users can save analyses to session history

---

## Architecture

### System Architecture
<img width="1189" height="934" alt="System Architecture" src="https://github.com/user-attachments/assets/29d8bada-ff07-4684-b200-309b293f5097" />

### Data Flow Diagram
<img width="648" height="944" alt="Data Flow Diagram" src="https://github.com/user-attachments/assets/d30dd66c-8325-46ef-987d-92d362d8df4c" />

### LSTM vs GRU Architecture Comparison
<img width="782" height="842" alt="LSTM vs GRU Architecture" src="https://github.com/user-attachments/assets/93f141de-138b-4b07-a37d-d45e1d73b0d2" />

---

## Folder Structure

```
invest_infinity/
├── app.py                          # Main Streamlit dashboard
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variable template
├── README.md                       # This file
│
├── model/
│   ├── __init__.py
│   ├── builders.py                 # build_lstm() and build_gru() functions
│   ├── trainer.py                  # train_model(), evaluate_model(), forecast_future()
│   └── portfolio_optimizer.py      # optimise(), backtest(), build_price_matrix()
│
├── utils/
│   ├── __init__.py
│   ├── data_fetcher.py             # yfinance download + time_split()
│   └── features.py                 # compute_features() — RSI, MACD, Bollinger, EMA
│
├── data/
│   └── stocks_data.csv             # CSV fallback (used if yfinance unreachable)
│
├── models/                         # Auto-created — saved .keras + .pkl scaler files
│   ├── AAPL_LSTM.keras
│   ├── AAPL_LSTM_scaler.pkl
│   ├── AAPL_GRU.keras
│   └── ...
│
├── assets/
│   └── logo.png                    # App logo (optional)
│
└── screenshots/
    ├── 01_home_dashboard.png
    ├── 02_model_selection.png
    ├── 03_forecast.png
    ├── 04_portfolio_maxsharpe.png
    ├── 05_portfolio_minvol.png
    ├── 06_portfolio_equalweight.png
    ├── 07_performance.png
    └── 08_about.png
```

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Language** | Python 3.9+ | Core implementation |
| **Dashboard** | Streamlit 1.28+ | Web UI, tab navigation, controls |
| **Deep Learning** | TensorFlow / Keras 2.10+ | LSTM and GRU model training |
| **Data** | yfinance 0.2.40+ | Live OHLCV market data (primary) |
| **Data processing** | Pandas, NumPy | Data manipulation and sequence building |
| **ML utilities** | scikit-learn | MinMaxScaler, RMSE/MAE metrics |
| **Portfolio optimisation** | PyPortfolioOpt 1.5.5+ | Efficient Frontier, Ledoit-Wolf covariance |
| **Convex optimisation** | CVXPY | Backend solver for PyPortfolioOpt |
| **Visualisation** | Plotly | Interactive charts (line, donut, heatmap, bar) |
| **Internationalisation** | Custom TRANSLATIONS dict | English / Hindi / French UI |

---

## Setup — Python 3.11.8 Environment

### 1. Install Python 3.11.8
Download and install Python 3.11.8 from the official website. Make sure to check **"Add Python to PATH"** during installation.

### 2. Clone the repository

```bash
git clone https://github.com/<your-username>/invest-infinity.git
cd invest-infinity
```

### 3. Create a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Requirements
```bash
pip install -r requirements.txt
```

### 5. Run the dashboard
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501` in your browser.

---

## Usage

### Workflow (5 steps in the app)

**1. Select Stocks**
- Open the **Control Panel** on the right side of the Home tab
- Select 5–10 stocks from the multiselect dropdown
- Stocks marked `[Trending]` are sorted to the top

**2. Fetch Data**
- Choose a historical data period (1y / 2y / 3y / 5y / 10y)
- Set your total investment amount and risk-free rate
- Click **Fetch Data** — downloads live OHLCV from Yahoo Finance

**3. Train Models**
- Click **Train Models**
- Trains one LSTM and one GRU per selected company
- Progress bar shows per-company status
- Evaluation metrics (RMSE, MSE, MAE, MAPE) are computed automatically on the test set

**4. Review & Select Models**
- Navigate to the **Model Selection** tab
- Compare LSTM vs GRU performance graphs and metrics for each company
- The recommended model is auto-highlighted in green
- Use the radio button to confirm or override the selection per company

**5. Run Analysis**
- Click **Run Analysis** on the Home tab
- Generates forecasts using your chosen models and forecast horizon
- Optimises portfolio allocation using the Efficient Frontier
- Explore results in **Forecast**, **Portfolio**, and **Performance** tabs

---

## Model Details

### Time-based Split (strict, no data leakage)

| Period | Date Range | Purpose |
|--------|-----------|---------|
| Training | Start of data → 2023-12-31 | Model fitting |
| Validation | Last 20% of training rows | EarlyStopping monitor |
| **Test** | **2024-01-01 → present** | **Out-of-sample evaluation only** |

Key properties:
- `MinMaxScaler` is **fitted on training data only** — never re-fitted on test data
- Test-period predictions use **true sliding windows** (genuine one-step-ahead, no future lookahead)
- Metrics (RMSE, MSE, MAE, MAPE) are computed **exclusively on the test period**

### Input Features (9 total)

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | Open | Daily open price |
| 1 | High | Daily high |
| 2 | Low | Daily low |
| 3 | **Close** | **Prediction target** |
| 4 | Volume | Trading volume |
| 5 | RSI_14 | Relative Strength Index (Wilder, 14-period) |
| 6 | MACD | EMA12 − EMA26 (trend / momentum) |
| 7 | Bollinger_B | %B position within 20-period Bollinger Bands |
| 8 | EMA_ratio | Close / EMA50 (trend strength indicator) |

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Timesteps | 60 | ~3 months of trading days; captures medium-term trends |
| LSTM / GRU units | 128 → 64 | Sufficient capacity; halved in layer 2 to force abstraction |
| Dropout | 0.30 | Prevents co-adaptation between units |
| Dense units | 32 → 1 | Lightweight regression head |
| Batch size | 32 | Good bias-variance trade-off |
| Max epochs | 150 | EarlyStopping terminates early in practice |
| Early stopping patience | 15 | Restores best weights automatically |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=7) | Adapts to loss plateaus |
| Optimiser | Adam lr=0.001 | Adaptive, works well for RNNs |
| Loss function | MSE | Standard for regression tasks |
| Forecast method | Iterative sliding window | One-step prediction repeated N times |

---

## Portfolio Optimisation

### Expected Returns Blend

```
μ_final  =  0.60 × μ_LSTM/GRU  +  0.40 × μ_historical
```

- **LSTM/GRU component** — annualised return from the 30-day price forecast
- **Historical component** — annualised mean daily return × 252
- The blend reduces forecast noise while retaining the model's signal

### Risk Metrics

| Metric | Definition |
|--------|-----------|
| **Sharpe Ratio** | (Annual Return − Risk-Free Rate) / Annual Volatility |
| **Annual Volatility** | Daily portfolio return std × √252 |
| **VaR 95%** | 5th percentile of daily historical returns |
| **CVaR 95%** | Mean of returns below the VaR threshold (Expected Shortfall) |
| **Max Drawdown** | Largest peak-to-trough decline in cumulative portfolio value |

---

## Architecture Notes

### Why LSTM + GRU (not just one)?

Both architectures share identical hyperparameters for a fair comparison. GRU has fewer parameters (no output gate) so it trains faster and is often competitive with LSTM on shorter sequences. By training both and evaluating on held-out test data, the dashboard lets the data decide rather than assuming one architecture is always better.

### Why Ledoit-Wolf covariance shrinkage?

Sample covariance is ill-conditioned with fewer than ~250 observations per feature. Ledoit-Wolf analytically shrinks the sample matrix toward a structured estimator, producing a positive-definite matrix required by the Efficient Frontier solver and reducing estimation error by 30–50% on typical finance datasets.

### Why blend LSTM/GRU + historical returns?

Pure model-forecasted returns can be erratic, especially for stocks with limited training data. The 60/40 blend gives the model's signal the majority weight while the historical mean acts as a regulariser, preventing the optimiser from concentrating all capital in one stock due to a single noisy forecast.

---

## Limitations

- **CSV fallback data** ends 2021-12-30; the live yfinance feed is required for 2024+ test-period evaluation
- The iterative forecast only updates the Close feature at each step; other features hold their last known value, which is a reasonable approximation for horizons under ~30 days but degrades for longer horizons
- Models are retrained from scratch each session (no persistent pre-trained weights shipped with the repo)
- Portfolio optimisation assumes log-normal returns and does not account for transaction costs or taxes
- Login functionality currently uses session-state only; there is no persistent database backend

---

## Future Improvements

- Add Transformer / Attention-based architecture as a third model option
- Implement Monte Carlo simulation for wider uncertainty bands
- Add database persistence (SQLite or Supabase) for saved user analyses
- Extend the stock universe to NSE/BSE Indian market tickers via `nsepy`
- Add scenario forecasting: "What if we apply a +10% growth shock?"
- Implement anomaly detection with Z-score flags on the forecast chart

---

## Environment Variables

```bash
# .env.example
# No API keys are required for the free-tier yfinance data source.

# Optional: override default model save directory
# MODEL_DIR=models

# Optional: override default CSV fallback path
# FALLBACK_CSV=data/stocks_data.csv
```

---

## License

This project is released under the **Apache License 2.0** in compliance with the NatWest Code for Purpose Hackathon requirements.

All commits are signed off per the Developer Certificate of Origin (DCO):

```bash
git commit -s -m "feat: add GRU model builder"
```

---

## Authors

| Name | GitHub |
|------|--------|
| Saisha Verma | @Saisha0512 |
| Ritisha Sood | @RitishaSood |

---

> **Disclaimer:** This tool is for educational purposes only and does not constitute financial advice. Past performance does not guarantee future results. Alex's story is illustrative — always do your own research before investing real money.

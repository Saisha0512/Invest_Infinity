# Invest Infinity — AI Predictive Forecasting Dashboard

> **NatWest - Code for Purpose India Hackathon**
> **Group : Code Infinity**
> **Team:** Saisha Verma · Ritisha Sood

---

## Overview

**Invest Infinity** is an AI-powered stock portfolio optimisation dashboard that transforms historical market data into actionable investment forecasts. It uses dual deep learning models (LSTM and GRU) to predict future stock prices, compares model performance on strict out-of-sample test data, and then constructs an optimally weighted portfolio using the Efficient Frontier method.

**Problem it solves:** Individual investors and analysts lack accessible tools to look ahead using AI — they rely on backward-looking data alone. Invest Infinity provides honest, uncertainty-aware forecasts with clear confidence bands, baseline comparisons, and anomaly signals, enabling users to understand *what the future may look like* rather than just what already happened.

**Intended users:** Retail investors, finance students, and analysts who want AI-assisted portfolio construction without needing to write code.

---

## Live Dashboard — Screenshots

### Home Page — Control Panel

<!-- SCREENSHOT 1: Paste Image 1 (home page with control panel on right) here -->
<img width="1600" height="761" alt="image" src="https://github.com/user-attachments/assets/e0dad959-98b2-4c05-9097-087ed0adf79d" />
<img width="400" height="692" alt="image" src="https://github.com/user-attachments/assets/191cd9f8-e54d-4c66-979b-791d33165912" />
<img width="450" height="692" alt="image" src="https://github.com/user-attachments/assets/1648f57b-526d-4ccc-9391-8f5cb5a0ecb2" />
<img width="450" height="692" alt="image" src="https://github.com/user-attachments/assets/4a2d84d7-0ce5-44b1-ac6b-d19174d0df40" />
<img width="450" height="692" alt="image" src="https://github.com/user-attachments/assets/875324b0-3300-4ab5-b189-44d12ceca243" />


> *Left-side shows workflow steps, right-side shows control panel with stock picker, data period selector, investment amount, risk-free rate slider and Fetch / Train / Run buttons.*

---

### Model Selection — LSTM vs GRU Comparison

<!-- SCREENSHOT 2: Paste Image 2 (model selection page) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/02_model_selection.png`]**  
> *Side-by-side RMSE / MSE / MAE / MAPE metrics displayed for LSTM and GRU per company. Green "Recommended" badge on the better model. Actual vs Predicted charts for each model.*

---

### Forecast Tab — Future Price Predictions

<!-- SCREENSHOT 3: Paste Image 3 (forecast page) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/03_forecast.png`]**  
> *30-day shows iterative price forecast per stock with ±2% confidence band, last known price and expected % change.*

---

### Portfolio Tab — Optimal Allocation (Max Sharpe)

<!-- SCREENSHOT 4: Paste Image 4 (portfolio — max sharpe) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/04_portfolio_maxsharpe.png`]**  
> *Donut allocation shows chart, shares-to-buy table, Sharpe ratio, annual return/volatility, VaR 95%, CVaR 95%, Max Drawdown.*

### Portfolio Tab — Min Volatility Strategy

<!-- SCREENSHOT 5: Paste Image 5 (portfolio — min volatility) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/05_portfolio_minvol.png`]**

### Portfolio Tab — Equal Weight Strategy

<!-- SCREENSHOT 6: Paste Image 6 (portfolio — equal weight) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/06_portfolio_equalweight.png`]**

---

### Performance Tab — Efficient Frontier & Backtest

<!-- SCREENSHOT 7: Paste Image 7 (performance page) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/07_performance.png`]**  
> *It shows efficient frontier curve with Max Sharpe ★ and Min Vol ◆ markers, historical backtest normalised to 100 (portfolio vs equal weight), strategy comparison table, weight breakdown bar chart.*

---

### About Tab — Methodology

<!-- SCREENSHOT 8: Paste Image 8 (about page) here -->
> 📌 **[INSERT SCREENSHOT: `screenshots/08_about.png`]**  
> *It shows Data pipeline, time-based split table, model architecture table, tech stack badges, authors.*

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
- **Anomaly / risk-free rate warning** — auto-adjusts and notifies if forecast returns fall below configured rate
- **Multilingual UI** — English, Hindi (हिन्दी), French (Français)
- **White-background professional UI** — fixed top navigation, black action buttons
- **Configurable forecast horizon** — 7 to 90 days via slider
- **Optional login** — users can save analyses to session history

---

## Architecture

### System Architecture
<img width="1189" height="934" alt="image" src="https://github.com/user-attachments/assets/29d8bada-ff07-4684-b200-309b293f5097" />


### Data Flow Diagram
<img width="648" height="944" alt="image" src="https://github.com/user-attachments/assets/d30dd66c-8325-46ef-987d-92d362d8df4c" />


### LSTM vs GRU Architecture Comparison
<img width="782" height="842" alt="image" src="https://github.com/user-attachments/assets/93f141de-138b-4b07-a37d-d45e1d73b0d2" />


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

## Setup Python 3.11.8 Environment (TensorFlow & Deep Learning)
Follow these steps to set up your environment:

### 1. Install Python 3.11.8
- Download and install Python 3.11.8 from the official website  
- Make sure to check **"Add Python to PATH"** during installation

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

### 3. Install Requirements 
```bash
pip install -r requirements.txt
```

## 4. Run the dashboard
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
- Generates 30-day forecasts using your chosen models
- Optimises portfolio allocation using the Efficient Frontier
- Explore results in **Forecast**, **Portfolio**, and **Performance** tabs

---

## Model Details

### Time-based Split (strict, no data leakage)

| Period | Date Range | Purpose |
|--------|-----------|---------|
| Training | Start of data → 2023-12-31 | Model fitting and hyperparameter search |
| Validation | Last 20% of training rows | EarlyStopping monitor during fit |
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

- **LSTM/GRU component** — annualised return from the 30-day price forecast:  
  `r_ann = (avg_forecast - last_price) / last_price × (252 / 30)`
- **Historical component** — annualised mean daily return × 252
- The blend reduces forecast noise while retaining the model's signal

### Risk Metrics

| Metric | Definition |
|--------|-----------|
| **Sharpe Ratio** | (Annual Return − Risk-Free Rate) / Annual Volatility |
| **Annual Volatility** | Daily portfolio return std × √252 |
| **VaR 95%** | 5th percentile of daily historical returns (worst daily loss, 95% confidence) |
| **CVaR 95%** | Mean of returns below the VaR threshold (Expected Shortfall) |
| **Max Drawdown** | Largest peak-to-trough decline in cumulative portfolio value |

---

## Architecture Notes

### Why LSTM + GRU (not just one)?

Both architectures have the same hyperparameters to enable a fair comparison. GRU has fewer parameters (no output gate) so it trains faster — it is often competitive with LSTM on shorter sequences. By training both and evaluating on the held-out test period, the dashboard lets the data decide rather than assuming one architecture is always better.

### Why Ledoit-Wolf covariance shrinkage?

Sample covariance is ill-conditioned with fewer than ~250 observations per feature. Ledoit-Wolf analytically shrinks the sample matrix toward a structured estimator, producing a positive-definite matrix required by the Efficient Frontier solver and reducing estimation error by 30–50% on typical finance datasets.

### Why blend LSTM/GRU + historical returns?

Pure model-forecasted returns can be erratic, especially for stocks with limited training data. The 60/40 blend gives the model's signal the majority weight while the historical mean acts as a regulariser, preventing the optimiser from concentrating all capital in one stock due to a single noisy forecast.

---

## Limitations

- **CSV fallback data** ends 2021-12-30; the live yfinance feed is required for 2024+ test-period evaluation
- The iterative forecast only updates the Close feature at each step; other features (RSI, MACD, etc.) hold their last known value, which is a reasonable approximation for horizons under ~30 days but degrades for longer horizons
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
# Add any future configuration here.

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

| Name |GitHub |
|------|--------|
| Saisha Verma | @Saisha0512 |
| Ritisha Sood | @RitishaSood |

---

> **Disclaimer:** This tool is for educational purposes only and does not constitute financial advice. Past performance does not guarantee future results.

"""
portfolio_optimizer.py
----------------------
Portfolio optimisation using PyPortfolioOpt (Efficient Frontier).

Expected returns : blended — 60% LSTM-forecasted + 40% historical mean
Covariance       : Ledoit-Wolf shrinkage (stable for small N, short history)
Strategies       : Max Sharpe, Min Volatility, Equal Weight
Risk metrics     : VaR 95%, CVaR 95%, Max Drawdown
Allocation       : Discrete share allocation for a given investment amount
"""

import warnings
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.05   # annualised, used for Sharpe computation
TRADING_DAYS   = 252


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────
def build_price_matrix(df: pd.DataFrame, companies: list) -> pd.DataFrame:
    """
    Pivot a long-format OHLCV dataframe into a Close price matrix.

    Input  : df with columns Date, Company, Close
    Output : DataFrame indexed by Date, columns = companies, values = Close
    """
    sub = df[df["Company"].isin(companies)].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    matrix = sub.pivot(index="Date", columns="Company", values="Close")
    matrix = matrix.sort_index().dropna()
    # Reorder columns to match the requested order
    matrix = matrix[[c for c in companies if c in matrix.columns]]
    return matrix


# ──────────────────────────────────────────────────────────────
# Expected returns: LSTM blend
# ──────────────────────────────────────────────────────────────
def lstm_blended_returns(
    forecast_results: dict,
    price_matrix:     pd.DataFrame,
    lstm_weight:      float = 0.60
) -> pd.Series:
    """
    Annualised expected returns = 60% LSTM forecast + 40% historical mean.

    LSTM 30-day return → annualised:  r_ann = r30 * (252 / 30)
    """
    hist_mu = expected_returns.mean_historical_return(price_matrix)
    blended = {}

    for company in price_matrix.columns:
        if company in forecast_results:
            res  = forecast_results[company]
            last = res["last_known_price"]
            avg  = float(np.mean(res["forecast_prices"]))
            r30  = (avg - last) / (last + 1e-10)
            r_ann_lstm = r30 * (TRADING_DAYS / 30)
            blended[company] = (
                lstm_weight * r_ann_lstm +
                (1 - lstm_weight) * hist_mu.get(company, r_ann_lstm)
            )
        else:
            blended[company] = hist_mu.get(company, 0.0)

    return pd.Series(blended)[price_matrix.columns]


# ──────────────────────────────────────────────────────────────
# Main optimisation function
# ──────────────────────────────────────────────────────────────
def optimise(
    forecast_results:   dict,
    price_matrix:       pd.DataFrame,
    investment_amount:  float = 10_000.0,
    risk_free_rate:     float = RISK_FREE_RATE
) -> dict:
    """
    Run full portfolio optimisation.

    Returns
    -------
    {
      "max_sharpe_weights": {ticker: weight},
      "min_vol_weights":    {ticker: weight},
      "equal_weights":      {ticker: weight},
      "performance": {
          "max_sharpe": {expected_return, volatility, sharpe_ratio},
          "min_vol":    {...},
          "equal":      {...}
      },
      "risk_metrics": {var_95, cvar_95, max_drawdown},
      "discrete_allocation": {ticker: n_shares},
      "leftover":     float,
      "frontier_data": {volatilities, returns},
      "correlation_matrix": pd.DataFrame,
      "mu":  pd.Series,   (expected returns used)
      "cov": pd.DataFrame (covariance matrix used)
    }
    """
    mu  = lstm_blended_returns(forecast_results, price_matrix)
    cov = risk_models.CovarianceShrinkage(price_matrix).ledoit_wolf()

    # ── Max Sharpe ──────────────────────────────────────────
    ef1 = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef1.max_sharpe(risk_free_rate=risk_free_rate)
    ms_weights = ef1.clean_weights()
    ms_perf    = ef1.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )

    # ── Min Volatility ──────────────────────────────────────
    ef2 = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef2.min_volatility()
    mv_weights = ef2.clean_weights()
    mv_perf    = ef2.portfolio_performance(
        verbose=False, risk_free_rate=risk_free_rate
    )

    # ── Equal Weight ────────────────────────────────────────
    n = len(price_matrix.columns)
    eq_w = {c: 1.0 / n for c in price_matrix.columns}
    daily_ret  = price_matrix.pct_change().dropna()
    eq_port    = (daily_ret * pd.Series(eq_w)).sum(axis=1)
    eq_ann_ret = float(eq_port.mean() * TRADING_DAYS)
    eq_ann_vol = float(eq_port.std()  * np.sqrt(TRADING_DAYS))
    eq_sharpe  = (eq_ann_ret - risk_free_rate) / (eq_ann_vol + 1e-10)

    # ── Risk metrics (on max-sharpe portfolio) ───────────────
    w_s        = pd.Series(ms_weights)
    port_ret   = (daily_ret * w_s).sum(axis=1)
    var_95     = float(np.percentile(port_ret, 5))
    cvar_95    = float(port_ret[port_ret <= var_95].mean())
    cumul      = (1 + port_ret).cumprod()
    max_dd     = float(((cumul - cumul.cummax()) / cumul.cummax()).min())

    # ── Discrete share allocation ────────────────────────────
    latest_prices = get_latest_prices(price_matrix)
    da = DiscreteAllocation(
        ms_weights, latest_prices,
        total_portfolio_value=investment_amount
    )
    allocation, leftover = da.greedy_portfolio()

    # ── Efficient frontier curve ─────────────────────────────
    target_rets = np.linspace(float(mu.min()), float(mu.max()), 60)
    ef_vols, ef_rets = [], []
    for tr in target_rets:
        try:
            ef_tmp = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef_tmp.efficient_return(target_return=float(tr))
            p = ef_tmp.portfolio_performance(
                verbose=False, risk_free_rate=risk_free_rate
            )
            ef_rets.append(p[0])
            ef_vols.append(p[1])
        except Exception:
            pass

    # ── Correlation matrix ───────────────────────────────────
    corr = daily_ret.corr()

    return {
        "max_sharpe_weights": dict(ms_weights),
        "min_vol_weights":    dict(mv_weights),
        "equal_weights":      eq_w,
        "performance": {
            "max_sharpe": {
                "expected_return": ms_perf[0],
                "volatility":      ms_perf[1],
                "sharpe_ratio":    ms_perf[2]
            },
            "min_vol": {
                "expected_return": mv_perf[0],
                "volatility":      mv_perf[1],
                "sharpe_ratio":    mv_perf[2]
            },
            "equal": {
                "expected_return": eq_ann_ret,
                "volatility":      eq_ann_vol,
                "sharpe_ratio":    eq_sharpe
            }
        },
        "risk_metrics": {
            "var_95":       var_95,
            "cvar_95":      cvar_95,
            "max_drawdown": max_dd
        },
        "discrete_allocation": allocation,
        "leftover":            float(leftover),
        "frontier_data": {
            "volatilities": ef_vols,
            "returns":      ef_rets
        },
        "correlation_matrix": corr,
        "mu":  mu,
        "cov": pd.DataFrame(cov)
    }


# ──────────────────────────────────────────────────────────────
# Backtest
# ──────────────────────────────────────────────────────────────
def backtest(
    price_matrix: pd.DataFrame,
    weights:      dict
) -> pd.DataFrame:
    """
    Simulate portfolio performance on historical Close prices.
    Normalises all series to 100 at start for easy comparison.

    Returns DataFrame: Date, Portfolio, EqualWeight
    """
    w_s       = pd.Series(weights).reindex(price_matrix.columns).fillna(0)
    daily_ret = price_matrix.pct_change().dropna()
    port_cum  = (1 + (daily_ret * w_s).sum(axis=1)).cumprod() * 100

    n         = len(price_matrix.columns)
    eq_w      = pd.Series({c: 1.0 / n for c in price_matrix.columns})
    eq_cum    = (1 + (daily_ret * eq_w).sum(axis=1)).cumprod() * 100

    return pd.DataFrame({
        "Date":        port_cum.index,
        "Portfolio":   port_cum.values,
        "EqualWeight": eq_cum.values
    }).reset_index(drop=True)
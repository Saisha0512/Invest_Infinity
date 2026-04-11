"""
model/portfolio_optimizer.py
-----------------------------
Portfolio optimisation using PyPortfolioOpt (Efficient Frontier).

Expected returns : 60% LSTM/GRU 30-day forecast (annualised) + 40% historical mean
Covariance       : Ledoit-Wolf shrinkage
"""

import warnings
import numpy as np
import pandas as pd
from pypfopt import EfficientFrontier, risk_models, expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

warnings.filterwarnings("ignore")

RISK_FREE_RATE = 0.05
TRADING_DAYS   = 252


def build_price_matrix(df: pd.DataFrame, companies: list) -> pd.DataFrame:
    sub    = df[df["Company"].isin(companies)].copy()
    sub["Date"] = pd.to_datetime(sub["Date"])
    matrix = sub.pivot(index="Date", columns="Company", values="Close")
    return matrix.sort_index().dropna()[[c for c in companies if c in matrix.columns]]


def lstm_blended_returns(
    forecast_results: dict,
    price_matrix:     pd.DataFrame
) -> pd.Series:
    hist_mu = expected_returns.mean_historical_return(price_matrix)
    blended = {}
    for company in price_matrix.columns:
        if company in forecast_results:
            res  = forecast_results[company]
            last = res["last_known_price"]
            avg  = float(np.mean(res["forecast_prices"]))
            r30  = (avg - last) / (last + 1e-10)
            r_ann= r30 * (TRADING_DAYS / 30)
            blended[company] = 0.6 * r_ann + 0.4 * hist_mu.get(company, r_ann)
        else:
            blended[company] = hist_mu.get(company, 0.0)
    return pd.Series(blended)[price_matrix.columns]


def optimise(
    forecast_results:  dict,
    price_matrix:      pd.DataFrame,
    investment_amount: float = 10_000.0,
    risk_free_rate:    float = RISK_FREE_RATE
) -> dict:
    mu  = lstm_blended_returns(forecast_results, price_matrix)
    cov = risk_models.CovarianceShrinkage(price_matrix).ledoit_wolf()

    # BUG FIX 4: If ALL forecast returns are below the risk-free rate,
    # max_sharpe() raises an error (no portfolio on the frontier above rfr).
    # Auto-adjust rfr to just below the minimum blended return so optimisation
    # can always proceed, and flag the adjustment in the returned dict.
    rfr_was_adjusted = False
    effective_rfr    = risk_free_rate
    if float(mu.max()) <= risk_free_rate:
        effective_rfr    = float(mu.min()) - 0.001
        rfr_was_adjusted = True

    ef1 = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef1.max_sharpe(risk_free_rate=effective_rfr)
    ms_w = ef1.clean_weights()
    ms_p = ef1.portfolio_performance(verbose=False, risk_free_rate=effective_rfr)

    ef2 = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef2.min_volatility()
    mv_w = ef2.clean_weights()
    mv_p = ef2.portfolio_performance(verbose=False, risk_free_rate=effective_rfr)

    n   = len(price_matrix.columns)
    eq_w = {c: 1.0 / n for c in price_matrix.columns}
    dr   = price_matrix.pct_change().dropna()
    eq_r = (dr * pd.Series(eq_w)).sum(axis=1)
    eq_ann_ret = float(eq_r.mean() * TRADING_DAYS)
    eq_ann_vol = float(eq_r.std()  * np.sqrt(TRADING_DAYS))
    eq_sharpe  = (eq_ann_ret - effective_rfr) / (eq_ann_vol + 1e-10)

    w_s    = pd.Series(ms_w)
    port_r = (dr * w_s).sum(axis=1)
    var95  = float(np.percentile(port_r, 5))
    cvar95 = float(port_r[port_r <= var95].mean())
    cumul  = (1 + port_r).cumprod()
    maxdd  = float(((cumul - cumul.cummax()) / cumul.cummax()).min())

    latest = get_latest_prices(price_matrix)
    da     = DiscreteAllocation(ms_w, latest, total_portfolio_value=investment_amount)
    alloc, leftover = da.greedy_portfolio()

    # Efficient frontier curve
    tgt = np.linspace(float(mu.min()), float(mu.max()), 60)
    ef_vols, ef_rets = [], []
    for tr in tgt:
        try:
            ef_tmp = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef_tmp.efficient_return(target_return=float(tr))
            p = ef_tmp.portfolio_performance(verbose=False, risk_free_rate=effective_rfr)
            ef_rets.append(p[0]); ef_vols.append(p[1])
        except Exception:
            pass

    corr = dr.corr()

    return {
        "max_sharpe_weights": dict(ms_w),
        "min_vol_weights":    dict(mv_w),
        "equal_weights":      eq_w,
        "performance": {
            "max_sharpe": {"expected_return": ms_p[0], "volatility": ms_p[1], "sharpe_ratio": ms_p[2]},
            "min_vol":    {"expected_return": mv_p[0], "volatility": mv_p[1], "sharpe_ratio": mv_p[2]},
            "equal":      {"expected_return": eq_ann_ret, "volatility": eq_ann_vol, "sharpe_ratio": eq_sharpe},
        },
        "risk_metrics": {"var_95": var95, "cvar_95": cvar95, "max_drawdown": maxdd},
        "discrete_allocation": alloc,
        "leftover":            float(leftover),
        "frontier_data":       {"volatilities": ef_vols, "returns": ef_rets},
        "correlation_matrix":  corr,
        # BUG FIX 4: expose adjustment flag so app.py warning banner works
        "rfr_was_adjusted":    rfr_was_adjusted,
        "effective_rfr":       effective_rfr,
    }


def backtest(price_matrix: pd.DataFrame, weights: dict) -> pd.DataFrame:
    w_s      = pd.Series(weights).reindex(price_matrix.columns).fillna(0)
    dr       = price_matrix.pct_change().dropna()
    port_cum = (1 + (dr * w_s).sum(axis=1)).cumprod() * 100
    n        = len(price_matrix.columns)
    eq_cum   = (1 + (dr * (1.0 / n)).sum(axis=1)).cumprod() * 100
    return pd.DataFrame({
        "Date":        port_cum.index,
        "Portfolio":   port_cum.values,
        "EqualWeight": eq_cum.values
    }).reset_index(drop=True)
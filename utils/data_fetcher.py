"""
utils/data_fetcher.py
---------------------
PRIMARY data source: yfinance (live, auto-adjusted OHLCV).
FALLBACK: local CSV if network is unavailable.

Time-based split strategy (strict, no leakage):
  TRAIN : data up to and including TRAIN_CUT_DATE  (default 2023-12-31)
  TEST  : data from  TEST_START_DATE onwards        (default 2024-01-01)

When the CSV fallback is used (data ends 2021-12-30):
  TRAIN : up to 2019-12-31
  TEST  : 2020-01-01 → 2021-12-30
"""

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ── Dates for strict temporal split ──────────────────────────
TRAIN_CUT_LIVE   = pd.Timestamp("2023-12-31")   # for live yfinance data
TEST_START_LIVE  = pd.Timestamp("2024-01-01")

TRAIN_CUT_CSV    = pd.Timestamp("2019-12-31")   # for CSV fallback
TEST_START_CSV   = pd.Timestamp("2020-01-01")

# ── Stock universe ─────────────────────────────────────────────
STOCK_UNIVERSE = {
    # Technology
    "AAPL":  {"name": "Apple Inc.",             "sector": "Technology",   "trending": True},
    "MSFT":  {"name": "Microsoft Corp.",         "sector": "Technology",   "trending": True},
    "GOOGL": {"name": "Alphabet Inc.",           "sector": "Technology",   "trending": True},
    "META":  {"name": "Meta Platforms Inc.",     "sector": "Technology",   "trending": True},
    "NVDA":  {"name": "NVIDIA Corp.",            "sector": "Technology",   "trending": True},
    "AMD":   {"name": "Advanced Micro Devices",  "sector": "Technology",   "trending": True},
    "INTC":  {"name": "Intel Corp.",             "sector": "Technology",   "trending": False},
    "CRM":   {"name": "Salesforce Inc.",         "sector": "Technology",   "trending": False},
    # Consumer / E-commerce
    "AMZN":  {"name": "Amazon.com Inc.",         "sector": "Consumer",     "trending": True},
    "TSLA":  {"name": "Tesla Inc.",              "sector": "Consumer",     "trending": True},
    "NFLX":  {"name": "Netflix Inc.",            "sector": "Consumer",     "trending": True},
    "NKE":   {"name": "Nike Inc.",               "sector": "Consumer",     "trending": False},
    # Finance
    "JPM":   {"name": "JPMorgan Chase & Co.",    "sector": "Finance",      "trending": True},
    "GS":    {"name": "Goldman Sachs Group",     "sector": "Finance",      "trending": False},
    "BAC":   {"name": "Bank of America",         "sector": "Finance",      "trending": False},
    "V":     {"name": "Visa Inc.",               "sector": "Finance",      "trending": False},
    # Healthcare
    "JNJ":   {"name": "Johnson & Johnson",       "sector": "Healthcare",   "trending": False},
    "UNH":   {"name": "UnitedHealth Group",      "sector": "Healthcare",   "trending": True},
    # Energy
    "XOM":   {"name": "Exxon Mobil Corp.",       "sector": "Energy",       "trending": False},
}

SORTED_TICKERS = (
    sorted([k for k, v in STOCK_UNIVERSE.items() if v["trending"]]) +
    sorted([k for k, v in STOCK_UNIVERSE.items() if not v["trending"]])
)


def fetch_stock_data(
    tickers:      list,
    period:       str  = "5y",
    fallback_csv: str  = None
) -> tuple:
    """
    Download OHLCV data via yfinance; fall back to CSV if unavailable.

    Returns
    -------
    df        : pd.DataFrame  — long-format OHLCV, cols: Date, Company, Open, High, Low, Close, Volume
    data_src  : str           — "yfinance" or "csv_fallback"
    train_cut : pd.Timestamp  — last training date
    test_start: pd.Timestamp  — first test date
    """
    try:
        import yfinance as yf
        frames  = []
        failed  = []
        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    period=period,
                    auto_adjust=True,
                    progress=False,
                    threads=False
                )
                if raw.empty:
                    failed.append(ticker)
                    continue
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)
                raw = raw.reset_index()
                raw = raw.rename(columns={"Adj Close": "Close", "index": "Date"})
                req = ["Date", "Open", "High", "Low", "Close", "Volume"]
                if any(c not in raw.columns for c in req):
                    failed.append(ticker)
                    continue
                raw["Company"] = ticker
                raw["Date"]    = pd.to_datetime(raw["Date"])
                raw = raw[req + ["Company"]].dropna(subset=["Close"])
                frames.append(raw)
            except Exception:
                failed.append(ticker)

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["Company", "Date"]).reset_index(drop=True)
            return df, "yfinance", TRAIN_CUT_LIVE, TEST_START_LIVE

        raise ConnectionError("All tickers failed.")

    except Exception as e:
        if fallback_csv and os.path.exists(fallback_csv):
            df = pd.read_csv(fallback_csv)
            df["Date"] = pd.to_datetime(df["Date"])
            subset = [t for t in tickers if t in df["Company"].unique()]
            if not subset:
                raise ValueError(f"Tickers {tickers} not in fallback CSV.")
            df = df[df["Company"].isin(subset)].copy()
            df = df.sort_values(["Company", "Date"]).reset_index(drop=True)
            return df, "csv_fallback", TRAIN_CUT_CSV, TEST_START_CSV
        raise


def time_split(
    df_company: pd.DataFrame,
    train_cut:  pd.Timestamp,
    test_start: pd.Timestamp
) -> tuple:
    """
    Strict time-based train/test split for a single-company dataframe.

    Returns
    -------
    train_df : rows with Date <= train_cut
    test_df  : rows with Date >= test_start
    """
    df = df_company.copy().sort_values("Date").reset_index(drop=True)
    train_df = df[df["Date"] <= train_cut].reset_index(drop=True)
    test_df  = df[df["Date"] >= test_start].reset_index(drop=True)
    return train_df, test_df
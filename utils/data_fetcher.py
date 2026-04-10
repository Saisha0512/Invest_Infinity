"""
data_fetcher.py
---------------
Fetches OHLCV data using yfinance as the PRIMARY data source.
Falls back to local CSV only if yfinance is unreachable (e.g. offline env).

Usage:
    from utils.data_fetcher import fetch_stock_data, get_trending_tickers

    df = fetch_stock_data(['AAPL', 'MSFT', 'NVDA'], period='5y')
"""

import os
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# Full stock universe — extend freely
# Sectors help the UI group / filter stocks
# trending=True → shown with 🔥 badge and sorted to top
# ──────────────────────────────────────────────────────────────
STOCK_UNIVERSE = {
    # ── Technology ──
    "AAPL":  {"name": "Apple Inc.",            "sector": "Technology",     "trending": True},
    "MSFT":  {"name": "Microsoft Corp.",        "sector": "Technology",     "trending": True},
    "GOOGL": {"name": "Alphabet Inc.",          "sector": "Technology",     "trending": True},
    "META":  {"name": "Meta Platforms Inc.",    "sector": "Technology",     "trending": True},
    "NVDA":  {"name": "NVIDIA Corp.",           "sector": "Technology",     "trending": True},
    "AMD":   {"name": "Advanced Micro Devices", "sector": "Technology",     "trending": True},
    "INTC":  {"name": "Intel Corp.",            "sector": "Technology",     "trending": False},
    "CRM":   {"name": "Salesforce Inc.",        "sector": "Technology",     "trending": False},
    "ORCL":  {"name": "Oracle Corp.",           "sector": "Technology",     "trending": False},
    # ── Consumer / E-commerce ──
    "AMZN":  {"name": "Amazon.com Inc.",        "sector": "Consumer",       "trending": True},
    "TSLA":  {"name": "Tesla Inc.",             "sector": "Consumer",       "trending": True},
    "NFLX":  {"name": "Netflix Inc.",           "sector": "Consumer",       "trending": True},
    "SHOP":  {"name": "Shopify Inc.",           "sector": "Consumer",       "trending": False},
    "NKE":   {"name": "Nike Inc.",              "sector": "Consumer",       "trending": False},
    # ── Finance ──
    "JPM":   {"name": "JPMorgan Chase & Co.",   "sector": "Finance",        "trending": True},
    "GS":    {"name": "Goldman Sachs Group",    "sector": "Finance",        "trending": False},
    "BAC":   {"name": "Bank of America Corp.",  "sector": "Finance",        "trending": False},
    "V":     {"name": "Visa Inc.",              "sector": "Finance",        "trending": False},
    "MA":    {"name": "Mastercard Inc.",        "sector": "Finance",        "trending": False},
    # ── Healthcare ──
    "JNJ":   {"name": "Johnson & Johnson",      "sector": "Healthcare",     "trending": False},
    "PFE":   {"name": "Pfizer Inc.",            "sector": "Healthcare",     "trending": False},
    "UNH":   {"name": "UnitedHealth Group",     "sector": "Healthcare",     "trending": True},
    # ── Energy ──
    "XOM":   {"name": "Exxon Mobil Corp.",      "sector": "Energy",         "trending": False},
    "CVX":   {"name": "Chevron Corp.",          "sector": "Energy",         "trending": False},
    # ── Industrials / Other ──
    "BA":    {"name": "Boeing Co.",             "sector": "Industrials",    "trending": False},
    "CAT":   {"name": "Caterpillar Inc.",       "sector": "Industrials",    "trending": False},
    "DIS":   {"name": "Walt Disney Co.",        "sector": "Entertainment",  "trending": False},
    "SBUX":  {"name": "Starbucks Corp.",        "sector": "Consumer",       "trending": False},
}

# Tickers sorted: trending first, then alphabetical
SORTED_TICKERS = (
    sorted([k for k, v in STOCK_UNIVERSE.items() if v["trending"]]) +
    sorted([k for k, v in STOCK_UNIVERSE.items() if not v["trending"]])
)


# ──────────────────────────────────────────────────────────────
# Core fetch function
# ──────────────────────────────────────────────────────────────
def fetch_stock_data(
    tickers: list,
    period: str = "5y",
    fallback_csv: str = None
) -> pd.DataFrame:
    """
    Download OHLCV data for a list of tickers using yfinance.

    Parameters
    ----------
    tickers     : list of ticker strings, e.g. ['AAPL', 'MSFT']
    period      : yfinance period string — '1y', '2y', '5y', '10y', 'max'
    fallback_csv: path to a CSV fallback if yfinance is unreachable

    Returns
    -------
    pd.DataFrame with columns: Date, Company, Open, High, Low, Close, Volume
    sorted by Company then Date, with no NaN rows in OHLCV columns.
    """
    try:
        import yfinance as yf
        frames = []
        failed = []

        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    period=period,
                    auto_adjust=True,     # adjusts for splits/dividends
                    progress=False,
                    threads=False
                )
                if raw.empty:
                    failed.append(ticker)
                    continue

                # yfinance returns MultiIndex columns when multiple tickers;
                # single ticker returns flat columns
                if isinstance(raw.columns, pd.MultiIndex):
                    raw.columns = raw.columns.get_level_values(0)

                raw = raw.reset_index()

                # Normalise column names (yfinance can return 'Adj Close' or 'Close')
                raw = raw.rename(columns={
                    "Adj Close": "Close",
                    "index":     "Date"
                })
                # Ensure required columns exist
                required = ["Date", "Open", "High", "Low", "Close", "Volume"]
                missing = [c for c in required if c not in raw.columns]
                if missing:
                    failed.append(ticker)
                    continue

                raw["Company"] = ticker
                raw["Date"]    = pd.to_datetime(raw["Date"])
                raw = raw[required + ["Company"]].dropna(subset=["Open", "High", "Low", "Close"])
                frames.append(raw)

            except Exception:
                failed.append(ticker)

        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df.sort_values(["Company", "Date"]).reset_index(drop=True)
            if failed:
                print(f"[data_fetcher] ⚠ Could not fetch: {failed}")
            return df

        # All tickers failed → use fallback
        raise ConnectionError("All yfinance downloads failed.")

    except Exception as e:
        if fallback_csv and os.path.exists(fallback_csv):
            print(f"[data_fetcher] yfinance unavailable ({e}). Loading CSV fallback.")
            df = pd.read_csv(fallback_csv)
            df["Date"] = pd.to_datetime(df["Date"])
            # Filter to only the requested tickers
            available = df["Company"].unique().tolist()
            subset = [t for t in tickers if t in available]
            if not subset:
                raise ValueError(
                    f"None of {tickers} found in fallback CSV. "
                    f"Available: {available}"
                )
            df = df[df["Company"].isin(subset)].copy()
            return df.sort_values(["Company", "Date"]).reset_index(drop=True)
        raise


def get_ticker_info(ticker: str) -> dict:
    """
    Return metadata dict for a ticker from STOCK_UNIVERSE,
    or a generic dict if not found.
    """
    return STOCK_UNIVERSE.get(ticker, {
        "name":     ticker,
        "sector":   "Unknown",
        "trending": False
    })


def get_trending_tickers() -> list:
    return [k for k, v in STOCK_UNIVERSE.items() if v["trending"]]


def get_sector_map() -> dict:
    """Return {sector: [tickers]} grouped dict."""
    sectors = {}
    for ticker, info in STOCK_UNIVERSE.items():
        s = info["sector"]
        sectors.setdefault(s, []).append(ticker)
    return sectors
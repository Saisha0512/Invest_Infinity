"""
features.py
-----------
Technical indicator computation for the LSTM input pipeline.

All indicators are computed from OHLCV columns and appended to
the dataframe.  The final FEATURE_COLS list defines exactly what
goes into the LSTM input tensor.

Features (9 total):
  0  Open
  1  High
  2  Low
  3  Close          ← prediction target (index TARGET_COL = 3)
  4  Volume
  5  RSI_14         — momentum oscillator
  6  MACD           — trend / momentum
  7  Bollinger_B    — mean-reversion position [0,1]
  8  EMA_ratio      — trend strength (Close / EMA50)
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "MACD", "Bollinger_B", "EMA_ratio"
]
TARGET_COL   = 3          # index of "Close" in FEATURE_COLS
N_FEATURES   = len(FEATURE_COLS)   # 9
WARMUP_BARS  = 50         # rows consumed by EMA50 warm-up


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicator columns to a single-company OHLCV dataframe.

    Input  : DataFrame with columns Date, Open, High, Low, Close, Volume
    Output : same dataframe + RSI_14, MACD, Bollinger_B, EMA_ratio columns,
             with the first WARMUP_BARS rows (NaN indicators) dropped.

    The dataframe is sorted by Date ascending before processing.
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    close  = df["Close"].astype(float)
    high   = df["High"].astype(float)
    low    = df["Low"].astype(float)
    volume = df["Volume"].astype(float)

    # ── RSI-14 ──────────────────────────────────────────────
    # Wilder's smoothed average (EWM with com=13)
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # ── MACD (12-26) ────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    # ── Bollinger %B (20 period, 2σ) ────────────────────────
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df["Bollinger_B"] = (close - lower) / (upper - lower + 1e-10)

    # ── EMA ratio ───────────────────────────────────────────
    # Close / EMA50 — values > 1 = above trend, < 1 = below trend
    ema50 = close.ewm(span=50, adjust=False).mean()
    df["EMA_ratio"] = close / (ema50 + 1e-10)

    # Drop warm-up rows where any indicator is NaN
    df = df.dropna(subset=["RSI_14", "MACD", "Bollinger_B", "EMA_ratio"])
    df = df.reset_index(drop=True)
    return df


def build_sequences(
    scaled_data: np.ndarray,
    timesteps:   int,
    target_col:  int = TARGET_COL
):
    """
    Create supervised learning sequences from a 2-D scaled array.

    Parameters
    ----------
    scaled_data : np.ndarray, shape (n_rows, n_features)
    timesteps   : lookback window length
    target_col  : column index of the prediction target (Close = 3)

    Returns
    -------
    X : np.ndarray, shape (n_samples, timesteps, n_features)
    y : np.ndarray, shape (n_samples,)  — next-day scaled Close
    """
    X, y = [], []
    for i in range(timesteps, len(scaled_data)):
        X.append(scaled_data[i - timesteps: i, :])   # full feature window
        y.append(scaled_data[i, target_col])          # next close (scaled)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
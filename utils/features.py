"""
utils/features.py
-----------------
Technical indicator computation for LSTM / GRU input.

9 features per timestep:
  0  Open
  1  High
  2  Low
  3  Close          <- prediction target (TARGET_COL)
  4  Volume
  5  RSI_14
  6  MACD
  7  Bollinger_B
  8  EMA_ratio
"""

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "RSI_14", "MACD", "Bollinger_B", "EMA_ratio"
]
TARGET_COL = 3      # index of "Close"
N_FEATURES = len(FEATURE_COLS)   # 9


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI_14, MACD, Bollinger_B, EMA_ratio to a single-company
    OHLCV dataframe.  Drops NaN warm-up rows.

    Input  : Date, Open, High, Low, Close, Volume
    Output : same + 4 indicator columns, NaN rows dropped
    """
    df = df.copy().sort_values("Date").reset_index(drop=True)
    close = df["Close"].astype(float)

    # RSI-14 (Wilder EWM)
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.ewm(com=13, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(com=13, min_periods=14, adjust=False).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # MACD (12-26)
    ema12       = close.ewm(span=12, adjust=False).mean()
    ema26       = close.ewm(span=26, adjust=False).mean()
    df["MACD"]  = ema12 - ema26

    # Bollinger %B (20-period, 2σ)
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    upper = sma20 + 2.0 * std20
    lower = sma20 - 2.0 * std20
    df["Bollinger_B"] = (close - lower) / (upper - lower + 1e-10)

    # EMA ratio: Close / EMA50
    ema50           = close.ewm(span=50, adjust=False).mean()
    df["EMA_ratio"] = close / (ema50 + 1e-10)

    df = df.dropna(subset=["RSI_14", "MACD", "Bollinger_B", "EMA_ratio"])
    return df.reset_index(drop=True)


def build_sequences(
    scaled_data: np.ndarray,
    timesteps:   int,
    target_col:  int = TARGET_COL
):
    """
    Sliding-window supervised sequences.

    Returns
    -------
    X : (n_samples, timesteps, n_features)
    y : (n_samples,)  — next-day scaled Close
    """
    X, y = [], []
    for i in range(timesteps, len(scaled_data)):
        X.append(scaled_data[i - timesteps: i, :])
        y.append(scaled_data[i, target_col])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
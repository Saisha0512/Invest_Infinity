"""
lstm_model.py
-------------
Refined LSTM model for per-company stock price forecasting.

Architecture
────────────
Input  → (TIMESTEPS=60, N_FEATURES=9)
LSTM-1 → 128 units, return_sequences=True,  Dropout 0.3
LSTM-2 →  64 units, return_sequences=False, Dropout 0.3
Dense  →  32 units, ReLU
Dense  →   1 unit,  Linear  (regression output)

Training details
────────────────
- Train/test split : 80% / 20% (no shuffle — time series)
- Scaler           : MinMaxScaler per company, fitted on TRAIN only
- Loss             : MSE
- Optimiser        : Adam (lr=0.001), ReduceLROnPlateau halves on plateau
- Early stopping   : patience=15, restores best weights
- Batch size       : 32
- Max epochs       : 150 (EarlyStopping terminates earlier in practice)

Forecasting
───────────
Iterative multi-step prediction over FORECAST_DAYS (default 30):
  - Start from last TIMESTEPS rows of data
  - At each step, predict next Close, slide the window, repeat
  - Only Close column is updated iteratively; other features hold
    their last known value (valid approximation for ~30-day horizon)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from utils.features import (
    compute_features, build_sequences,
    FEATURE_COLS, TARGET_COL, N_FEATURES
)

# ── Hyperparameters ────────────────────────────────────────────
TIMESTEPS     = 60
FORECAST_DAYS = 30
LSTM1_UNITS   = 128
LSTM2_UNITS   = 64
DENSE_UNITS   = 32
DROPOUT       = 0.3
BATCH_SIZE    = 32
MAX_EPOCHS    = 150
LR            = 0.001
TRAIN_RATIO   = 0.80


# ──────────────────────────────────────────────────────────────
# Model definition
# ──────────────────────────────────────────────────────────────
def build_model() -> Sequential:
    model = Sequential([
        Input(shape=(TIMESTEPS, N_FEATURES)),

        LSTM(LSTM1_UNITS, return_sequences=True),
        Dropout(DROPOUT),

        LSTM(LSTM2_UNITS, return_sequences=False),
        Dropout(DROPOUT),

        Dense(DENSE_UNITS, activation="relu"),
        Dense(1,           activation="linear")
    ])
    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


# ──────────────────────────────────────────────────────────────
# Training pipeline
# ──────────────────────────────────────────────────────────────
def train(
    df_company:  pd.DataFrame,
    company:     str,
    model_dir:   str = "models",
    verbose:     int = 0
) -> dict:
    """
    Full train pipeline for one company.

    Parameters
    ----------
    df_company : single-company OHLCV dataframe (from yfinance / CSV)
    company    : ticker string used for saving artefacts
    model_dir  : directory to save .keras model and .pkl scaler
    verbose    : Keras verbosity (0=silent, 1=progress bar)

    Returns
    -------
    dict with keys: company, model_path, scaler_path,
                    train_loss, val_loss, val_mae, epochs_run
    """
    os.makedirs(model_dir, exist_ok=True)

    # 1 ── feature engineering
    df = compute_features(df_company)
    if len(df) < TIMESTEPS + 20:
        raise ValueError(
            f"{company}: only {len(df)} usable rows after indicator warm-up "
            f"(need > {TIMESTEPS + 20})."
        )

    # 2 ── train / test split (temporal — no shuffle)
    split = int(len(df) * TRAIN_RATIO)
    train_df = df.iloc[:split]
    test_df  = df.iloc[split:]

    # 3 ── scale on TRAIN only — prevents data leakage into test / forecast
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_df[FEATURE_COLS].values)
    test_scaled  = scaler.transform(test_df[FEATURE_COLS].values)

    # 4 ── build supervised sequences
    X_train, y_train = build_sequences(train_scaled, TIMESTEPS)
    X_test,  y_test  = build_sequences(test_scaled,  TIMESTEPS)

    if len(X_train) == 0:
        raise ValueError(f"{company}: insufficient training rows.")

    # 5 ── train model
    model = build_model()
    callbacks = [
        EarlyStopping(
            monitor="val_loss", patience=15,
            restore_best_weights=True, verbose=0
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5,
            patience=7, min_lr=1e-6, verbose=0
        )
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose
    )

    # 6 ── save artefacts
    model_path  = os.path.join(model_dir, f"{company}_lstm.keras")
    scaler_path = os.path.join(model_dir, f"{company}_scaler.pkl")
    model.save(model_path)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    val_losses = history.history.get("val_loss", [float("nan")])
    val_maes   = history.history.get("val_mae",  [float("nan")])

    return {
        "company":    company,
        "model_path": model_path,
        "scaler_path":scaler_path,
        "train_loss": float(history.history["loss"][-1]),
        "val_loss":   float(min(val_losses)),
        "val_mae":    float(min(val_maes)),
        "epochs_run": len(history.history["loss"]),
    }


# ──────────────────────────────────────────────────────────────
# Forecasting pipeline
# ──────────────────────────────────────────────────────────────
def forecast(
    df_company: pd.DataFrame,
    company:    str,
    model_dir:  str = "models",
    n_days:     int = FORECAST_DAYS
) -> dict:
    """
    Generate test-period evaluation + n_days future price forecast.

    Returns
    -------
    dict with keys:
        company, last_known_price,
        forecast_prices  (list, n_days),
        forecast_dates   (list of ISO strings, n_days),
        actual_prices    (list, test period),
        predicted_prices (list, test period — model on true context),
        actual_dates     (list of ISO strings, test period),
        rmse, mape       (test-period accuracy metrics)
    """
    model_path  = os.path.join(model_dir, f"{company}_lstm.keras")
    scaler_path = os.path.join(model_dir, f"{company}_scaler.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No saved model for {company}. Please train first."
        )

    model = load_model(model_path)
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    df    = compute_features(df_company)
    n     = len(df)
    split = int(n * TRAIN_RATIO)

    # ── scale full dataset with the TRAIN-fitted scaler ──
    all_scaled = scaler.transform(df[FEATURE_COLS].values)

    # ── test-period evaluation (rolling, true context) ──
    pred_scaled   = []
    actual_scaled = []
    for i in range(split + TIMESTEPS, n):
        window = all_scaled[i - TIMESTEPS: i].reshape(1, TIMESTEPS, N_FEATURES)
        p = model.predict(window, verbose=0)[0, 0]
        pred_scaled.append(p)
        actual_scaled.append(all_scaled[i, TARGET_COL])

    def inv_close(vals: list) -> np.ndarray:
        """Inverse-transform a list of scaled Close values."""
        dummy = np.zeros((len(vals), N_FEATURES))
        dummy[:, TARGET_COL] = vals
        return scaler.inverse_transform(dummy)[:, TARGET_COL]

    actual_prices = inv_close(actual_scaled).tolist()
    pred_prices   = inv_close(pred_scaled).tolist()

    # ── test-period accuracy metrics ──
    act_arr  = np.array(actual_prices)
    pred_arr = np.array(pred_prices)
    rmse     = float(np.sqrt(np.mean((act_arr - pred_arr) ** 2)))
    mape     = float(np.mean(np.abs((act_arr - pred_arr) / (act_arr + 1e-10))) * 100)

    # ── actual dates for test period ──
    date_col = df["Date"].iloc[split + TIMESTEPS:]
    if pd.api.types.is_datetime64_any_dtype(date_col):
        actual_dates = date_col.dt.strftime("%Y-%m-%d").tolist()
    else:
        actual_dates = pd.to_datetime(date_col).dt.strftime("%Y-%m-%d").tolist()

    # ── future forecast (iterative sliding window) ──
    context = all_scaled[-TIMESTEPS:].copy()   # shape (60, 9)
    future_scaled = []

    for _ in range(n_days):
        window = context.reshape(1, TIMESTEPS, N_FEATURES)
        p = model.predict(window, verbose=0)[0, 0]
        future_scaled.append(p)
        # slide: drop oldest, append new row (only Close updated)
        new_row = context[-1].copy()
        new_row[TARGET_COL] = p
        context = np.vstack([context[1:], new_row])

    future_prices = inv_close(future_scaled).tolist()

    # ── future business-day dates ──
    last_date    = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=n_days
    ).strftime("%Y-%m-%d").tolist()

    return {
        "company":          company,
        "last_known_price": float(df["Close"].iloc[-1]),
        "forecast_prices":  future_prices,
        "forecast_dates":   future_dates,
        "actual_prices":    actual_prices,
        "predicted_prices": pred_prices,
        "actual_dates":     actual_dates,
        "rmse":             rmse,
        "mape":             mape,
    }
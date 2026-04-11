"""
model/trainer.py
----------------
Train, evaluate, and forecast using LSTM or GRU.

STRICT time-based split (no data leakage):
  - Scaler fitted ONLY on training rows
  - Test rows scaled with training scaler (no re-fit)
  - Metrics computed ONLY on test data

Key functions
─────────────
train_model()   — fit one model type for one company, save artefacts
evaluate_model()— load saved model, run test-period predictions + metrics
forecast_future()— iterative n-day forecast beyond last known date
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from tensorflow.keras.models import load_model as keras_load
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from utils.features import compute_features, build_sequences, FEATURE_COLS, TARGET_COL, N_FEATURES
from model.builders import MODEL_BUILDERS, TIMESTEPS, BATCH_SIZE, MAX_EPOCHS, PATIENCE_ES, PATIENCE_LR, FORECAST_DAYS


def _artefact_paths(company: str, model_type: str, model_dir: str) -> tuple:
    """Return (model_path, scaler_path) for a company+model combo."""
    tag = f"{company}_{model_type}"
    return (
        os.path.join(model_dir, f"{tag}.keras"),
        os.path.join(model_dir, f"{tag}_scaler.pkl"),
    )


def _inv_close(vals: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Inverse-transform a flat array of scaled Close values back to prices."""
    dummy = np.zeros((len(vals), N_FEATURES))
    dummy[:, TARGET_COL] = vals
    return scaler.inverse_transform(dummy)[:, TARGET_COL]


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────
def train_model(
    train_df:   pd.DataFrame,
    company:    str,
    model_type: str,           # "LSTM" or "GRU"
    model_dir:  str = "models",
    verbose:    int = 0
) -> dict:
    """
    Train one model on training data only (data up to TRAIN_CUT_DATE).

    Parameters
    ----------
    train_df   : single-company training DataFrame (already time-split)
    company    : ticker string
    model_type : "LSTM" or "GRU"
    model_dir  : directory for saving .keras + .pkl artefacts
    verbose    : Keras fit verbosity

    Returns
    -------
    dict: company, model_type, model_path, scaler_path,
          train_loss, val_loss, val_mae, epochs_run
    """
    os.makedirs(model_dir, exist_ok=True)

    if model_type not in MODEL_BUILDERS:
        raise ValueError(f"model_type must be one of {list(MODEL_BUILDERS)}")

    df = compute_features(train_df)

    # BUG FIX 1: Minimum rows must satisfy BOTH train_seq and val_seq having
    # at least TIMESTEPS+1 rows each after the 80/20 split.
    # train_seq needs: (TIMESTEPS+1) rows → val_split >= TIMESTEPS+1
    # val_seq   needs: (TIMESTEPS+1) rows → len(scaled)-val_split >= TIMESTEPS+1
    # With val_split = int(N*0.80): N*0.20 >= TIMESTEPS+1 → N >= (TIMESTEPS+1)/0.20
    # Minimum safe total: ceil((TIMESTEPS+1) / 0.20) + a small buffer = 310
    MIN_ROWS = int((TIMESTEPS + 1) / 0.20) + 10   # = 315
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"{company}/{model_type}: only {len(df)} usable rows after indicator "
            f"warm-up (need ≥ {MIN_ROWS} so both train and validation sequences "
            f"are non-empty with TIMESTEPS={TIMESTEPS})."
        )

    # Scale ONLY on training data — no leakage
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[FEATURE_COLS].values)

    # Use last 20% of training data as validation split (still within train period)
    val_split  = int(len(scaled) * 0.80)
    train_seq  = scaled[:val_split]
    val_seq    = scaled[val_split:]

    X_train, y_train = build_sequences(train_seq, TIMESTEPS)
    X_val,   y_val   = build_sequences(val_seq,   TIMESTEPS)

    # BUG FIX 1b: Guard against empty sequences (belt-and-suspenders)
    if len(X_train) == 0:
        raise ValueError(f"{company}/{model_type}: not enough training sequences.")
    if len(X_val) == 0:
        raise ValueError(
            f"{company}/{model_type}: validation split produced 0 sequences. "
            f"Increase training data period (currently {len(df)} rows)."
        )

    model = MODEL_BUILDERS[model_type]()
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=PATIENCE_ES,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=PATIENCE_LR, min_lr=1e-6, verbose=0),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose
    )

    model_path, scaler_path = _artefact_paths(company, model_type, model_dir)
    model.save(model_path)
    with open(scaler_path, "wb") as fh:
        pickle.dump(scaler, fh)

    return {
        "company":    company,
        "model_type": model_type,
        "model_path": model_path,
        "scaler_path":scaler_path,
        "train_loss": float(min(history.history["loss"])),
        "val_loss":   float(min(history.history.get("val_loss", [float("nan")]))),
        "val_mae":    float(min(history.history.get("val_mae",  [float("nan")]))),
        "epochs_run": len(history.history["loss"]),
    }


# ──────────────────────────────────────────────────────────────
# Evaluation on test data
# ──────────────────────────────────────────────────────────────
def evaluate_model(
    test_df:    pd.DataFrame,
    train_df:   pd.DataFrame,
    company:    str,
    model_type: str,
    model_dir:  str = "models"
) -> dict:
    """
    Load saved artefacts, predict on test_df, return metrics + chart data.

    The scaler was fitted on train_df features, so we:
      1. Re-compute features on BOTH train+test (for continuity),
         then transform with the saved scaler.
      2. Slide a 60-step window over the test portion using TRUE context.

    This is the correct evaluation protocol for time-series:
      - No re-fitting of scaler on test data.
      - Test predictions are one-step-ahead using true prior values.

    Returns
    -------
    dict: company, model_type,
          actual_dates, actual_prices, predicted_prices,
          rmse, mse, mae, mape
    """
    model_path, scaler_path = _artefact_paths(company, model_type, model_dir)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No saved model: {model_path}")

    model = keras_load(model_path)
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    # Compute features on full data (train+test combined for smooth context)
    full_df = pd.concat([train_df, test_df], ignore_index=True).sort_values("Date")
    full_df = compute_features(full_df)
    # IMPORTANT: reset_index so positional iloc and .index are aligned
    full_df = full_df.reset_index(drop=True)

    # BUG FIX 2: Compute boundary as a positional integer, not a label index.
    # After reset_index, full_df.index is 0..N-1 so .index[-1] is positional,
    # but if the mask is empty (test_start > last date) it raises IndexError.
    # Use searchsorted on the sorted Date array instead — always safe.
    test_start   = pd.to_datetime(test_df["Date"].min())
    dates_sorted = pd.to_datetime(full_df["Date"].values)
    boundary     = int(np.searchsorted(dates_sorted, test_start, side="left"))
    # Ensure boundary is far enough in that we have TIMESTEPS context rows
    boundary     = max(boundary, TIMESTEPS)

    if boundary >= len(full_df):
        raise ValueError(
            f"{company}/{model_type}: test period starts after the end of "
            f"available data. Check your train/test date split."
        )

    # Scale using training scaler (no re-fit)
    full_scaled = scaler.transform(full_df[FEATURE_COLS].values)

    # BUG FIX 3: Collect result row indices so dates, actuals, preds stay aligned.
    pred_scaled   = []
    actual_scaled = []
    result_indices = []   # positional indices into full_df for date lookup

    for i in range(boundary, len(full_scaled)):
        start = i - TIMESTEPS
        if start < 0:
            continue  # not enough context (already guarded by max() above)
        window = full_scaled[start: i].reshape(1, TIMESTEPS, N_FEATURES)
        p = model.predict(window, verbose=0)[0, 0]
        pred_scaled.append(p)
        actual_scaled.append(full_scaled[i, TARGET_COL])
        result_indices.append(i)

    actual_prices = _inv_close(np.array(actual_scaled), scaler).tolist()
    pred_prices   = _inv_close(np.array(pred_scaled),   scaler).tolist()

    # BUG FIX 3b: Use the same result_indices list for dates — guaranteed aligned
    result_dates = (
        full_df["Date"]
        .iloc[result_indices]
        .dt.strftime("%Y-%m-%d")
        .tolist()
    )

    # Metrics (test data only — no leakage)
    act  = np.array(actual_prices)
    pred = np.array(pred_prices)
    rmse = float(np.sqrt(mean_squared_error(act, pred)))
    mse  = float(mean_squared_error(act, pred))
    mae  = float(mean_absolute_error(act, pred))
    mape = float(np.mean(np.abs((act - pred) / (act + 1e-10))) * 100)

    return {
        "company":          company,
        "model_type":       model_type,
        "actual_dates":     result_dates,
        "actual_prices":    actual_prices,
        "predicted_prices": pred_prices,
        "rmse":             rmse,
        "mse":              mse,
        "mae":              mae,
        "mape":             mape,
    }


# ──────────────────────────────────────────────────────────────
# Future forecast (iterative sliding window)
# ──────────────────────────────────────────────────────────────
def forecast_future(
    full_df:    pd.DataFrame,
    company:    str,
    model_type: str,
    model_dir:  str = "models",
    n_days:     int = FORECAST_DAYS
) -> dict:
    """
    Predict the next n_days business-day close prices using the full
    available dataset as context.

    Only the Close column is updated iteratively at each step; all
    other feature columns hold their last known value (valid for ~30d).

    Returns
    -------
    dict: company, model_type, last_known_price,
          forecast_prices (list), forecast_dates (list of ISO strings)
    """
    model_path, scaler_path = _artefact_paths(company, model_type, model_dir)
    model  = keras_load(model_path)
    with open(scaler_path, "rb") as fh:
        scaler = pickle.load(fh)

    df = compute_features(full_df)
    all_scaled = scaler.transform(df[FEATURE_COLS].values)

    # Seed context: last TIMESTEPS rows of known data
    context = all_scaled[-TIMESTEPS:].copy()   # shape (60, 9)
    future_scaled = []

    for _ in range(n_days):
        window = context.reshape(1, TIMESTEPS, N_FEATURES)
        p      = model.predict(window, verbose=0)[0, 0]
        future_scaled.append(p)
        new_row              = context[-1].copy()
        new_row[TARGET_COL]  = p
        context              = np.vstack([context[1:], new_row])

    future_prices = _inv_close(np.array(future_scaled), scaler).tolist()

    # Business-day dates starting the day after the last known date
    last_date    = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = pd.bdate_range(
        start  = last_date + pd.Timedelta(days=1),
        periods= n_days
    ).strftime("%Y-%m-%d").tolist()

    return {
        "company":          company,
        "model_type":       model_type,
        "last_known_price": float(df["Close"].iloc[-1]),
        "last_known_date":  last_date.strftime("%Y-%m-%d"),
        "forecast_prices":  future_prices,
        "forecast_dates":   future_dates,
    }
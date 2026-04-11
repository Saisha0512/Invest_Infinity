"""
model/builders.py
-----------------
Modular, reproducible LSTM and GRU model builders.

Both architectures share the same hyperparameter set and output shape
so they are directly comparable on the same data.

Architecture:
  Input       : (TIMESTEPS, N_FEATURES)
  RecLayer-1  : 128 units, return_sequences=True,  Dropout 0.3
  RecLayer-2  :  64 units, return_sequences=False, Dropout 0.3
  Dense-1     :  32 units, ReLU
  Dense-2     :   1 unit,  Linear  (regression)
  Optimiser   : Adam lr=0.001
  Loss        : MSE
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam

# Fix TF random seed for reproducibility
tf.random.set_seed(42)

# ── Hyperparameters ────────────────────────────────────────────
TIMESTEPS    = 60
N_FEATURES   = 9
UNITS_1      = 128
UNITS_2      = 64
DENSE_UNITS  = 32
DROPOUT      = 0.3
LR           = 0.001
BATCH_SIZE   = 32
MAX_EPOCHS   = 150
PATIENCE_ES  = 15     # EarlyStopping patience
PATIENCE_LR  = 7      # ReduceLROnPlateau patience
FORECAST_DAYS = 30


def build_lstm(
    timesteps:  int = TIMESTEPS,
    n_features: int = N_FEATURES
) -> Sequential:
    """
    Two-layer stacked LSTM for stock price regression.

    return_sequences=True on Layer-1 so Layer-2 receives the full sequence.
    return_sequences=False on Layer-2 so Dense gets a single context vector.
    """
    model = Sequential([
        Input(shape=(timesteps, n_features)),

        LSTM(UNITS_1, return_sequences=True),
        Dropout(DROPOUT),

        LSTM(UNITS_2, return_sequences=False),
        Dropout(DROPOUT),

        Dense(DENSE_UNITS, activation="relu"),
        Dense(1,           activation="linear")
    ], name="LSTM_model")

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


def build_gru(
    timesteps:  int = TIMESTEPS,
    n_features: int = N_FEATURES
) -> Sequential:
    """
    Two-layer stacked GRU — identical hypers to LSTM for fair comparison.

    GRU has fewer parameters (no output gate) so it trains faster;
    useful to compare accuracy vs training time.
    """
    model = Sequential([
        Input(shape=(timesteps, n_features)),

        GRU(UNITS_1, return_sequences=True),
        Dropout(DROPOUT),

        GRU(UNITS_2, return_sequences=False),
        Dropout(DROPOUT),

        Dense(DENSE_UNITS, activation="relu"),
        Dense(1,           activation="linear")
    ], name="GRU_model")

    model.compile(
        optimizer=Adam(learning_rate=LR),
        loss="mean_squared_error",
        metrics=["mae"]
    )
    return model


MODEL_BUILDERS = {
    "LSTM": build_lstm,
    "GRU":  build_gru,
}
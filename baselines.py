"""
Baselines module for AQI forecasting.

Implements SARIMA and Vanilla LSTM baselines for comparison against the main ConvLSTM model.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


def sarima_order_search(
    series: pd.Series, seasonal_period: int = 12
) -> tuple[tuple[int, int, int] | None, tuple[int, int, int, int] | None, float]:
    """
    Grid search for the best SARIMA order by AIC.
    Searches (p,d,q) in {0,1,2}^3 and (P,D,Q) in {0,1}^3.
    """
    p = d = q = range(0, 3)
    P = D = Q = range(0, 2)

    best_aic = float("inf")
    best_order = None
    best_seasonal_order = None

    for pdq in itertools.product(p, d, q):
        for seasonal_pdq in itertools.product(P, D, Q):
            seasonal_order = seasonal_pdq + (seasonal_period,)
            try:
                mod = SARIMAX(
                    series,
                    order=pdq,
                    seasonal_order=seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                results = mod.fit(disp=False)
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = pdq
                    best_seasonal_order = seasonal_order
            except Exception:
                continue

    return best_order, best_seasonal_order, best_aic


def run_sarima_baseline(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    test_dates: pd.DatetimeIndex | list,
    city_positions: dict[str, tuple[int, int]],
    grid_shape: tuple[int, int],
    valid_mask: np.ndarray,
    seasonal_period: int = 12,
) -> tuple[np.ndarray, dict]:
    """
    Run per-city SARIMA baseline with auto-order selection and fallbacks.

    Trains on training data only and produces one-step-ahead forecasts
    for each test date using the actual observations iteratively.
    """
    warnings.filterwarnings("ignore")

    H, W = grid_shape
    T = len(test_dates)
    predictions = np.full((T, H, W, 1), 0.0, dtype=np.float32)
    logs = {}

    full_df = pd.concat([df_train, df_test])

    for city, (r, c) in city_positions.items():
        print(f"Fitting SARIMA for {city}...")
        train_series = df_train[city].astype(float)
        full_series = full_df[city].astype(float)

        best_order, best_seasonal, best_aic = sarima_order_search(train_series, seasonal_period)
        logs[city] = {"order": best_order, "seasonal_order": best_seasonal, "aic": best_aic}
        print(f"Best order for {city}: {best_order} x {best_seasonal}")

        preds = None

        if best_order is not None:
            try:
                mod = SARIMAX(
                    train_series,
                    order=best_order,
                    seasonal_order=best_seasonal,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False)
                res_full = res.apply(full_series)
                preds = res_full.predict(start=test_dates[0], end=test_dates[-1], dynamic=False)
            except Exception as e:
                print(f"Convergence failed for {city} with best order: {e}. Trying fallback 1...")
                best_order = None

        if best_order is None:
            try:
                mod = SARIMAX(
                    train_series,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                res = mod.fit(disp=False)
                res_full = res.apply(full_series)
                preds = res_full.predict(start=test_dates[0], end=test_dates[-1], dynamic=False)
            except Exception as e2:
                print(f"Fallback 1 failed for {city}: {e2}. Trying fallback 2...")
                try:
                    mod = SARIMAX(
                        train_series,
                        order=(1, 0, 0),
                        seasonal_order=(1, 0, 0, seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = mod.fit(disp=False)
                    res_full = res.apply(full_series)
                    preds = res_full.predict(start=test_dates[0], end=test_dates[-1], dynamic=False)
                except Exception as e3:
                    print(f"Fallback 2 failed for {city}: {e3}. Using persistence...")
                    # Persistence fallback
                    preds_list = []
                    for t in test_dates:
                        past_data = full_series.loc[:t].iloc[:-1]
                        if len(past_data) > 0:
                            preds_list.append(past_data.iloc[-1])
                        else:
                            preds_list.append(0.0)
                    preds = pd.Series(preds_list, index=test_dates)

        # Ensure correct alignment to test_dates
        preds = preds.reindex(test_dates).ffill().fillna(0)
        predictions[:, r, c, 0] = preds.values.astype(np.float32)

    warnings.filterwarnings("default")
    return predictions, logs


def _build_vanilla_lstm(seq_len: int) -> Model:
    """Builds and compiles the per-city vanilla LSTM model."""
    inputs = Input(shape=(seq_len, 1))
    x = LSTM(64, return_sequences=False)(inputs)
    x = Dropout(0.2)(x)
    x = Dense(32, activation="relu")(x)
    outputs = Dense(1, activation="linear")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse")
    return model


def predictions_to_grid_frames(
    city_predictions: dict[str, np.ndarray],
    city_positions: dict[str, tuple[int, int]],
    n_timesteps: int,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Convert per-city predictions dict to (T, H, W, 1) frames."""
    H, W = grid_shape
    frames = np.full((n_timesteps, H, W, 1), 0.0, dtype=np.float32)
    for city, preds in city_predictions.items():
        if city in city_positions:
            r, c = city_positions[city]
            frames[:, r, c, 0] = preds
    return frames


def run_vanilla_lstm_baseline(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    city_positions: dict[str, tuple[int, int]],
    grid_shape: tuple[int, int],
    valid_mask: np.ndarray,
    seq_len: int = 6,
    epochs: int = 100,
    batch_size: int = 16,
) -> tuple[np.ndarray, dict]:
    """
    Run per-city univariate Vanilla LSTM baseline.

    Each city gets its own LSTM model and standard scaler.
    """
    H, W = grid_shape
    T = len(df_test)
    city_predictions = {}
    histories = {}

    full_df = pd.concat([df_train, df_val, df_test])

    for city, (r, c) in city_positions.items():
        print(f"Training Vanilla LSTM for {city}...")
        scaler = StandardScaler()
        # Fit scaler on training data only
        scaler.fit(df_train[city].values.reshape(-1, 1))

        scaled_vals = scaler.transform(full_df[city].values.reshape(-1, 1)).flatten()

        X, y = [], []
        for i in range(len(scaled_vals) - seq_len):
            X.append(scaled_vals[i : i + seq_len])
            y.append(scaled_vals[i + seq_len])

        X = np.array(X, dtype=np.float32)[..., np.newaxis]
        y = np.array(y, dtype=np.float32)
        dates = full_df.index[seq_len:]

        # Create time-based masks to match splits exactly
        train_mask = dates.isin(df_train.index)
        val_mask = dates.isin(df_val.index)
        test_mask = dates.isin(df_test.index)

        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test = X[test_mask]

        model = _build_vanilla_lstm(seq_len)
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
        ]

        hist = model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
        )
        histories[city] = hist.history

        if len(X_test) > 0:
            preds_scaled = model.predict(X_test, verbose=0)
            preds = scaler.inverse_transform(preds_scaled).flatten()
            city_predictions[city] = preds
        else:
            city_predictions[city] = np.zeros(T, dtype=np.float32)

    predictions = predictions_to_grid_frames(city_predictions, city_positions, T, grid_shape)

    return predictions, histories

"""
Evaluation utilities for the AQI ConvLSTM forecasting pipeline.

Metrics computed
----------------
MAE     - Mean Absolute Error
RMSE    - Root Mean Squared Error
R2      - Coefficient of determination (sklearn)
MAPE    - Mean Absolute Percentage Error (%)
SMAPE   - Symmetric MAPE (%)
NSE     - Nash-Sutcliffe Efficiency  (1 = perfect, 0 = climatology mean)
Willmott_d - Willmott's Index of Agreement (0–1, higher is better)
Theil_U - Theil's U₂ statistic  (< 1 = better than persistence)
Bias    - Mean signed error (pred − true), positive = over-prediction
P95_AE  - 95th percentile of absolute errors
AQI_Category_Accuracy - fraction of samples where predicted AQI category matches

References
----------
Nash & Sutcliffe (1970) J. Hydrology 10(3):282-290
Willmott (1981) Physical Geography 2(2):184-194
Theil (1966) Applied Economic Forecasting, North-Holland
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


# India CPCB AQI breakpoints (upper bounds of each band)
AQI_BREAKPOINTS = np.array([50, 100, 200, 300, 400], dtype=np.float32)
AQI_LABELS = ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
AQI_COLORS = ("#00e400", "#92d050", "#ffff00", "#ff7e00", "#ff0000", "#7e0023")


def flatten_valid_cells(data: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Extract only valid city-cell values across all timesteps."""
    if data.ndim == 4 and data.shape[-1] == 1:
        data = data[..., 0]
    flat = []
    for i in range(data.shape[0]):
        flat.append(data[i][valid_mask])
    return np.concatenate(flat, axis=0)


def aqi_category(values: np.ndarray) -> np.ndarray:
    """Map AQI values to CPCB category indices (0=Good … 5=Severe)."""
    return np.digitize(np.asarray(values, dtype=np.float32), AQI_BREAKPOINTS, right=True)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, valid_mask: np.ndarray) -> dict:
    """
    Compute a comprehensive set of forecasting metrics over valid city cells.

    Parameters
    ----------
    y_true, y_pred : ndarray  (T, H, W, 1)
    valid_mask     : bool ndarray  (H, W)

    Returns
    -------
    dict of scalar floats plus 'true_values' and 'pred_values' arrays.
    """
    true_values = flatten_valid_cells(y_true, valid_mask)
    pred_values = flatten_valid_cells(y_pred, valid_mask)

    # Remove any NaN pairs (e.g., from non-city cells)
    mask = np.isfinite(true_values) & np.isfinite(pred_values)
    t = true_values[mask]
    p = pred_values[mask]

    mae   = float(mean_absolute_error(t, p))
    rmse  = float(np.sqrt(np.mean((t - p) ** 2)))
    r2    = float(r2_score(t, p))
    bias  = float(np.mean(p - t))
    p95   = float(np.percentile(np.abs(t - p), 95))
    mape  = float(np.mean(np.abs((t - p) / np.maximum(np.abs(t), 1.0))) * 100)
    smape = float(
        np.mean(2.0 * np.abs(p - t) / np.maximum(np.abs(t) + np.abs(p), 1.0)) * 100
    )
    cat_acc = float(np.mean(aqi_category(t) == aqi_category(p)))

    # Nash-Sutcliffe Efficiency
    ss_res = np.sum((t - p) ** 2)
    ss_tot = np.sum((t - np.mean(t)) ** 2)
    nse = float(1.0 - ss_res / max(ss_tot, 1e-10))

    # Willmott's Index of Agreement (d)
    mu_t = np.mean(t)
    num_w = np.sum((t - p) ** 2)
    den_w = np.sum((np.abs(p - mu_t) + np.abs(t - mu_t)) ** 2)
    willmott_d = float(1.0 - num_w / max(den_w, 1e-10))

    # Theil's U₂ statistic (persistence as reference)
    # U < 1 means model beats persistence; U = 0 is perfect
    persistence_err = np.sqrt(np.mean((t[1:] - t[:-1]) ** 2)) if len(t) > 1 else 1.0
    model_err = rmse
    theil_u = float(model_err / max(persistence_err, 1e-10))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Bias": bias,
        "P95_AE": p95,
        "MAPE_percent": mape,
        "SMAPE_percent": smape,
        "NSE": nse,
        "Willmott_d": willmott_d,
        "Theil_U": theil_u,
        "AQI_Category_Accuracy": cat_acc,
        "true_values": true_values,
        "pred_values": pred_values,
    }


def baseline_persistence(X_val: np.ndarray) -> np.ndarray:
    """Persistence baseline: predict last month of input sequence as next month."""
    return X_val[:, -1]


def per_city_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    city_positions: dict[str, tuple[int, int]],
) -> list[dict]:
    """
    Compute MAE, RMSE, Bias, NSE, and AQI category accuracy per city.

    Parameters
    ----------
    y_true, y_pred  : ndarray  (T, H, W, 1)
    city_positions  : dict mapping city name → (row, col)
    """
    rows = []
    for city, (r, c) in city_positions.items():
        t = y_true[:, r, c, 0]
        p = y_pred[:, r, c, 0]
        mask = np.isfinite(t) & np.isfinite(p)
        t, p = t[mask], p[mask]
        if len(t) == 0:
            continue

        ss_res = np.sum((t - p) ** 2)
        ss_tot = np.sum((t - np.mean(t)) ** 2)
        nse_city = float(1.0 - ss_res / max(ss_tot, 1e-10))

        rows.append(
            {
                "City": city,
                "MAE": float(np.mean(np.abs(t - p))),
                "RMSE": float(np.sqrt(np.mean((t - p) ** 2))),
                "Bias": float(np.mean(p - t)),
                "NSE": nse_city,
                "Mean_Actual_AQI": float(np.mean(t)),
                "Mean_Predicted_AQI": float(np.mean(p)),
                "AQI_Category_Accuracy": float(np.mean(aqi_category(t) == aqi_category(p))),
            }
        )
    return rows

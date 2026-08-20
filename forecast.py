"""
Recursive multi-step AQI forecasting with optional MC-Dropout uncertainty.

Standard forecast
-----------------
multi_step_forecast() runs the model deterministically (dropout disabled)
one month at a time, appending each predicted map back into the input
sequence to reach `steps` months ahead.

Uncertainty estimate (Monte Carlo Dropout)
------------------------------------------
mc_dropout_forecast() runs N stochastic forward passes (dropout active at
inference) and returns per-step mean and standard deviation over those
passes.  The std gives a rough confidence interval around each forecast.

Reference
---------
Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation: Representing
Model Uncertainty in Deep Learning." ICML 2016.
https://arxiv.org/abs/1506.02142
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from train import inverse_scale


def multi_step_forecast(
    model,
    recent_seq: np.ndarray,
    steps: int,
    scaler,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Deterministic recursive multi-step forecast.

    Parameters
    ----------
    model       : trained Keras model
    recent_seq  : ndarray (seq_len, H, W, 1) — scaled, last observed window
    steps       : number of future months to forecast
    scaler      : fitted scaler with inverse_transform
    valid_mask  : bool ndarray (H, W)

    Returns
    -------
    preds_real  : ndarray (steps, H, W, 1)  — AQI values in original scale
    preds_scaled: ndarray (steps, H, W, 1)  — normalized values
    """
    preds_scaled: list[np.ndarray] = []
    seq = recent_seq.copy()

    for _ in range(steps):
        pred = model.predict(seq[np.newaxis], verbose=0)[0]   # (H, W, 1)
        pred = np.clip(pred, 0.0, 1.5)                        # allow small overshoot
        pred[..., 0][~valid_mask] = 0.0
        preds_scaled.append(pred)
        seq = np.concatenate([seq[1:], pred[np.newaxis, ...]], axis=0)

    preds_scaled_arr = np.array(preds_scaled, dtype=np.float32)
    preds_real = inverse_scale(preds_scaled_arr, scaler)
    preds_real = np.maximum(preds_real, 0.0)
    preds_real[..., 0] = np.where(
        valid_mask[np.newaxis, :, :], preds_real[..., 0], np.nan
    )
    return preds_real, preds_scaled_arr


def mc_dropout_forecast(
    model,
    recent_seq: np.ndarray,
    steps: int,
    scaler,
    valid_mask: np.ndarray,
    n_passes: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Monte Carlo Dropout uncertainty-aware recursive forecast.

    Runs `n_passes` stochastic forward passes (dropout ACTIVE) and returns
    the mean and standard deviation across passes for each forecast step.

    Parameters
    ----------
    model       : trained Keras model (must have Dropout layers)
    recent_seq  : ndarray (seq_len, H, W, 1) — scaled window
    steps       : forecast horizon (months)
    scaler      : fitted scaler
    valid_mask  : bool ndarray (H, W)
    n_passes    : number of stochastic samples

    Returns
    -------
    mean_real : ndarray (steps, H, W, 1)  — posterior mean AQI
    std_real  : ndarray (steps, H, W, 1)  — posterior std (uncertainty)
    """
    # Build a callable that keeps dropout active at inference
    @tf.function
    def stochastic_predict(x):
        return model(x, training=True)

    all_runs: list[np.ndarray] = []

    for _ in range(n_passes):
        preds_scaled: list[np.ndarray] = []
        seq = recent_seq.copy()
        for _ in range(steps):
            inp = tf.constant(seq[np.newaxis], dtype=tf.float32)
            pred = stochastic_predict(inp).numpy()[0]
            pred = np.clip(pred, 0.0, 1.5)
            pred[..., 0][~valid_mask] = 0.0
            preds_scaled.append(pred)
            seq = np.concatenate([seq[1:], pred[np.newaxis, ...]], axis=0)
        all_runs.append(np.array(preds_scaled, dtype=np.float32))

    all_runs_arr = np.stack(all_runs, axis=0)        # (n_passes, steps, H, W, 1)
    mean_scaled = all_runs_arr.mean(axis=0)          # (steps, H, W, 1)
    std_scaled  = all_runs_arr.std(axis=0)

    mean_real = inverse_scale(mean_scaled, scaler)
    std_real  = inverse_scale(std_scaled, scaler)    # propagate through linear scaler
    mean_real = np.maximum(mean_real, 0.0)
    std_real  = np.maximum(std_real, 0.0)

    mean_real[..., 0] = np.where(
        valid_mask[np.newaxis], mean_real[..., 0], np.nan
    )
    std_real[..., 0] = np.where(
        valid_mask[np.newaxis], std_real[..., 0], np.nan
    )
    return mean_real, std_real

"""
Training utilities: scaling, augmentation, and model fitting.

Key design decision — per-cell normalization
---------------------------------------------
Cities have wildly different AQI ranges (Delhi ~200, Thiruvananthapuram ~62).
A single global MinMaxScaler collapses all city differences into one range and
makes the model chase high-AQI northern cities. Instead we fit one StandardScaler
per spatial cell (each city position in the 4×5 grid), which centres and scales
each city independently.  This dramatically improves convergence for low-AQI
southern cities and produces a balanced loss signal.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau


# ---------------------------------------------------------------------------
# Per-cell scaler (one StandardScaler per spatial position)
# ---------------------------------------------------------------------------

class PerCellScaler:
    """
    Fit one StandardScaler for every (row, col) grid cell.

    Works on tensors shaped (T, H, W, C) where T = timesteps, C = channels.
    For AQI C=1 so we simply standardise each city position independently.
    """

    def __init__(self):
        self._scalers: dict[tuple[int, int], StandardScaler] = {}
        self.H = 0
        self.W = 0

    def fit(self, data: np.ndarray) -> "PerCellScaler":
        """data: (T, H, W, 1)"""
        self.H, self.W = data.shape[1], data.shape[2]
        for r in range(self.H):
            for c in range(self.W):
                vals = data[:, r, c, 0].reshape(-1, 1)
                sc = StandardScaler()
                # Only fit on finite values
                finite_mask = np.isfinite(vals.ravel())
                if finite_mask.sum() > 0:
                    sc.fit(vals[finite_mask].reshape(-1, 1))
                    # Override mean_ / scale_ if all zero (empty cell)
                    if sc.scale_ == 0 or not np.isfinite(sc.scale_):
                        sc.mean_ = np.array([0.0])
                        sc.scale_ = np.array([1.0])
                else:
                    sc.mean_ = np.array([0.0])
                    sc.scale_ = np.array([1.0])
                self._scalers[(r, c)] = sc
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        out = data.copy().astype(np.float32)
        for r in range(self.H):
            for c in range(self.W):
                sc = self._scalers[(r, c)]
                vals = out[..., r, c, 0].ravel().reshape(-1, 1)
                out[..., r, c, 0] = sc.transform(vals).ravel().reshape(
                    out[..., r, c, 0].shape
                )
        return out

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        out = data.copy().astype(np.float32)
        for r in range(self.H):
            for c in range(self.W):
                sc = self._scalers[(r, c)]
                vals = out[..., r, c, 0].ravel().reshape(-1, 1)
                out[..., r, c, 0] = sc.inverse_transform(vals).ravel().reshape(
                    out[..., r, c, 0].shape
                )
        return out

    def fit_transform(self, data: np.ndarray) -> np.ndarray:
        return self.fit(data).transform(data)


# ---------------------------------------------------------------------------
# Public scaling API (called from experiments.py)
# ---------------------------------------------------------------------------

def scale_data(X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Fit a PerCellScaler on training inputs+targets, then transform all splits.

    Returns
    -------
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler
    """
    scaler = PerCellScaler()
    # Fit on all training frames: input sequences + targets
    # Stack along time axis: shape (T*seq+T, H, W, 1)
    fit_data = np.concatenate(
        [X_train.reshape(-1, *X_train.shape[2:]), y_train], axis=0
    )
    scaler.fit(fit_data)

    X_train_s = scaler.transform(X_train.reshape(-1, *X_train.shape[2:])).reshape(X_train.shape)
    X_val_s   = scaler.transform(X_val.reshape(-1, *X_val.shape[2:])).reshape(X_val.shape)
    X_test_s  = scaler.transform(X_test.reshape(-1, *X_test.shape[2:])).reshape(X_test.shape)
    y_train_s = scaler.transform(y_train)
    y_val_s   = scaler.transform(y_val)
    y_test_s  = scaler.transform(y_test)

    return X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler


def transform_with_scaler(data: np.ndarray, scaler: PerCellScaler) -> np.ndarray:
    """Scale an arbitrary (T, H, W, 1) tensor with a fitted PerCellScaler."""
    return scaler.transform(data)


def inverse_scale(data: np.ndarray, scaler: PerCellScaler) -> np.ndarray:
    """Invert scaling for any (T, H, W, 1) tensor."""
    return scaler.inverse_transform(data)


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    enabled: bool = True,
    copies: int = 2,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Expand the training set with noisy variants of each sample.

    For each copy we add independent Gaussian noise (mean 0, std noise_std)
    to the input sequence only (not the target).  Noise is in standardised
    units, so noise_std=0.05 equals ~5% of one standard deviation.
    """
    if not enabled or copies <= 0:
        return X_train, y_train

    rng = np.random.default_rng(seed)
    aug_X = [X_train]
    aug_y = [y_train]

    for i in range(copies):
        noise = rng.normal(0.0, noise_std, size=X_train.shape).astype(np.float32)
        noisy_X = X_train + noise
        aug_X.append(noisy_X)
        aug_y.append(y_train)

    return np.concatenate(aug_X, axis=0), np.concatenate(aug_y, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 250,
    batch_size: int = 8,
    checkpoint_path: str | None = None,
    early_stopping_patience: int = 40,
    lr_patience: int = 15,
):
    """
    Fit the ConvLSTM model with EarlyStopping, ReduceLROnPlateau, and
    optional ModelCheckpoint.

    Parameters
    ----------
    checkpoint_path : str or None
        If provided, the best-weights file is saved here during training.
    """
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.6,
            patience=lr_patience,
            min_lr=5e-6,
            verbose=1,
        ),
    ]
    if checkpoint_path is not None:
        callbacks.append(
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
                verbose=0,
            )
        )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )
    return history

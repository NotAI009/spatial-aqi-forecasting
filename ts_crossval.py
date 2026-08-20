"""
Time Series Cross-Validation module for the ConvLSTM AQI forecasting pipeline.

Implements an expanding-window cross-validation strategy.
"""
from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from statsmodels.tsa.statespace.sarimax import SARIMAX

from data_utils import create_sequences
from train import PerCellScaler, augment_training_data, train_model
from model import build_convlstm
from evaluate import compute_metrics, baseline_persistence

try:
    from experiments import PALETTE
except ImportError:
    PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def run_time_series_cv(
    frames: np.ndarray,
    valid_mask: np.ndarray,
    df: pd.DataFrame,
    city_positions: dict[str, tuple[int, int]],
    seq_len: int = 6,
    min_train_months: int = 48,
    step_size: int = 6,
    epochs: int = 100,
    batch_size: int = 8,
    use_augmentation: bool = True,
    run_baselines: bool = True,
) -> pd.DataFrame:
    """
    Run expanding-window time series cross-validation.
    """
    X_all, y_all = create_sequences(frames, seq_len)
    
    T_total = len(frames)
    cv_results = []
    fold = 1
    
    total_folds = max(0, (T_total - min_train_months + step_size - 1) // step_size)

    for train_end in range(min_train_months, T_total, step_size):
        test_end = min(train_end + step_size, T_total)
        
        train_idx_end = train_end - seq_len
        test_idx_start = train_end - seq_len
        test_idx_end = test_end - seq_len
        
        if train_idx_end <= 0:
            continue
        if test_idx_end <= test_idx_start:
            break
            
        val_size = max(1, int(0.1 * train_idx_end))
        train_idx_split = train_idx_end - val_size
        
        X_train = X_all[:train_idx_split]
        y_train = y_all[:train_idx_split]
        
        X_val = X_all[train_idx_split:train_idx_end]
        y_val = y_all[train_idx_split:train_idx_end]
        
        X_test = X_all[test_idx_start:test_idx_end]
        y_test = y_all[test_idx_start:test_idx_end]
        
        print(f"\nFold {fold}/{total_folds}: training ConvLSTM on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Scale data
        scaler = PerCellScaler()
        fit_data = np.concatenate([X_train.reshape(-1, *X_train.shape[2:]), y_train], axis=0)
        scaler.fit(fit_data)
        
        X_train_s = scaler.transform(X_train.reshape(-1, *X_train.shape[2:])).reshape(X_train.shape)
        X_val_s = scaler.transform(X_val.reshape(-1, *X_val.shape[2:])).reshape(X_val.shape)
        X_test_s = scaler.transform(X_test.reshape(-1, *X_test.shape[2:])).reshape(X_test.shape)
        y_train_s = scaler.transform(y_train)
        y_val_s = scaler.transform(y_val)
        
        if use_augmentation:
            X_train_s, y_train_s = augment_training_data(X_train_s, y_train_s)
            
        tf.keras.backend.clear_session()
        model = build_convlstm(input_shape=X_train_s.shape[1:], valid_mask=valid_mask)
        
        train_model(
            model, X_train_s, y_train_s, X_val_s, y_val_s,
            epochs=epochs, batch_size=batch_size, early_stopping_patience=20, lr_patience=10
        )
        
        y_pred_s = model.predict(X_test_s, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_s)
        
        metrics = compute_metrics(y_test, y_pred, valid_mask)
        
        cv_results.append({
            "Fold": fold,
            "Model": "ConvLSTM",
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"],
            "R2": metrics["R2"],
            "NSE": metrics["NSE"],
            "Train_Size": train_idx_end,
            "Test_Size": len(X_test)
        })
        
        if run_baselines:
            # Persistence
            y_pred_pers = baseline_persistence(X_test)
            metrics_pers = compute_metrics(y_test, y_pred_pers, valid_mask)
            cv_results.append({
                "Fold": fold,
                "Model": "Persistence",
                "MAE": metrics_pers["MAE"],
                "RMSE": metrics_pers["RMSE"],
                "R2": metrics_pers["R2"],
                "NSE": metrics_pers["NSE"],
                "Train_Size": train_idx_end,
                "Test_Size": len(X_test)
            })
            
            # SARIMA
            warnings.filterwarnings("ignore")
            sarima_pred_frame = np.zeros_like(y_test)
            
            for city, (r, c) in city_positions.items():
                city_series = df[city].values
                
                train_series = city_series[:train_end]
                full_series = city_series[:test_end]
                
                try:
                    s_model = SARIMAX(train_series, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
                    res = s_model.fit(disp=False)
                    res_full = res.apply(full_series)
                    preds = res_full.predict(start=train_end, end=test_end-1, dynamic=False)
                    
                    for test_step in range(len(X_test)):
                        sarima_pred_frame[test_step, r, c, 0] = preds[train_end + test_step]
                except Exception:
                    # Fallback to persistence if SARIMA fails to converge or errors
                    for test_step in range(len(X_test)):
                        t_target = test_idx_start + test_step + seq_len
                        sarima_pred_frame[test_step, r, c, 0] = city_series[t_target - 1]
                        
            metrics_sarima = compute_metrics(y_test, sarima_pred_frame, valid_mask)
            cv_results.append({
                "Fold": fold,
                "Model": "SARIMA",
                "MAE": metrics_sarima["MAE"],
                "RMSE": metrics_sarima["RMSE"],
                "R2": metrics_sarima["R2"],
                "NSE": metrics_sarima["NSE"],
                "Train_Size": train_idx_end,
                "Test_Size": len(X_test)
            })
            
        fold += 1

    return pd.DataFrame(cv_results)


def summarize_cv_results(cv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize cross-validation metrics across all folds by model.
    """
    summary = cv_df.groupby("Model")[["MAE", "RMSE", "R2", "NSE"]].agg(["mean", "std"])
    summary.columns = [f"{col[0]}_{col[1]}" for col in summary.columns]
    
    # Format as mean ± std for readability
    formatted = pd.DataFrame(index=summary.index)
    for col in ["MAE", "RMSE", "R2", "NSE"]:
        formatted[col] = summary.apply(
            lambda row: f"{row[col + '_mean']:.2f} ± {row[col + '_std']:.2f}", 
            axis=1
        )
    return formatted


def fig18_tscv_folds(cv_df: pd.DataFrame, path: str, show: bool = False):
    """
    Generate Figure 18: Expanding-Window Time Series Cross-Validation.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Expanding-Window Time Series Cross-Validation", fontsize=14, y=1.02)
    
    models = cv_df["Model"].unique()
    model_colors = {model: PALETTE[i % len(PALETTE)] for i, model in enumerate(models)}
    
    # Subplot 1: Line plot of MAE per fold
    ax1 = axes[0]
    for model in models:
        model_data = cv_df[cv_df["Model"] == model]
        ax1.plot(
            model_data["Fold"], 
            model_data["MAE"], 
            marker="o", 
            linestyle="-", 
            label=model, 
            color=model_colors[model]
        )
    
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("MAE")
    ax1.set_title("MAE across Folds")
    ax1.set_xticks(cv_df["Fold"].unique())
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend()
    
    # Subplot 2: Box plot of MAE distribution
    ax2 = axes[1]
    plot_data = [cv_df[cv_df["Model"] == model]["MAE"].values for model in models]
    
    bplot = ax2.boxplot(
        plot_data, 
        patch_artist=True, 
        labels=models,
        medianprops=dict(color="black")
    )
    
    for patch, model in zip(bplot["boxes"], models):
        patch.set_facecolor(model_colors[model])
        patch.set_alpha(0.7)
        
    ax2.set_ylabel("MAE")
    ax2.set_title("MAE Distribution")
    ax2.grid(True, linestyle="--", alpha=0.6, axis="y")
    
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    
    if show:
        plt.show()
    plt.close()

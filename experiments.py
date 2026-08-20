"""
20-city monthly AQI forecasting — ConvLSTM spatial-temporal pipeline.

Run the full experiment:
    python experiments.py --no-show-plots

Quick smoke test (3 epochs):
    python experiments.py --epochs 3 --forecast-steps 2 --no-show-plots

All outputs land in  outputs/  (or --output-dir).
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import tensorflow as tf
from scipy import stats

from data_utils import CITY_GRID_LAYOUT, build_grid, create_sequences, load_city_data, train_val_test_split
from evaluate import AQI_BREAKPOINTS, AQI_LABELS, AQI_COLORS, aqi_category, baseline_persistence, compute_metrics, per_city_metrics, flatten_valid_cells
from forecast import mc_dropout_forecast, multi_step_forecast
from model import build_convlstm
from train import augment_training_data, inverse_scale, scale_data, train_model, transform_with_scaler
from baselines import run_sarima_baseline, run_vanilla_lstm_baseline, run_seasonal_naive_baseline
from ablation import run_ablation_study
from dm_test import run_dm_tests, fig_dm_results
from ts_crossval import run_time_series_cv, summarize_cv_results, fig18_tscv_folds

# ── Matplotlib style ──────────────────────────────────────────────────────
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "legend.framealpha": 0.7,
    }
)

PALETTE = {
    "train": "#1d4ed8",
    "val": "#b91c1c",
    "pred": "#dc2626",
    "base": "#2563eb",
    "good": "#22c55e",
    "model": "#7c3aed",
    "orange": "#f97316",
    "teal": "#0d9488",
}


# ── Reproducibility ───────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)


# ── Figure save helper ────────────────────────────────────────────────────

def save_fig(fig, path: Path, show: bool):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    plt.close(fig)


# ── Grid annotation ───────────────────────────────────────────────────────

def city_label(name: str | None) -> str:
    if name is None:
        return ""
    return (
        name.replace("Thiruvananthapuram", "Thiru.")
        .replace("Visakhapatnam", "Vizag")
        .replace(" ", "\n")
    )


def annotate_grid(ax, data_2d: np.ndarray, valid_mask: np.ndarray, layout, fontsize: int = 7):
    for r in range(data_2d.shape[0]):
        for c in range(data_2d.shape[1]):
            city = layout[r][c]
            if city is None or not valid_mask[r, c]:
                continue
            val = data_2d[r, c]
            text = city_label(city)
            if np.isfinite(val):
                text = f"{text}\n{val:.0f}"
            ax.text(c, r, text, ha="center", va="center", fontsize=fontsize,
                    color="black", fontweight="semibold")


def _grid_lines(ax, shape):
    ax.set_xticks(np.arange(-0.5, shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    ax.tick_params(which="both", bottom=False, left=False,
                   labelbottom=False, labelleft=False)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 01  —  Seasonal monthly AQI pattern (city × month heatmap)
# ═══════════════════════════════════════════════════════════════════════════

def fig01_seasonal_pattern(df: pd.DataFrame, path: Path, show: bool):
    monthly = df.groupby(df.index.month).mean().reindex(range(1, 13))
    months  = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig, ax = plt.subplots(figsize=(14, 8))
    im = ax.imshow(monthly.T.to_numpy(), aspect="auto", cmap="YlOrRd")
    fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02, label="Mean AQI")
    ax.set_title("Seasonal Monthly AQI Pattern — 20 Cities (2018–2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("City")
    ax.set_xticks(range(12));  ax.set_xticklabels(months)
    ax.set_yticks(range(len(monthly.columns)))
    ax.set_yticklabels(monthly.columns)
    for r, city in enumerate(monthly.columns):
        for c, month in enumerate(monthly.index):
            val = monthly.loc[month, city]
            if np.isfinite(val):
                ax.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=6.5,
                        color="black" if val < 250 else "white")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 02  —  Average AQI per city (ranked bar chart)
# ═══════════════════════════════════════════════════════════════════════════

def fig02_city_bar(df: pd.DataFrame, path: Path, show: bool):
    mean_aqi = df.mean().sort_values(ascending=False)
    colors = [
        AQI_COLORS[min(int(np.digitize(v, AQI_BREAKPOINTS, right=True)), 5)]
        for v in mean_aqi.values
    ]
    fig, ax = plt.subplots(figsize=(13, 5.5))
    bars = ax.bar(mean_aqi.index, mean_aqi.values, color=colors, edgecolor="white", linewidth=0.6)
    ax.set_title("Long-Run Average Monthly AQI by City (2018–2025)")
    ax.set_xlabel("City")
    ax.set_ylabel("Mean AQI")
    ax.tick_params(axis="x", rotation=40)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, mean_aqi.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.0f}", ha="center", va="bottom", fontsize=8)
    # legend for AQI categories
    legend_patches = [mpatches.Patch(color=AQI_COLORS[i], label=AQI_LABELS[i])
                      for i in range(6)]
    ax.legend(handles=legend_patches, loc="upper right", ncol=3, fontsize=8)
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 03  —  Spatial AQI grid heatmap (one time-step)
# ═══════════════════════════════════════════════════════════════════════════

def fig03_spatial_map(data_2d, valid_mask, layout, title: str, path: Path, show: bool, cmap="YlOrRd"):
    masked = data_2d.astype(np.float32).copy()
    masked[~valid_mask] = np.nan
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=300)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="AQI")
    ax.set_title(title)
    annotate_grid(ax, masked, valid_mask, layout)
    _grid_lines(ax, masked.shape)
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 04  —  Training & validation loss curves
# ═══════════════════════════════════════════════════════════════════════════

def fig04_loss_curve(history, path: Path, show: bool):
    loss     = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs   = range(1, len(loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Loss
    axes[0].plot(epochs, loss,     color=PALETTE["train"], lw=2,   label="Train loss")
    axes[0].plot(epochs, val_loss, color=PALETTE["val"],   lw=2,   label="Val loss")
    axes[0].fill_between(epochs, loss, val_loss, alpha=0.08, color="#6366f1")
    axes[0].set_title("Masked MSE Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.25)

    # Log-scale loss
    axes[1].semilogy(epochs, loss,     color=PALETTE["train"], lw=2, label="Train loss")
    axes[1].semilogy(epochs, val_loss, color=PALETTE["val"],   lw=2, label="Val loss")
    axes[1].set_title("Loss (log scale)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("log(Loss)")
    axes[1].legend()
    axes[1].grid(alpha=0.25)

    best_ep = int(np.argmin(val_loss)) + 1
    for ax in axes:
        ax.axvline(best_ep, color="gray", linestyle="--", alpha=0.6, lw=1.2,
                   label=f"Best epoch {best_ep}")
    axes[0].legend()
    fig.suptitle("Model Training History", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 05  —  Observed vs predicted scatter (model vs baseline)
# ═══════════════════════════════════════════════════════════════════════════

def fig05_scatter(true_vals, base_vals, pred_vals, path: Path, show: bool):
    lo = float(min(true_vals.min(), base_vals.min(), pred_vals.min()))
    hi = float(max(true_vals.max(), base_vals.max(), pred_vals.max()))

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, preds, title, color in zip(
        axes,
        [base_vals, pred_vals],
        ["Persistence Baseline", "ConvLSTM Model"],
        [PALETTE["base"], PALETTE["pred"]],
    ):
        ax.scatter(true_vals, preds, s=18, alpha=0.4, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="Ideal 1:1")
        # Regression line
        z = np.polyfit(true_vals, preds, 1)
        xr = np.linspace(lo, hi, 200)
        ax.plot(xr, np.poly1d(z)(xr), color=color, lw=1.5, alpha=0.7, label="Regression fit")
        ax.set_title(title)
        ax.set_xlabel("Observed AQI")
        ax.set_ylabel("Predicted AQI")
        ax.set_xlim(lo - 5, hi + 5)
        ax.set_ylim(lo - 5, hi + 5)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Observed vs Predicted AQI — Test Set", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 06  —  Error distribution (histogram + box)
# ═══════════════════════════════════════════════════════════════════════════

def fig06_error_dist(true_vals, pred_vals, base_vals, path: Path, show: bool):
    err_m = pred_vals - true_vals
    err_b = base_vals - true_vals

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Signed error histogram
    axes[0].hist(err_m, bins=30, color=PALETTE["model"], alpha=0.7, label="ConvLSTM", edgecolor="white")
    axes[0].hist(err_b, bins=30, color=PALETTE["base"],  alpha=0.5, label="Persistence", edgecolor="white")
    axes[0].axvline(0, color="black", lw=1.2)
    axes[0].set_title("Signed Error Distribution")
    axes[0].set_xlabel("Predicted − Observed AQI")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    # Absolute error histogram
    axes[1].hist(np.abs(err_m), bins=30, color=PALETTE["model"], alpha=0.7, label="ConvLSTM", edgecolor="white")
    axes[1].hist(np.abs(err_b), bins=30, color=PALETTE["base"],  alpha=0.5, label="Persistence", edgecolor="white")
    axes[1].set_title("Absolute Error Distribution")
    axes[1].set_xlabel("| Predicted − Observed | AQI")
    axes[1].set_ylabel("Count")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    # Box plot comparison
    axes[2].boxplot(
        [np.abs(err_b), np.abs(err_m)],
        tick_labels=["Persistence", "ConvLSTM"],
        patch_artist=True,
        boxprops={"facecolor": "#e0e7ff"},
        medianprops={"color": "black", "lw": 2},
    )
    axes[2].set_title("Absolute Error Box Plot")
    axes[2].set_ylabel("Absolute Error (AQI)")
    axes[2].grid(axis="y", alpha=0.25)

    fig.suptitle("Prediction Error Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 07  —  Per-city MAE (ConvLSTM vs baseline)
# ═══════════════════════════════════════════════════════════════════════════

def fig07_per_city_mae(per_city_df: pd.DataFrame, base_per_city: dict, path: Path, show: bool):
    df = per_city_df.sort_values("MAE", ascending=True).reset_index(drop=True)
    cities = df["City"].tolist()
    model_mae = df["MAE"].tolist()
    base_mae  = [base_per_city.get(c, np.nan) for c in cities]

    x = np.arange(len(cities))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 7))
    b1 = ax.barh(x + width / 2, base_mae,  width, color=PALETTE["base"],  label="Persistence baseline", alpha=0.85)
    b2 = ax.barh(x - width / 2, model_mae, width, color=PALETTE["model"], label="ConvLSTM",              alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(cities)
    ax.set_xlabel("MAE (AQI units)")
    ax.set_title("Per-City Test MAE: ConvLSTM vs Persistence Baseline")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    # Annotate improvement
    for i, (m, b) in enumerate(zip(model_mae, base_mae)):
        if np.isfinite(b) and np.isfinite(m):
            delta = b - m
            color = "#16a34a" if delta > 0 else "#dc2626"
            ax.text(max(m, b) + 0.5, i, f"{delta:+.1f}", va="center", fontsize=7.5, color=color)
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 08  —  Actual vs predicted maps + error (3-panel)
# ═══════════════════════════════════════════════════════════════════════════

def fig08_eval_maps(y_true_frame, y_pred_frame, valid_mask, layout, label: str, path: Path, show: bool):
    true_2d = y_true_frame[..., 0].copy()
    pred_2d = y_pred_frame[..., 0].copy()
    err_2d  = pred_2d - true_2d           # signed error
    true_2d[~valid_mask] = np.nan
    pred_2d[~valid_mask] = np.nan
    err_2d[~valid_mask]  = np.nan

    vmin = np.nanmin(np.stack([true_2d, pred_2d]))
    vmax = np.nanmax(np.stack([true_2d, pred_2d]))
    elim = np.nanmax(np.abs(err_2d))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    im0 = axes[0].imshow(true_2d, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[0].set_title("Observed AQI")
    annotate_grid(axes[0], true_2d, valid_mask, layout)

    axes[1].imshow(pred_2d, cmap="YlOrRd", vmin=vmin, vmax=vmax)
    axes[1].set_title("ConvLSTM Prediction")
    annotate_grid(axes[1], pred_2d, valid_mask, layout)

    im2 = axes[2].imshow(err_2d, cmap="RdBu_r", vmin=-elim, vmax=elim)
    axes[2].set_title("Signed Error (pred − obs)")
    annotate_grid(axes[2], err_2d, valid_mask, layout)

    for ax in axes:
        _grid_lines(ax, true_2d.shape)

    fig.colorbar(im0, ax=axes[:2], fraction=0.025, pad=0.02, label="AQI")
    fig.colorbar(im2, ax=axes[2],  fraction=0.04,  pad=0.04, label="Error (AQI)")
    fig.suptitle(f"Spatial Evaluation — Test Month: {label}", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 09  —  Test period time series for best & worst cities
# ═══════════════════════════════════════════════════════════════════════════

def fig09_test_timeseries(test_results_df: pd.DataFrame, per_city_df: pd.DataFrame, path: Path, show: bool):
    best   = per_city_df.sort_values("MAE").head(4)["City"].tolist()
    worst  = per_city_df.sort_values("MAE", ascending=False).head(4)["City"].tolist()
    cities = list(dict.fromkeys(best + worst))

    fig, axes = plt.subplots(len(cities), 1, figsize=(13, 2.5 * len(cities)), sharex=True)
    if len(cities) == 1:
        axes = [axes]

    for ax, city in zip(axes, cities):
        sub = test_results_df[test_results_df["City"] == city].sort_values("Date")
        ax.plot(sub["Date"], sub["Actual_AQI"],    color="black", lw=1.6, label="Observed", zorder=3)
        ax.plot(sub["Date"], sub["Predicted_AQI"], color=PALETTE["model"], lw=1.6,
                linestyle="--", label="ConvLSTM", zorder=3)
        ax.fill_between(sub["Date"], sub["Actual_AQI"], sub["Predicted_AQI"],
                        alpha=0.15, color=PALETTE["model"])
        ax.set_ylabel("AQI")
        group = "Best" if city in best else "Hardest"
        ax.set_title(f"{city}  [{group}]", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper right")

    axes[-1].set_xlabel("Date")
    fig.suptitle("Test Period — Observed vs ConvLSTM Predicted AQI by City",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 10  —  Year-over-year annual trends
# ═══════════════════════════════════════════════════════════════════════════

def fig10_annual_trends(df: pd.DataFrame, path: Path, show: bool):
    annual = df.copy()
    annual.index = pd.to_datetime(annual.index)
    annual = annual.groupby(annual.index.year).mean()

    fig, ax = plt.subplots(figsize=(13, 6))
    for city in annual.columns:
        ax.plot(annual.index, annual[city], marker="o", markersize=4, lw=1.5, alpha=0.75, label=city)
    mean_annual = annual.mean(axis=1)
    ax.plot(annual.index, mean_annual, color="black", lw=2.8, marker="D", markersize=6,
            label="20-city mean", zorder=5)
    ax.set_title("Year-over-Year Annual Mean AQI Trends — 20 Cities")
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Mean AQI")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=6.5, ncol=4)
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 11  —  AQI category distribution per city (stacked bar)
# ═══════════════════════════════════════════════════════════════════════════

def fig11_category_distribution(df: pd.DataFrame, path: Path, show: bool):
    all_cities = df.columns.tolist()
    data = {}
    for city in all_cities:
        cats = aqi_category(df[city].dropna().values)
        counts = np.bincount(cats, minlength=6)
        data[city] = counts / counts.sum() * 100

    cat_df = pd.DataFrame(data, index=AQI_LABELS).T
    # sort by "Good" fraction
    cat_df = cat_df.sort_values("Good", ascending=False)

    fig, ax = plt.subplots(figsize=(14, 6))
    bottom = np.zeros(len(cat_df))
    for cat, color in zip(AQI_LABELS, AQI_COLORS):
        vals = cat_df[cat].values
        ax.bar(cat_df.index, vals, bottom=bottom, color=color, label=cat, edgecolor="white", linewidth=0.4)
        bottom += vals
    ax.set_title("AQI Category Distribution per City (% of Months, 2018–2025)")
    ax.set_xlabel("City")
    ax.set_ylabel("% of months")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 12  —  City-city AQI correlation matrix
# ═══════════════════════════════════════════════════════════════════════════

def fig12_correlation_heatmap(df: pd.DataFrame, path: Path, show: bool):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03, label="Pearson r")
    ax.set_xticks(range(len(corr)))
    ax.set_yticks(range(len(corr)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(corr.index, fontsize=8)
    for r in range(len(corr)):
        for c in range(len(corr)):
            val = corr.iloc[r, c]
            ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6.5,
                    color="black" if abs(val) < 0.7 else "white")
    ax.set_title("City-City Monthly AQI Pearson Correlation Matrix")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 13  —  Multi-month forecast heatmaps panel
# ═══════════════════════════════════════════════════════════════════════════

def fig13_forecast_heatmaps(future_maps, valid_mask, layout, forecast_dates, path: Path, show: bool):
    maps = future_maps[..., 0].copy()
    maps[:, ~valid_mask] = np.nan
    vmin = np.nanmin(maps)
    vmax = np.nanmax(maps)

    n = len(forecast_dates)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.5))
    if n == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        heat = maps[idx].copy()
        im = ax.imshow(heat, cmap="YlOrRd", vmin=vmin, vmax=vmax)
        ax.set_title(forecast_dates[idx].strftime("%b %Y"), fontsize=11, fontweight="bold")
        annotate_grid(ax, heat, valid_mask, layout)
        _grid_lines(ax, heat.shape)

    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="Forecast AQI")
    fig.suptitle("Recursive Multi-Month AQI Forecast", fontsize=13, fontweight="bold", y=1.02)
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 14  —  Forecast city trend lines
# ═══════════════════════════════════════════════════════════════════════════

def fig14_forecast_trends(forecast_df: pd.DataFrame, mean_real, std_real, valid_mask,
                           city_positions, forecast_dates, path: Path, show: bool):
    pivot = forecast_df.pivot(index="Date", columns="City", values="Forecast_AQI")
    mean_forecast = pivot.mean(axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))

    # Top: individual city lines
    ax = axes[0]
    for city in pivot.columns:
        ax.plot(pivot.index, pivot[city], lw=1.2, alpha=0.55)
    ax.plot(mean_forecast.index, mean_forecast, color="black", lw=2.8, label="20-city mean", zorder=5)
    ax.set_title("Recursive Multi-Month AQI Forecast by City")
    ax.set_ylabel("Forecast AQI")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    # Bottom: mean ± uncertainty band from MC-Dropout
    ax2 = axes[1]
    # compute city-averaged mean and std
    mc_mean_city = []
    mc_std_city  = []
    for i in range(len(forecast_dates)):
        vals = []
        stds = []
        for city, (r, c) in city_positions.items():
            v = mean_real[i, r, c, 0]
            s = std_real[i, r, c, 0]
            if np.isfinite(v):
                vals.append(v)
                stds.append(s)
        mc_mean_city.append(np.mean(vals) if vals else np.nan)
        mc_std_city.append(np.mean(stds) if stds else np.nan)

    mc_mean_arr = np.array(mc_mean_city)
    mc_std_arr  = np.array(mc_std_city)
    ax2.plot(forecast_dates, mc_mean_arr, color=PALETTE["model"], lw=2.5, marker="o",
             markersize=6, label="MC-Dropout mean")
    ax2.fill_between(
        forecast_dates,
        mc_mean_arr - 2 * mc_std_arr,
        mc_mean_arr + 2 * mc_std_arr,
        alpha=0.25, color=PALETTE["model"], label="±2σ uncertainty",
    )
    ax2.fill_between(
        forecast_dates,
        mc_mean_arr - mc_std_arr,
        mc_mean_arr + mc_std_arr,
        alpha=0.4, color=PALETTE["model"], label="±1σ uncertainty",
    )
    ax2.set_title("All-City Average Forecast with MC-Dropout Uncertainty")
    ax2.set_xlabel("Forecast Month")
    ax2.set_ylabel("Mean AQI (all cities)")
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8)

    fig.suptitle("Multi-Month AQI Forecast Analysis", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 15  —  Per-city NSE and R² comparison panel
# ═══════════════════════════════════════════════════════════════════════════

def fig15_per_city_metrics(per_city_df: pd.DataFrame, path: Path, show: bool):
    df = per_city_df.sort_values("NSE", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(17, 6))

    # NSE
    colors_nse = [PALETTE["good"] if v >= 0 else PALETTE["pred"] for v in df["NSE"]]
    axes[0].barh(df["City"], df["NSE"], color=colors_nse, edgecolor="white")
    axes[0].axvline(0, color="black", lw=1)
    axes[0].axvline(0.5, color="gray", lw=1, linestyle="--", alpha=0.6, label="NSE=0.5 threshold")
    axes[0].set_title("Nash-Sutcliffe Efficiency (NSE)")
    axes[0].set_xlabel("NSE  (> 0 = better than mean; 1 = perfect)")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="x", alpha=0.25)

    # Bias
    colors_bias = [PALETTE["pred"] if v > 0 else PALETTE["teal"] for v in df["Bias"]]
    axes[1].barh(df["City"], df["Bias"], color=colors_bias, edgecolor="white")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].set_title("Prediction Bias (Pred − Obs)")
    axes[1].set_xlabel("Bias (AQI)  [positive = over-prediction]")
    axes[1].grid(axis="x", alpha=0.25)

    # Category accuracy
    df_cat = df.sort_values("AQI_Category_Accuracy", ascending=True)
    axes[2].barh(df_cat["City"], df_cat["AQI_Category_Accuracy"] * 100,
                 color=PALETTE["teal"], edgecolor="white")
    axes[2].axvline(50, color="gray", lw=1, linestyle="--", alpha=0.6)
    axes[2].set_title("AQI Category Accuracy (%)")
    axes[2].set_xlabel("% months with correct AQI category")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("Per-City Model Quality Metrics", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# DataFrame builders
# ═══════════════════════════════════════════════════════════════════════════

def frame_to_city_values(frame, city_positions):
    d2 = frame[..., 0] if frame.ndim == 3 else frame
    return {city: float(d2[r, c]) for city, (r, c) in city_positions.items()}


def build_test_results_df(test_dates, y_true, y_pred, city_positions):
    rows = []
    for i, date in enumerate(test_dates):
        tv = frame_to_city_values(y_true[i], city_positions)
        pv = frame_to_city_values(y_pred[i], city_positions)
        for city in city_positions:
            actual    = tv[city]
            predicted = pv[city]
            rows.append(
                {
                    "Date": date,
                    "City": city,
                    "Actual_AQI": actual,
                    "Predicted_AQI": predicted,
                    "Absolute_Error": abs(actual - predicted),
                    "Signed_Error": predicted - actual,
                    "Actual_Category":    AQI_LABELS[int(aqi_category([actual])[0])],
                    "Predicted_Category": AQI_LABELS[int(aqi_category([predicted])[0])],
                }
            )
    return pd.DataFrame(rows)


def build_forecast_df(forecast_dates, future_maps, city_positions):
    rows = []
    for i, date in enumerate(forecast_dates):
        pv = frame_to_city_values(future_maps[i], city_positions)
        for city in city_positions:
            fc = pv[city]
            rows.append(
                {
                    "Date": date,
                    "City": city,
                    "Forecast_AQI": fc,
                    "Forecast_Category": AQI_LABELS[int(aqi_category([fc])[0])],
                }
            )
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# Journal-level diagnostics integrated into the main pipeline
# ═══════════════════════════════════════════════════════════════════════════

def train_monthly_climatology(df_train: pd.DataFrame, target_dates, city_positions):
    """Forecast each test month with that calendar month's training mean."""
    month_means = df_train.groupby(df_train.index.month).mean()
    city_order = list(city_positions)
    out = []
    for date in target_dates:
        row = []
        for city in city_order:
            if date.month in month_means.index and np.isfinite(month_means.loc[date.month, city]):
                row.append(float(month_means.loc[date.month, city]))
            else:
                row.append(float(df_train[city].mean()))
        out.append(row)
    return np.asarray(out, dtype=np.float32), city_order


def matrix_to_frames(values, city_order, city_positions, shape):
    frames = np.full(shape, np.nan, dtype=np.float32)
    for j, city in enumerate(city_order):
        r, c = city_positions[city]
        frames[:, r, c, 0] = values[:, j]
    return frames


def linear_trend_forecast(X_test):
    """One-step linear extrapolation from each 6-month input sequence."""
    seq_len = X_test.shape[1]
    x = np.arange(seq_len, dtype=np.float32)
    x_mean = x.mean()
    denom = np.sum((x - x_mean) ** 2)
    y_mean = X_test.mean(axis=1)
    slope = np.sum((x[None, :, None, None, None] - x_mean) * (X_test - y_mean[:, None]), axis=1) / denom
    pred = y_mean + slope * (seq_len - x_mean)
    return np.maximum(pred.astype(np.float32), 0.0)


def frame_to_long(dates, frames, city_positions, value_name):
    rows = []
    for i, date in enumerate(dates):
        for city, (r, c) in city_positions.items():
            rows.append({"Date": date, "City": city, value_name: float(frames[i, r, c, 0])})
    return pd.DataFrame(rows)


def dm_test_by_month(test_results: pd.DataFrame, baseline_long: pd.DataFrame, loss_power=2):
    """
    Diebold-Mariano style paired test using monthly average loss differences.

    Negative mean loss difference means ConvLSTM has lower loss than the
    baseline. We aggregate city losses by month so the test unit is forecast
    month, avoiding an inflated sample size from treating city cells as fully
    independent.
    """
    merged = test_results.merge(baseline_long, on=["Date", "City"], how="inner")
    merged["ConvLSTM_Error"] = merged["Predicted_AQI"] - merged["Actual_AQI"]
    merged["Baseline_Error"] = merged["Baseline_Predicted_AQI"] - merged["Actual_AQI"]
    merged["Loss_Diff"] = (
        np.abs(merged["ConvLSTM_Error"]) ** loss_power
        - np.abs(merged["Baseline_Error"]) ** loss_power
    )

    monthly_d = merged.groupby("Date")["Loss_Diff"].mean().to_numpy(dtype=np.float64)
    n = len(monthly_d)
    mean_d = float(np.mean(monthly_d))
    sd = float(np.std(monthly_d, ddof=1)) if n > 1 else np.nan
    if n > 1 and sd > 0:
        statistic = mean_d / (sd / np.sqrt(n))
        p_two_sided = float(2 * stats.t.sf(abs(statistic), df=n - 1))
    else:
        statistic = np.nan
        p_two_sided = np.nan

    return {
        "n_months": n,
        "loss_power": loss_power,
        "mean_loss_diff_model_minus_baseline": mean_d,
        "dm_t_statistic": float(statistic),
        "p_value_two_sided": p_two_sided,
        "interpretation": "negative favors ConvLSTM; positive favors baseline",
    }


def rolling_origin_baselines(df: pd.DataFrame, seq_len: int, min_train_months: int):
    """Rolling-origin CV for simple baselines only, without retraining ConvLSTM."""
    rows = []
    cities = df.columns.tolist()
    values = df.to_numpy(dtype=np.float32)
    for target_idx in range(min_train_months, len(df)):
        date = df.index[target_idx]
        if target_idx < seq_len:
            continue
        history = values[:target_idx]
        actual = values[target_idx]
        last_window = values[target_idx - seq_len: target_idx]
        preds = {
            "Persistence": last_window[-1],
            "Rolling3Mean": last_window[-3:].mean(axis=0),
            "TrainMean": history.mean(axis=0),
        }
        if target_idx >= 12:
            preds["SeasonalNaive"] = values[target_idx - 12]
        else:
            preds["SeasonalNaive"] = values[0]
        month_hist = df.iloc[:target_idx]
        month_mean = month_hist[month_hist.index.month == date.month].mean().reindex(cities)
        preds["SeasonalMonthlyMean"] = month_mean.fillna(month_hist.mean()).to_numpy(dtype=np.float32)

        for name, pred in preds.items():
            err = pred - actual
            rows.append(
                {
                    "Date": date,
                    "Baseline": name,
                    "MAE": float(np.mean(np.abs(err))),
                    "RMSE": float(np.sqrt(np.mean(err**2))),
                    "Bias": float(np.mean(err)),
                }
            )
    return pd.DataFrame(rows)


def fig16_baseline_comparison(metrics_df: pd.DataFrame, path: Path, show: bool):
    ordered = metrics_df.sort_values("MAE")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(ordered["Model"], ordered["MAE"], color=PALETTE["teal"], edgecolor="white")
    axes[0].set_title("Test MAE by Model/Baseline")
    axes[0].set_ylabel("MAE (AQI)")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(axis="y", alpha=0.25)

    ordered_rmse = metrics_df.sort_values("RMSE")
    axes[1].bar(ordered_rmse["Model"], ordered_rmse["RMSE"], color=PALETTE["model"], edgecolor="white")
    axes[1].set_title("Test RMSE by Model/Baseline")
    axes[1].set_ylabel("RMSE (AQI)")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(axis="y", alpha=0.25)
    fig.suptitle("Integrated Journal Baseline/Ablation Comparison", fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


def fig17_dm_summary(dm_df: pd.DataFrame, path: Path, show: bool):
    fig, ax = plt.subplots(figsize=(10, 4.8))
    colors = ["#16a34a" if v < 0 else "#dc2626" for v in dm_df["mean_loss_diff_model_minus_baseline"]]
    ax.bar(dm_df["Baseline"], dm_df["mean_loss_diff_model_minus_baseline"], color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Diebold-Mariano Style Paired Loss Difference")
    ax.set_ylabel("Mean squared loss diff: ConvLSTM - baseline")
    ax.set_xlabel("Baseline")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for i, row in dm_df.reset_index(drop=True).iterrows():
        p = row["p_value_two_sided"]
        label = f"p={p:.3f}" if np.isfinite(p) else "p=N/A"
        y = row["mean_loss_diff_model_minus_baseline"]
        ax.text(i, y, label, ha="center", va="bottom" if y >= 0 else "top", fontsize=8)
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 19  —  SARIMA vs ConvLSTM per-city MAE
# ═══════════════════════════════════════════════════════════════════════════

def fig19_sarima_comparison(per_city_conv: pd.DataFrame, sarima_per_city: list[dict],
                            lstm_per_city: list[dict], path: Path, show: bool):
    """Per-city MAE: ConvLSTM vs SARIMA vs Vanilla LSTM."""
    conv_mae = {row["City"]: row["MAE"] for row in per_city_conv.to_dict("records")}
    sarima_mae = {row["City"]: row["MAE"] for row in sarima_per_city}
    lstm_mae = {row["City"]: row["MAE"] for row in lstm_per_city}

    cities = sorted(conv_mae.keys(), key=lambda c: conv_mae.get(c, 999))
    x = np.arange(len(cities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(x - width, [sarima_mae.get(c, np.nan) for c in cities], width,
            color="#f97316", alpha=0.85, label="SARIMA")
    ax.barh(x, [lstm_mae.get(c, np.nan) for c in cities], width,
            color="#0d9488", alpha=0.85, label="Vanilla LSTM")
    ax.barh(x + width, [conv_mae[c] for c in cities], width,
            color=PALETTE["model"], alpha=0.85, label="ConvLSTM")
    ax.set_yticks(x)
    ax.set_yticklabels(cities)
    ax.set_xlabel("MAE (AQI units)")
    ax.set_title("Per-City Test MAE: ConvLSTM vs SARIMA vs Vanilla LSTM")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 20  —  Ablation study grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════

def fig20_ablation_study(ablation_df: pd.DataFrame, path: Path, show: bool):
    """Grouped bar chart of ablation study results."""
    df = ablation_df.sort_values("MAE", ascending=True)
    x = np.arange(len(df))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # MAE
    colors = [PALETTE["model"] if v == "full" else "#94a3b8" for v in df["Variant"]]
    colors_full = []
    for v in df["Variant"]:
        if v == "full":
            colors_full.append(PALETTE["model"])
        elif v == "no_augmentation":
            colors_full.append(PALETTE["orange"])
        elif v == "shallow":
            colors_full.append(PALETTE["teal"])
        else:
            colors_full.append("#94a3b8")
    axes[0].barh(x, df["MAE"], color=colors_full, edgecolor="white")
    axes[0].set_yticks(x)
    axes[0].set_yticklabels(df["Variant"])
    axes[0].set_xlabel("MAE (AQI)")
    axes[0].set_title("Mean Absolute Error")
    axes[0].grid(axis="x", alpha=0.25)

    # NSE
    colors_nse = [PALETTE["good"] if v >= 0 else PALETTE["pred"] for v in df["NSE"]]
    axes[1].barh(x, df["NSE"], color=colors_nse, edgecolor="white")
    axes[1].set_yticks(x)
    axes[1].set_yticklabels(df["Variant"])
    axes[1].set_xlabel("NSE")
    axes[1].set_title("Nash-Sutcliffe Efficiency")
    axes[1].axvline(0, color="black", lw=1)
    axes[1].grid(axis="x", alpha=0.25)

    # R²
    axes[2].barh(x, df["R2"], color=colors_full, edgecolor="white")
    axes[2].set_yticks(x)
    axes[2].set_yticklabels(df["Variant"])
    axes[2].set_xlabel("R²")
    axes[2].set_title("Coefficient of Determination")
    axes[2].grid(axis="x", alpha=0.25)

    fig.suptitle("ConvLSTM Ablation Study — Component Contribution Analysis",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_fig(fig, path, show)


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 21  —  Comprehensive model comparison radar chart
# ═══════════════════════════════════════════════════════════════════════════

def fig21_radar_comparison(comparison_df: pd.DataFrame, path: Path, show: bool):
    """Radar/spider chart comparing all models across multiple metrics."""
    metrics = ["MAE", "RMSE", "R2", "NSE", "AQI_Category_Accuracy"]
    metric_labels = ["MAE ↓", "RMSE ↓", "R² ↑", "NSE ↑", "Cat. Acc. ↑"]
    models = comparison_df["Model"].tolist()

    # Normalize each metric to [0, 1] for radar display
    # For MAE/RMSE lower is better, so invert
    values = comparison_df[metrics].to_numpy(dtype=np.float64).copy()
    normed = np.zeros_like(values)
    for i, m in enumerate(metrics):
        col = values[:, i]
        mi, ma = col.min(), col.max()
        if ma - mi > 1e-10:
            if m in ("MAE", "RMSE"):  # lower is better → invert
                normed[:, i] = (ma - col) / (ma - mi)
            else:
                normed[:, i] = (col - mi) / (ma - mi)
        else:
            normed[:, i] = 0.5

    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    model_colors = [PALETTE["model"], PALETTE["base"], PALETTE["orange"],
                    PALETTE["teal"], "#94a3b8", "#e11d48", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for idx, model in enumerate(models):
        vals = normed[idx].tolist()
        vals += vals[:1]
        color = model_colors[idx % len(model_colors)]
        ax.plot(angles, vals, "o-", lw=2, label=model, color=color, markersize=5)
        ax.fill(angles, vals, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_title("Model Comparison Radar Chart\n(normalized, outer = better)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
    fig.tight_layout()
    save_fig(fig, path, show)


def run_journal_diagnostics(
    df,
    X_test,
    y_test_real,
    valid_mask,
    city_positions,
    test_dates,
    df_train,
    metrics_df,
    test_results_df,
    output_dir: Path,
    seq_len: int,
    show_plots: bool,
    min_cv_train_months: int,
):
    """Create the scoped journal-level diagnostics inside the main run."""
    baseline_frames = {
        "Persistence": X_test[:, -1],
        "Rolling3Mean": X_test[:, -3:].mean(axis=1),
        "LinearTrend6M": linear_trend_forecast(X_test),
        "TrainMean": matrix_to_frames(
            np.repeat(df_train.mean().to_numpy(dtype=np.float32)[None, :], len(X_test), axis=0),
            df_train.columns.tolist(),
            city_positions,
            y_test_real.shape,
        ),
    }
    seasonal_vals, city_order = train_monthly_climatology(df_train, test_dates, city_positions)
    baseline_frames["SeasonalMonthlyMean"] = matrix_to_frames(
        seasonal_vals, city_order, city_positions, y_test_real.shape
    )
    baseline_frames["SeasonalNaive"] = run_seasonal_naive_baseline(
        df, test_dates, city_positions, y_test_real.shape[1:3]
    )

    rows = []
    baseline_longs = {}
    for name, pred in baseline_frames.items():
        metrics = compute_metrics(y_test_real, pred, valid_mask)
        rows.append({"Model": name, **{k: v for k, v in metrics.items() if not k.endswith("_values")}})
        baseline_longs[name] = frame_to_long(test_dates, pred, city_positions, "Baseline_Predicted_AQI")

    ablation_df = pd.concat([metrics_df, pd.DataFrame(rows)], ignore_index=True)
    ablation_df = ablation_df.drop_duplicates(subset=["Model"], keep="first")
    ablation_df.to_csv(output_dir / "baseline_ablation_metrics.csv", index=False)

    dm_rows = []
    for name, base_long in baseline_longs.items():
        dm_rows.append({"Baseline": name, **dm_test_by_month(test_results_df, base_long, loss_power=2)})
    dm_df = pd.DataFrame(dm_rows).sort_values("mean_loss_diff_model_minus_baseline")
    dm_df.to_csv(output_dir / "diebold_mariano_tests.csv", index=False)

    cv_df = rolling_origin_baselines(df, seq_len=seq_len, min_train_months=min_cv_train_months)
    cv_summary = cv_df.groupby("Baseline", as_index=False).agg(
        MAE_mean=("MAE", "mean"),
        MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"),
        Bias_mean=("Bias", "mean"),
        folds=("Date", "count"),
    )
    cv_summary.to_csv(output_dir / "rolling_origin_baselines.csv", index=False)

    fig16_baseline_comparison(ablation_df, output_dir / "16_journal_baseline_comparison.png", show_plots)
    fig17_dm_summary(dm_df, output_dir / "17_dm_test_summary.png", show_plots)

    best = ablation_df.sort_values("MAE").iloc[0]
    dm_persist = dm_df[dm_df["Baseline"] == "Persistence"].iloc[0]
    notes = f"""# Journal Readiness Notes

This file is generated by the integrated `experiments.py` pipeline.

## Integrated diagnostics
- Baseline/ablation table: `baseline_ablation_metrics.csv`
- Rolling-origin baseline cross-validation: `rolling_origin_baselines.csv`
- Diebold-Mariano style paired error tests: `diebold_mariano_tests.csv`
- Figures: `16_journal_baseline_comparison.png`, `17_dm_test_summary.png`

## Current evidence
- Best MAE in the comparison table: **{best['Model']}** with MAE = **{best['MAE']:.3f}** AQI.
- ConvLSTM vs persistence DM-style squared-error test:
  statistic = **{dm_persist['dm_t_statistic']:.3f}**, p = **{dm_persist['p_value_two_sided']:.4f}**.
- Negative mean loss difference favors ConvLSTM; positive favors the baseline.

## Scope note
This is a journal-level implementation layer without overextending the project:
preprocessing, {seq_len}-month spatial sequences, ConvLSTM, persistence and simple
classical baselines, rolling-origin checks, paired error testing, uncertainty
plots, and multi-month recursive forecasting.

For a stronger journal submission, the next expansion would be external
covariates such as meteorology, emissions, holidays/fire events, and possibly
daily-resolution experiments. GNNs, transformers, or graph construction are
future-work candidates, not required for this scoped deliverable.
"""
    (output_dir / "journal_readiness_notes.md").write_text(notes, encoding="utf-8")
    return {
        "baseline_ablation_metrics": ablation_df,
        "diebold_mariano_tests": dm_df,
        "rolling_origin_baselines": cv_summary,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Research brief writer
# ═══════════════════════════════════════════════════════════════════════════

def write_research_brief(
    output_dir: Path,
    report: dict,
    seq_len: int,
    forecast_steps: int,
    metrics_df: pd.DataFrame,
    per_city_df: pd.DataFrame,
    test_dates,
    forecast_dates,
    journal_results: dict | None = None,
):
    conv  = metrics_df.loc[metrics_df["Model"] == "ConvLSTM"].iloc[0]
    base  = metrics_df.loc[metrics_df["Model"] == "Baseline (Persistence)"].iloc[0]
    improvement = (base["MAE"] - conv["MAE"]) / max(base["MAE"], 1e-8) * 100

    # Pre-compute to avoid dict-literal issues inside f-strings
    missing_before_total = sum(report.get("missing_before", {}).values())
    missing_after_total  = sum(report.get("missing_after",  {}).values())

    best3   = per_city_df.sort_values("MAE").head(3)["City"].tolist()
    worst3  = per_city_df.sort_values("MAE", ascending=False).head(3)["City"].tolist()
    pos_nse = per_city_df[per_city_df["NSE"] > 0]["City"].tolist()
    journal_section = ""
    if journal_results:
        ablation_df = journal_results["baseline_ablation_metrics"].sort_values("MAE")
        dm_df = journal_results["diebold_mariano_tests"]
        dm_proper_df = journal_results.get("dm_proper_tests")
        best_row = ablation_df.iloc[0]
        persist_dm = dm_df[dm_df["Baseline"] == "Persistence"]
        if dm_proper_df is not None and not dm_proper_df.empty:
            persist_proper = dm_proper_df[
                (dm_proper_df["Baseline"] == "Persistence")
                & (dm_proper_df["Loss_Type"] == "MAE")
            ]
            if not persist_proper.empty:
                persist_proper = persist_proper.iloc[0]
                dm_line = (
                    "Proper Diebold-Mariano test vs persistence using MAE loss: "
                    f"HLN-adjusted statistic = {persist_proper['DM_HLN_Statistic']:.3f}, "
                    f"p = {persist_proper['P_Value_HLN']:.4f}. "
                    "This result is reported descriptively; statistical significance is "
                    "not claimed from the 12-month holdout."
                )
            else:
                dm_line = "Proper persistence DM comparison was not available."
        elif not persist_dm.empty:
            persist_dm = persist_dm.iloc[0]
            dm_line = (
                f"ConvLSTM vs persistence paired squared-error test: "
                f"t = {persist_dm['dm_t_statistic']:.3f}, "
                f"p = {persist_dm['p_value_two_sided']:.4f}. "
                "The loss difference is negative when ConvLSTM has lower error."
            )
        else:
            dm_line = "Persistence DM-style comparison was not available."
        baseline_lines = "\n".join(
            f"- {row.Model}: MAE = {row.MAE:.3f}, RMSE = {row.RMSE:.3f}, R² = {row.R2:.3f}"
            for row in ablation_df.itertuples(index=False)
        )
        journal_section = f"""
### 4.3 Journal-Level Baseline Diagnostics

To avoid relying on a single persistence comparison, the integrated pipeline also evaluates
rolling mean, linear trend, training mean, and seasonal monthly mean baselines. The best method
by MAE in this run is **{best_row['Model']}** with MAE = {best_row['MAE']:.3f}.

{baseline_lines}

{dm_line}

Full diagnostics are saved in `outputs/baseline_ablation_metrics.csv`,
`outputs/diebold_mariano_tests.csv`, `outputs/dm_test_proper.csv`, and
`outputs/rolling_origin_baselines.csv`.
"""

    brief = f"""\
# AQI Forecasting Research Brief
*Auto-generated by the spatial-aqi-forecasting pipeline*

---

## 1. Abstract

We present a spatiotemporal deep learning pipeline for forecasting monthly Air Quality Index (AQI)
across 20 major Indian cities using a Convolutional LSTM (ConvLSTM) network. Each city's monthly
mean AQI is embedded in a fixed 4 × 5 geographic grid. The ConvLSTM model learns to predict the
next month's AQI map from the preceding {seq_len} months. Recursive application of the one-step
predictor generates a {forecast_steps}-month future outlook. On the chronological test set covering
**{test_dates[0].strftime("%b %Y")} – {test_dates[-1].strftime("%b %Y")}**, the ConvLSTM achieves
MAE = {conv["MAE"]:.2f} AQI units, RMSE = {conv["RMSE"]:.2f}, R² = {conv["R2"]:.3f},
NSE = {conv["NSE"]:.3f}, and an AQI category accuracy of {conv["AQI_Category_Accuracy"]*100:.1f}%,
compared with a persistence baseline of MAE = {base["MAE"]:.2f}.
The model shows a **{improvement:.2f}% MAE improvement** relative to the persistence baseline.

---

## 2. Dataset

| Property | Value |
|---|---|
| Source file | `{report.get("source_file")}` |
| Format | {report.get("source_format")} |
| Date range | {report.get("start_month")} → {report.get("end_month")} |
| Total daily rows | {report.get("daily_rows", "N/A"):,} |
| Missing daily AQI values | {report.get("missing_daily_aqi", "N/A"):,} |
| Cities modeled | {len(report.get("cities", []))} |
| Monthly grid shape | 4 rows × 5 cols (all 20 cells occupied) |
| Monthly frames produced | {report.get("rows", "N/A")} |
| Missing monthly (before interpolation) | {missing_before_total} |
| Missing monthly (after interpolation) | {missing_after_total} |

Daily records were aggregated to monthly mean AQI per city. The resulting 87-month panel was
reindexed to a complete month-start timeline, with gaps filled by time-interpolation (pandas
`interpolate(method="time")`), linear fallback, and edge fill to produce a gap-free spatiotemporal
tensor.

---

## 3. Methodology

### 3.1 Spatial Frame Construction

The 20 cities are placed in a geographically motivated 4 × 5 grid (rows: North-India band,
North-Central, West/Deccan, South). Each month *t* yields a (4, 5, 1) AQI map tensor **M**_t.
The full dataset is the sequence **M**_1, **M**_2, …, **M**_T (T = {report.get("rows", "N/A")}).

### 3.2 Sequence Preparation

Supervised samples are constructed with a sliding window of length *L* = {seq_len}:
- Input: **X**_i = (**M**_(i), …, **M**_(i+L−1))   shape (L, 4, 5, 1)
- Target: **y**_i = **M**_(i+L)                     shape (4, 5, 1)

Total sequences produced: {report.get("rows", 0) - seq_len} samples.
Chronological 75 / 10 / 15 % split → no look-ahead leakage.

### 3.3 Normalization

Each grid cell (city) is independently standardised with a `StandardScaler` fitted on training
data only. This per-cell z-score normalization prevents high-AQI northern cities (Delhi, Gurugram)
from dominating the loss and biasing predictions for low-AQI southern cities.

### 3.4 Model Architecture

```
Input: (L, 4, 5, 1)
│
├─ ConvLSTM2D(64, 3×3, padding=same, return_sequences=True)
│  + BatchNormalization + Dropout(0.20)
│
├─ ConvLSTM2D(32, 3×3, padding=same, return_sequences=True)
│  + BatchNormalization + Dropout(0.20)
│
├─ ConvLSTM2D(16, 3×3, padding=same, return_sequences=False)
│  + BatchNormalization
│
├─ Spatial Attention Gate (2-layer Conv)
│
├─ Conv2D(32, 3×3, relu) + BatchNormalization
├─ Conv2D(1,  1×1, linear)  ← delta output
│
└─ Residual Add: last input frame + delta → final prediction
```

**Parameters**: see `outputs/model_summary.txt`

Loss: masked MSE (computed only on the 20 valid city cells).
Optimizer: Adam (initial LR = 8×10⁻⁴) with ReduceLROnPlateau.
Training: Early stopping (patience 40) on validation masked MSE.

### 3.5 Data Augmentation

{2} noisy copies of each training sample are created by adding Gaussian noise (σ = 0.05 in
standardised units) to the input sequence only (targets unchanged).  This triples the effective
training set without introducing target-side leakage.

### 3.6 Multi-Step Forecasting

One-step predictions are concatenated back into the input window and re-fed to the model to
produce the {forecast_steps}-step future outlook (recursive / autoregressive strategy).

### 3.7 Uncertainty Quantification (MC-Dropout)

Uncertainty bands are estimated via Monte Carlo Dropout (30 stochastic forward passes with
dropout active at inference). The resulting per-step standard deviation approximates Bayesian
posterior uncertainty, following Gal & Ghahramani (2016).

---

## 4. Evaluation Results

### 4.1 Overall Test-Set Metrics

| Metric | Persistence Baseline | ConvLSTM | Δ |
|---|---|---|---|
| MAE (AQI) | {base["MAE"]:.3f} | {conv["MAE"]:.3f} | {conv["MAE"]-base["MAE"]:+.3f} |
| RMSE | {base["RMSE"]:.3f} | {conv["RMSE"]:.3f} | {conv["RMSE"]-base["RMSE"]:+.3f} |
| R² | {base["R2"]:.3f} | {conv["R2"]:.3f} | {conv["R2"]-base["R2"]:+.3f} |
| MAPE (%) | {base["MAPE_percent"]:.2f} | {conv["MAPE_percent"]:.2f} | {conv["MAPE_percent"]-base["MAPE_percent"]:+.2f} |
| NSE | {base["NSE"]:.3f} | {conv["NSE"]:.3f} | {conv["NSE"]-base["NSE"]:+.3f} |
| Willmott d | {base["Willmott_d"]:.3f} | {conv["Willmott_d"]:.3f} | {conv["Willmott_d"]-base["Willmott_d"]:+.3f} |
| Theil U | {base["Theil_U"]:.3f} | {conv["Theil_U"]:.3f} | {conv["Theil_U"]-base["Theil_U"]:+.3f} |
| Category Acc. | {base["AQI_Category_Accuracy"]*100:.1f}% | {conv["AQI_Category_Accuracy"]*100:.1f}% | {(conv["AQI_Category_Accuracy"]-base["AQI_Category_Accuracy"])*100:+.1f}pp |

NSE > 0 indicates the model outperforms the climatological mean. Theil U < 1 indicates the model
outperforms a random-walk (persistence) benchmark.

### 4.2 Per-City Performance

- **Best predicted cities** (lowest MAE): {", ".join(best3)}
- **Hardest cities** (highest MAE): {", ".join(worst3)}
- Cities with positive NSE (model beats mean): {", ".join(pos_nse) if pos_nse else "None"}

Full per-city breakdown: `outputs/per_city_metrics.csv`

{journal_section}

---

## 5. Multi-Step Forecast

Forecast months: {forecast_dates[0].strftime("%b %Y")} – {forecast_dates[-1].strftime("%b %Y")}

See `outputs/forecast_next_months_long.csv` for city-level AQI values.
Spatial maps: `outputs/13_forecast_heatmaps.png`
Uncertainty bands: `outputs/14_forecast_trends.png`

---

## 6. Literature Review

The following publications directly inform or benchmark the methodology used here.

| # | Authors | Title | Venue | Year | Link |
|---|---|---|---|---|---|
| [1] | Shi et al. | Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting | NeurIPS | 2015 | [arXiv:1506.04214](https://arxiv.org/abs/1506.04214) |
| [2] | Le, Bui & Cha | Spatiotemporal deep learning model for citywide air pollution interpolation and prediction | IEEE Access | 2019 | [arXiv:1911.12919](https://arxiv.org/abs/1911.12919) |
| [3] | Liang et al. | AirFormer: Predicting Nationwide Air Quality in China with Transformers | AAAI | 2023 | [arXiv:2211.15979](https://arxiv.org/abs/2211.15979) |
| [4] | Gal & Ghahramani | Dropout as a Bayesian Approximation | ICML | 2016 | [arXiv:1506.02142](https://arxiv.org/abs/1506.02142) |
| [5] | Hochreiter & Schmidhuber | Long Short-Term Memory | Neural Computation | 1997 | [DOI:10.1162/neco.1997.9.8.1735](https://doi.org/10.1162/neco.1997.9.8.1735) |
| [6] | Zhang et al. | Deep learning for spatiotemporal air quality forecasting: a survey | Atmosphere | 2024 | [MDPI:10.3390/atmos15111352](https://www.mdpi.com/2073-4433/15/11/1352) |
| [7] | Dominick et al. | Spatial assessment of air quality using GIS and remote sensing | Chemosphere | 2012 | [DOI:10.1016/j.chemosphere.2012.07.070](https://doi.org/10.1016/j.chemosphere.2012.07.070) |
| [8] | Nash & Sutcliffe | River flow forecasting through conceptual models | J. Hydrology | 1970 | [DOI:10.1016/0022-1694(70)90255-6](https://doi.org/10.1016/0022-1694(70)90255-6) |

### Key Methodological Justifications

**ConvLSTM [1]**: The architectural backbone. ConvLSTM replaces fully connected LSTM transitions
with convolutional operations, making it directly suited to data defined on a spatial grid.

**Spatial AQI frames [2]**: Le et al. demonstrate that converting citywide pollution observations
into image-like spatial sequences enables ConvLSTM to learn both local (city-level) and regional
(inter-city) pollution dynamics simultaneously.

**Per-cell normalisation**: Motivated by the large inter-city AQI disparity in the dataset
(min ≈ 60 for southern cities, max ≈ 250 for northern cities). Global normalisation collapses this
structure and biases loss towards high-AQI cities.

**Residual skip connection**: Following the principle of ResNet (He et al., 2016), the model
predicts a *correction delta* over the last observed month, which is easier to learn than the
absolute target value, especially for smooth seasonal signals.

**MC-Dropout uncertainty [4]**: A practical Bayesian approximation that provides calibrated
uncertainty estimates without requiring an ensemble of separately trained models.

**NSE and Willmott d [8]**: Standard hydrological benchmarking metrics adopted here to provide
reference-independent model performance scores that are more informative than R² alone.

---

## 7. Outputs

All generated artifacts are in `outputs/`:

| File | Description |
|---|---|
| `convlstm_aqi_model.keras` | Trained model weights |
| `model_summary.txt` | Layer-by-layer parameter count |
| `metrics_summary.csv` | ConvLSTM vs baseline global metrics |
| `baseline_ablation_metrics.csv` | Integrated classical baseline/ablation metrics |
| `diebold_mariano_tests.csv` | Paired loss tests against baselines |
| `rolling_origin_baselines.csv` | Rolling-origin baseline CV summary |
| `per_city_metrics.csv` | MAE, RMSE, Bias, NSE per city |
| `prediction_results_test_long.csv` | Full test-set predictions |
| `forecast_next_months_long.csv` | {forecast_steps}-month recursive forecast |
| `01_seasonal_pattern.png` | City × month AQI heatmap |
| `02_city_bar.png` | Average AQI ranked bar |
| `03_spatial_map_first.png` | First-month AQI grid |
| `03b_spatial_map_avg.png` | Dataset-average AQI grid |
| `04_loss_curves.png` | Training & validation loss |
| `05_scatter_obs_pred.png` | Observed vs predicted scatter |
| `06_error_distribution.png` | Error histograms & box plot |
| `07_per_city_mae.png` | Per-city MAE vs baseline |
| `08_eval_maps.png` | Observed / predicted / error spatial maps |
| `09_test_timeseries.png` | Test-period time series by city |
| `10_annual_trends.png` | Year-over-year AQI trends |
| `11_category_distribution.png` | AQI category stacked bar |
| `12_correlation_heatmap.png` | City-city correlation matrix |
| `13_forecast_heatmaps.png` | Recursive forecast spatial maps |
| `14_forecast_trends.png` | Forecast city lines + uncertainty |
| `15_per_city_metrics_panel.png` | NSE / Bias / Category accuracy |
| `16_journal_baseline_comparison.png` | Baseline/ablation comparison |
| `17_dm_test_summary.png` | DM-style paired loss summary |
"""
    (output_dir / "research_brief.md").write_text(brief, encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════

def run_experiment(
    data_file,
    seq_len=12,
    forecast_steps=6,
    train_ratio=0.75,
    val_ratio=0.10,
    use_augmentation=True,
    augment_copies=2,
    augment_noise_std=0.05,
    epochs=250,
    batch_size=8,
    run_tscv=False,
    output_dir="outputs",
    show_plots=False,
    mc_passes=30,
    run_journal=True,
    min_cv_train_months=48,
):
    set_seed(42)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load & preprocess ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 1 — Data loading & preprocessing")
    print("=" * 60)
    df, report = load_city_data(data_file, return_report=True)
    report["grid_layout"] = [list(row) for row in CITY_GRID_LAYOUT]
    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"  Cities: {len(report['cities'])}  |  Months: {df.shape[0]}  "
          f"|  Range: {report['start_month']} → {report['end_month']}")
    df.to_csv(output_dir / "monthly_interpolated_city_aqi.csv")

    # EDA figures
    fig01_seasonal_pattern(df, output_dir / "01_seasonal_pattern.png", show_plots)
    fig02_city_bar(df, output_dir / "02_city_bar.png", show_plots)
    fig10_annual_trends(df, output_dir / "10_annual_trends.png", show_plots)
    fig11_category_distribution(df, output_dir / "11_category_distribution.png", show_plots)
    fig12_correlation_heatmap(df, output_dir / "12_correlation_heatmap.png", show_plots)
    print("  EDA figures saved.")

    # ── 2. Build grid frames ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 2 — Spatial frame construction")
    print("=" * 60)
    frames, valid_mask, city_positions = build_grid(df, layout=CITY_GRID_LAYOUT)
    np.save(output_dir / "monthly_aqi_frames.npy", frames)
    print(f"  Frame tensor: {frames.shape}")

    fig03_spatial_map(
        frames[0, ..., 0], valid_mask, CITY_GRID_LAYOUT,
        f"Spatial AQI Map ({df.index[0].strftime('%b %Y')})",
        output_dir / "03_spatial_map_first.png", show_plots,
    )
    fig03_spatial_map(
        frames[..., 0].mean(axis=0), valid_mask, CITY_GRID_LAYOUT,
        "Spatial AQI Map (Dataset Average)",
        output_dir / "03b_spatial_map_avg.png", show_plots,
    )

    # ── 3. Sequences & split ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 3 — Sequence preparation & train/val/test split")
    print("=" * 60)
    X, y = create_sequences(frames, seq_len=seq_len)
    target_dates = df.index[seq_len:]

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(
        X, y, train_ratio=train_ratio, val_ratio=val_ratio
    )
    print(f"  Sequences → train: {len(X_train)}  val: {len(X_val)}  test: {len(X_test)}")

    # ── 4. Normalise ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 4 — Per-cell normalization (StandardScaler per city)")
    print("=" * 60)
    X_train_s, X_val_s, X_test_s, y_train_s, y_val_s, y_test_s, scaler = scale_data(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    print("  Normalization complete.")

    # Save un-augmented scaled training data for ablation study
    X_train_s_orig = X_train_s.copy()
    y_train_s_orig = y_train_s.copy()

    # ── 5. Augmentation ───────────────────────────────────────────────────
    if use_augmentation:
        print("\n" + "=" * 60)
        print(" STEP 5 — Data augmentation")
        print("=" * 60)
        X_train_s, y_train_s = augment_training_data(
            X_train_s, y_train_s,
            enabled=True, copies=augment_copies,
            noise_std=augment_noise_std, seed=42,
        )
        print(f"  Augmented training samples: {len(X_train_s)}")
    else:
        X_train_s_orig = X_train_s
        y_train_s_orig = y_train_s

    # ── 6. & 7. Multi-Seed Training ────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 6 & 7 — Multi-Seed Model Training")
    print("=" * 60)
    
    n_seeds = 5 if run_journal else 1
    seed_metrics = []
    best_model = None
    best_mae = float('inf')
    best_pred_test_s = None
    best_baseline_pred_s = None
    best_pred_test_real = None
    best_baseline_real = None
    best_conv_metrics = None
    
    for seed_idx, seed in enumerate([42, 123, 456, 789, 999][:n_seeds]):
        print(f"\n  --- Training Seed {seed} ({seed_idx+1}/{n_seeds}) ---")
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Build
        model = build_convlstm(
            input_shape=(seq_len, frames.shape[1], frames.shape[2], 1),
            valid_mask=valid_mask,
            learning_rate=8e-4,
        )
        if seed_idx == 0:
            with (output_dir / "model_summary.txt").open("w", encoding="utf-8") as fh:
                model.summary(print_fn=lambda line: fh.write(line + "\n"))
                
        # Train
        ckpt_path = str(output_dir / f"best_weights_seed_{seed}.keras")
        history = train_model(
            model, X_train_s, y_train_s, X_val_s, y_val_s,
            epochs=epochs, batch_size=batch_size,
            checkpoint_path=ckpt_path,
        )
        if seed_idx == 0:
            fig04_loss_curve(history, output_dir / "04_loss_curves.png", show_plots)
            
        # Select the final seed using validation data only. The test set remains
        # untouched until the selected model is evaluated below.
        val_pred_s = model.predict(X_val_s, verbose=0).astype(np.float32)
        val_pred_s[..., 0] = np.where(valid_mask[np.newaxis], val_pred_s[..., 0], 0.0)
        y_val_real = inverse_scale(y_val_s, scaler)
        val_pred_real = inverse_scale(val_pred_s, scaler)
        for arr in (y_val_real, val_pred_real):
            arr[..., 0] = np.where(valid_mask[np.newaxis], arr[..., 0], np.nan)
        val_metrics = compute_metrics(y_val_real, val_pred_real, valid_mask)

        # Evaluate the held-out test set for reporting, but do not use it for
        # model selection.
        baseline_pred_s = baseline_persistence(X_test_s).astype(np.float32)
        pred_test_s     = model.predict(X_test_s, verbose=0).astype(np.float32)
        
        for arr in (baseline_pred_s, pred_test_s, y_test_s):
            arr[..., 0] = np.where(valid_mask[np.newaxis], arr[..., 0], 0.0)
            
        y_test_real    = inverse_scale(y_test_s, scaler)
        baseline_real  = inverse_scale(baseline_pred_s, scaler)
        pred_test_real = inverse_scale(pred_test_s, scaler)
        
        for arr in (y_test_real, baseline_real, pred_test_real):
            arr[..., 0] = np.where(valid_mask[np.newaxis], arr[..., 0], np.nan)
            
        c_metrics = compute_metrics(y_test_real, pred_test_real, valid_mask)
        seed_metrics.append({
            "seed": seed,
            "validation_MAE": val_metrics["MAE"],
            "validation_RMSE": val_metrics["RMSE"],
            "test_MAE": c_metrics["MAE"],
            "test_RMSE": c_metrics["RMSE"],
        })
        print(f"  Seed {seed} validation MAE: {val_metrics['MAE']:.3f}; test MAE: {c_metrics['MAE']:.3f}")
        
        if val_metrics["MAE"] < best_mae:
            best_mae = val_metrics["MAE"]
            best_model = model
            best_pred_test_s = pred_test_s
            best_baseline_pred_s = baseline_pred_s
            best_pred_test_real = pred_test_real
            best_baseline_real = baseline_real
            best_conv_metrics = c_metrics

    # Multi-seed stats
    print("\n  ── Multi-Seed Metrics ────────────────────────────────")
    maes = [m['test_MAE'] for m in seed_metrics]
    rmses = [m['test_RMSE'] for m in seed_metrics]
    print(f"  MAE:  {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"  RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    
    with (output_dir / "multi_seed_metrics.json").open("w", encoding="utf-8") as f:
        json.dump({
            "selection_metric": "validation_MAE",
            "seeds": seed_metrics,
            "MAE_mean": float(np.mean(maes)),
            "MAE_std": float(np.std(maes, ddof=1)),
            "RMSE_mean": float(np.mean(rmses)),
            "RMSE_std": float(np.std(rmses, ddof=1)),
        }, f, indent=2)

    # Use the best model for downstream analysis
    model = best_model
    pred_test_s = best_pred_test_s
    baseline_pred_s = best_baseline_pred_s
    pred_test_real = best_pred_test_real
    baseline_real = best_baseline_real
    conv_metrics = best_conv_metrics

    model.save(output_dir / "convlstm_aqi_model.keras")
    print(f"\n  Best model saved → {output_dir / 'convlstm_aqi_model.keras'}")
    
    # ── 8. Evaluate ────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 8 — Evaluation on test set (Using best seed)")
    print("=" * 60)
    
    base_metrics = compute_metrics(y_test_real, baseline_real, valid_mask)
    
    metrics_df = pd.DataFrame([
        {"Model": "Baseline (Persistence)", **{k: v for k, v in base_metrics.items() if not k.endswith("_values")}},
        {"Model": "ConvLSTM",              **{k: v for k, v in conv_metrics.items() if not k.endswith("_values")}},
    ])
    metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)

    per_city_df = pd.DataFrame(
        per_city_metrics(y_test_real, pred_test_real, city_positions)
    ).sort_values("MAE")
    per_city_df.to_csv(output_dir / "per_city_metrics.csv", index=False)
    per_city_df[["City", "MAE"]].to_csv(output_dir / "per_city_mae.csv", index=False)

    # Baseline per-city MAE for comparison plot
    base_per_city_rows = per_city_metrics(y_test_real, baseline_real, city_positions)
    base_per_city_dict = {row["City"]: row["MAE"] for row in base_per_city_rows}

    pd.DataFrame({
        "Observed_AQI":             conv_metrics["true_values"],
        "Persistence_Predicted_AQI": base_metrics["pred_values"],
        "ConvLSTM_Predicted_AQI":    conv_metrics["pred_values"],
    }).to_csv(output_dir / "observed_vs_predicted_values.csv", index=False)

    print("\n  ── Global metrics ──────────────────────────────────────")
    print(metrics_df.to_string(index=False))

    # Evaluation figures
    test_dates = target_dates[len(y_train) + len(y_val):]
    fig05_scatter(conv_metrics["true_values"], base_metrics["pred_values"],
                  conv_metrics["pred_values"], output_dir / "05_scatter_obs_pred.png", show_plots)
    fig06_error_dist(conv_metrics["true_values"], conv_metrics["pred_values"],
                     base_metrics["pred_values"], output_dir / "06_error_distribution.png", show_plots)
    fig07_per_city_mae(per_city_df, base_per_city_dict,
                       output_dir / "07_per_city_mae.png", show_plots)
    fig08_eval_maps(y_test_real[-1], pred_test_real[-1], valid_mask, CITY_GRID_LAYOUT,
                    test_dates[-1].strftime("%b %Y"), output_dir / "08_eval_maps.png", show_plots)

    test_results_df = build_test_results_df(test_dates, y_test_real, pred_test_real, city_positions)
    test_results_df.to_csv(output_dir / "prediction_results_test_long.csv", index=False)

    fig09_test_timeseries(test_results_df, per_city_df,
                          output_dir / "09_test_timeseries.png", show_plots)
    fig15_per_city_metrics(per_city_df, output_dir / "15_per_city_metrics_panel.png", show_plots)
    print("  All evaluation figures saved.")

    # ── 9. Journal-level diagnostics ──────────────────────────────────────
    journal_results = None
    if run_journal:
        print("\n" + "=" * 60)
        print(" STEP 9 — Journal-level baselines, ablations & paired tests")
        print("=" * 60)
        journal_results = run_journal_diagnostics(
            df=df,
            X_test=X_test,
            y_test_real=y_test_real,
            valid_mask=valid_mask,
            city_positions=city_positions,
            test_dates=test_dates,
            df_train=df.iloc[: seq_len + len(y_train)],
            metrics_df=metrics_df,
            test_results_df=test_results_df,
            output_dir=output_dir,
            seq_len=seq_len,
            show_plots=show_plots,
            min_cv_train_months=min_cv_train_months,
        )

    # ── 10. SARIMA Baseline ───────────────────────────────────────────────
    sarima_results = None
    if run_journal:
        print("\n" + "=" * 60)
        print(" STEP 10 — SARIMA baseline (per-city auto-order)")
        print("=" * 60)
        train_end_idx = seq_len + len(y_train)
        val_end_idx = train_end_idx + len(y_val)
        df_train_full = df.iloc[:train_end_idx]
        df_test_period = df.iloc[val_end_idx:]

        sarima_pred, sarima_logs = run_sarima_baseline(
            df_train=df_train_full,
            df_test=df_test_period,
            test_dates=test_dates,
            city_positions=city_positions,
            grid_shape=(frames.shape[1], frames.shape[2]),
            valid_mask=valid_mask,
        )
        sarima_metrics = compute_metrics(y_test_real, sarima_pred, valid_mask)
        sarima_per_city = per_city_metrics(y_test_real, sarima_pred, city_positions)
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame([{"Model": "SARIMA", **{k: v for k, v in sarima_metrics.items() if not k.endswith("_values")}}]),
        ], ignore_index=True).drop_duplicates(subset=["Model"], keep="first")
        metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(sarima_per_city).to_csv(output_dir / "sarima_per_city_metrics.csv", index=False)

        # Save SARIMA order log
        sarima_log_rows = [{"City": c, "Order": str(v.get("order")), "Seasonal": str(v.get("seasonal_order")), "AIC": v.get("aic")} for c, v in sarima_logs.items()]
        pd.DataFrame(sarima_log_rows).to_csv(output_dir / "sarima_orders.csv", index=False)
        print(f"  SARIMA  MAE={sarima_metrics['MAE']:.3f}  RMSE={sarima_metrics['RMSE']:.3f}  R²={sarima_metrics['R2']:.3f}")
        sarima_results = {"metrics": sarima_metrics, "per_city": sarima_per_city, "predictions": sarima_pred}

    # ── 11. Vanilla LSTM Baseline ─────────────────────────────────────────
    lstm_results = None
    if run_journal:
        print("\n" + "=" * 60)
        print(" STEP 11 — Vanilla LSTM baseline (per-city univariate)")
        print("=" * 60)
        train_end_idx = seq_len + len(y_train)
        val_end_idx = train_end_idx + len(y_val)
        df_train_full = df.iloc[:train_end_idx]
        df_val_period = df.iloc[train_end_idx:val_end_idx]
        df_test_period = df.iloc[val_end_idx:]

        lstm_pred, lstm_histories = run_vanilla_lstm_baseline(
            df_train=df_train_full,
            df_val=df_val_period,
            df_test=df_test_period,
            city_positions=city_positions,
            grid_shape=(frames.shape[1], frames.shape[2]),
            valid_mask=valid_mask,
            seq_len=seq_len,
            epochs=min(epochs, 100),
        )
        lstm_metrics = compute_metrics(y_test_real, lstm_pred, valid_mask)
        lstm_per_city = per_city_metrics(y_test_real, lstm_pred, city_positions)
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame([{"Model": "Vanilla LSTM", **{k: v for k, v in lstm_metrics.items() if not k.endswith("_values")}}]),
        ], ignore_index=True).drop_duplicates(subset=["Model"], keep="first")
        metrics_df.to_csv(output_dir / "metrics_summary.csv", index=False)
        pd.DataFrame(lstm_per_city).to_csv(output_dir / "vanilla_lstm_per_city_metrics.csv", index=False)
        print(f"  LSTM    MAE={lstm_metrics['MAE']:.3f}  RMSE={lstm_metrics['RMSE']:.3f}  R²={lstm_metrics['R2']:.3f}")
        lstm_results = {"metrics": lstm_metrics, "per_city": lstm_per_city, "predictions": lstm_pred}

    # ── 12. Ablation Study ────────────────────────────────────────────────
    ablation_results = None
    if run_journal:
        print("\n" + "=" * 60)
        print(" STEP 12 — ConvLSTM ablation study")
        print("=" * 60)
        input_shape = (seq_len, frames.shape[1], frames.shape[2], 1)

        def _augment_fn(X, y):
            return augment_training_data(X, y, enabled=True, copies=augment_copies, noise_std=augment_noise_std)

        ablation_df = run_ablation_study(
            X_train_s=X_train_s_orig, y_train_s=y_train_s_orig,
            X_val_s=X_val_s, y_val_s=y_val_s,
            X_test_s=X_test_s, y_test_real=y_test_real,
            valid_mask=valid_mask, scaler=scaler,
            input_shape=input_shape, seq_len=seq_len,
            epochs=epochs, batch_size=batch_size,
            augment_fn=_augment_fn, output_dir=str(output_dir),
        )
        ablation_df.to_csv(output_dir / "ablation_results.csv", index=False)
        fig20_ablation_study(ablation_df, output_dir / "20_ablation_study.png", show_plots)
        print("\n  Ablation results:")
        print(ablation_df[["Variant", "MAE", "RMSE", "R2", "NSE", "Epochs_Trained"]].to_string(index=False))
        ablation_results = ablation_df

    # ── 13. Proper Diebold-Mariano Tests ──────────────────────────────────
    dm_proper_results = None
    if run_journal:
        print("\n" + "=" * 60)
        print(" STEP 13 — Proper Diebold-Mariano tests (HAC + HLN correction)")
        print("=" * 60)
        preds_dict = {
            "ConvLSTM": pred_test_real,
            "Persistence": baseline_real,
        }
        if sarima_results is not None:
            preds_dict["SARIMA"] = sarima_results["predictions"]
        if lstm_results is not None:
            preds_dict["Vanilla LSTM"] = lstm_results["predictions"]

        dm_proper_df = run_dm_tests(y_test_real, preds_dict, valid_mask=valid_mask, h=1)
        dm_proper_df.to_csv(output_dir / "dm_test_proper.csv", index=False)
        fig_dm_results(dm_proper_df, str(output_dir / "21_dm_test_proper.png"), show_plots)
        print(dm_proper_df[["Baseline", "Loss_Type", "DM_HLN_Statistic", "P_Value_HLN", "Significant_5pct"]].to_string(index=False))
        dm_proper_results = dm_proper_df
        if journal_results is not None:
            journal_results["dm_proper_tests"] = dm_proper_df

    # ── 13b. Per-city comparison figure (SARIMA + LSTM) ───────────────────
    if run_journal and sarima_results is not None and lstm_results is not None:
        fig19_sarima_comparison(
            per_city_df, sarima_results["per_city"], lstm_results["per_city"],
            output_dir / "19_sarima_lstm_comparison.png", show_plots,
        )

    # ── 13c. Radar comparison chart ───────────────────────────────────────
    if run_journal:
        radar_rows = []
        for _, row in metrics_df.iterrows():
            radar_rows.append(row.to_dict())
        radar_df = pd.DataFrame(radar_rows)
        # Ensure required columns exist
        needed = ["Model", "MAE", "RMSE", "R2", "NSE", "AQI_Category_Accuracy"]
        if all(c in radar_df.columns for c in needed):
            fig21_radar_comparison(radar_df[needed], output_dir / "22_radar_comparison.png", show_plots)

    # ── 14. Time Series Cross-Validation (optional, long-running) ─────────
    tscv_results = None
    if run_tscv:
        print("\n" + "=" * 60)
        print(" STEP 14 — Expanding-window time series cross-validation")
        print("=" * 60)
        cv_df = run_time_series_cv(
            frames=frames, valid_mask=valid_mask, df=df,
            city_positions=city_positions, seq_len=seq_len,
            min_train_months=min_cv_train_months, step_size=6,
            epochs=min(epochs, 100), batch_size=batch_size,
            use_augmentation=use_augmentation, run_baselines=True,
        )
        cv_df.to_csv(output_dir / "tscv_results.csv", index=False)
        cv_summary = summarize_cv_results(cv_df)
        cv_summary.to_csv(output_dir / "tscv_summary.csv")
        fig18_tscv_folds(cv_df, str(output_dir / "18_tscv_folds.png"), show_plots)
        print("\n  Time Series CV Summary:")
        print(cv_summary.to_string())
        tscv_results = cv_df

    # ── 15. Forecast ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 15 — Multi-month recursive forecast + MC-Dropout uncertainty")
    print("=" * 60)
    all_frames_scaled = transform_with_scaler(frames, scaler)
    recent_seq = all_frames_scaled[-seq_len:]

    future_real, _ = multi_step_forecast(
        model, recent_seq, steps=forecast_steps, scaler=scaler, valid_mask=valid_mask
    )
    mc_mean_real, mc_std_real = mc_dropout_forecast(
        model, recent_seq, steps=forecast_steps, scaler=scaler,
        valid_mask=valid_mask, n_passes=mc_passes,
    )

    forecast_start = df.index[-1] + pd.offsets.MonthBegin(1)
    forecast_dates = pd.date_range(forecast_start, periods=forecast_steps, freq="MS")

    forecast_df = build_forecast_df(forecast_dates, future_real, city_positions)
    forecast_df.to_csv(output_dir / "forecast_next_months_long.csv", index=False)

    fig13_forecast_heatmaps(future_real, valid_mask, CITY_GRID_LAYOUT,
                             forecast_dates, output_dir / "13_forecast_heatmaps.png", show_plots)
    fig14_forecast_trends(forecast_df, mc_mean_real, mc_std_real, valid_mask,
                           city_positions, forecast_dates,
                           output_dir / "14_forecast_trends.png", show_plots)

    # ── 16. Research brief ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" STEP 16 — Writing research brief")
    print("=" * 60)
    write_research_brief(
        output_dir, report, seq_len, forecast_steps,
        metrics_df, per_city_df, test_dates, forecast_dates, journal_results,
    )

    print("\n" + "=" * 60)
    print(f" DONE — All outputs saved in: {output_dir.resolve()}")
    print("=" * 60)
    print(f"\n  ConvLSTM  MAE={conv_metrics['MAE']:.3f}  RMSE={conv_metrics['RMSE']:.3f}"
          f"  R²={conv_metrics['R2']:.3f}  NSE={conv_metrics['NSE']:.3f}"
          f"  Cat.Acc={conv_metrics['AQI_Category_Accuracy']*100:.1f}%")
    print(f"  Baseline  MAE={base_metrics['MAE']:.3f}  RMSE={base_metrics['RMSE']:.3f}"
          f"  R²={base_metrics['R2']:.3f}  NSE={base_metrics['NSE']:.3f}")
    improvement = (base_metrics["MAE"] - conv_metrics["MAE"]) / max(base_metrics["MAE"], 1e-8) * 100
    print(f"\n  MAE change vs persistence: {improvement:+.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="20-city monthly AQI ConvLSTM forecasting — complete research pipeline."
    )
    parser.add_argument("--data-file",         default="aqi_20cities_long.csv")
    parser.add_argument("--excel-file",         default=None, help="Legacy Excel alias")
    parser.add_argument("--seq-len",            type=int,   default=12)
    parser.add_argument("--forecast-steps",     type=int,   default=6)
    parser.add_argument("--train-ratio",        type=float, default=0.75)
    parser.add_argument("--val-ratio",          type=float, default=0.10)
    parser.add_argument("--epochs",             type=int,   default=250)
    parser.add_argument("--batch-size",         type=int,   default=8)
    parser.add_argument("--use-augmentation",   action="store_true", default=True)
    parser.add_argument("--no-augmentation",    action="store_false", dest="use_augmentation")
    parser.add_argument("--augment-copies",     type=int,   default=2)
    parser.add_argument("--augment-noise-std",  type=float, default=0.05)
    parser.add_argument("--mc-passes",          type=int,   default=30)
    parser.add_argument("--skip-journal-diagnostics", action="store_true",
                        help="Skip integrated baseline ablations, rolling-origin checks, and DM tests.")
    parser.add_argument("--run-tscv", action="store_true",
                        help="Run expanding-window time series cross-validation (long-running).")
    parser.add_argument("--min-cv-train-months", type=int, default=48,
                        help="Minimum history length for rolling-origin baseline diagnostics.")
    parser.add_argument("--output-dir",         default="outputs")
    parser.add_argument("--no-show-plots",      action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    data_file = args.excel_file if args.excel_file else args.data_file
    run_experiment(
        data_file=data_file,
        seq_len=args.seq_len,
        forecast_steps=args.forecast_steps,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        use_augmentation=args.use_augmentation,
        augment_copies=args.augment_copies,
        augment_noise_std=args.augment_noise_std,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        show_plots=not args.no_show_plots,
        mc_passes=args.mc_passes,
        run_journal=not args.skip_journal_diagnostics,
        run_tscv=(not args.skip_journal_diagnostics) or args.run_tscv,
        min_cv_train_months=args.min_cv_train_months,
    )


if __name__ == "__main__":
    main()

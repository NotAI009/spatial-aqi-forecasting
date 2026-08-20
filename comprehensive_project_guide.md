# Spatial AQI Forecasting: Comprehensive Project Guide

This guide explains the project as a complete research pipeline: raw AQI CSV to monthly spatial frames, ConvLSTM training, validation, baselines, figures, and IEEE-style reporting.

## 1. Project Purpose

The project forecasts monthly AQI across 20 Indian cities. Instead of treating each city as an isolated time series, it places all cities into a fixed 4 x 5 grid and predicts the next monthly AQI map from the previous 12 monthly maps.

The strongest paper narrative is:

> Monthly AQI forecasting benefits from treating city observations as a regional spatiotemporal field. A residual attention ConvLSTM provides stable and competitive performance across random seeds and rolling forecast origins, with strong gains over persistence on the primary chronological holdout.

## 2. Main Files

| File | Role |
|---|---|
| `aqi_20cities_long.csv` | Raw daily AQI data in long format |
| `data_utils.py` | Data loading, monthly aggregation, interpolation, and grid conversion |
| `train.py` | Per-cell scaling, augmentation, training utilities |
| `model.py` | Residual attention ConvLSTM architecture and masked losses |
| `forecast.py` | Recursive multi-step forecasting and MC-Dropout uncertainty |
| `baselines.py` | SARIMA, Vanilla LSTM, and seasonal-naive baselines |
| `evaluate.py` | Global and per-city evaluation metrics |
| `ablation.py` | Architecture ablation experiments |
| `dm_test.py` | Corrected month-level Diebold-Mariano tests |
| `ts_crossval.py` | Expanding-window time-series cross-validation |
| `experiments.py` | Full pipeline orchestration |

## 3. Data Flow

The pipeline performs these steps:

1. Read `aqi_20cities_long.csv`.
2. Parse dates, city names, and AQI values.
3. Aggregate daily AQI to monthly mean AQI.
4. Interpolate missing monthly values.
5. Arrange 20 cities into a fixed 4 x 5 grid.
6. Build 12-month input sequences and next-month targets.
7. Split chronologically into train, validation, and test periods.

The processed data contains 87 monthly frames and 75 supervised samples.

## 4. Model Design

The model input has shape `(12, 4, 5, 1)`. The target is the next `(4, 5, 1)` AQI frame.

Architecture summary:

- ConvLSTM2D(64) with batch normalization and dropout
- ConvLSTM2D(32) with batch normalization and dropout
- ConvLSTM2D(16) with batch normalization
- Spatial attention gate
- Conv2D refinement layer
- Delta prediction head
- Residual addition to the last observed frame

The residual design is important because monthly AQI often evolves from the previous month rather than changing from scratch. The model learns a correction term over the latest observed map.

## 5. Training Strategy

Training uses:

- Per-cell StandardScaler normalization fitted on training data only
- Masked MSE loss over valid city cells
- Adam optimizer
- Early stopping
- ReduceLROnPlateau
- Model checkpointing
- Optional Gaussian input augmentation

Five independent seeds are trained during the full journal run. The final model is selected by validation MAE, which keeps the test set reserved for final reporting.

## 6. Baselines

The project includes simple, classical, and neural baselines:

- Persistence: repeat the latest month.
- SeasonalNaive: repeat the same month from 12 months earlier.
- SeasonalMonthlyMean: historical average for the same calendar month.
- Rolling3Mean: average of the latest 3 months.
- TrainMean: training-set mean.
- LinearTrend6M: short linear trend extrapolation.
- SARIMA: per-city classical seasonal model.
- Vanilla LSTM: per-city neural recurrent model without spatial convolution.

The seasonal-naive baseline is especially important because the model uses a 12-month lookback. A reviewer will expect this comparison.

## 7. Validation Strategy

The current validation design is publication-appropriate because it avoids relying on one fragile test:

- Primary chronological holdout: April 2024 to March 2025.
- Multi-seed training: five independent random seeds.
- Rolling-origin CV: expanding training windows with repeated forecast periods.
- Corrected DM testing: spatially averaged loss per month before inference.
- Ablation study: tests the effect of architecture choices.

The corrected DM test should be interpreted descriptively because the final holdout has only 12 monthly observations.

## 8. Current Results

Primary chronological holdout:

| Model | MAE | RMSE | R2 | Category Accuracy |
|---|---:|---:|---:|---:|
| Persistence | 28.380 | 39.753 | 0.597 | 65.4% |
| ConvLSTM | 22.525 | 30.314 | 0.766 | 71.7% |
| SeasonalNaive | 22.391 | 31.808 | 0.742 | 70.4% |
| Vanilla LSTM | 23.964 | 32.031 | 0.738 | 65.8% |
| SARIMA | 50.537 | 65.403 | -0.091 | 45.4% |

Multi-seed ConvLSTM:

- MAE = 22.63 +/- 1.82
- RMSE = 30.71 +/- 1.86

Rolling-origin CV:

- ConvLSTM MAE = 26.53 +/- 6.44
- Persistence MAE = 27.92 +/- 5.70
- SARIMA MAE = 27.92 +/- 5.70

The current run supports a robust and competitive model, not a claim that ConvLSTM dominates every metric in every setting.

## 9. Known Caveats

- SeasonalNaive slightly beats ConvLSTM on test MAE.
- ConvLSTM has better RMSE, R2, NSE, and category accuracy than SeasonalNaive.
- The corrected DM test vs persistence is not significant at 5%.
- Rolling-origin folds are mixed, although ConvLSTM has the best average.
- SARIMA and persistence are identical in the current TSCV output; this should be explained as fallback behavior or revisited before final submission.
- No meteorology, emissions, wind, traffic, land-use, or satellite variables are included.

These caveats do not kill the paper. They make the paper more honest and help define future work.

## 10. Output Artifacts

Key output files:

- `outputs/metrics_summary.csv`: primary test metrics.
- `outputs/baseline_ablation_metrics.csv`: simple and seasonal baseline comparison.
- `outputs/multi_seed_metrics.json`: five-seed robustness results.
- `outputs/tscv_results.csv`: per-fold expanding-window CV results.
- `outputs/tscv_summary.csv`: summarized rolling-origin CV.
- `outputs/dm_test_proper.csv`: corrected DM tests with HAC and HLN correction.
- `outputs/ablation_results.csv`: architecture ablation results.
- `outputs/per_city_metrics.csv`: city-level metrics.
- `outputs/research_brief.md`: auto-generated research summary.

Important figures:

- `outputs/01_seasonal_pattern.png`
- `outputs/05_scatter_obs_pred.png`
- `outputs/07_per_city_mae.png`
- `outputs/08_eval_maps.png`
- `outputs/14_forecast_trends.png`
- `outputs/16_journal_baseline_comparison.png`
- `outputs/18_tscv_folds.png`
- `outputs/19_sarima_lstm_comparison.png`
- `outputs/20_ablation_study.png`
- `outputs/21_dm_test_proper.png`

## 11. IEEE Paper Positioning

The paper should be positioned as an applied machine learning study with a reproducible validation pipeline. The novelty is not just the ConvLSTM itself; it is the complete regional AQI forecasting workflow:

- 20-city monthly AQI grid construction.
- Annual-cycle 12-month lookback.
- Residual attention ConvLSTM.
- Per-cell normalization and masked losses.
- Comparison against seasonal, classical, and neural baselines.
- Multi-seed and rolling-origin validation.
- Multi-step forecasts with uncertainty.

The final manuscript should explicitly state that additional covariates would likely improve forecasting and are left for future work.

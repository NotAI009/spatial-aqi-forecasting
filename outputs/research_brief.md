# AQI Forecasting Research Brief

This brief summarizes the current manuscript-ready evidence from the spatial AQI forecasting pipeline.

## Abstract

This study presents a spatiotemporal deep learning framework for forecasting monthly Air Quality Index (AQI) across 20 major Indian cities. Daily AQI observations from January 2018 to March 2025 are aggregated into 87 monthly city-level frames and embedded into a fixed 4 x 5 grid. A residual attention ConvLSTM predicts the next monthly AQI map from the previous 12 months, explicitly using a full annual historical cycle. The framework is evaluated against persistence, seasonal-naive, SARIMA, Vanilla LSTM, and additional simple baselines. On the chronological April 2024 to March 2025 holdout, ConvLSTM achieves MAE = 22.525, RMSE = 30.314, R2 = 0.766, NSE = 0.766, and AQI category accuracy = 71.7%, improving MAE over persistence by 20.63%. Across five independent seeds, the model obtains MAE = 22.63 +/- 1.82 and RMSE = 30.71 +/- 1.86. Expanding-window cross-validation gives ConvLSTM the best average MAE, although individual folds are mixed. Corrected Diebold-Mariano testing does not indicate significant superiority over persistence on the 12-month holdout, so the results are interpreted as robust empirical evidence rather than a statistical significance claim.

## Dataset

| Property | Value |
|---|---|
| Source file | `aqi_20cities_long.csv` |
| Date range | January 2018 to March 2025 |
| Total daily rows | 52,940 |
| Missing daily AQI values | 1,383 |
| Cities modeled | 20 |
| Monthly frames | 87 |
| Supervised samples | 75 |
| Grid shape | 4 x 5 |
| Input window | 12 months |
| Test period | April 2024 to March 2025 |

## Method Summary

The model represents city AQI observations as monthly spatial frames. Each supervised sample uses 12 previous frames to predict the next frame. The ConvLSTM architecture includes stacked ConvLSTM layers, batch normalization, dropout, a spatial attention gate, and residual delta prediction over the latest observed frame. Per-cell scaling is fitted on training data only, and masked losses restrict optimization to real city cells.

## Primary Test Results

| Model | MAE | RMSE | R2 | NSE | Category Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence | 28.380 | 39.753 | 0.597 | 0.597 | 65.4% |
| ConvLSTM | 22.525 | 30.314 | 0.766 | 0.766 | 71.7% |
| SeasonalNaive | 22.391 | 31.808 | 0.742 | 0.742 | 70.4% |
| Vanilla LSTM | 23.964 | 32.031 | 0.738 | 0.738 | 65.8% |
| SARIMA | 50.537 | 65.403 | -0.091 | -0.091 | 45.4% |

ConvLSTM has the best RMSE, R2, NSE, and category accuracy. SeasonalNaive has the lowest MAE by a narrow margin, so the manuscript should describe ConvLSTM as competitive and balanced rather than universally dominant.

## Robustness Results

Five-seed ConvLSTM summary:

- MAE = 22.63 +/- 1.82
- RMSE = 30.71 +/- 1.86

Expanding-window CV:

| Model | MAE | RMSE | R2/NSE |
|---|---:|---:|---:|
| ConvLSTM | 26.53 +/- 6.44 | 36.13 +/- 9.80 | 0.69 +/- 0.11 |
| Persistence | 27.92 +/- 5.70 | 39.14 +/- 9.04 | 0.65 +/- 0.09 |
| SARIMA | 27.92 +/- 5.70 | 39.14 +/- 9.04 | 0.65 +/- 0.09 |

## Statistical Testing

The corrected DM test aggregates spatial losses across cities for each month before inference. This avoids treating city-month cells as independent temporal samples. ConvLSTM is significantly better than SARIMA under the corrected test, but it is not significantly better than persistence or Vanilla LSTM at the 5% level on the 12-month holdout.

## Manuscript Interpretation

The correct paper interpretation is:

> The proposed residual attention ConvLSTM offers stable and competitive regional monthly AQI forecasting, with strong persistence-baseline improvement on the primary holdout and the best average performance under rolling-origin validation. The evidence supports spatially structured deep learning as a promising modeling strategy, while seasonal-naive competitiveness and limited holdout length require cautious claims.

## Key Outputs

- `outputs/metrics_summary.csv`
- `outputs/baseline_ablation_metrics.csv`
- `outputs/multi_seed_metrics.json`
- `outputs/tscv_results.csv`
- `outputs/tscv_summary.csv`
- `outputs/dm_test_proper.csv`
- `outputs/ablation_results.csv`
- `outputs/per_city_metrics.csv`
- `ieee_paper_draft.md`

# Spatial AQI Forecasting Project: IEEE-Ready Research Summary

## 1. Project Overview

This project develops a monthly Air Quality Index (AQI) forecasting framework for 20 major Indian cities. The central modeling idea is to treat AQI as a spatiotemporal field rather than as 20 unrelated city-level time series. Daily AQI observations from `aqi_20cities_long.csv` are aggregated into monthly city means, embedded into a fixed 4 x 5 city grid, and modeled using a ConvLSTM network that predicts the next month's AQI map from the previous 12 monthly maps.

The current evidence supports a strong IEEE-style paper when the claims are framed carefully: the model is robust across multiple random seeds, competitive across rolling temporal splits, and substantially better than persistence on the primary chronological holdout. The paper should not claim universal superiority over every seasonal baseline, because the seasonal-naive baseline is slightly better on test MAE in the current run.

## 2. Dataset and Preprocessing

- Source file: `aqi_20cities_long.csv`
- Raw records: 52,940 daily AQI rows
- Cities: 20 Indian cities
- Date range: January 2018 to March 2025
- Missing daily AQI values: 1,383
- Processed panel: 87 monthly frames
- Supervised samples: 75 samples using a 12-month lookback window
- Forecast target: next-month AQI for all 20 cities

Daily records are parsed, cleaned, aggregated to monthly means, interpolated where necessary, and reindexed into a complete monthly sequence. Each month becomes a `(4, 5, 1)` AQI frame. The 12-month input window is deliberately chosen to capture the annual AQI cycle, including winter pollution accumulation and monsoon-period clearing.

## 3. Methodology

The ConvLSTM architecture combines temporal recurrence with spatial convolution:

- Three ConvLSTM2D layers learn month-to-month evolution while preserving the 4 x 5 grid.
- Batch normalization and dropout stabilize training.
- A spatial attention gate weights informative grid locations.
- A residual prediction head learns the delta from the most recent AQI map.
- Masked losses ensure that only valid city cells contribute to optimization.
- Per-cell scaling prevents high-AQI cities from dominating the loss.

The model is trained chronologically without shuffling across time. The final reported model is selected using validation MAE, not test MAE.

## 4. Validation Design

The evaluation was refactored away from an inflated single-split significance story and toward a more defensible empirical validation design:

- Corrected Diebold-Mariano testing aggregates spatial loss by month before testing.
- Five independent ConvLSTM seeds quantify initialization sensitivity.
- Expanding-window rolling-origin cross-validation tests temporal stability.
- Baselines include persistence, seasonal-naive, seasonal monthly mean, rolling mean, train mean, linear trend, SARIMA, and Vanilla LSTM.
- Ablations evaluate the contribution of architectural components.

This is a much stronger story for review because it emphasizes stability, benchmark coverage, and honest uncertainty.

## 5. Main Results

Primary chronological holdout, April 2024 to March 2025:

| Model | MAE | RMSE | R2 | NSE | Category Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence | 28.380 | 39.753 | 0.597 | 0.597 | 65.4% |
| ConvLSTM | 22.525 | 30.314 | 0.766 | 0.766 | 71.7% |
| SeasonalNaive | 22.391 | 31.808 | 0.742 | 0.742 | 70.4% |
| Vanilla LSTM | 23.964 | 32.031 | 0.738 | 0.738 | 65.8% |
| SARIMA | 50.537 | 65.403 | -0.091 | -0.091 | 45.4% |

The ConvLSTM improves test MAE over persistence by 20.63% and has the best RMSE, R2, NSE, and category accuracy among the main comparisons. SeasonalNaive is slightly better on MAE, so the manuscript should describe ConvLSTM as the best overall balance of accuracy and category discrimination, not as the best method on every single metric.

## 6. Multi-Seed Robustness

Across five independent random initializations:

| Metric | Mean | Std |
|---|---:|---:|
| Test MAE | 22.63 | 1.82 |
| Test RMSE | 30.71 | 1.86 |

The selected seed was chosen by validation MAE and achieved test MAE = 22.525 and RMSE = 30.314. The small spread across seeds supports training stability and reduces the risk that results depend on one lucky initialization.

## 7. Rolling-Origin Cross-Validation

Expanding-window time-series CV results:

| Model | MAE | RMSE | R2/NSE |
|---|---:|---:|---:|
| ConvLSTM | 26.53 +/- 6.44 | 36.13 +/- 9.80 | 0.69 +/- 0.11 |
| Persistence | 27.92 +/- 5.70 | 39.14 +/- 9.04 | 0.65 +/- 0.09 |
| SARIMA | 27.92 +/- 5.70 | 39.14 +/- 9.04 | 0.65 +/- 0.09 |

ConvLSTM has the best average cross-validation performance, but individual folds are mixed. This should be reported honestly as evidence of temporal robustness on average, not as proof that ConvLSTM wins every forecast origin.

## 8. Diebold-Mariano Interpretation

The corrected DM implementation uses month-level spatially averaged loss differentials, avoiding artificial inflation of sample size across cities. On the 12-month holdout:

- ConvLSTM vs persistence, MAE loss: HLN-adjusted p = 0.2441
- ConvLSTM vs persistence, MSE loss: HLN-adjusted p = 0.2037
- ConvLSTM vs SARIMA is significant under both MAE and MSE loss
- ConvLSTM vs Vanilla LSTM is not significant at 5%

The paper should not claim statistically significant superiority over persistence. Instead, it should report practical improvement, multi-seed stability, and rolling-origin evidence.

## 9. What Can Still Be Improved Using Only the Existing CSV

Without adding external meteorology, emissions, traffic, satellite, or station metadata, we can still strengthen the work in limited but useful ways:

- Add calendar-derived features such as month-of-year encodings.
- Compare against more seasonal baselines, such as previous-year same-month and month-wise climatology.
- Report median and interquartile range across rolling origins in addition to mean and standard deviation.
- Add a table showing which folds ConvLSTM wins and loses.
- Emphasize error geography: northern high-AQI cities remain hardest.
- Include a limitations section explaining that the dataset is monthly and moderately sized.

These improvements are documentation and evaluation refinements, not new-data breakthroughs. They make the paper more honest and reviewer-resistant.

## 10. Recommended Paper Claim

Use this central claim:

> This study proposes a 12-month residual attention ConvLSTM framework for regional monthly AQI forecasting across 20 Indian cities. The model provides stable performance across five random initializations, improves the primary chronological holdout MAE over persistence by 20.63%, and achieves the best average performance under expanding-window temporal validation. Results show that spatially structured deep forecasting is a competitive and robust approach for monthly regional AQI prediction, while seasonal-naive competitiveness and the limited holdout length motivate cautious interpretation.

Avoid claiming:

- Statistically significant superiority over persistence.
- Universal dominance over all baselines.
- Proof of generalization beyond the observed cities and time range.
- Causal pollution transport, since no wind or emissions covariates are included.

## 11. IEEE Readiness Verdict

The project is ready to support an IEEE-level manuscript if the writing is disciplined. The contribution is not "deep learning wins everything." The stronger contribution is a complete, reproducible spatiotemporal AQI forecasting pipeline with robust validation, honest baseline comparison, uncertainty estimation, and clear limitations.

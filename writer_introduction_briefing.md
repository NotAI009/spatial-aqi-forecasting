# Research Paper Introduction: Writer's Briefing

This briefing gives the exact story to use in the Introduction of the IEEE-style manuscript. The emphasis should be on robust empirical validation, not exaggerated statistical significance.

## 1. Background and Motivation

Air pollution is a major public-health and environmental risk in India. Reliable AQI forecasting can support early warnings, health advisories, traffic or industrial planning, and regional environmental monitoring. Monthly AQI forecasting is useful because many policy and health responses must be planned ahead of time rather than triggered only after unsafe conditions occur.

## 2. Forecasting Problem

This study forecasts next-month AQI for 20 major Indian cities. Daily AQI observations are aggregated into monthly city-level means. Each prediction uses the previous 12 monthly AQI maps, so the model sees one full annual cycle before forecasting the next month.

Verified dataset facts:

- Source: `aqi_20cities_long.csv`
- Period: January 2018 to March 2025
- Raw daily rows: 52,940
- Missing daily AQI values: 1,383, handled during preprocessing
- Monthly frames: 87
- Supervised 12-month sequences: 75
- Forecast evaluation: April 2024 to March 2025

## 3. Why City-by-City Forecasting Is Limited

AQI is not purely local. Nearby cities and regional corridors may share pollution behavior because of meteorology, seasonal conditions, transport patterns, industrial activity, traffic density, and large-scale pollution episodes. A separate SARIMA or Vanilla LSTM model can learn one city's temporal history, but it cannot directly model interactions across the city grid.

The paper should avoid claiming causal pollutant transport unless external wind or emissions data are added. The correct claim is that the spatial grid lets the model learn cross-city statistical dependencies present in the AQI observations.

## 4. Research Gap

Many AQI forecasting approaches rely on independent time-series modeling or use deep models without strong validation across random seeds and forecast origins. This study addresses that gap by combining a spatial grid representation, a 12-month ConvLSTM model, multi-seed evaluation, and expanding-window time-series validation.

## 5. Proposed Solution

The study represents the 20 cities as a fixed 4 x 5 grid and applies a residual attention ConvLSTM to predict the next AQI map from the previous 12 monthly maps.

Key mechanisms:

- Convolution learns local spatial patterns across the city grid.
- ConvLSTM recurrence learns temporal evolution across the annual cycle.
- Residual learning predicts the correction from the most recent AQI map.
- Spatial attention weights informative locations.
- Per-city scaling prevents high-AQI cities from dominating training.
- Masked loss ensures only valid city cells are evaluated.

## 6. Experimental Comparisons

The paper should present these comparisons:

- Persistence baseline: next month equals last month.
- SeasonalNaive baseline: next month equals the same month 12 months earlier.
- SeasonalMonthlyMean baseline: historical average for the same calendar month.
- SARIMA: classical univariate statistical baseline.
- Vanilla LSTM: univariate neural baseline without spatial convolution.
- Ablations: attention, residual connection, augmentation, depth, and batch normalization.
- Multi-seed runs: five independent ConvLSTM initializations.
- Rolling-origin CV: expanding-window temporal validation.

## 7. Main Results to Report

Primary chronological holdout:

- ConvLSTM: MAE = 22.525, RMSE = 30.314, R2 = 0.766, category accuracy = 71.7%.
- Persistence: MAE = 28.380, RMSE = 39.753, R2 = 0.597, category accuracy = 65.4%.
- MAE improvement over persistence: 20.63%.
- Multi-seed ConvLSTM: MAE = 22.63 +/- 1.82, RMSE = 30.71 +/- 1.86.
- Rolling-origin CV: ConvLSTM average MAE = 26.53 +/- 6.44; persistence average MAE = 27.92 +/- 5.70.

Important nuance:

- SeasonalNaive has slightly lower test MAE than ConvLSTM: 22.391 vs 22.525.
- ConvLSTM has better RMSE, R2, NSE, and category accuracy than SeasonalNaive.
- The corrected DM test vs persistence is not significant at 5%.

## 8. Safe Contribution Statement

Use this contribution framing:

1. A reproducible spatiotemporal AQI forecasting pipeline for 20 Indian cities using monthly AQI maps.
2. A 12-month residual attention ConvLSTM model designed around annual AQI seasonality.
3. A fixed 4 x 5 city-grid representation enabling joint regional modeling.
4. Comprehensive comparison against persistence, seasonal-naive, SARIMA, Vanilla LSTM, and ablated variants.
5. Robust validation through five random seeds and expanding-window rolling-origin CV.
6. Recursive multi-month forecasting with MC-Dropout uncertainty bands.

## 9. Claims to Avoid

Do not write:

- "ConvLSTM proves architectural superiority."
- "The model is statistically significantly better than persistence."
- "ConvLSTM beats every baseline."
- "The model captures physical pollutant transport."
- "The approach generalizes to all Indian cities."

Write instead:

> The proposed ConvLSTM framework achieves stable and competitive monthly AQI forecasting performance, with strong gains over persistence on the primary chronological holdout and the best average performance under rolling-origin validation. The results support spatially structured deep learning as a promising approach for regional AQI forecasting, while seasonal-naive competitiveness and the small number of holdout months motivate cautious interpretation.

## 10. Recommended Introduction Structure

1. Broad public-health importance of AQI forecasting.
2. Monthly AQI forecasting challenge: seasonality and regional variation.
3. Limitation of independent city-wise forecasting.
4. Need for spatial-temporal modeling and stronger validation.
5. Proposed 4 x 5 grid and 12-month ConvLSTM approach.
6. Evaluation design: baselines, ablations, multi-seed runs, rolling-origin CV.
7. Contributions and concise result preview with cautious language.

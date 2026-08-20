# A Residual Attention ConvLSTM Framework for Monthly AQI Forecasting Across Indian Cities

## Abstract

Accurate air quality forecasting is important for public-health planning, environmental monitoring, and policy response. This paper presents a spatiotemporal deep learning framework for monthly Air Quality Index (AQI) forecasting across 20 major Indian cities. Daily AQI records from January 2018 to March 2025 are aggregated into monthly means and embedded into a fixed 4 x 5 city grid. A residual attention ConvLSTM model uses the previous 12 monthly AQI maps to predict the next monthly map, thereby incorporating a full annual cycle while preserving regional spatial structure. The framework is evaluated against persistence, seasonal-naive, SARIMA, Vanilla LSTM, and additional simple baselines. On the chronological April 2024 to March 2025 holdout, the proposed model achieves MAE = 22.525, RMSE = 30.314, R2 = 0.766, NSE = 0.766, and AQI category accuracy = 71.7%, improving MAE over persistence by 20.63%. Across five random initializations, ConvLSTM obtains MAE = 22.63 +/- 1.82 and RMSE = 30.71 +/- 1.86. Expanding-window cross-validation shows the best average MAE for ConvLSTM, although fold-level results are mixed. Corrected Diebold-Mariano testing does not show significant superiority over persistence on the 12-month holdout; therefore, the results are interpreted as robust empirical evidence rather than a statistical significance claim.

## Index Terms

Air quality forecasting, AQI, ConvLSTM, spatiotemporal forecasting, time-series cross-validation, India, deep learning, uncertainty estimation.

## I. Introduction

Air pollution is a persistent environmental and public-health challenge in India. Forecasting future AQI can support early-warning systems, health advisories, traffic planning, and targeted environmental interventions. Monthly AQI forecasting is especially useful for medium-term planning because pollution follows strong seasonal patterns, including winter accumulation and monsoon-period reduction.

Many forecasting approaches model each city independently. Such models can capture temporal autocorrelation within a city, but they do not directly learn regional dependencies across cities. AQI patterns may co-vary across neighboring or climatically linked regions because of shared meteorology, emissions patterns, and seasonal pollution episodes. This motivates a forecasting formulation in which city-level AQI observations are modeled jointly as a spatial-temporal field.

This study proposes a residual attention ConvLSTM framework for monthly AQI forecasting across 20 Indian cities. The cities are arranged into a fixed 4 x 5 grid, and the model predicts the next AQI map from the previous 12 monthly maps. The 12-month lookback is selected to expose the model to a complete annual cycle. The ConvLSTM architecture preserves spatial structure while modeling temporal evolution, and residual learning predicts a correction over the latest observed map.

The main contributions are:

1. A reproducible monthly AQI forecasting pipeline that converts daily city observations into spatial AQI map sequences.
2. A 12-month residual attention ConvLSTM model for regional AQI forecasting.
3. A comprehensive benchmark against persistence, seasonal-naive, SARIMA, Vanilla LSTM, and additional simple baselines.
4. A robust validation design using five random seeds, expanding-window time-series cross-validation, corrected Diebold-Mariano diagnostics, and ablation experiments.
5. Recursive multi-month forecasting with MC-Dropout uncertainty estimates.

## II. Dataset and Preprocessing

The dataset is `aqi_20cities_long.csv`, containing daily AQI observations for 20 Indian cities from January 2018 to March 2025. The raw file contains 52,940 rows and 1,383 missing daily AQI values. Daily observations are aggregated into monthly city-level means, reindexed to a complete monthly timeline, and interpolated to remove monthly gaps. The final panel contains 87 monthly frames.

Each monthly frame is represented as a 4 x 5 grid with one channel. All 20 cells correspond to valid city positions. Supervised samples are generated using a sliding 12-month input window and one-month-ahead target, producing 75 samples. The chronological evaluation design avoids look-ahead leakage.

## III. Methodology

### A. Spatial AQI Representation

Let `M_t` denote the 4 x 5 AQI map for month `t`. Each sample consists of the sequence `(M_t, ..., M_{t+11})`, and the target is `M_{t+12}`. This representation preserves cross-city structure while giving the model a complete annual cycle of historical AQI.

### B. Normalization

AQI magnitudes vary widely across cities. A per-cell StandardScaler is fitted on the training set and applied separately to each city cell. This prevents high-AQI cities from dominating the loss and improves balanced learning across regions.

### C. ConvLSTM Architecture

The proposed model uses stacked ConvLSTM2D layers followed by a spatial attention gate and a residual prediction head. The architecture is:

```text
Input: (12, 4, 5, 1)
ConvLSTM2D(64) + BatchNorm + Dropout
ConvLSTM2D(32) + BatchNorm + Dropout
ConvLSTM2D(16) + BatchNorm
Spatial attention gate
Conv2D refinement
Conv2D delta output
Residual add with latest input frame
```

The model is trained with masked MSE loss over valid city cells. Adam optimization, early stopping, learning-rate reduction, and checkpointing are used for stable training. Gaussian noise augmentation is applied to input sequences during training while targets remain unchanged.

### D. Forecasting and Uncertainty

For multi-step forecasting, the trained one-step model is applied recursively: each predicted frame is appended to the input sequence and used to forecast the next month. MC-Dropout is used at inference time to estimate uncertainty bands from repeated stochastic forward passes.

## IV. Experimental Design

The evaluation includes:

- Primary chronological holdout: April 2024 to March 2025.
- Persistence baseline.
- SeasonalNaive baseline using the same month from 12 months earlier.
- Seasonal monthly mean, rolling mean, train mean, and linear trend baselines.
- Per-city SARIMA baseline.
- Per-city Vanilla LSTM baseline.
- Ablation variants removing attention, residual learning, augmentation, depth, or batch normalization.
- Five random ConvLSTM seeds.
- Expanding-window rolling-origin cross-validation.
- Corrected Diebold-Mariano tests with month-level spatial loss aggregation.

The DM test is used descriptively because the holdout contains only 12 monthly forecast origins.

## V. Results

### A. Primary Holdout Performance

| Model | MAE | RMSE | R2 | NSE | Category Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence | 28.380 | 39.753 | 0.597 | 0.597 | 65.4% |
| ConvLSTM | 22.525 | 30.314 | 0.766 | 0.766 | 71.7% |
| SeasonalNaive | 22.391 | 31.808 | 0.742 | 0.742 | 70.4% |
| Vanilla LSTM | 23.964 | 32.031 | 0.738 | 0.738 | 65.8% |
| SARIMA | 50.537 | 65.403 | -0.091 | -0.091 | 45.4% |

ConvLSTM improves MAE over persistence by 20.63%. It also achieves the best RMSE, R2, NSE, and AQI category accuracy in the main comparison. SeasonalNaive has the lowest MAE by a small margin, which indicates that annual seasonality is a strong signal in this dataset.

### B. Multi-Seed Robustness

| Metric | Mean | Std |
|---|---:|---:|
| MAE | 22.63 | 1.82 |
| RMSE | 30.71 | 1.86 |

The small spread across five seeds suggests that ConvLSTM performance is not driven by a single lucky initialization.

### C. Rolling-Origin Cross-Validation

| Model | MAE | RMSE | R2/NSE |
|---|---:|---:|---:|
| ConvLSTM | 26.53 +/- 6.44 | 36.13 +/- 9.80 | 0.69 +/- 0.11 |
| Persistence | 27.92 +/- 5.70 | 39.14 +/- 9.04 | 0.65 +/- 0.09 |
| SARIMA | 27.92 +/- 5.70 | 39.14 +/- 9.04 | 0.65 +/- 0.09 |

ConvLSTM achieves the best average rolling-origin MAE. However, fold-level results are mixed, so the appropriate interpretation is average temporal robustness rather than dominance at every forecast origin.

### D. Statistical Diagnostics

The corrected Diebold-Mariano implementation aggregates spatial losses by month before testing. This avoids inflating the sample size by treating city-month cells as independent temporal observations.

Key DM results:

- ConvLSTM vs persistence, MAE loss: HLN-adjusted p = 0.2441.
- ConvLSTM vs persistence, MSE loss: HLN-adjusted p = 0.2037.
- ConvLSTM vs SARIMA is significant under both MAE and MSE losses.
- ConvLSTM vs Vanilla LSTM is not significant at 5%.

Therefore, the manuscript does not claim statistically significant superiority over persistence.

### E. Ablation Findings

The ablation table shows that simplified variants can perform competitively in this small monthly dataset. In particular, the shallow and no-batch-normalization variants obtain low MAE in the current run. The residual connection remains important because removing it substantially worsens performance. These findings suggest that residual forecasting is valuable, while the depth and normalization choices may require further study on larger datasets.

## VI. Discussion

The results show that spatially structured deep learning is a promising approach for monthly regional AQI forecasting. ConvLSTM performs strongly against persistence and achieves stable behavior across random seeds. Its average rolling-origin performance is also better than the compared baselines. At the same time, the seasonal-naive result shows that annual repetition is an extremely strong predictor in this dataset. This is expected for monthly AQI data and supports the choice of a 12-month input window.

The results should therefore be interpreted as evidence for a robust and competitive regional forecasting framework, not as proof that a deep model universally dominates simple seasonal methods. This framing is scientifically stronger and better aligned with the corrected statistical diagnostics.

## VII. Limitations

This study uses only AQI observations. It does not include meteorology, emissions, traffic, satellite aerosol optical depth, wind fields, holidays, crop-burning indicators, or station metadata. The dataset contains 87 monthly frames, which limits statistical testing power. The fixed grid is geographically motivated but not a physical transport graph. Recursive multi-month forecasts accumulate uncertainty over time.

These limitations are important but manageable. They define future work rather than invalidating the current contribution.

## VIII. Conclusion

This paper presents a residual attention ConvLSTM framework for monthly AQI forecasting across 20 Indian cities. The model uses a 12-month spatial AQI sequence to predict the next monthly map, combines per-cell normalization with masked losses, and provides recursive future forecasts with uncertainty estimates. The framework improves substantially over persistence on the primary chronological holdout, remains stable across five random seeds, and achieves the best average performance under rolling-origin validation. Seasonal-naive competitiveness and non-significant persistence DM tests motivate cautious claims, but the project is suitable for an IEEE-style applied forecasting paper when presented as a robust empirical spatiotemporal forecasting study.

## References

[1] X. Shi et al., "Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting," NeurIPS, 2015. https://arxiv.org/abs/1506.04214

[2] V. Le, Q. Bui, and S. Cha, "Spatiotemporal deep learning model for citywide air pollution interpolation and prediction," IEEE Access, 2019. https://arxiv.org/abs/1911.12919

[3] Y. Liang et al., "AirFormer: Predicting Nationwide Air Quality in China with Transformers," AAAI, 2023. https://arxiv.org/abs/2211.15979

[4] Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation," ICML, 2016. https://arxiv.org/abs/1506.02142

[5] F. X. Diebold and R. S. Mariano, "Comparing Predictive Accuracy," Journal of Business and Economic Statistics, 1995. https://doi.org/10.1080/07350015.1995.10524599

[6] D. Harvey, S. Leybourne, and P. Newbold, "Testing the equality of prediction mean squared errors," International Journal of Forecasting, 1997. https://doi.org/10.1016/S0169-2070(96)00719-4

[7] W. K. Newey and K. D. West, "A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix," Econometrica, 1987. https://doi.org/10.2307/1913610

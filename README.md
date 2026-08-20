# spatial-aqi-forecasting

Research-grade monthly AQI forecasting for 20 Indian cities using a spatiotemporal ConvLSTM pipeline.

The project converts daily AQI records from `aqi_20cities_long.csv` into monthly 4 x 5 AQI map frames. A residual attention ConvLSTM uses the previous 12 monthly maps to predict the next month's AQI map. Multi-month forecasts are generated recursively, and uncertainty is estimated with Monte Carlo Dropout.

## Current Research Position

This repository is suitable as the empirical foundation for an IEEE-style paper when claims are framed carefully. The ConvLSTM shows strong gains over persistence on the primary chronological holdout and stable performance across multiple random seeds. Rolling-origin validation shows the best average performance for ConvLSTM, although individual folds are mixed. SeasonalNaive is a very competitive baseline and slightly outperforms ConvLSTM on test MAE, so the paper should avoid claiming universal dominance.

## Dataset

Source file:

```text
aqi_20cities_long.csv
```

Expected columns:

```text
Date, City, AQI
```

Current file summary:

- Daily rows: 52,940
- Date range: January 2018 to March 2025
- Cities: 20
- Daily AQI missing values: 1,383
- Processed monthly frames: 87
- Supervised 12-month sequences: 75

## Pipeline

1. Load daily long-format AQI data.
2. Aggregate daily AQI to monthly city means.
3. Interpolate missing monthly values.
4. Embed 20 cities into a fixed 4 x 5 grid.
5. Build 12-month input windows and next-month targets.
6. Fit per-cell StandardScaler objects on training data only.
7. Train the residual attention ConvLSTM.
8. Evaluate against persistence, seasonal, SARIMA, Vanilla LSTM, and ablation baselines.
9. Run corrected DM diagnostics, multi-seed training, and rolling-origin CV.
10. Generate figures, CSV outputs, and manuscript-ready summaries.

## Model

Input shape:

```text
(seq_len=12, rows=4, cols=5, channels=1)
```

Architecture:

```text
ConvLSTM2D(64) + BatchNorm + Dropout
ConvLSTM2D(32) + BatchNorm + Dropout
ConvLSTM2D(16) + BatchNorm
Spatial attention gate
Conv2D refinement
Conv2D delta head
Residual add with latest input frame
```

Important design choices:

- 12-month lookback captures annual AQI seasonality.
- Per-cell scaling prevents high-AQI cities from dominating training.
- Masked loss evaluates only valid city cells.
- Residual prediction stabilizes next-month forecasting.
- MC-Dropout provides uncertainty bands for recursive forecasts.

## Main Results

Primary chronological holdout, April 2024 to March 2025:

| Model | MAE | RMSE | R2 | NSE | Category Accuracy |
|---|---:|---:|---:|---:|---:|
| Persistence | 28.380 | 39.753 | 0.597 | 0.597 | 65.4% |
| ConvLSTM | 22.525 | 30.314 | 0.766 | 0.766 | 71.7% |
| SeasonalNaive | 22.391 | 31.808 | 0.742 | 0.742 | 70.4% |
| Vanilla LSTM | 23.964 | 32.031 | 0.738 | 0.738 | 65.8% |
| SARIMA | 50.537 | 65.403 | -0.091 | -0.091 | 45.4% |

ConvLSTM improves MAE over persistence by 20.63% and achieves the best RMSE, R2, NSE, and AQI category accuracy in the main comparison.

Multi-seed ConvLSTM robustness:

- Test MAE: 22.63 +/- 1.82
- Test RMSE: 30.71 +/- 1.86

Rolling-origin CV:

- ConvLSTM MAE: 26.53 +/- 6.44
- Persistence MAE: 27.92 +/- 5.70
- SARIMA MAE: 27.92 +/- 5.70

Corrected Diebold-Mariano testing does not show significant superiority over persistence on the 12-month holdout, so statistical significance is not claimed.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Full research run:

```bash
python experiments.py --no-show-plots
```

Quick smoke test:

```bash
python experiments.py --epochs 3 --forecast-steps 2 --mc-passes 3 --no-show-plots
```

Useful options:

- `--data-file aqi_20cities_long.csv`
- `--seq-len 12`
- `--forecast-steps 6`
- `--epochs 250`
- `--batch-size 8`
- `--mc-passes 30`
- `--no-augmentation`
- `--skip-journal-diagnostics`
- `--run-tscv`
- `--output-dir outputs`

## Key Outputs

| File | Description |
|---|---|
| `outputs/metrics_summary.csv` | Primary test metrics |
| `outputs/baseline_ablation_metrics.csv` | Simple and seasonal baseline comparison |
| `outputs/multi_seed_metrics.json` | Five-seed robustness results |
| `outputs/tscv_results.csv` | Per-fold rolling-origin CV results |
| `outputs/tscv_summary.csv` | CV mean +/- std summary |
| `outputs/dm_test_proper.csv` | Corrected DM tests with HAC and HLN correction |
| `outputs/ablation_results.csv` | Architecture ablation results |
| `outputs/per_city_metrics.csv` | City-level metrics |
| `outputs/research_brief.md` | Auto-generated research brief |
| `ieee_paper_draft.md` | Full IEEE-style manuscript draft |

## Key Figures

- `outputs/01_seasonal_pattern.png`: seasonal AQI heatmap
- `outputs/05_scatter_obs_pred.png`: observed vs predicted AQI
- `outputs/07_per_city_mae.png`: per-city MAE comparison
- `outputs/08_eval_maps.png`: observed, predicted, and error maps
- `outputs/14_forecast_trends.png`: recursive forecasts with uncertainty
- `outputs/16_journal_baseline_comparison.png`: baseline comparison
- `outputs/18_tscv_folds.png`: rolling-origin fold results
- `outputs/19_sarima_lstm_comparison.png`: SARIMA and LSTM comparison
- `outputs/20_ablation_study.png`: ablation results
- `outputs/21_dm_test_proper.png`: corrected DM diagnostics

## Manuscript Claim

Recommended claim:

> The proposed residual attention ConvLSTM provides stable and competitive monthly AQI forecasts across 20 Indian cities, improves substantially over persistence on the primary chronological holdout, and achieves the best average performance in rolling-origin validation. The results support spatially structured deep learning as a promising approach for regional AQI forecasting, while the competitiveness of seasonal-naive forecasting and the limited test horizon motivate cautious interpretation.

## References

1. Shi et al. (2015), ConvLSTM: https://arxiv.org/abs/1506.04214
2. Le, Bui, and Cha (2019), ConvLSTM for air pollution prediction: https://arxiv.org/abs/1911.12919
3. Liang et al. (2023), AirFormer: https://arxiv.org/abs/2211.15979
4. Gal and Ghahramani (2016), MC-Dropout: https://arxiv.org/abs/1506.02142
5. Diebold and Mariano (1995), predictive accuracy comparison: https://doi.org/10.1080/07350015.1995.10524599
6. Harvey, Leybourne, and Newbold (1997), finite-sample DM correction: https://doi.org/10.1016/S0169-2070(96)00719-4
7. Newey and West (1987), HAC covariance: https://doi.org/10.2307/1913610

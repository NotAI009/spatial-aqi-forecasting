# spatial-aqi-forecasting

Research-grade monthly AQI forecasting for 20 Indian cities using a deep spatiotemporal ConvLSTM pipeline.

The project converts daily city AQI records from `aqi_20cities_long.csv` into monthly 4×5 AQI map frames. The model uses the previous 12 monthly maps to predict the next month, capturing the dominant annual AQI seasonality. For multi-month outlooks, each predicted month is recursively fed back into the input sequence. Uncertainty is quantified with Monte Carlo Dropout.

## Core Idea

AQI is simultaneously temporal (seasonal cycles, emission trends) and spatial (inter-city pollution transport, regional meteorology). Instead of 20 independent time-series models, all cities are embedded in a fixed 4 × 5 geographic grid and a deep ConvLSTM model is trained on this spatiotemporal tensor:

- Convolutional filters learn inter-city spatial structure from each monthly AQI map
- Recurrent ConvLSTM state captures month-to-month temporal dynamics
- A residual skip connection (last observed frame + delta) eases training by learning corrections rather than absolute values
- A spatial attention gate focuses the model on the most informative grid regions
- Per-cell StandardScaler normalization (one scaler per city) prevents high-AQI northern cities from dominating the loss

## Dataset

```
aqi_20cities_long.csv
```

Expected columns: `Date, City, AQI`

Current file summary:
- Daily rows: 52,940
- Date range: 2018-01-01 to 2025-03-31
- Cities: 20
- Daily AQI missing values: 1,383 (filled by monthly interpolation)

## Pipeline Steps

### 1. Data Loading & Preprocessing
- Reads `aqi_20cities_long.csv` (daily long format)
- Parses dates, cities, and AQI; aggregates to monthly mean
- Reindexes to complete monthly timeline; interpolates gaps
- Saves `monthly_interpolated_city_aqi.csv` and `preprocessing_report.json`

### 2. Spatial Frame Construction
- Converts each month to a 4 × 5 × 1 AQI map frame
- Saves `monthly_aqi_frames.npy`

### 3. Sequence Preparation
- Input window: 12 previous monthly maps (configurable)
- Target: next monthly map
- Chronological split: no shuffling, no data leakage

### 4. Normalization
- **Per-cell StandardScaler**: one `StandardScaler` fitted per grid position (city)
- Eliminates scale imbalance between high-AQI northern cities (Delhi ~200) and low-AQI southern cities (Thiruvananthapuram ~62)

### 5. Data Augmentation
- Gaussian noise added to input sequences (2 copies per training sample)
- Noise in standardised units (σ = 0.05); targets unchanged

### 6. Model Architecture
```
Input: (seq_len=12, 4, 5, 1)
├─ ConvLSTM2D(64, 3×3) + BatchNorm + Dropout(0.20)    [return_sequences=True]
├─ ConvLSTM2D(32, 3×3) + BatchNorm + Dropout(0.20)    [return_sequences=True]
├─ ConvLSTM2D(16, 3×3) + BatchNorm                    [return_sequences=False]
├─ Spatial Attention Gate (2-layer Conv)
├─ Conv2D(32, 3×3, relu) + BatchNorm
├─ Conv2D(1,  1×1, linear)    ← delta output
└─ Residual Add: last input frame + delta  → prediction
```
- L2 regularization (1e-5) on all ConvLSTM/Conv layers
- Adam optimizer with initial learning rate 8e-4 and ReduceLROnPlateau

### 7. Training
- EarlyStopping (patience=40) restores best val weights
- ReduceLROnPlateau (factor=0.6, patience=15) handles plateaus
- ModelCheckpoint saves best weights during training

### 8. Evaluation
Compares ConvLSTM against persistence and journal-style classical baselines on the chronological test set:
- MAE, RMSE, R², Bias, P95 error
- MAPE, SMAPE
- Nash-Sutcliffe Efficiency (NSE)
- Willmott's Index of Agreement (d)
- Theil's U statistic (< 1 = beats persistence)
- AQI category accuracy (CPCB 6-class scheme)
- Baseline/ablation checks: persistence, rolling 3-month mean, 6-month linear trend, train mean, seasonal monthly mean
- Rolling-origin baseline cross-validation
- Diebold-Mariano style paired loss tests

### 9. Journal Diagnostics
- Integrated into `experiments.py` by default
- Saves `baseline_ablation_metrics.csv`, `diebold_mariano_tests.csv`, and `rolling_origin_baselines.csv`
- Produces publication figures `16_journal_baseline_comparison.png` and `17_dm_test_summary.png`
- Can be skipped with `--skip-journal-diagnostics` for fast debugging

### 10. SARIMA Baseline
- Per-city SARIMA(p,d,q)×(P,D,Q,12) with auto-order selection via AIC grid search
- Fallback chain: optimal → ARIMA(1,1,1)×(1,1,1,12) → AR(1)×(1,0,0,12) → persistence
- Saves per-city fitted orders and AIC values

### 11. Vanilla LSTM Baseline
- Per-city univariate LSTM(64) → Dropout(0.2) → Dense(32) → Dense(1)
- Independent StandardScaler per city, EarlyStopping, ReduceLROnPlateau
- Proves spatial convolution is the contribution, not just recurrence

### 12. Ablation Study
Systematic ablation of ConvLSTM design choices:
- `full` — original model (reference)
- `no_attention` — remove spatial attention gate
- `no_residual` — remove residual skip connection
- `no_augmentation` — train without Gaussian noise augmentation
- `shallow` — single ConvLSTM layer instead of 3
- `no_batchnorm` — remove BatchNormalization

### 13. Proper Diebold-Mariano Tests
- Publication-grade DM test with Newey-West HAC standard errors
- Harvey-Leybourne-Newbold (1997) finite-sample correction
- Tests ConvLSTM against all baselines (persistence, SARIMA, Vanilla LSTM)
- Both MSE and MAE loss differentials

### 14. Time Series Cross-Validation (Optional)
- Expanding-window CV with ConvLSTM retrained from scratch per fold
- Minimum training window: 48 months, step size: 6 months
- Also runs persistence and SARIMA on each fold for comparison
- Enabled with `--run-tscv` flag (long-running)

### 15. Multi-Step Forecasting
- Deterministic recursive forecast (6 months ahead by default)
- Monte Carlo Dropout forecast (30 stochastic passes) → mean + ±1σ/±2σ uncertainty bands

## Setup

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run

Full research run (IEEE level):
```bash
python experiments.py --no-show-plots
```

The full run includes preprocessing, training, evaluation, SARIMA/LSTM baselines, ablation study, proper DM tests, forecasting, and the research brief.

With time series cross-validation (long-running):
```bash
python experiments.py --no-show-plots --run-tscv
```

Quick smoke test:
```bash
python experiments.py --epochs 3 --forecast-steps 2 --mc-passes 3 --no-show-plots
```

Key options:
- `--data-file aqi_20cities_long.csv`
- `--seq-len 12`         (input window: months of history)
- `--forecast-steps 6`   (months to forecast ahead)
- `--epochs 250`         (max epochs; early stopping applies)
- `--batch-size 8`
- `--mc-passes 30`       (MC-Dropout forward passes for uncertainty)
- `--no-augmentation`    (disable noisy augmentation)
- `--skip-journal-diagnostics` (skip baselines/ablation/DM tests for fast debugging)
- `--run-tscv`           (enable expanding-window time series cross-validation)
- `--output-dir outputs`

## Outputs

All artifacts saved in `outputs/`:

### Data & Model
| File | Description |
|---|---|
| `convlstm_aqi_model.keras` | Trained model weights |
| `model_summary.txt` | Layer architecture and parameter count |
| `monthly_interpolated_city_aqi.csv` | Cleaned monthly city AQI table |
| `monthly_aqi_frames.npy` | Spatial frame tensor (T, 4, 5, 1) |
| `preprocessing_report.json` | Data quality and provenance report |
| `metrics_summary.csv` | ConvLSTM vs baseline global metrics |
| `baseline_ablation_metrics.csv` | ConvLSTM compared with persistence, rolling mean, linear trend, train mean, and seasonal monthly mean |
| `diebold_mariano_tests.csv` | Paired Diebold-Mariano style loss tests against baseline forecasts |
| `rolling_origin_baselines.csv` | Rolling-origin baseline cross-validation summary |
| `per_city_metrics.csv` | MAE, RMSE, Bias, NSE per city |
| `per_city_mae.csv` | Compact city MAE table |
| `observed_vs_predicted_values.csv` | All test-set prediction pairs |
| `prediction_results_test_long.csv` | Full test predictions with categories |
| `forecast_next_months_long.csv` | Future month forecasts |
| `research_brief.md` | Complete research paper section |
| `journal_readiness_notes.md` | Scoped notes for what is journal-ready and what remains future work |

### Figures (17 publication-quality plots)
| File | Description |
|---|---|
| `01_seasonal_pattern.png` | City × month AQI heatmap (seasonality) |
| `02_city_bar.png` | Average AQI per city, colour-coded by CPCB category |
| `03_spatial_map_first.png` | First-month spatial AQI grid |
| `03b_spatial_map_avg.png` | Dataset-average spatial AQI grid |
| `04_loss_curves.png` | Train/val loss (linear + log scale) |
| `05_scatter_obs_pred.png` | Observed vs predicted scatter (model vs baseline) |
| `06_error_distribution.png` | Signed/absolute error histograms + box plot |
| `07_per_city_mae.png` | Per-city MAE: ConvLSTM vs persistence (grouped bars) |
| `08_eval_maps.png` | Observed / Predicted / Error spatial maps (last test month) |
| `09_test_timeseries.png` | Test period time series — best & hardest cities |
| `10_annual_trends.png` | Year-over-year annual mean AQI trends |
| `11_category_distribution.png` | AQI category stacked bar per city |
| `12_correlation_heatmap.png` | City-city monthly AQI correlation matrix |
| `13_forecast_heatmaps.png` | Recursive forecast spatial maps (6 months) |
| `14_forecast_trends.png` | City forecast lines + MC-Dropout uncertainty bands |
| `15_per_city_metrics_panel.png` | NSE / Bias / Category accuracy per city |
| `16_journal_baseline_comparison.png` | Baseline/ablation MAE and RMSE comparison |
| `17_dm_test_summary.png` | Paired loss difference summary for DM-style tests |
| `18_tscv_folds.png` | Time series CV: per-fold MAE + box plot (if `--run-tscv`) |
| `19_sarima_lstm_comparison.png` | Per-city MAE: ConvLSTM vs SARIMA vs Vanilla LSTM |
| `20_ablation_study.png` | Ablation study: MAE / NSE / R² by variant |
| `21_dm_test_proper.png` | Proper DM tests with HAC + HLN correction |
| `22_radar_comparison.png` | Multi-metric radar chart across all models |

### IEEE-Level Data Files
| File | Description |
|---|---|
| `sarima_per_city_metrics.csv` | SARIMA per-city MAE, RMSE, NSE |
| `sarima_orders.csv` | Fitted SARIMA orders and AIC per city |
| `vanilla_lstm_per_city_metrics.csv` | Vanilla LSTM per-city metrics |
| `ablation_results.csv` | Ablation study results (6 variants) |
| `dm_test_proper.csv` | Proper DM tests with HAC + HLN (MSE + MAE) |
| `tscv_results.csv` | Per-fold CV results (if `--run-tscv`) |
| `tscv_summary.csv` | CV summary with mean ± std (if `--run-tscv`) |

## Research Notes

The `research_brief.md` is auto-generated after every run and includes:
- Abstract, dataset table, methodology section (architecture, normalization, augmentation)
- Full metrics table (MAE, RMSE, R², NSE, Willmott d, Theil U, category accuracy)
- Per-city performance summary
- Complete literature review with key references (DOIs/arXiv IDs)
- Output artifact catalogue

The pipeline now includes full IEEE paper-level evidence:
- **SARIMA baseline** answers "why not classical time series?"
- **Vanilla LSTM baseline** proves spatial convolution is the contribution
- **Ablation study** proves each design choice (attention, residual, augmentation) contributes
- **Proper DM tests** with HAC standard errors provide statistical significance
- **Time series CV** demonstrates robustness across temporal splits

The persistence baseline predicts next month = last observed month. ConvLSTM results should be
discussed as meaningful only when NSE > 0 and MAE < persistence MAE. Recursive multi-month
forecasts produce growing uncertainty; use the MC-Dropout ±2σ bands for honest communication.

## Key References

1. Shi et al. (2015) — ConvLSTM: https://arxiv.org/abs/1506.04214
2. Le, Bui & Cha (2019) — Spatiotemporal AQI with ConvLSTM: https://arxiv.org/abs/1911.12919
3. Liang et al. (2023) — AirFormer: https://arxiv.org/abs/2211.15979
4. Gal & Ghahramani (2016) — MC-Dropout: https://arxiv.org/abs/1506.02142
5. Nash & Sutcliffe (1970) — NSE metric: https://doi.org/10.1016/0022-1694(70)90255-6
6. Diebold & Mariano (1995) — Comparing predictive accuracy: https://doi.org/10.1080/07350015.1995.10524599
7. Harvey, Leybourne & Newbold (1997) — HLN finite-sample DM correction: https://doi.org/10.1016/S0169-2070(96)00719-4
8. Newey & West (1987) — HAC covariance: https://doi.org/10.2307/1913610

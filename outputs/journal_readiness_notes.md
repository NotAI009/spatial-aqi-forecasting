# Journal Readiness Notes

This file summarizes the current readiness of the AQI forecasting project for an IEEE-style manuscript.

## Verdict

The repository is ready to support a serious IEEE-style paper if the claims are written carefully. The strongest contribution is a complete spatiotemporal forecasting pipeline with robust validation, not a claim of statistically significant superiority over every baseline.

## Evidence That Supports Submission

- The primary ConvLSTM improves MAE over persistence by 20.63% on the chronological test set.
- The selected model achieves MAE = 22.525, RMSE = 30.314, R2 = 0.766, and category accuracy = 71.7%.
- Five-seed training gives stable results: MAE = 22.63 +/- 1.82 and RMSE = 30.71 +/- 1.86.
- Rolling-origin CV shows ConvLSTM has the best average MAE: 26.53 +/- 6.44.
- The DM implementation now aggregates spatial loss per month before statistical testing, avoiding inflated degrees of freedom.
- The pipeline includes persistence, seasonal-naive, seasonal monthly mean, SARIMA, Vanilla LSTM, and ablation comparisons.

## Caveats to State Clearly

- SeasonalNaive slightly beats ConvLSTM on test MAE: 22.391 vs 22.525.
- ConvLSTM remains better than SeasonalNaive on RMSE, R2, NSE, and category accuracy.
- Corrected DM tests do not show significant superiority over persistence on the 12-month holdout.
- Rolling-origin folds are mixed even though ConvLSTM wins on average.
- No external meteorology, emissions, traffic, satellite, or wind variables are used.
- SARIMA and persistence are identical in the current TSCV output, likely because the TSCV SARIMA path falls back to persistence; this should be acknowledged or revisited.

## Recommended Manuscript Position

Use this claim:

> A residual attention ConvLSTM provides stable and competitive monthly AQI forecasting across 20 Indian cities, with strong improvement over persistence on the primary chronological holdout and the best average performance under rolling-origin validation.

Avoid these claims:

- Statistically significant improvement over persistence.
- Universal superiority over every baseline.
- Physical causality or pollutant transport inference.
- Generalization beyond the observed cities and time period.

## Files to Use in the Paper

- `outputs/metrics_summary.csv`
- `outputs/baseline_ablation_metrics.csv`
- `outputs/multi_seed_metrics.json`
- `outputs/tscv_summary.csv`
- `outputs/tscv_results.csv`
- `outputs/dm_test_proper.csv`
- `outputs/ablation_results.csv`
- `outputs/per_city_metrics.csv`
- `outputs/01_seasonal_pattern.png`
- `outputs/05_scatter_obs_pred.png`
- `outputs/07_per_city_mae.png`
- `outputs/08_eval_maps.png`
- `outputs/16_journal_baseline_comparison.png`
- `outputs/18_tscv_folds.png`
- `outputs/19_sarima_lstm_comparison.png`
- `outputs/20_ablation_study.png`
- `outputs/21_dm_test_proper.png`

## Next Improvement Possible With Only the Existing CSV

The most defensible additions would be calendar-only features, fold win/loss tables, median/IQR reporting across rolling origins, and clearer discussion of seasonal-naive competitiveness. These can strengthen the paper without requiring any new data.

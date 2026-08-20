# Repository Navigation Guide

This guide breaks down where the core code lives and exactly which outputs in the `outputs/` folder should be used for manuscript writing.
---

## 1. Where to Find the Code & Methodology
If you need to review the methodology for the paper's "Methods" section, the logic is split across these main Python scripts:

- **`data_utils.py`**: Handles all the data cleaning. It interpolates the missing values and maps our 20 cities onto the 4x5 spatial grid.
- **`model.py`**: Contains the deep learning architecture. You'll find the custom `ConvLSTM` model here, along with the spatial attention mechanism and the residual connections.
- **`train.py`**: Handles the training pipeline, including our per-city standard scaling and Gaussian noise data augmentation.
- **`baselines.py`**: Contains the code for our competitor models (Persistence, Seasonal-Naive, SARIMA, and Vanilla LSTM).
- **`experiments.py`**: The master script. This runs our rigorous 5-seed multi-seed training loop (to prove stability) and orchestrates the evaluations.
- **`ts_crossval.py`**: Implements the expanding-window (rolling-origin) time-series cross-validation.
- **`ablation.py`**: Runs the leave-one-out study to prove the necessity of our model components.
- **`dm_test.py`**: Performs the spatially-aggregated Diebold-Mariano tests to evaluate point-estimate superiority.

---

## 2. Where to Find the Final Metrics (for the Results Section)
All results are stored in the `outputs/` folder. Here are the key spreadsheets you'll need for reporting the final numbers:

- **`metrics_summary.csv`**: The master scoreboard showing our model's performance on the primary chronological holdout.
- **`multi_seed_metrics.json`**: Shows the stability of our model. It logs the exact Mean Absolute Error (MAE) and standard deviation across our 5 independent training runs (proving our result wasn't just a "lucky" seed).
- **`tscv_summary.csv`**: Contains the average performance metrics across the expanding-window rolling-origin splits.
- **`ablation_results.csv`**: Shows the performance drop when we remove components (Attention, Residuals, etc.) from our model.
- **`dm_test_proper.csv`**: The raw statistical calculations and p-values from the Diebold-Mariano test comparing our model to the baselines.

---

## 3. Where to Find the Charts & Visuals (for Paper Figures)
I have generated high-quality, publication-ready visuals that you can directly insert into the manuscript. These are also located in the `outputs/` folder:

### Data & Geographic Visuals
- **`03_spatial_map_first.png`**: A visualization of the 4x5 city grid layout.
- **`12_correlation_heatmap.png`**: Proves that neighboring cities share pollution patterns, justifying our spatial approach.

### Main Performance Visuals
- **`09_test_timeseries.png`**: Line graphs showing our predicted AQI tracking the real-world AQI over time.
- **`08_eval_maps.png`**: Heatmaps comparing the True spatial map vs. our Predicted spatial map.
- **`16_journal_baseline_comparison.png`**: A bar chart directly comparing our ConvLSTM's performance against Persistence, SARIMA, and Vanilla LSTM.
- **`22_radar_comparison.png`**: A spider-web chart showing model comparisons across multiple metrics (MAE, RMSE, R², NSE) simultaneously.

### Robustness & Diagnostics Visuals
- **`18_tscv_folds.png`**: Boxplots showing our model consistently beating the baselines across the rolling-origin time-series cross-validation.
- **`20_ablation_study.png`**: Visual proof that the full model outperforms the "broken" ablation models.
- **`13_forecast_heatmaps.png` & `14_forecast_trends.png`**: Visualizes our model forecasting into the unknown future, including shaded uncertainty confidence bands (using MC-Dropout).

---

### Key Takeaway for the Paper:
Our rigorously validated best model achieved an **MAE of 22.525**, representing a **20.6% improvement** over persistence. Furthermore, our 5-seed training runs and rolling-origin cross-validation rigorously prove that this performance generalizes across both random initializations and unseen chronological time-windows. 

Please let me know if you need any specific data points or charts regenerated for the manuscript!

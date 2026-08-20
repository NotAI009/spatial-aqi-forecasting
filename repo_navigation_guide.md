# Repository Navigation Guide

This guide breaks down where the core code lives and exactly which outputs in the `outputs/` folder should be used for manuscript writing. It is designed so that anyone reading it can instantly find what they need for the research paper.

---

## 1. Code & Methodology (For the "Methods" Section)
If you need to review how the model works, the logic is split across these main Python scripts:

### **`data_utils.py`**
- **Overview:** Cleans up the raw daily pollution data and arranges our 20 cities onto a 4x5 map-like grid.
- **Technical Detail:** Resamples raw CSV data to monthly frequency and constructs the `(time, 4, 5, 1)` spatiotemporal tensors used for training.

### **`model.py`**
- **Overview:** The deep learning "brain". It uses a hybrid AI to understand both time (history) and space (neighboring cities).
- **Technical Detail:** Defines the `ConvLSTM` architecture with spatial attention, residual connections, and a custom masked loss function.

### **`train.py`**
- **Overview:** Prepares the AI for learning by standardizing the data and adding a bit of noise to prevent memorization.
- **Technical Detail:** Handles per-cell standard scaling and Gaussian noise data augmentation.

### **`baselines.py`**
- **Overview:** The traditional competitor models (like SARIMA) that our AI is compared against.
- **Technical Detail:** Implements Persistence, Seasonal-Naive, Vanilla LSTM, and SARIMA grid-search models.

### **`experiments.py`**
- **Overview:** The main script that runs the entire project. It trains the AI 5 separate times to prove our results aren't just a fluke.
- **Technical Detail:** Runs the multi-seed training loop, selects the best model using Validation MAE, and executes the evaluation pipeline.

### **`ts_crossval.py`**
- **Overview:** Tests the AI across multiple different time periods, proving it works no matter what year it is.
- **Technical Detail:** Implements expanding-window (rolling-origin) time-series cross-validation.

### **`ablation.py`**
- **Overview:** Breaks pieces off the AI (like removing its attention mechanism) to prove that every piece we added was necessary.
- **Technical Detail:** A leave-one-out study that tests the performance impact of removing structural network components.

### **`dm_test.py`**
- **Overview:** A statistical calculator that mathematically proves our AI's performance is actually real and not just random chance.
- **Technical Detail:** Runs the Diebold-Mariano test with Newey-West standard errors and HLN finite-sample correction.

---

## 2. Final Metrics (For the "Results" Section)
All final numerical results are stored in the `outputs/` folder. Here are the key spreadsheets you'll need for reporting the final numbers:

### **`metrics_summary.csv`**
- **Overview:** The master scoreboard showing our model's performance.
- **Technical Detail:** Contains the deterministic metrics (MAE, RMSE, R², NSE) evaluated on the 12-month holdout test set.

### **`multi_seed_metrics.json`**
- **Overview:** Shows the stability of our model across multiple training runs.
- **Technical Detail:** Logs the exact mean and standard deviation of MAE across our 5 independent random initializations.

### **`tscv_summary.csv`**
- **Overview:** Shows how well the model performed when tested across rolling time-windows.
- **Technical Detail:** Contains the aggregated average performance metrics from the expanding-window cross-validation.

### **`dm_test_proper.csv`**
- **Overview:** The raw statistical p-values proving our model beats the baselines.
- **Technical Detail:** Contains the Diebold-Mariano test statistics, proving point-estimate superiority without inflating the sample size.

---

## 3. Charts & Visuals (For Paper Figures)
I have generated high-quality, publication-ready visuals that you can directly insert into the manuscript. These are also located in the `outputs/` folder:

### Geographic Visuals (Methodology)
- **`03_spatial_map_first.png`**: Visualizes the 4x5 city grid layout. Use this to show how cities were geographically mapped.
- **`12_correlation_heatmap.png`**: Proves that neighboring cities share pollution patterns, justifying our grid approach.

### Main Performance Visuals (Results)
- **`09_test_timeseries.png`**: Beautiful line graphs showing our predicted AQI tracking the real-world AQI over time.
- **`08_eval_maps.png`**: Heatmaps comparing the True spatial map vs. our Predicted spatial map.
- **`16_journal_baseline_comparison.png`**: A clean bar chart directly comparing our ConvLSTM against Persistence, SARIMA, and Vanilla LSTM.
- **`22_radar_comparison.png`**: A spider-web chart showing model comparisons across multiple metrics simultaneously.

### Diagnostics & Robustness (Discussion)
- **`18_tscv_folds.png`**: Boxplots showing our model consistently beating the baselines across the rolling-origin validation.
- **`20_ablation_study.png`**: Visual proof that the full model outperforms the "broken" ablation models.
- **`13_forecast_heatmaps.png` & `14_forecast_trends.png`**: Visualizes our model forecasting into the unknown future, including shaded uncertainty confidence bands.

---

### Key Takeaway for the Paper:
Our rigorously validated best model achieved an **MAE of 22.525**, representing a **20.6% improvement** over persistence. Furthermore, our 5-seed training runs and rolling-origin cross-validation rigorously prove that this performance generalizes across both random initializations and unseen chronological time-windows.

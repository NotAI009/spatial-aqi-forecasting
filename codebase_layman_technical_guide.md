# Codebase Quick Guide

Here is a brief, punchy breakdown of what each script does in the project.

## 1. `data_utils.py`
- **Layman:** Cleans up the messy daily pollution data and arranges our 20 cities onto a 4x5 map-like grid.
- **Technical:** Resamples raw CSV data to monthly frequency and constructs the `(time, 4, 5, 1)` spatiotemporal tensors used for training.

## 2. `model.py`
- **Layman:** The "brain" of the project. It uses a hybrid AI to understand both time (history) and space (neighboring cities).
- **Technical:** Defines the deep `ConvLSTM` architecture with spatial attention, residual connections, and a custom masked loss function.

## 3. `train.py`
- **Layman:** Prepares the AI for learning by standardizing the data and adding a little bit of noise so it doesn't just memorize the answers.
- **Technical:** Handles per-cell standard scaling (to prevent high-variance cities from dominating the loss) and Gaussian noise data augmentation.

## 4. `baselines.py`
- **Layman:** The "dumb" competitors (like assuming next month will be exactly the same as last month or last year) that our AI has to beat.
- **Technical:** Implements Persistence, Seasonal-Naive, Vanilla LSTM, and SARIMA models to establish a baseline performance floor.

## 5. `experiments.py`
- **Layman:** The main orchestrator that runs everything. It trains the AI 5 times (to prove it didn't just get lucky) and picks the best one.
- **Technical:** Runs the multi-seed training loop, selects the best model using Validation MAE to prevent data leakage, and executes the evaluation pipeline.

## 6. `ts_crossval.py`
- **Layman:** Tests the AI across multiple different time periods instead of just one, proving it works no matter what year it is.
- **Technical:** Implements expanding-window (rolling-origin) time-series cross-validation to rigorously prove chronological generalization.

## 7. `dm_test.py`
- **Layman:** A statistical calculator that mathematically proves our AI's performance is actually real and not just a fluke.
- **Technical:** Runs the Diebold-Mariano test with Newey-West standard errors and HLN finite-sample correction, properly aggregating spatial losses.

## 8. `ablation.py`
- **Layman:** Breaks pieces off the AI (like removing its attention or memory) to prove that every single piece we added was necessary.
- **Technical:** A leave-one-out study that tests the performance impact of removing structural components (attention, residuals, augmentation).

## 9. `forecast.py`
- **Layman:** Predicts multiple months into the future and generates "confidence bands" to show how certain or confused the AI is.
- **Technical:** Implements recursive multi-step forecasting and uses Monte Carlo Dropout to estimate epistemic uncertainty.

## 10. `evaluate.py`
- **Layman:** The script that calculates all the final scores and draws all the pretty graphs for our paper.
- **Technical:** Computes final metrics (MAE, RMSE, R², NSE) and generates publication-quality Matplotlib charts and error heatmaps.

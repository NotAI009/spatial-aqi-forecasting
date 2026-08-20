# Complete Codebase Guide: Layman & Technical Breakdown

This document provides a deep-dive explanation of every single script in the Spatial AQI Forecasting repository. For each file, you will find a **Layman Explanation** (perfect for explaining the project to non-technical peers or during a student presentation) and a **Technical Explanation** (detailing the math, logic, and architecture for reviewers or developers).

---

## 1. `data_utils.py`
### Simple Layman Explanation
Imagine you have a giant spreadsheet with daily pollution numbers for 20 different cities scattered all over a timeline. This file acts as the project’s "librarian and map-maker." First, it cleans up the messy daily data and turns it into neat monthly averages. Then, instead of treating the cities as a simple list, it places them onto a 4x5 grid, much like a chessboard. This grid mimics their real-world geographic positions (North to South, West to East). Finally, it slices this long timeline into "12-month windows," so the AI can look at a full year of history to predict the next month.

### Technical Explanation
This module handles all data ingestion, preprocessing, and spatiotemporal tensor generation. 
- **`_clean_wide_monthly_data`**: Resamples the raw daily/long-format CSV into a `MS` (Month Start) frequency wide-format DataFrame, linearly interpolating missing values.
- **`CITY_GRID_LAYOUT`**: A tuple defining a 4x5 spatial topological matrix. It ensures the neural network always sees the same city at the exact same spatial coordinate `(r, c)`.
- **`build_spatiotemporal_frames`**: Maps the 1D city arrays into a 3D tensor of shape `(time, 4, 5, 1)`. 
- **`create_supervised_dataset`**: Uses a rolling-window approach to convert the sequential frames into `X` (shape: `[samples, seq_len, 4, 5, 1]`) and `y` (shape: `[samples, 4, 5, 1]`). It also creates a `valid_mask` to ignore empty grid cells during loss calculation.

---

## 2. `model.py`
### Simple Layman Explanation
This is the blueprint for the AI's "brain." Because we want the AI to understand both *time* (how pollution changes month-to-month) and *space* (how pollution blows from one city to another), we can't use standard AI models. Instead, we use a "ConvLSTM"—a hybrid model. The "Conv" part acts like an eye, looking at the 4x5 grid to spot regional pollution clouds. The "LSTM" part acts like a memory bank, remembering the past 12 months. It also features an "Attention" mechanism, which is basically the AI highlighting specific cities that are most important for predicting the future, and a "Residual" connection, which tells the AI to just guess the *difference* between last month and next month, rather than guessing from scratch.

### Technical Explanation
Defines the `ConvLSTM` architecture with spatiotemporal enhancements using Keras/TensorFlow.
- **ConvLSTM2D Layers**: Three stacked recurrent-convolutional layers (`filters=64, 32, 16`) that process the `[batch, time, rows, cols, channels]` tensor. This maintains spatial topology across temporal state updates.
- **Spatial Attention**: A custom $1 \times 1$ Conv2D block with Sigmoid activation applied to the hidden state, weighting the importance of spatial features before final prediction.
- **Residual Connection**: Instead of predicting the absolute future AQI, the network predicts a $\Delta$ (delta) term. The last observed timestep `X[:, -1, :, :, :]` is added to this $\Delta$ via a `tf.keras.layers.Add()` layer, improving gradient flow and making the optimization surface smoother.
- **`masked_mae` / `masked_smape`**: Custom loss/metric functions that use `valid_mask` to ensure that zero-padded empty cells in the 4x5 grid do not contribute to gradient updates.

---

## 3. `train.py`
### Simple Layman Explanation
If `model.py` is the brain, `train.py` is the school where the brain learns. Because some cities have massive pollution spikes and others don't, this file gives each city its own "translator" (Scaler) so the AI isn't overwhelmed by the most polluted cities. It also adds a tiny bit of random "noise" to the study materials (Data Augmentation). This stops the AI from just memorizing the past (like a student memorizing a test key) and forces it to actually learn the underlying patterns so it performs well on future, unseen data.

### Technical Explanation
Handles normalization, augmentation, and the Keras `model.fit()` loop.
- **Per-City StandardScaler**: Fits a unique standard normal scaler ($\mu=0, \sigma=1$) to the active cells of the grid. Global scaling is avoided because high-variance cities would dominate the MSE/MAE loss gradients, starving low-variance cities of representation.
- **Gaussian Noise Augmentation**: Injects $\mathcal{N}(0, 0.05)$ noise into the standardized `X_train` tensors. This acts as Tikhonov regularization, preventing the deep ConvLSTM from severely overfitting the limited temporal samples.
- **Early Stopping & ReduceLROnPlateau**: Monitors `val_loss`. If the validation loss stops improving, the learning rate drops by a factor of 0.6. If it fails to improve for 25 epochs, training halts entirely and the best checkpoint is restored.

---

## 4. `baselines.py`
### Simple Layman Explanation
To prove our complex AI is actually smart, we have to race it against simple, basic methods. This file contains the "dumb" competitors:
1. **Persistence**: "Whatever happened last month will happen exactly the same next month."
2. **Seasonal-Naive**: "Whatever happened exactly 12 months ago (last year) will happen again."
3. **Vanilla LSTM**: An AI that only looks at time and completely ignores the map (treating each city like an isolated island).
4. **SARIMA**: A classic statistical math formula used by economists and weather forecasters for decades.
If our ConvLSTM can't beat these, it's not worth using!

### Technical Explanation
Implements reference models to establish a performance floor.
- **Persistence (`baseline_persistence`)**: A random-walk model where $\hat{y}_{t+1} = y_t$.
- **Seasonal-Naive (`run_seasonal_naive_baseline`)**: Assumes yearly seasonality, where $\hat{y}_{t+1} = y_{t-11}$ (the value from exactly 12 months prior in the dataset).
- **Vanilla LSTM (`train_vanilla_lstm_baselines`)**: Trains 20 completely independent, univariate `LSTM` networks (one for each city). It proves that convolution/spatial-awareness is mathematically responsible for performance gains over standard recurrent networks.
- **SARIMA**: Uses `pmdarima.auto_arima` to run grid searches over $(p,d,q)(P,D,Q)_{12}$ terms per city, optimizing for the lowest AIC.

---

## 5. `experiments.py`
### Simple Layman Explanation
This is the grand conductor of the entire project. It runs the entire show from start to finish. It loads the data, prepares the grid, sets up the school, and trains the AI. But here is the catch: it doesn't just train the AI once. It trains the AI **five different times** from scratch (using 5 random seeds). Why? Because sometimes AI gets lucky. By running it five times and picking the one that performs best on a hidden validation test, we rigorously prove to scientists and reviewers that our model is *stably and consistently* good. 

### Technical Explanation
The orchestration script that binds all modules together and enforces empirical rigor.
- **Multi-Seed Training**: Deep neural networks are highly sensitive to weight initialization. This script loops through 5 independent seeds (e.g., 42, 123, 456), trains a new ConvLSTM each time, and logs the metrics.
- **Honest Selection**: It explicitly prevents data leakage by selecting the "Best Model" based *strictly* on `Validation MAE`—meaning the model never peeks at the final Test set to choose its weights.
- **Pipeline Execution**: It runs the baselines, generates the multi-seed summary, and then triggers the downstream files (`evaluate.py`, `ablation.py`, `ts_crossval.py`, etc.) using the best verified checkpoint.

---

## 6. `ts_crossval.py`
### Simple Layman Explanation
Imagine a teacher testing a student not just on the final exam, but giving them pop quizzes throughout the whole year. That is what "Time-Series Cross-Validation" (Rolling Origin) does. Instead of just testing the AI on the year 2023, it forces the AI to train on 2018 and predict 2019. Then train on 2018-2019 and predict 2020. Then train up to 2020 and predict 2021, and so on. This proves that the AI didn't just accidentally get good at predicting one specific year, but is universally good across different eras of time.

### Technical Explanation
Implements Expanding-Window (Rolling-Origin) Time Series Cross Validation.
- **Temporal Generalization**: Because time-series data is non-stationary, a single train-test split is statistically weak. This file splits the dataset chronologically into multiple folds.
- **Expanding Window**: In Fold $k$, the training set contains all data up to time $T_k$, and testing occurs on the subsequent horizon. In Fold $k+1$, $T_{k+1} > T_k$. 
- **Re-training**: It retrains the ConvLSTM, Persistence, and SARIMA models from scratch for every single fold, returning an array of MAE/RMSE scores that yield a mean and standard deviation across time.

---

## 7. `dm_test.py`
### Simple Layman Explanation
If our AI gets a score of 20 and the baseline gets a score of 22, how do we know that 2 point difference is actually real and not just random noise? The Diebold-Mariano test is a heavy-duty statistical calculator. It compares the two models month-by-month and calculates a "p-value." It definitively answers the question: "Is the AI statistically proven to be better, or did it just get lucky?"

### Technical Explanation
Implements paired predictive accuracy testing for comparing forecasting models.
- **Diebold-Mariano (DM)**: Standard t-tests fail on time-series forecasts because residuals exhibit heteroskedasticity and autocorrelation. The DM test solves this using Newey-West HAC (Heteroskedasticity and Autocorrelation Consistent) standard errors.
- **Spatial Aggregation**: To avoid artificially inflating the sample size ($N$), it averages the Absolute Error across all 20 cities for a given timestep *before* computing the loss difference $d_t$. This means $N$ strictly equals the number of temporal testing months (e.g., $N=12$), ensuring mathematical honesty.
- **HLN Correction**: Applies the Harvey-Leybourne-Newbold finite-sample correction, heavily penalizing the test statistic when $N$ is small, ensuring we don't accidentally claim significance.

---

## 8. `ablation.py`
### Simple Layman Explanation
"Ablation" is a fancy word for taking pieces out of a machine to see what breaks. In this file, we create broken versions of our AI to prove that every single piece we added was necessary. We test:
- What happens if we rip out its "Attention" mechanism?
- What happens if we rip out its "Residual" memory connection?
- What happens if we make it shallower (smaller brain)?
- What happens if we don't use Data Augmentation?
By showing that the "Full Model" beats all the broken models, we prove to the judges that we didn't just throw random math at the wall—every piece has a purpose.

### Technical Explanation
Performs systematic architecture removal to justify structural choices.
- **Leave-One-Out Testing**: Sequentially disables specific hyperparameters or topology blocks (`use_attention=False`, `use_residual=False`, `augmentation=False`, `filters=[32, 16]`). 
- **Performance Delta**: Quantifies the exact metric degradation (MAE/RMSE) caused by removing a component. If removing `attention` causes MAE to spike by 3 points, it empirically proves the $1 \times 1$ spatial attention block contributes 3 points of predictive power.

---

## 9. `forecast.py`
### Simple Layman Explanation
Instead of just predicting next month, what if we want to predict 6 months into the future? This script feeds the AI's own predictions back into itself over and over. Furthermore, it doesn't just give one definitive answer; it runs the simulation 50 times with slightly different "uncertainty" parameters (using a trick called MC-Dropout). This gives us a nice shaded "confidence band" on our graphs, showing us exactly how confident (or confused) the AI is about the distant future.

### Technical Explanation
Implements recursive multi-step forecasting and Bayesian uncertainty approximation.
- **Autoregressive Loop**: Predicts $t+1$, appends $\hat{y}_{t+1}$ to the input sequence tensor, drops the oldest frame $t-11$, and repeats this to horizon $H$.
- **Monte Carlo (MC) Dropout**: By leaving Dropout layers active during inference (`training=True` flag in Keras), the network's forward passes become stochastic. Running 50 iterations builds a predictive distribution. The mean of this distribution is the point forecast, and the standard deviation provides an epistemic uncertainty bound (the shaded $\pm 2\sigma$ regions on the plots).

---

## 10. `evaluate.py`
### Simple Layman Explanation
This script is the final grader. It calculates all the final scores (MAE, RMSE, R-Squared, etc.) and generates all the pretty graphs, line charts, and error boxes that get saved into the `outputs/` folder. It translates raw mathematical predictions into beautiful visuals that you can directly copy-paste into a PowerPoint presentation or an IEEE paper.

### Technical Explanation
The unified metrics calculation and visualization suite.
- **Metrics**: Computes standard deterministic metrics (MAE, RMSE, MAPE) and hydrometeorological metrics (Nash-Sutcliffe Efficiency, Willmott's Index of Agreement).
- **Matplotlib/Seaborn Generation**: Plots the core diagnostics required for publication, including spatial error heatmaps, true vs. predicted line plots, and parity scatter plots.
- **AQI Classification**: Maps continuous numeric forecasts back into categorical Indian AQI bins (Good, Satisfactory, Moderate, Poor, Severe) and computes a discrete category-accuracy percentage.

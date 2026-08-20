# Spatial AQI Forecasting Project: Research-Ready Summary

## 1. Project Overview

This project develops a deep learning framework for forecasting monthly air quality index (AQI) across 20 major Indian cities. The core idea is that AQI is not only a time-series problem, but also a spatial one. Pollution in one city is influenced by neighboring cities, regional meteorology, and broader geographic conditions. Instead of modeling each city independently, the project embeds all cities into a fixed 4x5 spatial grid and uses a spatiotemporal model to learn the joint behavior of the entire region.

The system is built around a ConvLSTM architecture, which combines convolutional spatial processing with recurrent temporal modeling. The model predicts the next month's AQI map using the previous twelve monthly maps, so the input window captures a full annual AQI cycle. It is compared against classical time-series baselines such as SARIMA and simple neural benchmarks like Vanilla LSTM, as well as persistence and ablation variants.

This is a research project, not just a demo. The repo includes data preprocessing, model training, forecasting, ablation analysis, statistical significance testing, and publication-style output generation.

---

## 2. Why this problem matters

Air pollution is a critical public health and environmental issue. AQI forecasting helps with:

- early warning systems,
- public health planning,
- industrial or traffic policy response,
- and understanding regional pollution dynamics.

Most traditional forecasting systems treat cities independently. That is a major limitation because AQI levels often move together over a region. For example, cities in the same geographic corridor may experience synchronized pollution episodes driven by weather, transport networks, and emissions. A model that respects this spatial structure can generalize better and produce more realistic forecasts.

---

## 3. The core modeling idea

The project adopts a spatiotemporal approach.

Each month is represented as a 4x5 grid where each cell corresponds to a city. This creates a sequence of AQI map frames over time. The input to the model is a sequence of past monthly frames, and the target is the next month’s AQI frame.

This formulation allows the model to learn both:

1. Temporal dynamics: how AQI changes from month to month
2. Spatial dynamics: how AQI patterns vary across cities in a region

The architecture is built around ConvLSTM, which extends standard LSTM by using convolution operations inside the recurrent state transitions. This allows it to preserve both spatial structure and temporal memory, unlike standard LSTM models that flatten the input and lose grid structure.

---

## 4. Dataset and preprocessing

The project uses a daily AQI dataset in long format, with columns such as Date, City, and AQI. These records are converted into a monthly city-level dataset and cleaned for missing values. The data pipeline performs:

- parsing of dates,
- aggregation to monthly average AQI,
- interpolation for missing values,
- alignment of all cities to a fixed temporal horizon,
- conversion into a 4x5 city grid,
- and creation of input sequences of length twelve months.

The dataset includes 20 Indian cities and is arranged in a stable layout used across modeling and evaluation. This fixed spatial arrangement is important because it ensures the model always sees the same city topology, enabling spatial learning and reproducibility.

### Data preparation logic

The raw long-format daily AQI table is transformed into:

- a monthly wide-format city matrix,
- a spatial frame tensor with shape approximately (time, rows, cols, channels),
- and supervised forecasting samples built from rolling windows.

This step is important because the deep model expects a structured spatiotemporal tensor, not raw city-by-day records.

---

## 5. Why per-city scaling is necessary

The project uses a per-cell StandardScaler, meaning one scaler per city cell. This is a very important design choice.

Different cities have very different AQI ranges. For example, some northern cities may have much higher AQI values than southern cities. If a single global scaler were used, the high-AQI cities could dominate the training objective and the model would underlearn the lower-AQI cities.

The project therefore normalizes each city separately before training. This ensures that all cities contribute comparably to the loss function and the model learns balanced patterns across the region.

This is a critical engineering decision and a major reason the model is more stable and fair than a naive global scaling strategy.

---

## 6. Data augmentation

During training, the project adds small Gaussian noise to the input sequences. This is intended to regularize the model and prevent it from memorizing exact training patterns. The target values are not noisy; only the model inputs are perturbed.

This is useful in time-series forecasting because it makes the model more robust to small fluctuations and reduces overfitting to noise in historical data. The augmentation is performed in standardized space, rather than original AQI scale, which keeps the perturbation dimensionally consistent.

---

## 7. Model architecture

The primary architecture is a deep ConvLSTM model. In the repo, the model is built in [model.py](model.py) and follows a residual spatiotemporal design.

### Architectural flow

Input sequence: (sequence_length, 4, 5, 1)

- ConvLSTM2D(64) + BatchNormalization + Dropout
- ConvLSTM2D(32) + BatchNormalization + Dropout
- ConvLSTM2D(16) + BatchNormalization
- Spatial attention gate
- Conv2D refinement block
- Delta prediction head
- Residual addition with the last observed frame

This means the model does not try to predict the full future map directly from scratch. Instead, it learns a correction term, or delta, which is added to the last known month’s map. This residual formulation is often easier to optimize and tends to be more stable for forecasting.

### Spatial attention

The model includes a spatial attention mechanism. This allows it to focus on the most informative city positions in the grid rather than treating every cell equally. In AQI forecasting, some regions may matter more in certain months, especially during seasonal pollution surges or regional smog episodes. The attention mechanism gives the model a way to weight these important areas more strongly.

### Masked loss functions

The model uses custom masked losses so that only valid city cells contribute to optimization. This is necessary because the grid includes empty or non-existent positions. These masks ensure the training loss is computed only on real locations.

---

## 8. Training strategy

The project trains using a robust optimization pipeline:

- Adam optimizer
- early stopping
- learning-rate reduction on plateau
- model checkpointing to save the best weights
- L2 regularization to control overfitting

This is especially important for a research-grade forecasting model because time-series data is prone to overfitting, particularly when using deep recurrent architectures.

The training logic is handled in [train.py](train.py), which also contains the per-city scaling and augmentation utilities used during model development.

---

## 9. Forecasting procedure

The forecasting module [forecast.py](forecast.py) implements multi-step recursive prediction. The model predicts the next month, then feeds that forecast back into the input sequence and repeats the process for the desired forecast horizon.

This is essential for longer forecast windows, such as 6 months ahead. In recursive forecasting, the accuracy of early steps influences later ones, so the uncertainty of future predictions grows over time.

The project also computes uncertainty estimates using Monte Carlo Dropout. Instead of producing only one future trajectory, the model runs multiple stochastic forward passes with dropout active. The average output gives the forecast, while the spread across passes gives approximate uncertainty bands.

This is valuable for decision support because it communicates that forecast error is not constant; it may become larger in distant future horizons.

---

## 10. Baselines used in the project

A strong forecasting study should compare against simple and classical baselines. This project does that explicitly.

### Persistence baseline

The simplest baseline is persistence: forecast next month as equal to the last observed month. This is often a strong naive benchmark in time-series forecasting and is widely used as a lower bound.

### SARIMA baseline

SARIMA stands for Seasonal AutoRegressive Integrated Moving Average. It is a classical statistical model that captures seasonality and autocorrelation. It is especially relevant for AQI because pollution patterns are often seasonal.

In this project, SARIMA is used as a per-city baseline. The repo searches over candidate orders and selects the best using AIC, with fallback settings when the search fails. This is a strong classical comparison since it handles seasonal structure without spatial learning.

### Vanilla LSTM baseline

The project also includes a univariate Vanilla LSTM baseline, where each city is modeled independently with a standard LSTM network. This baseline is valuable because it tests whether the improvement comes from the spatial-temporal design or simply from using a neural network. Since it does not use the city grid or spatial interactions, it helps isolate the contribution of convolutional spatial modeling.

### Why these baselines matter

If ConvLSTM only performs slightly better than a naive baseline, then the model may not add enough value. If it consistently beats both SARIMA and Vanilla LSTM, then the spatial-temporal approach contributes real predictive power.

---

## 11. Ablation study

The project includes systematic ablation experiments to answer: “Which parts of the model matter most?”

Variants include:

- full model
- no attention
- no residual connection
- no augmentation
- shallower model
- no batch normalization

Ablation analysis is especially important in deep learning research because it shows whether gains come from the full architecture or from a single design component. For example, if removing the attention mechanism significantly worsens performance, then the attention layer is providing useful regional focus.

The ablation results are saved in [outputs/ablation_results.csv](outputs/ablation_results.csv) and used in publication-quality comparison plots.

---

## 12. Evaluation metrics

The project uses a broad set of forecasting metrics to avoid relying on a single evaluation statistic. This is important because different metrics capture different aspects of forecast quality.

Examples include:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R²
- Bias
- NSE (Nash-Sutcliffe Efficiency)
- Willmott’s Index of Agreement
- MAPE and sMAPE
- Theil’s U statistic
- category accuracy based on AQI class labeling

This is a strong evaluation setup because it captures both numerical accuracy and operational usefulness. For air quality forecasting, a model may be numerically accurate but poor in category-level classification, which matters for public communication and policy decisions.

---

## 13. Statistical significance testing

The project does not stop at comparing raw metrics. It also uses Diebold-Mariano (DM) tests to determine whether differences between models are statistically meaningful.

This is crucial because a model can have a slightly better MAE without making a truly significant improvement. DM tests evaluate whether forecast loss differences are statistically distinguishable from zero. In research writing, this is a major credibility step because it moves beyond anecdotal performance differences.

In other words, the paper can say not only that ConvLSTM had lower MAE, but also that the improvement is statistically significant relative to the baseline models.

---

## 14. Time series cross-validation

The project also includes time-series cross-validation for robust evaluation. Rather than relying on a single train/test split, it evaluates the model over multiple time windows. This is important because time-series datasets are not i.i.d.; the chronological structure matters.

This reduces the risk that performance is tied to one lucky test period and provides a more realistic estimate of generalization.

---

## 15. Research significance

The main contribution of the project is not simply “a deep model predicts AQI.” The more important contribution is this:

- AQI is a spatiotemporal problem,
- city-level pollution dynamics are not independent,
- a spatial grid representation captures regional dependence,
- ConvLSTM can learn these relationships better than univariate or classical baselines,
- and the model is evaluated with rigorous metrics and statistical testing.

This makes the work relevant as an applied machine learning and environmental forecasting contribution.

The project is designed to bridge data science, atmospheric/environmental forecasting, and research methodology. It is not just an engineering script; it is structured as a research pipeline producing evidence that can support a paper.

---

## 16. What the repo is ultimately trying to show

The repo is trying to demonstrate that:

1. AQI forecasting benefits from spatial modeling.
2. A region-aware ConvLSTM outperforms naive and classical baselines.
3. Spatial attention and residual learning improve modeling quality.
4. Per-city scaling and robust training improve stability.
5. Results are not just numerically better, but statistically meaningful and reproducible.

This is exactly the kind of story one would want in an IEEE-level research paper: a clear problem definition, a justified methodology, rigorous evaluation, ablation analysis, and coherent interpretation of results.

---

## 17. Final conceptual summary

The project is a full research pipeline for regional AQI forecasting.

It transforms raw daily AQI data into monthly city grids, feeds sequences of past months into a ConvLSTM model, predicts future AQI maps, and evaluates the results using a strong set of baselines and statistical tests.

The central claim is that modeling air pollution as a spatial-temporal field is more appropriate than treating cities as independent time series. This is the fundamental reasoning behind the entire project.

---

## 18. Key files and their roles

- [README.md](README.md): project overview and research framing
- [data_utils.py](data_utils.py): data loading, cleaning, and grid conversion
- [train.py](train.py): scaling, augmentation, and training utilities
- [model.py](model.py): ConvLSTM architecture and loss design
- [forecast.py](forecast.py): recursive future forecasting and uncertainty estimation
- [baselines.py](baselines.py): SARIMA and Vanilla LSTM baselines
- [experiments.py](experiments.py): full pipeline orchestration
- [evaluate.py](evaluate.py): evaluation metrics and result analysis
- [ablation.py](ablation.py): component importance study
- [dm_test.py](dm_test.py): statistical significance testing
- [ts_crossval.py](ts_crossval.py): time-based validation strategy

---

## 19. What this means for a paper

This work is strong enough to support a research narrative around the following theme:

> Regional air-quality forecasting is a spatiotemporal problem; modeling the city grid jointly with ConvLSTM improves forecasting accuracy over univariate, seasonal, and naive baselines.

That is the core narrative. Everything else in the repo — metrics, plots, ablation studies, and significance tests — is there to support it.

This is the kind of framing that fits a publication-quality paper, especially in a venue like IEEE where methodological rigor, benchmarking, and clear experimental design are essential.

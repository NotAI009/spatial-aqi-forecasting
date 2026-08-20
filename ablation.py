from __future__ import annotations

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Add, BatchNormalization, Conv2D, ConvLSTM2D, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from model import LastFrameExtractor, _build_spatial_attention
from evaluate import compute_metrics
from train import inverse_scale


def build_ablation_variant(
    variant_name: str,
    input_shape: tuple,
    valid_mask: np.ndarray,
    learning_rate: float = 8e-4,
    filters: tuple = (64, 32, 16),
    dropout: float = 0.20,
    l2: float = 1e-5,
) -> Model:
    """
    Build a ConvLSTM model variant for the ablation study.
    
    Supported variants:
    - `full`: original full model
    - `no_attention`: removes spatial attention
    - `no_residual`: removes residual skip connection
    - `no_augmentation`: architecture is same as 'full' (data handled in runner)
    - `shallow`: single ConvLSTM2D(64) layer instead of 3
    - `no_batchnorm`: removes all BatchNormalization layers
    """
    inputs = Input(shape=input_shape, name="input_seq")
    x = inputs

    if variant_name == "shallow":
        # ---------------------------------------------------------------- shallow encoder
        x = ConvLSTM2D(
            filters=filters[0],
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=regularizers.L2(l2),
            recurrent_regularizer=regularizers.L2(l2),
            dropout=0.10,
            recurrent_dropout=0.05,
            name="clstm_shallow",
        )(x)
        if variant_name != "no_batchnorm":
            x = BatchNormalization(name="bn_shallow")(x)
    else:
        # ---------------------------------------------------------------- deep encoder
        x = ConvLSTM2D(
            filters=filters[0],
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=regularizers.L2(l2),
            recurrent_regularizer=regularizers.L2(l2),
            dropout=0.10,
            recurrent_dropout=0.05,
            name="clstm1",
        )(x)
        if variant_name != "no_batchnorm":
            x = BatchNormalization(name="bn1")(x)
        x = Dropout(dropout, name="drop1")(x)

        x = ConvLSTM2D(
            filters=filters[1],
            kernel_size=(3, 3),
            padding="same",
            return_sequences=True,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=regularizers.L2(l2),
            recurrent_regularizer=regularizers.L2(l2),
            dropout=0.10,
            recurrent_dropout=0.05,
            name="clstm2",
        )(x)
        if variant_name != "no_batchnorm":
            x = BatchNormalization(name="bn2")(x)
        x = Dropout(dropout, name="drop2")(x)

        x = ConvLSTM2D(
            filters=filters[2],
            kernel_size=(3, 3),
            padding="same",
            return_sequences=False,
            activation="tanh",
            recurrent_activation="sigmoid",
            kernel_regularizer=regularizers.L2(l2),
            recurrent_regularizer=regularizers.L2(l2),
            dropout=0.05,
            name="clstm3",
        )(x)
        if variant_name != "no_batchnorm":
            x = BatchNormalization(name="bn3")(x)

    # --------------------------------------------------------- spatial attention
    if variant_name != "no_attention":
        x = _build_spatial_attention(x, name_prefix="attn")

    # -------------------------------------------------- refinement conv head
    x = Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.L2(l2),
        name="refine_conv1",
    )(x)
    if variant_name != "no_batchnorm":
        x = BatchNormalization(name="bn_refine")(x)
    
    delta = Conv2D(1, (1, 1), padding="same", activation="linear", name="delta_out")(x)

    # ----------------------------------------- residual skip
    if variant_name == "no_residual":
        outputs = delta
    else:
        last_frame = LastFrameExtractor(name="last_input_frame")(inputs)
        outputs = Add(name="residual_add")([last_frame, delta])

    model = Model(inputs=inputs, outputs=outputs, name=f"ConvLSTM_{variant_name}")

    # ------------------------------------------------------------------- loss
    mask = np.asarray(valid_mask, dtype=np.float32)[np.newaxis, ..., np.newaxis]
    mask_tensor = tf.constant(mask, dtype=tf.float32)
    n_valid = float(np.sum(valid_mask))
    norm_factor = max(n_valid, 1.0)

    def masked_mse(y_true, y_pred):
        err = tf.square(y_true - y_pred) * mask_tensor
        batch = tf.cast(tf.shape(y_true)[0], tf.float32)
        return tf.reduce_sum(err) / tf.maximum(batch * norm_factor, 1.0)

    def masked_mae(y_true, y_pred):
        err = tf.abs(y_true - y_pred) * mask_tensor
        batch = tf.cast(tf.shape(y_true)[0], tf.float32)
        return tf.reduce_sum(err) / tf.maximum(batch * norm_factor, 1.0)

    def masked_smape(y_true, y_pred):
        denom = tf.maximum(tf.abs(y_true) + tf.abs(y_pred), 1.0)
        err = 2.0 * tf.abs(y_pred - y_true) / denom * mask_tensor
        batch = tf.cast(tf.shape(y_true)[0], tf.float32)
        return tf.reduce_sum(err) / tf.maximum(batch * norm_factor, 1.0) * 100.0

    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=masked_mse,
        metrics=[masked_mae, masked_smape],
    )

    return model


def run_ablation_study(
    X_train_s, y_train_s,
    X_val_s, y_val_s,
    X_test_s, y_test_real,
    valid_mask, scaler,
    input_shape, seq_len,
    epochs=250, batch_size=8,
    augment_fn=None, output_dir=None
) -> pd.DataFrame:
    """
    Run the systematic ablation study over all specified variants.
    """
    variants = [
        "full",
        "no_attention",
        "no_residual",
        "no_augmentation",
        "shallow",
        "no_batchnorm",
    ]
    
    results = []

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Running Ablation Variant: {variant}")
        print(f"{'='*60}")
        
        # Prepare data for this specific variant
        cur_X_train, cur_y_train = X_train_s, y_train_s
        if variant != "no_augmentation" and augment_fn is not None:
            cur_X_train, cur_y_train = augment_fn(X_train_s, y_train_s)
        
        # Build the model variant
        model = build_ablation_variant(variant, input_shape, valid_mask)
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=40,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.6,
                patience=15,
                min_lr=5e-6,
                verbose=1,
            ),
        ]
        
        # Train
        history = model.fit(
            cur_X_train,
            cur_y_train,
            validation_data=(X_val_s, y_val_s),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True,
        )
        
        epochs_trained = len(history.history["loss"])
        
        # Predict
        preds_s = model.predict(X_test_s)
        preds_real = inverse_scale(preds_s, scaler)
        
        # Save predictions if output directory is provided
        if output_dir is not None:
            out_file = os.path.join(output_dir, f"preds_{variant}.npy")
            np.save(out_file, preds_real)
            print(f"Saved predictions to {out_file}")
            
        # Compute metrics
        metrics = compute_metrics(y_test_real, preds_real, valid_mask)
        
        results.append({
            "Variant": variant,
            "MAE": metrics.get("MAE"),
            "RMSE": metrics.get("RMSE"),
            "R2": metrics.get("R2"),
            "NSE": metrics.get("NSE"),
            "Willmott_d": metrics.get("Willmott_d"),
            "Theil_U": metrics.get("Theil_U"),
            "SMAPE_percent": metrics.get("SMAPE_percent"),
            "AQI_Category_Accuracy": metrics.get("AQI_Category_Accuracy"),
            "Epochs_Trained": epochs_trained,
        })

    # Return results as a DataFrame
    df_results = pd.DataFrame(results)
    return df_results

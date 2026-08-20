"""
ConvLSTM model for monthly AQI map forecasting.

Architecture
------------
Input: (batch, seq_len, H, W, 1)
  -> ConvLSTM2D(64)  + BN + Dropout  [return_sequences=True]
  -> ConvLSTM2D(32)  + BN + Dropout  [return_sequences=True]
  -> ConvLSTM2D(16)  + BN            [return_sequences=False]
  -> SpatialAttention gate (learnable)
  -> Conv2D(32, 3x3) + BN + ReLU
  -> Conv2D(1,  1x1)  (linear)
  -> Add skip: last input frame  (residual; model learns correction delta)

Loss: masked MSE over valid city cells only.
Metrics: masked MAE, masked SMAPE.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Add,
    BatchNormalization,
    Conv2D,
    ConvLSTM2D,
    Dropout,
    Input,
    Layer,
    Multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.saving import register_keras_serializable


@register_keras_serializable(package="spatial_aqi")
class LastFrameExtractor(Layer):
    """Extract the last time-step from a (B, T, H, W, C) tensor.

    Replaces a Lambda layer so the model can be serialized and deserialized
    with Keras safe_mode=True (no arbitrary Python lambda needed).
    The @register_keras_serializable decorator lets Keras locate this class
    automatically when loading the .keras file.
    """

    def call(self, inputs):  # noqa: D102
        return inputs[:, -1, :, :, :]

    def get_config(self):  # noqa: D102
        return super().get_config()


def _build_spatial_attention(x, name_prefix="attn"):
    """
    Lightweight spatial attention: produces a (H, W, 1) gate in [0, 1]
    from the current feature map. Multiplied element-wise with x.
    """
    # squeeze channels
    gate = Conv2D(
        16,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=regularizers.L2(1e-5),
        name=f"{name_prefix}_conv1",
    )(x)
    gate = Conv2D(
        1,
        (1, 1),
        padding="same",
        activation="sigmoid",
        name=f"{name_prefix}_gate",
    )(gate)
    return Multiply(name=f"{name_prefix}_multiply")([x, gate])


def build_convlstm(
    input_shape,
    valid_mask,
    learning_rate=8e-4,
    filters=(64, 32, 16),
    dropout=0.20,
    l2=1e-5,
):
    """
    Build the deep ConvLSTM AQI forecasting model.

    Parameters
    ----------
    input_shape : tuple
        (sequence_length, grid_rows, grid_cols, channels)
    valid_mask : np.ndarray bool (H, W)
        True where a real city exists in the grid.
    learning_rate : float
        Peak Adam learning rate (cosine-decay restarts schedule).
    filters : tuple
        Number of filters for the three ConvLSTM layers.
    dropout : float
        Spatial dropout rate applied after each ConvLSTM block.
    l2 : float
        L2 kernel regularization weight.
    """
    # ------------------------------------------------------------------ input
    inputs = Input(shape=input_shape, name="input_seq")

    # ---------------------------------------------------------------- encoder
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
    )(inputs)
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
    x = BatchNormalization(name="bn2")(x)
    x = Dropout(dropout, name="drop2")(x)

    x = ConvLSTM2D(
        filters=filters[2],
        kernel_size=(3, 3),
        padding="same",
        return_sequences=False,          # collapse time axis
        activation="tanh",
        recurrent_activation="sigmoid",
        kernel_regularizer=regularizers.L2(l2),
        recurrent_regularizer=regularizers.L2(l2),
        dropout=0.05,
        name="clstm3",
    )(x)
    x = BatchNormalization(name="bn3")(x)

    # --------------------------------------------------------- spatial attention
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
    x = BatchNormalization(name="bn_refine")(x)
    delta = Conv2D(1, (1, 1), padding="same", activation="linear", name="delta_out")(x)

    # ----------------------------------------- residual skip: last input frame
    # LastFrameExtractor is a proper Layer (not a Lambda) so the model
    # serializes cleanly without safe_mode=False.
    last_frame = LastFrameExtractor(name="last_input_frame")(inputs)
    outputs = Add(name="residual_add")([last_frame, delta])

    model = Model(inputs=inputs, outputs=outputs, name="ConvLSTM_AQI")

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

    # Use a plain float LR so ReduceLROnPlateau can modify it during training.
    # CosineDecayRestarts conflicts with ReduceLROnPlateau (schedule LR is not settable).
    model.compile(
        optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=masked_mse,
        metrics=[masked_mae, masked_smape],
    )

    return model

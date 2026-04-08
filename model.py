"""
model.py — EfficientNetB4-based eye disease classifier.
Call build_model() to get the compiled Keras model.
"""

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,
)
from tensorflow.keras.models import Model

NUM_CLASSES = 10
IMG_SIZE = 380


def build_model(num_classes: int = NUM_CLASSES, freeze_until: int = -30) -> Model:
    """
    Build and compile the EfficientNetB4 transfer-learning model.

    Args:
        num_classes:   Number of output disease categories.
        freeze_until:  Freeze all base-model layers except the last N.
                       Use a negative index (e.g. -30) to unfreeze the tail.
    Returns:
        Compiled Keras Model.
    """
    base_model = EfficientNetB4(
        weights="imagenet",
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )

    # Fine-tune only the last |freeze_until| layers
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    for layer in base_model.layers[freeze_until:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(512, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"],
    )

    return model

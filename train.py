"""
train.py — Train the EfficientNetB4 eye disease classifier.
Run from the /training directory:
    python train.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB4
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    GlobalAveragePooling2D,
    BatchNormalization,          # ← was missing in original code
)
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    ReduceLROnPlateau,
)

from config import (
    IMG_SIZE,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    FREEZE_UNTIL,
    DATASET_DIR,
    MODEL_SAVE_PATH,
    ROTATION_RANGE,
    ZOOM_RANGE,
    HORIZONTAL_FLIP,
    VERTICAL_FLIP,
    SHEAR_RANGE,
    WIDTH_SHIFT_RANGE,
    HEIGHT_SHIFT_RANGE,
    BRIGHTNESS_RANGE,
    NUM_CLASSES,
)

# ── Data generators ────────────────────────────────────────────────────────────
train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=ROTATION_RANGE,
    zoom_range=ZOOM_RANGE,
    horizontal_flip=HORIZONTAL_FLIP,
    vertical_flip=VERTICAL_FLIP,
    shear_range=SHEAR_RANGE,
    width_shift_range=WIDTH_SHIFT_RANGE,
    height_shift_range=HEIGHT_SHIFT_RANGE,
    brightness_range=BRIGHTNESS_RANGE,
    fill_mode="nearest",
    validation_split=0.2
)

val_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.2
)

train_data = train_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
)

val_data = val_gen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False,
)

# ── Model ─────────────────────────────────────────────────────────────────────
base_model = EfficientNetB4(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)

# Freeze all layers except the last |FREEZE_UNTIL|
for layer in base_model.layers[:FREEZE_UNTIL]:
    layer.trainable = False
for layer in base_model.layers[FREEZE_UNTIL:]:
    layer.trainable = True

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dense(512, activation="relu")(x)
x = Dropout(0.5)(x)
output = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

callbacks = [
    ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1,
    ),
]

# ── Calculate Class Weights ───────────────────────────────────────────────────
class_counts = np.bincount(train_data.classes)
total_samples = np.sum(class_counts)
computed_weights = {}
for i, count in enumerate(class_counts):
    # Standard balancing heuristic: total / (num_classes * count)
    computed_weights[i] = total_samples / (NUM_CLASSES * count) if count > 0 else 1.0

print(f"\\n⚖️ Computed Class Weights: {computed_weights}\\n")

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=computed_weights
)

print(f"\n✅ Best model saved to {MODEL_SAVE_PATH}")
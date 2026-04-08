"""
config.py — Centralized training configuration.
Import this module in train.py instead of hardcoding values.
"""

# ── Image settings ─────────────────────────────────────────────────────────────
IMG_SIZE    = 380          # EfficientNetB4 native resolution
NUM_CLASSES = 10

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE     = 16
EPOCHS         = 5
LEARNING_RATE  = 1e-4
FREEZE_UNTIL   = -60       # unfreeze tail of base model (negative index)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATASET_DIR    = "dataset/Augmented_Dataset"
MODEL_SAVE_PATH = "../backend/save_model/model.h5"

# ── Augmentation ───────────────────────────────────────────────────────────────
ROTATION_RANGE    = 20
ZOOM_RANGE        = 0.2
HORIZONTAL_FLIP   = True
VERTICAL_FLIP     = True
SHEAR_RANGE       = 0.15
WIDTH_SHIFT_RANGE = 0.1
HEIGHT_SHIFT_RANGE = 0.1
BRIGHTNESS_RANGE  = [0.8, 1.2]

# ── Classes (must match dataset folder names, sorted alphabetically) ────────────
CLASSES = [
    "central_serous_chorioretinopathy",
    "diabetic_retinopathy",
    "disc_edema",
    "glaucoma",
    "healthy",
    "macular_scar",
    "myopia",
    "pterygium",
    "retinal_detachment",
    "retinitis_pigmentosa",
]

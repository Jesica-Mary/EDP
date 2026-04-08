# Reload trigger
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import io
import os

app = FastAPI(title="Eye Disease AI API", version="1.0.0")

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "save_model", "model.h5")

try:
    model = load_model(MODEL_PATH)
    print(f"[OK] Model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    model = None

# ── Classes ───────────────────────────────────────────────────────────────────
CLASSES = [
    "Retinitis Pigmentosa",
    "Retinal Detachment",
    "Pterygium",
    "Myopia",
    "Macular Scar",
    "Healthy",
    "Glaucoma",
    "Disc Edema",
    "Diabetic Retinopathy",
    "Central Serous Chorioretinopathy",
]

IMG_SIZE = 380

# ── Disease info for frontend ─────────────────────────────────────────────────
DISEASE_INFO = {
    "Retinitis Pigmentosa": "A genetic disorder that causes progressive retinal degeneration, leading to night blindness and tunnel vision.",
    "Retinal Detachment": "A serious condition where the retina separates from the back of the eye. Requires urgent medical attention.",
    "Pterygium": "A non-cancerous growth of the conjunctiva that can extend onto the cornea and affect vision.",
    "Myopia": "Nearsightedness — a common refractive error where close objects are clear but distant objects appear blurry.",
    "Macular Scar": "Scarring of the macula that can permanently impair central vision and color perception.",
    "Healthy": "No signs of disease detected. The eye appears to be in normal and healthy condition.",
    "Glaucoma": "A group of eye conditions damaging the optic nerve, often caused by abnormally high intraocular pressure.",
    "Disc Edema": "Swelling of the optic disc, often indicating increased intracranial pressure or inflammation.",
    "Diabetic Retinopathy": "Diabetes-related damage to blood vessels in the retina, a leading cause of blindness.",
    "Central Serous Chorioretinopathy": "Fluid accumulation under the retina causing blurry or distorted central vision.",
}

SEVERITY = {
    "Retinitis Pigmentosa": "high",
    "Retinal Detachment": "critical",
    "Pterygium": "low",
    "Myopia": "low",
    "Macular Scar": "medium",
    "Healthy": "none",
    "Glaucoma": "high",
    "Disc Edema": "high",
    "Diabetic Retinopathy": "high",
    "Central Serous Chorioretinopathy": "medium",
}


# ── Helpers ────────────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize, normalize, and batch-expand the image."""
    image = image.resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file")

    img = preprocess(image)
    preds = model.predict(img, verbose=0)[0]

    idx = int(np.argmax(preds))
    top_indices = np.argsort(preds)[::-1][:5].tolist()

    prediction = CLASSES[idx]

    return {
        "prediction": prediction,
        "confidence": round(float(preds[idx]) * 100, 2),
        "severity": SEVERITY.get(prediction, "unknown"),
        "info": DISEASE_INFO.get(prediction, ""),
        "top5": [
            {"label": CLASSES[i], "confidence": round(float(preds[i]) * 100, 2)}
            for i in top_indices
        ],
    }
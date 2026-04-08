"""
utils.py — Shared image preprocessing helpers.
"""

from PIL import Image
import numpy as np
import io

IMG_SIZE = 380
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def preprocess_image(image: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """
    Resize to (size, size), convert to RGB, normalize to [0, 1],
    and add a batch dimension.
    """
    image = image.convert("RGB").resize((size, size), Image.LANCZOS)
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def bytes_to_pil(raw: bytes) -> Image.Image:
    """Convert raw bytes to a PIL Image."""
    return Image.open(io.BytesIO(raw))


def is_valid_image_type(content_type: str) -> bool:
    """Return True if the MIME type is an accepted image format."""
    return content_type in ALLOWED_TYPES

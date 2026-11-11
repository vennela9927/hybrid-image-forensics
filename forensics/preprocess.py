"""Step 1-2 of the pipeline: image input and preprocessing."""
from __future__ import annotations

import io
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

MODEL_SIZE = 256          # resolution the CNN works at
MAX_ANALYSIS_SIDE = 768   # forensic maps are computed at (up to) this resolution


def load_image(src) -> np.ndarray:
    """Load an image from a path, bytes, file-like, PIL image or array.

    Returns an RGB uint8 array of shape (H, W, 3).
    """
    if isinstance(src, np.ndarray):
        img = src
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] == 4:
            img = img[:, :, :3]
        return np.ascontiguousarray(img.astype(np.uint8))
    if isinstance(src, Image.Image):
        pil = src
    elif isinstance(src, (str, Path)):
        pil = Image.open(src)
    elif isinstance(src, (bytes, bytearray)):
        pil = Image.open(io.BytesIO(src))
    else:  # file-like (e.g. Streamlit UploadedFile)
        pil = Image.open(src)
    pil = ImageOps.exif_transpose(pil).convert("RGB")
    return np.asarray(pil, dtype=np.uint8).copy()


def limit_size(img: np.ndarray, max_side: int = MAX_ANALYSIS_SIDE) -> np.ndarray:
    """Downscale very large images so forensic analysis stays fast.

    Area interpolation is used because it preserves local statistics best.
    """
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    f = max_side / s
    return cv2.resize(img, (round(w * f), round(h * f)), interpolation=cv2.INTER_AREA)


def resize_square(arr: np.ndarray, size: int = MODEL_SIZE, interp=cv2.INTER_AREA) -> np.ndarray:
    return cv2.resize(arr, (size, size), interpolation=interp)


def to_gray(img: np.ndarray) -> np.ndarray:
    """RGB uint8 -> float32 luminance in [0, 255]."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)


def normalize_rgb(img: np.ndarray) -> np.ndarray:
    """RGB uint8 (H,W,3) -> float32 (3,H,W) in [-1, 1]."""
    return (img.astype(np.float32) / 127.5 - 1.0).transpose(2, 0, 1)


def robust_normalize(m: np.ndarray, clip: float = 4.0) -> np.ndarray:
    """Standardise a forensic map with median / MAD so that *relative*
    inconsistencies inside one image stand out regardless of global level."""
    m = m.astype(np.float32)
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) * 1.4826 + 1e-6
    scale = max(mad, 0.25 * float(m.std()) + 1e-6)
    z = (m - med) / scale
    return np.clip(z, -clip, clip * 2) / clip

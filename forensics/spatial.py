"""Step 3a: spatial-domain forensic analysis.

Every function takes an RGB uint8 image and returns a float32 map (H, W) or a
visualisation image. These maps expose traces that editing tools leave in the
pixel domain: compression-level differences (ELA), sensor-noise
inconsistencies (SRM / noise residuals) and edge / texture anomalies.
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import to_gray

# ---------------------------------------------------------------- ELA ---------

def ela(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """Error Level Analysis.

    The image is re-saved as JPEG at a known quality and compared with the
    original. Regions that were pasted in (or compressed a different number of
    times) re-compress differently and show a different error level.
    Returns the per-pixel mean absolute error (float32, 0..255).
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    rec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    diff = np.abs(img.astype(np.float32) - rec.astype(np.float32))
    return diff.mean(axis=2)


def ela_visual(img: np.ndarray, quality: int = 90) -> np.ndarray:
    """Colour ELA image, brightness-scaled for display (RGB uint8)."""
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    rec = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    diff = np.abs(img.astype(np.float32) - rec.astype(np.float32))
    scale = 255.0 / max(np.percentile(diff, 99.5), 1.0)
    return np.clip(diff * scale, 0, 255).astype(np.uint8)


# ------------------------------------------------------- noise residuals ------

# Classic SRM (Spatial Rich Model) high-pass kernels (Fridrich & Kodovsky).
SRM_KERNELS = np.array([
    # 1st order edge residual (normalised)
    [[0, 0, 0, 0, 0],
     [0, -1, 2, -1, 0],
     [0, 2, -4, 2, 0],
     [0, -1, 2, -1, 0],
     [0, 0, 0, 0, 0]],
    # KV kernel
    [[-1, 2, -2, 2, -1],
     [2, -6, 8, -6, 2],
     [-2, 8, -12, 8, -2],
     [2, -6, 8, -6, 2],
     [-1, 2, -2, 2, -1]],
    # 2nd order horizontal
    [[0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0],
     [0, 1, -2, 1, 0],
     [0, 0, 0, 0, 0],
     [0, 0, 0, 0, 0]],
], dtype=np.float32)
SRM_SCALE = np.array([4.0, 12.0, 2.0], dtype=np.float32)


def srm_residuals(img: np.ndarray) -> np.ndarray:
    """Apply the three SRM filters to luminance. Returns (3, H, W) float32."""
    g = to_gray(img)
    out = [cv2.filter2D(g, cv2.CV_32F, k / s, borderType=cv2.BORDER_REFLECT)
           for k, s in zip(SRM_KERNELS, SRM_SCALE)]
    return np.stack(out)


def noise_residual(img: np.ndarray) -> np.ndarray:
    """Noise residual = image - denoised(image). |residual| per pixel."""
    g = to_gray(img)
    den = cv2.medianBlur(g.astype(np.uint8), 3).astype(np.float32)
    return np.abs(g - den)


def local_noise_map(img: np.ndarray, win: int = 15) -> np.ndarray:
    """Local standard deviation of the high-pass (SRM KV) residual.

    Authentic photos have a roughly uniform sensor-noise level; spliced,
    inpainted or AI-generated regions usually have a different one.
    """
    r = srm_residuals(img)[1]
    mu = cv2.boxFilter(r, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.boxFilter(r * r, -1, (win, win), borderType=cv2.BORDER_REFLECT)
    return np.sqrt(np.maximum(mu2 - mu * mu, 0))


# ------------------------------------------------------- edges / texture -----

def edge_map(img: np.ndarray) -> np.ndarray:
    """Sobel gradient magnitude (float32)."""
    g = to_gray(img)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def lbp_map(img: np.ndarray) -> np.ndarray:
    """8-neighbour Local Binary Pattern codes (uint8 map)."""
    g = to_gray(img)
    p = np.pad(g, 1, mode="edge")
    c = p[1:-1, 1:-1]
    shifts = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    code = np.zeros_like(c, dtype=np.uint8)
    h, w = c.shape
    for bit, (dy, dx) in enumerate(shifts):
        code |= ((p[dy:dy + h, dx:dx + w] >= c).astype(np.uint8) << bit)
    return code


def color_statistics(img: np.ndarray) -> np.ndarray:
    """Mean / std / skewness of each YCrCb channel (9 values)."""
    ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb).reshape(-1, 3).astype(np.float32)
    mu = ycc.mean(0)
    sd = ycc.std(0) + 1e-6
    sk = (((ycc - mu) / sd) ** 3).mean(0)
    return np.concatenate([mu / 255, sd / 128, sk / 5]).astype(np.float32)

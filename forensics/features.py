"""Multi-domain feature extraction.

* `forensic_maps`   - dense per-pixel maps from both domains (input to the
                      CNN's forensic stream and shown in the UI).
* `model_inputs`    - RGB + normalised forensic maps resized for the CNN.
* `global_features` - a fixed-length hand-crafted feature vector
                      (spatial + frequency + colour statistics) used by the
                      classical machine-learning baseline.
"""
from __future__ import annotations

import cv2
import numpy as np

from . import frequency as fq
from . import spatial as sp
from .preprocess import MODEL_SIZE, limit_size, normalize_rgb, resize_square, robust_normalize

# Order matters: this is the channel order of the CNN forensic stream.
FORENSIC_CHANNELS = [
    ("ela", "spatial"),            # error level analysis
    ("noise", "spatial"),          # local SRM noise level
    ("fft_hp", "frequency"),       # FFT high-pass energy
    ("dct_ac", "frequency"),       # 8x8 DCT high-frequency AC energy
    ("blockiness", "frequency"),   # JPEG blocking-artifact strength
    ("wavelet_noise", "frequency"),  # DWT-HH robust noise estimate
]
N_FORENSIC = len(FORENSIC_CHANNELS)


def forensic_maps(img: np.ndarray) -> dict[str, np.ndarray]:
    """Compute all dense forensic maps at the image's own resolution."""
    return {
        "ela": cv2.GaussianBlur(sp.ela(img, 90), (0, 0), 1.0),
        "noise": sp.local_noise_map(img, 15),
        "fft_hp": fq.fft_highpass_map(img),
        "dct_ac": fq.dct_ac_map(img),
        "blockiness": fq.blockiness_map(img),
        "wavelet_noise": fq.wavelet_noise_map(img),
    }


def model_inputs(img: np.ndarray, size: int = MODEL_SIZE, maps: dict | None = None):
    """Prepare CNN inputs.

    Returns
        rgb      float32 (3, size, size) in [-1, 1]
        forensic float32 (N_FORENSIC, size, size), robust-normalised
        maps     the raw native-resolution maps (re-used for visualisation)
    """
    img = limit_size(img)
    if maps is None:
        maps = forensic_maps(img)
    rgb = normalize_rgb(resize_square(img, size))
    chans = []
    for name, _ in FORENSIC_CHANNELS:
        m = resize_square(maps[name].astype(np.float32), size)
        chans.append(robust_normalize(m))
    return rgb, np.stack(chans).astype(np.float32), maps


# ------------------------------------------------------------- global ---------

def _map_stats(m: np.ndarray) -> list[float]:
    """Statistics that capture *inconsistency* of a forensic map."""
    m = m.astype(np.float32)
    mu, sd = float(m.mean()), float(m.std()) + 1e-6
    p50, p95, p99 = np.percentile(m, [50, 95, 99])
    # block-level heterogeneity: how different are 32x32 regions from each other
    h, w = m.shape
    bs = 32
    H, W = h - h % bs, w - w % bs
    if H >= bs and W >= bs:
        bm = m[:H, :W].reshape(H // bs, bs, W // bs, bs).mean((1, 3))
        het = float(bm.std() / (bm.mean() + 1e-6))
        rng = float((bm.max() - np.median(bm)) / (np.median(bm) + 1e-6))
        low = float((np.median(bm) - bm.min()) / (np.median(bm) + 1e-6))
    else:
        het = rng = low = 0.0
    z = (m - mu) / sd
    return [np.log1p(mu), sd / (abs(mu) + 1e-6), float((z ** 3).mean()), float((z ** 4).mean()),
            float((p95 + 1e-6) / (p50 + 1e-6)), float((p99 + 1e-6) / (p50 + 1e-6)), het, rng, low]


def global_features(img: np.ndarray, maps: dict | None = None) -> np.ndarray:
    """Hand-crafted multi-domain feature vector for the classical baseline."""
    img = limit_size(img)
    if maps is None:
        maps = forensic_maps(img)
    f: list[float] = []
    for name, _ in FORENSIC_CHANNELS:              # spatial + frequency maps
        f += _map_stats(maps[name])
    f += list(fq.radial_spectrum(img, 24))          # frequency: FFT profile
    f.append(fq.spectral_peak_score(img))           # frequency: periodic peaks
    f += list(fq.dct_histogram_features(img))       # frequency: DCT statistics
    lbp = sp.lbp_map(img)                           # spatial: texture
    hist = np.bincount(lbp.ravel(), minlength=256).astype(np.float32)
    hist = hist.reshape(16, 16).sum(1)
    f += list(hist / hist.sum())
    f += list(sp.color_statistics(img))             # colour statistics
    e = sp.edge_map(img)                            # spatial: edges
    f += [float(np.log1p(e.mean())), float(e.std() / (e.mean() + 1e-6))]
    return np.nan_to_num(np.array(f, np.float32), nan=0.0, posinf=10.0, neginf=-10.0)


def heuristic_heatmap(maps: dict, shape) -> np.ndarray:
    """Training-free localization: fuse the normalised forensic maps by taking
    the per-pixel mean absolute deviation from each map's typical level.
    Used by the baseline and as a fallback when no trained model exists."""
    h, w = shape
    acc = np.zeros((h, w), np.float32)
    for name, _ in FORENSIC_CHANNELS:
        m = cv2.resize(maps[name].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        acc += np.abs(robust_normalize(m))
    acc /= N_FORENSIC
    acc = cv2.GaussianBlur(acc, (0, 0), max(h, w) / 100)
    acc -= acc.min()
    return acc / (acc.max() + 1e-6)


def heuristic_score(maps: dict) -> float:
    """Training-free image-level score: how much the most anomalous region
    (98th percentile of the fused, smoothed inconsistency map) stands out from
    the typical level of the image."""
    acc = None
    for name, _ in FORENSIC_CHANNELS:
        m = cv2.resize(maps[name].astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
        z = np.abs(robust_normalize(m))
        acc = z if acc is None else acc + z
    acc = cv2.GaussianBlur(acc / N_FORENSIC, (0, 0), 3)
    return float(np.percentile(acc, 98) - np.median(acc))

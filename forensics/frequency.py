"""Step 3b: frequency-domain forensic analysis (FFT, DCT, DWT).

Editing and AI generation disturb the frequency content of an image:
  * FFT  - resampling / generative up-sampling leave periodic peaks and
           unnatural high-frequency energy.
  * DCT  - JPEG works on 8x8 DCT blocks; pasted regions carry a different (or
           mis-aligned) compression history, visible as changes in block AC
           energy and blocking-artifact strength.
  * DWT  - the diagonal (HH) wavelet sub-band is dominated by sensor noise; a
           local robust noise estimate reveals regions with a foreign noise
           level (inpainting, splicing, AI fill).
"""
from __future__ import annotations

import cv2
import numpy as np

from .preprocess import to_gray

try:  # PyWavelets is optional - a Haar transform is implemented as fallback
    import pywt
except Exception:  # pragma: no cover
    pywt = None


# ------------------------------------------------------------------ FFT -------

def fft_spectrum(img: np.ndarray) -> np.ndarray:
    """Centred log-magnitude spectrum, scaled to 0..1 (for display)."""
    g = to_gray(img)
    g = g - g.mean()
    win = np.outer(np.hanning(g.shape[0]), np.hanning(g.shape[1])).astype(np.float32)
    F = np.fft.fftshift(np.fft.fft2(g * win))
    mag = np.log1p(np.abs(F))
    mag -= mag.min()
    return (mag / (mag.max() + 1e-6)).astype(np.float32)


def fft_highpass_map(img: np.ndarray, cutoff: float = 0.25, win: int = 9) -> np.ndarray:
    """Local energy of the high-frequency component of the image.

    A circular high-pass filter removes everything below `cutoff` (fraction of
    Nyquist); the inverse FFT gives a high-frequency residual whose local
    energy is returned.
    """
    g = to_gray(img)
    h, w = g.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    r = np.sqrt(fx ** 2 + fy ** 2) / 0.5
    hp = 1.0 / (1.0 + (cutoff / (r + 1e-6)) ** 8)        # Butterworth high-pass
    res = np.real(np.fft.ifft2(np.fft.fft2(g) * hp)).astype(np.float32)
    return cv2.boxFilter(np.abs(res), -1, (win, win), borderType=cv2.BORDER_REFLECT)


def radial_spectrum(img: np.ndarray, bins: int = 24) -> np.ndarray:
    """Azimuthally averaged power spectrum (log), normalised. Used as a global
    frequency descriptor (GAN / resampling fingerprints show up here)."""
    g = to_gray(img)
    g = g - g.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    h, w = P.shape
    yy, xx = np.indices(P.shape)
    r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
    r = r / r.max()
    idx = np.minimum((r * bins).astype(int), bins - 1)
    prof = np.bincount(idx.ravel(), P.ravel(), bins) / (np.bincount(idx.ravel(), None, bins) + 1e-6)
    prof = np.log1p(prof)
    return ((prof - prof.mean()) / (prof.std() + 1e-6)).astype(np.float32)


def spectral_peak_score(img: np.ndarray) -> float:
    """Strength of isolated periodic peaks in the high-frequency spectrum
    (typical for up-sampling / generative artifacts)."""
    g = to_gray(img)
    g = g - cv2.GaussianBlur(g, (0, 0), 2)
    F = np.abs(np.fft.fftshift(np.fft.fft2(g)))
    F = np.log1p(F)
    bg = cv2.medianBlur(F.astype(np.float32), 5)
    h, w = F.shape
    yy, xx = np.indices(F.shape)
    mask = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) > (0.1 * min(h, w)) ** 2
    excess = (F - bg)[mask]
    return float(np.percentile(excess, 99.9))


# ------------------------------------------------------------------ DCT -------

_N = 8
_DCT = np.array([[np.sqrt((1 if k == 0 else 2) / _N) * np.cos((2 * n + 1) * k * np.pi / (2 * _N))
                  for n in range(_N)] for k in range(_N)], dtype=np.float32)
_UV = np.add.outer(np.arange(8), np.arange(8))


def _blocks(g: np.ndarray):
    h, w = g.shape
    H, W = h - h % 8, w - w % 8
    b = g[:H, :W].reshape(H // 8, 8, W // 8, 8).transpose(0, 2, 1, 3)
    return b, H, W


def block_dct(img: np.ndarray) -> np.ndarray:
    """8x8 block DCT (JPEG-aligned). Returns (H/8, W/8, 8, 8)."""
    g = to_gray(img) - 128.0
    b, _, _ = _blocks(g)
    return np.einsum("ij,abjk,lk->abil", _DCT, b, _DCT, optimize=True)


def _upsample_blocks(m: np.ndarray, shape) -> np.ndarray:
    h, w = shape
    up = np.kron(m, np.ones((8, 8), dtype=np.float32))
    out = np.empty((h, w), np.float32)
    out[:up.shape[0], :up.shape[1]] = up
    if up.shape[0] < h:
        out[up.shape[0]:, :up.shape[1]] = up[-1:, :]
    if up.shape[1] < w:
        out[:, up.shape[1]:] = out[:, up.shape[1] - 1:up.shape[1]]
    return out


def dct_ac_map(img: np.ndarray) -> np.ndarray:
    """Per-block high-frequency AC energy (log), upsampled to image size.
    JPEG quantisation removes high-frequency coefficients; regions compressed
    differently keep a different amount of them."""
    D = block_dct(img)
    hf = np.abs(D[..., _UV >= 5]).mean(-1)
    return _upsample_blocks(np.log1p(hf).astype(np.float32), img.shape[:2])


def dct_histogram_features(img: np.ndarray) -> np.ndarray:
    """Global DCT statistics: for 9 low-frequency AC positions, how strongly
    coefficient values cluster on integer multiples (double-quantisation
    periodicity) + their spread. 18 values."""
    D = block_dct(img)
    feats = []
    for (u, v) in [(0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1), (2, 2), (0, 3)]:
        c = D[..., u, v].ravel()
        c = c[np.abs(c) > 0.5]
        if c.size < 16:
            feats += [0.0, 0.0]
            continue
        hist, _ = np.histogram(np.abs(c), bins=40, range=(0, 40))
        spec = np.abs(np.fft.rfft(hist - hist.mean()))
        feats += [float(spec[1:].max() / (spec.sum() + 1e-6)), float(np.log1p(np.std(c)))]
    return np.array(feats, np.float32)


def blockiness_map(img: np.ndarray) -> np.ndarray:
    """Strength of 8x8 JPEG blocking artifacts per block.

    Ratio of gradient across block boundaries to gradient inside blocks.
    A spliced region whose JPEG grid is mis-aligned (or that was never
    compressed) shows a different ratio than the rest of the image."""
    g = to_gray(img)
    dx = np.abs(np.diff(g, axis=1))
    dy = np.abs(np.diff(g, axis=0))
    h, w = g.shape
    H, W = h - h % 8, w - w % 8
    if H < 16 or W < 16:
        return np.zeros((h, w), np.float32)
    dxb = np.zeros((h, w), np.float32)
    dxb[:, :-1] = dx
    dyb = np.zeros((h, w), np.float32)
    dyb[:-1, :] = dy
    col = np.arange(w) % 8 == 7
    row = np.arange(h) % 8 == 7
    bx, _, _ = _blocks(dxb * col[None, :])
    ix, _, _ = _blocks(dxb * (~col)[None, :])
    by, _, _ = _blocks(dyb * row[:, None])
    iy, _, _ = _blocks(dyb * (~row)[:, None])
    boundary = bx.sum((2, 3)) / 8 + by.sum((2, 3)) / 8
    inner = (ix.sum((2, 3)) + iy.sum((2, 3))) / 56
    ratio = np.log((boundary + 1.0) / (inner + 1.0))
    ratio = cv2.blur(ratio.astype(np.float32), (3, 3))
    return _upsample_blocks(ratio, (h, w))


# ------------------------------------------------------------------ DWT -------

def dwt_subbands(img: np.ndarray):
    """One-level 2-D Haar DWT of luminance: (LL, LH, HL, HH)."""
    g = to_gray(img)
    if pywt is not None:
        LL, (LH, HL, HH) = pywt.dwt2(g, "haar")
        return LL, LH, HL, HH
    h, w = g.shape
    g = g[: h - h % 2, : w - w % 2]
    a, b = g[0::2, 0::2], g[0::2, 1::2]
    c, d = g[1::2, 0::2], g[1::2, 1::2]
    return (a + b + c + d) / 2, (a + b - c - d) / 2, (a - b + c - d) / 2, (a - b - c + d) / 2


def wavelet_noise_map(img: np.ndarray, block: int = 8) -> np.ndarray:
    """Local noise standard deviation estimated from the HH wavelet sub-band
    with the robust median estimator  sigma = median(|HH|) / 0.6745
    (Donoho), computed on overlapping windows and upsampled to image size."""
    HH = np.abs(dwt_subbands(img)[3]).astype(np.float32)
    h2, w2 = HH.shape
    H, W = h2 - h2 % block, w2 - w2 % block
    if H == 0 or W == 0:
        return np.zeros(img.shape[:2], np.float32)
    b = HH[:H, :W].reshape(H // block, block, W // block, block).transpose(0, 2, 1, 3)
    sig = np.median(b.reshape(H // block, W // block, -1), axis=-1) / 0.6745
    sig = cv2.blur(sig.astype(np.float32), (3, 3))
    return cv2.resize(sig, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)

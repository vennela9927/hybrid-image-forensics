"""Synthetic manipulation dataset generator.

Public forensic datasets (CASIA v2, Columbia, COVERAGE, IMD2020 ...) are large
and need manual download, so the project ships with a generator that creates
authentic / manipulated image pairs *with pixel-accurate ground-truth masks*.

Base images come from either
  * a folder of real photos (`real_dir`, recommended - use your own photos), or
  * a procedural scene generator + simulated camera pipeline (default, offline).

Manipulations (the kinds produced by both classic editors and AI tools):
  splice        paste a region from another image (different camera noise /
                JPEG history)
  copy_move     duplicate a region inside the same image (scaled / rotated)
  inpaint       object removal by diffusion inpainting (no sensor noise)
  ai_generate   simulated generative fill: low-res content up-sampled with a
                periodic decoder (checkerboard) fingerprint and clean noise
  retouch       local blur / sharpen / median / contrast edit

Authentic and manipulated images get the *same* global post-processing
(global resize and final JPEG re-save), so the model cannot cheat by detecting
"was re-saved" - it must find *local inconsistencies*.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np

OPS = ["splice", "copy_move", "inpaint", "ai_generate", "retouch"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


# ------------------------------------------------------------- scenes --------

def _fractal_noise(rng, h, w, octaves=5, persistence=0.55):
    out = np.zeros((h, w), np.float32)
    amp, tot = 1.0, 0.0
    for o in range(octaves):
        s = 2 ** (o + 2)
        small = rng.standard_normal((max(2, h * s // max(h, w)), max(2, w * s // max(h, w)))).astype(np.float32)
        out += amp * cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        tot += amp
        amp *= persistence
    out /= tot
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def _rand_color(rng):
    return rng.uniform(0.05, 0.95, 3).astype(np.float32)


def procedural_scene(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """Natural-looking random scene (float32 RGB in [0,1])."""
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    horizon = rng.uniform(0.3, 0.7)
    sky = _rand_color(rng) * (1 - yy * 0.5) + _rand_color(rng) * yy * 0.5
    ground = _rand_color(rng) * 0.8
    tex = _fractal_noise(rng, h, w)[..., None]
    img = np.where(yy < horizon, sky, ground * (0.6 + 0.8 * tex)).astype(np.float32)
    img = img * np.ones((1, w, 1), np.float32)
    # objects
    for _ in range(rng.integers(4, 12)):
        col = _rand_color(rng)
        t = _fractal_noise(rng, h, w, octaves=rng.integers(3, 6))[..., None]
        layer = col * (0.55 + 0.9 * t * rng.uniform(0.3, 1.0))
        m = np.zeros((h, w), np.uint8)
        kind = rng.integers(0, 3)
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        if kind == 0:
            cv2.ellipse(m, (cx, cy), (int(rng.integers(w // 20, w // 4)), int(rng.integers(h // 20, h // 4))),
                        float(rng.uniform(0, 180)), 0, 360, 255, -1)
        elif kind == 1:
            pts = np.stack([rng.integers(0, w, 6), rng.integers(0, h, 6)], 1).astype(np.int32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(m, hull, 255)
        else:
            x0, y0 = int(rng.integers(0, w - 8)), int(rng.integers(0, h - 8))
            cv2.rectangle(m, (x0, y0), (x0 + int(rng.integers(8, w // 3)), y0 + int(rng.integers(8, h // 3))), 255, -1)
        a = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), rng.uniform(0.6, 2.0))[..., None]
        shade = 0.75 + 0.5 * _fractal_noise(rng, h, w, octaves=2)[..., None]
        img = img * (1 - a) + layer * shade * a
    # thin structures (edges / lines)
    for _ in range(rng.integers(0, 8)):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        c = tuple(float(v) for v in _rand_color(rng))
        cv2.line(img, p1, p2, c, int(rng.integers(1, 4)), cv2.LINE_AA)
    illum = 0.7 + 0.6 * _fractal_noise(rng, h, w, octaves=2)[..., None]
    return np.clip(img * illum, 0, 1).astype(np.float32)


def jpeg(img_u8: np.ndarray, q: int) -> np.ndarray:
    bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    _, enc = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def camera(rng, scene: np.ndarray, params: dict | None = None) -> np.ndarray:
    """Simulate a camera: optics blur, shot + read noise, gamma, JPEG.
    Different `params` emulate different cameras."""
    p = params or random_camera(rng)
    img = cv2.GaussianBlur(scene, (0, 0), p["blur"])
    lin = np.power(img, 2.2)
    shot = rng.standard_normal(lin.shape).astype(np.float32) * np.sqrt(lin + 1e-4) * p["shot"]
    read = rng.standard_normal(lin.shape).astype(np.float32) * p["read"]
    lin = np.clip(lin + shot + read, 0, 1)
    out = (np.power(lin, 1 / 2.2) * 255).round().clip(0, 255).astype(np.uint8)
    return jpeg(out, p["q"]) if p["q"] < 100 else out


def random_camera(rng) -> dict:
    return {"blur": float(rng.uniform(0.4, 1.1)), "shot": float(rng.uniform(0.004, 0.03)),
            "read": float(rng.uniform(0.002, 0.012)), "q": int(rng.choice([100, *range(70, 99)]))}


class ImageSource:
    """Gives 'camera' images: crops of real photos, or simulated ones."""

    def __init__(self, real_dir: str | None = None, real_ratio: float = 1.0):
        self.files = []
        if real_dir:
            self.files = [p for p in Path(real_dir).rglob("*") if p.suffix.lower() in IMG_EXTS]
        self.real_ratio = real_ratio if self.files else 0.0

    def get(self, rng, h, w, cam=None) -> np.ndarray:
        if self.files and rng.random() < self.real_ratio:
            for _ in range(5):
                im = cv2.imread(str(self.files[rng.integers(len(self.files))]), cv2.IMREAD_COLOR)
                if im is None:
                    continue
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                ih, iw = im.shape[:2]
                if ih < h or iw < w:
                    f = max(h / ih, w / iw) * 1.01
                    im = cv2.resize(im, (int(iw * f) + 1, int(ih * f) + 1), interpolation=cv2.INTER_CUBIC)
                    ih, iw = im.shape[:2]
                y, x = rng.integers(0, ih - h + 1), rng.integers(0, iw - w + 1)
                return im[y:y + h, x:x + w].copy()
        return camera(rng, procedural_scene(rng, h, w), cam)


# ------------------------------------------------------------- masks ---------

def random_region_mask(rng, h, w, min_area=0.02, max_area=0.22) -> np.ndarray:
    for _ in range(20):
        m = np.zeros((h, w), np.uint8)
        kind = rng.integers(0, 4)
        target = rng.uniform(min_area, max_area) * h * w
        r = np.sqrt(target / np.pi)
        cx, cy = int(rng.integers(int(r * 0.7), max(int(r * 0.7) + 1, w - int(r * 0.7)))), \
            int(rng.integers(int(r * 0.7), max(int(r * 0.7) + 1, h - int(r * 0.7))))
        if kind == 0:  # ellipse
            ar = rng.uniform(0.5, 2.0)
            cv2.ellipse(m, (cx, cy), (int(r * np.sqrt(ar)), int(r / np.sqrt(ar))), float(rng.uniform(0, 180)), 0, 360, 255, -1)
        elif kind == 1:  # polygon
            n = rng.integers(5, 10)
            ang = np.sort(rng.uniform(0, 2 * np.pi, n))
            rad = r * rng.uniform(0.6, 1.4, n)
            pts = np.stack([cx + rad * np.cos(ang), cy + rad * np.sin(ang)], 1).astype(np.int32)
            cv2.fillPoly(m, [pts], 255)
        elif kind == 2:  # rectangle
            ar = rng.uniform(0.5, 2.0)
            hw, hh = int(r * np.sqrt(ar) * 0.9), int(r / np.sqrt(ar) * 0.9)
            cv2.rectangle(m, (cx - hw, cy - hh), (cx + hw, cy + hh), 255, -1)
        else:  # organic blob ("object" shape)
            n = _fractal_noise(rng, h, w, octaves=3)
            d = np.sqrt((np.arange(w)[None] - cx) ** 2 + (np.arange(h)[:, None] - cy) ** 2) / (r * 1.3)
            m = ((n * 0.8 + (1 - d)) > 0.75).astype(np.uint8) * 255
        area = (m > 0).mean()
        if min_area * 0.5 <= area <= max_area * 1.6:
            return m
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (w // 2, h // 2), int(min(h, w) * 0.2), 255, -1)
    return m


def _alpha(rng, m: np.ndarray) -> np.ndarray:
    a = m.astype(np.float32) / 255
    if rng.random() < 0.6:  # feathered edges like a soft brush / AI blending
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(0.8, 3.0))
    return a[..., None]


# ------------------------------------------------------------- operations -----

def op_splice(rng, img, src: ImageSource):
    h, w = img.shape[:2]
    donor = src.get(rng, h, w, cam=random_camera(rng))
    if rng.random() < 0.5:  # donor at a different scale
        f = rng.uniform(0.6, 1.5)
        d = cv2.resize(donor, (int(w * f) + 1, int(h * f) + 1), interpolation=cv2.INTER_LINEAR)
        donor = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR) if f < 1 else d[:h, :w]
    if rng.random() < 0.5:  # colour match attempt (makes it harder)
        donor = np.clip(donor.astype(np.float32) * (img.mean((0, 1)) + 1) / (donor.mean((0, 1)) + 1), 0, 255).astype(np.uint8)
    m = random_region_mask(rng, h, w)
    a = _alpha(rng, m)
    out = img.astype(np.float32) * (1 - a) + donor.astype(np.float32) * a
    return out.round().astype(np.uint8), m


def op_copy_move(rng, img, src=None):
    h, w = img.shape[:2]
    m = random_region_mask(rng, h, w, 0.02, 0.15)
    ys, xs = np.nonzero(m)
    for _ in range(20):
        dx, dy = int(rng.integers(-w // 2, w // 2)), int(rng.integers(-h // 2, h // 2))
        if abs(dx) + abs(dy) > min(h, w) // 6:
            break
    s = rng.uniform(0.85, 1.2) if rng.random() < 0.5 else 1.0
    ang = rng.uniform(-20, 20) if rng.random() < 0.4 else 0.0
    cy, cx = ys.mean(), xs.mean()
    M = cv2.getRotationMatrix2D((cx, cy), ang, s)
    M[:, 2] += (dx, dy)
    moved = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    tm = cv2.warpAffine(m, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    if tm.sum() < 0.01 * 255 * h * w:
        return op_splice(rng, img, ImageSource())
    a = _alpha(rng, tm)
    out = img.astype(np.float32) * (1 - a) + moved.astype(np.float32) * a
    return out.round().astype(np.uint8), tm


def op_inpaint(rng, img, src=None):
    h, w = img.shape[:2]
    m = random_region_mask(rng, h, w, 0.015, 0.12)
    flag = cv2.INPAINT_TELEA if rng.random() < 0.5 else cv2.INPAINT_NS
    out = cv2.inpaint(img, m, int(rng.integers(3, 9)), flag)
    if rng.random() < 0.3:  # forger adds a little noise to hide the smooth fill
        n = rng.standard_normal(out.shape) * rng.uniform(0.5, 2.0)
        out = np.where(m[..., None] > 0, np.clip(out + n, 0, 255), out).astype(np.uint8)
    return out, m


def op_ai_generate(rng, img, src: ImageSource):
    """Simulated generative (diffusion / GAN) fill: content is synthesised at
    low resolution and decoded by up-sampling layers, which leave a periodic
    fingerprint, a too-clean noise floor and smoothed texture."""
    h, w = img.shape[:2]
    m = random_region_mask(rng, h, w, 0.03, 0.25)
    content = img if rng.random() < 0.5 else src.get(rng, h, w)
    f = int(rng.choice([2, 4, 8]))
    small = cv2.resize(content, (max(4, w // f), max(4, h // f)), interpolation=cv2.INTER_AREA)
    interp = cv2.INTER_NEAREST if rng.random() < 0.3 else cv2.INTER_CUBIC
    gen = cv2.resize(small, (w, h), interpolation=interp).astype(np.float32)
    if interp == cv2.INTER_NEAREST:
        gen = cv2.GaussianBlur(gen, (0, 0), f / 3)
    period = int(rng.choice([2, 4, 8]))
    yy, xx = np.indices((h, w))
    checker = (((yy // (period // 2 or 1)) + (xx // (period // 2 or 1))) % 2 * 2 - 1).astype(np.float32)
    gen += checker[..., None] * rng.uniform(0.8, 3.0)
    gen += rng.standard_normal(gen.shape).astype(np.float32) * rng.uniform(0.0, 0.8)
    a = _alpha(rng, m)
    out = img.astype(np.float32) * (1 - a) + gen * a
    return out.round().clip(0, 255).astype(np.uint8), m


def op_retouch(rng, img, src=None):
    h, w = img.shape[:2]
    m = random_region_mask(rng, h, w, 0.03, 0.25)
    k = rng.integers(0, 4)
    f = img.astype(np.float32)
    if k == 0:
        e = cv2.GaussianBlur(f, (0, 0), rng.uniform(1.0, 3.0))
    elif k == 1:
        e = f + (f - cv2.GaussianBlur(f, (0, 0), 1.5)) * rng.uniform(1.0, 2.5)
    elif k == 2:
        e = cv2.medianBlur(img, int(rng.choice([3, 5]))).astype(np.float32)
    else:
        e = (f - f.mean()) * rng.uniform(1.2, 1.6) + f.mean() + rng.uniform(-25, 25)
    a = _alpha(rng, m)
    out = f * (1 - a) + e * a
    return out.round().clip(0, 255).astype(np.uint8), m


OP_FUNCS = {"splice": op_splice, "copy_move": op_copy_move, "inpaint": op_inpaint,
            "ai_generate": op_ai_generate, "retouch": op_retouch}


def postprocess(rng, img: np.ndarray, mask: np.ndarray):
    """Global post-processing applied identically to authentic and fake."""
    h, w = img.shape[:2]
    if rng.random() < 0.25:
        f = rng.uniform(0.7, 0.95)
        nh, nw = int(h * f), int(w * f)
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if rng.random() < 0.75:
        img = jpeg(img, int(rng.integers(75, 99)))
    return img, mask


def make_sample(rng: np.random.Generator, src: ImageSource, manipulated: bool, op: str | None = None,
                size_range=(256, 448)):
    """Create one sample -> (image uint8 RGB, mask uint8 {0,255}, op name)."""
    h = int(rng.integers(*size_range)) // 8 * 8
    w = int(rng.integers(*size_range)) // 8 * 8
    base = src.get(rng, h, w)
    if not manipulated:
        img, mask, op = base, np.zeros((h, w), np.uint8), "authentic"
    else:
        op = op or OPS[rng.integers(len(OPS))]
        img, mask = OP_FUNCS[op](rng, base, src)
        mask = (mask > 127).astype(np.uint8) * 255
    img, mask = postprocess(rng, img, mask)
    return img, mask, op


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)

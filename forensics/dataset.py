"""PyTorch datasets.

Two on-disk layouts are supported:

1. Generated layout (scripts/generate_dataset.py)
       <root>/<split>/images/*.png
       <root>/<split>/masks/*.png
       <root>/<split>/labels.csv      (file,mask,label,op)

2. Generic real-dataset layout (e.g. CASIA v2, Columbia, your own photos)
       <root>/authentic/*.jpg
       <root>/tampered/*.jpg
       <root>/masks/<same stem as tampered image>*.png   (optional)
"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .features import model_inputs
from .preprocess import MODEL_SIZE, load_image, limit_size

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def read_index(split_dir: Path):
    rows = []
    with open(split_dir / "labels.csv", newline="") as f:
        for r in csv.DictReader(f):
            rows.append({"image": split_dir / "images" / r["file"],
                         "mask": split_dir / "masks" / r["mask"] if r["mask"] else None,
                         "label": int(r["label"]), "op": r["op"]})
    return rows


def folder_index(root: Path):
    """Index a generic authentic/ tampered/ masks/ folder."""
    rows = []
    for p in sorted((root / "authentic").rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            rows.append({"image": p, "mask": None, "label": 0, "op": "authentic"})
    masks = {}
    if (root / "masks").exists():
        for m in (root / "masks").rglob("*"):
            if m.suffix.lower() in IMG_EXTS:
                masks[m.stem.replace("_gt", "").replace("_mask", "")] = m
    for p in sorted((root / "tampered").rglob("*")):
        if p.suffix.lower() in IMG_EXTS:
            rows.append({"image": p, "mask": masks.get(p.stem), "label": 1, "op": "tampered"})
    return rows


def make_index(root: str | Path, split: str | None = None):
    root = Path(root)
    if split and (root / split / "labels.csv").exists():
        return read_index(root / split)
    if (root / "labels.csv").exists():
        return read_index(root)
    if (root / "authentic").exists() or (root / "tampered").exists():
        return folder_index(root)
    raise FileNotFoundError(f"No dataset found at {root} (split={split})")


class ForensicDataset(Dataset):
    def __init__(self, rows, size: int = MODEL_SIZE, augment: bool = False):
        self.rows = rows
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        img = limit_size(load_image(r["image"]))
        h, w = img.shape[:2]
        if r["mask"] is not None and Path(r["mask"]).exists():
            m = cv2.imread(str(r["mask"]), cv2.IMREAD_GRAYSCALE)
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            has_mask = 1.0
        else:
            m = np.zeros((h, w), np.uint8)
            # authentic -> empty mask is exact; tampered without GT -> ignore pixels
            has_mask = 1.0 if r["label"] == 0 else 0.0
        if self.augment and np.random.rand() < 0.5:
            img, m = img[:, ::-1].copy(), m[:, ::-1].copy()
        rgb, foren, _ = model_inputs(img, self.size)
        mask = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        mask = (mask > 127).astype(np.float32)[None]
        return {
            "rgb": torch.from_numpy(rgb), "forensic": torch.from_numpy(foren),
            "mask": torch.from_numpy(mask), "label": torch.tensor(float(r["label"])),
            "has_mask": torch.tensor(has_mask), "idx": i,
        }


# ------------------------------------------------------------- feature cache ----
# Computing the forensic maps is CPU work (~50-100 ms / image). On a GPU machine
# with few CPU cores (e.g. Google Colab) that would starve the GPU, so the
# inputs can be computed once and stored as uint8 memory-mapped arrays
# (RGB exact, forensic maps quantised to 1/85 of a z-unit).

def _q(f):  # forensic maps live in [-1, 2]
    return np.clip(np.round((f + 1.0) / 3.0 * 255.0), 0, 255).astype(np.uint8)


def _dq(u):
    return u.astype(np.float32) / 255.0 * 3.0 - 1.0


def build_cache(rows, cache_dir, size: int = MODEL_SIZE, workers: int = 2, batch: int = 32):
    """Pre-compute model inputs for `rows` into `cache_dir` (skipped if present)."""
    from torch.utils.data import DataLoader
    cache_dir = Path(cache_dir)
    done = cache_dir / "done.txt"
    if done.exists() and int(done.read_text().strip() or -1) == len(rows):
        return cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)
    n = len(rows)
    from .features import N_FORENSIC
    X = np.lib.format.open_memmap(cache_dir / "x.npy", "w+", np.uint8, (n, 3 + N_FORENSIC, size, size))
    M = np.lib.format.open_memmap(cache_dir / "m.npy", "w+", np.uint8, (n, size, size))
    meta = np.zeros((n, 2), np.float32)
    dl = DataLoader(ForensicDataset(rows, size), batch_size=batch, num_workers=workers)
    k = 0
    for b in dl:
        bs = len(b["label"])
        rgb = ((b["rgb"].numpy() + 1) * 127.5).round().clip(0, 255).astype(np.uint8)
        X[k:k + bs, :3] = rgb
        X[k:k + bs, 3:] = _q(b["forensic"].numpy())
        M[k:k + bs] = (b["mask"][:, 0].numpy() > 0.5).astype(np.uint8)
        meta[k:k + bs, 0] = b["label"].numpy()
        meta[k:k + bs, 1] = b["has_mask"].numpy()
        k += bs
        print(f"\r  caching {k}/{n}", end="", flush=True)
    print()
    X.flush()
    M.flush()
    np.save(cache_dir / "meta.npy", meta)
    done.write_text(str(n))
    return cache_dir


class CachedDataset(Dataset):
    def __init__(self, cache_dir, augment: bool = False):
        cache_dir = Path(cache_dir)
        self.X = np.load(cache_dir / "x.npy", mmap_mode="r")
        self.M = np.load(cache_dir / "m.npy", mmap_mode="r")
        self.meta = np.load(cache_dir / "meta.npy")
        self.augment = augment

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        x = np.array(self.X[i])
        m = np.array(self.M[i])
        if self.augment and np.random.rand() < 0.5:
            x, m = x[..., ::-1].copy(), m[:, ::-1].copy()
        return {
            "rgb": torch.from_numpy(x[:3].astype(np.float32) / 127.5 - 1.0),
            "forensic": torch.from_numpy(_dq(x[3:])),
            "mask": torch.from_numpy(m[None].astype(np.float32)),
            "label": torch.tensor(float(self.meta[i, 0])),
            "has_mask": torch.tensor(float(self.meta[i, 1])), "idx": i,
        }

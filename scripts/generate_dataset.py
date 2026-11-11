"""Generate a synthetic manipulation dataset with ground-truth masks.

    python scripts/generate_dataset.py --out data/synth --train 3000 --val 400 --test 600
    python scripts/generate_dataset.py --real-dir path/to/your/photos   # use real photos as base
"""
import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from forensics.synth import OPS, ImageSource, make_sample  # noqa: E402

_SRC = None


def _init(real_dir):
    global _SRC
    _SRC = ImageSource(real_dir)


def _work(args):
    split_dir, i, seed, manipulated, op = args
    rng = np.random.default_rng(seed)
    img, mask, op = make_sample(rng, _SRC, manipulated, op)
    name = f"{i:05d}.png"
    cv2.imwrite(str(split_dir / "images" / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    mname = ""
    if manipulated:
        mname = name
        cv2.imwrite(str(split_dir / "masks" / mname), mask)
    return name, mname, int(manipulated), op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--train", type=int, default=3000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--test", type=int, default=600)
    ap.add_argument("--real-dir", default=None, help="folder of authentic photos to use as base images")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    out = Path(a.out)
    for k, (split, n) in enumerate([("train", a.train), ("val", a.val), ("test", a.test)]):
        d = out / split
        (d / "images").mkdir(parents=True, exist_ok=True)
        (d / "masks").mkdir(parents=True, exist_ok=True)
        jobs = []
        for i in range(n):
            manipulated = i % 2 == 1
            op = OPS[(i // 2) % len(OPS)] if manipulated else None  # balanced op types
            jobs.append((d, i, a.seed * 1_000_003 + k * 100_000 + i, manipulated, op))
        with ProcessPoolExecutor(a.workers, initializer=_init, initargs=(a.real_dir,)) as ex:
            rows = list(ex.map(_work, jobs, chunksize=16))
        with open(d / "labels.csv", "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["file", "mask", "label", "op"])
            wr.writerows(rows)
        print(f"{split}: {n} images -> {d}")


if __name__ == "__main__":
    main()

"""Command-line analysis of one or more images.

    python analyze.py path/to/image.jpg [more images...] [--out reports]

Prints the verdict and saves a report figure (original, localization overlay,
mask and every spatial / frequency domain map) plus a JSON summary.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from forensics.detector import ForensicDetector  # noqa: E402


def save_report(res, path: Path, title: str):
    panels = [("Input image", res.image), ("Localization overlay", res.overlay),
              ("Predicted mask", res.mask)] + list(res.visuals.items())
    cols = 4
    rows = (len(panels) + cols - 1) // cols
    fig, ax = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    for k, a in enumerate(ax.ravel()):
        a.axis("off")
        if k < len(panels):
            name, im = panels[k]
            a.imshow(im, cmap="gray" if im.ndim == 2 else None)
            a.set_title(name, fontsize=9)
    color = {"Manipulated": "#c62828", "Authentic": "#2e7d32", "Uncertain": "#ef6c00"}[res.label]
    fig.suptitle(f"{title}  —  {res.label}  (P(manipulated) = {res.probability:.2f})", color=color, fontsize=14)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    det = ForensicDetector(a.ckpt) if a.ckpt else ForensicDetector()
    if det.model is None:
        print("! no trained checkpoint found - using training-free heuristic mode", file=sys.stderr)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in a.images:
        res = det.analyze(p)
        stem = Path(p).stem
        save_report(res, out / f"{stem}_report.png", Path(p).name)
        with open(out / f"{stem}_report.json", "w") as f:
            json.dump(res.summary(), f, indent=2)
        top = ", ".join(f"{e['feature']}={e['contrast']:.2f}" for e in res.evidence[:3])
        print(f"{Path(p).name:40s} {res.label:12s} P={res.probability:.3f}  area={100 * res.manipulated_area:.1f}%  "
              f"top evidence: {top}")
    print(f"reports saved to {out.resolve()}")


if __name__ == "__main__":
    main()

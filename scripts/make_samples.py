"""Create a few demo images (with ground-truth masks) in ./samples for the app."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from forensics.synth import OPS, ImageSource, make_sample  # noqa: E402


def main(real_dir=None):
    out = ROOT / "samples"
    out.mkdir(exist_ok=True)
    src = ImageSource(real_dir)
    rng = np.random.default_rng(2026)
    jobs = [("authentic_1", None), ("authentic_2", None)] + [(op, op) for op in OPS]
    for name, op in jobs:
        img, mask, _ = make_sample(rng, src, op is not None, op, size_range=(384, 512))
        cv2.imwrite(str(out / f"{name}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if op is not None:
            cv2.imwrite(str(out / f"{name}_mask.png"), mask)
    print(f"samples written to {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

"""Collect every results/*_metrics.json into a comparison table + bar chart.

    python scripts/compare.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

NAMES = {"ela_heuristic": "ELA heuristic", "handcrafted_rf": "Hand-crafted + RF",
         "rgb": "CNN (RGB only)", "forensic": "CNN (forensic only)", "hybrid": "Hybrid CNN (proposed)"}
ORDER = list(NAMES)


def main():
    res_dir = ROOT / "results"
    rows = {}
    for p in res_dir.glob("*_metrics.json"):
        key = p.stem.replace("_metrics", "")
        rows[key] = json.loads(p.read_text())
    keys = [k for k in ORDER if k in rows] + sorted(k for k in rows if k not in ORDER)
    if not keys:
        print("no results yet - run evaluate.py / baseline.py first")
        return
    cols = ["accuracy", "precision", "recall", "f1", "auc"]
    lines = ["| Method | Accuracy | Precision | Recall | F1 | AUC | Pixel F1 | Pixel IoU |",
             "|---|---|---|---|---|---|---|---|"]
    for k in keys:
        im, px = rows[k]["image_level"], rows[k].get("pixel_level", {})
        pf = f"{px['pixel_f1']:.3f}" if "pixel_f1" in px else "–"
        pi = f"{px['pixel_iou']:.3f}" if "pixel_iou" in px else "–"
        lines.append(f"| {NAMES.get(k, k)} | " + " | ".join(f"{im[c]:.3f}" for c in cols) + f" | {pf} | {pi} |")
    table = "\n".join(lines)
    (res_dir / "comparison.md").write_text(table + "\n", encoding="utf-8")
    print(table)

    metrics = cols + ["pixel_f1"]
    x = np.arange(len(metrics))
    wbar = 0.8 / len(keys)
    plt.figure(figsize=(10, 4.2))
    for i, k in enumerate(keys):
        vals = [rows[k]["image_level"].get(m, 0) if m != "pixel_f1" else rows[k].get("pixel_level", {}).get("pixel_f1", 0)
                for m in metrics]
        plt.bar(x + i * wbar - 0.4 + wbar / 2, vals, wbar, label=NAMES.get(k, k))
    plt.xticks(x, ["Accuracy", "Precision", "Recall", "F1", "AUC", "Pixel F1"])
    plt.ylim(0, 1.05)
    plt.ylabel("score")
    plt.title("Method comparison on the test set")
    plt.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.savefig(res_dir / "comparison.png", dpi=130)
    print(f"saved {res_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()

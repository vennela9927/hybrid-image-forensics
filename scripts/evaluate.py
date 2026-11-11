"""Evaluate a trained model on the test split.

    python scripts/evaluate.py --ckpt checkpoints/hybrid.pt --data data/synth
Writes results/<name>_metrics.json, ROC curve, confusion matrix, per-manipulation
breakdown and a qualitative grid of predictions.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from forensics.dataset import ForensicDataset, make_index  # noqa: E402
from forensics.detector import ForensicDetector  # noqa: E402
from forensics.metrics import best_f1_threshold, classification_metrics, pixel_metrics, roc_curve  # noqa: E402


def plot_roc(y, p, path, title):
    fpr, tpr = roc_curve(y, p)
    plt.figure(figsize=(4.5, 4.5))
    plt.plot(fpr, tpr, lw=2, label="model")
    plt.plot([0, 1], [0, 1], "--", c="gray", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def plot_confusion(m, path, title):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    plt.figure(figsize=(4, 3.6))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    plt.xticks([0, 1], ["Authentic", "Manipulated"])
    plt.yticks([0, 1], ["Authentic", "Manipulated"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "checkpoints" / "hybrid.pt"))
    ap.add_argument("--data", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    name = Path(a.ckpt).stem
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    det = ForensicDetector(a.ckpt, device=a.device)
    assert det.model is not None, f"checkpoint not found: {a.ckpt}"
    rows = make_index(a.data, a.split)
    loader = DataLoader(ForensicDataset(rows), batch_size=16, num_workers=a.workers)

    probs, labels, ops, pix, pix_best, examples = [], [], [], [], [], []
    with torch.no_grad():
        for b in loader:
            for j in range(len(b["label"])):
                p, m = det.predict_tensors(b["rgb"][j].numpy(), b["forensic"][j].numpy())
                i = int(b["idx"][j])
                probs.append(p)
                labels.append(int(b["label"][j]))
                ops.append(rows[i]["op"])
                if labels[-1] == 1 and b["has_mask"][j] > 0.5:
                    gt = b["mask"][j, 0].numpy()
                    pix.append(pixel_metrics(gt, m))
                    pix_best.append(best_f1_threshold(gt, m))
                if len(examples) < 16 and (i % 7 == 0):
                    examples.append((b["rgb"][j].numpy(), b["mask"][j, 0].numpy(), m, p, rows[i]["op"]))

    probs, labels, ops = np.array(probs), np.array(labels), np.array(ops)
    cm = classification_metrics(labels, probs)
    unc = ((probs > det.low) & (probs < det.high)).mean()
    res = {"model": name, "streams": det.meta.get("streams"), "n_images": int(len(labels)),
           "image_level": cm, "uncertain_rate": float(unc),
           "pixel_level": {k: float(np.mean([d[k] for d in pix])) for k in pix[0]} if pix else {},
           "per_manipulation": {}}
    if pix_best:
        res["pixel_level"]["pixel_f1_best_threshold"] = float(np.mean(pix_best))
    auth = probs[labels == 0]
    for op in sorted(set(ops) - {"authentic"}):
        sel = ops == op
        y = np.r_[np.zeros(len(auth)), np.ones(sel.sum())]
        s = np.r_[auth, probs[sel]]
        res["per_manipulation"][op] = {"n": int(sel.sum()), "detection_rate": float((probs[sel] >= 0.5).mean()),
                                       "auc_vs_authentic": classification_metrics(y, s)["auc"]}
    with open(out / f"{name}_metrics.json", "w") as f:
        json.dump(res, f, indent=2)
    plot_roc(labels, probs, out / f"{name}_roc.png", f"ROC - {name} (AUC {cm['auc']:.3f})")
    plot_confusion(cm, out / f"{name}_confusion.png", f"Confusion matrix - {name}")

    # qualitative grid
    if examples:
        n = len(examples)
        fig, ax = plt.subplots(n, 3, figsize=(7.5, 2.5 * n))
        for k, (rgb, gt, m, p, op) in enumerate(examples):
            ax[k, 0].imshow(((rgb.transpose(1, 2, 0) + 1) * 127.5).astype(np.uint8))
            ax[k, 0].set_title(f"{op}", fontsize=8)
            ax[k, 1].imshow(gt, cmap="gray", vmin=0, vmax=1)
            ax[k, 1].set_title("ground truth", fontsize=8)
            ax[k, 2].imshow(m, cmap="jet", vmin=0, vmax=1)
            ax[k, 2].set_title(f"prediction  P(manip)={p:.2f}", fontsize=8)
            for c in range(3):
                ax[k, c].axis("off")
        plt.tight_layout()
        plt.savefig(out / f"{name}_examples.png", dpi=90)
        plt.close()

    print(json.dumps({"image_level": {k: round(v, 4) for k, v in cm.items() if isinstance(v, float)},
                      "pixel_level": res["pixel_level"], "uncertain_rate": unc}, indent=2))
    print("per manipulation:", json.dumps(res["per_manipulation"], indent=1))


if __name__ == "__main__":
    main()

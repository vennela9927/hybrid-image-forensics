"""Classical baselines for comparison with the hybrid CNN.

1. ELA-only heuristic          (single spatial cue, no learning)
2. Hand-crafted multi-domain features + Random Forest  (classic ML)
   localization = training-free fusion of the forensic maps

    python scripts/baseline.py --data data/synth
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from forensics.dataset import make_index  # noqa: E402
from forensics.features import forensic_maps, global_features, heuristic_heatmap  # noqa: E402
from forensics.metrics import best_f1_threshold, classification_metrics, pixel_metrics  # noqa: E402
from forensics.preprocess import limit_size, load_image, robust_normalize  # noqa: E402


def _feat(row):
    img = limit_size(load_image(row["image"]))
    maps = forensic_maps(img)
    g = global_features(img, maps)
    ela = robust_normalize(maps["ela"])
    ela_score = float(np.percentile(ela, 99) - np.median(ela))
    out = {"g": g, "ela": ela_score}
    if row["label"] == 1 and row["mask"] is not None:
        h, w = 128, 128
        heat = heuristic_heatmap(maps, (h, w))
        gt = cv2.resize(cv2.imread(str(row["mask"]), cv2.IMREAD_GRAYSCALE), (w, h), interpolation=cv2.INTER_NEAREST) / 255.0
        thr = float(np.percentile(heat, 85))
        out["pix"] = pixel_metrics(gt, heat, thr)
        out["pix_best"] = best_f1_threshold(gt, heat)
    return out


def featurize(rows, workers):
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_feat, rows, chunksize=8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=3000)
    a = ap.parse_args()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tr = make_index(a.data, "train")[: a.max_train]
    te = make_index(a.data, "test")
    print(f"extracting features: train {len(tr)}  test {len(te)}")
    ftr, fte = featurize(tr, a.workers), featurize(te, a.workers)
    Xtr, ytr = np.stack([f["g"] for f in ftr]), np.array([r["label"] for r in tr])
    Xte, yte = np.stack([f["g"] for f in fte]), np.array([r["label"] for r in te])

    clf = make_pipeline(StandardScaler(), RandomForestClassifier(400, min_samples_leaf=2, n_jobs=-1, random_state=0))
    clf.fit(Xtr, ytr)
    p_rf = clf.predict_proba(Xte)[:, 1]

    # ELA-only: choose the threshold on the training set
    ela_tr = np.array([f["ela"] for f in ftr])
    ela_te = np.array([f["ela"] for f in fte])
    ths = np.quantile(ela_tr, np.linspace(0.05, 0.95, 50))
    best_t = max(ths, key=lambda t: ((ela_tr >= t) == ytr).mean())
    s_ela = 1 / (1 + np.exp(-(ela_te - best_t) * 4))

    pix = [f["pix"] for f in fte if "pix" in f]
    pixel = {k: float(np.mean([d[k] for d in pix])) for k in pix[0]}
    pixel["pixel_f1_best_threshold"] = float(np.mean([f["pix_best"] for f in fte if "pix_best" in f]))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {
        "ela_heuristic": {"model": "ELA heuristic (spatial only, no learning)",
                          "image_level": classification_metrics(yte, s_ela), "pixel_level": {}},
        "handcrafted_rf": {"model": "Hand-crafted multi-domain features + Random Forest",
                           "image_level": classification_metrics(yte, p_rf), "pixel_level": pixel},
    }
    for k, v in results.items():
        with open(out / f"{k}_metrics.json", "w") as f:
            json.dump(v, f, indent=2)
        im = v["image_level"]
        print(f"{k:16s} acc {im['accuracy']:.3f}  f1 {im['f1']:.3f}  auc {im['auc']:.3f}  {v['pixel_level']}")


if __name__ == "__main__":
    main()

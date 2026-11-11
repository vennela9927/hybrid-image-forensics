"""Evaluation metrics (image-level detection + pixel-level localization)."""
from __future__ import annotations

import numpy as np


def roc_auc(y_true, scores) -> float:
    """Area under the ROC curve (Mann-Whitney U), no sklearn needed."""
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):  # average ranks for ties
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(y_true, scores, n=200):
    y = np.asarray(y_true).astype(bool)
    s = np.asarray(scores)
    th = np.unique(np.concatenate([[-np.inf, np.inf], np.quantile(s, np.linspace(0, 1, n))]))[::-1]
    tpr = [(s[y] >= t).mean() if y.any() else 0 for t in th]
    fpr = [(s[~y] >= t).mean() if (~y).any() else 0 for t in th]
    return np.array(fpr), np.array(tpr)


def classification_metrics(y_true, prob, threshold=0.5) -> dict:
    y = np.asarray(y_true).astype(int)
    p = (np.asarray(prob) >= threshold).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"accuracy": (tp + tn) / max(len(y), 1), "precision": prec, "recall": rec, "f1": f1,
            "auc": roc_auc(y, prob), "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def pixel_metrics(gt: np.ndarray, pred_prob: np.ndarray, threshold=0.5) -> dict:
    """Per-image pixel-level F1 / IoU / MCC for a manipulated image."""
    g = gt > 0.5
    p = pred_prob >= threshold
    tp = float((g & p).sum())
    fp = float((~g & p).sum())
    fn = float((g & ~p).sum())
    tn = float((~g & ~p).sum())
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else 1.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else 0.0
    return {"pixel_f1": f1, "pixel_iou": iou, "pixel_mcc": mcc}


def best_f1_threshold(gt: np.ndarray, pred_prob: np.ndarray) -> float:
    """Oracle-threshold pixel F1 (commonly reported alongside fixed-0.5)."""
    best = 0.0
    for t in np.linspace(0.05, 0.95, 19):
        best = max(best, pixel_metrics(gt, pred_prob, t)["pixel_f1"])
    return best

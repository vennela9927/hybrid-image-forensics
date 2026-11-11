"""End-to-end inference: image in -> verdict + localization map + evidence."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from . import frequency as fq
from . import spatial as sp
from .features import FORENSIC_CHANNELS, heuristic_heatmap, heuristic_score, model_inputs
from .model import HybridForensicNet
from .preprocess import MODEL_SIZE, limit_size, load_image, robust_normalize

DEFAULT_CKPT = Path(__file__).resolve().parent.parent / "checkpoints" / "hybrid.pt"

DOMAIN_LABELS = {
    "ela": "Error Level Analysis (spatial / compression)",
    "noise": "SRM noise level (spatial)",
    "fft_hp": "FFT high-frequency energy (frequency)",
    "dct_ac": "DCT 8x8 AC energy (frequency / JPEG)",
    "blockiness": "JPEG blocking-grid strength (frequency / JPEG)",
    "wavelet_noise": "Wavelet HH noise estimate (frequency)",
}


@dataclass
class ForensicResult:
    label: str                      # "Authentic" | "Manipulated" | "Uncertain"
    probability: float              # P(manipulated)
    heatmap: np.ndarray             # float32 (H, W) in [0,1], original size
    mask: np.ndarray                # uint8 (H, W) {0,255}
    overlay: np.ndarray             # RGB uint8 visualisation
    image: np.ndarray               # analysed RGB image
    evidence: list = field(default_factory=list)   # per-domain contrast scores
    visuals: dict = field(default_factory=dict)    # domain maps for display
    manipulated_area: float = 0.0   # fraction of image flagged
    mode: str = "cnn"
    seconds: float = 0.0

    def summary(self) -> dict:
        return {"label": self.label, "probability_manipulated": round(self.probability, 4),
                "manipulated_area_percent": round(100 * self.manipulated_area, 2),
                "mode": self.mode, "seconds": round(self.seconds, 3),
                "evidence": self.evidence}


def colorize(m: np.ndarray, cmap=cv2.COLORMAP_JET) -> np.ndarray:
    m = m.astype(np.float32)
    lo, hi = np.percentile(m, 1), np.percentile(m, 99.5)
    u8 = np.clip((m - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(u8, cmap), cv2.COLOR_BGR2RGB)


def make_overlay(img: np.ndarray, heat: np.ndarray, mask: np.ndarray, alpha=0.5) -> np.ndarray:
    hm = cv2.cvtColor(cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    a = (alpha * np.clip(heat * 1.4, 0, 1))[..., None]
    out = (img.astype(np.float32) * (1 - a) + hm.astype(np.float32) * a).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 255, 255), max(1, img.shape[0] // 300))
    return out


class ForensicDetector:
    def __init__(self, checkpoint: str | Path | None = DEFAULT_CKPT, device: str = "cpu",
                 low: float = 0.35, high: float = 0.65, mask_threshold: float = 0.5):
        self.device = torch.device(device)
        self.low, self.high, self.mask_threshold = low, high, mask_threshold
        self.model = None
        self.meta = {}
        if checkpoint and Path(checkpoint).exists():
            ck = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.meta = ck.get("meta", {})
            self.model = HybridForensicNet(streams=self.meta.get("streams", "hybrid"),
                                           widths=tuple(self.meta.get("widths", (24, 48, 96, 160))))
            self.model.load_state_dict(ck["state_dict"])
            self.model.eval().to(self.device)
            torch.set_num_threads(max(1, torch.get_num_threads()))

    @property
    def mode(self) -> str:
        return "cnn" if self.model is not None else "heuristic"

    def decide(self, p: float) -> str:
        if p >= self.high:
            return "Manipulated"
        if p <= self.low:
            return "Authentic"
        return "Uncertain"

    @torch.no_grad()
    def predict_tensors(self, rgb: np.ndarray, foren: np.ndarray):
        r = torch.from_numpy(rgb)[None].to(self.device)
        f = torch.from_numpy(foren)[None].to(self.device)
        logit, mlog = self.model(r, f)
        # test-time augmentation: horizontal flip
        logit2, mlog2 = self.model(torch.flip(r, [3]), torch.flip(f, [3]))
        p = 0.5 * (torch.sigmoid(logit) + torch.sigmoid(logit2)).item()
        m = 0.5 * (torch.sigmoid(mlog) + torch.flip(torch.sigmoid(mlog2), [3]))
        return p, m[0, 0].cpu().numpy()

    def analyze(self, src, with_visuals: bool = True) -> ForensicResult:
        t0 = time.time()
        img = limit_size(load_image(src))
        h, w = img.shape[:2]
        rgb, foren, maps = model_inputs(img, MODEL_SIZE)
        if self.model is not None:
            p, m = self.predict_tensors(rgb, foren)
            heat = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        else:  # training-free fallback
            # weak training-free fallback (val AUC ~0.57): calibrated logistic on the
            # fused inconsistency score, so it mostly reports "Uncertain"
            heat = heuristic_heatmap(maps, (h, w))
            p = float(1 / (1 + np.exp(-(heuristic_score(maps) - 0.6) * 4)))
        label = self.decide(p)
        mask = (heat >= self.mask_threshold).astype(np.uint8) * 255
        if label == "Authentic":
            mask[:] = 0
        res = ForensicResult(label=label, probability=float(p), heatmap=heat.astype(np.float32), mask=mask,
                             overlay=make_overlay(img, heat if label != "Authentic" else heat * 0.3, mask),
                             image=img, mode=self.mode, manipulated_area=float((mask > 0).mean()))
        res.evidence = self.evidence(maps, heat, (h, w))
        if with_visuals:
            res.visuals = self.visuals(img, maps)
        res.seconds = time.time() - t0
        return res

    @staticmethod
    def evidence(maps: dict, heat: np.ndarray, shape) -> list:
        """How strongly each domain separates the suspicious region from the
        rest of the image (|mean_in - mean_out| in robust-z units)."""
        h, w = shape
        region = heat >= max(0.5, float(np.percentile(heat, 95)))
        if region.mean() < 0.002 or region.mean() > 0.9:
            region = heat >= np.percentile(heat, 90)
        out = []
        for name, domain in FORENSIC_CHANNELS:
            m = robust_normalize(cv2.resize(maps[name].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA))
            inside, outside = m[region], m[~region]
            score = float(abs(inside.mean() - outside.mean()) / (outside.std() + 1e-3)) if inside.size and outside.size else 0.0
            out.append({"feature": name, "domain": domain, "description": DOMAIN_LABELS[name],
                        "contrast": round(score, 3)})
        return sorted(out, key=lambda d: -d["contrast"])

    @staticmethod
    def visuals(img: np.ndarray, maps: dict) -> dict:
        lnoise = maps["noise"]
        return {
            "Spatial: Error Level Analysis": sp.ela_visual(img),
            "Spatial: Noise residual (SRM)": colorize(np.abs(sp.srm_residuals(img)[1]), cv2.COLORMAP_BONE),
            "Spatial: Local noise level": colorize(lnoise, cv2.COLORMAP_INFERNO),
            "Spatial: Edges (Sobel)": colorize(sp.edge_map(img), cv2.COLORMAP_BONE),
            "Frequency: FFT spectrum": colorize(fq.fft_spectrum(img), cv2.COLORMAP_JET),
            "Frequency: FFT high-pass energy": colorize(maps["fft_hp"], cv2.COLORMAP_INFERNO),
            "Frequency: DCT 8x8 AC energy": colorize(maps["dct_ac"], cv2.COLORMAP_VIRIDIS),
            "Frequency: JPEG blockiness": colorize(maps["blockiness"], cv2.COLORMAP_VIRIDIS),
            "Frequency: Wavelet (HH) noise": colorize(maps["wavelet_noise"], cv2.COLORMAP_INFERNO),
        }

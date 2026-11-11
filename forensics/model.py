"""Step 4-6: dual-stream CNN with multi-scale feature fusion.

             RGB image ──► RGB encoder ───────────┐
                                                   ├─► fusion (per scale, SE attention)
 forensic maps + SRM + Bayar ──► forensic encoder ┘         │
                                                   ┌────────┴────────┐
                                                   ▼                 ▼
                                        U-Net decoder         classification head
                                        (pixel-level mask)    (authentic / manipulated)

`streams` selects the full hybrid model or one of the single-stream ablations
("rgb" or "forensic") used for the comparison experiments.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .features import N_FORENSIC
from .spatial import SRM_KERNELS, SRM_SCALE


class SRMConv(nn.Module):
    """Fixed (non-trainable) SRM high-pass filters applied to luminance."""

    def __init__(self):
        super().__init__()
        k = torch.from_numpy(SRM_KERNELS / SRM_SCALE[:, None, None]).float().unsqueeze(1)
        self.register_buffer("weight", k)

    def forward(self, rgb):  # rgb in [-1,1]
        y = (0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]).unsqueeze(1) * 127.5
        r = F.conv2d(F.pad(y, (2, 2, 2, 2), mode="reflect"), self.weight)
        return torch.clamp(r / 8.0, -3, 3)


class BayarConv(nn.Module):
    """Constrained convolution (Bayar & Stamm 2018): the centre weight is fixed
    to -1 and the others sum to 1, forcing the layer to learn prediction-error
    (manipulation-trace) filters instead of image content."""

    def __init__(self, in_ch=1, out_ch=3, k=5):
        super().__init__()
        self.k = k
        self.w = nn.Parameter(torch.randn(out_ch, in_ch, k * k - 1) * 0.5)

    def forward(self, x):
        # softmax keeps the "sum to 1" constraint numerically stable
        w = torch.softmax(self.w, dim=-1)
        c = (self.k * self.k) // 2
        minus = -torch.ones(*w.shape[:2], 1, device=w.device)
        kernel = torch.cat([w[..., :c], minus, w[..., c:]], -1)
        kernel = kernel.view(*w.shape[:2], self.k, self.k)
        x = F.pad(x * 127.5, (self.k // 2,) * 4, mode="reflect")  # back to pixel units
        return torch.clamp(F.conv2d(x, kernel) / 8.0, -3, 3)


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """Stage 0 is a full-resolution conv followed by a stride-2 conv (keeps the
    high-frequency traces but makes the network ~4x cheaper); later stages
    halve the resolution with max-pooling."""

    def __init__(self, cin, widths):
        super().__init__()
        self.stages = nn.ModuleList()
        w0 = widths[0]
        self.stages.append(nn.Sequential(
            nn.Conv2d(cin, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
            nn.Conv2d(16, w0, 3, stride=2, padding=1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(inplace=True),
            nn.Conv2d(w0, w0, 3, padding=1, bias=False), nn.BatchNorm2d(w0), nn.ReLU(inplace=True)))
        c = w0
        for w in widths[1:]:
            self.stages.append(conv_block(c, w))
            c = w

    def forward(self, x):
        feats = []
        for i, st in enumerate(self.stages):
            if i > 0:
                x = F.max_pool2d(x, 2)
            x = st(x)
            feats.append(x)
        return feats


class SEFusion(nn.Module):
    """Concatenate the two streams and re-weight channels with
    squeeze-and-excitation attention (learned per-domain importance)."""

    def __init__(self, c_in, c_out):
        super().__init__()
        self.proj = nn.Sequential(nn.Conv2d(c_in, c_out, 1, bias=False), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c_out, max(c_out // 8, 4), 1), nn.ReLU(inplace=True),
                                nn.Conv2d(max(c_out // 8, 4), c_out, 1), nn.Sigmoid())

    def forward(self, *xs):
        x = self.proj(torch.cat(xs, 1))
        return x * self.se(x)


class HybridForensicNet(nn.Module):
    def __init__(self, n_forensic: int = N_FORENSIC, widths=(24, 48, 96, 160), streams: str = "hybrid"):
        super().__init__()
        assert streams in ("hybrid", "rgb", "forensic")
        self.streams = streams
        self.widths = widths
        if streams in ("hybrid", "forensic"):
            self.srm = SRMConv()
            self.bayar = BayarConv()
            self.f_enc = Encoder(n_forensic + 3 + 3, widths)
        if streams in ("hybrid", "rgb"):
            self.r_enc = Encoder(3, widths)
        mult = 2 if streams == "hybrid" else 1
        self.fuse = nn.ModuleList([SEFusion(w * mult, w) for w in widths])
        # decoder
        self.up = nn.ModuleList()
        self.dec = nn.ModuleList()
        for i in range(len(widths) - 1, 0, -1):
            self.up.append(nn.ConvTranspose2d(widths[i], widths[i - 1], 2, stride=2))
            self.dec.append(conv_block(widths[i - 1] * 2, widths[i - 1]))
        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        # classifier: deep fused features + statistics of the predicted mask
        self.cls_head = nn.Sequential(
            nn.Linear(widths[-1] * 2 + 3, 128), nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, rgb, forensic):
        fused = []
        if self.streams == "hybrid":
            g = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            ff = self.f_enc(torch.cat([forensic, self.srm(rgb), self.bayar(g)], 1))
            rf = self.r_enc(rgb)
            fused = [fu(a, b) for fu, a, b in zip(self.fuse, rf, ff)]
        elif self.streams == "forensic":
            g = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
            ff = self.f_enc(torch.cat([forensic, self.srm(rgb), self.bayar(g)], 1))
            fused = [fu(a) for fu, a in zip(self.fuse, ff)]
        else:
            rf = self.r_enc(rgb)
            fused = [fu(a) for fu, a in zip(self.fuse, rf)]

        x = fused[-1]
        for j, (up, dec) in enumerate(zip(self.up, self.dec)):
            skip = fused[len(fused) - 2 - j]
            x = dec(torch.cat([up(x), skip], 1))
        mask_logits = F.interpolate(self.mask_head(x), size=rgb.shape[-2:], mode="bilinear", align_corners=False)

        deep = fused[-1]
        mp = torch.sigmoid(mask_logits).flatten(1)
        topk = torch.topk(mp, k=max(1, mp.shape[1] // 50), dim=1).values.mean(1, keepdim=True)
        mstats = torch.cat([mp.mean(1, keepdim=True), topk, (mp > 0.5).float().mean(1, keepdim=True)], 1)
        g = torch.cat([F.adaptive_avg_pool2d(deep, 1).flatten(1), F.adaptive_max_pool2d(deep, 1).flatten(1), mstats], 1)
        cls_logit = self.cls_head(g).squeeze(1)
        return cls_logit, mask_logits


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    num = 2 * (p * target).sum((1, 2, 3)) + eps
    den = p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + eps
    return 1 - (num / den)


def forensic_loss(cls_logit, mask_logits, label, mask, has_mask=None, w_mask: float = 1.0):
    """Image-level BCE + pixel-level (BCE + Dice). Dice is only applied to
    manipulated images (it is undefined for empty masks); images whose
    ground-truth mask is unknown (`has_mask`=0) get no pixel loss."""
    if has_mask is None:
        has_mask = torch.ones_like(label)
    l_cls = F.binary_cross_entropy_with_logits(cls_logit, label)
    pos_w = torch.tensor(3.0, device=mask.device)
    bce = F.binary_cross_entropy_with_logits(mask_logits, mask, pos_weight=pos_w, reduction="none").mean((1, 2, 3))
    l_bce = (bce * has_mask).sum() / has_mask.sum().clamp(min=1)
    d = dice_loss(mask_logits, mask)
    w = label * has_mask
    l_dice = (d * w).sum() / w.sum().clamp(min=1)
    return l_cls + w_mask * (l_bce + l_dice), {"cls": l_cls.item(), "bce": l_bce.item(), "dice": l_dice.item()}


def count_params(model) -> int:
    return int(sum(np.prod(p.shape) for p in model.parameters() if p.requires_grad))

"""Train the hybrid (or ablation) forensic network.

    python scripts/train.py --data data/synth --epochs 15 --streams hybrid
    python scripts/train.py --data data/synth --streams rgb      --out checkpoints/rgb_only.pt
    python scripts/train.py --data data/synth --streams forensic --out checkpoints/forensic_only.pt
    python scripts/train.py --data path/to/CASIA2 --init checkpoints/hybrid.pt   # fine-tune on real data
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from forensics.dataset import CachedDataset, ForensicDataset, build_cache, make_index  # noqa: E402
from forensics.metrics import classification_metrics, pixel_metrics  # noqa: E402
from forensics.model import HybridForensicNet, count_params, forensic_loss  # noqa: E402


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    probs, labels, pf1 = [], [], []
    for b in loader:
        logit, mlog = model(b["rgb"].to(device), b["forensic"].to(device))
        p = torch.sigmoid(logit).cpu().numpy()
        mp = torch.sigmoid(mlog).cpu().numpy()
        probs += list(p)
        labels += list(b["label"].numpy())
        for j in range(len(p)):
            if b["label"][j] > 0.5 and b["has_mask"][j] > 0.5:
                pf1.append(pixel_metrics(b["mask"][j, 0].numpy(), mp[j, 0])["pixel_f1"])
    m = classification_metrics(labels, probs)
    m["pixel_f1"] = float(np.mean(pf1)) if pf1 else float("nan")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--streams", default="hybrid", choices=["hybrid", "rgb", "forensic"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-train", type=int, default=0, help="limit #train images (0 = all)")
    ap.add_argument("--init", default=None, help="checkpoint to fine-tune from")
    ap.add_argument("--out", default=None)
    ap.add_argument("--cache", default=None, help="dir for pre-computed inputs (much faster on GPU)")
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out or ROOT / "checkpoints" / f"{a.streams}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr_rows = make_index(a.data, "train")
    va_rows = make_index(a.data, "val")
    if a.max_train:
        tr_rows = tr_rows[: a.max_train]
    pin = device.type == "cuda"
    if a.cache:
        print("building feature cache (one-off)...", flush=True)
        tr_ds = CachedDataset(build_cache(tr_rows, Path(a.cache) / "train", workers=a.workers), augment=True)
        va_ds = CachedDataset(build_cache(va_rows, Path(a.cache) / "val", workers=a.workers))
    else:
        tr_ds, va_ds = ForensicDataset(tr_rows, augment=True), ForensicDataset(va_rows)
    tr = DataLoader(tr_ds, batch_size=a.batch, shuffle=True, num_workers=a.workers, pin_memory=pin,
                    persistent_workers=a.workers > 0, drop_last=True)
    va = DataLoader(va_ds, batch_size=a.batch, num_workers=a.workers, pin_memory=pin,
                    persistent_workers=a.workers > 0)

    model = HybridForensicNet(streams=a.streams).to(device)
    if a.init:
        model.load_state_dict(torch.load(a.init, map_location=device, weights_only=False)["state_dict"])
    print(f"model: {a.streams}  params: {count_params(model):,}  train: {len(tr_rows)}  val: {len(va_rows)}  device: {device}")
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=a.epochs * len(tr), pct_start=0.15)

    use_amp = bool(a.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    history, best = [], -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for step, b in enumerate(tr):
            with torch.autocast(device.type, enabled=use_amp):
                logit, mlog = model(b["rgb"].to(device, non_blocking=True), b["forensic"].to(device, non_blocking=True))
            loss, parts = forensic_loss(logit.float(), mlog.float(), b["label"].to(device), b["mask"].to(device),
                                        b["has_mask"].to(device))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            if step % 20 == 0:
                print(f"\r  ep {ep} step {step}/{len(tr)} loss {loss.item():.4f}", end="", flush=True)
            tot += loss.item() * len(b["label"])
            n += len(b["label"])
        vm = validate(model, va, device)
        score = vm["auc"] * 0.5 + (vm["pixel_f1"] if vm["pixel_f1"] == vm["pixel_f1"] else 0) * 0.5
        rec = {"epoch": ep, "train_loss": tot / n, **{f"val_{k}": v for k, v in vm.items()},
               "seconds": time.time() - t0}
        history.append(rec)
        print(f"\rep {ep:02d} loss {tot / n:.4f} | val acc {vm['accuracy']:.3f} auc {vm['auc']:.3f} "
              f"f1 {vm['f1']:.3f} pixF1 {vm['pixel_f1']:.3f} | {time.time() - t0:.0f}s", flush=True)
        if score > best:
            best = score
            torch.save({"state_dict": model.state_dict(),
                        "meta": {"streams": a.streams, "widths": list(model.widths), "epoch": ep,
                                 "val": vm, "data": str(a.data)}}, out)
            print(f"  saved best -> {out}", flush=True)
    with open(out.with_suffix(".history.json"), "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()

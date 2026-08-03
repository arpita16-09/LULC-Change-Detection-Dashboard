"""Evaluate OSCD-enhanced ResUNet on the enhanced test split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train_oscd_enhanced import ResUNet, CLASS_NAMES, NUM_CLASSES, IGNORE_INDEX, DEVICE

SCRIPT_DIR = Path(__file__).resolve().parent


@torch.no_grad()
def predict(model, img_hwc, tta=True):
    img = torch.from_numpy(img_hwc.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(DEVICE)
    _, _, h, w = img.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    def fwd(x):
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        return model(x)[:, :, :h, :w]

    if not tta:
        return fwd(img).argmax(1)[0].cpu().numpy().astype(np.uint8)

    probs = F.softmax(fwd(img), dim=1)
    probs = probs + torch.flip(F.softmax(fwd(torch.flip(img, [3])), 1), [3])
    probs = probs + torch.flip(F.softmax(fwd(torch.flip(img, [2])), 1), [2])
    probs = probs + torch.flip(F.softmax(fwd(torch.flip(img, [2, 3])), 1), [2, 3])
    return (probs / 4).argmax(1)[0].cpu().numpy().astype(np.uint8)


def update_conf(conf, pred, target):
    mask = (target >= 0) & (target < NUM_CLASSES)
    idx = NUM_CLASSES * target[mask].astype(np.int64) + pred[mask].astype(np.int64)
    conf += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics(conf):
    ious = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        d = tp + fp + fn
        ious.append(tp / d if d > 0 else float("nan"))
    valid = [x for x in ious if not np.isnan(x)]
    acc = conf.diagonal().sum() / conf.sum() if conf.sum() else 0.0
    return float(np.mean(valid)) if valid else 0.0, float(acc), ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=str(SCRIPT_DIR / "train" / "resunet_oscd_enhanced_best.pt"),
    )
    parser.add_argument("--test-root", default=str(SCRIPT_DIR / "enhanced" / "test"))
    parser.add_argument(
        "--legacy-masks",
        default="",
        help="Optional path to old masks dir for comparison against original pseudo-labels",
    )
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "test_eval_oscd_enhanced"))
    parser.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    model = ResUNet(in_channels=ckpt.get("in_channels", 10)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    test_root = Path(args.test_root)
    img_dir = test_root / "images_multiband_npy"
    mask_dir = Path(args.legacy_masks) if args.legacy_masks else test_root / "masks"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "pred_masks").mkdir(exist_ok=True)

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    rows = []
    samples = sorted(img_dir.glob("*_t*.npy"))
    print(f"Device={DEVICE} ckpt_epoch={ckpt.get('epoch')} val_miou={ckpt.get('val_miou')}")
    print(f"Images={img_dir} masks={mask_dir} n={len(samples)} tta={args.tta}")

    for npy_path in samples:
        sid = npy_path.stem
        mask_path = mask_dir / f"{sid}_mask.png"
        if not mask_path.exists():
            print("skip missing mask", sid)
            continue
        img = np.load(npy_path).astype(np.float32)
        target = np.array(Image.open(mask_path))
        if target.ndim == 3:
            target = target[..., 0]
        pred = predict(model, img, tta=args.tta)
        Image.fromarray(pred).save(out_dir / "pred_masks" / f"{sid}_pred.png")

        conf_i = np.zeros_like(conf)
        update_conf(conf_i, pred, target)
        update_conf(conf, pred, target)
        miou_i, acc_i, _ = metrics(conf_i)
        rows.append({"sample_id": sid, "pixel_acc": acc_i, "miou": miou_i})
        print(f"{sid:20s} acc={acc_i:.3f} mIoU={miou_i:.3f}")

    miou, acc, ious = metrics(conf)
    print("-" * 50)
    print(f"OVERALL acc={acc:.3f} mIoU={miou:.3f}")
    for n, iou in zip(CLASS_NAMES, ious):
        print(f"  {n:12s} IoU={iou:.3f}" if not np.isnan(iou) else f"  {n:12s} IoU=n/a")

    with open(out_dir / "per_image_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["sample_id", "pixel_acc", "miou"])
        w.writeheader()
        w.writerows(rows)
    summary = {
        "checkpoint": args.checkpoint,
        "epoch": ckpt.get("epoch"),
        "trained_val_miou": ckpt.get("val_miou"),
        "pixel_acc": acc,
        "mean_iou": miou,
        "per_class_iou": {n: (None if np.isnan(x) else float(x)) for n, x in zip(CLASS_NAMES, ious)},
        "mask_dir": str(mask_dir),
        "tta": args.tta,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with open(out_dir / "summary.txt", "w") as f:
        f.write(json.dumps(summary, indent=2))
        f.write("\n")


if __name__ == "__main__":
    main()

"""
Evaluate a trained ResUNet checkpoint on the held-out test AOIs.

Default paths match this repo layout:
  landcover_train/train/resunet_best.pt
  landcover_train/test/test/{images_multiband_npy,masks}/

Usage:
  python evaluate_resunet.py
  python evaluate_resunet.py --checkpoint path/to/resunet_best.pt --test-root path/to/test
"""

from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

CLASS_NAMES = ["Other", "Water", "Urban", "Forest", "Agriculture", "Barren"]
NUM_CLASSES = 6


def _load_train_module(script_dir: str):
    path = os.path.join(script_dir, "train", "train_resunet.py")
    spec = importlib.util.spec_from_file_location("train_resunet", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def list_test_samples(img_dir: str, mask_dir: str):
    samples = []
    for npy_path in sorted(glob.glob(os.path.join(img_dir, "*_t*.npy"))):
        sample_id = os.path.splitext(os.path.basename(npy_path))[0]
        mask_path = os.path.join(mask_dir, f"{sample_id}_mask.png")
        if os.path.exists(mask_path):
            samples.append((sample_id, npy_path, mask_path))
    return samples


@torch.no_grad()
def _forward_padded(model, img_bchw: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """Return logits cropped back to original HxW."""
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16
    x = img_bchw
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    logits = model(x)
    return logits[:, :, :h, :w]


@torch.no_grad()
def predict_full(model, img_chw: torch.Tensor, device: torch.device, tta: bool = False) -> np.ndarray:
    """Run the model on a full (C,H,W) image, padding to a multiple of 16.

    If tta=True, average softmax probs over identity + H/V/HV flips.
    """
    _, h, w = img_chw.shape
    x0 = img_chw.unsqueeze(0).to(device)
    if not tta:
        logits = _forward_padded(model, x0, h, w)
        return logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

    probs = F.softmax(_forward_padded(model, x0, h, w), dim=1)
    # horizontal flip
    logits = _forward_padded(model, torch.flip(x0, dims=[3]), h, w)
    probs = probs + torch.flip(F.softmax(logits, dim=1), dims=[3])
    # vertical flip
    logits = _forward_padded(model, torch.flip(x0, dims=[2]), h, w)
    probs = probs + torch.flip(F.softmax(logits, dim=1), dims=[2])
    # hv flip
    logits = _forward_padded(model, torch.flip(x0, dims=[2, 3]), h, w)
    probs = probs + torch.flip(F.softmax(logits, dim=1), dims=[2, 3])
    probs = probs / 4.0
    return probs[0].argmax(dim=0).cpu().numpy().astype(np.uint8)


def confusion_update(conf: np.ndarray, pred: np.ndarray, target: np.ndarray):
    mask = (target >= 0) & (target < NUM_CLASSES)
    idx = NUM_CLASSES * target[mask].astype(np.int64) + pred[mask].astype(np.int64)
    conf += np.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def metrics_from_confusion(conf: np.ndarray):
    ious, precisions, recalls, f1s = [], [], [], []
    for c in range(NUM_CLASSES):
        tp = conf[c, c]
        fp = conf[:, c].sum() - tp
        fn = conf[c, :].sum() - tp
        denom_iou = tp + fp + fn
        ious.append(tp / denom_iou if denom_iou > 0 else float("nan"))
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        precisions.append(prec)
        recalls.append(rec)
        if np.isnan(prec) or np.isnan(rec) or (prec + rec) == 0:
            f1s.append(float("nan"))
        else:
            f1s.append(2 * prec * rec / (prec + rec))
    valid_ious = [x for x in ious if not np.isnan(x)]
    miou = float(np.mean(valid_ious)) if valid_ious else 0.0
    pixel_acc = float(conf.diagonal().sum() / conf.sum()) if conf.sum() else 0.0
    return {
        "miou": miou,
        "pixel_acc": pixel_acc,
        "per_class_iou": ious,
        "per_class_precision": precisions,
        "per_class_recall": recalls,
        "per_class_f1": f1s,
    }


def colorize_mask(mask: np.ndarray) -> Image.Image:
    palette = {
        0: (60, 60, 60),
        1: (30, 144, 255),
        2: (220, 20, 60),
        3: (34, 139, 34),
        4: (173, 255, 47),
        5: (210, 180, 140),
    }
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, color in palette.items():
        rgb[mask == k] = color
    return Image.fromarray(rgb)


def fmt(x):
    return "n/a" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.3f}"


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="Evaluate ResUNet on landcover test set")
    parser.add_argument(
        "--checkpoint",
        default=os.path.join(script_dir, "train", "resunet_best.pt"),
    )
    parser.add_argument(
        "--test-root",
        default=os.path.join(script_dir, "test", "test"),
        help="Folder containing images_multiband_npy/ and masks/",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.join(script_dir, "test_eval_results"),
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--tta",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Flip test-time augmentation (default: on). Disable with --no-tta.",
    )
    args = parser.parse_args()

    img_dir = os.path.join(args.test_root, "images_multiband_npy")
    mask_dir = os.path.join(args.test_root, "masks")
    samples = list_test_samples(img_dir, mask_dir)
    if not samples:
        raise SystemExit(f"No test samples found under {args.test_root}")

    train_mod = _load_train_module(script_dir)
    device = torch.device(args.device)
    model = train_mod.ResUNet().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    pred_dir = os.path.join(args.out_dir, "pred_masks")
    pred_color_dir = os.path.join(args.out_dir, "pred_masks_colored")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(pred_color_dir, exist_ok=True)

    conf = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    per_image_rows = []

    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"  trained epoch={ckpt.get('epoch')} val_mIoU={ckpt.get('val_miou')}")
    print(f"Test root: {args.test_root}")
    print(f"Test samples: {len(samples)}")
    print(f"TTA: {args.tta}")
    print("-" * 60)

    for sample_id, npy_path, mask_path in samples:
        img = np.load(npy_path).astype(np.float32)  # H,W,4
        target = np.array(Image.open(mask_path))
        if target.ndim == 3:
            target = target[..., 0]
        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        pred = predict_full(model, img_t, device, tta=args.tta)

        Image.fromarray(pred).save(os.path.join(pred_dir, f"{sample_id}_pred.png"))
        colorize_mask(pred).save(os.path.join(pred_color_dir, f"{sample_id}_pred_color.png"))

        conf_img = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
        confusion_update(conf_img, pred, target)
        confusion_update(conf, pred, target)
        m_img = metrics_from_confusion(conf_img)
        per_image_rows.append(
            {
                "sample_id": sample_id,
                "height": target.shape[0],
                "width": target.shape[1],
                "pixel_acc": m_img["pixel_acc"],
                "miou": m_img["miou"],
            }
        )
        print(
            f"{sample_id:20s}  acc={m_img['pixel_acc']:.3f}  mIoU={m_img['miou']:.3f}  "
            f"shape={target.shape[0]}x{target.shape[1]}"
        )

    overall = metrics_from_confusion(conf)
    print("-" * 60)
    print(f"OVERALL pixel accuracy: {overall['pixel_acc']:.3f}")
    print(f"OVERALL mean IoU:       {overall['miou']:.3f}")
    print("Per-class metrics:")
    for i, name in enumerate(CLASS_NAMES):
        print(
            f"  {name:12s}  IoU={fmt(overall['per_class_iou'][i])}  "
            f"P={fmt(overall['per_class_precision'][i])}  "
            f"R={fmt(overall['per_class_recall'][i])}  "
            f"F1={fmt(overall['per_class_f1'][i])}"
        )

    metrics_csv = os.path.join(args.out_dir, "per_image_metrics.csv")
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "height", "width", "pixel_acc", "miou"])
        writer.writeheader()
        writer.writerows(per_image_rows)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"checkpoint: {args.checkpoint}\n")
        f.write(f"trained_epoch: {ckpt.get('epoch')}\n")
        f.write(f"trained_val_miou: {ckpt.get('val_miou')}\n")
        f.write(f"test_root: {args.test_root}\n")
        f.write(f"n_samples: {len(samples)}\n")
        f.write(f"pixel_acc: {overall['pixel_acc']:.6f}\n")
        f.write(f"mean_iou: {overall['miou']:.6f}\n")
        for i, name in enumerate(CLASS_NAMES):
            f.write(
                f"{name}: iou={fmt(overall['per_class_iou'][i])} "
                f"p={fmt(overall['per_class_precision'][i])} "
                f"r={fmt(overall['per_class_recall'][i])} "
                f"f1={fmt(overall['per_class_f1'][i])}\n"
            )
        f.write("\nconfusion_matrix (rows=gt, cols=pred):\n")
        f.write(np.array2string(conf) + "\n")

    np.save(os.path.join(args.out_dir, "confusion_matrix.npy"), conf)
    print(f"\nWrote results to: {args.out_dir}")
    print(f"  {metrics_csv}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()

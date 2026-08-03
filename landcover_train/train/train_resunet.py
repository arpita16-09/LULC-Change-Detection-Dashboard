"""
ResUNet training script for the OSCD-derived land-cover pseudo-label dataset.

Improvements vs. original baseline:
  - Dice + CrossEntropy loss (helps minority / confused classes)
  - Class weights from the full train set + extra boost for Urban/Barren
  - Stronger spectral / geometric augmentation
  - Multiple random crops per image each epoch
  - Weight decay, grad clipping, cosine LR, early stopping

Expects:
    train/
        images_multiband_npy/    <aoi>_t<1|2>.npy   (H, W, 4) float32  R,G,B,NIR
        masks/                   <aoi>_t<1|2>_mask.png  class ids 0-5
        train_resunet.py

Usage:
    pip install torch numpy pillow
    python train_resunet.py
"""

import os
import glob
import json
import random
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = SCRIPT_DIR
IMG_DIR = os.path.join(DATA_ROOT, "images_multiband_npy")
MASK_DIR = os.path.join(DATA_ROOT, "masks")
MANIFEST_PATH = os.path.join(os.path.dirname(SCRIPT_DIR), "split_manifest.json")

NUM_CLASSES = 6           # 0 Other, 1 Water, 2 Urban, 3 Forest, 4 Agriculture, 5 Barren
IN_CHANNELS = 4            # R, G, B, NIR
CROP_SIZE = 256
BATCH_SIZE = 4             # CPU-friendly; raise to 8 if you have a GPU
EPOCHS = 50
LR = 8e-4
WEIGHT_DECAY = 1e-4
VAL_FRACTION = 0.2
SEED = 42
CROPS_PER_IMAGE = 4        # random crops sampled per image each epoch
EARLY_STOP_PATIENCE = 12
DICE_WEIGHT = 0.5
CE_WEIGHT = 0.5
# Extra multipliers on top of inverse-frequency weights.
# Keep mild/empty by default: strong Barren boosts improved recall but hurt
# overall test accuracy/mIoU in experiments (pseudo-label noise).
CLASS_BOOST = {}
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "resunet_best.pt")
HISTORY_PATH = os.path.join(SCRIPT_DIR, "train_history.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Other", "Water", "Urban", "Forest", "Agriculture", "Barren"]


def _sanity_check_paths():
    problems = []
    if not os.path.isdir(DATA_ROOT):
        problems.append(f"DATA_ROOT does not exist: {DATA_ROOT}")
    if not os.path.isdir(IMG_DIR):
        problems.append(f"IMG_DIR does not exist: {IMG_DIR}")
    if not os.path.isdir(MASK_DIR):
        problems.append(f"MASK_DIR does not exist: {MASK_DIR}")
    if not problems:
        n_npy = len(glob.glob(os.path.join(IMG_DIR, "*.npy")))
        if n_npy == 0:
            problems.append(f"IMG_DIR exists but contains no .npy files: {IMG_DIR}")
    if problems:
        msg = "\n".join(problems)
        raise RuntimeError(
            "Dataset paths look wrong. Fix DATA_ROOT at the top of this script.\n"
            f"{msg}\n"
            f"SCRIPT_DIR is currently: {SCRIPT_DIR}"
        )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class LandCoverDataset(Dataset):
    """
    Loads (multiband image, mask) pairs for a given list of AOI names.
    When train=True, length is multiplied by CROPS_PER_IMAGE so each epoch
    sees several random crops / augmentations per sample.
    """

    def __init__(self, aoi_list, crop_size=CROP_SIZE, train=True, crops_per_image=1):
        self.crop_size = crop_size
        self.train = train
        self.crops_per_image = crops_per_image if train else 1
        self.samples = []
        for aoi in aoi_list:
            for npy_path in sorted(glob.glob(os.path.join(IMG_DIR, f"{aoi}_t*.npy"))):
                sample_id = os.path.splitext(os.path.basename(npy_path))[0]
                mask_path = os.path.join(MASK_DIR, f"{sample_id}_mask.png")
                if os.path.exists(mask_path):
                    self.samples.append((npy_path, mask_path))
        if not self.samples:
            raise RuntimeError(
                f"No samples found for AOIs: {aoi_list}\n"
                f"Looked in IMG_DIR: {IMG_DIR}\n"
                f"Looked in MASK_DIR: {MASK_DIR}"
            )

    def __len__(self):
        return len(self.samples) * self.crops_per_image

    def _pad_if_needed(self, img, mask):
        h, w = mask.shape
        cs = self.crop_size
        pad_h = max(0, cs - h)
        pad_w = max(0, cs - w)
        if pad_h > 0 or pad_w > 0:
            img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, pad_h), (0, pad_w)), mode="reflect")
        return img, mask

    def _augment_spectral(self, img):
        """Light photometric noise that preserves relative band structure."""
        # brightness / gain
        if random.random() < 0.7:
            gain = random.uniform(0.85, 1.15)
            img = img * gain
        # per-band bias
        if random.random() < 0.5:
            bias = np.random.uniform(-0.03, 0.03, size=(1, 1, img.shape[2])).astype(np.float32)
            img = img + bias
        # gaussian noise
        if random.random() < 0.5:
            noise = np.random.normal(0.0, 0.015, size=img.shape).astype(np.float32)
            img = img + noise
        # occasional channel dropout (forces NIR/RGB robustness)
        if random.random() < 0.15:
            c = random.randrange(img.shape[2])
            img = img.copy()
            img[:, :, c] = img[:, :, c].mean()
        return np.clip(img, 0.0, None)

    def __getitem__(self, idx):
        npy_path, mask_path = self.samples[idx % len(self.samples)]
        img = np.load(npy_path).astype(np.float32)              # (H, W, 4)
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask = mask.astype(np.uint8)

        img, mask = self._pad_if_needed(img, mask)
        h, w = mask.shape
        cs = self.crop_size

        if self.train:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2

        img = img[top:top + cs, left:left + cs, :]
        mask = mask[top:top + cs, left:left + cs]

        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1, :])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1, :, :])
                mask = np.ascontiguousarray(mask[::-1, :])
            k = random.randint(0, 3)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k, axes=(0, 1)))
            img = self._augment_spectral(img)

        img_t = torch.from_numpy(img.transpose(2, 0, 1).copy()).float()
        mask_t = torch.from_numpy(mask.copy()).long()
        return img_t, mask_t


def get_aoi_names():
    names = set()
    for f in glob.glob(os.path.join(IMG_DIR, "*_t*.npy")):
        base = os.path.basename(f)
        aoi = base.rsplit("_t", 1)[0]
        names.add(aoi)
    return sorted(names)


def split_aois(val_fraction=VAL_FRACTION, seed=SEED):
    """Split by AOI (not by sample) to avoid t1/t2 leakage."""
    aois = get_aoi_names()
    rng = random.Random(seed)
    rng.shuffle(aois)
    n_val = max(1, int(len(aois) * val_fraction))
    val_aois = aois[:n_val]
    train_aois = aois[n_val:]
    return train_aois, val_aois


# ---------------------------------------------------------------------------
# ResUNet model
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.res = ResBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        diff_h = skip.shape[2] - x.shape[2]
        diff_w = skip.shape[3] - x.shape[3]
        x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2])
        x = torch.cat([x, skip], dim=1)
        return self.res(x)


class ResUNet(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, base=32):
        super().__init__()
        self.stem = ResBlock(in_channels, base)
        self.enc1 = ResBlock(base, base * 2, stride=2)
        self.enc2 = ResBlock(base * 2, base * 4, stride=2)
        self.enc3 = ResBlock(base * 4, base * 8, stride=2)
        self.bottleneck = ResBlock(base * 8, base * 16, stride=2)

        self.up3 = UpBlock(base * 16, base * 8, base * 8)
        self.up2 = UpBlock(base * 8, base * 4, base * 4)
        self.up1 = UpBlock(base * 4, base * 2, base * 2)
        self.up0 = UpBlock(base * 2, base, base)

        self.head = nn.Conv2d(base, num_classes, kernel_size=1)
        self.dropout = nn.Dropout2d(p=0.1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.dropout(self.bottleneck(s3))

        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)
        return self.head(d0)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------
def dice_loss(logits, targets, num_classes=NUM_CLASSES, eps=1e-6):
    """Soft multi-class Dice loss, averaged over classes present in the batch."""
    probs = F.softmax(logits, dim=1)
    targets = targets.long()
    one_hot = F.one_hot(targets.clamp(0, num_classes - 1), num_classes)  # B,H,W,C
    one_hot = one_hot.permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    # ignore classes absent in this batch (denom ~ 0 after eps)
    present = one_hot.sum(dims) > 0
    if present.any():
        return 1.0 - dice[present].mean()
    return 1.0 - dice.mean()


def seg_loss(logits, targets, class_weights):
    ce = F.cross_entropy(logits, targets, weight=class_weights, label_smoothing=0.05)
    dsc = dice_loss(logits, targets)
    return CE_WEIGHT * ce + DICE_WEIGHT * dsc


@torch.no_grad()
def compute_confusion(preds, targets, num_classes):
    mask = (targets >= 0) & (targets < num_classes)
    idx = num_classes * targets[mask].long() + preds[mask].long()
    conf = torch.bincount(idx, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf


def mean_iou_from_confusion(conf):
    ious = []
    for c in range(conf.shape[0]):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        denom = tp + fp + fn
        ious.append(tp / denom if denom > 0 else float("nan"))
    valid = [x for x in ious if not np.isnan(x)]
    return float(np.mean(valid)) if valid else 0.0, ious


# ---------------------------------------------------------------------------
# Train / validate loops
# ---------------------------------------------------------------------------
def run_epoch(model, loader, optimizer, class_weights, train=True):
    model.train() if train else model.eval()
    total_loss = 0.0
    conf_total = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    n_seen = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

            if train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = seg_loss(logits, masks, class_weights)

            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n_seen += bs
            preds = logits.argmax(dim=1)
            conf_total += compute_confusion(preds.cpu(), masks.cpu(), NUM_CLASSES)

    avg_loss = total_loss / max(n_seen, 1)
    miou, per_class_iou = mean_iou_from_confusion(conf_total)
    pixel_acc = conf_total.diag().sum().item() / max(conf_total.sum().item(), 1)
    return avg_loss, miou, pixel_acc, per_class_iou


def compute_class_weights(dataset, num_classes=NUM_CLASSES):
    """Inverse-frequency weights from all masks, with minority-class boosts."""
    counts = np.zeros(num_classes, dtype=np.float64)
    # unique underlying samples (ignore crops_per_image multiplier)
    for npy_path, mask_path in dataset.samples:
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        for c in range(num_classes):
            counts[c] += np.sum(mask == c)
    counts = np.clip(counts, 1, None)
    weights = counts.sum() / (num_classes * counts)
    for c, boost in CLASS_BOOST.items():
        weights[c] *= boost
    # keep weights in a stable range
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32), counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    _sanity_check_paths()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train_aois, val_aois = split_aois()
    print(f"Device: {DEVICE}")
    print(f"Train AOIs ({len(train_aois)}): {train_aois}")
    print(f"Val AOIs   ({len(val_aois)}): {val_aois}")

    train_ds = LandCoverDataset(train_aois, train=True, crops_per_image=CROPS_PER_IMAGE)
    val_ds = LandCoverDataset(val_aois, train=False)
    print(
        f"Train samples: {len(train_ds.samples)} "
        f"({len(train_ds)} crops/epoch), Val samples: {len(val_ds)}"
    )

    num_workers = 0 if DEVICE.type == "cpu" else 2
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
    )

    class_weights, counts = compute_class_weights(train_ds)
    class_weights = class_weights.to(DEVICE)
    print("Pixel counts:", {n: int(c) for n, c in zip(CLASS_NAMES, counts)})
    print("Class weights:", {n: round(w, 3) for n, w in zip(CLASS_NAMES, class_weights.tolist())})

    model = ResUNet().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_miou = 0.0
    best_epoch = 0
    stale = 0
    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_miou, train_acc, _ = run_epoch(
            model, train_loader, optimizer, class_weights, train=True
        )
        val_loss, val_miou, val_acc, val_per_class = run_epoch(
            model, val_loader, optimizer, class_weights, train=False
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_acc": val_acc,
            "lr": optimizer.param_groups[0]["lr"],
            "per_class_iou": [
                None if (isinstance(x, float) and np.isnan(x)) else float(x)
                for x in val_per_class
            ],
        }
        history.append(row)

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"train loss {train_loss:.4f} mIoU {train_miou:.3f} acc {train_acc:.3f} | "
            f"val loss {val_loss:.4f} mIoU {val_miou:.3f} acc {val_acc:.3f}"
        )

        if val_miou > best_miou + 1e-4:
            best_miou = val_miou
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_miou": val_miou,
                    "val_acc": val_acc,
                    "epoch": epoch,
                    "class_weights": class_weights.detach().cpu(),
                    "config": {
                        "dice_weight": DICE_WEIGHT,
                        "ce_weight": CE_WEIGHT,
                        "class_boost": CLASS_BOOST,
                        "crops_per_image": CROPS_PER_IMAGE,
                    },
                },
                CHECKPOINT_PATH,
            )
            print(f"  -> new best model saved to {CHECKPOINT_PATH} (val mIoU {val_miou:.3f})")
            print("     per-class IoU:", {
                n: (f"{v:.3f}" if not np.isnan(v) else "n/a")
                for n, v in zip(CLASS_NAMES, val_per_class)
            })
        else:
            stale += 1
            if stale >= EARLY_STOP_PATIENCE:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}, mIoU {best_miou:.3f})")
                break

    with open(HISTORY_PATH, "w") as f:
        json.dump({"best_epoch": best_epoch, "best_val_miou": best_miou, "history": history}, f, indent=2)

    print("Training complete. Best val mIoU:", best_miou, "at epoch", best_epoch)
    print(f"History written to {HISTORY_PATH}")


if __name__ == "__main__":
    main()

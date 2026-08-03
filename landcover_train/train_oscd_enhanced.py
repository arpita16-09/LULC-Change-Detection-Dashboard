"""
Train ResUNet on the OSCD-enhanced 10-channel land-cover dataset.

Expects landcover_train/enhanced/{train,test}/ produced by
build_oscd_enhanced_dataset.py.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = Path(__file__).resolve().parent
ENHANCED = SCRIPT_DIR / "enhanced"
NUM_CLASSES = 6
IN_CHANNELS = 10
IGNORE_INDEX = 255
CROP_SIZE = 256
CLASS_NAMES = ["Other", "Water", "Urban", "Forest", "Agriculture", "Barren"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LandCoverDataset(Dataset):
    def __init__(self, root: Path, aoi_list, crop_size=CROP_SIZE, train=True, crops_per_image=1):
        self.root = Path(root)
        self.img_dir = self.root / "images_multiband_npy"
        self.mask_dir = self.root / "masks"
        self.crop_size = crop_size
        self.train = train
        self.crops_per_image = crops_per_image if train else 1
        self.samples = []
        for aoi in aoi_list:
            for npy_path in sorted(self.img_dir.glob(f"{aoi}_t*.npy")):
                sid = npy_path.stem
                mask_path = self.mask_dir / f"{sid}_mask.png"
                if mask_path.exists():
                    self.samples.append((npy_path, mask_path))
        if not self.samples:
            raise RuntimeError(f"No samples in {self.img_dir} for {aoi_list}")

    def __len__(self):
        return len(self.samples) * self.crops_per_image

    def _pad(self, img, mask):
        h, w = mask.shape
        cs = self.crop_size
        ph, pw = max(0, cs - h), max(0, cs - w)
        if ph or pw:
            img = np.pad(img, ((0, ph), (0, pw), (0, 0)), mode="reflect")
            mask = np.pad(mask, ((0, ph), (0, pw)), mode="constant", constant_values=IGNORE_INDEX)
        return img, mask

    def _aug_spectral(self, img):
        if random.random() < 0.7:
            img = img * random.uniform(0.85, 1.15)
        if random.random() < 0.5:
            bias = np.random.uniform(-0.03, 0.03, size=(1, 1, img.shape[2])).astype(np.float32)
            img = img + bias
        if random.random() < 0.5:
            img = img + np.random.normal(0, 0.015, img.shape).astype(np.float32)
        return img

    def __getitem__(self, idx):
        npy_path, mask_path = self.samples[idx % len(self.samples)]
        img = np.load(npy_path).astype(np.float32)
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 3:
            mask = mask[..., 0]
        img, mask = self._pad(img, mask)
        h, w = mask.shape
        cs = self.crop_size
        if self.train:
            top = random.randint(0, h - cs)
            left = random.randint(0, w - cs)
        else:
            top = (h - cs) // 2
            left = (w - cs) // 2
        img = img[top:top + cs, left:left + cs]
        mask = mask[top:top + cs, left:left + cs]
        if self.train:
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[:, ::-1])
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() < 0.5:
                img = np.ascontiguousarray(img[::-1])
                mask = np.ascontiguousarray(mask[::-1])
            k = random.randint(0, 3)
            if k:
                img = np.ascontiguousarray(np.rot90(img, k, axes=(0, 1)))
                mask = np.ascontiguousarray(np.rot90(mask, k))
            img = self._aug_spectral(img)
        return (
            torch.from_numpy(img.transpose(2, 0, 1).copy()).float(),
            torch.from_numpy(mask.copy()).long(),
        )


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.res = ResBlock(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        x = F.pad(x, [dw // 2, dw - dw // 2, dh // 2, dh - dh // 2])
        return self.res(torch.cat([x, skip], dim=1))


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
        self.drop = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(base, num_classes, 1)

    def forward(self, x):
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(s1)
        s3 = self.enc3(s2)
        b = self.drop(self.bottleneck(s3))
        d3 = self.up3(b, s3)
        d2 = self.up2(d3, s2)
        d1 = self.up1(d2, s1)
        d0 = self.up0(d1, s0)
        return self.head(d0)


def dice_loss(logits, targets, eps=1e-6):
    probs = F.softmax(logits, dim=1)
    valid = targets != IGNORE_INDEX
    if valid.sum() == 0:
        return logits.sum() * 0.0
    t = targets.clone()
    t[~valid] = 0
    one_hot = F.one_hot(t, NUM_CLASSES).permute(0, 3, 1, 2).float()
    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    one_hot = one_hot * valid_f
    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dims)
    denom = probs.sum(dims) + one_hot.sum(dims)
    dice = (2 * inter + eps) / (denom + eps)
    present = one_hot.sum(dims) > 0
    return 1 - (dice[present].mean() if present.any() else dice.mean())


def seg_loss(logits, targets, weights):
    ce = F.cross_entropy(logits, targets, weight=weights, ignore_index=IGNORE_INDEX, label_smoothing=0.05)
    return 0.6 * ce + 0.4 * dice_loss(logits, targets)


@torch.no_grad()
def confusion(preds, targets):
    mask = (targets >= 0) & (targets < NUM_CLASSES)
    idx = NUM_CLASSES * targets[mask].long() + preds[mask].long()
    return torch.bincount(idx, minlength=NUM_CLASSES ** 2).reshape(NUM_CLASSES, NUM_CLASSES)


def miou_from_conf(conf):
    ious = []
    for c in range(NUM_CLASSES):
        tp = conf[c, c].item()
        fp = conf[:, c].sum().item() - tp
        fn = conf[c, :].sum().item() - tp
        d = tp + fp + fn
        ious.append(tp / d if d > 0 else float("nan"))
    valid = [x for x in ious if not np.isnan(x)]
    return float(np.mean(valid)) if valid else 0.0, ious


def run_epoch(model, loader, opt, weights, train=True):
    model.train() if train else model.eval()
    total_loss, n = 0.0, 0
    conf = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, masks in loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            if train:
                opt.zero_grad()
            logits = model(imgs)
            loss = seg_loss(logits, masks, weights)
            if train:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            n += bs
            conf += confusion(logits.argmax(1).cpu(), masks.cpu())
    miou, per = miou_from_conf(conf)
    acc = conf.diag().sum().item() / max(conf.sum().item(), 1)
    return total_loss / max(n, 1), miou, acc, per


def class_weights(dataset, min_w=0.5, max_w=2.0):
    """Inverse-frequency weights, clipped so rare classes cannot dominate."""
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    for _, mask_path in dataset.samples:
        m = np.array(Image.open(mask_path))
        if m.ndim == 3:
            m = m[..., 0]
        for c in range(NUM_CLASSES):
            counts[c] += np.sum(m == c)
    counts = np.clip(counts, 1, None)
    w = counts.sum() / (NUM_CLASSES * counts)
    w = w / w.mean()
    w = np.clip(w, min_w, max_w)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32), counts


def split_aois(all_aois, val_fraction=0.2, seed=42):
    aois = list(all_aois)
    rng = random.Random(seed)
    rng.shuffle(aois)
    n_val = max(1, int(len(aois) * val_fraction))
    return aois[n_val:], aois[:n_val]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--crops-per-image", type=int, default=4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument(
        "--checkpoint",
        default=str(SCRIPT_DIR / "train" / "resunet_oscd_enhanced_best.pt"),
    )
    args = parser.parse_args()

    meta = json.loads((ENHANCED / "dataset_meta.json").read_text())
    train_aois_all = meta["splits"]["train"]
    test_aois = meta["splits"]["test"]
    train_aois, val_aois = split_aois(train_aois_all)

    print("Device:", DEVICE)
    print("Train AOIs", train_aois)
    print("Val AOIs", val_aois)
    print("Holdout test AOIs", test_aois)

    train_ds = LandCoverDataset(ENHANCED / "train", train_aois, train=True, crops_per_image=args.crops_per_image)
    val_ds = LandCoverDataset(ENHANCED / "train", val_aois, train=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights, counts = class_weights(train_ds)
    weights = weights.to(DEVICE)
    print("counts", {n: int(c) for n, c in zip(CLASS_NAMES, counts)})
    print("weights", {n: round(w, 3) for n, w in zip(CLASS_NAMES, weights.tolist())})

    model = ResUNet(in_channels=meta["n_channels"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best_miou, best_epoch, stale = -1.0, 0, 0
    history = []
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, opt, weights, train=True)
        va = run_epoch(model, val_loader, opt, weights, train=False)
        sched.step()
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train {tr[1]:.3f}/{tr[2]:.3f} | val {va[1]:.3f}/{va[2]:.3f}"
        )
        print("  per-class", {n: (f"{v:.3f}" if not np.isnan(v) else "n/a") for n, v in zip(CLASS_NAMES, va[3])})
        history.append(
            {
                "epoch": epoch,
                "train_miou": tr[1],
                "train_acc": tr[2],
                "val_miou": va[1],
                "val_acc": va[2],
                "per_class_iou": [None if np.isnan(x) else float(x) for x in va[3]],
            }
        )
        if va[1] > best_miou + 1e-4:
            best_miou, best_epoch, stale = va[1], epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "val_miou": va[1],
                    "val_acc": va[2],
                    "epoch": epoch,
                    "in_channels": meta["n_channels"],
                    "channels": meta["channels"],
                    "ignore_index": IGNORE_INDEX,
                },
                args.checkpoint,
            )
            print("  -> saved", args.checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping")
                break

    hist_path = SCRIPT_DIR / "train" / "train_history_oscd_enhanced.json"
    hist_path.write_text(json.dumps({"best_epoch": best_epoch, "best_val_miou": best_miou, "history": history}, indent=2))
    print("Done best", best_epoch, best_miou)


if __name__ == "__main__":
    main()

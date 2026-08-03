"""
Build an OSCD-enhanced land-cover training set.

Uses the uploaded OSCD Images (imgs_*_rect) plus official change masks to:
  1) create richer input tensors (RGB, NIR, SWIR1/2 + spectral indices)
  2) keep / lightly clean pseudo-label masks using temporal agreement on
     no-change pixels (ignore disagreements)

Outputs under landcover_train/enhanced/{train,test}/
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import tifffile as tf
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
OSCD_IMAGES = REPO / "oscd_images" / "Onera Satellite Change Detection dataset - Images"
OSCD_TRAIN_LABELS = REPO / "oscd_labels" / "Onera Satellite Change Detection dataset - Train Labels"
OSCD_TEST_LABELS = REPO / "oscd_labels" / "Onera Satellite Change Detection dataset - Test Labels"
OUT = SCRIPT_DIR / "enhanced"
MANIFEST = SCRIPT_DIR / "split_manifest.json"

BANDS_NEEDED = ["B02", "B03", "B04", "B08", "B11", "B12"]
IGNORE_INDEX = 255
CLASS_NAMES = {
    0: "Other",
    1: "Water",
    2: "Urban",
    3: "Forest",
    4: "Agriculture",
    5: "Barren",
}


def safe_div(a, b):
    return a / np.where(np.abs(b) < 1e-6, 1e-6, b)


def read_band(path: Path) -> np.ndarray:
    arr = tf.imread(str(path)).astype(np.float32)
    # Sentinel-2 L1C digital numbers ~ reflectance * 10000
    return arr / 10000.0


def load_bands(aoi_dir: Path) -> dict:
    bands = {}
    for b in BANDS_NEEDED:
        p = aoi_dir / f"{b}.tif"
        if not p.exists():
            raise FileNotFoundError(p)
        bands[b] = read_band(p)
    return bands


def classify(bands: dict) -> np.ndarray:
    """Improved spectral rules using SWIR2 as well."""
    B02, B03, B04 = bands["B02"], bands["B03"], bands["B04"]
    B08, B11, B12 = bands["B08"], bands["B11"], bands["B12"]

    ndvi = safe_div(B08 - B04, B08 + B04)
    ndwi = safe_div(B03 - B08, B03 + B08)
    ndbi = safe_div(B11 - B08, B11 + B08)
    bsi = safe_div((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02))
    # NDBSI-like using B12 helps separate barren/built-up a bit
    ndbsi = safe_div(B12 - B08, B12 + B08)

    label = np.zeros(ndvi.shape, dtype=np.uint8)
    water = ndwi > 0.10
    forest = (~water) & (ndvi > 0.50)
    agri = (~water) & (~forest) & (ndvi > 0.20)
    urban = (~water) & (~forest) & (~agri) & (ndbi > 0.05) & (ndvi < 0.18)
    barren = (
        (~water) & (~forest) & (~agri) & (~urban)
        & ((bsi > 0.10) | ((ndbsi > 0.05) & (ndvi < 0.20)))
    )

    label[water] = 1
    label[urban] = 2
    label[forest] = 3
    label[agri] = 4
    label[barren] = 5
    return label


def build_features(bands: dict) -> np.ndarray:
    """Return float32 HWC tensor with 10 channels."""
    B02, B03, B04 = bands["B02"], bands["B03"], bands["B04"]
    B08, B11, B12 = bands["B08"], bands["B11"], bands["B12"]
    ndvi = safe_div(B08 - B04, B08 + B04)
    ndwi = safe_div(B03 - B08, B03 + B08)
    ndbi = safe_div(B11 - B08, B11 + B08)
    bsi = safe_div((B11 + B04) - (B08 + B02), (B11 + B04) + (B08 + B02))
    feats = np.stack(
        [
            B04, B03, B02, B08, B11, B12,
            ndvi, ndwi, ndbi, bsi,
        ],
        axis=-1,
    ).astype(np.float32)
    # clip reflectance-like channels; indices already ~[-1,1]
    feats[..., :6] = np.clip(feats[..., :6], 0.0, 3.0)
    feats[..., 6:] = np.clip(feats[..., 6:], -1.0, 1.0)
    return feats


def load_change_mask(aoi: str, shape) -> np.ndarray | None:
    for root in (OSCD_TRAIN_LABELS, OSCD_TEST_LABELS):
        png = root / aoi / "cm" / "cm.png"
        tif = root / aoi / "cm" / f"{aoi}-cm.tif"
        if png.exists():
            m = np.array(Image.open(png))
            if m.ndim == 3:
                m = m[..., 0]
            # png: 0 no-change, 255 change
            return (m > 127).astype(np.uint8)
        if tif.exists():
            m = tf.imread(str(tif))
            if m.ndim == 3:
                m = m[..., 0]
            return (m > 0).astype(np.uint8)
    return None


def clean_pair_labels(lab1: np.ndarray, lab2: np.ndarray, change: np.ndarray | None):
    """On no-change pixels, disagreeing labels become IGNORE_INDEX."""
    out1, out2 = lab1.copy(), lab2.copy()
    if change is None:
        return out1, out2, 0
    # resize change if needed
    if change.shape != lab1.shape:
        change_img = Image.fromarray(change)
        change = np.array(change_img.resize((lab1.shape[1], lab1.shape[0]), Image.NEAREST))
    stable = change == 0
    disagree = stable & (lab1 != lab2)
    n = int(disagree.sum())
    out1[disagree] = IGNORE_INDEX
    out2[disagree] = IGNORE_INDEX
    return out1, out2, n


def ensure_dirs(split: str):
    base = OUT / split
    for sub in ("images_multiband_npy", "masks", "masks_colored", "images_rgb"):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def colorize(label: np.ndarray) -> np.ndarray:
    colors = {
        0: (60, 60, 60),
        1: (0, 90, 255),
        2: (230, 30, 30),
        3: (0, 130, 0),
        4: (190, 230, 60),
        5: (180, 140, 90),
        255: (0, 0, 0),
    }
    out = np.zeros((*label.shape, 3), dtype=np.uint8)
    for k, c in colors.items():
        out[label == k] = c
    return out


def to_rgb(bands: dict) -> np.ndarray:
    rgb = np.stack([bands["B04"], bands["B03"], bands["B02"]], axis=-1)
    lo, hi = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def main():
    manifest = json.loads(MANIFEST.read_text())
    splits = {
        "train": manifest["train_aois"],
        "test": manifest["test_aois"],
    }
    channel_names = [
        "B04", "B03", "B02", "B08", "B11", "B12",
        "NDVI", "NDWI", "NDBI", "BSI",
    ]
    summary = []
    stats = {"ignored_pixels": 0, "total_pixels": 0}

    for split, aois in splits.items():
        out_root = ensure_dirs(split)
        for aoi in aois:
            bands_t = {}
            labels_t = {}
            for t, sub in (("1", "imgs_1_rect"), ("2", "imgs_2_rect")):
                aoi_dir = OSCD_IMAGES / aoi / sub
                bands = load_bands(aoi_dir)
                bands_t[t] = bands
                labels_t[t] = classify(bands)

            change = load_change_mask(aoi, labels_t["1"].shape)
            lab1, lab2, n_ign = clean_pair_labels(labels_t["1"], labels_t["2"], change)
            labels_t = {"1": lab1, "2": lab2}
            stats["ignored_pixels"] += n_ign
            stats["total_pixels"] += lab1.size + lab2.size

            for t in ("1", "2"):
                sid = f"{aoi}_t{t}"
                feats = build_features(bands_t[t])
                label = labels_t[t]
                np.save(out_root / "images_multiband_npy" / f"{sid}.npy", feats)
                Image.fromarray(label).save(out_root / "masks" / f"{sid}_mask.png")
                Image.fromarray(colorize(label)).save(
                    out_root / "masks_colored" / f"{sid}_mask_color.png"
                )
                Image.fromarray(to_rgb(bands_t[t])).save(out_root / "images_rgb" / f"{sid}.png")

                valid = label != IGNORE_INDEX
                total_valid = int(valid.sum()) or 1
                row = {
                    "sample_id": sid,
                    "split": split,
                    "aoi": aoi,
                    "height": int(label.shape[0]),
                    "width": int(label.shape[1]),
                    "n_ignore": int((label == IGNORE_INDEX).sum()),
                    "has_change_mask": change is not None,
                }
                for cid, name in CLASS_NAMES.items():
                    row[f"pct_{name}"] = round(
                        100.0 * int(((label == cid) & valid).sum()) / total_valid, 2
                    )
                summary.append(row)
                print(
                    f"{sid:20s} split={split} shape={label.shape} "
                    f"ignore={(label == IGNORE_INDEX).mean():.3f} change={change is not None}"
                )

    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "labels_summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    meta = {
        "channels": channel_names,
        "n_channels": len(channel_names),
        "ignore_index": IGNORE_INDEX,
        "source_images": str(OSCD_IMAGES),
        "change_labels": [str(OSCD_TRAIN_LABELS), str(OSCD_TEST_LABELS)],
        "stats": stats,
        "splits": splits,
        "notes": (
            "Features rebuilt from OSCD imgs_*_rect. Pseudo-labels regenerated with "
            "SWIR2-aware rules; no-change disagreements set to ignore_index=255."
        ),
    }
    (OUT / "dataset_meta.json").write_text(json.dumps(meta, indent=2))
    print("Wrote", OUT)
    print("Ignored pixels:", stats["ignored_pixels"], "/", stats["total_pixels"])


if __name__ == "__main__":
    main()

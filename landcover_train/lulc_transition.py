"""
LULC transition (from → to) change detection.

Runs the OSCD-enhanced land-cover model on t1 and t2, then computes:
  - per-pixel transition map (encoded as from*6 + to)
  - transition counts / percentages
  - optional intersection with OSCD binary change masks

Usage:
  python landcover_train/lulc_transition.py --aoi bordeaux
  python landcover_train/lulc_transition.py --all-test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from train_oscd_enhanced import ResUNet, CLASS_NAMES, NUM_CLASSES, DEVICE

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
ENHANCED = SCRIPT_DIR / "enhanced"
DEFAULT_CKPT = SCRIPT_DIR / "train" / "resunet_oscd_enhanced_best.pt"
OSCD_IMG = REPO / "oscd_images" / "Onera Satellite Change Detection dataset - Images"
OSCD_TRAIN_L = REPO / "oscd_labels" / "Onera Satellite Change Detection dataset - Train Labels"
OSCD_TEST_L = REPO / "oscd_labels" / "Onera Satellite Change Detection dataset - Test Labels"


def _parse_aoi_list(txt_name: str) -> list[str]:
    p = OSCD_IMG / txt_name
    if not p.exists():
        return []
    return [a.strip() for a in p.read_text().replace("\n", ",").split(",") if a.strip()]


def official_aoi_splits() -> dict:
    """All OSCD cities from train.txt / test.txt (24 total)."""
    train = sorted(_parse_aoi_list("train.txt"))
    test = sorted(_parse_aoi_list("test.txt"))
    if not train and not test:
        # fallback: every folder under OSCD images
        all_aois = sorted(
            d.name
            for d in OSCD_IMG.iterdir()
            if d.is_dir() and (d / "imgs_1_rect").exists()
        )
        return {"train": all_aois, "test": []}
    return {"train": train, "test": test}


def _load_oscd_features(aoi: str, which: str) -> np.ndarray:
    """Build 10-channel features directly from OSCD imgs_*_rect."""
    import build_oscd_enhanced_dataset as bed

    aoi_dir = OSCD_IMG / aoi / f"imgs_{which}_rect"
    if not aoi_dir.is_dir():
        raise FileNotFoundError(f"Missing OSCD imagery: {aoi_dir}")
    bands = bed.load_bands(aoi_dir)
    return bed.build_features(bands)


def load_aoi_pair(aoi: str, split: str | None = None):
    """Load t1/t2 features from enhanced cache, else from raw OSCD imagery."""
    splits = official_aoi_splits()
    if split is None:
        if aoi in splits["test"]:
            split = "test"
        elif aoi in splits["train"]:
            split = "train"
        else:
            split = "train"

    t1_path = ENHANCED / split / "images_multiband_npy" / f"{aoi}_t1.npy"
    t2_path = ENHANCED / split / "images_multiband_npy" / f"{aoi}_t2.npy"
    # also check the other split folder
    if not t1_path.exists():
        alt = "train" if split == "test" else "test"
        t1_alt = ENHANCED / alt / "images_multiband_npy" / f"{aoi}_t1.npy"
        t2_alt = ENHANCED / alt / "images_multiband_npy" / f"{aoi}_t2.npy"
        if t1_alt.exists() and t2_alt.exists():
            t1_path, t2_path, split = t1_alt, t2_alt, alt

    if t1_path.exists() and t2_path.exists():
        t1 = np.load(t1_path).astype(np.float32)
        t2 = np.load(t2_path).astype(np.float32)
        return t1, t2, split, "enhanced_cache"

    t1 = _load_oscd_features(aoi, "1")
    t2 = _load_oscd_features(aoi, "2")
    return t1, t2, split, "oscd_images"

# Distinct colors for common transitions (for viz)
TRANSITION_COLORS = {
    (3, 2): (220, 20, 60),      # Forest -> Urban
    (4, 2): (255, 140, 0),      # Agriculture -> Urban
    (1, 2): (255, 20, 147),     # Water -> Urban
    (5, 2): (178, 34, 34),      # Barren -> Urban
    (3, 4): (154, 205, 50),     # Forest -> Agriculture
    (4, 3): (34, 139, 34),      # Agriculture -> Forest
    (2, 3): (0, 128, 0),        # Urban -> Forest (rare/greening)
    (1, 4): (0, 191, 255),      # Water -> Agriculture
    (4, 1): (30, 144, 255),     # Agriculture -> Water
    (5, 4): (238, 232, 170),    # Barren -> Agriculture
}


def load_model(checkpoint: Path | str = DEFAULT_CKPT):
    ckpt = torch.load(checkpoint, map_location=DEVICE, weights_only=False)
    model = ResUNet(in_channels=ckpt.get("in_channels", 10)).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def predict_mask(model, img_hwc: np.ndarray, tta: bool = True) -> np.ndarray:
    x = torch.from_numpy(img_hwc.transpose(2, 0, 1).copy()).float().unsqueeze(0).to(DEVICE)
    _, _, h, w = x.shape
    pad_h = (16 - h % 16) % 16
    pad_w = (16 - w % 16) % 16

    def fwd(t):
        if pad_h or pad_w:
            t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect")
        return model(t)[:, :, :h, :w]

    if not tta:
        return fwd(x).argmax(1)[0].cpu().numpy().astype(np.uint8)

    probs = F.softmax(fwd(x), dim=1)
    probs = probs + torch.flip(F.softmax(fwd(torch.flip(x, [3])), 1), [3])
    probs = probs + torch.flip(F.softmax(fwd(torch.flip(x, [2])), 1), [2])
    probs = probs + torch.flip(F.softmax(fwd(torch.flip(x, [2, 3])), 1), [2, 3])
    return (probs / 4).argmax(1)[0].cpu().numpy().astype(np.uint8)


def load_change_mask(aoi: str, shape) -> np.ndarray | None:
    for root in (OSCD_TRAIN_L, OSCD_TEST_L):
        png = root / aoi / "cm" / "cm.png"
        if png.exists():
            m = np.array(Image.open(png))
            if m.ndim == 3:
                m = m[..., 0]
            m = (m > 127).astype(np.uint8)
            if m.shape != shape:
                m = np.array(Image.fromarray(m).resize((shape[1], shape[0]), Image.NEAREST))
            return m
    return None


def class_fractions(mask: np.ndarray) -> dict:
    total = mask.size
    return {
        CLASS_NAMES[c]: round(100.0 * float(np.mean(mask == c)), 3)
        for c in range(NUM_CLASSES)
    }


def transition_stats(mask_t1: np.ndarray, mask_t2: np.ndarray, change_mask: np.ndarray | None = None):
    """Return counts/pct for from→to, optionally only inside OSCD change pixels."""
    assert mask_t1.shape == mask_t2.shape
    if change_mask is not None:
        sel = change_mask.astype(bool)
    else:
        sel = mask_t1 != mask_t2

    changed = (mask_t1 != mask_t2) & sel
    n_changed = int(changed.sum())
    n_total = int(mask_t1.size)
    n_sel = int(sel.sum()) if change_mask is not None else n_total

    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            matrix[a, b] = int(((mask_t1 == a) & (mask_t2 == b) & sel).sum())

    transitions = []
    for a in range(NUM_CLASSES):
        for b in range(NUM_CLASSES):
            if a == b:
                continue
            cnt = int(matrix[a, b])
            if cnt == 0:
                continue
            transitions.append(
                {
                    "from": CLASS_NAMES[a],
                    "to": CLASS_NAMES[b],
                    "from_id": a,
                    "to_id": b,
                    "pixels": cnt,
                    "pct_of_image": round(100.0 * cnt / n_total, 4),
                    "pct_of_changed": round(100.0 * cnt / max(n_changed, 1), 4),
                }
            )
    transitions.sort(key=lambda r: r["pixels"], reverse=True)

    # dashboard-friendly summary keys
    summary = {
        "forest_to_urban": _pct(matrix, 3, 2, n_total),
        "agriculture_to_urban": _pct(matrix, 4, 2, n_total),
        "water_to_urban": _pct(matrix, 1, 2, n_total),
        "barren_to_urban": _pct(matrix, 5, 2, n_total),
        "forest_to_agriculture": _pct(matrix, 3, 4, n_total),
        "agriculture_to_forest": _pct(matrix, 4, 3, n_total),
        "any_to_urban": round(100.0 * float(matrix[:, 2].sum() - matrix[2, 2]) / n_total, 4),
        "urban_to_any": round(100.0 * float(matrix[2, :].sum() - matrix[2, 2]) / n_total, 4),
        "vegetation_loss": round(
            100.0
            * float(
                matrix[3, 2] + matrix[3, 5] + matrix[4, 2] + matrix[4, 5] + matrix[3, 0] + matrix[4, 0]
            )
            / n_total,
            4,
        ),
    }

    return {
        "n_total_pixels": n_total,
        "n_changed_pixels": n_changed,
        "change_pct": round(100.0 * n_changed / n_total, 4),
        "selection_pixels": n_sel,
        "used_oscd_change_mask": change_mask is not None,
        "transition_matrix": matrix.tolist(),
        "transitions": transitions,
        "summary": summary,
        "t1_fractions": class_fractions(mask_t1),
        "t2_fractions": class_fractions(mask_t2),
    }


def _pct(matrix, a, b, n_total):
    return round(100.0 * float(matrix[a, b]) / n_total, 4)


def colorize_lulc(mask: np.ndarray) -> np.ndarray:
    palette = {
        0: (60, 60, 60),
        1: (30, 144, 255),
        2: (220, 20, 60),
        3: (34, 139, 34),
        4: (173, 255, 47),
        5: (210, 180, 140),
    }
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, c in palette.items():
        rgb[mask == k] = c
    return rgb


def colorize_transitions(mask_t1: np.ndarray, mask_t2: np.ndarray, change_mask: np.ndarray | None = None) -> np.ndarray:
    rgb = np.zeros((*mask_t1.shape, 3), dtype=np.uint8)
    rgb[:] = (40, 40, 40)  # stable background
    changed = mask_t1 != mask_t2
    if change_mask is not None:
        changed = changed & change_mask.astype(bool)
    # default changed color
    rgb[changed] = (200, 200, 200)
    for (a, b), color in TRANSITION_COLORS.items():
        m = changed & (mask_t1 == a) & (mask_t2 == b)
        rgb[m] = color
    return rgb


def run_aoi(
    aoi: str,
    model=None,
    checkpoint: Path = DEFAULT_CKPT,
    split: str | None = None,
    use_oscd_change_mask: bool = True,
    tta: bool = True,
    out_dir: Path | None = None,
):
    if model is None:
        model, _ = load_model(checkpoint)

    t1, t2, split, source = load_aoi_pair(aoi, split=split)

    pred1 = predict_mask(model, t1, tta=tta)
    pred2 = predict_mask(model, t2, tta=tta)
    cm = load_change_mask(aoi, pred1.shape) if use_oscd_change_mask else None
    stats = transition_stats(pred1, pred2, cm)

    result = {
        "aoi": aoi,
        "split": split,
        "feature_source": source,
        "shape": list(pred1.shape),
        **stats,
    }

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred1).save(out_dir / f"{aoi}_t1_lulc.png")
        Image.fromarray(pred2).save(out_dir / f"{aoi}_t2_lulc.png")
        Image.fromarray(colorize_lulc(pred1)).save(out_dir / f"{aoi}_t1_lulc_color.png")
        Image.fromarray(colorize_lulc(pred2)).save(out_dir / f"{aoi}_t2_lulc_color.png")
        Image.fromarray(colorize_transitions(pred1, pred2, cm)).save(out_dir / f"{aoi}_transition_color.png")
        # encode transition id map: from*NUM + to (ignore stable as 255)
        tmap = np.where(pred1 != pred2, pred1.astype(np.int16) * NUM_CLASSES + pred2, 255).astype(np.uint8)
        Image.fromarray(tmap).save(out_dir / f"{aoi}_transition_code.png")
        (out_dir / f"{aoi}_transition.json").write_text(json.dumps(result, indent=2))

    return result, pred1, pred2


def main():
    parser = argparse.ArgumentParser(description="LULC transition change detection")
    parser.add_argument("--aoi", default="")
    parser.add_argument("--all-test", action="store_true")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "transition_results"))
    parser.add_argument("--no-oscd-mask", action="store_true")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    model, ckpt = load_model(args.checkpoint)
    print(f"Loaded epoch={ckpt.get('epoch')} val_mIoU={ckpt.get('val_miou')}")

    if args.all_test:
        aois = official_aoi_splits()["test"]
        if not aois:
            aois = sorted(
                {
                    p.name.replace("_t1.npy", "")
                    for p in (ENHANCED / "test" / "images_multiband_npy").glob("*_t1.npy")
                }
            )
    elif args.aoi:
        aois = [args.aoi]
    else:
        raise SystemExit("Pass --aoi <name> or --all-test")

    out = Path(args.out_dir)
    summary = []
    for aoi in aois:
        result, _, _ = run_aoi(
            aoi,
            model=model,
            use_oscd_change_mask=not args.no_oscd_mask,
            tta=not args.no_tta,
            out_dir=out,
        )
        summary.append(
            {
                "aoi": aoi,
                "change_pct": result["change_pct"],
                "top": result["transitions"][:5],
                "summary": result["summary"],
            }
        )
        print(
            f"{aoi:14s} change={result['change_pct']:.2f}%  "
            f"top={result['transitions'][0] if result['transitions'] else None}"
        )

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Wrote", out)


if __name__ == "__main__":
    main()

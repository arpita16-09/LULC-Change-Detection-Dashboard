"""Backend wrapper around landcover_train.lulc_transition."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
LANDCOVER = REPO / "landcover_train"
if str(LANDCOVER) not in sys.path:
    sys.path.insert(0, str(LANDCOVER))

import lulc_transition as lt  # noqa: E402

UPLOADS = Path(__file__).resolve().parents[1] / "uploads"
RESULTS = UPLOADS / "transitions"
RESULTS.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_model():
    return lt.load_model(lt.DEFAULT_CKPT)


def list_aois():
    """Return all official OSCD AOIs (24), not only cached enhanced .npy files."""
    return lt.official_aoi_splits()


def run_aoi_transition(aoi: str, use_oscd_mask: bool = True, tta: bool = True):
    model, ckpt = get_model()
    out_dir = RESULTS / aoi
    result, pred1, pred2 = lt.run_aoi(
        aoi,
        model=model,
        use_oscd_change_mask=use_oscd_mask,
        tta=tta,
        out_dir=out_dir,
    )
    result["artifacts"] = {
        "t1_lulc_color": f"/uploads/transitions/{aoi}/{aoi}_t1_lulc_color.png",
        "t2_lulc_color": f"/uploads/transitions/{aoi}/{aoi}_t2_lulc_color.png",
        "transition_color": f"/uploads/transitions/{aoi}/{aoi}_transition_color.png",
        "json": f"/uploads/transitions/{aoi}/{aoi}_transition.json",
    }
    result["model"] = {
        "checkpoint": str(lt.DEFAULT_CKPT.name),
        "epoch": ckpt.get("epoch"),
        "val_miou": ckpt.get("val_miou"),
    }
    # map summary into ChangePanel-friendly keys
    result["changes"] = {
        "forest_to_urban": result["summary"]["forest_to_urban"],
        "agriculture_to_urban": result["summary"]["agriculture_to_urban"],
        "water_to_urban": result["summary"]["water_to_urban"],
        "barren_to_urban": result["summary"]["barren_to_urban"],
        "forest_to_agriculture": result["summary"]["forest_to_agriculture"],
        "agriculture_to_forest": result["summary"]["agriculture_to_forest"],
        "vegetation_loss": result["summary"]["vegetation_loss"],
    }
    result["stats"] = {
        k.lower().replace(" ", "_"): v for k, v in result["t2_fractions"].items()
    }
    return result


def run_array_transition(t1: np.ndarray, t2: np.ndarray, name: str = "upload"):
    model, ckpt = get_model()
    if t1.shape != t2.shape:
        raise ValueError(f"t1/t2 shape mismatch: {t1.shape} vs {t2.shape}")
    if t1.ndim != 3 or t1.shape[-1] != 10:
        raise ValueError("Expected enhanced tensors with shape (H,W,10)")
    pred1 = lt.predict_mask(model, t1.astype(np.float32), tta=True)
    pred2 = lt.predict_mask(model, t2.astype(np.float32), tta=True)
    stats = lt.transition_stats(pred1, pred2, None)
    out_dir = RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(lt.colorize_lulc(pred1)).save(out_dir / f"{name}_t1_lulc_color.png")
    Image.fromarray(lt.colorize_lulc(pred2)).save(out_dir / f"{name}_t2_lulc_color.png")
    Image.fromarray(lt.colorize_transitions(pred1, pred2, None)).save(
        out_dir / f"{name}_transition_color.png"
    )
    result = {"aoi": name, **stats}
    result["artifacts"] = {
        "t1_lulc_color": f"/uploads/transitions/{name}/{name}_t1_lulc_color.png",
        "t2_lulc_color": f"/uploads/transitions/{name}/{name}_t2_lulc_color.png",
        "transition_color": f"/uploads/transitions/{name}/{name}_transition_color.png",
    }
    result["model"] = {
        "checkpoint": str(lt.DEFAULT_CKPT.name),
        "epoch": ckpt.get("epoch"),
        "val_miou": ckpt.get("val_miou"),
    }
    result["changes"] = {
        "forest_to_urban": result["summary"]["forest_to_urban"],
        "agriculture_to_urban": result["summary"]["agriculture_to_urban"],
        "water_to_urban": result["summary"]["water_to_urban"],
        "barren_to_urban": result["summary"]["barren_to_urban"],
        "forest_to_agriculture": result["summary"]["forest_to_agriculture"],
        "agriculture_to_forest": result["summary"]["agriculture_to_forest"],
        "vegetation_loss": result["summary"]["vegetation_loss"],
    }
    return result

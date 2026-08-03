import os
import shutil
import uuid
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from routes.change_detection import router as change_router
from services import lulc_transition_service as transition_svc

app = FastAPI(title="LULC Transition Change Detection")

_default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]
_extra = os.getenv("CORS_ORIGINS", "")
allow_origins = _default_origins + [o.strip() for o in _extra.split(",") if o.strip()]
# Allow all in demo deploys if explicitly requested
if os.getenv("CORS_ALLOW_ALL", "0") == "1":
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(change_router)

UPLOAD_DIR = Path(__file__).parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

DATA_FILE = Path(__file__).parent / "data" / "lulc_data.json"

# Dates are always labeled t1/t2 in the API (no calendar years exposed).
DEFAULT_DATA = {
    "metadata": {
        "study_area": "OSCD multi-city (24 AOIs)",
        "year1": "t1",
        "year2": "t2",
        "model": "resunet_oscd_enhanced_best.pt",
        "task": "LULC transition change detection",
    },
    "stats": {
        "water": 0,
        "urban": 0,
        "forest": 0,
        "agriculture": 0,
        "bare_soil": 0,
        "other": 0,
    },
    "changes": {
        "forest_to_urban": 0,
        "agriculture_to_urban": 0,
        "water_to_urban": 0,
        "barren_to_urban": 0,
        "forest_to_agriculture": 0,
        "agriculture_to_forest": 0,
        "vegetation_loss": 0,
    },
}


def load_data():
    import json

    if not DATA_FILE.exists():
        return DEFAULT_DATA
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Force date labels to t1/t2 even if the JSON is stale.
    meta = data.setdefault("metadata", {})
    meta["year1"] = "t1"
    meta["year2"] = "t2"
    return data


@app.get("/")
def root():
    return {
        "message": "LULC Transition API",
        "endpoints": [
            "/status",
            "/aois",
            "/predict/transition/{aoi}",
            "/predict/transition",
            "/changes",
            "/stats",
        ],
    }


@app.get("/status")
def status():
    data = load_data()
    aois = transition_svc.list_aois()
    return {
        "study_area": data["metadata"].get("study_area", "OSCD multi-city (24 AOIs)"),
        "year1": "t1",
        "year2": "t2",
        "status": "ready",
        "task": "lulc_transition",
        "n_train_aois": len(aois["train"]),
        "n_test_aois": len(aois["test"]),
    }


@app.get("/aois")
def aois():
    return transition_svc.list_aois()


@app.get("/stats")
def get_stats():
    return load_data()["stats"]


@app.get("/changes")
def get_changes():
    """Default summary; UI should call /predict/transition/{aoi} for real results."""
    return load_data()["changes"]


@app.post("/upload")
async def upload_image(image: UploadFile = File(...)):
    ext = os.path.splitext(image.filename or "")[1] or ".bin"
    file_id = str(uuid.uuid4())
    filename = f"{file_id}{ext}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(image.file, f)
    return {
        "status": "uploaded",
        "file_id": file_id,
        "filename": filename,
        "url": f"/uploads/{filename}",
    }


@app.get("/predict/transition/{aoi}")
def predict_transition_aoi(aoi: str, use_oscd_mask: bool = True, tta: bool = True):
    aois = transition_svc.list_aois()
    if aoi not in aois["train"] and aoi not in aois["test"]:
        raise HTTPException(404, f"Unknown AOI '{aoi}'. Use /aois")
    try:
        return transition_svc.run_aoi_transition(aoi, use_oscd_mask=use_oscd_mask, tta=tta)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Transition failed: {e}") from e


@app.post("/predict/transition")
async def predict_transition_upload(
    t1: UploadFile = File(..., description="t1 enhanced .npy (H,W,10)"),
    t2: UploadFile = File(..., description="t2 enhanced .npy (H,W,10)"),
):
    name = str(uuid.uuid4())[:8]
    try:
        arr1 = np.load(t1.file)
        arr2 = np.load(t2.file)
        return transition_svc.run_array_transition(arr1, arr2, name=name)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, f"Transition failed: {e}") from e


@app.post("/predict")
async def predict_legacy(image: UploadFile = File(...)):
    """Backward-compatible stub pointing clients to transition endpoints."""
    return {
        "status": "use_transition_endpoints",
        "message": "Use GET /predict/transition/{aoi} or POST /predict/transition with t1/t2 .npy",
        "lulc_classes": load_data()["stats"],
    }

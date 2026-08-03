# LULC Change Detection Dashboard

Land Use / Land Cover (**LULC**) transition change detection on the [Onera Satellite Change Detection (OSCD)](https://rcdaudt.github.io/oscd/) dataset.

The app predicts land-cover classes at two dates (**t1** and **t2**), then reports **from→to** changes (e.g. Forest→Urban, Agriculture→Urban) with maps and statistics for **24 OSCD cities**.

## What this project does

1. Runs an **OSCD-enhanced ResUNet** on t1 and t2 imagery (10 spectral channels).
2. Builds per-pixel LULC maps (Other, Water, Urban, Forest, Agriculture, Barren).
3. Computes transition stats and optional filtering with official OSCD change masks.
4. Serves results through a **FastAPI** backend and a **React** dashboard.

> This is **not** a Siamese U-Net. A single ResUNet classifies each date; transitions are derived by comparing the two predicted maps.

## Repository layout

| Path | Description |
|---|---|
| `lulc-dashboard/` | React + Vite frontend |
| `backend/` | FastAPI API (`/aois`, `/predict/transition/{aoi}`, …) |
| `landcover_train/` | Training, evaluation, transition pipeline, model checkpoints |
| `oscd_images/` | OSCD Sentinel-2 image pairs (24 AOIs) |
| `oscd_labels/` | Official OSCD binary change masks |
| `DEPLOY.md` | Free hosting notes (Hugging Face / Vercel / Render) |

## Requirements

- Python **3.10+** (PyTorch, FastAPI)
- Node.js **18+** and npm
- Git LFS (large `.npy` / model / imagery files)

```bash
git lfs install
git clone <your-fork-or-repo-url>
cd LULC-Change-Detection-Dashboard
git lfs pull
```

## Local deployment

> The Python API is **FastAPI** (not Flask). It fills the same role as a Flask backend:
> serve `/aois`, run the model, return transition JSON/images. The React UI is separate
> and will later deploy to **Vercel**.

### Option A — helper scripts

**Windows (PowerShell)** — two terminals:

```powershell
# Terminal 1 — API
.\scripts\start-backend.ps1

# Terminal 2 — UI
.\scripts\start-frontend.ps1
```

**macOS / Linux / Git Bash:**

```bash
# Terminal 1
bash scripts/start-backend.sh

# Terminal 2
bash scripts/start-frontend.sh
```

### Option B — manual commands

#### 1. Backend API (FastAPI + uvicorn)

```bash
cd backend
python -m pip install -r requirements.txt
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

- Health: http://127.0.0.1:8000/status  
- Docs: http://127.0.0.1:8000/docs  

#### 2. Frontend dashboard (Vite)

```bash
cd lulc-dashboard
npm install
# Windows PowerShell:
$env:VITE_API_BASE="http://127.0.0.1:8000"
npm run dev
```

Open http://127.0.0.1:5173 → choose an AOI → **Run LULC Transition**.  
Header should show **API ● online** and **24** AOIs.

### Next: Vercel (frontend only)

After local works, the dashboard deploys to Vercel with `VITE_API_BASE` pointing at your
public API. See [DEPLOY.md](./DEPLOY.md).

## Model

| Item | Detail |
|---|---|
| Architecture | ResUNet (single-date land-cover segmentation) |
| Checkpoint | `landcover_train/train/resunet_oscd_enhanced_best.pt` |
| Inputs | 10 channels: B04, B03, B02, B08, B11, B12, NDVI, NDWI, NDBI, BSI |
| Classes | Other, Water, Urban, Forest, Agriculture, Barren |
| Held-out test (enhanced labels + TTA) | ~**95.8%** pixel accuracy / **0.755** mIoU |

Labels used for training are **spectral pseudo-labels** cleaned with OSCD change masks — useful for demos and research bootstrapping, not surveyed ground truth. See `landcover_train/README.md` and `landcover_train/MODEL_IMPROVEMENTS.md`.

### Rebuild enhanced features (if needed)

Large enhanced `.npy` caches may be gitignored. Rebuild after clone:

```bash
python landcover_train/build_oscd_enhanced_dataset.py
```

The API can also build features on the fly from raw OSCD `imgs_*_rect` folders when cache files are missing (slower on first run).

### Train / evaluate

```bash
python landcover_train/train_oscd_enhanced.py
python landcover_train/evaluate_oscd_enhanced.py --tta
python landcover_train/lulc_transition.py --aoi mumbai
```

## Areas of interest (24 OSCD cities)

**Train (14):** aguasclaras, bercy, bordeaux, nantes, paris, rennes, saclay_e, abudhabi, cupertino, pisa, beihai, hongkong, beirut, mumbai  

**Test (10):** brasilia, montpellier, norcia, rio, saclay_w, valencia, dubai, lasvegas, milano, chongqing

## API reference

| Method | Path | Description |
|---|---|---|
| `GET` | `/status` | Health / study metadata |
| `GET` | `/aois` | Train/test AOI lists |
| `GET` | `/predict/transition/{aoi}` | Run LULC transition (`use_oscd_mask`, `tta` query flags) |
| `POST` | `/predict/transition` | Upload t1/t2 enhanced `.npy` tensors |

Artifacts (PNG maps, JSON) are served under `/uploads/transitions/...`.

## Dashboard features

- All **24** OSCD AOIs in the selector (train + test groups)
- Per-class **t1 vs t2** land-cover percentages
- Transition summary (e.g. Forest→Urban) and preview maps
- Online/offline API status banner

## Deploy

See **[DEPLOY.md](./DEPLOY.md)** for free hosting options (Hugging Face Spaces + Vercel recommended).

## Notes / limitations

- Transition percentages are **model estimates**, not official land-cover surveys.
- **Barren** remains a weak class (few stable pseudo-label pixels).
- Backend must be running at `http://127.0.0.1:8000` (or `VITE_API_BASE`) or the UI will show offline / failed-to-fetch.

## License / data

OSCD imagery and change labels belong to their original authors / distributors. Use and cite the OSCD dataset according to its terms. This repository’s application code is provided for academic / project use unless otherwise stated.

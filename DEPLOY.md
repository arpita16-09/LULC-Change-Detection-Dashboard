# Deploy LULC Transition Dashboard (free)

This repo has:
- **Backend**: FastAPI (`backend/`) with LULC transition API
- **Frontend**: React/Vite (`lulc-dashboard/`)
- **Model**: `landcover_train/train/resunet_oscd_enhanced_best.pt`

## What the agent already did
- LULC transition pipeline (t1/t2 land-cover → from→to stats + maps)
- API endpoints (`/predict/transition/{aoi}`, `/aois`, …)
- Dashboard wired to real transition results
- Results generated for test AOIs under `landcover_train/transition_results/`

## What YOU must do (accounts / clicks)

### 1) Merge or use the feature branch
Branch: `main`

Locally:
```bash
git fetch origin
git checkout main
git lfs pull
python landcover_train/build_oscd_enhanced_dataset.py   # rebuilds enhanced .npy (gitignored)
```

### 2) Run locally (to see it now)
```bash
# terminal A — API
cd backend
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# terminal B — UI
cd lulc-dashboard
npm install
npm run dev
```
Open http://localhost:5173 → choose an AOI → **Run LULC Transition**.

### 3) Free public deploy (recommended)

**A. Backend on Hugging Face Spaces (Docker)**  
1. Create a free account at https://huggingface.co  
2. Create a new **Space** → SDK **Docker**  
3. Upload/copy the repo (or connect GitHub)  
4. Ensure Space can see model + `landcover_train/enhanced` features (run build script in Dockerfile or include prebuilt features)  
5. Set Space secret/env if needed: `CORS_ALLOW_ALL=1`  
6. After build, copy the Space URL, e.g. `https://YOURNAME-lulc-api.hf.space`

**B. Frontend on Vercel**  
1. Create account at https://vercel.com  
2. Import the GitHub repo  
3. Root directory: `lulc-dashboard`  
4. Framework: Vite  
5. Environment variable:  
   `VITE_API_BASE=https://YOURNAME-lulc-api.hf.space`  
6. Deploy → open the Vercel URL

### 4) Alternative free backend: Render
1. https://render.com → New Web Service → connect GitHub  
2. Root: `backend`  
3. Build: `pip install -r requirements.txt`  
4. Start: `uvicorn main:app --host 0.0.0.0 --port $PORT`  
5. Free tier sleeps when idle; cold start may be slow with PyTorch

## API quick reference
- `GET /aois`
- `GET /predict/transition/{aoi}`
- `GET /status`
- Artifacts served under `/uploads/transitions/...`

## Notes
- Enhanced `.npy` features are gitignored (large). Always run `build_oscd_enhanced_dataset.py` after clone.
- Transition “change %” uses predicted LULC disagreement, optionally intersected with OSCD official change masks.
- Labels are still pseudo-labels for LULC classes; treat transitions as model estimates, not surveyed ground truth.

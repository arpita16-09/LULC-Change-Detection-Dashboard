#!/usr/bin/env bash
# Local API server (FastAPI + uvicorn) on http://127.0.0.1:8000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/backend"
python -m pip install -r requirements.txt
exec python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

#!/usr/bin/env bash
# Local dashboard (Vite) — expects API at http://127.0.0.1:8000
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/lulc-dashboard"
export VITE_API_BASE="${VITE_API_BASE:-http://127.0.0.1:8000}"
npm install
exec npm run dev -- --host 127.0.0.1 --port 5173

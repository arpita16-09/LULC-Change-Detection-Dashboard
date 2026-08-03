# Local dashboard (Vite) — expects API at http://127.0.0.1:8000
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location (Join-Path $Root "lulc-dashboard")
if (-not $env:VITE_API_BASE) { $env:VITE_API_BASE = "http://127.0.0.1:8000" }
npm install
npm run dev -- --host 127.0.0.1 --port 5173

@echo off
setlocal
cd /d "%~dp0"

echo [ClipGenius] Starting sidecar...
set CLIPGENIUS_TOKEN=dev-local-token
set CLIPGENIUS_PORT=8089

if not exist ".venv\Scripts\python.exe" (
    echo Error: .venv not found. Run scripts\setup.bat first.
    exit /b 1
)

start "ClipGenius Sidecar" /min .venv\Scripts\python.exe -m uvicorn pipeline.server:app --host 127.0.0.1 --port %CLIPGENIUS_PORT%

echo [ClipGenius] Sidecar listening on http://127.0.0.1:%CLIPGENIUS_PORT%
echo [ClipGenius] Token: %CLIPGENIUS_TOKEN%
echo [ClipGenius] Health check: curl -H "X-ClipGenius-Token: %CLIPGENIUS_TOKEN%" http://127.0.0.1:%CLIPGENIUS_PORT%/health

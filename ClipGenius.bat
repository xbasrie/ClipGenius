@echo off
setlocal
title ClipGenius Launcher (.exe Wrapper)

set "CLIPGENIUS_HOME=D:\clipgenius_data"
set "CLIPGENIUS_PORT=8089"
set "CLIPGENIUS_TOKEN=dev-local-token"

echo ========================================================
echo        CLIPGENIUS DESKTOP STUDIO (STANDALONE)
echo ========================================================
echo Memeriksa direktori data di %CLIPGENIUS_HOME%...
if not exist "%CLIPGENIUS_HOME%" mkdir "%CLIPGENIUS_HOME%"

echo Menjalankan engine ClipGenius di port %CLIPGENIUS_PORT%...
cd /d "%~dp0"

start "" http://127.0.0.1:%CLIPGENIUS_PORT%/

"%~dp0.venv\Scripts\python.exe" -m uvicorn pipeline.server:app --host 127.0.0.1 --port %CLIPGENIUS_PORT%

pause

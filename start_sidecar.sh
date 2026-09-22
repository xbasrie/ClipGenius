#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

export CLIPGENIUS_TOKEN="dev-local-token"
export CLIPGENIUS_PORT="${CLIPGENIUS_PORT:-8089}"

echo "[ClipGenius] Starting sidecar on http://127.0.0.1:${CLIPGENIUS_PORT}..."
exec ./.venv/Scripts/python.exe -m uvicorn pipeline.server:app --host 127.0.0.1 --port "${CLIPGENIUS_PORT}"

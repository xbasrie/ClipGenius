"""Integration test for pipeline/server.py via httpx TestClient (in-process).

Runs without network, exercises the REST API surface end-to-end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ["CLIPGENIUS_TOKEN"] = "test-secret-token"

from fastapi.testclient import TestClient  # noqa: E402
from pipeline import config, db  # noqa: E402
from pipeline.server import app  # noqa: E402

client = TestClient(app)
AUTH = {"X-ClipGenius-Token": "test-secret-token"}


def test_auth_rejection():
    r = client.get("/health")
    assert r.status_code == 401


def test_health():
    r = client.get("/health", headers=AUTH)
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert "llm" in data
    assert "ffmpeg" in data


def test_presets():
    r = client.get("/presets", headers=AUTH)
    assert r.status_code == 200
    p = r.json()["presets"]
    assert "classic_white" in p
    assert "creator_pop" in p


def test_create_and_query_job(tmp_path):
    # use the smoke fixture as an input file
    src = ROOT / "fixtures" / "smoke_out" / "source.mp4"
    if not src.exists():
        return  # skipped if smoke not run yet

    r = client.post("/jobs", headers=AUTH, json={
        "file_path": str(src),
        "clip_max_s": 30,
        "clip_count": 2,
        "reframe": False,  # fast test
        "hook_overlay": False,
    })
    assert r.status_code == 200
    jid = r.json()["job_id"]

    r = client.get(f"/jobs/{jid}", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["job"]["state"] in ("queued", "processing", "completed")

    r = client.get("/jobs", headers=AUTH)
    assert r.status_code == 200
    assert any(j["id"] == jid for j in r.json()["jobs"])


def test_settings():
    r = client.put("/settings/test_key", headers=AUTH, json={"foo": 42})
    assert r.status_code == 200
    r = client.get("/settings/test_key", headers=AUTH)
    assert r.status_code == 200
    assert r.json()["value"] == {"foo": 42}


if __name__ == "__main__":
    test_auth_rejection()
    test_health()
    test_presets()
    test_settings()
    test_create_and_query_job(Path())
    print("ALL SERVER TESTS PASSED")
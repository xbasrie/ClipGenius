"""Persistent store: SQLite (WAL). Replaces OpenShorts' in-memory dicts -> restart-safe."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from . import config

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    duration_s REAL NOT NULL DEFAULT 0,
    meta TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    project_id TEXT NOT NULL,
    state TEXT NOT NULL,
    stage TEXT NOT NULL DEFAULT '',
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    options TEXT NOT NULL DEFAULT '{}',
    bookmark TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    idx INTEGER NOT NULL,
    start_s REAL NOT NULL,
    end_s REAL NOT NULL,
    trim_start_s REAL,
    trim_end_s REAL,
    score REAL NOT NULL DEFAULT 0,
    hook TEXT NOT NULL DEFAULT '',
    title TEXT NOT NULL DEFAULT '',
    transcript TEXT NOT NULL DEFAULT '{}',
    subtitle TEXT NOT NULL DEFAULT '{}',
    included INTEGER NOT NULL DEFAULT 1,
    file TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS transcript_cache (
    key TEXT PRIMARY KEY, payload TEXT NOT NULL, created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id);
CREATE INDEX IF NOT EXISTS idx_clips_job ON clips(job_id, idx);
"""


def connect() -> sqlite3.Connection:
    global _conn
    with _lock:
        if _conn is None:
            config.ensure_dirs()
            _conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
            _conn.row_factory = sqlite3.Row
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.execute("PRAGMA synchronous=NORMAL")
            _conn.executescript(SCHEMA)
            _conn.commit()
        return _conn


def close() -> None:
    global _conn
    with _lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def new_id(prefix: str = "") -> str:
    return f"{prefix}{uuid.uuid4().hex[:12]}"


def execute(sql: str, params: tuple = ()) -> sqlite3.Cursor:
    with _lock:
        cur = connect().execute(sql, params)
        connect().commit()
        return cur


def query(sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    with _lock:
        rows = connect().execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def one(sql: str, params: tuple = ()) -> dict[str, Any] | None:
    rows = query(sql, params)
    return rows[0] if rows else None


# ---------- projects ----------

def create_project(source: str, source_kind: str, title: str = "", duration_s: float = 0.0,
                   meta: dict | None = None) -> str:
    pid = new_id("prj_")
    execute(
        "INSERT INTO projects (id,title,source,source_kind,duration_s,meta,created_at)"
        " VALUES (?,?,?,?,?,?,?)",
        (pid, title, source, source_kind, duration_s, json.dumps(meta or {}), time.time()),
    )
    return pid


def get_project(pid: str) -> dict | None:
    row = one("SELECT * FROM projects WHERE id=?", (pid,))
    if row:
        row["meta"] = json.loads(row["meta"])
    return row


def list_projects(limit: int = 50) -> list[dict]:
    rows = query("SELECT * FROM projects ORDER BY created_at DESC LIMIT ?", (limit,))
    for r in rows:
        r["meta"] = json.loads(r["meta"])
    return rows


# ---------- jobs ----------

def create_job(project_id: str, options: dict) -> str:
    jid = new_id("job_")
    now = time.time()
    execute(
        "INSERT INTO jobs (id,project_id,state,stage,progress,message,error,options,bookmark,"
        "created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (jid, project_id, "queued", "", 0.0, "Menunggu antrean", "", json.dumps(options),
         "{}", now, now),
    )
    return jid


def update_job(jid: str, **fields) -> None:
    if not fields:
        return
    allowed = {"state", "stage", "progress", "message", "error", "bookmark", "options"}
    parts, params = [], []
    for k, v in fields.items():
        if k not in allowed:
            continue
        parts.append(f"{k}=?")
        params.append(json.dumps(v) if k in {"bookmark", "options"} else v)
    if not parts:
        return
    parts.append("updated_at=?")
    params.append(time.time())
    params.append(jid)
    execute(f"UPDATE jobs SET {','.join(parts)} WHERE id=?", tuple(params))


def get_job(jid: str) -> dict | None:
    row = one("SELECT * FROM jobs WHERE id=?", (jid,))
    if row:
        row["bookmark"] = json.loads(row["bookmark"])
        row["options"] = json.loads(row["options"])
    return row


def list_jobs(limit: int = 50, project_id: str | None = None) -> list[dict]:
    if project_id:
        rows = query("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at DESC LIMIT ?",
                     (project_id, limit))
    else:
        rows = query("SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,))
    for r in rows:
        r["bookmark"] = json.loads(r["bookmark"])
        r["options"] = json.loads(r["options"])
    return rows


def recoverable_jobs() -> list[dict]:
    """Jobs killed mid-flight — the resume path."""
    rows = query("SELECT * FROM jobs WHERE state IN ('queued','processing','paused')"
                 " ORDER BY created_at ASC")
    for r in rows:
        r["bookmark"] = json.loads(r["bookmark"])
        r["options"] = json.loads(r["options"])
    return rows


# ---------- clips ----------

def add_clip(job_id: str, idx: int, start_s: float, end_s: float, score: float,
             hook: str, title: str, transcript: dict, file: str = "") -> str:
    cid = new_id("clp_")
    execute(
        "INSERT INTO clips (id,job_id,idx,start_s,end_s,score,hook,title,transcript,file,"
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (cid, job_id, idx, start_s, end_s, score, hook, title, json.dumps(transcript), file,
         time.time()),
    )
    return cid


def list_clips(job_id: str) -> list[dict]:
    rows = query("SELECT * FROM clips WHERE job_id=? ORDER BY idx ASC", (job_id,))
    for r in rows:
        r["transcript"] = json.loads(r["transcript"])
        r["subtitle"] = json.loads(r["subtitle"])
        r["included"] = bool(r["included"])
        r["start_s"] = r["trim_start_s"] if r["trim_start_s"] is not None else r["start_s"]
        r["end_s"] = r["trim_end_s"] if r["trim_end_s"] is not None else r["end_s"]
    return rows


def update_clip(cid: str, **fields) -> None:
    allowed = {"trim_start_s", "trim_end_s", "included", "hook", "subtitle", "title", "file",
               "score"}
    parts, params = [], []
    for k, v in fields.items():
        if k in allowed:
            parts.append(f"{k}=?")
            params.append(json.dumps(v) if k == "subtitle" else v)
    if not parts:
        return
    params.append(cid)
    execute(f"UPDATE clips SET {','.join(parts)} WHERE id=?", tuple(params))


def delete_clips(job_id: str) -> None:
    execute("DELETE FROM clips WHERE job_id=?", (job_id,))


# ---------- settings & cache ----------

def set_setting(key: str, value: Any) -> None:
    execute("INSERT INTO settings (key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET"
            " value=excluded.value", (key, json.dumps(value)))


def get_setting(key: str, default: Any = None) -> Any:
    row = one("SELECT value FROM settings WHERE key=?", (key,))
    return json.loads(row["value"]) if row else default


def cache_get(key: str) -> dict | None:
    row = one("SELECT payload FROM transcript_cache WHERE key=?", (key,))
    return json.loads(row["payload"]) if row else None


def cache_put(key: str, payload: dict) -> None:
    execute("INSERT INTO transcript_cache (key,payload,created_at) VALUES (?,?,?)"
            " ON CONFLICT(key) DO UPDATE SET payload=excluded.payload,"
            " created_at=excluded.created_at", (key, json.dumps(payload), time.time()))


def cache_purge(days: int = 30) -> int:
    cutoff = time.time() - days * 86400
    cur = execute("DELETE FROM transcript_cache WHERE created_at < ?", (cutoff,))
    return cur.rowcount


def workdir_for(job_id: str) -> Path:
    p = config.WORK_DIR / job_id
    p.mkdir(parents=True, exist_ok=True)
    return p
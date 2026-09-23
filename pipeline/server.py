"""ClipGenius sidecar: localhost FastAPI that owns the pipeline and job state.

Bound to 127.0.0.1 only, guarded by a one-time token minted by the Electron main
process (env CLIPGENIUS_TOKEN). Single-threaded worker queue — heavier concurrency
just fights over the GPU/CPU anyway.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from . import config, db
from .modulos import clip as clip_mod
from .modulos import curation, export as export_mod, media, stt, subtitle as sub_mod
from .providers import llm

config.ensure_dirs()

logger = logging.getLogger("clipgenius")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.FileHandler(config.LOG_DIR / "sidecar.log", encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(_h)
    logger.addHandler(logging.StreamHandler())

TOKEN = os.environ.get("CLIPGENIUS_TOKEN", "")
_jobs: "queue.Queue[str]" = queue.Queue()
_WORKER: threading.Thread | None = None
_STOP = threading.Event()

STAGES = ["download", "transcribe", "curate", "clip", "subtitle", "review"]


# --------------------------------------------------------------------------- auth

def require_token(request: Request, x_clipgenius_token: str = Header(default="")) -> None:
    if not TOKEN:
        return
    # Public routes for browser / health check / docs
    if request.url.path in ("/", "/health", "/docs", "/openapi.json", "/favicon.ico"):
        return
    if x_clipgenius_token != TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing sidecar token")


app = FastAPI(title="ClipGenius Sidecar", version="0.1.0", dependencies=[Depends(require_token)])


# --------------------------------------------------------------------------- models

class CreateJob(BaseModel):
    url: str | None = None
    file_path: str | None = None
    clip_max_s: int = Field(default=config.DEFAULT_CLIP_MAX_S, ge=5, le=120)
    clip_count: int = Field(default=config.DEFAULT_CLIP_COUNT, ge=1, le=15)
    model_tier: str = Field(default="fast")
    language: str | None = None
    aspect: str = Field(default="9:16")
    resolution: str = Field(default="1080p")
    subtitle_preset: str = Field(default="classic_white")
    hook_overlay: bool = True
    reframe: bool = True
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_key: str | None = None
    llm_base_url: str | None = None


class ClipPatch(BaseModel):
    trim_start_s: float | None = None
    trim_end_s: float | None = None
    included: bool | None = None
    hook: str | None = None
    title: str | None = None
    subtitle_preset: str | None = None
    subtitle_style: dict | None = None
    subtitle_edits: list[dict] | None = None


class ExportRequest(BaseModel):
    output_dir: str
    aspect: str = "9:16"
    resolution: str = "1080p"
    clip_ids: list[str] | None = None


# --------------------------------------------------------------------------- job runner

def _progress(job_id: str, stage: str, pct: float, msg: str = "") -> None:
    stages_done = STAGES.index(stage) if stage in STAGES else 0
    overall = (stages_done / len(STAGES)) * 100 + (pct / 100) * (100 / len(STAGES))
    db.update_job(job_id, state="processing", stage=stage,
                  progress=round(min(99.5, overall), 1), message=msg or stage)


def run_job(job_id: str) -> None:
    job = db.get_job(job_id)
    if not job or job.get("state") in ("failed", "canceled", "completed"):
        return
    opts = job["options"]
    work = db.workdir_for(job_id)
    project = db.get_project(job["project_id"]) or {}
    bookmark = job["bookmark"]
    done = set(bookmark.get("stages_done", []))

    def _check_canceled() -> None:
        j = db.get_job(job_id)
        if not j:
            raise RuntimeError("Job tidak ditemukan")
        if j.get("state") == "failed":
            raise RuntimeError("Job dibatalkan oleh pengguna")
        if j.get("state") == "paused":
            raise RuntimeError("Job dijeda oleh pengguna")

    try:
        db.update_job(job_id, state="processing", error="")

        # ---- download ------------------------------------------------------
        video_path = Path(bookmark.get("video_path", "")) if bookmark.get("video_path") else None
        if "download" not in done or not (video_path and video_path.exists()):
            src_url, src_file = opts.get("url"), opts.get("file_path")
            if src_file:
                video_path = Path(src_file)
                title = opts.get("title") or video_path.stem
                duration = media.get_duration(video_path)
                meta = {}
            else:
                _progress(job_id, "download", 5, "Membaca metadata video")
                _check_canceled()
                meta = media.get_video_info(src_url)
                title = meta.get("title") or "video"
                duration = float(meta.get("duration") or 0) or 0.0
                _progress(job_id, "download", 15, f"Mengunduh: {title[:60]}")
                def _dl_cb(pct: float, speed: str, eta: float | None) -> None:
                    _check_canceled()
                    _progress(job_id, "download", pct, f"Unduh {pct:.0f}% ({speed or '-'})")

                video_path = media.download_video(src_url, work, progress_cb=_dl_cb)
                duration = duration or media.get_duration(video_path)
            db.execute(
                "UPDATE projects SET title=?, duration_s=?, meta=? WHERE id=?",
                (title, duration, json.dumps(meta), job["project_id"]),
            )
            bookmark.update({"video_path": str(video_path), "title": title,
                             "duration": duration, "meta": meta})
            db.update_job(job_id, bookmark=bookmark)
            done.add("download")

        title = bookmark.get("title") or "video"
        duration = float(bookmark.get("duration") or media.get_duration(video_path))

        # ---- transcribe ----------------------------------------------------
        transcript = bookmark.get("transcript")
        if "transcribe" not in done or not transcript:
            _check_canceled()
            src_url = opts.get("url")
            if src_url:
                _progress(job_id, "transcribe", 10, "Mengecek subtitle bawaan YouTube...")
                transcript = media.fetch_youtube_transcript(src_url, work)
                if transcript and transcript.get("segments"):
                    logger.info("Pakai subtitle bawaan YouTube untuk %s (%d segments)",
                                job_id, len(transcript["segments"]))
                    _progress(job_id, "transcribe", 100,
                              f"Menggunakan subtitle YouTube ({len(transcript['segments'])} baris)")
            if not transcript:
                _progress(job_id, "transcribe", 2, "Mentranskripsi audio dengan Whisper")
                def _stt_cb(pct: float, msg: str | None = None) -> None:
                    _check_canceled()
                    _progress(job_id, "transcribe", pct, msg or f"Transkripsi ({pct:.0f}%)")

                transcript = stt.transcribe(
                    video_path, model_tier=opts.get("model_tier", "fast"),
                    language=opts.get("language"),
                    progress_cb=_stt_cb,
                )
            bookmark["transcript"] = transcript
            db.update_job(job_id, bookmark=bookmark)
            done.add("transcribe")

        # ---- curate --------------------------------------------------------
        plan = bookmark.get("plan")
        if "curate" not in done or not plan:
            _progress(job_id, "curate", 10, "AI memilih momen terbaik")
            cfg = config.LLMConfig.load()
            if opts.get("llm_provider"):
                cfg = config.LLMConfig(
                    provider=opts["llm_provider"],
                    model=opts.get("llm_model") or cfg.model,
                    api_key=opts.get("llm_key") or cfg.api_key,
                    base_url=opts.get("llm_base_url") or cfg.base_url,
                )
            plan = curation.curate_clips(
                transcript, duration, clip_max_s=opts.get("clip_max_s", 30),
                clip_count=opts.get("clip_count", 5), llm_config=cfg,
            )
            bookmark["plan"] = plan
            db.update_job(job_id, bookmark=bookmark)
            done.add("curate")

        shorts = plan.get("shorts") or []
        if not shorts:
            raise RuntimeError("Tidak ada klip yang bisa dibuat dari video ini")

        # ---- clips + reframe + subtitles -----------------------------------
        clips_meta = bookmark.get("clips")
        if "clip" not in done or not clips_meta:
            db.delete_clips(job_id)
            clips_meta = []
            reframed = None
            if opts.get("reframe", True) and opts.get("aspect", "9:16") == "9:16":
                _progress(job_id, "clip", 5, "Reframe ke 9:16 (tracking subjek)")
                reframed = work / "reframed.mp4"
                if not reframed.exists():
                    strat = clip_mod.scene_strategy_for(video_path)
                    clip_mod.reframe_to_vertical(
                        video_path, reframed, scene_strategy=strat,
                        progress_cb=lambda pct: _progress(job_id, "clip", pct * 100,
                                                          "Reframe 9:16"),
                    )
                bookmark["reframed"] = str(reframed)

            source_for_cut = reframed or video_path
            for i, s in enumerate(shorts, start=1):
                _progress(job_id, "clip", (i / len(shorts)) * 100,
                          f"Memotong klip {i}/{len(shorts)}")
                cut = work / f"clip_{i}.mp4"
                clip_mod.cut_clip(source_for_cut, s["start"], s["end"], cut)
                if opts.get("hook_overlay", True) and s.get("hook"):
                    hooked = work / f"clip_{i}_hook.mp4"
                    clip_mod.text_overlay(cut, s["hook"], hooked)
                    cut = hooked
                clips_meta.append({"idx": i, "start_s": s["start"], "end_s": s["end"],
                                   "score": s.get("score", 0), "hook": s.get("hook", ""),
                                   "title": s.get("title", ""), "file": str(cut)})
            bookmark["clips"] = clips_meta
            db.update_job(job_id, bookmark=bookmark)
            done.add("clip")

        # persist clip rows (idempotent)
        if not db.list_clips(job_id):
            for cm in clips_meta:
                cid = db.add_clip(job_id, cm["idx"], cm["start_s"], cm["end_s"],
                                  cm["score"], cm["hook"], cm["title"],
                                  _slice_transcript(transcript, cm["start_s"], cm["end_s"]),
                                  cm["file"])
                srt = work / f"clip_{cm['idx']}.srt"
                tc = {"segments": _slice_transcript(transcript, cm["start_s"], cm["end_s"])}
                if sub_mod.generate_srt(tc, cm["start_s"], cm["end_s"], srt):
                    db.update_clip(cid, subtitle={"srt": str(srt),
                                                  "preset": opts.get("subtitle_preset",
                                                                     "classic_white")})
                thumb = work / f"thumb_{cm['idx']}.jpg"
                if clip_mod.make_thumbnail(cm["file"], thumb):
                    db.update_clip(cid, subtitle={**({"srt": str(srt)} if srt.exists() else {}),
                                                  "thumb": str(thumb),
                                                  "preset": opts.get("subtitle_preset",
                                                                     "classic_white")})

        _progress(job_id, "review", 100, "Siap ditinjau")
        bookmark["stages_done"] = sorted(done | {"clip", "subtitle", "review"})
        db.update_job(job_id, state="completed", stage="review", progress=100,
                      message=f"{len(clips_meta)} klip siap ditinjau", bookmark=bookmark)
        logger.info("job %s completed with %d clips", job_id, len(clips_meta))

    except Exception as e:  # noqa: BLE001 — job boundary: record and move on
        current = db.get_job(job_id)
        if current and current.get("state") == "paused":
            logger.info("job %s paused cleanly", job_id)
            return
        logger.exception("job %s failed", job_id)
        db.update_job(job_id, state="failed", error=str(e), message=f"Gagal: {e}")


def _slice_transcript(transcript: dict, start_s: float, end_s: float) -> list[dict]:
    out = []
    for seg in transcript.get("segments", []):
        words = [w for w in seg.get("words", []) if w["end"] > start_s and w["start"] < end_s]
        if words or (seg["end"] > start_s and seg["start"] < end_s):
            out.append({"start": seg["start"], "end": seg["end"],
                        "text": seg.get("text", ""), "words": words})
    return out


# --------------------------------------------------------------------------- worker

def _worker_loop() -> None:
    while not _STOP.is_set():
        try:
            job_id = _jobs.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            run_job(job_id)
        finally:
            _jobs.task_done()


def start_worker() -> None:
    global _WORKER
    if _WORKER and _WORKER.is_alive():
        return
    _STOP.clear()
    _WORKER = threading.Thread(target=_worker_loop, name="clipgenius-worker", daemon=True)
    _WORKER.start()
    for job in db.recoverable_jobs():
        db.update_job(job["id"], state="queued", message="Dipulihkan setelah restart")
        _jobs.put(job["id"])
    logger.info("worker started; %d job dipulihkan", _jobs.qsize())


@app.on_event("startup")
def _startup() -> None:
    db.connect()
    start_worker()


@app.on_event("shutdown")
def _shutdown() -> None:
    _STOP.set()


# --------------------------------------------------------------------------- endpoints

@app.get("/")
def root():
    index_path = Path(__file__).resolve().parent / "static" / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "app": "ClipGenius Sidecar API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "jobs": "/jobs",
            "presets": "/presets"
        }
    }


@app.get("/debug/threads")
def debug_threads():
    import sys, traceback
    return {str(th_id): "".join(traceback.format_stack(frame))
            for th_id, frame in sys._current_frames().items()}


@app.get("/health")
def health() -> dict:
    import sys, traceback
    stacks = {th.name: "".join(traceback.format_stack(sys._current_frames()[th.ident]))
              for th in threading.enumerate() if th.ident in sys._current_frames()}
    return {"ok": True, "worker_alive": bool(_WORKER and _WORKER.is_alive()),
            "queued": _jobs.qsize(), "llm": llm.provider_status(),
            "ffmpeg": config.ffmpeg(), "thread_stacks": stacks}


@app.get("/settings/{key}")
def read_setting(key: str) -> dict:
    return {"key": key, "value": db.get_setting(key)}


@app.put("/settings/{key}")
async def write_setting(key: str, request: Request) -> dict:
    db.set_setting(key, await request.json())
    return {"ok": True}


@app.post("/jobs")
def create_job(payload: CreateJob) -> dict:
    if not payload.url and not payload.file_path:
        raise HTTPException(400, "url atau file_path wajib diisi")
    opts = payload.model_dump()
    if payload.url:
        pid = db.create_project(payload.url, "url")
    else:
        p = Path(payload.file_path or "")
        if not p.exists():
            raise HTTPException(400, f"File tidak ditemukan: {p}")
        pid = db.create_project(str(p), "file", duration_s=media.get_duration(p))
    jid = db.create_job(pid, opts)
    _jobs.put(jid)
    return {"job_id": jid, "project_id": pid, "state": "queued"}


@app.get("/jobs")
def list_jobs(limit: int = 50, project_id: str | None = None) -> dict:
    return {"jobs": db.list_jobs(limit, project_id)}


@app.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    return {"job": job, "project": db.get_project(job["project_id"])}


@app.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    db.update_job(job_id, state="queued", message="Dipulihkan")
    _jobs.put(job_id)
    return {"ok": True, "job_id": job_id}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    db.delete_clips(job_id)
    db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
    work = config.WORK_DIR / job_id
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    return {"ok": True}


@app.get("/jobs/{job_id}/clips")
def get_clips(job_id: str) -> dict:
    return {"clips": db.list_clips(job_id)}


@app.patch("/clips/{clip_id}")
def patch_clip(clip_id: str, payload: ClipPatch) -> dict:
    fields = payload.model_dump(exclude_none=True)
    edits = fields.pop("subtitle_edits", None)
    preset = fields.pop("subtitle_preset", None)
    style = fields.pop("subtitle_style", None)

    if fields:
        db.update_clip(clip_id, **fields)

    stored = db.one("SELECT subtitle FROM clips WHERE id=?", (clip_id,))
    sub = json.loads(stored["subtitle"]) if stored and stored["subtitle"] else {}
    if preset:
        sub["preset"] = preset
    if style:
        sub["style"] = style
    if edits is not None:
        srt = sub.get("srt")
        if srt and Path(srt).exists():
            _apply_subtitle_edits(Path(srt), edits)
    if preset or style or edits is not None:
        db.update_clip(clip_id, subtitle=sub)

    row = db.one("SELECT * FROM clips WHERE id=?", (clip_id,))
    if not row:
        raise HTTPException(404, "Klip tidak ditemukan")
    return {"ok": True}


def _apply_subtitle_edits(srt_path: Path, edits: list[dict]) -> None:
    """edits: [{'index': 1, 'text': 'new text'}] — rewrite the SRT in place."""
    blocks = srt_path.read_text(encoding="utf-8").strip().split("\n\n")
    want = {int(e["index"]): str(e["text"]) for e in edits if "index" in e}
    out = []
    for b in blocks:
        lines = b.split("\n")
        if len(lines) >= 3 and lines[0].strip().isdigit():
            i = int(lines[0].strip())
            if i in want:
                lines = [lines[0], lines[1], want[i]]
        out.append("\n".join(lines))
    srt_path.write_text("\n\n".join(out) + "\n", encoding="utf-8")


@app.get("/clips/{clip_id}/srt")
def get_srt(clip_id: str) -> dict:
    row = db.one("SELECT subtitle FROM clips WHERE id=?", (clip_id,))
    if not row:
        raise HTTPException(404, "Klip tidak ditemukan")
    sub = json.loads(row["subtitle"]) if row["subtitle"] else {}
    srt = sub.get("srt")
    if not srt or not Path(srt).exists():
        raise HTTPException(404, "SRT belum dibuat")
    blocks = [b for b in Path(srt).read_text(encoding="utf-8").strip().split("\n\n") if b]
    cues = []
    for b in blocks:
        lines = b.split("\n")
        if len(lines) >= 3:
            cues.append({"index": int(lines[0]), "time": lines[1], "text": "\n".join(lines[2:])})
    return {"srt_path": srt, "cues": cues, "preset": sub.get("preset", "classic_white")}


@app.get("/media/{job_id}/{name}")
def serve_media(job_id: str, name: str) -> FileResponse:
    path = (db.workdir_for(job_id) / name).resolve()
    if not str(path).startswith(str(config.WORK_DIR.resolve())) or not path.exists():
        raise HTTPException(404, "Tidak ditemukan")
    return FileResponse(str(path))


@app.post("/jobs/{job_id}/export")
def export_job(job_id: str, payload: ExportRequest) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    clips = db.list_clips(job_id)
    if payload.clip_ids:
        clips = [c for c in clips if c["id"] in set(payload.clip_ids)]

    project = db.get_project(job["project_id"]) or {}
    title = project.get("title") or "video"
    bookmark = job["bookmark"]
    reframed = bookmark.get("reframed")

    items = []
    for c in clips:
        sub = c.get("subtitle") or {}
        items.append({**c, "source_video": bookmark.get("video_path"),
                      "srt_path": sub.get("srt"),
                      "subtitle_preset": sub.get("preset", "classic_white"),
                      "subtitle_style": sub.get("style")})

    results = export_mod.export_batch(
        items, payload.output_dir, title, aspect=payload.aspect,
        resolution=payload.resolution, reframed_source=reframed,
    )
    ok = [r for r in results if r["file"]]
    return {"exported": len(ok), "failed": len(results) - len(ok), "results": results}


@app.post("/jobs/{job_id}/pause")
def pause_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    if job["state"] in ("processing", "queued"):
        db.update_job(job_id, state="paused", message="Dijeda oleh pengguna")
    return {"ok": True}


@app.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job tidak ditemukan")
    db.update_job(job_id, state="failed", error="Dibatalkan oleh pengguna", message="Dibatalkan")
    return {"ok": True}


@app.get("/presets")
def presets() -> dict:
    return {"presets": sub_mod.PRESETS, "aspects": list(export_mod.ASPECTS),
            "resolutions": export_mod.RESOLUTIONS}


@app.exception_handler(Exception)
async def unhandled(_request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": str(exc)})
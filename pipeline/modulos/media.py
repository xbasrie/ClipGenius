"""FFmpeg / ffprobe / yt-dlp helpers. All subprocess calls go through config."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

from .. import config


class MediaError(RuntimeError):
    pass


def _run(cmd: list[str], *, timeout_s: int | None = None) -> str:
    """Run a command, return stdout; raise MediaError with stderr tail on failure."""
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    except FileNotFoundError as e:
        raise MediaError(f"Binary not found: {cmd[0]} ({e})") from e
    except subprocess.TimeoutExpired as e:
        raise MediaError(f"Timed out after {timeout_s}s: {' '.join(cmd[:4])} …") from e
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()
        raise MediaError(f"Command failed ({proc.returncode}): {' '.join(cmd[:4])} …\n{err[-800:]}")
    return proc.stdout.decode("utf-8", "replace")


# ---------- download ----------

def download_video(url: str, output_dir: Path,
                   progress_cb: Callable[[float, str, float], None] | None = None) -> Path:
    """Download to mp4 via yt-dlp; progress_cb(percent, speed_str, eta_seconds)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outtmpl = str(output_dir / "source.%(ext)s")
    cmd = [
        *config.ytdlp(),
        "--format", "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]/b",
        "--merge-output-format", "mp4",
        "--recode-video", "mp4",
        "--newline",
        "--no-playlist",
        "--quiet",
        "--progress",
        "--progress-template",
        "download:PROG %(progress._percent_str)s %(progress._speed_str)s %(progress._eta_str)s",
        "-o", outtmpl,
        url,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace", bufsize=1)
    assert proc.stdout is not None and proc.stderr is not None
    for line in proc.stdout:
        line = line.strip()
        if progress_cb and line.startswith("PROG "):
            parts = line[5:].split()
            pct = _parse_percent(parts[0]) if parts else None
            speed = parts[1] if len(parts) > 1 else ""
            eta = _parse_eta(parts[2]) if len(parts) > 2 else None
            if pct is not None:
                progress_cb(pct, speed, eta)
    if progress_cb:
        progress_cb(100.0, "", 0.0)
    proc.wait()
    if proc.returncode != 0:
        err = (proc.stderr.read() or "").strip()[-800:]
        raise MediaError(f"yt-dlp failed ({url}): {err}")

    candidates = sorted(output_dir.glob("source.*"),
                        key=lambda p: p.stat().st_size, reverse=True)
    videos = [p for p in candidates
              if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov", ".m4v"}]
    if not videos:
        raise MediaError(f"yt-dlp finished but no video file found in {output_dir}")
    result = videos[0]
    if result.suffix.lower() != ".mp4":
        # ponytail: remux non-mp4 to mp4 here; upgrade path -> codec-aware remux
        mp4 = result.with_suffix(".mp4")
        _run([config.ffmpeg(), "-y", "-i", str(result), "-c", "copy", str(mp4)], timeout_s=600)
        result.unlink(missing_ok=True)
        result = mp4
    return result


def _parse_percent(s: str) -> float | None:
    s = s.strip().replace("%", "").replace("n/a", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_eta(s: str) -> float | None:
    """'00:05:23' or '01:23' -> seconds."""
    parts = s.strip().split(":")
    if not all(p.isdigit() for p in parts):
        return None
    secs = 0.0
    for p in parts:
        secs = secs * 60 + int(p)
    return secs


def get_video_info(url: str) -> dict:
    """title, duration, thumbnail_url, channel via yt-dlp --dump-json."""
    cmd = [*config.ytdlp(), "--dump-json", "--no-playlist", "--skip-download", url]
    raw = _run(cmd, timeout_s=120)
    line = raw.strip().splitlines()[0]
    try:
        data = json.loads(line)
    except json.JSONDecodeError as e:
        raise MediaError(f"yt-dlp --dump-json returned invalid JSON for {url}") from e
    return {
        "title": data.get("title") or "",
        "duration": float(data.get("duration") or 0.0),
        "thumbnail_url": data.get("thumbnail") or "",
        "channel": data.get("channel") or data.get("uploader") or "",
    }


# ---------- probe ----------

def get_duration(path: str | Path) -> float:
    out = _run_json_probe(path)
    try:
        dur = float(out["format"]["duration"])
    except (KeyError, ValueError, TypeError) as e:
        raise MediaError(f"ffprobe: no duration for {path}") from e
    return dur


def get_resolution(path: str | Path) -> tuple[int, int]:
    out = _run_json_probe(path)
    streams = out.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise MediaError(f"ffprobe: no video stream in {path}")
    w, h = int(video["width"]), int(video["height"])
    # display aspect-aware height (e.g. anamorphic sources)
    sample = video.get("sample_aspect_ratio", "1:1")
    if sample not in ("1:1", "0:1", ""):
        num, _, den = sample.partition(":")
        try:
            w = int(round(w * float(num) / float(den or 1)))
        except (ValueError, ZeroDivisionError):
            pass
    return w, h


def _run_json_probe(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"File not found: {p}")
    cmd = [
        config.ffprobe(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(p),
    ]
    raw = _run(cmd, timeout_s=60)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise MediaError(f"ffprobe returned invalid JSON for {p}") from e


# ---------- transcode / hash ----------

def extract_audio(video_path: str | Path, audio_path: str | Path) -> Path:
    vp, ap = Path(video_path), Path(audio_path)
    if not vp.is_file():
        raise MediaError(f"Video not found: {vp}")
    ap.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        config.ffmpeg(), "-y",
        "-i", str(vp),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", "16000",
        "-ac", "1",
        str(ap),
    ]
    _run(cmd, timeout_s=1200)
    if not ap.is_file():
        raise MediaError(f"ffmpeg produced no audio file: {ap}")
    return ap


def file_hash(path: str | Path) -> str:
    """sha256 of first 64KB — fast cache key for transcripts."""
    p = Path(path)
    if not p.is_file():
        raise MediaError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with p.open("rb") as f:
        h.update(f.read(64 * 1024))
    return h.hexdigest()

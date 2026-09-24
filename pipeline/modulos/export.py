"""Export engine: batch encode selected clips -> MP4 (9:16 or 16:9, 720p/1080p).

Single FFmpeg pass per clip: cut -> (reframe) -> subtitles -> hook. Everything else
uses `-c:v copy` so we never decode twice.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Callable, Iterable

from ..config import ffmpeg
from . import clip as clip_mod
from . import subtitle as sub_mod
from .media import get_resolution

RESOLUTIONS = {
    "1080p": {"portrait": (1080, 1920), "landscape": (1920, 1080)},
    "720p": {"portrait": (720, 1280), "landscape": (1280, 720)},
}
ASPECTS = ("9:16", "16:9")

_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def sanitize_name(name: str, fallback: str = "clip") -> str:
    cleaned = _ILLEGAL.sub("_", (name or "").strip())
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" .")
    return (cleaned or fallback)[:120]


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(1, 1000):
        cand = path.with_name(f"{path.stem}_{i}{path.suffix}")
        if not cand.exists():
            return cand
    return path


def build_output_name(title: str, index: int, start_s: float, end_s: float, ext: str = "mp4") -> str:
    dur = max(1, round(end_s - start_s))
    return f"{sanitize_name(title, 'video')}_Clip{index:02d}_{dur}s.{ext}"


def export_clip(
    source_video: str | Path,
    start_s: float,
    end_s: float,
    output_path: str | Path,
    *,
    aspect: str = "9:16",
    resolution: str = "1080p",
    srt_path: str | Path | None = None,
    preset: str = "classic_white",
    custom_style: dict | None = None,
    hook_text: str = "",
    reframed_source: str | Path | None = None,
    watermark: dict | None = None,
    progress_cb: Callable[[float], None] | None = None,
) -> Path:
    """Encode one clip. `reframed_source` is an already-9:16 file covering the whole
    timeline (cheaper than reframing per clip when the job produced several)."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if aspect not in ASPECTS:
        raise ValueError(f"aspect must be one of {ASPECTS}, got {aspect!r}")
    if resolution not in RESOLUTIONS:
        raise ValueError(f"resolution must be one of {tuple(RESOLUTIONS)}, got {resolution!r}")

    target_w, target_h = RESOLUTIONS[resolution]["portrait" if aspect == "9:16" else "landscape"]
    duration = max(0.1, end_s - start_s)

    # Stage 1: cut to a temp file using the best available source.
    tmp_cut = output_path.with_suffix(".cut.mp4")
    if aspect == "9:16" and reframed_source and Path(reframed_source).exists():
        clip_mod.cut_clip(reframed_source, start_s, end_s, tmp_cut, reencode=True)
        vf = f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2"
    else:
        clip_mod.cut_clip(source_video, start_s, end_s, tmp_cut, reencode=True)
        if aspect == "9:16":
            vf = (f"crop='min(iw,ih*9/16)':'min(ih,iw*16/9)',"
                  f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase,"
                  f"crop={target_w}:{target_h}")
        else:
            vf = (f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                  f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black")

    # Stage 2: subtitles + hook + watermark, then final encode.
    filters = [vf]
    extra_inputs = []

    if srt_path and Path(srt_path).exists():
        filters.append(_subtitle_filter(srt_path, preset, custom_style))
    if hook_text:
        filters.append(_hook_filter(hook_text))

    # Watermark Handling
    if watermark and watermark.get("enabled"):
        wm_type = watermark.get("type", "text")
        opacity = float(watermark.get("opacity", 0.8))
        pos = watermark.get("position", "top_right")

        # Position mappings for FFmpeg
        pos_map = {
            "top_left": ("40", "40"),
            "top_right": ("w-tw-40", "40"),
            "bottom_left": ("40", "h-th-180"),
            "bottom_right": ("w-tw-40", "h-th-180"),
            "center_top": ("(w-tw)/2", "50"),
        }
        x_expr, y_expr = pos_map.get(pos, ("w-tw-40", "40"))

        if wm_type == "text" and watermark.get("text"):
            raw_text = str(watermark.get("text", "")).strip()
            safe_text = (raw_text.replace("\\", "").replace("'", "")
                         .replace(":", "\\:").replace("%", "").replace(",", "\\,"))
            font_size = int(watermark.get("font_size", 32))
            alpha_hex = f"{int(opacity * 255):02x}"
            font_color = f"white@{opacity:.2f}"
            filters.append(
                f"drawtext=text='{safe_text}':fontcolor={font_color}:fontsize={font_size}:"
                f"shadowcolor=black@{max(0.2, opacity*0.8):.2f}:shadowx=2:shadowy=2:x={x_expr}:y={y_expr}"
            )
        elif wm_type == "image" and watermark.get("image_path") and Path(watermark["image_path"]).exists():
            img_path = Path(watermark["image_path"])
            extra_inputs.extend(["-i", str(img_path)])
            scale_w = int(watermark.get("width", 160))
            ov_pos = {
                "top_left": ("40", "40"),
                "top_right": ("W-w-40", "40"),
                "bottom_left": ("40", "H-h-180"),
                "bottom_right": ("W-w-40", "H-h-180"),
                "center_top": ("(W-w)/2", "50"),
            }
            ox, oy = ov_pos.get(pos, ("W-w-40", "40"))
            # We can use filter_complex when extra_inputs present
            filter_chain = ",".join(filters)
            fc = f"[0:v]{filter_chain}[base];[1:v]scale={scale_w}:-1,format=rgba,colorchannelmixer=aa={opacity:.2f}[wm];[base][wm]overlay={ox}:{oy}[outv]"
            cmd = [ffmpeg(), "-y", "-loglevel", "error", "-i", str(tmp_cut), *extra_inputs,
                   "-filter_complex", fc, "-map", "[outv]", "-map", "0:a?",
                   "-c:v", "libx264", "-preset", "medium", "-crf", "18",
                   "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
                   "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
                   "-movflags", "+faststart", str(output_path)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            tmp_cut.unlink(missing_ok=True)
            if r.returncode != 0:
                raise RuntimeError(f"export failed for clip {start_s:.2f}-{end_s:.2f}: {r.stderr[-400:]}")
            if progress_cb:
                progress_cb(1.0)
            return output_path

    cmd = [ffmpeg(), "-y", "-loglevel", "error", "-i", str(tmp_cut),
           "-vf", ",".join(filters),
           "-c:v", "libx264", "-preset", "medium", "-crf", "18",
           "-b:v", "10M", "-maxrate", "14M", "-bufsize", "20M",
           "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "256k",
           "-movflags", "+faststart", str(output_path)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    tmp_cut.unlink(missing_ok=True)
    if r.returncode != 0:
        raise RuntimeError(f"export failed for clip {start_s:.2f}-{end_s:.2f}: {r.stderr[-400:]}")
    if progress_cb:
        progress_cb(1.0)
    return output_path


def _subtitle_filter(srt_path, preset: str, custom_style: dict | None) -> str:
    style_str = sub_mod.build_force_style(preset, custom_style)
    safe = str(srt_path).replace("\\", "/").replace(":", "\\:")
    return f"subtitles='{safe}':force_style='{style_str}'"


def _hook_filter(text: str) -> str:
    safe = (text.replace("\\", "").replace("'", "").replace(":", "\\:")
            .replace("%", "").replace(",", "\\,"))
    return (f"drawtext=text='{safe}':fontcolor=white:fontsize=52:"
            f"box=1:boxcolor=black@0.55:boxborderw=14:x=(w-text_w)/2:y=h*0.08")


def export_batch(
    clips: Iterable[dict],
    output_dir: str | Path,
    title: str,
    *,
    aspect: str = "9:16",
    resolution: str = "1080p",
    reframed_source: str | Path | None = None,
    watermark: dict | None = None,
    progress_cb: Callable[[int, int, str], None] | None = None,
) -> list[dict]:
    """Encode every included clip. Returns [{clip_id, file, name, error}]."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = [c for c in clips if c.get("included", True)]
    results: list[dict] = []
    for i, c in enumerate(wanted, start=1):
        name = build_output_name(title, i, c["start_s"], c["end_s"])
        dest = unique_path(output_dir / name)
        entry = {"clip_id": c.get("id"), "name": dest.name, "file": "", "error": ""}
        try:
            clip_wm = c.get("watermark") or watermark
            export_clip(
                c.get("source_video") or c.get("file") or "",
                c["start_s"], c["end_s"], dest,
                aspect=aspect, resolution=resolution,
                srt_path=c.get("srt_path"), preset=c.get("subtitle_preset", "classic_white"),
                custom_style=c.get("subtitle_style"), hook_text=c.get("hook", ""),
                reframed_source=reframed_source,
                watermark=clip_wm,
            )
            entry["file"] = str(dest)
        except Exception as e:  # noqa: BLE001 — per-clip isolation, batch must continue
            entry["error"] = str(e)
        results.append(entry)
        if progress_cb:
            progress_cb(i, len(wanted), dest.name)
    return results
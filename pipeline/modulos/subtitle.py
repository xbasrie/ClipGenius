"""SRT generation from word timestamps + FFmpeg subtitle burn-in with style presets."""
from __future__ import annotations

from pathlib import Path

from .. import config


class SubtitleError(RuntimeError):
    pass


# ---------- colour ----------

def hex_to_ass_color(hex_color: str, opacity: float = 1.0) -> str:
    """'#RRGGBB' -> ASS '&HAABBGGRR'. opacity 1.0 = opaque, 0.0 = invisible."""
    h = hex_color.strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise SubtitleError(f"Invalid hex colour: {hex_color!r} (expected #RRGGBB)")
    r, g, b = h[0:2].upper(), h[2:4].upper(), h[4:6].upper()
    alpha = round((1.0 - max(0.0, min(1.0, float(opacity)))) * 255)
    return f"&H{alpha:02X}{b}{g}{r}"


# ---------- presets ----------

PRESETS: dict[str, dict] = {
    "classic_white": {
        "fontname": "Arial",
        "fontsize": 20,
        "font_color": "#FFFFFF",
        "border_color": "#000000",
        "border_width": 2.0,
        "bg_color": "#000000",
        "bg_opacity": 0.0,
        "alignment": 2,
        "bold": 1,
        "shadow": 0.5,
        "margin_v": 60,
    },
    "bold_yellow": {
        "fontname": "Arial Black",
        "fontsize": 24,
        "font_color": "#FFE600",
        "border_color": "#000000",
        "border_width": 3.0,
        "bg_color": "#000000",
        "bg_opacity": 0.0,
        "alignment": 2,
        "bold": 1,
        "shadow": 1.0,
        "margin_v": 70,
    },
    "neon_glow": {
        "fontname": "Arial",
        "fontsize": 22,
        "font_color": "#39FF14",
        "border_color": "#00FF88",
        "border_width": 4.0,
        "bg_color": "#001100",
        "bg_opacity": 0.15,
        "alignment": 2,
        "bold": 1,
        "shadow": 3.0,
        "margin_v": 66,
    },
    "minimalist": {
        "fontname": "Helvetica",
        "fontsize": 18,
        "font_color": "#F2F2F2",
        "border_color": "#000000",
        "border_width": 0.0,
        "bg_color": "#000000",
        "bg_opacity": 0.45,
        "alignment": 2,
        "bold": 0,
        "shadow": 0.0,
        "margin_v": 48,
    },
    "creator_pop": {
        "fontname": "Arial Black",
        "fontsize": 26,
        "font_color": "#FFFFFF",
        "border_color": "#FF2D95",
        "border_width": 2.5,
        "bg_color": "#FF2D95",
        "bg_opacity": 0.85,
        "alignment": 2,
        "bold": 1,
        "shadow": 1.5,
        "margin_v": 72,
    },
}

_SENTENCE_END = (".", "!", "?", "…", "。", "！", "？")


def _ass_time(seconds: float) -> str:
    """ASS timestamp H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, float(seconds))
    total_cs = round(seconds * 100)
    h, rem = divmod(total_cs, 360_000)
    m, rem = divmod(rem, 6_000)
    s, cs = divmod(rem, 100)
    return f"{h:d}:{m:02d}:{s:02d}.{cs:02d}"


def build_force_style(preset_name: str, custom_style: dict | None = None) -> str:
    """Force_style option string for the FFmpeg ass/subtitles filter."""
    if preset_name not in PRESETS:
        raise SubtitleError(f"Unknown preset {preset_name!r}; available: {sorted(PRESETS)}")
    style = dict(PRESETS[preset_name])
    if custom_style:
        style.update({k: v for k, v in custom_style.items() if v is not None})
    bg_opacity = float(style.get("bg_opacity", 0.0))
    parts = [
        f"FontName={style['fontname']}",
        f"FontSize={style['fontsize']}",
        f"PrimaryColour={hex_to_ass_color(style['font_color'], 1.0)}",
        f"OutlineColour={hex_to_ass_color(style['border_color'], 1.0)}",
        f"BackColour={hex_to_ass_color(style.get('bg_color', '#000000'), bg_opacity)}",
        f"BorderStyle={3 if bg_opacity > 0 else 1}",
        f"Outline={style.get('border_width', 2.0)}",
        f"Shadow={style.get('shadow', 0.0)}",
        f"Bold={style.get('bold', 0)}",
        f"Alignment={style['alignment']}",
        f"MarginV={style.get('margin_v', 60)}",
    ]
    return ",".join(parts)


def _escape_filter_path(path: Path) -> str:
    """Forward slashes + escaped colon; safe inside an FFmpeg single-quoted value."""
    return str(path).replace("\\", "/").replace("'", r"\'").replace(":", r"\:")


# ---------- SRT ----------

def _srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = round(seconds * 1000)
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_srt_block(index: int, start: float, end: float, text: str) -> str:
    """One SRT cue block, newline separated, no trailing blank line."""
    if end <= start:
        end = start + 0.05
    return f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n{text.strip()}\n"


def _cues(transcript: dict, clip_start: float, clip_end: float,
          max_chars: int, max_duration: float, time_offset: float = 0.0) -> list[tuple[float, float, str]]:
    """Word stream inside the clip window -> short relative-time cues."""
    words: list[dict] = []
    for seg in transcript.get("segments") or []:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                words.append({"word": str(w.get("word", "")).strip(),
                              "start": float(w["start"]), "end": float(w["end"])})
        elif seg.get("text"):
            toks = str(seg["text"]).split()
            if not toks:
                continue
            span = max(float(seg["end"]) - float(seg["start"]), 0.001) / len(toks)
            for i, tok in enumerate(toks):
                s = float(seg["start"]) + i * span
                words.append({"word": tok, "start": s, "end": s + span})

    window = [w for w in words
              if w["word"] and w["end"] > clip_start and w["start"] < clip_end]
    if not window:
        return []

    cues: list[tuple[float, float, str]] = []
    line: list[dict] = []

    def flush() -> None:
        if not line:
            return
        text = " ".join(w["word"] for w in line)
        if not text:
            line.clear()
            return
        start = line[0]["start"] - clip_start + time_offset
        end = line[-1]["end"] - clip_start + time_offset
        line.clear()
        cues.append((round(max(0.0, start), 3), round(max(0.05, end), 3),
                     text[0].upper() + text[1:] if len(text) > 1 else text.upper()))

    for w in window:
        if line:
            gap = w["start"] - line[-1]["end"]
            projected_len = sum(len(x["word"]) + 1 for x in line) + len(w["word"])
            projected_dur = w["end"] - line[0]["start"]
            if gap > 0.7 or projected_len > max_chars or projected_dur > max_duration:
                flush()
        line.append(w)
        if line[-1]["word"].rstrip().endswith(_SENTENCE_END) and len(line) >= 2:
            flush()
    flush()

    # merge cues too short to read (<0.4 s) into the next one
    merged: list[tuple[float, float, str]] = []
    for cue in cues:
        if merged and cue[0] - merged[-1][1] < 0.05 and (cue[1] - cue[0]) < 0.4:
            p = merged.pop()
            merged.append((p[0], max(p[1], cue[1]), f"{p[2]} {cue[2]}"))
        else:
            merged.append(cue)
    return merged


def generate_srt(transcript: dict, clip_start: float, clip_end: float,
                 output_path: str | Path,
                 max_chars: int = config.SRT_MAX_CHARS,
                 max_duration: float = config.SRT_MAX_DURATION_S,
                 time_offset: float = 0.0) -> bool:
    """
    Write an SRT for the clip window [clip_start, clip_end] with timestamps
    relative to clip_start. Returns False when the window has no words.
    """
    if clip_end <= clip_start:
        raise SubtitleError(f"clip_end ({clip_end}) must be > clip_start ({clip_start})")
    if not transcript or not transcript.get("segments"):
        raise SubtitleError("generate_srt needs a transcript with segments")

    cues = _cues(transcript, float(clip_start), float(clip_end),
                 int(max_chars), float(max_duration), float(time_offset))
    out = Path(output_path)
    if not cues:
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(format_srt_block(i, s, e, t) for i, (s, e, t) in enumerate(cues, 1))
    try:
        out.write_text(body + "\n", encoding="utf-8")
    except OSError as e:
        raise SubtitleError(f"Cannot write SRT {out}: {e}") from e
    return True


# ---------- burn ----------

def burn_subtitles(video_path: str | Path, srt_path: str | Path,
                   output_path: str | Path, preset_name: str = "classic_white",
                   custom_style: dict | None = None) -> bool:
    """Hard-code subtitles via FFmpeg's subtitles filter + force_style."""
    import subprocess

    vp, sp, op = Path(video_path), Path(srt_path), Path(output_path)
    if not vp.is_file():
        raise SubtitleError(f"Video not found: {vp}")
    if not sp.is_file():
        raise SubtitleError(f"SRT not found: {sp}")
    op.parent.mkdir(parents=True, exist_ok=True)
    style = build_force_style(preset_name, custom_style)

    parts = [f"filename='{_escape_filter_path(sp)}'"]
    if config.FONTS_DIR.is_dir():
        parts.append(f"fontsdir='{_escape_filter_path(config.FONTS_DIR)}'")
    parts.append(f"charenc=UTF-8")
    parts.append(f"force_style='{style}'")
    vf = "subtitles=" + ":".join(parts)

    cmd = [
        config.ffmpeg(), "-y",
        "-i", str(vp),
        "-vf", vf,
        "-c:a", "copy",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        str(op),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=3600)
    except FileNotFoundError as e:
        raise SubtitleError(f"ffmpeg not found ({config.ffmpeg()}): {e}") from e
    except subprocess.TimeoutExpired as e:
        raise SubtitleError("ffmpeg subtitle burn timed out after 3600s") from e
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", "replace").strip()[-800:]
        raise SubtitleError(f"ffmpeg subtitle burn failed: {err}")
    if not op.is_file():
        raise SubtitleError(f"ffmpeg reported success but {op} is missing")
    return True
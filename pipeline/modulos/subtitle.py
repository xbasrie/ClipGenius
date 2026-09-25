"""SRT generation from word timestamps + FFmpeg subtitle burn-in with style presets."""
from __future__ import annotations

import re
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


# Supported entrance animations:
# 'none', 'pop', 'fade', 'bounce', 'slide_up'
ENTRANCE_ANIMATIONS = ("none", "pop", "fade", "bounce", "slide_up")


def get_animation_ass_tag(animation: str, duration_ms: int = 140) -> str:
    """Return inline ASS tag string for entrance animation."""
    if not animation or animation == "none":
        return ""
    dur = max(60, min(500, int(duration_ms)))
    if animation == "pop":
        # Scale up from 70% to 105% then settle to 100%
        return f"{{\\fscx75\\fscy75\\t(0,{dur},\\fscx100\\fscy100)}}"
    elif animation == "fade":
        # Fade in alpha from invisible to visible
        return f"{{\\alpha&HFF&\\t(0,{dur},\\alpha&H00&)}}"
    elif animation == "bounce":
        # Bounce: zoom from 60% to 118% then settle to 100%
        half = dur // 2
        return f"{{\\fscx60\\fscy60\\t(0,{half},\\fscx118\\fscy118)\\t({half},{dur},\\fscx100\\fscy100)}}"
    elif animation == "slide_up":
        # Slight vertical rise with fade
        return f"{{\\fscy80\\alpha&HBB&\\t(0,{dur},\\fscy100\\alpha&H00&)}}"
    return ""


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
        preset_name = "classic_white"
    style = dict(PRESETS.get(preset_name, PRESETS["classic_white"]))
    if custom_style:
        style.update({k: v for k, v in custom_style.items() if v is not None})
    bg_opacity = float(style.get("bg_opacity", 0.0))

    # Glow effect handling:
    # If glow is active, enlarge outline border and use neon glow color with subtle shadow
    glow = bool(style.get("glow", False))
    border_width = float(style.get("border_width", 2.0))
    shadow = float(style.get("shadow", 0.0))
    border_color = style.get("border_color", "#000000")

    if glow:
        border_width = max(border_width, 4.5)
        shadow = max(shadow, 2.5)
        # Use glow color if specified, else use font color or high saturation cyan/amber
        glow_col = style.get("glow_color") or style.get("font_color") or "#00FFFF"
        border_color = glow_col

    parts = [
        f"FontName={style['fontname']}",
        f"FontSize={style['fontsize']}",
        f"PrimaryColour={hex_to_ass_color(style['font_color'], 1.0)}",
        f"OutlineColour={hex_to_ass_color(border_color, 0.9 if glow else 1.0)}",
        f"BackColour={hex_to_ass_color(style.get('bg_color', '#000000'), bg_opacity)}",
        f"BorderStyle={3 if bg_opacity > 0 else 1}",
        f"Outline={border_width}",
        f"Shadow={shadow}",
        f"Bold={style.get('bold', 0)}",
        f"Alignment={style.get('alignment', 2)}",
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


def _is_keyword(word: str) -> bool:
    """Detect if a token is a high-energy keyword (numbers, symbols, emphasized words)."""
    clean = re.sub(r"[^\w\s]", "", word).strip()
    if not clean:
        return False
    # Numbers, percentages, currencies, or symbols
    if any(ch.isdigit() for ch in word) or any(c in word for c in ("$", "%", "!", "?")):
        return True
    # Indonesian & English high-impact trigger words
    triggers = {
        "jangan", "rahasia", "sukses", "kaya", "cepat", "fakta", "gila", "bahaya",
        "rugi", "untung", "stop", "viral", "trik", "tips", "penting", "wajib",
        "hancur", "terbaik", "paling", "modal", "omset", "cuci", "dosa", "mati",
        "never", "always", "secret", "rich", "fast", "money", "crazy", "stop",
        "best", "worst", "huge", "insane", "danger", "free", "hack"
    }
    return clean.lower() in triggers or (len(clean) >= 6 and clean.isupper())


def _format_chunk_keywords(chunk_tokens: list[str], highlight_color_bgr: str = "&H0022FF&") -> str:
    """Format chunk text highlighting keywords using ASS color tags."""
    formatted = []
    for tok in chunk_tokens:
        clean = tok.strip()
        if not clean:
            continue
        if _is_keyword(clean):
            # Highlight with contrasting accent (e.g. bright green/yellow)
            formatted.append(f"{{\\c{highlight_color_bgr}}}{clean}{{\
}}")
        else:
            formatted.append(clean)
    return " ".join(formatted)


def _cues(transcript: dict, clip_start: float, clip_end: float,
          max_chars: int, max_duration: float, time_offset: float = 0.0,
          word_by_word: bool = True, words_per_chunk: int = 1,
          animation: str = "none",
          keyword_pop: bool = False,
          highlight_color: str = "#22c55e") -> list[tuple[float, float, str]]:
    """Word stream inside the clip window -> relative-time cues."""
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

    anim_tag = get_animation_ass_tag(animation)
    hl_bgr = hex_to_ass_color(highlight_color)

    # Word-by-word or N-words per chunk mode (1 to 5 words)
    words_per_chunk = max(1, min(10, int(words_per_chunk or 1)))
    if word_by_word or words_per_chunk >= 1:
        cues: list[tuple[float, float, str]] = []
        for i in range(0, len(window), words_per_chunk):
            chunk = window[i:i + words_per_chunk]
            start = chunk[0]["start"] - clip_start + time_offset
            end = chunk[-1]["end"] - clip_start + time_offset
            if end <= start:
                end = start + 0.25 * len(chunk)
            
            raw_tokens = [w["word"].strip().upper() for w in chunk if w["word"].strip()]
            if not raw_tokens:
                continue

            if keyword_pop and len(raw_tokens) > 1:
                chunk_text = _format_chunk_keywords(raw_tokens, hl_bgr)
            elif keyword_pop and len(raw_tokens) == 1 and _is_keyword(raw_tokens[0]):
                chunk_text = f"{{\\c{hl_bgr}}}{raw_tokens[0]}{{\
}}"
            else:
                chunk_text = " ".join(raw_tokens)

            full_text = f"{anim_tag}{chunk_text}" if anim_tag else chunk_text
            cues.append((round(max(0.0, start), 3), round(max(0.05, end), 3), full_text))
        return cues

    # Sentence / phrase grouping mode fallback
    cues = []
    line = []

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
        content = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        full_text = f"{anim_tag}{content}" if anim_tag else content
        cues.append((round(max(0.0, start), 3), round(max(0.05, end), 3), full_text))

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

    return cues


def generate_srt(transcript: dict, clip_start: float, clip_end: float,
                 output_path: str | Path,
                 max_chars: int = config.SRT_MAX_CHARS,
                 max_duration: float = config.SRT_MAX_DURATION_S,
                 time_offset: float = 0.0,
                 word_by_word: bool = True,
                 words_per_chunk: int = 1,
                 animation: str = "none",
                 keyword_pop: bool = False,
                 highlight_color: str = "#22c55e") -> bool:
    """
    Write an SRT for the clip window [clip_start, clip_end] with timestamps
    relative to clip_start. Returns False when the window has no words.
    """
    if clip_end <= clip_start:
        raise SubtitleError(f"clip_end ({clip_end}) must be > clip_start ({clip_start})")
    if not transcript or not transcript.get("segments"):
        raise SubtitleError("generate_srt needs a transcript with segments")

    cues = _cues(transcript, float(clip_start), float(clip_end),
                 int(max_chars), float(max_duration), float(time_offset),
                 word_by_word=word_by_word, words_per_chunk=words_per_chunk,
                 animation=animation, keyword_pop=keyword_pop,
                 highlight_color=highlight_color)
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
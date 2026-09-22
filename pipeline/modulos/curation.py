"""Viral moment detection: LLM prompt -> validated clip windows, heuristic fallback."""
from __future__ import annotations

import json
import logging
import re

from .. import config
from ..providers import llm

logger = logging.getLogger(__name__)

MIN_CLIP_S = 5.0

# Full OpenShorts (mutonby/openshorts) Gemini prompt, verbatim except that the
# three insertion points are token placeholders instead of str.format braces —
# the JSON contract is brace-heavy and .format() would need doubling everywhere.
# ponytail: token replace instead of .format(); upgrade path -> str.format_map if
# the prompt ever needs arithmetic placeholders.
PROMPT_TEMPLATE = """\
You are a senior short-form video editor. Read the ENTIRE transcript and word-level timestamps to choose the 3–15 MOST VIRAL moments for TikTok/IG Reels/YouTube Shorts. Each clip must be between 15 and 60 seconds long.

⚠️ FFMPEG TIME CONTRACT — STRICT REQUIREMENTS:
- Return timestamps in ABSOLUTE SECONDS from the start of the video (usable in: ffmpeg -ss <start> -to <end> -i <input> ...).
- Only NUMBERS with decimal point, up to 3 decimals (examples: 0, 1.250, 17.350).
- Ensure 0 ≤ start < end ≤ VIDEO_DURATION_SECONDS.
- Each clip between 15 and 60 s (inclusive).
- Prefer starting 0.2–0.4 s BEFORE the hook and ending 0.2–0.4 s AFTER the payoff.
- Use silence moments for natural cuts; never cut in the middle of a word or phrase.
- STRICTLY FORBIDDEN to use time formats other than absolute seconds.

VIDEO_DURATION_SECONDS: __VIDEO_DURATION__

TRANSCRIPT_TEXT (raw):
__TRANSCRIPT_TEXT__

WORDS_JSON (array of {w, s, e} where s/e are seconds):
__WORDS_JSON__

STRICT EXCLUSIONS:
- No generic intros/outros or purely sponsorship segments unless they contain the hook.
- No clips < 15 s or > 60 s.

OUTPUT — RETURN ONLY VALID JSON (no markdown, no comments). Order clips by predicted performance (best to worst). In the descriptions, ALWAYS include a CTA like "Follow me and comment X and I'll send you the workflow" (especially if discussing an n8n workflow):
{
  "shorts": [
    {
      "start": <number in seconds, e.g., 12.340>,
      "end": <number in seconds, e.g., 37.900>,
      "video_description_for_tiktok": "<description for TikTok oriented to get views>",
      "video_description_for_instagram": "<description for Instagram oriented to get views>",
      "video_title_for_youtube_short": "<title for YouTube Short oriented to get views 100 chars max>",
      "viral_hook_text": "<SHORT punchy text overlay (max 10 words) with 1-2 fitting emojis. MUST BE IN THE SAME LANGUAGE AS THE VIDEO TRANSCRIPT. Examples: 'POV: You realized... 😳', 'Did you know? 🤯', 'Stop doing this! 🚫'>"
    }
  ]
}
"""

# USD per 1M tokens (input, output); unknown model -> the Gemini Flash default.
_RATES: dict[str, tuple[float, float]] = {
    "gemini-2.5-flash": (0.10, 0.40),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-flash": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
}
_DEFAULT_RATE = (0.10, 0.40)

HOOK_MARKERS = (
    "?", "!", "you", "your", "how", "why", "what", "never", "always", "secret",
    "mistake", "free", "best", "worst", "nobody", "everyone", "stop", "start",
    "cara", "kenapa", "jangan", "gratis", "rahasia", "salah", "paling",
)


def build_prompt(transcript: dict, video_duration: float) -> str:
    """Fill PROMPT_TEMPLATE with the transcript and compact word list."""
    words = _words(transcript)
    words_json = json.dumps(
        [{"w": w["word"], "s": round(w["start"], 3), "e": round(w["end"], 3)} for w in words],
        ensure_ascii=False, separators=(",", ":"),
    )
    text = (transcript.get("text") or "").strip()
    if len(text) > 120_000:
        text = text[:120_000] + " …[truncated]"
    return (PROMPT_TEMPLATE
            .replace("__VIDEO_DURATION__", f"{video_duration:.3f}")
            .replace("__TRANSCRIPT_TEXT__", text)
            .replace("__WORDS_JSON__", words_json))


def _words(transcript: dict) -> list[dict]:
    """Flatten word records; synthesize from segment bounds when the STT gave none."""
    out: list[dict] = []
    for seg in transcript.get("segments") or []:
        seg_words = seg.get("words") or []
        if seg_words:
            for w in seg_words:
                out.append({"word": w["word"], "start": float(w["start"]),
                            "end": float(w["end"])})
        elif seg.get("text"):
            toks = seg["text"].split()
            if not toks:
                continue
            span = max(float(seg["end"]) - float(seg["start"]), 0.001) / len(toks)
            for i, tok in enumerate(toks):
                s = float(seg["start"]) + i * span
                out.append({"word": tok, "start": s, "end": s + span})
    return out


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _cost(prompt: str, response: dict, model: str) -> dict:
    """Local estimate — providers here return no usage block."""
    in_tok = _estimate_tokens(prompt)
    out_tok = _estimate_tokens(json.dumps(response, ensure_ascii=False))
    rate_in, rate_out = _RATES.get(model, _DEFAULT_RATE)
    in_cost = round(in_tok / 1_000_000 * rate_in, 8)
    out_cost = round(out_tok / 1_000_000 * rate_out, 8)
    return {
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "input_cost": in_cost,
        "output_cost": out_cost,
        "total_cost": round(in_cost + out_cost, 8),
        "model": model,
        "estimated": True,
    }


def _clean_shorts(raw: list, duration: float, clip_max_s: float, clip_count: int) -> list[dict]:
    """Enforce the time contract; drop anything unfixable, keep order, cap count."""
    cleaned: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            start = float(item.get("start"))
            end = float(item.get("end"))
        except (TypeError, ValueError):
            continue
        if start != start or end != end:  # NaN
            continue
        start = max(0.0, min(start, duration))
        end = max(0.0, min(end, duration))
        if end <= start:
            continue
        if end - start > clip_max_s:
            end = start + clip_max_s
        if end - start < MIN_CLIP_S:
            continue
        cleaned.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "video_description_for_tiktok": str(item.get("video_description_for_tiktok") or "")[:400],
            "video_description_for_instagram": str(item.get("video_description_for_instagram") or "")[:400],
            "video_title_for_youtube_short": str(item.get("video_title_for_youtube_short") or "")[:100],
            "viral_hook_text": str(item.get("viral_hook_text") or "")[:120],
        })
    cleaned.sort(key=lambda c: c["start"])
    return cleaned[:max(0, clip_count)] or []


def curate_clips(transcript: dict, video_duration: float, clip_max_s: int,
                 clip_count: int, llm_config: config.LLMConfig | None = None) -> dict:
    """
    Pick viral moments. Returns {'shorts': [...], 'cost_analysis': {...},
    'source': 'llm'|'heuristic'}. Never raises on LLM problems — falls back.
    """
    if not transcript or not transcript.get("segments"):
        raise ValueError("curate_clips needs a transcript with segments")
    duration = float(video_duration) or 0.0
    if duration <= 0:
        duration = max((float(s["end"]) for s in transcript["segments"]), default=0.0)
    if duration <= 0:
        raise ValueError("video_duration must be > 0 (or inferable from transcript)")
    cfg = llm_config or config.LLMConfig.from_env()

    prompt = build_prompt(transcript, duration)
    if llm.available(cfg):
        try:
            response = llm.generate_json(prompt, cfg)
            shorts = _clean_shorts(response.get("shorts") or [], duration, clip_max_s, clip_count)
            if shorts:
                return {
                    "shorts": shorts,
                    "cost_analysis": _cost(prompt, response, cfg.model),
                    "source": "llm",
                }
            logger.warning("LLM returned no valid clips — heuristic fallback")
        except Exception as e:  # noqa: BLE001 — any provider failure falls back
            logger.warning("LLM curation failed (%s) — heuristic fallback", e)
    else:
        logger.info("No LLM configured (provider=%s) — heuristic curation", cfg.provider)

    result = heuristic_curate(transcript, duration, clip_max_s, clip_count)
    result["cost_analysis"]["model"] = cfg.model or "heuristic"
    return result


def heuristic_curate(transcript: dict, duration: float, clip_max_s: int,
                     clip_count: int) -> dict:
    """
    No-LLM curation: score evenly spaced windows by word density + hook markers,
    keep the best non-overlapping ones.
    """
    duration = float(duration)
    length = float(min(clip_max_s, max(MIN_CLIP_S, duration)))
    words = _words(transcript)
    if not words or duration <= 0:
        raise ValueError("heuristic_curate needs words and a positive duration")

    n_slots = max(clip_count * 3, 6)
    step = max(MIN_CLIP_S, duration / n_slots)
    starts: list[float] = []
    s = 0.0
    while s + MIN_CLIP_S <= duration and len(starts) < n_slots * 2:
        starts.append(round(s, 3))
        s += step
    if not starts:
        starts = [0.0]

    candidates: list[dict] = []
    for start in starts:
        start = min(start, max(0.0, duration - length))
        end = min(duration, start + length)
        window = [w for w in words if w["start"] >= start and w["end"] <= end]
        if not window:
            continue
        span = max(end - start, 0.001)
        density = len(window) / span
        text = " ".join(w["word"] for w in window).lower()
        hooks = sum(text.count(m) for m in HOOK_MARKERS)
        numbers = len(re.findall(r"\d", text))
        score = round(density * 10 + hooks * 0.6 + min(numbers, 8) * 0.3, 4)
        candidates.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "duration": round(end - start, 3),
            "score": score,
            "text": " ".join(w["word"] for w in window)[:280],
        })

    if not candidates:
        raise ValueError("heuristic_curate found no scoreable windows")

    candidates.sort(key=lambda c: c["score"], reverse=True)
    picked: list[dict] = []
    for cand in candidates:
        overlap_ok = all(
            min(cand["end"], p["end"]) - max(cand["start"], p["start"])
            <= 0.5 * min(cand["duration"], p["duration"])
            for p in picked
        )
        if overlap_ok:
            picked.append(cand)
        if len(picked) >= max(1, clip_count):
            break
    if not picked:
        picked = candidates[:max(1, clip_count)]
    picked.sort(key=lambda c: c["start"])

    shorts = []
    for c in picked:
        hook_src = " ".join(c["text"].split()[:10])
        shorts.append({
            "start": c["start"],
            "end": c["end"],
            "duration": c["duration"],
            "score": c["score"],
            "video_description_for_tiktok": hook_src,
            "video_description_for_instagram": hook_src,
            "video_title_for_youtube_short": hook_src[:100],
            "viral_hook_text": c["text"].split(" ")[0][:40] if c["text"] else "",
        })
    return {
        "shorts": shorts,
        "cost_analysis": {
            "input_tokens": 0, "output_tokens": 0, "input_cost": 0.0,
            "output_cost": 0.0, "total_cost": 0.0, "model": "heuristic", "estimated": False,
        },
        "source": "heuristic",
    }
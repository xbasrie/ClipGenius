"""Speech-to-text via faster-whisper with word timestamps + DB cache."""
from __future__ import annotations

import logging
from pathlib import Path

from .. import config, db
from .media import file_hash

logger = logging.getLogger(__name__)


class STTError(RuntimeError):
    pass


def _model_name(tier: str) -> str:
    try:
        return config.STT_MODELS[tier]
    except KeyError as e:
        raise STTError(f"Unknown STT tier {tier!r}; expected one of "
                       f"{sorted(config.STT_MODELS)}") from e


def _pick_device() -> tuple[str, str]:
    """(device, compute_type): cuda when available and libraries loadable, else cpu/int8."""
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            import ctypes
            ctypes.CDLL("cublas64_12.dll")
            supported = ctranslate2.get_supported_compute_types("cuda")
            if "float16" in supported:
                return "cuda", "float16"
            if "int8" in supported:
                return "cuda", "int8"
            if supported:
                return "cuda", next(iter(supported))
    except Exception:  # noqa: BLE001 — probing CUDA is best-effort
        pass
    return "cpu", "int8"


def transcribe(video_path: str | Path, model_tier: str = "fast",
               language: str | None = None,
               progress_cb=None) -> dict:
    """
    Transcribe a video file with word-level timestamps.

    Returns {'text', 'language', 'segments': [{'text','start','end','words':
    [{'word','start','end','probability'}]}]}. Cached by file hash.
    """
    vp = Path(video_path)
    if not vp.is_file():
        raise STTError(f"Video not found: {vp}")
    model_ref = _model_name(model_tier)

    key = f"stt:{model_ref}:{language or 'auto'}:{file_hash(vp)}"
    cached = db.cache_get(key)
    if cached:
        logger.info("STT cache hit: %s", key[:40])
        return cached

    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise STTError("faster-whisper not installed — pip install faster-whisper") from e

    device, compute_type = _pick_device()
    logger.info("STT: model=%s device=%s compute=%s", model_ref, device, compute_type)
    try:
        segments_iter, info = WhisperModel(model_ref, device=device, compute_type=compute_type
                                           ).transcribe(
            str(vp),
            language=language,
            word_timestamps=True,
            vad_filter=True,
        )
    except Exception as e:
        if device != "cpu":
            logger.warning("STT on %s failed (%s); falling back to CPU int8", device, e)
            device, compute_type = "cpu", "int8"
            segments_iter, info = WhisperModel(model_ref, device=device, compute_type=compute_type
                                               ).transcribe(
                str(vp),
                language=language,
                word_timestamps=True,
                vad_filter=True,
            )
        else:
            raise

    total = float(getattr(info, "duration", 0.0)) or 0.0
    out_segments: list[dict] = []
    chunks: list[str] = []
    last_end = 0.0
    for seg in segments_iter:
        words = [
            {"word": w.word.strip(), "start": round(w.start, 3),
             "end": round(w.end, 3), "probability": round(w.probability, 3)}
            for w in (seg.words or [])
            if w.word.strip()
        ]
        out_segments.append({
            "text": (seg.text or "").strip(),
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "words": words,
        })
        chunks.append(seg.text or "")
        last_end = seg.end
        if progress_cb and total > 0:
            progress_cb(min(99.0, max(0.0, last_end / total * 100.0)))

    result = {
        "text": " ".join(t.strip() for t in chunks if t and t.strip()),
        "segments": out_segments,
        "language": getattr(info, "language", "") or language or "",
    }
    if progress_cb:
        progress_cb(100.0)
    if not out_segments:
        raise STTError(f"No speech detected in {vp}")
    db.cache_put(key, result)
    return result

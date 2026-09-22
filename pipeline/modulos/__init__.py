"""ClipGenius pipeline modules: media, STT, curation, subtitles."""
from . import curation, media, stt, subtitle
from .curation import PROMPT_TEMPLATE, curate_clips, heuristic_curate
from .media import (MediaError, download_video, extract_audio, file_hash, get_duration,
                    get_resolution, get_video_info)
from .stt import STTError, transcribe
from .subtitle import (PRESETS, SubtitleError, burn_subtitles, format_srt_block,
                       generate_srt, hex_to_ass_color)

__all__ = [
    "curation", "media", "stt", "subtitle",
    "MediaError", "STTError", "SubtitleError",
    "download_video", "get_video_info", "get_duration", "get_resolution",
    "extract_audio", "file_hash",
    "transcribe",
    "PROMPT_TEMPLATE", "curate_clips", "heuristic_curate",
    "PRESETS", "generate_srt", "format_srt_block", "burn_subtitles", "hex_to_ass_color",
]
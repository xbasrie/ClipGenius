"""ClipGenius pipeline — paths & settings. Single source of truth."""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

APP_NAME = "ClipGenius"


def _root() -> Path:
    if os.environ.get("CLIPGENIUS_HOME"):
        return Path(os.environ["CLIPGENIUS_HOME"])
    if os.path.exists("D:\\"):
        return Path("D:/clipgenius_data")
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~/.local/share")
    return Path(base) / APP_NAME


ROOT = _root()
WORK_DIR = ROOT / "work"
LOG_DIR = ROOT / "logs"
DB_PATH = ROOT / "clipgenius.db"
FONTS_DIR = Path(__file__).resolve().parent.parent / "resources" / "fonts"
BIN_DIR = Path(__file__).resolve().parent.parent / "resources" / "bin"

DEFAULT_CLIP_MIN_S = 15
DEFAULT_CLIP_MAX_S = 30
DEFAULT_CLIP_COUNT = 5
# ponytail: ceiling = always 2 lines max in vertical; upgrade path -> per-style line rules
SRT_MAX_CHARS = 28
SRT_MAX_DURATION_S = 2.6


def ensure_dirs() -> None:
    for d in (ROOT, WORK_DIR, LOG_DIR, FONTS_DIR, BIN_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _bundled(name: str) -> str | None:
    """Prefer bundled binary in resources/bin, next to exe, else PATH."""
    import sys
    base_dirs = [
        BIN_DIR,
        Path(__file__).resolve().parent.parent / "resources" / "bin",
        Path(sys.executable).parent / "resources" / "bin",
        Path(sys.executable).parent,
    ]
    if hasattr(sys, "_MEIPASS"):
        base_dirs.extend([
            Path(sys._MEIPASS) / "resources" / "bin",
            Path(sys._MEIPASS) / "bin",
            Path(sys._MEIPASS),
        ])
    for bdir in base_dirs:
        for candidate in (bdir / f"{name}.exe", bdir / name):
            if candidate.exists():
                return str(candidate)
    return shutil.which(name)


def ffmpeg() -> str:
    return _bundled("ffmpeg") or "ffmpeg"


def ffprobe() -> str:
    return _bundled("ffprobe") or "ffprobe"


def ytdlp() -> list[str]:
    """yt-dlp invocation: bundled exe, venv scripts, else python module."""
    exe = _bundled("yt-dlp")
    if exe:
        return [exe]
    import sys

    # Check venv or local scripts
    candidates = [
        Path(__file__).resolve().parent.parent / ".venv" / "Scripts" / "yt-dlp.exe",
        Path.home() / "clipgenius" / ".venv" / "Scripts" / "yt-dlp.exe",
    ]
    for c in candidates:
        if c.exists():
            return [str(c)]

    if not getattr(sys, "frozen", False):
        return [sys.executable, "-m", "yt_dlp"]

    py = shutil.which("python") or shutil.which("python3")
    if py:
        return [py, "-m", "yt_dlp"]

    return ["yt-dlp"]



STT_MODELS = {
    "fast": "base",
    "balanced": "small",
    "accurate": "medium",
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "offline"  # offline | gemini | openai | ollama
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    timeout_s: int = 180

    @classmethod
    def load(cls) -> "LLMConfig":
        try:
            from . import db
            saved = db.get_setting("llm")
            if isinstance(saved, dict) and saved.get("provider"):
                prov = saved.get("provider", "offline")
                defaults = {
                    "gemini": ("gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta"),
                    "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
                    "ollama": ("llama3.1", "http://127.0.0.1:11434/v1"),
                }
                mdl_def, url_def = defaults.get(prov, ("", ""))
                return cls(
                    provider=prov,
                    model=saved.get("model") or mdl_def,
                    api_key=saved.get("api_key") or "",
                    base_url=saved.get("base_url") or url_def,
                )
        except Exception:
            pass
        return cls.from_env()

    @staticmethod
    def from_env() -> "LLMConfig":
        provider = os.environ.get("CLIPGENIUS_LLM_PROVIDER", "").lower()
        if not provider:
            if os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
                provider = "gemini"
            elif os.environ.get("OPENAI_API_KEY"):
                provider = "openai"
            else:
                provider = "offline"
        defaults = {
            "gemini": ("gemini-2.0-flash", "https://generativelanguage.googleapis.com/v1beta"),
            "openai": ("gpt-4o-mini", "https://api.openai.com/v1"),
            "ollama": ("llama3.1", "http://127.0.0.1:11434/v1"),
        }
        model_default, url_default = defaults.get(provider, ("", ""))
        return LLMConfig(
            provider=provider,
            model=os.environ.get("CLIPGENIUS_LLM_MODEL", model_default),
            api_key=(
                os.environ.get("CLIPGENIUS_LLM_KEY")
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
                or os.environ.get("OPENAI_API_KEY", "")
            ),
            base_url=os.environ.get("CLIPGENIUS_LLM_BASE_URL", url_default),
        )
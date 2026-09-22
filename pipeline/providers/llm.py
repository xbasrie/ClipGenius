"""LLM providers behind one interface: generate_json(prompt) -> dict.

Rung 4 of the ladder: no new SDKs. Gemini/OpenAI/Ollama all speak HTTP via httpx
which is already installed. Offline heuristic covers the no-key case so the
pipeline never hard-fails on a missing API key.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import httpx

from .. import config

RETRY_MARKERS = ("429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED",
                 "INTERNAL", "DEADLINE_EXCEEDED", "timed out", "Connection")


class LLMError(RuntimeError):
    pass


def _extract_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError as e:
            raise LLMError(f"Invalid JSON from model: {e}") from e
    raise LLMError("Model returned no JSON object")


def _post_with_retry(url: str, *, headers: dict, payload: dict, timeout: int,
                     attempts: int = 4) -> dict:
    last: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            with httpx.Client(timeout=timeout) as client:
                r = client.post(url, headers=headers, json=payload)
            if r.status_code >= 400:
                body = r.text[:300]
                if r.status_code in (429, 500, 502, 503, 504) and attempt < attempts:
                    raise httpx.HTTPStatusError(body, request=r.request, response=r)
                raise LLMError(f"HTTP {r.status_code}: {body}")
            return r.json()
        except Exception as e:  # noqa: BLE001 — retry/backoff boundary
            last = e
            msg = str(e)
            transient = any(m in msg for m in RETRY_MARKERS) or isinstance(e, httpx.TransportError)
            if attempt >= attempts or not transient:
                break
            delay = min(2 * (2 ** (attempt - 1)), 30) + 0.2 * attempt
            time.sleep(delay)
    raise LLMError(f"LLM request failed: {last}")


# ---------- providers ----------

def _call_gemini(prompt: str, cfg: config.LLMConfig) -> str:
    if not cfg.api_key:
        raise LLMError("Gemini API key not configured")
    url = f"{cfg.base_url}/models/{cfg.model}:generateContent?key={cfg.api_key}"
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.4},
    }
    data = _post_with_retry(url, headers={"Content-Type": "application/json"}, payload=payload,
                            timeout=cfg.timeout_s)
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected Gemini response shape: {str(data)[:300]}") from e


def _call_openai_compatible(prompt: str, cfg: config.LLMConfig) -> str:
    url = f"{cfg.base_url}/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": [
            {"role": "system", "content": "Return ONLY valid JSON. No markdown."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    data = _post_with_retry(url, headers=headers, payload=payload, timeout=cfg.timeout_s)
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise LLMError(f"Unexpected chat response shape: {str(data)[:300]}") from e


def generate_json(prompt: str, cfg: config.LLMConfig | None = None) -> dict:
    """Raises LLMError when no provider is usable — callers fall back to heuristics."""
    cfg = cfg or config.LLMConfig.load()
    if cfg.provider == "gemini":
        return _extract_json(_call_gemini(prompt, cfg))
    if cfg.provider in ("openai", "ollama"):
        return _extract_json(_call_openai_compatible(prompt, cfg))
    raise LLMError(f"No LLM provider configured (provider={cfg.provider!r})")


def available(cfg: config.LLMConfig | None = None) -> bool:
    cfg = cfg or config.LLMConfig.load()
    if cfg.provider == "ollama":
        return True
    return bool(cfg.api_key) and cfg.provider in ("gemini", "openai")


def provider_status() -> dict[str, Any]:
    cfg = config.LLMConfig.load()
    return {
        "provider": cfg.provider,
        "model": cfg.model,
        "configured": available(cfg),
        "has_key": bool(cfg.api_key),
        "base_url": cfg.base_url,
    }
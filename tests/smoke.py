"""Smoke test: generate a synthetic video, inject transcript, run the full
pipeline (curate -> clip -> reframe -> subtitle -> export) without network.

Run: ./.venv/Scripts/python.exe -m tests.smoke
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline import config, db  # noqa: E402
from pipeline.modulos import clip as clip_mod  # noqa: E402
from pipeline.modulos import curation, export as export_mod  # noqa: E402
from pipeline.modulos import subtitle as sub_mod  # noqa: E402

FIXTURES = ROOT / "fixtures"
OUT = FIXTURES / "smoke_out"


def make_video(path: Path, duration_s: float = 60.0, w: int = 1280, h: int = 720) -> Path:
    """testsrc video + sine audio; no real speech, but the container is valid."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    r = subprocess.run([
        config.ffmpeg(), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", f"testsrc=duration={duration_s}:size={w}x{h}:rate=30",
        "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-c:a", "aac", "-shortest", str(path),
    ], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"fixture gen failed: {r.stderr[-300:]}")
    return path


def fake_transcript(duration_s: float) -> dict:
    """Synthetic word timestamps every ~0.4s across the whole video."""
    segments, words_all, t = [], [], 0.0
    sentence = "ini adalah video contoh untuk menguji pipeline pemotongan klip otomatis"
    i = 0
    while t < duration_s:
        word = sentence.split()[i % len(sentence.split())]
        w = {"word": word, "start": round(t, 3), "end": round(t + 0.38, 3),
             "probability": 0.99}
        words_all.append(w)
        t += 0.4
        i += 1
    # group into segments of ~5s
    seg, seg_start = [], 0.0
    for w in words_all:
        if w["end"] - seg_start >= 5.0 or not seg:
            if seg:
                segments.append({"start": seg_start, "end": w["start"],
                                 "text": " ".join(x["word"] for x in seg), "words": seg})
            seg, seg_start = [w], w["start"]
        else:
            seg.append(w)
    if seg:
        last = seg[-1]["end"]
        segments.append({"start": seg_start, "end": last,
                         "text": " ".join(x["word"] for x in seg), "words": seg})
    return {"text": " ".join(x["word"] for x in words_all),
            "segments": segments, "language": "id"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. fixture
    src = make_video(OUT / "source.mp4")
    dur = clip_mod.probe_duration(src)
    print(f"[1/6] fixture ok {src.name} dur={dur:.1f}s")

    # 2. transcript (injected — real STT needs speech)
    transcript = fake_transcript(dur)
    print(f"[2/6] transcript ok segs={len(transcript['segments'])} words="
          f"{sum(len(s['words']) for s in transcript['segments'])}")

    # 3. curation — offline, expect heuristic fallback
    plan = curation.curate_clips(transcript, dur, clip_max_s=30, clip_count=5,
                                 llm_config=config.LLMConfig(provider="offline"))
    shorts = plan.get("shorts", [])
    assert 1 <= len(shorts) <= 5, f"expected 1-5 shorts, got {len(shorts)}"
    for s in shorts:
        assert 0 <= s["start"] < s["end"] <= dur + 0.5
        assert s["end"] - s["start"] <= 30.5
    print(f"[3/6] curation ok shorts={len(shorts)} "
          f"(provider={plan.get('model', 'heuristic')})")

    # 4. cut + reframe 9:16
    reframed = OUT / "reframed.mp4"
    if not reframed.exists():
        clip_mod.reframe_to_vertical(src, reframed,
                                     scene_strategy=clip_mod.scene_strategy_for(src))
    r_w, r_h = clip_mod.cv2.VideoCapture(str(reframed)).get(3), \
        clip_mod.cv2.VideoCapture(str(reframed)).get(4)
    print(f"[4/6] reframe ok {int(r_w)}x{int(r_h)} (expect ~1080x1920)")
    assert abs(r_w / r_h - 9 / 16) < 0.02

    # 5. subtitles + srt
    cuts = []
    for i, s in enumerate(shorts[:3], start=1):
        cut = OUT / f"clip_{i}.mp4"
        clip_mod.cut_clip(reframed, s["start"], s["end"], cut)
        srt = OUT / f"clip_{i}.srt"
        tc = {"segments": [sg for sg in transcript["segments"]
                           if sg["end"] > s["start"] and sg["start"] < s["end"]]}
        ok = sub_mod.generate_srt(tc, s["start"], s["end"], srt)
        assert ok
        burnt = OUT / f"clip_{i}_sub.mp4"
        sub_mod.burn_subtitles(cut, srt, burnt, preset_name="bold_yellow")
        cuts.append({"start_s": s["start"], "end_s": s["end"], "hook": s.get("hook", ""),
                     "file": str(burnt), "srt_path": str(srt)})
    print(f"[5/6] subtitle burn ok clips={len(cuts)}")

    # 6. export batch
    items = [{**c, "included": True, "subtitle_preset": "classic_white"} for c in cuts]
    res = export_mod.export_batch(items, OUT / "export", "FixtureTest",
                                  aspect="9:16", resolution="720p")
    for r in res:
        assert r["file"] and Path(r["file"]).exists(), f"export failed: {r}"
    print(f"[6/6] export ok files={[Path(r['file']).name for r in res]}")

    print(f"\nALL OK in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
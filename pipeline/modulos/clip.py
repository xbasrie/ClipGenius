"""Clip cutting + vertical reframing. Ported from OpenShorts main.py, de-torch'd.

Ladder note: OpenShorts used MediaPipe + YOLOv8 (torch, ~2GB of deps) for subject
tracking. We use OpenCV's bundled Haar cascade instead — zero extra bytes, same
"don't jitter the camera" behaviour. ponytail: ceiling = frontal faces only;
upgrade path -> re-enable MediaPipe/YOLO behind a settings flag if accuracy demands it.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Iterable

import cv2

from ..config import ffmpeg

ASPECT_9_16 = 9 / 16
VERTICAL_H = 1920

_CASCADE = None


def _cascade() -> cv2.CascadeClassifier:
    global _CASCADE
    if _CASCADE is None:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        _CASCADE = cv2.CascadeClassifier(str(path))
    return _CASCADE


# --------------------------------------------------------------------------- camera


class SmoothedCameraman:
    """Heavy-tripod logic: hold still inside a safe zone, pan slowly when the subject
    leaves it, snap only on scene cuts. Same behaviour as OpenShorts' version."""

    def __init__(self, out_w: int, out_h: int, vid_w: int, vid_h: int) -> None:
        self.out_w, self.out_h = out_w, out_h
        self.vid_w, self.vid_h = vid_w, vid_h
        self.current_center_x = vid_w / 2
        self.target_center_x = vid_w / 2
        self.crop_h = vid_h
        self.crop_w = min(int(vid_h * ASPECT_9_16), vid_w)
        if self.crop_w >= vid_w:
            self.crop_h = int(self.crop_w / ASPECT_9_16)
        self.safe_zone_radius = self.crop_w * 0.25

    def update_target(self, box: tuple[int, int, int, int] | None) -> None:
        if box:
            x, _y, w, _h = box
            self.target_center_x = x + w / 2

    def get_crop_box(self, force_snap: bool = False) -> tuple[int, int, int, int]:
        if force_snap:
            self.current_center_x = self.target_center_x
        else:
            diff = self.target_center_x - self.current_center_x
            if abs(diff) > self.safe_zone_radius:
                direction = 1 if diff > 0 else -1
                speed = 15.0 if abs(diff) > self.crop_w * 0.5 else 3.0
                self.current_center_x += direction * speed
                new_diff = self.target_center_x - self.current_center_x
                if (direction == 1 and new_diff < 0) or (direction == -1 and new_diff > 0):
                    self.current_center_x = self.target_center_x

        half = self.crop_w / 2
        self.current_center_x = max(half, min(self.current_center_x, self.vid_w - half))
        x1 = max(0, int(self.current_center_x - half))
        x2 = min(self.vid_w, int(self.current_center_x + half))
        return x1, 0, x2, self.vid_h


class SpeakerTracker:
    """Sticky speaker selection: score decays, current speaker gets a bonus, switches
    are rate-limited. Prevents the crop from ping-ponging between faces."""

    def __init__(self, cooldown_frames: int = 30) -> None:
        self.active_id: int | None = None
        self.scores: dict[int, float] = {}
        self.known: list[dict] = []
        self.last_switch = -10_000
        self.cooldown = cooldown_frames
        self.next_id = 0

    def get_target(self, boxes: list[tuple[int, int, int, int]], frame_no: int,
                   width: int) -> tuple[int, int, int, int] | None:
        if not boxes:
            return None

        cands = []
        for box in boxes:
            x, _y, w, _h = box
            cx = x + w / 2
            best_id, best_dist = -1, width * 0.15
            for kf in self.known:
                if frame_no - kf["last"] > 30:
                    continue
                d = abs(cx - kf["center"])
                if d < best_dist:
                    best_dist, best_id = d, kf["id"]
            if best_id == -1:
                best_id = self.next_id
                self.next_id += 1
            self.known = [k for k in self.known if k["id"] != best_id]
            self.known.append({"id": best_id, "center": cx, "last": frame_no})
            cands.append({"id": best_id, "box": box, "area": w * _h})

        for pid in list(self.scores):
            self.scores[pid] *= 0.85
            if self.scores[pid] < 0.1:
                del self.scores[pid]
        for c in cands:
            self.scores[c["id"]] = self.scores.get(c["id"], 0) + c["area"] / (width * width * 0.05)

        best, best_score = None, -1.0
        for c in cands:
            s = self.scores.get(c["id"], 0) * (3.0 if c["id"] == self.active_id else 1.0)
            if s > best_score:
                best, best_score = c, s

        if best is None:
            return None
        if best["id"] != self.active_id:
            if frame_no - self.last_switch < self.cooldown:
                old = next((c for c in cands if c["id"] == self.active_id), None)
                if old:
                    return old["box"]
            self.active_id = best["id"]
            self.last_switch = frame_no
        return best["box"]


# --------------------------------------------------------------------------- reframe


def detect_faces(frame) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = _cascade().detectMultiScale(gray, scaleFactor=1.15, minNeighbors=6,
                                        minSize=(60, 60))
    return [tuple(int(v) for v in f) for f in faces]


def blurred_general_frame(frame, out_w: int, out_h: int):
    """GENERAL mode: keep the full width, fill the rest with a blurred crop."""
    h, w = frame.shape[:2]
    bg = cv2.resize(frame, (out_w, out_h))
    bg = cv2.GaussianBlur(bg, (0, 0), 25)
    fit_w = out_w
    fit_h = int(h * out_w / w)
    if fit_h > out_h:
        fit_h = out_h
        fit_w = int(w * out_h / h)
    fg = cv2.resize(frame, (fit_w, fit_h))
    y0 = (out_h - fit_h) // 2
    x0 = (out_w - fit_w) // 2
    bg[y0:y0 + fit_h, x0:x0 + fit_w] = fg
    return bg


def reframe_to_vertical(input_video: str | Path, output_video: str | Path,
                        scene_strategy: Callable[[float], str] | None = None,
                        progress_cb: Callable[[float], None] | None = None) -> Path:
    """Stream frames -> ffmpeg stdin. Never loads the whole video into RAM."""
    input_video, output_video = Path(input_video), Path(output_video)
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    out_h = 1920  # High Definition 1080x1920
    out_w = 1080
    if out_w % 2:
        out_w += 1

    cameraman = SmoothedCameraman(out_w, out_h, vid_w, vid_h)
    tracker = SpeakerTracker()
    scenes = detect_scenes(input_video, fps)

    output_video.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_video.with_suffix(".tmp.mp4")
    cmd = [
        ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-b:v", "8M", "-maxrate", "12M", "-bufsize", "16M",
        "-pix_fmt", "yuv420p", "-an", str(tmp_out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE)
    frame_no = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t = frame_no / fps
            mode = scene_strategy(t) if scene_strategy else "TRACK"
            if mode == "GENERAL":
                out = blurred_general_frame(frame, out_w, out_h)
                cameraman.current_center_x = vid_w / 2
                cameraman.target_center_x = vid_w / 2
            else:
                if frame_no % 2 == 0:
                    cameraman.update_target(tracker.get_target(detect_faces(frame), frame_no, vid_w))
                x1, y1, x2, y2 = cameraman.get_crop_box(force_snap=is_scene_start(frame_no, fps, scenes))
                if x2 > x1 and y2 > y1:
                    out = cv2.resize(frame[y1:y2, x1:x2], (out_w, out_h))
                else:
                    out = cv2.resize(frame, (out_w, out_h))
            proc.stdin.write(out.tobytes())  # type: ignore[union-attr]
            frame_no += 1
            if progress_cb and frame_no % 15 == 0:
                progress_cb(frame_no / total)
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        proc.wait()

    if proc.returncode != 0:
        tmp_out.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg reframe failed: {stderr[-400:]}")

    audio_tmp = tmp_out.with_name(tmp_out.stem + ".aac")
    has_audio = _has_audio_stream(input_video)
    if has_audio:
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(input_video),
                        "-vn", "-c:a", "aac", "-b:a", "128k", str(audio_tmp)],
                       capture_output=True)
        subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(tmp_out),
                        "-i", str(audio_tmp), "-c:v", "copy", "-c:a", "copy",
                        "-shortest", str(output_video)], capture_output=True)
        tmp_out.unlink(missing_ok=True)
        audio_tmp.unlink(missing_ok=True)
    else:
        tmp_out.replace(output_video)
    return output_video


def _has_audio_stream(path: Path) -> bool:
    from ..config import ffprobe

    r = subprocess.run([ffprobe(), "-v", "error", "-select_streams", "a:0",
                        "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    return "audio" in (r.stdout or "")


def detect_scenes(video_path: Path, fps: float | None = None) -> list[tuple[float, float]]:
    """Scene boundaries in seconds. Empty list on failure — caller treats video as one scene."""
    try:
        from scenedetect import ContentDetector, SceneManager, open_video

        video = open_video(str(video_path))
        manager = SceneManager()
        manager.add_detector(ContentDetector())
        manager.detect_scenes(video, show_progress=False)
        return [(s.get_seconds(), e.get_seconds()) for s, e in manager.get_scene_list()]
    except Exception:  # noqa: BLE001 — scene detection is best-effort
        return []


def is_scene_start(frame_no: int, fps: float, scenes: list[tuple[float, float]]) -> bool:
    if not scenes or fps <= 0:
        return False
    t = frame_no / fps
    return any(abs(t - start) < (1.5 / fps) for start, _ in scenes)


def scene_strategy_for(video_path: Path, sample_frames: int = 8) -> Callable[[float], str]:
    """Sample the video; if faces dominate -> TRACK, else GENERAL (groups/screen share)."""
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    hits = 0
    checked = 0
    for i in range(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * (i + 0.5) / sample_frames))
        ok, frame = cap.read()
        if not ok:
            continue
        checked += 1
        if detect_faces(frame):
            hits += 1
    cap.release()
    mode = "TRACK" if checked and hits / checked >= 0.5 else "GENERAL"
    return lambda _t: mode


# --------------------------------------------------------------------------- cut


def cut_clip(source: str | Path, start_s: float, end_s: float, output: str | Path,
             reencode: bool = True) -> Path:
    """Precise cut. Re-encode by default so the cut lands on the requested frame."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    dur = max(0.1, end_s - start_s)
    if reencode:
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(source),
               "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "fast", "-crf", "20",
               "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)]
    else:
        cmd = [ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{start_s:.3f}", "-i", str(source),
               "-t", f"{dur:.3f}", "-c", "copy", "-movflags", "+faststart", str(output)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg cut failed: {r.stderr[-400:]}")
    return output


def probe_duration(path: str | Path) -> float:
    from ..config import ffprobe

    r = subprocess.run([ffprobe(), "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return float((r.stdout or "0").strip())
    except ValueError:
        return 0.0


def text_overlay(video_path: str | Path, text: str, output_path: str | Path,
                 fontsize: int = 54, position: str = "top") -> Path:
    """Hook text via drawtext — no PIL, no extra font download, no overlay PNG step."""
    video_path, output_path = Path(video_path), Path(output_path)
    safe = text.replace("\\", "").replace("'", "").replace(":", "\\:").replace("%", "")
    y_expr = "h*0.10" if position == "top" else "h*0.80"
    vf = (
        f"drawtext=text='{safe}':fontcolor=white:fontsize={fontsize}:"
        f"box=1:boxcolor=black@0.55:boxborderw=14:"
        f"x=(w-text_w)/2:y={y_expr}"
    )
    r = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-i", str(video_path),
                        "-vf", vf, "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                        "-c:a", "copy", str(output_path)], capture_output=True, text=True)
    if r.returncode != 0:
        shutil.copy2(video_path, output_path)
    return output_path


def make_thumbnail(video_path: str | Path, output_path: str | Path, at_s: float = 1.0) -> Path | None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([ffmpeg(), "-y", "-loglevel", "error", "-ss", f"{max(0.0, at_s):.2f}",
                        "-i", str(video_path), "-frames:v", "1", "-q:v", "3", str(output_path)],
                       capture_output=True)
    return output_path if r.returncode == 0 and output_path.exists() else None


def safe_temp_dir(prefix: str = "cg_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix))
"""FFmpeg helpers for the shorts pipeline.

Ported from AutoShorts' Rust ``media.rs``. The binaries come from the
``static-ffmpeg`` package rather than the system PATH, so this works on a bare
container (Hugging Face Spaces, a VM, CI) with no apt-get and no admin rights -
which is the whole reason the original desktop app's "install ffmpeg first"
prerequisite does not survive a move to hosting.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_ffmpeg_lock = threading.Lock()
_ffmpeg_ready = False


class ShortsMediaError(RuntimeError):
    """Raised when ffmpeg/ffprobe is unavailable or a command fails."""


def ensure_ffmpeg() -> None:
    """Put the bundled ffmpeg/ffprobe on PATH once per process.

    ``static_ffmpeg`` downloads the binaries on first call, so this is slow the
    very first time and instant afterwards.
    """
    global _ffmpeg_ready

    if _ffmpeg_ready:
        return

    with _ffmpeg_lock:
        if _ffmpeg_ready:
            return

        if shutil.which("ffmpeg") and shutil.which("ffprobe"):
            _ffmpeg_ready = True
            return

        try:
            import static_ffmpeg

            static_ffmpeg.add_paths()
        except Exception as error:
            raise ShortsMediaError(
                "ffmpeg is unavailable. Install the static-ffmpeg package or put " "ffmpeg and ffprobe on PATH."
            ) from error

        if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
            raise ShortsMediaError("ffmpeg/ffprobe could not be located after setup.")

        _ffmpeg_ready = True
        logger.info(
            "ffmpeg ready",
            extra={"module_name": "shorts", "action": "ffmpeg-ready", "path": shutil.which("ffmpeg")},
        )


def _run(command: list[str], *, what: str, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired as error:
        raise ShortsMediaError(f"{what} timed out after {timeout}s.") from error
    except OSError as error:
        raise ShortsMediaError(f"{what} could not start: {error}") from error

    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout or "").strip()
        raise ShortsMediaError(f"{what} failed: {detail[-800:]}")

    return completed


@dataclass
class MediaProbe:
    duration_sec: float | None
    has_video: bool
    has_audio: bool
    width: int | None
    height: int | None
    video_codec: str | None
    audio_codec: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "duration_sec": round(self.duration_sec, 2) if self.duration_sec else None,
            "has_video": self.has_video,
            "has_audio": self.has_audio,
            "width": self.width,
            "height": self.height,
            "video_codec": self.video_codec,
            "audio_codec": self.audio_codec,
        }


def probe_media(path: Path) -> MediaProbe:
    ensure_ffmpeg()
    completed = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        what="ffprobe",
        timeout=120,
    )

    try:
        payload = json.loads(completed.stdout)
    except ValueError as error:
        raise ShortsMediaError("ffprobe returned output that could not be parsed.") from error

    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = payload.get("format", {}).get("duration")
    try:
        duration_sec = float(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_sec = None

    return MediaProbe(
        duration_sec=duration_sec,
        has_video=video is not None,
        has_audio=audio is not None,
        width=video.get("width") if video else None,
        height=video.get("height") if video else None,
        video_codec=video.get("codec_name") if video else None,
        audio_codec=audio.get("codec_name") if audio else None,
    )


def extract_audio(source: Path, destination_dir: Path) -> Path:
    """Extract 16 kHz mono WAV - the format both Whisper and Deepgram expect."""
    ensure_ffmpeg()
    destination_dir.mkdir(parents=True, exist_ok=True)
    output = destination_dir / "transcription_audio.wav"

    _run(
        ["ffmpeg", "-y", "-i", str(source), "-vn", "-ac", "1", "-ar", "16000", str(output)],
        what="ffmpeg audio extraction",
    )
    return output


def _escape_drawtext(text: str) -> str:
    """Escape a caption for ffmpeg's drawtext filter."""
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "’")
        .replace("%", "\\%")
        .replace("[", "\\[")
        .replace("]", "\\]")
        .replace(",", "\\,")
    )


def _group_caption_phrases(
    words: list[dict], start_sec: float, end_sec: float, *, max_words: int = 4, max_gap: float = 0.7
) -> list[tuple[float, float, str]]:
    """Group word timings into short phrases rebased to the clip start.

    One drawtext filter per word produced ~150 filters on a 60s clip, which is
    slow to encode and can blow past the OS command-line length limit. Grouping
    into a few-word phrases keeps captions readable with far fewer filters.
    """
    phrases: list[tuple[float, float, str]] = []
    bucket: list[str] = []
    bucket_start: float | None = None
    bucket_end: float | None = None

    for word in words:
        w_start = float(word.get("start", 0.0))
        w_end = float(word.get("end", 0.0))
        text = str(word.get("text", "")).strip()
        if not text or w_end <= start_sec or w_start >= end_sec:
            continue

        gap = bucket_end is not None and (w_start - bucket_end) > max_gap
        if bucket and (len(bucket) >= max_words or gap):
            phrases.append((bucket_start, bucket_end, " ".join(bucket)))
            bucket, bucket_start, bucket_end = [], None, None

        if bucket_start is None:
            bucket_start = max(0.0, w_start - start_sec)
        bucket_end = max((bucket_start or 0.0) + 0.05, w_end - start_sec)
        bucket.append(text)

    if bucket:
        phrases.append((bucket_start, bucket_end, " ".join(bucket)))
    return phrases


def build_caption_filter(words: list[dict], start_sec: float, end_sec: float) -> str:
    """Burn-in captions as time-gated drawtext filters, one per short phrase.

    Times are rebased to the clip, because the clip starts at 0 once cut.
    """
    parts: list[str] = []
    for phrase_start, phrase_end, text in _group_caption_phrases(words, start_sec, end_sec):
        parts.append(
            f"drawtext=text='{_escape_drawtext(text.upper())}':fontcolor=white:fontsize=h/18:borderw=4:bordercolor=black@0.85"
            f":x=(w-text_w)/2:y=h*0.72:enable='between(t,{phrase_start:.2f},{phrase_end:.2f})'"
        )

    return ",".join(parts)


def render_clip(
    source: Path,
    start_sec: float,
    end_sec: float,
    output: Path,
    *,
    caption_filter: str | None = None,
    portrait: bool = True,
    probe: MediaProbe | None = None,
) -> Path:
    """Cut a segment and centre-crop it to 9:16 portrait H.264.

    ``probe`` may be supplied by the caller so a multi-clip render probes the
    source once instead of re-running ffprobe for every clip.
    """
    ensure_ffmpeg()
    output.parent.mkdir(parents=True, exist_ok=True)

    if probe is None:
        probe = probe_media(source)
    command = ["ffmpeg", "-y", "-ss", f"{start_sec:.3f}", "-to", f"{end_sec:.3f}", "-i", str(source)]

    if probe.has_video:
        filters: list[str] = []
        if portrait:
            # Even dimensions are required by yuv420p, hence the 2*trunc(...).
            filters.append("crop=w='2*trunc(min(iw,ih*9/16)/2)':h='2*trunc(min(ih,iw*16/9)/2)'")
        if caption_filter:
            filters.append(caption_filter)

        if filters:
            command += ["-vf", ",".join(filters)]
        command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
    else:
        command.append("-vn")

    command += ["-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart", str(output)]
    _run(command, what="ffmpeg clip render")
    return output

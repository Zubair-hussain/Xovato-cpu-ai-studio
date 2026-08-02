"""Orchestration for the shorts pipeline.

Analysis (probe -> extract audio -> transcribe -> rank) and rendering both run in
the background against the shared ``JobStore``, because they take minutes on a
long video. A synchronous request would be cut off by Vercel's ~60s and
Cloudflare's ~100s proxy limits long before the work finished, so the browser
submits a job and polls it.

Note the job store is in-memory: jobs do not survive a restart and will not be
visible across multiple instances. Moving it to Postgres/Redis is a prerequisite
for running more than one replica.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import asdict
from pathlib import Path

from app.config import Settings
from app.models import JobCreate, MediaType
from app.services import job_store
from app.shorts_media import ShortsMediaError, build_caption_filter, extract_audio, probe_media, render_clip
from app.shorts_pipeline import (
    ShortsPipelineError,
    Transcript,
    rank_moments,
    transcribe_audio,
)

logger = logging.getLogger(__name__)

# Transcripts are kept in memory per project so rendering can reuse word
# timings for captions without transcribing twice.
_transcripts: dict[str, Transcript] = {}


def project_dir(settings: Settings, project_id: str) -> Path:
    return settings.data_dir / "shorts" / project_id


def save_upload(settings: Settings, project_id: str, filename: str, source) -> Path:
    directory = project_dir(settings, project_id)
    directory.mkdir(parents=True, exist_ok=True)

    suffix = Path(filename or "source.mp4").suffix.lower() or ".mp4"
    destination = directory / f"source{suffix}"
    with destination.open("wb") as handle:
        shutil.copyfileobj(source, handle)
    return destination


def create_analysis_job(project_id: str, filename: str) -> str:
    job = job_store.create(
        JobCreate(
            module="shorts-studio",
            media_type=MediaType.video,
            input_uri=f"upload://{filename}",
            model_chain=["ffprobe", "ffmpeg-extract", "transcribe", "rank-moments"],
            settings={"project_id": project_id},
        )
    )
    return job.id


def run_analysis(
    settings: Settings,
    *,
    job_id: str,
    project_id: str,
    source: Path,
    target_count: int,
    min_len: float,
    max_len: float,
    transcribe_provider: str | None,
    ranking_provider: str | None,
) -> None:
    """Probe, extract audio, transcribe and rank. Safe to run in a worker thread."""
    directory = project_dir(settings, project_id)

    try:
        job_store.set_progress(job_id, 5, "Probing media")
        probe = probe_media(source)
        if not probe.has_audio:
            raise ShortsPipelineError("This file has no audio track, so there is nothing to transcribe.")
        job_store.set_result(job_id, {"probe": probe.as_dict(), "project_id": project_id})

        job_store.set_progress(job_id, 15, "Extracting audio")
        audio = extract_audio(source, directory)

        job_store.set_progress(job_id, 30, "Transcribing")
        transcript = transcribe_audio(audio, provider=transcribe_provider)
        _transcripts[project_id] = transcript
        job_store.set_result(job_id, {"transcript": transcript.as_dict()})

        job_store.set_progress(job_id, 80, "Ranking moments")
        candidates, engine = rank_moments(
            transcript,
            target_count=target_count,
            min_len=min_len,
            max_len=max_len,
            provider=ranking_provider,
        )

        if not candidates:
            raise ShortsPipelineError(
                "No moments long enough were found. Try lowering the minimum clip length."
            )

        job_store.set_result(
            job_id,
            {
                "candidates": [candidate.as_dict() for candidate in candidates],
                "ranking_engine": engine,
                "transcribe_engine": transcript.engine,
            },
        )
        job_store.set_progress(job_id, 100, "Complete")
        job_store.mark_completed(job_id, f"local://shorts/{project_id}")
        logger.info(
            "shorts analysis complete",
            extra={
                "module_name": "shorts-studio",
                "action": "analyse",
                "job_id": job_id,
                "candidates": len(candidates),
                "ranking_engine": engine,
            },
        )
    except (ShortsMediaError, ShortsPipelineError) as error:
        job_store.mark_failed(job_id, str(error))
        logger.error(
            "shorts analysis failed",
            extra={"module_name": "shorts-studio", "action": "analyse-failed", "job_id": job_id, "warning": str(error)},
        )
    except Exception as error:  # pragma: no cover - unexpected failure guard
        job_store.mark_failed(job_id, f"Unexpected failure: {error}")
        logger.exception("shorts analysis crashed", extra={"module_name": "shorts-studio", "action": "analyse-crash"})


def run_render(
    settings: Settings,
    *,
    job_id: str,
    project_id: str,
    source: Path,
    clips: list[dict],
    captions: bool,
) -> None:
    """Render the chosen moments to 9:16 clips."""
    directory = project_dir(settings, project_id)
    transcript = _transcripts.get(project_id)
    rendered: list[dict] = []

    try:
        for index, clip in enumerate(clips):
            start = float(clip["start_sec"])
            end = float(clip["end_sec"])
            label = f"Rendering clip {index + 1} of {len(clips)}"
            job_store.set_progress(job_id, int(5 + (index / max(len(clips), 1)) * 90), label)

            caption_filter = None
            if captions and transcript and transcript.words:
                caption_filter = build_caption_filter(
                    [asdict(word) for word in transcript.words], start, end
                ) or None

            output = directory / f"clip_{index + 1:02d}.mp4"
            render_clip(source, start, end, output, caption_filter=caption_filter)

            rendered.append(
                {
                    "index": index + 1,
                    "start_sec": round(start, 2),
                    "end_sec": round(end, 2),
                    "hook": clip.get("hook", ""),
                    "uri": f"local://shorts/{project_id}/{output.name}",
                    "size_bytes": output.stat().st_size,
                }
            )
            job_store.set_result(job_id, {"clips": rendered})

        job_store.set_progress(job_id, 100, "Complete")
        job_store.mark_completed(job_id, f"local://shorts/{project_id}")
        logger.info(
            "shorts render complete",
            extra={"module_name": "shorts-studio", "action": "render", "job_id": job_id, "clips": len(rendered)},
        )
    except (ShortsMediaError, ShortsPipelineError) as error:
        job_store.set_result(job_id, {"clips": rendered})
        job_store.mark_failed(job_id, str(error))
    except Exception as error:  # pragma: no cover
        job_store.set_result(job_id, {"clips": rendered})
        job_store.mark_failed(job_id, f"Unexpected failure: {error}")

import base64
import json
import logging
import os
import re
import threading
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import soundfile as sf
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Response, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.audio_processing import (
    OUTPUT_MIME_TYPES,
    SUPPORTED_INPUT_FORMATS,
    AudioProcessingError,
    enhance_audio,
)
from app.config import get_settings
from app.face_restoration import FaceRestorationError, face_models_available
from app.image_processing import (
    MIME_TYPES,
    ImageProcessingError,
    enhance_image,
    remove_background,
    resolve_format,
    restore_faces_image,
    segmentation_model_available,
)
from app.logging_config import configure_logging
from app.models import (
    AudiobookRequest,
    DubbingRequest,
    Job,
    JobCreate,
    LogEvent,
    MediaType,
    OpenAISpeechRequest,
    ShortsRenderRequest,
    SpeechRequest,
    StoryRequest,
    TranscriptionRequest,
    VoiceDesignRequest,
    VoiceProfile,
    VoiceProfileCreate,
    VoiceSample,
)
from app.omnivoice_runtime import model_ready
from app.services import job_store, omnivoice_breaker, segmentation_breaker, system_health, voice_store
from app.shorts_media import ShortsMediaError, ensure_ffmpeg
from app.shorts_service import (
    UploadTooLarge,
    create_analysis_job,
    run_analysis,
    run_render,
    save_upload,
)
from app.shorts_service import (
    project_dir as shorts_project_dir,
)
from app.supabase_output import output_bucket_name, record_completed_output, supabase_output_enabled
from app.voice_engine import VoiceEngineError, voice_engine
from app.voice_library import get_stock_voice, library_payload
from app.voice_styles import (
    EMOTION_PRESETS,
    VALID_INSTRUCT_ITEMS,
    VoiceStyleError,
    instruct_string,
    resolve_style,
)

settings = get_settings()
configure_logging(settings.log_level, settings.log_dir)
omnivoice_breaker.recovery_seconds = settings.omnivoice_breaker_recovery_seconds
segmentation_breaker.recovery_seconds = settings.segmentation_breaker_recovery_seconds
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Local core backend for image, audio, and voice AI media workflows.",
)

# Bound the number of shorts jobs (ffmpeg + Whisper) that run at once so a burst
# of uploads cannot saturate the CPU and starve every other endpoint.
_shorts_semaphore = threading.BoundedSemaphore(max(1, settings.max_shorts_jobs))

# Project ids are minted as ``proj_<12 hex>``; reject anything else so the value
# cannot be used to walk the filesystem from the render endpoint.
_PROJECT_ID_RE = re.compile(r"^proj_[0-9a-f]{12}$")


def _guarded_shorts_task(func, *args, **kwargs) -> None:
    """Run a shorts background task while holding the concurrency semaphore."""
    with _shorts_semaphore:
        func(*args, **kwargs)


@app.on_event("startup")
def _prune_old_outputs() -> None:
    """Delete generated files older than the retention window, if one is set."""
    hours = settings.output_retention_hours
    if hours <= 0:
        return
    from app.utils.cleanup import remove_expired_files

    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    for root in (voice_store.outputs_dir, voice_store.samples_dir, settings.data_dir / "shorts"):
        if root.exists():
            removed = remove_expired_files(root, cutoff)
            if removed:
                logger.info(
                    "pruned expired files",
                    extra={"module_name": "system-health", "action": "retention-prune", "removed": removed},
                )


_cors_origins = settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Credentials cannot be combined with a wildcard origin per the CORS spec.
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Image-Analysis", "X-Audio-Analysis", "X-Job-Id", "X-Output-Format", "X-Module"],
)


def _local_runtime_warning() -> str | None:
    if settings.environment.lower() == "local":
        return "Running locally: CPU-only OmniVoice can use heavy system resources. Circuit breaker may pause synthesis and use preview fallback."
    return None


if _local_runtime_warning():
    logger.warning(
        _local_runtime_warning(),
        extra={"module_name": "system-health", "action": "health-check", "warning": _local_runtime_warning()},
    )


def _apply_health_circuit_breaker() -> dict[str, object]:
    """Report CPU health.

    This deliberately does NOT open the breaker on CPU load. It used to: any
    reading at or above the 85% warning threshold tripped it, and synthesis
    itself saturates the CPU by design. The result was that real synthesis
    tripped the breaker, every later request silently fell back to a Windows
    SAPI/tone preview, and the fallback persisted for the 30 minute recovery
    window - so users heard a generic robotic voice instead of their clone.

    The breaker now trips only on genuine synthesis failures, which is what a
    circuit breaker is for. High CPU during TTS is expected, not a fault.
    """
    health = system_health.snapshot()
    breaker_status = omnivoice_breaker.status()
    logger.info(
        "omnivoice health preflight",
        extra={
            "module_name": "text-to-speech",
            "action": "omnivoice-health-preflight",
            "cpu_percent": health["cpu_percent"],
            "circuit_state": breaker_status["state"],
            "retry_after_seconds": breaker_status["retry_after_seconds"],
            "warning": health["warning"],
        },
    )
    return health


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "ready"}


@app.get(f"{settings.api_prefix}/health")
def health() -> dict[str, object]:
    runtime_health = _apply_health_circuit_breaker()
    breaker_status = omnivoice_breaker.status()
    services = [
        {"name": "fastapi", "state": "operational"},
        {"name": "job-store", "state": "operational", "mode": "in-memory"},
        {"name": "postgres", "state": "not-configured"},
        {"name": "redis", "state": "not-configured"},
        {"name": "cloudflare-r2", "state": "not-configured"},
        {
            "name": "supabase-outputs",
            "state": "configured" if supabase_output_enabled() else "not-configured",
            "bucket": output_bucket_name(),
        },
    ]
    logger.info(
        "health check completed",
        extra={
            "module_name": "system-health",
            "action": "health-check",
            "cpu_percent": runtime_health["cpu_percent"],
            "circuit_state": breaker_status["state"],
            "retry_after_seconds": breaker_status["retry_after_seconds"],
            "warning": runtime_health["warning"] or _local_runtime_warning(),
        },
    )
    return {
        "status": "local-ready",
        "environment": settings.environment,
        "warning": _local_runtime_warning(),
        "runtime_health": runtime_health,
        "circuit_breaker": breaker_status,
        "services": services,
    }


@app.post(f"{settings.api_prefix}/logs", status_code=status.HTTP_202_ACCEPTED)
def create_log(event: LogEvent) -> dict[str, object]:
    level = event.level.lower()
    log_method = logger.error if level == "error" else logger.warning if level == "warn" else logger.info
    log_method(
        event.message,
        extra={"module_name": event.module, "action": event.action},
    )
    return {"ok": True, "event": event.model_dump()}


@app.post(f"{settings.api_prefix}/jobs", response_model=Job, status_code=status.HTTP_201_CREATED)
def create_job(payload: JobCreate) -> Job:
    job = job_store.create(payload)
    logger.info(
        "job queued",
        extra={"module_name": job.module, "action": "job-create", "job_id": job.id, "media_type": job.media_type},
    )
    return job


@app.get(f"{settings.api_prefix}/jobs", response_model=list[Job])
def list_jobs() -> list[Job]:
    return job_store.list()


@app.get(f"{settings.api_prefix}/jobs/{{job_id}}", response_model=Job)
def get_job(job_id: str) -> Job:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@app.post(f"{settings.api_prefix}/jobs/{{job_id}}/run", response_model=Job)
def run_job(job_id: str) -> Job:
    job = job_store.mark_running(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    if job.module == "background-removal":
        breaker_status = segmentation_breaker.status()
        if not segmentation_breaker.allow_request():
            job_store.mark_failed(job_id, "Segmentation circuit breaker is open.")
            logger.warning(
                "segmentation circuit breaker open",
                extra={
                    "module_name": "background-removal",
                    "action": "segmentation-breaker-open",
                    "job_id": job.id,
                    "retry_after_seconds": breaker_status["retry_after_seconds"],
                },
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "message": "Segmentation is paused after repeated failures.",
                    "circuit_breaker": breaker_status,
                },
            )
        # NB: do NOT record a breaker success here. This endpoint does no real
        # segmentation work, so counting it as a success would let anyone reset
        # the breaker with an empty job. The breaker is managed only where the
        # model actually runs (the background-removal endpoint).

    completed = job_store.mark_completed(job_id, output_uri=f"local://outputs/{job_id}")
    if completed is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    logger.info(
        "local job completed",
        extra={"module_name": completed.module, "action": "job-run-local", "job_id": completed.id},
    )
    return completed


@app.get(f"{settings.api_prefix}/voice/engines")
def list_voice_engines() -> dict[str, object]:
    ready = model_ready(settings.omnivoice_model_dir)
    breaker_status = omnivoice_breaker.status()
    return {
        "active_tts": "omnivoice-cpu"
        if ready and settings.use_real_omnivoice and breaker_status["state"] != "open"
        else "preview-wav",
        "active_asr": "whisper-local",
        "hardware": {"cuda": False, "mps": False, "cpu": True},
        "model_ready": ready,
        "model_dir": str(settings.omnivoice_model_dir),
        "circuit_breaker": breaker_status,
        "tts_engines": ["omnivoice-local", "xtts-local", "kokoro-local", "f5-tts-local"],
        "asr_engines": ["whisper-local", "faster-whisper-local", "sherpa-onnx-local"],
    }


@app.post(f"{settings.api_prefix}/voice/profiles", response_model=VoiceProfile, status_code=status.HTTP_201_CREATED)
def create_voice_profile(payload: VoiceProfileCreate) -> VoiceProfile:
    # Cloning a voice requires explicit consent from the person being cloned.
    if not payload.consent_verified:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "consent_verified must be true. Only clone a voice with the explicit "
                "consent of the person it belongs to."
            ),
        )
    profile = voice_store.create_clone(payload)
    logger.info("voice profile cloned", extra={"module_name": "voice-cloning", "action": "profile-create"})
    return profile


@app.get(f"{settings.api_prefix}/voice/profiles", response_model=list[VoiceProfile])
def list_voice_profiles() -> list[VoiceProfile]:
    return voice_store.list()


@app.post(f"{settings.api_prefix}/voice/samples", response_model=VoiceSample, status_code=status.HTTP_201_CREATED)
def upload_voice_sample(sample: UploadFile = File(...)) -> VoiceSample:
    if not (sample.content_type or "").startswith("audio/"):
        allowed = (".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm")
        if not (sample.filename or "").lower().endswith(allowed):
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Upload an audio sample file")

    try:
        saved = voice_store.save_sample(
            sample.filename or "voice-sample.wav",
            sample.content_type or "",
            sample.file,
            max_bytes=settings.max_voice_sample_bytes,
        )
    except ValueError as error:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail=str(error)) from error
    logger.info(
        "voice sample stored",
        extra={"module_name": "voice-cloning", "action": "sample-upload", "sample_id": saved.id},
    )
    return saved


@app.post(f"{settings.api_prefix}/voice/design", response_model=VoiceProfile, status_code=status.HTTP_201_CREATED)
def design_voice(payload: VoiceDesignRequest) -> VoiceProfile:
    profile = voice_store.create_design(payload)
    logger.info("voice designed", extra={"module_name": "voice-design", "action": "profile-design"})
    return profile


@app.post(f"{settings.api_prefix}/voice/speech", response_model=Job, status_code=status.HTTP_201_CREATED)
def synthesize_speech(payload: SpeechRequest, background: BackgroundTasks) -> Job:
    """Synthesise speech, cloning a stored voice profile when one is selected."""
    # A voice id is either one of the built-in stock voices (voice design mode,
    # no reference audio needed) or a cloned profile the user created.
    stock_voice = get_stock_voice(payload.voice_id)
    profile = None if stock_voice else (voice_store.get(payload.voice_id) if payload.voice_id else None)

    if payload.voice_id and stock_voice is None and profile is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice '{payload.voice_id}' was not found in the stock library or your cloned profiles.",
        )

    # A stock voice's identity, or a design profile's derived attributes, form
    # the base instruct layer the emotion and per-request overrides build on.
    if stock_voice:
        base_instruct = list(stock_voice.instruct_items)
    elif profile and profile.design_instruct:
        base_instruct = list(profile.design_instruct)
    else:
        base_instruct = None

    try:
        style = resolve_style(
            payload.emotion,
            base_instruct=base_instruct,
            extra_instruct=payload.instruct,
            speed=payload.speed,
            speed_multiplier=stock_voice.default_speed if stock_voice else 1.0,
            tilt_db=payload.tilt_db,
            dynamics=payload.dynamics,
            gain_db=payload.gain_db,
        )
    except VoiceStyleError as error:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    job = job_store.create(
        JobCreate(
            module="text-to-speech",
            media_type=MediaType.voice,
            input_uri="local://text/request",
            model_chain=["omnivoice-local"],
            settings={
                "text": payload.text,
                "voice_id": payload.voice_id or "auto",
                "voice_source": "stock" if stock_voice else "clone" if profile else "auto",
                "language": payload.language,
                "response_format": payload.response_format,
                "emotion": style.emotion,
                "instruct": instruct_string(style) or "none",
                "speed": style.speed,
            },
        )
    )

    if payload.response_format != "wav":
        job_store.mark_failed(job.id, "Only wav speech output is supported by the local OmniVoice backend.")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only wav speech output is supported by the local OmniVoice backend.",
        )

    ref_audio_path = voice_store.resolve_local_uri(profile.reference_uri) if profile else None
    if ref_audio_path is not None and not ref_audio_path.exists():
        ref_audio_path = None

    if not settings.use_real_omnivoice:
        completed = job_store.mark_completed(job.id, voice_store.write_preview_wav(job.id, payload.text)) or job
        # Supabase sync can take up to ~80s on a slow network; run it after the
        # response is sent instead of blocking the caller.
        background.add_task(record_completed_output, completed, voice_store)
        return completed

    if not omnivoice_breaker.allow_request():
        breaker_status = omnivoice_breaker.status()
        job_store.mark_failed(job.id, "OmniVoice is paused after repeated failures.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Speech synthesis is paused after repeated failures. "
                "The previous behaviour silently returned a robotic system voice instead; "
                "this now fails loudly so the problem is visible.",
                "circuit_breaker": breaker_status,
            },
        )

    job_store.mark_running(job.id)
    output_path = voice_store.output_path(job.id, ".wav")

    try:
        result = voice_engine.synthesize(
            settings,
            text=payload.text,
            style=style,
            language=payload.language,
            ref_audio=ref_audio_path,
            ref_text=(profile.reference_text if profile else None),
            cache_key=(profile.id if profile else None),
        )
    except VoiceEngineError as error:
        omnivoice_breaker.record_failure(error)
        job_store.mark_failed(job.id, str(error))
        logger.error(
            "omnivoice synthesis failed",
            extra={
                "module_name": "text-to-speech",
                "action": "synthesis-failed",
                "job_id": job.id,
                "warning": str(error),
            },
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    omnivoice_breaker.record_success()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), result.audio, result.sample_rate, subtype="PCM_16")

    duration = len(result.audio) / max(result.sample_rate, 1)
    logger.info(
        "speech synthesised",
        extra={
            "module_name": "text-to-speech",
            "action": "synthesis",
            "job_id": job.id,
            "cloned": result.cloned,
            "emotion": style.emotion,
            "generate_seconds": round(result.generate_seconds, 2),
            "realtime_factor": round(result.generate_seconds / duration, 3) if duration else None,
        },
    )
    completed = job_store.mark_completed(job.id, voice_store.uri_for_path(output_path)) or job
    background.add_task(record_completed_output, completed, voice_store)
    return completed


@app.get(f"{settings.api_prefix}/voice/library")
def list_stock_voices() -> dict[str, object]:
    """Ready-made voices that need no reference audio."""
    return {
        "voices": library_payload(),
        "note": (
            "Original voice personas rendered from the model's own attributes. "
            "They are not impersonations of real people, and because they need no "
            "reference audio they skip the cloning step entirely."
        ),
    }


@app.get(f"{settings.api_prefix}/voice/styles")
def list_voice_styles() -> dict[str, object]:
    """Emotion presets and the exact attribute vocabulary the model accepts."""
    return {
        "emotions": [
            {
                "id": preset.emotion,
                "description": preset.description,
                "instruct_items": preset.instruct_items,
                "speed": preset.speed,
                "tilt_db": preset.tilt_db,
                "dynamics": preset.dynamics,
                "gain_db": preset.gain_db,
            }
            for preset in EMOTION_PRESETS.values()
        ],
        "instruct_vocabulary": list(VALID_INSTRUCT_ITEMS),
        "note": (
            "OmniVoice has no emotion tokens. Emotions are built from the attributes it "
            "does support (pitch, whisper), its speed control, and a prosody pass over "
            "the rendered audio. For truly acted emotion, clone a reference clip that is "
            "already spoken that way."
        ),
    }


@app.get(f"{settings.api_prefix}/voice/engine")
def voice_engine_status() -> dict[str, object]:
    return {**voice_engine.status(), "circuit_breaker": omnivoice_breaker.status()}


@app.post(f"{settings.api_prefix}/voice/engine/warmup")
def warm_up_voice_engine() -> dict[str, object]:
    """Load the model ahead of the first request so it does not pay the cost."""
    try:
        voice_engine.load(settings)
    except VoiceEngineError as error:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(error)) from error
    return voice_engine.status()


@app.post(f"{settings.api_prefix}/voice/engine/unload")
def unload_voice_engine() -> dict[str, object]:
    voice_engine.unload()
    return voice_engine.status()


# These modules have a UI and a job shape but no wired-up backend yet. Rather
# than reporting a plain "completed" with an output file that does not exist,
# the job carries an explicit preview flag so callers can tell it is a stub.
_PREVIEW_NOTE = (
    "This module is a preview: the job is recorded but no media is produced yet. "
    "Wire the real backend before relying on the output_uri."
)


def _preview_job(job: Job, output_uri: str) -> Job:
    job_store.set_result(job.id, {"preview": True, "detail": _PREVIEW_NOTE})
    completed = job_store.mark_completed(job.id, output_uri) or job
    completed.stage = "preview"
    return completed


@app.post(f"{settings.api_prefix}/voice/transcriptions", response_model=Job, status_code=status.HTTP_201_CREATED)
def transcribe_audio(payload: TranscriptionRequest) -> Job:
    job = job_store.create(
        JobCreate(
            module="speech-to-text",
            media_type=MediaType.audio,
            input_uri=payload.audio_uri,
            model_chain=["whisper-local"],
            settings={
                "language": payload.language or "auto",
                "response_format": payload.response_format,
                "diarize": payload.diarize,
            },
        )
    )
    return _preview_job(job, f"local://outputs/{job.id}.{payload.response_format}")


@app.post(f"{settings.api_prefix}/voice/dubbing", response_model=Job, status_code=status.HTTP_201_CREATED)
def create_dub(payload: DubbingRequest) -> Job:
    settings_payload = {key: value for key, value in payload.model_dump().items() if value is not None}
    job = job_store.create(
        JobCreate(
            module="dubbing-studio",
            media_type=MediaType.voice,
            input_uri=payload.source_uri,
            model_chain=["whisper-local", "translation-local", "omnivoice-local", "ffmpeg"],
            settings=settings_payload,
        )
    )
    return _preview_job(job, f"local://outputs/{job.id}.{payload.export_format}")


@app.post(f"{settings.api_prefix}/voice/audiobooks", response_model=Job, status_code=status.HTTP_201_CREATED)
def create_audiobook(payload: AudiobookRequest) -> Job:
    settings_payload = {key: value for key, value in payload.model_dump().items() if value is not None}
    job = job_store.create(
        JobCreate(
            module="audiobook-editor",
            media_type=MediaType.voice,
            input_uri=payload.source_uri,
            model_chain=["chapter-detect", "omnivoice-local", "loudnorm"],
            settings=settings_payload,
        )
    )
    return _preview_job(job, f"local://outputs/{job.id}.m4b")


@app.post(f"{settings.api_prefix}/voice/stories", response_model=Job, status_code=status.HTTP_201_CREATED)
def create_story(payload: StoryRequest) -> Job:
    job = job_store.create(
        JobCreate(
            module="stories",
            media_type=MediaType.voice,
            input_uri="local://stories/script",
            model_chain=["multi-voice-render", "omnivoice-local"],
            settings={"title": payload.title, "line_count": len(payload.lines), "export_format": payload.export_format},
        )
    )
    return _preview_job(job, f"local://outputs/{job.id}.{payload.export_format}")


@app.get(f"{settings.api_prefix}/voice/diagnostics")
def voice_diagnostics() -> dict[str, object]:
    ready = model_ready(settings.omnivoice_model_dir)
    source_ready = (settings.omnivoice_source_dir / "omnivoice").exists()
    runtime_health = _apply_health_circuit_breaker()
    breaker_status = omnivoice_breaker.status()
    return {
        "status": "local-ready",
        "warning": runtime_health["warning"] or _local_runtime_warning(),
        "runtime_health": runtime_health,
        "checks": [
            {
                "name": "omnivoice-model",
                "state": "ready" if ready else "missing",
                "device": "cpu",
                "enabled": settings.use_real_omnivoice,
                "path": str(settings.omnivoice_model_dir),
            },
            {
                "name": "omnivoice-source",
                "state": "ready" if source_ready else "missing",
                "path": str(settings.omnivoice_source_dir),
            },
            {
                "name": "omnivoice-circuit-breaker",
                "state": breaker_status["state"],
                "failure_count": breaker_status["failure_count"],
                "retry_after_seconds": breaker_status["retry_after_seconds"],
            },
            {"name": "tts-engine", "state": "available", "engine": "omnivoice-cpu" if ready else "preview-wav"},
            {"name": "asr-engine", "state": "available", "engine": "whisper-local"},
            {"name": "cpu-preflight", "state": "cpu-mode"},
            {"name": "voice-store", "state": "operational", "profiles": len(voice_store.list())},
            {"name": "voice-data-dir", "state": "operational", "path": str(settings.data_dir)},
        ],
    }


def _read_upload(image: UploadFile) -> bytes:
    content_type = (image.content_type or "").lower()
    filename = (image.filename or "").lower()
    if not content_type.startswith("image/") and not filename.endswith(
        (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Upload a PNG, JPG, WEBP, BMP, GIF, or TIFF image.",
        )

    # Read one byte past the limit: if we get it, the upload is over the cap and
    # we reject it without pulling the whole (possibly huge) body into memory.
    limit = settings.max_image_upload_bytes
    data = image.file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The uploaded image was empty.")
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image exceeds the {limit // (1024 * 1024)} MB upload limit.",
        )
    return data


def _image_response(payload: bytes, analysis, pillow_format: str, module: str, job_id: str) -> Response:
    # The processed bytes are the body; the measurements ride along in a header
    # so the dashboard can render a real analysis without a second round trip.
    return Response(
        content=payload,
        media_type=MIME_TYPES.get(pillow_format, "application/octet-stream"),
        headers={
            "X-Image-Analysis": base64.b64encode(json.dumps(analysis.as_dict()).encode("utf-8")).decode("ascii"),
            "X-Job-Id": job_id,
            "X-Output-Format": pillow_format,
            "X-Module": module,
            "Cache-Control": "no-store",
        },
    )


@app.post(f"{settings.api_prefix}/images/enhance")
def enhance_image_endpoint(
    image: UploadFile = File(...),
    resolution: str = Form("source"),
    output_format: str = Form("jpg"),
    quality: int = Form(90),
    strength: float = Form(1.0),
) -> Response:
    data = _read_upload(image)
    job = job_store.create(
        JobCreate(
            module="image-enhancement",
            media_type=MediaType.image,
            input_uri=f"upload://{image.filename or 'image'}",
            model_chain=["auto-levels", "median-denoise", "lanczos-resample", "unsharp-mask"],
            settings={"resolution": resolution, "output_format": output_format, "quality": quality},
        )
    )
    job_store.mark_running(job.id)

    try:
        payload, analysis = enhance_image(
            data, resolution=resolution, output_format=output_format, quality=quality, strength=strength
        )
    except ImageProcessingError as error:
        job_store.mark_failed(job.id, str(error))
        logger.error(
            "image enhancement failed",
            extra={"module_name": "image-enhancement", "action": "enhance-failed", "job_id": job.id},
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    pillow_format = resolve_format(output_format, keep_alpha=False)
    job_store.mark_completed(job.id, f"inline://{job.id}.{pillow_format.lower()}")
    logger.info(
        "image enhanced",
        extra={
            "module_name": "image-enhancement",
            "action": "enhance",
            "job_id": job.id,
            "output_bytes": len(payload),
        },
    )
    return _image_response(payload, analysis, pillow_format, "image-enhancement", job.id)


@app.post(f"{settings.api_prefix}/images/background-removal")
def remove_background_endpoint(
    image: UploadFile = File(...),
    resolution: str = Form("source"),
    output_format: str = Form("png"),
    quality: int = Form(95),
    edge_softness: float = Form(1.0),
) -> Response:
    data = _read_upload(image)

    # A missing model is a configuration problem, not a runtime failure - check
    # it *before* the breaker and do not record it as a failure. Otherwise two
    # requests with no model file would open the breaker and keep background
    # removal disabled for 15 minutes even after the file is added.
    if not segmentation_model_available(settings.segmentation_model_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Segmentation model is missing. Download u2netp.onnx to "
                f"{settings.segmentation_model_path} to enable background removal."
            ),
        )

    breaker_status = segmentation_breaker.status()
    if not segmentation_breaker.allow_request():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "message": "Segmentation is paused after repeated failures.",
                "circuit_breaker": breaker_status,
            },
        )

    job = job_store.create(
        JobCreate(
            module="background-removal",
            media_type=MediaType.image,
            input_uri=f"upload://{image.filename or 'image'}",
            model_chain=["u2netp-onnx", "guided-filter-matting", "alpha-composite"],
            settings={"resolution": resolution, "output_format": output_format, "quality": quality},
        )
    )
    job_store.mark_running(job.id)

    try:
        payload, analysis = remove_background(
            data,
            model_path=settings.segmentation_model_path,
            resolution=resolution,
            output_format=output_format,
            quality=quality,
            edge_softness=edge_softness,
        )
    except ImageProcessingError as error:
        job_store.mark_failed(job.id, str(error))
        segmentation_breaker.record_failure(error)
        logger.error(
            "background removal failed",
            extra={
                "module_name": "background-removal",
                "action": "cutout-failed",
                "job_id": job.id,
                "circuit_state": segmentation_breaker.status()["state"],
            },
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    pillow_format = resolve_format(output_format, keep_alpha=True)
    segmentation_breaker.record_success()
    job_store.mark_completed(job.id, f"inline://{job.id}.{pillow_format.lower()}")
    logger.info(
        "background removed",
        extra={
            "module_name": "background-removal",
            "action": "cutout",
            "job_id": job.id,
            "output_bytes": len(payload),
        },
    )
    return _image_response(payload, analysis, pillow_format, "background-removal", job.id)


@app.post(f"{settings.api_prefix}/images/face-restoration")
def restore_faces_endpoint(
    image: UploadFile = File(...),
    resolution: str = Form("source"),
    output_format: str = Form("jpg"),
    quality: int = Form(92),
    blend: float = Form(1.0),
    enhance_whole_image: bool = Form(True),
) -> Response:
    data = _read_upload(image)

    if not face_models_available(settings.face_detector_path, settings.face_restorer_path):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Face models are missing. Download yunet.onnx and gpen_bfr_256.onnx into "
                f"{settings.face_detector_path.parent} to enable face restoration."
            ),
        )

    job = job_store.create(
        JobCreate(
            module="face-restoration",
            media_type=MediaType.image,
            input_uri=f"upload://{image.filename or 'image'}",
            model_chain=["yunet-detect", "ffhq-align", "gpen-bfr-256", "feathered-composite"],
            settings={"resolution": resolution, "output_format": output_format, "blend": blend},
        )
    )
    job_store.mark_running(job.id)

    try:
        payload, analysis = restore_faces_image(
            data,
            detector_path=settings.face_detector_path,
            restorer_path=settings.face_restorer_path,
            resolution=resolution,
            output_format=output_format,
            quality=quality,
            blend=blend,
            enhance_whole_image=enhance_whole_image,
        )
    except (ImageProcessingError, FaceRestorationError) as error:
        job_store.mark_failed(job.id, str(error))
        logger.error(
            "face restoration failed",
            extra={"module_name": "face-restoration", "action": "restore-failed", "job_id": job.id},
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    pillow_format = resolve_format(output_format, keep_alpha=False)
    job_store.mark_completed(job.id, f"inline://{job.id}.{pillow_format.lower()}")
    logger.info(
        "faces restored",
        extra={
            "module_name": "face-restoration",
            "action": "restore",
            "job_id": job.id,
            "faces_detected": analysis.faces_detected,
        },
    )
    return _image_response(payload, analysis, pillow_format, "face-restoration", job.id)


@app.post(f"{settings.api_prefix}/audio/enhance")
def enhance_audio_endpoint(
    audio: UploadFile = File(...),
    denoise_strength: float = Form(0.6),
    target_lufs: float = Form(-16.0),
    peak_ceiling_dbfs: float = Form(-1.0),
    highpass_hz: float = Form(80.0),
    output_format: str = Form("wav"),
) -> Response:
    filename = (audio.filename or "").lower()
    content_type = (audio.content_type or "").lower()
    if not content_type.startswith("audio/") and not filename.endswith(
        tuple(f".{extension}" for extension in SUPPORTED_INPUT_FORMATS)
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Upload a WAV, MP3, FLAC, OGG, or AIFF file. M4A/AAC is not supported.",
        )

    limit = settings.max_audio_upload_bytes
    data = audio.file.read(limit + 1)
    if not data:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="The uploaded audio was empty.")
    if len(data) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Audio exceeds the {limit // (1024 * 1024)} MB upload limit.",
        )

    job = job_store.create(
        JobCreate(
            module="audio-enhancement",
            media_type=MediaType.audio,
            input_uri=f"upload://{audio.filename or 'audio'}",
            model_chain=["spectral-gate", "highpass", "bs1770-loudnorm", "peak-limiter"],
            settings={"denoise_strength": denoise_strength, "target_lufs": target_lufs},
        )
    )
    job_store.mark_running(job.id)

    try:
        payload, resolved_format, analysis = enhance_audio(
            data,
            denoise_strength=denoise_strength,
            target_lufs=target_lufs,
            peak_ceiling_dbfs=peak_ceiling_dbfs,
            highpass_hz=highpass_hz,
            output_format=output_format,
        )
    except AudioProcessingError as error:
        job_store.mark_failed(job.id, str(error))
        logger.error(
            "audio enhancement failed",
            extra={"module_name": "audio-enhancement", "action": "enhance-failed", "job_id": job.id},
        )
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error

    job_store.mark_completed(job.id, f"inline://{job.id}.{resolved_format}")
    logger.info(
        "audio enhanced",
        extra={
            "module_name": "audio-enhancement",
            "action": "enhance",
            "job_id": job.id,
            "output_bytes": len(payload),
        },
    )
    return Response(
        content=payload,
        media_type=OUTPUT_MIME_TYPES.get(resolved_format, "application/octet-stream"),
        headers={
            "X-Audio-Analysis": base64.b64encode(json.dumps(analysis.as_dict()).encode("utf-8")).decode("ascii"),
            "X-Job-Id": job.id,
            "X-Output-Format": resolved_format,
            "X-Module": "audio-enhancement",
            "Cache-Control": "no-store",
        },
    )


@app.post(f"{settings.api_prefix}/shorts/analyze", status_code=status.HTTP_202_ACCEPTED)
def analyze_shorts(
    background: BackgroundTasks,
    media: UploadFile = File(...),
    target_count: int = Form(5),
    min_len: float = Form(15.0),
    max_len: float = Form(60.0),
    transcribe_provider: str | None = Form(None),
    ranking_provider: str | None = Form(None),
) -> dict[str, object]:
    """Accept a long recording and start analysis in the background.

    Returns immediately with a job id. Analysis of a long video takes minutes,
    which no hosting proxy will hold a request open for, so the client polls
    /shorts/jobs/{id}.
    """
    filename = (media.filename or "").lower()
    allowed = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wav", ".mp3", ".flac", ".ogg", ".m4a")
    if not filename.endswith(allowed):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Upload a video or audio file ({', '.join(allowed)}).",
        )

    project_id = f"proj_{uuid4().hex[:12]}"
    try:
        # Enforce the size limit while streaming to disk, so an oversized upload
        # is rejected before it can fill the disk.
        source = save_upload(
            settings,
            project_id,
            media.filename or "source.mp4",
            media.file,
            max_bytes=settings.max_shorts_upload_bytes,
        )
    except UploadTooLarge as error:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Media exceeds the {settings.max_shorts_upload_bytes // (1024 * 1024)} MB upload limit.",
        ) from error

    size = source.stat().st_size

    # Clamp min_len first, then derive max_len from the clamped value - otherwise
    # a tiny min_len with a smaller max_len could yield max_len < min_len.
    clamped_min = max(3.0, min_len)
    clamped_max = max(clamped_min + 1.0, max_len)

    job_id = create_analysis_job(project_id, media.filename or "source")
    background.add_task(
        _guarded_shorts_task,
        run_analysis,
        settings,
        job_id=job_id,
        project_id=project_id,
        source=source,
        target_count=max(1, min(target_count, 20)),
        min_len=clamped_min,
        max_len=clamped_max,
        transcribe_provider=transcribe_provider,
        ranking_provider=ranking_provider,
    )

    logger.info(
        "shorts analysis queued",
        extra={"module_name": "shorts-studio", "action": "analyse-queued", "job_id": job_id},
    )
    return {"job_id": job_id, "project_id": project_id, "size_bytes": size, "status": "queued"}


@app.post(f"{settings.api_prefix}/shorts/{{project_id}}/render", status_code=status.HTTP_202_ACCEPTED)
def render_shorts(
    project_id: str,
    background: BackgroundTasks,
    payload: ShortsRenderRequest,
) -> dict[str, object]:
    # Validate the id shape before touching the filesystem so it cannot be used
    # to escape the shorts directory (e.g. "..").
    if not _PROJECT_ID_RE.match(project_id):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Invalid project id.")

    directory = shorts_project_dir(settings, project_id)
    source = next((path for path in directory.glob("source.*")), None)
    if source is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Project {project_id} was not found.")

    if not payload.clips:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Select at least one moment.")

    job = job_store.create(
        JobCreate(
            module="shorts-studio",
            media_type=MediaType.video,
            input_uri=f"local://shorts/{project_id}",
            model_chain=["ffmpeg-crop-9x16", "drawtext-captions"],
            settings={"project_id": project_id, "clip_count": len(payload.clips), "captions": payload.captions},
        )
    )

    background.add_task(
        _guarded_shorts_task,
        run_render,
        settings,
        job_id=job.id,
        project_id=project_id,
        source=source,
        clips=[clip.model_dump() for clip in payload.clips],
        captions=payload.captions,
    )
    return {"job_id": job.id, "project_id": project_id, "status": "queued"}


@app.get(f"{settings.api_prefix}/shorts/jobs/{{job_id}}", response_model=Job)
def get_shorts_job(job_id: str) -> Job:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
    return job


@app.get(f"{settings.api_prefix}/shorts/capabilities")
def shorts_capabilities() -> dict[str, object]:
    try:
        ensure_ffmpeg()
        ffmpeg_ready = True
        ffmpeg_detail = None
    except ShortsMediaError as error:
        ffmpeg_ready = False
        ffmpeg_detail = str(error)

    try:
        import faster_whisper  # noqa: F401

        local_asr = True
    except ImportError:
        local_asr = False

    return {
        "ffmpeg": {"available": ffmpeg_ready, "detail": ffmpeg_detail, "source": "static-ffmpeg (bundled)"},
        "transcription": {
            "local_whisper": local_asr,
            "deepgram": bool(os.getenv("DEEPGRAM_API_KEY")),
            "default": os.getenv("SHORTS_TRANSCRIBE_PROVIDER", "local"),
        },
        "ranking": {
            "heuristic": True,
            "llm": bool(
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("ANTHROPIC_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or os.getenv("OPENROUTER_API_KEY")
                or os.getenv("GROQ_API_KEY")
            ),
            "default": os.getenv("SHORTS_RANKING_PROVIDER", "heuristic"),
        },
        "output": {"aspect": "9:16", "codec": "h264/aac", "captions": True},
    }


@app.get(f"{settings.api_prefix}/audio/capabilities")
def audio_capabilities() -> dict[str, object]:
    return {
        "enhancement": {
            "available": True,
            "engine": "spectral gate + BS.1770 loudness normalisation (NumPy, CPU)",
        },
        "input_formats": list(SUPPORTED_INPUT_FORMATS),
        "output_formats": list(OUTPUT_MIME_TYPES),
        "unsupported": ["m4a", "aac"],
    }


@app.get(f"{settings.api_prefix}/images/capabilities")
def image_capabilities() -> dict[str, object]:
    ready = segmentation_model_available(settings.segmentation_model_path)
    breaker_status = segmentation_breaker.status()
    return {
        "enhancement": {"available": True, "engine": "auto-levels + unsharp (Pillow/NumPy, CPU)"},
        "background_removal": {
            "available": ready and breaker_status["state"] != "open",
            "engine": "u2netp ONNX (CPU)",
            "model_path": str(settings.segmentation_model_path),
            "detail": None if ready else "Model file missing; download u2netp.onnx.",
            "circuit_breaker": breaker_status,
        },
        "face_restoration": {
            "available": face_models_available(settings.face_detector_path, settings.face_restorer_path),
            "engine": "YuNet detect + GPEN-BFR-256 (CPU)",
            "note": "Generative: synthesises plausible detail rather than recovering original pixels.",
        },
        "formats": ["png", "jpg", "webp"],
        "resolutions": ["source", "1080p", "2k", "4k"],
        "aspect_ratio": "always preserved; presets fit inside the target box",
    }


# Only these subtrees of data/ are downloadable. Logs and the voice-profile
# index (which lists every user's voices) live elsewhere under data/ and must
# not be reachable through this endpoint.
_SERVABLE_ROOTS = ("outputs", "shorts")


@app.get(f"{settings.api_prefix}/files/{{local_path:path}}")
def read_local_file(local_path: str) -> FileResponse:
    path = voice_store.resolve_local_uri(f"local://{local_path}")
    if path is None or not path.exists() or not path.is_file():
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    data_root = voice_store.data_dir.resolve()
    allowed = any(path == (data_root / root) or (data_root / root) in path.parents for root in _SERVABLE_ROOTS)
    if not allowed:
        # 404 rather than 403 so we do not confirm the existence of files (logs,
        # profile index) outside the servable roots.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")
    return FileResponse(path)


@app.get("/v1/audio/voices")
def openai_list_voices() -> dict[str, object]:
    voices = voice_store.list()
    return {
        "object": "list",
        "data": [{"id": voice.id, "name": voice.name, "language": voice.language} for voice in voices],
    }


@app.post("/v1/audio/speech")
def openai_speech(payload: OpenAISpeechRequest, background: BackgroundTasks) -> Response:
    """OpenAI-compatible speech endpoint.

    Only ``wav`` is produced (the local OmniVoice backend's single format). This
    routes through the real synthesis path instead of returning text bytes
    mislabelled as audio, which corrupted any OpenAI client that called it.
    """
    if payload.response_format != "wav":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only 'wav' response_format is supported by the local backend.",
        )

    speech = synthesize_speech(
        SpeechRequest(text=payload.input, voice_id=payload.voice or None, response_format="wav"),
        background,
    )
    audio_path = voice_store.resolve_local_uri(speech.output_uri or "")
    if audio_path is None or not audio_path.exists():
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Speech output was not produced.")
    return Response(content=audio_path.read_bytes(), media_type="audio/wav", headers={"X-Job-Id": speech.id})


@app.post("/v1/audio/transcriptions")
def openai_transcriptions(file: UploadFile = File(...)) -> dict[str, str]:
    """OpenAI-compatible transcription using the local faster-whisper engine."""
    import tempfile
    from pathlib import Path as _Path

    from app.shorts_media import ShortsMediaError, extract_audio, probe_media
    from app.shorts_pipeline import ShortsPipelineError
    from app.shorts_pipeline import transcribe_audio as _transcribe

    suffix = _Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmp:
        source = _Path(tmp) / f"input{suffix}"
        source.write_bytes(file.file.read())
        try:
            probe = probe_media(source)
            audio = extract_audio(source, _Path(tmp)) if probe.has_video else source
            transcript = _transcribe(audio, provider="local")
        except (ShortsMediaError, ShortsPipelineError) as error:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(error)) from error
    return {"text": transcript.full_text}

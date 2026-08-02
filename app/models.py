from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, Field

SettingValue = str | int | float | bool | list[str]


class MediaType(StrEnum):
    image = "image"
    audio = "audio"
    voice = "voice"
    video = "video"


class JobStatus(StrEnum):
    queued = "queued"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobCreate(BaseModel):
    module: str = Field(..., examples=["image-enhancement"])
    media_type: MediaType
    input_uri: str = Field(..., examples=["local://uploads/sample.png"])
    model_chain: list[str] = Field(default_factory=list)
    settings: dict[str, SettingValue] = Field(default_factory=dict)


class Job(BaseModel):
    id: str = Field(default_factory=lambda: f"job_{uuid4().hex[:12]}")
    module: str
    media_type: MediaType
    input_uri: str
    model_chain: list[str]
    settings: dict[str, SettingValue]
    status: JobStatus = JobStatus.queued
    output_uri: str | None = None
    error: str | None = None
    # Long-running jobs (shorts rendering) report progress so the browser can
    # poll instead of holding an HTTP request open past a proxy timeout.
    progress: int = 0
    stage: str | None = None
    result: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class LogEvent(BaseModel):
    level: str = "info"
    module: str
    action: str
    message: str
    meta: dict[str, str | int | float | bool] = Field(default_factory=dict)


class VoiceProfileCreate(BaseModel):
    name: str = Field(..., examples=["Creator voice profile"])
    language: str = Field(default="en")
    reference_uri: str = Field(..., examples=["local://uploads/reference.wav"])
    consent_verified: bool = True
    tags: list[str] = Field(default_factory=list)
    # The real transcript of the reference clip. A wrong transcript measurably
    # degrades clone quality; leaving it empty lets OmniVoice auto-transcribe.
    reference_text: str | None = None


class VoiceProfile(BaseModel):
    id: str = Field(default_factory=lambda: f"voice_{uuid4().hex[:10]}")
    name: str
    language: str
    reference_uri: str
    consent_verified: bool
    tags: list[str] = Field(default_factory=list)
    reference_text: str | None = None
    engine: str = "omnivoice-local"
    portable_bundle_uri: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VoiceSample(BaseModel):
    id: str = Field(default_factory=lambda: f"sample_{uuid4().hex[:10]}")
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int
    uri: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class VoiceDesignRequest(BaseModel):
    name: str
    gender: str = "neutral"
    age: str = "adult"
    accent: str = "general"
    pitch: float = 1.0
    speed: float = 1.0
    emotion: str = "natural"
    dialect: str = "standard"
    language: str = "en"


class SpeechRequest(BaseModel):
    text: str
    voice_id: str | None = None
    language: str = "en"
    response_format: str = "wav"
    stream: bool = False
    # Style controls. `emotion` selects a preset; the rest override it.
    emotion: str = "neutral"
    instruct: list[str] = Field(default_factory=list)
    speed: float | None = None
    tilt_db: float | None = None
    dynamics: float | None = None
    gain_db: float | None = None


class TranscriptionRequest(BaseModel):
    audio_uri: str
    language: str | None = None
    response_format: str = "json"
    diarize: bool = False


class DubbingRequest(BaseModel):
    source_uri: str
    target_language: str
    voice_id: str | None = None
    preserve_background: bool = True
    export_format: str = "mp4"


class AudiobookRequest(BaseModel):
    title: str
    source_uri: str
    voice_id: str | None = None
    chapter_detection: bool = True
    loudnorm: bool = True


class StoryLine(BaseModel):
    speaker: str
    text: str
    voice_id: str | None = None


class StoryRequest(BaseModel):
    title: str
    lines: list[StoryLine]
    export_format: str = "wav"


class ShortsClip(BaseModel):
    start_sec: float
    end_sec: float
    hook: str = ""


class ShortsRenderRequest(BaseModel):
    clips: list[ShortsClip]
    captions: bool = True


class OpenAISpeechRequest(BaseModel):
    model: str = "tts-1"
    voice: str = "alloy"
    input: str
    response_format: str = "wav"

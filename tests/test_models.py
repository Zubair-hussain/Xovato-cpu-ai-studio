import pytest
from pydantic import ValidationError

from app.models import Job, JobCreate, MediaType, ShortsClip, ShortsRenderRequest, VoiceProfileCreate


def test_consent_defaults_to_false():
    payload = VoiceProfileCreate(name="v", reference_uri="local://x.wav")
    assert payload.consent_verified is False


def test_shorts_clip_rejects_reversed_window():
    with pytest.raises(ValidationError):
        ShortsClip(start_sec=10.0, end_sec=5.0)


def test_shorts_clip_rejects_negative_start():
    with pytest.raises(ValidationError):
        ShortsClip(start_sec=-1.0, end_sec=5.0)


def test_shorts_clip_accepts_valid_window():
    clip = ShortsClip(start_sec=2.0, end_sec=8.0, hook="hi")
    assert clip.end_sec > clip.start_sec


def test_render_request_requires_at_least_one_clip():
    with pytest.raises(ValidationError):
        ShortsRenderRequest(clips=[])


def test_render_request_caps_clip_count():
    clips = [ShortsClip(start_sec=float(i), end_sec=float(i) + 1.0) for i in range(21)]
    with pytest.raises(ValidationError):
        ShortsRenderRequest(clips=clips)


def test_job_defaults():
    job = Job(**JobCreate(module="m", media_type=MediaType.image, input_uri="u").model_dump())
    assert job.status == "queued"
    assert job.progress == 0
    assert job.id.startswith("job_")

"""End-to-end API tests against the FastAPI app via TestClient.

Runs in preview mode (ENHANCEAI_USE_REAL_OMNIVOICE=false, set in conftest), so
speech uses the tone/SAPI preview and no heavy model is loaded.
"""

import base64
import json


def test_root_and_health(client):
    assert client.get("/").json()["status"] == "ready"
    health = client.get("/api/v1/health").json()
    assert health["status"] == "local-ready"
    assert any(s["name"] == "fastapi" for s in health["services"])


def test_logs_endpoint(client):
    response = client.post("/api/v1/logs", json={"module": "ui", "action": "click", "message": "hi"})
    assert response.status_code == 202
    assert response.json()["ok"] is True


def test_job_lifecycle(client):
    created = client.post(
        "/api/v1/jobs",
        json={"module": "image-enhancement", "media_type": "image", "input_uri": "local://x.png"},
    )
    assert created.status_code == 201
    job_id = created.json()["id"]

    assert client.get(f"/api/v1/jobs/{job_id}").status_code == 200
    assert client.get("/api/v1/jobs/does-not-exist").status_code == 404

    ran = client.post(f"/api/v1/jobs/{job_id}/run")
    assert ran.status_code == 200
    assert ran.json()["status"] == "completed"


def test_voice_engines_and_library_and_styles(client):
    engines = client.get("/api/v1/voice/engines").json()
    assert engines["active_asr"] == "whisper-local"
    library = client.get("/api/v1/voice/library").json()
    assert len(library["voices"]) >= 5
    styles = client.get("/api/v1/voice/styles").json()
    assert any(e["id"] == "neutral" for e in styles["emotions"])


def test_voice_profile_requires_consent(client):
    rejected = client.post(
        "/api/v1/voice/profiles",
        json={"name": "Test", "reference_uri": "local://r.wav", "consent_verified": False},
    )
    assert rejected.status_code == 422

    accepted = client.post(
        "/api/v1/voice/profiles",
        json={"name": "Consented", "reference_uri": "local://r.wav", "consent_verified": True},
    )
    assert accepted.status_code == 201
    assert client.get("/api/v1/voice/profiles").status_code == 200


def test_voice_design(client):
    response = client.post(
        "/api/v1/voice/design",
        json={"name": "Designed", "gender": "female", "age": "young", "accent": "british", "pitch": 1.4},
    )
    assert response.status_code == 201
    assert "female" in response.json()["design_instruct"]


def test_voice_sample_rejects_non_audio(client):
    files = {"sample": ("note.txt", b"hello", "text/plain")}
    assert client.post("/api/v1/voice/samples", files=files).status_code == 422


def test_speech_preview_completes(client):
    response = client.post("/api/v1/voice/speech", json={"text": "hello world", "response_format": "wav"})
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["output_uri"].startswith("local://")


def test_speech_rejects_non_wav(client):
    response = client.post("/api/v1/voice/speech", json={"text": "hi", "response_format": "mp3"})
    assert response.status_code == 422


def test_speech_unknown_voice_404(client):
    response = client.post("/api/v1/voice/speech", json={"text": "hi", "voice_id": "ghost-voice-xyz"})
    assert response.status_code == 404


def test_preview_modules_are_flagged(client):
    response = client.post("/api/v1/voice/transcriptions", json={"audio_uri": "local://a.wav"})
    assert response.status_code == 201
    assert response.json()["result"]["preview"] is True


def test_image_enhance_endpoint(client, png_bytes):
    files = {"image": ("in.png", png_bytes, "image/png")}
    response = client.post("/api/v1/images/enhance", files=files, data={"output_format": "jpg"})
    assert response.status_code == 200
    assert response.headers["X-Module"] == "image-enhancement"
    analysis = json.loads(base64.b64decode(response.headers["X-Image-Analysis"]))
    assert "summary" in analysis


def test_image_enhance_rejects_non_image(client):
    files = {"image": ("x.txt", b"nope", "text/plain")}
    assert client.post("/api/v1/images/enhance", files=files).status_code == 422


def test_background_removal(client, png_bytes):
    # If the u2netp model is present it should segment (200); if not, the endpoint
    # must report it clearly (503). Both are correct depending on the environment.
    from app.config import get_settings
    from app.image_processing import segmentation_model_available

    files = {"image": ("in.png", png_bytes, "image/png")}
    response = client.post("/api/v1/images/background-removal", files=files)
    if segmentation_model_available(get_settings().segmentation_model_path):
        assert response.status_code == 200
        assert response.headers["X-Module"] == "background-removal"
    else:
        assert response.status_code == 503


def test_face_restoration_model_missing(client, png_bytes):
    files = {"image": ("in.png", png_bytes, "image/png")}
    response = client.post("/api/v1/images/face-restoration", files=files)
    assert response.status_code == 503


def test_audio_enhance_endpoint(client, wav_bytes):
    files = {"audio": ("in.wav", wav_bytes, "audio/wav")}
    response = client.post("/api/v1/audio/enhance", files=files, data={"output_format": "wav"})
    assert response.status_code == 200
    assert response.headers["X-Module"] == "audio-enhancement"


def test_capabilities_endpoints(client):
    assert client.get("/api/v1/audio/capabilities").json()["enhancement"]["available"] is True
    assert "background_removal" in client.get("/api/v1/images/capabilities").json()
    assert client.get("/api/v1/shorts/capabilities").json()["output"]["aspect"] == "9:16"


def test_files_endpoint_blocks_logs(client):
    # The logs live under data/logs, which is not a servable root.
    assert client.get("/api/v1/files/logs/error.log").status_code == 404
    assert client.get("/api/v1/files/voice_profiles/index.json").status_code == 404


def test_shorts_analyze_rejects_bad_extension(client):
    files = {"media": ("notes.txt", b"data", "text/plain")}
    assert client.post("/api/v1/shorts/analyze", files=files).status_code == 422


def test_shorts_render_invalid_project_id(client):
    response = client.post(
        "/api/v1/shorts/not-a-valid-id/render",
        json={"clips": [{"start_sec": 0.0, "end_sec": 5.0}], "captions": True},
    )
    assert response.status_code == 422


def test_openai_voices_and_speech(client):
    assert client.get("/v1/audio/voices").json()["object"] == "list"
    response = client.post("/v1/audio/speech", json={"input": "hello", "voice": "aria", "response_format": "wav"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("audio/wav")
    assert response.content[:4] == b"RIFF"  # a real WAV, not text bytes

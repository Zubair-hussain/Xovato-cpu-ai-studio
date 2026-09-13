# `app/` - the FastAPI backend

This package is the whole backend. It is CPU-first and dependency-light (NumPy,
Pillow, onnxruntime, soundfile, faster-whisper) - no GPU required.

## Entry point

`main.py` builds the FastAPI `app` and defines every route. Start it with:

```bash
uvicorn app.main:app --reload --port 8000   # docs at /docs
```

## Module map

| Module | Responsibility |
| --- | --- |
| `main.py` | FastAPI app, all HTTP routes, request validation, wiring. |
| `config.py` | `Settings` (env-driven, `ENHANCEAI_*`), cached via `get_settings()`. |
| `models.py` | Pydantic request/response models (`Job`, `SpeechRequest`, ...). |
| `image_processing.py` | Enhancement, background removal (u2netp), encoding, guided filter. |
| `face_restoration.py` | YuNet detection + GPEN-BFR-256 restoration + align/blend. |
| `audio_processing.py` | Spectral denoise, BS.1770 loudness, true-peak limiter (pure NumPy). |
| `voice_engine.py` | Warm, in-process OmniVoice engine + clone-prompt cache. |
| `voice_styles.py` | Emotion presets, instruct vocabulary, prosody shaping, design mapping. |
| `voice_library.py` | The 12 built-in stock voices. |
| `omnivoice_runtime.py` | Model-presence check for the OmniVoice checkpoint. |
| `shorts_media.py` | ffmpeg/ffprobe helpers: probe, extract audio, crop 9:16, captions. |
| `shorts_pipeline.py` | Transcription (whisper/Deepgram) + moment ranking (heuristic/LLM). |
| `shorts_service.py` | Background orchestration for shorts analysis and rendering. |
| `supabase_output.py` | Optional best-effort upload of outputs to Supabase. |
| `logging_config.py` | Structured JSON logging. |
| `model_manager.py` | Reports which model files are present. |
| `services/` | `JobStore`, `VoiceStore`, circuit breakers, health monitor. See [services/README](services/README.md). |
| `utils/` | Small file/cleanup helpers. See [utils/README](utils/README.md). |

## Conventions

- Every endpoint validates input and returns a clear 4xx/5xx on bad requests.
- Heavy work (shorts) runs in background tasks under a concurrency semaphore.
- Safety defaults (consent-gated cloning, upload limits, `/files` allowlist,
  circuit breakers) are load-bearing - do not remove them.
- Model-dependent features return `503` with a clear message when the model file
  is absent, and never trip a circuit breaker for a missing file.

See the top-level [README](../README.md) and [docs/](../docs/README.md) for
features, samples, benchmarks and the GPU path.

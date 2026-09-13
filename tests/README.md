# `tests/` - test suite

111 tests, ~70% line coverage. Runs on CPU with no models, no GPU, no network.

```bash
pip install -r requirements-dev.txt
pytest                     # config + coverage come from ../pytest.ini
# HTML coverage report -> ../htmlcov/index.html
```

## Layout

| File | Covers |
| --- | --- |
| `conftest.py` | Redirects data/log dirs to a temp folder before import; shared `client`, `png_bytes`, `wav_bytes` fixtures. |
| `test_api.py` | Every HTTP endpoint via `TestClient` (health, jobs, voice, images, audio, shorts, OpenAI-compat, `/files` allowlist). |
| `test_config.py` | Settings defaults, CORS parsing, limits. |
| `test_models.py` | Pydantic validation (consent default, clip windows, render caps). |
| `test_services.py` | `JobStore` eviction, `CircuitBreaker`, `VoiceStore`, health. |
| `test_audio_processing.py` | LUFS, denoise, limiter, running-min/box-filter equivalence. |
| `test_image_processing.py` | Enhance, encode, target-size, guided filter, missing-model paths. |
| `test_voice_styles.py` | Instruct validation, style resolution, design mapping, prosody. |
| `test_voice_library.py` | Stock-voice lookup and payload shape. |
| `test_voice_engine.py` | Guard paths that need no torch (empty text, missing transcript). |
| `test_shorts_pipeline.py` | Heuristic ranking, overlap suppression, LLM JSON parsing, fallback. |
| `test_shorts_media.py` | Caption phrase grouping, drawtext escaping. |
| `test_shorts_service.py` | Upload streaming + size limit, job creation, project dir. |
| `test_supabase_output.py` | Config gating (service-role key required). |
| `test_utils.py` | File helpers and retention prune. |
| `test_deployment_package.py` | Required files present, CPU defaults, model manager. |

## What is not covered, and why

The heavy runtime paths - torch/OmniVoice synthesis, ffmpeg encode/probe, ONNX
model inference, Supabase network calls - are exercised through their guard
clauses but not with the real runtimes, since those need a GPU, ffmpeg binaries,
model files, or a live Supabase. That is the gap between 70% and 100%. New pure
logic should ship with tests; see [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md).

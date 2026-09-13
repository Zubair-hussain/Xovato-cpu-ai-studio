# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased] - Hardening & test pass

A single pass that resolved 38 tracked issues (see `ISSUES.md`), added a real
test suite, and produced measured benchmarks. Grouped by theme:

### Fixed - critical
- **Backend now starts.** `app/services.py` was shadowed by the `app/services/`
  package, so `from app.services import job_store` raised `ImportError` on every
  machine. The real implementation moved to `app/services/core.py` and is
  re-exported from the package `__init__`.
- Declared the Python 3.11+ requirement (`.python-version`, README, requirements).

### Fixed - security & stability
- Upload size limits are now enforced **while streaming**, before the file can
  fill memory or disk (images, audio, voice samples, shorts video).
- The `/files` endpoint serves only `outputs/` and `shorts/`; logs and the voice
  profile index are no longer reachable.
- Voice-clone `consent_verified` defaults to `false` and the API rejects clones
  without explicit consent.
- CORS origins are configurable via `ENHANCEAI_CORS_ORIGINS`.
- The in-memory job store evicts oldest finished jobs past a cap.
- Shorts analysis/render run under a concurrency semaphore.
- Optional retention prune of old generated files on startup.
- `project_id` is validated before any filesystem access in the render endpoint.

### Fixed - correctness
- The audio limiter no longer reads past the end of its buffer (van Herk running
  minimum); output is numerically identical to before.
- Missing segmentation model no longer trips the circuit breaker.
- The empty `/jobs/{id}/run` no longer resets the segmentation breaker.
- Shorts `max_len` is derived from the clamped `min_len`.
- Voice profiles with the same name no longer overwrite each other's bundle.
- Shorts clips validate `end_sec > start_sec`; render caps clip count.
- The LLM ranking provider is passed explicitly instead of leaking through
  `os.environ`.
- Preview TTS only launches PowerShell on Windows.

### Fixed - honesty
- `/v1/audio/speech` returns real WAV audio; `/v1/audio/transcriptions` runs
  faster-whisper. Transcription/dubbing/audiobook/story jobs are flagged
  `preview` instead of reporting a completed file that does not exist.
- Voice design maps gender/age/accent/pitch onto real instruct attributes.

### Performance
- Vectorised `_smooth`, `_istft` overlap-add and `measure_lufs` block powers.
- `render_clip` reuses one ffprobe across all clips.
- Captions are grouped into phrases (far fewer drawtext filters).
- Supabase output sync moved off the request path.
- `/health` CPU reading is non-blocking; `psutil` added to requirements.

### Added
- Test suite: 111 tests, 70% line coverage (`pytest-cov`).
- Benchmarks with measured CPU numbers and reproducible scripts
  (`scripts/benchmark.py`, `scripts/generate_charts.py`, `docs/benchmarks.md`).
- Industry scaffolding: CI workflow, `Makefile`, `CONTRIBUTING.md`,
  `requirements-dev.txt`, richer JSON logging.

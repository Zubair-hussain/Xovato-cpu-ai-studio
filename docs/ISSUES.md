# Xovato CPU AI Studio: Issue List

> ✅ **All 38 issues below are resolved in the current codebase.** See
> [CHANGELOG.md](./CHANGELOG.md) for the grouped summary and
> [benchmarks.md](./benchmarks.md) for measured results. Verified by a
> 111-test suite (`pytest`, ~70% line coverage).

Found by reading every backend file and running the app on Python 3.12.

| Severity | Count |
| --- | --- |
| 🔴 Critical | 3 |
| 🟠 Security & Stability | 10 |
| 🟡 Logic Bugs | 8 |
| 🟡 Fake or Placeholder Features | 4 |
| 🟢 Performance | 7 |
| ⚪ Code Quality & Docs | 6 |
| **Total** | **38** |

---

## 🔴 Critical

### 1. Backend fails to start: `app/services.py` is shadowed by the `app/services/` package
**Labels:** `bug` `critical`

**Description:**
The repo has both a module, `app/services.py`, and a package folder with the same name, `app/services/`. Python loads the package first. That package's `__init__.py` defines nothing, so every `from app.services import job_store` fails:

```
ImportError: cannot import name 'job_store' from 'app.services' (app/services/__init__.py)
```

The error is raised from `app/main.py:55`, `app/shorts_service.py:23` and `app/services/voice_cloning.py`. As a result `uvicorn app.main:app` crashes on startup and **no API endpoint works**.

**Suggested fix:** Move `services.py` into the package (for example `app/services/core.py`) and re-export `job_store`, `voice_store`, `omnivoice_breaker`, `segmentation_breaker` and `system_health` from `app/services/__init__.py`.

---

### 2. Python 3.11+ is required but not declared anywhere
**Labels:** `bug` `critical` `docs`

**Description:**
The code uses `datetime.UTC` and `enum.StrEnum`, which only exist in Python 3.11 and later (`app/models.py`, `app/services.py`, `app/logging_config.py`). On Python 3.10 the app crashes on import. The README, `requirements.txt` and the start command never mention a minimum Python version.

**Suggested fix:** Add a `.python-version` file or `python_requires`, and state "Python 3.11+" in the README.

---

### 3. Tests pass while the backend is broken
**Labels:** `bug` `testing`

**Description:**
`tests/test_deployment_package.py` only checks that files exist and imports `app.config` and `app.model_manager`. It never imports `app.main`, so the suite passes (3/3) even though the server can't start (issue #1).

**Suggested fix:** Add a smoke test that imports `app.main` and calls `GET /api/v1/health` with FastAPI's `TestClient`.

---

## 🟠 Security & Stability

### 4. Shorts upload writes the whole file to disk before checking the size limit
**Labels:** `security` `bug`

**Description:**
In `POST /api/v1/shorts/analyze` (`main.py:912-920`), `save_upload()` copies the entire upload to disk first and only then compares it with `max_shorts_upload_bytes` (2 GB). An attacker can stream a very large file and fill the server's disk before the check runs.

**Suggested fix:** Copy in chunks and abort once the byte count passes the limit.

---

### 5. Image and audio uploads are read fully into memory before the size check
**Labels:** `security` `performance`

**Description:**
`_read_upload()` (`main.py:590`) and `enhance_audio_endpoint` (`main.py:827`) call `file.read()` with no limit, then compare the length with the max. A large upload is loaded into RAM in full, so a few concurrent requests can exhaust memory.

**Suggested fix:** Use `file.read(limit + 1)` and reject the upload if more bytes than the limit come back.

---

### 6. Voice sample upload has no size limit
**Labels:** `security`

**Description:**
`POST /api/v1/voice/samples` (`main.py:277`) streams the file straight to disk through `VoiceStore.save_sample()` with no maximum size. Anyone can upload files of unlimited size.

**Suggested fix:** Apply `max_audio_upload_bytes`, or a dedicated limit for voice samples.

---

### 7. `/api/v1/files/{path}` exposes logs and private voice data
**Labels:** `security`

**Description:**
`read_local_file` (`main.py:1063`) serves any file under `data/`. Path traversal is blocked, but the endpoint still exposes:

- `logs/error.log` and `logs/health.log`
- `voice_profiles/index.json`, which lists every user's voice profile
- raw uploaded voice samples

**Suggested fix:** Serve only from an allowlist of folders (for example `outputs/` and `shorts/`).

---

### 8. Voice cloning consent defaults to `True`
**Labels:** `security` `legal`

**Description:**
`VoiceProfileCreate.consent_verified` defaults to `True` (`models.py:63`). A client that leaves the field out creates a voice clone that is recorded as consented. For a public voice-cloning app, this is a legal and abuse risk.

**Suggested fix:** Default to `False` and reject clone creation unless consent is explicitly `true`.

---

### 9. CORS origins are hardcoded to localhost
**Labels:** `bug` `deployment`

**Description:**
`allow_origins` is fixed to `http://localhost:3000` and `http://127.0.0.1:3000` (`main.py:83`). Once the frontend is deployed to a real domain, every browser request will be blocked by CORS.

**Suggested fix:** Read the origins from an environment variable such as `ENHANCEAI_CORS_ORIGINS`.

---

### 10. The in-memory job store and transcript cache grow without limit
**Labels:** `performance` `stability`

**Description:**
Every API call creates a `Job` in `JobStore._jobs` (`services.py`), and every shorts analysis stores a full transcript in `_transcripts` (`shorts_service.py`). Nothing is ever evicted, so under steady traffic memory keeps growing until the process is killed.

**Suggested fix:** Cap the store size (for example 1,000 jobs, evicting the oldest finished ones) or add a TTL.

---

### 11. No limit on concurrent shorts jobs
**Labels:** `performance` `stability`

**Description:**
Shorts analysis and rendering run through `BackgroundTasks` (`main.py:923`, `main.py:967`) with no concurrency cap. Each job runs ffmpeg and Whisper, which use a lot of CPU. A handful of simultaneous users can saturate the CPU and stall every other endpoint.

**Suggested fix:** Put a semaphore around the jobs (for example `ENHANCEAI_MAX_SHORTS_JOBS=2`) and show extra jobs as "queued".

---

### 12. Generated files are never cleaned up
**Labels:** `stability`

**Description:**
Outputs, voice samples and shorts projects (source videos and rendered clips) pile up in `data/` forever. `app/utils/cleanup.py:remove_expired_files` exists but is never called. The disk will eventually fill up.

**Suggested fix:** Add a retention setting and run the cleanup on startup or on a schedule.

---

### 13. `project_id` in the render endpoint isn't validated
**Labels:** `security`

**Description:**
`POST /api/v1/shorts/{project_id}/render` (`main.py:949`) builds a filesystem path straight from the URL parameter. Values like `..` make the server look for `source.*` files outside the shorts folder.

**Suggested fix:** Validate the ID against `^proj_[0-9a-f]{12}$`.

---

## 🟡 Logic Bugs

### 14. Fake job runs reset the segmentation circuit breaker
**Labels:** `bug`

**Description:**
`POST /api/v1/jobs/{id}/run` (`main.py:235`) calls `segmentation_breaker.record_success()` for background-removal jobs without doing any real work. Anyone can create a fake job and run it to reset the breaker. That defeats its purpose of pausing a failing model.

**Suggested fix:** Remove the `record_success()` call from this endpoint.

---

### 15. A missing model file trips the circuit breaker
**Labels:** `bug`

**Description:**
If `u2netp.onnx` is missing, `remove_background_endpoint` (`main.py:686`) records a breaker failure. After 2 requests the breaker opens, and background removal stays blocked for 15 minutes **even after the model file is added**. A missing file is a configuration problem, not a runtime failure.

**Suggested fix:** Check whether the model exists before checking the breaker, and don't record that case as a failure.

---

### 16. The first request's LLM ranking provider sticks for the whole process
**Labels:** `bug`

**Description:**
`rank_moments` (`shorts_pipeline.py:496`) calls `os.environ.setdefault("SHORTS_LLM_PROVIDER", chosen)`. The first request sets the provider for the whole process. Later requests that ask for a different provider (for example `anthropic` after `openai`) silently use the first one.

**Suggested fix:** Pass the provider into `rank_with_llm()` as an argument instead of changing global environment variables.

---

### 17. `max_len` can end up smaller than `min_len` in shorts analysis
**Labels:** `bug`

**Description:**
In `main.py:930-931`, `min_len` is clamped to at least 3.0, but `max_len` is computed from the **unclamped** `min_len`. With `min_len=1, max_len=2` the result is `min_len=3, max_len=2`, so no moments can ever be found.

**Suggested fix:** Clamp `min_len` first, then compute `max_len` from the clamped value.

---

### 18. Shorts clip requests aren't validated
**Labels:** `bug`

**Description:**
`ShortsClip` (`models.py:154`) doesn't check that `end_sec > start_sec` or that `start_sec >= 0`, and `ShortsRenderRequest` has no maximum clip count. Bad values reach ffmpeg and fail late, and one request can queue hundreds of renders.

**Suggested fix:** Add Pydantic validators and a maximum clip count (for example 20).

---

### 19. Voice profiles with the same name overwrite each other
**Labels:** `bug` `data-loss`

**Description:**
`VoiceStore._bundle_path()` (`services.py:337`) names the bundle file after the profile name only (`my-voice.ovsvoice`). Creating a second profile with the same name overwrites the first profile's bundle file.

**Suggested fix:** Include the profile ID in the filename.

---

### 20. The audio limiter reads past the end of its array
**Labels:** `bug` `memory-safety`

**Description:**
In `_limit()` (`audio_processing.py:301`), the `as_strided` view has `len(required) + window` rows, but only `len(required) + 1` rows fit inside the padded buffer. The extra rows read memory past the end of the array. The result is sliced off afterwards, but reading it can still return garbage or crash the process.

**Suggested fix:** Replace it with a correct running-minimum filter that runs in linear time (for example the van Herk / Gil-Werman algorithm).

---

### 21. Preview TTS launches `powershell.exe` on every platform
**Labels:** `bug` `performance`

**Description:**
`VoiceStore._write_windows_tts_wav()` (`services.py:261`) always tries to start `powershell.exe`, even on Linux hosts. There it fails and falls back to a tone, but it still costs a failed process launch on every speech request.

**Suggested fix:** Only try it when `os.name == "nt"`.

---

## 🟡 Fake or Placeholder Features

### 22. `/v1/audio/speech` returns text instead of audio
**Labels:** `bug` `api`

**Description:**
The OpenAI-compatible speech endpoint (`main.py:1077`) returns the bytes of `"Simulated wav audio for voice ..."` with `Content-Type: audio/wav`. Any OpenAI SDK or client that calls it gets a corrupt audio file.

**Suggested fix:** Route it through the real synthesis path, or return `501 Not Implemented`.

---

### 23. `/v1/audio/transcriptions` returns a hardcoded fake transcript
**Labels:** `bug` `api`

**Description:**
`main.py:1084` doesn't accept an audio file and always returns `"Simulated local transcription from the active ASR engine."`. faster-whisper is already installed and used by the shorts pipeline, but this endpoint doesn't use it.

**Suggested fix:** Accept an uploaded file and call `transcribe_audio()`, or return `501`.

---

### 24. Transcription, dubbing, audiobook and story jobs report success without doing any work
**Labels:** `bug` `api`

**Description:**
`/voice/transcriptions`, `/voice/dubbing`, `/voice/audiobooks` and `/voice/stories` (`main.py:483-538`) immediately mark jobs as `completed`, with `output_uri` values that point to files that don't exist. The frontend and users are told the work succeeded when nothing was done.

**Suggested fix:** Return `501`, or mark these jobs clearly as `simulated` / `preview`.

---

### 25. Voice design only saves tags
**Labels:** `enhancement`

**Description:**
`POST /voice/design` (`services.py:219`) stores gender, age, accent and similar values as tags. `pitch` and `speed` are ignored, and none of these values are ever mapped to OmniVoice `instruct` items when synthesising. A "designed" voice sounds the same as the default voice.

**Suggested fix:** Map the design fields to valid instruct items (see `voice_styles.VALID_INSTRUCT_ITEMS`) and use them during synthesis.

---

## 🟢 Performance

### 26. The audio limiter does work proportional to samples × window size
**Labels:** `performance`

**Description:**
`_limit()` takes the minimum over a 480-sample window for every sample. For 20 minutes of 48 kHz stereo that is tens of billions of operations, which makes audio enhancement very slow whenever the limiter kicks in (and it usually does after loudness normalisation).

**Suggested fix:** Use a linear-time running-minimum algorithm.

---

### 27. `_smooth()` uses `np.apply_along_axis`, which loops in Python
**Labels:** `performance`

**Description:**
The denoise gain mask is smoothed with `np.apply_along_axis` (`audio_processing.py:230`). That makes one `np.convolve` call per frame (about 100k calls for long audio) plus one per frequency bin.

**Suggested fix:** Use a vectorised cumulative-sum box filter along each axis.

---

### 28. `_istft()` and `measure_lufs()` loop in Python over every frame
**Labels:** `performance`

**Description:**
Overlap-add in `_istft` and block power in `measure_lufs` both iterate frame by frame in Python. Enhancement measures LUFS three times, so the cost multiplies on long recordings.

**Suggested fix:** Vectorise the overlap-add (split into non-overlapping hop groups), and compute block powers from cumulative sums.

---

### 29. `render_clip()` re-runs ffprobe for every clip
**Labels:** `performance`

**Description:**
`shorts_media.render_clip` (`shorts_media.py:207`) calls `probe_media(source)` on every call. Rendering 10 clips probes the same source file 10 times, even though the analysis step already probed it.

**Suggested fix:** Probe once in `run_render` and pass the result into `render_clip`.

---

### 30. Supabase sync blocks the speech response
**Labels:** `performance`

**Description:**
`record_completed_output()` runs inline in `synthesize_speech` (`main.py:358`, `main.py:419`). It performs a storage upload (60 s timeout) and a REST insert (20 s timeout), so a slow Supabase can hold each TTS response for up to about 80 seconds.

**Suggested fix:** Run it through `BackgroundTasks`.

---

### 31. The health check blocks for 100 ms and never gets a CPU reading
**Labels:** `performance` `bug`

**Description:**
`SystemHealthMonitor._cpu_percent()` calls `psutil.cpu_percent(interval=0.1)`, which blocks every health or diagnostics request for 100 ms. `psutil` also isn't in `requirements.txt`, so in a normal install the health state is always `"unknown"` with an "Install psutil" warning.

**Suggested fix:** Add `psutil` to requirements and use `interval=None`, which doesn't block.

---

### 32. Captions use one ffmpeg `drawtext` filter per word
**Labels:** `performance`

**Description:**
`build_caption_filter()` (`shorts_media.py:169`) emits a separate `drawtext` filter for every word. A 60-second clip gets about 150 chained filters. That slows encoding, and on Windows the command can pass the 32k-character command-line limit.

**Suggested fix:** Group words into short phrases, or write captions to an `.ass` subtitle file and use the `subtitles` filter.

---

## ⚪ Code Quality & Docs

### 33. The JSON log formatter drops most structured fields
**Labels:** `observability`

**Description:**
`JsonFormatter` (`logging_config.py`) only outputs a fixed list of 8 `extra` keys. Fields logged across the app, such as `output_bytes`, `generate_seconds`, `realtime_factor`, `cloned`, `candidates`, `faces_detected` and `path`, are silently dropped. That makes production debugging and performance tracking much harder.

**Suggested fix:** Include every non-standard `LogRecord` attribute in the output.

---

### 34. Dead code and unused settings
**Labels:** `cleanup`

**Description:**

- `omnivoice_runtime.synthesize_cpu()` is the old subprocess-per-request synthesis path. It has been replaced by `voice_engine` and is never called.
- The `omnivoice_breaker_heavy_recovery_seconds` setting is never used.
- The `app/services/*` facade modules are not imported anywhere.

---

### 35. README and docs point to paths that don't exist
**Labels:** `docs`

**Description:**

- The test command is `..\backend\.venv\Scripts\python -m pytest tests`, but `../backend` isn't in this repo.
- The docs keep referring to a `deployment/` folder and `../xovato-app`.
- `models/copy-models.ps1` and `MODEL_MANIFEST.json` copy from `../backend/data/models`, which doesn't exist.

New contributors can't follow the setup steps as written.

---

### 36. Supabase config reads outside the repo and falls back to the anon key
**Labels:** `security` `config`

**Description:**
`supabase_output._local_env_values()` reads `../xovato-app/.env.local` at import time. `_supabase_config()` falls back to `SUPABASE_ANON_KEY` or `NEXT_PUBLIC_SUPABASE_ANON_KEY` for **server-side writes and uploads**. That either fails silently under RLS or requires permissive table and bucket policies for anonymous users.

**Suggested fix:** Require `SUPABASE_SERVICE_ROLE_KEY` for server writes, and only read the backend's own `.env`.

---

### 37. `requirements.txt` mixes production and test dependencies
**Labels:** `deployment`

**Description:**
`pytest` and `httpx` (the latter only needed in production for shorts) sit in the production requirements, while `psutil` is missing. There is no dev requirements file, so every production image carries test tooling.

**Suggested fix:** Split into `requirements.txt` and `requirements-dev.txt` (with `pytest`, `pytest-cov`).

---

### 38. Test coverage is close to zero
**Labels:** `testing`

**Description:**
The only 3 tests check that files exist. There are no tests for:

- API endpoints (health, jobs, voice, images, audio, shorts)
- image enhancement, background removal, face restoration
- audio DSP (LUFS, denoise, limiter)
- voice styles and the stock voice library
- shorts ranking, LLM response parsing, caption filters
- the circuit breaker and the job store

Every bug listed above could merge unnoticed.

**Suggested fix:** Add unit and API tests with `pytest-cov`, and require a minimum coverage level in CI.

---

## Recommended Fix Order

1. **#1, #2**: nothing runs without these.
2. **#4 to #11**: must be fixed before public launch.
3. **#14 to #24**: correctness and honest API behaviour.
4. **#26 to #32**: performance under load.
5. **#33 to #38**: cleanup, docs, tests.

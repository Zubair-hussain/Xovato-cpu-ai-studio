# `app/services/` - service layer

Shared, process-wide state and the deployment facade.

## Real implementation

`core.py` holds the actual classes and singletons:

| Object | Purpose |
| --- | --- |
| `JobStore` / `job_store` | Thread-safe in-memory job registry with progress, results, and eviction of the oldest finished jobs past a cap. |
| `VoiceStore` / `voice_store` | Voice profiles, samples, generated outputs; local URI resolution with traversal protection. |
| `CircuitBreaker` / `omnivoice_breaker`, `segmentation_breaker` | Pause a feature after repeated real failures, with timed recovery. |
| `SystemHealthMonitor` / `system_health` | Non-blocking CPU health snapshot (via psutil when installed). |

`__init__.py` re-exports all of these, so `from app.services import job_store`
works everywhere.

> **History:** the project once shipped both `app/services.py` and this package,
> which shadowed each other and broke every import. The module now lives at
> `core.py` and is re-exported here - see [docs/CHANGELOG.md](../../docs/CHANGELOG.md).

## Facade modules

`audio_enhancement.py`, `background_removal.py`, `face_enhancement.py`,
`image_enhancement.py`, `transcription.py`, `video_processing.py`,
`voice_cloning.py` are thin re-exports that match the deployment packaging layout
described in [docs/app-package.md](../../docs/app-package.md). They import from the
real modules; nothing depends on them internally.

The job store is **in-memory**: jobs do not survive a restart or span replicas.
Moving it to Postgres/Redis is the prerequisite for running more than one worker.

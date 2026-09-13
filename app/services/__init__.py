"""Service layer for the Xovato CPU backend.

Historically the project shipped both a module ``app/services.py`` (the real
``JobStore`` / ``VoiceStore`` / circuit breakers / health monitor) *and* an
``app/services/`` facade package. Python resolves a package before a module of
the same name, so ``from app.services import job_store`` picked the empty
package and raised ``ImportError`` on every machine.

The real implementation now lives in :mod:`app.services.core`, and this package
re-exports it so ``from app.services import job_store`` keeps working. The thin
per-feature facade modules (``image_enhancement``, ``audio_enhancement`` …) stay
for the deployment packaging layout described in the docs.
"""

from app.services.core import (
    CircuitBreaker,
    JobStore,
    SystemHealthMonitor,
    VoiceStore,
    job_store,
    omnivoice_breaker,
    segmentation_breaker,
    system_health,
    voice_store,
)

__all__ = [
    "CircuitBreaker",
    "JobStore",
    "SystemHealthMonitor",
    "VoiceStore",
    "job_store",
    "omnivoice_breaker",
    "segmentation_breaker",
    "system_health",
    "voice_store",
]

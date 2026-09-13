"""Shared test fixtures.

The data directory and log directory are redirected to a per-run temp folder
*before* any ``app`` module is imported, so tests never write into the repo's
``data/`` tree and every run starts from a clean state.
"""

from __future__ import annotations

import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Must happen before app.config is imported anywhere.
_TMP_ROOT = Path(tempfile.mkdtemp(prefix="xovato-tests-"))
os.environ.setdefault("ENHANCEAI_DATA_DIR", str(_TMP_ROOT / "data"))
os.environ.setdefault("ENHANCEAI_LOG_DIR", str(_TMP_ROOT / "data" / "logs"))
os.environ.setdefault("ENHANCEAI_USE_REAL_OMNIVOICE", "false")
os.environ.setdefault("ENHANCEAI_ENVIRONMENT", "test")


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG for image-endpoint tests."""
    from PIL import Image

    rng = np.random.default_rng(0)
    image = Image.fromarray((rng.random((64, 96, 3)) * 255).astype("uint8"))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def wav_bytes() -> bytes:
    """One second of stereo tone-plus-noise as a WAV."""
    import soundfile as sf

    sr = 24000
    t = np.arange(sr) / sr
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    stereo = np.stack([tone, tone * 0.8], axis=1).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, stereo, sr, format="WAV", subtype="PCM_16")
    return buffer.getvalue()

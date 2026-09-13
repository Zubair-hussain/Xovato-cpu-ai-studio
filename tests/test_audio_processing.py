import io

import numpy as np
import pytest
import soundfile as sf

from app import audio_processing as ap


def _wav(seconds=1.0, sr=24000, channels=2, amp=0.3):
    t = np.arange(int(seconds * sr)) / sr
    tone = amp * np.sin(2 * np.pi * 220 * t)
    data = np.stack([tone] * channels, axis=1).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, data, sr, format="WAV", subtype="FLOAT")
    return buffer.getvalue()


def test_running_min_matches_reference():
    rng = np.random.default_rng(3)
    values = rng.random(500)
    radius = 7
    fast = ap._running_min(values, radius)
    # Reference: explicit centred min with edge padding.
    padded = np.pad(values, (radius, radius), mode="edge")
    reference = np.array([padded[i : i + 2 * radius + 1].min() for i in range(len(values))])
    assert np.allclose(fast, reference)


def test_running_min_radius_zero_is_identity():
    values = np.array([3.0, 1.0, 2.0])
    assert np.allclose(ap._running_min(values, 0), values)


def test_box_filter_matches_convolve():
    rng = np.random.default_rng(4)
    matrix = rng.random((40, 12)).astype(np.float32)
    fast = ap._box_filter_axis(matrix, 5, axis=0)
    kernel = np.ones(5, dtype=np.float32) / 5
    reference = np.apply_along_axis(lambda c: np.convolve(c, kernel, mode="same"), 0, matrix)
    assert np.allclose(fast, reference, atol=1e-5)


def test_measure_lufs_reasonable():
    sr = 24000
    samples = (0.2 * np.sin(2 * np.pi * 300 * np.arange(sr * 2) / sr))[:, None]
    lufs = ap.measure_lufs(samples.astype(np.float32), sr)
    assert -40.0 < lufs < 0.0


def test_enhance_audio_normalises_to_target():
    payload, fmt, analysis = ap.enhance_audio(_wav(), target_lufs=-16.0, output_format="wav")
    assert fmt == "wav"
    assert abs(analysis.output_lufs - (-16.0)) < 2.0
    decoded, _ = sf.read(io.BytesIO(payload))
    assert np.max(np.abs(decoded)) <= 1.0


def test_enhance_audio_limiter_no_clipping():
    payload, _, analysis = ap.enhance_audio(_wav(amp=0.9), target_lufs=-6.0, peak_ceiling_dbfs=-1.0)
    assert analysis.output_peak_dbfs <= -0.5


def test_enhance_audio_rejects_empty():
    buffer = io.BytesIO()
    sf.write(buffer, np.zeros((0, 1), dtype=np.float32), 24000, format="WAV")
    with pytest.raises(ap.AudioProcessingError):
        ap.enhance_audio(buffer.getvalue())


def test_load_audio_rejects_garbage():
    with pytest.raises(ap.AudioProcessingError):
        ap.load_audio(b"not audio at all")

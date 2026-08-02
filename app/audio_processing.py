"""CPU-only audio enhancement: denoise, de-rumble, normalise loudness, limit.

The chain is classical DSP rather than a model, which is the right trade here:
stationary background noise (hiss, fans, air conditioning, mains hum) is exactly
what spectral gating handles well, and loudness normalisation is a measurement
problem with a published standard, not a learning problem.

Everything is vectorised numpy. SciPy is not installed and per-sample Python
loops would be far too slow, so all filtering is done in the frequency domain
and the limiter works on a vectorised envelope.

Loudness follows ITU-R BS.1770 / EBU R128: K-weighting, 400 ms blocks, absolute
gate at -70 LUFS and a relative gate 10 LU below the ungated mean.
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_MAX_SECONDS = 20 * 60
_EPSILON = 1e-12

# Formats libsndfile can decode here. M4A/AAC is deliberately excluded - it is
# not supported by this libsndfile build and must fail with a clear message.
SUPPORTED_INPUT_FORMATS = ("wav", "mp3", "flac", "ogg", "aiff", "aif", "aifc")
OUTPUT_MIME_TYPES = {"wav": "audio/wav", "flac": "audio/flac", "ogg": "audio/ogg"}


class AudioProcessingError(RuntimeError):
    """Raised when audio cannot be decoded or processed."""


@dataclass
class AudioAnalysis:
    duration_seconds: float
    sample_rate: int
    channels: int
    input_lufs: float
    output_lufs: float
    input_peak_dbfs: float
    output_peak_dbfs: float
    noise_floor_dbfs: float
    estimated_snr_db: float
    noise_reduction_db: float
    summary: str = ""
    actions: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {
            "duration_seconds": round(self.duration_seconds, 2),
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "input_lufs": round(self.input_lufs, 2),
            "output_lufs": round(self.output_lufs, 2),
            "input_peak_dbfs": round(self.input_peak_dbfs, 2),
            "output_peak_dbfs": round(self.output_peak_dbfs, 2),
            "noise_floor_dbfs": round(self.noise_floor_dbfs, 2),
            "estimated_snr_db": round(self.estimated_snr_db, 2),
            "noise_reduction_db": round(self.noise_reduction_db, 2),
            "summary": self.summary,
            "actions": self.actions,
        }


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #


def load_audio(data: bytes) -> tuple[np.ndarray, int]:
    """Decode to float32 samples shaped (frames, channels)."""
    try:
        import soundfile as sf
    except ImportError as error:  # pragma: no cover - dependency guard
        raise AudioProcessingError("soundfile is not installed.") from error

    try:
        samples, sample_rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    except Exception as error:
        raise AudioProcessingError(
            "Could not decode this audio. Supported formats: WAV, MP3, FLAC, OGG, AIFF. "
            "M4A/AAC is not supported - convert it to WAV or MP3 first."
        ) from error

    if samples.size == 0:
        raise AudioProcessingError("The uploaded audio contained no samples.")

    if samples.shape[0] / sample_rate > _MAX_SECONDS:
        raise AudioProcessingError(f"Audio is longer than the {_MAX_SECONDS // 60} minute limit.")

    return np.ascontiguousarray(samples, dtype=np.float32), int(sample_rate)


def encode_audio(samples: np.ndarray, sample_rate: int, output_format: str) -> tuple[bytes, str]:
    import soundfile as sf

    fmt = output_format.lower()
    if fmt not in OUTPUT_MIME_TYPES:
        fmt = "wav"

    buffer = io.BytesIO()
    subtype = "PCM_16" if fmt in {"wav", "flac"} else None
    sf.write(buffer, samples, sample_rate, format=fmt.upper(), subtype=subtype)
    return buffer.getvalue(), fmt


# --------------------------------------------------------------------------- #
# Loudness (ITU-R BS.1770)
# --------------------------------------------------------------------------- #


def _biquad_magnitude(b: tuple[float, float, float], a: tuple[float, float, float], freqs: np.ndarray, sample_rate: int) -> np.ndarray:
    """Magnitude response of a biquad on the given frequency grid."""
    omega = 2.0 * np.pi * freqs / sample_rate
    z1 = np.exp(-1j * omega)
    z2 = z1 * z1
    numerator = b[0] + b[1] * z1 + b[2] * z2
    denominator = a[0] + a[1] * z1 + a[2] * z2
    return np.abs(numerator / (denominator + _EPSILON))


def _k_weighting_response(freqs: np.ndarray, sample_rate: int) -> np.ndarray:
    """BS.1770 K-weighting: a high shelf followed by a high-pass."""
    # Stage 1 - high shelf.
    gain_db = 3.999843853973347
    q = 0.7071752369554196
    fc = 1681.974450955533
    amplitude = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * fc / sample_rate
    alpha = np.sin(w0) / (2.0 * q)
    cos_w0 = np.cos(w0)
    root = np.sqrt(amplitude)
    shelf_b = (
        amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 + 2 * root * alpha),
        -2 * amplitude * ((amplitude - 1) + (amplitude + 1) * cos_w0),
        amplitude * ((amplitude + 1) + (amplitude - 1) * cos_w0 - 2 * root * alpha),
    )
    shelf_a = (
        (amplitude + 1) - (amplitude - 1) * cos_w0 + 2 * root * alpha,
        2 * ((amplitude - 1) - (amplitude + 1) * cos_w0),
        (amplitude + 1) - (amplitude - 1) * cos_w0 - 2 * root * alpha,
    )

    # Stage 2 - high-pass.
    q2 = 0.5003270373238773
    fc2 = 38.13547087602444
    w2 = 2.0 * np.pi * fc2 / sample_rate
    alpha2 = np.sin(w2) / (2.0 * q2)
    cos_w2 = np.cos(w2)
    hp_b = ((1 + cos_w2) / 2.0, -(1 + cos_w2), (1 + cos_w2) / 2.0)
    hp_a = (1 + alpha2, -2 * cos_w2, 1 - alpha2)

    return _biquad_magnitude(shelf_b, shelf_a, freqs, sample_rate) * _biquad_magnitude(hp_b, hp_a, freqs, sample_rate)


def _apply_response(samples: np.ndarray, response: np.ndarray) -> np.ndarray:
    """Apply a zero-phase magnitude response to every channel."""
    spectrum = np.fft.rfft(samples, axis=0)
    return np.fft.irfft(spectrum * response[:, None], n=samples.shape[0], axis=0).astype(np.float32)


def measure_lufs(samples: np.ndarray, sample_rate: int) -> float:
    """Integrated loudness in LUFS, with the BS.1770 absolute and relative gates."""
    frequencies = np.fft.rfftfreq(samples.shape[0], 1.0 / sample_rate)
    weighted = _apply_response(samples, _k_weighting_response(frequencies, sample_rate))

    block = max(1, int(0.4 * sample_rate))
    step = max(1, block // 4)  # 75% overlap, per the standard
    if weighted.shape[0] < block:
        mean_square = float(np.mean(weighted**2)) + _EPSILON
        return -0.691 + 10.0 * np.log10(mean_square)

    starts = np.arange(0, weighted.shape[0] - block + 1, step)
    powers = np.empty(len(starts), dtype=np.float64)
    for index, start in enumerate(starts):
        segment = weighted[start : start + block]
        powers[index] = float(np.sum(np.mean(segment**2, axis=0)))

    loudness = -0.691 + 10.0 * np.log10(powers + _EPSILON)

    # Absolute gate, then a gate 10 LU below the mean of what survived.
    keep = loudness > -70.0
    if not np.any(keep):
        return float(np.mean(loudness))

    ungated_mean = -0.691 + 10.0 * np.log10(np.mean(powers[keep]) + _EPSILON)
    keep &= loudness > (ungated_mean - 10.0)
    if not np.any(keep):
        return float(ungated_mean)

    return float(-0.691 + 10.0 * np.log10(np.mean(powers[keep]) + _EPSILON))


# --------------------------------------------------------------------------- #
# Spectral processing
# --------------------------------------------------------------------------- #


def _stft(signal: np.ndarray, n_fft: int, hop: int) -> tuple[np.ndarray, int]:
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    padded = np.pad(signal, (n_fft // 2, n_fft), mode="reflect")
    frame_count = 1 + (len(padded) - n_fft) // hop

    strides = (padded.strides[0] * hop, padded.strides[0])
    frames = np.lib.stride_tricks.as_strided(padded, shape=(frame_count, n_fft), strides=strides)
    return np.fft.rfft(frames * window, axis=1), len(padded)


def _istft(spectrum: np.ndarray, n_fft: int, hop: int, length: int, output_length: int) -> np.ndarray:
    window = np.hanning(n_fft + 1)[:-1].astype(np.float32)
    frames = np.fft.irfft(spectrum, n=n_fft, axis=1) * window

    signal = np.zeros(length, dtype=np.float64)
    weight = np.zeros(length, dtype=np.float64)
    for index in range(frames.shape[0]):
        start = index * hop
        signal[start : start + n_fft] += frames[index]
        weight[start : start + n_fft] += window**2

    signal /= np.maximum(weight, 1e-8)
    return signal[n_fft // 2 : n_fft // 2 + output_length].astype(np.float32)


def _smooth(matrix: np.ndarray, time_span: int, freq_span: int) -> np.ndarray:
    """Box-smooth a gain mask over time and frequency to suppress musical noise."""
    smoothed = matrix
    if time_span > 1:
        kernel = np.ones(time_span, dtype=np.float32) / time_span
        smoothed = np.apply_along_axis(lambda column: np.convolve(column, kernel, mode="same"), 0, smoothed)
    if freq_span > 1:
        kernel = np.ones(freq_span, dtype=np.float32) / freq_span
        smoothed = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, smoothed)
    return smoothed


def _denoise_channel(
    channel: np.ndarray,
    sample_rate: int,
    strength: float,
    highpass_hz: float,
) -> tuple[np.ndarray, float, float]:
    """Spectral-gate one channel. Returns audio, noise floor dBFS and SNR estimate."""
    n_fft = 1024 if sample_rate <= 24000 else 2048
    hop = n_fft // 4

    spectrum, padded_length = _stft(channel, n_fft, hop)
    magnitude = np.abs(spectrum)

    # A low percentile across time is a robust stationary-noise estimate: speech
    # and music are intermittent, the noise floor is not.
    noise_profile = np.percentile(magnitude, 15.0, axis=0)

    frame_energy = magnitude.mean(axis=1)
    noise_level = float(np.mean(noise_profile) + _EPSILON)
    signal_level = float(np.percentile(frame_energy, 95.0) + _EPSILON)
    snr_db = 20.0 * np.log10(signal_level / noise_level)
    noise_floor_dbfs = 20.0 * np.log10(noise_level / (np.max(magnitude) + _EPSILON) + _EPSILON)

    # Spectral subtraction with over-subtraction and a floor, so removed noise
    # is attenuated rather than nulled - nulling is what causes the watery
    # "musical noise" artefact.
    over_subtraction = 1.0 + 1.5 * strength
    floor_gain = 10.0 ** (-(6.0 + 16.0 * strength) / 20.0)

    gain = 1.0 - over_subtraction * (noise_profile[None, :] / (magnitude + _EPSILON))
    gain = np.clip(gain, floor_gain, 1.0)
    gain = _smooth(gain.astype(np.float32), time_span=3, freq_span=5)

    # Fold the high-pass into the same mask - cheaper than a separate filter and
    # it removes rumble before the limiter ever sees it.
    if highpass_hz > 0:
        frequencies = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)
        rolloff = np.clip((frequencies - highpass_hz * 0.5) / max(highpass_hz * 0.5, 1.0), 0.0, 1.0)
        gain *= rolloff[None, :].astype(np.float32)

    cleaned = _istft(spectrum * gain, n_fft, hop, padded_length, len(channel))
    return cleaned, noise_floor_dbfs, snr_db


def _limit(samples: np.ndarray, ceiling_dbfs: float, sample_rate: int) -> tuple[np.ndarray, bool]:
    """Vectorised soft limiter: smoothed gain reduction, no per-sample loop."""
    ceiling = 10.0 ** (ceiling_dbfs / 20.0)
    peak_envelope = np.max(np.abs(samples), axis=1)

    if float(np.max(peak_envelope, initial=0.0)) <= ceiling:
        return samples, False

    required = np.minimum(1.0, ceiling / np.maximum(peak_envelope, _EPSILON))

    # Take a running minimum over a short window so gain dips slightly before
    # each peak, then smooth it so the reduction is inaudible.
    window = max(1, int(0.005 * sample_rate))
    padded = np.pad(required, (window, window), mode="edge")
    strides = (padded.strides[0], padded.strides[0])
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=(len(required) + window, 2 * window), strides=strides
    )
    running_min = windows.min(axis=1)[: len(required)]

    kernel = np.ones(window * 2 + 1, dtype=np.float64) / (window * 2 + 1)
    smoothed = np.convolve(running_min, kernel, mode="same")
    smoothed = np.minimum(smoothed, required)

    return (samples * smoothed[:, None]).astype(np.float32), True


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def enhance_audio(
    data: bytes,
    *,
    denoise_strength: float = 0.6,
    target_lufs: float = -16.0,
    peak_ceiling_dbfs: float = -1.0,
    highpass_hz: float = 80.0,
    output_format: str = "wav",
) -> tuple[bytes, str, AudioAnalysis]:
    """Clean up a recording: remove noise and rumble, then normalise loudness.

    Order matters. Noise is removed before loudness is measured, otherwise the
    noise floor inflates the measurement and the result comes out too quiet. The
    limiter runs last so nothing after it can reintroduce clipping.
    """
    samples, sample_rate = load_audio(data)
    strength = float(np.clip(denoise_strength, 0.0, 1.0))
    actions: list[str] = []

    duration = samples.shape[0] / sample_rate
    channels = samples.shape[1]
    input_peak = float(np.max(np.abs(samples), initial=0.0))
    input_peak_dbfs = 20.0 * np.log10(input_peak + _EPSILON)
    input_lufs = measure_lufs(samples, sample_rate)

    # 1. Denoise + de-rumble, per channel.
    noise_floors: list[float] = []
    snrs: list[float] = []
    if strength > 0.01:
        cleaned = np.empty_like(samples)
        for channel in range(channels):
            cleaned[:, channel], floor_dbfs, snr_db = _denoise_channel(
                samples[:, channel], sample_rate, strength, highpass_hz
            )
            noise_floors.append(floor_dbfs)
            snrs.append(snr_db)
        actions.append(f"spectral noise gate at {strength * 100:.0f}% strength")
        if highpass_hz > 0:
            actions.append(f"removed rumble below {highpass_hz:.0f} Hz")
    else:
        cleaned = samples.copy()
        noise_floors.append(-90.0)
        snrs.append(0.0)
        actions.append("denoise disabled, left the noise floor untouched")

    noise_floor_dbfs = float(np.mean(noise_floors))
    estimated_snr = float(np.mean(snrs))

    # 2. Measure how much noise actually went away.
    residual = samples[: cleaned.shape[0]] - cleaned
    residual_energy = float(np.mean(residual**2))
    noise_reduction_db = (
        10.0 * np.log10((float(np.mean(samples**2)) + _EPSILON) / (float(np.mean(cleaned**2)) + _EPSILON))
        if residual_energy > 0
        else 0.0
    )

    # 3. Loudness normalisation to the target.
    measured = measure_lufs(cleaned, sample_rate)
    gain_db = float(np.clip(target_lufs - measured, -30.0, 30.0))
    cleaned = (cleaned * (10.0 ** (gain_db / 20.0))).astype(np.float32)
    actions.append(f"normalised loudness {measured:.1f} -> {target_lufs:.1f} LUFS ({gain_db:+.1f} dB)")

    # 4. Limit so the gain stage cannot clip.
    cleaned, limited = _limit(cleaned, peak_ceiling_dbfs, sample_rate)
    if limited:
        actions.append(f"limited true peaks to {peak_ceiling_dbfs:.1f} dBFS")
    else:
        actions.append(f"peaks already below {peak_ceiling_dbfs:.1f} dBFS")

    cleaned = np.clip(cleaned, -1.0, 1.0)
    output_lufs = measure_lufs(cleaned, sample_rate)
    output_peak_dbfs = 20.0 * np.log10(float(np.max(np.abs(cleaned), initial=0.0)) + _EPSILON)

    analysis = AudioAnalysis(
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        input_lufs=input_lufs,
        output_lufs=output_lufs,
        input_peak_dbfs=input_peak_dbfs,
        output_peak_dbfs=output_peak_dbfs,
        noise_floor_dbfs=noise_floor_dbfs,
        estimated_snr_db=estimated_snr,
        noise_reduction_db=noise_reduction_db,
        actions=actions,
    )
    analysis.summary = (
        f"{duration:.1f}s, {sample_rate} Hz, {channels}ch. "
        f"Loudness {input_lufs:.1f} -> {output_lufs:.1f} LUFS, "
        f"peak {input_peak_dbfs:.1f} -> {output_peak_dbfs:.1f} dBFS, "
        f"estimated SNR {estimated_snr:.1f} dB."
    )

    payload, resolved_format = encode_audio(cleaned, sample_rate, output_format)
    return payload, resolved_format, analysis

import json
import math
import os
import re
import struct
import subprocess
import time
import wave
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from app.config import get_settings
from app.models import Job, JobCreate, JobStatus, VoiceDesignRequest, VoiceProfile, VoiceProfileCreate, VoiceSample


class JobStore:
    def __init__(self, max_retained: int | None = None) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = Lock()
        # 0/None means unbounded. Configured lazily from settings on first use
        # so tests can construct a store without a settings object.
        self._max_retained = max_retained

    @property
    def max_retained(self) -> int:
        if self._max_retained is None:
            self._max_retained = get_settings().max_jobs_retained
        return self._max_retained

    def create(self, payload: JobCreate) -> Job:
        job = Job(**payload.model_dump())
        with self._lock:
            self._jobs[job.id] = job
            self._evict_locked()
        return job

    def _evict_locked(self) -> None:
        """Drop the oldest *finished* jobs once the store passes its cap.

        Running/queued jobs are never evicted - only completed and failed ones,
        oldest first - so an active job cannot vanish from under a poller.
        """
        cap = self.max_retained
        if cap <= 0 or len(self._jobs) <= cap:
            return

        finished = sorted(
            (job for job in self._jobs.values() if job.status in {JobStatus.completed, JobStatus.failed}),
            key=lambda job: job.updated_at,
        )
        removable = len(self._jobs) - cap
        for job in finished[:removable]:
            self._jobs.pop(job.id, None)

    def list(self) -> list[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.created_at, reverse=True)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def mark_running(self, job_id: str) -> Job | None:
        return self._update_status(job_id, JobStatus.running)

    def set_progress(self, job_id: str, progress: int, stage: str) -> Job | None:
        """Report progress on a long job so the client can poll it."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.progress = max(0, min(100, progress))
            job.stage = stage
            job.status = JobStatus.running
            job.updated_at = datetime.now(UTC)
            return job

    def set_result(self, job_id: str, result: dict) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.result = {**job.result, **result}
            job.updated_at = datetime.now(UTC)
            return job

    def mark_completed(self, job_id: str, output_uri: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = JobStatus.completed
            job.output_uri = output_uri
            job.updated_at = datetime.now(UTC)
            return job

    def mark_failed(self, job_id: str, error: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = JobStatus.failed
            job.error = error
            job.updated_at = datetime.now(UTC)
            return job

    def _update_status(self, job_id: str, status: JobStatus) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.status = status
            job.updated_at = datetime.now(UTC)
            return job


job_store = JobStore()


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 3, recovery_seconds: int = 300) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self._failure_count = 0
        self._opened_at: float | None = None
        self._last_error: str | None = None
        self._lock = Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self._opened_at is None:
                return True
            return time.monotonic() - self._opened_at >= self.recovery_seconds

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            self._opened_at = None
            self._last_error = None

    def record_failure(self, error: Exception | str) -> None:
        with self._lock:
            self._failure_count += 1
            self._last_error = str(error)
            if self._failure_count >= self.failure_threshold:
                self._opened_at = time.monotonic()

    def open(self, reason: str, recovery_seconds: int | None = None) -> None:
        with self._lock:
            self._failure_count = max(self._failure_count, self.failure_threshold)
            self._last_error = reason
            self._opened_at = time.monotonic()
            if recovery_seconds is not None:
                self.recovery_seconds = recovery_seconds

    def status(self) -> dict[str, object]:
        with self._lock:
            if self._opened_at is None:
                state = "closed"
                retry_after_seconds = 0
            else:
                elapsed = time.monotonic() - self._opened_at
                retry_after_seconds = max(0, int(self.recovery_seconds - elapsed))
                state = "half-open" if retry_after_seconds == 0 else "open"

            return {
                "state": state,
                "failure_count": self._failure_count,
                "failure_threshold": self.failure_threshold,
                "recovery_seconds": self.recovery_seconds,
                "retry_after_seconds": retry_after_seconds,
                "last_error": self._last_error,
            }


omnivoice_breaker = CircuitBreaker()
segmentation_breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=900)


class SystemHealthMonitor:
    def __init__(self) -> None:
        self._last_cpu_percent: float | None = None

    def snapshot(self) -> dict[str, object]:
        cpu_percent = self._cpu_percent()
        self._last_cpu_percent = cpu_percent
        settings = get_settings()
        if cpu_percent is None:
            state = "unknown"
            warning = "Install psutil for local CPU health based circuit breaking."
        elif cpu_percent >= settings.local_cpu_critical_percent:
            state = "critical"
            warning = "Local CPU is critically high; OmniVoice CPU synthesis should pause."
        elif cpu_percent >= settings.local_cpu_warning_percent:
            state = "warning"
            warning = "Local CPU is high; OmniVoice CPU synthesis may be paused."
        else:
            state = "healthy"
            warning = None

        return {
            "state": state,
            "cpu_percent": cpu_percent,
            "warning": warning,
            "warning_threshold": settings.local_cpu_warning_percent,
            "critical_threshold": settings.local_cpu_critical_percent,
        }

    def _cpu_percent(self) -> float | None:
        try:
            import psutil  # type: ignore[import-not-found]
        except ImportError:
            return None
        # interval=None is non-blocking: it reports load since the previous
        # call instead of sleeping 100 ms inside every health request. The very
        # first reading is 0.0, which the caller treats as "healthy".
        return float(psutil.cpu_percent(interval=None))


system_health = SystemHealthMonitor()


class VoiceStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self._voices: dict[str, VoiceProfile] = {}
        self._lock = Lock()
        self.data_dir = data_dir or get_settings().data_dir
        self.samples_dir = self.data_dir / "voice_samples"
        self.profiles_dir = self.data_dir / "voice_profiles"
        self.outputs_dir = self.data_dir / "outputs"
        self.index_path = self.profiles_dir / "index.json"
        for directory in (self.samples_dir, self.profiles_dir, self.outputs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._load()

    def create_clone(self, payload: VoiceProfileCreate) -> VoiceProfile:
        # Build the profile first so its unique id can be folded into the bundle
        # filename; naming the bundle after the display name alone let two
        # profiles called "My Voice" overwrite each other's data.
        profile = VoiceProfile(**payload.model_dump())
        bundle_path = self._bundle_path(payload.name, profile.id)
        profile = profile.model_copy(update={"portable_bundle_uri": self._to_local_uri(bundle_path)})
        self._write_bundle(profile, bundle_path, kind="clone")
        with self._lock:
            self._voices[profile.id] = profile
            self._save()
        return profile

    def create_design(self, payload: VoiceDesignRequest) -> VoiceProfile:
        from app.voice_styles import design_to_instruct

        design_instruct = design_to_instruct(
            gender=payload.gender, age=payload.age, accent=payload.accent, pitch=payload.pitch
        )
        profile = VoiceProfile(
            name=payload.name,
            language=payload.language,
            reference_uri="local://generated/voice-design",
            consent_verified=True,
            tags=[payload.gender, payload.age, payload.accent, payload.emotion, payload.dialect],
            engine="voice-design-local",
            design_instruct=design_instruct,
        )
        bundle_path = self._bundle_path(payload.name, profile.id)
        profile = profile.model_copy(update={"portable_bundle_uri": self._to_local_uri(bundle_path)})
        self._write_bundle(profile, bundle_path, kind="design", design=payload.model_dump())
        with self._lock:
            self._voices[profile.id] = profile
            self._save()
        return profile

    def save_sample(self, filename: str, content_type: str, source, *, max_bytes: int | None = None) -> VoiceSample:
        suffix = Path(filename or "sample.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm"}:
            suffix = ".wav"
        sample = VoiceSample(
            filename=filename or f"voice-sample{suffix}",
            content_type=content_type or "application/octet-stream",
            size_bytes=0,
            uri="",
        )
        destination = self.samples_dir / f"{sample.id}{suffix}"
        # Stream in chunks with an optional cap so an unbounded upload cannot
        # fill the disk.
        written = 0
        with destination.open("wb") as handle:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if max_bytes is not None and written > max_bytes:
                    handle.close()
                    destination.unlink(missing_ok=True)
                    raise ValueError(f"Voice sample exceeds the {max_bytes} byte limit.")
                handle.write(chunk)
        sample.size_bytes = destination.stat().st_size
        sample.uri = self._to_local_uri(destination)
        return sample

    def write_preview_wav(self, job_id: str, text: str = "preview audio") -> str:
        destination = self.outputs_dir / f"{job_id}.wav"
        # The SAPI path only exists on Windows; attempting it elsewhere just
        # burns a failed subprocess launch on every request.
        if os.name == "nt" and self._write_windows_tts_wav(destination, text):
            return self._to_local_uri(destination)

        self._write_tone_wav(destination, text)
        return self._to_local_uri(destination)

    def _write_windows_tts_wav(self, destination: Path, text: str) -> bool:
        env = {
            **os.environ,
            "ENHANCEAI_TTS_TEXT": text[:4000],
            "ENHANCEAI_TTS_OUTPUT": str(destination),
        }
        command = [
            "powershell.exe",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-Command",
            """
$text = [Environment]::GetEnvironmentVariable('ENHANCEAI_TTS_TEXT')
$output = [Environment]::GetEnvironmentVariable('ENHANCEAI_TTS_OUTPUT')
Add-Type -AssemblyName System.Speech
$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer
$speaker.Rate = 0
$speaker.Volume = 100
$speaker.SetOutputToWaveFile($output)
$speaker.Speak($text)
$speaker.Dispose()
""",
        ]
        try:
            completed = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False, env=env)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return completed.returncode == 0 and destination.exists() and destination.stat().st_size > 44

    def _write_tone_wav(self, destination: Path, text: str) -> None:
        sample_rate = 16000
        amplitude = 9000
        frames = bytearray()
        words = [word for word in re.split(r"\s+", text.strip()) if word] or ["preview"]

        for word in words[:12]:
            frequency = 360 + (sum(ord(character) for character in word) % 420)
            sample_count = int(sample_rate * 0.22)
            for index in range(sample_count):
                envelope = min(1.0, index / 600, (sample_count - index) / 600)
                sample = int(amplitude * envelope * math.sin(2 * math.pi * frequency * index / sample_rate))
                frames.extend(struct.pack("<h", sample))
            frames.extend(b"\x00\x00" * int(sample_rate * 0.05))

        with wave.open(str(destination), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(bytes(frames))

    def output_path(self, job_id: str, suffix: str = ".wav") -> Path:
        return self.outputs_dir / f"{job_id}{suffix}"

    def uri_for_path(self, path: Path) -> str:
        return self._to_local_uri(path)

    def list(self) -> list[VoiceProfile]:
        with self._lock:
            return sorted(self._voices.values(), key=lambda voice: voice.created_at, reverse=True)

    def get(self, voice_id: str) -> VoiceProfile | None:
        with self._lock:
            return self._voices.get(voice_id)

    def resolve_local_uri(self, uri: str) -> Path | None:
        prefix = "local://"
        if not uri.startswith(prefix):
            return None
        relative = Path(uri[len(prefix) :])
        path = (self.data_dir / relative).resolve()
        data_root = self.data_dir.resolve()
        if path == data_root or data_root in path.parents:
            return path
        return None

    def _bundle_path(self, name: str, profile_id: str) -> Path:
        slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "voice"
        # The id keeps two same-named profiles from clobbering one another.
        return self.profiles_dir / f"{slug}-{profile_id}.ovsvoice"

    def _to_local_uri(self, path: Path) -> str:
        return f"local://{path.resolve().relative_to(self.data_dir.resolve()).as_posix()}"

    def _write_bundle(self, profile: VoiceProfile, path: Path, **extra: object) -> None:
        payload = {
            "format": "omnivoice-profile",
            "version": 1,
            "profile": profile.model_dump(mode="json"),
            **extra,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        try:
            profiles = json.loads(self.index_path.read_text(encoding="utf-8"))
            self._voices = {item["id"]: VoiceProfile(**item) for item in profiles}
        except (OSError, ValueError, TypeError, KeyError):
            self._voices = {}

    def _save(self) -> None:
        payload = [profile.model_dump(mode="json") for profile in self._voices.values()]
        self.index_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


voice_store = VoiceStore()

import io

import pytest

from app.models import JobCreate, MediaType, VoiceDesignRequest, VoiceProfileCreate
from app.services.core import CircuitBreaker, JobStore, SystemHealthMonitor, VoiceStore


def _job(store, module="m"):
    return store.create(JobCreate(module=module, media_type=MediaType.image, input_uri="u"))


def test_jobstore_crud_and_ordering():
    store = JobStore(max_retained=0)
    a = _job(store)
    b = _job(store)
    assert store.get(a.id).id == a.id
    listed = store.list()
    assert {job.id for job in listed} == {a.id, b.id}
    assert store.mark_completed(a.id, "local://outputs/a").status == "completed"
    assert store.mark_failed(b.id, "boom").error == "boom"
    assert store.get("missing") is None


def test_jobstore_eviction_keeps_running_jobs():
    store = JobStore(max_retained=5)
    finished = [_job(store) for _ in range(5)]
    for job in finished:
        store.mark_completed(job.id, "local://outputs/x")
    running = _job(store)
    store.mark_running(running.id)
    # Push well past the cap with more finished jobs.
    for _ in range(10):
        job = _job(store)
        store.mark_completed(job.id, "local://outputs/y")
    assert len(store.list()) <= 5
    assert store.get(running.id) is not None  # a running job is never evicted


def test_jobstore_progress_clamped():
    store = JobStore(max_retained=0)
    job = _job(store)
    assert store.set_progress(job.id, 150, "done").progress == 100
    assert store.set_progress(job.id, -5, "x").progress == 0


def test_circuit_breaker_opens_and_recovers():
    breaker = CircuitBreaker(failure_threshold=2, recovery_seconds=0)
    assert breaker.allow_request() is True
    breaker.record_failure("a")
    assert breaker.status()["state"] == "closed"
    breaker.record_failure("b")
    # recovery_seconds=0 means it immediately reports half-open and allows again.
    assert breaker.status()["state"] in {"half-open", "open"}
    assert breaker.allow_request() is True
    breaker.record_success()
    assert breaker.status()["state"] == "closed"
    assert breaker.status()["failure_count"] == 0


def test_circuit_breaker_manual_open():
    breaker = CircuitBreaker(recovery_seconds=999)
    breaker.open("maintenance", recovery_seconds=999)
    assert breaker.status()["state"] == "open"
    assert breaker.allow_request() is False


def test_health_monitor_snapshot_keys():
    snapshot = SystemHealthMonitor().snapshot()
    assert {"state", "cpu_percent", "warning", "warning_threshold", "critical_threshold"} <= set(snapshot)


def test_voicestore_clone_and_design_unique_bundles(tmp_path):
    store = VoiceStore(data_dir=tmp_path)
    a = store.create_clone(VoiceProfileCreate(name="My Voice", reference_uri="local://r.wav", consent_verified=True))
    b = store.create_clone(VoiceProfileCreate(name="My Voice", reference_uri="local://r.wav", consent_verified=True))
    assert a.portable_bundle_uri != b.portable_bundle_uri  # same name, different bundle

    design = store.create_design(VoiceDesignRequest(name="Designed", gender="female", age="young", accent="british"))
    assert "female" in design.design_instruct
    assert store.get(design.id).id == design.id


def test_voicestore_sample_size_limit(tmp_path):
    store = VoiceStore(data_dir=tmp_path)
    big = io.BytesIO(b"\x00" * 2048)
    with pytest.raises(ValueError):
        store.save_sample("s.wav", "audio/wav", big, max_bytes=1024)


def test_voicestore_resolve_local_uri_blocks_traversal(tmp_path):
    store = VoiceStore(data_dir=tmp_path)
    assert store.resolve_local_uri("local://../../etc/passwd") is None
    inside = store.resolve_local_uri("local://outputs/ok.wav")
    assert inside is not None

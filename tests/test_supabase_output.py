from app import supabase_output as so


def test_disabled_without_config(monkeypatch):
    for name in ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY", "NEXT_PUBLIC_SUPABASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(so, "_LOCAL_ENV", {})
    assert so.supabase_output_enabled() is False


def test_requires_service_role_key(monkeypatch):
    monkeypatch.setattr(so, "_LOCAL_ENV", {})
    monkeypatch.setenv("SUPABASE_URL", "https://demo.supabase.co")
    monkeypatch.setenv("SUPABASE_ANON_KEY", "anon-key")
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    # The anon key must NOT enable server-side writes.
    assert so.supabase_output_enabled() is False

    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", "service-key")
    assert so.supabase_output_enabled() is True


def test_default_bucket_name(monkeypatch):
    monkeypatch.setattr(so, "_LOCAL_ENV", {})
    monkeypatch.delenv("SUPABASE_OUTPUTS_BUCKET", raising=False)
    assert so.output_bucket_name() == "outputs"


def test_record_completed_output_noop_without_uri():
    class _Job:
        output_uri = None
        module = "m"
        id = "job_x"

    # Should return quietly, not raise, when there is nothing to sync.
    so.record_completed_output(_Job(), voice_store=None)

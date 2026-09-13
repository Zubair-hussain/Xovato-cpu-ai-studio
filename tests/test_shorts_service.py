import io

import pytest

from app.config import get_settings
from app.shorts_service import UploadTooLarge, create_analysis_job, project_dir, save_upload


def test_project_dir_under_data():
    settings = get_settings()
    path = project_dir(settings, "proj_abc")
    assert path.parent.name == "shorts"


def test_save_upload_writes_file():
    settings = get_settings()
    source = io.BytesIO(b"hello world video bytes")
    dest = save_upload(settings, "proj_testwrite", "clip.mp4", source)
    assert dest.exists()
    assert dest.read_bytes() == b"hello world video bytes"
    assert dest.suffix == ".mp4"


def test_save_upload_enforces_limit():
    settings = get_settings()
    source = io.BytesIO(b"\x00" * 5000)
    with pytest.raises(UploadTooLarge):
        save_upload(settings, "proj_toobig", "clip.mp4", source, max_bytes=1024)
    # The partial file must be cleaned up on rejection.
    assert not (project_dir(settings, "proj_toobig") / "source.mp4").exists()


def test_create_analysis_job_registers_job():
    from app.services import job_store

    job_id = create_analysis_job("proj_job", "clip.mp4")
    assert job_store.get(job_id) is not None
    assert job_store.get(job_id).module == "shorts-studio"

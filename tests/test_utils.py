from datetime import UTC, datetime, timedelta

from app.utils.cleanup import remove_expired_files
from app.utils.files import ensure_directory


def test_ensure_directory_creates(tmp_path):
    target = tmp_path / "a" / "b"
    result = ensure_directory(target)
    assert result.exists() and result.is_dir()


def test_remove_expired_files(tmp_path):
    old = tmp_path / "old.txt"
    new = tmp_path / "new.txt"
    old.write_text("x")
    new.write_text("y")

    # Backdate the "old" file well past the cutoff.
    import os

    past = (datetime.now(UTC) - timedelta(days=2)).timestamp()
    os.utime(old, (past, past))

    removed = remove_expired_files(tmp_path, datetime.now(UTC) - timedelta(days=1))
    assert removed == 1
    assert not old.exists()
    assert new.exists()

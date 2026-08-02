from datetime import UTC, datetime
from pathlib import Path


def remove_expired_files(root: str | Path, older_than: datetime) -> int:
    removed = 0
    for path in Path(root).rglob("*"):
        if not path.is_file():
            continue
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if modified < older_than:
            path.unlink()
            removed += 1
    return removed


# `app/utils/` - small helpers

Tiny, dependency-free utilities used by the deployment layer.

| File | Function | Purpose |
| --- | --- | --- |
| `files.py` | `ensure_directory(path)` | Create a directory (and parents) if missing, return it. |
| `cleanup.py` | `remove_expired_files(root, older_than)` | Delete files under `root` last modified before `older_than`; returns the count removed. |

`remove_expired_files` powers the optional startup retention prune in
`app/main.py` (`ENHANCEAI_OUTPUT_RETENTION_HOURS`). Both are covered by
`tests/test_utils.py`.

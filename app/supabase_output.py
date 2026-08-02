from __future__ import annotations

import json
import logging
import mimetypes
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from app.models import Job

logger = logging.getLogger(__name__)


def _local_env_values() -> dict[str, str]:
    roots = [
        Path(__file__).resolve().parents[1] / ".env",
        Path(__file__).resolve().parents[2] / "xovato-app" / ".env.local",
    ]
    values: dict[str, str] = {}
    for path in roots:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("//") or "=" not in stripped:
                continue
            name, value = stripped.split("=", 1)
            values.setdefault(name.strip(), value.strip().strip('"').strip("'"))
    return values


_LOCAL_ENV = _local_env_values()


def _env(name: str) -> str:
    return os.getenv(name) or os.getenv(f"ENHANCEAI_{name}") or _LOCAL_ENV.get(name) or _LOCAL_ENV.get(f"ENHANCEAI_{name}") or ""


def supabase_output_enabled() -> bool:
    return _supabase_config() is not None


def output_bucket_name() -> str:
    return _env("SUPABASE_OUTPUTS_BUCKET") or "outputs"


def _supabase_config() -> tuple[str, str] | None:
    url = _env("SUPABASE_URL") or _env("NEXT_PUBLIC_SUPABASE_URL")
    key = _env("SUPABASE_SERVICE_ROLE_KEY") or _env("SUPABASE_ANON_KEY") or _env("NEXT_PUBLIC_SUPABASE_ANON_KEY")
    if not url or not key:
        return None
    return url.rstrip("/"), key.strip()


def _request_json(url: str, key: str, payload: dict[str, object]) -> None:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        },
    )
    with urlopen(request, timeout=20):
        return


def _upload_storage(url: str, key: str, bucket: str, storage_path: str, path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None

    encoded_path = "/".join(quote(part) for part in storage_path.split("/"))
    mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    request = Request(
        f"{url}/storage/v1/object/{quote(bucket)}/{encoded_path}",
        data=path.read_bytes(),
        method="POST",
        headers={
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": mime_type,
            "x-upsert": "true",
        },
    )
    with urlopen(request, timeout=60):
        return f"{bucket}/{storage_path}"


def record_completed_output(job: Job, voice_store) -> None:
    """Best-effort upload/insert for generated outputs.

    Local audio generation must still work when Supabase is not configured or
    the table/bucket has not been created yet, so failures are logged only.
    """
    if not job.output_uri:
        return

    config = _supabase_config()
    if config is None:
        logger.info(
            "supabase output sync skipped",
            extra={"module_name": job.module, "action": "supabase-output-skip", "job_id": job.id},
        )
        return

    url, key = config
    local_path = voice_store.resolve_local_uri(job.output_uri)
    storage_bucket = output_bucket_name()
    storage_path = f"{job.module}/{job.id}{local_path.suffix if local_path else ''}"
    stored_object = None

    try:
        if local_path is not None:
            stored_object = _upload_storage(url, key, storage_bucket, storage_path, local_path)

        _request_json(
            f"{url}/rest/v1/media_outputs",
            key,
            {
                "job_id": job.id,
                "module": job.module,
                "media_type": str(job.media_type),
                "status": str(job.status),
                "output_uri": job.output_uri,
                "storage_bucket": storage_bucket if stored_object else None,
                "storage_path": storage_path if stored_object else None,
                "settings": job.settings,
                "result": job.result,
            },
        )
        logger.info(
            "supabase output synced",
            extra={"module_name": job.module, "action": "supabase-output-sync", "job_id": job.id},
        )
    except (HTTPError, URLError, OSError, TimeoutError) as error:
        logger.warning(
            "supabase output sync failed",
            extra={
                "module_name": job.module,
                "action": "supabase-output-failed",
                "job_id": job.id,
                "warning": str(error),
            },
        )

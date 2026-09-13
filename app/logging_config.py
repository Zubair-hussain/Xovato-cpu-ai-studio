import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class HealthLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        action = getattr(record, "action", "")
        module_name = getattr(record, "module_name", "")
        return module_name in {"system-health", "text-to-speech"} and action in {
            "health-check",
            "omnivoice-breaker-open",
            "omnivoice-breaker-trip",
            "omnivoice-fallback",
            "omnivoice-health-preflight",
        }


# Attributes present on every LogRecord; anything else attached via ``extra=``
# is application context we want to keep in the JSON line.
_STANDARD_RECORD_KEYS = frozenset(
    {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
        "message",
        "asctime",
    }
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "service": "enhanceai-core-backend",
            "message": record.getMessage(),
        }

        # Keep every extra field the caller attached (output_bytes,
        # generate_seconds, realtime_factor, cloned, faces_detected, path, …),
        # not just a fixed allowlist. Values must be JSON-serialisable; anything
        # exotic is stringified so a stray object never drops the whole line.
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_KEYS or key.startswith("_") or value is None:
                continue
            try:
                json.dumps(value)
                payload[key] = value
            except (TypeError, ValueError):
                payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging(level: str = "INFO", log_dir: Path | None = None) -> None:
    formatter = JsonFormatter()
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(stream_handler)
    root_logger.setLevel(level)

    if log_dir is None:
        return

    log_dir.mkdir(parents=True, exist_ok=True)

    error_handler = logging.FileHandler(log_dir / "error.log", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    root_logger.addHandler(error_handler)

    health_handler = logging.FileHandler(log_dir / "health.log", encoding="utf-8")
    health_handler.setLevel(logging.INFO)
    health_handler.addFilter(HealthLogFilter())
    health_handler.setFormatter(formatter)
    root_logger.addHandler(health_handler)

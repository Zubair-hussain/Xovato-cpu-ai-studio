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


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname.lower(),
            "service": "enhanceai-core-backend",
            "message": record.getMessage(),
        }

        for key in (
            "module_name",
            "action",
            "job_id",
            "media_type",
            "cpu_percent",
            "circuit_state",
            "retry_after_seconds",
            "warning",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value

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

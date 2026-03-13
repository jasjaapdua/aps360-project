"""Custom file-based logging utilities for CLI command runs.

This module provides a small structured logger that writes one JSON object per
line to a configured log file. It is designed for experiment traceability.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def _utc_now_iso() -> str:
    """Return current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    """Convert non-JSON-native objects to serializable representations."""
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    return repr(value)


class JsonLineFormatter(logging.Formatter):
    """Format log records as one compact JSON document per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": _utc_now_iso(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        event = getattr(record, "event", None)
        data = getattr(record, "data", None)
        if event is not None:
            payload["event"] = event
        if data is not None:
            payload["data"] = data

        return json.dumps(payload, default=_json_default, ensure_ascii=True)


class RunLogger:
    """Structured logger that records lifecycle events for one CLI run."""

    def __init__(self, logger: logging.Logger, run_id: str):
        self._logger = logger
        self.run_id = run_id
        self._start_time = time.monotonic()

    def info(self, event: str, **data: Any) -> None:
        """Write an informational structured event."""
        self._logger.info(event, extra={"event": event, "data": self._with_run_meta(data)})

    def error(self, event: str, **data: Any) -> None:
        """Write an error structured event."""
        self._logger.error(event, extra={"event": event, "data": self._with_run_meta(data)})

    def finish(self, status: str, **data: Any) -> None:
        """Write final completion event with duration metadata."""
        duration_seconds = round(time.monotonic() - self._start_time, 4)
        payload = {"status": status, "duration_seconds": duration_seconds, **data}
        self.info("run_finished", **payload)

    def _with_run_meta(self, data: dict[str, Any]) -> dict[str, Any]:
        return {"run_id": self.run_id, **data}


def build_run_logger(
    log_file: str,
    level: str = "INFO",
    logger_name: str = "lyrics_cli",
) -> RunLogger:
    """Create a file logger and wrap it in :class:`RunLogger`.

    Args:
        log_file: Log file path. Parent directories are created automatically.
        level: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`).
        logger_name: Underlying Python logger name.

    Returns:
        Initialized :class:`RunLogger` for the current command execution.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    absolute_path = str(log_path.resolve())
    existing_handler = None
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == Path(absolute_path):
            existing_handler = handler
            break

    if existing_handler is None:
        file_handler = logging.FileHandler(absolute_path, encoding="utf-8")
        file_handler.setFormatter(JsonLineFormatter())
        logger.addHandler(file_handler)

    run_logger = RunLogger(logger=logger, run_id=str(uuid4()))
    run_logger.info(
        "run_started",
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        executable=sys.executable,
    )
    return run_logger

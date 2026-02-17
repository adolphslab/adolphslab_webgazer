"""Logging utilities.

Requirements
- REV-REPRO-002: capture logs per run; avoid logging raw sensitive data.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Optional


class JsonFormatter(logging.Formatter):
    """JSON formatter aligned with stage-1/2 structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(*, log_path: Path, level: str = "INFO") -> logging.Logger:
    """Configure stage logger with console + JSONL file handlers."""
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("alabwebgazer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = JsonFormatter()

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str, level: Optional[int] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(level)
    return logger

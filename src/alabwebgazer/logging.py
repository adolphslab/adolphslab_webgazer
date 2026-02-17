"""Structured logging and run metadata capture."""

from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from alabwebgazer.io import write_json


class JsonFormatter(logging.Formatter):
    """Minimal JSON formatter for reproducible logs."""

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


def _git_sha(cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip() or None


def _safe_pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_run_metadata(*, cwd: Path, random_seed: int) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "random_seed": random_seed,
        "git_sha": _git_sha(cwd),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "numpy": _safe_pkg_version("numpy"),
            "pandas": _safe_pkg_version("pandas"),
            "statsmodels": _safe_pkg_version("statsmodels"),
        },
    }


def configure_logging(*, log_path: Path, level: str = "INFO") -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("alabwebgazer")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(JsonFormatter())

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger


def persist_run_metadata(metadata_payload: dict[str, Any], metadata_path: Path) -> None:
    write_json(metadata_payload, metadata_path)

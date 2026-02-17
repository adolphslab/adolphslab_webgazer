"""I/O helpers for tabular artifacts and snapshots.

This module intentionally avoids project-specific assumptions: callers are expected
to provide validated config objects and explicit paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class IOErrorConfig(ValueError):
    """Raised on unsupported or missing paths."""


def resolve_path(base_dir: Path, path_like: Path) -> Path:
    """Resolve relative paths against base_dir."""
    if path_like.is_absolute():
        return path_like
    return (base_dir / path_like).resolve()


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise IOErrorConfig(f"Input table not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise IOErrorConfig(f"Unsupported table format: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    raise IOErrorConfig(f"Unsupported output table format: {path}")


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

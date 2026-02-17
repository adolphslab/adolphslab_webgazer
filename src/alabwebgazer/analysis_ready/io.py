"""I/O helpers (CSV).

Data policy:
- Never write raw sensitive data into logs.
- Writes are deterministic (sorted columns where appropriate) and atomic where feasible.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path, *, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=index)
    tmp.replace(path)


def maybe_read_csv(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    return read_csv(path)

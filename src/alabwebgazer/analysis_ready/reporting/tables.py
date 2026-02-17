"""Tabulation utilities for outputs (CSV).

Requirement anchors:
- SAP-IMPL-OUT-001: publication-grade tables with Δp, OR, p, q.
"""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


def bh_fdr(pvals: Iterable[float]) -> List[float]:
    p = np.asarray(list(pvals), dtype=float)
    # Treat NaN as NaN (no adjustment); multipletests can't handle NaN
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan, dtype=float)
    if mask.any():
        q[mask] = multipletests(p[mask], method="fdr_bh")[1]
    return list(q)


def ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

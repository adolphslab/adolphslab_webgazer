"""Data validation utilities.

These checks are intended to fail fast on schema/key/count issues.

Key requirements
- SAP-IMPL-VALID-001: required columns exist; key uniqueness; no silent leakage of excluded tasks.
- SAP-IMPL-TRIAL0-001: trial==0 rows must be dropped with explicit reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import pandas as pd

from .errors import DataValidationError


def require_columns(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise DataValidationError(f"{name}: missing required columns: {missing}")


def require_unique_keys(df: pd.DataFrame, keys: Sequence[str], *, name: str) -> None:
    require_columns(df, keys, name=name)
    dup = df.duplicated(subset=list(keys), keep=False)
    if dup.any():
        examples = df.loc[dup, list(keys)].head(5).to_dict(orient="records")
        raise DataValidationError(
            f"{name}: keys not unique on {list(keys)}; example duplicates: {examples}"
        )


def validate_binomial_counts(
    df: pd.DataFrame, *, successes_col: str, trials_col: str, name: str
) -> None:
    require_columns(df, [successes_col, trials_col], name=name)
    s = df[successes_col]
    t = df[trials_col]
    if (t < 0).any():
        raise DataValidationError(f"{name}: negative trials detected in {trials_col}.")
    # Allow NA rows (they may be filtered later)
    bad = (s.notna() & t.notna()) & ((s < 0) | (s > t))
    if bad.any():
        ex = df.loc[bad, [successes_col, trials_col]].head(5).to_dict(orient="records")
        raise DataValidationError(
            f"{name}: invalid successes/trials (require 0<=s<=t). Examples: {ex}"
        )


@dataclass(frozen=True)
class FilterReport:
    n_in: int
    n_out: int
    n_dropped: int
    drop_reasons: Dict[str, int]


def apply_common_filters(
    df: pd.DataFrame,
    *,
    name: str,
    tasks_keep: List[str],
    use_run_only: bool,
    groups_keep: List[str],
    sex_keep: List[str],
) -> Tuple[pd.DataFrame, FilterReport]:
    """Apply common confirmatory filters and return filtered df + audit counts."""
    n_in = len(df)
    drop_reasons: Dict[str, int] = {}

    # task filter
    if "task" in df.columns:
        mask = df["task"].isin(tasks_keep)
        drop_reasons["task_not_kept"] = int((~mask).sum())
        df = df.loc[mask].copy()

    # use_run gate
    if use_run_only and "use_run" in df.columns:
        mask = df["use_run"].astype(bool)
        drop_reasons["use_run_false"] = int((~mask).sum())
        df = df.loc[mask].copy()

    # group filter
    if "Group" in df.columns:
        mask = df["Group"].isin(groups_keep)
        drop_reasons["group_not_kept"] = int((~mask).sum())
        df = df.loc[mask].copy()

    # sex filter (explicitly drops missing/other)
    if "Sex" in df.columns:
        mask = df["Sex"].isin(sex_keep)
        drop_reasons["sex_not_kept"] = int((~mask).sum())
        df = df.loc[mask].copy()

    n_out = len(df)
    return df, FilterReport(n_in=n_in, n_out=n_out, n_dropped=n_in - n_out, drop_reasons=drop_reasons)


def drop_zero_trials(
    df: pd.DataFrame,
    *,
    trials_col: str,
    group_keys: Sequence[str],
    name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Drop rows where trials==0 and return (filtered_df, drop_counts_table)."""
    require_columns(df, [trials_col], name=name)
    zero = df[trials_col] == 0
    if not zero.any():
        return df, pd.DataFrame(columns=list(group_keys) + ["n_dropped_zero_trials"])

    require_columns(df, group_keys, name=name)
    drop_counts = (
        df.loc[zero, list(group_keys)]
        .value_counts(dropna=False)
        .rename("n_dropped_zero_trials")
        .reset_index()
    )
    df_f = df.loc[~zero].copy()
    return df_f, drop_counts

"""Tabular validation logic with explicit, early-failing checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from alabwebgazer.contracts import ColumnContract


class ValidationError(ValueError):
    """Raised when validation errors are present."""


@dataclass(frozen=True)
class ValidationIssue:
    level: str  # "error" or "warning"
    message: str


def _missing_columns(df: pd.DataFrame, required: Iterable[str]) -> list[str]:
    return [c for c in required if c not in df.columns]


def _check_non_negative(df: pd.DataFrame, cols: Iterable[str]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    for col in cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        if series.isna().any():
            issues.append(ValidationIssue("error", f"Column {col} contains non-numeric values"))
            continue
        if (series < 0).any():
            issues.append(ValidationIssue("error", f"Column {col} contains negative values"))
    return issues


def validate_contract(df: pd.DataFrame, contract: ColumnContract, table_name: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    missing = _missing_columns(df, contract.required)
    if missing:
        issues.append(
            ValidationIssue(
                "error",
                f"{table_name}: missing required columns: {', '.join(missing)}",
            )
        )
        return issues

    issues.extend(_check_non_negative(df, contract.non_negative))
    return issues


def validate_feature_table(
    df: pd.DataFrame,
    *,
    allowed_groups: tuple[str, ...],
    required_features: tuple[str, ...],
    id_col: str,
    video_col: str,
    feature_col: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if df.empty:
        issues.append(ValidationIssue("error", "feature table is empty"))
        return issues

    if "Group" in df.columns:
        observed_groups = set(df["Group"].dropna().astype(str).unique())
        unexpected = sorted(observed_groups - set(allowed_groups))
        if unexpected:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"Unexpected groups present: {', '.join(unexpected)}",
                )
            )

    if feature_col in df.columns:
        observed_features = set(df[feature_col].dropna().astype(str).unique())
        missing_features = sorted(set(required_features) - observed_features)
        if missing_features:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"Required features absent in this input: {', '.join(missing_features)}",
                )
            )

    denom_check_cols = {"dwell_bins", "n_present_any_valid", "n_y_valid"}
    if denom_check_cols.issubset(df.columns):
        dwell = pd.to_numeric(df["dwell_bins"], errors="coerce")
        n_present = pd.to_numeric(df["n_present_any_valid"], errors="coerce")
        n_valid = pd.to_numeric(df["n_y_valid"], errors="coerce")

        if (dwell > n_present).any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "Some rows have dwell_bins > n_present_any_valid",
                )
            )
        if (dwell > n_valid).any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "Some rows have dwell_bins > n_y_valid",
                )
            )

    key_cols = [c for c in (id_col, video_col, feature_col) if c in df.columns]
    if len(key_cols) == 3:
        n_dups = int(df.duplicated(subset=key_cols).sum())
        if n_dups > 0:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"Detected {n_dups} duplicate rows by key {key_cols}",
                )
            )

    return issues


def validate_run_table(
    df: pd.DataFrame,
    *,
    id_col: str,
    video_col: str,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if df.empty:
        issues.append(ValidationIssue("error", "run table is empty"))
        return issues

    if {"n_switches", "n_contig_steps"}.issubset(df.columns):
        n_switches = pd.to_numeric(df["n_switches"], errors="coerce")
        n_steps = pd.to_numeric(df["n_contig_steps"], errors="coerce")
        if (n_switches > n_steps).any():
            issues.append(
                ValidationIssue(
                    "warning",
                    "Some rows have n_switches > n_contig_steps",
                )
            )

    key_cols = [c for c in (id_col, video_col) if c in df.columns]
    if len(key_cols) == 2:
        n_dups = int(df.duplicated(subset=key_cols).sum())
        if n_dups > 0:
            issues.append(
                ValidationIssue(
                    "warning",
                    f"Detected {n_dups} duplicate rows by key {key_cols}",
                )
            )

    return issues


def raise_if_needed(issues: list[ValidationIssue], *, fail_on_warnings: bool) -> None:
    errors = [i for i in issues if i.level == "error"]
    warnings = [i for i in issues if i.level == "warning"]

    if errors:
        message = "\n".join(f"ERROR: {issue.message}" for issue in errors)
        raise ValidationError(message)

    if fail_on_warnings and warnings:
        message = "\n".join(f"WARNING: {issue.message}" for issue in warnings)
        raise ValidationError(message)

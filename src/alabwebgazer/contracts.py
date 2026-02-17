"""Data contracts for pipeline inputs and intermediate artifacts."""

from __future__ import annotations

from dataclasses import dataclass


# MS-017 / SM-006 / SAP-003
REQUIRED_FEATURE_COLUMNS: tuple[str, ...] = (
    "SID",
    "Video",
    "Group",
    "feature",
    "dwell_bins",
    "n_present_any_valid",
    "n_y_valid",
)

# MS-017 / SM-007 / SAP-003
REQUIRED_RUN_COLUMNS: tuple[str, ...] = (
    "SID",
    "Video",
    "Group",
    "n_switches",
    "n_contig_steps",
)

NON_NEGATIVE_COLUMNS: tuple[str, ...] = (
    "dwell_bins",
    "n_present_any_valid",
    "n_y_valid",
    "n_switches",
    "n_contig_steps",
)


@dataclass(frozen=True)
class ColumnContract:
    required: tuple[str, ...]
    non_negative: tuple[str, ...]


FEATURE_TABLE_CONTRACT = ColumnContract(
    required=REQUIRED_FEATURE_COLUMNS,
    non_negative=("dwell_bins", "n_present_any_valid", "n_y_valid"),
)

RUN_TABLE_CONTRACT = ColumnContract(
    required=REQUIRED_RUN_COLUMNS,
    non_negative=("n_switches", "n_contig_steps"),
)

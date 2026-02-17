"""Stable requirement identifiers for traceability.

These IDs are intended to be referenced in:
- step implementations (as metadata)
- run manifests
- tests (acceptance checks)

They describe *engineering* requirements (auditability, determinism, schema checks),
not scientific claims.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class Requirement:
    requirement_id: str
    description: str


REQUIREMENTS: Dict[str, Requirement] = {
    # RAW -> WINDOW
    "RAW-WINDOW-001": Requirement(
        "RAW-WINDOW-001",
        "Convert raw jsPsych/WebGazer exports into 0.5s-binned quadrant time series with config snapshots and QC summaries.",
    ),
    "RAW-WINDOW-002": Requirement(
        "RAW-WINDOW-002",
        "Record preprocessing parameters and diagnostics (exclusion rates, method used) per output file.",
    ),
    # WINDOW -> TABLES
    "WINDOW-TABLES-001": Requirement(
        "WINDOW-TABLES-001",
        "Compute per-run dwell/switching metrics from window CSVs + per-quadrant feature streams.",
    ),
    "WINDOW-TABLES-002": Requirement(
        "WINDOW-TABLES-002",
        "Build analysis-ready long and run-level tables (df_mergedmain.csv, df_runs.csv) with explicit filters and schema checks.",
    ),
    # TABLES -> INFERENCE (analysis_ready subpackage)
    "TABLES-INFER-001": Requirement(
        "TABLES-INFER-001",
        "Run analysis-ready modeling pipeline from analysis-ready CSVs with run manifests and reproducible artifacts.",
    ),
    # Cross-cutting
    "ENG-AUDIT-001": Requirement(
        "ENG-AUDIT-001",
        "Every run writes a manifest including config snapshot, environment metadata, and input hashes.",
    ),
    "ENG-PRIV-001": Requirement(
        "ENG-PRIV-001",
        "Do not write raw participant data to logs; avoid logging secrets; allow optional subject-id hashing.",
    ),
    "ORCH-E2E-001": Requirement(
        "ORCH-E2E-001",
        "Optional unified runner executes stage modules in one auditable run with deterministic stage subdirectories.",
    ),
}


def require(ids: Iterable[str]) -> List[Requirement]:
    out: List[Requirement] = []
    for rid in ids:
        if rid not in REQUIREMENTS:
            raise KeyError(f"Unknown requirement_id: {rid}")
        out.append(REQUIREMENTS[rid])
    return out

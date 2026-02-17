"""Ingest analysis-ready tables.

Requirement anchors:
- SAP-IMPL-INGEST-001: analysis consumes df_mergedmain.csv and df_runs.csv; keys unique.
"""

from __future__ import annotations

from typing import List

from .. import io
from ..logging import get_logger
from ..validation import require_unique_keys, require_columns
from .base import RunContext, Step

log = get_logger(__name__)


class IngestAnalysisReadyTablesStep(Step):
    step_id = "ingest_analysis_ready"
    requirement_ids = ("SAP-IMPL-INGEST-001", "SAP-IMPL-VALID-001")

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config
        stage = "00_ingest"

        log.info("Reading feature-long CSV: %s", cfg.paths.feature_long_csv)
        df_feat = io.read_csv(cfg.paths.feature_long_csv)

        log.info("Reading run-level CSV: %s", cfg.paths.run_level_csv)
        df_runs = io.read_csv(cfg.paths.run_level_csv)

        # Base schema validation
        require_unique_keys(df_feat, ["SID", "Video", "task", "feature"], name="df_mergedmain")
        require_unique_keys(df_runs, ["SID", "Video", "task"], name="df_runs")

        # Minimal required columns for filtering
        require_columns(df_feat, ["Group", "Sex"], name="df_mergedmain")
        require_columns(df_runs, ["Group", "Sex"], name="df_runs")

        # Required columns for all configured endpoints
        needed_feat: List[str] = []
        needed_runs: List[str] = []
        for ep_id, ep in cfg.confirmatory.endpoints.items():
            cols = [ep.successes_col, ep.trials_col] + [c for c in ep.predictors if ":" not in c]
            if ep.table == "feature_long":
                needed_feat.extend(cols)
                if ep.feature_filter:
                    needed_feat.append("feature")
                # NOTE: state_col may be derived from feature (e.g., averted interaction); do not require here.
            else:
                needed_runs.extend(cols)

        needed_feat = sorted(set([c for c in needed_feat if c not in ["{y}", "State"]]))
        needed_runs = sorted(set([c for c in needed_runs if c not in ["{y}"]]))

        require_columns(df_feat, needed_feat, name="df_mergedmain")
        require_columns(df_runs, needed_runs, name="df_runs")

        # Store in context (do not write full data by default; privacy)
        ctx.tables["feature_long"] = df_feat
        ctx.tables["run_level"] = df_runs

        # Write schema summaries (safe)
        ctx.artifacts.write_json(
            stage,
            "feature_long_schema.json",
            {"n_rows": len(df_feat), "columns": {c: str(t) for c, t in df_feat.dtypes.items()}},
        )
        ctx.artifacts.write_json(
            stage,
            "run_level_schema.json",
            {"n_rows": len(df_runs), "columns": {c: str(t) for c, t in df_runs.dtypes.items()}},
        )

        log.info("Ingest complete: feature_long=%d rows; run_level=%d rows", len(df_feat), len(df_runs))

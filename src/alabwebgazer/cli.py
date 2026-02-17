"""Command-line interface for alabwebgazer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional


def _parse_steps_csv(raw: Optional[str]) -> list[str] | None:
    if raw is None:
        return None
    values = [x.strip() for x in raw.split(",")]
    values = [x for x in values if x]
    return values if values else None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="adolphslab_webgazer")
    sub = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # Optional unified end-to-end orchestration
    # ------------------------------------------------------------------
    val_run = sub.add_parser("validate-run-config", help="validate unified end-to-end YAML config")
    val_run.add_argument("--config", required=True, help="Path to YAML config")

    run_all = sub.add_parser("run", help="run optional unified end-to-end pipeline")
    run_all.add_argument("--config", required=True, help="Path to YAML config")

    # ------------------------------------------------------------------
    # RAW -> WINDOW
    # ------------------------------------------------------------------
    val_prep = sub.add_parser(
        "validate-preprocess-config",
        help="validate preprocess-jspsych YAML config",
    )
    val_prep.add_argument("--config", required=True, help="Path to YAML config")

    run_prep = sub.add_parser(
        "preprocess-jspsych",
        help="run raw jsPsych/WebGazer -> window CSV preprocessing",
    )
    run_prep.add_argument("--config", required=True, help="Path to YAML config")

    # ------------------------------------------------------------------
    # WINDOW -> TABLES
    # ------------------------------------------------------------------
    val_tables = sub.add_parser(
        "validate-build-tables-config",
        help="validate build-tables YAML config",
    )
    val_tables.add_argument("--config", required=True, help="Path to YAML config")

    run_tables = sub.add_parser(
        "build-tables",
        help="run window -> analysis-ready table construction",
    )
    run_tables.add_argument("--config", required=True, help="Path to YAML config")

    # ------------------------------------------------------------------
    # TABLES -> INFERENCE (analysis_ready)
    # ------------------------------------------------------------------
    run_ar_cmd = sub.add_parser(
        "run-analysis-ready",
        help="run analysis-ready modeling pipeline (YAML config)",
    )
    run_ar_cmd.add_argument("--config", required=True, help="Path to YAML config")
    run_ar_cmd.add_argument(
        "--steps",
        required=False,
        help="Comma-separated subset of analysis-ready step IDs",
    )
    run_ar_cmd.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing run directory if it exists",
    )

    val_ar_cmd = sub.add_parser(
        "validate-analysis-ready-config",
        help="validate analysis-ready YAML config only",
    )
    val_ar_cmd.add_argument("--config", required=True, help="Path to YAML config")

    sub.add_parser("list-analysis-ready-steps", help="list available analysis-ready step IDs")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "validate-run-config":
        from alabwebgazer.core.config import ConfigError as RunConfigError
        from alabwebgazer.core.config import load_config as load_run_config

        config_path = Path(args.config)
        try:
            _ = load_run_config(config_path)
        except RunConfigError as exc:
            print(f"RUN_CONFIG_ERROR: {exc}")
            return 2
        print(f"Config valid: {config_path}")
        return 0

    if args.command == "run":
        from alabwebgazer.core.config import ConfigError as RunConfigError
        from alabwebgazer.core.config import load_config as load_run_config
        from alabwebgazer.core.pipeline import PipelineError as UnifiedPipelineError
        from alabwebgazer.core.pipeline import run_pipeline as run_unified_pipeline

        config_path = Path(args.config)
        try:
            cfg = load_run_config(config_path)
            run_dir = run_unified_pipeline(config_path=config_path, cfg=cfg)
        except (UnifiedPipelineError, RunConfigError) as exc:
            print(f"RUN_ERROR: {exc}")
            return 2
        except Exception as exc:  # noqa: BLE001
            print(f"RUN_ERROR: {exc}")
            return 2
        print(f"Unified run completed: {run_dir}")
        return 0

    if args.command == "validate-preprocess-config":
        from alabwebgazer.preprocessing.config import ConfigError, load_config

        config_path = Path(args.config)
        try:
            _ = load_config(config_path)
        except ConfigError as exc:
            print(f"PREPROCESS_CONFIG_ERROR: {exc}")
            return 2
        print(f"Config valid: {config_path}")
        return 0

    if args.command == "preprocess-jspsych":
        from alabwebgazer.preprocessing.batch import PipelineError, run_preprocess_pipeline

        config_path = Path(args.config)
        try:
            run_dir = run_preprocess_pipeline(config_path=config_path)
        except PipelineError as exc:
            print(f"PREPROCESS_ERROR: {exc}")
            return 2
        print(f"Preprocessing run completed: {run_dir}")
        return 0

    if args.command == "validate-build-tables-config":
        from alabwebgazer.derive_tables.config import ConfigError, load_config

        config_path = Path(args.config)
        try:
            _ = load_config(config_path)
        except ConfigError as exc:
            print(f"BUILD_TABLES_CONFIG_ERROR: {exc}")
            return 2
        print(f"Config valid: {config_path}")
        return 0

    if args.command == "build-tables":
        from alabwebgazer.derive_tables.pipeline import run_pipeline

        config_path = Path(args.config)
        try:
            run_dir = run_pipeline(config_path=config_path)
        except Exception as exc:  # noqa: BLE001
            print(f"BUILD_TABLES_ERROR: {exc}")
            return 2
        print(f"Build-tables run completed: {run_dir}")
        return 0

    if args.command == "list-analysis-ready-steps":
        from alabwebgazer.analysis_ready.steps.registry import STEP_REGISTRY

        for step_id in sorted(STEP_REGISTRY):
            print(step_id)
        return 0

    if args.command == "validate-analysis-ready-config":
        from alabwebgazer.analysis_ready.config import ConfigError as ArConfigError
        from alabwebgazer.analysis_ready.config import load_config as load_ar_config

        ar_config_path = Path(args.config)
        try:
            _ = load_ar_config(ar_config_path)
        except ArConfigError as exc:
            print(f"ANALYSIS_READY_CONFIG_ERROR: {exc}")
            return 2
        print(f"Analysis-ready config valid: {ar_config_path}")
        return 0

    if args.command == "run-analysis-ready":
        from alabwebgazer.analysis_ready.config import ConfigError as ArConfigError
        from alabwebgazer.analysis_ready.errors import PipelineError as ArPipelineError
        from alabwebgazer.analysis_ready.pipeline import run_pipeline as run_analysis_ready_pipeline

        ar_config_path = Path(args.config)
        step_ids = _parse_steps_csv(args.steps)
        try:
            run_dir = run_analysis_ready_pipeline(
                config_path=ar_config_path,
                steps=step_ids,
                overwrite=bool(args.overwrite),
            )
        except (ArPipelineError, ArConfigError) as exc:
            print(f"ANALYSIS_READY_ERROR: {exc}")
            return 2
        print(f"Analysis-ready run completed: {run_dir}")
        return 0

    parser.print_help()
    return 1

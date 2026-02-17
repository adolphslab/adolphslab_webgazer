"""Build analysis-ready tables from window CSVs and feature PKLs."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

from alabwebgazer.contracts import FEATURE_TABLE_CONTRACT, RUN_TABLE_CONTRACT
from alabwebgazer.core.run import (
    make_run_id,
    prepare_run_dirs,
    snapshot_config,
    write_metadata_and_inputs,
)
from alabwebgazer.io import write_json
from alabwebgazer.logging import configure_logging
from alabwebgazer.requirements import require
from alabwebgazer.validation import (
    validate_contract,
    validate_feature_table,
    validate_run_table,
)

from alabwebgazer.derive_tables.config import load_config
from alabwebgazer.derive_tables.legacy_adapters import (
    DEFAULT_GROUPS_KEEP,
    run_cluster_build,
    run_dwell_task,
)

TABLE_FILE_NAMES: tuple[str, ...] = (
    "df_mergedmain.csv",
    "df_runs.csv",
    "df_runlevel.csv",
    "df_feat_long.csv",
    "df_feat_wide.csv",
    "df_cluster.csv",
)

TABLE_SORT_KEYS: dict[str, tuple[str, ...]] = {
    "df_mergedmain.csv": ("SID", "task", "run_num", "feature", "Video"),
    "df_runs.csv": ("SID", "task", "run_num", "Video"),
    "df_runlevel.csv": ("SID",),
    "df_feat_long.csv": ("SID", "feature"),
    "df_feat_wide.csv": ("SID",),
    "df_cluster.csv": ("SID",),
}


def _resolve(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def _stable_sort_table(df: pd.DataFrame, *, table_name: str) -> pd.DataFrame:
    keys = [k for k in TABLE_SORT_KEYS.get(table_name, ()) if k in df.columns]
    if not keys:
        return df
    return df.sort_values(keys, kind="stable").reset_index(drop=True)


def _stabilize_table_csv(path: Path) -> None:
    if not path.exists():
        return
    df = pd.read_csv(path)
    stable = _stable_sort_table(df, table_name=path.name)
    stable.to_csv(path, index=False)


def _validate_output_tables(
    *,
    tables_dir: Path,
    allowed_groups: Sequence[str],
    required_features: Sequence[str],
) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {"errors": [], "warnings": []}

    mergedmain_path = tables_dir / "df_mergedmain.csv"
    if mergedmain_path.exists():
        df_m = pd.read_csv(mergedmain_path)
        issues_m = validate_contract(df_m, FEATURE_TABLE_CONTRACT, "df_mergedmain")
        issues_m.extend(
            validate_feature_table(
                df_m,
                allowed_groups=tuple(allowed_groups),
                required_features=tuple(required_features),
                id_col="SID",
                video_col="Video",
                feature_col="feature",
            )
        )
        report["errors"].extend([i.message for i in issues_m if i.level == "error"])
        report["warnings"].extend([i.message for i in issues_m if i.level == "warning"])

    runs_path = tables_dir / "df_runs.csv"
    if runs_path.exists():
        df_r = pd.read_csv(runs_path)
        issues_r = validate_contract(df_r, RUN_TABLE_CONTRACT, "df_runs")
        issues_r.extend(validate_run_table(df_r, id_col="SID", video_col="Video"))
        report["errors"].extend([i.message for i in issues_r if i.level == "error"])
        report["warnings"].extend([i.message for i in issues_r if i.level == "warning"])

    return report


def run_pipeline(*, config_path: Path) -> Path:
    cfg = load_config(config_path)
    config_dir = config_path.parent.resolve()

    output_root = _resolve(config_dir, cfg.paths.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = cfg.run.run_id or make_run_id(prefix="tables", seed=cfg.run.random_seed)
    paths = prepare_run_dirs(output_root=output_root, run_id=run_id, overwrite=cfg.run.overwrite)

    logger = configure_logging(
        log_path=paths.logs_dir / "build_tables.jsonl",
        level=str(cfg.run.log_level),
    )
    logger.info("build_tables_start")

    config_text = snapshot_config(config_path=config_path, out_dir=paths.artifacts_dir)

    # Input hash capture: configs + metadata CSVs (if any).
    # We do not hash entire data directories by default.
    input_files: List[Path] = [config_path]
    for t in cfg.dwell_tasks:
        if t.subject_metadata_csv:
            input_files.append(_resolve(config_dir, t.subject_metadata_csv))
        if t.runlist_csv:
            input_files.append(_resolve(config_dir, t.runlist_csv))

    write_metadata_and_inputs(
        paths=paths,
        repo_root=Path(__file__).resolve().parents[3],
        run_id=run_id,
        random_seed=cfg.run.random_seed,
        config_text=config_text,
        input_paths=input_files,
        packages=(
            "adolphslab-webgazer",
            "numpy",
            "pandas",
            "scipy",
            "statsmodels",
            "scikit-learn",
            "tqdm",
        ),
    )

    # ---------------------------------------------------------------------
    # Step 1: compute dwell metrics (via legacy adapters)
    # ---------------------------------------------------------------------
    dwell_out_root = paths.artifacts_dir / "dwelltime"
    dwell_out_root.mkdir(parents=True, exist_ok=True)

    dwell_outputs: Dict[str, Path] = {}
    for task_cfg in cfg.dwell_tasks:
        task = task_cfg.task
        out_dir = dwell_out_root / task
        out_dir.mkdir(parents=True, exist_ok=True)

        master = run_dwell_task(
            config_dir=config_dir,
            task_cfg=task_cfg,
            dwell_options=cfg.dwell_options,
            random_seed=cfg.run.random_seed,
            out_dir=out_dir,
            logger=logger,
        )
        dwell_outputs[task] = master

    write_json(
        {k: str(v) for k, v in dwell_outputs.items()},
        paths.artifacts_dir / "dwell_outputs.json",
    )

    # ---------------------------------------------------------------------
    # Step 2: build analysis-ready tables (via legacy adapters)
    # ---------------------------------------------------------------------
    analysis_ready_dir = paths.artifacts_dir / "analysis_ready"
    if cfg.cluster.enabled:
        analysis_ready_dir.mkdir(parents=True, exist_ok=True)

        run_cluster_build(
            cluster_options=cfg.cluster,
            dwell_outputs=dwell_outputs,
            out_dir=analysis_ready_dir,
            logger=logger,
        )

        # Stabilize and copy tables into canonical tables/ directory.
        for name in TABLE_FILE_NAMES:
            p = analysis_ready_dir / name
            if p.exists():
                _stabilize_table_csv(p)
                shutil.copy2(p, paths.tables_dir / name)

        validation_report = _validate_output_tables(
            tables_dir=paths.tables_dir,
            allowed_groups=(
                tuple(cfg.cluster.groups_keep)
                if cfg.cluster.groups_keep
                else DEFAULT_GROUPS_KEEP
            ),
            required_features=(
                tuple(cfg.cluster.features_keep)
                if cfg.cluster.features_keep
                else ()
            ),
        )
        write_json(validation_report, paths.artifacts_dir / "table_validation.json")

    # Manifest
    require(["WINDOW-TABLES-001", "WINDOW-TABLES-002", "ENG-AUDIT-001"])
    manifest = {
        "run_id": run_id,
        "requirements": ["WINDOW-TABLES-001", "WINDOW-TABLES-002", "ENG-AUDIT-001"],
        "dwell_outputs": {
            k: (
                str(Path(v).relative_to(paths.run_dir))
                if Path(v).is_relative_to(paths.run_dir)
                else str(v)
            )
            for k, v in dwell_outputs.items()
        },
        "analysis_ready_dir": (
            str(analysis_ready_dir.relative_to(paths.run_dir))
            if cfg.cluster.enabled
            else None
        ),
        "tables_dir": str(paths.tables_dir.relative_to(paths.run_dir)),
    }
    write_json(manifest, paths.run_dir / "manifest.json")

    logger.info("build_tables_done")
    return paths.run_dir

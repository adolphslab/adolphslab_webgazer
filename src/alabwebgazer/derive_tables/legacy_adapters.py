"""Adapters for invoking vendored legacy dwell/clustering implementations.

The legacy modules remain the single source of algorithmic truth:
- `alabwebgazer.legacy.compute_dwelltime`
- `alabwebgazer.legacy.clustering_dataprep`

This adapter module centralizes how derive_tables interacts with them so the
rest of the codebase does not depend on legacy module internals.
"""

from __future__ import annotations

import io
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from alabwebgazer.io import write_json

from alabwebgazer.derive_tables.config import (
    ClusterOptionsConfig,
    DwellOptionsConfig,
    DwellTaskConfig,
)

DEFAULT_GROUPS_KEEP: tuple[str, ...] = ("ASD", "SPARK", "Control", "NonASD_Psych")


def _resolve(base: Path, p: Path) -> Path:
    return p if p.is_absolute() else (base / p).resolve()


def _legacy_dwell_tag(cfg: Any) -> str:
    """Build deterministic compute_dwelltime output tag."""
    tag = "dwell_times"
    if int(cfg.min_run) > 1:
        tag += f"_mrun-{int(cfg.min_run)}"
    if cfg.bin_seconds is not None:
        tag += f"_binsz-{float(cfg.bin_seconds):g}s"
    tag += (
        f"_bdry-{cfg.boundary_mode}-t{float(cfg.boundary_tau):g}-p{float(cfg.boundary_power):g}"
    )
    if bool(cfg.include_joint_features):
        tag += "_jointfeat"
    return tag


def run_dwell_task(
    *,
    config_dir: Path,
    task_cfg: DwellTaskConfig,
    dwell_options: DwellOptionsConfig,
    random_seed: int,
    out_dir: Path,
    logger: Optional[object] = None,
) -> Path:
    """Run legacy `compute_dwelltime` for one task and return its master CSV."""
    from alabwebgazer.legacy import compute_dwelltime as dwell

    task = task_cfg.task
    legacy_cfg = dwell.RunConfig(
        video_feats_dir=str(_resolve(config_dir, task_cfg.video_feats_dir)),
        window_data_dir=str(_resolve(config_dir, task_cfg.window_data_dir)),
        output_root=str(out_dir),
        task=str(task),
        window_glob=str(task_cfg.window_glob),
        subject_metadata_csv=(
            str(_resolve(config_dir, task_cfg.subject_metadata_csv))
            if task_cfg.subject_metadata_csv
            else None
        ),
        runlist_csv=(
            str(_resolve(config_dir, task_cfg.runlist_csv))
            if task_cfg.runlist_csv
            else None
        ),
        presence_threshold=float(dwell_options.presence_threshold),
        min_run=int(dwell_options.min_run),
        bin_seconds=(
            float(dwell_options.bin_seconds)
            if dwell_options.bin_seconds is not None
            else None
        ),
        include_joint_features=bool(dwell_options.include_joint_features),
        derived_require_speaker_present=bool(dwell_options.derived_require_speaker_present),
        derived_require_speaker_unique=bool(dwell_options.derived_require_speaker_unique),
        task5_require_speaker_unique=bool(dwell_options.task5_require_speaker_unique),
        include_speaker_not_averted_loose=bool(dwell_options.include_speaker_not_averted_loose),
        boundary_mode=str(dwell_options.boundary_mode),
        boundary_tau=float(dwell_options.boundary_tau),
        boundary_power=float(dwell_options.boundary_power),
        save_bin_level=bool(dwell_options.save_bin_level),
        random_state=int(random_seed),
    )
    write_json(asdict(legacy_cfg), out_dir / "legacy_run_config.json")

    if logger is not None:
        logger.info(f"compute_dwelltime_start:{task}")
    dwell.main(legacy_cfg)
    if logger is not None:
        logger.info(f"compute_dwelltime_done:{task}")

    expected_csv = out_dir / f"{task}_{_legacy_dwell_tag(legacy_cfg)}.csv"
    if expected_csv.exists():
        return expected_csv

    produced_csvs = sorted(p for p in out_dir.glob(f"{task}_*.csv") if p.suffix.lower() == ".csv")
    if not produced_csvs:
        raise RuntimeError(
            f"No dwell output CSV produced for task={task} in {out_dir}. "
            f"Expected: {expected_csv.name}"
        )
    if len(produced_csvs) == 1:
        return produced_csvs[0]

    raise RuntimeError(
        "Ambiguous dwell outputs; deterministic selection failed. "
        f"Expected: {expected_csv.name}; found: {[p.name for p in produced_csvs]}"
    )


def run_cluster_build(
    *,
    cluster_options: ClusterOptionsConfig,
    dwell_outputs: Dict[str, Path],
    out_dir: Path,
    logger: Optional[object] = None,
) -> None:
    """Run legacy clustering_dataprep over selected dwell outputs."""
    from alabwebgazer.legacy import clustering_dataprep as cluster

    pro_task = cluster_options.pro_task or "task5_pro"
    spark_task = cluster_options.spark_task or "task5_spark"

    if pro_task not in dwell_outputs or spark_task not in dwell_outputs:
        raise RuntimeError(
            "cluster.enabled=true but required dwell outputs missing. "
            f"Have: {sorted(dwell_outputs)}; "
            f"need pro_task={pro_task!r} and spark_task={spark_task!r}"
        )

    legacy_cluster_cfg = cluster.BuildConfig(
        pro_csv=dwell_outputs[pro_task],
        spark_csv=dwell_outputs[spark_task],
        out_dir=out_dir,
        use_run_only=bool(cluster_options.use_run_only),
        groups_keep=(
            tuple(cluster_options.groups_keep)
            if cluster_options.groups_keep
            else DEFAULT_GROUPS_KEEP
        ),
        task_prefixes_keep=(
            tuple(cluster_options.task_prefixes_keep)
            if cluster_options.task_prefixes_keep
            else None
        ),
        features_keep=(
            tuple(cluster_options.features_keep)
            if cluster_options.features_keep
            else None
        ),
        default_bin_seconds=float(cluster_options.default_bin_seconds),
        force_group_by_cohort=dict(cluster_options.force_group_by_cohort),
        add_stability_block=bool(cluster_options.add_stability_block),
        keep_support_counts=bool(cluster_options.keep_support_counts),
        write_spearman_corr=bool(cluster_options.write_spearman_corr),
    )
    write_json(asdict(legacy_cluster_cfg), out_dir / "legacy_cluster_config.json")

    if logger is not None:
        logger.info("clustering_dataprep_start")
    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        cluster.build_df_cluster(legacy_cluster_cfg)
    legacy_stream = buf.getvalue().strip()
    if legacy_stream:
        (out_dir / "legacy_cluster_stdout.log").write_text(legacy_stream + "\n", encoding="utf-8")
        if logger is not None:
            for line in legacy_stream.splitlines():
                logger.info("legacy_cluster:%s", line)
    if logger is not None:
        logger.info("clustering_dataprep_done")

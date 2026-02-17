"""YAML config for building analysis-ready tables from window CSVs and feature PKLs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigError(RuntimeError):
    """Raised when the build-tables YAML config is invalid."""


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    run_id: Optional[str] = None
    random_seed: int = Field(0, ge=0)
    log_level: str = "INFO"
    overwrite: bool = False


class PathsConfig(StrictBaseModel):
    output_root: Path = Path("runs_tables")


class DwellTaskConfig(StrictBaseModel):
    task: str = Field(..., description="e.g., task5_pro / task5_spark / task5_lab / task5_tobii")
    window_data_dir: Path
    video_feats_dir: Path
    window_glob: str = "videoview*.csv"
    subject_metadata_csv: Optional[Path] = None
    runlist_csv: Optional[Path] = None


class DwellOptionsConfig(StrictBaseModel):
    presence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    min_run: int = Field(1, ge=1)
    bin_seconds: Optional[float] = None

    include_joint_features: bool = True
    derived_require_speaker_present: bool = True
    derived_require_speaker_unique: bool = False
    task5_require_speaker_unique: bool = True
    include_speaker_not_averted_loose: bool = False

    boundary_mode: Literal["posthoc", "off"] = "posthoc"
    boundary_tau: float = Field(0.05, gt=0)
    boundary_power: float = Field(1.0, gt=0)

    save_bin_level: bool = False


class ClusterOptionsConfig(StrictBaseModel):
    enabled: bool = True
    # If omitted, we attempt to use dwell outputs for tasks named task5_pro and task5_spark.
    pro_task: Optional[str] = "task5_pro"
    spark_task: Optional[str] = "task5_spark"

    # Filters and output variants (subset of legacy BuildConfig options)
    use_run_only: bool = True
    task_prefixes_keep: Optional[List[str]] = Field(default_factory=lambda: ["task5_"])
    groups_keep: Optional[List[str]] = None
    features_keep: Optional[List[str]] = None
    default_bin_seconds: float = 0.5
    add_stability_block: bool = True
    keep_support_counts: bool = True
    write_spearman_corr: bool = True

    # Force group label by cohort (e.g., keep SPARK separate)
    force_group_by_cohort: Dict[str, str] = Field(default_factory=lambda: {"SPARK": "SPARK"})


class AppConfig(StrictBaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    dwell_tasks: List[DwellTaskConfig]
    dwell_options: DwellOptionsConfig = Field(default_factory=DwellOptionsConfig)
    cluster: ClusterOptionsConfig = Field(default_factory=ClusterOptionsConfig)

    @model_validator(mode="after")
    def _check_tasks(self) -> "AppConfig":
        if not self.dwell_tasks:
            raise ValueError("dwell_tasks must be non-empty")
        return self


def load_config(path: Path) -> AppConfig:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to read config YAML: {path}: {exc}") from exc
    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid build-tables config: {path}: {exc}") from exc

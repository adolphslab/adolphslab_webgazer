"""YAML config for optional unified end-to-end orchestration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigError(RuntimeError):
    """Raised when the unified pipeline YAML config is invalid."""


StepId = Literal["preprocess_jspsych", "build_tables", "analysis_ready"]


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    output_root: Path = Field(
        Path("runs_pipeline"),
        description="Root directory under which unified run directories are written.",
    )
    run_id: Optional[str] = Field(
        default=None,
        description="Optional explicit run id; if omitted, a timestamped id is generated.",
    )
    random_seed: int = Field(0, ge=0, description="Master seed for deterministic run-id suffixes.")
    overwrite: bool = Field(
        False,
        description="If true, overwrite an existing unified run directory.",
    )
    log_level: str = Field("INFO", description="Pipeline-level log level.")
    notes: str = Field("", description="Free-text notes (no secrets).")


class PipelineConfig(StrictBaseModel):
    steps: List[StepId] = Field(
        default_factory=lambda: ["preprocess_jspsych", "build_tables", "analysis_ready"],
        description="Ordered list of stage ids to execute.",
    )

    @model_validator(mode="after")
    def _validate_steps(self) -> "PipelineConfig":
        if not self.steps:
            raise ValueError("pipeline.steps must be non-empty.")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("pipeline.steps must not contain duplicates.")
        return self


class PreprocessStageConfig(StrictBaseModel):
    enabled: bool = True
    config_path: Path = Field(..., description="Path to preprocess-jspsych YAML config.")


class BuildTablesStageConfig(StrictBaseModel):
    enabled: bool = True
    config_path: Path = Field(..., description="Path to build-tables YAML config.")
    use_preprocess_windows: bool = Field(
        True,
        description="If true, patch dwell_tasks[*].window_data_dir to Stage 1 windows output.",
    )


class AnalysisReadyStageConfig(StrictBaseModel):
    enabled: bool = True
    config_path: Path = Field(..., description="Path to analysis-ready YAML config.")
    use_build_tables_outputs: bool = Field(
        True,
        description="If true, patch paths.feature_long_csv/run_level_csv to Stage 2 outputs.",
    )
    steps: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of analysis-ready step ids.",
    )


class AppConfig(StrictBaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    preprocess_jspsych: PreprocessStageConfig
    build_tables: BuildTablesStageConfig
    analysis_ready: AnalysisReadyStageConfig


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env_vars(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env_vars(v) for v in obj]
    if isinstance(obj, str):

        def repl(match: re.Match[str]) -> str:
            var = match.group(1)
            return os.environ.get(var, match.group(0))

        return _ENV_VAR_PATTERN.sub(repl, obj)
    return obj


def load_config(path: Path) -> AppConfig:
    """Load and validate a unified end-to-end YAML config."""
    try:
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to read run config YAML: {path}: {exc}") from exc

    raw = _interpolate_env_vars(raw)

    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid run config: {path}: {exc}") from exc

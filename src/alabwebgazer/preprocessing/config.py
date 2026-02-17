"""Configuration for batch preprocessing of raw jsPsych/WebGazer exports.

Primary goal: reproducibility + auditability for raw preprocessing runs.

This config is intentionally conservative:
- All study-specific mapping (stimulus -> condition + video length) is expressed
  via `condition_rules`.
- Output naming is configurable, but defaults are compatible with downstream dwell-time scripts.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigError(RuntimeError):
    """Raised when a preprocessing YAML config is invalid."""


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    run_id: Optional[str] = Field(
        default=None,
        description="Optional run id. If omitted, the runner will generate one.",
    )
    random_seed: int = Field(0, ge=0, description="Random seed for deterministic components.")
    log_level: str = Field("INFO", description="Logging level (INFO, WARNING, ...).")
    overwrite: bool = Field(False, description="If True, overwrite an existing run directory.")
    notes: str = Field("", description="Free-text notes (no secrets).")


class PathsConfig(StrictBaseModel):
    input_glob: str = Field(
        ..., description="Glob for jsPsych JSON exports (relative to config file by default)."
    )
    output_root: Path = Field(
        Path("runs_preprocess"), description="Root directory under which run outputs are written."
    )
    windows_subdir: str = Field(
        "windows", description="Subdirectory within the run dir for videoview_window_*.csv outputs."
    )
    summary_csv: str = Field(
        "preprocess_summary.csv",
        description="Filename (within run dir) for the preprocessing summary table.",
    )


class FilenameParsingConfig(StrictBaseModel):
    """How to derive participant/session identifiers from input filenames.

    Prefer regex parsing when file naming is inconsistent.
    """

    delimiter: str = "_"

    # Split-based parsing
    id_index: int = Field(1, ge=0)
    session_start: int = Field(2, ge=0)
    session_end: int = Field(7, ge=0)

    # Regex overrides (capture group 1)
    id_regex: Optional[str] = None
    session_regex: Optional[str] = None


class TrialSelectionConfig(StrictBaseModel):
    trial_type: str = Field(
        "video-keyboard-response",
        description="Trial type label for video trials.",
    )
    require_webgazer_data: bool = True

    # Optional fallbacks if trials do not record screen dimensions.
    default_inner_width: Optional[float] = None
    default_inner_height: Optional[float] = None


class ConditionRule(StrictBaseModel):
    """Rule for mapping a trial to (condition, video length)."""

    pattern: str
    condition: str
    video_length_seconds: float = Field(..., gt=0)

    match_field: Literal["stimname", "stimulus0", "stimulus"] = "stimname"
    case_sensitive: bool = True
    use_regex: bool = False


class PreprocessParamsConfig(StrictBaseModel):
    median_kernel: int = Field(7, ge=1)
    median_method: Literal["scipy_medfilt", "rolling"] = "scipy_medfilt"

    x_prob_threshold: float = Field(0.9, ge=0.0, le=1.0)
    y_prob_threshold: float = Field(0.7, ge=0.0, le=1.0)

    bin_seconds: float = Field(0.5, gt=0.0)
    drift_mode: Literal["mean", "median", "none"] = "mean"
    # Deprecated compatibility knob. If provided, it overrides drift_mode:
    # True -> "mean", False -> "none".
    apply_drift_correction: Optional[bool] = None

    quadrant_method: Literal["gmm_axis", "midpoint"] = "gmm_axis"
    allow_midpoint_fallback: bool = False

    random_state: int = Field(0, ge=0)

    # Resampling parity controls
    resample_edge_policy: Literal["notebook_strict", "half_open"] = "notebook_strict"
    resample_include_endpoint: bool = True

    @model_validator(mode="after")
    def _normalize_legacy_drift_flag(self) -> "PreprocessParamsConfig":
        if self.apply_drift_correction is not None:
            self.drift_mode = "mean" if self.apply_drift_correction else "none"
        return self


class OutputNamingConfig(StrictBaseModel):
    """Output naming convention.

    Default is compatible with `compute_dwelltime.py::parse_subject_and_condition` when the
    session string does not contain underscores.
    Supported template fields: {id}, {session}, {condition}, {trial_index}.
    """

    template: str = "videoview_window_{id}_{session}_{condition}.csv"
    add_trial_index_suffix: bool = False


class PrivacyConfig(StrictBaseModel):
    """Privacy/safety knobs.

    Hashing subject IDs is OFF by default because it can break downstream joins.
    """

    hash_subject_ids: bool = False
    hash_salt_env: str = "ALABWEBGAZER_HASH_SALT"


class QuadrantImageQCConfig(StrictBaseModel):
    """Optional quadrant-image QC detection patterns."""

    enabled: bool = False
    q1_pattern: str = "top:25%; left:25%"
    q2_pattern: str = "top:25%; right:25%"
    q3_pattern: str = "bottom:25%; left:25%"
    q4_pattern: str = "bottom:25%; right:25%"


class AppConfig(StrictBaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    paths: PathsConfig
    filename_parsing: FilenameParsingConfig = Field(default_factory=FilenameParsingConfig)
    trials: TrialSelectionConfig = Field(default_factory=TrialSelectionConfig)
    condition_rules: List[ConditionRule] = Field(default_factory=list)
    preprocess: PreprocessParamsConfig = Field(default_factory=PreprocessParamsConfig)
    output_naming: OutputNamingConfig = Field(default_factory=OutputNamingConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    quadrant_image_qc: QuadrantImageQCConfig = Field(default_factory=QuadrantImageQCConfig)

    @model_validator(mode="after")
    def _check_rules_nonempty(self) -> "AppConfig":
        if not self.condition_rules:
            raise ValueError(
                "condition_rules must be non-empty (cannot infer video length otherwise)."
            )
        return self


_ENV_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate_env_vars(obj: Any) -> Any:
    """Recursively expand ${VAR} in strings within a YAML-loaded object."""
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
    """Load YAML config and validate with pydantic."""
    try:
        raw: Dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Failed to read config YAML: {path}: {exc}") from exc

    raw = _interpolate_env_vars(raw)

    try:
        return AppConfig.model_validate(raw)
    except Exception as exc:  # noqa: BLE001
        raise ConfigError(f"Invalid preprocess config: {path}: {exc}") from exc

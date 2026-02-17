"""Configuration models and loading utilities.

Key requirements
- SAP-IMPL-INGEST-001: config-driven execution with auditable snapshots.
- SAP-IMPL-CONF-001: explicit filter contracts for task/group/video in confirmatory runs.
- REV-REPRO-001: record config + run metadata for every run.

The config file format is YAML.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .errors import ConfigError


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunConfig(StrictBaseModel):
    run_id: str = Field(..., description="Unique run identifier (used for output directory).")
    random_seed: int = Field(0, ge=0, description="Master random seed for deterministic runs.")
    sap_name: str = Field(
        "SAP_implemented_all",
        description="Human label of SAP used for this run.",
    )
    sap_version: str = Field(
        "TODO",
        description="Version string for SAP (commit hash, date, etc.).",
    )
    notes: str = Field("", description="Free-text run notes (no secrets).")


class PathsConfig(StrictBaseModel):
    feature_long_csv: Path = Field(
        ...,
        description="Path to df_mergedmain.csv (feature-long table).",
    )
    run_level_csv: Path = Field(..., description="Path to df_runs.csv (run-level table).")
    output_root: Path = Field(Path("runs"), description="Directory to write run outputs under.")


class FiltersConfig(StrictBaseModel):
    tasks_keep: List[str] = Field(
        ...,
        description="Explicit keep-list for task values in confirmatory analyses.",
    )
    use_run_only: bool = True
    groups_keep: List[str] = Field(
        ...,
        description="Explicit keep-list for Group values in confirmatory analyses.",
    )
    sex_keep: List[str] = Field(default_factory=lambda: ["Female", "Male"])
    design_videos: List[str] = Field(
        ...,
        description="Explicit design-video list used for standardization and baselines.",
    )

    @staticmethod
    def _clean_unique_nonempty(values: List[str], *, field_name: str) -> List[str]:
        cleaned = [str(v).strip() for v in values]
        if not cleaned:
            raise ValueError(f"filters.{field_name} must be non-empty.")
        if any(v == "" for v in cleaned):
            raise ValueError(f"filters.{field_name} cannot contain blank values.")
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"filters.{field_name} must not contain duplicates.")
        return cleaned

    @model_validator(mode="after")
    def _validate_lists(self) -> "FiltersConfig":
        self.tasks_keep = self._clean_unique_nonempty(
            self.tasks_keep,
            field_name="tasks_keep",
        )
        self.groups_keep = self._clean_unique_nonempty(
            self.groups_keep,
            field_name="groups_keep",
        )
        self.sex_keep = self._clean_unique_nonempty(
            self.sex_keep,
            field_name="sex_keep",
        )
        self.design_videos = self._clean_unique_nonempty(
            self.design_videos,
            field_name="design_videos",
        )
        return self


class EndpointConfig(StrictBaseModel):
    table: Literal["feature_long", "run_level"]
    feature_filter: Optional[List[str]] = None
    state_col: Optional[str] = None
    successes_col: str
    trials_col: str
    predictors: List[str]

    @model_validator(mode="after")
    def _validate_endpoint_contract(self) -> "EndpointConfig":
        if not str(self.successes_col).strip():
            raise ValueError("successes_col must be a non-empty string.")
        if not str(self.trials_col).strip():
            raise ValueError("trials_col must be a non-empty string.")
        if not self.predictors:
            raise ValueError("predictors must be non-empty.")

        if self.table == "feature_long":
            if not self.feature_filter:
                raise ValueError(
                    "feature_long endpoints must set non-empty feature_filter explicitly."
                )
        else:
            if self.feature_filter is not None:
                raise ValueError(
                    "run_level endpoints must not set feature_filter."
                )
        return self


class ConfirmatoryConfig(StrictBaseModel):
    contrasts: List[Tuple[str, str]] = Field(
        default_factory=lambda: [("Control", "ASD"), ("Control", "SPARK"), ("ASD", "SPARK")]
    )
    endpoints: Dict[str, EndpointConfig]

    @model_validator(mode="after")
    def _check_endpoints(self) -> "ConfirmatoryConfig":
        if not self.endpoints:
            raise ValueError("confirmatory.endpoints must be non-empty.")
        if not self.contrasts:
            raise ValueError("confirmatory.contrasts must be non-empty.")
        for a, b in self.contrasts:
            if str(a).strip() == "" or str(b).strip() == "":
                raise ValueError("confirmatory.contrasts cannot contain blank group labels.")
            if a == b:
                raise ValueError(
                    "confirmatory.contrasts contains invalid self-contrast: "
                    f"({a}, {b})"
                )
        return self


class BootstrapConfig(StrictBaseModel):
    enabled: bool = False
    n_resamples: int = Field(500, ge=100)
    seed: int = 0


class ModelsConfig(StrictBaseModel):
    primary_backend: Literal["glm_cluster", "gee_exchangeable"] = "glm_cluster"
    cluster_col: str = "SID"
    bootstrap: BootstrapConfig = Field(default_factory=BootstrapConfig)


class MultiplicityConfig(StrictBaseModel):
    method: Literal["bh_fdr"] = "bh_fdr"


class AppConfig(StrictBaseModel):
    run: RunConfig
    paths: PathsConfig
    filters: FiltersConfig
    confirmatory: ConfirmatoryConfig
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    multiplicity: MultiplicityConfig = Field(default_factory=MultiplicityConfig)

    @model_validator(mode="after")
    def _check_paths_exist(self) -> "AppConfig":
        # Paths may not exist in CI/tests; allow opt-out via env var
        if os.environ.get("ALABWEBGAZER_SKIP_INPUT_EXISTS_CHECK") == "1":
            return self
        for p in [self.paths.feature_long_csv, self.paths.run_level_csv]:
            if not p.exists():
                raise ValueError(f"Input path does not exist: {p}")
        return self

    @model_validator(mode="after")
    def _check_filter_contract(self) -> "AppConfig":
        groups_keep = set(self.filters.groups_keep)
        contrast_groups = {g for pair in self.confirmatory.contrasts for g in pair}
        missing = sorted(contrast_groups - groups_keep)
        if missing:
            raise ValueError(
                "All confirmatory contrast groups must be present in filters.groups_keep. "
                f"Missing: {missing}"
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


def _resolve_path_like(base_dir: Path, value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, str):
        return value
    if "${" in value:
        # Keep unresolved placeholders unchanged.
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def load_config(path: Path) -> AppConfig:
    """Load YAML config and validate with pydantic."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Failed to read config YAML: {path}: {e}") from e

    raw = _interpolate_env_vars(raw)
    config_dir = path.parent.resolve()

    if isinstance(raw, dict):
        paths = raw.get("paths")
        if isinstance(paths, dict):
            for key in ("feature_long_csv", "run_level_csv", "output_root"):
                if key in paths:
                    paths[key] = _resolve_path_like(config_dir, paths[key])

    # Convert contrast lists to tuples if provided as lists
    if "confirmatory" in raw and "contrasts" in raw["confirmatory"]:
        raw["confirmatory"]["contrasts"] = [tuple(x) for x in raw["confirmatory"]["contrasts"]]

    try:
        return AppConfig.model_validate(raw)
    except Exception as e:  # noqa: BLE001
        raise ConfigError(f"Invalid config: {path}: {e}") from e


@dataclass(frozen=True)
class ConfigSnapshot:
    """Serializable config snapshot written into each run directory."""

    config_path: Path
    raw_text: str

    @classmethod
    def from_file(cls, path: Path) -> "ConfigSnapshot":
        return cls(config_path=path, raw_text=path.read_text(encoding="utf-8"))

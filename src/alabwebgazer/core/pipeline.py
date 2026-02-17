"""Optional unified end-to-end orchestration.

This runner executes the existing stage pipelines in sequence:
1) preprocess-jspsych
2) build-tables
3) run-analysis-ready

It does not re-implement scientific logic. It only patches stage configs so each
stage writes to deterministic subdirectories inside one unified run directory.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from alabwebgazer.core.config import AppConfig
from alabwebgazer.core.hashing import sha256_file, sha256_text
from alabwebgazer.core.provenance import build_run_metadata
from alabwebgazer.core.run import make_run_id
from alabwebgazer.io import write_json, write_text


class PipelineError(RuntimeError):
    """Raised when unified orchestration fails."""


PREPROCESS_STEP_ID = "preprocess_jspsych"
BUILD_TABLES_STEP_ID = "build_tables"
ANALYSIS_READY_STEP_ID = "analysis_ready"

PREPROCESS_STAGE_RUN_ID = "01_preprocess_windows"
BUILD_TABLES_STAGE_RUN_ID = "02_build_tables"
ANALYSIS_READY_STAGE_RUN_ID = "03_analysis_ready"


def _now_utc_iso() -> str:
    return datetime.now(UTC).isoformat()


def _resolve(base_dir: Path, path_like: Path) -> Path:
    if path_like.is_absolute():
        return path_like
    return (base_dir / path_like).resolve()


def _relative_or_abs(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _log_event(log_path: Path, *, level: str, msg: str, **fields: Any) -> None:
    payload: Dict[str, Any] = {
        "ts": _now_utc_iso(),
        "level": level,
        "msg": msg,
    }
    payload.update(fields)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True, default=str) + "\n")


def _load_yaml_map(path: Path) -> Dict[str, Any]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise PipelineError(f"Failed to read YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise PipelineError(f"YAML must be a mapping: {path}")
    return raw


def _write_yaml_map(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _as_abs_path_str(base_dir: Path, value: Any) -> Any:
    if value is None:
        return None
    text = str(value)
    if "${" in text:
        # Preserve env-var interpolation placeholders for stage-level loaders.
        return text
    p = Path(text)
    if p.is_absolute():
        return str(p)
    return str((base_dir / p).resolve())


def _collect_stage_requirements(stage_run_dir: Path) -> list[str]:
    out: set[str] = set()
    manifest = stage_run_dir / "manifest.json"
    if manifest.exists():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            reqs = payload.get("requirements", [])
            if isinstance(reqs, list):
                out.update([str(x) for x in reqs])
        except Exception:  # noqa: BLE001
            pass

    run_meta = stage_run_dir / "run_metadata.json"
    if run_meta.exists():
        try:
            payload = json.loads(run_meta.read_text(encoding="utf-8"))
            reqs = payload.get("requirements_used", [])
            if isinstance(reqs, list):
                out.update([str(x) for x in reqs])
        except Exception:  # noqa: BLE001
            pass
    return sorted(out)


def _ensure_safe_stage_run_dir(*, unified_run_dir: Path, stage_run_dir: Path) -> None:
    u = unified_run_dir.resolve()
    s = stage_run_dir.resolve()
    if s == u:
        raise PipelineError(
            "Unsafe stage run path: stage run directory resolves to unified run directory."
        )
    if u not in s.parents:
        raise PipelineError(
            "Unsafe stage run path: stage run directory is outside the unified run directory."
        )


def _prepare_preprocess_stage_config(
    *,
    original_config_path: Path,
    stage_configs_dir: Path,
    stage_output_root: Path,
) -> tuple[Path, Path, Path]:
    raw = _load_yaml_map(original_config_path)
    cfg_dir = original_config_path.parent.resolve()

    run_raw = raw.setdefault("run", {})
    paths_raw = raw.setdefault("paths", {})
    if not isinstance(run_raw, dict) or not isinstance(paths_raw, dict):
        raise PipelineError(f"Invalid preprocess config structure: {original_config_path}")

    input_glob = paths_raw.get("input_glob")
    if input_glob is not None:
        paths_raw["input_glob"] = _as_abs_path_str(cfg_dir, input_glob)

    run_raw["run_id"] = PREPROCESS_STAGE_RUN_ID
    run_raw["overwrite"] = False
    paths_raw["output_root"] = str(stage_output_root)

    windows_subdir = str(paths_raw.get("windows_subdir", "windows"))

    patched_path = stage_configs_dir / f"{PREPROCESS_STEP_ID}.yml"
    _write_yaml_map(patched_path, raw)

    stage_run_dir = stage_output_root / PREPROCESS_STAGE_RUN_ID
    windows_dir = stage_run_dir / windows_subdir
    return patched_path, stage_run_dir, windows_dir


def _prepare_build_tables_stage_config(
    *,
    original_config_path: Path,
    stage_configs_dir: Path,
    stage_output_root: Path,
    preprocess_windows_dir: Optional[Path],
    use_preprocess_windows: bool,
) -> tuple[Path, Path]:
    raw = _load_yaml_map(original_config_path)
    cfg_dir = original_config_path.parent.resolve()

    run_raw = raw.setdefault("run", {})
    paths_raw = raw.setdefault("paths", {})
    dwell_tasks = raw.get("dwell_tasks", [])

    if not isinstance(run_raw, dict) or not isinstance(paths_raw, dict):
        raise PipelineError(f"Invalid build-tables config structure: {original_config_path}")
    if not isinstance(dwell_tasks, list):
        raise PipelineError(f"dwell_tasks must be a list: {original_config_path}")
    if use_preprocess_windows and preprocess_windows_dir is None:
        raise PipelineError(
            "build_tables.use_preprocess_windows=true requires preprocess_jspsych to run first."
        )

    for row in dwell_tasks:
        if not isinstance(row, dict):
            continue
        for key in ("window_data_dir", "video_feats_dir", "subject_metadata_csv", "runlist_csv"):
            if key in row and row[key] is not None:
                row[key] = _as_abs_path_str(cfg_dir, row[key])
        if use_preprocess_windows and preprocess_windows_dir is not None:
            row["window_data_dir"] = str(preprocess_windows_dir)

    run_raw["run_id"] = BUILD_TABLES_STAGE_RUN_ID
    run_raw["overwrite"] = False
    paths_raw["output_root"] = str(stage_output_root)

    patched_path = stage_configs_dir / f"{BUILD_TABLES_STEP_ID}.yml"
    _write_yaml_map(patched_path, raw)

    stage_run_dir = stage_output_root / BUILD_TABLES_STAGE_RUN_ID
    return patched_path, stage_run_dir


def _prepare_analysis_ready_stage_config(
    *,
    original_config_path: Path,
    stage_configs_dir: Path,
    stage_output_root: Path,
    feature_long_csv: Optional[Path],
    run_level_csv: Optional[Path],
) -> tuple[Path, Path]:
    raw = _load_yaml_map(original_config_path)
    cfg_dir = original_config_path.parent.resolve()

    run_raw = raw.setdefault("run", {})
    paths_raw = raw.setdefault("paths", {})
    if not isinstance(run_raw, dict) or not isinstance(paths_raw, dict):
        raise PipelineError(f"Invalid analysis-ready config structure: {original_config_path}")

    for key in ("feature_long_csv", "run_level_csv"):
        if key in paths_raw and paths_raw[key] is not None:
            paths_raw[key] = _as_abs_path_str(cfg_dir, paths_raw[key])

    if feature_long_csv is not None:
        paths_raw["feature_long_csv"] = str(feature_long_csv)
    if run_level_csv is not None:
        paths_raw["run_level_csv"] = str(run_level_csv)

    run_raw["run_id"] = ANALYSIS_READY_STAGE_RUN_ID
    run_raw["overwrite"] = False
    paths_raw["output_root"] = str(stage_output_root)

    patched_path = stage_configs_dir / f"{ANALYSIS_READY_STEP_ID}.yml"
    _write_yaml_map(patched_path, raw)

    stage_run_dir = stage_output_root / ANALYSIS_READY_STAGE_RUN_ID
    return patched_path, stage_run_dir


def run_pipeline(*, config_path: Path, cfg: AppConfig) -> Path:
    """Run unified end-to-end orchestration and return the unified run directory."""
    cfg_path = config_path.resolve()
    cfg_dir = cfg_path.parent

    output_root = _resolve(cfg_dir, cfg.run.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = cfg.run.run_id or make_run_id(prefix="pipeline", seed=cfg.run.random_seed)
    run_dir = output_root / run_id

    if run_dir.exists():
        if cfg.run.overwrite:
            shutil.rmtree(run_dir)
        else:
            raise PipelineError(
                f"Unified run directory already exists: {run_dir} (set run.overwrite=true)"
            )

    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    stage_cfg_dir = run_dir / "stage_configs"
    stages_root = run_dir / "stages"
    for d in (logs_dir, artifacts_dir, stage_cfg_dir, stages_root):
        d.mkdir(parents=True, exist_ok=True)

    pipeline_log = logs_dir / "pipeline.jsonl"
    started_utc = _now_utc_iso()
    _log_event(
        pipeline_log,
        level="INFO",
        msg="pipeline_start",
        run_id=run_id,
        steps=cfg.pipeline.steps,
    )

    input_hashes: Dict[str, str] = {}
    executed_steps: list[Dict[str, Any]] = []
    stage_run_dirs: Dict[str, str] = {}
    stage_config_paths: Dict[str, str] = {}
    requirements_union: set[str] = {"ENG-AUDIT-001", "ORCH-E2E-001"}

    pipeline_status = "running"
    failure_error: Optional[str] = None
    failed_step_id: Optional[str] = None
    current_step_id: Optional[str] = None

    preprocess_windows_dir: Optional[Path] = None
    feature_long_csv: Optional[Path] = None
    run_level_csv: Optional[Path] = None

    try:
        cfg_text = cfg_path.read_text(encoding="utf-8")
        write_text(cfg_text, artifacts_dir / "config.snapshot.yml")
        run_meta = build_run_metadata(
            run_id=run_id,
            repo_root=Path(__file__).resolve().parents[3],
            random_seed=cfg.run.random_seed,
            config_text=cfg_text,
            packages=[
                "adolphslab-webgazer",
                "numpy",
                "pandas",
                "scipy",
                "statsmodels",
                "scikit-learn",
                "pydantic",
                "PyYAML",
            ],
        )
        write_json(run_meta.to_dict(), artifacts_dir / "run_metadata.json")

        input_hashes = {
            "unified_config_path": sha256_file(cfg_path),
            "unified_config_text": sha256_text(cfg_text),
        }

        for step_id in cfg.pipeline.steps:
            current_step_id = step_id
            _log_event(pipeline_log, level="INFO", msg="step_start", step_id=step_id)

            if step_id == PREPROCESS_STEP_ID:
                if not cfg.preprocess_jspsych.enabled:
                    executed_steps.append({"step_id": step_id, "skipped": True})
                    _log_event(pipeline_log, level="INFO", msg="step_skip", step_id=step_id)
                    continue

                source_cfg = _resolve(cfg_dir, cfg.preprocess_jspsych.config_path)
                patched_cfg, expected_stage_run_dir, windows_dir = _prepare_preprocess_stage_config(
                    original_config_path=source_cfg,
                    stage_configs_dir=stage_cfg_dir,
                    stage_output_root=stages_root,
                )
                _ensure_safe_stage_run_dir(
                    unified_run_dir=run_dir,
                    stage_run_dir=expected_stage_run_dir,
                )

                from alabwebgazer.preprocessing.batch import run_preprocess_pipeline

                actual_stage_run_dir = run_preprocess_pipeline(config_path=patched_cfg)
                preprocess_windows_dir = windows_dir
                if not preprocess_windows_dir.exists():
                    raise PipelineError(
                        f"Missing preprocess windows directory: {preprocess_windows_dir}"
                    )
                step_reqs = _collect_stage_requirements(actual_stage_run_dir)
                requirements_union.update(step_reqs)
                executed_steps.append(
                    {
                        "step_id": step_id,
                        "skipped": False,
                        "run_dir": _relative_or_abs(actual_stage_run_dir, run_dir),
                        "requirements": step_reqs,
                    }
                )
                stage_run_dirs[step_id] = _relative_or_abs(actual_stage_run_dir, run_dir)
                stage_config_paths[step_id] = _relative_or_abs(patched_cfg, run_dir)
                input_hashes[f"{step_id}.source_config"] = sha256_file(source_cfg)
                input_hashes[f"{step_id}.patched_config"] = sha256_file(patched_cfg)

            elif step_id == BUILD_TABLES_STEP_ID:
                if not cfg.build_tables.enabled:
                    executed_steps.append({"step_id": step_id, "skipped": True})
                    _log_event(pipeline_log, level="INFO", msg="step_skip", step_id=step_id)
                    continue

                source_cfg = _resolve(cfg_dir, cfg.build_tables.config_path)
                patched_cfg, expected_stage_run_dir = _prepare_build_tables_stage_config(
                    original_config_path=source_cfg,
                    stage_configs_dir=stage_cfg_dir,
                    stage_output_root=stages_root,
                    preprocess_windows_dir=preprocess_windows_dir,
                    use_preprocess_windows=cfg.build_tables.use_preprocess_windows,
                )
                _ensure_safe_stage_run_dir(
                    unified_run_dir=run_dir,
                    stage_run_dir=expected_stage_run_dir,
                )

                from alabwebgazer.derive_tables.pipeline import (
                    run_pipeline as run_build_tables_pipeline,
                )

                actual_stage_run_dir = run_build_tables_pipeline(config_path=patched_cfg)
                feature_long_csv = actual_stage_run_dir / "tables" / "df_mergedmain.csv"
                run_level_csv = actual_stage_run_dir / "tables" / "df_runs.csv"

                step_reqs = _collect_stage_requirements(actual_stage_run_dir)
                requirements_union.update(step_reqs)
                executed_steps.append(
                    {
                        "step_id": step_id,
                        "skipped": False,
                        "run_dir": _relative_or_abs(actual_stage_run_dir, run_dir),
                        "requirements": step_reqs,
                    }
                )
                stage_run_dirs[step_id] = _relative_or_abs(actual_stage_run_dir, run_dir)
                stage_config_paths[step_id] = _relative_or_abs(patched_cfg, run_dir)
                input_hashes[f"{step_id}.source_config"] = sha256_file(source_cfg)
                input_hashes[f"{step_id}.patched_config"] = sha256_file(patched_cfg)

            elif step_id == ANALYSIS_READY_STEP_ID:
                if not cfg.analysis_ready.enabled:
                    executed_steps.append({"step_id": step_id, "skipped": True})
                    _log_event(pipeline_log, level="INFO", msg="step_skip", step_id=step_id)
                    continue

                if cfg.analysis_ready.use_build_tables_outputs:
                    if feature_long_csv is None or run_level_csv is None:
                        raise PipelineError(
                            "analysis_ready.use_build_tables_outputs=true requires build_tables "
                            "to run first in pipeline.steps."
                        )
                    if not feature_long_csv.exists() or not run_level_csv.exists():
                        raise PipelineError(
                            "Missing Stage 2 outputs required for analysis-ready: "
                            f"{feature_long_csv}, {run_level_csv}"
                        )

                source_cfg = _resolve(cfg_dir, cfg.analysis_ready.config_path)
                patched_cfg, expected_stage_run_dir = _prepare_analysis_ready_stage_config(
                    original_config_path=source_cfg,
                    stage_configs_dir=stage_cfg_dir,
                    stage_output_root=stages_root,
                    feature_long_csv=(
                        feature_long_csv if cfg.analysis_ready.use_build_tables_outputs else None
                    ),
                    run_level_csv=(
                        run_level_csv if cfg.analysis_ready.use_build_tables_outputs else None
                    ),
                )
                _ensure_safe_stage_run_dir(
                    unified_run_dir=run_dir,
                    stage_run_dir=expected_stage_run_dir,
                )

                from alabwebgazer.analysis_ready.pipeline import (
                    run_pipeline as run_analysis_ready_pipeline,
                )

                actual_stage_run_dir = run_analysis_ready_pipeline(
                    config_path=patched_cfg,
                    steps=cfg.analysis_ready.steps,
                    overwrite=False,
                )

                step_reqs = _collect_stage_requirements(actual_stage_run_dir)
                requirements_union.update(step_reqs)
                executed_steps.append(
                    {
                        "step_id": step_id,
                        "skipped": False,
                        "run_dir": _relative_or_abs(actual_stage_run_dir, run_dir),
                        "requirements": step_reqs,
                    }
                )
                stage_run_dirs[step_id] = _relative_or_abs(actual_stage_run_dir, run_dir)
                stage_config_paths[step_id] = _relative_or_abs(patched_cfg, run_dir)
                input_hashes[f"{step_id}.source_config"] = sha256_file(source_cfg)
                input_hashes[f"{step_id}.patched_config"] = sha256_file(patched_cfg)

            else:  # pragma: no cover - guarded by config validator
                raise PipelineError(f"Unsupported step id: {step_id}")

            _log_event(pipeline_log, level="INFO", msg="step_end", step_id=step_id)

        pipeline_status = "completed"
        _log_event(pipeline_log, level="INFO", msg="pipeline_done", run_id=run_id)
    except Exception as exc:
        pipeline_status = "failed"
        failure_error = f"{type(exc).__name__}: {exc}"
        failed_step_id = current_step_id
        _log_event(
            pipeline_log,
            level="ERROR",
            msg="pipeline_failed",
            run_id=run_id,
            failed_step_id=failed_step_id,
            error=failure_error,
        )
        raise
    finally:
        finished_utc = _now_utc_iso()
        if input_hashes:
            write_json(input_hashes, artifacts_dir / "input_hashes.json")
        manifest = {
            "run_id": run_id,
            "created_utc": started_utc,
            "started_utc": started_utc,
            "finished_utc": finished_utc,
            "status": pipeline_status,
            "notes": cfg.run.notes,
            "config_path": str(cfg_path),
            "steps_requested": cfg.pipeline.steps,
            "steps_executed": executed_steps,
            "stage_run_dirs": stage_run_dirs,
            "stage_configs": stage_config_paths,
            "requirements": sorted(requirements_union),
        }
        if failure_error is not None:
            manifest["error"] = failure_error
        if failed_step_id is not None:
            manifest["failed_step_id"] = failed_step_id
        write_json(manifest, run_dir / "manifest.json")

    return run_dir

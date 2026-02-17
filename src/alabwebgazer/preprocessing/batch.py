"""Batch preprocessing for jsPsych/WebGazer exports.

Produces (per run directory):
- ``windows/``: ``videoview_window_*.csv`` binned quadrant time series
- ``tables/preprocess_summary.csv``: one row per output file
- ``artifacts/config.snapshot.yml``: exact config text
- ``artifacts/run_metadata.json`` and ``artifacts/input_hashes.json``
- ``manifest.json``: high-level summary of inputs/outputs

This module is the bridge from notebook-style preprocessing to downstream dwell-time extraction.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from alabwebgazer.core.run import (
    make_run_id,
    prepare_run_dirs,
    snapshot_config,
    write_metadata_and_inputs,
)
from alabwebgazer.io import write_json
from alabwebgazer.logging import configure_logging
from alabwebgazer.preprocessing.config import (
    AppConfig,
    ConditionRule,
    FilenameParsingConfig,
    load_config,
)
from alabwebgazer.preprocessing.pipeline import WebGazerPreprocessParams, preprocess_webgazer_trial
from alabwebgazer.preprocessing.qc import QuadrantImageQC, compute_quadrant_image_qc
from alabwebgazer.requirements import require


class PipelineError(RuntimeError):
    """Raised when a preprocessing run cannot complete."""


@dataclass(frozen=True)
class TrialSpec:
    file: Path
    trial_index: int
    participant_id: str
    session_id: str
    condition: str
    video_length_seconds: float
    inner_width: float
    inner_height: float


def _load_jspsych_json(path: Path) -> List[Dict[str, Any]]:
    """Load a jsPsych JSON export.

    Expected: a JSON list of trial dicts.
    Also tolerated: dict-wrapped export like {"data": [...]}.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        for key in ("data", "trials", "raw", "rows"):
            if key in raw and isinstance(raw[key], list):
                return [x for x in raw[key] if isinstance(x, dict)]
        raise ValueError(f"Unsupported jsPsych JSON dict format (keys: {sorted(raw.keys())})")
    raise ValueError(f"Unsupported jsPsych JSON type: {type(raw)}")


def _parse_id_session(path: Path, cfg: FilenameParsingConfig) -> Tuple[str, str]:
    stem = path.stem

    if cfg.id_regex:
        m = re.search(cfg.id_regex, stem)
        if not m:
            raise ValueError(f"id_regex did not match filename stem: {stem}")
        participant_id = m.group(1)
    else:
        parts = stem.split(cfg.delimiter)
        participant_id = parts[cfg.id_index] if cfg.id_index < len(parts) else "UNKNOWN"

    if cfg.session_regex:
        m = re.search(cfg.session_regex, stem)
        if not m:
            raise ValueError(f"session_regex did not match filename stem: {stem}")
        session_id = m.group(1)
    else:
        parts = stem.split(cfg.delimiter)
        start = min(cfg.session_start, len(parts))
        end = min(cfg.session_end, len(parts))
        session_id = "".join(parts[start:end]) if end > start else ""

    return str(participant_id), str(session_id)


def _trial_field_text(trial: Dict[str, Any], match_field: str) -> str:
    if match_field == "stimname":
        return str(trial.get("stimname", ""))
    stim = trial.get("stimulus", "")
    if match_field == "stimulus0":
        if isinstance(stim, list) and stim:
            return str(stim[0])
        return str(stim)
    if match_field == "stimulus":
        if isinstance(stim, list):
            return " ".join([str(x) for x in stim])
        return str(stim)
    raise ValueError(f"Unsupported match_field: {match_field}")


def _infer_condition(trial: Dict[str, Any], rules: Iterable[ConditionRule]) -> Tuple[str, float]:
    for rule in rules:
        text = _trial_field_text(trial, rule.match_field)
        pat = rule.pattern
        if not rule.case_sensitive:
            text = text.lower()
            pat = pat.lower()
        if rule.use_regex:
            if re.search(pat, text):
                return rule.condition, float(rule.video_length_seconds)
        else:
            if pat in text:
                return rule.condition, float(rule.video_length_seconds)
    raise ValueError("No condition_rules matched this trial (cannot infer condition/video length).")


def _hash_id(raw_id: str, *, salt: str) -> str:
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(raw_id.encode("utf-8"))
    return h.hexdigest()[:12]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Expected numeric value, got {value!r}") from exc


def discover_video_trials(
    *,
    jspsych_trials: List[Dict[str, Any]],
    file_path: Path,
    participant_id: str,
    session_id: str,
    cfg: AppConfig,
) -> List[TrialSpec]:
    out: List[TrialSpec] = []
    for idx, trial in enumerate(jspsych_trials):
        if str(trial.get("trial_type", "")) != cfg.trials.trial_type:
            continue
        if cfg.trials.require_webgazer_data and "webgazer_data" not in trial:
            continue

        condition, vid_len = _infer_condition(trial, cfg.condition_rules)

        if "innerWidth" not in trial or "innerHeight" not in trial:
            if cfg.trials.default_inner_width is None or cfg.trials.default_inner_height is None:
                raise ValueError(
                    "Trial missing innerWidth/innerHeight "
                    f"(file={file_path.name}, trial_index={idx}). "
                    "Provide trials.default_inner_width/default_inner_height in the config "
                    "as a fallback."
                )
            trial_inner_w = float(cfg.trials.default_inner_width)
            trial_inner_h = float(cfg.trials.default_inner_height)
        else:
            trial_inner_w = _safe_float(trial["innerWidth"])
            trial_inner_h = _safe_float(trial["innerHeight"])

        out.append(
            TrialSpec(
                file=file_path,
                trial_index=int(idx),
                participant_id=participant_id,
                session_id=session_id,
                condition=str(condition),
                video_length_seconds=float(vid_len),
                inner_width=trial_inner_w,
                inner_height=trial_inner_h,
            )
        )

    return out


def _load_webgazer_df(trial: Dict[str, Any]) -> pd.DataFrame:
    wg = trial.get("webgazer_data")
    if wg is None:
        return pd.DataFrame(columns=["t", "x", "y"])
    if isinstance(wg, str):
        wg = json.loads(wg)
    if isinstance(wg, list):
        return pd.DataFrame(wg)
    raise ValueError(f"Unsupported webgazer_data type: {type(wg)}")


def run_preprocess_pipeline(*, config_path: Path) -> Path:
    cfg = load_config(config_path)
    config_dir = config_path.parent.resolve()

    output_root = (
        cfg.paths.output_root
        if cfg.paths.output_root.is_absolute()
        else (config_dir / cfg.paths.output_root)
    )
    output_root.mkdir(parents=True, exist_ok=True)

    run_id = cfg.run.run_id or make_run_id(prefix="preprocess", seed=cfg.run.random_seed)
    paths = prepare_run_dirs(output_root=output_root, run_id=run_id, overwrite=cfg.run.overwrite)

    logger = configure_logging(
        log_path=paths.logs_dir / "preprocess.jsonl",
        level=str(cfg.run.log_level),
    )
    logger.info("preprocess_start")

    config_text = snapshot_config(config_path=config_path, out_dir=paths.artifacts_dir)

    # Expand input glob relative to config file by default
    pattern = cfg.paths.input_glob
    glob_path = pattern if Path(pattern).is_absolute() else str((config_dir / pattern).resolve())
    input_files = [Path(p) for p in sorted(glob(glob_path))]
    if not input_files:
        raise PipelineError(
            f"No input files matched: {cfg.paths.input_glob} (resolved: {glob_path})"
        )

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
            "scikit-learn",
            "pydantic",
            "PyYAML",
        ),
    )

    windows_dir = paths.run_dir / cfg.paths.windows_subdir
    windows_dir.mkdir(parents=True, exist_ok=True)

    # Preprocess parameters
    p = cfg.preprocess
    params = WebGazerPreprocessParams(
        median_kernel=p.median_kernel,
        median_method=p.median_method,
        x_prob_threshold=p.x_prob_threshold,
        y_prob_threshold=p.y_prob_threshold,
        bin_seconds=p.bin_seconds,
        drift_mode=p.drift_mode,
        apply_drift_correction=p.apply_drift_correction,
        quadrant_method=p.quadrant_method,
        allow_midpoint_fallback=p.allow_midpoint_fallback,
        random_state=p.random_state,
        resample_edge_policy=p.resample_edge_policy,
        resample_include_endpoint=p.resample_include_endpoint,
    )

    salt = ""
    if cfg.privacy.hash_subject_ids:
        salt = os.environ.get(cfg.privacy.hash_salt_env, "")
        if not salt:
            raise PipelineError(
                f"privacy.hash_subject_ids=true but env var {cfg.privacy.hash_salt_env} is not set"
            )

    summary_rows: List[Dict[str, Any]] = []

    for fp in input_files:
        pid, session = _parse_id_session(fp, cfg.filename_parsing)
        pid_out = _hash_id(pid, salt=salt) if cfg.privacy.hash_subject_ids else pid

        trials = _load_jspsych_json(fp)
        qc_result: Optional[QuadrantImageQC] = None
        if cfg.quadrant_image_qc.enabled:
            try:
                qc_result = compute_quadrant_image_qc(
                    trials,
                    q1_pattern=cfg.quadrant_image_qc.q1_pattern,
                    q2_pattern=cfg.quadrant_image_qc.q2_pattern,
                    q3_pattern=cfg.quadrant_image_qc.q3_pattern,
                    q4_pattern=cfg.quadrant_image_qc.q4_pattern,
                    median_kernel=5,
                    x_prob_thres=params.x_prob_threshold,
                    y_prob_thres=params.y_prob_threshold,
                    random_state=params.random_state,
                )
                safe_pid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(pid_out)).strip("._-") or "NA"
                safe_sess = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(session)).strip("._-") or "NA"
                qc_json = paths.artifacts_dir / f"qc_quadrant_images_{safe_pid}_{safe_sess}.json"
                write_json(
                    {
                        "participant_id": pid_out,
                        "session_id": session,
                        "n_image_trials": qc_result.n_image_trials,
                        "raw_accuracy": qc_result.raw_accuracy,
                        "preproc_accuracy": qc_result.preproc_accuracy,
                        "confusion_raw": qc_result.confusion_raw,
                        "confusion_preproc": qc_result.confusion_preproc,
                    },
                    qc_json,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"quadrant_image_qc_failed file={fp.name}: {exc}")

        specs = discover_video_trials(
            jspsych_trials=trials,
            file_path=fp,
            participant_id=pid_out,
            session_id=session,
            cfg=cfg,
        )

        for spec in specs:
            trial = trials[spec.trial_index]
            gaze = _load_webgazer_df(trial)

            res = preprocess_webgazer_trial(
                gaze,
                inner_width=spec.inner_width,
                inner_height=spec.inner_height,
                video_length_seconds=spec.video_length_seconds,
                params=params,
            )

            try:
                out_name = cfg.output_naming.template.format(
                    id=spec.participant_id,
                    session=spec.session_id,
                    condition=spec.condition,
                    trial_index=spec.trial_index,
                )
            except KeyError as exc:
                raise PipelineError(
                    f"output_naming.template contains unknown field: {exc!s}. "
                    "Allowed fields: {id}, {session}, {condition}, {trial_index}."
                ) from exc

            if (
                cfg.output_naming.add_trial_index_suffix
                and "{trial_index}" not in cfg.output_naming.template
            ):
                if out_name.endswith(".csv"):
                    out_name = out_name[:-4] + f"_trial{spec.trial_index}.csv"
                else:
                    out_name = out_name + f"_trial{spec.trial_index}"

            out_csv = windows_dir / out_name
            res.frame.to_csv(out_csv, index=False)

            summary_rows.append(
                {
                    "input_file": fp.name,
                    "participant_id": spec.participant_id,
                    "session_id": spec.session_id,
                    "condition": spec.condition,
                    "trial_index": spec.trial_index,
                    "video_length_seconds": spec.video_length_seconds,
                    "inner_width": spec.inner_width,
                    "inner_height": spec.inner_height,
                    "out_csv": str(out_csv.relative_to(paths.run_dir)),
                    "prop_exclusion_pct": res.prop_exclusion_pct,
                    "n_raw": res.n_raw,
                    "n_after_border": res.n_after_border,
                    "n_after_confidence": res.n_after_confidence,
                    "quadrant_method_used": res.quadrant_method_used,
                    "qc_quadimg_raw_acc": qc_result.raw_accuracy if qc_result else None,
                    "qc_quadimg_preproc_acc": qc_result.preproc_accuracy if qc_result else None,
                    "qc_quadimg_n_trials": qc_result.n_image_trials if qc_result else None,
                }
            )

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(paths.tables_dir / cfg.paths.summary_csv, index=False)

    # Basic manifest
    require(["RAW-WINDOW-001", "RAW-WINDOW-002", "ENG-AUDIT-001", "ENG-PRIV-001"])
    manifest = {
        "run_id": run_id,
        "requirements": [
            "RAW-WINDOW-001",
            "RAW-WINDOW-002",
            "ENG-AUDIT-001",
            "ENG-PRIV-001",
        ],
        "inputs": [str(p) for p in input_files],
        "outputs": {
            "windows_dir": str(windows_dir.relative_to(paths.run_dir)),
            "summary_csv": str(
                (paths.tables_dir / cfg.paths.summary_csv).relative_to(paths.run_dir)
            ),
        },
    }
    write_json(manifest, paths.run_dir / "manifest.json")

    logger.info("preprocess_done")
    return paths.run_dir

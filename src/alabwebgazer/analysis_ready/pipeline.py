"""Pipeline orchestrator.

Key requirements:
- REV-REPRO-001: deterministic execution + run manifest + config snapshot.
- REV-AUDIT-001: stable run directory with stage artifacts.

This orchestrator is intentionally simple; complex DAG scheduling can be added later.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np

from alabwebgazer.core.hashing import sha256_file, sha256_text
from alabwebgazer.core.run import prepare_run_dirs, snapshot_config

from .artifacts import ArtifactStore
from .config import load_config
from .errors import PipelineError
from .logging import configure_logging, get_logger
from .provenance import RunManifest
from .steps.base import RunContext
from .steps.registry import STEP_REGISTRY

log = get_logger(__name__)


DEFAULT_STEPS: List[str] = [
    "ingest_analysis_ready",
    "confirmatory_models",
]


def run_pipeline(
    *,
    config_path: Path,
    steps: Optional[List[str]] = None,
    overwrite: bool = False,
) -> Path:
    """Run the pipeline and return the run directory."""
    cfg = load_config(config_path)

    try:
        paths = prepare_run_dirs(
            output_root=cfg.paths.output_root,
            run_id=cfg.run.run_id,
            overwrite=overwrite,
        )
    except FileExistsError as exc:
        raise PipelineError(str(exc)) from exc
    run_dir = paths.run_dir

    configure_logging(
        log_path=paths.logs_dir / "analysis_ready.jsonl",
        level="INFO",
    )
    log.info("Starting run: %s", cfg.run.run_id)

    # Save config snapshot (canonical stage location + backward-compatible root aliases)
    config_text = snapshot_config(config_path=config_path, out_dir=paths.artifacts_dir)
    (run_dir / "config.snapshot.yml").write_text(config_text, encoding="utf-8")
    # Backward-compatible alias
    (run_dir / "config_snapshot.yml").write_text(config_text, encoding="utf-8")

    # Run manifest (canonical filename across all stages)
    manifest = RunManifest.create(
        run_id=cfg.run.run_id,
        repo_root=Path(__file__).resolve().parents[3],
    )
    manifest.write_json(run_dir / "manifest.json")
    # Backward-compatible alias
    manifest.write_json(run_dir / "run_manifest.json")

    # RNG
    rng = np.random.default_rng(cfg.run.random_seed)

    artifacts = ArtifactStore(run_dir=paths.tables_dir)
    ctx = RunContext(config=cfg, artifacts=artifacts, rng=rng)

    step_ids = steps if steps is not None else DEFAULT_STEPS

    for step_id in step_ids:
        if step_id not in STEP_REGISTRY:
            raise PipelineError(f"Unknown step_id: {step_id}. Known: {sorted(STEP_REGISTRY)}")

    step_requirements = {
        step_id: list(STEP_REGISTRY[step_id].requirement_ids) for step_id in step_ids
    }
    requirements_used = sorted({rid for rids in step_requirements.values() for rid in rids})

    input_hashes: dict[str, str] = {}
    for p in (config_path, cfg.paths.feature_long_csv, cfg.paths.run_level_csv):
        if p.exists() and p.is_file():
            input_hashes[str(p)] = sha256_file(p)
    (paths.artifacts_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Backward-compatible alias
    (run_dir / "input_hashes.json").write_text(
        json.dumps(input_hashes, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    run_meta = {
        "run_id": cfg.run.run_id,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "config_sha256": sha256_text(config_text),
        "steps_requested": step_ids,
        "step_requirements": step_requirements,
        "requirements_used": requirements_used,
        "input_hashes": input_hashes,
        "logs_dir": "logs",
        "artifacts_dir": "artifacts",
        "tables_dir": "tables",
        "status": "running",
        "executed_steps": [],
    }
    (paths.artifacts_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Backward-compatible alias
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    executed_steps: List[str] = []
    for step_id in step_ids:
        step_cls = STEP_REGISTRY[step_id]
        step = step_cls()
        log.info("Running step: %s", step_id)
        step.run(ctx)
        executed_steps.append(step_id)

    run_meta["status"] = "completed"
    run_meta["completed_utc"] = datetime.now(timezone.utc).isoformat()
    run_meta["executed_steps"] = executed_steps
    (paths.artifacts_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Backward-compatible alias
    (run_dir / "run_metadata.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Backward-compatible stage directory copies at run root.
    for stage_dir in sorted(paths.tables_dir.iterdir()):
        if not stage_dir.is_dir():
            continue
        legacy_stage_dir = run_dir / stage_dir.name
        if legacy_stage_dir.exists():
            shutil.rmtree(legacy_stage_dir)
        shutil.copytree(stage_dir, legacy_stage_dir)

    log.info("Run complete: %s", run_dir)
    return run_dir

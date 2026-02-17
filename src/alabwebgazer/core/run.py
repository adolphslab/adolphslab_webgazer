"""Run directory helpers used by CLI pipelines."""

from __future__ import annotations

import random
import shutil
import string
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from alabwebgazer.core.hashing import sha256_file
from alabwebgazer.core.provenance import build_run_metadata
from alabwebgazer.io import write_json, write_text


def _rand_suffix(seed: int, n: int = 6) -> str:
    rng = random.Random(int(seed))
    alphabet = string.ascii_lowercase + string.digits
    return "".join(rng.choice(alphabet) for _ in range(n))


def make_run_id(*, prefix: str, seed: int) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{ts}_{_rand_suffix(seed)}"


@dataclass(frozen=True)
class RunPaths:
    run_dir: Path
    logs_dir: Path
    artifacts_dir: Path
    tables_dir: Path


def prepare_run_dirs(*, output_root: Path, run_id: str, overwrite: bool) -> RunPaths:
    run_dir = output_root / run_id
    if run_dir.exists():
        if overwrite:
            shutil.rmtree(run_dir)
        else:
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. Choose a new run_id or set overwrite=true."
            )

    logs_dir = run_dir / "logs"
    artifacts_dir = run_dir / "artifacts"
    tables_dir = run_dir / "tables"
    for d in (logs_dir, artifacts_dir, tables_dir):
        d.mkdir(parents=True, exist_ok=True)

    return RunPaths(run_dir=run_dir, logs_dir=logs_dir, artifacts_dir=artifacts_dir, tables_dir=tables_dir)


def snapshot_config(*, config_path: Path, out_dir: Path) -> str:
    text = config_path.read_text(encoding="utf-8")
    write_text(text, out_dir / "config.snapshot.yml")
    return text


def write_metadata_and_inputs(
    *,
    paths: RunPaths,
    repo_root: Path,
    run_id: str,
    random_seed: int,
    config_text: str,
    input_paths: Sequence[Path],
    packages: Sequence[str],
) -> Dict[str, Any]:
    meta = build_run_metadata(
        run_id=run_id,
        repo_root=repo_root,
        random_seed=random_seed,
        config_text=config_text,
        packages=packages,
    )
    write_json(meta.to_dict(), paths.artifacts_dir / "run_metadata.json")

    hashes: Dict[str, str] = {}
    for p in input_paths:
        if p.exists() and p.is_file():
            hashes[str(p)] = sha256_file(p)
    write_json(hashes, paths.artifacts_dir / "input_hashes.json")

    return {"run_metadata": meta.to_dict(), "input_hashes": hashes}

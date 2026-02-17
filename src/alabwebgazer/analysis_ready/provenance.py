"""Provenance capture utilities.

Requirements
- REV-REPRO-001: capture run metadata (timestamp, git SHA, versions, config snapshot).
- REV-SEC-001: do not leak secrets in logs/metadata.

This module intentionally avoids capturing arbitrary environment variables.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Optional


_SAFE_ENV_ALLOWLIST = [
    "PYTHONHASHSEED",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]


def _get_git_sha(repo_root: Path) -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or None
    except Exception:  # noqa: BLE001
        return None


def _get_versions(packages: list[str]) -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for pkg in packages:
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


@dataclass(frozen=True)
class RunManifest:
    run_id: str
    created_utc: str
    git_sha: Optional[str]
    python_version: str
    platform: str
    package_versions: Dict[str, Optional[str]]
    safe_env: Dict[str, Optional[str]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "git_sha": self.git_sha,
            "python_version": self.python_version,
            "platform": self.platform,
            "package_versions": self.package_versions,
            "safe_env": self.safe_env,
        }

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def create(cls, run_id: str, repo_root: Path) -> "RunManifest":
        created = datetime.now(timezone.utc).isoformat()
        return cls(
            run_id=run_id,
            created_utc=created,
            git_sha=_get_git_sha(repo_root),
            python_version=platform.python_version(),
            platform=platform.platform(),
            package_versions=_get_versions(
                [
                    "numpy",
                    "pandas",
                    "scipy",
                    "statsmodels",
                    "pydantic",
                    "PyYAML",
                    "matplotlib",
                    "scikit-learn",
                    "bambi",
                    "pingouin",
                ]
            ),
            safe_env={k: os.environ.get(k) for k in _SAFE_ENV_ALLOWLIST},
        )

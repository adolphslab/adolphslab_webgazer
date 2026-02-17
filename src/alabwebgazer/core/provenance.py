"""Run metadata capture and provenance helpers."""

from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from alabwebgazer.core.hashing import sha256_text


def _git_sha(cwd: Path) -> Optional[str]:
    try:
        import subprocess

        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip() or None
    except Exception:
        return None


def collect_versions(packages: Iterable[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for pkg in packages:
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def safe_env_allowlist(
    prefixes: Iterable[str] = ("ALABWEBGAZER_",),
    sensitive_markers: Iterable[str] = ("SALT", "SECRET", "TOKEN", "PASSWORD", "KEY"),
) -> Dict[str, str]:
    """Return an allowlisted subset of environment variables with secret redaction."""
    out: Dict[str, str] = {}
    markers = tuple(m.upper() for m in sensitive_markers)
    for k, v in os.environ.items():
        if not any(k.startswith(p) for p in prefixes):
            continue
        key_upper = k.upper()
        if any(marker in key_upper for marker in markers):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class RunMetadata:
    run_id: str
    created_utc: str
    random_seed: int
    git_sha: Optional[str]
    python_version: str
    platform: str
    packages: Dict[str, str]
    env_allowlisted: Dict[str, str]
    config_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "created_utc": self.created_utc,
            "random_seed": self.random_seed,
            "git_sha": self.git_sha,
            "python_version": self.python_version,
            "platform": self.platform,
            "packages": self.packages,
            "env_allowlisted": self.env_allowlisted,
            "config_sha256": self.config_sha256,
        }


def build_run_metadata(
    *,
    run_id: str,
    repo_root: Path,
    random_seed: int,
    config_text: str,
    packages: Iterable[str],
) -> RunMetadata:
    return RunMetadata(
        run_id=run_id,
        created_utc=datetime.now(UTC).isoformat(),
        random_seed=int(random_seed),
        git_sha=_git_sha(repo_root),
        python_version=sys.version,
        platform=platform.platform(),
        packages=collect_versions(packages),
        env_allowlisted=safe_env_allowlist(),
        config_sha256=sha256_text(config_text),
    )

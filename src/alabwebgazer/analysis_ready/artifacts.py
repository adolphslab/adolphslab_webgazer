"""Artifact storage utilities.

Requirements
- REV-AUDIT-001: save intermediate datasets + outputs with stable naming and run IDs.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


@dataclass(frozen=True)
class ArtifactStore:
    run_dir: Path

    def stage_dir(self, stage: str) -> Path:
        d = self.run_dir / stage
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_json(self, stage: str, name: str, obj: Dict[str, Any]) -> Path:
        path = self.stage_dir(stage) / name
        path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def write_text(self, stage: str, name: str, text: str) -> Path:
        path = self.stage_dir(stage) / name
        path.write_text(text, encoding="utf-8")
        return path

    def write_csv(self, stage: str, name: str, df: pd.DataFrame, index: bool = False) -> Path:
        path = self.stage_dir(stage) / name
        df.to_csv(path, index=index)
        return path

    def maybe_write_csv(self, stage: str, name: str, df: Optional[pd.DataFrame]) -> Optional[Path]:
        if df is None:
            return None
        return self.write_csv(stage=stage, name=name, df=df)

"""Pipeline step interfaces.

Design: simple step registry (plugin pattern) so new steps can be added with minimal wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..artifacts import ArtifactStore
from ..config import AppConfig


@dataclass
class RunContext:
    config: AppConfig
    artifacts: ArtifactStore
    rng: np.random.Generator
    # Data tables carried between steps (avoid logging these!)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)


class Step:
    """Abstract pipeline step."""

    step_id: str = "BASE"
    requirement_ids: tuple[str, ...] = ()

    def run(self, ctx: RunContext) -> None:  # pragma: no cover
        raise NotImplementedError

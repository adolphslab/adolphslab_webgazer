"""Step registry.

To add a new step:
- Create a module in `alabwebgazer.analysis_ready.steps`.
- Subclass `Step` and set `step_id`.
- Register it here.
"""

from __future__ import annotations

from typing import Dict, Type

from .base import Step
from .confirmatory import ConfirmatoryModelsStep
from .ingest import IngestAnalysisReadyTablesStep

STEP_REGISTRY: Dict[str, Type[Step]] = {
    IngestAnalysisReadyTablesStep.step_id: IngestAnalysisReadyTablesStep,
    ConfirmatoryModelsStep.step_id: ConfirmatoryModelsStep,
}

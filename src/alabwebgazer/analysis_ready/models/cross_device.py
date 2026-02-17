"""Cross-device agreement and validation analyses.

Scope:
- WebGazer vs lab-based eye tracker agreement
- ranking and ordering consistency summaries

TODO:
- Define required paired-device table contract.
- Implement agreement metrics and reporting outputs.
- Add analysis-ready step integration with manifest artifacts.
"""

from __future__ import annotations

import pandas as pd


def evaluate_cross_device_agreement(*args, **kwargs) -> pd.DataFrame:
    """Return cross-device validation summary tables."""
    raise NotImplementedError

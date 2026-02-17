"""Oculomotor feature analyses.

Scope:
- first-gaze spatial bias
- all-gaze spatial bias
- dispersion and temporal quality summaries

TODO:
- Finalize feature contract and naming conventions.
- Implement metric computation and uncertainty summaries.
- Add analysis-ready step integration and table artifacts.
"""

from __future__ import annotations

import pandas as pd


def compute_oculomotor_features(*args, **kwargs) -> pd.DataFrame:
    """Return oculomotor feature summaries for analysis-ready tables."""
    raise NotImplementedError

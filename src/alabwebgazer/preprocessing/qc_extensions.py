"""Participant-level preprocessing quality-control extensions.

Scope:
- comprehension-check aggregation
- viewing-distance and head-movement summaries
- exclusion diagnostics for downstream filtering

TODO:
- Define raw-input contract and required fields.
- Implement deterministic QC aggregation tables.
- Wire outputs into preprocess stage artifacts.
"""

from __future__ import annotations

import pandas as pd


def build_qc_extension_table(*args, **kwargs) -> pd.DataFrame:
    """Return participant-level QC extension summaries."""
    raise NotImplementedError

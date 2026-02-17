"""Inter-subject correlation / agreement (ISC).

Requirement anchors:
- MS-ISC-001..002 (pairwise agreement + sliding window)
- REV-ISC-001 (avoid pseudoreplication; valid-overlap thresholding; chance correction)

TODO: Implement once quadrant time-series format and filtering rules are finalized.
"""

from __future__ import annotations

import pandas as pd


def compute_isc_pairwise(*args, **kwargs):
    raise NotImplementedError

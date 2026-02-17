"""Cross-video generalizability (ICC).

Requirement anchors:
- MS-GENER-001 (ICC(2,1) two-way random, absolute agreement, single measurement)
- REV-RELIAB-001 (report ICC + uncertainty)

TODO: Implement in Python (pingouin) or via R-psych wrapper, depending on locked spec.
"""

from __future__ import annotations

import pandas as pd


def compute_icc_2_1(*args, **kwargs):
    raise NotImplementedError

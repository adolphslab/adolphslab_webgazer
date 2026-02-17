from __future__ import annotations

import pandas as pd
import pytest

from alabwebgazer.analysis_ready.errors import DataValidationError
from alabwebgazer.analysis_ready.validation import require_unique_keys


def test_analysis_ready_require_unique_keys_raises_on_dupes() -> None:
    df = pd.DataFrame(
        [
            {"SID": "S1", "Video": "S1", "task": "task5_pro", "feature": "speaker"},
            {"SID": "S1", "Video": "S1", "task": "task5_pro", "feature": "speaker"},
        ]
    )
    with pytest.raises(DataValidationError):
        require_unique_keys(df, ["SID", "Video", "task", "feature"], name="df_mergedmain")

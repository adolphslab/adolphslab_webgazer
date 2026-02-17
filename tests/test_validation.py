import pandas as pd
import pytest

from alabwebgazer.contracts import FEATURE_TABLE_CONTRACT
from alabwebgazer.validation import (
    ValidationError,
    ValidationIssue,
    raise_if_needed,
    validate_contract,
)


def test_validate_contract_missing_columns() -> None:
    df = pd.DataFrame({"SID": ["S1"], "Video": ["V1"]})
    issues = validate_contract(df, FEATURE_TABLE_CONTRACT, "feature_table")
    assert issues
    assert issues[0].level == "error"


def test_raise_if_needed_errors() -> None:
    # No issues should not raise.
    raise_if_needed([], fail_on_warnings=False)

    with pytest.raises(ValidationError):
        raise_if_needed([ValidationIssue(level="error", message="boom")], fail_on_warnings=False)

    with pytest.raises(ValidationError):
        raise_if_needed([ValidationIssue(level="warning", message="warn")], fail_on_warnings=True)

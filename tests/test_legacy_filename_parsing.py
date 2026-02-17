from __future__ import annotations

from pathlib import Path

from alabwebgazer.legacy.compute_dwelltime import parse_subject_and_condition


def test_parse_subject_and_condition_handles_trial_suffix_token() -> None:
    sid, cond = parse_subject_and_condition(
        Path("videoview_window_S001_demoSession_demo_trial0.csv"),
        task="task5_pro",
    )
    assert sid == "S001"
    assert cond == "demo"


def test_parse_subject_and_condition_legacy_pattern_unchanged() -> None:
    sid, cond = parse_subject_and_condition(
        Path("videoview_window_S001_demoSession_demo.csv"),
        task="task5_pro",
    )
    assert sid == "S001"
    assert cond == "demo"

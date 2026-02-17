from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alabwebgazer.preprocessing import quadrant as quadrant_mod
from alabwebgazer.preprocessing.filters import border_exclusion, median_filter_xy
from alabwebgazer.preprocessing.pipeline import (
    WebGazerPreprocessParams,
    preprocess_webgazer_trial,
)
from alabwebgazer.preprocessing.quadrant import assign_quadrants_axis


def test_border_exclusion_drops_out_of_bounds() -> None:
    x = [10.0, -1.0, 55.0, 30.0]
    y = [10.0, 10.0, 200.0, 30.0]

    x_kept, y_kept, excluded = border_exclusion(
        x,
        y,
        inner_width=100.0,
        inner_height=100.0,
    )

    assert excluded.tolist() == [1, 2]
    assert x_kept.tolist() == [10.0, 30.0]
    assert y_kept.tolist() == [10.0, 30.0]


def test_assign_quadrants_midpoint_mapping() -> None:
    assign = assign_quadrants_axis(
        x=[10.0, 90.0, 10.0, 90.0],
        y=[10.0, 10.0, 90.0, 90.0],
        inner_width=100.0,
        inner_height=100.0,
        method="midpoint",
    )

    assert assign.window.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_preprocess_webgazer_trial_outputs_expected_columns() -> None:
    # 4 samples over 2 bins (0-0.5s, 0.5-1.0s)
    df = pd.DataFrame(
        {
            "t": [100, 300, 600, 800],
            "x": [10.0, 12.0, 90.0, 88.0],
            "y": [10.0, 14.0, 10.0, 12.0],
        }
    )

    res = preprocess_webgazer_trial(
        df,
        inner_width=100.0,
        inner_height=100.0,
        video_length_seconds=1.0,
        params=WebGazerPreprocessParams(
            median_kernel=3,
            bin_seconds=0.5,
            quadrant_method="midpoint",
            drift_mode="none",
        ),
    )

    out = res.frame
    assert list(out.columns) == [
        "Tstart",
        "Tend",
        "window",
        "x_res",
        "y_res",
        "x_ratio",
        "y_ratio",
    ]
    assert len(out) == 2
    assert set(out["window"].dropna().astype(int).tolist()).issubset({1, 2, 3, 4})
    assert 0.0 <= res.prop_exclusion_pct <= 100.0


def test_median_filter_accepts_even_kernel_by_promoting_to_odd() -> None:
    x = [1.0, 2.0, 100.0, 2.0, 1.0]
    y = [1.0, 1.0, 1.0, 1.0, 1.0]
    x_out, y_out = median_filter_xy(x, y, kernel_size=4)

    assert len(x_out) == len(x)
    assert len(y_out) == len(y)
    assert np.isfinite(x_out).all()


def test_resample_include_endpoint_toggle_changes_tail_bin_behavior() -> None:
    df = pd.DataFrame(
        {
            "t": [100, 300, 600, 800],
            "x": [10.0, 12.0, 90.0, 88.0],
            "y": [10.0, 14.0, 10.0, 12.0],
        }
    )

    p_base = dict(
        median_kernel=1,
        bin_seconds=0.5,
        quadrant_method="midpoint",
        drift_mode="none",
    )

    out_no_end = preprocess_webgazer_trial(
        df,
        inner_width=100.0,
        inner_height=100.0,
        video_length_seconds=1.0,
        params=WebGazerPreprocessParams(**p_base, resample_include_endpoint=False),
    ).frame
    out_with_end = preprocess_webgazer_trial(
        df,
        inner_width=100.0,
        inner_height=100.0,
        video_length_seconds=1.0,
        params=WebGazerPreprocessParams(**p_base, resample_include_endpoint=True),
    ).frame

    assert len(out_no_end) == 1
    assert len(out_with_end) == 2


def test_gmm_midpoint_fallback_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    def _raise(*args: object, **kwargs: object) -> object:  # noqa: ARG001
        raise RuntimeError("forced-gmm-failure")

    monkeypatch.setattr(quadrant_mod, "_fit_gmm_labels", _raise)
    x = [10.0, 90.0, 10.0, 90.0]
    y = [10.0, 10.0, 90.0, 90.0]

    with pytest.raises(RuntimeError):
        assign_quadrants_axis(
            x=x,
            y=y,
            inner_width=100.0,
            inner_height=100.0,
            method="gmm_axis",
            allow_midpoint_fallback=False,
        )

    out = assign_quadrants_axis(
        x=x,
        y=y,
        inner_width=100.0,
        inner_height=100.0,
        method="gmm_axis",
        allow_midpoint_fallback=True,
    )
    assert out.method_used.startswith("midpoint_fallback")
    assert out.window.tolist() == [1.0, 2.0, 3.0, 4.0]

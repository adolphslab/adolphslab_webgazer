"""Single-trial preprocessing pipeline for raw WebGazer x/y streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from alabwebgazer.preprocessing.filters import (
    border_exclusion,
    drift_adjust_to_screen_center,
    median_filter_xy,
)
from alabwebgazer.preprocessing.quadrant import assign_quadrants_axis
from alabwebgazer.preprocessing.resample import resample_to_bins


@dataclass(frozen=True)
class WebGazerPreprocessParams:
    """Preprocessing parameters based on the notebook defaults."""

    median_kernel: int = 7
    median_method: Literal["scipy_medfilt", "rolling"] = "scipy_medfilt"

    x_prob_threshold: float = 0.9
    y_prob_threshold: float = 0.7

    bin_seconds: float = 0.5
    drift_mode: Literal["mean", "median", "none"] = "mean"
    # Legacy compatibility knob. If provided, this overrides drift_mode.
    apply_drift_correction: Optional[bool] = None

    quadrant_method: Literal["gmm_axis", "midpoint"] = "gmm_axis"
    allow_midpoint_fallback: bool = False
    random_state: int = 0

    # Resampling parity controls
    resample_edge_policy: Literal["notebook_strict", "half_open"] = "notebook_strict"
    resample_include_endpoint: bool = True


@dataclass(frozen=True)
class PreprocessResult:
    """Preprocessed trial output and diagnostics."""

    frame: pd.DataFrame
    prop_exclusion_pct: float
    n_raw: int
    n_after_border: int
    n_after_confidence: int
    quadrant_method_used: str


def preprocess_webgazer_trial(
    gazedata_df: pd.DataFrame,
    *,
    inner_width: float,
    inner_height: float,
    video_length_seconds: float,
    params: WebGazerPreprocessParams | None = None,
) -> PreprocessResult:
    """Preprocess one raw WebGazer trial to binned 4-quadrant assignments.

    Required columns in ``gazedata_df``: ``t``, ``x``, ``y``.
    """
    p = params or WebGazerPreprocessParams()
    drift_mode = p.drift_mode
    if p.apply_drift_correction is not None:
        drift_mode = "mean" if p.apply_drift_correction else "none"

    required = {"t", "x", "y"}
    missing = sorted(required - set(gazedata_df.columns))
    if missing:
        raise ValueError(f"gazedata_df missing required columns: {missing}")

    t_arr = pd.to_numeric(gazedata_df["t"], errors="coerce").to_numpy(dtype=float)
    x_arr = pd.to_numeric(gazedata_df["x"], errors="coerce").to_numpy(dtype=float)
    y_arr = pd.to_numeric(gazedata_df["y"], errors="coerce").to_numpy(dtype=float)

    x_filt, y_filt = median_filter_xy(
        x_arr, y_arr, kernel_size=p.median_kernel, method=p.median_method
    )

    x_adj, y_adj = drift_adjust_to_screen_center(
        x_filt,
        y_filt,
        inner_width=inner_width,
        inner_height=inner_height,
        mode=drift_mode,
    )

    x_border, y_border, border_excluded_idx = border_exclusion(
        x_adj,
        y_adj,
        inner_width=inner_width,
        inner_height=inner_height,
    )
    t_border = np.delete(t_arr, border_excluded_idx, axis=0)

    assign = assign_quadrants_axis(
        x_border,
        y_border,
        inner_width=inner_width,
        inner_height=inner_height,
        method=p.quadrant_method,
        x_prob_threshold=p.x_prob_threshold,
        y_prob_threshold=p.y_prob_threshold,
        random_state=p.random_state,
        allow_midpoint_fallback=p.allow_midpoint_fallback,
    )
    t_used = t_border[assign.keep_mask]

    out_df = resample_to_bins(
        t_ms=t_used,
        window=assign.window,
        x=assign.x_kept,
        y=assign.y_kept,
        video_length_seconds=video_length_seconds,
        bin_seconds=p.bin_seconds,
        inner_width=inner_width,
        inner_height=inner_height,
        edge_policy=p.resample_edge_policy,
        include_endpoint=p.resample_include_endpoint,
    )

    # Notebook parity: exclusions were summarized as % bins with NaN x_res.
    prop_exclusion_pct = float(np.mean(out_df["x_res"].isna().to_numpy()) * 100.0)

    return PreprocessResult(
        frame=out_df,
        prop_exclusion_pct=prop_exclusion_pct,
        n_raw=int(len(gazedata_df)),
        n_after_border=int(len(x_border)),
        n_after_confidence=int(len(assign.window)),
        quadrant_method_used=assign.method_used,
    )

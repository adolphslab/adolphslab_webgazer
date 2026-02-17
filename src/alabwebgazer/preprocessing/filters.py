"""Filtering and coordinate-adjustment helpers for raw WebGazer streams."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd

try:
    # SciPy is a base dependency for this repo.
    from scipy.signal import medfilt  # type: ignore
except Exception:  # noqa: BLE001
    medfilt = None


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _odd_kernel(kernel_size: int) -> int:
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def median_filter_xy(
    x: Iterable[float],
    y: Iterable[float],
    *,
    kernel_size: int = 7,
    method: Literal["scipy_medfilt", "rolling"] = "scipy_medfilt",
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a median filter to x/y streams.

    Notebook parity note:
    - Historical notebook code uses ``scipy.signal.medfilt``.
    - We support a rolling-median fallback for environments without SciPy.
    """
    k = _odd_kernel(kernel_size)
    x_arr = _to_float_array(x)
    y_arr = _to_float_array(y)

    if k == 1:
        return x_arr, y_arr

    if method == "scipy_medfilt":
        if medfilt is None:
            raise RuntimeError("SciPy medfilt not available; use method='rolling'.")
        x_f = medfilt(x_arr, kernel_size=k).astype(float)
        y_f = medfilt(y_arr, kernel_size=k).astype(float)
        return x_f, y_f

    x_out = (
        pd.Series(x_arr)
        .rolling(window=k, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    y_out = (
        pd.Series(y_arr)
        .rolling(window=k, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    return x_out, y_out


def drift_adjust_to_screen_center(
    x: Iterable[float],
    y: Iterable[float],
    *,
    inner_width: float,
    inner_height: float,
    mode: Literal["mean", "median", "none"] = "mean",
) -> tuple[np.ndarray, np.ndarray]:
    """Center x/y onto screen center with configurable drift statistic.

    ``mode='mean'`` mirrors notebook ``Med_adjust`` behavior:
      x_adj = x - (mean(x) - inner_width/2)
      y_adj = y - (mean(y) - inner_height/2)
    """
    x_arr = _to_float_array(x)
    y_arr = _to_float_array(y)

    if mode == "none":
        return x_arr, y_arr

    if mode == "mean":
        x_center = np.nanmean(x_arr)
        y_center = np.nanmean(y_arr)
    elif mode == "median":
        x_center = np.nanmedian(x_arr)
        y_center = np.nanmedian(y_arr)
    else:
        raise ValueError(f"Unsupported drift mode: {mode}")

    x_adj = x_arr - (x_center - (inner_width / 2.0))
    y_adj = y_arr - (y_center - (inner_height / 2.0))
    return x_adj, y_adj


def mean_adjust_to_screen_center(
    x: Iterable[float],
    y: Iterable[float],
    *,
    inner_width: float,
    inner_height: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Backward-compatible alias for mean drift adjustment."""
    return drift_adjust_to_screen_center(
        x,
        y,
        inner_width=inner_width,
        inner_height=inner_height,
        mode="mean",
    )


def border_exclusion(
    x: Iterable[float],
    y: Iterable[float],
    *,
    inner_width: float,
    inner_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Drop out-of-bounds gaze coordinates.

    Returns
    -------
    x_kept, y_kept, excluded_idx
    """
    x_arr = _to_float_array(x)
    y_arr = _to_float_array(y)

    in_bounds = (
        np.isfinite(x_arr)
        & np.isfinite(y_arr)
        & (x_arr >= 0.0)
        & (x_arr <= inner_width)
        & (y_arr >= 0.0)
        & (y_arr <= inner_height)
    )
    excluded_idx = np.flatnonzero(~in_bounds)
    return x_arr[in_bounds], y_arr[in_bounds], excluded_idx

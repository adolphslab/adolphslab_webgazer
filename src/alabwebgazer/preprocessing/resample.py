"""Time-bin resampling for quadrant-labeled gaze streams."""

from __future__ import annotations

from typing import Iterable, Literal

import numpy as np
import pandas as pd


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _mode_int(values: np.ndarray) -> float:
    """Deterministic mode for small integer labels.

    In case of ties, returns the smallest label (consistent with SciPy's default
    behavior in typical settings).
    """
    if len(values) == 0:
        return float("nan")
    vals = values.astype(int, copy=False)
    uniq, counts = np.unique(vals, return_counts=True)
    return float(uniq[np.argmax(counts)])


def resample_to_bins(
    *,
    t_ms: Iterable[float],
    window: Iterable[float],
    x: Iterable[float],
    y: Iterable[float],
    video_length_seconds: float,
    bin_seconds: float = 0.5,
    inner_width: float,
    inner_height: float,
    edge_policy: Literal["notebook_strict", "half_open"] = "notebook_strict",
    include_endpoint: bool = True,
) -> pd.DataFrame:
    """Resample points into fixed-width time bins.

    Output columns mirror historical notebook outputs:
    ``Tstart``, ``Tend``, ``window``, ``x_res``, ``y_res``, ``x_ratio``, ``y_ratio``.

    Notes on edge handling
    ----------------------
    - ``notebook_strict`` matches the notebook's strict inequalities:
        (t > t0) & (t < t1)
      This can drop points exactly on bin edges.
    - ``half_open`` uses [t0, t1) which is generally easier to reason about.

    If ``include_endpoint`` is True, we extend the bin edges to include the final
    tail up to ``video_length_seconds``.
    """
    if bin_seconds <= 0:
        raise ValueError("bin_seconds must be > 0")
    if video_length_seconds <= 0:
        raise ValueError("video_length_seconds must be > 0")

    t_arr = _to_float_array(t_ms)
    w_arr = _to_float_array(window)
    x_arr = _to_float_array(x)
    y_arr = _to_float_array(y)

    if not (len(t_arr) == len(w_arr) == len(x_arr) == len(y_arr)):
        raise ValueError("t_ms/window/x/y must have equal length")

    stop = float(video_length_seconds + bin_seconds) if include_endpoint else float(video_length_seconds)
    time_bins = np.arange(0.0, stop, float(bin_seconds))

    t_s = t_arr / 1000.0

    start_all: list[float] = []
    end_all: list[float] = []
    window_all: list[float] = []
    x_all: list[float] = []
    y_all: list[float] = []

    for idx in range(len(time_bins) - 1):
        t0 = float(time_bins[idx])
        t1 = float(time_bins[idx + 1])

        if edge_policy == "notebook_strict":
            in_bin = (t_s > t0) & (t_s < t1)
        else:
            in_bin = (t_s >= t0) & (t_s < t1)

        w_bin = w_arr[in_bin]
        x_bin = x_arr[in_bin]
        y_bin = y_arr[in_bin]

        if len(w_bin) == 0:
            w_out = float("nan")
            x_out = float("nan")
            y_out = float("nan")
        else:
            finite_w = w_bin[np.isfinite(w_bin)]
            w_out = _mode_int(finite_w) if len(finite_w) > 0 else float("nan")
            x_out = float(np.nanmean(x_bin))
            y_out = float(np.nanmean(y_bin))

        start_all.append(t0)
        end_all.append(t1)
        window_all.append(w_out)
        x_all.append(x_out)
        y_all.append(y_out)

    out = pd.DataFrame(
        {
            "Tstart": start_all,
            "Tend": end_all,
            "window": window_all,
            "x_res": x_all,
            "y_res": y_all,
        }
    )
    out["x_ratio"] = out["x_res"] / float(inner_width)
    out["y_ratio"] = out["y_res"] / float(inner_height)
    return out

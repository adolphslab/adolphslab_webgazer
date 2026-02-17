"""Quadrant assignment helpers for processed x/y gaze streams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class QuadrantAssignment:
    """Assignment output for a single x/y stream."""

    x_kept: np.ndarray
    y_kept: np.ndarray
    window: np.ndarray
    keep_mask: np.ndarray
    method_used: str
    x_confidence: Optional[np.ndarray] = None
    y_confidence: Optional[np.ndarray] = None


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _midpoint_assign(x: np.ndarray, y: np.ndarray, *, inner_width: float, inner_height: float) -> np.ndarray:
    half_x = inner_width / 2.0
    half_y = inner_height / 2.0

    window = np.full(shape=len(x), fill_value=np.nan, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)

    tl = finite & (x < half_x) & (y < half_y)
    tr = finite & (x >= half_x) & (y < half_y)
    bl = finite & (x < half_x) & (y >= half_y)
    br = finite & (x >= half_x) & (y >= half_y)

    window[tl] = 1.0
    window[tr] = 2.0
    window[bl] = 3.0
    window[br] = 4.0
    return window


def _fit_gmm_labels(values: np.ndarray, *, random_state: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        from sklearn.mixture import GaussianMixture
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scikit-learn is required for quadrant_method='gmm_axis'") from exc

    arr = values.reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=random_state)
    gm.fit(arr)

    labels = gm.predict(arr)
    probs = gm.predict_proba(arr)

    # Enforce left/top as 0 and right/bottom as 1 by mean ordering.
    means = gm.means_.reshape(-1)
    if means[0] > means[1]:
        labels = 1 - labels
        probs = probs[:, ::-1]

    confidence = probs.max(axis=1)
    return labels.astype(int), confidence.astype(float)


def assign_quadrants_axis(
    x: Iterable[float],
    y: Iterable[float],
    *,
    inner_width: float,
    inner_height: float,
    method: Literal["gmm_axis", "midpoint"] = "gmm_axis",
    x_prob_threshold: float = 0.9,
    y_prob_threshold: float = 0.7,
    random_state: int = 0,
    allow_midpoint_fallback: bool = False,
) -> QuadrantAssignment:
    """Assign 4-quadrant labels (1..4) to x/y points.

    Methods
    -------
    - ``gmm_axis``: 2-component GMM on each axis; low-confidence points dropped.
    - ``midpoint``: deterministic half-screen split (always available).

    Fallback policy
    ---------------
    If ``method='gmm_axis'`` and the fit fails, the behavior is controlled by
    ``allow_midpoint_fallback``. By default it is False to avoid silent changes
    in the measurement step.

    Returns
    -------
    QuadrantAssignment with ``window`` labels and ``keep_mask`` indicating points
    retained after confidence filtering.
    """
    x_arr = _to_float_array(x)
    y_arr = _to_float_array(y)

    finite = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_kept = x_arr[finite]
    y_kept = y_arr[finite]

    if len(x_kept) == 0:
        return QuadrantAssignment(
            x_kept=x_kept,
            y_kept=y_kept,
            window=np.array([], dtype=float),
            keep_mask=np.array([], dtype=bool),
            method_used=method,
        )

    if method == "midpoint":
        window_mid = _midpoint_assign(x_kept, y_kept, inner_width=inner_width, inner_height=inner_height)
        keep = np.isfinite(window_mid)
        return QuadrantAssignment(
            x_kept=x_kept[keep],
            y_kept=y_kept[keep],
            window=window_mid[keep],
            keep_mask=keep,
            method_used="midpoint",
        )

    try:
        x_label, x_conf = _fit_gmm_labels(x_kept, random_state=random_state)
        y_label, y_conf = _fit_gmm_labels(y_kept, random_state=random_state)

        keep = (x_conf >= x_prob_threshold) & (y_conf >= y_prob_threshold)
        window = (1 + x_label + 2 * y_label).astype(float)

        return QuadrantAssignment(
            x_kept=x_kept[keep],
            y_kept=y_kept[keep],
            window=window[keep],
            keep_mask=keep,
            method_used="gmm_axis",
            x_confidence=x_conf,
            y_confidence=y_conf,
        )
    except Exception as exc:
        if not allow_midpoint_fallback:
            raise
        window_mid = _midpoint_assign(x_kept, y_kept, inner_width=inner_width, inner_height=inner_height)
        keep = np.isfinite(window_mid)
        return QuadrantAssignment(
            x_kept=x_kept[keep],
            y_kept=y_kept[keep],
            window=window_mid[keep],
            keep_mask=keep,
            method_used=f"midpoint_fallback:{exc.__class__.__name__}",
        )

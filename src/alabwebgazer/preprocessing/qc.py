"""QC helpers for jsPsych/WebGazer exports.

This module includes:
- general per-trial quality summaries used in notebook-style checks
- quadrant-image calibration QC (raw and preprocessed confusion matrices)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.signal import medfilt  # type: ignore
except Exception:  # noqa: BLE001
    medfilt = None

try:
    from sklearn.mixture import GaussianMixture  # type: ignore
except Exception:  # noqa: BLE001
    GaussianMixture = None


@dataclass(frozen=True)
class QuadrantImageQC:
    """Summary metrics for quadrant-image trials."""

    n_image_trials: int
    confusion_raw: Optional[List[List[float]]]
    confusion_preproc: Optional[List[List[float]]]
    raw_accuracy: Optional[float]
    preproc_accuracy: Optional[float]


def _to_float_array(values: Iterable[float]) -> np.ndarray:
    return pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)


def _odd_kernel(kernel_size: int) -> int:
    if kernel_size < 1:
        raise ValueError("kernel_size must be >= 1")
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1


def _median_filter_xy(x: np.ndarray, y: np.ndarray, kernel: int) -> Tuple[np.ndarray, np.ndarray]:
    k = _odd_kernel(kernel)
    if k == 1:
        return x.astype(float), y.astype(float)
    if medfilt is not None:
        return (
            medfilt(x.astype(float), kernel_size=k).astype(float),
            medfilt(y.astype(float), kernel_size=k).astype(float),
        )
    x_out = (
        pd.Series(x.astype(float))
        .rolling(window=k, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    y_out = (
        pd.Series(y.astype(float))
        .rolling(window=k, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )
    return x_out, y_out


def _drift_adjust_mean(
    x: np.ndarray,
    y: np.ndarray,
    half_x: float,
    half_y: float,
) -> Tuple[np.ndarray, np.ndarray]:
    cx = float(np.nanmean(x.astype(float)))
    cy = float(np.nanmean(y.astype(float)))
    return x.astype(float) - (cx - half_x), y.astype(float) - (cy - half_y)


def _border_keep_mask(x: np.ndarray, y: np.ndarray, inner_w: float, inner_h: float) -> np.ndarray:
    return (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= inner_w)
        & (y >= 0.0)
        & (y <= inner_h)
    )


def quad_acc_validation(
    raw_gaze: Sequence[Sequence[Mapping[str, Any]]],
) -> Tuple[List[List[float]], List[float], float]:
    """Compute a quadrant-level validation accuracy summary."""
    df_allpoints = pd.concat([pd.DataFrame(point) for point in raw_gaze], ignore_index=True)
    if df_allpoints.empty:
        return [[0.0, 0.0, 0.0, 0.0] for _ in range(4)], [0.0, 0.0, 0.0, 0.0], 0.0

    x_mid = float(np.mean(pd.to_numeric(df_allpoints.get("x"), errors="coerce")))
    y_mid = float(np.mean(pd.to_numeric(df_allpoints.get("y"), errors="coerce")))

    confusion_mat: List[List[float]] = []
    for point in raw_gaze:
        df_point = pd.DataFrame(point)
        if len(df_point) == 0:
            confusion_mat.append([0.0, 0.0, 0.0, 0.0])
            continue

        t0 = float(df_point["t"].iloc[0])
        df_point = df_point[pd.to_numeric(df_point["t"], errors="coerce") > (t0 + 200.0)]

        x = pd.to_numeric(df_point.get("x"), errors="coerce")
        y = pd.to_numeric(df_point.get("y"), errors="coerce")
        n = float(len(df_point)) if len(df_point) else 1.0

        tl = float(np.sum((x < x_mid) & (y < y_mid)) / n * 100.0)
        tr = float(np.sum((x > x_mid) & (y < y_mid)) / n * 100.0)
        bl = float(np.sum((x < x_mid) & (y > y_mid)) / n * 100.0)
        br = float(np.sum((x > x_mid) & (y > y_mid)) / n * 100.0)

        # Notebook row order: [TL, BL, TR, BR]
        confusion_mat.append([tl, bl, tr, br])

    diagonals = [confusion_mat[i][i] for i in range(4)]
    quad_acc = float(np.mean(diagonals))
    return confusion_mat, diagonals, quad_acc


def video_trial_qc_info(vid_trial: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute sampling and completeness metrics for a video trial."""
    df_gaze = pd.DataFrame(data=vid_trial.get("webgazer_data", []))
    if df_gaze.empty or "t" not in df_gaze.columns:
        return {
            "samp_int_avg_ms": np.nan,
            "samp_int_median_ms": np.nan,
            "samp_rate_avg_hz": np.nan,
            "samp_rate_median_hz": np.nan,
            "last_timepoint_min": np.nan,
            "vid_name": str(vid_trial.get("stimname", "missing")),
        }

    t = pd.to_numeric(df_gaze["t"], errors="coerce").to_numpy(dtype=float)
    if len(t) < 2:
        return {
            "samp_int_avg_ms": np.nan,
            "samp_int_median_ms": np.nan,
            "samp_rate_avg_hz": np.nan,
            "samp_rate_median_hz": np.nan,
            "last_timepoint_min": float(t[-1] / (60.0 * 1000.0)) if len(t) else np.nan,
            "vid_name": str(vid_trial.get("stimname", "missing")),
        }

    samp_int_all = np.diff(t)
    samp_int_avg = float(np.mean(samp_int_all))
    samp_int_median = float(np.median(samp_int_all))
    samp_rate_avg = float(1.0 / (samp_int_avg / 1000.0)) if samp_int_avg > 0 else np.nan
    samp_rate_median = float(1.0 / (samp_int_median / 1000.0)) if samp_int_median > 0 else np.nan
    last_timepoint_min = float(t[-1] / (60.0 * 1000.0))
    vid_name = str(vid_trial.get("stimname", "missing"))

    return {
        "samp_int_avg_ms": samp_int_avg,
        "samp_int_median_ms": samp_int_median,
        "samp_rate_avg_hz": samp_rate_avg,
        "samp_rate_median_hz": samp_rate_median,
        "last_timepoint_min": last_timepoint_min,
        "vid_name": vid_name,
    }


def find_quadrant_image_trials(
    trials: Sequence[Mapping[str, Any]],
    *,
    q1_pattern: str,
    q2_pattern: str,
    q3_pattern: str,
    q4_pattern: str,
) -> Dict[int, List[int]]:
    """Identify quadrant-image trials by stimulus-position patterns."""

    out: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}

    for idx, t in enumerate(trials):
        if "webgazer_data" not in t or "stimulus" not in t:
            continue
        stim = str(t.get("stimulus", ""))
        if q1_pattern in stim:
            out[1].append(idx)
        elif q2_pattern in stim:
            out[2].append(idx)
        elif q3_pattern in stim:
            out[3].append(idx)
        elif q4_pattern in stim:
            out[4].append(idx)

    return out


def quad_assign_raw(img_trial: Mapping[str, Any]) -> List[float]:
    """Assign quadrants from raw coordinates (no preprocessing)."""

    half_x = float(img_trial["innerWidth"]) / 2.0
    half_y = float(img_trial["innerHeight"]) / 2.0

    df_gaze = pd.DataFrame(img_trial.get("webgazer_data", []))
    if "t" in df_gaze.columns:
        df_gaze = df_gaze[df_gaze["t"] > 500]

    x_all = _to_float_array(df_gaze.get("x", pd.Series(dtype=float)))
    y_all = _to_float_array(df_gaze.get("y", pd.Series(dtype=float)))

    out: List[float] = []
    for x, y in zip(x_all, y_all):
        quad = float("nan")
        if (x < half_x) and (y < half_y):
            quad = 1.0
        elif (x > half_x) and (y < half_y):
            quad = 2.0
        elif (x < half_x) and (y > half_y):
            quad = 3.0
        elif (x > half_x) and (y > half_y):
            quad = 4.0
        elif (x < 0) or (y < 0) or (x > half_x * 2) or (y > half_y * 2):
            quad = float("nan")
        out.append(quad)
    return out


def quad_percentage(quad_array: List[float]) -> Tuple[List[float], int]:
    """Percent of samples in each quadrant for an array of labels."""

    n = int(len(quad_array))
    if n == 0:
        return [float("nan")] * 4, 0
    return [quad_array.count(float(i)) / n * 100.0 for i in (1, 2, 3, 4)], n


def confusion_raw(
    trials: Sequence[Mapping[str, Any]],
    idx_by_quad: Dict[int, List[int]],
) -> Optional[List[List[float]]]:
    """Compute weighted confusion matrix for raw quadrant assignments."""

    confusion: List[List[float]] = []
    any_found = False

    for true_quad in (1, 2, 3, 4):
        quad_perc_all: List[List[float]] = []
        n_all: List[int] = []

        for trial_idx in idx_by_quad.get(true_quad, []):
            t = trials[trial_idx]
            if not t.get("webgazer_data"):
                continue
            any_found = True
            q_raw = quad_assign_raw(t)
            q_perc, n = quad_percentage(q_raw)
            if n > 0:
                quad_perc_all.append(q_perc)
                n_all.append(n)

        if sum(n_all) > 0:
            confusion.append(
                list(
                    np.average(
                        np.asarray(quad_perc_all),
                        axis=0,
                        weights=np.asarray(n_all),
                    )
                )
            )
        else:
            confusion.append([float("nan")] * 4)

    return confusion if any_found else None


def _accuracy_from_confusion(conf: Optional[List[List[float]]]) -> Optional[float]:
    if conf is None:
        return None
    arr = np.asarray(conf, dtype=float)
    if arr.shape != (4, 4):
        return None
    diag = np.diag(arr)
    if np.all(~np.isfinite(diag)):
        return None
    return float(np.nanmean(diag) / 100.0)


def confusion_preproc(
    trials: Sequence[Mapping[str, Any]],
    idx_by_quad: Dict[int, List[int]],
    *,
    median_kernel: int,
    x_prob_thres: float,
    y_prob_thres: float,
    random_state: int,
) -> Optional[List[List[float]]]:
    """Compute confusion matrix after GMM-based preprocessing."""

    if GaussianMixture is None:
        return None

    rows: List[pd.DataFrame] = []
    any_found = False

    for true_quad in (1, 2, 3, 4):
        for trial_idx in idx_by_quad.get(true_quad, []):
            t = trials[trial_idx]
            if not t.get("webgazer_data"):
                continue
            any_found = True
            df = pd.DataFrame(t.get("webgazer_data", []))
            if "t" in df.columns:
                df = df[df["t"] > 500]
            df = df.copy()
            df["actual_quadrant"] = int(true_quad)
            rows.append(df)

    if not any_found:
        return None

    df_all = pd.concat(rows, axis=0, ignore_index=True)
    if df_all.empty:
        return None

    first_trial_idx = None
    for q in (1, 2, 3, 4):
        if idx_by_quad.get(q):
            first_trial_idx = idx_by_quad[q][0]
            break
    if first_trial_idx is None:
        return None

    inner_w = float(trials[first_trial_idx].get("innerWidth", 0.0) or 0.0)
    inner_h = float(trials[first_trial_idx].get("innerHeight", 0.0) or 0.0)
    if inner_w <= 0 or inner_h <= 0:
        return None

    x = _to_float_array(df_all.get("x", pd.Series(dtype=float)))
    y = _to_float_array(df_all.get("y", pd.Series(dtype=float)))

    x_f, y_f = _median_filter_xy(x, y, kernel=median_kernel)
    x_adj, y_adj = _drift_adjust_mean(x_f, y_f, inner_w / 2.0, inner_h / 2.0)

    keep = _border_keep_mask(x_adj, y_adj, inner_w, inner_h)
    x_k = x_adj[keep]
    y_k = y_adj[keep]
    actual_k = df_all.loc[keep, "actual_quadrant"].to_numpy(dtype=int)

    if len(x_k) == 0:
        return None

    x_r = x_k.reshape(-1, 1)
    y_r = y_k.reshape(-1, 1)

    gm_x = GaussianMixture(n_components=2, random_state=int(random_state)).fit(x_r)
    gm_y = GaussianMixture(n_components=2, random_state=int(random_state)).fit(y_r)

    x_pred = gm_x.predict(x_r)
    y_pred = gm_y.predict(y_r)

    if float(gm_x.means_[0]) > float(gm_x.means_[1]):
        x_pred = 1 - x_pred
    if float(gm_y.means_[0]) > float(gm_y.means_[1]):
        y_pred = 1 - y_pred

    window = 1 + 1 * x_pred + 2 * y_pred

    max_px = np.max(gm_x.predict_proba(x_r), axis=1)
    max_py = np.max(gm_y.predict_proba(y_r), axis=1)
    keep2 = (max_px >= float(x_prob_thres)) & (max_py >= float(y_prob_thres))

    if not np.any(keep2):
        return None

    window2 = window[keep2]
    actual2 = actual_k[keep2]

    conf: List[List[float]] = []
    for q in (1, 2, 3, 4):
        sel = actual2 == q
        if not np.any(sel):
            conf.append([float("nan")] * 4)
            continue
        vals = window2[sel].astype(int)
        row = [float(np.mean(vals == k) * 100.0) for k in (1, 2, 3, 4)]
        conf.append(row)

    return conf


def compute_quadrant_image_qc(
    trials: Sequence[Mapping[str, Any]],
    *,
    q1_pattern: str,
    q2_pattern: str,
    q3_pattern: str,
    q4_pattern: str,
    median_kernel: int = 5,
    x_prob_thres: float = 0.9,
    y_prob_thres: float = 0.7,
    random_state: int = 0,
) -> QuadrantImageQC:
    """Compute quadrant-image QC metrics from a full jsPsych trial list."""

    idx = find_quadrant_image_trials(
        list(trials),
        q1_pattern=q1_pattern,
        q2_pattern=q2_pattern,
        q3_pattern=q3_pattern,
        q4_pattern=q4_pattern,
    )
    n_trials = int(sum(len(v) for v in idx.values()))

    conf_r = confusion_raw(list(trials), idx)
    conf_p = confusion_preproc(
        list(trials),
        idx,
        median_kernel=median_kernel,
        x_prob_thres=x_prob_thres,
        y_prob_thres=y_prob_thres,
        random_state=random_state,
    )

    return QuadrantImageQC(
        n_image_trials=n_trials,
        confusion_raw=conf_r,
        confusion_preproc=conf_p,
        raw_accuracy=_accuracy_from_confusion(conf_r),
        preproc_accuracy=_accuracy_from_confusion(conf_p),
    )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dwell-time computation with availability (raw/valid), chance, enrichment,
retention, QC, and optional boundary-aware confidence weighting (BAW).

This version performs:
  * Naive (unweighted) dwell metrics
  * Optional boundary-aware confidence weighting (BAW), applied post-hoc
    to valid bins using the boundary_distance_weight() function.
  - post-hoc boundary-aware metrics (suffix `_baw`), using per-bin weights
   derived from distance to the vertical/horizontal midlines.

Per (subject, condition, feature) we compute:
  * coverage / missingness (finite, valid, offscreen, tail dropout)
  * availability (present-any RAW/VALID)
  * chance baselines (unweighted and BAW-weighted)
  * dwell metrics and enrichment vs chance
  * retention RAW -> VALID and conditional on feature presence
  * feature-dependent missingness (present vs absent)
  * boundary diagnostics
  * rich gaze QC:
      - flips, flip_rate_per_min, flip_rate_per_min_valid
      - TL/TR/BL/BR base rates and entropy
      - early/late and quartile y_valid fractions
      - per-quadrant counts among y-valid
      - median_run_len
      - flicker_index (same as singleton_run_frac), abab_rate, short_run_share_lt3
      - switch_rate, diagonal_rate
  * BAW diagnostics (ESS, top-1% weight share)

Notes
-----
- RAW video grid:
    x_feature_len = min length across the 4 quadrant feature streams.
    If lengths differ, we log a WARNING and truncate to the minimum.

- Gaze labels:
    y_slice is WebGazer "window" labels truncated/padded to length T.
    Valid gaze is y in {1,2,3,4}. Late-video dropout stays as missingness;
    we do NOT truncate at last finite window.

- Features:
    Feature NaNs are treated as "no feature":
      * Presence tests only count finite values >= presence_threshold.
      * Attended-feature NaNs remain but never count as present.

- Boundary-aware weighting:
    * boundary_mode = "posthoc":
         boundary weights are used in BAW-adjusted metrics
    * boundary_mode = "off":
         boundary weighting disabled in metrics (diagnostic cols still reported)

  - min_run is applied over contiguous valid frames on the RAW timeline:
      two valid frames separated by missing bins do NOT belong to the same run
      (we enforce t[i] - t[i-1] <= 1 using t_valid).
"""

import argparse
import json
import logging
import pickle
from dataclasses import dataclass, asdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# ============================== Config ======================================

@dataclass(frozen=True)
class RunConfig:
    # IO
    video_feats_dir: str
    window_data_dir: str
    output_root: str
    task: str  # e.g., task4_pro / task5_spark / task5_lab / task5_tobii
    window_glob: str = "videoview*.csv"

    # Optional metadata
    subject_metadata_csv: Optional[str] = None
    runlist_csv: Optional[str] = None

    # Core behavior
    presence_threshold: float = 0.5
    min_run: int = 1
    bin_seconds: Optional[float] = None

    # Derived / joint-feature endpoints (additive; does not change base metrics)
    include_joint_features: bool = True
    derived_require_speaker_present: bool = True
    # attention to off-speaker/on-speaker distraction given that exactly one speaker is labeled in the bin.
    derived_require_speaker_unique: bool = False # keep False as default. 

    # Task5 speaker-gaze splits: if True, only compute speaker_averted/speaker_not_averted
    # when there is exactly one speaker quadrant in the bin (recommended).
    task5_require_speaker_unique: bool = True

    # Sensitivity (Task5 only): if True, also emit `speaker_not_averted_loose`,
    # which treats NaN/unknown `averted` on the speaker quadrant as "not averted".
    # Keep this OFF for primary analyses to avoid analytic degrees of freedom.
    include_speaker_not_averted_loose: bool = False

    # Boundary-aware confidence weighting
    boundary_mode: str = "posthoc"  # {"posthoc", "off"}
    boundary_tau: float = 0.05
    boundary_power: float = 1.0

    # Debug / logging
    save_bin_level: bool = False
    random_state: int = 12345
    loglevel: str = "INFO"


# Spatial ordering TL,TR,BL,BR (for diagonal switch stats and quadrant base rates)
GRID_MAP_DEFAULT: Tuple[int, int, int, int] = (1, 2, 3, 4)


# ============================ Feature naming ================================

FEAT_NAMES_TASK4 = [
    "speaker", "distraction",
    "low_color", "low_intensity", "low_motion", "low_orientation",
    "pyfeat_facesize", "pyfeat_AU1", "pyfeat_AU2", "pyfeat_AU3", "pyfeat_AU4", "pyfeat_AU5",
    "pyfeat_parallel", "pyfeat_rotation", "pyfeat_depth",
    "densepose_hand",
    "mfcc1", "mfcc2", "mfcc3", "mfcc4",
]
FEAT_NAMES_TASK5 = ["speaker", "distraction", "averted"] + FEAT_NAMES_TASK4[2:]


def base_feat_names_for_task(task: str) -> List[str]:
    return FEAT_NAMES_TASK5 if task.startswith("task5_") else FEAT_NAMES_TASK4


def features_of_interest_for_task(task: str) -> List[str]:
    feats = ["speaker", "distraction"]
    if task.startswith("task5_"):
        feats.append("averted")
    return feats




# -------------------------- Derived / joint features --------------------------

DERIVED_FEATS_TASK4 = [
    # Distraction split relative to speaker quadrant
    "distraction_offspeaker",
    "distraction_onspeaker",
]

# Task5 adds speaker-conditioned gaze-direction states (on the speaker quadrant).
DERIVED_FEATS_TASK5_BASE = DERIVED_FEATS_TASK4 + [
    "speaker_averted",
    # Strict complement: requires finite/annotated averted value on the speaker quadrant.
    # (interpret as: "speaker_not_averted_annotated", not necessarily "direct gaze".)
    "speaker_not_averted",
]

# Optional sensitivity endpoint (off by default): treat NaN/unknown averted as "not averted".
DERIVED_FEATS_TASK5_LOOSE = DERIVED_FEATS_TASK5_BASE + [
    "speaker_not_averted_loose",
]


def derived_features_for_task(task: str, include_loose: bool = False) -> List[str]:
    """Return derived feature names to compute for this task."""
    if str(task).startswith("task5_"):
        feats = list(DERIVED_FEATS_TASK5_BASE)
        if include_loose:
            feats.append("speaker_not_averted_loose")
        return feats
    return list(DERIVED_FEATS_TASK4)


def build_derived_feature_matrices(
    *,
    task: str,
    X_feature: List[np.ndarray],
    T: int,
    name_to_idx: Dict[str, int],
    presence_threshold: float,
    require_speaker_present: bool = True,
    require_speaker_unique: bool = False,
    task5_require_speaker_unique: bool = True,
    include_speaker_not_averted_loose: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Build additive, joint-feature matrices aligned to the RAW timeline.

    Each returned matrix is shape (T, 4) with float values:
      - values >= presence_threshold indicate the derived feature is present
      - values  < presence_threshold indicate absent
      - For most derived features we emit only finite values (present/absent).
      - For strict Task5 speaker-gaze splits (speaker_averted / speaker_not_averted),
        we set NaN on the *speaker quadrant* in bins where the speaker is in-gate
        but the upstream `averted` annotation is missing. This marks those bins as
        undefined for that endpoint (so attended_ok excludes them) while missingness
        is still quantified via the speaker-label diagnostics.

    IMPORTANT DESIGN CHOICE (robustness):
      - Distraction splits are computed only if both (speaker, distraction) exist.
      - Task5 speaker-gaze splits are computed only if both (speaker, averted) exist.
      - These blocks are *independent*, so Task5 endpoints do not accidentally
        disappear when 'distraction' is not part of a task's feature set.

    Derived endpoints included (when inputs exist)
    ---------------------------------------------

    Task4 + Task5 (speaker-relative distraction):
      * distraction_offspeaker: distraction ∧ ¬speaker (same bin), optionally gated
        to bins with a speaker present / uniquely defined.
      * distraction_onspeaker:  distraction ∧ speaker (same bin), same gating.

    Task5 only (speaker gaze-direction via joint features):
      * speaker_averted:      speaker ∧ averted (strict; requires finite averted on
        the speaker quadrant; NaN if missing).
      * speaker_not_averted:  speaker ∧ ¬averted (strict; requires finite averted on
        the speaker quadrant; NaN if missing).
      * speaker_not_averted_loose: speaker ∧ ¬averted, but treats NaN/unknown averted
        as "not averted" (sensitivity only).

    Notes on gating
    ---------------
    - For distraction_* endpoints, `require_speaker_present=True` is recommended so
      "off-speaker" is interpreted relative to bins where a speaker exists.
    - For Task5 speaker_* endpoints, `task5_require_speaker_unique=True` is recommended
      so the comparison is interpretable as *the* speaker.
    """
    derived: Dict[str, np.ndarray] = {}

    if T <= 0 or ("speaker" not in name_to_idx):
        return derived

    th = float(presence_threshold)
    if not np.isfinite(th):
        raise ValueError(f"presence_threshold must be finite; got {presence_threshold!r}")

    # Encode derived features so the script's generic '>= presence_threshold' presence test works
    # for any (reasonable) threshold value.
    val_present = th
    val_absent = th - 1.0

    # --- Speaker presence / uniqueness (used for gating and Task5 conditioning) ---
    sp_idx = name_to_idx["speaker"]
    speaker = np.column_stack([X_feature[q][:T, sp_idx] for q in range(4)]).astype(float, copy=False)
    speaker_pres = np.isfinite(speaker) & (speaker >= th)  # (T,4) bool

    speaker_count = speaker_pres.sum(axis=1)
    speaker_any = (speaker_count > 0)      # (T,) bool
    speaker_unique = (speaker_count == 1)  # (T,) bool

    # --- Distraction split relative to speaker quadrant (Task4+Task5) ---
    if "distraction" in name_to_idx:
        ds_idx = name_to_idx["distraction"]
        distract = np.column_stack([X_feature[q][:T, ds_idx] for q in range(4)]).astype(float, copy=False)
        distraction_pres = np.isfinite(distract) & (distract >= th)

        gate = np.ones(T, dtype=bool)
        if require_speaker_present:
            gate &= speaker_any
        if require_speaker_unique:
            gate &= speaker_unique
        gate4 = gate[:, None]

        off_present = gate4 & distraction_pres & (~speaker_pres)
        on_present = gate4 & distraction_pres & speaker_pres

        derived["distraction_offspeaker"] = np.where(off_present, val_present, val_absent).astype(float)
        derived["distraction_onspeaker"] = np.where(on_present, val_present, val_absent).astype(float)

    # --- Task5: speaker gaze direction via joint features (speaker quadrant only) ---
    if task.startswith("task5_") and ("averted" in name_to_idx):
        av_idx = name_to_idx["averted"]
        av = np.column_stack([X_feature[q][:T, av_idx] for q in range(4)]).astype(float, copy=False)

        av_finite = np.isfinite(av)
        av_pres = av_finite & (av >= th)
        av_abs = av_finite & (av < th)

        # Speaker gating for Task5 endpoints: always require speaker_any; usually require unique speaker.
        gate_t5 = speaker_any.copy()
        # Task5 gating is intentionally independent of `require_speaker_unique` (distraction split).
        if task5_require_speaker_unique:
            gate_t5 &= speaker_unique

        speaker_t5 = speaker_pres & gate_t5[:, None]  # (T,4) True only on speaker quadrants in-gate

        # speaker_averted: speaker ∧ averted (strict; NaN where speaker but averted is missing)
        sp_av = np.where(speaker_t5, np.where(av_pres, val_present, val_absent), val_absent).astype(float)
        sp_av[speaker_t5 & (~av_finite)] = np.nan
        derived["speaker_averted"] = sp_av

        # speaker_not_averted: speaker ∧ ¬averted (strict; NaN where speaker but averted is missing)
        sp_not = np.where(speaker_t5, np.where(av_abs, val_present, val_absent), val_absent).astype(float)
        sp_not[speaker_t5 & (~av_finite)] = np.nan
        derived["speaker_not_averted"] = sp_not
        # speaker_not_averted_loose: treat NaN/unknown averted as "not averted" (sensitivity only)
        if include_speaker_not_averted_loose:
            sp_not_loose = np.where(speaker_t5, np.where(~av_pres, val_present, val_absent), val_absent).astype(float)
            derived["speaker_not_averted_loose"] = sp_not_loose

    return derived
def require_feature_mapping(task: str, X_feature: List[np.ndarray], base_names: List[str]) -> List[str]:
    """
    Sanity-check feature layout and ensure required names exist.
    """
    k_cols = [arr.shape[1] for arr in X_feature]
    if len(set(k_cols)) != 1:
        raise ValueError(
            f"Quadrant feature arrays have different column counts: {k_cols}. "
            "All four quadrants must share identical feature layout."
        )
    k_base = k_cols[0]
    if len(base_names) != k_base:
        raise ValueError(
            f"Feature-name mismatch: feature arrays have {k_base} columns, "
            f"but task map has {len(base_names)} names."
        )
    if len(set(base_names)) != len(base_names):
        dupes = sorted([n for n in set(base_names) if base_names.count(n) > 1])
        raise ValueError(f"Duplicate feature names in base map: {dupes}")
    required = features_of_interest_for_task(task)
    missing = sorted(set(required) - set(base_names))
    if missing:
        raise ValueError(f"Required feature(s) missing for {task}: {missing}")
    return required


# ============================== IO & parsing ================================

def _read_csv(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, engine="pyarrow")
    except Exception:
        return pd.read_csv(p)


def parse_subject_and_condition(csv_path: Path, task: str) -> Tuple[str, str]:
    """Parse participant ID and condition from a window CSV filename.

    Historical files often look like:
        videoview_window_{ID}_{session}_{condition}.csv

    Where `{session}` may contain underscores. To reduce brittle parsing, we:
    - if the stem starts with `videoview_window_`, take ID=parts[2] and condition=last part
      (treating the middle as session tokens).
    - otherwise fall back to task-specific index patterns for backward compatibility.
    """
    parts = csv_path.stem.split("_")

    # Some preprocess configs append `_trial{index}` to avoid collisions.
    # Strip that suffix token before condition parsing so Stage 2 still maps
    # feature_4quad_{condition}.pkl correctly.
    if len(parts) >= 2:
        last = parts[-1].lower()
        if last.startswith("trial") and last[5:].isdigit():
            parts = parts[:-1]

    # Robust common pattern:
    # videoview_window_{ID}_{...session...}_{condition}
    if len(parts) >= 4 and parts[0] == "videoview" and parts[1] == "window":
        subject_id = parts[2]
        condition = parts[-1]
        return subject_id, condition

    patterns: Dict[str, Tuple[int, List[int]]] = {
        "task4_pro":   (2, [4, 5]),
        "task4_spark": (2, [5, 6]),
        "task4_lab":   (2, [4, 5]),
        "task5_pro":   (2, [4]),
        "task5_spark": (2, [4]),
        "task5_prospark": (2, [4]),
        "task5_lab":   (2, [3]),
        "task5_tobii": (2, [3]),
    }
    id_idx, cond_idxs = patterns.get(task, (2, [4, 5]))
    subject_id = parts[id_idx] if id_idx < len(parts) else "UNKNOWN"
    cond_parts = [parts[i] for i in cond_idxs if i < len(parts)]
    condition = "_".join(cond_parts) if cond_parts else "UNKNOWN_COND"
    return subject_id, condition


def load_feature_pkl(features_dir: Path, condition: str) -> List[np.ndarray]:
    """
    Load per-quadrant feature arrays and warn if lengths differ.

    Returns
    -------
    X_feature : list of 4 np.ndarray
        Each array is shape (T_q, k). We do not force equal T_q here;
        downstream alignment is done via align_labels_and_features,
        but we emit a warning for diagnostics.
    """
    pkl_path = features_dir / f"feature_4quad_{condition}.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        X_feature: List[np.ndarray] = pickle.load(f)
    if not isinstance(X_feature, list) or len(X_feature) != 4:
        raise ValueError(f"{pkl_path.name}: expected list of 4 (T,k) arrays.")

    quad_lens = [arr.shape[0] for arr in X_feature]
    if len(set(quad_lens)) != 1:
        min_len = int(min(quad_lens))
        logging.warning(
            "[%s] quadrant feature length mismatch: %s; "
            "metrics will be truncated to min length (%d).",
            pkl_path.name, quad_lens, min_len,
        )
    return X_feature


def maybe_read_subject_meta(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read subject-level metadata and normalize key columns.

    Expected (case-insensitive):
      - ID  : participant id
      - group : group label (e.g., Controls / ASD / NonASD_Psych / SPARK)
      - pool  : optional pool label

    We normalize to:
      - 'ID' (capitalized)
      - 'group', 'pool' (lower-case)
    """
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        if "id" not in cols:
            logging.warning("subject_metadata_csv missing an 'ID' column; ignoring metadata.")
            return None

        # Canonicalize ID/group/pool column names (case-insensitive).
        if cols["id"] != "ID":
            if "ID" in df.columns:
                # merge then drop duplicate
                df["ID"] = df["ID"].astype(str).combine_first(df[cols["id"]].astype(str))
                df = df.drop(columns=[cols["id"]])
            else:
                df = df.rename(columns={cols["id"]: "ID"})

        if "group" in cols and cols["group"] != "group":
            if "group" in df.columns:
                df["group"] = df["group"].combine_first(df[cols["group"]])
                df = df.drop(columns=[cols["group"]])
            else:
                df = df.rename(columns={cols["group"]: "group"})

        if "pool" in cols and cols["pool"] != "pool":
            if "pool" in df.columns:
                df["pool"] = df["pool"].combine_first(df[cols["pool"]])
                df = df.drop(columns=[cols["pool"]])
            else:
                df = df.rename(columns={cols["pool"]: "pool"})

        if "group" not in df.columns:
            logging.warning(
                "subject_metadata_csv has no 'group' column; groups will come only from runlist if provided."
            )
        if "pool" not in df.columns:
            logging.info("subject_metadata_csv has no 'pool' column; 'Pool' will be omitted from outputs.")

        # Precompute normalized join key for faster, consistent lookups downstream.
        if "ID" in df.columns:
            # Always recompute for consistency (avoid stale keys from previous preprocessing steps)
            df["_id_key"] = df["ID"].astype(str).map(_norm_key)

        return df
    except Exception as e:
        logging.warning("Failed to read subject_metadata_csv: %s", e)
        return None
def maybe_read_runlist(csv_path: Optional[str]) -> Optional[pd.DataFrame]:
    """Read per-run metadata (runlist) and normalize key columns.

    Join keys (case-insensitive):
      - ID
      - condition  (or Video, used as fallback)

    We normalize to canonical columns used downstream:
      - ID, condition
      - Group, Video, AQ, SRS, Sex, Age  (added as NaN if missing)

    This avoids silent metadata loss when the CSV uses different capitalization
    (e.g., 'group' instead of 'Group').
    """
    if not csv_path:
        return None
    try:
        df = pd.read_csv(csv_path)
        cols = {c.lower(): c for c in df.columns}

        if "id" not in cols:
            logging.warning("runlist_csv missing an 'ID' column; ignoring runlist.")
            return None

        # Canonicalize ID
        if cols["id"] != "ID":
            if "ID" in df.columns:
                df["ID"] = df["ID"].astype(str).combine_first(df[cols["id"]].astype(str))
                df = df.drop(columns=[cols["id"]])
            else:
                df = df.rename(columns={cols["id"]: "ID"})

        # Canonicalize / create condition join column
        cols = {c.lower(): c for c in df.columns}  # refresh after renames
        if "condition" in cols:
            if cols["condition"] != "condition":
                if "condition" in df.columns:
                    df["condition"] = df["condition"].astype(str).combine_first(df[cols["condition"]].astype(str))
                    df = df.drop(columns=[cols["condition"]])
                else:
                    df = df.rename(columns={cols["condition"]: "condition"})
        elif "video" in cols:
            join_col = cols["video"]  # e.g. "Video" / "video"
            df = df.rename(columns={join_col: "condition"})
            # Preserve a 'Video' column for reporting, if not present
            if "Video" not in df.columns:
                df["Video"] = df["condition"]
        else:
            logging.warning("runlist_csv missing 'condition' or 'Video'; ignoring runlist.")
            return None

        # Canonicalize common metadata columns (case-insensitive) to expected names
        canon = {
            "group": "Group",
            "video": "Video",
            "aq": "AQ",
            "srs": "SRS",
            "sex": "Sex",
            "age": "Age",
        }
        cols = {c.lower(): c for c in df.columns}
        for low, target in canon.items():
            if low in cols:
                orig = cols[low]
                if orig != target:
                    if target in df.columns:
                        df[target] = df[target].combine_first(df[orig])
                        df = df.drop(columns=[orig])
                    else:
                        df = df.rename(columns={orig: target})

        # Ensure required cols exist
        for c in ("Group", "Video", "AQ", "SRS", "Sex", "Age"):
            if c not in df.columns:
                df[c] = np.nan

        # If Video is missing/blank, fall back to condition for reporting + feature lookup.
        # (Some CSVs contain the literal string "nan" rather than a real NaN.)
        if "condition" in df.columns:
            if "Video" not in df.columns:
                df["Video"] = df["condition"]
            else:
                df["Video"] = df["Video"].combine_first(df["condition"])
                vid = df["Video"].astype(str)
                cond = df["condition"].astype(str)
                m = vid.str.strip().eq("") | vid.str.lower().isin({"nan", "none", "null"})
                if m.any():
                    df.loc[m, "Video"] = cond.loc[m]

        # Precompute normalized join keys for faster lookups downstream.
        if "ID" in df.columns:
            df["_id_key"] = df["ID"].astype(str).map(_norm_key)
        if "condition" in df.columns:
            df["_cond_key"] = df["condition"].astype(str).map(_norm_key)

        return df
    except Exception as e:
        logging.warning("Failed to read runlist_csv: %s", e)
        return None

# ============================== Utilities ===================================

def run_length_filter_contig(
    b: np.ndarray,
    t_idx: np.ndarray,
    min_run: int,
    max_gap: int | None = None,
    use_time_span: bool = False,
) -> np.ndarray:
    """
    Keep runs of True in b where:
      - consecutive True indices satisfy t_idx[i] - t_idx[i_prev_true] <= max_gap
        (if max_gap is not None)
      - run length >= min_run, where length is:
          * number of indices if use_time_span is False
          * t_idx[end] - t_idx[start] + 1 if use_time_span is True
    """
    b = np.asarray(b, dtype=bool)
    t_idx = np.asarray(t_idx)
    n = b.size

    if n == 0:
        return np.zeros(0, dtype=bool)

    # For trivial runs, just return b
    if min_run <= 1 and max_gap is None:
        return b.copy()

    out = np.zeros(n, dtype=bool)
    run_start = None
    last_true_t = None

    def commit_run(end_idx: int):
        nonlocal run_start
        if run_start is None:
            return
        if use_time_span:
            length = t_idx[end_idx] - t_idx[run_start] + 1
        else:
            length = end_idx - run_start + 1
        if length >= min_run:
            out[run_start:end_idx + 1] = True
        run_start = None

    for i in range(n):
        if b[i]:
            if run_start is None:
                # start new run
                run_start = i
                last_true_t = t_idx[i]
            else:
                # check time gap between consecutive True indices
                if max_gap is not None and (t_idx[i] - last_true_t) > max_gap:
                    # close previous run at i-1
                    commit_run(i - 1)
                    # start new run at i
                    run_start = i
                last_true_t = t_idx[i]
        else:
            # end of a run
            if run_start is not None:
                commit_run(i - 1)

    # handle run reaching the end
    if run_start is not None:
        commit_run(n - 1)

    return out


def _safe_div(num: float | int, den: float | int) -> float:
    den = float(den)
    return float(num / den) if den > 0 else np.nan


def _chance(feat_mat: np.ndarray, thr: float) -> float:
    """Unweighted mean_t(#present_quads / 4)."""
    if feat_mat.size == 0:
        return np.nan
    present = (np.isfinite(feat_mat) & (feat_mat >= thr))
    return float(np.mean(present.sum(axis=1) / 4.0))


def _chance_weighted(feat_mat: np.ndarray, thr: float, w: np.ndarray) -> float:
    """Weighted mean_t(#present_quads / 4) under weights w (len = T)."""
    assert feat_mat.shape[0] == w.size, "Weight length mismatch in _chance_weighted"

    mask = w > 0
    if feat_mat.size == 0 or not np.any(mask):
        return np.nan
    present = (np.isfinite(feat_mat[mask]) & (feat_mat[mask] >= thr))
    per_t = present.sum(axis=1).astype(float) / 4.0
    num = float(np.sum(w[mask] * per_t))
    den = float(np.sum(w[mask]))
    return num / den if den > 0 else np.nan


# ------------------------ sequence QC utilities ------------------------

def contiguous_flags(t_idx: np.ndarray) -> np.ndarray:
    """contig[i]=True if t[i] == t[i-1] + 1 (for i>0)."""
    n = t_idx.size
    contig = np.zeros(n, dtype=bool)
    if n > 1:
        contig[1:] = (t_idx[1:] == (t_idx[:-1] + 1))
    return contig


def run_lengths_contig(y: np.ndarray, t_idx: np.ndarray) -> List[int]:
    """Run lengths of identical labels across contiguous steps only."""
    n = y.size
    if n == 0:
        return []
    contig = contiguous_flags(t_idx)
    runs: List[int] = []
    run = 1
    for i in range(1, n):
        if contig[i] and (y[i] == y[i - 1]):
            run += 1
        else:
            runs.append(run)
            run = 1
    runs.append(run)
    return runs


def flicker_index(y: np.ndarray, t_idx: np.ndarray) -> float:
    runs = run_lengths_contig(y, t_idx)
    return np.nan if len(runs) == 0 else float(np.sum(np.array(runs) == 1) / len(runs))


def short_run_share(y: np.ndarray, t_idx: np.ndarray, m: int) -> float:
    runs = run_lengths_contig(y, t_idx)
    return np.nan if len(runs) == 0 else float(np.sum(np.array(runs) < m) / len(runs))


def abab_rate(y: np.ndarray, t_idx: np.ndarray) -> float:
    """Share of ABAB alternations over 3-step contiguous windows."""
    n = y.size
    contig = contiguous_flags(t_idx)
    if n < 3:
        return np.nan
    cnt = 0
    tot = 0
    for i in range(2, n):
        if contig[i] and contig[i - 1]:
            tot += 1
            if (y[i] == y[i - 2]) and (y[i] != y[i - 1]):
                cnt += 1
    return np.nan if tot == 0 else float(cnt / tot)


def quad_grid_pairs(order: Tuple[int, int, int, int]) -> Tuple[set[Tuple[int, int]], set[Tuple[int, int]]]:
    """
    order is (qTL, qTR, qBL, qBR) in 1-based labels.
    Returns (adjacent_pairs, diagonal_pairs) in 0-based indices.
    """
    qTL, qTR, qBL, qBR = [q - 1 for q in order]
    adjacent = {
        (qTL, qTR), (qTR, qTL),
        (qBL, qBR), (qBR, qBL),
        (qTL, qBL), (qBL, qTL),
        (qTR, qBR), (qBR, qTR),
    }
    diagonal = {(qTL, qBR), (qBR, qTL), (qTR, qBL), (qBL, qTR)}
    return adjacent, diagonal


def switch_stats(
    y: np.ndarray,
    t_idx: np.ndarray,
    diag_pairs: set[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """
    Switching metrics on a 0-based quadrant label sequence y observed at raw indices t_idx.

    Key design choice:
      - switch_rate is computed CONDITIONAL on contiguity, i.e., only over steps where
        successive valid observations are adjacent in raw time (no gaps).
      - This avoids missingness-induced deflation of switching (common in WebGazer).

    Returns
    -------
    dict with:
      - switch_rate      : P(switch | contig_step)
      - diagonal_rate    : P(diagonal | switch & contig_step)  (if diag_pairs provided)
      - n_contig_steps   : number of contiguous steps (denominator for switch_rate)
      - n_switches       : number of observed switches on contiguous steps
      - contig_rate      : P(contig_step) among all steps (n-1)
      - gap_rate         : 1 - contig_rate
    """
    y = np.asarray(y)
    t_idx = np.asarray(t_idx)

    n = int(y.size)
    if n <= 1:
        return {
            "switch_rate": np.nan,
            "diagonal_rate": np.nan,
            "n_contig_steps": 0,
            "n_switches": 0,
            "contig_rate": np.nan,
            "gap_rate": np.nan,
        }

    contig = contiguous_flags(t_idx)   # length n, with contig[0]=True by convention
    contig_steps = contig[1:]          # length n-1
    ychg = (y[1:] != y[:-1])           # length n-1
    sw = ychg & contig_steps           # switches only on contiguous steps

    n_steps = int(n - 1)
    n_contig = int(np.sum(contig_steps))
    n_switches = int(np.sum(sw))

    switch_rate = float(n_switches / n_contig) if n_contig > 0 else np.nan
    contig_rate = float(n_contig / n_steps) if n_steps > 0 else np.nan
    gap_rate = float(1.0 - contig_rate) if np.isfinite(contig_rate) else np.nan

    diagonal_rate = np.nan
    if diag_pairs is not None and n_switches > 0:
        diag = 0
        # iterate only over observed switches (already contiguous-gated)
        for a, b in zip(y[1:][sw], y[:-1][sw]):
            aa = int(a)
            bb = int(b)
            if (aa, bb) in diag_pairs or (bb, aa) in diag_pairs:
                diag += 1
        diagonal_rate = float(diag / n_switches)

    return {
        "switch_rate": switch_rate,
        "diagonal_rate": diagonal_rate,
        "n_contig_steps": n_contig,
        "n_switches": n_switches,
        "contig_rate": contig_rate,
        "gap_rate": gap_rate,
    }


def boundary_distance_weight(xr: np.ndarray, yr: np.ndarray, tau: float, power: float) -> np.ndarray:
    """
    xr, yr in [0,1]. Distance to nearest midline: d = min(|x-0.5|, |y-0.5|).
    
    For finite coords:
        if tau > 0:  w = min(1, d/tau) ** power
        if tau <= 0: w = 1   (no boundary downweighting)
    
    Frames with non-finite xr/yr always get weight 0 (excluded from BAW).
    """
    xr = np.asarray(xr, float)
    yr = np.asarray(yr, float)
    ok = np.isfinite(xr) & np.isfinite(yr)

    w = np.zeros_like(xr, float)     # default 0 => excluded
    
    if tau is None or tau <= 0:
        w[ok] = 1.0
        return w
    
    # distance to nearest midline
    d = np.minimum(np.abs(xr - 0.5), np.abs(yr - 0.5))
    
    # normalized distance, clipped to [0, 1]
    w[ok] = np.minimum(1.0, d[ok] / float(tau))

    if power < 0:
        raise ValueError("boundary_distance_weight: power must be >= 0.")
    if power != 1.0:
        w[ok] = w[ok] ** float(power)

    return w


def align_labels_and_features(
    y_all: np.ndarray,
    X_feature: List[np.ndarray],
    source_label: str = "",
) -> Tuple[int, int, np.ndarray, np.ndarray]:
    """Align a RAW window-label series to the feature timeline.

    Returns
    -------
    x_feature_len : int
        The aligned timeline length (min length across quadrant feature arrays).
    T : int
        Same as x_feature_len (kept for historical reasons).
    y_slice : np.ndarray, shape (T,)
        Window labels truncated/padded to length T.
    y_valid_mask : np.ndarray[bool], shape (T,)
        True for *valid* window labels on the aligned timeline.

    Notes
    -----
    * Window labels are expected to be discrete {1,2,3,4}. Non-integer finite values are treated
      as invalid (and logged), because truncation/casting would silently misclassify bins.
    * This function does **not** mutate y_slice; it only reports which bins are valid.
    """
    quad_lens = [int(len(X_feature[q])) for q in range(4)]
    x_feature_len = int(min(quad_lens)) if quad_lens else 0
    if x_feature_len <= 0:
        return 0, 0, np.array([]), np.array([], bool)

    if len(set(quad_lens)) > 1:
        context = f" [{source_label}]" if source_label else ""
        logging.warning(
            "Quadrant length mismatch%s: %s; truncating to min length = %d.",
            context, quad_lens, x_feature_len
        )

    T = x_feature_len

    y_all = np.asarray(y_all)
    n_y = int(y_all.shape[0])
    if n_y >= T:
        y_slice = y_all[:T]
    else:
        y_slice = np.full(T, np.nan, dtype=float)
        if n_y > 0:
            y_slice[:n_y] = y_all

    # Window labels should be discrete 1..4; treat non-integer numeric values as invalid
    y_finite = np.isfinite(y_slice)
    y_round = np.zeros(T, dtype=int)
    if np.any(y_finite):
        y_round[y_finite] = np.rint(y_slice[y_finite]).astype(int)

    nonint = y_finite & (np.abs(y_slice - y_round) > 1e-6)
    if np.any(nonint):
        context = f" [{source_label}]" if source_label else ""
        logging.warning(
            "Non-integer window labels%s: %d/%d; treating as invalid.",
            context, int(nonint.sum()), int(y_finite.sum())
        )

    y_valid_mask = y_finite & (~nonint) & (y_round >= 1) & (y_round <= 4)

    return x_feature_len, T, y_slice, y_valid_mask


# ----------------- generic weight diagnostics (used for BAW) -----------------

def ess(w: np.ndarray) -> float:
    """Effective sample size under weights."""
    if w.size == 0:
        return np.nan
    s1 = float(np.sum(w))
    s2 = float(np.sum(w * w))
    return (s1 * s1 / s2) if s2 > 0 else np.nan


def top_k_share(w: np.ndarray, frac: float = 0.01) -> float:
    """Share of weight mass in the largest top 'frac' frames."""
    if w.size == 0:
        return np.nan
    k = max(1, int(np.ceil(frac * w.size)))
    ws = np.sort(w)[::-1]
    total = float(np.sum(ws))
    return float(np.sum(ws[:k]) / total) if total > 0 else np.nan


# ========================== Per-subject computation =========================

def _norm_key(x: object) -> str:
    """Normalize join keys for participant IDs / conditions.

    Unlike `_clean_str`, this is intentionally *lossy* and used only for joins.
    Missing values (NaN/None/blank) normalize to the empty string.

    Note: Some upstream CSVs may contain string sentinels like "nan"/"none"/"null"
    (especially after `astype(str)`); we treat those as missing here as well so joins
    do not silently fail.
    """
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    if x is None:
        return ""
    s = str(x).strip().lower()
    return "" if s in {"", "nan", "none", "null"} else s


def _clean_str(x: object) -> Optional[str]:
    """Return a stripped string, or None if missing/blank.

    Treats actual NaN/NA as missing, and also treats common string sentinels
    like 'nan'/'none'/'null' (case-insensitive) as missing.
    """
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    if x is None:
        return None
    s = str(x).strip()
    if not s:
        return None
    if s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _canonical_feature_key(x: object) -> Optional[str]:
    """Normalize a runlist/video/condition value to a `{key}` for feature_4quad_{key}.pkl.

    We try to be permissive about inputs:
      - full paths
      - filenames with extensions (e.g., .mp4 / .pkl)
      - already prefixed names (feature_4quad_{key}[.pkl])

    Returns None if the input is missing/blank.
    """
    s = _clean_str(x)
    if s is None:
        return None

    # Strip directories and (one) file extension.
    try:
        s = Path(s).name
        s = Path(s).stem
    except Exception:
        # Fallback: manual basename + extension strip
        s = str(s).rsplit("/", 1)[-1]
        if "." in s:
            s = s.rsplit(".", 1)[0]

    # Strip the feature prefix if present.
    pref = "feature_4quad_"
    if s.lower().startswith(pref):
        s = s[len(pref):]

    s = s.strip()
    return s or None


def _feature_key_candidates(video: object, condition: object) -> List[str]:
    """Generate candidate `{key}` strings for load_feature_pkl().

    Feature files are looked up as: `feature_4quad_{key}.pkl`.

    The runlist `Video` field (and sometimes the parsed `condition`) may contain:
      - a plain key (e.g., "task4_pro_condA")
      - a filename with extension (e.g., "task4_pro_condA.mp4" or "...condA.pkl")
      - a full path (e.g., "/some/dir/feature_4quad_task4_pro_condA.pkl")
      - an already-prefixed name (e.g., "feature_4quad_task4_pro_condA")

    This helper returns a de-duplicated list of plausible `{key}` candidates, with an
    ordering that tries the *most likely correct* normalization first (canonical key),
    then progressively more permissive fallbacks.

    Rationale: load_feature_pkl() itself re-applies the `feature_4quad_` prefix and
    the `.pkl` extension, so candidates that *still* include the prefix/extension are
    typically wrong but included as late fallbacks in case your directory naming differs.
    """
    candidates: List[str] = []
    pref = "feature_4quad_"

    def _add_one(s: object) -> None:
        s2 = _clean_str(s)
        if s2 is None:
            return
        if s2 and (s2.lower() != pref) and (s2 not in candidates):
            candidates.append(s2)

    def _add_canonical_first(raw: object) -> None:
        c = _canonical_feature_key(raw)
        if c:
            _add_one(c)
            # Also try lower-case variant as a robustness fallback (case mismatches do happen).
            cl = c.lower()
            if cl != c:
                _add_one(cl)

    # 1) Best-effort canonical keys (video first, then condition).
    _add_canonical_first(video)
    _add_canonical_first(condition)

    # 2) Additional variants (paths, stems, prefix stripping, extension stripping).
    def _add_variants(raw: object) -> None:
        s0 = _clean_str(raw)
        if s0 is None:
            return

        variants: List[str] = [s0]
        try:
            p = Path(s0)
            variants.extend([p.name, p.stem])
        except Exception:
            pass

        for v in variants:
            v = _clean_str(v)
            if v is None:
                continue

            _add_one(v)

            # Prefix stripping (case-insensitive)
            if v.lower().startswith(pref):
                suffix = v[len(pref):]
                _add_one(suffix)
                try:
                    _add_one(Path(suffix).stem)
                except Exception:
                    if "." in suffix:
                        _add_one(suffix.rsplit(".", 1)[0])

            # Extension stripping (defensive)
            try:
                _add_one(Path(v).stem)
            except Exception:
                if "." in v:
                    _add_one(v.rsplit(".", 1)[0])

            # Canonicalize each variant too (adds stripped key ahead of noisy strings)
            _add_canonical_first(v)

    _add_variants(video)
    _add_variants(condition)

    return candidates



def compute_dwell_for_subject(
    *,
    csv_file: Path,
    video_feats_dir: Path,
    cfg: RunConfig,
    subj_meta_df: Optional[pd.DataFrame],
    runlist_df: Optional[pd.DataFrame],
) -> Optional[pd.DataFrame]:
    """
    Compute naive and boundary-weighted dwell metrics for a single (subject, condition).
    """
    sid, condition = parse_subject_and_condition(csv_file, cfg.task)
    sid_key = _norm_key(sid)
    cond_key = _norm_key(condition)

    # --- Attach optional run metadata ---
    group_meta = np.nan
    pool_meta = np.nan
    aq = np.nan
    srs = np.nan
    sex = np.nan
    age = np.nan
    video = condition
    runlist_provided = (runlist_df is not None) and isinstance(runlist_df, pd.DataFrame) and (not runlist_df.empty)
    runlist_row_found = False
    # If no runlist is provided (or it is empty), include runs by default.
    use_run = (not runlist_provided)
    has_pool_meta = (subj_meta_df is not None) and ("pool" in subj_meta_df.columns)

    # 1) Subject-level metadata fallback
    if subj_meta_df is not None:
        if "_id_key" in subj_meta_df.columns:
            row_s = subj_meta_df.loc[subj_meta_df["_id_key"] == sid_key]
        else:
            row_s = subj_meta_df.loc[subj_meta_df["ID"].astype(str).map(_norm_key) == sid_key]
        if len(row_s) > 1:
            logging.warning(
                "[%s] subject_metadata_csv has %d rows for ID=%s; using first.",
                csv_file.name, len(row_s), sid
            )
        if len(row_s) > 0:
            row_s = row_s.iloc[0]

            g = _clean_str(row_s.get("group", None))
            if g is not None:
                group_meta = g

            p = _clean_str(row_s.get("pool", None))
            if p is not None:
                pool_meta = p

    # 2) Runlist overrides / adds per-run metadata
    if runlist_provided:
        if "_id_key" in runlist_df.columns and "_cond_key" in runlist_df.columns:
            RL = runlist_df[(runlist_df["_id_key"] == sid_key) & (runlist_df["_cond_key"] == cond_key)]
        else:
            RL = runlist_df[
                (runlist_df["ID"].astype(str).map(_norm_key) == sid_key) &
                (runlist_df["condition"].astype(str).map(_norm_key) == cond_key)
            ]
        if len(RL) > 1:
            logging.warning(
                "[%s] runlist_csv has %d rows for ID=%s, condition=%s; using first.",
                csv_file.name, len(RL), sid, condition
            )
        if len(RL) > 0:
            row_r = RL.iloc[0]

            g = _clean_str(row_r.get("Group", None))
            if g is not None:
                group_meta = g

            aq = row_r.get("AQ", aq)
            srs = row_r.get("SRS", srs)
            sex = row_r.get("Sex", sex)
            age = row_r.get("Age", age)

            v = _clean_str(row_r.get("Video", None))
            video = v if v is not None else condition
            runlist_row_found = True
            use_run = True
    # --- Gaze windows & ratios ---
    gaze = _read_csv(csv_file)
    if "window" not in gaze.columns:
        logging.error("[%s] missing 'window' col", csv_file.name)
        return None
    y_all = pd.to_numeric(gaze["window"], errors="coerce").to_numpy()
    xr_all = pd.to_numeric(gaze.get("x_ratio", pd.Series(np.nan, index=gaze.index)), errors="coerce").to_numpy()
    yr_all = pd.to_numeric(gaze.get("y_ratio", pd.Series(np.nan, index=gaze.index)), errors="coerce").to_numpy()

    # --- Features ---
    # Prefer runlist-provided `video` (if any) to select the feature PKL, but fall back
    # to `condition`. This avoids a silent mismatch where Video metadata is reported
    # but features are always loaded by condition.
    feature_key_used: Optional[str] = None
    feature_candidates: List[str] = _feature_key_candidates(video, condition)

    last_err: Optional[Exception] = None
    X_feature: Optional[List[np.ndarray]] = None
    for key in feature_candidates:
        try:
            X_feature = load_feature_pkl(video_feats_dir, key)
            feature_key_used = key
            break
        except Exception as e:
            last_err = e
            continue

    if X_feature is None:
        logging.warning(
            "[%s] features not found; tried keys=%s. Last error: %s",
            csv_file.name, feature_candidates, last_err
        )
        return None

    # Alignment over unified video length (T == x_feature_len)
    x_feature_len, T, y_slice, y_valid_mask = align_labels_and_features(
        y_all, X_feature, source_label=f"{sid}_{condition}"
    )
    if T == 0:
        logging.info("[%s] empty or misaligned after alignment; skipping.", csv_file.name)
        return None

    n_raw = int(T)

    # Quadrant-length QC
    quad_lens = [arr.shape[0] for arr in X_feature]
    quad_len_min = int(min(quad_lens)) if quad_lens else 0
    quad_len_max = int(max(quad_lens)) if quad_lens else 0
    quad_len_range = quad_len_max - quad_len_min
    quad_len_mismatch = quad_len_range > 0

    # Coverage / masks on RAW grid
    y_finite_mask = np.isfinite(y_slice)
    n_y_finite = int(y_finite_mask.sum())
    n_y_nonfinite = int(n_raw - n_y_finite)
    frac_y_finite = _safe_div(n_y_finite, n_raw)
    frac_y_nonfinite = _safe_div(n_y_nonfinite, n_raw)

    n_y_valid = int(y_valid_mask.sum())
    n_y_offscreen = int((y_finite_mask & ~y_valid_mask).sum())
    frac_y_valid = _safe_div(n_y_valid, n_raw)
    frac_y_offscreen = _safe_div(n_y_offscreen, n_raw)
    frac_y_missing = float(1.0 - _safe_div(n_y_valid, n_raw))

    # Tail dropout: frames after last finite window sample
    if n_y_finite > 0:
        last_finite_idx = int(np.where(y_finite_mask)[0][-1])
        tail_y_missing_bins = int(n_raw - (last_finite_idx + 1))
    else:
        last_finite_idx = -1
        tail_y_missing_bins = int(n_raw)
    tail_y_missing_frac_raw = _safe_div(tail_y_missing_bins, n_raw)
    T_trim = last_finite_idx + 1 if last_finite_idx >= 0 else 0

    # Early vs late missingness + quartiles
    if n_raw > 0:
        mid = n_raw // 2
        if mid > 0:
            frac_y_valid_first_half = _safe_div(int(y_valid_mask[:mid].sum()), mid)
        else:
            frac_y_valid_first_half = np.nan

        second_len = n_raw - mid
        if second_len > 0:
            frac_y_valid_second_half = _safe_div(int(y_valid_mask[mid:].sum()), second_len)
        else:
            frac_y_valid_second_half = np.nan

        idx_all = np.arange(n_raw)
        q_splits = np.array_split(idx_all, 4)
        frac_quarts: List[float] = []
        for q_idx in q_splits:
            if len(q_idx) > 0:
                frac_quarts.append(_safe_div(int(y_valid_mask[q_idx].sum()), len(q_idx)))
            else:
                frac_quarts.append(np.nan)
        frac_y_valid_qrt1, frac_y_valid_qrt2, frac_y_valid_qrt3, frac_y_valid_qrt4 = frac_quarts
    else:
        frac_y_valid_first_half = frac_y_valid_second_half = np.nan
        frac_y_valid_qrt1 = frac_y_valid_qrt2 = frac_y_valid_qrt3 = frac_y_valid_qrt4 = np.nan

    # No valid gaze -> skip
    if n_y_valid == 0:
        logging.info("[%s] no usable bins (no valid 1..4 labels); skipping.", csv_file.name)
        return None

    # Indices and labels on valid frames
    t_valid = np.where(y_valid_mask)[0].astype(int)
    y_valid = np.rint(y_slice[y_valid_mask]).astype(int)
    y_idx_valid = y_valid - 1  # 0..3
    n_valid = int(y_idx_valid.size)
    if n_valid != n_y_valid:
        logging.warning("[%s] n_valid (%d) != n_y_valid (%d); this should not happen.",
                        csv_file.name, n_valid, n_y_valid)

    # Coordinates aligned to RAW timeline T
    if xr_all.size >= T:
        xr_T = xr_all[:T]
    else:
        xr_T = np.full(T, np.nan, dtype=float)
        xr_T[:xr_all.size] = xr_all

    if yr_all.size >= T:
        yr_T = yr_all[:T]
    else:
        yr_T = np.full(T, np.nan, dtype=float)
        yr_T[:yr_all.size] = yr_all

    # Names/mapping
    base_names = base_feat_names_for_task(cfg.task)
    try:
        feats_interest = require_feature_mapping(cfg.task, X_feature, base_names)
    except Exception as e:
        logging.warning("[%s] feature mapping issue (condition=%s): %s",
                        csv_file.name, condition, e)
        return None

    name_to_idx: Dict[str, int] = {n: i for i, n in enumerate(base_names)}
    for req in feats_interest:
        if req not in name_to_idx:
            logging.error("[%s] required feature '%s' missing; skipping file.", csv_file.name, req)
            return None

    # Boundary weights (post-hoc metrics)
    w_boundary_all = boundary_distance_weight(xr_T, yr_T, cfg.boundary_tau, cfg.boundary_power)
    w_boundary_valid = w_boundary_all[y_valid_mask]
    use_baw = (str(cfg.boundary_mode).lower() == "posthoc")
    w_baw_valid = w_boundary_valid.copy() if use_baw else np.ones_like(w_boundary_valid)

    # BAW diagnostics
    baw_ess = ess(w_baw_valid)
    baw_top1 = top_k_share(w_baw_valid, 0.01)

    # Sequence QC / flicker metrics on valid scope
    _, diag_pairs = quad_grid_pairs(GRID_MAP_DEFAULT)
    q_counts = np.bincount(y_idx_valid, minlength=4).astype(float)
    if q_counts.sum() > 0:
        q_base = q_counts / q_counts.sum()
        q_base_TL = float(q_base[GRID_MAP_DEFAULT[0] - 1])
        q_base_TR = float(q_base[GRID_MAP_DEFAULT[1] - 1])
        q_base_BL = float(q_base[GRID_MAP_DEFAULT[2] - 1])
        q_base_BR = float(q_base[GRID_MAP_DEFAULT[3] - 1])
        q_nonzero = q_base[q_base > 0]
        quad_entropy = float(-np.sum(q_nonzero * np.log(q_nonzero))) if q_nonzero.size else np.nan
    else:
        q_base = np.full(4, np.nan, dtype=float)  
        q_base_TL = q_base_TR = q_base_BL = q_base_BR = np.nan
        quad_entropy = np.nan

    if n_valid > 0:
        runs = run_lengths_contig(y_idx_valid, t_valid)
        median_run_len = float(np.median(runs)) if runs else np.nan
        flicker_idx = float(flicker_index(y_idx_valid, t_valid))
        abab = float(abab_rate(y_idx_valid, t_valid))
        short_lt3 = float(short_run_share(y_idx_valid, t_valid, 3))
        sw_stats = switch_stats(y_idx_valid, t_valid, diag_pairs=diag_pairs)
        switch_rate = float(sw_stats.get("switch_rate", np.nan))
        diagonal_rate = float(sw_stats.get("diagonal_rate", np.nan))
        n_contig_steps = int(sw_stats.get("n_contig_steps", 0))
        n_switches = int(sw_stats.get("n_switches", 0))
        contig_rate = float(sw_stats.get("contig_rate", np.nan))
        gap_rate = float(sw_stats.get("gap_rate", np.nan))

    else:
        median_run_len = np.nan
        flicker_idx = abab = short_lt3 = np.nan
        switch_rate = diagonal_rate = np.nan
        contig_rate = gap_rate = np.nan
        n_contig_steps = n_switches = 0

    # flips per minute (contiguous valid steps, normalized by RAW and valid durations)
    if n_valid > 1:
        contig_valid = contiguous_flags(t_valid)
        sw = (y_idx_valid[1:] != y_idx_valid[:-1]) & contig_valid[1:]
        flips = int(sw.sum())
        flip_idx_valid = np.where(sw)[0] + 1
    else:
        flips = 0
        flip_idx_valid = np.array([], dtype=int)
    n_flips = int(flips)

    if cfg.bin_seconds and np.isfinite(cfg.bin_seconds) and cfg.bin_seconds > 0 and n_raw > 0:
        minutes = (n_raw * cfg.bin_seconds) / 60.0
        flip_rate_per_min = _safe_div(flips, minutes)
    else:
        flip_rate_per_min = np.nan

    if cfg.bin_seconds and np.isfinite(cfg.bin_seconds) and cfg.bin_seconds > 0 and n_valid > 0:
        minutes_valid = (n_valid * cfg.bin_seconds) / 60.0
        flip_rate_per_min_valid = _safe_div(flips, minutes_valid)
    else:
        flip_rate_per_min_valid = np.nan

    # Gaze time per quadrant among y-valid frames
    if n_y_valid > 0:
        yv = y_valid  # already integer labels on y-valid frames
        counts = np.bincount(yv, minlength=5)
        n_q1_y_valid = int(counts[1])
        n_q2_y_valid = int(counts[2])
        n_q3_y_valid = int(counts[3])
        n_q4_y_valid = int(counts[4])
        frac_q1_y_valid = _safe_div(n_q1_y_valid, n_y_valid)
        frac_q2_y_valid = _safe_div(n_q2_y_valid, n_y_valid)
        frac_q3_y_valid = _safe_div(n_q3_y_valid, n_y_valid)
        frac_q4_y_valid = _safe_div(n_q4_y_valid, n_y_valid)
    else:
        n_q1_y_valid = n_q2_y_valid = n_q3_y_valid = n_q4_y_valid = 0
        frac_q1_y_valid = frac_q2_y_valid = frac_q3_y_valid = frac_q4_y_valid = np.nan

    mean_w_boundary_all = float(np.nanmean(w_boundary_all)) if w_boundary_all.size else np.nan
    mean_w_boundary_valid = float(np.nanmean(w_boundary_valid)) if w_boundary_valid.size else np.nan
    mean_w_boundary_at_flips = np.nan
    if flip_idx_valid.size > 0:
        mean_w_boundary_at_flips = float(np.nanmean(w_boundary_valid[flip_idx_valid]))

    # Optional durations in seconds
    if cfg.bin_seconds and np.isfinite(cfg.bin_seconds) and cfg.bin_seconds > 0:
        x_feature_seconds = float(x_feature_len * cfg.bin_seconds)
    else:
        x_feature_seconds = np.nan

    # Precompute center distance for near-boundary metrics
    d_center = np.minimum(np.abs(xr_T - 0.5), np.abs(yr_T - 0.5))
    coord_ok = np.isfinite(xr_T) & np.isfinite(yr_T)
    
    coord_ok_valid = coord_ok & y_valid_mask
    n_coord_ok_valid = int(coord_ok_valid.sum())
    frac_coord_ok_valid = _safe_div(n_coord_ok_valid, float(n_y_valid))
    
    near_valid = coord_ok_valid & (d_center < cfg.boundary_tau)
    frac_near_boundary_valid = _safe_div(int(near_valid.sum()), float(n_coord_ok_valid))


    # ---------------- Speaker-label diagnostics (construct validity) ----------------
    # These are repeated on every output row (per feature) so downstream models can
    # stratify / adjust for speaker-label quality and Task5 averted annotation missingness.
    n_speaker_any_raw = np.nan
    frac_speaker_any_raw = np.nan
    n_speaker_unique_raw = np.nan
    frac_speaker_unique_raw = np.nan
    n_speaker_any_valid = np.nan
    frac_speaker_any_valid = np.nan
    n_speaker_unique_valid = np.nan
    frac_speaker_unique_valid = np.nan

    n_speaker_averted_unknown_raw = np.nan
    frac_speaker_averted_unknown_raw = np.nan
    n_speaker_averted_unknown_valid = np.nan
    frac_speaker_averted_unknown_valid = np.nan

    if "speaker" in name_to_idx:
        sp_idx = name_to_idx["speaker"]
        speaker_raw = np.column_stack([X_feature[q][:T, sp_idx] for q in range(4)])
        speaker_pres = (np.isfinite(speaker_raw) & (speaker_raw >= cfg.presence_threshold))
        speaker_count = speaker_pres.sum(axis=1)
        speaker_any = (speaker_count > 0)
        speaker_unique = (speaker_count == 1)

        n_speaker_any_raw = int(speaker_any.sum())
        frac_speaker_any_raw = _safe_div(n_speaker_any_raw, n_raw)
        n_speaker_unique_raw = int(speaker_unique.sum())
        frac_speaker_unique_raw = _safe_div(n_speaker_unique_raw, n_raw)

        n_speaker_any_valid = int((speaker_any & y_valid_mask).sum())
        frac_speaker_any_valid = _safe_div(n_speaker_any_valid, n_y_valid)
        n_speaker_unique_valid = int((speaker_unique & y_valid_mask).sum())
        frac_speaker_unique_valid = _safe_div(n_speaker_unique_valid, n_y_valid)

        # Task5: how often is averted missing on the speaker quadrant (among unique-speaker bins)?
        if cfg.task.startswith("task5_") and ("averted" in name_to_idx):
            av_idx = name_to_idx["averted"]
            averted_raw = np.column_stack([X_feature[q][:T, av_idx] for q in range(4)])
            # For unique-speaker bins, pick the averted value on the (unique) speaker quadrant
            spk_quad = np.argmax(speaker_pres, axis=1)
            averted_at_speaker = np.full(T, np.nan, dtype=float)
            mask_u = speaker_unique
            if np.any(mask_u):
                averted_at_speaker[mask_u] = averted_raw[mask_u, spk_quad[mask_u]]

            unknown_mask = mask_u & (~np.isfinite(averted_at_speaker))
            n_speaker_averted_unknown_raw = int(unknown_mask.sum())
            frac_speaker_averted_unknown_raw = _safe_div(n_speaker_averted_unknown_raw, n_speaker_unique_raw)
            n_speaker_averted_unknown_valid = int((unknown_mask & y_valid_mask).sum())
            frac_speaker_averted_unknown_valid = _safe_div(n_speaker_averted_unknown_valid, n_speaker_unique_valid)

    rows: List[dict] = []
    feats: List[str] = list(feats_interest)

    # Derived / joint features (optional, additive)
    derived_feat_raw: Dict[str, np.ndarray] = {}
    if cfg.include_joint_features:
        derived_feat_raw = build_derived_feature_matrices(
            task=cfg.task,
            X_feature=X_feature,
            T=T,
            name_to_idx=name_to_idx,
            presence_threshold=cfg.presence_threshold,
            require_speaker_present=cfg.derived_require_speaker_present,
            require_speaker_unique=cfg.derived_require_speaker_unique,
            task5_require_speaker_unique=cfg.task5_require_speaker_unique,
            include_speaker_not_averted_loose=cfg.include_speaker_not_averted_loose,
        )
        # Append in a stable, human-readable order
        for nm in derived_features_for_task(cfg.task, cfg.include_speaker_not_averted_loose):
            if nm in derived_feat_raw and nm not in feats:
                feats.append(nm)

    for feat in feats:
        if feat in name_to_idx:
            j = name_to_idx[feat]
            feat_raw = np.column_stack([X_feature[q][:T, j] for q in range(4)])
        elif feat in derived_feat_raw:
            feat_raw = derived_feat_raw[feat]
        else:
            logging.warning("[%s] unknown feature '%s' (condition=%s); skipping",
                            csv_file.name, feat, condition)
            continue

        feat_valid = feat_raw[y_valid_mask, :]

        # Availability (supply)
        present_raw = np.any(np.isfinite(feat_raw) & (feat_raw >= cfg.presence_threshold), axis=1)
        present_valid = np.any(np.isfinite(feat_valid) & (feat_valid >= cfg.presence_threshold), axis=1)
        n_present_any_raw = int(present_raw.sum())
        n_present_any_valid = int(present_valid.sum())
        frac_present_any_raw = _safe_div(n_present_any_raw, float(n_raw))
        frac_present_any_valid = _safe_div(n_present_any_valid, float(n_y_valid))

        # Chance baselines
        e_random_raw = _chance(feat_raw, cfg.presence_threshold)
        e_random_valid = _chance(feat_valid, cfg.presence_threshold)
        e_random_valid_baw = _chance_weighted(feat_valid, cfg.presence_threshold, w_baw_valid)

        # Attended & dwell (on valid frames)
        attended_vals = feat_valid[np.arange(n_valid), y_idx_valid]
        attended_ok = np.isfinite(attended_vals)
        n_attended_valid = int(attended_ok.sum())
        n_attended_nan = int((~attended_ok).sum())

        chosen_has_feat = np.zeros(n_valid, dtype=bool)
        chosen_has_feat[attended_ok] = attended_vals[attended_ok] >= cfg.presence_threshold

        chosen_after = run_length_filter_contig(
            chosen_has_feat, t_valid, min_run=cfg.min_run, max_gap=1, use_time_span=True
        )
        dwell_bins = int(chosen_after.sum())
        n_attended_present_valid = int(np.sum(chosen_has_feat & present_valid))

        # Build a raw-grid dwell indicator: 1 at raw index t if valid & dwell, else 0
        dwell_raw = np.zeros(T, dtype=float)
        dwell_raw[y_valid_mask] = chosen_after.astype(float)

        # Presence-trimmed RAW metrics: truncate at last finite gaze bin (T_trim)
        if T_trim > 0:
            n_raw_trim = float(T_trim)
            dwell_bins_trim = int(dwell_raw[:T_trim].sum())
            rate_over_raw_trim = _safe_div(dwell_bins_trim, n_raw_trim)
            e_random_raw_trim = _chance(feat_raw[:T_trim, :], cfg.presence_threshold)
        else:
            n_raw_trim = 0.0
            dwell_bins_trim = 0
            rate_over_raw_trim = np.nan
            e_random_raw_trim = np.nan

        
        # Quadrant-bias adjusted chance baseline on valid frames.
        #
        # IMPORTANT: `feat_valid` columns are always in *window label order* (q1..q4),
        # and `q_base` is computed in that same order. We therefore use `q_base` directly
        # for the chance baseline; the TL/TR/BL/BR mapping is only for reporting.
        e_random_valid_qbase = np.nan
        if np.all(np.isfinite(q_base)):
            s_q = float(np.sum(q_base))
            if s_q > 0:
                q_weights = q_base / s_q  # (4,) in window order
                present_valid_bin = (np.isfinite(feat_valid) & (feat_valid >= cfg.presence_threshold))
                per_t_qbase = np.sum(present_valid_bin * q_weights[None, :], axis=1)
                e_random_valid_qbase = float(np.mean(per_t_qbase))

        if cfg.bin_seconds and np.isfinite(cfg.bin_seconds) and cfg.bin_seconds > 0:
            dwell_seconds = float(dwell_bins * cfg.bin_seconds)
        else:
            dwell_seconds = np.nan

        # Weighted dwell (BAW)
        dwell_baw = float(np.sum(w_baw_valid[chosen_after])) if n_valid > 0 else np.nan

        # Denominators
        den_raw = float(n_raw)
        den_valid = float(n_y_valid)
        den_att = float(n_attended_valid)
        den_pres = float(n_present_any_valid)

        den_valid_baw = float(np.sum(w_baw_valid))
        den_pres_baw = float(np.sum(w_baw_valid[present_valid])) if n_present_any_valid > 0 else np.nan

        # Interpretation:
        #   rate_over_raw        = P(dwell) over all video bins (including missing gaze)
        #   rate_over_valid      = P(dwell | gaze valid anywhere)
        #   rate_given_present   = P(dwell | gaze valid & feature present in >=1 quadrant)
        #   rate_over_valid_baw  = same as above, but downweighting near-boundary gaze

        # Rates (unweighted)
        rate_over_raw = _safe_div(dwell_bins, den_raw)
        rate_over_valid = _safe_div(dwell_bins, den_valid)
        rate_over_attended_valid = _safe_div(dwell_bins, den_att)
        rate_given_present_any = _safe_div(dwell_bins, den_pres)

        # Rates (BAW-weighted)
        rate_over_valid_baw = _safe_div(dwell_baw, den_valid_baw)
        rate_given_present_any_baw = _safe_div(dwell_baw, den_pres_baw) if np.isfinite(den_pres_baw) else np.nan

        # Enrichment helpers
        def enrich(obs: float, chance: float) -> Tuple[float, float]:
            if not np.isfinite(chance):
                return np.nan, np.nan
            diff = obs - chance
            denom = 1.0 - chance
            norm = (diff / denom) if denom > 0 else np.nan
            return float(diff), float(norm)

        enrich_over_valid_diff, enrich_over_valid_norm = enrich(rate_over_valid, e_random_valid)
        _, enrich_over_valid_norm_baw = enrich(rate_over_valid_baw, e_random_valid_baw)
        _, enrich_over_valid_norm_qbase = enrich(rate_over_valid, e_random_valid_qbase)

        enrich_over_raw_diff, enrich_over_raw_norm = enrich(rate_over_raw, e_random_raw)
        enrich_over_raw_trim_diff, enrich_over_raw_trim_norm = enrich(rate_over_raw_trim, e_random_raw_trim)

        # Retention ratio RAW -> VALID (conditional on feature presence)
        valid_over_raw = _safe_div(n_present_any_valid, float(n_present_any_raw))

        # Missingness given feature present/absent at RAW
        y_valid_raw = y_valid_mask[:n_raw]
        y_missing_raw = ~y_valid_raw

        n_y_missing_given_present_any_raw = int((y_missing_raw & present_raw).sum())
        denom_present_any_raw = int(present_raw.sum()) if present_raw.size > 0 else 0
        frac_y_missing_given_present_any_raw = _safe_div(
            n_y_missing_given_present_any_raw,
            denom_present_any_raw,
        )

        absent_raw = ~present_raw
        denom_absent_raw = int(absent_raw.sum()) if absent_raw.size > 0 else 0
        n_y_missing_given_absent_raw = int((y_missing_raw & absent_raw).sum())
        frac_y_missing_given_absent_raw = _safe_div(
            n_y_missing_given_absent_raw,
            denom_absent_raw,
        )

        if np.isfinite(frac_y_missing_given_present_any_raw) and np.isfinite(frac_y_missing):
            missingness_differential = float(frac_y_missing_given_present_any_raw - frac_y_missing)
        else:
            missingness_differential = np.nan

        # Exactly-one-present precision + how often exact1 occurs
        present_count_valid = (np.isfinite(feat_valid) & (feat_valid >= cfg.presence_threshold)).sum(axis=1)
        exact1_idx = np.where(present_count_valid == 1)[0]
        frac_exact1_valid = _safe_div(exact1_idx.size, n_valid)

        precision_exact1 = precision_exact1_baw = np.nan
        if exact1_idx.size > 0:
            present_valid_bin = (np.isfinite(feat_valid) & (feat_valid >= cfg.presence_threshold))
            q_star = np.argmax(present_valid_bin[exact1_idx, :], axis=1)
            correct = (y_idx_valid[exact1_idx] == q_star).astype(float)

            def wmean(b, w_vec):
                s = float(np.sum(w_vec))
                return float(np.sum(w_vec * b) / s) if s > 0 else np.nan

            precision_exact1 = float(np.mean(correct))
            precision_exact1_baw = wmean(correct, w_baw_valid[exact1_idx])

        # Near-boundary fractions, conditioned on supply
        # present_valid is already on valid scope (length n_valid)
        coord_ok_valid_vec = coord_ok[y_valid_mask]          # length n_valid
        den = int((coord_ok_valid_vec & present_valid).sum())
        num = int(((coord_ok_valid_vec & present_valid) & (d_center[y_valid_mask] < cfg.boundary_tau)).sum())
        frac_near_boundary_given_present = _safe_div(num, float(den))

        if np.isfinite(frac_near_boundary_given_present) and np.isfinite(frac_near_boundary_valid):
            center_bias_differential_given_present = float(
                frac_near_boundary_given_present - frac_near_boundary_valid
            )
        else:
            center_bias_differential_given_present = np.nan


        # additional metrics to facilitate cluster analysis
        den_present_any_baw = float(np.sum(w_baw_valid[present_valid])) if n_present_any_valid > 0 else 0.0
        chance_valid_baw_num = float(e_random_valid_baw) * den_valid_baw if np.isfinite(e_random_valid_baw) else np.nan
        
        n_exact1_valid = int(exact1_idx.size)
        n_exact1_correct = int(np.sum(correct)) if exact1_idx.size > 0 else 0
        den_exact1_baw = float(np.sum(w_baw_valid[exact1_idx])) if exact1_idx.size > 0 else 0.0
        num_exact1_correct_baw = float(np.sum(w_baw_valid[exact1_idx] * correct)) if exact1_idx.size > 0 else 0.0


        row = {
            # IDs / meta
            "ID": sid,
            "condition": condition,
            "task": cfg.task,
            "feature": feat,
            "Video": video,
            "feature_key": feature_key_used,
            "Group": group_meta,
            "AQ": aq,
            "SRS": srs,
            "Sex": sex,
            "Age": age,
            "use_run": bool(use_run),
            "runlist_provided": bool(runlist_provided),
            "runlist_row_found": bool(runlist_row_found),

            # Speaker-label diagnostics (repeated per-row)
            "n_speaker_any_raw": int(n_speaker_any_raw) if np.isfinite(n_speaker_any_raw) else np.nan,
            "frac_speaker_any_raw": float(frac_speaker_any_raw) if np.isfinite(frac_speaker_any_raw) else np.nan,
            "n_speaker_unique_raw": int(n_speaker_unique_raw) if np.isfinite(n_speaker_unique_raw) else np.nan,
            "frac_speaker_unique_raw": float(frac_speaker_unique_raw) if np.isfinite(frac_speaker_unique_raw) else np.nan,
            "n_speaker_any_valid": int(n_speaker_any_valid) if np.isfinite(n_speaker_any_valid) else np.nan,
            "frac_speaker_any_valid": float(frac_speaker_any_valid) if np.isfinite(frac_speaker_any_valid) else np.nan,
            "n_speaker_unique_valid": int(n_speaker_unique_valid) if np.isfinite(n_speaker_unique_valid) else np.nan,
            "frac_speaker_unique_valid": float(frac_speaker_unique_valid) if np.isfinite(frac_speaker_unique_valid) else np.nan,
            "n_speaker_averted_unknown_raw": int(n_speaker_averted_unknown_raw) if np.isfinite(n_speaker_averted_unknown_raw) else np.nan,
            "frac_speaker_averted_unknown_raw": float(frac_speaker_averted_unknown_raw) if np.isfinite(frac_speaker_averted_unknown_raw) else np.nan,
            "n_speaker_averted_unknown_valid": int(n_speaker_averted_unknown_valid) if np.isfinite(n_speaker_averted_unknown_valid) else np.nan,
            "frac_speaker_averted_unknown_valid": float(frac_speaker_averted_unknown_valid) if np.isfinite(frac_speaker_averted_unknown_valid) else np.nan,

            # Stimulus length / coverage
            "x_feature_len": int(x_feature_len),
            "x_feature_seconds": float(x_feature_seconds) if np.isfinite(x_feature_seconds) else np.nan,
            "n_raw": int(n_raw),
            "n_raw_trim": int(n_raw_trim),
            "n_y_finite": int(n_y_finite),
            "n_y_nonfinite": int(n_y_nonfinite),
            "n_y_valid": int(n_y_valid),
            "n_y_offscreen": int(n_y_offscreen),
            "frac_y_finite": float(frac_y_finite) if np.isfinite(frac_y_finite) else np.nan,
            "frac_y_nonfinite": float(frac_y_nonfinite) if np.isfinite(frac_y_nonfinite) else np.nan,
            "frac_y_valid": float(frac_y_valid) if np.isfinite(frac_y_valid) else np.nan,
            "frac_y_offscreen": float(frac_y_offscreen) if np.isfinite(frac_y_offscreen) else np.nan,
            "frac_y_missing": float(frac_y_missing) if np.isfinite(frac_y_missing) else np.nan,
            "tail_y_missing_bins": int(tail_y_missing_bins),
            "tail_y_missing_frac_raw": float(tail_y_missing_frac_raw)
                if np.isfinite(tail_y_missing_frac_raw) else np.nan,
            "frac_y_valid_first_half": float(frac_y_valid_first_half)
                if np.isfinite(frac_y_valid_first_half) else np.nan,
            "frac_y_valid_second_half": float(frac_y_valid_second_half)
                if np.isfinite(frac_y_valid_second_half) else np.nan,
            "frac_y_valid_qrt1": float(frac_y_valid_qrt1) if np.isfinite(frac_y_valid_qrt1) else np.nan,
            "frac_y_valid_qrt2": float(frac_y_valid_qrt2) if np.isfinite(frac_y_valid_qrt2) else np.nan,
            "frac_y_valid_qrt3": float(frac_y_valid_qrt3) if np.isfinite(frac_y_valid_qrt3) else np.nan,
            "frac_y_valid_qrt4": float(frac_y_valid_qrt4) if np.isfinite(frac_y_valid_qrt4) else np.nan,

            # Quadrant length QC
            "quad_len_min": int(quad_len_min),
            "quad_len_max": int(quad_len_max),
            "quad_len_range": int(quad_len_range),
            "quad_len_mismatch": bool(quad_len_mismatch),

            # Gaze-quadrant distribution (QC)
            "n_flips": int(n_flips),
            "n_q1_y_valid": int(n_q1_y_valid),
            "n_q2_y_valid": int(n_q2_y_valid),
            "n_q3_y_valid": int(n_q3_y_valid),
            "n_q4_y_valid": int(n_q4_y_valid),
            "frac_q1_y_valid": float(frac_q1_y_valid) if np.isfinite(frac_q1_y_valid) else np.nan,
            "frac_q2_y_valid": float(frac_q2_y_valid) if np.isfinite(frac_q2_y_valid) else np.nan,
            "frac_q3_y_valid": float(frac_q3_y_valid) if np.isfinite(frac_q3_y_valid) else np.nan,
            "frac_q4_y_valid": float(frac_q4_y_valid) if np.isfinite(frac_q4_y_valid) else np.nan,
            "q_base_TL": float(q_base_TL) if np.isfinite(q_base_TL) else np.nan,
            "q_base_TR": float(q_base_TR) if np.isfinite(q_base_TR) else np.nan,
            "q_base_BL": float(q_base_BL) if np.isfinite(q_base_BL) else np.nan,
            "q_base_BR": float(q_base_BR) if np.isfinite(q_base_BR) else np.nan,
            "quad_entropy": float(quad_entropy) if np.isfinite(quad_entropy) else np.nan,

            # Availability
            "n_present_any_raw": int(n_present_any_raw),
            "frac_present_any_raw": float(frac_present_any_raw) if np.isfinite(frac_present_any_raw) else np.nan,
            "n_present_any_valid": int(n_present_any_valid),
            "frac_present_any_valid": float(frac_present_any_valid) if np.isfinite(frac_present_any_valid) else np.nan,

            # Chance baselines
            "e_random_raw": float(e_random_raw) if np.isfinite(e_random_raw) else np.nan,
            "e_random_raw_trim": float(e_random_raw_trim) if np.isfinite(e_random_raw_trim) else np.nan,
            "e_random_valid": float(e_random_valid) if np.isfinite(e_random_valid) else np.nan,
            "e_random_valid_qbase": float(e_random_valid_qbase) if np.isfinite(e_random_valid_qbase) else np.nan,
            "e_random_valid_baw": float(e_random_valid_baw) if np.isfinite(e_random_valid_baw) else np.nan,

            # Attended & dwell
            "n_attended_valid": int(n_attended_valid),
            "n_attended_nan": int(n_attended_nan),
            "dwell_bins": int(dwell_bins),
            "dwell_seconds": float(dwell_seconds) if np.isfinite(dwell_seconds) else np.nan,
            "n_attended_present_valid": n_attended_present_valid,

            # Rates (unweighted)
            "rate_given_present_any": float(rate_given_present_any)
                if np.isfinite(rate_given_present_any) else np.nan,
            "rate_over_valid": float(rate_over_valid)
                if np.isfinite(rate_over_valid) else np.nan,
            "rate_over_raw": float(rate_over_raw) if np.isfinite(rate_over_raw) else np.nan,
            "rate_over_raw_trim": float(rate_over_raw_trim) if np.isfinite(rate_over_raw_trim) else np.nan,
            "rate_over_attended_valid": float(rate_over_attended_valid)
                if np.isfinite(rate_over_attended_valid) else np.nan,

            # Rates (BAW-weighted)
            "rate_given_present_any_baw": float(rate_given_present_any_baw)
                if np.isfinite(rate_given_present_any_baw) else np.nan,
            "rate_over_valid_baw": float(rate_over_valid_baw)
                if np.isfinite(rate_over_valid_baw) else np.nan,

            # Enrichment (valid)
            "enrich_over_valid_diff": float(enrich_over_valid_diff)
                if np.isfinite(enrich_over_valid_diff) else np.nan,
            "enrich_over_valid_norm": float(enrich_over_valid_norm)
                if np.isfinite(enrich_over_valid_norm) else np.nan,
            "enrich_over_valid_norm_baw": float(enrich_over_valid_norm_baw)
                if np.isfinite(enrich_over_valid_norm_baw) else np.nan,
            "enrich_over_valid_norm_qbase": float(enrich_over_valid_norm_qbase)
                if np.isfinite(enrich_over_valid_norm_qbase) else np.nan,

            # Enrichment (raw, QC)
            "enrich_over_raw_diff": float(enrich_over_raw_diff)
                if np.isfinite(enrich_over_raw_diff) else np.nan,
            "enrich_over_raw_norm": float(enrich_over_raw_norm)
                if np.isfinite(enrich_over_raw_norm) else np.nan,
            "enrich_over_raw_trim_diff": float(enrich_over_raw_trim_diff)
                if np.isfinite(enrich_over_raw_trim_diff) else np.nan,
            "enrich_over_raw_trim_norm": float(enrich_over_raw_trim_norm)
                if np.isfinite(enrich_over_raw_trim_norm) else np.nan,

            # Retention & missingness-by-feature
            "valid_over_raw": float(valid_over_raw) if np.isfinite(valid_over_raw) else np.nan,
            "n_y_missing_given_present_any_raw": int(n_y_missing_given_present_any_raw),
            "frac_y_missing_given_present_any_raw": float(frac_y_missing_given_present_any_raw)
                if np.isfinite(frac_y_missing_given_present_any_raw) else np.nan,
            "n_y_missing_given_absent_raw": int(n_y_missing_given_absent_raw),
            "frac_y_missing_given_absent_raw": float(frac_y_missing_given_absent_raw)
                if np.isfinite(frac_y_missing_given_absent_raw) else np.nan,
            "missingness_differential": float(missingness_differential)
                if np.isfinite(missingness_differential) else np.nan,

            # Boundary diagnostics
            "mean_w_boundary_all": float(mean_w_boundary_all) if np.isfinite(mean_w_boundary_all) else np.nan,
            "mean_w_boundary_valid": float(mean_w_boundary_valid) if np.isfinite(mean_w_boundary_valid) else np.nan,
            "mean_w_boundary_at_flips": float(mean_w_boundary_at_flips)
                if np.isfinite(mean_w_boundary_at_flips) else np.nan,
            "frac_coord_ok_valid": float(frac_coord_ok_valid)
                if np.isfinite(frac_coord_ok_valid) else np.nan,
            "frac_near_boundary_valid": float(frac_near_boundary_valid)
                if np.isfinite(frac_near_boundary_valid) else np.nan,
            "frac_near_boundary_given_present": float(frac_near_boundary_given_present)
                if np.isfinite(frac_near_boundary_given_present) else np.nan,
            "center_bias_differential_given_present": float(center_bias_differential_given_present)
                if np.isfinite(center_bias_differential_given_present) else np.nan,

            # exactly-one-present precision
            "frac_exact1_valid": float(frac_exact1_valid) if np.isfinite(frac_exact1_valid) else np.nan,
            "precision_exact1_valid": float(precision_exact1) if np.isfinite(precision_exact1) else np.nan,
            "precision_exact1_valid_baw": float(precision_exact1_baw)
                if np.isfinite(precision_exact1_baw) else np.nan,

            # Settings echo + QC churn
            "min_run": int(cfg.min_run),
            "presence_threshold": float(cfg.presence_threshold),
            "include_joint_features": bool(cfg.include_joint_features),
            "derived_require_speaker_present": bool(cfg.derived_require_speaker_present),
            "derived_require_speaker_unique": bool(cfg.derived_require_speaker_unique),
            "task5_require_speaker_unique": bool(cfg.task5_require_speaker_unique),
            "flip_rate_per_min": float(flip_rate_per_min) if np.isfinite(flip_rate_per_min) else np.nan,
            "flip_rate_per_min_valid": float(flip_rate_per_min_valid)
                if np.isfinite(flip_rate_per_min_valid) else np.nan,
            "median_run_len": float(median_run_len) if np.isfinite(median_run_len) else np.nan,
            "flicker_index": float(flicker_idx) if np.isfinite(flicker_idx) else np.nan,
            "abab_rate": float(abab) if np.isfinite(abab) else np.nan,
            "short_run_share_lt3": float(short_lt3) if np.isfinite(short_lt3) else np.nan,
            "switch_rate": float(switch_rate) if np.isfinite(switch_rate) else np.nan,
            "diagonal_rate": float(diagonal_rate) if np.isfinite(diagonal_rate) else np.nan,
            "n_switches": n_switches,
            "n_contig_steps": n_contig_steps,
            "contig_rate": float(contig_rate) if np.isfinite(contig_rate) else np.nan,
            "gap_rate": float(gap_rate) if np.isfinite(gap_rate) else np.nan,

            # Boundary mode / BAW diagnostics
            "boundary_mode": str(cfg.boundary_mode),
            "boundary_tau": float(cfg.boundary_tau),
            "boundary_power": float(cfg.boundary_power),
            "ess_baw": float(baw_ess) if np.isfinite(baw_ess) else np.nan,
            "top1pct_share_baw": float(baw_top1) if np.isfinite(baw_top1) else np.nan,

            # additional metrics to facilitate cluster analysis
            "dwell_bins_trim": int(dwell_bins_trim),
            "dwell_baw": float(dwell_baw),
            "den_valid_baw": float(den_valid_baw),
            "den_present_any_baw": float(den_present_any_baw),
            "chance_valid_baw_num": chance_valid_baw_num,
            
            "n_exact1_valid": n_exact1_valid,
            "n_exact1_correct": n_exact1_correct,
            "den_exact1_baw": den_exact1_baw,
            "num_exact1_correct_baw": num_exact1_correct_baw,
            
            "bin_seconds": float(cfg.bin_seconds) if cfg.bin_seconds is not None else np.nan,
        }

        # Only include Pool column if metadata actually had it
        if has_pool_meta:
            row["Pool"] = pool_meta

        rows.append(row)

    # Optional per-bin debug CSV (only boundary weights + validity mask)
    if cfg.save_bin_level and n_raw > 0:
        out_dir = Path(cfg.output_root) / "perbin"
        out_dir.mkdir(parents=True, exist_ok=True)
    
        perbin = pd.DataFrame({
            "t": np.arange(T, dtype=int),
            "y_valid": y_valid_mask.astype(int),
            "w_boundary": w_boundary_all.astype(float),
            "w_baw_valid": np.nan,
        })
        perbin.loc[y_valid_mask, "w_baw_valid"] = w_baw_valid
        perbin.to_csv(out_dir / f"{sid}_{condition}_perbin.csv", index=False)
    

    return pd.DataFrame(rows) if rows else None


# ================================ CLI / main ================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute dwell-times with availability (raw/valid), chance, enrichment, "
            "retention, and optional boundary-aware confidence weighting (BAW). "
        )
    )
    p.add_argument("--video_feats_dir", required=True)
    p.add_argument("--window_data_dir", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--task", choices=[
        "task4_pro", "task4_spark", "task4_lab",
        "task5_pro", "task5_spark", "task5_prospark", "task5_lab", "task5_tobii"], required=True)
    p.add_argument("--window_glob", default="videoview*.csv")

    # Metadata (optional)
    p.add_argument("--subject_metadata_csv", type=str, default=None,
                   help="Optional CSV mapping ID -> group, pool.")
    p.add_argument("--runlist_csv", type=str, default=None,
                   help="Optional CSV mapping (ID, condition/Video) -> Group, Video, AQ, SRS, Sex, Age.")

    # Base options
    p.add_argument("--presence_threshold", type=float, default=0.5)
    p.add_argument("--min_run", type=int, default=1)
    p.add_argument("--bin_seconds", type=float, default=None)
    # Derived / joint-feature endpoints (optional, additive)
    p.add_argument("--no_joint_features", action="store_true", default=False,
                   help="Disable derived joint-feature endpoints (e.g., distraction_offspeaker, speaker_averted).")
    p.add_argument("--derived_allow_no_speaker", action="store_true", default=False,
                   help=("Allow distraction_offspeaker/onspeaker derivations even when no speaker is labeled "
                         "in a bin (not recommended for speaker-relative interpretation)."))
    p.add_argument("--derived_require_speaker_unique", action="store_true", default=False,
                   help=("Require exactly one speaker quadrant when computing distraction_offspeaker/onspeaker. "
                         "Useful if rare multi-speaker labeling exists."))
    p.add_argument("--task5_allow_nonunique_speaker", action="store_true", default=False,
                   help=("(Task5 only) Allow speaker_averted/speaker_not_averted derivations even when "
                         "speaker is not uniquely labeled in a bin (not recommended)."))
    p.add_argument("--task5_include_not_averted_loose", action="store_true", default=False,
                   help=("(Task5 only) Also emit speaker_not_averted_loose sensitivity endpoint (treat unknown/NaN averted on the speaker quadrant as \"not averted\")."))



    p.add_argument("--loglevel", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])

    # Boundary
    p.add_argument("--boundary_mode", choices=["posthoc", "off"], default="posthoc")
    p.add_argument("--boundary_tau", type=float, default=0.05)
    p.add_argument("--boundary_power", type=float, default=1.0)

    # Debug
    p.add_argument("--save_bin_level", action="store_true", default=False)
    p.add_argument("--random_state", type=int, default=12345)

    return p


def main(cfg: RunConfig) -> None:
    level = getattr(logging, cfg.loglevel.upper(), logging.INFO)
    logging.basicConfig(level=level,
                        format="%(asctime)s | %(levelname)s | %(message)s")

    if cfg.boundary_mode not in {"posthoc", "off"}:
        raise ValueError(f"Unsupported boundary_mode={cfg.boundary_mode!r}; use 'posthoc' or 'off'.")

    video_feats_path = Path(cfg.video_feats_dir)
    window_dir = Path(cfg.window_data_dir)
    out_root = Path(cfg.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    tag = "dwell_times"
    if cfg.min_run > 1:
        tag += f"_mrun-{cfg.min_run}"
    if cfg.bin_seconds is not None:
        tag += f"_binsz-{cfg.bin_seconds:g}s"
    tag += f"_bdry-{cfg.boundary_mode}-t{cfg.boundary_tau:g}-p{cfg.boundary_power:g}"
    if cfg.include_joint_features:
        tag += "_jointfeat"

    # Save run config
    (out_root / f"{cfg.task}_{tag}_run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    files = sorted(glob(str(window_dir / cfg.window_glob)))
    if not files:
        logging.warning("No files matched %s in %s", cfg.window_glob, window_dir)

    # Optional metadata
    subj_meta = maybe_read_subject_meta(cfg.subject_metadata_csv)
    if subj_meta is not None and subj_meta.empty:
        logging.warning("subject_metadata_csv read but contained no rows; ignoring metadata.")
        subj_meta = None

    runlist = maybe_read_runlist(cfg.runlist_csv)
    if runlist is not None and runlist.empty:
        logging.warning("runlist_csv read but contained no rows; treating runlist as None.")
        runlist = None

    # ---------------- Single pass: compute outputs ----------------
    all_rows: List[pd.DataFrame] = []
    for f in tqdm(files, desc="Compute per-subject dwell metrics"):
        csv_path = Path(f)
        try:
            df = compute_dwell_for_subject(
                csv_file=csv_path,
                video_feats_dir=video_feats_path,
                cfg=cfg,
                subj_meta_df=subj_meta,
                runlist_df=runlist,
            )
        except Exception as e:
            logging.exception("Failed on %s: %s", csv_path.name, e)
            df = None
        if df is not None and not df.empty:
            all_rows.append(df)

    if all_rows:
        master = pd.concat(all_rows, ignore_index=True)
        out_csv = out_root / f"{cfg.task}_{tag}.csv"
        master.to_csv(out_csv, index=False)
        logging.info("[ok] wrote %s (rows=%d)", out_csv, len(master))
    else:
        logging.warning("No rows produced; MASTER not written.")


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()

    cfg = RunConfig(
        video_feats_dir=args.video_feats_dir,
        window_data_dir=args.window_data_dir,
        output_root=args.output_root,
        task=args.task,
        window_glob=args.window_glob,
        subject_metadata_csv=args.subject_metadata_csv,
        runlist_csv=args.runlist_csv,
        presence_threshold=args.presence_threshold,
        min_run=args.min_run,
        bin_seconds=args.bin_seconds,
        include_joint_features=(not args.no_joint_features),
        derived_require_speaker_present=(not args.derived_allow_no_speaker),
        derived_require_speaker_unique=args.derived_require_speaker_unique,
        task5_require_speaker_unique=(not args.task5_allow_nonunique_speaker),
        include_speaker_not_averted_loose=args.task5_include_not_averted_loose,
        boundary_mode=args.boundary_mode,
        boundary_tau=args.boundary_tau,
        boundary_power=args.boundary_power,
        save_bin_level=args.save_bin_level,
        random_state=args.random_state,
        loglevel=args.loglevel,
    )
    main(cfg)

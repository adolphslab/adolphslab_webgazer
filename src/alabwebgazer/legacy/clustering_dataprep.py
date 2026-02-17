#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build participant-level clustering tables from per-run × per-feature dwell-time CSVs
produced by compute_dwelltime_feats.py.

Core outputs
- df_runs     : one row per run (per participant), run-level QC/supply columns
- df_runlevel : one row per participant, pooled totals + QC weighted means + subject meta
- df_feat_long: one row per (participant, feature) with pooled feature metrics
- df_feat_wide: df_feat_long pivoted wide
- df_cluster  : df_runlevel merged with df_feat_wide (+ optional stability block)

Pooling rules (across runs, within participant and feature)
- Count-like sufficient stats: SUM across runs (support columns)
- Rates/proportions: ratio-of-sums = sum(num) / sum(den)                  [*_rosum]
- Chance baselines e_random_*: exposure-weighted mean                      [*_wtime]
- Equal-run sensitivity: simple mean of per-run metrics                    [*_meanrun]
- Enrichment: recomputed from pooled (rate, chance); never pooled by averaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# =============================
# Configuration
# =============================
@dataclass(frozen=True)
class BuildConfig:
    """
    Configuration for merging dwell-time exports and computing subject-level metrics.

    Inputs are two CSVs produced by compute_dwelltime_feats.py (one per cohort), in "long" format:
      one row per (run, feature).

    The pipeline:
      1) Read + normalize + filter rows -> df_mergedmain (long)
      2) Collapse to one row per run -> df_runs
      3) Aggregate run-level QC + totals to one row per subject -> df_runlevel
      4) Pool feature endpoints across runs to one row per (subject, feature) -> df_feat_long
      5) Pivot to wide (subject rows, feature-suffixed columns) -> df_feat_wide
      6) Merge runlevel + feature-wide -> df_cluster

    Notes
    -----
    - For pooled rates/precisions, we use ratio-of-sums (rosum) wherever the numerator/denominator
      are available. For chance baselines (e_random_*), we use exposure-weighted means.
    - For Task5 overlap-aware derived features (speaker_averted, speaker_not_averted), we treat them
      as ordinary features; speaker-label quality covariates are pooled separately at runlevel.
    """
    pro_csv: Path
    spark_csv: Path
    out_dir: Optional[Path] = None

    # Filtering
    use_run_only: bool = True
    groups_keep: Tuple[str, ...] = ("ASD", "SPARK", "Control", "NonASD_Psych")
    features_keep: Optional[Tuple[str, ...]] = (
        "speaker",
        "distraction",
        "averted",
        # derived / joint features from compute_dwelltime_feats.py
        "distraction_offspeaker",
        "distraction_onspeaker",
        "speaker_averted",
        "speaker_not_averted",
        # optional sensitivity endpoint (only present if exported)
        "speaker_not_averted_loose",
    )
    task_prefixes_keep: Optional[Tuple[str, ...]] = ("task5_",)

    min_frac_y_valid_run: Optional[float] = None
    min_n_y_valid_run: Optional[int] = None

    default_bin_seconds: Optional[float] = 0.5
    default_boundary_mode: str = "posthoc"

    # Force group label by cohort (keep SPARK separate)
    force_group_by_cohort: Dict[str, str] = field(default_factory=lambda: {"SPARK": "SPARK"})

    # Output variants
    compute_rosum: bool = True
    compute_meanrun: bool = True

    # Per-run metrics to also average with equal weight across runs (meanrun).
    # Enrichment metrics are intentionally excluded: pooled enrichment is recomputed
    # from pooled rate + pooled chance.
    meanrun_cols: Tuple[str, ...] = (
        "rate_over_valid",
        "rate_over_raw",
        "rate_over_raw_trim",
        "rate_given_present_any",
        "rate_over_attended_valid",
        "rate_over_valid_baw",
        "rate_given_present_any_baw",
        "frac_present_any_raw",
        "frac_present_any_valid",
        "valid_over_raw",
        "frac_y_missing_given_present_any_raw",
        "frac_y_missing_given_absent_raw",
        "missingness_differential",
        "frac_exact1_valid",
        "precision_exact1_valid",
        "precision_exact1_valid_baw",
        "e_random_raw",
        "e_random_raw_trim",
        "e_random_valid",
        "e_random_valid_qbase",
        "e_random_valid_baw",
        # boundary diagnostics (per-run)
        "frac_near_boundary_valid",
        "frac_near_boundary_given_present",
        "center_bias_differential_given_present",
    )

    # Run QC columns to weighted-mean at subject level
    run_qc_valid_cols: Tuple[str, ...] = (
        "median_run_len",
        "flicker_index",
        "short_run_share_lt3",
        "quad_entropy",
        "ess_baw",
        "top1pct_share_baw",
        "mean_w_boundary_all",
        "mean_w_boundary_valid",
        "mean_w_boundary_at_flips",
        "frac_near_boundary_valid",
        "switch_rate",
        "contig_rate",
        "gap_rate",
        "diagonal_rate",
        "abab_rate",
    )
    run_qc_raw_cols: Tuple[str, ...] = (
        "frac_y_valid_first_half",
        "frac_y_valid_second_half",
        "frac_y_valid_qrt1",
        "frac_y_valid_qrt2",
        "frac_y_valid_qrt3",
        "frac_y_valid_qrt4",
        "tail_y_missing_frac_raw",
    )

    # Optional stability block (run-to-run variability per feature)
    add_stability_block: bool = False
    stability_weight_col: str = "n_y_valid"
    stability_metrics: Tuple[str, ...] = (
        "rate_over_valid",
        "rate_given_present_any",
        "rate_over_valid_baw",
        "precision_exact1_valid",
        "precision_exact1_valid_baw",
        "enrich_over_valid_norm",
        "enrich_over_valid_norm_baw",
    )

    keep_support_counts: bool = True
    write_spearman_corr: bool = False

    totals_suffix: str = "_total"

    # Validation / diagnostics
    warn_on_suspicious_values: bool = True


# =============================
# Low-level helpers
# =============================
_TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on"}


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with best-effort engine selection."""
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path)


def _norm_str(s: pd.Series) -> pd.Series:
    """Normalize string-like columns: strip and convert sentinel strings to NA."""
    s2 = s.astype("string").str.strip()
    return s2.replace(
        {
            "": pd.NA,
            "nan": pd.NA,
            "NaN": pd.NA,
            "none": pd.NA,
            "None": pd.NA,
            "NA": pd.NA,
            "N/A": pd.NA,
        }
    )


def _coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def _to_bool_series(s: pd.Series) -> pd.Series:
    """
    Robust boolean parsing:
      - numeric: 0 -> False; nonzero -> True
      - bool: preserved
      - string: matches common truthy strings
      - missing -> False
    """
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    # Fast path: numeric-ish
    sn = pd.to_numeric(s, errors="coerce")
    if sn.notna().any():
        return sn.fillna(0).astype(float).ne(0.0)

    ss = _norm_str(s).fillna("")
    return ss.astype(str).str.lower().isin(_TRUE_STRINGS)


def _mode_str(s: pd.Series) -> object:
    s = _norm_str(s).dropna()
    if s.empty:
        return pd.NA
    vc = s.value_counts(dropna=True)
    return vc.index[0] if not vc.empty else pd.NA


def _first_nonnull(s: pd.Series) -> object:
    """
    Return the first non-null value in a group.
    This is safer than pandas' "first" when some feature-rows in a run
    have NaNs for run-level numeric columns (e.g., messy merges).
    """
    x = s.dropna()
    if x.empty:
        return np.nan
    return x.iloc[0]


def _join_uniq(s: pd.Series) -> str:
    s = _norm_str(s).dropna()
    if s.empty:
        return ""
    vals = sorted(set(map(str, s.tolist())))
    return "|".join(vals)


def _is_numeric_or_bool(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


def _safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    den2 = den.replace(0, np.nan)
    return num / den2


def _is_sum_like_name(col: str) -> bool:
    """Heuristic: count/exposure/sufficient-stat fields that should be summed across runs."""
    c = col.lower()
    if c == "bin_seconds":
        return False
    return (
        c.startswith(("n_", "den_", "num_"))
        or c.endswith(("_bins", "_seconds", "_num"))
        or c in {
            "dwell_bins",
            "dwell_bins_trim",
            "dwell_seconds",
            "dwell_baw",
            "den_valid_baw",
            "den_present_any_baw",
            "chance_valid_baw_num",
            "den_exact1_baw",
            "num_exact1_correct_baw",
            "tail_y_missing_bins",
            "x_feature_len",
            "x_feature_seconds",
            "raw_seconds_est",
            "valid_seconds_est",
        }
    )


def _is_feature_specific_name(col: str) -> bool:
    """
    Name-based guard: treat these as feature-level even if constant within a run by chance.
    """
    c = col.lower()
    return (
        c.startswith(("rate_", "enrich_", "e_random_", "dwell_", "precision_", "chance_"))
        or "present_any" in c
        or "exact1" in c
        or "missing_given_" in c
        or "attended" in c
        or c in {"valid_over_raw", "missingness_differential", "center_bias_differential_given_present"}
    )


# =============================
# Aggregation primitives
# =============================
def ratio_of_sums_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    num_col: str,
    den_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Compute pooled ratio-of-sums: sum(num)/sum(den) within each group.

    Important: rows with missing numerator/denominator (or non-positive denominator)
    should not contribute to either sum (prevents biased pooling when one side is NaN).
    """
    group_cols = list(group_cols)
    base = df[group_cols].drop_duplicates()
    if num_col not in df.columns or den_col not in df.columns:
        return base.assign(**{out_col: np.nan}).reset_index(drop=True)

    tmp = df[group_cols + [num_col, den_col]].copy()
    tmp[num_col] = pd.to_numeric(tmp[num_col], errors="coerce")
    tmp[den_col] = pd.to_numeric(tmp[den_col], errors="coerce")

    # Drop missing pairs and denominators that cannot define a rate.
    tmp = tmp.dropna(subset=[num_col, den_col])
    tmp = tmp.loc[tmp[den_col] > 0]
    if tmp.empty:
        return base.assign(**{out_col: np.nan}).reset_index(drop=True)

    sums = tmp.groupby(group_cols, dropna=False)[[num_col, den_col]].sum(min_count=1).reset_index()
    sums[out_col] = _safe_div(sums[num_col], sums[den_col])
    return base.merge(sums[group_cols + [out_col]], on=group_cols, how="left", validate="one_to_one")


def exposure_weighted_mean_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    exposure_col: str,
    out_col: str,
) -> pd.DataFrame:
    """
    Exposure-weighted mean: sum(value * exposure) / sum(exposure), skipping NaN value rows.

    This is the correct pooling for run-level chance baselines e_random_* when exposures differ.
    """
    group_cols = list(group_cols)
    if value_col not in df.columns or exposure_col not in df.columns:
        return df[group_cols].drop_duplicates().assign(**{out_col: np.nan})

    tmp = df[group_cols + [value_col, exposure_col]].copy()
    v = pd.to_numeric(tmp[value_col], errors="coerce")
    e = pd.to_numeric(tmp[exposure_col], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Ignore exposure where value is missing
    e_eff = e.where(v.notna(), other=0.0)
    num = (v.fillna(0.0) * e_eff)

    sums = (
        pd.concat([tmp[group_cols], num.rename("_num"), e_eff.rename("_den")], axis=1)
        .groupby(group_cols, dropna=False)[["_num", "_den"]]
        .sum(min_count=1)
        .reset_index()
    )
    sums[out_col] = _safe_div(sums["_num"], sums["_den"])
    return sums[group_cols + [out_col]]


def weighted_mean_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    weight_col: str,
) -> pd.DataFrame:
    """NaN-safe weighted mean with per-column denominators (weights ignored where x is NaN)."""
    group_cols = list(group_cols)
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols or weight_col not in df.columns:
        return df[group_cols].drop_duplicates().reset_index(drop=True)

    tmp = df[group_cols + value_cols + [weight_col]].copy()
    w = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    x = tmp[value_cols].apply(pd.to_numeric, errors="coerce")

    w_eff = x.notna().mul(w, axis=0)
    sw = pd.concat([tmp[group_cols], w_eff], axis=1).groupby(group_cols, dropna=False)[value_cols].sum(min_count=1)
    swx = pd.concat([tmp[group_cols], x.mul(w, axis=0)], axis=1).groupby(group_cols, dropna=False)[value_cols].sum(min_count=1)

    mu = swx.div(sw.replace(0, np.nan))
    return mu.reset_index()


def _weighted_std_by_group(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_cols: Sequence[str],
    weight_col: str,
) -> pd.DataFrame:
    """NaN-safe weighted population std: sqrt(Ew[x^2] - Ew[x]^2)."""
    group_cols = list(group_cols)
    value_cols = [c for c in value_cols if c in df.columns]
    if not value_cols or weight_col not in df.columns:
        return df[group_cols].drop_duplicates().reset_index(drop=True)

    tmp = df[group_cols + value_cols + [weight_col]].copy()
    w = pd.to_numeric(tmp[weight_col], errors="coerce").fillna(0.0).clip(lower=0.0).astype(float)
    x = tmp[value_cols].apply(pd.to_numeric, errors="coerce")

    w_eff = x.notna().mul(w, axis=0)
    sw = pd.concat([tmp[group_cols], w_eff], axis=1).groupby(group_cols, dropna=False)[value_cols].sum(min_count=1)
    swx = pd.concat([tmp[group_cols], x.mul(w, axis=0)], axis=1).groupby(group_cols, dropna=False)[value_cols].sum(min_count=1)
    swxx = pd.concat([tmp[group_cols], x.pow(2).mul(w, axis=0)], axis=1).groupby(group_cols, dropna=False)[value_cols].sum(min_count=1)

    mu = swx.div(sw.replace(0, np.nan))
    ex2 = swxx.div(sw.replace(0, np.nan))
    var = (ex2 - mu.pow(2)).clip(lower=0.0)
    return np.sqrt(var).reset_index()


def _entropy_from_counts(counts: np.ndarray) -> np.ndarray:
    s = np.nansum(counts, axis=1, keepdims=True)
    p = np.divide(counts, s, out=np.full_like(counts, np.nan, dtype=float), where=s > 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)


def _enrich(rate: pd.Series, chance: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """
    Enrichment:
      diff = rate - chance
      norm = (rate - chance) / (1 - chance)
    """
    diff = rate - chance
    denom = (1.0 - chance).replace(0, np.nan)
    norm = diff / denom
    return diff, norm


# =============================
# Run identity + schema utilities
# =============================
def infer_run_id_cols(df: pd.DataFrame) -> List[str]:
    """Best-effort inference of columns that uniquely define a run within a subject."""
    cols: List[str] = []
    for c in ["SID", "task"]:
        if c in df.columns:
            cols.append(c)

    for c in ["run", "run_id", "run_idx", "run_index", "trial", "trial_id", "session", "session_id", "block"]:
        if c in df.columns:
            cols.append(c)

    # Need a stimulus identifier
    if "Video" in df.columns:
        cols.append("Video")
    elif "condition" in df.columns:
        cols.append("condition")
    else:
        raise ValueError("Need at least one of ['Video', 'condition'] to define a run.")

    # If both exist and differ, keep both
    if "Video" in df.columns and "condition" in df.columns:
        v = _norm_str(df["Video"]).str.lower()
        c = _norm_str(df["condition"]).str.lower()
        if not v.fillna("").equals(c.fillna("")):
            cols.append("condition")

    out: List[str] = []
    for c in cols:
        if c in df.columns and c not in out:
            out.append(c)
    return out


def split_runlevel_vs_featurelevel_cols(
    df: pd.DataFrame,
    run_id_cols: Sequence[str],
    *,
    feature_col: str = "feature",
) -> Tuple[List[str], List[str]]:
    """
    Run-level columns: constant within run across feature rows (ignoring NaNs),
    AND not name-marked as feature-specific.
    """
    run_id_cols = list(run_id_cols)
    missing = [c for c in run_id_cols + [feature_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    candidate = [c for c in df.columns if c not in set(run_id_cols) | {feature_col}]
    if not candidate:
        return [], []

    nunique_nonnull = df.groupby(run_id_cols, dropna=False)[candidate].nunique(dropna=True)
    run_level = [c for c in candidate if (nunique_nonnull[c].max() <= 1) and (not _is_feature_specific_name(c))]
    feat_level = [c for c in candidate if c not in run_level]
    return run_level, feat_level


def dedup_run_feature_rows(df: pd.DataFrame, run_id_cols: Sequence[str], feature_col: str = "feature") -> pd.DataFrame:
    """Ensure a single row per (run_id_cols, feature). Keeps first if duplicates exist."""
    key = list(run_id_cols) + [feature_col]
    if df.duplicated(key).any():
        log.warning("Duplicate (run, feature) rows detected; keeping first per key.")
        return df.groupby(key, dropna=False, as_index=False).first()
    return df


def ensure_support_cols(df: pd.DataFrame, cfg: BuildConfig) -> pd.DataFrame:
    """
    Validate + normalize the minimum set of support columns used downstream.

    This script expects "jointfeat" exports from compute_dwelltime_feats.py.
    """
    out = df.copy()

    out["boundary_mode"] = (
        _norm_str(out["boundary_mode"]).str.lower()
        if "boundary_mode" in out.columns
        else cfg.default_boundary_mode
    )

    required = [
        "dwell_bins", "dwell_bins_trim", "dwell_baw",
        "n_raw", "n_raw_trim", "n_y_valid",
        "n_present_any_raw", "n_present_any_valid",
        "den_valid_baw", "den_present_any_baw",
        "e_random_valid_baw", "chance_valid_baw_num",
        "n_exact1_valid", "n_exact1_correct",
        "den_exact1_baw", "num_exact1_correct_baw",
    ]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(
            f"CSV schema mismatch: missing {missing}. "
            "Re-export with the current compute_dwelltime_feats.py (jointfeat)."
        )

    # ADDED for correctness: guard against NaN denominators when the corresponding count is 0
    def zero_if_empty(count_col: str, cols: Sequence[str]) -> None:
        if count_col not in out.columns:
            return
        m = pd.to_numeric(out[count_col], errors="coerce").fillna(0) <= 0
        for c in cols:
            if c in out.columns:
                out.loc[m, c] = pd.to_numeric(out.loc[m, c], errors="coerce").fillna(0.0)

    zero_if_empty("n_present_any_valid", ["den_present_any_baw"])
    zero_if_empty("n_exact1_valid", ["den_exact1_baw", "num_exact1_correct_baw"])

    return out


# =============================
# IO + filtering
# =============================
def _canonicalize_group_labels(df: pd.DataFrame, groups_keep: Sequence[str]) -> None:
    """
    Canonicalize Group labels to match groups_keep (case-insensitive).

    Example: "control" -> "Control" if "Control" in groups_keep.
    """
    if "Group" not in df.columns or not groups_keep:
        return

    mapping = {str(g).strip().upper(): str(g).strip() for g in groups_keep if pd.notna(g)}
    g = _norm_str(df["Group"])
    g_up = g.astype("string").str.upper()
    df["Group"] = g_up.map(mapping).astype("string")


def load_and_filter(cfg: BuildConfig) -> pd.DataFrame:
    """Read Prolific + SPARK CSVs, normalize, and apply filters."""
    if not Path(cfg.pro_csv).exists():
        raise FileNotFoundError(cfg.pro_csv)
    if not Path(cfg.spark_csv).exists():
        raise FileNotFoundError(cfg.spark_csv)

    df_pro = _read_csv(Path(cfg.pro_csv))
    df_sp = _read_csv(Path(cfg.spark_csv))

    df_pro["Cohort"] = "Prolific"
    df_sp["Cohort"] = "SPARK"
    df = pd.concat([df_pro, df_sp], ignore_index=True)

    # Normalize key string columns (defensive)
    for c in ["ID", "task", "feature", "Group", "Sex", "Cohort", "condition", "Video", "feature_key", "boundary_mode"]:
        if c in df.columns:
            df[c] = _norm_str(df[c])

    # Lowercase task + feature names for stable matching
    if "task" in df.columns:
        df["task"] = df["task"].astype("string").str.lower()
    if "feature" in df.columns:
        df["feature"] = df["feature"].astype("string").str.lower()

    # Ensure Video exists for run identity / diagnostics
    if "Video" not in df.columns and "condition" in df.columns:
        df["Video"] = df["condition"]
    elif "Video" in df.columns and "condition" in df.columns:
        df["Video"] = df["Video"].fillna(df["condition"])

    # Force cohort->group mapping (e.g., SPARK cohort labeled as "SPARK")
    if cfg.force_group_by_cohort and {"Cohort", "Group"}.issubset(df.columns):
        for cohort_name, group_label in cfg.force_group_by_cohort.items():
            m = df["Cohort"].astype("string").str.upper() == str(cohort_name).upper()
            df.loc[m, "Group"] = str(group_label)

    # Canonicalize group labels to match cfg.groups_keep (case-insensitive)
    _canonicalize_group_labels(df, cfg.groups_keep)

    # Coerce common numerics used downstream
    _coerce_numeric(df, ["AQ", "SRS", "Age", "bin_seconds", "n_raw", "n_raw_trim", "n_y_valid", "frac_y_valid"])
    _coerce_numeric(df, ["n_flips", "n_switches", "n_contig_steps", "contig_rate", "gap_rate", "switch_rate"])

    # Backward/forward compatibility for switching metrics
    if ("n_switches" in df.columns) and ("n_flips" not in df.columns):
        df["n_flips"] = pd.to_numeric(df["n_switches"], errors="coerce")
    if ("switch_rate" not in df.columns) and {"n_switches", "n_contig_steps"}.issubset(df.columns):
        den = pd.to_numeric(df["n_contig_steps"], errors="coerce").replace(0, np.nan)
        df["switch_rate"] = pd.to_numeric(df["n_switches"], errors="coerce") / den
    if ("contig_rate" not in df.columns) and {"n_contig_steps", "n_y_valid"}.issubset(df.columns):
        den = (pd.to_numeric(df["n_y_valid"], errors="coerce") - 1).clip(lower=0).replace(0, np.nan)
        df["contig_rate"] = pd.to_numeric(df["n_contig_steps"], errors="coerce") / den
    if ("gap_rate" not in df.columns) and ("contig_rate" in df.columns):
        df["gap_rate"] = 1.0 - pd.to_numeric(df["contig_rate"], errors="coerce")

    # Fill bin_seconds if requested
    if cfg.default_bin_seconds is not None:
        if "bin_seconds" not in df.columns:
            df["bin_seconds"] = float(cfg.default_bin_seconds)
        else:
            df["bin_seconds"] = df["bin_seconds"].fillna(float(cfg.default_bin_seconds))

    # Filters
    if cfg.use_run_only and "use_run" in df.columns:
        df = df[_to_bool_series(df["use_run"])].copy()

    if cfg.groups_keep and "Group" in df.columns:
        df = df[df["Group"].isin(list(cfg.groups_keep))].copy()

    if cfg.task_prefixes_keep is not None and "task" in df.columns:
        prefixes = tuple(str(p).lower() for p in cfg.task_prefixes_keep)
        df = df[df["task"].astype(str).str.startswith(prefixes)].copy()

    if cfg.features_keep is not None and "feature" in df.columns:
        feats = [str(f).lower() for f in cfg.features_keep]
        df = df[df["feature"].isin(feats)].copy()

        # Fail loudly if a required feature silently disappears due to wrong input file
        expected = set(feats)
        present = set(df["feature"].dropna().unique().tolist())
        optional = {"speaker_not_averted_loose"}
        missing_expected = sorted(expected - present - optional)
        if missing_expected:
            raise ValueError(
                "After filtering, expected features are missing: "
                f"{missing_expected}. "
                "Likely causes: using a non-jointfeat CSV, task mismatch, or upstream export settings."
            )

    # Run-level validity filters (prefer explicit frac_y_valid column; fall back to n_y_valid/n_raw)
    if cfg.min_frac_y_valid_run is not None:
        if "frac_y_valid" in df.columns:
            df = df[pd.to_numeric(df["frac_y_valid"], errors="coerce") >= float(cfg.min_frac_y_valid_run)].copy()
        elif {"n_y_valid", "n_raw"}.issubset(df.columns):
            frac = pd.to_numeric(df["n_y_valid"], errors="coerce") / pd.to_numeric(df["n_raw"], errors="coerce").replace(0, np.nan)
            df = df[frac >= float(cfg.min_frac_y_valid_run)].copy()

    if cfg.min_n_y_valid_run is not None and "n_y_valid" in df.columns:
        df = df[pd.to_numeric(df["n_y_valid"], errors="coerce") >= int(cfg.min_n_y_valid_run)].copy()

    if "ID" not in df.columns:
        raise ValueError("Expected column 'ID' in dwell-time CSVs.")

    df["SID"] = df["Cohort"].astype("string") + ":" + df["ID"].astype("string")

    if df.empty:
        raise ValueError(
            "No rows remain after filtering. Check groups_keep / features_keep / task_prefixes_keep / use_run_only."
        )

    # Consistency check: Group should be constant within SID (after canonicalization)
    if "Group" in df.columns:
        g_nu = df.groupby("SID", dropna=False)["Group"].nunique(dropna=True)
        if (g_nu > 1).any():
            bad = g_nu[g_nu > 1].index[:5].tolist()
            raise ValueError(f"Inconsistent Group labels within SID for {len(bad)} subjects (e.g., {bad}).")

    if cfg.warn_on_suspicious_values:
        _warn_basic_sanity(df)

    return df.reset_index(drop=True)


def _warn_basic_sanity(df: pd.DataFrame) -> None:
    """Cheap sanity checks; warnings only."""
    for c in ["n_raw", "n_y_valid", "dwell_bins", "n_present_any_valid", "n_present_any_raw", "n_flips", "n_switches", "n_contig_steps"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if (x < 0).any():
                log.warning("Column %s has negative values; check upstream exports.", c)

    for c in ["frac_y_valid", "frac_present_any_valid", "rate_over_valid", "rate_over_raw", "switch_rate", "diagonal_rate", "abab_rate", "contig_rate", "gap_rate"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce")
            if ((x < -1e-6) | (x > 1 + 1e-6)).any():
                log.warning("Column %s has values outside [0,1]; check upstream exports.", c)

    # Warn if compute-config flags differ within file (likely indicates mixed exports)
    flag_cols = ["include_joint_features", "derived_require_speaker_present", "derived_require_speaker_unique", "task5_require_speaker_unique", "boundary_mode"]
    present = [c for c in flag_cols if c in df.columns]
    for c in present:
        nu = df[c].nunique(dropna=True)
        if nu > 1:
            log.warning("Column %s has %d distinct non-null values; check for mixed export settings.", c, int(nu))


# =============================
# Build tables
# =============================
def build_df_runs(df: pd.DataFrame, *, run_id_cols: Sequence[str], run_level_cols: Sequence[str]) -> pd.DataFrame:
    """One row per run (deduplicated across features)."""
    run_id_cols = list(run_id_cols)
    run_level_cols = [c for c in run_level_cols if c in df.columns]

    num_cols = [c for c in run_level_cols if _is_numeric_or_bool(df[c])]
    cat_cols = [c for c in run_level_cols if c not in num_cols]

    agg: Dict[str, object] = {c: _first_nonnull for c in num_cols}
    agg.update({c: _mode_str for c in cat_cols})

    df_runs = df.groupby(run_id_cols, dropna=False).agg(agg).reset_index()

    diag = (
        df.groupby(run_id_cols, dropna=False)
        .agg(n_rows_in_run=("feature", "size"), n_feature_rows_in_run=("feature", "nunique"))
        .reset_index()
    )
    df_runs = df_runs.merge(diag, on=run_id_cols, how="left", validate="one_to_one")

    # seconds estimates (allow bin_seconds varying by run)
    if {"n_raw", "bin_seconds"}.issubset(df_runs.columns):
        df_runs["raw_seconds_est"] = pd.to_numeric(df_runs["n_raw"], errors="coerce") * pd.to_numeric(df_runs["bin_seconds"], errors="coerce")
    if {"n_y_valid", "bin_seconds"}.issubset(df_runs.columns):
        df_runs["valid_seconds_est"] = pd.to_numeric(df_runs["n_y_valid"], errors="coerce") * pd.to_numeric(df_runs["bin_seconds"], errors="coerce")

    return df_runs


def build_df_runlevel(df_runs: pd.DataFrame, *, run_id_cols: Sequence[str], cfg: BuildConfig) -> pd.DataFrame:
    """One row per subject with pooled run-level QC and totals."""
    if "SID" not in df_runs.columns:
        raise ValueError("df_runs must contain 'SID'.")

    run_id_cols = list(run_id_cols)

    df_runlevel = df_runs.groupby("SID", dropna=False).size().reset_index(name="n_runs")

    # Sum-like totals (avoid feature double-count by using df_runs)
    diag_cols = {"n_rows_in_run", "n_feature_rows_in_run"}
    candidate = [c for c in df_runs.columns if c not in set(run_id_cols) | {"SID"} | diag_cols]
    sum_cols = [c for c in candidate if _is_numeric_or_bool(df_runs[c]) and _is_sum_like_name(c)]
    if sum_cols:
        sums = df_runs.groupby("SID", dropna=False)[sum_cols].sum(min_count=1).reset_index()
        sums = sums.rename(columns={c: f"{c}{cfg.totals_suffix}" for c in sum_cols})
        df_runlevel = df_runlevel.merge(sums, on="SID", how="left", validate="one_to_one")

    # Video list diagnostic
    if "Video" in df_runs.columns:
        vids = df_runs.groupby("SID", dropna=False)["Video"].agg(video_list=_join_uniq).reset_index()
        df_runlevel = df_runlevel.merge(vids, on="SID", how="left", validate="one_to_one")

    # Overall validity/missingness (pooled)
    nraw = f"n_raw{cfg.totals_suffix}"
    nyv = f"n_y_valid{cfg.totals_suffix}"
    if {nraw, nyv}.issubset(df_runlevel.columns):
        df_runlevel["frac_y_valid_overall"] = df_runlevel[nyv] / df_runlevel[nraw].replace(0, np.nan)
        df_runlevel["frac_y_missing_overall"] = 1.0 - df_runlevel["frac_y_valid_overall"]

    # Quadrant entropy from pooled quadrant counts
    qcols = [f"n_q{i}_y_valid{cfg.totals_suffix}" for i in range(1, 5)]
    if all(c in df_runlevel.columns for c in qcols):
        df_runlevel["quad_entropy_overall"] = _entropy_from_counts(df_runlevel[qcols].to_numpy(float))

    # Flip rate per minute from pooled seconds
    flips = f"n_flips{cfg.totals_suffix}"
    if flips not in df_runlevel.columns:
        alt = f"n_switches{cfg.totals_suffix}"
        if alt in df_runlevel.columns:
            flips = alt
    raw_sec = f"raw_seconds_est{cfg.totals_suffix}"
    val_sec = f"valid_seconds_est{cfg.totals_suffix}"
    if {flips, raw_sec}.issubset(df_runlevel.columns):
        df_runlevel["flip_rate_per_min_overall"] = df_runlevel[flips] / (df_runlevel[raw_sec] / 60.0).replace(0, np.nan)
    if {flips, val_sec}.issubset(df_runlevel.columns):
        df_runlevel["flip_rate_per_min_valid_overall"] = df_runlevel[flips] / (df_runlevel[val_sec] / 60.0).replace(0, np.nan)

    # ---- Speaker-label diagnostics (optional; present only in newer dwell exports) ----
    sp_any_raw = f"n_speaker_any_raw{cfg.totals_suffix}"
    sp_unique_raw = f"n_speaker_unique_raw{cfg.totals_suffix}"
    sp_any_valid = f"n_speaker_any_valid{cfg.totals_suffix}"
    sp_unique_valid = f"n_speaker_unique_valid{cfg.totals_suffix}"
    sp_unk_raw = f"n_speaker_averted_unknown_raw{cfg.totals_suffix}"
    sp_unk_valid = f"n_speaker_averted_unknown_valid{cfg.totals_suffix}"

    if {sp_any_raw, sp_unique_raw, sp_any_valid, sp_unique_valid}.issubset(df_runlevel.columns) and {nraw, nyv}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_any_raw_overall"] = df_runlevel[sp_any_raw] / df_runlevel[nraw].replace(0, np.nan)
        df_runlevel["frac_speaker_unique_raw_overall"] = df_runlevel[sp_unique_raw] / df_runlevel[nraw].replace(0, np.nan)
        df_runlevel["frac_speaker_any_valid_overall"] = df_runlevel[sp_any_valid] / df_runlevel[nyv].replace(0, np.nan)
        df_runlevel["frac_speaker_unique_valid_overall"] = df_runlevel[sp_unique_valid] / df_runlevel[nyv].replace(0, np.nan)

    if {sp_unk_raw, sp_unique_raw}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_averted_unknown_raw_overall"] = df_runlevel[sp_unk_raw] / df_runlevel[sp_unique_raw].replace(0, np.nan)
    if {sp_unk_valid, sp_unique_valid}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_averted_unknown_valid_overall"] = df_runlevel[sp_unk_valid] / df_runlevel[sp_unique_valid].replace(0, np.nan)

    if {sp_unique_raw, sp_any_raw}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_unique_given_speaker_any_raw_overall"] = df_runlevel[sp_unique_raw] / df_runlevel[sp_any_raw].replace(0, np.nan)
    if {sp_unique_valid, sp_any_valid}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_unique_given_speaker_any_valid_overall"] = df_runlevel[sp_unique_valid] / df_runlevel[sp_any_valid].replace(0, np.nan)

    if {sp_unk_raw, sp_any_raw}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_averted_unknown_given_speaker_any_raw_overall"] = df_runlevel[sp_unk_raw] / df_runlevel[sp_any_raw].replace(0, np.nan)
    if {sp_unk_valid, sp_any_valid}.issubset(df_runlevel.columns):
        df_runlevel["frac_speaker_averted_unknown_given_speaker_any_valid_overall"] = df_runlevel[sp_unk_valid] / df_runlevel[sp_any_valid].replace(0, np.nan)

    # ---- Weighted means for run-level QC (weights aligned to metric denominator) ----
    df_w = df_runs.copy()

    if "n_y_valid" in df_w.columns:
        w_valid = pd.to_numeric(df_w["n_y_valid"], errors="coerce").fillna(0.0).clip(lower=0.0)
        df_w["_w_valid"] = w_valid
        df_w["_w_steps"] = (w_valid - 1.0).clip(lower=0.0)  # ~ adjacent steps
        df_w["_w_abab"] = (w_valid - 2.0).clip(lower=0.0)   # ~ 3-step windows

    if "n_raw" in df_w.columns:
        df_w["_w_raw"] = pd.to_numeric(df_w["n_raw"], errors="coerce").fillna(0.0).clip(lower=0.0)

    if "n_contig_steps" in df_w.columns:
        df_w["_w_contig_steps"] = pd.to_numeric(df_w["n_contig_steps"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif "_w_steps" in df_w.columns:
        df_w["_w_contig_steps"] = df_w["_w_steps"]

    if "n_switches" in df_w.columns:
        df_w["_w_switches"] = pd.to_numeric(df_w["n_switches"], errors="coerce").fillna(0.0).clip(lower=0.0)
    elif "n_flips" in df_w.columns:
        df_w["_w_switches"] = pd.to_numeric(df_w["n_flips"], errors="coerce").fillna(0.0).clip(lower=0.0)

    # Derive switch_rate if missing but counts present
    if "switch_rate" not in df_w.columns and {"n_switches", "n_contig_steps"}.issubset(df_w.columns):
        df_w["switch_rate"] = pd.to_numeric(df_w["n_switches"], errors="coerce") / pd.to_numeric(df_w["n_contig_steps"], errors="coerce").replace(0, np.nan)

    # Choose weights per metric class
    weights_map: Dict[str, str] = {}
    for c in cfg.run_qc_valid_cols:
        if c == "abab_rate":
            weights_map[c] = "_w_abab"
        elif c in {"contig_rate", "gap_rate"}:
            weights_map[c] = "_w_steps"
        elif c == "switch_rate":
            weights_map[c] = "_w_contig_steps"
        elif c in {"diagonal_rate", "mean_w_boundary_at_flips"}:
            weights_map[c] = "_w_switches"
        elif c == "mean_w_boundary_all":
            # This is "all bins" in the exporter; best available weight is n_raw.
            weights_map[c] = "_w_raw"
        else:
            weights_map[c] = "_w_valid"

    qc_valid = [c for c in cfg.run_qc_valid_cols if c in df_w.columns]
    if qc_valid:
        # group metrics by weight for fewer groupby passes
        by_w: Dict[str, List[str]] = {}
        for c in qc_valid:
            w = weights_map.get(c, "_w_valid")
            if w in df_w.columns:
                by_w.setdefault(w, []).append(c)

        for wcol, cols in by_w.items():
            wmean = weighted_mean_by_group(df_w, ["SID"], cols, wcol).rename(columns={c: f"{c}_wmean" for c in cols})
            df_runlevel = df_runlevel.merge(wmean, on="SID", how="left", validate="one_to_one")

    qc_raw = [c for c in cfg.run_qc_raw_cols if c in df_w.columns]
    if qc_raw and "_w_raw" in df_w.columns:
        wmean = weighted_mean_by_group(df_w, ["SID"], qc_raw, "_w_raw").rename(columns={c: f"{c}_wmean" for c in qc_raw})
        df_runlevel = df_runlevel.merge(wmean, on="SID", how="left", validate="one_to_one")

    # Subject meta
    meta_agg: Dict[str, object] = {}
    for c in ["Cohort", "Group", "Sex"]:
        if c in df_runs.columns:
            meta_agg[c] = _mode_str
    # Dwell-export compute-config flags (keeps provenance when merging cohorts)
    for c in ["include_joint_features", "derived_require_speaker_present", "derived_require_speaker_unique", "task5_require_speaker_unique", "boundary_mode"]:
        if c in df_runs.columns:
            meta_agg[c] = "first"
    for c in ["AQ", "SRS", "Age"]:
        if c in df_runs.columns:
            meta_agg[c] = "mean"
    if meta_agg:
        meta = df_runs.groupby("SID", dropna=False).agg(meta_agg).reset_index()
        df_runlevel = df_runlevel.merge(meta, on="SID", how="left", validate="one_to_one")

    return df_runlevel


def build_df_feat_long(
    df: pd.DataFrame,
    *,
    cfg: BuildConfig,
    run_id_cols: Sequence[str],
    feature_level_num_cols: Sequence[str],
) -> pd.DataFrame:
    """
    One row per (SID, feature) with pooled feature endpoints across runs.
    """
    gcols = ["SID", "feature"]
    run_id_cols = list(run_id_cols)

    df_f = ensure_support_cols(df, cfg)
    df_f = dedup_run_feature_rows(df_f, run_id_cols, feature_col="feature")

    # Coerce numerics broadly for pooling-related fields (defensive for newer columns)
    numeric_need = set(
        [
            "dwell_bins",
            "dwell_bins_trim",
            "dwell_baw",
            "n_raw",
            "n_raw_trim",
            "n_y_valid",
            "n_present_any_raw",
            "n_present_any_valid",
            "n_attended_valid",
            "n_attended_nan",
            "n_attended_present_valid",
            "n_y_missing_given_present_any_raw",
            "n_y_missing_given_absent_raw",
            "den_valid_baw",
            "den_present_any_baw",
            "chance_valid_baw_num",
            "n_exact1_valid",
            "n_exact1_correct",
            "den_exact1_baw",
            "num_exact1_correct_baw",
            "frac_near_boundary_given_present",
            "e_random_raw",
            "e_random_raw_trim",
            "e_random_valid",
            "e_random_valid_qbase",
            "e_random_valid_baw",
            *cfg.meanrun_cols,
        ]
    )
    _coerce_numeric(df_f, [c for c in numeric_need if c in df_f.columns])

    # Support sums (optional diagnostics / for later weighting)
    support_cols = [c for c in feature_level_num_cols if (c in df_f.columns) and _is_numeric_or_bool(df_f[c]) and _is_sum_like_name(c)]
    must_support = [
        "dwell_bins",
        "dwell_bins_trim",
        "dwell_baw",
        "n_raw",
        "n_raw_trim",
        "n_y_valid",
        "n_present_any_raw",
        "n_present_any_valid",
        "n_attended_valid",
        "n_attended_nan",
        "n_attended_present_valid",
        "n_y_missing_given_present_any_raw",
        "n_y_missing_given_absent_raw",
        "den_valid_baw",
        "den_present_any_baw",
        "chance_valid_baw_num",
        "n_exact1_valid",
        "n_exact1_correct",
        "den_exact1_baw",
        "num_exact1_correct_baw",
    ]
    for c in must_support:
        if c in df_f.columns:
            support_cols.append(c)
    support_cols = sorted(set(support_cols))

    df_sum = (
        df_f.groupby(gcols, dropna=False)[support_cols].sum(min_count=1).reset_index()
        if support_cols
        else df_f[gcols].drop_duplicates()
    )

    # ---- Ratio-of-sums pooled metrics ----
    df_rosum_base = df_f[gcols].drop_duplicates()
    rosum_parts: List[pd.DataFrame] = []

    def add_rosum(out: str, num: str, den: str) -> None:
        if num in df_f.columns and den in df_f.columns:
            rosum_parts.append(ratio_of_sums_by_group(df_f, gcols, num, den, out))

    if cfg.compute_rosum:
        add_rosum("rate_over_valid_rosum", "dwell_bins", "n_y_valid")
        add_rosum("rate_over_raw_rosum", "dwell_bins", "n_raw")
        add_rosum("rate_over_raw_trim_rosum", "dwell_bins_trim", "n_raw_trim")
        add_rosum("rate_given_present_any_rosum", "dwell_bins", "n_present_any_valid")
        add_rosum("rate_over_attended_valid_rosum", "dwell_bins", "n_attended_valid")

        add_rosum("frac_present_any_raw_rosum", "n_present_any_raw", "n_raw")
        add_rosum("frac_present_any_valid_rosum", "n_present_any_valid", "n_y_valid")
        add_rosum("valid_over_raw_rosum", "n_present_any_valid", "n_present_any_raw")
        add_rosum("frac_y_missing_given_present_any_raw_rosum", "n_y_missing_given_present_any_raw", "n_present_any_raw")

        # absent-raw denom is exact: n_raw - n_present_any_raw
        if {"n_y_missing_given_absent_raw", "n_raw", "n_present_any_raw"}.issubset(df_f.columns):
            df_tmp = df_f.copy()
            df_tmp["den_absent_raw"] = (
                pd.to_numeric(df_tmp["n_raw"], errors="coerce")
                - pd.to_numeric(df_tmp["n_present_any_raw"], errors="coerce")
            ).clip(lower=0)
            rosum_parts.append(
                ratio_of_sums_by_group(
                    df_tmp,
                    gcols,
                    "n_y_missing_given_absent_raw",
                    "den_absent_raw",
                    "frac_y_missing_given_absent_raw_rosum",
                )
            )

        # BAW rates
        add_rosum("rate_over_valid_baw_rosum", "dwell_baw", "den_valid_baw")
        add_rosum("rate_given_present_any_baw_rosum", "dwell_baw", "den_present_any_baw")

        # Exact1 precision
        add_rosum("frac_exact1_valid_rosum", "n_exact1_valid", "n_y_valid")
        add_rosum("precision_exact1_valid_rosum", "n_exact1_correct", "n_exact1_valid")
        add_rosum("precision_exact1_valid_baw_rosum", "num_exact1_correct_baw", "den_exact1_baw")

        # Boundary: best available pooling for frac_near_boundary_given_present is to reconstruct
        # a numerator using n_present_any_valid as a proxy denominator.
        if {"frac_near_boundary_given_present", "n_present_any_valid"}.issubset(df_f.columns):
            df_tmp = df_f.copy()
            df_tmp["_num_near"] = pd.to_numeric(df_tmp["frac_near_boundary_given_present"], errors="coerce") * pd.to_numeric(
                df_tmp["n_present_any_valid"], errors="coerce"
            )
            rosum_parts.append(
                ratio_of_sums_by_group(
                    df_tmp,
                    gcols,
                    "_num_near",
                    "n_present_any_valid",
                    "frac_near_boundary_given_present_rosum",
                )
            )

        # Chance baseline (BAW) as ratio-of-sums of the stored numerator
        add_rosum("chance_valid_baw_rosum", "chance_valid_baw_num", "den_valid_baw")

    df_rosum = (
        reduce(lambda a, b: a.merge(b, on=gcols, how="left", validate="one_to_one"), [df_rosum_base] + rosum_parts)
        if rosum_parts
        else df_rosum_base
    )

    # ---- Chance baselines (exposure-weighted mean) ----
    chance_parts: List[pd.DataFrame] = []
    if cfg.compute_rosum:
        if {"e_random_raw", "n_raw"}.issubset(df_f.columns):
            chance_parts.append(exposure_weighted_mean_by_group(df_f, gcols, "e_random_raw", "n_raw", "e_random_raw_wtime"))
        if {"e_random_raw_trim", "n_raw_trim"}.issubset(df_f.columns):
            chance_parts.append(exposure_weighted_mean_by_group(df_f, gcols, "e_random_raw_trim", "n_raw_trim", "e_random_raw_trim_wtime"))
        if {"e_random_valid", "n_y_valid"}.issubset(df_f.columns):
            chance_parts.append(exposure_weighted_mean_by_group(df_f, gcols, "e_random_valid", "n_y_valid", "e_random_valid_wtime"))
        if {"e_random_valid_qbase", "n_y_valid"}.issubset(df_f.columns):
            chance_parts.append(exposure_weighted_mean_by_group(df_f, gcols, "e_random_valid_qbase", "n_y_valid", "e_random_valid_qbase_wtime"))
        # IMPORTANT: BAW chance must be weighted by den_valid_baw (sum of boundary weights), not n_y_valid.
        if {"e_random_valid_baw", "den_valid_baw"}.issubset(df_f.columns):
            chance_parts.append(exposure_weighted_mean_by_group(df_f, gcols, "e_random_valid_baw", "den_valid_baw", "e_random_valid_baw_wtime"))

    df_chance = (
        reduce(lambda a, b: a.merge(b, on=gcols, how="left", validate="one_to_one"), [df_rosum_base] + chance_parts)
        if chance_parts
        else df_rosum_base
    )

    # ---- Meanrun metrics (equal-weighted across runs) ----
    df_meanrun = df_rosum_base
    if cfg.compute_meanrun:
        cols = [c for c in cfg.meanrun_cols if c in df_f.columns]
        if cols:
            meanrun = df_f.groupby(gcols, dropna=False)[cols].mean(numeric_only=True).reset_index()
            meanrun = meanrun.rename(columns={c: f"{c}_meanrun" for c in cols})
            df_meanrun = df_meanrun.merge(meanrun, on=gcols, how="left", validate="one_to_one")

    # ---- Merge feature-level blocks ----
    df_feat_long = (
        df_sum.merge(df_rosum, on=gcols, how="left", validate="one_to_one")
        .merge(df_chance, on=gcols, how="left", validate="one_to_one", suffixes=("", "_dup"))
        .merge(df_meanrun, on=gcols, how="left", validate="one_to_one", suffixes=("", "_dup2"))
    )

    # Recompute enrichments from pooled rate + pooled chance
    if "rate_over_valid_rosum" in df_feat_long.columns and "e_random_valid_wtime" in df_feat_long.columns:
        diff, norm = _enrich(df_feat_long["rate_over_valid_rosum"], df_feat_long["e_random_valid_wtime"])
        df_feat_long["enrich_over_valid_diff_rosum"] = diff
        df_feat_long["enrich_over_valid_norm_rosum"] = norm

    if "rate_over_valid_rosum" in df_feat_long.columns and "e_random_valid_qbase_wtime" in df_feat_long.columns:
        diff, norm = _enrich(df_feat_long["rate_over_valid_rosum"], df_feat_long["e_random_valid_qbase_wtime"])
        df_feat_long["enrich_over_valid_diff_qbase_rosum"] = diff
        df_feat_long["enrich_over_valid_norm_qbase_rosum"] = norm

    # Prefer e_random_valid_baw_wtime; fall back to chance_valid_baw_rosum
    if "rate_over_valid_baw_rosum" in df_feat_long.columns:
        if "e_random_valid_baw_wtime" in df_feat_long.columns:
            diff, norm = _enrich(df_feat_long["rate_over_valid_baw_rosum"], df_feat_long["e_random_valid_baw_wtime"])
            df_feat_long["enrich_over_valid_diff_baw_rosum"] = diff
            df_feat_long["enrich_over_valid_norm_baw_rosum"] = norm
        elif "chance_valid_baw_rosum" in df_feat_long.columns:
            diff, norm = _enrich(df_feat_long["rate_over_valid_baw_rosum"], df_feat_long["chance_valid_baw_rosum"])
            df_feat_long["enrich_over_valid_diff_baw_rosum"] = diff
            df_feat_long["enrich_over_valid_norm_baw_rosum"] = norm

    if "rate_over_raw_rosum" in df_feat_long.columns and "e_random_raw_wtime" in df_feat_long.columns:
        diff, norm = _enrich(df_feat_long["rate_over_raw_rosum"], df_feat_long["e_random_raw_wtime"])
        df_feat_long["enrich_over_raw_diff_rosum"] = diff
        df_feat_long["enrich_over_raw_norm_rosum"] = norm

    if "rate_over_raw_trim_rosum" in df_feat_long.columns and "e_random_raw_trim_wtime" in df_feat_long.columns:
        diff, norm = _enrich(df_feat_long["rate_over_raw_trim_rosum"], df_feat_long["e_random_raw_trim_wtime"])
        df_feat_long["enrich_over_raw_trim_diff_rosum"] = diff
        df_feat_long["enrich_over_raw_trim_norm_rosum"] = norm

    return df_feat_long


def pivot_feature_wide(df_feat_long: pd.DataFrame, *, id_col: str = "SID", feature_col: str = "feature") -> pd.DataFrame:
    """Pivot df_feat_long to wide with columns like {metric}_{feature}."""
    if df_feat_long.empty:
        return df_feat_long.copy()

    metric_cols = [c for c in df_feat_long.columns if c not in {id_col, feature_col}]
    df_w = df_feat_long.pivot(index=id_col, columns=feature_col, values=metric_cols)

    df_w.columns = [f"{m}_{f}" for m, f in df_w.columns.to_list()]
    df_w = df_w.reset_index()

    # Deterministic column order
    keep_first = [id_col]
    rest = sorted([c for c in df_w.columns if c not in keep_first])
    return df_w[keep_first + rest]


def drop_feature_support_counts(df_cluster: pd.DataFrame, *, features: Sequence[str]) -> pd.DataFrame:
    """
    Drop wide feature-suffixed "support" columns (counts/denominators) to reduce width for clustering,
    keeping higher-level rates/enrichments.
    """
    feats = sorted({str(f).lower() for f in features}, key=len, reverse=True)

    def metric_base(col: str) -> Optional[str]:
        cl = col.lower()
        for feat in feats:
            suf = f"_{feat}"
            if cl.endswith(suf):
                return cl[:-len(suf)]
        return None

    def is_drop(col: str) -> bool:
        base = metric_base(col)
        if base is None:
            return False
        return (
            base.startswith(("n_", "den_", "num_", "dwell_", "chance_"))
            or base.endswith(("_bins", "_seconds", "_num"))
            or base in {"x_feature_len", "x_feature_seconds", "raw_seconds_est", "valid_seconds_est"}
        )

    return df_cluster.drop(columns=[c for c in df_cluster.columns if is_drop(c)], errors="ignore")


# =============================
# Optional stability block
# =============================
def compute_feature_stability_block(
    df_long: pd.DataFrame,
    *,
    run_id_cols: Sequence[str],
    id_col: str = "SID",
    feature_col: str = "feature",
    weight_col: str = "n_y_valid",
    metrics: Sequence[str] = (),
) -> pd.DataFrame:
    """
    Run-to-run variability per (subject, feature) for selected per-run metrics.
    Produces wide columns with suffixes:
      - _run_std: unweighted std across runs (ddof=1)
      - _run_iqr: interquartile range across runs
      - _run_wstd: weighted (population) std with weights = weight_col
      - n_runs_feat: number of runs contributing per feature
    """
    metrics = [m for m in metrics if m in df_long.columns]
    if not metrics:
        return pd.DataFrame(index=pd.Index(df_long[id_col].astype(str).unique(), name=id_col))

    key_cols = list(run_id_cols) + [feature_col]
    keep_cols = metrics + ([weight_col] if weight_col in df_long.columns else [])
    df_rf = df_long[key_cols + keep_cols].groupby(key_cols, dropna=False, as_index=False).first()

    gcols = [id_col, feature_col]
    gb = df_rf.groupby(gcols, dropna=False)

    cnt = gb[metrics].count()
    std = gb[metrics].std(ddof=1)
    iqr = gb[metrics].quantile(0.75) - gb[metrics].quantile(0.25)

    if weight_col in df_rf.columns:
        wstd = _weighted_std_by_group(df_rf, gcols, metrics, weight_col).set_index(gcols)
    else:
        wstd = std * np.nan

    std = std.where(cnt >= 2)
    iqr = iqr.where(cnt >= 2)
    wstd = wstd.where(cnt >= 2)

    n_runs_feat = gb.size().rename("n_runs_feat")
    out = (
        n_runs_feat.to_frame()
        .join(std.add_suffix("_run_std"))
        .join(iqr.add_suffix("_run_iqr"))
        .join(wstd.add_suffix("_run_wstd"))
        .reset_index()
    )

    value_cols = [c for c in out.columns if c not in gcols]
    wide = out.pivot(index=id_col, columns=feature_col, values=value_cols)
    wide.columns = [f"{metric}_{feat}" for (metric, feat) in wide.columns]
    wide.index.name = id_col
    return wide


# =============================
# Main entrypoint
# =============================
def build_df_cluster(cfg: BuildConfig) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build the full set of outputs:
      df_cluster, df_feat_wide, df_runlevel, df_runs, df_feat_long

    Raises
    ------
    FileNotFoundError
        If input CSVs do not exist.
    ValueError
        If required columns are missing, filtering removes all rows, or IDs/groups are inconsistent.
    """
    df = load_and_filter(cfg)

    run_id_cols = infer_run_id_cols(df)
    df = dedup_run_feature_rows(df, run_id_cols, feature_col="feature")
    log.info("---> df_mergedmain columns: %s", ", ".join(df.columns.to_list()))
    log.info("df_mergedmain shape: %s", df.shape)
    
    run_level_cols, feat_level_cols = split_runlevel_vs_featurelevel_cols(df, run_id_cols, feature_col="feature")
    feat_level_num = [c for c in feat_level_cols if c in df.columns and _is_numeric_or_bool(df[c])]

    # Run-level tables
    df_runs = build_df_runs(df, run_id_cols=run_id_cols, run_level_cols=run_level_cols)
    df_runlevel = build_df_runlevel(df_runs, run_id_cols=run_id_cols, cfg=cfg)

    # Feature-level tables
    df_feat_long = build_df_feat_long(df, cfg=cfg, run_id_cols=run_id_cols, feature_level_num_cols=feat_level_num)
    df_feat_wide = pivot_feature_wide(df_feat_long)
    log.info("---> df_feat_wide columns: %s", ", ".join(df_feat_wide.columns.to_list()))
    
    df_cluster = df_runlevel.merge(df_feat_wide, on="SID", how="left", validate="one_to_one")
    log.info("---> df_cluster columns: %s", ", ".join(df_cluster.columns.to_list()))
    
    # Feature-specific missingness differential vs overall missingness (raw scope)
    if "frac_y_missing_overall" in df_cluster.columns:
        if cfg.features_keep is not None:
            features_for_loops = [str(f).lower() for f in cfg.features_keep]
        else:
            features_for_loops = sorted({str(f).lower() for f in df["feature"].dropna().unique()})

        for feat in features_for_loops:
            col = f"frac_y_missing_given_present_any_raw_rosum_{feat}"
            if col in df_cluster.columns:
                df_cluster[f"missingness_differential_rosum_{feat}"] = df_cluster[col] - df_cluster["frac_y_missing_overall"]
            col2 = f"frac_y_missing_given_present_any_raw_meanrun_{feat}"
            if col2 in df_cluster.columns:
                df_cluster[f"missingness_differential_from_overall_meanrun_{feat}"] = df_cluster[col2] - df_cluster["frac_y_missing_overall"]

    # Optional stability block
    if cfg.add_stability_block:
        wcol = cfg.stability_weight_col if cfg.stability_weight_col in df.columns else ("n_y_valid" if "n_y_valid" in df.columns else "n_raw")
        stab = compute_feature_stability_block(
            df_long=df,
            run_id_cols=run_id_cols,
            id_col="SID",
            feature_col="feature",
            weight_col=wcol,
            metrics=cfg.stability_metrics,
        )
        if not stab.empty:
            df_cluster = df_cluster.merge(stab.reset_index(), on="SID", how="left", validate="one_to_one")

    if not cfg.keep_support_counts and cfg.features_keep:
        df_cluster = drop_feature_support_counts(df_cluster, features=cfg.features_keep)

    # # Stabilize schema ordering for reproducible downstream reads (CSV column order can
    # # otherwise drift across pandas versions/merge orders).
    # if "SID" in df_cluster.columns:
    #     cols = ["SID"] + sorted([c for c in df_cluster.columns if c != "SID"])
    #     df_cluster = df_cluster.reindex(columns=cols)

    # Save outputs
    if cfg.out_dir is not None:
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.out_dir / "df_mergedmain.csv", index=False)
        df_cluster.to_csv(cfg.out_dir / "df_cluster.csv", index=False)
        df_runs.to_csv(cfg.out_dir / "df_runs.csv", index=False)
        df_runlevel.to_csv(cfg.out_dir / "df_runlevel.csv", index=False)
        df_feat_long.to_csv(cfg.out_dir / "df_feat_long.csv", index=False)
        df_feat_wide.to_csv(cfg.out_dir / "df_feat_wide.csv", index=False)

        if cfg.write_spearman_corr:
            num_cols = [c for c in df_cluster.columns if pd.api.types.is_numeric_dtype(df_cluster[c])]
            df_cluster[num_cols].corr(method="spearman").to_csv(cfg.out_dir / "corr_spearman.csv")

    return df_cluster, df_feat_wide, df_runlevel, df_runs, df_feat_long

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    MAIN_DATA_DIR = Path("/home/umit/Documents/Research_ET/WebGazer/output")
    BASE_DIR = MAIN_DATA_DIR / "dwell_times_batch_feats" / "naive_bdry"

    cfg = BuildConfig(
        pro_csv=BASE_DIR / "task5_pro_dwell_times_binsz-0.5s_bdry-posthoc-t0.05-p1_jointfeat.csv",
        spark_csv=BASE_DIR / "task5_spark_dwell_times_binsz-0.5s_bdry-posthoc-t0.05-p1_jointfeat.csv",
        out_dir=MAIN_DATA_DIR / "cluster_dataprep_feats",
        force_group_by_cohort={"SPARK": "SPARK"},
        add_stability_block=True,
        keep_support_counts=True,
        default_bin_seconds=0.5,
        write_spearman_corr=True,
    )

    df_cluster, df_feat_wide, *_ = build_df_cluster(cfg)
    log.info("df_cluster shape: %s", df_cluster.shape)
    
    
    

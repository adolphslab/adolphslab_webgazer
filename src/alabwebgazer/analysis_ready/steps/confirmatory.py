"""Confirmatory models step (SAP implemented).

Implements binomial-logit models on successes/trials (denominator-aware), with:
- task + use_run gating
- Video fixed effects
- Sex covariate (primary)
- planned contrasts
- BH-FDR across confirmatory tests

Requirement anchors:
- SAP-IMPL-ENDP-001, SAP-IMPL-CONT-001, SAP-IMPL-EST-001
- SAP-IMPL-VID-001, SAP-IMPL-COV-001
- SAP-IMPL-INF-001, SAP-IMPL-MULTI-001
- SAP-IMPL-TRIAL0-001
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..errors import DataValidationError
from ..logging import get_logger
from ..models.binomial import (
    FittedBinomial,
    fit_gee_exchangeable,
    fit_glm_cluster,
    simulate_contrast_delta_p,
    simulate_difference_in_differences_p,
)
from ..reporting.tables import bh_fdr
from ..validation import (
    apply_common_filters,
    drop_zero_trials,
    validate_binomial_counts,
)
from .base import RunContext, Step

log = get_logger(__name__)


def _set_categories(df: pd.DataFrame, *, groups: List[str], sexes: List[str], videos: List[str]) -> pd.DataFrame:
    d = df.copy()
    if "Group" in d.columns:
        d["Group"] = pd.Categorical(d["Group"], categories=groups)
    if "Sex" in d.columns:
        d["Sex"] = pd.Categorical(d["Sex"], categories=sexes)
    if "Video" in d.columns:
        d["Video"] = pd.Categorical(d["Video"], categories=videos)
    return d


def _term_map(ref_group: str, ref_sex: str, ref_video: str, ref_state: str) -> Dict[str, str]:
    return {
        "Group": f'C(Group, Treatment(reference="{ref_group}"))',
        "Sex": f'C(Sex, Treatment(reference="{ref_sex}"))',
        "Video": f'C(Video, Treatment(reference="{ref_video}"))',
        "State": f'C(State, Treatment(reference="{ref_state}"))',
    }


def _build_formula(predictors: List[str], *, ref_group: str, ref_sex: str, ref_video: str, ref_state: str) -> str:
    tm = _term_map(ref_group, ref_sex, ref_video, ref_state)

    def term(x: str) -> str:
        if ":" in x:
            a, b = x.split(":", 1)
            return f"{tm.get(a, a)}:{tm.get(b, b)}"
        return tm.get(x, x)

    rhs = " + ".join([term(p) for p in predictors])
    return "{y} ~ " + rhs


def _fit_backend(ctx: RunContext, df: pd.DataFrame, *, formula: str, successes_col: str, trials_col: str) -> FittedBinomial:
    backend = ctx.config.models.primary_backend
    if backend == "glm_cluster":
        return fit_glm_cluster(
            df,
            formula=formula,
            successes_col=successes_col,
            trials_col=trials_col,
            cluster_col=ctx.config.models.cluster_col,
        )
    return fit_gee_exchangeable(
        df,
        formula=formula,
        successes_col=successes_col,
        trials_col=trials_col,
        cluster_col=ctx.config.models.cluster_col,
    )


def _standardization_grid(df: pd.DataFrame, *, groups: List[str], sexes: List[str], videos: List[str]) -> pd.DataFrame:
    # Equal-weight standardization across design videos and sex (SAP-IMPL-VID-001, SAP-IMPL-COV-001)
    grid = pd.DataFrame(
        [(g, s, v) for g in groups for s in sexes for v in videos],
        columns=["Group", "Sex", "Video"],
    )
    return _set_categories(grid, groups=groups, sexes=sexes, videos=videos)


class ConfirmatoryModelsStep(Step):
    step_id = "confirmatory_models"
    requirement_ids = (
        "SAP-IMPL-ENDP-001",
        "SAP-IMPL-CONT-001",
        "SAP-IMPL-INF-001",
        "SAP-IMPL-MULTI-001",
    )

    def run(self, ctx: RunContext) -> None:
        cfg = ctx.config
        stage = "01_confirmatory"

        df_feat = ctx.tables["feature_long"]
        df_runs = ctx.tables["run_level"]

        # Common filters with audit counts
        df_feat_f, rep_feat = apply_common_filters(
            df_feat,
            name="feature_long",
            tasks_keep=cfg.filters.tasks_keep,
            use_run_only=cfg.filters.use_run_only,
            groups_keep=cfg.filters.groups_keep,
            sex_keep=cfg.filters.sex_keep,
        )
        df_runs_f, rep_runs = apply_common_filters(
            df_runs,
            name="run_level",
            tasks_keep=cfg.filters.tasks_keep,
            use_run_only=cfg.filters.use_run_only,
            groups_keep=cfg.filters.groups_keep,
            sex_keep=cfg.filters.sex_keep,
        )

        ctx.artifacts.write_json(stage, "filter_report_feature_long.json", asdict(rep_feat))
        ctx.artifacts.write_json(stage, "filter_report_run_level.json", asdict(rep_runs))

        if df_feat_f.empty:
            raise DataValidationError(
                "feature_long: no rows remain after common filters. "
                f"tasks_keep={cfg.filters.tasks_keep}, "
                f"groups_keep={cfg.filters.groups_keep}, "
                f"sex_keep={cfg.filters.sex_keep}, "
                f"use_run_only={cfg.filters.use_run_only}"
            )
        if df_runs_f.empty:
            raise DataValidationError(
                "run_level: no rows remain after common filters. "
                f"tasks_keep={cfg.filters.tasks_keep}, "
                f"groups_keep={cfg.filters.groups_keep}, "
                f"sex_keep={cfg.filters.sex_keep}, "
                f"use_run_only={cfg.filters.use_run_only}"
            )

        # Enforce categories for consistent baselines
        df_feat_f = _set_categories(
            df_feat_f,
            groups=cfg.filters.groups_keep,
            sexes=cfg.filters.sex_keep,
            videos=cfg.filters.design_videos,
        )
        df_runs_f = _set_categories(
            df_runs_f,
            groups=cfg.filters.groups_keep,
            sexes=cfg.filters.sex_keep,
            videos=cfg.filters.design_videos,
        )

        results_rows: List[Dict[str, object]] = []
        pvals: List[float] = []
        keys_for_q = []

        for ep_id, ep in cfg.confirmatory.endpoints.items():
            log.info("Running endpoint: %s", ep_id)
            if ep.table == "feature_long":
                d = df_feat_f.copy()
                if ep.feature_filter is not None:
                    d = d.loc[d["feature"].isin(ep.feature_filter)].copy()
                if d.empty:
                    raise DataValidationError(
                        f"{ep_id}: no feature_long rows remain after "
                        f"feature_filter={ep.feature_filter}"
                    )

                # Special handling for aversion interaction: derive State from feature
                if ep.state_col:
                    if ep.state_col not in d.columns:
                        # derive from 'feature'
                        d[ep.state_col] = np.where(
                            d["feature"].astype(str).str.contains("not_averted"),
                            "speaker_not_averted",
                            np.where(
                                d["feature"].astype(str).str.contains("averted"),
                                "speaker_averted",
                                np.nan,
                            ),
                        )
                    d = d.loc[d[ep.state_col].notna()].copy()
                    d[ep.state_col] = pd.Categorical(
                        d[ep.state_col], categories=["speaker_not_averted", "speaker_averted"]
                    )

                group_keys = ["Group", "Video", "task"]
                if ep.feature_filter is not None:
                    group_keys.append("feature")
                d, zero_tbl = drop_zero_trials(
                    d,
                    trials_col=ep.trials_col,
                    group_keys=group_keys,
                    name=ep_id,
                )
                ctx.artifacts.maybe_write_csv(stage, f"{ep_id}_drop_zero_trials.csv", zero_tbl)

                validate_binomial_counts(
                    d,
                    successes_col=ep.successes_col,
                    trials_col=ep.trials_col,
                    name=ep_id,
                )

            else:
                d = df_runs_f.copy()
                group_keys = ["Group", "Video", "task"]
                d, zero_tbl = drop_zero_trials(
                    d,
                    trials_col=ep.trials_col,
                    group_keys=group_keys,
                    name=ep_id,
                )
                ctx.artifacts.maybe_write_csv(stage, f"{ep_id}_drop_zero_trials.csv", zero_tbl)
                validate_binomial_counts(
                    d,
                    successes_col=ep.successes_col,
                    trials_col=ep.trials_col,
                    name=ep_id,
                )

            if d.empty:
                raise DataValidationError(
                    f"{ep_id}: no rows remain after filtering and zero-trial exclusion."
                )

            # Build formula
            formula = _build_formula(
                ep.predictors,
                ref_group="Control",
                ref_sex="Male",
                ref_video=cfg.filters.design_videos[0] if cfg.filters.design_videos else "S1",
                ref_state="speaker_not_averted",
            )

            fitted = _fit_backend(
                ctx,
                d,
                formula=formula,
                successes_col=ep.successes_col,
                trials_col=ep.trials_col,
            )

            # Planned contrasts
            for a, b in cfg.confirmatory.contrasts:
                if ep_id == "H1a_averted_interaction":
                    # Difference-in-differences on probability scale:
                    # (p_not - p_averted)_A - (p_not - p_averted)_B
                    grid_base = _standardization_grid(
                        d,
                        groups=[a, b],
                        sexes=cfg.filters.sex_keep,
                        videos=cfg.filters.design_videos,
                    )
                    # Expand with state
                    g_not = grid_base.copy()
                    g_not["State"] = "speaker_not_averted"
                    g_av = grid_base.copy()
                    g_av["State"] = "speaker_averted"

                    # For group a
                    ga_not = g_not.loc[g_not["Group"] == a].copy()
                    ga_av = g_av.loc[g_av["Group"] == a].copy()
                    # For group b
                    gb_not = g_not.loc[g_not["Group"] == b].copy()
                    gb_av = g_av.loc[g_av["Group"] == b].copy()

                    delta_hat, (ci_low, ci_high) = simulate_difference_in_differences_p(
                        fitted,
                        group_a_state_ref=ga_not,
                        group_a_state_alt=ga_av,
                        group_b_state_ref=gb_not,
                        group_b_state_alt=gb_av,
                        n_sims=2000,
                        seed=cfg.run.random_seed,
                    )

                    # p-value for interaction term
                    term_a = f'C(Group, Treatment(reference="Control"))[T.{a}]:C(State, Treatment(reference="speaker_not_averted"))[T.speaker_averted]'
                    term_b = f'C(Group, Treatment(reference="Control"))[T.{b}]:C(State, Treatment(reference="speaker_not_averted"))[T.speaker_averted]'
                    # For baseline group, interaction term doesn't exist; handle accordingly
                    p_val = np.nan
                    try:
                        if a == "Control":
                            p_val = float(fitted.result.pvalues.get(term_b, np.nan))
                        elif b == "Control":
                            p_val = float(fitted.result.pvalues.get(term_a, np.nan))
                        else:
                            tt = fitted.result.t_test(f"{term_a} = {term_b}")
                            p_val = float(tt.pvalue)
                    except Exception:  # noqa: BLE001
                        p_val = np.nan

                else:
                    grid = _standardization_grid(
                        d,
                        groups=[a, b],
                        sexes=cfg.filters.sex_keep,
                        videos=cfg.filters.design_videos,
                    )
                    # Split grid by group for Δp
                    ga = grid.loc[grid["Group"] == a].copy()
                    gb = grid.loc[grid["Group"] == b].copy()

                    delta_hat, (ci_low, ci_high) = simulate_contrast_delta_p(
                        fitted,
                        new_df_a=ga,
                        new_df_b=gb,
                        n_sims=2000,
                        seed=cfg.run.random_seed,
                    )

                    # p-value for log-odds contrast (Wald)
                    p_val = np.nan
                    try:
                        if b == "Control" and a != "Control":
                            term = f'C(Group, Treatment(reference="Control"))[T.{a}]'
                            p_val = float(fitted.result.pvalues.get(term, np.nan))
                        elif a == "Control" and b != "Control":
                            term = f'C(Group, Treatment(reference="Control"))[T.{b}]'
                            p_val = float(fitted.result.pvalues.get(term, np.nan))
                        elif a != "Control" and b != "Control":
                            t1 = f'C(Group, Treatment(reference="Control"))[T.{a}]'
                            t2 = f'C(Group, Treatment(reference="Control"))[T.{b}]'
                            tt = fitted.result.t_test(f"{t1} = {t2}")
                            p_val = float(tt.pvalue)
                    except Exception:  # noqa: BLE001
                        p_val = np.nan

                row = {
                    "endpoint": ep_id,
                    "contrast": f"{a} vs {b}",
                    "group_a": a,
                    "group_b": b,
                    "backend": fitted.backend,
                    "delta_p": delta_hat,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "p_value": p_val,
                    "n_rows": int(len(d)),
                    "n_subjects": int(d["SID"].nunique()) if "SID" in d.columns else np.nan,
                }
                results_rows.append(row)
                pvals.append(p_val)
                keys_for_q.append((ep_id, a, b))

        res_df = pd.DataFrame(results_rows)

        # Multiplicity adjustment across confirmatory tests
        res_df["q_value"] = bh_fdr(res_df["p_value"].tolist())

        ctx.artifacts.write_csv(stage, "confirmatory_results.csv", res_df, index=False)
        log.info("Confirmatory models complete: %d rows", len(res_df))

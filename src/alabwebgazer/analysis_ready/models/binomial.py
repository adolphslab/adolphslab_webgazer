"""Binomial modeling backends for confirmatory analyses.

Primary confirmatory backend: GLM Binomial with cluster-robust SEs by participant (SID).
Secondary backend: GEE Binomial with exchangeable working correlation.

Implementation note (engineering):
- statsmodels emits a warning when using cluster-robust covariances with freq_weights in GLM.
  To avoid ambiguous behavior, the GLM backend here fits binomial data using a 2-column endog
  (successes, failures) without weights, while still using patsy to build the design matrix.

Requirement anchors:
- SAP-IMPL-INF-001 (glm_cluster, gee_exchangeable)
- SAP-IMPL-EST-001 (successes/trials)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm


@dataclass(frozen=True)
class FittedBinomial:
    backend: Literal["glm_cluster", "gee_exchangeable"]
    formula: str
    successes_col: str
    trials_col: str
    cluster_col: str
    design_info: patsy.DesignInfo
    result: object  # statsmodels result type


def _rhs(formula: str) -> str:
    parts = formula.split("~", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid formula (missing '~'): {formula}")
    return parts[1].strip()


def fit_glm_cluster(
    df: pd.DataFrame,
    *,
    formula: str,
    successes_col: str,
    trials_col: str,
    cluster_col: str,
) -> FittedBinomial:
    """Fit Binomial GLM with cluster-robust SEs using count endog (successes, failures)."""
    d = df.copy()

    rhs = _rhs(formula.replace("{y}", "y"))
    X = patsy.dmatrix(rhs, data=d, return_type="dataframe")

    successes = np.asarray(d[successes_col], dtype=float)
    trials = np.asarray(d[trials_col], dtype=float)
    failures = trials - successes
    endog = np.column_stack([successes, failures])

    model = sm.GLM(endog=endog, exog=X, family=sm.families.Binomial())
    res = model.fit(cov_type="cluster", cov_kwds={"groups": d[cluster_col]})

    return FittedBinomial(
        backend="glm_cluster",
        formula=formula,
        successes_col=successes_col,
        trials_col=trials_col,
        cluster_col=cluster_col,
        design_info=X.design_info,
        result=res,
    )


def fit_gee_exchangeable(
    df: pd.DataFrame,
    *,
    formula: str,
    successes_col: str,
    trials_col: str,
    cluster_col: str,
) -> FittedBinomial:
    """Fit Binomial GEE with exchangeable working correlation.

    Uses proportion endog with weights=trials. (Alternative backend; see notes in docs.)
    """
    d = df.copy()
    d["_y_prop"] = d[successes_col] / d[trials_col]

    ind = sm.cov_struct.Exchangeable()
    model = sm.GEE.from_formula(
        formula=formula.replace("{y}", "_y_prop"),
        groups=d[cluster_col],
        data=d,
        family=sm.families.Binomial(),
        cov_struct=ind,
        weights=d[trials_col],
    )
    res = model.fit()

    return FittedBinomial(
        backend="gee_exchangeable",
        formula=formula,
        successes_col=successes_col,
        trials_col=trials_col,
        cluster_col=cluster_col,
        design_info=res.model.data.design_info,
        result=res,
    )


def _design_matrix(model: FittedBinomial, new_df: pd.DataFrame) -> np.ndarray:
    X = patsy.build_design_matrices([model.design_info], new_df, return_type="dataframe")[0]
    return np.asarray(X)


def simulate_contrast_delta_p(
    model: FittedBinomial,
    *,
    new_df_a: pd.DataFrame,
    new_df_b: pd.DataFrame,
    n_sims: int,
    seed: int,
) -> Tuple[float, Tuple[float, float]]:
    """Parametric simulation for Δp = p(A) - p(B).

    Uses asymptotic Normal(params, cov) draws. For cluster-robust, uses the robust cov.

    Returns:
      (point_estimate, (ci_low, ci_high)) where CI is 2.5%/97.5% percentiles.
    """
    res = model.result
    params_hat = np.asarray(res.params)
    cov = np.asarray(res.cov_params())

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=params_hat, cov=cov, size=n_sims)  # (n_sims, n_params)

    Xa = _design_matrix(model, new_df_a)  # (n_a, n_params)
    Xb = _design_matrix(model, new_df_b)  # (n_b, n_params)

    # inverse link
    inv_link = sm.families.Binomial().link.inverse

    # sims
    eta_a = Xa @ draws.T  # (n_a, n_sims)
    eta_b = Xb @ draws.T  # (n_b, n_sims)
    p_a = inv_link(eta_a)
    p_b = inv_link(eta_b)
    delta = p_a.mean(axis=0) - p_b.mean(axis=0)

    # point estimate
    p_a_hat = inv_link(Xa @ params_hat).mean()
    p_b_hat = inv_link(Xb @ params_hat).mean()
    delta_hat = float(p_a_hat - p_b_hat)

    ci = (float(np.quantile(delta, 0.025)), float(np.quantile(delta, 0.975)))
    return delta_hat, ci


def simulate_difference_in_differences_p(
    model: FittedBinomial,
    *,
    group_a_state_ref: pd.DataFrame,
    group_a_state_alt: pd.DataFrame,
    group_b_state_ref: pd.DataFrame,
    group_b_state_alt: pd.DataFrame,
    n_sims: int,
    seed: int,
) -> Tuple[float, Tuple[float, float]]:
    """Parametric simulation for ΔΔp:
    (p_ref - p_alt)_A - (p_ref - p_alt)_B.
    """
    res = model.result
    params_hat = np.asarray(res.params)
    cov = np.asarray(res.cov_params())

    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(mean=params_hat, cov=cov, size=n_sims)  # (n_sims, n_params)

    xa_ref = _design_matrix(model, group_a_state_ref)
    xa_alt = _design_matrix(model, group_a_state_alt)
    xb_ref = _design_matrix(model, group_b_state_ref)
    xb_alt = _design_matrix(model, group_b_state_alt)

    inv_link = sm.families.Binomial().link.inverse

    # Simulated standardized probabilities
    pa_ref = inv_link(xa_ref @ draws.T).mean(axis=0)
    pa_alt = inv_link(xa_alt @ draws.T).mean(axis=0)
    pb_ref = inv_link(xb_ref @ draws.T).mean(axis=0)
    pb_alt = inv_link(xb_alt @ draws.T).mean(axis=0)

    did = (pa_ref - pa_alt) - (pb_ref - pb_alt)

    # Point estimate
    pa_ref_hat = inv_link(xa_ref @ params_hat).mean()
    pa_alt_hat = inv_link(xa_alt @ params_hat).mean()
    pb_ref_hat = inv_link(xb_ref @ params_hat).mean()
    pb_alt_hat = inv_link(xb_alt @ params_hat).mean()
    did_hat = float((pa_ref_hat - pa_alt_hat) - (pb_ref_hat - pb_alt_hat))

    ci = (float(np.quantile(did, 0.025)), float(np.quantile(did, 0.975)))
    return did_hat, ci

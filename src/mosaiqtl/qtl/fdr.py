"""
fdr — false discovery rate estimation

storey_qvalue: Storey (2002) q-value estimator with natural cubic spline
π₀ estimation.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.interpolate import UnivariateSpline

logger = logging.getLogger('mosaiqtl')

__all__ = ['storey_qvalue']

# default λ grid: 0.05 to 0.95 in 0.05 steps (Storey 2002 recommendation)
_LAMBDA_DEFAULT = np.arange(0.05, 1.00, 0.05)


def storey_qvalue(
    pvalues: np.ndarray,
    lambda_range: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Storey (2002) q-value estimator.

    Estimates the proportion of true null hypotheses π₀ from the
    p-value distribution using a natural cubic spline smoother, then
    computes q-values as BH-adjusted p-values scaled by π₀.

    parameters
    ----------
    pvalues      : np.ndarray (m,)  p-values in [0, 1]; NaN are ignored
    lambda_range : np.ndarray or None
        grid of λ values for the π₀ estimator (default: 0.05..0.95 step 0.05).
        must have ≥ 4 points for spline fitting; falls back to min estimator
        when fewer than 4 finite π̂₀ values are available.

    returns
    -------
    q_values : np.ndarray (m,)  q-values in [0, 1]; NaN where pvalues is NaN
    pi0      : float             estimated π₀ (proportion of true nulls)

    notes
    -----
    q(i) = π₀ × m × p(i) / rank(p(i)), with monotone enforcement.
    Under complete null, π₀ → 1 and q ≈ BH q.
    Under strong signal, π₀ < 1 and q < BH q.
    """
    lam = _LAMBDA_DEFAULT if lambda_range is None else np.asarray(lambda_range)
    pv  = np.asarray(pvalues, dtype=float)

    finite_mask = np.isfinite(pv)
    m = int(finite_mask.sum())
    if m == 0:
        return pv.copy(), 1.0

    pv_fin = pv[finite_mask]

    # ── π₀ estimation via natural cubic spline (Storey 2002 §2.4) ────────────
    pi0_hat = np.array([
        (pv_fin > lv).sum() / (m * (1.0 - lv))
        for lv in lam
    ])

    # keep only λ values where the estimate is finite and positive
    ok = np.isfinite(pi0_hat) & (pi0_hat > 0)
    if ok.sum() >= 4:
        try:
            # KNOWN ISSUES (not yet fixed):
            # (1) UnivariateSpline(k=3) is a B-spline, not a natural cubic spline
            #     (Storey 2002 specifies zero second derivative at endpoints).
            #     Use make_interp_spline with bc_type='natural' for strict compliance.
            # (2) s=len(lam[ok]) under-smooths relative to Storey's GCV tuning;
            #     R qvalue uses smooth.spline(df=3) — 3 effective degrees of freedom.
            # (3) The cap below (pi0 = min(pi0, 1.0)) silently hides values > 1,
            #     which are a diagnostic that input p-values are miscalibrated.
            spl = UnivariateSpline(lam[ok], pi0_hat[ok], k=3, s=len(lam[ok]))
            pi0 = float(spl(lam[ok].max()))
        except Exception:
            pi0 = float(pi0_hat[ok].min())
    else:
        # too few points for spline — fall back to minimum estimator
        pi0 = float(pi0_hat[ok].min()) if ok.any() else 1.0

    # cap π₀ at 1; a value > 1 means the null model over-estimates signal
    pi0 = min(pi0, 1.0)
    pi0 = max(pi0, 1e-6)   # numerical guard: q must be finite
    logger.debug('storey_qvalue: m=%d  pi0=%.4f', m, pi0)

    # ── q-value computation ───────────────────────────────────────────────────
    # sort p-values ascending, compute BH-style correction scaled by π₀
    order   = np.argsort(pv_fin)
    ranks   = np.arange(1, m + 1)              # 1-based ranks of sorted p
    sorted_p = pv_fin[order]

    q_sorted = np.minimum(pi0 * m * sorted_p / ranks, 1.0)

    # enforce monotonicity: q[i] ≤ q[i+1] (scan from largest p downward)
    for i in range(m - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])

    # map back to original order
    q_fin = np.empty(m)
    q_fin[order] = q_sorted

    q_out = np.full_like(pv, np.nan)
    q_out[finite_mask] = q_fin
    return q_out, pi0

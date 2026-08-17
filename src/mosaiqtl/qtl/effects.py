"""
effects — allele haplotype effect estimation and locus characterization

two estimation methods for haplotype effects:

  fixef: γ̂ = G⁺ c   (pseudoinverse; identical to genome scan betas at that locus)
  ranef: γ̂ = (G + λI)⁻¹ c   (ridge/BLUP; λ = σ²_e / σ²_a from empirical Bayes)

both use the FWL-rotated gram matrix G = H_perp'H_perp and cross-term
c = H_perp'e₀ derived from the LOCO-LMM eigendecomp at the requested locus.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import stats

from ..models.records import FounderEffect
from ..utils import rint, build_z_matrix
from . import lmm as lmm_module

if TYPE_CHECKING:
    from ..genotype.genomecache import GenomeCache

logger = logging.getLogger("mosaiqtl")

__all__ = ["get_allele_effects", "qtl_gxe_lrt"]


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _locus_fwl(
    y: np.ndarray,
    x: np.ndarray,
    h: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    h2: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """FWL gram matrix and residual cross-term at one locus.

    returns (c, G, rss0) where:
      c    = H_perp' e₀  — (J,) cross-term
      G    = H_perp' H_perp — (J, J) weighted gram matrix
      rss0 = null model RSS
    """
    delta = h2 / max(1.0 - h2, 1e-10)
    w_sqrt = 1.0 / np.sqrt(delta * eigvals + 1.0)

    yt = (eigvecs.T @ y) * w_sqrt
    xt = (eigvecs.T @ x) * w_sqrt[:, None]
    ht = (eigvecs.T @ h) * w_sqrt[:, None]

    xtwx = xt.T @ xt
    try:
        beta_null = np.linalg.solve(xtwx, xt.T @ yt)
    except np.linalg.LinAlgError:
        J = h.shape[1]
        return np.zeros(J), np.zeros((J, J)), 0.0

    e0 = yt - xt @ beta_null
    rss0 = float(e0 @ e0)

    xt_pinv = np.linalg.solve(xtwx, xt.T)  # (p, n)
    h_perp = ht - xt @ (xt_pinv @ ht)  # (n, J)

    c = h_perp.T @ e0  # (J,)
    G = h_perp.T @ h_perp  # (J, J)
    return c, G, rss0


# ─────────────────────────────────────────────────────────────────────────────
# public API
# ─────────────────────────────────────────────────────────────────────────────


def get_allele_effects(
    phenotype: np.ndarray,
    genomecache: GenomeCache,
    locus_id: str,
    chrom: str,
    kinship_decomp: dict,
    covariates: np.ndarray | None = None,
    sample_ids: list[str] | None = None,
    use_rint: bool = True,
    method: str = "fixef",
    h2: float | None = None,
    founder_names: list[str] | None = None,
) -> list[FounderEffect]:
    """allele haplotype effects at a single locus.

    parameters
    ----------
    phenotype      : np.ndarray (n,)
    genomecache   : GenomeCache
    locus_id       : str    marker identifier
    chrom          : str    chromosome of the locus
    kinship_decomp : dict   {chrom: {'values':, 'vectors':}}
    covariates     : np.ndarray (n, p) or None
    sample_ids     : list of str or None
        replicate sample IDs; when provided phenotype/covariates are
        reduced to strain means via Z before the refit.
    use_rint       : bool   apply RINT transform
    method         : str    'fixef' (default) or 'ranef'
        fixef: γ̂ = G⁺ c  (pseudoinverse; matches scan betas exactly)
        ranef: γ̂ = (G + λI)⁻¹ c  (ridge/BLUP; empirical-Bayes λ)
    h2             : float or None   pre-estimated null h²; re-estimated if None
    founder_names  : list of str or None   defaults to genomecache.founders

    returns
    -------
    list[FounderEffect]  one per founder in FOUNDER_ORDER
    """
    if founder_names is None:
        founder_names = list(genomecache.founders)

    logger.info(
        "get_allele_effects: gene/locus=%s  chr=%s  method=%s  n_founders=%d",
        locus_id,
        chrom,
        method,
        len(founder_names),
    )

    # ── resolve Z and validity ────────────────────────────────────────────────
    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    cache_ids = list(genomecache.sample_ids)
    if sid == cache_ids:
        z_mat, valid_sid = None, np.ones(len(sid), dtype=bool)
    else:
        z_mat, valid_sid = build_z_matrix(sid, cache_ids)

    valid = (~np.isnan(phenotype)) & valid_sid
    y = phenotype[valid].copy()
    if use_rint:
        y = rint(y)

    z_sub = z_mat[valid] if z_mat is not None else None

    if covariates is not None:
        x = covariates[valid].copy()
    else:
        x = np.ones((len(y), 1), dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((len(y), 1)), x])

    # replicate-aware: strain-mean transform
    if z_sub is not None:
        r_counts = z_sub.sum(axis=0)
        y = (z_sub.T @ y) / r_counts
        x = (z_sub.T @ x) / r_counts[:, None]

    # ── kinship eigendecomp (already at strain level) ─────────────────────────
    ev = kinship_decomp[chrom]["values"]
    evc = kinship_decomp[chrom]["vectors"]

    # ── estimate h² if not provided ──────────────────────────────────────────
    if h2 is None:
        h2, _ = lmm_module.estimate_h2_null(evc.T @ y, evc.T @ x, ev)

    # ── haplotype probs at locus (strain-level, J columns) ───────────────────
    lids, _, probs_chr = genomecache.get_chromosome(chrom)
    matches = np.where(lids == locus_id)[0]
    if len(matches) == 0:
        raise ValueError(f"locus {locus_id!r} not found on chr {chrom!r}")
    h = probs_chr[int(matches[0])]  # (m, J) — strain level

    # ── FWL gram matrix and cross-term ────────────────────────────────────────
    c, G, rss0 = _locus_fwl(y, x, h, ev, evc, h2)
    n = len(y)
    p = x.shape[1]
    J = h.shape[1]
    q = int(np.linalg.matrix_rank(G, tol=1e-10))

    if q < J - 1:
        logger.warning(
            "get_allele_effects: G rank=%d < J-1=%d at locus %s — "
            "rcond clamps applied via pinv; one or more founders have near-zero representation",
            q,
            J - 1,
            locus_id,
        )

    # ── solve for betas and compute covariance ────────────────────────────────
    if method == "fixef":
        G_pinv = np.linalg.pinv(G, rcond=1e-10)
        betas = G_pinv @ c
        rss1 = max(rss0 - float(c @ betas), 1e-30)
        sigma2 = rss1 / max(n - p - q, 1)
        cov_b = sigma2 * G_pinv
    elif method == "ranef":
        # ridge with λ = σ²_e / σ²_a (empirical-Bayes signal-to-noise)
        G_pinv = np.linalg.pinv(G, rcond=1e-10)
        b_fix = G_pinv @ c
        rss1 = max(rss0 - float(c @ b_fix), 1e-30)
        sigma2_e = rss1 / max(n - p - q, 1)
        signal = max(rss0 - rss1, 0.0)
        # allele-effect variance: empirical Bayes estimate
        sigma2_a = max(signal / max(q, 1) / n, sigma2_e * 1e-4)
        lam = sigma2_e / sigma2_a
        G_ridge = G + lam * np.eye(J)
        try:
            G_r_inv = np.linalg.inv(G_ridge)
        except np.linalg.LinAlgError:
            G_r_inv = np.linalg.pinv(G_ridge)
        betas = G_r_inv @ c
        # ridge covariance: σ²_e (G+λI)⁻¹ G (G+λI)⁻¹
        cov_b = sigma2_e * G_r_inv @ G @ G_r_inv
    else:
        raise ValueError(f"method must be 'fixef' or 'ranef', got {method!r}")

    se = np.sqrt(np.maximum(np.diag(cov_b), 0.0))
    df = max(n - p - q, 1)

    effects = []
    for fi, fname in enumerate(founder_names[:J]):
        b = float(betas[fi])
        s = max(float(se[fi]), 1e-15)
        z = b / s
        pv = float(2.0 * stats.t.sf(abs(z), df=df))
        effects.append(
            FounderEffect(
                founder=fname,
                estimate=b,
                se=s,
                z_score=z,
                p_value=pv,
                ci_lo=b - 1.96 * s,
                ci_hi=b + 1.96 * s,
            )
        )
    return effects


# ─────────────────────────────────────────────────────────────────────────────
# locus characterization
# ─────────────────────────────────────────────────────────────────────────────


def qtl_gxe_lrt(
    y: np.ndarray,
    h_locus: np.ndarray,
    sex: np.ndarray,
    covariates: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    lod_additive: float,
    h2: float | None = None,
) -> dict[str, float]:
    """Single-locus LMM likelihood-ratio test for haplotype × sex interaction.

    Computes LOD_gxe = LOD_full − LOD_additive at the peak locus, where
    LOD_full adds J interaction columns [H ⊙ sex] to the test matrix.
    Equivalent to miQTL ``scan.h2lmm(intcovar=sex)`` evaluated at one locus
    instead of genome-wide — same statistical model, lower compute.

    The 1-way haplotype effect (``chi2_haplotype``, ``p_haplotype``) is
    derived at zero cost from the existing scan LOD via the LRT identity:
    ``chi2 = 2 ln(10) × LOD``, ``p ~ χ²(J−1)``.

    parameters
    ----------
    y            : np.ndarray (n,)    strain-level phenotype
    h_locus      : np.ndarray (n, J)  haplotype probabilities at peak locus
    sex          : np.ndarray (n,)    binary sex covariate (0 = male, 1 = female)
    covariates   : np.ndarray (n, p)  covariate matrix including intercept
    eigvals      : np.ndarray (n,)    LOCO kinship eigenvalues (peak chromosome)
    eigvecs      : np.ndarray (n, n)  LOCO kinship eigenvectors
    lod_additive : float              peak LOD from the additive scan
    h2           : float or None      null h²; re-estimated if None

    returns
    -------
    dict with keys:
        chi2_haplotype  — LRT χ²(J−1) for 1-way haplotype effect (from scan LOD)
        p_haplotype     — corresponding p-value
        lod_gxe_sex     — LOD_full − LOD_additive at peak locus
        p_gxe_sex       — p-value for interaction, χ²(J−1)
    """
    from scipy.stats import chi2 as chi2_dist

    n = len(y)
    J = h_locus.shape[1]
    # 1-way haplotype test: derived from scan LOD at zero cost
    chi2_hap = 2.0 * np.log(10) * max(lod_additive, 0.0)
    p_hap = float(1.0 - chi2_dist.cdf(chi2_hap, df=J - 1))

    # estimate h² from null model if not provided
    if h2 is None:
        x_rot = eigvecs.T @ covariates
        y_rot = eigvecs.T @ y
        h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, eigvals)

    # build interaction matrix: [H, H ⊙ sex] — adds J interaction columns
    sex_c = sex.astype(float)
    h_full = np.column_stack([h_locus, h_locus * sex_c[:, None]])

    try:
        c, G, rss0 = _locus_fwl(y, covariates, h_full, eigvals, eigvecs, h2)
        rss1 = rss0 - float(c @ np.linalg.pinv(G, rcond=1e-10) @ c)
        rss1 = max(rss1, rss0 * 1e-12)
        lod_full = float(n) / 2.0 * np.log10(rss0 / rss1)
    except (np.linalg.LinAlgError, ValueError):
        return {
            "chi2_haplotype": chi2_hap,
            "p_haplotype": p_hap,
            "lod_gxe_sex": np.nan,
            "p_gxe_sex": np.nan,
        }

    lod_gxe = lod_full - lod_additive
    chi2_gxe = 2.0 * np.log(10) * max(lod_gxe, 0.0)
    p_gxe = float(1.0 - chi2_dist.cdf(chi2_gxe, df=J - 1))

    return {
        "chi2_haplotype": float(chi2_hap),
        "p_haplotype": float(p_hap),
        "lod_gxe_sex": float(lod_gxe),
        "p_gxe_sex": float(p_gxe),
    }

"""
mosaiqtl.mediation — genome-wide mediation scan

tests whether a trans-eQTL effect on outcome gene y is mediated through
an intermediate (cis) gene m. the qr decomposition is precomputed once
per mediator to make genome-wide scanning efficient.

equivalent to miqtl:
  mediation.scan.qr()       →  mediation_scan_qr()
  fixef.mediation.scan()    →  fixef_mediation_scan()
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats as sp_stats

logger = logging.getLogger('mosaiqtl')


def _wls_beta(x, y, w_inv):
    """weighted least squares: β = (XᵀWX)⁻¹ XᵀWy."""
    xw = x * w_inv[:, np.newaxis]
    return np.linalg.lstsq(xw.T @ x, xw.T @ y, rcond=None)[0]


def mediation_scan_qr(outcome, mediator, genome_cache,
                      covariates=None, kinship_decomp=None,
                      use_rint=True, use_reml=True,
                      outcome_id='outcome', mediator_id='mediator'):
    """genome-wide mediation scan via qr decomposition.

    tests: does the haplotype effect on outcome persist after conditioning
    on mediator? if the haplotype effect shrinks to zero when mediator is
    included, the effect is fully mediated.

    model:
      null:  y ~ mediator + X + u     (no haplotype)
      alt:   y ~ H + mediator + X + u (haplotype added)
      LOD = n/2 * log10(RSS_null / RSS_alt)

    the qr decomposition of [mediator | X] is computed once and reused
    across all loci, matching miQTL's efficient implementation.

    equivalent to miqtl::mediation.scan.qr().

    parameters
    ----------
    outcome : np.ndarray (n,)   y — the trans target gene
    mediator : np.ndarray (n,)  m — the cis mediator gene expression
    genome_cache : GenomeCache
    covariates : np.ndarray (n, p) or None
    kinship_decomp : dict or None
    use_rint : bool   apply rint to outcome
    use_reml : bool
    outcome_id : str
    mediator_id : str

    returns
    -------
    pd.DataFrame  columns: locus, chr, pos_mb, lod, p_value, mediation_lod
      mediation_lod = lod_without_mediator - lod_with_mediator
                    (positive = mediation detected)
    """
    from . import kinship as kin_module
    from . import lmm as lmm_module
    from .utils import rint as _rint

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    # align samples
    valid  = (~np.isnan(outcome)) & (~np.isnan(mediator))
    y      = outcome[valid].copy()
    m      = mediator[valid].copy()
    if use_rint:
        y = _rint(y)
    n      = len(y)

    x_base = (covariates[valid].copy() if covariates is not None
              else np.ones((n, 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((n, 1)), x_base])

    # build mediation covariate: [mediator | X]
    m_std  = (m - m.mean()) / max(m.std(), 1e-10)
    x_med  = np.hstack([m_std.reshape(-1, 1), x_base])   # (n, p+1)

    rows = []

    for chrom in genome_cache.chromosomes:
        if chrom not in kinship_decomp:
            continue
        eigvals = kinship_decomp[chrom]['values']
        eigvecs = kinship_decomp[chrom]['vectors']
        ev_v    = eigvecs[valid]

        y_rot   = ev_v.T @ y
        xb_rot  = ev_v.T @ x_base
        xm_rot  = ev_v.T @ x_med

        h2_base, _ = lmm_module.estimate_h2_null(y_rot, xb_rot, eigvals)
        h2_med,  _ = lmm_module.estimate_h2_null(y_rot, xm_rot, eigvals)

        lids, pos, probs = genome_cache.get_chromosome(chrom)
        nf1 = genome_cache.n_founders - 1

        for i_l, lid in enumerate(lids):
            h_l    = probs[i_l][valid, :nf1]
            h_rot  = ev_v.T @ h_l

            # scan without mediator (base model)
            res_base = lmm_module.fit_lmm_at_locus(
                y_rot, h_rot, xb_rot, eigvals, h2_base, use_reml)
            # scan with mediator
            res_med  = lmm_module.fit_lmm_at_locus(
                y_rot, h_rot, xm_rot, eigvals, h2_med, use_reml)

            rows.append({
                'locus':         lid,
                'chr':           chrom,
                'pos_mb':        float(pos[i_l]),
                'lod':           res_med.lod,         # LOD after conditioning on mediator
                'p_value':       res_med.p_value,
                'lod_base':      res_base.lod,         # LOD without mediator
                'mediation_lod': max(res_base.lod - res_med.lod, 0.0),
                'outcome_id':    outcome_id,
                'mediator_id':   mediator_id,
            })

    return pd.DataFrame(rows)


def fixef_mediation_scan(outcome, mediator, genome_cache,
                         covariates=None, use_rint=True,
                         outcome_id='outcome', mediator_id='mediator',
                         just_these_loci=None):
    """fixed-effect mediation scan — no kinship, ols only.

    fast version for exploratory analysis or large-scale screening.
    equivalent to miqtl::fixef.mediation.scan().

    parameters
    ----------
    outcome : np.ndarray (n,)
    mediator : np.ndarray (n,)
    genome_cache : GenomeCache
    covariates : np.ndarray or None
    use_rint : bool
    outcome_id : str
    mediator_id : str
    just_these_loci : set or None

    returns
    -------
    pd.DataFrame  same columns as mediation_scan_qr()
    """
    from .utils import rint as _rint

    valid  = (~np.isnan(outcome)) & (~np.isnan(mediator))
    y      = outcome[valid].copy()
    m      = mediator[valid].copy()
    if use_rint:
        y = _rint(y)
    n      = len(y)

    x_base = (covariates[valid].copy() if covariates is not None
              else np.ones((n, 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((n, 1)), x_base])

    m_std  = (m - m.mean()) / max(m.std(), 1e-10)
    x_med  = np.hstack([m_std.reshape(-1, 1), x_base])
    p_b    = x_base.shape[1]
    p_m    = x_med.shape[1]

    # null rss for base and mediation models
    def _rss_ols(x, y):
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        e = y - x @ b; return float(e @ e)

    rss0_b = _rss_ols(x_base, y)
    rss0_m = _rss_ols(x_med,  y)

    rows = []

    for chrom in genome_cache.chromosomes:
        lids, pos, probs = genome_cache.get_chromosome(chrom)
        nf1 = genome_cache.n_founders - 1

        for i_l, lid in enumerate(lids):
            if just_these_loci and lid not in just_these_loci:
                continue
            h   = probs[i_l][valid, :nf1]
            q   = nf1

            zb  = np.hstack([h, x_base])
            zm  = np.hstack([h, x_med])

            rss1_b = _rss_ols(zb, y); rss1_m = _rss_ols(zm, y)
            lod_b  = max((n/2.0)*np.log10(max(rss0_b,1e-30)/max(rss1_b,1e-30)), 0.0)
            lod_m  = max((n/2.0)*np.log10(max(rss0_m,1e-30)/max(rss1_m,1e-30)), 0.0)

            df2 = n - p_m - q
            if df2 > 0 and rss1_m > 0:
                f  = ((rss0_m - rss1_m)/q) / (rss1_m/df2)
                pv = 1.0 - sp_stats.f.cdf(max(f,0), q, df2)
            else:
                pv = 1.0

            rows.append({'locus': lid, 'chr': chrom, 'pos_mb': float(pos[i_l]),
                         'lod': lod_m, 'p_value': pv,
                         'lod_base': lod_b,
                         'mediation_lod': max(lod_b - lod_m, 0.0),
                         'outcome_id': outcome_id, 'mediator_id': mediator_id})

    return pd.DataFrame(rows)

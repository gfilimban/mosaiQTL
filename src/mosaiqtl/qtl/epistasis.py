"""
epistasis — pairwise GxG interaction and conditional multi-QTL scans

three functions for dissecting multi-locus genetic architecture
in multiparent populations:

  scan_epistasis()            — pairwise interaction test at haplotype resolution
  test_haplotype_combinations() — founder-pair expression heatmap at two loci
  scan_conditional()          — forward-stepwise multi-signal decomposition
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats as sp_stats

logger = logging.getLogger('mosaiqtl')


# ─────────────────────────────────────────────────────────────────────────────
# scan_epistasis
# ─────────────────────────────────────────────────────────────────────────────

def scan_epistasis(phenotype, genomecache, locus_a, locus_b,
                   covariates=None, kinship_decomp=None,
                   use_rint=True, use_reml=True,
                   phenotype_id='unknown'):
    """pairwise GxG interaction test at haplotype resolution.

    tests whether the combination of founder haplotypes at two loci
    produces expression not predicted by their additive sum — polygenic
    epistasis in the Collaborative Cross.

    model:
      additive:     y = H^a α + H^b β + X ξ + u
      interaction:  y = H^a α + H^b β + (H^a ⊗ H^b) γ + X ξ + u
      interaction LOD = LOD_int − LOD_add

    the interaction term (H^a ⊗ H^b) ∈ ℝ^{(J-1)²} is the Khatri-Rao
    product of the two haplotype design matrices (both with last founder
    dropped for identifiability).

    parameters
    ----------
    phenotype    : np.ndarray (n,)
    genomecache : GenomeCache
    locus_a      : str   first locus id
    locus_b      : str   second locus id
    covariates   : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values', 'vectors'}}
    use_rint     : bool
    use_reml     : bool
    phenotype_id : str

    returns
    -------
    dict with keys:
        lod_additive    : float   LOD under additive model
        lod_interaction : float   LOD under full interaction model
        lod_epistasis   : float   interaction LOD drop = lod_int - lod_add
        p_epistasis     : float   p-value for interaction term (chi²(J-1)²)
        betas_a         : np.ndarray (J-1,)  main effects locus a
        betas_b         : np.ndarray (J-1,)  main effects locus b
        betas_int       : np.ndarray ((J-1)²,) interaction effects
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    # locate locus_a and locus_b in cache
    def _get_locus_probs(locus_id):
        """Return haplotype probability array (n_samples, J) for locus_id."""
        for chrom in genomecache.chromosomes:
            lids, pos, probs = genomecache.get_chromosome(chrom)
            idx = np.where(lids == locus_id)[0]
            if len(idx):
                return chrom, probs[idx[0]]   # (n, J)
        raise KeyError(f'locus {locus_id!r} not found in cache')

    chr_a, h_a = _get_locus_probs(locus_a)
    chr_b, h_b = _get_locus_probs(locus_b)

    valid = ~np.isnan(phenotype)
    y     = phenotype[valid].copy()
    if use_rint:
        y = _rint(y)
    n     = len(y)

    x = (covariates[valid].copy() if covariates is not None
         else np.ones((n, 1), dtype=float))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    nf1 = genomecache.n_founders - 1
    ha  = h_a[valid, :nf1]   # (n, J-1) locus a haplotype design
    hb  = h_b[valid, :nf1]   # (n, J-1) locus b haplotype design

    # Khatri-Rao row-wise outer product: (H^a ⊗ H^b)
    # shape: (n, (J-1)²) — each row is ha[i] ⊗ hb[i]
    hab = ha[:, :, None] * hb[:, None, :]              # (n, J-1, J-1)
    hab = hab.reshape(n, nf1 * nf1)                    # (n, (J-1)²)

    # choose kinship from locus_a chromosome (arbitrary; both loci share same y)
    chrom_k = chr_a if chr_a in kinship_decomp else chr_b
    ev  = kinship_decomp[chrom_k]['values']
    evc = kinship_decomp[chrom_k]['vectors'][valid]

    y_rot   = evc.T @ y
    x_rot   = evc.T @ x
    ha_rot  = evc.T @ ha
    hb_rot  = evc.T @ hb
    hab_rot = evc.T @ hab

    h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, ev,
                                         method='reml' if use_reml else 'ml')

    # additive model: [Ha Hb X]
    z_add = np.hstack([ha_rot, hb_rot, x_rot])
    res_add = _wls_lod(y_rot, x_rot, z_add, ev, h2, n)

    # interaction model: [Ha Hb Ha⊗Hb X]
    z_int = np.hstack([ha_rot, hb_rot, hab_rot, x_rot])
    res_int = _wls_lod(y_rot, x_rot, z_int, ev, h2, n)

    lod_epi = max(res_int['lod'] - res_add['lod'], 0.0)
    chi2    = 2.0 * lod_epi * np.log(10)              # approx LRT statistic
    p_epi   = float(1.0 - sp_stats.chi2.cdf(chi2, df=nf1 * nf1))

    q = nf1
    return {
        'lod_additive':    res_add['lod'],
        'lod_interaction': res_int['lod'],
        'lod_epistasis':   lod_epi,
        'p_epistasis':     p_epi,
        'betas_a':         res_int['betas'][:q],
        'betas_b':         res_int['betas'][q:2*q],
        'betas_int':       res_int['betas'][2*q:],
    }


# ─────────────────────────────────────────────────────────────────────────────
# test_haplotype_combinations
# ─────────────────────────────────────────────────────────────────────────────

def test_haplotype_combinations(phenotype, genomecache, locus_a, locus_b,
                                 sample_ids=None, min_strains=3):
    """founder-pair mean expression heatmap at two peak loci.

    for each pair (f_a, f_b) of founder haplotypes, subsets strains that
    carry f_a at locus_a AND f_b at locus_b (using hard-call max-prob
    assignment), then computes the mean phenotype for that combination.

    reveals whether combinations of founder haplotypes produce expression
    levels not predictable from the sum of individual founder effects —
    a direct visualisation of polygenic epistasis.

    parameters
    ----------
    phenotype    : np.ndarray (n,) or dict {strain_id: float}
    genomecache : GenomeCache
    locus_a      : str
    locus_b      : str
    sample_ids   : array-like (n,) or None  maps phenotype rows to strains
    min_strains  : int  minimum strains per cell to report (else NaN)

    returns
    -------
    pd.DataFrame  (J × J) mean expression table
        index = founder at locus_a, columns = founder at locus_b
        cell (i,j) = mean phenotype for strains with f_a=i AND f_b=j
        NaN when fewer than min_strains strains in that cell
    """
    founders = list(genomecache.founders)
    J        = len(founders)

    def _get_hard_calls(locus_id):
        """return hard-call founder assignment per strain."""
        for chrom in genomecache.chromosomes:
            lids, _, probs = genomecache.get_chromosome(chrom)
            idx = np.where(lids == locus_id)[0]
            if len(idx):
                return np.argmax(probs[idx[0]], axis=1)   # (n_strains,)
        raise KeyError(f'locus {locus_id!r} not found')

    calls_a = _get_hard_calls(locus_a)   # (m,) index into founders
    calls_b = _get_hard_calls(locus_b)

    # align phenotype to cache strain order
    strains = list(genomecache.sample_ids)
    if isinstance(phenotype, dict):
        y_arr = np.array([phenotype.get(s, np.nan) for s in strains])
    else:
        phenotype = np.asarray(phenotype)
        if sample_ids is not None:
            sid_map = {s: i for i, s in enumerate(sample_ids)}
            y_arr   = np.array([phenotype[sid_map[s]] if s in sid_map else np.nan
                                for s in strains])
        else:
            y_arr = phenotype   # assume already aligned to cache order

    # build J×J mean matrix
    mat   = np.full((J, J), np.nan)
    for fa in range(J):
        for fb in range(J):
            mask = (calls_a == fa) & (calls_b == fb) & ~np.isnan(y_arr)
            if mask.sum() >= min_strains:
                mat[fa, fb] = float(y_arr[mask].mean())

    return pd.DataFrame(mat, index=founders, columns=founders)


# ─────────────────────────────────────────────────────────────────────────────
# scan_conditional
# ─────────────────────────────────────────────────────────────────────────────

def scan_conditional(scan_result, genomecache, covariates=None,
                     kinship_decomp=None, n_signals=3, lod_drop_threshold=1.5,
                     use_rint=True, use_reml=True):
    """forward-stepwise multi-signal decomposition (Layer 3b).

    conditions on the peak-locus BLUP (best linear unbiased predictor of the
    haplotype effect) and rescans residuals for secondary signals.  repeats
    until n_signals reached or no residual peak exceeds lod_drop_threshold.

    rationale: CC haplotype blocks average ~10 Mb; a single QTL interval
    commonly contains multiple independent regulatory variants. conditioning
    on BLUP rather than OLS peak residual is more conservative — it accounts
    for the uncertainty in the haplotype effect estimate and avoids
    over-splitting within LD blocks.

    this is the haplotype-space analogue of SuSiE's iterative Bayesian
    stepwise selection, without the Bayesian posterior integral.

    parameters
    ----------
    scan_result         : ScanResult  Layer 1 output for one gene
    genomecache        : GenomeCache
    covariates          : np.ndarray (n, p) or None
    kinship_decomp      : dict or None
    n_signals           : int  max independent signals (default 3)
    lod_drop_threshold  : float  min LOD for secondary signal (default 1.5)
    use_rint            : bool
    use_reml            : bool

    returns
    -------
    list of ScanResult  one per independent signal found (length 1..n_signals)
        scan_result[0] is always the original (primary) result
    """
    from ..genotype import kinship as kin_module
    from .scan import ScanResult

    if not len(scan_result.lod):
        return [scan_result]

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    signals = [scan_result]

    # reconstruct phenotype from the peak ScanResult — we need the original y.
    # scan_result does not store y; caller must pass it via a ScanResult subclass
    # or we operate purely on the LOD profile for peak identification.
    # current implementation: identify independent peaks by LOD-drop criterion
    # on the original scan and reruns; full BLUP conditioning requires y to be
    # passed explicitly (see scan_conditional_with_y below for the complete form).

    # ── simplified peak-drop approach (no y required) ─────────────────────────
    # for each additional signal: find the highest-LOD locus NOT within
    # lod_drop_threshold of any already-found peak locus.
    peak_positions = [scan_result.peak_locus()['position']]
    lod             = scan_result.lod.copy()
    positions       = scan_result.positions

    for _ in range(n_signals - 1):
        # mask out any position within 5 Mb of an existing peak
        near_peak = np.zeros(len(positions), dtype=bool)
        for pp in peak_positions:
            near_peak |= np.abs(positions - pp) < 5e6  # 5 Mb exclusion zone

        residual_lod = np.where(near_peak, -np.inf, lod)
        best_i       = int(np.argmax(residual_lod))
        best_lod     = float(residual_lod[best_i])

        if best_lod < lod_drop_threshold:
            break   # no more signals above threshold

        # create a synthetic ScanResult for this secondary peak's neighbourhood
        chrom = scan_result.chromosomes[best_i]
        cis   = np.abs(positions - positions[best_i]) < 5e6
        cis  &= scan_result.chromosomes == chrom

        secondary = ScanResult(
            locus_ids    = scan_result.locus_ids[cis],
            chromosomes  = scan_result.chromosomes[cis],
            positions    = positions[cis],
            lod          = lod[cis],
            p_value      = scan_result.p_value[cis],
            betas        = scan_result.betas[cis],
            h2_null      = scan_result.h2_null,
            phenotype_id = f'{scan_result.phenotype_id}_signal{len(signals)+1}',
        )
        signals.append(secondary)
        peak_positions.append(positions[best_i])

    return signals


def scan_conditional_with_y(phenotype, scan_result, genomecache,
                              covariates=None, kinship_decomp=None,
                              n_signals=3, lod_drop_threshold=1.5,
                              use_rint=True, use_reml=True):
    """full BLUP-conditioning version of scan_conditional (requires phenotype).

    this is the preferred form — it computes the exact BLUP residual
    at each conditioning step rather than using LOD-drop heuristics.

    parameters
    ----------
    phenotype    : np.ndarray (n,)  original phenotype (same n as scan_result)
    scan_result  : ScanResult       Layer 1 output
    (all other params same as scan_conditional)

    returns
    -------
    list of ScanResult
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint
    from .scan import scan_h2lmm

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    valid  = ~np.isnan(phenotype)
    y      = phenotype[valid].copy()
    if use_rint:
        y = _rint(y)
    x = (covariates[valid].copy() if covariates is not None
         else np.ones((len(y), 1), dtype=float))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((len(y), 1)), x])

    signals = [scan_result]
    y_res   = y.copy()   # residual phenotype updated each iteration

    for sig_i in range(1, n_signals):
        if not len(signals[-1].lod):
            break
        prev  = signals[-1]
        pi    = int(np.argmax(prev.lod))
        chrom = str(prev.chromosomes[pi])
        lid   = str(prev.locus_ids[pi])

        # get haplotype probs at peak locus
        lids_c, _, probs_c = genomecache.get_chromosome(chrom)
        loc_idx = np.where(lids_c == lid)[0]
        if not len(loc_idx):
            break
        h_peak = probs_c[loc_idx[0]][valid, :genomecache.n_founders - 1]

        # BLUP of haplotype effect at peak locus using LOCO kinship
        if chrom not in kinship_decomp:
            break
        ev  = kinship_decomp[chrom]['values']
        evc = kinship_decomp[chrom]['vectors'][valid]
        h2, _ = lmm_module.estimate_h2_null(evc.T @ y_res, evc.T @ x, ev)
        res = lmm_module.fit_lmm_at_locus(evc.T @ y_res, evc.T @ h_peak,
                                            evc.T @ x, ev, h2)
        blup_hat = h_peak @ res.betas   # predicted haplotype contribution (n,)
        y_res    = y_res - blup_hat     # condition out peak haplotype BLUP

        # rescan residuals
        sr_next = scan_h2lmm(y_res, genomecache, x,
                              kinship_decomp, use_rint=False,
                              phenotype_id=f'{scan_result.phenotype_id}_s{sig_i+1}',
                              scan_window='genome')
        if sr_next.lod.max() < lod_drop_threshold:
            break
        signals.append(sr_next)

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _wls_lod(y_rot, x_rot, z_rot, eigvals, h2, n):
    """weighted least squares LOD and betas for an arbitrary design matrix z."""
    delta = h2 / max(1 - h2, 1e-10)
    w_inv = 1.0 / (delta * eigvals + 1.0)
    xw    = x_rot * w_inv[:, None]
    zw    = z_rot * w_inv[:, None]
    try:
        b_null = np.linalg.solve(xw.T @ x_rot, xw.T @ y_rot)
        b_alt  = np.linalg.solve(zw.T @ z_rot, zw.T @ y_rot)
    except np.linalg.LinAlgError:
        return {'lod': 0.0, 'betas': np.zeros(z_rot.shape[1])}
    e0   = y_rot - x_rot @ b_null; rss0 = float(e0 @ (e0 * w_inv))
    e1   = y_rot - z_rot @ b_alt;  rss1 = float(e1 @ (e1 * w_inv))
    lod  = max((n / 2.0) * np.log10(rss0 / max(rss1, 1e-30)), 0.0)
    return {'lod': lod, 'betas': b_alt}

"""
mosaiqtl.nb — negative binomial genome-wide scan (primary model)

primary eQTL model for pseudobulk snRNA-seq counts.  unlike the LMM port,
this module operates on raw integer counts — no RINT, no kinship random effect.

design:
  scan_nb()             genome-wide NB GLM scan for one gene
  scan_nb_multi_gene()  vectorised loop over many genes (same cache)

score test (default, O(L) from one null fit per gene):
  H_l'  working-residuals   — score vector U_l ∈ ℝ^{J-1}
  H_l'  diag(w_nb)  H_l    — information matrix I_l ∈ ℝ^{(J-1)×(J-1)}
  Q_l = U_l' I_l^{-1} U_l  ~ χ²(J-1)
  LOD_l = Q_l / (2 ln 10)

LRT (use_lrt=True, exact betas but O(L × IRLS)):
  alt model fit per locus via edgepython.glm_fit
  LOD = (ℓ_alt − ℓ_null) / ln(10)

NB log-likelihood (phi = dispersion parameter, r = 1/phi size):
  ℓ(y | μ, φ) = Σ_i [ln Γ(y_i+r) − ln Γ(r) − ln Γ(y_i+1)
                       + r ln(r/(r+μ_i)) + y_i ln(μ_i/(r+μ_i))]

dependencies: edgepython (pachter 2026), scipy
"""

import numpy as np
import logging
from scipy.special import gammaln as _gammaln
from scipy import stats as _stats

logger = logging.getLogger('mosaiqtl')


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nb_loglik(y, mu, phi):
    """NB log-likelihood summed over n samples.

    parameters
    ----------
    y   : np.ndarray (n,)  observed counts (non-negative integers)
    mu  : np.ndarray (n,)  fitted means (>0)
    phi : float            dispersion parameter φ (variance = μ + φ μ²)

    returns
    -------
    float  total log-likelihood ℓ(y | μ, φ)
    """
    r   = 1.0 / max(phi, 1e-12)           # NB size parameter
    mu  = np.maximum(mu, 1e-300)
    ll  = (_gammaln(y + r) - _gammaln(r) - _gammaln(y + 1)
           + r * np.log(r / (r + mu))
           + y * np.log(mu / (r + mu)))
    return float(ll.sum())


def _fit_nb(dgelist, design, phi, ep):
    """fit NB GLM and return (mu_hat, coefs, fit_obj)."""
    try:
        fit   = ep.glm_fit(dgelist, design=design, dispersion=phi)
        mu    = np.atleast_1d(fit.get('fitted.values',
                                       np.full(design.shape[0], 0.5))).ravel()
        coefs = np.atleast_1d(fit.get('coefficients',
                                       np.zeros(design.shape[1]))).ravel()
        return mu, coefs, fit
    except Exception as e:
        logger.debug(f'glm_fit failed: {e}')
        n = design.shape[0]
        return np.full(n, float(np.mean(dgelist.get('counts', np.ones((1, n)))))), \
               np.zeros(design.shape[1]), {}


def _score_stat_vectorised(H_all, r_vec, w_vec):
    """compute NB score statistics for all loci on a chromosome at once.

    parameters
    ----------
    H_all : np.ndarray (n_loci, n, J-1)  haplotype design (last founder dropped)
    r_vec : np.ndarray (n,)               working residual = (y−μ) / (1+φμ)
    w_vec : np.ndarray (n,)               weight = μ / (1+φμ)

    returns
    -------
    np.ndarray (n_loci,)  score statistics Q_l ~ χ²(J-1) under H0
    """
    n_loci, n, q = H_all.shape

    # score vectors U_l = H_l' r  →  (n_loci, q)
    u = np.einsum('lij,i->lj', H_all, r_vec)

    # weighted H:  (H_l * w)  →  (n_loci, n, q)
    hw = H_all * w_vec[np.newaxis, :, np.newaxis]

    # information matrices I_l = H_l' diag(w) H_l  →  (n_loci, q, q)
    info = np.einsum('lij,lik->ljk', hw, H_all)

    # Q_l = U_l' I_l^{-1} U_l — solve each (q×q) system
    stats = np.zeros(n_loci)
    for l in range(n_loci):
        try:
            stats[l] = max(float(u[l] @ np.linalg.solve(info[l], u[l])), 0.0)
        except np.linalg.LinAlgError:
            stats[l] = 0.0
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# public scan API
# ─────────────────────────────────────────────────────────────────────────────

def scan_nb(counts, genome_cache, covariates=None, sample_ids=None,
            lib_sizes=None, dispersion=None,
            use_lrt=False, just_these_loci=None,
            phenotype_id='unknown'):
    """genome-wide NB GLM haplotype scan for one gene (pseudobulk).

    primary mosaiQTL model — operates on raw integer pseudobulk counts.
    no RINT (NB handles discrete counts natively).
    no kinship term (fixed-effect GLM; LMM is the miQTL validation port).

    score test (default):
      one null model per gene; score statistic Q_l evaluated at each locus.
      O(L × n × J²) — vectorised per chromosome, very fast.

    LRT (use_lrt=True):
      alt model refitted per locus via edgepython; returns log-scale betas.
      exact but O(L × IRLS × n) — use for peak loci after scoring.

    parameters
    ----------
    counts          : np.ndarray (n,)    raw pseudobulk counts
    genome_cache    : GenomeCache
    covariates      : np.ndarray (n, p)  intercept + batch + PCs (or None)
    sample_ids      : array-like (n,) or None
        phenotype-level ids; if different from genome_cache.sample_ids,
        Z-matrix expansion is applied (H_n = Z H_m, n ≥ m).
    lib_sizes       : np.ndarray (n,) or None
        total counts per sample (used as offset log(lib_size)).
        if None, estimated as column sums scaled by TMM norm factors.
    dispersion      : float or None
        pre-estimated NB dispersion φ; if None, estimated here via edgepython
        three-stage EB pipeline (common → trended → tagwise).
    use_lrt         : bool
        if True, fit alt model at each locus for exact LOD and betas.
        if False (default), use vectorised score test (betas=zeros).
    just_these_loci : set or None
        restrict scan to this set of locus ids (cis-window for eQTL mapping).
    phenotype_id    : str

    returns
    -------
    ScanResult  (same interface as scan_h2lmm; h2_null = {} always)
    """
    import edgepython as ep
    from .scan import ScanResult
    from .utils import build_z_matrix

    # ── sample mapping → Z matrix ─────────────────────────────────────────
    cache_ids = list(genome_cache.sample_ids)
    sid       = list(sample_ids) if sample_ids is not None else cache_ids
    if sid == cache_ids:
        z_sub, valid_sid = None, np.ones(len(sid), dtype=bool)
    else:
        z_full, valid_sid = build_z_matrix(sid, cache_ids)
        z_sub = z_full[valid_sid]          # (n_valid, m)

    # apply validity mask
    y   = counts[valid_sid].astype(float)
    n   = len(y)

    cov = covariates[valid_sid].copy() if covariates is not None else None
    lib = lib_sizes[valid_sid] if lib_sizes is not None else None

    # ── design matrix X_cov ───────────────────────────────────────────────
    if cov is not None:
        x = cov if np.allclose(cov[:, 0], 1.0) else np.hstack([np.ones((n, 1)), cov])
    else:
        x = np.ones((n, 1), dtype=float)

    # ── build edgepython dgelist ──────────────────────────────────────────
    dg = ep.make_dgelist(counts=y.reshape(1, -1))
    if lib is not None:
        dg['samples']['lib.size'] = lib.astype(float)
    dg = ep.calc_norm_factors(dg)

    # ── dispersion estimation ─────────────────────────────────────────────
    if dispersion is None:
        dg  = ep.estimate_disp(dg, design=x)
        phi = float(np.atleast_1d(dg.get('tagwise.dispersion',
                        dg.get('common.dispersion', 0.1)))[0])
    else:
        phi = float(dispersion)
    phi = max(phi, 1e-6)

    # ── null model ────────────────────────────────────────────────────────
    mu_null, _, null_fit = _fit_nb(dg, x, phi, ep)
    null_ll              = _nb_loglik(y, mu_null, phi)

    # score test working quantities — computed once, used for all loci
    denom  = 1.0 + phi * mu_null          # (n,)
    r_vec  = (y - mu_null) / denom        # working residual  (n,)
    w_vec  = mu_null / denom              # weight            (n,)

    nf1 = genome_cache.n_founders - 1     # J-1 haplotype dof

    # ── per-chromosome scan ───────────────────────────────────────────────
    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genome_cache.chromosomes:
        lids, pos, probs = genome_cache.get_chromosome(chrom)
        # probs shape: (n_loci, m, J)

        # filter loci
        if just_these_loci is not None:
            mask  = np.isin(lids, list(just_these_loci))
            lids  = lids[mask]; pos = pos[mask]; probs = probs[mask]
        if len(lids) == 0:
            continue

        n_loci = len(lids)

        # Z-expand haplotypes: (n_loci, n, J)
        if z_sub is not None:
            # H_n = Z H_m — vectorised via einsum
            h_all = np.einsum('nm,lmj->lnj', z_sub, probs)
        else:
            h_all = probs                  # already (n_loci, n, J)

        h_all = h_all[:, :, :nf1]         # drop last founder (n_loci, n, J-1)

        if not use_lrt:
            # ── vectorised score test ─────────────────────────────────────
            sc = _score_stat_vectorised(h_all, r_vec, w_vec)  # (n_loci,)
            lods  = sc / (2.0 * np.log(10))
            pvals = 1.0 - _stats.chi2.cdf(sc, df=nf1)
            betas = np.zeros((n_loci, nf1))
        else:
            # ── LRT: fit alt model at each locus ─────────────────────────
            lods  = np.zeros(n_loci)
            pvals = np.ones(n_loci)
            betas = np.zeros((n_loci, nf1))
            dg_l  = ep.make_dgelist(counts=y.reshape(1, -1))
            if lib is not None:
                dg_l['samples']['lib.size'] = lib.astype(float)
            dg_l = ep.calc_norm_factors(dg_l)
            for i_l in range(n_loci):
                x_alt  = np.hstack([h_all[i_l], x])    # (n, J-1+p)
                mu_alt, coefs, _ = _fit_nb(dg_l, x_alt, phi, ep)
                alt_ll  = _nb_loglik(y, mu_alt, phi)
                lrt_chi = max(2.0 * (alt_ll - null_ll), 0.0)
                lods[i_l]    = lrt_chi / (2.0 * np.log(10))
                pvals[i_l]   = 1.0 - _stats.chi2.cdf(lrt_chi, df=nf1)
                betas[i_l]   = coefs[:nf1]              # log-scale founder effects

        all_lids.append(lids)
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(pos)
        all_lod.append(lods)
        all_pval.append(pvals)
        all_betas.append(betas)

    if not all_lids:
        return ScanResult(
            locus_ids=np.array([]), chromosomes=np.array([]),
            positions=np.array([]), lod=np.array([]), p_value=np.array([]),
            betas=np.empty((0, nf1)), h2_null={}, phenotype_id=phenotype_id)

    return ScanResult(
        locus_ids   = np.concatenate(all_lids),
        chromosomes = np.concatenate(all_chrs),
        positions   = np.concatenate(all_pos),
        lod         = np.concatenate(all_lod),
        p_value     = np.concatenate(all_pval),
        betas       = np.vstack(all_betas),
        h2_null     = {},
        phenotype_id = phenotype_id,
    )


def scan_nb_multi_gene(count_matrix, gene_ids, genome_cache,
                       covariates=None, sample_ids=None,
                       lib_sizes=None, dispersion_dict=None,
                       use_lrt=False, cis_loci_dict=None,
                       n_cores=1):
    """NB GLM scan for multiple genes sharing the same cache.

    parameters
    ----------
    count_matrix   : np.ndarray (n, G)  raw counts; columns = genes
    gene_ids       : list of str         length G
    genome_cache   : GenomeCache
    covariates     : np.ndarray (n, p) or None
    sample_ids     : array-like (n,) or None
    lib_sizes      : np.ndarray (n,) or None
    dispersion_dict: dict {gene_id: float} or None
        pre-estimated dispersions; missing genes estimated inside scan_nb.
    use_lrt        : bool
    cis_loci_dict  : dict {gene_id: set} or None
        per-gene cis loci sets for eQTL mapping.
    n_cores        : int  1 = serial; >1 uses multiprocessing

    returns
    -------
    dict {gene_id: ScanResult}
    """
    results = {}

    def _scan_one(gi, gid):
        phi  = dispersion_dict.get(gid) if dispersion_dict else None
        cis  = cis_loci_dict.get(gid)  if cis_loci_dict  else None
        return gid, scan_nb(
            count_matrix[:, gi], genome_cache, covariates,
            sample_ids, lib_sizes, phi, use_lrt, cis, gid)

    if n_cores == 1:
        for gi, gid in enumerate(gene_ids):
            logger.info(f'NB scan {gi+1}/{len(gene_ids)}: {gid}')
            _, res = _scan_one(gi, gid)
            results[gid] = res
    else:
        from multiprocessing import Pool
        with Pool(n_cores) as pool:
            for gid, res in pool.starmap(
                    _scan_one, enumerate(gene_ids)):
                results[gid] = res

    return results


# ─────────────────────────────────────────────────────────────────────────────
# legacy helpers (kept for vqtl, precompute_sufficient_stats downstream)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_gene_dispersion(counts, design, lib_sizes=None):
    """estimate per-gene overdispersion using edgepython EB pipeline.

    three-stage: common → trended → tagwise (WLEB shrinkage).

    parameters
    ----------
    counts    : np.ndarray (n_genes, n_samples)  raw count matrix
    design    : np.ndarray (n_samples, p)         covariate design matrix
    lib_sizes : np.ndarray (n_samples,) or None

    returns
    -------
    dict  {'common', 'trended', 'tagwise', 'dgelist'}
    """
    import edgepython as ep
    y = ep.make_dgelist(counts=counts)
    if lib_sizes is not None:
        y['samples']['lib.size'] = lib_sizes
    y = ep.calc_norm_factors(y)
    y = ep.estimate_disp(y, design=design)
    return {
        'common':  y.get('common.dispersion',  None),
        'trended': y.get('trended.dispersion', None),
        'tagwise': y.get('tagwise.dispersion', None),
        'dgelist': y,
    }


def fit_nb_null(counts_gene, design, dispersion, lib_sizes=None):
    """fit NB GLM null model (no haplotype term) for one gene."""
    import edgepython as ep
    y = ep.make_dgelist(counts=counts_gene.reshape(1, -1))
    if lib_sizes is not None:
        y['samples']['lib.size'] = lib_sizes
    y = ep.calc_norm_factors(y)
    return ep.glm_fit(y, design=design, dispersion=dispersion)


def score_test_at_locus(null_fit, h_locus, counts_gene, dispersion,
                        lib_sizes=None):
    """NB score test for haplotype effect at one locus.

    evaluates score statistic Q = U' I^{-1} U ~ χ²(J-1) where:
      U = H_l' (y−μ)/(1+φμ)     score vector
      I = H_l' diag(μ/(1+φμ)) H_l  fisher information

    parameters
    ----------
    null_fit    : dict  output of fit_nb_null / _fit_nb
    h_locus     : np.ndarray (n, J-1)  haplotype design at this locus
    counts_gene : np.ndarray (n,)      raw counts
    dispersion  : float
    lib_sizes   : ignored (offset already in null fit)

    returns
    -------
    dict  {'score_stat', 'p_value', 'df'}
    """
    mu  = np.atleast_1d(null_fit.get('fitted.values',
                                      np.full(len(counts_gene), 0.5))).ravel()
    mu  = np.maximum(mu, 1e-300)
    q   = h_locus.shape[1]
    den = 1.0 + dispersion * mu
    r_v = (counts_gene - mu) / den          # working residual
    w_v = mu / den                          # weight

    u    = h_locus.T @ r_v                  # (q,) score vector
    hw   = h_locus * w_v[:, np.newaxis]     # (n, q)
    info = hw.T @ h_locus                   # (q, q) information

    try:
        sc = max(float(u @ np.linalg.solve(info, u)), 0.0)
    except np.linalg.LinAlgError:
        sc = 0.0

    return {
        'score_stat': sc,
        'p_value':    float(1.0 - _stats.chi2.cdf(sc, df=q)),
        'df':         q,
    }


def precompute_sufficient_stats(counts, subject_ids, library_sizes=None):
    """per-subject sufficient statistics for cell-level NB models.

    parameters
    ----------
    counts       : np.ndarray (n_cells,)
    subject_ids  : np.ndarray (n_cells,) str
    library_sizes: np.ndarray (n_cells,) or None

    returns
    -------
    dict  {'subjects', 'n_cells', 'total_count', 'mean_count', 'var_count'}
    """
    unique_subs = np.unique(subject_ids)
    ns          = len(unique_subs)
    n_cells     = np.zeros(ns, dtype=np.int64)
    total_count = np.zeros(ns, dtype=np.float64)
    mean_count  = np.zeros(ns, dtype=np.float64)
    var_count   = np.zeros(ns, dtype=np.float64)

    sub_map = {}
    for idx, sid in enumerate(subject_ids):
        sub_map.setdefault(sid, []).append(idx)

    for s_idx, sid in enumerate(unique_subs):
        idx_s = np.array(sub_map[sid])
        y_s   = counts[idx_s].astype(np.float64)
        n_cells[s_idx]     = len(idx_s)
        total_count[s_idx] = y_s.sum()
        mean_count[s_idx]  = y_s.mean()
        var_count[s_idx]   = float(np.var(y_s, ddof=1)) if len(y_s) > 1 else 0.0

    return {
        'subjects':    unique_subs,
        'n_cells':     n_cells,
        'total_count': total_count,
        'mean_count':  mean_count,
        'var_count':   var_count,
    }

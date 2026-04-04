"""
mosaiqtl.nb — negative binomial genome-wide scan (primary model)

two-stage design:
  stage 1 — null model + dispersion (cell level)
      glm_sc_fit(counts_cells, sample=strain_ids, design=X_cells)
          NEBULA-LN: NB-gamma mixed model with per-subject random effect
          absorbs within-strain cell-to-cell variability and between-strain
          overdispersion simultaneously. sigma_sample = per-gene subject
          variance component; dispersion = cell-level NB overdispersion.
      shrink_sc_disp(fit, counts_cells)
          empirical Bayes shrinkage of cell-level dispersion toward a
          trended prior (same WLEB framework as edgeR's tagwise dispersion).

  stage 2 — genome-wide scan (pseudobulk, vectorised)
      for each gene: pseudobulk counts (one per strain) + shrunk dispersion
      score test over all loci via einsum — O(L × n × J²) per chromosome.
      sigma_sample optionally inflates score test denominator to account for
      residual subject-level variance not captured by fixed covariates.

model:
  cell level (stage 1):
      Y_ij ~ NB(μ_ij, φ_g),   log μ_ij = X_j β_g + log s_j + u_i
      u_i ~ N(0, σ²_g)         subject random effect per gene g

  pseudobulk scan (stage 2):
      S_ig = Σ_j Y_ij           pseudobulk sum per strain i, gene g
      H₀: log E[S_ig] = X_i β₀ + log(n_i s̄_i)
      H₁: log E[S_ig] = H_iℓ γ + X_i β + log(n_i s̄_i)
      Q_ℓ = U_ℓᵀ I_ℓ⁻¹ U_ℓ ~ χ²(J−1)   score test

  kinship note: the per-subject random effect u_i from glm_sc_fit is an
  unstructured exchangeable random effect, not a LOCO-kinship random effect.
  for CC strains with controlled admixture and 10 expression PCs as fixed
  covariates, this is adequate for the eQTL proof-of-concept.  a full
  NB-LMM with LOCO kinship covariance is a planned future extension.

dependencies: edgepython (pachter 2026, GPL-3.0)
"""

import numpy as np
import logging
from scipy.special import gammaln as _gammaln
from scipy import stats as _stats

logger = logging.getLogger('mosaiqtl')


# ─────────────────────────────────────────────────────────────────────────────
# stage 1 helpers
# ─────────────────────────────────────────────────────────────────────────────

def fit_null_sc(counts_cells, strain_ids, covariates_cells=None,
                norm_method='TMM', shrink=True, verbose=False):
    """fit NEBULA-LN null model on cell-level counts for all genes.

    wraps glm_sc_fit + shrink_sc_disp from edgepython.
    run once per cell type before the genome-wide scan.

    parameters
    ----------
    counts_cells      : np.ndarray (n_genes, n_cells)  raw UMI counts
    strain_ids        : array-like (n_cells,)  strain/subject per cell
    covariates_cells  : np.ndarray (n_cells, p) or None
        cell-level covariates (intercept + batch + PCs).
        if None, intercept only.
    norm_method       : str  'TMM' (default) or 'none'
    shrink            : bool  apply EB dispersion shrinkage (default True)
    verbose           : bool

    returns
    -------
    dict  edgepython fit object, augmented with:
        'dispersion_use' : np.ndarray (n_genes_filtered,)
            dispersion values to use in the scan:
            dispersion_shrunk if shrink=True, else dispersion
    """
    import edgepython as ep
    import pandas as pd

    strain_ids = np.asarray(strain_ids)

    # design: intercept + covariates (cell level)
    if covariates_cells is not None:
        x = np.asarray(covariates_cells, dtype=np.float64)
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((len(x), 1)), x])
        design = pd.DataFrame(x, columns=[f'V{i+1}' for i in range(x.shape[1])])
    else:
        design = None

    fit = ep.glm_sc_fit(
        counts_cells,
        sample=strain_ids,
        design=design,
        norm_method=norm_method,
        verbose=verbose,
    )

    if shrink:
        fit = ep.shrink_sc_disp(fit, counts=counts_cells)
        fit['dispersion_use'] = np.where(
            np.isfinite(fit['dispersion_shrunk']) & (fit['dispersion_shrunk'] > 0),
            fit['dispersion_shrunk'],
            fit['dispersion'],
        )
    else:
        fit['dispersion_use'] = fit['dispersion']

    n_pass = int(fit['gene_mask'].sum())
    logger.info(f'fit_null_sc: {n_pass}/{len(fit["gene_mask"])} genes pass filter  '
                f'med_disp={float(np.nanmedian(fit["dispersion_use"])):.3f}  '
                f'med_sigma={float(np.nanmedian(fit["sigma_sample"])):.3f}')
    return fit


def pseudobulk_sum(counts_cells, strain_ids, gene_mask=None):
    """aggregate cell-level counts to per-strain pseudobulk sums.

    parameters
    ----------
    counts_cells : np.ndarray (n_genes, n_cells)
    strain_ids   : array-like (n_cells,)
    gene_mask    : np.ndarray (n_genes,) bool or None
        if provided, subset to genes that passed glm_sc_fit filter

    returns
    -------
    pb_counts  : np.ndarray (n_genes_kept, n_strains)  pseudobulk sums
    pb_ncells  : np.ndarray (n_strains,)  cells per strain
    strains    : np.ndarray (n_strains,)  ordered strain ids
    """
    strain_ids = np.asarray(strain_ids)
    strains, inv = np.unique(strain_ids, return_inverse=True)
    n_strains = len(strains)

    mat = counts_cells[gene_mask] if gene_mask is not None else counts_cells
    n_genes = mat.shape[0]

    pb = np.zeros((n_genes, n_strains), dtype=np.float64)
    np.add.at(pb.T, inv, mat.T)  # (n_strains, n_genes) += counts

    pb_ncells = np.bincount(inv, minlength=n_strains).astype(np.float64)
    return pb, pb_ncells, strains


# ─────────────────────────────────────────────────────────────────────────────
# stage 2 — genome-wide score test
# ─────────────────────────────────────────────────────────────────────────────

def _nb_loglik(y, mu, phi):
    """NB log-likelihood summed over samples."""
    r  = 1.0 / max(phi, 1e-12)
    mu = np.maximum(mu, 1e-300)
    return float((_gammaln(y + r) - _gammaln(r) - _gammaln(y + 1)
                  + r * np.log(r / (r + mu))
                  + y * np.log(mu / (r + mu))).sum())


def _score_stats_chromosome(H_all, r_vec, w_vec):
    """vectorised NB score statistics for all loci on one chromosome.

    parameters
    ----------
    H_all : (L, n, J-1)  haplotype design at all loci
    r_vec : (n,)          working residual under null
    w_vec : (n,)          working weight under null

    returns
    -------
    (L,) score statistics Q_ℓ ~ χ²(J-1) under H₀
    """
    u    = np.einsum('lnj,n->lj',   H_all, r_vec,    optimize=True)  # (L,J-1)
    hw   = H_all * w_vec[None, :, None]
    info = np.einsum('lnj,lnk->ljk', hw, H_all,      optimize=True)  # (L,J-1,J-1)
    info += np.eye(H_all.shape[2])[None] * 1e-10
    sc   = np.zeros(H_all.shape[0])
    for l in range(len(sc)):
        try:
            sc[l] = max(float(u[l] @ np.linalg.solve(info[l], u[l])), 0.0)
        except np.linalg.LinAlgError:
            sc[l] = 0.0
    return sc


def scan_nb_gene(counts_pb, genome_cache, strain_order,
                  phi, sigma_sample=0.0,
                  covariates=None, lib_sizes=None,
                  use_lrt=False, just_these_loci=None,
                  phenotype_id='unknown'):
    """NB score-test scan for one gene on pseudobulk counts.

    parameters
    ----------
    counts_pb     : np.ndarray (n_strains,)  pseudobulk sum counts
    genome_cache  : GenomeCache
    strain_order  : array-like (n_strains,)  strain IDs in counts_pb order
                    must match genome_cache.sample_ids order
    phi           : float  NB dispersion (from fit_null_sc, shrunk)
    sigma_sample  : float  subject-level variance from glm_sc_fit (≥0)
                    inflates score denominator when >0
    covariates    : np.ndarray (n_strains, p) or None
    lib_sizes     : np.ndarray (n_strains,) or None  log-offset per strain
    use_lrt       : bool  fit alt model per locus for exact betas (slow)
    just_these_loci : set or None
    phenotype_id  : str

    returns
    -------
    ScanResult
    """
    import edgepython as ep
    from .scan import ScanResult
    from .utils import build_z_matrix

    phi  = max(float(phi), 1e-6)
    y    = np.asarray(counts_pb, dtype=np.float64)
    sids = list(strain_order)

    # align to cache order via Z-matrix
    cache_ids = list(genome_cache.sample_ids)
    if sids == cache_ids:
        z_sub, valid = None, np.ones(len(sids), dtype=bool)
    else:
        z_full, valid = build_z_matrix(sids, cache_ids)
        z_sub = z_full[valid]

    y = y[valid]
    n = len(y)

    x = (covariates[valid].copy() if covariates is not None
         else np.ones((n, 1), dtype=float))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    s = lib_sizes[valid] if lib_sizes is not None else np.ones(n)

    # ── null model via edgepython glm_fit ─────────────────────────────────
    dg = ep.make_dgelist(counts=y.reshape(1, -1))
    dg['samples']['lib.size'] = s.astype(float)
    dg = ep.calc_norm_factors(dg)
    mu_null, _, _ = _fit_nb_ep(dg, x, phi, ep)

    # score working quantities
    denom  = 1.0 + phi * mu_null
    # inflate denominator by sigma_sample: increases effective variance at each cell
    # approximation: var(S_i) ≈ (φ + σ²) μ_i² + μ_i   (pseudobulk, sum over cells)
    w_inf  = max(1.0 + sigma_sample, 1.0)
    r_vec  = (y - mu_null) / (denom * w_inf)
    w_vec  = mu_null / (denom * w_inf)

    null_ll = _nb_loglik(y, mu_null, phi)
    nf1     = genome_cache.n_founders - 1

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genome_cache.chromosomes:
        lids, pos, probs = genome_cache.get_chromosome(chrom)

        if just_these_loci is not None:
            mask = np.isin(lids, list(just_these_loci))
            lids, pos, probs = lids[mask], pos[mask], probs[mask]
        if not len(lids):
            continue

        # Z-expand haplotypes (n_loci, n, J-1)
        H = (np.einsum('nm,lmj->lnj', z_sub, probs[:, :, :nf1], optimize=True)
             if z_sub is not None else probs[:, valid, :nf1])

        if not use_lrt:
            sc   = _score_stats_chromosome(H, r_vec, w_vec)
            lods = sc / (2.0 * np.log(10))
            pv   = 1.0 - _stats.chi2.cdf(sc, df=nf1)
            betas = np.zeros((len(lids), nf1))
        else:
            lods  = np.zeros(len(lids)); pv = np.ones(len(lids))
            betas = np.zeros((len(lids), nf1))
            dg_l  = ep.make_dgelist(counts=y.reshape(1, -1))
            dg_l['samples']['lib.size'] = s.astype(float)
            dg_l = ep.calc_norm_factors(dg_l)
            for i_l in range(len(lids)):
                x_alt = np.hstack([H[i_l], x])
                mu_alt, coefs, _ = _fit_nb_ep(dg_l, x_alt, phi, ep)
                alt_ll = _nb_loglik(y, mu_alt, phi)
                chi2   = max(2.0 * (alt_ll - null_ll), 0.0)
                lods[i_l]  = chi2 / (2.0 * np.log(10))
                pv[i_l]    = 1.0 - _stats.chi2.cdf(chi2, df=nf1)
                betas[i_l] = coefs[:nf1]

        all_lids.append(lids); all_chrs.append(np.full(len(lids), chrom))
        all_pos.append(pos); all_lod.append(lods)
        all_pval.append(pv); all_betas.append(betas)

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


def scan_nb(counts_cells, genome_cache, strain_ids_cells,
             gene_ids=None, covariates_cells=None,
             covariates_strains=None, lib_sizes=None,
             norm_method='TMM', shrink_disp=True,
             use_lrt=False, cis_loci_dict=None,
             n_cores=1, verbose=False):
    """genome-wide NB GLM scan — full pipeline for one cell type.

    stage 1: glm_sc_fit + shrink_sc_disp (cell level, once)
    stage 2: score test per gene on pseudobulk counts (vectorised)

    parameters
    ----------
    counts_cells      : np.ndarray (n_genes, n_cells)  raw UMI counts
    genome_cache      : GenomeCache
    strain_ids_cells  : array-like (n_cells,)  strain per cell
    gene_ids          : list of str or None  row names for counts_cells
    covariates_cells  : np.ndarray (n_cells, p) or None
    covariates_strains: np.ndarray (n_strains, p) or None
        strain-level covariates for the scan (batch + PCs).
        if None, derived from covariates_cells by strain mean.
    lib_sizes         : np.ndarray (n_strains,) or None
    norm_method       : str  'TMM' or 'none'
    shrink_disp       : bool  EB dispersion shrinkage (default True)
    use_lrt           : bool  exact betas via LRT (slow, use for peak loci)
    cis_loci_dict     : dict {gene_id: set} or None
    n_cores           : int
    verbose           : bool

    returns
    -------
    dict {gene_id: ScanResult}
    """
    # stage 1: null model
    fit_null = fit_null_sc(
        counts_cells, strain_ids_cells, covariates_cells,
        norm_method=norm_method, shrink=shrink_disp, verbose=verbose)

    gene_mask = fit_null['gene_mask']
    n_genes_pass = int(gene_mask.sum())
    if n_genes_pass == 0:
        logger.warning('scan_nb: no genes passed glm_sc_fit filter')
        return {}

    gene_ids_all  = (np.asarray(gene_ids) if gene_ids is not None
                     else np.arange(counts_cells.shape[0]).astype(str))
    gene_ids_pass = gene_ids_all[gene_mask]
    disp_pass     = fit_null['dispersion_use']              # (n_genes_pass,)
    sigma_pass    = fit_null['sigma_sample']                # (n_genes_pass,)

    # pseudobulk aggregation
    pb_counts, pb_ncells, strains = pseudobulk_sum(
        counts_cells, strain_ids_cells, gene_mask)           # (n_genes_pass, n_strains)

    # strain-level covariates
    if covariates_strains is None and covariates_cells is not None:
        # average cell-level covariates per strain
        sid_arr = np.asarray(strain_ids_cells)
        _, inv  = np.unique(sid_arr, return_inverse=True)
        n_s     = len(strains)
        cov_sum = np.zeros((n_s, covariates_cells.shape[1]))
        cnt_s   = np.zeros(n_s)
        np.add.at(cov_sum, inv, covariates_cells)
        np.add.at(cnt_s, inv, 1)
        covariates_strains = cov_sum / cnt_s[:, None]

    # stage 2: per-gene scan
    def _one(gi):
        gid  = gene_ids_pass[gi]
        cis  = cis_loci_dict.get(gid) if cis_loci_dict else None
        return gid, scan_nb_gene(
            pb_counts[gi], genome_cache, strains,
            phi=disp_pass[gi], sigma_sample=float(sigma_pass[gi]),
            covariates=covariates_strains, lib_sizes=lib_sizes,
            use_lrt=use_lrt, just_these_loci=cis,
            phenotype_id=gid)

    results = {}
    if n_cores == 1:
        for gi in range(n_genes_pass):
            gid, res = _one(gi)
            results[gid] = res
            if verbose and gi % max(1, n_genes_pass // 10) == 0:
                logger.info(f'scan_nb: {gi+1}/{n_genes_pass} genes')
    else:
        from multiprocessing import Pool
        with Pool(n_cores) as pool:
            for gid, res in pool.imap_unordered(_one, range(n_genes_pass)):
                results[gid] = res

    return results


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fit_nb_ep(dgelist, design, phi, ep):
    """safe glm_fit wrapper returning (mu_hat, coefs, fit)."""
    try:
        fit   = ep.glm_fit(dgelist, design=design, dispersion=phi)
        mu    = np.atleast_1d(
            fit.get('fitted.values', np.full(design.shape[0], 0.5))).ravel()
        coefs = np.atleast_1d(
            fit.get('coefficients', np.zeros(design.shape[1]))).ravel()
        return mu, coefs, fit
    except Exception as e:
        logger.debug(f'glm_fit failed: {e}')
        n = design.shape[0]
        return np.full(n, float(np.mean(
            dgelist.get('counts', np.ones((1, n)))))), \
               np.zeros(design.shape[1]), {}


# ─────────────────────────────────────────────────────────────────────────────
# legacy exports (kept for vqtl.py compatibility)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_gene_dispersion(counts, design, lib_sizes=None):
    """estimate per-gene overdispersion via edgepython EB pipeline.
    prefer fit_null_sc() for cell-level data.
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
    """fit pseudobulk NB null model (no haplotype) for one gene."""
    import edgepython as ep
    y = ep.make_dgelist(counts=counts_gene.reshape(1, -1))
    if lib_sizes is not None:
        y['samples']['lib.size'] = lib_sizes
    y = ep.calc_norm_factors(y)
    return ep.glm_fit(y, design=design, dispersion=dispersion)


def score_test_at_locus(null_fit, h_locus, counts_gene, dispersion):
    """NB score test at a single locus (convenience wrapper)."""
    mu  = np.atleast_1d(null_fit.get(
        'fitted.values', np.full(len(counts_gene), 0.5))).ravel()
    mu  = np.maximum(mu, 1e-300)
    q   = h_locus.shape[1]
    den = 1.0 + dispersion * mu
    r   = (counts_gene - mu) / den
    w   = mu / den
    u   = h_locus.T @ r
    hw  = h_locus * w[:, None]
    info = hw.T @ h_locus
    try:
        sc = max(float(u @ np.linalg.solve(info, u)), 0.0)
    except np.linalg.LinAlgError:
        sc = 0.0
    return {'score_stat': sc,
            'p_value': float(1.0 - _stats.chi2.cdf(sc, df=q)),
            'df': q}

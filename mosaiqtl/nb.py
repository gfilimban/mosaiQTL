"""
mosaiqtl.nb — negative binomial models via edgepython

single wrapper around edgepython for all nb-based scans.
the input structure determines which backend is used:

  pseudobulked counts (n_samples,)  →  edgepython glm_fit()
  cell-level counts (n_cells,) + subject_ids  →  edgepython glm_sc_fit()

the haplotype design matrix H_l^{(-J)} is identical in both cases —
only the likelihood function changes.

dependencies:
  edgepython (pachter 2026) — provides:
    - glm_fit / glm_ql_fit: nb glm fitting via irls
    - estimate_disp: common + trended + tagwise dispersion (wleb shrinkage)
    - calc_norm_factors: tmm normalization
    - glm_sc_fit / shrink_sc_disp / glm_sc_test: nebula-ln mixed model
"""

import numpy as np
import logging

logger = logging.getLogger('mosaiqtl')


def estimate_gene_dispersion(counts, design, lib_sizes=None):
    """estimate per-gene overdispersion using edgepython's eb pipeline.

    wraps edgepython's three-stage empirical bayes dispersion estimation:
      1. common phi across all genes
      2. trended phi as function of mean expression
      3. tagwise phi shrunk toward trend via wleb

    parameters
    ----------
    counts : np.ndarray (n_genes, n_samples)
        raw count matrix (genes x samples)
    design : np.ndarray (n_samples, p)
        design matrix (covariates only — no haplotype term)
    lib_sizes : np.ndarray (n_samples,) or None
        library sizes for offset. if none, column sums of counts.

    returns
    -------
    dict with keys:
        'common'  : float — common dispersion
        'trended' : np.ndarray (n_genes,) — trended dispersion
        'tagwise' : np.ndarray (n_genes,) — shrunk per-gene dispersion
    """
    import edgepython as ep

    # build dgelist
    y = ep.make_dgelist(counts=counts)

    # set library sizes
    if lib_sizes is not None:
        y['samples']['lib.size'] = lib_sizes

    # tmm normalization
    y = ep.calc_norm_factors(y)

    # three-stage dispersion estimation
    y = ep.estimate_disp(y, design=design)

    return {
        'common': y.get('common.dispersion', None),
        'trended': y.get('trended.dispersion', None),
        'tagwise': y.get('tagwise.dispersion', None),
        'dgelist': y,
    }


def fit_nb_null(counts_gene, design, dispersion, lib_sizes=None):
    """fit nb glm null model for one gene (no haplotype term).

    parameters
    ----------
    counts_gene : np.ndarray (n_samples,)
        raw counts for one gene
    design : np.ndarray (n_samples, p)
        covariate design matrix (intercept + sex + batch + pcs)
    dispersion : float
        gene-level dispersion (from estimate_gene_dispersion)
    lib_sizes : np.ndarray (n_samples,) or None

    returns
    -------
    dict with null model fit info (coefficients, deviance, etc.)
    """
    import edgepython as ep

    y = ep.make_dgelist(counts=counts_gene.reshape(1, -1))
    if lib_sizes is not None:
        y['samples']['lib.size'] = lib_sizes
    y = ep.calc_norm_factors(y)

    fit = ep.glm_fit(y, design=design, dispersion=dispersion)
    return fit


def score_test_at_locus(null_fit, h_locus, counts_gene, dispersion,
                        lib_sizes=None):
    """nb score test for haplotype effect at one locus.

    evaluates the score (gradient of log-likelihood) under the null
    model at the haplotype design matrix h_locus. does not require
    re-fitting the full model — only matrix products.

    parameters
    ----------
    null_fit : dict
        output of fit_nb_null
    h_locus : np.ndarray (n_samples, j-1)
        founder haplotype probabilities at this locus (drop one)
    counts_gene : np.ndarray (n_samples,)
        raw counts for one gene
    dispersion : float
        gene-level dispersion
    lib_sizes : np.ndarray (n_samples,) or None

    returns
    -------
    dict with keys:
        'score_stat' : float — chi-squared score statistic
        'p_value'    : float — p-value from chi2(j-1) distribution
        'df'         : int — degrees of freedom (j-1)
    """
    from scipy import stats

    n = len(counts_gene)
    q = h_locus.shape[1]  # j-1

    # extract null fitted values (mu_hat under null)
    # edgepython stores fitted values in the fit object
    mu_hat = null_fit.get('fitted.values', None)
    if mu_hat is None:
        # fallback: compute from coefficients
        design = null_fit.get('design', None)
        coefs = null_fit.get('coefficients', None)
        offset = null_fit.get('offset', np.zeros(n))
        if design is not None and coefs is not None:
            eta = design @ coefs.flatten() + offset
            mu_hat = np.exp(eta)
        else:
            logger.warning('cannot extract null fitted values')
            return {'score_stat': 0.0, 'p_value': 1.0, 'df': q}

    mu_hat = mu_hat.flatten()

    # nb variance: var = mu + phi * mu^2
    r = 1.0 / dispersion  # nb size parameter
    var_hat = mu_hat + dispersion * mu_hat ** 2

    # working weights: w_i = mu_i^2 / var_i
    w = mu_hat ** 2 / var_hat

    # pearson residuals under null: (y - mu) / sqrt(var)
    resid = (counts_gene - mu_hat) / np.sqrt(var_hat)

    # score vector: U = H' W^{1/2} resid_pearson
    # but more precisely for nb glm:
    # U_j = sum_i h_ij * (y_i - mu_i) / (1 + dispersion * mu_i)
    working_resid = (counts_gene - mu_hat) / (1 + dispersion * mu_hat)
    u = h_locus.T @ working_resid  # (q,)

    # fisher information for haplotype block:
    # I = H' W H where W = diag(mu_i / (1 + dispersion * mu_i))
    w_nb = mu_hat / (1 + dispersion * mu_hat)
    hw = h_locus * w_nb[:, np.newaxis]
    info = hw.T @ h_locus  # (q, q)

    # score statistic: U' I^{-1} U ~ chi2(q)
    try:
        score_stat = float(u @ np.linalg.solve(info, u))
    except np.linalg.LinAlgError:
        score_stat = 0.0

    score_stat = max(score_stat, 0.0)
    p_value = 1.0 - stats.chi2.cdf(score_stat, df=q)

    return {
        'score_stat': score_stat,
        'p_value': p_value,
        'df': q,
    }


def precompute_sufficient_stats(counts, subject_ids, library_sizes=None):
    """precompute per-subject sufficient statistics for cell-level scans.

    used by the vqtl module for per-subject dispersion estimation,
    and by scan_nb() when operating in cell-level mode.

    parameters
    ----------
    counts : np.ndarray (n_cells,)
        raw umi counts for one gene across all cells
    subject_ids : np.ndarray (n_cells,)
        subject identifier for each cell
    library_sizes : np.ndarray (n_cells,) or None
        total umi per cell

    returns
    -------
    dict with keys:
        'subjects'    : np.ndarray of str — unique subject ids
        'n_cells'     : np.ndarray of int — cells per subject
        'total_count' : np.ndarray of float — S_i
        'mean_count'  : np.ndarray of float — S_i / n_i
        'var_count'   : np.ndarray of float — sample variance of Y_ij
    """
    from scipy.special import gammaln

    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    n_cells = np.zeros(n_subjects, dtype=np.int64)
    total_count = np.zeros(n_subjects, dtype=np.float64)
    mean_count = np.zeros(n_subjects, dtype=np.float64)
    var_count = np.zeros(n_subjects, dtype=np.float64)

    if library_sizes is None:
        library_sizes = np.ones(len(counts), dtype=np.float64)

    # build subject-to-cell index map
    subject_map = {}
    for idx, sid in enumerate(subject_ids):
        if sid not in subject_map:
            subject_map[sid] = []
        subject_map[sid].append(idx)

    for s_idx, sid in enumerate(unique_subjects):
        cell_indices = np.array(subject_map[sid])
        y_s = counts[cell_indices].astype(np.float64)

        n_cells[s_idx] = len(cell_indices)
        total_count[s_idx] = np.sum(y_s)
        mean_count[s_idx] = np.mean(y_s)

        if len(y_s) > 1:
            var_count[s_idx] = np.var(y_s, ddof=1)
        else:
            var_count[s_idx] = 0.0

    return {
        'subjects': unique_subjects,
        'n_cells': n_cells,
        'total_count': total_count,
        'mean_count': mean_count,
        'var_count': var_count,
    }

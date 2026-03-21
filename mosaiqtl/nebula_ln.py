"""
mosaiqtl.nebula_ln — nebula-ln negative binomial mixed model

implements the nebula-ln approximation from he et al. (2021,
communications biology) for cell-level eqtl mapping in
multi-subject single-cell data.

model:
  Y_ij | u_i ~ NB(μ_ij, φ)           # cell j from subject i
  log(μ_ij) = x_ij' β + u_i + log(L_ij)  # linear predictor
  exp(u_i) ~ Gamma(1/σ², 1/σ²)       # subject random effect

where:
  φ  = cell-level overdispersion (NB parameter)
  σ² = subject-level overdispersion (random effect variance)

the ln approximation leverages large cells-per-subject (cps):
when cps is large, the marginal likelihood per subject can be
analytically approximated without laplace or quadrature methods.

key sufficient statistics per subject (precomputed once):
  S_i = Σ_j Y_ij                    (total count)
  n_i = number of cells              (cell count)
  L_i = Σ_j L_ij                    (total library size)
  T_i = Σ_j log(Γ(Y_ij + 1/φ))     (log-gamma sum)

note: genotype is a subject-level predictor in our eqtl context,
so the nebula recommendation is κ=800 for accurate σ² estimation.
"""

import numpy as np
from scipy.special import gammaln, digamma, polygamma
from scipy.optimize import minimize
import logging

logger = logging.getLogger('mosaiqtl')


class NebulaResult:
    """container for nebula-ln fit results.

    attributes
    ----------
    beta : np.ndarray
        estimated fixed effect coefficients (log-scale)
    se : np.ndarray
        standard errors of beta
    p_values : np.ndarray
        wald p-values for each coefficient
    phi : float
        estimated cell-level overdispersion
    sigma2 : float
        estimated subject-level overdispersion
    converged : bool
        whether optimization converged
    loglik : float
        marginal log-likelihood at optimum
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def precompute_sufficient_stats(counts, subject_ids, library_sizes=None):
    """precompute per-subject sufficient statistics for nebula-ln.

    this is the expensive step — done once per gene, then reused
    across all loci in the genome scan.

    parameters
    ----------
    counts : np.ndarray (n_cells,)
        raw umi counts for one gene across all cells
    subject_ids : np.ndarray (n_cells,)
        subject identifier for each cell
    library_sizes : np.ndarray (n_cells,) or None
        total umi per cell (if none, computed as sum of counts
        but this should be the full-library size, not per-gene)

    returns
    -------
    dict with keys:
        'subjects'    : np.ndarray of str — unique subject ids
        'n_cells'     : np.ndarray of int — cells per subject
        'total_count' : np.ndarray of float — S_i = Σ_j Y_ij
        'total_lib'   : np.ndarray of float — L_i = Σ_j L_ij
        'mean_count'  : np.ndarray of float — S_i / n_i
        'var_count'   : np.ndarray of float — sample variance of Y_ij
        'log_gamma_sum': np.ndarray of float — Σ_j log(Y_ij!)
    """
    # identify unique subjects and their cell indices
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    # preallocate output arrays
    n_cells = np.zeros(n_subjects, dtype=np.int64)
    total_count = np.zeros(n_subjects, dtype=np.float64)
    total_lib = np.zeros(n_subjects, dtype=np.float64)
    mean_count = np.zeros(n_subjects, dtype=np.float64)
    var_count = np.zeros(n_subjects, dtype=np.float64)
    log_gamma_sum = np.zeros(n_subjects, dtype=np.float64)

    # default library sizes to 1 if not provided
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
        l_s = library_sizes[cell_indices].astype(np.float64)

        n_cells[s_idx] = len(cell_indices)
        total_count[s_idx] = np.sum(y_s)
        total_lib[s_idx] = np.sum(l_s)
        mean_count[s_idx] = np.mean(y_s)

        # sample variance (unbiased)
        if len(y_s) > 1:
            var_count[s_idx] = np.var(y_s, ddof=1)
        else:
            var_count[s_idx] = 0.0

        # log-factorial sum: Σ log(y!) = Σ gammaln(y+1)
        log_gamma_sum[s_idx] = np.sum(gammaln(y_s + 1))

    return {
        'subjects': unique_subjects,
        'n_cells': n_cells,
        'total_count': total_count,
        'total_lib': total_lib,
        'mean_count': mean_count,
        'var_count': var_count,
        'log_gamma_sum': log_gamma_sum,
    }


def estimate_dispersion_moments(suff_stats):
    """moment-based estimator for cell-level overdispersion φ.

    uses the relationship Var(Y) = μ + φ μ² to estimate φ from
    the per-subject sample means and variances.

    this is a quick initial estimate — nebula-ln refines it via ml.

    parameters
    ----------
    suff_stats : dict
        output of precompute_sufficient_stats

    returns
    -------
    phi_hat : float
        estimated cell-level overdispersion (clamped to [1e-4, 100])
    phi_per_subject : np.ndarray
        per-subject moment estimates of φ
    """
    mu = suff_stats['mean_count']
    var = suff_stats['var_count']
    n = suff_stats['n_cells']

    # only use subjects with enough cells and nonzero mean
    valid = (n >= 5) & (mu > 0.01)

    if valid.sum() < 3:
        logger.warning('too few valid subjects for moment estimation')
        return 1.0, np.full(len(mu), 1.0)

    # φ = (Var - μ) / μ² for each subject
    phi_per_subject = np.full(len(mu), np.nan)
    phi_per_subject[valid] = (var[valid] - mu[valid]) / (mu[valid] ** 2)

    # take median of valid estimates (robust to outliers)
    phi_valid = phi_per_subject[valid]
    phi_hat = float(np.nanmedian(phi_valid))

    # clamp to reasonable range
    phi_hat = np.clip(phi_hat, 1e-4, 100.0)

    return phi_hat, phi_per_subject


def estimate_sigma2_moments(suff_stats, phi):
    """moment-based estimator for subject-level overdispersion σ².

    the between-subject variance of the mean counts, after
    removing the expected within-subject component, gives σ².

    parameters
    ----------
    suff_stats : dict
        output of precompute_sufficient_stats
    phi : float
        cell-level overdispersion estimate

    returns
    -------
    sigma2_hat : float
        estimated subject-level overdispersion (clamped to [1e-6, 10])
    """
    mu = suff_stats['mean_count']
    n = suff_stats['n_cells']

    valid = (n >= 5) & (mu > 0.01)
    if valid.sum() < 3:
        return 0.1

    # log-scale: log(mean_count) should have variance ≈ σ² + 1/(n·μ) + φ/n
    log_mu = np.log(mu[valid])
    total_var = np.var(log_mu, ddof=1)

    # expected within-subject variance on log scale
    within_var = np.mean(1.0 / (n[valid] * mu[valid]) + phi / n[valid])

    sigma2_hat = max(total_var - within_var, 1e-6)
    sigma2_hat = np.clip(sigma2_hat, 1e-6, 10.0)

    return float(sigma2_hat)


def nebula_ln_loglik(params, suff_stats, x_design, offset=None):
    """approximate marginal log-likelihood for nebula-ln.

    implements the ln (large-number) approximation to the
    nb-gamma marginal likelihood.

    parameters
    ----------
    params : np.ndarray
        [log(φ), log(σ²), β_0, β_1, ..., β_p]
        overdispersions on log scale, fixed effects on natural scale
    suff_stats : dict
        precomputed sufficient statistics
    x_design : np.ndarray (n_subjects, p)
        subject-level design matrix (haplotype + covariates)
    offset : np.ndarray (n_subjects,) or None
        log(mean library size per subject) — if none, computed from suff_stats

    returns
    -------
    float
        negative log-likelihood (for minimization)
    """
    n_subjects = len(suff_stats['subjects'])
    p = x_design.shape[1]

    # unpack parameters
    log_phi = params[0]
    log_sigma2 = params[1]
    beta = params[2:2 + p]

    # transform to natural scale
    phi = np.exp(log_phi)        # cell-level overdispersion
    sigma2 = np.exp(log_sigma2)  # subject-level overdispersion

    # clamp for numerical stability
    phi = np.clip(phi, 1e-6, 500.0)
    sigma2 = np.clip(sigma2, 1e-8, 50.0)

    # linear predictor (subject level)
    eta = x_design @ beta  # (n_subjects,)

    # offset: log of mean library size per cell per subject
    if offset is None:
        # use mean library size per cell
        offset = np.log(
            suff_stats['total_lib'] / suff_stats['n_cells'] + 1e-10
        )

    # expected count per cell: μ_i = exp(η_i + offset_i)
    log_mu = eta + offset
    mu = np.exp(log_mu)

    # sufficient statistics
    s_i = suff_stats['total_count']  # total count per subject
    n_i = suff_stats['n_cells']      # cells per subject

    # nb parameters
    r = 1.0 / phi  # nb 'size' parameter (r = 1/φ)

    # --- ln approximation to marginal likelihood ---
    # for each subject, the marginal likelihood integrates out u_i.
    # when n_i is large, the sum of nb(μ_i exp(u_i), r) over cells
    # can be approximated via cumulant expansion.

    # gamma mixing: exp(u_i) ~ Gamma(1/σ², 1/σ²)
    # shape = rate = 1/σ² = α
    alpha = 1.0 / sigma2

    # for the ln approximation, the key result is:
    # log P(Y_i | β, φ, σ²) ≈
    #   Σ_j [gammaln(Y_ij + r) - gammaln(r) - gammaln(Y_ij + 1)]
    #   + n_i r log(r)
    #   - (n_i r + S_i) log(n_i r + S_i)  [normalizing term]
    #   + gammaln(alpha + S_i) - gammaln(alpha)
    #   + alpha log(alpha)
    #   - (alpha + S_i) log(alpha + n_i μ_i)

    ll = 0.0

    for i in range(n_subjects):
        if n_i[i] == 0:
            continue

        mu_i = mu[i]   # expected count per cell
        s = s_i[i]     # total count
        ni = n_i[i]    # number of cells

        # nb normalizing constants (cell-level)
        # Σ_j [gammaln(Y_ij + r) - gammaln(r)] is hard without cell data
        # use the nb-gamma conjugacy result instead:

        # under nb-gamma conjugacy, the marginal is negative binomial
        # with modified parameters. for the ln approximation:

        # term 1: cell-level nb contribution (approximated)
        # use the sufficient statistic approximation
        ll_i = -suff_stats['log_gamma_sum'][i]  # -Σ log(Y_ij!)

        # term 2: nb rate terms
        # each cell contributes: r log(r/(r + μ_i)) + Y_ij log(μ_i/(r + μ_i))
        log_p = np.log(mu_i / (r + mu_i) + 1e-300)  # log(prob of success)
        log_1mp = np.log(r / (r + mu_i) + 1e-300)    # log(1 - p)

        ll_i += s * log_p + ni * r * log_1mp

        # term 3: nb normalizing (gammaln terms)
        # Σ_j gammaln(Y_ij + r) - n_i gammaln(r)
        # approximate as: gammaln(s + ni*r) - ni*gammaln(r) + correction
        # (this is the ln approximation — valid when n_i is large)
        ll_i += gammaln(s + ni * r) - ni * gammaln(r)

        # term 4: gamma mixing (subject-level random effect)
        # marginal over u_i: Gamma(α, α) mixing on the rate
        ll_i += gammaln(alpha + s) - gammaln(alpha)
        ll_i += alpha * np.log(alpha + 1e-300)
        ll_i -= (alpha + s) * np.log(alpha + ni * mu_i + 1e-300)

        ll += ll_i

    return -ll  # return negative for minimization


def fit_nebula_ln(suff_stats, x_design, offset=None,
                  phi_init=None, sigma2_init=None, beta_init=None,
                  max_iter=200, tol=1e-6):
    """fit nebula-ln model to single-cell count data for one gene.

    parameters
    ----------
    suff_stats : dict
        output of precompute_sufficient_stats
    x_design : np.ndarray (n_subjects, p)
        subject-level design matrix
    offset : np.ndarray (n_subjects,) or None
        log library size offset
    phi_init : float or None
        initial cell-level overdispersion
    sigma2_init : float or None
        initial subject-level overdispersion
    beta_init : np.ndarray or None
        initial fixed effects
    max_iter : int
        maximum optimization iterations
    tol : float
        convergence tolerance

    returns
    -------
    NebulaResult
    """
    n_subjects = len(suff_stats['subjects'])
    p = x_design.shape[1]

    # initialize parameters via moment estimators if not provided
    if phi_init is None:
        phi_init, _ = estimate_dispersion_moments(suff_stats)
    if sigma2_init is None:
        sigma2_init = estimate_sigma2_moments(suff_stats, phi_init)
    if beta_init is None:
        # initialize intercept from mean log count
        mean_count = np.mean(suff_stats['mean_count'][
            suff_stats['mean_count'] > 0
        ])
        beta_init = np.zeros(p)
        if offset is not None:
            beta_init[0] = np.log(max(mean_count, 0.01)) - np.mean(offset)
        else:
            beta_init[0] = np.log(max(mean_count, 0.01))

    # pack initial parameters: [log(φ), log(σ²), β]
    params_init = np.concatenate([
        [np.log(phi_init), np.log(sigma2_init)],
        beta_init
    ])

    # optimize via l-bfgs-b
    result = minimize(
        nebula_ln_loglik,
        params_init,
        args=(suff_stats, x_design, offset),
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-5},
    )

    # unpack results
    params_opt = result.x
    phi_hat = np.exp(params_opt[0])
    sigma2_hat = np.exp(params_opt[1])
    beta_hat = params_opt[2:2 + p]

    # estimate standard errors via numerical hessian
    try:
        from scipy.optimize import approx_fprime

        # compute hessian numerically
        eps = 1e-5
        n_params = len(params_opt)
        hessian = np.zeros((n_params, n_params))
        f0 = nebula_ln_loglik(params_opt, suff_stats, x_design, offset)

        for i in range(n_params):
            ei = np.zeros(n_params)
            ei[i] = eps
            for j in range(i, n_params):
                ej = np.zeros(n_params)
                ej[j] = eps
                fpp = nebula_ln_loglik(
                    params_opt + ei + ej, suff_stats, x_design, offset
                )
                fpm = nebula_ln_loglik(
                    params_opt + ei - ej, suff_stats, x_design, offset
                )
                fmp = nebula_ln_loglik(
                    params_opt - ei + ej, suff_stats, x_design, offset
                )
                fmm = nebula_ln_loglik(
                    params_opt - ei - ej, suff_stats, x_design, offset
                )
                hessian[i, j] = (fpp - fpm - fmp + fmm) / (4 * eps * eps)
                hessian[j, i] = hessian[i, j]

        # covariance = inverse hessian
        cov = np.linalg.inv(hessian)
        se_all = np.sqrt(np.maximum(np.diag(cov), 0))
        se_beta = se_all[2:2 + p]
    except Exception:
        se_beta = np.full(p, np.nan)
        logger.warning('hessian computation failed; se set to nan')

    # wald p-values for fixed effects
    z_scores = beta_hat / np.maximum(se_beta, 1e-10)
    p_values = 2 * (1 - stats.norm.cdf(np.abs(z_scores)))

    from scipy import stats

    return NebulaResult(
        beta=beta_hat,
        se=se_beta,
        p_values=p_values,
        phi=phi_hat,
        sigma2=sigma2_hat,
        converged=result.success,
        loglik=-result.fun,
        n_iter=result.nit if hasattr(result, 'nit') else -1,
    )

"""
nb — negative binomial genome-wide scan (primary model)

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
  NB-LMM with LOCO kinship covariance implemented via PQL + Woodbury identity.
  stage 1 (fit_null_sc) dispatches to native NEBULA-LN by default (use_native=True)
  or edgepython (use_native=False, requires edgepython installed).

references: He et al. 2021 (NEBULA-LN); Pachter 2026 edgePython (design ref);
            Chen et al. 2016 AJHG (PQL score test); Robinson et al. 2010 (TMM)
"""

import numpy as np
import logging
from scipy.special import gammaln as _gammaln, digamma as _dg, polygamma as _pg
from scipy import stats as _stats
from scipy.optimize import minimize_scalar as _minimize_scalar
from scipy.linalg import cho_factor as _cho_fac, cho_solve as _cho_sv

logger = logging.getLogger('mosaiqtl')


# ─────────────────────────────────────────────────────────────────────────────
# native NEBULA-LN implementation (no edgepython needed)
# design follows He et al. 2021 (NEBULA-LN); Pachter 2026 edgePython (reference).
# ─────────────────────────────────────────────────────────────────────────────


def _tmm_offsets(counts_mat, lib_sizes=None):
    """Log library size offsets (simple: log total counts per sample).
    Full TMM (Robinson et al. 2010) is a planned refinement; for QTL mapping
    the simpler version introduces negligible bias.
    counts_mat : (n_genes, n_samples)
    returns    : (n_samples,) log offsets
    """
    ls = counts_mat.sum(axis=0).astype(float)  # (n_samples,)
    ls = np.maximum(ls, 1.0)
    return np.log(ls)


def _irls_nb_pseudobulk(S, x, phi_size, offsets, max_iter=25, tol=1e-7):
    """IRLS for NB GLM on pseudobulk counts.
    phi_size  : NB size parameter r (= 1/sigma2 for subject-level NB)
    returns mu (n,), beta (p,)
    """
    eta = offsets.copy()                       # init: mu = exp(offset)
    mu  = np.exp(eta)
    for _ in range(max_iter):
        W  = mu / (1.0 + mu / phi_size)        # NB working weight
        z  = eta + (S - mu) / mu               # working response
        Wx = x * W[:, None]
        try:
            beta = np.linalg.solve(Wx.T @ x, Wx.T @ (z - offsets))
        except np.linalg.LinAlgError:
            break
        eta_new = x @ beta + offsets
        mu_new  = np.exp(eta_new)
        if np.max(np.abs(mu_new - mu) / (np.abs(mu) + 1e-8)) < tol:
            mu = mu_new; break
        mu = mu_new; eta = eta_new
    return mu, beta


def _nr_k(S, M, k_init=1.0, max_iter=25, tol=1e-6):
    """Newton-Raphson for k = 1/sigma2 in NB pseudobulk marginal likelihood.
    Marginal: S_i ~ NB(M_i, 1/k).  Gradient + Hessian use digamma/polygamma.
    """
    k = max(float(k_init), 1e-4)
    S = np.asarray(S, float); M = np.asarray(M, float)
    for _ in range(max_iter):
        g = float(np.sum(_dg(k+S) - _dg(k) + np.log(k/(k+M)) + (M-S)/(k+M)))
        h = float(np.sum(_pg(1,k+S) - _pg(1,k) + 1/k - 1/(k+M) - (M-S)/(k+M)**2))
        if abs(h) < 1e-12:
            break
        k_new = k - g / h
        if k_new <= 0:
            k_new = k * 0.5
        if abs(k_new - k) / (abs(k) + 1e-8) < tol:
            k = k_new; break
        k = k_new
    return max(k, 1e-6)


def _phi_mom(counts_g, mu_cell):
    """Method-of-moments cell-level NB dispersion estimate.
    counts_g : (n_cells,)  raw UMI counts
    mu_cell  : (n_cells,)  expected counts per cell
    returns phi (cell-level dispersion = 1/r_cell)
    """
    r  = (counts_g - mu_cell) ** 2
    phi = np.mean((r - mu_cell) / (mu_cell ** 2 + 1e-8))
    return max(float(phi), 1e-6)


def _wleb_shrink(phi_hat, mean_expr, df_prior=10.0):
    """Weighted likelihood empirical Bayes dispersion shrinkage.
    Shrinks per-gene phi toward a trended prior as a function of mean expression.
    Follows Robinson & Smyth 2007 (tagwise dispersion in edgeR).
    phi_hat   : (n_genes,)
    mean_expr : (n_genes,)  mean pseudobulk count per gene
    returns   : (n_genes,) shrunk phi
    """
    ok = np.isfinite(phi_hat) & (phi_hat > 0) & (mean_expr > 0)
    if ok.sum() < 5:
        return phi_hat.copy()
    # trended prior: lowess on log(phi) ~ log(mean)
    log_m = np.log(mean_expr[ok] + 1e-8)
    # simple: smooth via binned mean (robust to outliers)
    n_bins = min(10, ok.sum() // 5)
    bins   = np.percentile(log_m, np.linspace(0, 100, n_bins+1))
    prior  = np.full(len(phi_hat), np.nanmedian(phi_hat[ok]))
    for b in range(n_bins):
        mask_b = ok & (log_m >= bins[b]) & (log_m <= bins[b+1])
        if mask_b.sum() >= 3:
            bin_idx = np.where(ok)[0]
            subset  = bin_idx[(log_m[bin_idx] >= bins[b]) & (log_m[bin_idx] <= bins[b+1])]
            prior_b = float(np.median(phi_hat[subset]))
            # assign prior to all genes in this expression range
            log_expr  = np.log(mean_expr + 1e-8)
            full_mask = (log_expr >= bins[b]) & (log_expr <= bins[b+1])
            prior[full_mask] = prior_b
    # posterior: (df_prior * prior + df_gene * phi) / (df_prior + df_gene)
    # df_gene = 1 (single per-gene estimate), df_prior as specified
    shrunk = (df_prior * prior + phi_hat) / (df_prior + 1.0)
    shrunk[~ok] = phi_hat[~ok]
    return np.maximum(shrunk, 1e-6)


def fit_null_sc_native(counts_cells, sample_ids, covariates_cells=None,
                       verbose=False):
    """Native NEBULA-LN Stage 1 null fit — no edgepython required.

    implements the NB-gamma mixed model from He et al. 2021, using the LN
    approximation for large CPS. design follows Pachter 2026 edgePython.

    parameters
    ----------
    counts_cells      : (n_genes, n_cells)  raw UMI counts
    sample_ids        : (n_cells,)          sample/subject per cell
    covariates_cells  : (n_cells, p) or None

    returns dict compatible with fit_null_sc output:
        dispersion_use : (n_genes_pass,)  EB-shrunk cell-level phi
        sigma_sample   : (n_genes_pass,)  subject-level overdispersion sigma2
        gene_mask      : (n_genes,) bool
        fitted_values  : (n_genes_pass, n_cells) optional — None here
    """
    counts_cells = np.asarray(counts_cells, dtype=np.float64)
    n_genes, n_cells = counts_cells.shape
    sample_ids   = np.asarray(sample_ids)
    samples, inv = np.unique(sample_ids, return_inverse=True)
    n_samp       = len(samples)

    # subject-level offsets and pseudobulk
    lib_cell = counts_cells.sum(axis=0) + 1.0                # (n_cells,)
    lib_samp = np.zeros(n_samp)
    ncell_s  = np.zeros(n_samp)
    np.add.at(lib_samp, inv, lib_cell)
    np.add.at(ncell_s,  inv, 1.0)
    offsets_samp = np.log(lib_samp.clip(1))                  # (n_samp,)

    # subject-level design matrix
    if covariates_cells is not None:
        cov_c = np.asarray(covariates_cells, float)
        cov_s = np.zeros((n_samp, cov_c.shape[1]))
        np.add.at(cov_s, inv, cov_c)
        cov_s /= ncell_s[:, None].clip(1)
        if not np.allclose(cov_s[:, 0], 1.0):
            x_samp = np.hstack([np.ones((n_samp, 1)), cov_s])
        else:
            x_samp = cov_s
    else:
        x_samp = np.ones((n_samp, 1))

    # gene-level filter: cpc > 0.005 and expressed in >= 5 cells
    mean_cpc = counts_cells.mean(axis=1)
    n_expr   = (counts_cells > 0).sum(axis=1)
    gene_mask = (mean_cpc > 0.005) & (n_expr >= 5)
    idx       = np.where(gene_mask)[0]
    n_pass    = len(idx)
    if verbose:
        logger.info(f'fit_null_sc_native: {n_pass}/{n_genes} genes pass filter')

    phi_hat   = np.full(n_pass, 0.1)
    sigma2    = np.full(n_pass, 0.1)
    mean_pb   = np.zeros(n_pass)

    for gi, g in enumerate(idx):
        counts_g = counts_cells[g]
        # pseudobulk sum per subject
        S = np.zeros(n_samp)
        np.add.at(S, inv, counts_g)

        # initial phi: MOM on pseudobulk residuals
        pb_mean    = S.mean()
        pb_var     = S.var()
        phi_pb_init = max((pb_var - pb_mean) / (pb_mean**2 + 1e-8), 1e-3)

        # NEBULA-LN: alternate beta (IRLS) and k (NR)
        k = 1.0 / max(phi_pb_init, 1e-3)
        for _ in range(8):
            mu_s, _  = _irls_nb_pseudobulk(S, x_samp, k, offsets_samp)
            k        = _nr_k(S, mu_s, k_init=k)

        sigma2[gi]  = 1.0 / max(k, 1e-6)
        mean_pb[gi] = float(mu_s.mean())

        # cell-level phi: MOM after removing subject scale factor
        v_hat   = (S / mu_s.clip(1e-8))[inv]                 # (n_cells,)
        mu_cell = v_hat * (mu_s / ncell_s.clip(1))[inv]
        phi_hat[gi] = _phi_mom(counts_g, mu_cell)

        if verbose and gi % max(1, n_pass//10) == 0:
            logger.info(f'  {gi+1}/{n_pass}  phi={phi_hat[gi]:.3f}  sigma2={sigma2[gi]:.3f}')

    # EB shrinkage
    phi_shrunk = _wleb_shrink(phi_hat, mean_pb)

    return {
        'dispersion':      phi_hat,
        'dispersion_shrunk': phi_shrunk,
        'dispersion_use':  phi_shrunk,
        'sigma_sample':    sigma2,
        'gene_mask':       gene_mask,
        'fitted_values':   None,
    }

# ─────────────────────────────────────────────────────────────────────────────
# stage 1 helpers
# ─────────────────────────────────────────────────────────────────────────────


def fit_null_sc(counts_cells, strain_ids, covariates_cells=None,
                norm_method='TMM', shrink=True, verbose=False,
                use_native=True):
    """fit NEBULA-LN null model on cell-level counts for all genes.

    use_native=True  — native numpy/scipy implementation (recommended, no edgepython).
    use_native=False — legacy edgepython path (requires edgepython installed).
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
    if use_native:
        return fit_null_sc_native(
            counts_cells, strain_ids, covariates_cells, verbose=verbose)

    # ── legacy edgepython path ────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# woodbury helpers — Woodbury identity for V = diag(1/W) + tau * Z K_m Z^T
# never form K_n = Z K_m Z^T explicitly; all operations stay at (m×m) cost.
# ─────────────────────────────────────────────────────────────────────────────


def _wb(v, W, Z, M_inv):
    """V^{-1} v via Woodbury.  V = diag(1/W) + tau*Z K Z^T.
    V^{-1}v = W*v - W*(Z M^{-1} Z^T W v)   (M = K^{-1}/tau + Z^T diag(W) Z)
    """
    Dv = W * v                                    # D^{-1}v = W*v
    return Dv - W * (Z @ (M_inv @ (Z.T @ Dv)))   # (n,)


def _wb_mat(A, W, Z, M_inv):
    """V^{-1} A via Woodbury.  A: (n, k)."""
    DA = W[:, None] * A
    return DA - W[:, None] * (Z @ (M_inv @ (Z.T @ DA)))   # (n, k)


def _wb_batch(H_all, W, Z, M_inv):
    """V^{-1} H_all via Woodbury.  H_all: (L, n, J-1) — all loci at once.
    returns (L, n, J-1)."""
    DH   = W[None, :, None] * H_all                                    # (L, n, J-1)
    ZtDH = np.einsum('mn,lnj->lmj', Z.T, DH, optimize=True)            # (L, m, J-1)
    MiZtDH = np.einsum('mk,lkj->lmj', M_inv, ZtDH, optimize=True)     # (L, m, J-1)
    return DH - W[None, :, None] * np.einsum('nm,lmj->lnj', Z, MiZtDH, optimize=True)


def _build_M(K_m, tau, W, Z):
    """Build M = K^{-1}/tau + Z^T diag(W) Z  (m×m) for Woodbury."""
    return np.linalg.inv(K_m) / tau + Z.T @ (W[:, None] * Z)


def _pql_tau_reml_wb(z, x, K_m, Z, W, tau_init=0.1):
    """REML for tau via Woodbury identity (replaces _pql_tau_reml).

    working LMM: z ~ N(X*beta, D + tau*Z K_m Z^T),  D = diag(1/W)
    log|V| via matrix det lemma: log|M| + m*log(tau) + log|K_m| + log|D|
    all linear algebra stays at m×m cost (never form n×n matrices).
    """
    m         = K_m.shape[0]
    K_inv     = np.linalg.inv(K_m)
    log_det_K = np.linalg.slogdet(K_m)[1]
    log_det_D = -np.sum(np.log(W))                   # log|D| = -sum log W
    ZtWZ      = Z.T @ (W[:, None] * Z)               # (m, m), precomputed

    def _neg_reml(log_tau):
        tau = np.exp(log_tau)
        M   = K_inv / tau + ZtWZ                     # (m, m)
        try:
            L_cho, low = _cho_fac(M)
        except np.linalg.LinAlgError:
            return 1e10
        M_inv     = _cho_sv((L_cho, low), np.eye(m))
        log_det_M = 2.0 * np.sum(np.log(np.diag(L_cho)))
        log_det_V = log_det_M + m*np.log(tau) + log_det_K + log_det_D

        Vinvz  = _wb(z, W, Z, M_inv)
        VinvX  = _wb_mat(x, W, Z, M_inv)
        A      = x.T @ VinvX                         # (p, p)
        sign, log_det_A = np.linalg.slogdet(A)
        if sign <= 0: return 1e10
        Pz  = Vinvz - VinvX @ np.linalg.solve(A, x.T @ Vinvz)
        return 0.5 * (log_det_V + log_det_A + float(z @ Pz))

    log_t0 = float(np.log(max(tau_init, 1e-6)))
    res    = _minimize_scalar(_neg_reml, bounds=(log_t0-6, log_t0+6), method='bounded')
    return float(max(np.exp(res.x), 1e-8))


def _pql_tau_reml(z, x, k_mat, w_vec, tau_init=0.1):
    """REML estimate of kinship variance component tau in the PQL working LMM.

    working model: z ~ N(X*beta, diag(1/w) + tau*K)
    with n=48 strains a direct Cholesky is cheap — no eigendecomp trick needed.

    parameters
    ----------
    z       : (n,)    working response
    x       : (n, p)  covariate matrix
    k_mat   : (n, n)  LOCO kinship matrix for current chromosome
    w_vec   : (n,)    NB working weights W_ii = mu_i/(1 + phi*mu_i)
    tau_init: float   initial value for warm start

    returns
    -------
    tau : float  REML estimate, clipped to [1e-8, inf)
    """
    def _neg_reml(log_tau):
        tau = np.exp(log_tau)
        V   = np.diag(1.0 / w_vec) + tau * k_mat
        try:
            chol = np.linalg.cholesky(V)          # raises if not PD
        except np.linalg.LinAlgError:
            return 1e10
        V_inv              = np.linalg.inv(V)      # cheap at n<=48
        VX                 = V_inv @ x
        A                  = x.T @ VX              # (p, p)
        sign, logdet_xtVx  = np.linalg.slogdet(A)
        if sign <= 0:
            return 1e10
        P      = V_inv - VX @ np.linalg.solve(A, VX.T)
        logdet = 2.0 * np.sum(np.log(np.diag(chol)))
        return 0.5 * (logdet + logdet_xtVx + float(z @ P @ z))

    log_tau0 = float(np.log(max(tau_init, 1e-6)))
    res = _minimize_scalar(_neg_reml, bounds=(log_tau0 - 6, log_tau0 + 6),
                           method='bounded')
    return float(max(np.exp(res.x), 1e-8))


def fit_null_nb_glmm_loco(S, x, offsets, K_m, Z, phi, max_iter=5, tol=1e-4):
    """PQL null NB-GLMM with LOCO kinship via Woodbury identity.

    model: S_i ~ NB(mu_i, phi),  log mu_i = x_i^T beta + u_i + offset_i
           u   ~ N(0, tau * K_m(sample)),  K_m(sample) = Z K_m Z^T

    Woodbury: V = diag(1/W) + tau*Z K_m Z^T.  never forms the n×n matrix.
    all linear algebra lives at m×m cost (m = n_strains ≤ 48).

    parameters
    ----------
    S     : (n,)    pseudobulk counts per sample
    x     : (n, p)  covariate matrix
    offsets: (n,)   log library size offsets
    K_m   : (m, m)  strain-level LOCO kinship
    Z     : (n, m)  sample-to-strain mapping  (one-hot rows)
    phi   : float   NB dispersion (held fixed from Stage 1)

    returns dict with keys: mu, tau, W, M_inv, Z, VinvX, XtVinvX_inv, r_adj
    (P is never stored — applied implicitly via _wb / _wb_batch)
    """
    phi = max(float(phi), 1e-6)
    mu  = np.maximum(S.astype(np.float64), 0.5)
    tau = 0.1

    for it in range(max_iter):
        W   = mu / (1.0 + phi * mu)
        z   = (np.log(mu) - offsets) + (S - mu) / mu

        # REML for tau via Woodbury (m×m cost)
        tau = _pql_tau_reml_wb(z, x, K_m, Z, W, tau_init=tau)

        # Woodbury working quantities
        M     = _build_M(K_m, tau, W, Z)             # (m, m)
        M_inv = np.linalg.inv(M)

        # GLS for beta
        Vinvz = _wb(z, W, Z, M_inv)
        VinvX = _wb_mat(x, W, Z, M_inv)              # (n, p)
        A     = x.T @ VinvX                           # (p, p)
        beta  = np.linalg.solve(A, VinvX.T @ z)

        # BLUP for u
        resid = z - x @ beta
        Vinvr = _wb(resid, W, Z, M_inv)
        u_hat = tau * (Z @ (K_m @ (Z.T @ Vinvr)))    # u = tau K_n V^{-1} residual

        mu_new = np.exp(x @ beta + u_hat + offsets)
        delta  = float(np.max(np.abs(mu_new - mu) / (np.abs(mu) + 1e-8)))
        mu     = mu_new
        if delta < tol and it > 0:
            break

    # final working quantities for score test
    W     = mu / (1.0 + phi * mu)
    z_fin = (np.log(mu) - offsets) + (S - mu) / mu
    M     = _build_M(K_m, tau, W, Z)
    M_inv = np.linalg.inv(M)
    VinvX = _wb_mat(x, W, Z, M_inv)
    A     = x.T @ VinvX                               # (p, p)
    A_inv = np.linalg.inv(A)
    Vinvz = _wb(z_fin, W, Z, M_inv)
    r_adj = Vinvz - VinvX @ (A_inv @ (x.T @ Vinvz))  # P z  (n,)

    return {'mu': mu, 'tau': float(tau), 'W': W,
            'M_inv': M_inv, 'Z': Z,
            'VinvX': VinvX, 'XtVinvX_inv': A_inv,
            'r_adj': r_adj}


def _score_stats_loco(H_all, r_adj, W, Z, M_inv, VinvX, XtVinvX_inv):
    """Vectorised NB-GLMM score statistics via Woodbury (no P matrix stored).

    I_l = H_l^T P H_l = H_l^T V^{-1} H_l - (H_l^T V^{-1}X)(X^T V^{-1}X)^{-1}(X^T V^{-1}H_l)

    parameters
    ----------
    H_all        : (L, n, J-1)  haplotype design — all loci
    r_adj        : (n,)         P z — kinship-adjusted working residuals
    W            : (n,)         NB working weights from PQL null
    Z            : (n, m)       sample-to-strain Z-matrix
    M_inv        : (m, m)       Woodbury inner matrix inverse
    VinvX        : (n, p)       V^{-1} X (precomputed)
    XtVinvX_inv  : (p, p)       (X^T V^{-1} X)^{-1} (precomputed)

    returns (L,) score statistics ~ chi2(J-1) under H_0
    """
    L, n, nf1 = H_all.shape

    # U_l = H_l^T r_adj  (L, J-1)
    U = np.einsum('lnj,n->lj', H_all, r_adj, optimize=True)

    # V^{-1} H_all via Woodbury  (L, n, J-1)
    VinvH = _wb_batch(H_all, W, Z, M_inv)

    # H_l^T V^{-1} H_l  (L, J-1, J-1)
    HtVinvH = np.einsum('lnj,lnk->ljk', H_all, VinvH, optimize=True)

    # correction: (H_l^T V^{-1}X)(X^T V^{-1}X)^{-1}(X^T V^{-1}H_l)
    HtVinvX = np.einsum('lnj,np->ljp', H_all, VinvX, optimize=True)   # (L, J-1, p)
    corr    = np.einsum('ljp,pk,lqk->ljq', HtVinvX, XtVinvX_inv, HtVinvX, optimize=True)

    info_mat = HtVinvH - corr + np.eye(nf1)[None] * 1e-10   # (L, J-1, J-1)
    sc = np.zeros(L)
    for li in range(L):
        try:
            sc[li] = max(float(U[li] @ np.linalg.solve(info_mat[li], U[li])), 0.0)
        except np.linalg.LinAlgError:
            sc[li] = 0.0
    return sc


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
    for li in range(len(sc)):
        try:
            sc[li] = max(float(u[li] @ np.linalg.solve(info[li], u[li])), 0.0)
        except np.linalg.LinAlgError:
            sc[li] = 0.0
    return sc


def scan_nb_gene(counts_pb, genomecache, strain_order,
                  phi, sigma_sample=0.0,
                  covariates=None, lib_sizes=None,
                  kinship_loco=None,
                  use_lrt=False, just_these_loci=None,
                  phenotype_id='unknown'):
    """NB score-test scan for one gene on pseudobulk counts.

    parameters
    ----------
    counts_pb     : np.ndarray (n_strains,)  pseudobulk sum counts
    genomecache  : GenomeCache
    strain_order  : array-like (n_strains,)  strain IDs in counts_pb order
                    must match genomecache.sample_ids order
    phi           : float  NB dispersion (from fit_null_sc, shrunk)
    sigma_sample  : float  subject-level variance from glm_sc_fit (≥0)
                    inflates score denominator when >0
    covariates    : np.ndarray (n_strains, p) or None
    lib_sizes     : np.ndarray (n_strains,) or None  log-offset per strain
    kinship_loco  : dict {chr: np.ndarray (n_strains, n_strains)} or None
                    LOCO kinship matrices from kinship.calc_kinship(method='loco').
                    when provided, replaces the exchangeable u_i with a structured
                    N(0, tau K^(-c)) random effect via PQL (Chen et al. 2016 AJHG).
                    this is the recommended setting for multiparent populations.
    use_lrt       : bool  fit alt model per locus for exact betas (slow)
    just_these_loci : set or None
    phenotype_id  : str

    returns
    -------
    ScanResult
    """
    import edgepython as ep
    from .scan import ScanResult
    from ..utils import build_z_matrix

    phi  = max(float(phi), 1e-6)
    y    = np.asarray(counts_pb, dtype=np.float64)
    sids = list(strain_order)

    # align to cache order via Z-matrix
    cache_ids = list(genomecache.sample_ids)
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
    offsets = np.log(np.maximum(s, 1e-8))                 # log library size offset

    # ── null model ────────────────────────────────────────────────────────
    # with kinship: PQL NB-GLMM per chromosome (fit_null_nb_glmm_loco)
    # without kinship: standard NB GLM via edgePython (original path)
    use_kinship = kinship_loco is not None and len(kinship_loco) > 0

    if not use_kinship:
        # original path — exchangeable RE via sigma_sample inflation
        dg = ep.make_dgelist(counts=y.reshape(1, -1))
        dg['samples']['lib.size'] = s.astype(float)
        dg = ep.calc_norm_factors(dg)
        mu_null, _, _ = _fit_nb_ep(dg, x, phi, ep)
        denom  = 1.0 + phi * mu_null
        w_inf  = max(1.0 + sigma_sample, 1.0)
        r_vec_global = (y - mu_null) / (denom * w_inf)
        w_vec_global = mu_null / (denom * w_inf)
        null_ll      = _nb_loglik(y, mu_null, phi)

    nf1 = genomecache.n_founders - 1

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)

        if just_these_loci is not None:
            mask = np.isin(lids, list(just_these_loci))
            lids, pos, probs = lids[mask], pos[mask], probs[mask]
        if not len(lids):
            continue

        # Z-expand haplotypes (n_loci, n, J-1)
        H = (np.einsum('nm,lmj->lnj', z_sub, probs[:, :, :nf1], optimize=True)
             if z_sub is not None else probs[:, valid, :nf1])

        # per-chromosome null (LOCO kinship path)
        if use_kinship and str(chrom) in {str(k) for k in kinship_loco}:
            _kkey  = (chrom if chrom in kinship_loco
                      else str(chrom))
            K_c = np.asarray(kinship_loco[_kkey], dtype=np.float64)  # (m, m) strain-level
            # Z_pql: sample→strain mapping for valid samples
            # z_sub is (n_valid, m); when z_sub is None, n==m so Z=I
            Z_pql = z_sub if z_sub is not None else np.eye(K_c.shape[0])
            glmm_fit = fit_null_nb_glmm_loco(y, x, offsets, K_c, Z_pql, phi)
            mu_chr   = glmm_fit['mu']
            null_ll  = _nb_loglik(y, mu_chr, phi)
        else:
            # no kinship for this chromosome — fall back to global null
            r_adj_chr = r_vec_global

        if not use_lrt:
            if use_kinship and str(chrom) in {str(k) for k in kinship_loco}:
                sc = _score_stats_loco(H, glmm_fit['r_adj'],
                                        glmm_fit['W'], glmm_fit['Z'],
                                        glmm_fit['M_inv'], glmm_fit['VinvX'],
                                        glmm_fit['XtVinvX_inv'])
            else:
                sc = _score_stats_chromosome(H, r_adj_chr, w_vec_global)
            lods  = sc / (2.0 * np.log(10))
            pv    = 1.0 - _stats.chi2.cdf(sc, df=nf1)
            betas = np.zeros((len(lids), nf1))
        else:
            lods  = np.zeros(len(lids)); pv = np.ones(len(lids))
            betas = np.zeros((len(lids), nf1))
            dg_l  = ep.make_dgelist(counts=y.reshape(1, -1))
            dg_l['samples']['lib.size'] = s.astype(float)
            dg_l  = ep.calc_norm_factors(dg_l)
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


def scan_nb(counts_cells, genomecache, strain_ids_cells,
             gene_ids=None, covariates_cells=None,
             covariates_strains=None, lib_sizes=None,
             kinship_loco=None,
             norm_method='TMM', shrink_disp=True,
             use_lrt=False, cis_loci_dict=None,
             n_cores=1, verbose=False):
    """genome-wide NB GLM scan — full pipeline for one cell type.

    stage 1: glm_sc_fit + shrink_sc_disp (cell level, once)
    stage 2: score test per gene on pseudobulk counts (vectorised)

    parameters
    ----------
    counts_cells      : np.ndarray (n_genes, n_cells)  raw UMI counts
    genomecache      : GenomeCache
    strain_ids_cells  : array-like (n_cells,)  strain per cell
    gene_ids          : list of str or None  row names for counts_cells
    covariates_cells  : np.ndarray (n_cells, p) or None
    covariates_strains: np.ndarray (n_strains, p) or None
        strain-level covariates for the scan (batch + PCs).
        if None, derived from covariates_cells by strain mean.
    lib_sizes         : np.ndarray (n_strains,) or None
    kinship_loco      : dict {chr: np.ndarray} or None
                        LOCO kinship from kinship.calc_kinship(method='loco').
                        strongly recommended for multiparent populations.
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
            pb_counts[gi], genomecache, strains,
            phi=disp_pass[gi], sigma_sample=float(sigma_pass[gi]),
            covariates=covariates_strains, lib_sizes=lib_sizes,
            kinship_loco=kinship_loco,
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

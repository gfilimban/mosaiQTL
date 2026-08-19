"""
nb — negative binomial genome-wide scan (primary model)

two-stage design:
  stage 1 — null model + dispersion (cell level)
      fit_null_sc_native(counts_cells, sample_ids, ...)
          NEBULA-LN: NB-gamma mixed model with per-subject random effect
          absorbs within-strain cell-to-cell variability and between-strain
          overdispersion simultaneously. sigma_sample = per-gene subject
          variance component; dispersion = cell-level NB overdispersion.
      shrinkage via _wleb_shrink (WLEB framework).

  stage 2 — genome-wide scan (pseudobulk per sample, vectorised)
      for each gene: pseudobulk counts (one per individual sample) + shrunk dispersion
      score test over all loci via einsum — O(L × n × J²) per chromosome.
      sigma_sample optionally inflates score test denominator to account for
      residual within-strain variability not captured by fixed covariates.

model:
  cell level (stage 1):
      Y_ij ~ NB(μ_ij, φ_g),   log μ_ij = X_j β_g + log s_j + u_i
      u_i ~ N(0, σ²_g)         subject random effect per gene g

  pseudobulk scan (stage 2):
      S_ig = Σ_j Y_ij           pseudobulk sum per sample i, gene g
      H₀: log E[S_ig] = X_i β₀ + log(n_i s̄_i)
      H₁: log E[S_ig] = H_iℓ γ + X_i β + log(n_i s̄_i)
      Q_ℓ = U_ℓᵀ I_ℓ⁻¹ U_ℓ ~ χ²(J−1)   score test

  note: stage 2 operates on n individual-sample pseudobulk observations,
  not m strain means. multiple samples sharing a genotype are kept distinct;
  the Z-matrix (n × m) expands the m-row haplotype matrix to n rows.
  when kinship_loco is provided, the per-chromosome null uses a PQL NB-GLMM
  with LOCO kinship random effect via the Woodbury identity.

references: He et al. 2021 (NEBULA-LN); Chen et al. 2016 AJHG (PQL score test)
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor as _ThreadPool
from pathlib import Path

import numpy as np
from scipy.special import gammaln as _gammaln, digamma as _dg, polygamma as _pg
from scipy import stats as _stats
from scipy.optimize import minimize_scalar as _minimize_scalar
from scipy.linalg import cho_factor as _cho_fac, cho_solve as _cho_sv

from .scan import ScanResult
from ..utils import build_z_matrix
from ..genotype.genomecache import GenomeCache

logger = logging.getLogger("mosaiqtl")

__all__ = [
    "fit_null_sc_native",
    "fit_null_sc",
    "pseudobulk_sum",
    "precompute_sufficient_stats",
    "fit_null_nb_glmm_loco",
    "scan_nb_gene",
    "scan_nb_multi_gene",
    "scan_nb",
]


# ─────────────────────────────────────────────────────────────────────────────
# stage 1 — cell-level null model
# ─────────────────────────────────────────────────────────────────────────────


def _tmm_norm_factors(counts_mat, ref_col=None, trim_m=0.3, trim_a=0.05):
    """Per-sample TMM scaling factors via trimmed mean of M-values.

    Reference sample: the sample whose 75th-percentile count is closest
    to the median 75th-percentile across all samples. This minimises the
    average M-value magnitude and therefore the trimming loss.
    Computes log-ratios (M) and average log-expression (A) relative to
    the reference, trims the most extreme M and A fractions, and returns
    weighted-mean M per sample as the log2 scaling factor. Add to
    log(lib_size) to get the full log-offset for each sample.

    counts_mat : (n_genes, n_samples)  raw integer counts
    ref_col    : int or None  reference column index; if None uses the
                 sample whose 75th-percentile count is closest to the
                 dataset median (default qtl2/edgeR heuristic).
    trim_m     : float  fraction of M-values to trim from each tail (default 0.3)
    trim_a     : float  fraction of A-values to trim from each tail (default 0.05)
    returns    : (n_samples,) log-scale TMM normalisation factors
    """
    counts_mat = np.asarray(counts_mat, dtype=np.float64)
    n_genes, n_samples = counts_mat.shape
    lib_sizes = counts_mat.sum(axis=0).clip(1.0)  # (n_samples,)

    # choose reference sample: closest 75th-pctile count to dataset median
    if ref_col is None:
        q75 = np.percentile(counts_mat, 75, axis=0)
        ref_col = int(np.argmin(np.abs(q75 - np.median(q75))))

    ref_counts = counts_mat[:, ref_col]
    ref_lib = lib_sizes[ref_col]

    log_factors = np.zeros(n_samples)
    for k in range(n_samples):
        if k == ref_col:
            continue
        obs = counts_mat[:, k]
        obs_lib = lib_sizes[k]

        # keep genes with counts > 0 in both samples
        keep = (obs > 0) & (ref_counts > 0)
        if keep.sum() < 10:
            continue

        obs_k = obs[keep]
        ref_k = ref_counts[keep]

        # M: log-ratio of relative expression
        M = np.log2(obs_k / obs_lib) - np.log2(ref_k / ref_lib)
        # A: average log-expression
        A = 0.5 * (np.log2(obs_k / obs_lib) + np.log2(ref_k / ref_lib))

        # asymptotic variance for weighting
        w = (obs_lib - obs_k) / (obs_lib * obs_k) + (ref_lib - ref_k) / (
            ref_lib * ref_k
        )
        w = np.maximum(w, 1e-12)

        # trim by M and A simultaneously
        lo_m, hi_m = np.percentile(M, [trim_m * 100, (1 - trim_m) * 100])
        lo_a, hi_a = np.percentile(A, [trim_a * 100, (1 - trim_a) * 100])
        use = (M >= lo_m) & (M <= hi_m) & (A >= lo_a) & (A <= hi_a)
        if use.sum() < 5:
            continue

        log_factors[k] = float(np.sum(M[use] / w[use]) / np.sum(1.0 / w[use]))

    return log_factors  # log2 scale; add to log(lib_size) to get full offset


def _tmm_offsets(counts_mat, lib_sizes=None):
    """Log library size per sample (total counts, floor 1).

    counts_mat : (n_genes, n_samples)
    returns    : (n_samples,) log offsets
    """
    ls = counts_mat.sum(axis=0).astype(float)
    ls = np.maximum(ls, 1.0)
    return np.log(ls)


def _irls_nb_pseudobulk(S, x, phi_size, offsets, max_iter=25, tol=1e-7):
    """IRLS for NB GLM on pseudobulk counts.

    phi_size  : NB size parameter r (= 1/sigma2 for subject-level NB)
    returns mu (n,), beta (p,)
    """
    eta = offsets.copy()  # init: mu = exp(offset)
    mu = np.exp(eta)
    for _ in range(max_iter):
        W = mu / (1.0 + mu / phi_size)  # NB working weight
        z = eta + (S - mu) / mu  # working response
        Wx = x * W[:, None]
        try:
            beta = np.linalg.solve(Wx.T @ x, Wx.T @ (z - offsets))
        except np.linalg.LinAlgError:
            break
        eta_new = x @ beta + offsets
        mu_new = np.exp(eta_new)
        if np.max(np.abs(mu_new - mu) / (np.abs(mu) + 1e-8)) < tol:
            mu = mu_new
            break
        mu = mu_new
        eta = eta_new
    return mu, beta


def _nr_k(S, M, k_init=1.0, max_iter=25, tol=1e-6):
    """Newton-Raphson for k = 1/sigma2 in NB pseudobulk marginal likelihood.

    Marginal: S_i ~ NB(M_i, 1/k).  Gradient + Hessian use digamma/polygamma.
    """
    k = max(float(k_init), 1e-4)
    S = np.asarray(S, float)
    M = np.asarray(M, float)
    for _ in range(max_iter):
        g = float(np.sum(_dg(k + S) - _dg(k) + np.log(k / (k + M)) + (M - S) / (k + M)))
        h = float(
            np.sum(
                _pg(1, k + S) - _pg(1, k) + 1 / k - 1 / (k + M) - (M - S) / (k + M) ** 2
            )
        )
        if abs(h) < 1e-12:
            break
        k_new = k - g / h
        if k_new <= 0:
            k_new = k * 0.5
        if abs(k_new - k) / (abs(k) + 1e-8) < tol:
            k = k_new
            break
        k = k_new
    return max(k, 1e-6)


def _phi_mom(counts_g, mu_cell):
    """Method-of-moments cell-level NB dispersion estimate.

    counts_g : (n_cells,)  raw UMI counts
    mu_cell  : (n_cells,)  expected counts per cell
    returns phi (cell-level dispersion = 1/r_cell)
    """
    r = (counts_g - mu_cell) ** 2
    phi = np.mean((r - mu_cell) / (mu_cell**2 + 1e-8))
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
    bins = np.percentile(log_m, np.linspace(0, 100, n_bins + 1))
    prior = np.full(len(phi_hat), np.nanmedian(phi_hat[ok]))
    for b in range(n_bins):
        mask_b = ok & (log_m >= bins[b]) & (log_m <= bins[b + 1])
        if mask_b.sum() >= 3:
            bin_idx = np.where(ok)[0]
            subset = bin_idx[
                (log_m[bin_idx] >= bins[b]) & (log_m[bin_idx] <= bins[b + 1])
            ]
            prior_b = float(np.median(phi_hat[subset]))
            # assign prior to all genes in this expression range
            log_expr = np.log(mean_expr + 1e-8)
            full_mask = (log_expr >= bins[b]) & (log_expr <= bins[b + 1])
            prior[full_mask] = prior_b
    # posterior: (df_prior * prior + df_gene * phi) / (df_prior + df_gene)
    # df_gene = 1 (single per-gene estimate), df_prior as specified
    shrunk = (df_prior * prior + phi_hat) / (df_prior + 1.0)
    shrunk[~ok] = phi_hat[~ok]
    return np.maximum(shrunk, 1e-6)


def fit_null_sc_native(
    counts_cells: np.ndarray,
    sample_ids: "np.ndarray | list[str]",
    covariates_cells: "np.ndarray | None" = None,
    verbose: bool = False,
) -> dict:
    """Native NEBULA-LN Stage 1 null fit — no edgepython required.

    Implements the NB-gamma mixed model from He et al. 2021, using the LN
    approximation for large CPS. Alternates IRLS for beta and Newton-Raphson
    for k=1/sigma2, followed by EB shrinkage of cell-level dispersion.

    Args:
        counts_cells: Raw UMI count matrix, shape (n_genes, n_cells).
        sample_ids: Sample/subject label per cell, length n_cells.
        covariates_cells: Cell-level covariate matrix (n_cells, p) or None.
            If None, intercept only.
        verbose: Log progress at INFO level if True.

    Returns:
        Dict compatible with fit_null_sc output, containing:
            dispersion: (n_genes_pass,) raw MOM cell-level phi,
            dispersion_shrunk: (n_genes_pass,) EB-shrunk phi,
            dispersion_use: (n_genes_pass,) same as dispersion_shrunk,
            sigma_sample: (n_genes_pass,) subject-level overdispersion sigma2,
            gene_mask: (n_genes,) bool pass/fail filter,
            fitted_values: None (not computed at cell level).
    """
    counts_cells = np.asarray(counts_cells, dtype=np.float64)
    n_genes, n_cells = counts_cells.shape
    sample_ids = np.asarray(sample_ids)
    samples, inv = np.unique(sample_ids, return_inverse=True)
    n_samp = len(samples)

    # subject-level pseudobulk and library sizes
    lib_cell = counts_cells.sum(axis=0) + 1.0  # (n_cells,) per-cell total + 1
    lib_samp = np.zeros(n_samp)
    ncell_s = np.zeros(n_samp)
    np.add.at(lib_samp, inv, lib_cell)
    np.add.at(ncell_s, inv, 1.0)

    # pseudobulk count matrix for TMM: sum gene counts per subject
    pb_for_tmm = np.zeros((counts_cells.shape[0], n_samp), dtype=np.float64)
    np.add.at(pb_for_tmm.T, inv, counts_cells.T)

    # TMM factors (log2 scale) → convert to natural log and add to log(lib_samp)
    tmm_log2 = _tmm_norm_factors(pb_for_tmm)  # (n_samp,) log2
    offsets_samp = np.log(lib_samp.clip(1)) + tmm_log2 * np.log(2)  # (n_samp,)
    del pb_for_tmm

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
    n_expr = (counts_cells > 0).sum(axis=1)
    gene_mask = (mean_cpc > 0.005) & (n_expr >= 5)
    idx = np.where(gene_mask)[0]
    n_pass = len(idx)
    if verbose:
        logger.info("fit_null_sc_native: %d/%d genes pass filter", n_pass, n_genes)

    phi_hat = np.full(n_pass, 0.1)
    sigma2 = np.full(n_pass, 0.1)
    mean_pb = np.zeros(n_pass)

    for gi, g in enumerate(idx):
        counts_g = counts_cells[g]
        # pseudobulk sum per subject
        S = np.zeros(n_samp)
        np.add.at(S, inv, counts_g)

        # initial phi: MOM on pseudobulk residuals
        pb_mean = S.mean()
        pb_var = S.var()
        phi_pb_init = max((pb_var - pb_mean) / (pb_mean**2 + 1e-8), 1e-3)

        # NEBULA-LN: alternate beta (IRLS) and k (NR)
        k = 1.0 / max(phi_pb_init, 1e-3)
        mu_s = np.ones(n_samp)
        for _ in range(8):
            mu_s, _ = _irls_nb_pseudobulk(S, x_samp, k, offsets_samp)
            k = _nr_k(S, mu_s, k_init=k)

        sigma2[gi] = 1.0 / max(k, 1e-6)
        mean_pb[gi] = float(mu_s.mean())

        # cell-level phi: MOM after removing subject scale factor
        v_hat = (S / mu_s.clip(1e-8))[inv]  # (n_cells,)
        mu_cell = v_hat * (mu_s / ncell_s.clip(1))[inv]
        phi_hat[gi] = _phi_mom(counts_g, mu_cell)

        if verbose and gi % max(1, n_pass // 10) == 0:
            logger.info(
                "  %d/%d  phi=%.3f  sigma2=%.3f",
                gi + 1,
                n_pass,
                phi_hat[gi],
                sigma2[gi],
            )

    # EB shrinkage
    phi_shrunk = _wleb_shrink(phi_hat, mean_pb)

    return {
        "dispersion": phi_hat,
        "dispersion_shrunk": phi_shrunk,
        "dispersion_use": phi_shrunk,
        "sigma_sample": sigma2,
        "gene_mask": gene_mask,
        "fitted_values": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# stage 1 helpers
# ─────────────────────────────────────────────────────────────────────────────


def fit_null_sc(
    counts_cells: np.ndarray,
    strain_ids: "np.ndarray | list[str]",
    covariates_cells: "np.ndarray | None" = None,
    norm_method: str = "TMM",
    shrink: bool = True,
    verbose: bool = False,
    use_native: bool = True,
) -> dict:
    """Fit NEBULA-LN null model on cell-level counts for all genes.

    Dispatches to the native numpy/scipy implementation by default
    (use_native=True). The legacy edgepython path (use_native=False) is
    retained for backward compatibility but requires edgepython installed.
    Run once per cell type before the genome-wide scan.

    Args:
        counts_cells: Raw UMI count matrix, shape (n_genes, n_cells).
        strain_ids: Strain/subject label per cell, length n_cells.
        covariates_cells: Cell-level covariate matrix (n_cells, p) or None.
            If None, intercept only.
        norm_method: Normalisation method passed to edgepython; ignored by
            the native path. Either 'TMM' or 'none'.
        shrink: Apply EB dispersion shrinkage (default True).
        verbose: Log progress at INFO level if True.
        use_native: Use native implementation (recommended). Set False to
            use the legacy edgepython path (requires edgepython installed).

    Returns:
        Dict containing dispersion_use (n_genes_filtered,), sigma_sample
        (n_genes_filtered,), and gene_mask (n_genes,) bool, plus raw and
        shrunk dispersion arrays.
    """
    if use_native:
        return fit_null_sc_native(
            counts_cells, strain_ids, covariates_cells, verbose=verbose
        )

    # ── legacy edgepython path ────────────────────────────────────────────
    import edgepython as ep
    import pandas as pd

    strain_ids = np.asarray(strain_ids)

    # design: intercept + covariates (cell level)
    if covariates_cells is not None:
        x = np.asarray(covariates_cells, dtype=np.float64)
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((len(x), 1)), x])
        design = pd.DataFrame(x, columns=[f"V{i+1}" for i in range(x.shape[1])])
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
        fit["dispersion_use"] = np.where(
            np.isfinite(fit["dispersion_shrunk"]) & (fit["dispersion_shrunk"] > 0),
            fit["dispersion_shrunk"],
            fit["dispersion"],
        )
    else:
        fit["dispersion_use"] = fit["dispersion"]

    n_pass = int(fit["gene_mask"].sum())
    logger.info(
        "fit_null_sc: %d/%d genes pass filter  " "med_disp=%.3f  med_sigma=%.3f",
        n_pass,
        len(fit["gene_mask"]),
        float(np.nanmedian(fit["dispersion_use"])),
        float(np.nanmedian(fit["sigma_sample"])),
    )
    return fit


def pseudobulk_sum(
    counts_cells: np.ndarray,
    strain_ids: "np.ndarray | list[str]",
    gene_mask: "np.ndarray | None" = None,
) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":
    """Aggregate cell-level counts to per-strain pseudobulk sums.

    Args:
        counts_cells: Raw UMI count matrix, shape (n_genes, n_cells).
        strain_ids: Strain/subject label per cell, length n_cells.
        gene_mask: Boolean mask (n_genes,) selecting genes to keep, or None.
            If None, all genes are kept.

    Returns:
        Tuple of (pb_counts, pb_ncells, strains) where pb_counts has shape
        (n_genes_kept, n_strains), pb_ncells is (n_strains,) cells per
        strain, and strains is the sorted unique strain ID array.
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


def precompute_sufficient_stats(
    counts: np.ndarray,
    subject_ids: "np.ndarray | list[str]",
    library_sizes: "np.ndarray | None" = None,
) -> dict:
    """Compute per-subject count statistics from cell-level data (1 gene).

    Used by vqtl.py to estimate per-subject overdispersion.

    Args:
        counts: Raw UMI counts for one gene, shape (n_cells,).
        subject_ids: Subject label per cell, length n_cells.
        library_sizes: Per-cell total UMI counts (n_cells,) or None.
            Reserved for future use; currently unused.

    Returns:
        Dict with keys: subjects (n_subjects,) unique IDs, n_cells
        (n_subjects,) cells per subject, mean_count (n_subjects,) mean
        count per subject, var_count (n_subjects,) variance per subject.
    """
    counts = np.asarray(counts, dtype=np.float64)
    subject_ids = np.asarray(subject_ids)
    subjects, inv = np.unique(subject_ids, return_inverse=True)
    n_subj = len(subjects)

    n_cells = np.bincount(inv, minlength=n_subj).astype(np.float64)
    sum_counts = np.bincount(inv, weights=counts, minlength=n_subj)
    mean_count = np.where(n_cells > 0, sum_counts / np.maximum(n_cells, 1), 0.0)

    # variance via E[X^2] - (E[X])^2
    sum_sq = np.bincount(inv, weights=counts**2, minlength=n_subj)
    var_count = np.where(
        n_cells > 1,
        sum_sq / np.maximum(n_cells, 1) - mean_count**2,
        0.0,
    )

    return {
        "subjects": subjects,
        "n_cells": n_cells,
        "mean_count": mean_count,
        "var_count": var_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# stage 2 — genome-wide score test
# ─────────────────────────────────────────────────────────────────────────────


def _nb_loglik(y, mu, phi):
    """NB log-likelihood summed over samples."""
    r = 1.0 / max(phi, 1e-12)
    mu = np.maximum(mu, 1e-300)
    return float(
        (
            _gammaln(y + r)
            - _gammaln(r)
            - _gammaln(y + 1)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        ).sum()
    )


# ─────────────────────────────────────────────────────────────────────────────
# woodbury helpers — Woodbury identity for V = diag(1/W) + tau * Z K_m Z^T
# never form K_n = Z K_m Z^T explicitly; all operations stay at (m×m) cost.
# ─────────────────────────────────────────────────────────────────────────────


def _wb(v, W, Z, M_inv):
    """V^{-1} v via Woodbury.  V = diag(1/W) + tau*Z K Z^T.

    V^{-1}v = W*v - W*(Z M^{-1} Z^T W v)   (M = K^{-1}/tau + Z^T diag(W) Z)
    """
    Dv = W * v  # D^{-1}v = W*v
    return Dv - W * (Z @ (M_inv @ (Z.T @ Dv)))  # (n,)


def _wb_mat(A, W, Z, M_inv):
    """V^{-1} A via Woodbury.  A: (n, k)."""
    DA = W[:, None] * A
    return DA - W[:, None] * (Z @ (M_inv @ (Z.T @ DA)))  # (n, k)


def _wb_batch(H_all, W, Z, M_inv):
    """V^{-1} H_all via Woodbury.  H_all: (L, n, J-1) — all loci at once.

    returns (L, n, J-1).
    """
    DH = W[None, :, None] * H_all  # (L, n, J-1)
    ZtDH = np.einsum("mn,lnj->lmj", Z.T, DH, optimize=True)  # (L, m, J-1)
    MiZtDH = np.einsum("mk,lkj->lmj", M_inv, ZtDH, optimize=True)  # (L, m, J-1)
    return DH - W[None, :, None] * np.einsum("nm,lmj->lnj", Z, MiZtDH, optimize=True)


def _build_M(K_m, tau, W, Z):
    """Build M = K^{-1}/tau + Z^T diag(W) Z  (m×m) for Woodbury."""
    return np.linalg.inv(K_m) / tau + Z.T @ (W[:, None] * Z)


def _pql_tau_reml_wb(z, x, K_m, Z, W, tau_init=0.1):
    """REML for tau via Woodbury identity (replaces _pql_tau_reml).

    working LMM: z ~ N(X*beta, D + tau*Z K_m Z^T),  D = diag(1/W)
    log|V| via matrix det lemma: log|M| + m*log(tau) + log|K_m| + log|D|
    all linear algebra stays at m×m cost (never form n×n matrices).
    """
    m = K_m.shape[0]
    K_inv = np.linalg.inv(K_m)
    log_det_K = np.linalg.slogdet(K_m)[1]
    log_det_D = -np.sum(np.log(W))  # log|D| = -sum log W
    ZtWZ = Z.T @ (W[:, None] * Z)  # (m, m), precomputed

    def _neg_reml(log_tau):
        """Negative REML log-likelihood (Woodbury form) as a function of log(tau)."""
        tau = np.exp(log_tau)
        M = K_inv / tau + ZtWZ  # (m, m)
        try:
            L_cho, low = _cho_fac(M)
        except np.linalg.LinAlgError:
            return 1e10
        M_inv = _cho_sv((L_cho, low), np.eye(m))
        log_det_M = 2.0 * np.sum(np.log(np.diag(L_cho)))
        log_det_V = log_det_M + m * np.log(tau) + log_det_K + log_det_D

        Vinvz = _wb(z, W, Z, M_inv)
        VinvX = _wb_mat(x, W, Z, M_inv)
        A = x.T @ VinvX  # (p, p)
        sign, log_det_A = np.linalg.slogdet(A)
        if sign <= 0:
            return 1e10
        Pz = Vinvz - VinvX @ np.linalg.solve(A, x.T @ Vinvz)
        return 0.5 * (log_det_V + log_det_A + float(z @ Pz))

    log_t0 = float(np.log(max(tau_init, 1e-6)))
    res = _minimize_scalar(_neg_reml, bounds=(log_t0 - 6, log_t0 + 6), method="bounded")
    return float(max(np.exp(res.x), 1e-8))


def fit_null_nb_glmm_loco(
    S: np.ndarray,
    x: np.ndarray,
    offsets: np.ndarray,
    K_m: np.ndarray,
    Z: np.ndarray,
    phi: float,
    max_iter: int = 5,
    tol: float = 1e-4,
) -> dict:
    """PQL null NB-GLMM with LOCO kinship via Woodbury identity.

    Model: S_i ~ NB(mu_i, phi),  log mu_i = x_i^T beta + u_i + offset_i
           u   ~ N(0, tau * K_m(sample)),  K_m(sample) = Z K_m Z^T

    Woodbury: V = diag(1/W) + tau*Z K_m Z^T.  Never forms the n×n matrix.
    All kinship solves operate at m×m cost where m = number of unique strains.

    Args:
        S: Pseudobulk counts per sample, shape (n,).
        x: Covariate matrix, shape (n, p).
        offsets: Log library size offsets, shape (n,).
        K_m: Strain-level LOCO kinship, shape (m, m).
        Z: Sample-to-strain mapping (one-hot rows), shape (n, m).
        phi: NB dispersion held fixed from Stage 1.
        max_iter: Maximum PQL iterations.
        tol: Convergence tolerance on relative mu change.

    Returns:
        Dict with keys: mu, tau, W, M_inv, Z, VinvX, XtVinvX_inv, r_adj.
        P is never stored — applied implicitly via _wb / _wb_batch.
    """
    phi = max(float(phi), 1e-6)
    mu = np.maximum(S.astype(np.float64), 0.5)
    tau = 0.1

    for it in range(max_iter):
        W = mu / (1.0 + phi * mu)
        z = (np.log(mu) - offsets) + (S - mu) / mu

        # REML for tau via Woodbury (m×m cost)
        tau = _pql_tau_reml_wb(z, x, K_m, Z, W, tau_init=tau)

        # Woodbury working quantities
        M = _build_M(K_m, tau, W, Z)  # (m, m)
        M_inv = np.linalg.inv(M)

        # GLS for beta
        Vinvz = _wb(z, W, Z, M_inv)
        VinvX = _wb_mat(x, W, Z, M_inv)  # (n, p)
        A = x.T @ VinvX  # (p, p)
        beta = np.linalg.solve(A, VinvX.T @ z)

        # BLUP for u
        resid = z - x @ beta
        Vinvr = _wb(resid, W, Z, M_inv)
        u_hat = tau * (Z @ (K_m @ (Z.T @ Vinvr)))  # u = tau K_n V^{-1} residual

        mu_new = np.exp(x @ beta + u_hat + offsets)
        delta = float(np.max(np.abs(mu_new - mu) / (np.abs(mu) + 1e-8)))
        mu = mu_new
        if delta < tol and it > 0:
            break

    # final working quantities for score test
    W = mu / (1.0 + phi * mu)
    z_fin = (np.log(mu) - offsets) + (S - mu) / mu
    M = _build_M(K_m, tau, W, Z)
    M_inv = np.linalg.inv(M)
    VinvX = _wb_mat(x, W, Z, M_inv)
    A = x.T @ VinvX  # (p, p)
    A_inv = np.linalg.inv(A)
    Vinvz = _wb(z_fin, W, Z, M_inv)
    r_adj = Vinvz - VinvX @ (A_inv @ (x.T @ Vinvz))  # P z  (n,)

    return {
        "mu": mu,
        "tau": float(tau),
        "W": W,
        "M_inv": M_inv,
        "Z": Z,
        "VinvX": VinvX,
        "XtVinvX_inv": A_inv,
        "r_adj": r_adj,
    }


def _score_stats_loco(H_all, r_adj, W, Z, M_inv, VinvX, XtVinvX_inv):
    """Vectorised NB-GLMM score statistics via Woodbury (no P matrix stored).

    I_l = H_l^T P H_l = H_l^T V^{-1} H_l - (H_l^T V^{-1}X)(X^T V^{-1}X)^{-1}(X^T V^{-1}H_l)

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
    U = np.einsum("lnj,n->lj", H_all, r_adj, optimize=True)

    # V^{-1} H_all via Woodbury  (L, n, J-1)
    VinvH = _wb_batch(H_all, W, Z, M_inv)

    # H_l^T V^{-1} H_l  (L, J-1, J-1)
    HtVinvH = np.einsum("lnj,lnk->ljk", H_all, VinvH, optimize=True)

    # correction: (H_l^T V^{-1}X)(X^T V^{-1}X)^{-1}(X^T V^{-1}H_l)
    HtVinvX = np.einsum("lnj,np->ljp", H_all, VinvX, optimize=True)  # (L, J-1, p)
    corr = np.einsum("ljp,pk,lqk->ljq", HtVinvX, XtVinvX_inv, HtVinvX, optimize=True)

    info_mat = HtVinvH - corr + np.eye(nf1)[None] * 1e-10  # (L, J-1, J-1)
    try:
        sol = np.linalg.solve(info_mat, U[..., None]).squeeze(-1)  # (L, J-1)
        sc = np.maximum(np.einsum("lj,lj->l", U, sol, optimize=True), 0.0)
    except np.linalg.LinAlgError:
        sc = np.zeros(L)
        for li in range(L):
            try:
                sc[li] = max(float(U[li] @ np.linalg.solve(info_mat[li], U[li])), 0.0)
            except np.linalg.LinAlgError:
                pass
    return sc


def _score_stats_chromosome(H_all, r_vec, w_vec):
    """Vectorised NB score statistics for all loci on one chromosome.

    H_all : (L, n, J-1)  haplotype design at all loci
    r_vec : (n,)          working residual under null
    w_vec : (n,)          working weight under null

    returns (L,) score statistics Q_ℓ ~ χ²(J-1) under H₀
    """
    u = np.einsum("lnj,n->lj", H_all, r_vec, optimize=True)  # (L,J-1)
    hw = H_all * w_vec[None, :, None]
    info = np.einsum("lnj,lnk->ljk", hw, H_all, optimize=True)  # (L,J-1,J-1)
    info += np.eye(H_all.shape[2])[None] * 1e-10
    try:
        sol = np.linalg.solve(info, u[..., None]).squeeze(-1)  # (L, J-1)
        sc = np.maximum(np.einsum("lj,lj->l", u, sol, optimize=True), 0.0)
    except np.linalg.LinAlgError:
        sc = np.zeros(H_all.shape[0])
        for li in range(len(sc)):
            try:
                sc[li] = max(float(u[li] @ np.linalg.solve(info[li], u[li])), 0.0)
            except np.linalg.LinAlgError:
                pass
    return sc


def scan_nb_gene(
    counts_pb: np.ndarray,
    genomecache: "GenomeCache",
    sample_ids: "list[str]",
    phi: float,
    sigma_sample: float = 0.0,
    covariates: "np.ndarray | None" = None,
    lib_sizes: "np.ndarray | None" = None,
    kinship_loco: "dict | None" = None,
    loci_subset: "set | None" = None,
    phenotype_id: str = "unknown",
) -> ScanResult:
    """NB score-test scan for one gene on pseudobulk counts.

    Runs the Stage 2 NB-GLMM score test across all loci in the genome cache
    for a single gene. When kinship_loco is provided, fits a per-chromosome
    PQL NB-GLMM null with LOCO kinship via the Woodbury identity. Otherwise
    falls back to a standard NB GLM null with optional sigma_sample inflation.

    Args:
        counts_pb: Pseudobulk sum counts per sample, shape (n_samples,).
        genomecache: Loaded genome cache.
        sample_ids: Strain/sample IDs corresponding to counts_pb rows.
            Must match genomecache.sample_ids order or be alignable via
            the Z-matrix.
        phi: NB dispersion from Stage 1 (EB-shrunk). Must be > 0.
        sigma_sample: Subject-level variance from Stage 1 (>= 0). Inflates
            the score test denominator when > 0 to account for residual
            within-strain variability.
        covariates: Sample-level covariate matrix (n_samples, p) or None.
            Intercept column is added if not already present.
        lib_sizes: Log library size offset per sample (n_samples,) or None.
            If None, uniform offsets are used.
        kinship_loco: LOCO kinship dict {chr: ndarray (m, m)} or None.
            When provided, each chromosome's null is fit with the LOCO
            kinship random effect via PQL + Woodbury. Strongly recommended
            for multiparent populations.
        loci_subset: Restrict scan to this set of locus IDs or None.
        phenotype_id: Label stored in the returned ScanResult.

    Returns:
        ScanResult with fields lod (n_loci,), p_value (n_loci,),
        betas (n_loci, J) zero-padded (score test does not estimate per-founder
        effects; use effects.py at the peak locus for effect estimates),
        chromosomes, positions, locus_ids, and h2_null.
    """
    phi = max(float(phi), 1e-6)
    y = np.asarray(counts_pb, dtype=np.float64)
    sids = list(sample_ids)

    # align to cache order via Z-matrix
    cache_ids = list(genomecache.sample_ids)
    if sids == cache_ids:
        z_sub, valid = None, np.ones(len(sids), dtype=bool)
    else:
        z_full, valid = build_z_matrix(sids, cache_ids)
        z_sub = z_full[valid]

    y = y[valid]
    n = len(y)

    x = (
        covariates[valid].copy()
        if covariates is not None
        else np.ones((n, 1), dtype=float)
    )
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    s = lib_sizes[valid] if lib_sizes is not None else np.ones(n)
    offsets = np.log(np.maximum(s, 1e-8))  # log library size offset

    # ── null model ────────────────────────────────────────────────────────
    # with kinship: PQL NB-GLMM per chromosome (fit_null_nb_glmm_loco)
    # without kinship: standard NB GLM via native IRLS
    use_kinship = kinship_loco is not None and len(kinship_loco) > 0
    chr_keys = {str(k) for k in kinship_loco} if kinship_loco else set()

    # global null (no-kinship path only)
    mu_null = None
    r_vec_global = None
    w_vec_global = None
    if not use_kinship:
        phi_size = 1.0 / max(phi, 1e-12)
        mu_null, _ = _irls_nb_pseudobulk(y, x, phi_size, offsets)
        denom = 1.0 + phi * mu_null
        w_inf = max(1.0 + sigma_sample, 1.0)
        r_vec_global = (y - mu_null) / (denom * w_inf)
        w_vec_global = mu_null / (denom * w_inf)

    nf = genomecache.n_founders  # J — full founder count
    nf1 = nf - 1  # J-1 — score test columns (rank constraint)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)

        if loci_subset is not None:
            mask = np.isin(lids, list(loci_subset))
            lids, pos, probs = lids[mask], pos[mask], probs[mask]
        if not len(lids):
            continue

        # Z-expand haplotypes (n_loci, n, J-1)
        H = (
            np.einsum("nm,lmj->lnj", z_sub, probs[:, :, :nf1], optimize=True)
            if z_sub is not None
            else probs[:, valid, :nf1]
        )

        # per-chromosome null (LOCO kinship path)
        glmm_fit = None
        if use_kinship and str(chrom) in chr_keys:
            _kkey = chrom if chrom in kinship_loco else str(chrom)
            K_c = np.asarray(kinship_loco[_kkey], dtype=np.float64)  # (m, m)
            # Z_pql: sample→strain mapping for valid samples
            Z_pql = z_sub if z_sub is not None else np.eye(K_c.shape[0])
            glmm_fit = fit_null_nb_glmm_loco(y, x, offsets, K_c, Z_pql, phi)
        elif use_kinship:
            # kinship requested but this chr absent — skip
            logger.debug("scan_nb_gene: chr %s not in kinship_loco — skipping", chrom)
            continue

        # score test (only supported path)
        if use_kinship and glmm_fit is not None:
            sc = _score_stats_loco(
                H,
                glmm_fit["r_adj"],
                glmm_fit["W"],
                glmm_fit["Z"],
                glmm_fit["M_inv"],
                glmm_fit["VinvX"],
                glmm_fit["XtVinvX_inv"],
            )
        else:
            sc = _score_stats_chromosome(H, r_vec_global, w_vec_global)

        lods = sc / (2.0 * np.log(10))
        pv = 1.0 - _stats.chi2.cdf(sc, df=nf1)
        betas = np.zeros(
            (len(lids), nf)
        )  # (n_loci, J) — score test gives no per-locus estimates

        all_lids.append(lids)
        all_chrs.append(np.full(len(lids), chrom))
        all_pos.append(pos)
        all_lod.append(lods)
        all_pval.append(pv)
        all_betas.append(betas)

        del H, sc, lods, pv, betas
        gc.collect()

    if not all_lids:
        return ScanResult(
            locus_ids=np.array([]),
            chromosomes=np.array([]),
            positions=np.array([]),
            lod=np.array([]),
            p_value=np.array([]),
            betas=np.empty((0, nf)),
            h2_null={},
            phenotype_id=phenotype_id,
        )

    return ScanResult(
        locus_ids=np.concatenate(all_lids),
        chromosomes=np.concatenate(all_chrs),
        positions=np.concatenate(all_pos),
        lod=np.concatenate(all_lod),
        p_value=np.concatenate(all_pval),
        betas=np.vstack(all_betas),
        h2_null={},
        phenotype_id=phenotype_id,
    )


def scan_nb_multi_gene(
    pb_matrix: np.ndarray,
    gene_ids: "list[str]",
    phi_vec: np.ndarray,
    sigma_vec: np.ndarray,
    genomecache: "GenomeCache",
    sample_ids: "list[str]",
    covariates: "np.ndarray | None" = None,
    lib_sizes: "np.ndarray | None" = None,
    kinship_loco: "dict | None" = None,
    cis_loci_dict: "dict | None" = None,
    output_path: "Path | None" = None,
) -> "dict[str, ScanResult]":
    """Stream NB score-test scan across multiple genes, one gene at a time.

    Iterates over genes in pb_matrix, calling scan_nb_gene for each. After
    each gene, the result is released and gc.collect() is called to keep
    peak RAM bounded to approximately one gene's working arrays.

    Args:
        pb_matrix: Pseudobulk count matrix, shape (n_genes, n_samples).
        gene_ids: Gene identifiers, length n_genes.
        phi_vec: Per-gene NB dispersion from Stage 1, shape (n_genes,).
        sigma_vec: Per-gene subject-level variance from Stage 1, shape (n_genes,).
        genomecache: Loaded genome cache.
        sample_ids: Genotype IDs (strain labels) per sample, length n_samples.
        covariates: Sample-level covariate matrix (n_samples, p) or None.
        lib_sizes: Log library size offsets per sample, shape (n_samples,) or None.
        kinship_loco: LOCO kinship dict {chrom: (m, m) array} or None.
        cis_loci_dict: Restrict scan to these loci per gene or None.
            Keyed by gene ID string.
        output_path: If set, append each gene's peak result to this CSV
            incrementally. File is created if it does not exist.

    Returns:
        Dict mapping gene_id to ScanResult.
    """
    results: dict[str, ScanResult] = {}
    n = len(gene_ids)
    for gi, gid in enumerate(gene_ids):
        cis = cis_loci_dict.get(str(gid)) if cis_loci_dict else None
        res = scan_nb_gene(
            pb_matrix[gi],
            genomecache,
            sample_ids,
            phi=float(phi_vec[gi]),
            sigma_sample=float(sigma_vec[gi]),
            covariates=covariates,
            lib_sizes=lib_sizes,
            kinship_loco=kinship_loco,
            loci_subset=cis,
            phenotype_id=str(gid),
        )
        results[str(gid)] = res
        if (gi + 1) % max(1, n // 10) == 0:
            logger.info("scan_nb_multi_gene: %d/%d genes", gi + 1, n)
        del res
        gc.collect()
    return results


def _scan_nb_chromosome(
    chrom: str,
    pb_matrix: np.ndarray,
    gene_ids: "list[str]",
    phi_vec: np.ndarray,
    sigma_vec: np.ndarray,
    cache_path: str,
    sample_ids: "list[str]",
    covariates: "np.ndarray | None",
    lib_sizes: "np.ndarray | None",
    kinship_loco: "dict | None",
    cis_loci_dict: "dict | None",
) -> "dict[str, tuple]":
    """Scan all genes on one chromosome. Opens its own cache handle — thread-safe.

    Each thread worker calls GenomeCache.from_hdf5(cache_path) independently so
    that concurrent threads hold independent h5py file descriptors (h5py is safe
    for concurrent reads when each thread owns its own handle).

    Args:
        chrom: Chromosome identifier (str).
        pb_matrix: Pseudobulk counts, shape (n_genes, n_strains).
        gene_ids: Gene identifiers, length n_genes.
        phi_vec: Per-gene NB dispersion, shape (n_genes,).
        sigma_vec: Per-gene subject variance, shape (n_genes,).
        cache_path: HDF5 path to reopen per thread.
        sample_ids: Strain IDs matching pb_matrix columns.
        covariates: Strain-level covariate matrix (n_strains, p) or None.
        lib_sizes: Log library size offset per strain or None.
        kinship_loco: LOCO kinship dict {chr: ndarray (m, m)} or None.
        cis_loci_dict: Per-gene cis loci restriction or None.

    Returns:
        Dict mapping gene_id to (lids, pos, lod, pval) — all arrays for this
        chromosome. Empty if chromosome absent from cache or kinship.
    """
    cache = GenomeCache.from_hdf5(cache_path)

    try:
        lids_full, pos_full, probs_full = cache.get_chromosome(chrom)
    except (KeyError, OSError, RuntimeError):
        logger.debug("_scan_nb_chromosome: chr %s not in cache", chrom)
        return {}

    chr_keys = {str(k) for k in kinship_loco} if kinship_loco else set()
    use_kinship = kinship_loco is not None and str(chrom) in chr_keys
    if kinship_loco is not None and not use_kinship:
        logger.debug(
            "_scan_nb_chromosome: chr %s absent from kinship_loco — skipping", chrom
        )
        return {}

    # Z-matrix alignment (computed once per chromosome)
    cache_ids = list(cache.sample_ids)
    sids = list(sample_ids)
    if sids == cache_ids:
        z_sub: "np.ndarray | None" = None
        valid = np.ones(len(sids), dtype=bool)
    else:
        z_full, valid = build_z_matrix(sids, cache_ids)
        z_sub = z_full[valid]

    n = int(valid.sum())
    nf = cache.n_founders
    nf1 = nf - 1

    # LOCO null (shared across genes; only for kinship path)
    K_c: "np.ndarray | None" = None
    Z_pql: "np.ndarray | None" = None
    if use_kinship:
        _kkey = chrom if chrom in kinship_loco else str(chrom)  # type: ignore[index]
        K_c = np.asarray(kinship_loco[_kkey], dtype=np.float64)  # type: ignore[index]
        Z_pql = z_sub if z_sub is not None else np.eye(K_c.shape[0])

    out: dict[str, tuple] = {}

    for gi, gid in enumerate(gene_ids):
        cis = cis_loci_dict.get(str(gid)) if cis_loci_dict else None
        if cis is not None:
            mask = np.isin(lids_full, list(cis))
            lids, pos, probs = lids_full[mask], pos_full[mask], probs_full[mask]
        else:
            lids, pos, probs = lids_full, pos_full, probs_full

        if not len(lids):
            continue

        phi = max(float(phi_vec[gi]), 1e-6)
        sigma_sample = float(sigma_vec[gi])
        y = pb_matrix[gi][valid].astype(np.float64)
        x = (
            covariates[valid].copy()
            if covariates is not None
            else np.ones((n, 1), dtype=float)
        )
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((n, 1)), x])
        s = lib_sizes[valid] if lib_sizes is not None else np.ones(n)
        offsets = np.log(np.maximum(s, 1e-8))

        H = (
            np.einsum("nm,lmj->lnj", z_sub, probs[:, :, :nf1], optimize=True)
            if z_sub is not None
            else probs[:, valid, :nf1]
        )

        if use_kinship and K_c is not None and Z_pql is not None:
            glmm_fit = fit_null_nb_glmm_loco(y, x, offsets, K_c, Z_pql, phi)
            sc = _score_stats_loco(
                H,
                glmm_fit["r_adj"],
                glmm_fit["W"],
                glmm_fit["Z"],
                glmm_fit["M_inv"],
                glmm_fit["VinvX"],
                glmm_fit["XtVinvX_inv"],
            )
        else:
            phi_size = 1.0 / max(phi, 1e-12)
            mu_null, _ = _irls_nb_pseudobulk(y, x, phi_size, offsets)
            denom = 1.0 + phi * mu_null
            w_inf = max(1.0 + sigma_sample, 1.0)
            r_vec = (y - mu_null) / (denom * w_inf)
            w_vec = mu_null / (denom * w_inf)
            sc = _score_stats_chromosome(H, r_vec, w_vec)

        lod = sc / (2.0 * np.log(10))
        pval = 1.0 - _stats.chi2.cdf(sc, df=nf1)
        out[str(gid)] = (lids.copy(), pos.copy(), lod, pval)

        del H, sc, lod, pval
        gc.collect()

    return out


def scan_nb(
    counts_cells: np.ndarray,
    genomecache: "GenomeCache",
    strain_ids_cells: "np.ndarray | list[str]",
    gene_ids: "list[str] | None" = None,
    covariates_cells: "np.ndarray | None" = None,
    covariates_strains: "np.ndarray | None" = None,
    lib_sizes: "np.ndarray | None" = None,
    kinship_loco: "dict | None" = None,
    norm_method: str = "TMM",
    shrink_disp: bool = True,
    cis_loci_dict: "dict | None" = None,
    output_path: "Path | None" = None,
    n_cores: int = 1,
    verbose: bool = False,
) -> "dict[str, ScanResult]":
    """Genome-wide NB GLM scan — full two-stage pipeline for one cell type.

    Stage 1: fit_null_sc (cell level, once per cell type).
    Stage 2: NB score test per gene on pseudobulk counts (vectorised over loci).

    When n_cores=1 (default), streams one gene at a time — peak RAM is bounded
    to approximately one gene's working arrays (constant-memory streaming).
    When n_cores>1, uses a ThreadPoolExecutor where each thread processes one
    chromosome for all genes simultaneously; peak RAM scales as
    n_cores × per_chromosome_peak_ram. Requires genomecache._h5_path to be set
    (automatic when loaded via GenomeCache.from_hdf5).

    Args:
        counts_cells: Raw UMI count matrix, shape (n_genes, n_cells).
        genomecache: Loaded genome cache.
        strain_ids_cells: Strain label per cell, length n_cells.
        gene_ids: Row identifiers for counts_cells, or None (uses row indices).
        covariates_cells: Cell-level covariate matrix (n_cells, p) or None.
        covariates_strains: Strain-level covariate matrix (n_strains, p) or None.
            If None and covariates_cells is provided, derived by strain mean.
        lib_sizes: Log library size offset per strain (n_strains,) or None.
        kinship_loco: LOCO kinship dict {chr: ndarray (m, m)} or None.
            Strongly recommended for multiparent populations.
        norm_method: Normalisation method for the legacy edgepython path;
            ignored by the native Stage 1. Either 'TMM' or 'none'.
        shrink_disp: Apply EB dispersion shrinkage (default True).
        cis_loci_dict: Restrict each gene's scan to its cis loci or None.
        output_path: If set, scan_nb_multi_gene writes incremental CSV output
            (n_cores=1 only; ignored when n_cores>1).
        n_cores: Number of parallel chromosome workers (default 1). Values > 1
            enable chromosome-level parallelism via ThreadPoolExecutor. Requires
            genomecache._h5_path (set automatically by from_hdf5).
        verbose: Log progress at INFO level if True.

    Returns:
        Dict mapping gene_id string to ScanResult.
    """
    # stage 1: null model
    fit_null = fit_null_sc(
        counts_cells,
        strain_ids_cells,
        covariates_cells,
        norm_method=norm_method,
        shrink=shrink_disp,
        verbose=verbose,
    )

    gene_mask = fit_null["gene_mask"]
    n_genes_pass = int(gene_mask.sum())
    if n_genes_pass == 0:
        logger.warning("scan_nb: no genes passed glm_sc_fit filter")
        return {}

    gene_ids_all = (
        np.asarray(gene_ids)
        if gene_ids is not None
        else np.arange(counts_cells.shape[0]).astype(str)
    )
    gene_ids_pass = gene_ids_all[gene_mask]
    disp_pass = fit_null["dispersion_use"]  # (n_genes_pass,)
    sigma_pass = fit_null["sigma_sample"]  # (n_genes_pass,)

    # pseudobulk aggregation
    pb_counts, pb_ncells, strains = pseudobulk_sum(
        counts_cells, strain_ids_cells, gene_mask
    )  # (n_genes_pass, n_strains)

    # strain-level covariates
    if covariates_strains is None and covariates_cells is not None:
        sid_arr = np.asarray(strain_ids_cells)
        _, inv = np.unique(sid_arr, return_inverse=True)
        n_s = len(strains)
        cov_sum = np.zeros((n_s, covariates_cells.shape[1]))
        cnt_s = np.zeros(n_s)
        np.add.at(cov_sum, inv, covariates_cells)
        np.add.at(cnt_s, inv, 1)
        covariates_strains = cov_sum / cnt_s[:, None]

    # stage 2: sequential (constant-memory streaming) or chromosome-parallel
    if n_cores <= 1:
        return scan_nb_multi_gene(
            pb_counts,
            list(gene_ids_pass),
            disp_pass,
            sigma_pass,
            genomecache,
            list(strains),
            covariates=covariates_strains,
            lib_sizes=lib_sizes,
            kinship_loco=kinship_loco,
            cis_loci_dict=cis_loci_dict,
            output_path=output_path,
        )

    # chromosome-parallel path — each thread opens its own HDF5 handle
    if genomecache._h5_path is None:
        logger.warning(
            "scan_nb: n_cores=%d requested but genomecache._h5_path is None "
            "(cache was not loaded via from_hdf5); falling back to n_cores=1",
            n_cores,
        )
        return scan_nb_multi_gene(
            pb_counts,
            list(gene_ids_pass),
            disp_pass,
            sigma_pass,
            genomecache,
            list(strains),
            covariates=covariates_strains,
            lib_sizes=lib_sizes,
            kinship_loco=kinship_loco,
            cis_loci_dict=cis_loci_dict,
            output_path=output_path,
        )

    cache_path = genomecache._h5_path
    chromosomes = genomecache.chromosomes
    nf = genomecache.n_founders
    gene_ids_str = [str(g) for g in gene_ids_pass]

    # submit one task per chromosome
    chr_results: dict[str, dict] = {}
    with _ThreadPool(max_workers=n_cores) as ex:
        futures = {
            chrom: ex.submit(
                _scan_nb_chromosome,
                chrom,
                pb_counts,
                gene_ids_str,
                disp_pass,
                sigma_pass,
                cache_path,
                list(strains),
                covariates_strains,
                lib_sizes,
                kinship_loco,
                cis_loci_dict,
            )
            for chrom in chromosomes
        }
        for chrom, fut in futures.items():
            chr_results[chrom] = fut.result()
            if verbose:
                logger.info("scan_nb: chr %s done", chrom)

    # merge per-chromosome results into per-gene ScanResult
    gene_parts: dict[str, list] = {g: [] for g in gene_ids_str}
    for chrom in chromosomes:
        for gid, (lids, pos, lod, pval) in chr_results.get(chrom, {}).items():
            gene_parts[gid].append((chrom, lids, pos, lod, pval))

    results: dict[str, ScanResult] = {}
    for gid in gene_ids_str:
        parts = gene_parts[gid]
        if not parts:
            continue
        all_lids = np.concatenate([p[1] for p in parts])
        all_chrs = np.concatenate([np.full(len(p[1]), p[0]) for p in parts])
        all_pos = np.concatenate([p[2] for p in parts])
        all_lod = np.concatenate([p[3] for p in parts])
        all_pval = np.concatenate([p[4] for p in parts])
        results[gid] = ScanResult(
            locus_ids=all_lids,
            chromosomes=all_chrs,
            positions=all_pos,
            lod=all_lod,
            p_value=all_pval,
            betas=np.zeros((len(all_lids), nf)),
            h2_null={},
            phenotype_id=gid,
        )

    del chr_results, gene_parts
    gc.collect()
    return results

"""
permutation — permutation thresholds and parametric bootstrap CI

implements significance thresholds for genome-wide scans via permutation
and positional confidence intervals via parametric bootstrap.

design notes:
  - fix_h2=True (default) re-uses h² estimated from the real data across all
    permutations, saving n_perms × n_chrom calls to estimate_h2_null.
  - chrom kwarg restricts threshold scans to a single chromosome (fast single-
    gene mode used in validation); leave None for full-genome threshold.
  - run_positional_scans operates in genotype space (m strains), consistent
    with kinship_decomp which is always (m×m) before Z-expansion.
  - permutation_method='parametric' (default) draws a kinship-informed null:
    u~N(0,K·σ²_g), e~N(0,σ²_e·I), then rank-permutes y by the null draw's
    order. correct for real LOCO kinship (Phase 2+).
    permutation_method='plain' shuffles y labels — valid only when τ²≈0
    (identity kinship / OLS); used by Phase 1 validation only.
"""

import gc
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy.stats import genextreme

from ..utils import _build_gm_idx, _enrich_scan_df

logger = logging.getLogger("mosaiqtl")

CI_COLS: list[str] = ["gene_id", "chr", "peak_mb", "ci_lo", "ci_hi", "ci_width_mb"]

__all__ = [
    "KinshipSingularError",
    "generate_null_outcomes",
    "run_threshold_scans",
    "calibrate_gene_pvalue",
    "get_gev_thresholds",
    "load_per_chr_thresholds",
    "run_positional_scans",
]


class KinshipSingularError(ValueError):
    """raised when a LOCO kinship matrix is genuinely not positive semi-definite.

    a matrix counts as non-PSD when its smallest eigenvalue drops below
    -(largest eigenvalue × n² × machine epsilon). the n² tolerance allows for
    floating-point rounding that accumulates during the matrix decomposition
    used to draw the null; a tighter n×eps tolerance missed real failures.
    """


# ─────────────────────────────────────────────────────────────────────────────
# null outcome generation
# ─────────────────────────────────────────────────────────────────────────────


def generate_null_outcomes(y, n_perms=1000, method="permutation", seed=1):
    """generate null phenotype columns for threshold estimation.

    for permutation: shuffles phenotype labels (breaks geno-pheno link).
    for bootstrap: resamples with replacement from centred phenotype.


    parameters
    ----------
    y       : np.ndarray (n,)  original phenotype (NaN-free)
    n_perms : int              number of permuted/bootstrapped columns to draw
    method  : str              'permutation' (default) or 'bootstrap'
    seed    : int

    returns
    -------
    np.ndarray (n, n_perms)  null phenotype matrix
    """
    rng = np.random.default_rng(seed)

    if method == "permutation":
        return np.column_stack([rng.permutation(y) for _ in range(n_perms)])
    elif method == "bootstrap":
        y_c = y - y.mean()
        return np.column_stack(
            [rng.choice(y_c, size=len(y), replace=True) for _ in range(n_perms)]
        )
    else:
        raise ValueError(f"method must be permutation or bootstrap, got {method!r}")


# ─────────────────────────────────────────────────────────────────────────────
# parametric null helpers
# ─────────────────────────────────────────────────────────────────────────────


def _null_fit_params(
    y_rot: np.ndarray,
    x_rot: np.ndarray,
    eigvals: np.ndarray,
    h2: float,
) -> tuple[np.ndarray, float, float]:
    """recover beta_hat and variance components at a given h².

    parameters
    ----------
    y_rot   : (n,)    U'y — rotated phenotype
    x_rot   : (n, p)  U'X — rotated covariates
    eigvals : (n,)    eigenvalues of kinship matrix
    h2      : float   heritability

    returns
    -------
    beta      : (p,)  fixed-effect estimates
    sigma2_g  : float genetic variance  (= h2 * sigma2_total)
    sigma2_e  : float residual variance (= (1-h2) * sigma2_total)
    """
    if x_rot.ndim == 1:
        x_rot = x_rot[:, None]
    n = len(y_rot)
    w_inv = 1.0 / (h2 * eigvals + (1.0 - h2))
    xw = x_rot * w_inv[:, None]
    beta = np.linalg.solve(xw.T @ x_rot, xw.T @ y_rot)
    resid = y_rot - x_rot @ beta
    wrss = float(resid @ (resid * w_inv))
    sigma2_total = wrss / n
    return beta, h2 * sigma2_total, (1.0 - h2) * sigma2_total


def _draw_kinship_re(
    K: np.ndarray,
    sigma2_g: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """draw a random effect u ~ N(0, K·sigma2_g), checking K is positive semi-definite first.

    raises KinshipSingularError when K's smallest eigenvalue is more negative
    than the largest eigenvalue × n² × machine epsilon (see KinshipSingularError).
    """
    eigs = np.linalg.eigvalsh(K)
    min_eig = float(eigs.min())
    psd_tol = float(eigs.max()) * len(K) ** 2 * float(np.finfo(float).eps)
    if min_eig < -psd_tol:
        raise KinshipSingularError(
            f"LOCO kinship not PSD (min_eig={min_eig:.2e}); "
            "skipping chromosome — mirrors R behaviour on singular K"
        )
    return rng.multivariate_normal(np.zeros(len(K)), K * sigma2_g)


def _draw_parametric_batch(
    y: np.ndarray,
    x: np.ndarray,
    K: np.ndarray,
    evc: np.ndarray,
    ev: np.ndarray,
    h2: float,
    n_perms: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """draw n_perms parametric null phenotypes at once.

    draws all the random-effect and residual-noise vectors together, then
    rank-permutes y by each null column's order — one positive-semi-definite
    check up front instead of one per permutation.

    raises KinshipSingularError if K is not positive semi-definite.

    returns
    -------
    np.ndarray (n, n_perms)  each column is y rank-permuted by one null draw
    """
    y_rot = evc.T @ y
    x_rot = evc.T @ x
    beta, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, ev, h2)

    eigs = np.linalg.eigvalsh(K)
    min_eig = float(eigs.min())
    psd_tol = float(eigs.max()) * len(K) ** 2 * float(np.finfo(float).eps)
    if min_eig < -psd_tol:
        raise KinshipSingularError(
            f"LOCO kinship not PSD (min_eig={min_eig:.2e}); "
            "skipping chromosome — mirrors R behaviour on singular K"
        )

    # draw every random effect at once rather than looping over permutations
    U_mat = rng.multivariate_normal(np.zeros(len(y)), K * sigma2_g, size=n_perms).T
    E_mat = rng.normal(0.0, np.sqrt(max(sigma2_e, 0.0)), size=(len(y), n_perms))
    Y_null = (x @ beta)[:, None] + U_mat + E_mat

    # each column's rank order gives one null permutation of the real phenotype
    return y[np.argsort(Y_null, axis=0)]


# ─────────────────────────────────────────────────────────────────────────────
# threshold scans
# ─────────────────────────────────────────────────────────────────────────────


def run_threshold_scans(
    null_outcomes,
    genomecache,
    phenotype=None,
    covariates=None,
    kinship_decomp=None,
    kinship_loco=None,
    chrom=None,
    ref_chrom=None,
    permutation_method="parametric",
    use_rint=True,
    sample_ids=None,
    z_mat=None,
    seed=1,
    batch_size=500,
    gene_pos_mb=None,
    window_mb=None,
):
    """scan null phenotype columns to build the peak-LOD null distribution.

    h² is always fixed from the real phenotype (ML estimate) and reused across
    all permutations — the LOD scan is conditional on fixed h², so the permutation
    null must be too.

    parameters
    ----------
    null_outcomes  : np.ndarray (n, n_perms)  from generate_null_outcomes().
        in 'parametric' mode the array is used only to determine n_perms;
        its contents are ignored.
    genomecache   : GenomeCache
    phenotype      : np.ndarray (n,)  required; used to estimate h² per chromosome.
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
        if None, computed from genomecache (loco).
    kinship_loco   : dict or None  {chrom: np.ndarray (m, m)}
        raw LOCO kinship matrices (not decomposed).  used when
        permutation_method='parametric'; if None, reconstructed from
        kinship_decomp automatically.  ignored for 'plain'.
    chrom          : str or None
        if given, restrict scans to this chromosome only (validation mode).
        if None, scan all chromosomes (full-genome threshold).
    ref_chrom      : str or None
        chromosome whose LOCO kinship and h² are used to draw the null
        phenotype matrix in 'parametric' mode.  all chromosomes are then
        scanned with this single null matrix — this is correct for a
        genome-wide FWE threshold (see notes), since drawing an independent
        null per chromosome would inflate the genome-wide max by taking the
        max over many independent nulls instead of many scans of one null.
        if None, the first chromosome with a valid (positive semi-definite)
        kinship matrix is used.
    permutation_method : str  'parametric' (default) or 'plain'
        'parametric' — kinship-informed rank-permutation: draws u~N(0,K·σ²_g),
                       rank-permutes y by the null order.  correct default.
        'plain'      — simple shuffle; equivalent only when τ²≈0 (identity
                       kinship / OLS).  used by Phase 1 validation only.
    use_rint       : bool  apply rank-inverse normal transform to phenotype
    sample_ids     : list of str or None
        replicate sample IDs (length n).  when provided, each permuted
        phenotype is averaged to strain means via the Z-matrix before
        scanning; kinship_decomp must be at strain level (m×m).
        Mutually exclusive with z_mat: use one or the other, not both.
    z_mat          : np.ndarray (n, m) or None
        sample-to-strain design matrix for n-level scanning.  when provided,
        null_outcomes and kinship_decomp must already be at n-level (i.e.
        kinship_decomp holds eigdecomp of K_n = Z K_m Z^T).  z_mat is passed
        through to scan_chromosome_batch so haplotype probs are expanded from
        m cache strains to n samples inside each scan.  this ensures the null
        LOD scale matches a sample-level scan (LOD = (n/2)·log10(rss0/rss1))
        rather than a strain-mean scan (LOD = (m/2)·log10(rss0/rss1)).
        sample_ids must be None when z_mat is provided.
    seed           : int  rng seed for all internal random draws.
        'parametric' — seeds the null-draw rng inside _draw_parametric_batch.
        'plain'      — seed is unused here; reproducibility is controlled by
                       the seed passed to generate_null_outcomes by the caller.
    batch_size     : int  mini-batch size for the fast vectorised path
        (fix_h2=True + plain).  500 balances memory and loop overhead;
        reduce if RAM is tight.
    gene_pos_mb    : float or None  representative gene's TSS position in Mb.
        with window_mb, restricts the null scan on `chrom` to loci within
        ±window_mb of this position — matching a local/cis real scan so its
        permutation threshold is drawn from the same locus set instead of a
        genome-wide (or whole-chromosome) null, which would be miscalibrated
        for a scan that never considered those other loci in the first place.
        requires `chrom` to be given (the window anchors to one chromosome) —
        raises ValueError if `chrom` is None.
    window_mb      : float or None  half-window in Mb around gene_pos_mb.

    returns
    -------
    max_lods       : np.ndarray (n_perms,)  per-permutation genome-wide max LOD
    chr_null_lods  : dict {chr: np.ndarray (n_perms,)}
        per-chromosome null max-LOD distributions — already computed during the
        scan at no extra cost. use these to derive calibrated per-gene empirical
        p-values: p_g = mean(chr_null_lods[gene_chr] >= peak_lod_g), which avoids
        the conservative bias of comparing against the genome-wide maximum.

    notes
    -----
    fast path (fix_h2=True + permutation_method='plain'):
        precomputes H_perp and G_pinv once per chromosome then vectorises
        over all permutations in mini-batches via scan_chromosome_batch.
        O(n_perms × n_loci × J) vs O(n_perms × n_loci × J²) for slow path.
    slow path (fix_h2=False or permutation_method='parametric'):
        calls scan_chromosome once per permutation per chromosome.
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if permutation_method not in ("plain", "parametric"):
        raise ValueError(
            f"permutation_method must be 'plain' or 'parametric', got {permutation_method!r}"
        )
    if gene_pos_mb is not None and window_mb is not None and chrom is None:
        raise ValueError(
            "gene_pos_mb/window_mb require chrom (the local window anchors to "
            "one chromosome) — pass chrom=<gene's own chromosome> for a local scan"
        )

    _t0 = time.perf_counter()
    n_perms_req = null_outcomes.shape[1]
    _chrs_req = [chrom] if chrom is not None else [c for c in genomecache.chromosomes]
    logger.info(
        "run_threshold_scans: n_perms=%d  n_chr=%d  ref_chrom=%s  method=%s  fix_h2=True",
        n_perms_req,
        len(_chrs_req),
        ref_chrom if ref_chrom is not None else "auto",
        permutation_method,
    )

    # compute LOCO kinship if not already supplied
    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd)

    # rebuild the raw kinship matrix from its eigendecomposition when not supplied
    if permutation_method == "parametric" and kinship_loco is None:
        kinship_loco = {
            c: (
                kinship_decomp[c]["vectors"]
                * kinship_decomp[c]["values"][None, :]
                @ kinship_decomp[c]["vectors"].T
            )
            for c in kinship_decomp
        }

    chrs = (
        [chrom]
        if chrom is not None
        else [c for c in genomecache.chromosomes if c in kinship_decomp]
    )

    # default covariates to an intercept-only column if none given
    n = null_outcomes.shape[0]
    x = covariates if covariates is not None else np.ones((n, 1), dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    # sample_ids path: average phenotype/covariates to strain means and scan
    # at strain level. used when no z_mat is supplied (no replicate design).
    _z_red: "np.ndarray | None" = None
    _r_counts: "np.ndarray | None" = None
    x_scan = x
    if sample_ids is not None and z_mat is None:
        from ..utils import build_z_matrix

        _z_red, _ = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        _r_counts = _z_red.sum(axis=0)
        x_scan = (_z_red.T @ x) / _r_counts[:, None]

    # z_mat path: expand kinship from strain level to sample level per chromosome
    # (K_n = z_mat @ K_m @ z_mat.T) so the null LOD scale matches a sample-level
    # scan (LOD = (n/2)·log10(rss0/rss1)) instead of a strain-mean scan.
    _n_kd: "dict | None" = None
    if z_mat is not None:
        _n_kd = {}
        for c in chrs:
            ev_m = kinship_decomp[c]["values"]
            evc_m = kinship_decomp[c]["vectors"]
            K_m = (evc_m * ev_m) @ evc_m.T
            K_n = z_mat @ K_m @ z_mat.T
            ev_n, evc_n = np.linalg.eigh(K_n)
            _n_kd[c] = {"values": ev_n, "vectors": evc_n, "K_n": K_n}

    def _ev_evc(c: str) -> "tuple[np.ndarray, np.ndarray]":
        """return the eigendecomposition at sample level when z_mat is given, else strain level."""
        if _n_kd is not None:
            return _n_kd[c]["values"], _n_kd[c]["vectors"]
        return kinship_decomp[c]["values"], kinship_decomp[c]["vectors"]

    def _K_raw(c: str) -> np.ndarray:
        """return the raw kinship matrix used for parametric null draws."""
        if _n_kd is not None:
            return _n_kd[c]["K_n"]
        return kinship_loco[c]

    # estimate h² per chromosome once, from the real phenotype
    if phenotype is None:
        raise ValueError("phenotype required for h² estimation")
    y0 = _rint(phenotype) if use_rint else phenotype.copy()
    if _r_counts is not None:
        y0 = (_z_red.T @ y0) / _r_counts
    h2_fixed = {}
    for c in chrs:
        ev, evc = _ev_evc(c)
        h2_fixed[c], _ = lmm_module.estimate_h2_null(
            evc.T @ y0, evc.T @ x_scan, ev, method="ml"
        )

    _ref_chr_used = next(iter(h2_fixed)) if h2_fixed else "none"
    _ref_h2 = h2_fixed.get(_ref_chr_used, float("nan"))
    logger.info(
        "run_threshold_scans: h² estimated on %d chromosomes — ref_chr=%s  h2=%.4f",
        len(h2_fixed),
        _ref_chr_used,
        _ref_h2,
    )

    n_perms = null_outcomes.shape[1]
    max_lods = np.zeros(n_perms)
    rng = np.random.default_rng(seed) if permutation_method == "parametric" else None

    plain_mat = None
    if permutation_method == "plain":
        if use_rint:
            plain_mat = np.column_stack(
                [_rint(null_outcomes[:, i]) for i in range(n_perms)]
            )
        else:
            plain_mat = null_outcomes.astype(float)
        if _r_counts is not None:
            plain_mat = (_z_red.T @ plain_mat) / _r_counts[:, None]

    # parametric: draw the null phenotypes once, from the first valid chromosome's
    # kinship, then reuse the same draw for every chromosome (see ref_chrom above).
    if permutation_method == "parametric":
        null_mat_gw = None
        for c in chrs:
            try:
                ev_r, evc_r = _ev_evc(c)
                null_mat_gw = _draw_parametric_batch(
                    y0, x_scan, _K_raw(c), evc_r, ev_r, h2_fixed[c], n_perms, rng
                )
                break
            except KinshipSingularError as exc:
                logger.warning(
                    "run_threshold_scans: chr%s skipped for ref draw — %s", c, exc
                )
        if null_mat_gw is None:
            logger.warning(
                "run_threshold_scans: all chromosomes singular — returning zeros"
            )
            return max_lods, {c: np.zeros(len(max_lods)) for c in chrs}
    else:
        null_mat_gw = plain_mat

    chr_null_lods: dict = {}
    for c in chrs:
        ev, evc = _ev_evc(c)
        h2 = h2_fixed[c]
        loci_mask = None
        if gene_pos_mb is not None and window_mb is not None and c == chrom:
            _, positions_c, _ = genomecache.get_chromosome(c)
            loci_mask = np.abs(positions_c - gene_pos_mb) <= window_mb
        try:
            chr_max = lmm_module.scan_chromosome_batch(
                null_mat_gw,
                genomecache,
                c,
                x_scan,
                ev,
                evc,
                h2,
                z_mat=z_mat,
                batch_size=batch_size,
                loci_mask=loci_mask,
            )
        except KinshipSingularError as exc:
            logger.warning("run_threshold_scans: chr%s skipped — %s", c, exc)
            chr_null_lods[c] = np.zeros(n_perms_req)
            continue
        chr_null_lods[c] = chr_max
        max_lods = np.maximum(max_lods, chr_max)
        logger.info(
            "run_threshold_scans: chr%s  max_lod=%.3f  elapsed=%.1fs",
            c,
            float(chr_max.max()),
            time.perf_counter() - _t0,
        )

    _gw_threshold_95 = (
        float(np.quantile(max_lods[max_lods > 0], 0.95))
        if (max_lods > 0).any()
        else float("nan")
    )
    logger.info(
        "run_threshold_scans: done  n_perms=%d  genome-wide 95th pct LOD=%.3f  elapsed=%.1fs",
        n_perms_req,
        _gw_threshold_95,
        time.perf_counter() - _t0,
    )
    return max_lods, chr_null_lods


# ─────────────────────────────────────────────────────────────────────────────
# per-gene p-value calibration
# ─────────────────────────────────────────────────────────────────────────────


def calibrate_gene_pvalue(
    y_gene,
    genomecache,
    chrom,
    peak_lod,
    covariates=None,
    kinship_decomp=None,
    kinship_loco=None,
    n_founders=None,
    n_perms=200,
    gene_pos_mb=None,
    window_mb=None,
    use_rint=True,
    sample_ids=None,
    z_mat=None,
    seed=1,
):
    """calibrate one gene's peak LOD into a Beta-adjusted, genome-comparable p-value.

    a genome-wide threshold treats every gene as though it shares one null
    distribution, but each gene has its own fixed nuisance parameter (h² here)
    and therefore its own null LOD distribution. this draws a small parametric
    null specific to this gene — same h² and same cis window used for its real
    scan — then hands the resulting per-permutation max LOD to
    beta_calibrate_pvalue() for the Beta-smoothed adjustment. the returned
    p-value is Uniform(0, 1) under the null for this gene specifically, so
    p-values from many genes can be pooled and fed to storey_qvalue() for a
    genome-wide q-value.

    parameters
    ----------
    y_gene         : np.ndarray (n,)  this gene's expression/phenotype vector
    genomecache    : GenomeCache
    chrom          : str  chromosome to scan (the gene's cis chromosome)
    peak_lod       : float  this gene's real observed peak LOD on chrom
    covariates     : np.ndarray (n, p) or None  (intercept added automatically)
    kinship_decomp : dict  {chrom: {'values':, 'vectors':}} — required; build
        once genome-wide via kinship.calc_kinship + kinship.decomp_kinship and
        reuse across genes.
    kinship_loco   : dict  {chrom: np.ndarray (m, m)} — raw LOCO kinship,
        required for the parametric null draw.
    n_founders     : int or None  df for the chi² LRT (n_founders - 1); when
        None, taken from genomecache.n_founders.
    n_perms        : int  permutations for this gene's null (default 200 — far
        fewer than the genome-wide threshold's n_perms, since the Beta fit
        extrapolates the tail).
    gene_pos_mb    : float or None  gene TSS position in Mb; with window_mb,
        restricts the null scan to the cis window used for the real scan.
        None scans the whole chromosome.
    window_mb      : float or None  half-window in Mb around gene_pos_mb.
    use_rint       : bool  rank-inverse normal transform y_gene before scanning.
    sample_ids     : list of str or None  replicate sample IDs; see
        run_threshold_scans for the strain-mean reduction this triggers.
    z_mat          : np.ndarray (n, m) or None  sample-to-strain design matrix;
        mutually exclusive with sample_ids — see run_threshold_scans.
    seed           : int  rng seed for the null draw.

    returns
    -------
    p_adj : float  Beta-calibrated p-value for this gene's peak_lod
    a_hat : float  fitted Beta shape parameter
    b_hat : float  fitted Beta shape parameter
    h2_g  : float  this gene's own fixed h² used for the null draw
    """
    from . import fdr as fdr_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if kinship_decomp is None or kinship_loco is None:
        raise ValueError(
            "kinship_decomp and kinship_loco are required — build once "
            "genome-wide and reuse across genes rather than recomputing per gene"
        )

    n = len(y_gene)
    x = covariates if covariates is not None else np.ones((n, 1), dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    y0 = _rint(y_gene) if use_rint else np.asarray(y_gene, dtype=float).copy()

    x_scan = x
    _z_red = None
    if sample_ids is not None and z_mat is None:
        from ..utils import build_z_matrix

        _z_red, _ = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        r_counts = _z_red.sum(axis=0)
        y0 = (_z_red.T @ y0) / r_counts
        x_scan = (_z_red.T @ x) / r_counts[:, None]

    ev = kinship_decomp[chrom]["values"]
    evc = kinship_decomp[chrom]["vectors"]
    K = kinship_loco[chrom]

    h2_g, _ = lmm_module.estimate_h2_null(evc.T @ y0, evc.T @ x_scan, ev, method="ml")

    rng = np.random.default_rng(seed)
    null_mat = _draw_parametric_batch(y0, x_scan, K, evc, ev, h2_g, n_perms, rng)

    loci_mask = None
    if gene_pos_mb is not None and window_mb is not None:
        _, positions_c, _ = genomecache.get_chromosome(chrom)
        loci_mask = np.abs(positions_c - gene_pos_mb) <= window_mb

    null_max_lods = lmm_module.scan_chromosome_batch(
        null_mat,
        genomecache,
        chrom,
        x_scan,
        ev,
        evc,
        h2_g,
        z_mat=z_mat,
        loci_mask=loci_mask,
    )

    df = (n_founders if n_founders is not None else genomecache.n_founders) - 1
    p_adj, a_hat, b_hat = fdr_module.beta_calibrate_pvalue(peak_lod, null_max_lods, df)
    logger.debug(
        "calibrate_gene_pvalue: chr%s  n_perms=%d  h2=%.4f  peak_lod=%.3f  p_adj=%.3g",
        chrom,
        n_perms,
        h2_g,
        peak_lod,
        p_adj,
    )
    return p_adj, a_hat, b_hat, h2_g


# ─────────────────────────────────────────────────────────────────────────────
# gev threshold
# ─────────────────────────────────────────────────────────────────────────────


def get_gev_thresholds(
    max_lods: "np.ndarray",
    percentile: float = 0.95,
) -> "tuple[float, dict] | tuple[None, None]":
    """fit a GEV distribution to the permutation peak-LOD distribution.

    fitting a GEV curve gives a more stable threshold than an empirical
    quantile when n_perms is small (100-500).

    parameters
    ----------
    max_lods :
        per-permutation maximum LOD scores.
    percentile :
        significance percentile (default ``0.95`` = 5% genome-wide FWE).

    returns
    -------
    tuple[float, dict]
        LOD threshold at the requested percentile and a parameter dict with
        keys ``c``, ``loc``, ``scale``, ``threshold``, ``percentile``,
        ``n_perms``, ``empirical_quantile``.
    tuple[None, None]
        when there are fewer than 10 finite values or the GEV fit fails.
    """
    finite = max_lods[np.isfinite(max_lods)]
    if len(finite) < 10:
        logger.warning(
            "get_gev_thresholds: only %d finite values — need ≥10; returning None",
            len(finite),
        )
        return None, None
    try:
        c, loc, scale = genextreme.fit(finite)
        threshold = float(genextreme.ppf(percentile, c, loc=loc, scale=scale))
    except Exception as exc:
        _emp = float(np.quantile(finite, percentile))
        logger.warning(
            "get_gev_thresholds: GEV fit failed (%s) — returning None "
            "(empirical %.0f%% quantile was %.3f)",
            exc,
            percentile * 100,
            _emp,
        )
        return None, None

    params = {
        "c": float(c),
        "loc": float(loc),
        "scale": float(scale),
        "threshold": threshold,
        "percentile": percentile,
        "n_perms": len(finite),
        "empirical_quantile": float(np.quantile(finite, percentile)),
    }
    logger.debug(
        "get_gev_thresholds: n=%d  shape=%.4f  loc=%.4f  scale=%.4f  threshold=%.3f",
        len(finite),
        float(c),
        float(loc),
        float(scale),
        threshold,
    )
    return threshold, params


def load_per_chr_thresholds(
    perm_dir: "Path",
    percentile: float = 0.95,
) -> "dict[str, float | None]":
    """load per-chromosome null distributions and return GEV thresholds.

    reads ``perm_max_lods_chr{chrom}.csv`` files written by
    :func:`run_threshold_scans`, fits a GEV to each chromosome's max-LOD
    null, and returns per-chromosome FWER thresholds at *percentile*.

    useful when ``threshold.csv`` was written with only a ``genome_wide``
    summary row (checkpointed scans) and per-chromosome bars are needed for
    diagnostic plots.

    parameters
    ----------
    perm_dir :
        directory containing ``perm_max_lods_chr*.csv`` files (typically
        the ``core/`` subdirectory of a scan output).
    percentile :
        significance percentile; default ``0.95`` (5% FWER).

    returns
    -------
    dict[str, float | None]
        ``{chrom: threshold}``; value is ``None`` when the GEV fit fails.
    """
    from pathlib import Path as _Path

    perm_dir = _Path(perm_dir)
    result: "dict[str, float | None]" = {}
    for pf in sorted(perm_dir.glob("perm_max_lods_chr*.csv")):
        chrom = pf.stem.replace("perm_max_lods_chr", "")
        try:
            nulls = pd.read_csv(pf)["max_lod"].values
            thr, _ = get_gev_thresholds(nulls, percentile=percentile)
            result[chrom] = thr
        except Exception:
            result[chrom] = None
    return result


# ─────────────────────────────────────────────────────────────────────────────
# positional CI via parametric bootstrap
# ─────────────────────────────────────────────────────────────────────────────


def run_positional_scans(
    bootstrap_outcomes,
    genomecache,
    phenotype,
    covariates=None,
    kinship_decomp=None,
    peak_chrom=None,
    use_rint=True,
    ci_alpha=0.05,
    sample_ids=None,
):
    """run single-chromosome scans on parametric bootstrap samples to compute
    a positional confidence interval for a QTL.


    parameters
    ----------
    bootstrap_outcomes : np.ndarray (n, n_boot)  from generate_null_outcomes(method='bootstrap')
    genomecache       : GenomeCache
    phenotype          : np.ndarray (n,)  original phenotype for h² pre-estimation
    covariates         : np.ndarray (n, p) or None
    kinship_decomp     : dict or None  {chrom: {'values':, 'vectors':}}
    peak_chrom         : str or None
        chromosome to scan; if None, scans all and takes overall peak position.
    use_rint           : bool
    ci_alpha           : float  confidence level = 1 - ci_alpha (default 0.05 = 95% CI)
    sample_ids         : list of str or None
        replicate sample IDs (length n).  when provided, each bootstrap
        phenotype is averaged to strain means before scanning; kinship_decomp
        must be at strain level (m×m).

    returns
    -------
    np.ndarray (n_boot,)   bootstrap peak positions in Mb
    dict  {'ci_lo', 'ci_hi', 'ci': (lo, hi), 'n_boot', 'n_valid', 'alpha'}
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd)

    chrs = (
        [peak_chrom]
        if peak_chrom is not None
        else [c for c in genomecache.chromosomes if c in kinship_decomp]
    )

    _n_boot = bootstrap_outcomes.shape[1]
    logger.info(
        "run_positional_scans: n_boot=%d  n_chr=%d  chr=%s",
        _n_boot,
        len(chrs),
        peak_chrom if peak_chrom is not None else "all",
    )

    n = bootstrap_outcomes.shape[0]
    x = covariates if covariates is not None else np.ones((n, 1), dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    # replicate path: reduce to strain means if sample_ids provided
    z_mat = None
    r_counts = None
    x_scan = x
    if sample_ids is not None:
        from ..utils import build_z_matrix

        z_mat, _ = build_z_matrix(list(sample_ids), list(genomecache.sample_ids))
        r_counts = z_mat.sum(axis=0)
        x_scan = (z_mat.T @ x) / r_counts[:, None]

    # pre-compute h² from original phenotype
    y0 = _rint(phenotype) if use_rint else phenotype.copy()
    if z_mat is not None:
        y0 = (z_mat.T @ y0) / r_counts
    h2_fixed = {}
    for c in chrs:
        ev = kinship_decomp[c]["values"]
        evc = kinship_decomp[c]["vectors"]
        h2_fixed[c], _ = lmm_module.estimate_h2_null(
            evc.T @ y0, evc.T @ x_scan, ev, method="ml"
        )

    n_boot = bootstrap_outcomes.shape[1]

    # build the full boot matrix at strain level once — shape (n_scan, n_boot)
    Y_boot = bootstrap_outcomes.copy().astype(float)
    if use_rint:
        Y_boot = np.apply_along_axis(_rint, 0, Y_boot)
    if z_mat is not None:
        Y_boot = (z_mat.T @ Y_boot) / r_counts[:, None]

    # accumulate per-boot best (lod, pos) across chromosomes
    best_lods = np.full(n_boot, -np.inf)
    peak_pos = np.full(n_boot, np.nan)

    for c in chrs:
        ev = kinship_decomp[c]["values"]
        evc = kinship_decomp[c]["vectors"]
        h2 = h2_fixed[c]
        chr_max, chr_pos = lmm_module.scan_chromosome_batch(
            Y_boot,
            genomecache,
            c,
            x_scan,
            ev,
            evc,
            h2,
            return_peak_positions=True,
        )
        better = chr_max > best_lods
        best_lods = np.where(better, chr_max, best_lods)
        peak_pos = np.where(better, chr_pos, peak_pos)

    valid = peak_pos[np.isfinite(peak_pos)]
    lo = float(np.quantile(valid, ci_alpha / 2))
    hi = float(np.quantile(valid, 1 - ci_alpha / 2))
    ci_info = {
        "ci_lo": lo,
        "ci_hi": hi,
        "ci": (lo, hi),
        "n_boot": n_boot,
        "n_valid": len(valid),
        "alpha": ci_alpha,
    }
    logger.info(
        "run_positional_scans: done  n_boot=%d  CI=[%.3f, %.3f] Mb  width=%.3f Mb",
        _n_boot,
        lo,
        hi,
        hi - lo,
    )
    return peak_pos, ci_info


def ci_scan_multi_gene(
    expr_cache: "dict[str, np.ndarray]",
    gene_chr_map: "dict[str, str]",
    peak_mb_map: "dict[str, float]",
    genomecache,
    covariates: "np.ndarray | None" = None,
    kinship_decomp: "dict | None" = None,
    sample_ids: "list[str] | None" = None,
    n_boot: int = 200,
    seed: int = 1,
    use_rint: bool = True,
    output_path: "Path | None" = None,
    gene_meta: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """bootstrap positional CI for a set of genes, streaming to disk.

    calls generate_null_outcomes + run_positional_scans per gene, streams each
    CI row immediately — no accumulation of bootstrap matrices across genes.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map   : dict {gene_id: str}  chromosome per gene
    peak_mb_map    : dict {gene_id: float}  peak position in Mb (from S4 scan)
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids     : list[str] or None  genotype-level strain IDs
    n_boot         : int  bootstrap replicates per gene (default 200)
    seed           : int  base seed; incremented per gene
    use_rint       : bool
    output_path    : Path or None  stream CSV here; None accumulates in RAM
    gene_meta      : DataFrame keyed by 'gene_id' (the identity key; carries
                     gene_symbol and any other display columns) — extra
                     columns joined into each row

    returns
    -------
    pd.DataFrame  columns: gene_id, chr [+ gene_meta cols], peak_mb, ci_lo, ci_hi, ci_width_mb
    """
    _gm_idx = _build_gm_idx(gene_meta)
    _id_cols = ["gene_id", "chr"]
    _data_cols = ["peak_mb", "ci_lo", "ci_hi", "ci_width_mb"]
    _first = True
    _buf: "list[pd.DataFrame]" = []
    _n_genes_ci = len(expr_cache)
    logger.info(
        "ci_scan_multi_gene: n_genes=%d  n_boot=%d",
        _n_genes_ci,
        n_boot,
    )
    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, "")
        peak_mb = peak_mb_map.get(gene_id)
        if not gene_chr or peak_mb is None:
            logger.warning(
                "ci_scan_multi_gene: missing chr/peak_mb for %s — skipped", gene_id
            )
            continue
        logger.debug(
            "ci_scan_multi_gene: gene %d/%d (%.1f%%) %s",
            _g + 1,
            _n_genes_ci,
            100.0 * (_g + 1) / _n_genes_ci,
            gene_id,
        )
        _kd_g = (
            {gene_chr: kinship_decomp[gene_chr]} if kinship_decomp is not None else None
        )
        try:
            boot_out = generate_null_outcomes(
                y, n_perms=n_boot, seed=seed + _g, method="bootstrap"
            )
            ci_mbs, ci_info = run_positional_scans(
                boot_out,
                genomecache,
                phenotype=y,
                covariates=covariates,
                kinship_decomp=_kd_g,
                peak_chrom=gene_chr,
                use_rint=use_rint,
                sample_ids=sample_ids,
            )
            del boot_out, ci_mbs
            _df = pd.DataFrame(
                [
                    {
                        "gene_id": gene_id,
                        "chr": gene_chr,
                        "peak_mb": float(peak_mb),
                        "ci_lo": ci_info["ci_lo"],
                        "ci_hi": ci_info["ci_hi"],
                        "ci_width_mb": ci_info["ci_hi"] - ci_info["ci_lo"],
                    }
                ]
            )
            _df = _enrich_scan_df(_df, gene_id, _gm_idx, _id_cols, _data_cols)
            if output_path is not None:
                _df.to_csv(output_path, mode="a", header=_first, index=False)
                _first = False
            else:
                _buf.append(_df)
            del _df
        except Exception as exc:
            logger.warning("ci_scan_multi_gene: %s failed — %s", gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()

"""
permutation — genome-wide significance thresholds via permutation

implements genome-wide significance thresholds (GEV fit on max-LOD distribution)
and the parametric null draw used per-gene by qtl.fdr.calibrate_gene_pvalue.

design notes:
  - fix_h2=True (default) re-uses h² estimated from the real data across all
    permutations, saving n_perms × n_chrom calls to estimate_h2_null.
  - chrom kwarg restricts threshold scans to a single chromosome (fast single-
    gene mode used in validation); leave None for full-genome threshold.
  - permutation_method='parametric' (default) draws a kinship-informed null:
    u~N(0,K·σ²_g), e~N(0,σ²_e·I), then rank-permutes y by the null draw's
    order. correct for real LOCO kinship (Phase 2+).
    permutation_method='plain' shuffles y labels — valid only when τ²≈0
    (identity kinship / OLS); used by Phase 1 validation only.
"""

import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path


logger = logging.getLogger("mosaiqtl")

__all__ = [
    "KinshipSingularError",
    "generate_null_outcomes",
    "run_threshold_scans",
    "draw_parametric_null_batch",
    "pick_rep_gene",
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
    """draw a random effect u ~ N(0, K·sigma2_g) via eigendecomposition.

    projects K onto the PSD cone before drawing (clamps near-zero negative
    eigenvalues that arise from floating-point error in rank-deficient LOCO
    kinship matrices).  raises KinshipSingularError only when min_eig < -1e-4,
    which indicates a genuine singularity rather than numerical noise.
    """
    ev, evc = np.linalg.eigh(K)
    if float(ev.min()) < -1e-4:
        raise KinshipSingularError(
            f"LOCO kinship genuinely singular (min_eig={float(ev.min()):.2e})"
        )
    # project onto PSD cone; draw u = evc @ diag(sqrt(ev_+ * sigma2_g)) @ z
    sqrt_ev = np.sqrt(np.maximum(ev, 0.0) * sigma2_g)
    return evc @ (sqrt_ev * rng.standard_normal(len(ev)))


def draw_parametric_null_batch(
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

    uses the eigendecomposition (ev, evc) already in hand to project K onto the
    PSD cone, draw all random-effect vectors in one matrix multiply, then
    rank-permute y by each null column's order.

    raises KinshipSingularError only when min(ev) < -1e-4 (genuine singularity).

    returns
    -------
    np.ndarray (n, n_perms)  each column is y rank-permuted by one null draw
    """
    y_rot = evc.T @ y
    x_rot = evc.T @ x
    beta, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, ev, h2)

    if float(ev.min()) < -1e-4:
        raise KinshipSingularError(
            f"LOCO kinship genuinely singular (min_eig={float(ev.min()):.2e})"
        )
    # project onto PSD cone; draw U = evc @ diag(sqrt(ev_+ * sigma2_g)) @ Z
    sqrt_ev = np.sqrt(np.maximum(ev, 0.0) * sigma2_g)
    U_mat = evc @ (sqrt_ev[:, None] * rng.standard_normal((len(ev), n_perms)))
    E_mat = rng.normal(0.0, np.sqrt(max(sigma2_e, 0.0)), size=(len(y), n_perms))
    Y_null = (x @ beta)[:, None] + U_mat + E_mat

    # sample i takes the y value whose rank equals the rank of Y_null[i], so each
    # column keeps the null draw's kinship-induced ordering of the real values
    ranks = np.argsort(np.argsort(Y_null, axis=0), axis=0)
    return np.sort(y)[ranks]


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
    h2=None,
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
        primary kinship input: eigendecomposition of the strain-level LOCO
        kinship.  supplies the eigenvectors for the h² estimate and the scan,
        and, with z_mat, the K_m that is expanded to K_n.  if None, computed
        from genomecache (loco).
    kinship_loco   : dict or None  {chrom: np.ndarray (m, m)}
        optional pre-built raw form of the same LOCO kinship that
        kinship_decomp factors; it must describe the same matrices.  used
        only for the strain-level parametric draw, where it avoids rebuilding
        K from kinship_decomp.  if None, rebuilt from kinship_decomp.
        ignored for 'plain' and whenever z_mat is given (K_n is formed from
        kinship_decomp).
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
        'parametric' — seeds the null-draw rng inside draw_parametric_null_batch.
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
    h2             : float or None  fixed h² to use for all chromosomes.
        when provided, skips the internal ML re-estimation of h² from the
        phenotype (which uses the null model and can differ substantially from
        the scan h² when there is a strong QTL on the ref chromosome).  pass
        the scan-estimated h² (e.g. median h² across genes from allele_effects)
        to ensure the null draw is calibrated at the same heritability as the
        real scans.

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

        # Rare cell types may have zero cells for some strains; dividing by zero
        # produces NaN that propagates to the scan.  Replace zero counts with 1
        # (safe divisor) — those strains' Z.T@y and Z.T@x sums are already 0, so
        # 0/1 = 0 = RINT mean, which is the correct mean imputation.  Covariates
        # for missing strains are then set to the valid-strain mean with the
        # intercept forced to 1 so the FWL projection stays well-defined.
        _valid_strains = _r_counts > 0
        if not _valid_strains.all():
            n_dropped = int((~_valid_strains).sum())
            logger.warning(
                'run_threshold_scans: %d strain(s) with no replicates — '
                'mean-imputing for permutation null; remaining=%d',
                n_dropped, int(_valid_strains.sum()),
            )
            _r_counts = np.where(_valid_strains, _r_counts, 1.0)
        x_scan = (_z_red.T @ x) / _r_counts[:, None]
        if not _valid_strains.all():
            x_scan[~_valid_strains] = x_scan[_valid_strains].mean(axis=0)
            x_scan[~_valid_strains, 0] = 1.0

    # z_mat path: expand kinship to sample level (K_n = Z K_m Z^T)
    _n_kd: "dict | None" = None
    if z_mat is not None:
        _n_kd = {}
        for c in chrs:
            ev_m = kinship_decomp[c]["values"]
            evc_m = kinship_decomp[c]["vectors"]
            K_m = (evc_m * ev_m) @ evc_m.T
            K_n = z_mat @ K_m @ z_mat.T
            ev_n, evc_n = np.linalg.eigh(K_n)
            ev_n = np.maximum(ev_n, 0.0)  # project onto PSD cone
            K_n = (evc_n * ev_n) @ evc_n.T
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
        if h2 is not None:
            h2_fixed[c] = float(h2)
        else:
            ev, evc = _ev_evc(c)
            h2_fixed[c], _ = lmm_module.estimate_h2_null(
                evc.T @ y0, evc.T @ x_scan, ev, method="ml"
            )

    _ref_chr_used = (
        ref_chrom if ref_chrom in h2_fixed else next(iter(h2_fixed), "none")
    )
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
        ref_order = ([ref_chrom] if ref_chrom in chrs else []) + [
            c for c in chrs if c != ref_chrom
        ]
        for c in ref_order:
            try:
                ev_r, evc_r = _ev_evc(c)
                null_mat_gw = draw_parametric_null_batch(
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
# representative gene selection for permutation threshold calibration
# ─────────────────────────────────────────────────────────────────────────────


def _strand_aware_tss_mb(gene_id: str, gene_tss_df: "pd.DataFrame") -> "float | None":
    """Return the strand-aware TSS in Mb for gene_id, or None if not found.

    Prefers a pre-computed tss_mb column when present (avoids re-derivation).
    Otherwise derives from start/end + strand: TSS = start for + strand genes,
    end for − strand genes.  gene_tss_df must carry gene_id plus either tss_mb
    (Mb) or (start, end, strand) in bp.  Returns None when the gene is absent.
    """
    rows = gene_tss_df[gene_tss_df["gene_id"] == gene_id]
    if rows.empty:
        return None
    row = rows.iloc[0]
    if "tss_mb" in rows.columns and not pd.isna(row["tss_mb"]):
        return float(row["tss_mb"])
    has_strand = "strand" in rows.columns and not pd.isna(row["strand"])
    if has_strand and str(row["strand"]) == "-":
        return float(row["end"]) / 1e6
    return float(row["start"]) / 1e6


def pick_rep_gene(
    peak_lod_df: "pd.DataFrame",
    chrom: "str | None" = None,
    gene_tss_df: "pd.DataFrame | None" = None,
) -> "tuple[str, str, float | None, float]":
    """Pick the gene whose h² is nearest the median as the permutation representative.

    The permutation threshold is drawn from a null parameterised by h² (via
    fix_h2=True in run_threshold_scans).  Using the median-h² gene ensures the
    null max-LOD distribution is representative of the population, not biased
    toward high-signal or low-signal extremes.

    peak_lod_df must carry the columns written by PEAK_LOD_COLS: gene_id, chr,
    start_mb, h2.  h² is written by the scan engine (scan_h2lmm /
    scan_multi_gene) and is present in every peak_lod.csv.

    TSS resolution (used by L-mode local-window null):
      - When gene_tss_df is provided (gene_tss.vm36.csv or gene_info.vm36.csv),
        the strand-aware TSS is computed: end/1e6 for − strand, start/1e6 for +.
        This is more accurate than start_mb for genes whose TSS is at the 3' end
        of the chromosome coordinate.
      - When gene_tss_df is None, start_mb from peak_lod_df is used as a proxy.
        For ±1 Mb windows the error is at most one gene-body length (~0.05 Mb
        typical), so this fallback is acceptable when gene_tss is unavailable.

    The chrom argument also governs C-mode: for scan_type='C', pass the
    representative gene's chromosome to restrict the permutation null to that
    chromosome (matching the per-chromosome scan each gene actually runs).  For
    scan_type='G' pass None (genome-wide null).  For scan_type='L' pass the
    scan chromosome and also use the returned tss_mb as gene_pos_mb in
    run_threshold_scans.

    Args:
        peak_lod_df: per-gene peak-LOD table; must have gene_id, chr, start_mb,
            h2 columns.  Genes with NaN h2 are excluded.
        chrom: restrict candidate genes to this chromosome when not None.  For
            L-mode pass the scan chromosome; for C-mode pass the representative
            gene's chromosome; for G-mode pass None.
        gene_tss_df: optional table with gene_id, chr, start, end, strand
            columns (bp coordinates) used to resolve the strand-aware TSS.
            gene_tss.vm36.csv and gene_info.vm36.csv both satisfy this contract.
            When None, tss_mb falls back to start_mb from peak_lod_df.

    Returns:
        Tuple (gene_id, chrom, tss_mb, h2_median).
        gene_id  : Ensembl ID of the representative gene.
        chrom    : chromosome of that gene (str).
        tss_mb   : strand-aware TSS in Mb when gene_tss_df is provided, else
                   start_mb from peak_lod_df (None only when gene_tss_df is
                   provided but the gene is absent from it — callers that need a
                   position should fall back to start_mb in that case).
        h2_median: median h² across the candidate gene set.

    Raises:
        ValueError: if peak_lod_df is missing required columns or no gene with
            finite h2 remains after filtering.
    """
    required = {"gene_id", "chr", "start_mb", "h2"}
    missing = required - set(peak_lod_df.columns)
    if missing:
        raise ValueError(
            f"pick_rep_gene: peak_lod_df missing columns: {sorted(missing)}"
        )

    df = peak_lod_df.copy()
    df["chr"] = df["chr"].astype(str)
    if chrom is not None:
        df = df[df["chr"] == str(chrom)]

    df = df[df["h2"].notna() & np.isfinite(df["h2"].astype(float))]
    if df.empty:
        raise ValueError(
            f"pick_rep_gene: no genes with finite h² "
            f"{'on chr' + str(chrom) if chrom else 'in peak_lod_df'}"
        )

    h2_vals = df["h2"].astype(float)
    h2_med = float(h2_vals.median())
    idx = (h2_vals - h2_med).abs().values.argmin()
    row = df.iloc[idx]

    gene_id  = str(row["gene_id"])
    gene_chr = str(row["chr"])
    start_mb = float(row["start_mb"])

    tss_mb: "float | None"
    if gene_tss_df is not None:
        tss_mb = _strand_aware_tss_mb(gene_id, gene_tss_df)
        if tss_mb is None:
            logger.warning(
                "pick_rep_gene: %s not found in gene_tss_df — using start_mb=%.3fMb",
                gene_id, start_mb,
            )
            tss_mb = start_mb
    else:
        tss_mb = start_mb

    logger.info(
        "pick_rep_gene: median h²=%.3f  gene=%s  chr%s  TSS=%.3fMb  h²=%.3f",
        h2_med, gene_id, gene_chr, tss_mb, float(row["h2"]),
    )
    return gene_id, gene_chr, tss_mb, h2_med



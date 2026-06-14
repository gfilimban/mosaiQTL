"""
mediation — genome-wide mediation scan

tests whether a haplotype QTL effect on outcome gene y is mediated through
an intermediate gene m. the qr decomposition of [m | X] is precomputed once;
the per-locus loop is vectorised via the FWL identity (same trick as scan.py).

lod_drop = lod_base − lod_mediated
  > 0  mediation detected (haplotype effect shrinks when m is added)
  < 0  amplification (shared upstream regulator)
"""

import gc
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy import stats as sp_stats
from ..utils import build_z_matrix

logger = logging.getLogger('mosaiqtl')


def select_cis_mediation_pairs(
    egene_table: pd.DataFrame,
    gene_info: pd.DataFrame,
    window_mb: float = 10.0,
    gene_col: str = 'gene',
    chr_col: str = 'chr',
    peak_mb_col: str = 'peak_mb',
    info_gene_col: str = 'gene_id',
    info_chr_col: str = 'chr',
    info_start_col: str = 'start',
) -> 'list[tuple[str, str, str]]':
    """Select (outcome, mediator, chr) pairs for cis-mediation analysis.

    For each eGene (outcome), candidate mediators are ALL expressed genes on the
    same chromosome whose TSS falls within window_mb of the outcome's peak locus —
    not limited to other eGenes. This avoids circular filtering (using significance
    to choose who can mediate significance).

    Parameters
    ----------
    egene_table  : DataFrame with [gene, chr, peak_mb] columns (minimum).
        Build via pipeline.call_egenes(effects_df=...) to add chr/peak_mb.
    gene_info    : DataFrame of all expressed genes with columns
        [gene_id, chr, start] where start is the TSS position in base-pairs.
        Typically loaded from gene_info.csv produced by convert_gastroc_h5ad.py.
    window_mb    : half-window in Mb around the outcome's peak locus; any gene
        whose TSS/start falls within [peak_mb - window_mb, peak_mb + window_mb]
        is a candidate mediator (default 10.0 Mb).
    gene_col     : egene_table column for gene name
    chr_col      : egene_table column for chromosome
    peak_mb_col  : egene_table column for peak locus position in Mb
    info_gene_col: gene_info column for gene identifier
    info_chr_col : gene_info column for chromosome
    info_start_col: gene_info column for TSS / gene start (base-pairs)

    Returns
    -------
    list of (outcome_gene, mediator_gene, chr) tuples — outcome != mediator
    """
    required = [chr_col, peak_mb_col]
    missing = [c for c in required if c not in egene_table.columns]
    if missing:
        raise ValueError(
            f'egene_table is missing columns {missing}; '
            f'call pipeline.call_egenes(effects_df=...) to add them')

    gi = gene_info.copy()
    gi[info_chr_col] = gi[info_chr_col].astype(str)
    gi['_tss_mb'] = gi[info_start_col] / 1e6

    pairs: list[tuple[str, str, str]] = []
    for _, row in egene_table.iterrows():
        outcome = row[gene_col]
        ch = str(row[chr_col])
        if pd.isna(row.get(peak_mb_col)):
            continue
        peak = float(row[peak_mb_col])
        lo, hi = peak - window_mb, peak + window_mb

        candidates = gi[
            (gi[info_chr_col] == ch) &
            (gi['_tss_mb'] >= lo) &
            (gi['_tss_mb'] <= hi) &
            (gi[info_gene_col] != outcome)
        ]
        for med in candidates[info_gene_col]:
            pairs.append((outcome, str(med), ch))
    return pairs


def _wls_null_rss(y_rot, x_rot, w_inv):
    """null RSS under weighted OLS: RSS₀ = ||M_X y||²_W."""
    xw = x_rot * w_inv[:, None]
    beta = np.linalg.lstsq(xw.T @ x_rot, xw.T @ y_rot, rcond=None)[0]
    e = y_rot - x_rot @ beta
    return float(e @ (e * w_inv)), beta


def _batch_lod(y_rot, x_rot, probs_chr, w_inv, n, rss0, valid):
    """vectorised LOD over all loci on one chromosome.

    probs_chr : (L, n_strains, J) — caller passes all J founders; row-subsetted
                by valid inside. pinv handles rank J-1 (sum-to-one) or J-2 (one
                near-zero founder), matching r's rank-tolerant QR in lm.fit().
    returns   : lod (L,), p_value (L,), betas (L, J)
    """
    p  = x_rot.shape[1]

    H  = probs_chr[:, valid, :]               # (L, n, J) all founders

    # FWL: residualise H on X
    xt_pinv  = np.linalg.lstsq(x_rot.T @ x_rot, x_rot.T, rcond=None)[0]  # (p,n)
    proj     = np.einsum('pn,lnj->lpj', xt_pinv, H, optimize=True)        # (L,p,J)
    H_perp   = H - np.einsum('np,lpj->lnj', x_rot, proj, optimize=True)   # (L,n,J)

    c    = np.einsum('lnj,n->lj', H_perp, y_rot * w_inv, optimize=True)   # (L,J)
    # Gram matrices (L, J, J)
    G    = np.einsum('lnj,lnk->ljk', H_perp * w_inv[None,:,None], H_perp,
                     optimize=True)
    # pinv handles rank J-1 (normal) and J-2 (near-zero founder) per locus
    betas = (np.linalg.pinv(G, rcond=1e-10) @ c[..., None]).squeeze(-1)   # (L,J)

    rss_red = np.einsum('lj,lj->l', c, betas)                              # (L,)
    rss1    = np.maximum(rss0 - rss_red, 1e-30)
    lod     = np.where(rss1 < rss0, (n / 2.0) * np.log10(rss0 / rss1), 0.0)

    q_eff   = np.linalg.matrix_rank(G)                                     # (L,)
    df2_arr = np.maximum(n - p - q_eff, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        f_stat = np.where(q_eff > 0,
                          np.maximum((rss0-rss1)/np.maximum(q_eff,1)/(rss1/df2_arr),
                                     0.0), 0.0)
        pval   = np.where(df2_arr > 0,
                          1.0 - sp_stats.f.cdf(f_stat, q_eff, df2_arr),
                          1.0)

    return lod, pval, betas


def mediation_scan_qr(outcome, mediator, genomecache,
                       covariates=None, kinship_decomp=None,
                       use_rint=True, use_reml=True,
                       outcome_id='outcome', mediator_id='mediator',
                       sample_ids=None,
                       scan_window='genome', gene_chr=None,
                       gene_pos_mb=None, window_mb=1.0,
                       base_lod: 'dict[str, np.ndarray] | None' = None):
    """genome-wide mediation scan — vectorised per chromosome.

    tests: does the haplotype effect on outcome persist after conditioning
    on mediator? lod_drop > 0 means effect is (partially) mediated.

    parameters
    ----------
    outcome, mediator : np.ndarray (n,)
    genomecache      : GenomeCache
    covariates        : np.ndarray (n, p) or None
    kinship_decomp    : dict or None
    use_rint          : bool  apply RINT to outcome
    use_reml          : bool
    outcome_id        : str
    mediator_id       : str
    sample_ids        : array-like (n,) or None
        genotype-level strain IDs for each sample. when provided and
        different from genomecache.sample_ids, builds a Z matrix and
        reduces outcome/mediator/covariates to strain means before scanning.
    scan_window       : str  'genome' | 'cis' | 'local'  (default 'genome')
    gene_chr          : str or None  required for 'cis' and 'local'
    gene_pos_mb       : float or None  required for 'local'
    window_mb         : float  half-window for 'local' (default 1.0)
    base_lod          : dict {chrom: np.ndarray (n_loci,)} or None
        pre-computed base LOD scores for each chromosome.  when supplied the
        base scan is skipped entirely — only the mediated scan is run.  use
        this when calling for multiple mediators of the same outcome to avoid
        recomputing the identical base scan each time.

    returns
    -------
    pd.DataFrame  locus, chr, pos_mb, lod_base, lod_mediated, lod_drop, p_value
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if scan_window in ('cis', 'local') and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    # --- sample alignment via Z matrix ---
    cache_ids = list(genomecache.sample_ids)
    sid_list  = list(sample_ids) if sample_ids is not None else cache_ids
    if sid_list != cache_ids:
        z_mat, valid_sid = build_z_matrix(sid_list, cache_ids)
    else:
        z_mat, valid_sid = None, np.ones(len(sid_list), dtype=bool)

    valid  = (~np.isnan(outcome)) & (~np.isnan(mediator)) & valid_sid
    y      = outcome[valid].copy()
    m      = mediator[valid].copy()
    if use_rint:
        y = _rint(y)

    x_base = (covariates[valid].copy() if covariates is not None
              else np.ones((valid.sum(), 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(y), 1)), x_base])

    # --- reduce to strain means when Z matrix present ---
    z_sub = z_mat[valid] if z_mat is not None else None
    if z_sub is not None:
        r_counts = z_sub.sum(axis=0)                      # (m,)
        y        = (z_sub.T @ y) / r_counts               # (m,)
        m        = (z_sub.T @ m) / r_counts               # (m,)
        x_base   = (z_sub.T @ x_base) / r_counts[:, None]  # (m, p)
        ev_valid = slice(None)       # eigenvectors and probs already at strain level
    else:
        ev_valid = valid             # subset sample-level arrays by NaN mask

    n  = len(y)

    m_std = (m - m.mean()) / max(m.std(), 1e-10)
    x_med = np.hstack([m_std[:, None], x_base])   # (n, p+1)

    # --- chromosome list ---
    chroms = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ('cis', 'local'):
        _gc = str(gene_chr)
        chroms = [c for c in chroms if c == _gc]

    rows_base = []

    for chrom in chroms:
        ev  = kinship_decomp[chrom]['values']
        evc = kinship_decomp[chrom]['vectors']
        ev_v = evc[ev_valid]                         # (n, n_eigs)

        # rotate + weight — sqrt(w_inv) applied for WLS formulation

        def _rot_wt(x_cov, h2):
            """Rotate and weight x_cov by eigendecomp of kinship for given h2."""
            delta = h2 / max(1 - h2, 1e-10)
            w_inv = 1.0 / (delta * ev + 1.0)
            w_sqrt = np.sqrt(w_inv)
            return (ev_v.T @ y) * w_sqrt, (ev_v.T @ x_cov) * w_sqrt[:, None], w_inv, w_sqrt

        h2_med, _ = lmm_module.estimate_h2_null(ev_v.T @ y, ev_v.T @ x_med, ev)
        y_rot_m, x_rot_m, w_inv_m, w_sqrt_m = _rot_wt(x_med, h2_med)
        rss0_m, _ = _wls_null_rss(y_rot_m, x_rot_m, np.ones(n))

        lids, pos, probs = genomecache.get_chromosome(chrom)  # (L, n, J)
        H_raw   = probs[:, ev_valid, :]                        # (L, n, J)
        H_rot_m = np.einsum('nk,lnj->lkj', ev_v, H_raw, optimize=True) \
                  * w_sqrt_m[None, :, None]

        lod_m, pv, _ = _batch_lod(
            y_rot_m, x_rot_m, H_rot_m, np.ones(n), n, rss0_m, slice(None))

        if base_lod is not None and chrom in base_lod:
            lod_b = base_lod[chrom]
        else:
            h2_base, _ = lmm_module.estimate_h2_null(ev_v.T @ y, ev_v.T @ x_base, ev)
            y_rot_b, x_rot_b, w_inv_b, w_sqrt_b = _rot_wt(x_base, h2_base)
            rss0_b, _ = _wls_null_rss(y_rot_b, x_rot_b, np.ones(n))
            H_rot_b   = np.einsum('nk,lnj->lkj', ev_v, H_raw, optimize=True) \
                        * w_sqrt_b[None, :, None]
            lod_b, _, _ = _batch_lod(
                y_rot_b, x_rot_b, H_rot_b, np.ones(n), n, rss0_b, slice(None))

        lod_drop = np.maximum(lod_b - lod_m, 0.0)

        df = pd.DataFrame({
            'locus':        lids,
            'chr':          chrom,
            'pos_mb':       pos.astype(float),
            'lod_base':     lod_b,
            'lod_mediated': lod_m,
            'lod_drop':     lod_drop,
            'p_value':      pv,
            'outcome_id':   outcome_id,
            'mediator_id':  mediator_id,
        })
        rows_base.append(df)

    return pd.concat(rows_base, ignore_index=True) if rows_base else pd.DataFrame()


def mediation_scan_qr_batch(
        outcome: 'np.ndarray',
        mediators: 'dict[str, np.ndarray]',
        genomecache,
        covariates: 'np.ndarray | None' = None,
        kinship_decomp: 'dict | None' = None,
        use_rint: bool = True,
        outcome_id: str = 'outcome',
        sample_ids: 'list | None' = None,
        scan_window: str = 'genome',
        gene_chr: 'str | None' = None,
        gene_pos_mb: 'float | None' = None,
        window_mb: float = 1.0,
        base_lod: 'dict[str, np.ndarray] | None' = None,
) -> pd.DataFrame:
    """Batch mediation scan — all mediators of one outcome in a single pass.

    Vectorises over mediators using the Woodbury rank-1 identity, avoiding
    repeated FWL projection and REML estimation per mediator.

    h2 is fixed from the base model ``y ~ X`` (same approximation as
    ``fix_h2=True`` in permutation).  Empirically the h2 drift from adding
    one mediator covariate is ≤ 0.03, and Pearson r(exact, fixed) ≥ 0.9999
    across tested pairs (see check_mediation_h2_approx.py).

    Parameters
    ----------
    outcome   : np.ndarray (n,) — outcome gene expression vector.
    mediators : dict {mediator_id: np.ndarray (n,)} — all candidate mediators.
    All other parameters match ``mediation_scan_qr``.

    Returns
    -------
    pd.DataFrame with columns:
        locus, chr, pos_mb, lod_base, lod_mediated, lod_drop,
        outcome_id, mediator_id
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    if scan_window in ('cis', 'local') and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    # --- sample alignment ---
    cache_ids = list(genomecache.sample_ids)
    sid_list  = list(sample_ids) if sample_ids is not None else cache_ids
    if sid_list != cache_ids:
        z_mat, valid_sid = build_z_matrix(sid_list, cache_ids)
    else:
        z_mat, valid_sid = None, np.ones(len(sid_list), dtype=bool)

    med_ids   = list(mediators.keys())
    med_arrs  = [mediators[k] for k in med_ids]
    n_med     = len(med_ids)

    # stack mediators → (n_med, n_samples) for joint masking
    M_raw = np.stack(med_arrs, axis=0)                          # (n_med, n_samples)

    valid = (~np.isnan(outcome)) & valid_sid
    for i in range(n_med):
        valid &= ~np.isnan(M_raw[i])

    y = outcome[valid].copy()
    M = M_raw[:, valid].copy()                                  # (n_med, n_valid)
    if use_rint:
        y = _rint(y)

    x_base = (covariates[valid].copy() if covariates is not None
              else np.ones((valid.sum(), 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(y), 1)), x_base])

    # reduce to strain means when Z matrix present
    z_sub = z_mat[valid] if z_mat is not None else None
    if z_sub is not None:
        r_counts = z_sub.sum(axis=0)
        y        = (z_sub.T @ y) / r_counts
        M        = (z_sub.T @ M.T).T / r_counts[None, :]       # (n_med, n_strains)
        x_base   = (z_sub.T @ x_base) / r_counts[:, None]
        ev_valid  = slice(None)
    else:
        ev_valid  = valid

    n = len(y)

    # standardise mediators once
    M_std = (M - M.mean(axis=1, keepdims=True)) / np.maximum(
        M.std(axis=1, keepdims=True), 1e-10)                    # (n_med, n)

    # chromosome list
    chroms = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ('cis', 'local'):
        chroms = [c for c in chroms if c == str(gene_chr)]

    rows: list[pd.DataFrame] = []
    n_chroms = len(chroms)

    for _ci, chrom in enumerate(chroms):
        _t0_chr = time.perf_counter()
        ev  = kinship_decomp[chrom]['values']
        evc = kinship_decomp[chrom]['vectors']
        ev_v = evc[ev_valid]                                     # (n, n)

        # fix h2 from base model
        h2, _ = lmm_module.estimate_h2_null(ev_v.T @ y, ev_v.T @ x_base, ev)
        delta  = h2 / max(1 - h2, 1e-10)
        w_inv  = 1.0 / (delta * ev + 1.0)
        w_sqrt = np.sqrt(w_inv)

        y_rot  = (ev_v.T @ y)      * w_sqrt                     # (n,)
        x_rot  = (ev_v.T @ x_base) * w_sqrt[:, None]            # (n, p)

        # --- base FWL quantities (shared across all mediators) ---
        xt_gram  = x_rot.T @ x_rot                              # (p, p)
        xt_pinv  = np.linalg.lstsq(xt_gram, x_rot.T, rcond=None)[0]   # (p, n)

        y_perp   = y_rot - x_rot @ (xt_pinv @ y_rot)           # (n,)
        rss0_base = float(y_perp @ y_perp)

        lids, pos, probs = genomecache.get_chromosome(chrom)    # (L, n_strains, J)
        H_rot  = np.einsum('nk,lnj->lkj', ev_v, probs,
                            optimize=True) * w_sqrt[None, :, None]  # (L, n, J)

        proj_H   = np.einsum('pn,lnj->lpj', xt_pinv, H_rot, optimize=True)
        H_perp_X = H_rot - np.einsum('np,lpj->lnj', x_rot, proj_H, optimize=True)
        # (L, n, J)

        G_base = np.einsum('lnj,lnk->ljk', H_perp_X, H_perp_X, optimize=True)
        c_base = np.einsum('lnj,n->lj',    H_perp_X, y_perp,    optimize=True)
        G_inv  = np.linalg.pinv(G_base, rcond=1e-10)            # (L, J, J)
        q_base = np.einsum('ljk,lk->lj',  G_inv, c_base,       optimize=True)

        # base LOD (reuse precomputed if available)
        if base_lod is not None and chrom in base_lod:
            lod_b = base_lod[chrom]
        else:
            rss_red_base = np.einsum('lj,lj->l', c_base, q_base)
            rss1_base    = np.maximum(rss0_base - rss_red_base, 1e-30)
            lod_b = np.where(rss1_base < rss0_base,
                             (n / 2.0) * np.log10(rss0_base / rss1_base), 0.0)

        # --- mediator batch ---
        # rotate and residualise all mediators on X at once
        M_rot    = np.einsum('in,nk->ik', M_std, ev_v,
                              optimize=True) * w_sqrt[None, :]   # (n_med, n)
        proj_M   = xt_pinv @ M_rot.T                            # (p, n_med)
        M_perp_X = M_rot - (x_rot @ proj_M).T                  # (n_med, n)

        norms    = np.einsum('in,in->i', M_perp_X, M_perp_X)   # (n_med,)
        safe_n   = np.maximum(norms, 1e-30)
        y_dot_m  = M_perp_X @ y_perp                           # (n_med,)
        alpha    = y_dot_m / safe_n                             # (n_med,)
        rss0_m   = rss0_base - y_dot_m ** 2 / safe_n           # (n_med,)

        # Woodbury rank-1: H_perp_X projected onto each mediator residual
        # H_dot_M[l, j, i] = H_perp_X[l, :, j] . M_perp_X[i, :]
        H_dot_M = np.einsum('lnj,in->lji', H_perp_X, M_perp_X, optimize=True)
        # (L, J, n_med)

        # V[l, j, i] = (G_inv[l] @ H_dot_M[l, :, i])[j]
        V = np.einsum('ljk,lki->lji', G_inv, H_dot_M, optimize=True)   # (L, J, n_med)

        denom = safe_n[None, :] - np.einsum('lji,lji->li', H_dot_M, V)  # (L, n_med)
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

        # mediated c and q per (locus, mediator)
        c_lm = c_base[:, :, None] - alpha[None, None, :] * H_dot_M     # (L, J, n_med)
        q_lm = q_base[:, :, None] - alpha[None, None, :] * V           # (L, J, n_med)

        base_quad  = np.einsum('lji,lji->li', c_lm, q_lm)              # (L, n_med)
        v_dot_q    = np.einsum('lji,lji->li', H_dot_M, q_lm)           # (L, n_med)
        correction = v_dot_q ** 2 / denom                               # (L, n_med)
        rss_red    = base_quad + correction                              # (L, n_med)

        rss1_m = np.maximum(rss0_m[None, :] - rss_red, 1e-30)          # (L, n_med)
        lod_m  = np.where(
            rss1_m < rss0_m[None, :],
            (n / 2.0) * np.log10(np.maximum(rss0_m[None, :], 1e-30) / rss1_m),
            0.0,
        )                                                                # (L, n_med)

        lod_drop = np.maximum(lod_b[:, None] - lod_m, 0.0)             # (L, n_med)

        # unstack to long DataFrame
        for i, mid in enumerate(med_ids):
            rows.append(pd.DataFrame({
                'locus':        lids,
                'chr':          chrom,
                'pos_mb':       pos.astype(float),
                'lod_base':     lod_b,
                'lod_mediated': lod_m[:, i],
                'lod_drop':     lod_drop[:, i],
                'outcome_id':   outcome_id,
                'mediator_id':  mid,
            }))
        logger.debug('[%d/%d] mediation chr%s  %d loci × %d mediators  %.1fs',
                     _ci + 1, n_chroms, chrom, len(lids), n_med,
                     time.perf_counter() - _t0_chr)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fixef_mediation_scan(outcome, mediator, genomecache,
                          covariates=None, use_rint=True,
                          outcome_id='outcome', mediator_id='mediator',
                          just_these_loci=None):
    """fixed-effect mediation scan — OLS, no kinship, vectorised.

    fast screening version. compatible with just_these_loci filter
    for cis-window restriction.

    returns pd.DataFrame with same columns as mediation_scan_qr().
    """
    from ..utils import rint as _rint

    valid  = (~np.isnan(outcome)) & (~np.isnan(mediator))
    y      = outcome[valid].copy()
    m      = mediator[valid].copy()
    if use_rint:
        y = _rint(y)
    n  = len(y)

    x_base = (covariates[valid].copy() if covariates is not None
              else np.ones((n, 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((n, 1)), x_base])
    m_std  = (m - m.mean()) / max(m.std(), 1e-10)
    x_med  = np.hstack([m_std[:, None], x_base])

    def _rss0(x):
        """Return residual sum of squares from OLS of y on x."""
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        e = y - x @ b; return float(e @ e)

    rss0_b = _rss0(x_base); rss0_m = _rss0(x_med)
    p_b = x_base.shape[1];  p_m = x_med.shape[1]

    rows = []
    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)  # (L, m, J)
        H = probs[:, valid, :]                                  # (L, n, J) all founders

        if just_these_loci is not None:
            mask = np.isin(lids, list(just_these_loci))
            lids, pos, H = lids[mask], pos[mask], H[mask]
        if not len(lids):
            continue

        # vectorised OLS via FWL for both models; pinv handles rank J-1 (sum-to-one)
        for x_cov, rss0, p_cov, model_tag in [
            (x_base, rss0_b, p_b, 'base'),
            (x_med,  rss0_m, p_m, 'med'),
        ]:
            xt_pinv = np.linalg.lstsq(x_cov.T @ x_cov, x_cov.T, rcond=None)[0]
            proj    = np.einsum('pn,lnj->lpj', xt_pinv, H, optimize=True)
            H_perp  = H - np.einsum('np,lpj->lnj', x_cov, proj, optimize=True)
            c       = np.einsum('lnj,n->lj', H_perp, y, optimize=True)
            G       = np.einsum('lnj,lnk->ljk', H_perp, H_perp, optimize=True)
            betas   = (np.linalg.pinv(G, rcond=1e-10) @ c[..., None]).squeeze(-1)
            rss_red = np.einsum('lj,lj->l', c, betas)
            rss1    = np.maximum(rss0 - rss_red, 1e-30)
            lod     = np.where(rss1 < rss0, (n/2.0)*np.log10(rss0/rss1), 0.0)

            if model_tag == 'base':
                lod_b = lod
            else:
                lod_m = lod
                q_eff = np.linalg.matrix_rank(G)
                df2_arr = np.maximum(n - p_cov - q_eff, 1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    f_stat = np.where(
                        q_eff > 0,
                        np.maximum((rss0-rss1)/np.maximum(q_eff,1)/(rss1/df2_arr), 0.0),
                        0.0)
                    pval   = np.where(df2_arr > 0,
                                      1.0 - sp_stats.f.cdf(f_stat, q_eff, df2_arr),
                                      1.0)

        df = pd.DataFrame({
            'locus':        lids, 'chr': chrom, 'pos_mb': pos.astype(float),
            'lod_base':     lod_b, 'lod_mediated': lod_m,
            'lod_drop':     np.maximum(lod_b - lod_m, 0.0),
            'p_value':      pval,
            'outcome_id':   outcome_id, 'mediator_id': mediator_id,
        })
        rows.append(df)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def mediation_scan_multi_outcome(
    pairs_by_outcome: 'dict[str, list[tuple[str, str]]]',
    expr_cache: 'dict[str, np.ndarray]',
    genomecache,
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    sample_ids: 'list[str] | None' = None,
    lod_base_min: float = 1.5,
    lod_drop_min: float = 0.5,
    output_path: 'Path | None' = None,
) -> pd.DataFrame:
    """mediation scan over all outcome genes, streaming to disk.

    iterates outcomes from pairs_by_outcome, calls mediation_scan_qr_batch
    for each, applies lod_base and lod_drop filters, and writes filtered rows
    immediately — peak memory is one outcome's chromosome worth of loci.

    parameters
    ----------
    pairs_by_outcome : dict {outcome_id: [(mediator_id, chr), ...]}
        all mediators for one outcome share the same chromosome.
        build with select_cis_mediation_pairs then group by outcome.
    expr_cache       : dict {gene_id: np.ndarray (n,)}  outcome and mediator vectors
    genomecache      : GenomeCache
    covariates       : np.ndarray (n, p) or None
    kinship_decomp   : dict or None
    sample_ids       : list[str] or None  genotype-level strain IDs
    lod_base_min     : float  minimum base LOD to retain (default 1.5)
    lod_drop_min     : float  minimum LOD drop to retain (default 0.5)
    output_path      : Path or None  stream CSV here; None accumulates in RAM

    returns
    -------
    pd.DataFrame  columns: gene, mediator, chr, locus, pos_mb, lod_base, lod_mediated
    """
    _COLS = ['gene', 'mediator', 'chr', 'locus', 'pos_mb', 'lod_base', 'lod_mediated']
    _first = True
    _buf: 'list[pd.DataFrame]' = []
    _n = len(pairs_by_outcome)
    for _i, (outcome_id, med_list) in enumerate(pairs_by_outcome.items()):
        if outcome_id not in expr_cache:
            continue
        ch = med_list[0][1]
        med_vecs = {
            mid: expr_cache[mid] for mid, _ in med_list if mid in expr_cache
        }
        if not med_vecs:
            continue
        logger.debug(
            'mediation_scan_multi_outcome: [%d/%d] %s → %d mediator(s)',
            _i + 1, _n, outcome_id, len(med_vecs),
        )
        try:
            _df = mediation_scan_qr_batch(
                expr_cache[outcome_id], med_vecs, genomecache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                sample_ids=sample_ids,
                outcome_id=outcome_id,
                scan_window='cis', gene_chr=ch,
            )
            if not _df.empty:
                _df = _df.rename(columns={
                    'outcome_id': 'gene', 'mediator_id': 'mediator'})
                _df = _df[
                    (_df['lod_base'] >= lod_base_min) &
                    (_df['lod_base'] - _df['lod_mediated'] >= lod_drop_min)
                ][_COLS]
                if not _df.empty:
                    if output_path is not None:
                        _df.to_csv(
                            output_path, mode='a', header=_first, index=False)
                        _first = False
                    else:
                        _buf.append(_df.copy())
            del _df
        except Exception as exc:
            logger.warning(
                'mediation_scan_multi_outcome: %s failed — %s', outcome_id, exc)
        del med_vecs
    gc.collect()
    if output_path is not None:
        return (pd.read_csv(output_path)
                if output_path.exists() and not _first else pd.DataFrame())
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()

"""
mediation — genome-wide mediation scan

tests whether a haplotype QTL effect on outcome gene y is mediated through
an intermediate gene m. the qr decomposition of [m | X] is precomputed once;
the per-locus loop is vectorised via the FWL identity (same trick as scan.py).

lod_drop = lod_base − lod_mediated
  > 0  mediation detected (haplotype effect shrinks when m is added)
  < 0  amplification (shared upstream regulator)
"""

import numpy as np
import pandas as pd
import logging
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
        x_base   = (z_sub.T @ x_base) / r_counts[:, None] # (m, p)
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

"""
mediation — genome-wide mediation scan

tests whether a haplotype QTL effect on outcome gene y is mediated through
an intermediate gene m.  the Woodbury rank-1 identity is used to share
FWL quantities across all mediators on each chromosome: H_perp_X and G_inv
are computed once per chromosome; per-mediator LOD is a rank-1 update.

lod_drop = lod_base - lod_mediated
  > 0  mediation detected (haplotype effect shrinks when m is added)
  < 0  amplification (shared upstream regulator)
"""

import gc
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from scipy.stats import chi2 as _chi2
from ..utils import build_z_matrix

logger = logging.getLogger('mosaiqtl')

_MED_COLS = [
    'outcome_id', 'mediator_id', 'chr', 'locus', 'pos_mb',
    'lod_base', 'lod_mediated', 'lod_drop', 'prop_mediated',
    'p_value', 'h2', 'n',
]

_FIXEF_COLS = [
    'outcome_id', 'mediator_id', 'chr', 'locus', 'pos_mb',
    'lod_base', 'lod_mediated', 'lod_drop', 'prop_mediated',
    'p_value', 'n',
]


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


def mediation_scan_qr(
    outcome: np.ndarray,
    mediators: 'np.ndarray | dict[str, np.ndarray]',
    genomecache,
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    outcome_id: str = 'outcome',
    mediator_id: str = 'mediator',
    sample_ids: 'list | None' = None,
    scan_window: str = 'genome',
    gene_chr: 'str | None' = None,
    gene_pos_mb: 'float | None' = None,
    window_mb: float = 10.0,
    base_lod: 'dict[str, np.ndarray] | None' = None,
) -> pd.DataFrame:
    """Genome-wide mediation scan — one or multiple mediators per call.

    Tests whether the haplotype QTL effect on outcome is mediated through
    candidate mediator genes. The Woodbury rank-1 identity is used to share
    FWL quantities across mediators on each chromosome: H_perp_X and G_inv
    are computed once per chromosome; per-mediator LOD is a rank-1 update.
    h2 is fixed from the base model ``y ~ X`` for all mediators on the same
    chromosome (drift ≤ 0.03 empirically, Pearson r ≥ 0.9999 vs exact).

    Parameters
    ----------
    outcome   : np.ndarray (n,)
        Outcome expression vector. Pre-scaled by the caller — RINT-transformed
        for the LMM path; Pearson residuals (~N(0,1)) for the NB path.
        No internal transformation is applied.
    mediators : np.ndarray (n,) or dict {mediator_id: np.ndarray (n,)}
        Single mediator array or mapping of mediator IDs to arrays. A plain
        array is treated as a single mediator with id 'mediator'.
    genomecache : GenomeCache
    covariates : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values', 'vectors'}}
    outcome_id : str
    mediator_id : str
        Label used when ``mediators`` is a plain array (ignored when
        ``mediators`` is a dict — keys are used directly).
    sample_ids : list or None
        Genotype-level strain IDs. When provided and different from
        genomecache.sample_ids, reduces outcome/mediators/covariates to
        strain means via Z-matrix before scanning.
    scan_window : 'genome' | 'cis' | 'local'
    gene_chr : str or None  required for 'cis' and 'local'
    gene_pos_mb : float or None  required for 'local'
    window_mb : float  half-window for 'local' (default 10.0)
    base_lod : dict {chrom: np.ndarray (n_loci,)} or None
        Pre-computed base LOD scores (y ~ H | X). When supplied the base
        scan is skipped. Use when the same outcome is tested against many
        mediators to avoid recomputing the identical base scan.

    Returns
    -------
    pd.DataFrame columns:
        outcome_id, mediator_id, chr, locus, pos_mb,
        lod_base, lod_mediated, lod_drop, prop_mediated,
        p_value, h2, n

        p_value  : chi-squared(1) significance of lod_drop
        h2       : heritability from base null model (LOCO, per chromosome)
        n        : effective sample size after NaN filtering and Z-reduction
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module

    if isinstance(mediators, np.ndarray):
        med_dict: dict[str, np.ndarray] = {mediator_id: mediators}
    else:
        med_dict = dict(mediators)

    if scan_window in ('cis', 'local') and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    cache_ids = list(genomecache.sample_ids)
    sid_list  = list(sample_ids) if sample_ids is not None else cache_ids
    if sid_list != cache_ids:
        z_mat, valid_sid = build_z_matrix(sid_list, cache_ids)
    else:
        z_mat, valid_sid = None, np.ones(len(sid_list), dtype=bool)

    med_ids  = list(med_dict.keys())
    med_arrs = [med_dict[k] for k in med_ids]
    n_med    = len(med_ids)
    M_raw    = np.stack(med_arrs, axis=0)               # (n_med, n_samples)

    valid = (~np.isnan(outcome)) & valid_sid
    for i in range(n_med):
        valid &= ~np.isnan(M_raw[i])

    y = outcome[valid].copy()
    M = M_raw[:, valid].copy()                           # (n_med, n_valid)

    x_base = (covariates[valid].copy() if covariates is not None
              else np.ones((valid.sum(), 1), dtype=float))
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(y), 1)), x_base])

    z_sub = z_mat[valid] if z_mat is not None else None
    if z_sub is not None:
        r_counts = z_sub.sum(axis=0)
        y        = (z_sub.T @ y) / r_counts
        M        = (z_sub.T @ M.T).T / r_counts[None, :]
        x_base   = (z_sub.T @ x_base) / r_counts[:, None]
        ev_valid  = slice(None)
    else:
        ev_valid  = valid

    n = len(y)

    M_std = (M - M.mean(axis=1, keepdims=True)) / np.maximum(
        M.std(axis=1, keepdims=True), 1e-10)             # (n_med, n)

    chroms = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ('cis', 'local'):
        chroms = [c for c in chroms if c == str(gene_chr)]

    rows: list[pd.DataFrame] = []
    n_chroms = len(chroms)

    for _ci, chrom in enumerate(chroms):
        _t0 = time.perf_counter()
        ev  = kinship_decomp[chrom]['values']
        evc = kinship_decomp[chrom]['vectors']
        ev_v = evc[ev_valid]                             # (n, n_eigs)

        h2, _ = lmm_module.estimate_h2_null(ev_v.T @ y, ev_v.T @ x_base, ev)
        w_inv  = 1.0 / (h2 * ev + (1.0 - h2))
        w_sqrt = np.sqrt(w_inv)

        y_rot  = (ev_v.T @ y)      * w_sqrt             # (n,)
        x_rot  = (ev_v.T @ x_base) * w_sqrt[:, None]    # (n, p)

        xt_gram  = x_rot.T @ x_rot
        xt_pinv  = np.linalg.lstsq(xt_gram, x_rot.T, rcond=None)[0]   # (p, n)

        y_perp    = y_rot - x_rot @ (xt_pinv @ y_rot)
        rss0_base = float(y_perp @ y_perp)

        lids, pos, probs = genomecache.get_chromosome(chrom)   # (L, n_strains, J)
        H_rot = np.einsum('nk,lnj->lkj', ev_v, probs,
                           optimize=True) * w_sqrt[None, :, None]  # (L, n, J)

        proj_H   = np.einsum('pn,lnj->lpj', xt_pinv, H_rot, optimize=True)
        H_perp_X = H_rot - np.einsum('np,lpj->lnj', x_rot, proj_H, optimize=True)

        G_base = np.einsum('lnj,lnk->ljk', H_perp_X, H_perp_X, optimize=True)
        c_base = np.einsum('lnj,n->lj',    H_perp_X, y_perp,    optimize=True)
        G_inv  = np.linalg.pinv(G_base, rcond=1e-10)            # (L, J, J)
        q_base = np.einsum('ljk,lk->lj', G_inv, c_base, optimize=True)

        if base_lod is not None and chrom in base_lod:
            lod_b = base_lod[chrom]
        else:
            rss_red_base = np.einsum('lj,lj->l', c_base, q_base)
            rss1_base    = np.maximum(rss0_base - rss_red_base, 1e-30)
            lod_b = np.where(rss1_base < rss0_base,
                             (n / 2.0) * np.log10(rss0_base / rss1_base), 0.0)

        M_rot    = np.einsum('in,nk->ik', M_std, ev_v,
                              optimize=True) * w_sqrt[None, :]    # (n_med, n)
        proj_M   = xt_pinv @ M_rot.T                              # (p, n_med)
        M_perp_X = M_rot - (x_rot @ proj_M).T                    # (n_med, n)

        norms    = np.einsum('in,in->i', M_perp_X, M_perp_X)    # (n_med,)
        safe_n   = np.maximum(norms, 1e-30)
        y_dot_m  = M_perp_X @ y_perp                             # (n_med,)
        alpha    = y_dot_m / safe_n
        rss0_m   = rss0_base - y_dot_m ** 2 / safe_n            # (n_med,)

        H_dot_M = np.einsum('lnj,in->lji', H_perp_X, M_perp_X, optimize=True)
        V = np.einsum('ljk,lki->lji', G_inv, H_dot_M, optimize=True)
        denom = safe_n[None, :] - np.einsum('lji,lji->li', H_dot_M, V)
        denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)

        c_lm = c_base[:, :, None] - alpha[None, None, :] * H_dot_M
        q_lm = q_base[:, :, None] - alpha[None, None, :] * V

        base_quad  = np.einsum('lji,lji->li', c_lm, q_lm)
        v_dot_q    = np.einsum('lji,lji->li', H_dot_M, q_lm)
        correction = v_dot_q ** 2 / denom
        rss_red    = base_quad + correction

        rss1_m = np.maximum(rss0_m[None, :] - rss_red, 1e-30)    # (L, n_med)
        lod_m  = np.where(
            rss1_m < rss0_m[None, :],
            (n / 2.0) * np.log10(np.maximum(rss0_m[None, :], 1e-30) / rss1_m),
            0.0,
        )                                                           # (L, n_med)

        lod_drop  = np.maximum(lod_b[:, None] - lod_m, 0.0)       # (L, n_med)
        prop_med  = lod_drop / np.maximum(lod_b[:, None], 1e-10)
        chi2_stat = 2.0 * np.log(10) * lod_drop
        p_val     = _chi2.sf(chi2_stat, df=1)

        for i, mid in enumerate(med_ids):
            rows.append(pd.DataFrame({
                'outcome_id':    outcome_id,
                'mediator_id':   mid,
                'chr':           chrom,
                'locus':         lids,
                'pos_mb':        pos.astype(float),
                'lod_base':      lod_b,
                'lod_mediated':  lod_m[:, i],
                'lod_drop':      lod_drop[:, i],
                'prop_mediated': prop_med[:, i],
                'p_value':       p_val[:, i],
                'h2':            h2,
                'n':             n,
            }))

        logger.debug('[%d/%d] mediation chr%s  %d loci × %d mediators  %.1fs',
                     _ci + 1, n_chroms, chrom, len(lids), n_med,
                     time.perf_counter() - _t0)

    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=_MED_COLS))


def fixef_mediation_scan(outcome, mediator, genomecache,
                          covariates=None,
                          outcome_id='outcome', mediator_id='mediator',
                          just_these_loci=None):
    """Fixed-effect mediation scan — OLS, no kinship, vectorised.

    Fast screening version. Compatible with just_these_loci filter
    for cis-window restriction.  Use mediation_scan_qr for kinship-
    corrected results; this function is for rapid candidate filtering only.

    Parameters
    ----------
    outcome    : np.ndarray (n,)  pre-scaled by the caller
    mediator   : np.ndarray (n,)
    genomecache : GenomeCache
    covariates : np.ndarray (n, p) or None
    outcome_id : str
    mediator_id : str
    just_these_loci : array-like or None

    Returns
    -------
    pd.DataFrame columns:
        outcome_id, mediator_id, chr, locus, pos_mb,
        lod_base, lod_mediated, lod_drop, prop_mediated,
        p_value, n
    """
    from scipy import stats as sp_stats

    valid  = (~np.isnan(outcome)) & (~np.isnan(mediator))
    y      = outcome[valid].copy()
    m      = mediator[valid].copy()
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
        H = probs[:, valid, :]                                  # (L, n, J)

        if just_these_loci is not None:
            mask = np.isin(lids, list(just_these_loci))
            lids, pos, H = lids[mask], pos[mask], H[mask]
        if not len(lids):
            continue

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
                q_eff   = np.linalg.matrix_rank(G)
                df2_arr = np.maximum(n - p_cov - q_eff, 1)
                with np.errstate(invalid='ignore', divide='ignore'):
                    f_stat = np.where(
                        q_eff > 0,
                        np.maximum(
                            (rss0 - rss1) / np.maximum(q_eff, 1)
                            / (rss1 / df2_arr), 0.0),
                        0.0)
                    pval = np.where(df2_arr > 0,
                                    1.0 - sp_stats.f.cdf(f_stat, q_eff, df2_arr),
                                    1.0)

        lod_drop = np.maximum(lod_b - lod_m, 0.0)
        prop_med = lod_drop / np.maximum(lod_b, 1e-10)

        rows.append(pd.DataFrame({
            'outcome_id':    outcome_id,
            'mediator_id':   mediator_id,
            'chr':           chrom,
            'locus':         lids,
            'pos_mb':        pos.astype(float),
            'lod_base':      lod_b,
            'lod_mediated':  lod_m,
            'lod_drop':      lod_drop,
            'prop_mediated': prop_med,
            'p_value':       pval,
            'n':             n,
        }))

    return (pd.concat(rows, ignore_index=True) if rows
            else pd.DataFrame(columns=_FIXEF_COLS))


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
    """Mediation scan over all outcome genes, streaming to disk.

    Iterates outcomes from pairs_by_outcome, calls mediation_scan_qr for
    each, applies lod_base and lod_drop filters, and writes filtered rows
    immediately — peak memory is one outcome's chromosome worth of loci.

    Parameters
    ----------
    pairs_by_outcome : dict {outcome_id: [(mediator_id, chr), ...]}
        All mediators for one outcome share the same chromosome.
        Build with select_cis_mediation_pairs then group by outcome.
    expr_cache       : dict {gene_id: np.ndarray (n,)}
    genomecache      : GenomeCache
    covariates       : np.ndarray (n, p) or None
    kinship_decomp   : dict or None
    sample_ids       : list[str] or None  genotype-level strain IDs
    lod_base_min     : float  minimum base LOD to retain (default 1.5)
    lod_drop_min     : float  minimum LOD drop to retain (default 0.5)
    output_path      : Path or None  stream CSV here; None accumulates in RAM

    Returns
    -------
    pd.DataFrame columns: gene, mediator, chr, locus, pos_mb,
        lod_base, lod_mediated, lod_drop, prop_mediated, p_value, h2, n
    """
    _OUT_COLS = [
        'gene', 'mediator', 'chr', 'locus', 'pos_mb',
        'lod_base', 'lod_mediated', 'lod_drop', 'prop_mediated',
        'p_value', 'h2', 'n',
    ]
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
            _df = mediation_scan_qr(
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
                    (_df['lod_drop'] >= lod_drop_min)
                ][_OUT_COLS]
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

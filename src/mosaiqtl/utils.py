"""
utils — shared utility functions

rank-inverse normal transformation, sample alignment, Z-matrix construction,
kinship/haplotype Z-expansion, and design matrix builder.
"""

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger('mosaiqtl')


def rint(x):
    """rank-based inverse normal transformation (Blom formula).

    maps continuous values to standard normal quantiles via rank ordering.
    ties are averaged; NaN values are preserved.

    parameters
    ----------
    x : array-like (1-d)

    returns
    -------
    np.ndarray  transformed values, mean ≈ 0, sd ≈ 1
    """
    x     = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    n     = valid.sum()
    out   = np.full_like(x, np.nan)
    ranks = stats.rankdata(x[valid], method='average')
    out[valid] = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
    return out


def align_samples(ids_a, ids_b):
    """find shared samples and return alignment indices.

    parameters
    ----------
    ids_a, ids_b : array-like

    returns
    -------
    idx_a, idx_b : np.ndarray (int)  indices into a and b for shared samples
    shared       : list              ordered shared ids
    """
    b_map = {sid: i for i, sid in enumerate(ids_b)}
    idx_a, idx_b, shared = [], [], []
    for i, sid in enumerate(ids_a):
        if sid in b_map:
            idx_a.append(i); idx_b.append(b_map[sid]); shared.append(sid)
    return np.array(idx_a), np.array(idx_b), shared


def build_z_matrix(sample_ids, geno_ids):
    """build the (n × m) sample-to-genotype mapping matrix Z.

    Z_{ik} = 1 if sample i belongs to strain k, else 0.
    when n == m and ids match, Z is the identity matrix.
    samples with no matching genotype are flagged via valid_mask.

    parameters
    ----------
    sample_ids : array-like (n,)  one per phenotype observation (may repeat)
    geno_ids   : array-like (m,)  genome cache order

    returns
    -------
    Z          : np.ndarray (n, m)
    valid_mask : np.ndarray (n,) bool
    """
    sample_ids = list(sample_ids)
    geno_ids   = list(geno_ids)
    geno_map   = {g: j for j, g in enumerate(geno_ids)}
    n, m       = len(sample_ids), len(geno_ids)
    Z          = np.zeros((n, m), dtype=np.float64)
    valid      = np.zeros(n, dtype=bool)
    for i, sid in enumerate(sample_ids):
        if sid in geno_map:
            Z[i, geno_map[sid]] = 1.0
            valid[i]            = True
    n_miss = int((~valid).sum())
    if n_miss:
        logger.warning(f'build_z_matrix: {n_miss}/{n} samples missing genotype')
    return Z, valid


def expand_kinship(k_geno, z_mat):
    """expand m×m kinship to n×n sample space: K_n = Z K_m Z'.

    parameters
    ----------
    k_geno : np.ndarray (m, m)
    z_mat  : np.ndarray (n, m)

    returns
    -------
    np.ndarray (n, n)
    """
    return z_mat @ k_geno @ z_mat.T


def expand_probs(h_geno, z_mat):
    """expand (m, J) haplotype probs to (n, J) via Z: H_n = Z H_m.

    parameters
    ----------
    h_geno : np.ndarray (m, J)
    z_mat  : np.ndarray (n, m)

    returns
    -------
    np.ndarray (n, J)
    """
    return z_mat @ h_geno


def make_design_matrix(covariates_df, formula_cols=None):
    """construct numeric design matrix from a DataFrame.

    adds intercept; one-hot encodes categorical columns (first level dropped).

    parameters
    ----------
    covariates_df : pd.DataFrame
    formula_cols  : list of str or None

    returns
    -------
    np.ndarray (n, p), list of str (column names)
    """
    import pandas as pd
    df = (covariates_df[formula_cols] if formula_cols else covariates_df).copy()
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    cat = df.select_dtypes(exclude=[np.number]).columns.tolist()
    parts = [np.ones((len(df), 1))]; names = ['intercept']
    for col in num:
        parts.append(df[col].values[:, None].astype(np.float64)); names.append(col)
    for col in cat:
        d = pd.get_dummies(df[col], prefix=col, drop_first=True)
        parts.append(d.values.astype(np.float64)); names.extend(d.columns.tolist())
    return np.hstack(parts), names

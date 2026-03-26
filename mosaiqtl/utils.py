"""
mosaiqtl.utils — shared utility functions

includes rank-inverse normal transformation, data alignment,
sample-to-genotype mapping (Z matrix), and numerical helpers
used across all modules.

the Z-matrix design supports datasets where n phenotype samples
map to m < n unique genotypes (e.g. replicated CC strains):

    H_locus = Z @ H_cache       # (n, J)  ← (n, m) @ (m, J)
    K_n     = Z @ K_m @ Z.T     # (n, n)  ← expand kinship

when n == m (one sample per genotype), Z == I and all results
are identical to the 1:1 case.
"""

import numpy as np
from scipy import stats
import logging

logger = logging.getLogger('mosaiqtl')


def rint(x):
    """rank-based inverse normal transformation.

    maps arbitrary continuous values to standard normal quantiles
    via rank ordering. ties are averaged. missing values (nan) are
    preserved.

    equivalent to miqtl::rint(proportion = 0.5) in R.

    parameters
    ----------
    x : array-like  input values (1d)

    returns
    -------
    np.ndarray  transformed values with mean ~0 and sd ~1
    """
    x     = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    n     = valid.sum()
    out   = np.full_like(x, np.nan)
    ranks = stats.rankdata(x[valid], method='average')
    # blom offset: (r − 3/8) / (n + 1/4) — standard for RINT
    out[valid] = stats.norm.ppf((ranks - 0.375) / (n + 0.25))
    return out


def align_samples(sample_ids_a, sample_ids_b):
    """find shared samples and return index arrays for alignment.

    parameters
    ----------
    sample_ids_a : array-like  sample ids from first dataset
    sample_ids_b : array-like  sample ids from second dataset

    returns
    -------
    idx_a  : np.ndarray (int)  indices into a for shared samples
    idx_b  : np.ndarray (int)  indices into b for shared samples
    shared : list              ordered list of shared sample ids
    """
    b_map = {sid: i for i, sid in enumerate(sample_ids_b)}
    idx_a, idx_b, shared = [], [], []
    for i, sid in enumerate(sample_ids_a):
        if sid in b_map:
            idx_a.append(i); idx_b.append(b_map[sid]); shared.append(sid)
    return np.array(idx_a), np.array(idx_b), shared


def build_z_matrix(sample_ids, geno_ids):
    """build the (n × m) sample-to-genotype mapping matrix Z.

    each row of Z has exactly one nonzero entry (1.0) at the
    column corresponding to the sample's genotype. samples whose
    id does not appear in geno_ids are silently excluded (their row
    is all zeros and will not contribute to the scan).

    when n == m and sample_ids == geno_ids, Z is the identity matrix.

    parameters
    ----------
    sample_ids : array-like (n,)
        phenotype-level identifiers — one per observation.
        may repeat (multiple replicates per strain).
    geno_ids : array-like (m,)
        genotype-level identifiers — must match genome cache order.

    returns
    -------
    Z : np.ndarray (n, m)  mapping matrix, dtype float64
    valid_mask : np.ndarray (n,) bool
        true for rows that have a matching genotype
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
    n_missing = int((~valid).sum())
    if n_missing:
        logger.warning(f'build_z_matrix: {n_missing}/{n} samples '
                       f'have no matching genotype — excluded from scan')
    return Z, valid


def expand_kinship(k_geno, z_mat):
    """expand m×m genotype-level kinship to n×n sample-level.

    K_n = Z K_m Z^T

    parameters
    ----------
    k_geno : np.ndarray (m, m)  genotype-level realized relationship matrix
    z_mat  : np.ndarray (n, m)  mapping matrix from build_z_matrix()

    returns
    -------
    np.ndarray (n, n)
    """
    return z_mat @ k_geno @ z_mat.T


def expand_probs(h_geno, z_mat):
    """expand m × J haplotype probs to n × J via Z.

    H_n = Z H_m

    parameters
    ----------
    h_geno : np.ndarray (m, J)  haplotype probs from genome cache
    z_mat  : np.ndarray (n, m)

    returns
    -------
    np.ndarray (n, J)
    """
    return z_mat @ h_geno


def drop_colinear_founders(h_matrix, tol=1e-8):
    """drop last founder column (sum-to-one constraint makes it redundant).

    parameters
    ----------
    h_matrix : np.ndarray (n, J)
    tol : float  (unused; kept for API stability)

    returns
    -------
    np.ndarray (n, J-1)
    """
    return h_matrix[:, :-1]


def safe_log(x, floor=1e-300):
    """element-wise log with floor to avoid -inf."""
    return np.log(np.maximum(x, floor))


def logfactorial_sum(counts):
    """sum of log(y!) via gammaln — used in NEBULA-LN sufficient statistics."""
    from scipy.special import gammaln
    return np.sum(gammaln(counts + 1))


def make_design_matrix(covariates_df, formula_cols=None):
    """construct a numeric design matrix from a DataFrame.

    adds intercept column. categorical columns are one-hot encoded
    with the first level dropped (matches R's model.matrix convention).

    parameters
    ----------
    covariates_df : pd.DataFrame  rows = samples
    formula_cols  : list of str or None  columns to include; all if None

    returns
    -------
    np.ndarray (n, p)  design matrix (intercept in column 0)
    list of str        column names
    """
    import pandas as pd
    df = (covariates_df[formula_cols] if formula_cols is not None
          else covariates_df).copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    parts = [np.ones((len(df), 1))]; names = ['intercept']
    for col in num_cols:
        parts.append(df[col].values.reshape(-1, 1).astype(np.float64))
        names.append(col)
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        parts.append(dummies.values.astype(np.float64))
        names.extend(dummies.columns.tolist())
    return np.hstack(parts), names

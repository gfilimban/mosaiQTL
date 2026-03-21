"""
mosaiqtl.utils — shared utility functions

includes rank-inverse normal transformation, data alignment,
and numerical helpers used across modules.
"""

import numpy as np
from scipy import stats
import logging

# configure module-level logger
logger = logging.getLogger('mosaiqtl')


def rint(x):
    """rank-based inverse normal transformation.

    maps arbitrary continuous values to standard normal quantiles
    via rank ordering. ties are averaged. missing values (nan)
    are preserved.

    equivalent to miqtl::rint() in R.

    parameters
    ----------
    x : array-like
        input values (1d)

    returns
    -------
    np.ndarray
        transformed values with mean ~0 and sd ~1
    """
    # cast to float array, preserving nan
    x = np.asarray(x, dtype=np.float64)

    # mask for non-nan values
    valid = ~np.isnan(x)
    n = valid.sum()

    # allocate output, fill nan
    out = np.full_like(x, np.nan)

    # rank with average tie-breaking (1-based)
    ranks = stats.rankdata(x[valid], method='average')

    # blom offset: (rank - 3/8) / (n + 1/4) — standard for RINT
    out[valid] = stats.norm.ppf((ranks - 0.375) / (n + 0.25))

    return out


def align_samples(sample_ids_a, sample_ids_b):
    """find shared samples and return index arrays for alignment.

    parameters
    ----------
    sample_ids_a : array-like
        sample identifiers from first dataset
    sample_ids_b : array-like
        sample identifiers from second dataset

    returns
    -------
    idx_a : np.ndarray (int)
        indices into a for shared samples
    idx_b : np.ndarray (int)
        indices into b for shared samples
    shared : list
        ordered list of shared sample ids
    """
    # build lookup for b
    b_map = {sid: i for i, sid in enumerate(sample_ids_b)}

    idx_a = []  # matching indices in a
    idx_b = []  # matching indices in b
    shared = []  # shared sample ids

    for i, sid in enumerate(sample_ids_a):
        if sid in b_map:
            idx_a.append(i)
            idx_b.append(b_map[sid])
            shared.append(sid)

    return np.array(idx_a), np.array(idx_b), shared


def drop_colinear_founders(h_matrix, tol=1e-8):
    """drop one founder column to avoid collinearity.

    haplotype probability matrices have 8 columns summing to 1,
    so one column is redundant. we drop the last founder (wsbj
    by convention) and return 7 columns.

    parameters
    ----------
    h_matrix : np.ndarray (n_samples, 8)
        founder haplotype probabilities at one locus
    tol : float
        tolerance for detecting perfect collinearity

    returns
    -------
    np.ndarray (n_samples, 7)
        reduced haplotype matrix with last column dropped
    """
    if h_matrix.shape[1] == 8:
        return h_matrix[:, :7]
    return h_matrix


def safe_log(x, floor=1e-300):
    """element-wise log with floor to avoid -inf.

    parameters
    ----------
    x : np.ndarray
        input values (must be >= 0)
    floor : float
        minimum value before taking log

    returns
    -------
    np.ndarray
        log(max(x, floor))
    """
    return np.log(np.maximum(x, floor))


def logfactorial_sum(counts):
    """sum of log(y!) for a vector of counts.

    used in nebula-ln sufficient statistics. uses
    scipy.special.gammaln for numerical stability:
    log(y!) = gammaln(y + 1)

    parameters
    ----------
    counts : np.ndarray (int or float)
        count values (non-negative)

    returns
    -------
    float
        sum of log(y!) across all elements
    """
    from scipy.special import gammaln
    return np.sum(gammaln(counts + 1))


def make_design_matrix(covariates_df, formula_cols=None):
    """construct a numeric design matrix from a dataframe.

    adds intercept column. categorical columns are one-hot encoded
    with the first level dropped.

    parameters
    ----------
    covariates_df : pd.DataFrame
        covariate data, rows = samples
    formula_cols : list of str or None
        columns to include. if none, use all columns.

    returns
    -------
    np.ndarray (n_samples, p)
        design matrix with intercept in column 0
    col_names : list of str
        column names for the design matrix
    """
    import pandas as pd

    if formula_cols is not None:
        df = covariates_df[formula_cols].copy()
    else:
        df = covariates_df.copy()

    # separate numeric and categorical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    parts = [np.ones((len(df), 1))]  # intercept
    names = ['intercept']

    # add numeric columns as-is
    for col in numeric_cols:
        parts.append(df[col].values.reshape(-1, 1).astype(np.float64))
        names.append(col)

    # one-hot encode categoricals, drop first level
    for col in cat_cols:
        dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
        parts.append(dummies.values.astype(np.float64))
        names.extend(dummies.columns.tolist())

    return np.hstack(parts), names

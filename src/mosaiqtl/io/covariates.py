"""
covariates — covariate matrix construction for QTL scan

builds the fixed-effect covariate matrix from user-supplied sample metadata.
only required column: sample_id (used as index).
all other covariates are opt-in via function parameters.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.sparse.linalg import svds

from mosaiqtl.models.config import Population

logger = logging.getLogger('mosaiqtl')


class MissingCovariateError(ValueError):
    """raised when a requested covariate column contains NaN values.

    signals a data completeness problem in sample metadata —
    distinct from configuration errors (plain ValueError).
    message lists affected column and sample IDs for actionable diagnosis.
    """


def build_covariate_matrix(
    sample_metadata: pd.DataFrame,
    covariates: list[str] | None = None,
    n_expression_pcs: int = 0,
    n_genotype_pcs: int = 0,
    population: Optional[Union[str, Population]] = None,
    pseudobulk_counts: Optional[pd.DataFrame] = None,
    kinship_matrix: Optional[np.ndarray] = None,
    kinship_sample_ids: Optional[list[str]] = None,
) -> pd.DataFrame:
    """build the fixed-effect covariate matrix for a QTL scan.

    parameters
    ----------
    sample_metadata : pd.DataFrame
        must contain sample_id column (used as index).
        all other columns are optional — include only what your design requires.
    covariates : list[str]
        columns from sample_metadata to include as fixed effects.
        categorical columns are one-hot encoded; numeric passed through.
        default empty — caller decides what to include.
    n_expression_pcs : int
        number of expression PCs (PEER-equivalent confounders).
        captures library size, batch, and other latent confounders jointly.
        0 to disable. requires pseudobulk_counts when > 0.
    n_genotype_pcs : int
        number of genotype PCs for population structure correction.
        0 to disable. requires kinship_matrix and kinship_sample_ids when > 0.
        informative for DO, MPP, F2, and structured populations only —
        not permitted for CC: LOCO kinship already captures all structure.
        requires population to be set when > 0.
    population : str, Population, or None
        population design. accepts a Population instance (preferred) or a
        string name ('cc', 'do', 'mpp', 'f2', 'structured').
        required when n_genotype_pcs > 0 to enforce informativeness.
    pseudobulk_counts : pd.DataFrame or None
        samples × genes pseudobulk count matrix; required when
        n_expression_pcs > 0. index must match sample_metadata.sample_id.
    kinship_matrix : np.ndarray or None
        (n_samples, n_samples) genome-wide kinship matrix; required when
        n_genotype_pcs > 0.
    kinship_sample_ids : list[str] or None
        sample IDs corresponding to rows/cols of kinship_matrix; required
        when n_genotype_pcs > 0.

    returns
    -------
    pd.DataFrame
        index = sample_id (sorted ascending), columns = covariate names,
        dtype float64, no NaN values.

    raises
    ------
    MissingCovariateError
        if any requested covariate column contains NaN, with the affected
        column name and sample IDs listed.
    ValueError
        if sample_id column is absent from sample_metadata.
        if a requested column is not found in sample_metadata.
        if n_expression_pcs > 0 and pseudobulk_counts is None.
        if n_genotype_pcs > 0 and kinship_matrix or kinship_sample_ids is None.
        if n_genotype_pcs > 0 and population is a LOCO population (e.g. 'cc').
    """
    if 'sample_id' not in sample_metadata.columns:
        raise ValueError("'sample_id' column is required in sample_metadata")

    if n_expression_pcs > 0 and pseudobulk_counts is None:
        raise ValueError(
            f'n_expression_pcs={n_expression_pcs} requires pseudobulk_counts; '
            f'pass pseudobulk_counts or set n_expression_pcs=0'
        )

    if n_genotype_pcs > 0:
        if kinship_matrix is None or kinship_sample_ids is None:
            raise ValueError(
                f'n_genotype_pcs={n_genotype_pcs} requires kinship_matrix and '
                f'kinship_sample_ids; pass both or set n_genotype_pcs=0'
            )
        if _loco_sufficient(population):
            pop_label = population.name if isinstance(population, Population) else population
            raise ValueError(
                f'n_genotype_pcs={n_genotype_pcs} is not informative for '
                f"population={pop_label!r}: LOCO kinship already captures all "
                f'population structure. Set n_genotype_pcs=0.'
            )

    if covariates is None:
        covariates = []

    meta = sample_metadata.set_index('sample_id').sort_index()
    parts: list[pd.DataFrame] = []

    missing_cols = [c for c in covariates if c not in meta.columns]
    if missing_cols:
        raise ValueError(
            f'requested covariate columns not found in sample_metadata: {missing_cols}'
        )

    for col in covariates:
        parts.append(_encode_column(meta, col))

    if n_expression_pcs > 0:
        parts.append(_expression_pcs(pseudobulk_counts, meta.index, n_expression_pcs))

    if n_genotype_pcs > 0:
        parts.append(_genotype_pcs(kinship_matrix, kinship_sample_ids, meta.index, n_genotype_pcs))

    if not parts:
        return pd.DataFrame(index=meta.index, dtype=np.float64)

    result = pd.concat(parts, axis=1).astype(np.float64)

    for col in result.columns[result.isna().any()]:
        bad = result.index[result[col].isna()].tolist()
        raise MissingCovariateError(
            f"NaN in covariate '{col}' for samples: {bad}"
        )

    return result


def encode_sex(series: 'pd.Series') -> 'pd.Series':
    """Encode a sex column to float (Male/M/0 → 0.0, Female/F/1 → 1.0).

    Accepts any of: 'Male'/'Female', 'M'/'F', 'm'/'f', 0/1, 0.0/1.0.
    Unknown values map to 0.0 (treated as male); a warning is logged.

    Parameters
    ----------
    series : pd.Series
        Raw sex column from sample metadata.

    Returns
    -------
    pd.Series
        Float64 series with same index, values in {0.0, 1.0}.
    """
    def _map(v: object) -> float:
        """Map a sex value (string label or numeric) to 0.0 (male) or 1.0 (female)."""
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        s = str(v).strip().lower()
        if s in ('male', 'm', '0'):
            return 0.0
        if s in ('female', 'f', '1'):
            return 1.0
        logger.warning('encode_sex: unrecognised value %r — defaulting to 0.0', v)
        return 0.0

    return series.map(_map).astype(np.float64)


def build_sex_dataset_covariates(
    meta: 'pd.DataFrame',
    sample_order: 'list[str]',
    dataset_col: str = 'dataset',
    dataset_categories: 'list[str] | None' = None,
) -> 'np.ndarray':
    """Build a covariate matrix aligned to ``sample_order`` for sex + dataset design.

    Matches R formula ``~ sex + dataset`` with Founder as reference level.

    Parameters
    ----------
    meta :
        Sample metadata DataFrame; must contain ``sample_id``, ``sex``, and
        the column named by ``dataset_col``.
    sample_order :
        Ordered list of sample IDs; defines the row order of the output.
    dataset_col :
        Column in ``meta`` holding the dataset/batch label (default ``'dataset'``).
    dataset_categories :
        Ordered category list for the dataset column; the first entry becomes
        the reference level (dropped in dummies).  Defaults to
        ``['Founder', 'CC1', 'CC2', 'CC3', 'F1']``.

    Returns
    -------
    np.ndarray  (n_samples, n_covariates)
        Columns: sex (0=F, 1=M), then one-hot dataset dummies (reference dropped).
    """
    import pandas as pd
    if dataset_categories is None:
        dataset_categories = ['Founder', 'CC1', 'CC2', 'CC3', 'F1']
    rows = meta.set_index('sample_id').loc[sample_order]
    sex_vec = encode_sex(rows['sex']).values.astype(np.float64)
    cat = pd.Categorical(rows[dataset_col].values, categories=dataset_categories)
    dummies = pd.get_dummies(cat, drop_first=True).values.astype(np.float64)
    return np.column_stack([sex_vec, dummies])


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------

def _loco_sufficient(population: Optional[Union[str, Population]]) -> bool:
    """Return True when LOCO kinship fully absorbs population structure.

    accepts a Population instance (preferred) or a string name for backward
    compatibility. the string path treats 'cc' as loco_sufficient=True since
    that was the only population supported before Population was introduced.
    """
    if population is None:
        return False
    if isinstance(population, Population):
        return population.loco_sufficient
    return population == 'cc'


def _check_nan(df: pd.DataFrame, col: str) -> None:
    """Raise MissingCovariateError if any value in col is NaN."""
    bad = df.index[df[col].isna()].tolist()
    if bad:
        raise MissingCovariateError(
            f"NaN in covariate '{col}' for samples: {bad}"
        )


def _encode_column(meta: pd.DataFrame, col: str) -> pd.DataFrame:
    """one-hot encode categorical columns; pass numeric columns through."""
    _check_nan(meta, col)
    series = meta[col]
    if pd.api.types.is_numeric_dtype(series):
        return series.rename(col).to_frame()
    dummies = pd.get_dummies(series, prefix=col, drop_first=True, dtype=float)
    return dummies


def _expression_pcs(
    pseudobulk_counts: pd.DataFrame,
    sample_ids: pd.Index,
    n_pcs: int,
) -> pd.DataFrame:
    """compute top-n expression PCs from log1p-normalised pseudobulk counts."""
    counts = pseudobulk_counts.loc[sample_ids]
    lib = counts.sum(axis=1).values.astype(float)
    lib = np.where(lib == 0, 1.0, lib)
    normed = np.log1p(counts.values.astype(float) / lib[:, None] * 1e4)
    normed -= normed.mean(axis=0)

    n_components = min(n_pcs, min(normed.shape) - 1)
    _, _, vt = svds(normed, k=n_components)
    scores = normed @ vt[::-1].T   # svds returns ascending order — reverse

    cols = [f'expr_pc{i + 1}' for i in range(n_components)]
    return pd.DataFrame(scores, index=sample_ids, columns=cols)


def _genotype_pcs(
    kinship_matrix: np.ndarray,
    kinship_sample_ids: list[str],
    sample_ids: pd.Index,
    n_pcs: int,
) -> pd.DataFrame:
    """top-n eigenvectors of the genome-wide kinship matrix.

    captures population structure for DO, MPP, F2, and structured populations.
    eigh returns ascending eigenvalues — reversed to get top structure first.
    """
    k_df = pd.DataFrame(kinship_matrix, index=kinship_sample_ids, columns=kinship_sample_ids)
    k_sub = k_df.loc[sample_ids, sample_ids].values.astype(float)

    n_components = min(n_pcs, len(sample_ids) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(k_sub)
    idx = np.argsort(eigenvalues)[::-1]
    top_vecs = eigenvectors[:, idx[:n_components]]

    cols = [f'geno_pc{i + 1}' for i in range(n_components)]
    return pd.DataFrame(top_vecs, index=sample_ids, columns=cols)

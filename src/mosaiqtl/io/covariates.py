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

logger = logging.getLogger("mosaiqtl")


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
        number of genotype PCs to correct for population structure.
        0 to disable. requires kinship_matrix and kinship_sample_ids when > 0.
        only meaningful for populations with residual structure beyond LOCO
        kinship (e.g. DO, F2); not permitted for CC, where LOCO kinship
        already captures all structure. requires population to be set when > 0.
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
    if "sample_id" not in sample_metadata.columns:
        raise ValueError("'sample_id' column is required in sample_metadata")

    if n_expression_pcs > 0 and pseudobulk_counts is None:
        raise ValueError(
            f"n_expression_pcs={n_expression_pcs} requires pseudobulk_counts; "
            f"pass pseudobulk_counts or set n_expression_pcs=0"
        )

    if n_genotype_pcs > 0:
        if kinship_matrix is None or kinship_sample_ids is None:
            raise ValueError(
                f"n_genotype_pcs={n_genotype_pcs} requires kinship_matrix and "
                f"kinship_sample_ids; pass both or set n_genotype_pcs=0"
            )
        if _loco_sufficient(population):
            pop_label = (
                population.name if isinstance(population, Population) else population
            )
            raise ValueError(
                f"n_genotype_pcs={n_genotype_pcs} is not informative for "
                f"population={pop_label!r}: LOCO kinship already captures all "
                f"population structure. Set n_genotype_pcs=0."
            )

    if covariates is None:
        covariates = []

    meta = sample_metadata.set_index("sample_id").sort_index()
    parts: list[pd.DataFrame] = []

    missing_cols = [c for c in covariates if c not in meta.columns]
    if missing_cols:
        raise ValueError(
            f"requested covariate columns not found in sample_metadata: {missing_cols}"
        )

    for col in covariates:
        parts.append(_encode_column(meta, col))

    if n_expression_pcs > 0:
        parts.append(_expression_pcs(pseudobulk_counts, meta.index, n_expression_pcs))

    if n_genotype_pcs > 0:
        parts.append(
            _genotype_pcs(
                kinship_matrix, kinship_sample_ids, meta.index, n_genotype_pcs
            )
        )

    if not parts:
        return pd.DataFrame(index=meta.index, dtype=np.float64)

    result = pd.concat(parts, axis=1).astype(np.float64)

    for col in result.columns[result.isna().any()]:
        bad = result.index[result[col].isna()].tolist()
        raise MissingCovariateError(f"NaN in covariate '{col}' for samples: {bad}")

    return result


def build_sample_geno_map(
    meta: "pd.DataFrame",
    sample_col: str = "sample_id",
    geno_col: str = "genotype",
) -> "dict[str, str]":
    """build a sample_id → genotype_id lookup from sample metadata.

    returns a dict so callers build it once and apply it to many sample lists
    without re-scanning the metadata DataFrame on each call.

    parameters
    ----------
    meta :
        sample metadata; must contain ``sample_col`` and ``geno_col``.
    sample_col :
        column holding sample identifiers (default ``'sample_id'``).
    geno_col :
        column holding the corresponding genotype/strain identifier
        (default ``'genotype'``); used to index into the genomecache.

    returns
    -------
    dict[str, str]
        maps each sample_id to its genotype_id.
        ``mapping.get(s, s)`` returns ``s`` unchanged when a sample has no
        separate genotype entry (e.g. inbred lines where sample == genotype).
    """
    return dict(zip(meta[sample_col], meta[geno_col]))


def encode_sex(series: "pd.Series") -> "pd.Series":
    """encode a sex column to float (male → 0.0, female → 1.0).

    accepts any of: 'Male'/'Female', 'M'/'F', 'm'/'f', 0/1, 0.0/1.0.
    unrecognised values default to 0.0 (male) with a logged warning.

    parameters
    ----------
    series : pd.Series
        raw sex column from sample metadata.

    returns
    -------
    pd.Series
        float64 series with same index, values in {0.0, 1.0}.
    """

    def _map(v: object) -> float:
        """map a sex value (string label or numeric) to 0.0 (male) or 1.0 (female)."""
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        s = str(v).strip().lower()
        if s in ("male", "m", "0"):
            return 0.0
        if s in ("female", "f", "1"):
            return 1.0
        logger.warning("encode_sex: unrecognised value %r — defaulting to 0.0", v)
        return 0.0

    return series.map(_map).astype(np.float64)


# ---------------------------------------------------------------------------
def build_cov_array(
    meta: "pd.DataFrame",
    sample_ids: "list[str]",
    cov_cols: "list[str]",
) -> "np.ndarray | None":
    """build a covariate array aligned to ``sample_ids``.

    convenience wrapper that combines :func:`build_covariate_matrix` with
    the reindex + missing-sample check used in every scan script.  Returns
    ``None`` when *cov_cols* is empty.

    parameters
    ----------
    meta :
        sample metadata; must contain a ``sample_id`` column plus any
        columns named in *cov_cols*.
    sample_ids :
        ordered list of sample identifiers that the covariate rows must
        align to.
    cov_cols :
        covariate column names to include.

    returns
    -------
    np.ndarray (n_samples, n_covariates) or None
        ``None`` when *cov_cols* is empty.

    raises
    ------
    ValueError
        when any sample in *sample_ids* has no covariate entry.
    """
    if not cov_cols:
        return None
    meta_scan = meta[meta['sample_id'].isin(sample_ids)].drop_duplicates('sample_id')
    cov_df = build_covariate_matrix(meta_scan, covariates=cov_cols)
    cov_aligned = cov_df.reindex(sample_ids)
    missing = cov_aligned.index[cov_aligned.isna().any(axis=1)].tolist()
    if missing:
        raise ValueError(
            f'covariates {cov_cols} missing for {len(missing)} sample(s): {missing}'
        )
    return cov_aligned.values


# private helpers
# ---------------------------------------------------------------------------


def _loco_sufficient(population: Optional[Union[str, Population]]) -> bool:
    """return True when LOCO kinship fully absorbs population structure.

    accepts a Population instance (preferred) or a string name for backward
    compatibility. the string path treats 'cc' as loco_sufficient=True since
    that was the only population supported before Population was introduced.
    """
    if population is None:
        return False
    if isinstance(population, Population):
        return population.loco_sufficient
    return population == "cc"


def _check_nan(df: pd.DataFrame, col: str) -> None:
    """raise MissingCovariateError if any value in col is NaN."""
    bad = df.index[df[col].isna()].tolist()
    if bad:
        raise MissingCovariateError(f"NaN in covariate '{col}' for samples: {bad}")


def _encode_column(meta: pd.DataFrame, col: str) -> pd.DataFrame:
    """one-hot encode categorical columns; pass numeric columns through.

    for categorical columns, the reference level (dropped dummy) is the first
    category whose value starts with 'founder' (case-insensitive) — the
    biological baseline for CC/DO mixed-origin datasets.  when no such
    category exists the alphabetically first value is dropped instead.
    """
    _check_nan(meta, col)
    series = meta[col]
    if pd.api.types.is_numeric_dtype(series):
        return series.rename(col).to_frame()
    unique_vals = sorted(series.unique(), key=str.lower)
    ref = next(
        (v for v in unique_vals if str(v).lower().startswith('founder')),
        unique_vals[0],
    )
    ordered = [ref] + [v for v in unique_vals if v != ref]
    cat_series = series.astype(pd.CategoricalDtype(categories=ordered))
    dummies = pd.get_dummies(cat_series, prefix=col, drop_first=True, dtype=float)
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
    scores = normed @ vt[::-1].T  # svds ranks smallest-to-largest, so reverse

    cols = [f"expr_pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(scores, index=sample_ids, columns=cols)


def _genotype_pcs(
    kinship_matrix: np.ndarray,
    kinship_sample_ids: list[str],
    sample_ids: pd.Index,
    n_pcs: int,
) -> pd.DataFrame:
    """top-n eigenvectors of the genome-wide kinship matrix.

    captures leftover population structure not already absorbed by LOCO
    kinship. eigenvalues come back smallest-to-largest, so they're reversed
    to put the strongest structure first.
    """
    k_df = pd.DataFrame(
        kinship_matrix, index=kinship_sample_ids, columns=kinship_sample_ids
    )
    k_sub = k_df.loc[sample_ids, sample_ids].values.astype(float)

    n_components = min(n_pcs, len(sample_ids) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(k_sub)
    idx = np.argsort(eigenvalues)[::-1]
    top_vecs = eigenvectors[:, idx[:n_components]]

    cols = [f"geno_pc{i + 1}" for i in range(n_components)]
    return pd.DataFrame(top_vecs, index=sample_ids, columns=cols)

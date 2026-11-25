"""
utils — shared utility functions

rank-inverse normal transformation, sample alignment, Z-matrix construction,
kinship/haplotype Z-expansion, design matrix builder, chromosome ordering,
and streaming-output helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy import stats
import logging

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger("mosaiqtl")


def chr_sort_key(c: str) -> int:
    """Numeric sort key for chromosome strings.

    Returns the integer value for autosomes, 20 for X, 21 for Y, 99 for
    anything else.  Use as ``key=chr_sort_key`` wherever chromosome lists
    or file paths keyed by chromosome need genomic ordering.

    Parameters
    ----------
    c : str
        Chromosome label, e.g. ``'1'``, ``'19'``, ``'X'``.

    Returns
    -------
    int
    """
    s = str(c).strip()
    if s.isdigit():
        return int(s)
    return {"X": 20, "Y": 21}.get(s.upper(), 99)


def add_chr_prefix(values: "pd.Series") -> "pd.Series":
    """Return chromosome labels as ``'chr{N}'`` strings.

    Labels that already start with ``'chr'`` and non-chromosome labels
    (e.g. ``'genome'``) pass through unchanged, so the call is idempotent.

    Parameters
    ----------
    values : pd.Series
        Chromosome labels such as ``'1'``, ``'19'``, ``'X'``.

    Returns
    -------
    pd.Series of str
    """
    s = values.astype(str)
    skip = s.str.startswith("chr") | (s == "genome") | (s == "nan")
    return s.where(skip, "chr" + s)


def strip_chr_prefix(values: "pd.Series") -> "pd.Series":
    """Return chromosome labels without a leading ``'chr'`` (``'chr19'`` → ``'19'``)."""
    return values.astype(str).str.replace(r"^chr", "", regex=True)


def strip_gene_version(values: "pd.Series") -> "pd.Series":
    """Return Ensembl gene IDs without the version suffix (``'ENSMUSG1.2'`` → ``'ENSMUSG1'``).

    IDs without a version pass through unchanged, so versioned and unversioned
    tables join on the same key.
    """
    return values.astype(str).str.replace(r"\.\d+$", "", regex=True)


def nearest_transcript_tss_mb(
    peak_df: "pd.DataFrame",
    gene_tss_df: "pd.DataFrame",
) -> "pd.Series":
    """Strand-aware transcript TSS (Mb) nearest each gene's peak, joined on unversioned gene_id.

    Parameters
    ----------
    peak_df :
        ``gene_id`` and ``peak_mb`` columns.
    gene_tss_df :
        One row per transcript: ``gene_id``, ``start``, ``end`` (bp) and ``strand``.
        The TSS is ``start`` on ``+`` and ``end`` on ``-``.

    Returns
    -------
    pd.Series
        Indexed as ``peak_df``; ``NaN`` where the gene has no transcript row.
    """
    import numpy as np
    import pandas as pd

    tss = pd.DataFrame(
        {
            "key": strip_gene_version(gene_tss_df["gene_id"]),
            "tss_mb": np.where(
                gene_tss_df["strand"].astype(str) == "-",
                gene_tss_df["end"],
                gene_tss_df["start"],
            ).astype(float)
            / 1e6,
        }
    ).drop_duplicates()
    left = pd.DataFrame(
        {"key": strip_gene_version(peak_df["gene_id"]), "peak_mb": peak_df["peak_mb"].to_numpy()},
        index=peak_df.index,
    )
    cand = left.reset_index(names="row").merge(tss, on="key", how="inner")
    if cand.empty:
        return pd.Series(np.nan, index=peak_df.index)
    cand["gap"] = (cand["tss_mb"] - cand["peak_mb"]).abs()
    best = cand.loc[cand.groupby("row")["gap"].idxmin()].set_index("row")["tss_mb"]
    return best.reindex(peak_df.index)


def gene_tss_anchor(gene_tss_df: "pd.DataFrame") -> "pd.DataFrame":
    """Most upstream strand-aware TSS (Mb) of each gene, keyed on unversioned gene_id.

    The anchor is the smallest ``start`` over a ``+`` gene's transcripts and the
    largest ``end`` over a ``-`` gene's, so it depends only on the gene and is the
    same in every cell type and for every peak.

    Parameters
    ----------
    gene_tss_df :
        One row per transcript: ``gene_id``, ``start``, ``end`` (bp) and ``strand``.

    Returns
    -------
    pd.DataFrame
        Indexed by unversioned ``gene_id`` with ``tss_mb`` and ``sign`` (-1 for
        ``-`` strand genes, else 1).
    """
    import numpy as np
    import pandas as pd

    minus = gene_tss_df["strand"].astype(str).eq("-").to_numpy()
    tx = pd.DataFrame(
        {
            "key": strip_gene_version(gene_tss_df["gene_id"]).to_numpy(),
            "pos": np.where(minus, gene_tss_df["end"], gene_tss_df["start"]).astype(float) / 1e6,
            "minus": minus,
        }
    )
    grouped = tx.groupby("key")
    tss = np.where(grouped["minus"].first(), grouped["pos"].max(), grouped["pos"].min())
    return pd.DataFrame(
        {"tss_mb": tss, "sign": np.where(grouped["minus"].first(), -1.0, 1.0)},
        index=grouped["pos"].min().index,
    )


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
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    n = valid.sum()
    out = np.full_like(x, np.nan)
    ranks = stats.rankdata(x[valid], method="average")
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
            idx_a.append(i)
            idx_b.append(b_map[sid])
            shared.append(sid)
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
    geno_ids = list(geno_ids)
    geno_map = {g: j for j, g in enumerate(geno_ids)}
    n, m = len(sample_ids), len(geno_ids)
    Z = np.zeros((n, m), dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, sid in enumerate(sample_ids):
        if sid in geno_map:
            Z[i, geno_map[sid]] = 1.0
            valid[i] = True
    n_miss = int((~valid).sum())
    if n_miss:
        logger.warning(f"build_z_matrix: {n_miss}/{n} samples missing genotype")
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


def get_rss_mb() -> float:
    """Return the current process RSS in MB (macOS: bytes → MB; Linux: kB → MB)."""
    import resource
    import sys

    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == "darwin":
        return usage.ru_maxrss / 1024**2
    return usage.ru_maxrss / 1024


def git_hash() -> str:
    """Return the short git commit hash of HEAD, or ``'unknown'`` on failure."""
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


def build_tss_lookup(genes_df: "pd.DataFrame") -> "dict[str, float]":
    """Build a gene_id → TSS position (Mb) mapping from a gene-info DataFrame.

    Uses strand-aware boundary selection: ``start`` column for ``+`` strand,
    ``end`` column for ``-`` strand.  Falls back to ``start`` when the
    ``strand`` column is absent.

    Parameters
    ----------
    genes_df :
        DataFrame with at minimum a ``gene_id`` column and ``start`` column
        (base-pair coordinates).  Optional: ``end``, ``strand``.

    Returns
    -------
    dict[str, float]
        Mapping from gene_id to TSS position in Mb.
    """
    if "start" not in genes_df.columns or "gene_id" not in genes_df.columns:
        return {}
    tss = genes_df["start"]
    if {"strand", "end"} <= set(genes_df.columns):
        tss = tss.where(genes_df["strand"] != "-", genes_df["end"])
    return dict(zip(genes_df["gene_id"].astype(str), tss.astype(float) / 1e6))


def reduce_to_strain_means(
    y: np.ndarray,
    Z: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    """Reduce a sample-level vector to strain means via the Z design matrix.

    Parameters
    ----------
    y :
        Sample-level vector of shape ``(n_samples,)``.
    Z :
        Design matrix of shape ``(n_samples, m)`` mapping samples to strains,
        as returned by :func:`build_z_matrix`.
    valid_mask :
        Boolean mask of length ``n_samples`` marking samples that have a valid
        strain assignment, as returned by :func:`build_z_matrix`.

    Returns
    -------
    np.ndarray
        Strain-mean vector of shape ``(m,)``.  Strains with no valid replicates
        receive ``0.0``.
    """
    Z_sub = Z[valid_mask]
    rep_counts = Z_sub.sum(axis=0)
    s = Z_sub.T @ y[valid_mask]
    return np.where(rep_counts > 0, s / rep_counts, 0.0)


def compute_near_tss_mb(
    peak_df: "pd.DataFrame",
    gene_tss_df: "pd.DataFrame",
) -> "pd.Series":
    """For each gene in ``peak_df``, find the TSS in ``gene_tss_df`` nearest to the peak.

    Parameters
    ----------
    peak_df :
        DataFrame with ``gene`` and ``peak_mb`` columns.
    gene_tss_df :
        DataFrame with ``gene_symbol`` and ``tss_mb`` columns listing all known
        transcript TSSs per gene.

    Returns
    -------
    pd.Series
        Indexed as ``peak_df``, values are the nearest TSS position in Mb
        (``NaN`` when the gene is absent from ``gene_tss_df``).
    """
    near: dict[str, float] = {}
    for _, row in peak_df.iterrows():
        cands = gene_tss_df[gene_tss_df["gene_symbol"] == row["gene_symbol"]]
        if cands.empty:
            near[row["gene_symbol"]] = float("nan")
        else:
            idx = (cands["tss_mb"] - row["peak_mb"]).abs().idxmin()
            near[row["gene_symbol"]] = float(cands.loc[idx, "tss_mb"])
    return peak_df["gene_symbol"].map(near)


def _build_gm_idx(
    gene_meta: "pd.DataFrame | None",
) -> "dict[str, dict] | None":
    """Build a gene → {col: val} lookup from a gene_meta DataFrame.

    Parameters
    ----------
    gene_meta : DataFrame with a 'gene_id' column (the stable identity key;
        gene_symbol travels along as a plain data column, joined in for
        display), or None.

    Returns
    -------
    dict keyed by gene_id, or None when gene_meta is absent.
    """
    if gene_meta is not None and not gene_meta.empty and "gene_id" in gene_meta.columns:
        return gene_meta.set_index("gene_id").to_dict("index")
    return None


def _enrich_scan_df(
    df: "pd.DataFrame",
    gene_id: str,
    gm_idx: "dict[str, dict] | None",
    id_cols: "list[str]",
    data_cols: "list[str]",
) -> "pd.DataFrame":
    """Inject gene_meta columns and reorder a per-gene scan DataFrame.

    Column order: id_cols → gene_meta columns → data_cols → remaining.
    Gene_meta columns are injected only when not already present in df,
    so the function's own columns always take precedence.

    Parameters
    ----------
    df        : per-gene scan result (modified in-place before reorder).
    gene_id   : key into gm_idx.
    gm_idx    : output of _build_gm_idx, or None.
    id_cols   : identifier columns that lead the output (e.g. ['gene_id', 'chr']).
    data_cols : scan-result columns that follow the gene_meta columns.

    Returns
    -------
    pd.DataFrame with columns in the canonical order.
    """
    meta_cols: "list[str]" = []
    if gm_idx is not None and gene_id in gm_idx:
        for col, val in gm_idx[gene_id].items():
            if col not in df.columns:
                df[col] = val
                meta_cols.append(col)
    ordered = id_cols + meta_cols + data_cols
    ordered_set = set(ordered)
    return df[
        [c for c in ordered if c in df.columns]
        + [c for c in df.columns if c not in ordered_set]
    ]


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
    parts = [np.ones((len(df), 1))]
    names = ["intercept"]
    for col in num:
        parts.append(df[col].values[:, None].astype(np.float64))
        names.append(col)
    for col in cat:
        d = pd.get_dummies(df[col], prefix=col, drop_first=True)
        parts.append(d.values.astype(np.float64))
        names.extend(d.columns.tolist())
    return np.hstack(parts), names

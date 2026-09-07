"""anndata_io — helpers for loading expression data from HDF5 and AnnData files."""

from __future__ import annotations

from pathlib import Path

import numpy as np

__all__ = [
    "load_expression_h5",
]


def load_expression_h5(
    h5_path: Path,
    genes_wanted: list[str],
    counts_key: str = "counts",
    genes_key: str = "genes",
    samples_key: str = "samples",
) -> tuple[np.ndarray, list[str], list[str]]:
    """load a subset of genes from an expression HDF5 file, by name.

    only the requested genes are read off disk — the whole expression
    matrix is never loaded into memory at once. the file is expected to
    contain::

        f['genes']   : (n_genes,)   gene identifier strings
        f['samples'] : (n_samples,) sample identifier strings
        f['counts']  : (n_genes, n_samples)  expression matrix

    parameters
    ----------
    h5_path :
        path to the HDF5 file (opened read-only).
    genes_wanted :
        gene identifiers to fetch. the returned matrix is ordered by each
        gene's position in the file, not by the order of ``genes_wanted``.
    counts_key :
        dataset name for the expression matrix (default ``'counts'``).
    genes_key :
        dataset name for gene identifiers (default ``'genes'``).
    samples_key :
        dataset name for sample identifiers (default ``'samples'``).

    returns
    -------
    counts : np.ndarray (n_found, n_samples) float32
        expression values for the found genes.
    genes : list[str]
        gene identifiers in the order they appear in the file.
        length equals the number of genes in ``genes_wanted`` that
        were present in the file.
    samples : list[str]
        sample identifiers in file order. identical for every call on
        the same file — the sample axis never changes.
    """
    import h5py  # optional dependency, only needed if this function is called

    with h5py.File(h5_path, "r") as f:
        all_genes = np.array(f[genes_key]).astype(str).tolist()
        all_samples = np.array(f[samples_key]).astype(str).tolist()
        counts_ds = f[counts_key]
        wanted_set = set(genes_wanted)
        idx = [i for i, g in enumerate(all_genes) if g in wanted_set]
        if not idx:
            return np.empty((0, len(all_samples)), dtype=np.float32), [], all_samples
        counts = np.array(counts_ds[idx, :], dtype=np.float32)
    genes_out = [all_genes[i] for i in idx]
    return counts, genes_out, all_samples

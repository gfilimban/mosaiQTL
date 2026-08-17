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
    """Load a subset of genes from a row-major expression HDF5 file.

    Only the requested gene rows are read from disk — no full-matrix
    materialisation.  The file layout expected is::

        f['genes']   : (n_genes,)   gene identifier strings
        f['samples'] : (n_samples,) sample identifier strings
        f['counts']  : (n_genes, n_samples)  float or int expression matrix

    Parameters
    ----------
    h5_path :
        Path to the HDF5 file (opened read-only).
    genes_wanted :
        Gene identifiers to fetch.  Order of the returned matrix matches
        the order in which each gene first appears in the file, not the
        order of ``genes_wanted``.
    counts_key :
        Dataset name for the expression matrix (default ``'counts'``).
    genes_key :
        Dataset name for gene identifiers (default ``'genes'``).
    samples_key :
        Dataset name for sample identifiers (default ``'samples'``).

    Returns
    -------
    counts : np.ndarray (n_found, n_samples) float32
        Expression values for the found genes.
    genes : list[str]
        Gene identifiers in the order they appear in the file.
        Length equals the number of genes in ``genes_wanted`` that
        were present in the file.
    samples : list[str]
        Sample identifiers in file order.  Identical for every call on
        the same file — the sample axis never changes.
    """
    import h5py  # optional dep — only needed when this function is called

    with h5py.File(h5_path, "r") as f:
        all_genes = np.array(f[genes_key]).astype(str).tolist()
        all_samples = np.array(f[samples_key]).astype(str).tolist()
        counts_ds = f[counts_key]  # (n_genes_total, n_samples)
        wanted_set = set(genes_wanted)
        idx = [i for i, g in enumerate(all_genes) if g in wanted_set]
        if not idx:
            return np.empty((0, len(all_samples)), dtype=np.float32), [], all_samples
        counts = np.array(counts_ds[idx, :], dtype=np.float32)
    genes_out = [all_genes[i] for i in idx]
    return counts, genes_out, all_samples

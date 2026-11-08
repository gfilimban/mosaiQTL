"""
kinship — genetic relatedness matrix computation

a kinship matrix estimates how genetically similar every pair of animals is,
based on how much of the genome they share the same founder ancestry at:

    K_{ij} = Σ_{l} p_{il} · p_{jl}  /  n_loci

where p_{il} is the founder-probability vector for animal i at marker l
(the 8 probabilities sum to 1). with this formula, an inbred founder is
K_ii = 1 (identical to itself), an F1 hybrid is K_ii = 0.5, and a Collaborative
Cross mouse falls around K_ii ≈ 0.2-0.35 depending on how heterozygous its
genome still is.

for "leave one chromosome out" (LOCO) kinship, each chromosome's contribution
is tracked separately and subtracted out when that chromosome is the one being
scanned for QTL. this avoids the test marker inflating its own relatedness
estimate (proximal contamination):

    K_loco(c)_{ij} = (Σ_{l not on c} p_{il} · p_{jl})  /  (n_loci - n_loci_c)

supports three modes: kinship from the whole genome, LOCO kinship (one matrix
per chromosome, excluding that chromosome), and per-chromosome kinship (using
only that chromosome).
"""

import time
from pathlib import Path
from typing import Optional
import numpy as np
import logging

logger = logging.getLogger("mosaiqtl")

__all__ = [
    "calc_kinship",
    "decomp_kinship",
    "kinship_from_decomp",
    "load_kinship_decomp",
    "save_kinship_decomp",
]


def calc_kinship(genomecache, method="loco"):
    """compute kinship matrices from haplotype probabilities.

    parameters
    ----------
    genomecache : GenomeCache
    method : str
        'overall' — single matrix from all loci
        'loco'    — one matrix per chromosome, excluding that chr
        'chr'     — one matrix per chromosome, using only that chr

    returns
    -------
    dict  {'overall': K} | {chrom: K}  each K is (n, n)
    """
    n_chr = len(genomecache.chromosomes)
    n_strains = len(genomecache.sample_ids)
    logger.info(
        "calc_kinship: method=%r  n_chr=%d  n_strains=%d",
        method,
        n_chr,
        n_strains,
    )
    _t0 = time.perf_counter()

    chr_num = {}  # {chrom: Σ_l p_l p_l^T}
    chr_nloci = {}  # {chrom: number of loci}

    for ci, chrom in enumerate(genomecache.chromosomes):
        _, _, probs = genomecache.get_chromosome(chrom)
        # probs: (L, n, J)  allele probabilities summing to 1 per sample per locus
        chr_num[chrom] = np.einsum("lni,lmi->nm", probs, probs, optimize=True)
        chr_nloci[chrom] = probs.shape[0]
        logger.debug(
            "calc_kinship: chr%s (%d/%d)  n_loci=%d  elapsed=%.1fs",
            chrom,
            ci + 1,
            n_chr,
            chr_nloci[chrom],
            time.perf_counter() - _t0,
        )

    num_tot = sum(chr_num.values())
    nloci_tot = sum(chr_nloci.values())

    def _norm(K: np.ndarray) -> np.ndarray:
        """normalise K so the mean diagonal entry equals 1."""
        d = np.diag(K).mean()
        return K / d if d > 0 else K

    if method == "overall":
        result = {"overall": _norm(num_tot / nloci_tot)}
    elif method == "loco":
        result = {
            c: _norm((num_tot - chr_num[c]) / (nloci_tot - chr_nloci[c]))
            for c in chr_num
        }
    elif method == "chr":
        result = {c: _norm(chr_num[c] / chr_nloci[c]) for c in chr_num}
    else:
        raise ValueError(f"unknown kinship method: {method!r}")

    _example = next(iter(result.values()))
    _mean_diag = float(np.diag(_example).mean())
    logger.info(
        "calc_kinship: done  n_keys=%d  mean_diag=%.4f  elapsed=%.1fs",
        len(result),
        _mean_diag,
        time.perf_counter() - _t0,
    )
    return result


def decomp_kinship(kinship_dict):
    """eigendecompose kinship matrices for fast LMM fitting.

    K = U D U' enables diagonal V^{-1} in the rotated coordinate system,
    avoiding repeated n×n inversions during the h² grid search and scan.

    parameters
    ----------
    kinship_dict : dict  {key: K}  from calc_kinship()

    returns
    -------
    dict  {key: {'values': λ, 'vectors': U}}  eigenvalues descending
    """
    n_keys = len(kinship_dict)
    _example_k = next(iter(kinship_dict.values()))
    logger.info(
        "decomp_kinship: n_matrices=%d  n_strains=%d",
        n_keys,
        _example_k.shape[0],
    )
    result = {}
    for key, k in kinship_dict.items():
        vals, vecs = np.linalg.eigh((k + k.T) / 2.0)  # symmetrize → eigh
        order = np.argsort(vals)[::-1]
        result[key] = {
            "values": np.maximum(vals[order], 0.0),  # floor negative noise
            "vectors": vecs[:, order],
        }
        _ev = result[key]["values"]
        logger.debug(
            "decomp_kinship[%s]: n=%d  λ_range=[%.4f, %.4f]  top-3=%s",
            key,
            len(_ev),
            float(_ev[-1]),
            float(_ev[0]),
            ", ".join(f"{v:.4f}" for v in _ev[:3]),
        )
    return result


def kinship_from_decomp(decomp_chr: dict) -> np.ndarray:
    """reconstruct the kinship matrix K from a per-chromosome decomp dict.

    given the eigendecomposition produced by :func:`decomp_kinship` for one
    chromosome, recover the symmetric kinship matrix::

        K = (vectors * values) @ vectors.T

    parameters
    ----------
    decomp_chr :
        ``{'values': ndarray (m,), 'vectors': ndarray (m, m)}`` for one chromosome.

    returns
    -------
    K : ndarray (m, m)
    """
    vecs = decomp_chr['vectors']
    vals = decomp_chr['values']
    return (vecs * vals) @ vecs.T


def save_kinship_decomp(kinship_decomp: dict, h5_path: Path) -> None:
    """write a LOCO kinship decomp dict to an HDF5 file.

    writes the same layout as build_kinship_loco.py so the file can
    be re-loaded by :func:`load_kinship_decomp`.

    parameters
    ----------
    kinship_decomp :
        ``{chrom_str: {'values': ndarray, 'vectors': ndarray}}``
    h5_path :
        destination path; parent directories must exist.
    """
    import h5py
    with h5py.File(h5_path, 'w') as f:
        for chrom, d in kinship_decomp.items():
            grp = f.create_group(f'chr{chrom}')
            grp.create_dataset('values',  data=d['values'].astype(np.float64))
            grp.create_dataset('vectors', data=d['vectors'].astype(np.float64))


def load_kinship_decomp(
    h5_path: Path,
    chroms: Optional[list[str]] = None,
) -> dict:
    """load pre-decomposed LOCO kinship matrices from HDF5.

    reads the file written by build_kinship_loco.py: each group
    ``chr{c}`` holds ``values`` (eigenvalues) and ``vectors`` (eigenvectors).

    parameters
    ----------
    h5_path :
        path to the kinship HDF5 file (opened read-only).
    chroms :
        chromosomes to load; ``None`` loads all present chromosomes.
        pass a single-element list to load one chromosome without reading
        the whole file (avoids the I/O cost for per-chromosome LMM scans).

    returns
    -------
    dict  {chrom_str: {'values': ndarray, 'vectors': ndarray}}
        chromosome strings match those stored in the file (no ``chr`` prefix).
    """
    import h5py
    want = set(chroms) if chroms is not None else None
    result: dict = {}
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            if not key.startswith('chr'):
                continue
            chrom = key[3:]
            if want is not None and chrom not in want:
                continue
            result[chrom] = {
                'values':  f[key]['values'][:],
                'vectors': f[key]['vectors'][:],
            }
    return result

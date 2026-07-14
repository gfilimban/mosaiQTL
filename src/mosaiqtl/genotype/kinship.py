"""
kinship — kinship matrix computation

Normalized dot-product kinship, consistent with qtl2::calc_kinship(type='loco').

    K_{ij} = Σ_{l} p_{il} · p_{jl}  /  n_loci

where p_{il} is the J-dimensional allele-probability vector for sample i at
locus l (sums to 1).  This formula gives inbred founders K_ii = 1, F1 hybrids
K_ii = 0.5, and CC mice K_ii ≈ 0.2–0.35 depending on heterozygosity —
matching qtl2::calc_kinship output exactly.

For LOCO: numerators and locus counts are accumulated per chromosome; the
target chromosome contribution is subtracted before dividing.

    K_loco(c)_{ij} = (Σ_{l ∉ c} p_{il} · p_{jl})  /  (n_loci − n_loci_c)

supports: overall, loco, and per-chromosome kinship.
"""

import time
import numpy as np
import logging

logger = logging.getLogger('mosaiqtl')

__all__ = [
    'calc_kinship',
    'decomp_kinship',
]


def calc_kinship(genomecache, method='loco'):
    """Compute kinship matrices from haplotype probabilities.

    Consistent with qtl2::calc_kinship(genoprob_to_alleleprob(probs), type=...).

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
        'calc_kinship: method=%r  n_chr=%d  n_strains=%d',
        method, n_chr, n_strains,
    )
    _t0 = time.perf_counter()

    chr_num   = {}   # {chrom: Σ_l p_l p_l^T}
    chr_nloci = {}   # {chrom: number of loci}

    for chrom in genomecache.chromosomes:
        _, _, probs = genomecache.get_chromosome(chrom)
        # probs: (L, n, J)  allele probabilities summing to 1 per sample per locus
        chr_num[chrom]   = np.einsum('lni,lmi->nm', probs, probs, optimize=True)
        chr_nloci[chrom] = probs.shape[0]

    num_tot   = sum(chr_num.values())
    nloci_tot = sum(chr_nloci.values())

    def _norm(K: np.ndarray) -> np.ndarray:
        """Normalise K so the mean diagonal entry equals 1."""
        d = np.diag(K).mean()
        return K / d if d > 0 else K

    if method == 'overall':
        result = {'overall': _norm(num_tot / nloci_tot)}
    elif method == 'loco':
        result = {c: _norm((num_tot - chr_num[c]) / (nloci_tot - chr_nloci[c]))
                  for c in chr_num}
    elif method == 'chr':
        result = {c: _norm(chr_num[c] / chr_nloci[c]) for c in chr_num}
    else:
        raise ValueError(f'unknown kinship method: {method!r}')

    _example = next(iter(result.values()))
    _mean_diag = float(np.diag(_example).mean())
    logger.info(
        'calc_kinship: done  n_keys=%d  mean_diag=%.4f  elapsed=%.1fs',
        len(result), _mean_diag, time.perf_counter() - _t0,
    )
    return result


def decomp_kinship(kinship_dict):
    """Eigendecompose kinship matrices for fast LMM fitting.

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
        'decomp_kinship: n_matrices=%d  n_strains=%d',
        n_keys, _example_k.shape[0],
    )
    result = {}
    for key, k in kinship_dict.items():
        vals, vecs = np.linalg.eigh((k + k.T) / 2.0)   # symmetrize → eigh
        order = np.argsort(vals)[::-1]
        result[key] = {
            'values':  np.maximum(vals[order], 0.0),    # floor negative noise
            'vectors': vecs[:, order],
        }
        _ev = result[key]['values']
        logger.debug(
            'decomp_kinship[%s]: n=%d  λ_range=[%.4f, %.4f]  top-3=%s',
            key, len(_ev), float(_ev[-1]), float(_ev[0]),
            ', '.join(f'{v:.4f}' for v in _ev[:3]),
        )
    return result

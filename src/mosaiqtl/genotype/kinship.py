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

import numpy as np
import logging

logger = logging.getLogger('mosaiqtl')


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
        d = np.diag(K).mean()
        return K / d if d > 0 else K

    if method == 'overall':
        return {'overall': _norm(num_tot / nloci_tot)}

    elif method == 'loco':
        return {c: _norm((num_tot - chr_num[c]) / (nloci_tot - chr_nloci[c]))
                for c in chr_num}

    elif method == 'chr':
        return {c: _norm(chr_num[c] / chr_nloci[c]) for c in chr_num}

    else:
        raise ValueError(f'unknown kinship method: {method!r}')


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
    result = {}
    for key, k in kinship_dict.items():
        vals, vecs = np.linalg.eigh((k + k.T) / 2.0)   # symmetrize → eigh
        order = np.argsort(vals)[::-1]
        result[key] = {
            'values':  np.maximum(vals[order], 0.0),    # floor negative noise
            'vectors': vecs[:, order],
        }
        logger.debug(f'kinship[{key}]: top-3 λ={result[key]["values"][:3]}')
    return result

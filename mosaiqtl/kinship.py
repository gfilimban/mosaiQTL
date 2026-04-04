"""
mosaiqtl.kinship — kinship matrix computation

realized genomic relationship matrix (van Raden) from founder haplotype
probabilities.

    K_ij = (1/L) Σ_l h_il · h_jl

where h_il ∈ ℝ^J is the founder probability vector for sample i at locus l.
this is allele-sharing kinship, positive semi-definite by construction.

supports: overall, loco, and per-chromosome kinship.
"""

import numpy as np
import logging

logger = logging.getLogger('mosaiqtl')


def calc_kinship(genome_cache, method='loco'):
    """compute kinship matrices from haplotype probabilities.

    parameters
    ----------
    genome_cache : GenomeCache
    method : str
        'overall' — single matrix from all loci
        'loco'    — one matrix per chromosome, excluding that chr
        'chr'     — one matrix per chromosome, using only that chr

    returns
    -------
    dict  {'overall': K} | {chrom: K}  each K is (n, n)
    """
    n = len(genome_cache.sample_ids)
    chr_kin = {}   # {chrom: (K_contribution, n_loci)}

    for chrom in genome_cache.chromosomes:
        _, _, probs = genome_cache.get_chromosome(chrom)
        # probs: (L, n, J) — vectorised: K_c = sum_l H_l H_l' = einsum
        # H[l] is (n,J); H[l] @ H[l].T summed over l = einsum('lni,lmi->nm')
        k = np.einsum('lni,lmi->nm', probs, probs, optimize=True)
        chr_kin[chrom] = (k, probs.shape[0])

    if method == 'overall':
        k_tot = sum(k for k, _ in chr_kin.values())
        n_tot = sum(n for _, n in chr_kin.values())
        return {'overall': k_tot / n_tot}

    elif method == 'loco':
        k_tot = sum(k for k, _ in chr_kin.values())
        n_tot = sum(n for _, n in chr_kin.values())
        return {c: (k_tot - k) / (n_tot - nl)
                for c, (k, nl) in chr_kin.items()}

    elif method == 'chr':
        return {c: k / nl for c, (k, nl) in chr_kin.items()}

    else:
        raise ValueError(f'unknown kinship method: {method!r}')


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

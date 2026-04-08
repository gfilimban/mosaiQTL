"""
mosaiqtl.kinship — kinship matrix computation

van Raden (2008) realized genomic relationship matrix, consistent with
qtl2::calc_kinship(probs_allele, type='loco').

    K_{ij} = Σ_{l,f} (p_{ilf} − p̄_{lf})(p_{jlf} − p̄_{lf})
             ─────────────────────────────────────────────────
                     Σ_{l,f} p̄_{lf} (1 − p̄_{lf})

where p_{ilf} = allele probability ∈ [0,1] for sample i, locus l, founder f.
p̄_{lf} = mean over samples.  Python cache stores probs already in [0,1]
(= qtl2 allele dosage / 2), so no additional scaling is needed.

For LOCO the numerator and denominator are accumulated per chromosome and
the target chromosome is subtracted — exact because p̄ is computed from
samples (fixed) not from the locus set.

supports: overall, loco, and per-chromosome kinship.
"""

import numpy as np
import logging

logger = logging.getLogger('mosaiqtl')


def calc_kinship(genome_cache, method='loco'):
    """Compute van Raden kinship matrices from haplotype probabilities.

    Consistent with qtl2::calc_kinship(genoprob_to_alleleprob(probs), type=...).

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
    chr_num   = {}   # {chrom: K_numerator}   sum_l M_l M_l'
    chr_denom = {}   # {chrom: scalar}         sum_{l,f} p̄(1-p̄)

    for chrom in genome_cache.chromosomes:
        _, _, probs = genome_cache.get_chromosome(chrom)
        # probs: (L, n, J)  already in [0,1]
        p_bar = probs.mean(axis=1)          # (L, J)
        M     = probs - p_bar[:, None, :]   # (L, n, J)  centred

        chr_num[chrom]   = np.einsum('lni,lmi->nm', M, M, optimize=True)
        chr_denom[chrom] = float((p_bar * (1.0 - p_bar)).sum())

    num_tot   = sum(chr_num.values())
    denom_tot = sum(chr_denom.values())

    if method == 'overall':
        return {'overall': _safe_div(num_tot, denom_tot)}

    elif method == 'loco':
        return {c: _safe_div(num_tot - chr_num[c], denom_tot - chr_denom[c])
                for c in chr_num}

    elif method == 'chr':
        return {c: _safe_div(chr_num[c], chr_denom[c]) for c in chr_num}

    else:
        raise ValueError(f'unknown kinship method: {method!r}')


def _safe_div(num, denom):
    """Return num / denom; warn and fall back to num if denom ~ 0."""
    if abs(denom) < 1e-12:
        logger.warning('kinship denominator near zero — returning raw numerator')
        return num
    return num / denom


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

"""
mosaiqtl.kinship — kinship matrix computation

computes realized genomic relationship matrices from founder
haplotype probabilities, matching qtl2::calc_kinship().

the kinship between subjects i and j is:
  K_ij = (1 / n_loci) Σ_l Σ_f H_ilf · H_jlf

where H_ilf is the probability that subject i carries founder f
at locus l. this is the allele-sharing kinship.

supports:
  - overall kinship (all loci)
  - loco (leave-one-chromosome-out) kinship
  - precomputed eigendecompositions for fast lmm fitting
"""

import numpy as np
import logging

logger = logging.getLogger('mosaiqtl')


def calc_kinship(genome_cache, method='loco'):
    """compute kinship matrices from haplotype probabilities.

    parameters
    ----------
    genome_cache : GenomeCache
        haplotype probability data
    method : str
        'overall' — single kinship from all loci
        'loco'    — one kinship per chromosome, excluding that chr
        'chr'     — one kinship per chromosome, using only that chr

    returns
    -------
    dict
        if 'overall': {'overall': K}
        if 'loco': {chrom: K_without_chrom}
        if 'chr': {chrom: K_from_chrom}
        each K is np.ndarray (n_subjects, n_subjects)
    """
    chromosomes = genome_cache.chromosomes
    n_subjects = len(genome_cache.sample_ids)

    # step 1: compute per-chromosome kinship contributions
    # K_chr[c] = sum over loci on chr c of H_l H_l'
    # and count of loci per chromosome
    chr_kinship = {}  # {chrom: (K_contribution, n_loci)}

    for chrom in chromosomes:
        _, _, probs = genome_cache.get_chromosome(chrom)
        # probs shape: (n_loci, n_subjects, n_founders)
        n_loci = probs.shape[0]

        # compute kinship contribution for this chromosome
        # K_c = sum_l (H_l @ H_l') where H_l is (n_subjects, n_founders)
        k_contrib = np.zeros((n_subjects, n_subjects))

        for l_idx in range(n_loci):
            # haplotype probability matrix at this locus
            h_l = probs[l_idx]  # (n_subjects, n_founders)
            # allele-sharing: H_l @ H_l.T
            k_contrib += h_l @ h_l.T

        chr_kinship[chrom] = (k_contrib, n_loci)

    # step 2: assemble final kinship based on method
    if method == 'overall':
        # sum all chromosome contributions, divide by total loci
        k_total = np.zeros((n_subjects, n_subjects))
        n_total = 0
        for chrom in chromosomes:
            k_c, n_c = chr_kinship[chrom]
            k_total += k_c
            n_total += n_c
        return {'overall': k_total / n_total}

    elif method == 'loco':
        # for each chromosome, sum all OTHER chromosomes
        # precompute total
        k_total = np.zeros((n_subjects, n_subjects))
        n_total = 0
        for chrom in chromosomes:
            k_c, n_c = chr_kinship[chrom]
            k_total += k_c
            n_total += n_c

        result = {}
        for chrom in chromosomes:
            k_c, n_c = chr_kinship[chrom]
            # subtract this chromosome's contribution
            k_loco = (k_total - k_c) / (n_total - n_c)
            result[chrom] = k_loco

        return result

    elif method == 'chr':
        result = {}
        for chrom in chromosomes:
            k_c, n_c = chr_kinship[chrom]
            result[chrom] = k_c / n_c
        return result

    else:
        raise ValueError(f'unknown kinship method: {method}')


def decomp_kinship(kinship_dict):
    """eigendecompose kinship matrices for fast lmm fitting.

    the eigendecomposition K = U D U' allows the lmm to be
    solved as weighted least squares in rotated coordinates,
    avoiding repeated n×n matrix inversions.

    parameters
    ----------
    kinship_dict : dict
        {key: K_matrix} from calc_kinship()

    returns
    -------
    dict
        {key: {'values': eigenvalues, 'vectors': eigenvectors}}
        eigenvalues sorted descending, vectors as columns of U
    """
    result = {}

    for key, k_mat in kinship_dict.items():
        # symmetrize (numerical safety)
        k_sym = (k_mat + k_mat.T) / 2.0

        # eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(k_sym)

        # sort descending
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        # floor small negative eigenvalues (numerical noise)
        eigvals = np.maximum(eigvals, 0.0)

        result[key] = {
            'values': eigvals,
            'vectors': eigvecs,
        }

        logger.debug(
            f'kinship[{key}]: top 3 eigenvalues = '
            f'{eigvals[:3]}, condition = {eigvals[0] / max(eigvals[-1], 1e-15):.1f}'
        )

    return result


def kinship_to_correlation(k_mat):
    """normalize kinship to correlation scale (diagonal = 1).

    parameters
    ----------
    k_mat : np.ndarray (n, n)
        raw kinship matrix

    returns
    -------
    np.ndarray (n, n)
        correlation-scaled kinship
    """
    # diagonal elements
    d = np.sqrt(np.diag(k_mat))
    # outer product of diagonal
    d_outer = np.outer(d, d)
    # avoid division by zero
    d_outer = np.maximum(d_outer, 1e-15)
    return k_mat / d_outer

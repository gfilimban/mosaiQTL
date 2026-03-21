"""
mosaiqtl.scan — genome-wide haplotype qtl scan

orchestrates the scan across all chromosomes for a single
phenotype (gene), combining genome cache, kinship, and the
lmm or nebula-ln engine.

equivalent to miqtl::scan.h2lmm() with loco kinship.
"""

import numpy as np
import pandas as pd
import logging
from . import kinship as kin_module
from . import lmm as lmm_module
from .utils import rint, drop_colinear_founders

logger = logging.getLogger('mosaiqtl')


class ScanResult:
    """container for genome-wide scan results.

    attributes
    ----------
    locus_ids : np.ndarray of str
        marker identifiers across all chromosomes
    chromosomes : np.ndarray of str
        chromosome for each locus
    positions : np.ndarray of float
        physical position (bp) for each locus
    lod : np.ndarray of float
        lod scores
    p_value : np.ndarray of float
        p-values
    betas : np.ndarray (n_loci, n_founders-1)
        founder haplotype effects
    h2_null : dict
        {chrom: h2_estimate} under null model per loco chromosome
    phenotype_id : str
        gene/phenotype name
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def peak_locus(self):
        """return the locus with highest lod score."""
        idx = np.argmax(self.lod)
        return {
            'locus_id': self.locus_ids[idx],
            'chromosome': self.chromosomes[idx],
            'position': self.positions[idx],
            'lod': self.lod[idx],
            'p_value': self.p_value[idx],
            'betas': self.betas[idx],
        }

    def to_dataframe(self):
        """convert scan results to pandas dataframe."""
        df = pd.DataFrame({
            'locus_id': self.locus_ids,
            'chromosome': self.chromosomes,
            'position': self.positions,
            'lod': self.lod,
            'p_value': self.p_value,
        })
        # add founder effect columns
        n_founders = self.betas.shape[1]
        for f_idx in range(n_founders):
            df[f'beta_f{f_idx}'] = self.betas[:, f_idx]
        return df


def scan_h2lmm(phenotype, genome_cache, covariates=None,
               kinship_decomp=None, use_rint=True,
               use_reml=True, phenotype_id='unknown'):
    """run haplotype-based genome scan with lmm and loco kinship.

    this is the primary scan function, equivalent to
    miqtl::scan.h2lmm().

    parameters
    ----------
    phenotype : np.ndarray (n_subjects,)
        phenotype values (one gene). nan values are excluded.
    genome_cache : GenomeCache
        haplotype probability data
    covariates : np.ndarray (n_subjects, p) or None
        covariate design matrix. if none, intercept-only model.
    kinship_decomp : dict or None
        {chrom: {'values': eigvals, 'vectors': eigvecs}}
        precomputed loco kinship eigendecompositions.
        if none, computed internally.
    use_rint : bool
        apply rank-inverse normal transformation to phenotype
    use_reml : bool
        use reml (True) or ml (False) for variance estimation
    phenotype_id : str
        name of the phenotype (for logging)

    returns
    -------
    ScanResult
    """
    n = len(phenotype)

    # handle missing values
    valid = ~np.isnan(phenotype)
    if valid.sum() < n:
        logger.info(
            f'{phenotype_id}: excluding {n - valid.sum()} missing values'
        )
    y = phenotype[valid].copy()
    n_valid = len(y)

    # apply rint if requested
    if use_rint:
        y = rint(y)

    # build covariate matrix (with intercept)
    if covariates is not None:
        x = covariates[valid].copy()
        # add intercept if not present
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((n_valid, 1)), x])
    else:
        x = np.ones((n_valid, 1))

    # compute kinship if not provided
    if kinship_decomp is None:
        logger.info(f'{phenotype_id}: computing loco kinship')
        kinship_dict = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kinship_dict)

    # scan each chromosome
    all_locus_ids = []
    all_chroms = []
    all_positions = []
    all_lod = []
    all_pval = []
    all_betas = []
    h2_null_dict = {}

    for chrom in genome_cache.chromosomes:
        if chrom not in kinship_decomp:
            logger.warning(f'no kinship for chr{chrom}, skipping')
            continue

        eigvals = kinship_decomp[chrom]['values']
        eigvecs = kinship_decomp[chrom]['vectors']

        # subset eigvals/eigvecs for valid samples only
        eigvals_v = eigvals  # already aligned
        eigvecs_v = eigvecs[valid][:, :] if valid.sum() < n else eigvecs

        # estimate null h² for this chromosome
        y_rot = eigvecs_v.T @ y
        x_rot = eigvecs_v.T @ x

        h2_null, ll_null = lmm_module.estimate_h2_null(
            y_rot, x_rot, eigvals_v, method='reml' if use_reml else 'ml'
        )
        h2_null_dict[chrom] = h2_null

        logger.debug(
            f'{phenotype_id} chr{chrom}: h2_null = {h2_null:.4f}'
        )

        # scan all loci on this chromosome
        chr_result = lmm_module.scan_chromosome(
            y, genome_cache, chrom, x,
            eigvals_v, eigvecs_v, h2_null, use_reml
        )

        n_loci = len(chr_result['locus_ids'])
        all_locus_ids.append(chr_result['locus_ids'])
        all_chroms.append(np.full(n_loci, chrom))
        all_positions.append(chr_result['positions'])
        all_lod.append(chr_result['lod'])
        all_pval.append(chr_result['p_value'])
        all_betas.append(chr_result['betas'])

    # concatenate across chromosomes
    return ScanResult(
        locus_ids=np.concatenate(all_locus_ids),
        chromosomes=np.concatenate(all_chroms),
        positions=np.concatenate(all_positions),
        lod=np.concatenate(all_lod),
        p_value=np.concatenate(all_pval),
        betas=np.vstack(all_betas),
        h2_null=h2_null_dict,
        phenotype_id=phenotype_id,
    )


def scan_multi_gene(phenotype_matrix, gene_ids, genome_cache,
                    covariates=None, kinship_decomp=None,
                    use_rint=True, n_cores=1):
    """scan multiple genes (columns of phenotype_matrix).

    parameters
    ----------
    phenotype_matrix : np.ndarray (n_subjects, n_genes)
        phenotype matrix (columns are genes)
    gene_ids : list of str
        gene names
    genome_cache : GenomeCache
        haplotype probability data
    covariates : np.ndarray (n_subjects, p) or None
    kinship_decomp : dict or None
    use_rint : bool
    n_cores : int
        number of parallel processes (1 = serial)

    returns
    -------
    dict of {gene_id: ScanResult}
    """
    # precompute kinship once for all genes
    if kinship_decomp is None:
        logger.info('precomputing loco kinship for multi-gene scan')
        kinship_dict = kin_module.calc_kinship(genome_cache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kinship_dict)

    results = {}

    if n_cores == 1:
        # serial execution
        for g_idx, gene_id in enumerate(gene_ids):
            logger.info(
                f'scanning gene {g_idx + 1}/{len(gene_ids)}: {gene_id}'
            )
            results[gene_id] = scan_h2lmm(
                phenotype_matrix[:, g_idx],
                genome_cache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                use_rint=use_rint,
                phenotype_id=gene_id,
            )
    else:
        # parallel execution via multiprocessing
        from multiprocessing import Pool
        from functools import partial

        def _scan_one_gene(args):
            g_idx, gene_id = args
            return gene_id, scan_h2lmm(
                phenotype_matrix[:, g_idx],
                genome_cache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                use_rint=use_rint,
                phenotype_id=gene_id,
            )

        with Pool(n_cores) as pool:
            for gene_id, scan_result in pool.imap_unordered(
                _scan_one_gene, enumerate(gene_ids)
            ):
                results[gene_id] = scan_result
                logger.info(
                    f'completed: {gene_id} '
                    f'(peak lod = {scan_result.peak_locus()["lod"]:.2f})'
                )

    return results

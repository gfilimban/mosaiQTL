"""Unit tests for qtl/lmm.py — Gaussian LMM validated port."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.lmm import estimate_h2_null, precompute_chromosome, scan_chromosome
from mosaiqtl.qtl.scan import scan_h2lmm


class TestEstimateH2Null:
    def test_returns_float_in_unit_interval(self, tiny_kinship,
                                            tiny_strain_phenotype, tiny_strain_covariates):
        K = tiny_kinship['1']
        eigvals, eigvecs = np.linalg.eigh(K)
        y_rot = eigvecs.T @ tiny_strain_phenotype
        x_rot = eigvecs.T @ tiny_strain_covariates
        h2, _ = estimate_h2_null(y_rot, x_rot, eigvals)
        assert 0.0 <= h2 <= 1.0

    def test_reml_and_ml_both_return(self, tiny_kinship,
                                     tiny_strain_phenotype, tiny_strain_covariates):
        K = tiny_kinship['1']
        eigvals, eigvecs = np.linalg.eigh(K)
        y_rot = eigvecs.T @ tiny_strain_phenotype
        x_rot = eigvecs.T @ tiny_strain_covariates
        h2_reml, ll_reml = estimate_h2_null(y_rot, x_rot, eigvals, method='reml')
        h2_ml,   ll_ml   = estimate_h2_null(y_rot, x_rot, eigvals, method='ml')
        assert np.isfinite(h2_reml) and np.isfinite(h2_ml)
        assert np.isfinite(ll_reml) and np.isfinite(ll_ml)


class TestScanChromosome:
    def test_returns_dict_with_required_keys(self, tiny_cache, tiny_kinship,
                                             tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        for key in ('locus_ids', 'positions', 'lod', 'p_value', 'betas'):
            assert key in result

    def test_lod_shape(self, tiny_cache, tiny_kinship,
                       tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        n_loci = len(tiny_cache._data[chrom]['locus_ids'])
        assert result['lod'].shape == (n_loci,)

    def test_lod_nonnegative(self, tiny_cache, tiny_kinship,
                             tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        assert np.all(result['lod'] >= -1e-8)

    def test_lod_finite(self, tiny_cache, tiny_kinship,
                        tiny_strain_phenotype, tiny_strain_covariates):
        # LOD = (n/2) log10(RSS0/RSS1) — all values must be finite
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        assert np.all(np.isfinite(result['lod']))

    def test_p_values_in_unit_interval(self, tiny_cache, tiny_kinship,
                                       tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        pv = result['p_value']
        assert np.all(pv >= 0.0) and np.all(pv <= 1.0 + 1e-8)


class TestScanChromosomeFull:
    def test_full_betas_shape(self, tiny_cache, tiny_kinship,
                              tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs, model='full')
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        n_loci = len(tiny_cache._data[chrom]['locus_ids'])
        J = tiny_cache.n_founders
        nf_full = J * (J + 1) // 2
        assert result['betas'].shape == (n_loci, nf_full)

    def test_full_lod_nonnegative(self, tiny_cache, tiny_kinship,
                                   tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs, model='full')
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        assert np.all(result['lod'] >= -1e-8)

    def test_full_lod_finite(self, tiny_cache, tiny_kinship,
                              tiny_strain_phenotype, tiny_strain_covariates):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs, model='full')
        result = scan_chromosome(
            tiny_strain_phenotype, cd, tiny_strain_covariates, eigvals, eigvecs, h2,
        )
        assert np.all(np.isfinite(result['lod']))


class TestReplicatePath:
    """scan_h2lmm with replicated samples uses the sample-level Z K Z^T model."""

    def test_replicate_path_uses_sample_level_model(
        self, tiny_cache, tiny_kinship, tiny_phenotype, sample_strain_map,
    ):
        """Replicate path (n=40, Z K Z^T) gives different LODs than the strain-mean
        path (m=10), confirming the n-dimensional model is active.  LODs must be
        finite, non-negative, and rank-consistent with the strain-mean path."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.utils import build_z_matrix

        kd = decomp_kinship(tiny_kinship)

        # replicate path: n=40 sample-level model with K_n = Z K_m Z^T
        res_rep = scan_h2lmm(
            tiny_phenotype, tiny_cache,
            kinship_decomp=kd,
            sample_ids=list(sample_strain_map),
            use_rint=False,
            scan_window='genome',
        )

        # strain-mean path: m=10 model (no replication)
        Z, _ = build_z_matrix(list(sample_strain_map), list(tiny_cache.sample_ids))
        r     = Z.sum(axis=0)
        y_bar = (Z.T @ tiny_phenotype) / r
        res_strain = scan_h2lmm(
            y_bar, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )

        # models are intentionally different — LODs must not be equal
        assert not np.allclose(res_rep.lod, res_strain.lod), \
            'replicate and strain-mean LODs must differ (Z K Z^T vs K_m models)'
        # sanity: replicate-path LODs are finite and non-negative
        assert np.all(np.isfinite(res_rep.lod))
        assert np.all(res_rep.lod >= 0.0)
        # locus ranking must be consistent across the two models
        rank_rep    = np.argsort(np.argsort(res_rep.lod))
        rank_strain = np.argsort(np.argsort(res_strain.lod))
        rho = np.corrcoef(rank_rep, rank_strain)[0, 1]
        assert rho > 0.7, f'rank correlation between models too low: {rho:.3f}'

    def test_replicate_path_lod_finite(
        self, tiny_cache, tiny_kinship, tiny_phenotype, sample_strain_map,
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        res = scan_h2lmm(
            tiny_phenotype, tiny_cache,
            kinship_decomp=kd,
            sample_ids=list(sample_strain_map),
            use_rint=False,
            scan_window='genome',
        )
        assert np.all(np.isfinite(res.lod))
        assert np.all(res.lod >= 0.0)

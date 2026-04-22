"""Unit tests for qtl/lmm.py — Gaussian LMM validated port."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.lmm import estimate_h2_null, scan_chromosome


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
        result = scan_chromosome(
            tiny_strain_phenotype, tiny_cache, chrom, tiny_strain_covariates,
            eigvals, eigvecs, h2,
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
        result = scan_chromosome(
            tiny_strain_phenotype, tiny_cache, chrom, tiny_strain_covariates,
            eigvals, eigvecs, h2,
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
        result = scan_chromosome(
            tiny_strain_phenotype, tiny_cache, chrom, tiny_strain_covariates,
            eigvals, eigvecs, h2,
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
        result = scan_chromosome(
            tiny_strain_phenotype, tiny_cache, chrom, tiny_strain_covariates,
            eigvals, eigvecs, h2,
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
        result = scan_chromosome(
            tiny_strain_phenotype, tiny_cache, chrom, tiny_strain_covariates,
            eigvals, eigvecs, h2,
        )
        pv = result['p_value']
        assert np.all(pv >= 0.0) and np.all(pv <= 1.0 + 1e-8)

"""Unit tests for qtl/epistasis.py — pairwise GxG interaction and conditional scans."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.epistasis import (
    scan_epistasis,
    test_haplotype_combinations as hap_combinations,
    scan_conditional,
    scan_conditional_with_y,
    _wls_lod,
)


# ── _wls_lod ──────────────────────────────────────────────────────────────────

class TestWlsLod:
    def test_returns_dict(self):
        rng = np.random.default_rng(0)
        n = 20
        y_rot = rng.standard_normal(n)
        x_rot = np.column_stack([np.ones(n), rng.standard_normal(n)])
        z_rot = np.column_stack([rng.standard_normal(n)])
        eigvals = np.abs(rng.standard_normal(n))
        result = _wls_lod(y_rot, x_rot, z_rot, eigvals, h2=0.3, n=n)
        assert isinstance(result, dict)

    def test_required_keys(self):
        rng = np.random.default_rng(1)
        n = 15
        y_rot = rng.standard_normal(n)
        x_rot = np.ones((n, 1))
        z_rot = rng.standard_normal((n, 2))
        eigvals = np.ones(n)
        result = _wls_lod(y_rot, x_rot, z_rot, eigvals, h2=0.2, n=n)
        assert 'lod' in result
        assert 'betas' in result

    def test_lod_nonnegative(self):
        rng = np.random.default_rng(2)
        n = 20
        y_rot = rng.standard_normal(n)
        x_rot = np.ones((n, 1))
        z_rot = rng.standard_normal((n, 3))
        eigvals = np.ones(n)
        result = _wls_lod(y_rot, x_rot, z_rot, eigvals, h2=0.0, n=n)
        assert result['lod'] >= -1e-8


# ── scan_epistasis ────────────────────────────────────────────────────────────

class TestScanEpistasis:
    def _get_two_loci(self, tiny_cache):
        lids, _, _ = tiny_cache.get_chromosome('1')
        return lids[0], lids[5]

    def test_returns_dict(self, tiny_cache, tiny_kinship,
                          tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        la, lb = self._get_two_loci(tiny_cache)
        result = scan_epistasis(
            tiny_strain_phenotype, tiny_cache, la, lb,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert isinstance(result, dict)

    def test_required_keys(self, tiny_cache, tiny_kinship,
                            tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        la, lb = self._get_two_loci(tiny_cache)
        result = scan_epistasis(
            tiny_strain_phenotype, tiny_cache, la, lb,
            kinship_decomp=kd, use_rint=False,
        )
        for key in ('lod_additive', 'lod_interaction', 'lod_epistasis',
                    'p_epistasis', 'betas_a', 'betas_b', 'betas_int'):
            assert key in result

    def test_lod_additive_nonnegative(self, tiny_cache, tiny_kinship,
                                       tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        la, lb = self._get_two_loci(tiny_cache)
        result = scan_epistasis(
            tiny_strain_phenotype, tiny_cache, la, lb,
            kinship_decomp=kd, use_rint=False,
        )
        assert result['lod_additive'] >= -1e-8

    def test_lod_epistasis_nonnegative(self, tiny_cache, tiny_kinship,
                                        tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        la, lb = self._get_two_loci(tiny_cache)
        result = scan_epistasis(
            tiny_strain_phenotype, tiny_cache, la, lb,
            kinship_decomp=kd, use_rint=False,
        )
        assert result['lod_epistasis'] >= -1e-8

    def test_p_value_in_unit_interval(self, tiny_cache, tiny_kinship,
                                       tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        la, lb = self._get_two_loci(tiny_cache)
        result = scan_epistasis(
            tiny_strain_phenotype, tiny_cache, la, lb,
            kinship_decomp=kd, use_rint=False,
        )
        assert 0.0 <= result['p_epistasis'] <= 1.0 + 1e-8

    def test_missing_locus_raises(self, tiny_cache, tiny_kinship,
                                   tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        with pytest.raises(KeyError):
            scan_epistasis(
                tiny_strain_phenotype, tiny_cache,
                'nonexistent_locus_a', 'nonexistent_locus_b',
                kinship_decomp=kd, use_rint=False,
            )


# ── test_haplotype_combinations ───────────────────────────────────────────────

class TestHaplotypeCombinations:
    def test_returns_dataframe(self, tiny_cache, tiny_strain_phenotype):
        lids, _, _ = tiny_cache.get_chromosome('1')
        df = hap_combinations(
            tiny_strain_phenotype, tiny_cache,
            lids[0], lids[5],
            sample_ids=tiny_cache.sample_ids,
        )
        assert hasattr(df, 'columns')

    def test_shape_j_by_j(self, tiny_cache, tiny_strain_phenotype):
        lids, _, _ = tiny_cache.get_chromosome('1')
        df = hap_combinations(
            tiny_strain_phenotype, tiny_cache,
            lids[0], lids[3],
            sample_ids=tiny_cache.sample_ids,
        )
        J = tiny_cache.n_founders
        assert df.shape == (J, J)


# ── scan_conditional ──────────────────────────────────────────────────────────

class TestScanConditional:
    def test_returns_list_of_scan_results(self, tiny_cache, tiny_kinship,
                                          tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm, ScanResult
        kd = decomp_kinship(tiny_kinship)
        base_scan = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
            scan_window='genome',
        )
        signals = scan_conditional(
            base_scan, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert isinstance(signals, list)
        assert isinstance(signals[0], ScanResult)

    def test_primary_signal_lod_nonneg(self, tiny_cache, tiny_kinship,
                                        tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        kd = decomp_kinship(tiny_kinship)
        base_scan = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
            scan_window='genome',
        )
        signals = scan_conditional(
            base_scan, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert signals[0].lod.shape[0] > 0
        assert (signals[0].lod >= -1e-8).all()


# ── scan_conditional_with_y ───────────────────────────────────────────────────

class TestScanConditionalWithY:
    def test_returns_list_of_scan_results(self, tiny_cache, tiny_kinship,
                                          tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm, ScanResult
        kd = decomp_kinship(tiny_kinship)
        base_scan = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
            scan_window='genome',
        )
        signals = scan_conditional_with_y(
            tiny_strain_phenotype, base_scan, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert isinstance(signals, list)
        assert isinstance(signals[0], ScanResult)

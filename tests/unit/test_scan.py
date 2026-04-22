"""Unit tests for qtl/scan.py — ScanResult and scan_h2lmm."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.scan import (
    ScanResult, scan_h2lmm, scan_multi_gene,
    grab_locus_from_scan, lod_drop_ci,
    fixef_eqtl_scan, instability_lm_scan,
)


# ── ScanResult ────────────────────────────────────────────────────────────────

def _make_scan_result(n_loci=20, n_founders=8):
    """Helper: build a minimal ScanResult with known values."""
    rng = np.random.default_rng(0)
    lods = np.abs(rng.standard_normal(n_loci))
    return ScanResult(
        locus_ids   = np.array([f'L{i}' for i in range(n_loci)]),
        chromosomes = np.array(['1'] * (n_loci // 2) + ['2'] * (n_loci - n_loci // 2)),
        positions   = np.linspace(1.0, 100.0, n_loci),
        lod         = lods,
        p_value     = np.clip(rng.uniform(0, 0.5, n_loci), 0, 1),
        betas       = rng.standard_normal((n_loci, n_founders - 1)),
        h2_null     = {'1': 0.3, '2': 0.4},
        phenotype_id = 'test_gene',
    )


class TestScanResult:
    def test_peak_locus_keys(self):
        sr = _make_scan_result()
        peak = sr.peak_locus()
        for key in ('locus_id', 'chromosome', 'position', 'lod', 'p_value', 'betas'):
            assert key in peak

    def test_peak_locus_is_max(self):
        sr = _make_scan_result()
        peak = sr.peak_locus()
        assert peak['lod'] == pytest.approx(sr.lod.max())

    def test_to_dataframe_shape(self):
        n = 10
        sr = _make_scan_result(n_loci=n)
        df = sr.to_dataframe()
        assert len(df) == n

    def test_to_dataframe_columns(self):
        sr = _make_scan_result()
        df = sr.to_dataframe()
        for col in ('locus_id', 'chromosome', 'position', 'lod', 'p_value'):
            assert col in df.columns

    def test_to_dataframe_beta_columns(self):
        sr = _make_scan_result(n_founders=8)
        df = sr.to_dataframe()
        for fi in range(7):
            assert f'beta_f{fi}' in df.columns

    def test_attributes_accessible(self):
        sr = _make_scan_result()
        assert hasattr(sr, 'lod')
        assert hasattr(sr, 'chromosomes')
        assert hasattr(sr, 'h2_null')
        assert hasattr(sr, 'phenotype_id')


# ── scan_h2lmm ────────────────────────────────────────────────────────────────

class TestScanH2Lmm:
    def test_returns_scan_result(self, tiny_cache, tiny_kinship,
                                  tiny_strain_phenotype, tiny_strain_covariates):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            use_rint=False,
            phenotype_id='test',
        )
        assert isinstance(result, ScanResult)

    def test_lod_shape(self, tiny_cache, tiny_kinship,
                       tiny_strain_phenotype, tiny_strain_covariates):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert result.lod.shape == (total_loci,)

    def test_lod_finite(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert np.all(np.isfinite(result.lod))

    def test_lod_nonnegative(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert np.all(result.lod >= -1e-8)

    def test_p_values_in_unit_interval(self, tiny_cache, tiny_kinship,
                                        tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert np.all(result.p_value >= 0.0)
        assert np.all(result.p_value <= 1.0 + 1e-8)

    def test_h2_null_per_chrom(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        for c in tiny_cache.chromosomes:
            assert c in result.h2_null
            assert 0.0 <= result.h2_null[c] <= 1.0

    def test_phenotype_id_stored(self, tiny_cache, tiny_kinship,
                                  tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            phenotype_id='my_gene',
        )
        assert result.phenotype_id == 'my_gene'

    def test_covariates_affect_lod(self, tiny_cache, tiny_kinship,
                                    tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(99)
        # use a non-trivial covariate (intercept + random noise) so it differs
        # from the default intercept-only model
        x_extra = np.column_stack([
            np.ones(len(tiny_strain_phenotype)),
            rng.standard_normal(len(tiny_strain_phenotype)),
        ])
        r_no_cov   = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                  kinship_decomp=kd, use_rint=False)
        r_with_cov = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                  covariates=x_extra,
                                  kinship_decomp=kd, use_rint=False)
        # LOD profiles differ when informative covariates are included
        assert not np.allclose(r_no_cov.lod, r_with_cov.lod)

    def test_betas_shape(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert result.betas.shape == (total_loci, tiny_cache.n_founders)

    def test_no_kinship_auto_computes(self, tiny_cache, tiny_strain_phenotype):
        # kinship_decomp=None → computed internally from genomecache
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=None,
            use_rint=False,
        )
        assert isinstance(result, ScanResult)

    def test_rint_changes_lod(self, tiny_cache, tiny_kinship,
                               tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        r_no_rint   = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                   kinship_decomp=kd, use_rint=False)
        r_with_rint = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                   kinship_decomp=kd, use_rint=True)
        assert not np.allclose(r_no_rint.lod, r_with_rint.lod)


# ── grab_locus_from_scan ──────────────────────────────────────────────────────

class TestGrabLocus:
    def test_returns_locus_with_max_lod(self):
        sr = _make_scan_result()
        lid = grab_locus_from_scan(sr)
        assert lid == sr.locus_ids[np.argmax(sr.lod)]


# ── lod_drop_ci ───────────────────────────────────────────────────────────────

class TestLodDropCi:
    def test_returns_three_floats(self, tiny_cache, tiny_kinship,
                                   tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                         kinship_decomp=kd, use_rint=False)
        lo, hi, pk = lod_drop_ci(sr, chrom='1', drop=1.5)
        assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(pk)

    def test_lo_le_peak_le_hi(self, tiny_cache, tiny_kinship,
                                tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                         kinship_decomp=kd, use_rint=False)
        lo, hi, pk = lod_drop_ci(sr, chrom='1', drop=1.5)
        assert lo <= pk <= hi

    def test_empty_chrom_returns_nan(self):
        sr = _make_scan_result()
        lo, hi, pk = lod_drop_ci(sr, chrom='99')
        assert np.isnan(lo) and np.isnan(hi) and np.isnan(pk)


# ── fixef_eqtl_scan ───────────────────────────────────────────────────────────

class TestFixefEqtlScan:
    def test_returns_scan_result(self, tiny_cache, tiny_strain_phenotype):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache, use_rint=False)
        assert isinstance(sr, ScanResult)

    def test_lod_nonneg(self, tiny_cache, tiny_strain_phenotype):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache, use_rint=False)
        assert (sr.lod >= -1e-8).all()

    def test_just_these_loci(self, tiny_cache, tiny_strain_phenotype):
        lids, _, _ = tiny_cache.get_chromosome('1')
        subset = {lids[0], lids[1]}
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache,
                              use_rint=False, just_these_loci=subset)
        assert set(sr.locus_ids) == subset

    def test_with_covariates(self, tiny_cache, tiny_strain_phenotype,
                              tiny_strain_covariates):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache,
                              covariates=tiny_strain_covariates,
                              use_rint=False)
        assert isinstance(sr, ScanResult)

    def test_phenotype_id_stored(self, tiny_cache, tiny_strain_phenotype):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache,
                              phenotype_id='gene_x', use_rint=False)
        assert sr.phenotype_id == 'gene_x'


# ── scan_multi_gene ───────────────────────────────────────────────────────────

class TestScanMultiGene:
    def test_returns_dict(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        n = len(tiny_strain_phenotype)
        pheno_mat = np.column_stack([
            tiny_strain_phenotype,
            np.random.default_rng(1).standard_normal(n),
        ])
        results = scan_multi_gene(
            pheno_mat, ['gene_a', 'gene_b'], tiny_cache,
            kinship_decomp=kd,
        )
        assert isinstance(results, dict)
        assert set(results.keys()) == {'gene_a', 'gene_b'}

    def test_each_value_is_scan_result(self, tiny_cache, tiny_kinship,
                                        tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        pheno_mat = tiny_strain_phenotype[:, None]
        results = scan_multi_gene(
            pheno_mat, ['gene_a'], tiny_cache, kinship_decomp=kd,
        )
        assert isinstance(results['gene_a'], ScanResult)


# ── instability_lm_scan ───────────────────────────────────────────────────────

class TestInstabilityLmScan:
    def test_returns_scan_result(self, tiny_cache, tiny_strain_phenotype):
        sr = instability_lm_scan(
            tiny_strain_phenotype, tiny_cache, use_rint=False,
            n_samples=5, seed=0,
        )
        assert isinstance(sr, ScanResult)

    def test_lod_shape(self, tiny_cache, tiny_strain_phenotype):
        sr = instability_lm_scan(
            tiny_strain_phenotype, tiny_cache, use_rint=False,
            n_samples=3, seed=0,
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert sr.lod.shape[0] == total_loci

    def test_p_value_in_unit_interval(self, tiny_cache, tiny_strain_phenotype):
        sr = instability_lm_scan(
            tiny_strain_phenotype, tiny_cache, use_rint=False,
            n_samples=3, seed=1,
        )
        assert ((sr.p_value >= 0) & (sr.p_value <= 1 + 1e-8)).all()

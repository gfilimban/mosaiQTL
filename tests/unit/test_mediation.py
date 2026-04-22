"""Unit tests for qtl/mediation.py — QR-based mediation scan."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.mediation import (
    _wls_null_rss,
    mediation_scan_qr,
    fixef_mediation_scan,
)


# ── _wls_null_rss ─────────────────────────────────────────────────────────────

class TestWlsNullRss:
    def test_returns_float_and_beta(self):
        rng = np.random.default_rng(0)
        n = 20
        y = rng.standard_normal(n)
        x = np.column_stack([np.ones(n), rng.standard_normal(n)])
        w_inv = np.ones(n)
        rss, beta = _wls_null_rss(y, x, w_inv)
        assert isinstance(rss, float)
        assert beta.shape == (2,)

    def test_rss_nonnegative(self):
        rng = np.random.default_rng(1)
        n = 15
        y = rng.standard_normal(n)
        x = np.ones((n, 1))
        rss, _ = _wls_null_rss(y, x, np.ones(n))
        assert rss >= 0.0

    def test_perfect_fit_zero_rss(self):
        n = 10
        x = np.column_stack([np.ones(n), np.arange(n, dtype=float)])
        y = 2.0 * x[:, 1] + 1.0  # exact linear
        rss, _ = _wls_null_rss(y, x, np.ones(n))
        assert rss < 1e-10


# ── fixef_mediation_scan ──────────────────────────────────────────────────────

class TestFixefMediationScan:
    def test_returns_dataframe(self, tiny_cache, tiny_strain_phenotype,
                               tiny_strain_covariates):
        rng = np.random.default_rng(5)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            covariates=tiny_strain_covariates,
            use_rint=False,
        )
        assert hasattr(df, 'columns')

    def test_required_columns(self, tiny_cache, tiny_strain_phenotype,
                               tiny_strain_covariates):
        rng = np.random.default_rng(6)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False,
        )
        for col in ('locus', 'chr', 'pos_mb', 'lod_base',
                    'lod_mediated', 'lod_drop', 'p_value'):
            assert col in df.columns

    def test_lod_drop_nonnegative(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(7)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False,
        )
        assert (df['lod_drop'] >= 0).all()

    def test_row_count(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(8)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False,
        )
        # one row per locus across all chromosomes
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert len(df) == total_loci

    def test_just_these_loci_filter(self, tiny_cache, tiny_strain_phenotype):
        lids_ch1 = tiny_cache.get_chromosome('1')[0]
        target = set([lids_ch1[0], lids_ch1[1]])
        rng = np.random.default_rng(9)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False, just_these_loci=target,
        )
        assert set(df['locus']) == target

    def test_outcome_mediator_ids(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(10)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False,
            outcome_id='gene_A', mediator_id='gene_B',
        )
        assert (df['outcome_id'] == 'gene_A').all()
        assert (df['mediator_id'] == 'gene_B').all()

    def test_lod_base_finite(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(11)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False,
        )
        assert np.all(np.isfinite(df['lod_base']))

    def test_p_values_in_unit_interval(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(12)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype, mediator, tiny_cache,
            use_rint=False,
        )
        assert (df['p_value'] >= 0).all() and (df['p_value'] <= 1.0 + 1e-8).all()


# ── mediation_scan_qr ─────────────────────────────────────────────────────────

class TestMediationScanQr:
    def test_returns_dataframe(self, tiny_cache, tiny_kinship,
                               tiny_strain_phenotype, tiny_strain_covariates):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(20)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype, mediator, tiny_cache,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert hasattr(df, 'columns')

    def test_required_columns(self, tiny_cache, tiny_kinship,
                              tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(21)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype, mediator, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        for col in ('locus', 'chr', 'pos_mb', 'lod_base',
                    'lod_mediated', 'lod_drop'):
            assert col in df.columns

    def test_lod_drop_nonnegative(self, tiny_cache, tiny_kinship,
                                   tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(22)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype, mediator, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert (df['lod_drop'] >= 0).all()

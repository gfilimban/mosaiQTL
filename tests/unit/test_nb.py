"""Unit tests for qtl/nb.py — NB-GLMM Stage 1 and Stage 2."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.nb import (
    pseudobulk_sum,
    _tmm_offsets,
    _nb_loglik,
    _irls_nb_pseudobulk,
    fit_null_sc_native,
)


# ── pseudobulk_sum ────────────────────────────────────────────────────────────

class TestPseudobulkSum:
    def test_shape(self, counts_cells, cell_sample_ids):
        pb, ncells, strains = pseudobulk_sum(counts_cells, cell_sample_ids)
        n_strains = len(np.unique(cell_sample_ids))
        assert pb.shape == (counts_cells.shape[0], n_strains)

    def test_ncells_sums_to_total(self, counts_cells, cell_sample_ids):
        _, ncells, _ = pseudobulk_sum(counts_cells, cell_sample_ids)
        assert int(ncells.sum()) == counts_cells.shape[1]

    def test_pb_sums_match_raw(self, counts_cells, cell_sample_ids):
        pb, _, strains = pseudobulk_sum(counts_cells, cell_sample_ids)
        # for each strain, sum of raw counts should equal pseudobulk
        for i, sid in enumerate(strains):
            mask = cell_sample_ids == sid
            expected = counts_cells[:, mask].sum(axis=1)
            np.testing.assert_allclose(pb[:, i], expected, atol=1e-6)

    def test_gene_mask(self, counts_cells, cell_sample_ids):
        n_genes = counts_cells.shape[0]
        mask = np.array([True, False, True, False][:n_genes])
        pb, _, _ = pseudobulk_sum(counts_cells, cell_sample_ids, gene_mask=mask)
        assert pb.shape[0] == mask.sum()

    def test_nonnegative(self, counts_cells, cell_sample_ids):
        pb, _, _ = pseudobulk_sum(counts_cells, cell_sample_ids)
        assert np.all(pb >= 0)

    def test_strains_sorted(self, counts_cells, cell_sample_ids):
        _, _, strains = pseudobulk_sum(counts_cells, cell_sample_ids)
        assert list(strains) == sorted(strains)


# ── _tmm_offsets ──────────────────────────────────────────────────────────────

class TestTmmOffsets:
    def test_shape(self):
        rng = np.random.default_rng(0)
        counts = rng.integers(1, 100, size=(50, 20))
        offsets = _tmm_offsets(counts)
        assert offsets.shape == (20,)

    def test_log_scale(self):
        counts = np.ones((10, 5), dtype=float) * 100.0
        offsets = _tmm_offsets(counts)
        np.testing.assert_allclose(offsets, np.log(1000.0), atol=1e-8)

    def test_handles_zero_library(self):
        counts = np.zeros((5, 3), dtype=float)
        offsets = _tmm_offsets(counts)
        np.testing.assert_allclose(offsets, 0.0)  # log(1) = 0


# ── _nb_loglik ────────────────────────────────────────────────────────────────

class TestNbLoglik:
    def test_returns_float(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 20, 10).astype(float)
        mu = np.full(10, 10.0)
        ll = _nb_loglik(y, mu, 0.5)
        assert isinstance(ll, float)

    def test_finite(self):
        y = np.array([5.0, 10.0, 8.0, 12.0])
        mu = np.array([8.0, 10.0, 9.0, 11.0])
        ll = _nb_loglik(y, mu, 0.3)
        assert np.isfinite(ll)

    def test_higher_loglik_for_better_fit(self):
        y = np.array([10.0, 10.0, 10.0, 10.0])
        mu_good = np.array([10.0, 10.0, 10.0, 10.0])
        mu_bad  = np.array([1.0,  1.0,  1.0,  1.0])
        ll_good = _nb_loglik(y, mu_good, 0.5)
        ll_bad  = _nb_loglik(y, mu_bad,  0.5)
        assert ll_good > ll_bad


# ── _irls_nb_pseudobulk ───────────────────────────────────────────────────────

class TestIrlsNbPseudobulk:
    def test_returns_mu_beta(self):
        rng = np.random.default_rng(0)
        n = 15
        S = rng.integers(5, 50, n).astype(float)
        x = np.column_stack([np.ones(n), rng.standard_normal(n)])
        offsets = np.zeros(n)
        mu, beta = _irls_nb_pseudobulk(S, x, phi_size=2.0, offsets=offsets)
        assert mu.shape == (n,)
        assert beta.shape == (2,)

    def test_mu_positive(self):
        rng = np.random.default_rng(1)
        n = 10
        S = rng.integers(1, 30, n).astype(float)
        x = np.ones((n, 1))
        offsets = np.zeros(n)
        mu, _ = _irls_nb_pseudobulk(S, x, phi_size=1.0, offsets=offsets)
        assert np.all(mu > 0)

    def test_intercept_only_mean_approx(self):
        n = 20
        S = np.full(n, 10.0)
        x = np.ones((n, 1))
        offsets = np.zeros(n)
        mu, beta = _irls_nb_pseudobulk(S, x, phi_size=100.0, offsets=offsets)
        np.testing.assert_allclose(mu, 10.0, rtol=0.05)


# ── fit_null_sc_native (Stage 1) ──────────────────────────────────────────────

class TestFitNullScNative:
    def test_returns_dict(self, counts_cells, cell_sample_ids):
        result = fit_null_sc_native(counts_cells, cell_sample_ids)
        assert isinstance(result, dict)

    def test_required_keys(self, counts_cells, cell_sample_ids):
        result = fit_null_sc_native(counts_cells, cell_sample_ids)
        for key in ('dispersion', 'dispersion_shrunk', 'dispersion_use',
                    'sigma_sample', 'gene_mask'):
            assert key in result

    def test_phi_positive(self, counts_cells, cell_sample_ids):
        result = fit_null_sc_native(counts_cells, cell_sample_ids)
        phi = result['dispersion']
        assert np.all(phi[~np.isnan(phi)] > 0)

    def test_gene_mask_shape(self, counts_cells, cell_sample_ids):
        result = fit_null_sc_native(counts_cells, cell_sample_ids)
        assert result['gene_mask'].shape == (counts_cells.shape[0],)

    def test_gene_mask_bool(self, counts_cells, cell_sample_ids):
        result = fit_null_sc_native(counts_cells, cell_sample_ids)
        assert result['gene_mask'].dtype == bool

    def test_with_covariates(self, counts_cells, cell_sample_ids, sample_ids,
                              sample_sex):
        # build per-cell covariate (sex of the sample each cell belongs to)
        sample_to_sex = dict(zip(sample_ids, sample_sex))
        cov_cells = np.array([sample_to_sex[s] for s in cell_sample_ids])
        cov_cells = cov_cells.reshape(-1, 1)
        result = fit_null_sc_native(
            counts_cells, cell_sample_ids, covariates_cells=cov_cells)
        assert 'dispersion' in result

    def test_gene_mask_is_all_true_with_enough_cells(self, counts_cells,
                                                       cell_sample_ids):
        result = fit_null_sc_native(counts_cells, cell_sample_ids)
        # all 4 genes have many cells; mask should be all True
        assert result['gene_mask'].sum() == counts_cells.shape[0]

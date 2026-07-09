# tests/test_permutation.py — tests for generate_null_outcomes, get_gev_thresholds
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.qtl.permutation import generate_null_outcomes, get_gev_thresholds


class TestGenerateNullOutcomes:
    def setup_method(self):
        self.rng = np.random.default_rng(0)
        self.y   = self.rng.standard_normal(50)

    def test_permutation_shape(self):
        M = generate_null_outcomes(self.y, n_perms=20, method='permutation', seed=1)
        assert M.shape == (50, 20)

    def test_permutation_preserves_values(self):
        '''each permuted column is a rearrangement of y.'''
        M = generate_null_outcomes(self.y, n_perms=5, method='permutation', seed=1)
        for col in range(5):
            np.testing.assert_array_equal(np.sort(M[:, col]), np.sort(self.y))

    def test_permutation_reproducible(self):
        M1 = generate_null_outcomes(self.y, n_perms=10, seed=7)
        M2 = generate_null_outcomes(self.y, n_perms=10, seed=7)
        np.testing.assert_array_equal(M1, M2)

    def test_bootstrap_shape(self):
        M = generate_null_outcomes(self.y, n_perms=20, method='bootstrap', seed=1)
        assert M.shape == (50, 20)

    def test_bootstrap_centred(self):
        '''bootstrap resamples from centred y → column means near 0.'''
        M = generate_null_outcomes(self.y - self.y.mean(), n_perms=200,
                                    method='bootstrap', seed=1)
        assert abs(M.mean()) < 0.2

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            generate_null_outcomes(self.y, method='jackknife')


class TestGetGevThresholds:
    def setup_method(self):
        rng = np.random.default_rng(42)
        # simulate max-LOD distribution from permutations
        self.max_lods = rng.gamma(shape=4.0, scale=1.5, size=500)

    def test_returns_float_and_dict(self):
        thr, params = get_gev_thresholds(self.max_lods)
        assert isinstance(thr, float)
        assert isinstance(params, dict)

    def test_threshold_above_median(self):
        '''95th pct threshold must exceed median of max-LOD dist.'''
        thr, _ = get_gev_thresholds(self.max_lods, percentile=0.95)
        assert thr > float(np.median(self.max_lods))

    def test_params_keys(self):
        _, p = get_gev_thresholds(self.max_lods)
        for k in ('c', 'loc', 'scale', 'threshold', 'n_perms', 'empirical_quantile'):
            assert k in p

    def test_higher_percentile_higher_threshold(self):
        t90, _ = get_gev_thresholds(self.max_lods, percentile=0.90)
        t99, _ = get_gev_thresholds(self.max_lods, percentile=0.99)
        assert t99 > t90

    def test_too_few_values_returns_none(self):
        thr, params = get_gev_thresholds(np.array([1.0, 2.0, 3.0]))
        assert thr is None and params is None

    def test_ignores_nan(self):
        lods_with_nan = np.concatenate([self.max_lods, [np.nan, np.inf]])
        thr, _ = get_gev_thresholds(lods_with_nan)
        assert np.isfinite(thr)

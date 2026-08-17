"""Unit tests for qtl/permutation.py — permutation thresholds and bootstrap CI."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.permutation import (
    generate_null_outcomes,
    run_threshold_scans,
    get_gev_thresholds,
    run_positional_scans,
    KinshipSingularError,
    _draw_kinship_re,
    _null_fit_params,
)

# ── generate_null_outcomes ────────────────────────────────────────────────────


class TestGenerateNullOutcomes:
    def test_permutation_shape(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(20)
        null = generate_null_outcomes(y, n_perms=50, method="permutation")
        assert null.shape == (20, 50)

    def test_bootstrap_shape(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(15)
        null = generate_null_outcomes(y, n_perms=30, method="bootstrap")
        assert null.shape == (15, 30)

    def test_permutation_preserves_values(self):
        y = np.arange(10, dtype=float)
        null = generate_null_outcomes(y, n_perms=20, method="permutation")
        for col in range(null.shape[1]):
            assert set(null[:, col]) == set(y)

    def test_bootstrap_centred(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(50) + 10.0  # large non-zero mean
        null = generate_null_outcomes(y, n_perms=100, method="bootstrap")
        # each column is drawn from centred distribution
        assert null.mean() == pytest.approx(0.0, abs=2.0)

    def test_invalid_method_raises(self):
        y = np.ones(10)
        with pytest.raises(ValueError, match="method"):
            generate_null_outcomes(y, n_perms=5, method="invalid")

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(20)
        a = generate_null_outcomes(y, n_perms=10, seed=42)
        b = generate_null_outcomes(y, n_perms=10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(20)
        a = generate_null_outcomes(y, n_perms=10, seed=1)
        b = generate_null_outcomes(y, n_perms=10, seed=2)
        assert not np.array_equal(a, b)


# ── get_gev_thresholds ────────────────────────────────────────────────────────


class TestGetGevThresholds:
    def test_returns_float_and_dict(self):
        rng = np.random.default_rng(10)
        max_lods = rng.exponential(2.0, 200)
        thresh, params = get_gev_thresholds(max_lods)
        assert isinstance(thresh, float)
        assert isinstance(params, dict)

    def test_threshold_above_median(self):
        rng = np.random.default_rng(11)
        max_lods = rng.exponential(2.0, 200)
        thresh, _ = get_gev_thresholds(max_lods, percentile=0.95)
        assert thresh > float(np.median(max_lods))

    def test_params_keys(self):
        rng = np.random.default_rng(12)
        max_lods = rng.exponential(2.0, 100)
        _, params = get_gev_thresholds(max_lods)
        for key in (
            "c",
            "loc",
            "scale",
            "threshold",
            "percentile",
            "n_perms",
            "empirical_quantile",
        ):
            assert key in params

    def test_too_few_values_returns_none(self):
        thr, params = get_gev_thresholds(np.array([1.0, 2.0, 3.0]))
        assert thr is None and params is None

    def test_higher_percentile_higher_threshold(self):
        rng = np.random.default_rng(13)
        max_lods = rng.exponential(2.0, 300)
        t90, _ = get_gev_thresholds(max_lods, percentile=0.90)
        t95, _ = get_gev_thresholds(max_lods, percentile=0.95)
        assert t95 >= t90

    def test_ignores_inf(self):
        rng = np.random.default_rng(14)
        max_lods = rng.exponential(2.0, 100)
        max_lods_inf = np.append(max_lods, [np.inf, np.nan])
        thresh, params = get_gev_thresholds(max_lods_inf)
        assert np.isfinite(thresh)
        assert params["n_perms"] == 100


# ── run_threshold_scans ───────────────────────────────────────────────────────


class TestRunThresholdScans:
    def test_returns_array(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        n_perms = 5
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=n_perms, seed=42)
        max_lods, chr_nulls = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            use_rint=False,
        )
        assert max_lods.shape == (n_perms,)
        assert isinstance(chr_nulls, dict)

    def test_max_lods_nonnegative(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=3, seed=1)
        max_lods, _ = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            use_rint=False,
        )
        assert np.all(max_lods >= -1e-8)

    def test_single_chrom_mode(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=3, seed=2)
        max_lods, chr_nulls = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            chrom="1",
            use_rint=False,
        )
        assert max_lods.shape == (3,)
        assert list(chr_nulls.keys()) == ["1"]

    def test_no_phenotype_raises(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=2, seed=4)
        with pytest.raises(ValueError, match="phenotype required"):
            run_threshold_scans(
                null,
                tiny_cache,
                kinship_decomp=kd,
                kinship_loco=tiny_kinship,
                phenotype=None,
            )


# ── run_positional_scans ──────────────────────────────────────────────────────


class TestRunPositionalScans:
    def test_returns_array_and_dict(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        boot = generate_null_outcomes(
            tiny_strain_phenotype, n_perms=10, method="bootstrap", seed=99
        )
        peak_pos, ci = run_positional_scans(
            boot,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert peak_pos.shape == (10,)
        assert "ci_lo" in ci and "ci_hi" in ci

    def test_ci_ordered(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        boot = generate_null_outcomes(
            tiny_strain_phenotype, n_perms=20, method="bootstrap", seed=77
        )
        _, ci = run_positional_scans(
            boot,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            use_rint=False,
        )
        assert ci["ci_lo"] <= ci["ci_hi"]


# ── parametric permutation helpers ───────────────────────────────────────────


class TestParametricHelpers:
    """unit tests for _null_fit_params, _draw_kinship_re, _parametric_null_col."""

    def _make_psd_kinship(self, n, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        K = A @ A.T / n + np.eye(n) * 0.1
        return K

    def test_null_fit_params_shape(self):
        rng = np.random.default_rng(0)
        n, p = 20, 2
        y_rot = rng.standard_normal(n)
        x_rot = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eigvals = np.abs(rng.standard_normal(n)) + 0.1
        beta, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, eigvals, h2=0.3)
        assert beta.shape == (p,)
        assert sigma2_g > 0
        assert sigma2_e > 0

    def test_null_fit_params_variance_partition(self):
        rng = np.random.default_rng(1)
        n = 30
        y_rot = rng.standard_normal(n)
        x_rot = np.ones((n, 1))
        eigvals = np.ones(n)
        h2 = 0.4
        _, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, eigvals, h2=h2)
        # sigma2_g / (sigma2_g + sigma2_e) should equal h2
        assert abs(sigma2_g / (sigma2_g + sigma2_e) - h2) < 1e-10

    def test_draw_kinship_re_shape(self):
        rng = np.random.default_rng(42)
        n = 15
        K = self._make_psd_kinship(n)
        u = _draw_kinship_re(K, sigma2_g=0.5, rng=rng)
        assert u.shape == (n,)

    def test_draw_kinship_re_singular_raises(self):
        rng = np.random.default_rng(0)
        # genuinely non-PSD: shift eigenvalues negative by subtracting 0.5I
        # min_eig ≈ -0.4 >> psd_tol (≈ max_eig * n² * eps << 1)
        n = 10
        A = rng.standard_normal((n, n))
        K_bad = A @ A.T / n - 0.5 * np.eye(n)
        with pytest.raises(KinshipSingularError):
            _draw_kinship_re(K_bad, sigma2_g=1.0, rng=rng)

    def test_run_threshold_scans_parametric_shape(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """parametric mode returns (n_perms,) max-LOD array."""
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        n_perms = 5
        # null_outcomes used only for n_perms in parametric mode
        dummy = np.empty((len(tiny_strain_phenotype), n_perms))
        c0 = list(tiny_cache.chromosomes)[0]
        max_lods, chr_nulls = run_threshold_scans(
            dummy,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            chrom=c0,
            permutation_method="parametric",
            use_rint=False,
            seed=99,
        )
        assert max_lods.shape == (n_perms,)
        assert np.all(max_lods >= 0)
        assert c0 in chr_nulls and chr_nulls[c0].shape == (n_perms,)

    def test_run_threshold_scans_parametric_without_kinship_loco(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        n_perms = 3
        dummy = np.empty((len(tiny_strain_phenotype), n_perms))
        max_lods, _ = run_threshold_scans(
            dummy,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            permutation_method="parametric",
        )
        assert max_lods.shape == (n_perms,)
        assert np.all(max_lods >= 0)

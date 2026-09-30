"""Unit tests for qtl/imputation.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.imputation import (
    wakefield_abf,
    abf_pip,
    credible_set,
    lod_beta_to_z,
    fine_map_region,
)

_LOG10_TO_LN = np.log(10.0)


# ── wakefield_abf ─────────────────────────────────────────────────────────────


class TestWakefieldAbf:
    def test_z_zero_returns_value_less_than_one(self):
        # z=0 → no signal; ABF < 1 under any positive prior
        abf = wakefield_abf(np.array([0.0]), se=1.0)
        assert float(abf[0]) < 1.0

    def test_abf_increases_with_z(self):
        z = np.array([0.0, 1.0, 2.0, 3.0, 5.0])
        abf = wakefield_abf(z, se=1.0)
        assert np.all(np.diff(abf) > 0)

    def test_abf_symmetric_in_z(self):
        z_pos = np.array([1.5, 2.5, 3.5])
        z_neg = -z_pos
        np.testing.assert_allclose(
            wakefield_abf(z_pos, se=1.0),
            wakefield_abf(z_neg, se=1.0),
            atol=1e-14,
        )

    def test_scalar_se_broadcasts(self):
        z = np.array([1.0, 2.0, 3.0])
        abf_scalar = wakefield_abf(z, se=0.5)
        se_array = np.full(3, 0.5)
        abf_array = wakefield_abf(z, se=se_array)
        np.testing.assert_allclose(abf_scalar, abf_array, atol=1e-14)

    def test_output_shape(self):
        z = np.arange(10, dtype=float)
        assert wakefield_abf(z, se=1.0).shape == (10,)

    def test_abf_positive(self):
        z = np.linspace(-5, 5, 50)
        assert np.all(wakefield_abf(z, se=1.0) > 0)

    def test_larger_prior_var_gives_higher_abf_for_nonzero_z(self):
        # wider prior → more mass under the alternative → higher ABF when there is signal
        z = np.array([3.0])
        abf_small = wakefield_abf(z, se=1.0, prior_var=0.01)
        abf_large = wakefield_abf(z, se=1.0, prior_var=0.10)
        assert float(abf_large[0]) > float(abf_small[0])


# ── abf_pip ───────────────────────────────────────────────────────────────────


class TestAbfPip:
    def test_sums_to_one(self):
        abf = np.array([0.5, 1.0, 2.5, 0.1])
        pip = abf_pip(abf)
        assert float(pip.sum()) == pytest.approx(1.0, abs=1e-12)

    def test_argmax_matches_abf_argmax(self):
        abf = np.array([0.3, 5.0, 1.2, 0.8])
        pip = abf_pip(abf)
        assert int(np.argmax(pip)) == int(np.argmax(abf))

    def test_single_snp_pip_is_one(self):
        pip = abf_pip(np.array([3.7]))
        assert float(pip[0]) == pytest.approx(1.0, abs=1e-12)

    def test_uniform_abf_gives_uniform_pip(self):
        abf = np.ones(8)
        pip = abf_pip(abf)
        np.testing.assert_allclose(pip, 1.0 / 8, atol=1e-12)

    def test_output_in_unit_interval(self):
        rng = np.random.default_rng(0)
        abf = rng.exponential(1.0, 50)
        pip = abf_pip(abf)
        assert np.all(pip >= 0) and np.all(pip <= 1)

    def test_prior_prob_does_not_change_result(self):
        # uniform scalar prior cancels in normalization
        abf = np.array([1.0, 2.0, 3.0])
        pip_default = abf_pip(abf)
        pip_explicit = abf_pip(abf, prior_prob=0.5)
        np.testing.assert_allclose(pip_default, pip_explicit, atol=1e-14)

    def test_near_zero_abf_handled(self):
        abf = np.array([1e-310, 1.0, 0.5])
        pip = abf_pip(abf)
        assert np.all(np.isfinite(pip))
        assert float(pip.sum()) == pytest.approx(1.0, abs=1e-10)


# ── credible_set ──────────────────────────────────────────────────────────────


class TestCredibleSet:
    def test_contains_top_pip_snp(self):
        pip = np.array([0.1, 0.5, 0.05, 0.3, 0.05])
        cs = credible_set(pip)
        assert int(np.argmax(pip)) in cs

    def test_achieves_coverage(self):
        pip = np.array([0.4, 0.3, 0.15, 0.1, 0.05])
        cs = credible_set(pip, coverage=0.95)
        assert pip[cs].sum() >= 0.95

    def test_smallest_possible_set(self):
        # top SNP alone has pip=0.99 → CS of size 1
        pip = np.array([0.99, 0.005, 0.005])
        cs = credible_set(pip, coverage=0.95)
        assert len(cs) == 1

    def test_full_set_when_signal_spread(self):
        pip = np.full(10, 0.1)  # uniform → need all 10 to reach 0.95
        cs = credible_set(pip, coverage=0.95)
        assert len(cs) == 10

    def test_returns_sorted_indices(self):
        pip = np.array([0.1, 0.6, 0.3])
        cs = credible_set(pip)
        assert list(cs) == sorted(cs)

    def test_100_percent_coverage_returns_all(self):
        pip = abf_pip(np.array([1.0, 2.0, 3.0]))
        cs = credible_set(pip, coverage=1.0)
        assert len(cs) == 3

    def test_output_dtype_int(self):
        pip = np.array([0.7, 0.2, 0.1])
        cs = credible_set(pip)
        assert cs.dtype.kind == "i"


# ── lod_beta_to_z ─────────────────────────────────────────────────────────────


class TestLodBetaToZ:
    def test_zero_lod_gives_zero_z(self):
        z = lod_beta_to_z(np.array([0.0]), np.array([1.0]))
        assert float(z[0]) == pytest.approx(0.0, abs=1e-14)

    def test_negative_lod_clamped_to_zero(self):
        z = lod_beta_to_z(np.array([-5.0]), np.array([1.0]))
        assert float(z[0]) == pytest.approx(0.0, abs=1e-14)

    def test_sign_from_beta(self):
        lod = np.array([4.0, 4.0])
        beta = np.array([+1.0, -1.0])
        z = lod_beta_to_z(lod, beta)
        assert float(z[0]) > 0
        assert float(z[1]) < 0

    def test_z_squared_equals_chi2(self):
        lod = np.array([3.0])
        z = lod_beta_to_z(lod, np.array([1.0]))
        expected_chi2 = 3.0 * 2.0 * _LOG10_TO_LN
        assert float(z[0] ** 2) == pytest.approx(expected_chi2, rel=1e-10)

    def test_output_shape(self):
        lod = np.arange(5, dtype=float)
        beta = np.ones(5)
        assert lod_beta_to_z(lod, beta).shape == (5,)

    def test_zero_beta_treated_as_positive(self):
        z = lod_beta_to_z(np.array([2.0]), np.array([0.0]))
        assert float(z[0]) > 0


# ── fine_map_region ───────────────────────────────────────────────────────────


class TestFineMapRegion:
    @pytest.fixture
    def region(self):
        rng = np.random.default_rng(7)
        n_snps = 20
        lod = np.abs(rng.normal(2.0, 1.5, n_snps))
        beta = rng.choice([-1.0, 1.0], size=n_snps)
        return lod, beta

    def test_output_columns_default(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, n=50)
        assert list(df.columns) == ["snp", "lod", "z", "abf", "pip", "in_cs"]

    def test_output_columns_with_pos_mb(self, region):
        lod, beta = region
        pos = np.linspace(1.0, 5.0, len(lod))
        df = fine_map_region(lod, beta, n=50, pos_mb=pos)
        assert "pos_mb" in df.columns
        assert df.columns.tolist().index("pos_mb") == 1

    def test_sorted_by_pip_descending(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, n=50)
        assert list(df["pip"]) == sorted(df["pip"], reverse=True)

    def test_in_cs_count_matches_credible_set(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, n=50, coverage=0.95)
        cs_size = int(df["in_cs"].sum())
        pip = df["pip"].to_numpy()
        pip_sorted = np.sort(pip)[::-1]
        cumsum = np.cumsum(pip_sorted)
        expected = int(np.searchsorted(cumsum, 0.95, side="left")) + 1
        assert cs_size == expected

    def test_pip_sums_to_one(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, n=50)
        assert float(df["pip"].sum()) == pytest.approx(1.0, abs=1e-10)

    def test_snp_ids_used_when_provided(self, region):
        lod, beta = region
        ids = np.array([f"rs{i}" for i in range(len(lod))])
        df = fine_map_region(lod, beta, n=50, snp_ids=ids)
        assert set(df["snp"]) == set(ids)

    def test_n_row_count(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, n=50)
        assert len(df) == len(lod)

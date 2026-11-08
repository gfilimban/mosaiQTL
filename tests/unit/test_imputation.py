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
    merge_ld_groups,
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

    def test_nan_abf_does_not_dominate(self):
        # A NaN ABF (failed scan) must not collapse all other PIPs to zero.
        # Before the fix, np.isfinite(NaN)=False → log_abf=700 → NaN SNP gets
        # PIP≈1 and all others get PIP≈0.
        abf_clean = np.array([1.0, 2.0, 3.0])
        pip_clean = abf_pip(abf_clean)
        abf_with_nan = np.array([1.0, 2.0, 3.0, np.nan])
        pip_with_nan = abf_pip(abf_with_nan)
        # the NaN SNP receives near-zero PIP and the others preserve their ratios
        assert pip_with_nan[3] < 1e-10, "NaN ABF must not dominate"
        np.testing.assert_allclose(
            pip_with_nan[:3] / pip_with_nan[:3].sum(),
            pip_clean,
            atol=1e-10,
        )

    def test_nan_lod_excluded_from_fine_map_region(self):
        # A NaN LOD row must not produce a valid PIP through fine_map_region.
        lod = np.array([5.0, np.nan, 1.0, 0.5])
        beta = np.array([0.3, 0.1, 0.2, -0.1])
        df = fine_map_region(lod, beta, n=418)
        # NaN-LOD SNP gets near-zero pip — not in the 95% CS
        nan_row = df[df["snp"] == 1]  # index 1 has NaN lod
        assert len(nan_row) == 0 or float(nan_row["pip"].iloc[0]) < 1e-6


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


# ── merge_ld_groups ───────────────────────────────────────────────────────────


class TestMergeLdGroups:
    def _make_correlated_dosages(self, n: int = 20, rng=None):
        if rng is None:
            rng = np.random.default_rng(0)
        base = rng.uniform(0.0, 1.0, n)
        return base, base + rng.uniform(-0.02, 0.02, n)

    def test_identical_patterns_always_merged(self):
        # same pattern => r²=1 => merged regardless of threshold
        pat = (1, 0, 0, 0, 0, 0, 0, 0)
        rep_patterns = [pat]
        r2 = np.array([[1.0]])
        snp_patterns = [pat, pat, pat]
        groups = merge_ld_groups(rep_patterns, r2, snp_patterns)
        assert len(set(groups.values())) == 1

    def test_two_correlated_patterns_merge(self):
        rng = np.random.default_rng(1)
        d1, d2 = self._make_correlated_dosages(rng=rng)
        r2_val = float(np.corrcoef(d1, d2)[0, 1] ** 2)
        pat_a = (1, 0, 0, 0, 0, 0, 0, 0)
        pat_b = (1, 0, 0, 0, 0, 0, 0, 1)
        rep_patterns = [pat_a, pat_b]
        r2 = np.array([[1.0, r2_val], [r2_val, 1.0]])
        snp_patterns = [pat_a] * 3 + [pat_b] * 3
        groups = merge_ld_groups(rep_patterns, r2, snp_patterns, r2_threshold=0.8)
        if r2_val >= 0.8:
            assert len(set(groups.values())) == 1, 'high-r² patterns must merge'
        else:
            assert len(set(groups.values())) == 2, 'low-r² patterns must not merge'

    def test_uncorrelated_patterns_stay_separate(self):
        # r²=0 between patterns => two distinct groups
        pat_a = (1, 0, 0, 0, 0, 0, 0, 0)
        pat_b = (0, 1, 0, 0, 0, 0, 0, 0)
        rep_patterns = [pat_a, pat_b]
        r2 = np.array([[1.0, 0.0], [0.0, 1.0]])
        snp_patterns = [pat_a, pat_a, pat_b, pat_b]
        groups = merge_ld_groups(rep_patterns, r2, snp_patterns, r2_threshold=0.5)
        assert len(set(groups.values())) == 2

    def test_all_snps_assigned(self):
        pat_a = (1, 0, 0, 0, 0, 0, 0, 0)
        pat_b = (0, 0, 1, 0, 0, 0, 0, 0)
        rep_patterns = [pat_a, pat_b]
        r2 = np.array([[1.0, 0.1], [0.1, 1.0]])
        snp_patterns = [pat_a, pat_b, pat_a, pat_b, pat_b]
        groups = merge_ld_groups(rep_patterns, r2, snp_patterns)
        assert sorted(groups.keys()) == list(range(5))

    def test_group_ids_are_integers(self):
        pat = (1, 0, 0, 0, 0, 0, 0, 0)
        r2 = np.array([[1.0]])
        groups = merge_ld_groups([pat], r2, [pat, pat])
        assert all(isinstance(v, int) for v in groups.values())

    def test_three_patterns_partial_merge(self):
        # A-B correlated, B-C not => A and B merge, C stays separate
        pat_a = (1, 0, 0, 0, 0, 0, 0, 0)
        pat_b = (1, 0, 0, 0, 0, 0, 0, 1)
        pat_c = (0, 1, 0, 0, 0, 0, 0, 0)
        rep_patterns = [pat_a, pat_b, pat_c]
        r2 = np.array([
            [1.0, 0.95, 0.02],
            [0.95, 1.0, 0.03],
            [0.02, 0.03, 1.0],
        ])
        snp_patterns = [pat_a, pat_b, pat_c]
        groups = merge_ld_groups(rep_patterns, r2, snp_patterns, r2_threshold=0.8)
        assert groups[0] == groups[1], 'A and B should be in same group'
        assert groups[2] != groups[0], 'C should be in its own group'


# ── fine_map_region with ld_groups ────────────────────────────────────────────


class TestFineMapRegionLdAware:
    def test_pip_sums_to_one_with_ld_groups(self):
        n_snps = 10
        lod = np.ones(n_snps) * 2.0
        lod[3] = 7.0
        beta = np.ones(n_snps)
        ld_groups = {i: i // 2 for i in range(n_snps)}  # pairs
        df = fine_map_region(lod, beta, n=100, ld_groups=ld_groups)
        assert float(df['pip'].sum()) == pytest.approx(1.0, abs=1e-8)

    def test_all_same_group_distributes_evenly(self):
        n_snps = 5
        lod = np.array([1.0, 1.0, 8.0, 1.0, 1.0])
        beta = np.ones(n_snps)
        ld_groups = {i: 0 for i in range(n_snps)}  # all one group
        df = fine_map_region(lod, beta, n=200, ld_groups=ld_groups)
        # one group, mass=1 split equally => each PIP = 1/n_snps
        np.testing.assert_allclose(
            sorted(df['pip'].tolist()),
            sorted([1.0 / n_snps] * n_snps),
            atol=1e-10,
        )

    def test_group_cs_excludes_unrelated_snps(self):
        # Two LD groups: group 0 (SNPs 0–4, LOD=5) and group 1 (SNPs 5–9, LOD=0).
        # LD-aware CS should include all of group 0 and none of group 1.
        n_snps = 10
        lod = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        beta = np.ones(n_snps)
        ld_groups = {i: 0 if i < 5 else 1 for i in range(n_snps)}
        df = fine_map_region(lod, beta, n=100, ld_groups=ld_groups, coverage=0.90)
        cs_snps = set(df.loc[df['in_cs'], 'snp'].tolist())
        # signal group (0–4) must be in CS
        assert cs_snps.issuperset({0, 1, 2, 3, 4}), f'signal group missing: CS={cs_snps}'
        # noise group (5–9) must NOT be in CS when signal group alone covers >90%
        assert cs_snps.isdisjoint({5, 6, 7, 8, 9}), f'noise group in CS: {cs_snps}'

    def test_cs_members_within_ld_groups(self):
        n_snps = 8
        lod = np.array([1.0, 1.0, 1.0, 1.0, 6.0, 6.0, 1.0, 1.0])
        beta = np.ones(n_snps)
        # Group 0: SNPs 4+5 (signal), Group 1: remainder
        ld_groups = {4: 0, 5: 0, 0: 1, 1: 1, 2: 1, 3: 1, 6: 1, 7: 1}
        df = fine_map_region(lod, beta, n=100, ld_groups=ld_groups)
        cs_snps = set(df.loc[df['in_cs'], 'snp'].tolist())
        # SNPs 4 and 5 are the signal group; they should be in the CS
        assert 4 in cs_snps and 5 in cs_snps

    def test_in_cs_flag_consistent_with_coverage(self):
        n_snps = 12
        rng = np.random.default_rng(42)
        lod = rng.uniform(1.0, 8.0, n_snps)
        beta = np.ones(n_snps)
        ld_groups = {i: i // 3 for i in range(n_snps)}  # groups of 3
        df = fine_map_region(lod, beta, n=150, ld_groups=ld_groups, coverage=0.95)
        cs_pip_sum = float(df.loc[df['in_cs'], 'pip'].sum())
        assert cs_pip_sum >= 0.95 - 1e-8

    def test_output_row_count_unchanged(self):
        n_snps = 15
        lod = np.ones(n_snps) * 2.0
        beta = np.ones(n_snps)
        ld_groups = {i: i % 3 for i in range(n_snps)}
        df = fine_map_region(lod, beta, n=100, ld_groups=ld_groups)
        assert len(df) == n_snps

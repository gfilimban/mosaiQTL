"""Unit tests for qtl/imputation.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.imputation import (
    compute_log_abf,
    log_abf_to_pip,
    credible_set,
    lod_beta_to_z,
    fine_map_region,
    merge_ld_groups,
    ld_group_summary,
    LD_GROUP_COLS,
)

_LOG10_TO_LN = np.log(10.0)


def _se_from_lod(lod, beta):
    """Per-SNP SE whose Wald z = beta/SE has z² = 2·ln(10)·LOD."""
    lod = np.asarray(lod, dtype=float)
    return np.abs(beta) / np.sqrt(2.0 * _LOG10_TO_LN * np.maximum(lod, 1e-12))


# ── compute_log_abf ───────────────────────────────────────────────────────────


class TestComputeLogAbf:
    def test_hand_computed_values(self):
        # r = W/(V+W) = 0.04/1.04; log ABF = 0.5·ln(1-r) + 0.5·z²·r
        r = 0.04 / 1.04
        out = compute_log_abf(np.array([0.0, 3.0]), se=1.0, prior_var=0.04)
        np.testing.assert_allclose(
            out, [-0.5 * np.log(1.04), -0.5 * np.log(1.04) + 4.5 * r], atol=1e-12
        )

    def test_z_zero_is_negative(self):
        # no signal: the Bayes factor favours the null, so log ABF < 0
        assert float(compute_log_abf(np.array([0.0]), se=1.0)[0]) < 0.0

    def test_increases_with_abs_z(self):
        out = compute_log_abf(np.array([0.0, 1.0, 2.0, 3.0, 5.0]), se=1.0)
        assert np.all(np.diff(out) > 0)

    def test_symmetric_in_z(self):
        z = np.array([1.5, 2.5, 3.5])
        np.testing.assert_allclose(
            compute_log_abf(z, se=1.0), compute_log_abf(-z, se=1.0), atol=1e-14
        )

    def test_scalar_se_broadcasts(self):
        z = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            compute_log_abf(z, se=0.5), compute_log_abf(z, se=np.full(3, 0.5)), atol=1e-14
        )

    def test_output_shape(self):
        assert compute_log_abf(np.arange(10, dtype=float), se=1.0).shape == (10,)

    def test_larger_prior_var_raises_log_abf_for_nonzero_z(self):
        z = np.array([3.0])
        small = compute_log_abf(z, se=1.0, prior_var=0.01)
        large = compute_log_abf(z, se=1.0, prior_var=0.10)
        assert float(large[0]) > float(small[0])

    def test_large_z_is_finite_where_exp_would_overflow(self):
        out = compute_log_abf(np.array([300.0]), se=1.0)
        assert np.isfinite(out[0]) and out[0] > 709.0


# ── log_abf_to_pip ────────────────────────────────────────────────────────────


class TestLogAbfToPip:
    def test_sums_to_one(self):
        pip = log_abf_to_pip(np.log([0.5, 1.0, 2.5, 0.1]))
        assert float(pip.sum()) == pytest.approx(1.0, abs=1e-12)

    def test_matches_ratio_of_bayes_factors(self):
        pip = log_abf_to_pip(np.log([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(pip, [1 / 6, 2 / 6, 3 / 6], atol=1e-12)

    def test_argmax_matches_log_abf_argmax(self):
        x = np.array([-1.2, 1.6, 0.2, -0.2])
        assert int(np.argmax(log_abf_to_pip(x))) == int(np.argmax(x))

    def test_single_snp_pip_is_one(self):
        assert float(log_abf_to_pip(np.array([3.7]))[0]) == pytest.approx(1.0, abs=1e-12)

    def test_equal_log_abf_gives_uniform_pip(self):
        np.testing.assert_allclose(log_abf_to_pip(np.zeros(8)), 1.0 / 8, atol=1e-12)

    def test_shift_invariant(self):
        x = np.array([0.3, -1.0, 2.0])
        np.testing.assert_allclose(log_abf_to_pip(x), log_abf_to_pip(x + 1000.0), atol=1e-12)

    def test_huge_log_abf_does_not_overflow(self):
        pip = log_abf_to_pip(np.array([5000.0, 4990.0, 0.0]))
        assert np.all(np.isfinite(pip))
        assert float(pip[0]) == pytest.approx(1.0 / (1.0 + np.exp(-10.0)), rel=1e-9)

    def test_matches_posterior_from_null_over_alternative_form(self):
        z = np.array([0.5, 2.0, 5.0, 1.0])
        se = np.array([0.1, 0.2, 0.15, 0.3])
        r = 0.04 / (se**2 + 0.04)
        null_over_alt = np.exp(-0.5 * z**2 * r) / np.sqrt(1.0 - r)
        expected = (1.0 / null_over_alt) / np.sum(1.0 / null_over_alt)
        np.testing.assert_allclose(log_abf_to_pip(compute_log_abf(z, se)), expected, rtol=1e-12)

    def test_very_negative_log_abf_gets_zero_weight(self):
        pip = log_abf_to_pip(np.array([-1e4, 0.0, -2e4]))
        assert float(pip[1]) == pytest.approx(1.0, abs=1e-12)

    def test_nan_gets_zero_weight_and_others_keep_ratios(self):
        clean = log_abf_to_pip(np.log([1.0, 2.0, 3.0]))
        with_nan = log_abf_to_pip(np.append(np.log([1.0, 2.0, 3.0]), np.nan))
        assert with_nan[3] == 0.0
        np.testing.assert_allclose(with_nan[:3], clean, atol=1e-12)

    def test_no_finite_value_returns_zeros(self):
        np.testing.assert_array_equal(log_abf_to_pip(np.array([np.nan, np.nan])), [0.0, 0.0])

    def test_ld_group_with_nan_first_member_keeps_its_valid_evidence(self):
        lod = np.array([np.nan, 6.0, 0.5, 0.4])
        beta = np.array([0.3, 0.3, 0.2, 0.2])
        df = fine_map_region(
            lod, beta, _se_from_lod(lod, beta), ld_groups={0: 0, 1: 0, 2: 1, 3: 2}
        )
        by_snp = df.set_index("snp")
        assert by_snp.loc[1, "pip"] > by_snp.loc[2, "pip"]
        assert bool(by_snp.loc[1, "in_cs"])
        assert float(df["pip"].sum()) == pytest.approx(1.0, abs=1e-9)

    def test_nan_lod_excluded_from_fine_map_region(self):
        # A NaN LOD row must not produce a valid PIP through fine_map_region.
        lod = np.array([5.0, np.nan, 1.0, 0.5])
        beta = np.array([0.3, 0.1, 0.2, -0.1])
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta))
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
        pip = log_abf_to_pip(np.log([1.0, 2.0, 3.0]))
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
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta))
        assert list(df.columns) == ["snp", "lod", "z", "log_abf", "pip", "in_cs"]

    def test_output_columns_with_pos_mb(self, region):
        lod, beta = region
        pos = np.linspace(1.0, 5.0, len(lod))
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), pos_mb=pos)
        assert "pos_mb" in df.columns
        assert df.columns.tolist().index("pos_mb") == 1

    def test_sorted_by_pip_descending(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta))
        assert list(df["pip"]) == sorted(df["pip"], reverse=True)

    def test_tied_pip_ordered_by_lod_then_tss_distance(self):
        lod = np.array([3.0, 5.0, 4.0, 5.0])
        beta = np.ones(4)
        pos = np.array([1.0, 3.0, 2.0, 1.2])
        groups = {i: 1 for i in range(4)}
        snps = ['a', 'b', 'c', 'd']
        df = fine_map_region(
            lod, beta, _se_from_lod(lod, beta), snp_ids=snps, pos_mb=pos,
            ld_groups=groups, tss_mb=1.0,
        )
        assert df['pip'].nunique() == 1
        assert list(df['snp']) == ['d', 'b', 'c', 'a']

    def test_tie_order_independent_of_input_order(self):
        lod = np.array([5.0, 5.0, 5.0])
        beta = np.ones(3)
        pos = np.array([1.5, 1.1, 0.7])
        groups = {i: 1 for i in range(3)}
        snps = np.array(['x', 'y', 'z'])
        first = fine_map_region(
            lod, beta, _se_from_lod(lod, beta), snp_ids=list(snps), pos_mb=pos,
            ld_groups=groups, tss_mb=1.0,
        )
        perm = np.array([2, 0, 1])
        second = fine_map_region(
            lod[perm], beta[perm], _se_from_lod(lod[perm], beta[perm]),
            snp_ids=list(snps[perm]), pos_mb=pos[perm], ld_groups=groups,
            tss_mb=1.0,
        )
        assert list(first['snp']) == list(second['snp']) == ['y', 'z', 'x']

    def test_in_cs_count_matches_credible_set(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), coverage=0.95)
        cs_size = int(df["in_cs"].sum())
        pip = df["pip"].to_numpy()
        pip_sorted = np.sort(pip)[::-1]
        cumsum = np.cumsum(pip_sorted)
        expected = int(np.searchsorted(cumsum, 0.95, side="left")) + 1
        assert cs_size == expected

    def test_pip_sums_to_one(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta))
        assert float(df["pip"].sum()) == pytest.approx(1.0, abs=1e-10)

    def test_snp_ids_used_when_provided(self, region):
        lod, beta = region
        ids = np.array([f"rs{i}" for i in range(len(lod))])
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), snp_ids=ids)
        assert set(df["snp"]) == set(ids)

    def test_n_row_count(self, region):
        lod, beta = region
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta))
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
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), ld_groups=ld_groups)
        assert float(df['pip'].sum()) == pytest.approx(1.0, abs=1e-8)

    def test_all_same_group_distributes_evenly(self):
        n_snps = 5
        lod = np.array([1.0, 1.0, 8.0, 1.0, 1.0])
        beta = np.ones(n_snps)
        ld_groups = {i: 0 for i in range(n_snps)}  # all one group
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), ld_groups=ld_groups)
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
        df = fine_map_region(
            lod, beta, _se_from_lod(lod, beta), ld_groups=ld_groups, coverage=0.90
        )
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
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), ld_groups=ld_groups)
        cs_snps = set(df.loc[df['in_cs'], 'snp'].tolist())
        # SNPs 4 and 5 are the signal group; they should be in the CS
        assert 4 in cs_snps and 5 in cs_snps

    def test_in_cs_flag_consistent_with_coverage(self):
        n_snps = 12
        rng = np.random.default_rng(42)
        lod = rng.uniform(1.0, 8.0, n_snps)
        beta = np.ones(n_snps)
        ld_groups = {i: i // 3 for i in range(n_snps)}  # groups of 3
        df = fine_map_region(
            lod, beta, _se_from_lod(lod, beta), ld_groups=ld_groups, coverage=0.95
        )
        cs_pip_sum = float(df.loc[df['in_cs'], 'pip'].sum())
        assert cs_pip_sum >= 0.95 - 1e-8

    def test_output_row_count_unchanged(self):
        n_snps = 15
        lod = np.ones(n_snps) * 2.0
        beta = np.ones(n_snps)
        ld_groups = {i: i % 3 for i in range(n_snps)}
        df = fine_map_region(lod, beta, _se_from_lod(lod, beta), ld_groups=ld_groups)
        assert len(df) == n_snps


class TestLdGroupSummary:
    @staticmethod
    def _fm(coverage=0.95):
        lod = np.array([1.0, 1.0, 6.0, 6.0, 6.0, 0.5, 0.5, 0.5])
        beta = np.ones(len(lod))
        ld_groups = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2}
        patterns = np.array(['10', '10', '01', '01', '11', '00', '00', '00'], dtype=object)
        pos = np.arange(len(lod), dtype=float) * 1e-3
        return fine_map_region(
            lod, beta, _se_from_lod(lod, beta), pos_mb=pos, ld_groups=ld_groups,
            patterns=patterns, coverage=coverage,
        )

    def test_columns_and_group_pips_sum_to_one(self):
        out = ld_group_summary(self._fm())
        assert list(out.columns) == LD_GROUP_COLS
        assert float(out['group_pip'].sum()) == pytest.approx(1.0, abs=1e-9)

    def test_ranked_by_group_pip_with_rank_ids(self):
        out = ld_group_summary(self._fm())
        assert out['ld_group'].tolist() == [1, 2, 3]
        assert out['group_pip'].is_monotonic_decreasing
        assert out.loc[0, 'n_variants'] == 3
        assert out.loc[0, 'n_patterns'] == 2
        assert out.loc[0, 'patterns'] == '01,11'
        assert out.loc[0, 'span_bp'] == 2000

    def test_group_membership_matches_variant_credible_set(self):
        fm = self._fm()
        out = ld_group_summary(fm)
        cs_variants = set(fm.loc[fm['in_cs'], 'snp'])
        top_members = set(fm.loc[fm['ld_group'] == fm.loc[fm['pip'].idxmax(), 'ld_group'], 'snp'])
        assert bool(out.loc[0, 'in_cs'])
        assert top_members <= cs_variants
        assert int(out.loc[out['in_cs'], 'n_variants'].sum()) == len(cs_variants)

    def test_best_lod_variant_belongs_to_group(self):
        out = ld_group_summary(self._fm())
        assert out.loc[0, 'best_lod'] == pytest.approx(6.0)
        assert out.loc[0, 'best_lod_variant'] in {2, 3, 4}

    def test_empty_when_no_valid_variant(self):
        fm = self._fm().assign(pip=0.0)
        assert ld_group_summary(fm).empty

    def test_raises_when_pips_do_not_sum_to_one(self):
        fm = self._fm()
        fm['pip'] = fm['pip'] * 0.5
        with pytest.raises(ValueError, match='sum to'):
            ld_group_summary(fm)


# ── simulation checks ─────────────────────────────────────────────────────────


def _ols_beta_se(d_std, y):
    """Per-SNP simple-regression slope and SE for standardized dosages (n, m)."""
    n = len(y)
    yc = y - y.mean()
    beta = d_std.T @ yc / n
    rss = ((yc[:, None] - d_std * beta) ** 2).sum(axis=0)
    se = np.sqrt(rss / (n - 2) / n)
    return beta, se


class TestFineMappingSimulation:
    def test_credible_set_coverage_matches_nominal(self):
        rng = np.random.default_rng(11)
        n, n_snps, n_rep = 300, 40, 200
        hits = 0
        for _ in range(n_rep):
            shared = rng.normal(size=(n, n_snps // 4))
            noise = rng.normal(size=(n, n_snps))
            d = np.repeat(shared, 4, axis=1) * 0.8 + noise * 0.6
            d = (d - d.mean(axis=0)) / d.std(axis=0)
            causal = int(rng.integers(n_snps))
            y = 0.4 * d[:, causal] + rng.normal(size=n)
            beta, se = _ols_beta_se(d, y)
            lod = (n / 2.0) * np.log10(1.0 + (beta / se) ** 2 / (n - 2))
            df = fine_map_region(
                lod, beta, se, snp_ids=np.arange(n_snps), coverage=0.95
            )
            cs = set(df.loc[df["in_cs"], "snp"].tolist())
            hits += causal in cs
        assert hits / n_rep >= 0.90

    def test_null_phenotype_calls_nothing_at_fdr(self):
        from scipy import stats

        from mosaiqtl.qtl.fdr import qvalue

        rng = np.random.default_rng(5)
        n, n_genes = 100, 600
        geno = rng.normal(size=(n, 1))
        geno = (geno - geno.mean()) / geno.std()
        pvals = np.empty(n_genes)
        for g in range(n_genes):
            y = rng.permutation(rng.normal(size=n))
            beta, se = _ols_beta_se(geno, y)
            pvals[g] = 2 * stats.t.sf(abs(beta[0] / se[0]), df=n - 2)
        q, pi0 = qvalue(pvals)
        assert pi0 > 0.85
        assert int((q < 0.05).sum()) <= 2


# ── missing founder alleles in the SNP scan ───────────────────────────────────


class TestMissingFounderAlleles:
    @pytest.fixture
    def scan_inputs(self, tiny_cache, tiny_kinship, strain_ids):
        import pandas as pd

        from mosaiqtl.genotype.kinship import decomp_kinship

        founders = list(tiny_cache.founders)
        rng = np.random.default_rng(3)
        rows = {}
        for i, pos in enumerate((20.0, 40.0)):
            g = rng.integers(0, 2, size=len(founders)).astype(float)
            g[0], g[1] = 0.0, 1.0
            rows[f'full{i}'] = {'chr': '1', 'pos_mb': pos, **dict(zip(founders, g))}
        withheld = {'chr': '1', 'pos_mb': 60.0}
        withheld.update(
            {f: (np.nan if j == 2 else float(j % 2)) for j, f in enumerate(founders)}
        )
        rows['missing'] = withheld
        pheno = rng.normal(size=len(strain_ids))
        return (
            pheno,
            pd.DataFrame.from_dict(rows, orient='index'),
            decomp_kinship(tiny_kinship),
        )

    def test_snp_with_missing_founder_is_skipped(self, tiny_cache, scan_inputs):
        from mosaiqtl.qtl.snp_scan import imputed_snp_scan_h2lmm

        pheno, snp_genos, kd = scan_inputs
        df = imputed_snp_scan_h2lmm(
            pheno, tiny_cache, snp_genos, kinship_decomp=kd, use_rint=False
        )
        assert 'missing' not in set(df['snp'])
        assert {'full0', 'full1'} <= set(df['snp'])

    def test_missing_call_is_not_treated_as_ref(self, tiny_cache, scan_inputs):
        from mosaiqtl.qtl import snp_scan

        pheno, snp_genos, kd = scan_inputs
        founders = list(tiny_cache.founders)
        as_ref = snp_genos.copy()
        as_ref[founders] = as_ref[founders].fillna(0.0)
        df_missing = snp_scan.imputed_snp_scan_h2lmm(
            pheno, tiny_cache, snp_genos, kinship_decomp=kd, use_rint=False
        )
        df_ref = snp_scan.imputed_snp_scan_h2lmm(
            pheno, tiny_cache, as_ref, kinship_decomp=kd, use_rint=False
        )
        assert 'missing' in set(df_ref['snp'])
        assert 'missing' not in set(df_missing['snp'])

    def test_replicates_are_fit_at_sample_level_with_the_sample_kinship(
        self, tiny_cache, scan_inputs, strain_ids
    ):
        from mosaiqtl.qtl import lmm
        from mosaiqtl.qtl.snp_scan import imputed_snp_scan_h2lmm
        from mosaiqtl.utils import build_z_matrix, rint

        _, snp_genos, kd = scan_inputs
        sample_ids = list(strain_ids) * 2
        y = np.random.default_rng(9).normal(size=len(sample_ids))
        df = imputed_snp_scan_h2lmm(
            y, tiny_cache, snp_genos, kinship_decomp=kd, sample_ids=sample_ids
        )

        z, _ = build_z_matrix(sample_ids, list(tiny_cache.sample_ids))
        k_m = (kd['1']['vectors'] * kd['1']['values']) @ kd['1']['vectors'].T
        evals, evecs = np.linalg.eigh(z @ k_m @ z.T)
        x = np.ones((len(y), 1))
        h2, _ = lmm.estimate_h2_null(evecs.T @ rint(y), evecs.T @ x, evals, method='ml')
        assert (df['n'] == len(sample_ids)).all()
        assert df['h2'].to_numpy() == pytest.approx(h2)


# ── SNP scan: interpolated haplotype probabilities and minor-strain filter ────


class TestSnpScanInterpolationAndFilter:
    @pytest.fixture
    def setup(self, tiny_cache, tiny_kinship, strain_ids):
        import pandas as pd

        from mosaiqtl.genotype.kinship import decomp_kinship

        founders = list(tiny_cache.founders)
        _, pos, probs = tiny_cache.get_chromosome('1')
        g = np.zeros(len(founders))
        g[[0, 3, 5]] = 1.0
        x = float(pos[4] + 0.25 * (pos[5] - pos[4]))
        snp = pd.DataFrame(
            {'chr': ['1'], 'pos_mb': [x], **{f: [v] for f, v in zip(founders, g)}},
            index=['mid'],
        )
        y = np.random.default_rng(5).normal(size=len(strain_ids))
        return snp, g, x, pos, probs, y, decomp_kinship(tiny_kinship)

    def test_dosage_uses_probabilities_interpolated_to_snp_position(
        self, tiny_cache, setup, monkeypatch
    ):
        from mosaiqtl.qtl import snp_scan

        snp, g, x, pos, probs, y, kd = setup
        monkeypatch.setattr(snp_scan, 'MIN_MINOR_STRAINS', 0)
        df = snp_scan.imputed_snp_scan_h2lmm(
            y, tiny_cache, snp, kinship_decomp=kd, use_rint=False
        )
        t = (x - pos[4]) / (pos[5] - pos[4])
        expected = ((1 - t) * probs[4] + t * probs[5]) @ g
        nearest = probs[int(np.argmin(np.abs(pos - x)))] @ g
        assert df['dosage_mean'].iloc[0] == pytest.approx(expected.mean())
        assert not np.isclose(expected.mean(), nearest.mean())

    def test_minor_strains_reported_and_threshold_enforced(
        self, tiny_cache, setup, monkeypatch
    ):
        from mosaiqtl.qtl import snp_scan

        snp, g, x, pos, probs, y, kd = setup
        monkeypatch.setattr(snp_scan, 'MIN_MINOR_STRAINS', 0)
        df = snp_scan.imputed_snp_scan_h2lmm(
            y, tiny_cache, snp, kinship_decomp=kd, use_rint=False
        )
        t = (x - pos[4]) / (pos[5] - pos[4])
        tot = (((1 - t) * probs[4] + t * probs[5]) @ g).sum()
        n = probs.shape[1]
        assert df['minor_strains'].iloc[0] == pytest.approx(min(tot, n - tot))

        monkeypatch.setattr(snp_scan, 'MIN_MINOR_STRAINS', n)
        gone = snp_scan.imputed_snp_scan_h2lmm(
            y, tiny_cache, snp, kinship_decomp=kd, use_rint=False
        )
        assert gone.empty

    def test_minor_strains_flows_into_scan_and_imputed_schemas(
        self, tiny_cache, setup, monkeypatch
    ):
        from mosaiqtl.pipeline import IMPUTED_SNP_COLS
        from mosaiqtl.qtl import snp_scan

        snp, _, _, _, _, y, kd = setup
        monkeypatch.setattr(snp_scan, 'MIN_MINOR_STRAINS', 0)
        df = snp_scan.imputed_snp_scan_h2lmm(
            y, tiny_cache, snp, kinship_decomp=kd, use_rint=False
        )
        assert 'minor_strains' in snp_scan.SNP_SCAN_COLS
        assert 'minor_strains' in IMPUTED_SNP_COLS
        assert set(snp_scan.SNP_SCAN_COLS) <= set(df.columns)

    def test_allele_frequency_is_over_samples_not_strains(
        self, tiny_cache, setup, strain_ids, monkeypatch
    ):
        from mosaiqtl.qtl import snp_scan

        snp, g, x, pos, probs, _, kd = setup
        monkeypatch.setattr(snp_scan, 'MIN_MINOR_STRAINS', 0)
        sample_ids = list(strain_ids) + list(strain_ids[:3])
        y = np.random.default_rng(6).normal(size=len(sample_ids))
        df = snp_scan.imputed_snp_scan_h2lmm(
            y, tiny_cache, snp, kinship_decomp=kd, sample_ids=sample_ids
        )
        t = (x - pos[4]) / (pos[5] - pos[4])
        d = ((1 - t) * probs[4] + t * probs[5]) @ g
        per_sample = np.concatenate([d, d[:3]])
        assert df['dosage_mean'].iloc[0] == pytest.approx(per_sample.mean())
        assert df['n'].iloc[0] == len(sample_ids)


# ── end-to-end fine-mapping simulation (real scan → real fine-mapping) ────────


class TestFineMappingEndToEnd:
    """Phenotypes simulated from a planted SNP run through the actual SNP scan
    and Bayesian fine-mapping, not through hand-built z-scores."""

    N_REPS = 20
    CAUSAL = 7

    @pytest.fixture
    def region(self, tiny_cache, tiny_kinship):
        import pandas as pd

        from mosaiqtl.genotype.kinship import decomp_kinship

        founders = list(tiny_cache.founders)
        _, pos, probs = tiny_cache.get_chromosome('1')
        rng = np.random.default_rng(11)
        rows = {}
        for i, p in enumerate(pos):
            g = rng.integers(0, 2, size=len(founders)).astype(float)
            g[0], g[1] = 0.0, 1.0
            rows[f'snp{i}'] = {'chr': '1', 'pos_mb': float(p), **dict(zip(founders, g))}
        snp_genos = pd.DataFrame.from_dict(rows, orient='index')
        causal_geno = snp_genos.loc[f'snp{self.CAUSAL}', founders].to_numpy(dtype=float)
        dosage = probs[self.CAUSAL] @ causal_geno
        return snp_genos, dosage, decomp_kinship(tiny_kinship)

    def _fine_map(self, tiny_cache, snp_genos, kd, y):
        from mosaiqtl.qtl.snp_scan import imputed_snp_scan_h2lmm

        scan = imputed_snp_scan_h2lmm(
            y, tiny_cache, snp_genos, kinship_decomp=kd, use_rint=False
        )
        return fine_map_region(
            scan['lod'].to_numpy(),
            scan['beta_dosage'].to_numpy(),
            scan['se_dosage'].to_numpy(),
            snp_ids=scan['snp'].to_numpy(),
        )

    def test_planted_snp_recovered_in_credible_set(self, tiny_cache, region):
        snp_genos, dosage, kd = region
        z = (dosage - dosage.mean()) / dosage.std()
        hits = 0
        for r in range(self.N_REPS):
            noise = np.random.default_rng(100 + r).normal(scale=0.3, size=len(z))
            fm = self._fine_map(tiny_cache, snp_genos, kd, 3.0 * z + noise)
            hits += f'snp{self.CAUSAL}' in set(fm.loc[fm['in_cs'], 'snp'])
        assert hits / self.N_REPS >= 0.8

    def test_null_phenotype_gives_diffuse_posterior(self, tiny_cache, region):
        snp_genos, dosage, kd = region
        z = (dosage - dosage.mean()) / dosage.std()
        null_top, planted_top = [], []
        for r in range(self.N_REPS):
            y_null = np.random.default_rng(500 + r).standard_normal(len(z))
            noise = np.random.default_rng(900 + r).normal(scale=0.3, size=len(z))
            null_top.append(self._fine_map(tiny_cache, snp_genos, kd, y_null)['pip'].max())
            planted_top.append(
                self._fine_map(tiny_cache, snp_genos, kd, 3.0 * z + noise)['pip'].max()
            )
        assert np.median(null_top) < np.median(planted_top)


# ── allele-frequency columns of the imputed SNP table ─────────────────────────


class TestImputedAlleleCountColumns:
    def test_mac_and_ma_sample_are_populated_from_sample_count(
        self, tiny_cache, tiny_kinship, strain_ids, monkeypatch
    ):
        import pandas as pd

        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.pipeline import impute_egene
        from mosaiqtl.qtl import snp_scan

        monkeypatch.setattr(snp_scan, 'MIN_MINOR_STRAINS', 0)
        founders = list(tiny_cache.founders)
        _, pos, probs = tiny_cache.get_chromosome('1')
        rng = np.random.default_rng(21)
        rows = []
        for i, p in enumerate(pos):
            alt = rng.integers(0, 2, size=len(founders))
            alt[0], alt[1] = 0, 1
            rows.append(
                {
                    'marker': f'snp{i}',
                    'chr': '1',
                    'pos_bp': int(round(p * 1e6)),
                    'pos_mb': float(p),
                    'rsid': f'rs{i}',
                    **{f: ('G' if a else 'A') for f, a in zip(founders, alt)},
                }
            )
        alleles = pd.DataFrame(rows)
        sample_ids = list(strain_ids) + list(strain_ids[:3])
        y = np.random.default_rng(22).normal(size=len(sample_ids))

        df = impute_egene(
            'g1', '1', y, tiny_cache, alleles,
            int(pos[0] * 1e6), 'snp0', 1.0,
            int(pos[0] * 1e6), int(pos[-1] * 1e6) + 1,
            kinship_decomp=decomp_kinship(tiny_kinship),
            sample_ids=sample_ids,
            ref_founder=founders[0],
            use_rint=False,
        )

        assert df['mac'].notna().all()
        assert df['ma_sample'].notna().all()
        expected = (np.minimum(df['af'], 1.0 - df['af']) * len(sample_ids)).round()
        assert df['mac'].astype(float).to_numpy() == pytest.approx(expected.to_numpy())
        assert (df['ma_sample'] == df['mac']).all()

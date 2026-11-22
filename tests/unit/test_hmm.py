"""Unit tests for genotype/hmm.py — forward-backward HMM."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.genotype.hmm import (
    build_diplotype_states,
    diplotype_to_allele_probs,
    haldane_recomb,
    transition_matrix,
    build_emission_matrix,
    forward_backward,
    uniform_init,
    haplotype_transition_matrix,
    _augmented_positions,
    interpolate_posteriors,
    calc_genoprob_cc,
    x_haplotype_prior,
    x_haplotype_transition_matrix,
)

N_FOUNDERS = 8
N_STATES = N_FOUNDERS * (N_FOUNDERS + 1) // 2  # 36


# ── diplotype state index ─────────────────────────────────────────────────────


class TestBuildDiplotype:
    def test_state_count(self):
        states, _ = build_diplotype_states(N_FOUNDERS)
        assert len(states) == N_STATES

    def test_states_ordered(self):
        states, _ = build_diplotype_states(N_FOUNDERS)
        for f1, f2 in states:
            assert f1 <= f2

    def test_state_to_idx_roundtrip(self):
        states, idx = build_diplotype_states(N_FOUNDERS)
        for i, s in enumerate(states):
            assert idx[s] == i

    def test_homozygous_states_present(self):
        states, idx = build_diplotype_states(N_FOUNDERS)
        for f in range(N_FOUNDERS):
            assert (f, f) in idx


class TestDiplotypeToAlleleProbs:
    def test_homozygous_maps_to_one_founder(self):
        states, idx = build_diplotype_states(N_FOUNDERS)
        for f in range(N_FOUNDERS):
            dipl = np.zeros(N_STATES)
            dipl[idx[(f, f)]] = 1.0
            allele = diplotype_to_allele_probs(dipl, N_FOUNDERS)
            assert allele[f] == pytest.approx(1.0, abs=1e-10)
            assert allele.sum() == pytest.approx(1.0, abs=1e-10)

    def test_heterozygous_splits_equally(self):
        states, idx = build_diplotype_states(N_FOUNDERS)
        dipl = np.zeros(N_STATES)
        dipl[idx[(0, 1)]] = 1.0
        allele = diplotype_to_allele_probs(dipl, N_FOUNDERS)
        assert allele[0] == pytest.approx(0.5, abs=1e-10)
        assert allele[1] == pytest.approx(0.5, abs=1e-10)
        assert allele[2:].sum() == pytest.approx(0.0, abs=1e-10)

    def test_batch_shape(self):
        rng = np.random.default_rng(0)
        n_loci, n_strains = 5, 3
        dipl = rng.dirichlet(np.ones(N_STATES), size=(n_loci, n_strains))
        allele = diplotype_to_allele_probs(dipl, N_FOUNDERS)
        assert allele.shape == (n_loci, n_strains, N_FOUNDERS)

    def test_sum_to_one(self):
        rng = np.random.default_rng(1)
        dipl = rng.dirichlet(np.ones(N_STATES), size=(10,))
        allele = diplotype_to_allele_probs(dipl, N_FOUNDERS)
        np.testing.assert_allclose(allele.sum(axis=-1), 1.0, atol=1e-10)


# ── transition / emission ─────────────────────────────────────────────────────


class TestHaldane:
    def test_zero_distance(self):
        assert haldane_recomb(0.0) == pytest.approx(0.0)

    def test_large_distance_approaches_half(self):
        r = haldane_recomb(10.0)
        assert abs(r - 0.5) < 1e-4

    def test_symmetric(self):
        assert haldane_recomb(0.1) == pytest.approx(haldane_recomb(-0.1))

    def test_range(self):
        for d in [0.01, 0.1, 0.5, 1.0, 2.0]:
            r = haldane_recomb(d)
            assert 0.0 <= r <= 0.5


class TestTransitionMatrix:
    def test_shape(self):
        T = transition_matrix(0.01)
        assert T.shape == (N_STATES, N_STATES)

    def test_rows_sum_to_one(self):
        T = transition_matrix(0.05)
        np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegative(self):
        T = transition_matrix(0.1)
        assert np.all(T >= -1e-12)

    def test_zero_recomb_is_identity(self):
        T = transition_matrix(0.0)
        np.testing.assert_allclose(T, np.eye(N_STATES), atol=1e-10)

    def test_rows_sum_to_one_without_renormalisation(self):
        for r in [1e-4, 0.005, 0.05, 0.2, 0.5]:
            T = transition_matrix(r)
            np.testing.assert_allclose(T.sum(axis=1), 1.0, atol=1e-12)

    def test_homozygous_to_heterozygous_matches_ordered_outcomes(self):
        r = 0.05
        J = N_FOUNDERS
        T = transition_matrix(r)
        states, idx = build_diplotype_states(J)
        H = haplotype_transition_matrix(r, J)
        expected = 2 * H[0, 0] * H[0, 1]
        np.testing.assert_allclose(T[idx[(0, 0)], idx[(0, 1)]], expected, rtol=1e-12)

    def test_uniform_haplotype_mixture_is_stationary(self):
        T = transition_matrix(0.3)
        states, _ = build_diplotype_states(N_FOUNDERS)
        pi = np.array([1.0 if a == b else 2.0 for a, b in states])
        pi /= pi.sum()
        np.testing.assert_allclose(pi @ T, pi, atol=1e-12)


class TestHaplotypeTransitionMatrix:
    def test_rows_sum_to_one_and_symmetric(self):
        for r in [0.0, 0.01, 0.2, 0.5]:
            H = haplotype_transition_matrix(r, N_FOUNDERS)
            np.testing.assert_allclose(H.sum(axis=1), 1.0, atol=1e-12)
            np.testing.assert_allclose(H, H.T, atol=1e-15)

    def test_limits(self):
        np.testing.assert_allclose(
            haplotype_transition_matrix(0.0, N_FOUNDERS), np.eye(N_FOUNDERS)
        )
        np.testing.assert_allclose(
            haplotype_transition_matrix(0.5, N_FOUNDERS),
            np.full((N_FOUNDERS, N_FOUNDERS), 1.0 / N_FOUNDERS),
            atol=1e-12,
        )

    def test_closed_form_entries(self):
        r = 0.07
        H = haplotype_transition_matrix(r, N_FOUNDERS)
        np.testing.assert_allclose(H[0, 0], (1 - r) / (1 + 6 * r), rtol=1e-12)
        np.testing.assert_allclose(H[0, 1], r / (1 + 6 * r), rtol=1e-12)

    def test_small_r_switch_probability_is_sevenfold_expansion(self):
        r = 1e-5
        H = haplotype_transition_matrix(r, N_FOUNDERS)
        np.testing.assert_allclose(1.0 - H[0, 0], 7 * r, rtol=1e-4)

    def test_requires_eight_founders(self):
        with pytest.raises(ValueError):
            haplotype_transition_matrix(0.1, 4)


def _simulate_sibmated_lines(r, n_lines, n_gen, rng):
    """founder carried at two loci by lines inbred by sibling mating.

    forward simulation of the eight-founder funnel followed by sibling
    mating with Haldane gametes; a chromosome is a pair of founder labels
    (locus 0, locus 1) and an individual is two chromosomes.
    """

    def gamete(ind):
        first = rng.integers(0, 2, n_lines)
        rec = rng.random(n_lines) < r
        idx = np.arange(n_lines)
        other = np.where(rec, 1 - first, first)
        return np.stack([ind[idx, first, 0], ind[idx, other, 1]], axis=1)

    def mate(p1, p2):
        return np.stack([gamete(p1), gamete(p2)], axis=1)

    order = np.argsort(rng.random((n_lines, N_FOUNDERS)), axis=1)
    founders = [
        np.broadcast_to(order[:, i, None, None], (n_lines, 2, 2)).copy()
        for i in range(N_FOUNDERS)
    ]
    q1 = mate(mate(founders[0], founders[1]), mate(founders[2], founders[3]))
    q2 = mate(mate(founders[4], founders[5]), mate(founders[6], founders[7]))
    a, b = q1, q2
    for _ in range(n_gen):
        a, b = mate(a, b), mate(a, b)
    return a[:, 0, :]


class TestHaplotypeKernelAgainstSimulation:
    @pytest.mark.parametrize('r', [0.01, 0.1, 0.3])
    def test_same_founder_probability_matches_sibling_mating(self, r):
        n_lines = 40_000
        rng = np.random.default_rng(11)
        hap = _simulate_sibmated_lines(r, n_lines, 80, rng)
        observed = float(np.mean(hap[:, 0] == hap[:, 1]))
        expected = haplotype_transition_matrix(r, N_FOUNDERS)[0, 0]
        se = np.sqrt(expected * (1 - expected) / n_lines)
        assert abs(observed - expected) < 4 * se


class TestInbredHaplotypePath:
    def _data(self, n_markers=30):
        rng = np.random.default_rng(3)
        fg = np.array(
            [
                [rng.choice(["A", "B"]) for _ in range(N_FOUNDERS)]
                for _ in range(n_markers)
            ],
            dtype=object,
        )
        gmap = np.linspace(0, 20, n_markers)
        truth = np.where(np.arange(n_markers) < 15, 2, 5)
        geno = np.array([[fg[m, truth[m]]] for m in range(n_markers)], dtype=object)
        return geno, fg, gmap, truth

    def test_probs_sum_to_one_and_recover_founder(self):
        geno, fg, gmap, truth = self._data()
        probs = calc_genoprob_cc(geno, fg, gmap, n_founders=N_FOUNDERS)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        assert probs[0, 2, 3] > 0.5 and probs[0, 5, -3] > 0.5

    def test_no_heterozygous_mass_far_from_markers_with_diplotype_output(self):
        geno, fg, gmap, _ = self._data()
        dip = calc_genoprob_cc(
            geno, fg, gmap, n_founders=N_FOUNDERS, return_additive=False
        )
        states, _ = build_diplotype_states(N_FOUNDERS)
        het = np.array([a != b for a, b in states])
        assert dip[:, het, :].sum() == 0.0
        np.testing.assert_allclose(dip.sum(axis=1), 1.0, atol=1e-9)

    def test_pseudomarkers_sum_to_one(self):
        geno, fg, gmap, _ = self._data()
        probs, pos, is_real = calc_genoprob_cc(
            geno, fg, gmap, n_founders=N_FOUNDERS, step_cM=0.5
        )
        assert probs.shape[1] == N_FOUNDERS
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)

    def test_non_inbred_path_keeps_diplotype_states(self):
        geno, fg, gmap, _ = self._data()
        dip = calc_genoprob_cc(
            geno,
            fg,
            gmap,
            n_founders=N_FOUNDERS,
            return_additive=False,
            cc_inbred=False,
        )
        assert dip.shape[1] == N_STATES
        np.testing.assert_allclose(dip.sum(axis=1), 1.0, atol=1e-9)


class TestBuildEmissionMatrix:
    def _make_founder_genos(self, n_markers):
        rng = np.random.default_rng(0)
        choices = ["A", "B"]
        return np.array(
            [
                [rng.choice(choices) for _ in range(N_FOUNDERS)]
                for _ in range(n_markers)
            ],
            dtype=object,
        )

    def test_shape(self):
        n = 10
        fg = self._make_founder_genos(n)
        E = build_emission_matrix(fg, None, n_founders=N_FOUNDERS)
        assert E.shape == (n, N_STATES, 3)

    def test_rows_sum_to_one(self):
        n = 5
        fg = self._make_founder_genos(n)
        E = build_emission_matrix(fg, None, n_founders=N_FOUNDERS)
        np.testing.assert_allclose(E.sum(axis=2), 1.0, atol=1e-10)

    def test_nonnegative(self):
        fg = self._make_founder_genos(5)
        E = build_emission_matrix(fg, None, n_founders=N_FOUNDERS)
        assert np.all(E >= 0)

    def test_missing_founder_uniform(self):
        # marker with all-missing founders → uniform 1/3 per obs
        fg = np.full((1, N_FOUNDERS), "-", dtype=object)
        E = build_emission_matrix(fg, None, n_founders=N_FOUNDERS)
        np.testing.assert_allclose(E[0], 1 / 3, atol=1e-10)


# ── forward-backward ──────────────────────────────────────────────────────────


def _homozygous_init(n_founders):
    states, state_to_idx = build_diplotype_states(n_founders)
    init = np.zeros(len(states))
    for f in range(n_founders):
        init[state_to_idx[(f, f)]] = 1.0 / n_founders
    return init


def _make_hmm_inputs(n_markers=10, seed=42):
    """Helper: build obs_seq, E, T_list, init for forward-backward tests."""
    rng = np.random.default_rng(seed)
    init = _homozygous_init(N_FOUNDERS)

    fg = np.array(
        [[rng.choice(["A", "B"]) for _ in range(N_FOUNDERS)] for _ in range(n_markers)],
        dtype=object,
    )
    E = build_emission_matrix(fg, None, n_founders=N_FOUNDERS)

    T_list = [transition_matrix(0.01) for _ in range(n_markers - 1)]
    obs_seq = rng.integers(0, 3, size=n_markers)
    return obs_seq, E, T_list, init


class TestForwardBackward:
    def test_gamma_shape(self):
        obs, E, T, init = _make_hmm_inputs(10)
        gamma, alpha, beta = forward_backward(obs, E, T, init)
        assert gamma.shape == (10, N_STATES)

    def test_gamma_rows_sum_to_one(self):
        obs, E, T, init = _make_hmm_inputs(10)
        gamma, _, _ = forward_backward(obs, E, T, init)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-8)

    def test_gamma_nonnegative(self):
        obs, E, T, init = _make_hmm_inputs(10)
        gamma, _, _ = forward_backward(obs, E, T, init)
        assert np.all(gamma >= -1e-12)

    def test_alpha_beta_shapes(self):
        obs, E, T, init = _make_hmm_inputs(8)
        gamma, alpha, beta = forward_backward(obs, E, T, init)
        assert alpha.shape == gamma.shape
        assert beta.shape == gamma.shape

    def test_handles_missing_obs(self):
        obs, E, T, init = _make_hmm_inputs(6)
        obs_missing = obs.copy()
        obs_missing[2] = -1  # missing observation
        gamma, _, _ = forward_backward(obs_missing, E, T, init)
        np.testing.assert_allclose(gamma.sum(axis=1), 1.0, atol=1e-8)


# ── init distributions ────────────────────────────────────────────────────────


class TestInitDistributions:
    def test_uniform_sum_to_one(self):
        init = uniform_init(N_FOUNDERS)
        assert init.sum() == pytest.approx(1.0)

    def test_uniform_equal(self):
        init = uniform_init(N_FOUNDERS)
        np.testing.assert_allclose(init, 1.0 / N_STATES)


# ── pseudomarker grid ─────────────────────────────────────────────────────────


class TestAugmentedPositions:
    def test_returns_sorted(self):
        gmap = np.array([0.0, 10.0, 20.0, 30.0])
        pos, is_real = _augmented_positions(gmap, step_cM=5.0)
        assert np.all(np.diff(pos) >= 0)

    def test_real_markers_included(self):
        gmap = np.array([0.0, 10.0, 20.0])
        pos, is_real = _augmented_positions(gmap, step_cM=5.0)
        for g in gmap:
            assert g in pos

    def test_pseudomarkers_present(self):
        gmap = np.array([0.0, 20.0])
        pos, is_real = _augmented_positions(gmap, step_cM=5.0)
        assert np.sum(~is_real) > 0

    def test_is_real_bool(self):
        gmap = np.array([0.0, 10.0])
        pos, is_real = _augmented_positions(gmap, step_cM=2.0)
        assert is_real.dtype == bool


class TestInterpolatePosteriors:
    def test_shape_matches_augmented(self):
        n_markers = 6
        obs, E, T, init = _make_hmm_inputs(n_markers, seed=7)
        gamma, alpha, beta = forward_backward(obs, E, T, init)
        gmap_cM = np.linspace(0.0, 50.0, n_markers)
        step_cM = 10.0
        gamma_all, all_pos, is_real = interpolate_posteriors(
            gamma, alpha, beta, E, obs, gmap_cM, step_cM, N_FOUNDERS
        )
        expected = len(_augmented_positions(gmap_cM, step_cM)[0])
        assert gamma_all.shape == (expected, N_STATES)

    def test_posteriors_sum_to_one(self):
        n_markers = 5
        obs, E, T, init = _make_hmm_inputs(n_markers, seed=9)
        gamma, alpha, beta = forward_backward(obs, E, T, init)
        gmap_cM = np.linspace(0.0, 40.0, n_markers)
        gamma_all, _, _ = interpolate_posteriors(
            gamma, alpha, beta, E, obs, gmap_cM, 8.0, N_FOUNDERS
        )
        np.testing.assert_allclose(gamma_all.sum(axis=1), 1.0, atol=1e-6)


# ── calc_genoprob_cc (top-level) ──────────────────────────────────────────────


class TestCalcGenoprobCC:
    def _make_inputs(self, n_markers=8, n_cc=3):
        rng = np.random.default_rng(42)
        choices = ["A", "B"]
        geno = np.array(
            [[rng.choice(choices) for _ in range(n_cc)] for _ in range(n_markers)],
            dtype=object,
        )
        fg = np.array(
            [
                [rng.choice(choices) for _ in range(N_FOUNDERS)]
                for _ in range(n_markers)
            ],
            dtype=object,
        )
        gmap = np.linspace(0.0, 0.5, n_markers)  # Morgans
        return geno, fg, gmap

    def test_shape_additive(self):
        geno, fg, gmap = self._make_inputs(8, 3)
        probs = calc_genoprob_cc(geno, fg, gmap, n_founders=N_FOUNDERS)
        assert probs.shape == (3, N_FOUNDERS, 8)

    def test_probs_sum_to_one(self):
        geno, fg, gmap = self._make_inputs(8, 3)
        probs = calc_genoprob_cc(geno, fg, gmap, n_founders=N_FOUNDERS)
        # sum over founders dim for each strain × locus
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_shape_diplotype(self):
        geno, fg, gmap = self._make_inputs(6, 2)
        probs = calc_genoprob_cc(
            geno, fg, gmap, n_founders=N_FOUNDERS, return_additive=False
        )
        assert probs.shape == (2, N_STATES, 6)

    def test_with_pseudomarkers(self):
        geno, fg, gmap = self._make_inputs(5, 2)
        result = calc_genoprob_cc(geno, fg, gmap, n_founders=N_FOUNDERS, step_cM=10.0)
        probs, aug_pos, is_real = result
        assert probs.shape[0] == 2
        assert probs.shape[1] == N_FOUNDERS
        assert probs.shape[2] == len(aug_pos)


class TestXChromosomeModel:
    """X chromosome of a fixed eight-founder-funnel line: carriers at funnel positions 1, 2, 3, 5, 6."""

    FUNNEL = np.array([3, 0, 5, 1, 7, 2, 4, 6])
    CARRIERS = FUNNEL[[0, 1, 2, 4, 5]]
    FAVOURED = FUNNEL[2]

    def test_prior_weights(self):
        prior = x_haplotype_prior(self.FUNNEL)
        assert prior.sum() == pytest.approx(1.0)
        assert prior[self.FAVOURED] == pytest.approx(1.0 / 3.0)
        others = [f for f in self.CARRIERS if f != self.FAVOURED]
        np.testing.assert_allclose(prior[others], 1.0 / 6.0)
        assert prior[self.FUNNEL[[3, 6, 7]]].sum() == 0.0

    @pytest.mark.parametrize("r", [0.0, 0.01, 0.2, 0.5])
    def test_kernel_rows_sum_to_one(self, r):
        H = x_haplotype_transition_matrix(r, self.FUNNEL)
        np.testing.assert_allclose(H.sum(axis=1), 1.0)

    @pytest.mark.parametrize("r", [0.01, 0.2, 0.5])
    def test_kernel_reversible_wrt_prior(self, r):
        H = x_haplotype_transition_matrix(r, self.FUNNEL)
        prior = x_haplotype_prior(self.FUNNEL)
        flow = prior[:, None] * H
        np.testing.assert_allclose(flow, flow.T, atol=1e-12)

    def test_kernel_limits(self):
        np.testing.assert_allclose(
            x_haplotype_transition_matrix(0.0, self.FUNNEL), np.eye(8)
        )
        H = x_haplotype_transition_matrix(0.5, self.FUNNEL)
        prior = x_haplotype_prior(self.FUNNEL)
        np.testing.assert_allclose(H[self.CARRIERS], np.tile(prior, (5, 1)))

    def test_non_carriers_get_no_mass(self):
        rng = np.random.default_rng(0)
        n_m = 12
        geno = np.array(
            [[rng.choice(["A", "B"]) for _ in range(2)] for _ in range(n_m)],
            dtype=object,
        )
        fg = np.array(
            [[rng.choice(["A", "B"]) for _ in range(8)] for _ in range(n_m)],
            dtype=object,
        )
        gmap = np.linspace(0.0, 30.0, n_m)
        probs = calc_genoprob_cc(
            geno, fg, gmap, n_founders=8, x_funnel=np.tile(self.FUNNEL, (2, 1))
        )
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-9)
        assert probs[:, self.FUNNEL[[3, 6, 7]], :].max() == 0.0

    def test_funnel_shape_checked(self):
        geno = np.array([["A"], ["B"]], dtype=object)
        fg = np.array([["A"] * 8, ["B"] * 8], dtype=object)
        with pytest.raises(ValueError):
            calc_genoprob_cc(
                geno, fg, np.array([0.0, 1.0]), n_founders=8, x_funnel=np.zeros((2, 8))
            )

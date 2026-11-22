# tests/test_hmm.py — unit tests for mosaiqtl.genotype.hmm (pure-math functions only)
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.genotype.hmm import (
    build_diplotype_states,
    diplotype_to_allele_probs,
    haldane_recomb,
    transition_matrix,
    uniform_init,
)


class TestBuildDiplotypeStates:
    def test_count_8_founders(self):
        states, _ = build_diplotype_states(8)
        assert len(states) == 36  # 8*(8+1)/2

    def test_count_2_founders(self):
        states, _ = build_diplotype_states(2)
        # (0,0),(0,1),(1,1) → 3
        assert len(states) == 3

    def test_all_ordered(self):
        states, _ = build_diplotype_states(8)
        assert all(f1 <= f2 for f1, f2 in states)

    def test_state_to_idx_invertible(self):
        states, idx = build_diplotype_states(8)
        assert all(idx[s] == i for i, s in enumerate(states))


class TestDiplotypeToAlleleProbs:
    def test_homozygous_maps_to_one(self):
        """uniform prob on (0,0) diplotype → allele 0 prob = 1.0."""
        states, idx = build_diplotype_states(8)
        dp = np.zeros(len(states))
        dp[idx[(0, 0)]] = 1.0
        ap = diplotype_to_allele_probs(dp, n_founders=8)
        np.testing.assert_allclose(ap[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(ap[1:], 0.0, atol=1e-10)

    def test_heterozygous_splits_half(self):
        states, idx = build_diplotype_states(8)
        dp = np.zeros(len(states))
        dp[idx[(0, 1)]] = 1.0
        ap = diplotype_to_allele_probs(dp, n_founders=8)
        np.testing.assert_allclose(ap[0], 0.5, atol=1e-10)
        np.testing.assert_allclose(ap[1], 0.5, atol=1e-10)
        np.testing.assert_allclose(ap[2:], 0.0, atol=1e-10)

    def test_output_sums_to_one(self):
        states, _ = build_diplotype_states(8)
        dp = np.random.dirichlet(np.ones(len(states)))
        ap = diplotype_to_allele_probs(dp, n_founders=8)
        np.testing.assert_allclose(ap.sum(), 1.0, atol=1e-10)

    def test_batched_shape(self):
        n_l, n_s, n_states = 5, 10, 36
        dp = np.random.dirichlet(np.ones(n_states), size=(n_l, n_s))
        ap = diplotype_to_allele_probs(dp, n_founders=8)
        assert ap.shape == (n_l, n_s, 8)


class TestHaldaneRecomb:
    def test_zero_distance(self):
        assert haldane_recomb(0.0) == 0.0

    def test_max_half(self):
        """r → 0.5 as d → ∞."""
        assert haldane_recomb(100.0) == pytest.approx(0.5, abs=1e-6)

    def test_symmetric(self):
        assert haldane_recomb(0.1) == pytest.approx(haldane_recomb(-0.1), abs=1e-12)

    def test_known_value(self):
        # r = 0.5*(1 - exp(-0.2)) ≈ 0.09063
        np.testing.assert_allclose(
            haldane_recomb(0.1), 0.5 * (1 - np.exp(-0.2)), rtol=1e-10
        )


class TestTransitionMatrix:
    def setup_method(self):
        self.r = 0.1
        self.T = transition_matrix(self.r, n_founders=8)
        self.ns = 36

    def test_shape(self):
        assert self.T.shape == (self.ns, self.ns)

    def test_row_sums_one(self):
        np.testing.assert_allclose(self.T.sum(axis=1), 1.0, atol=1e-10)

    def test_nonnegative(self):
        assert np.all(self.T >= 0)

    def test_identity_at_r0(self):
        """r=0 → no recombination → transition matrix = identity."""
        T0 = transition_matrix(0.0, n_founders=8)
        np.testing.assert_allclose(T0, np.eye(self.ns), atol=1e-10)


class TestUniformInit:
    def test_shape(self):
        p = uniform_init(8)
        assert p.shape == (36,)

    def test_sums_to_one(self):
        np.testing.assert_allclose(uniform_init(8).sum(), 1.0, atol=1e-10)

    def test_uniform(self):
        p = uniform_init(8)
        np.testing.assert_allclose(p, p[0], atol=1e-10)

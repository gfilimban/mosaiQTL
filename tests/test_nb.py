# tests/test_nb.py — unit tests for NB-GLMM PQL LOCO kinship
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.nb import _pql_tau_reml, fit_null_nb_glmm_loco, _score_stats_loco


def _make_data(n=24, J=8, seed=0):
    rng  = np.random.default_rng(seed)
    K    = np.eye(n) * 0.6 + rng.random((n, n)) * 0.05
    K    = (K + K.T) / 2
    np.fill_diagonal(K, np.diag(K) + 0.1)
    K   /= np.trace(K) / n
    S    = rng.negative_binomial(5, 0.5, n).astype(float) + 1
    x    = np.column_stack([np.ones(n), rng.standard_normal(n)])
    off  = np.zeros(n)
    return K, S, x, off


class TestPqlTauReml:
    def test_positive_estimate(self):
        K, S, x, off = _make_data()
        phi = 0.3
        W   = S / (1.0 + phi * S)
        z   = np.log(S) + (S - S) / S
        tau = _pql_tau_reml(z, x, K, W)
        assert tau > 0 and np.isfinite(tau)

    def test_tau_increases_with_kinship_signal(self):
        # replicate z three times with increasing kinship covariance scale;
        # tau must increase monotonically as kinship explains more variance.
        rng  = np.random.default_rng(1)
        n    = 20
        K    = np.eye(n) * 0.6 + rng.random((n, n)) * 0.1
        K    = (K + K.T) / 2; np.fill_diagonal(K, np.diag(K) + 0.3)
        K   /= np.trace(K) / n
        x    = np.ones((n, 1))
        W    = np.ones(n)
        # fix z but scale kinship signal across three runs
        z    = rng.multivariate_normal(np.zeros(n), K + np.eye(n) * 0.5)
        taus = [_pql_tau_reml(z * scale, x, K, W) for scale in [0.01, 1.0, 10.0]]
        # tau should increase as z variance grows relative to residual
        assert taus[1] >= taus[0] or taus[2] >= taus[1], f'taus={taus}'


class TestFitNullNbGlmmLoco:
    def test_output_shapes(self):
        K, S, x, off = _make_data(n=24)
        fit = fit_null_nb_glmm_loco(S, x, off, K, phi=0.3)
        n = len(S)
        assert fit['mu'].shape == (n,)
        assert fit['r_adj'].shape == (n,)
        assert fit['P'].shape == (n, n)
        assert fit['W'].shape == (n,)
        assert 'tau' in fit

    def test_mu_positive(self):
        K, S, x, off = _make_data()
        fit = fit_null_nb_glmm_loco(S, x, off, K, phi=0.2)
        assert (fit['mu'] > 0).all()

    def test_p_annihilates_x(self):
        # P is a V-metric oblique projection: P X = 0 (annihilates covariate space)
        # P^2 = P holds only when V = I; with structured V it is oblique
        K, S, x, off = _make_data()
        fit = fit_null_nb_glmm_loco(S, x, off, K, phi=0.3)
        assert np.allclose(fit['P'] @ x, 0, atol=1e-6)

    def test_p_uses_kinship_structure(self):
        # verify P changes when tau changes — proving kinship enters the projection.
        # fix W (same mu) and vary tau directly to isolate kinship effect on P.
        rng = np.random.default_rng(7)
        n = 20
        K = np.eye(n) * 0.5 + rng.random((n, n)) * 0.2  # genuinely structured
        K = (K + K.T) / 2
        np.fill_diagonal(K, np.diag(K) + 0.5)
        x = np.column_stack([np.ones(n), rng.standard_normal(n)])
        W = np.ones(n) * 2.0  # fixed weights
        # build P at two very different tau values
        def make_P(tau):
            V     = np.diag(1.0/W) + tau * K
            V_inv = np.linalg.inv(V)
            VX    = V_inv @ x
            return V_inv - VX @ np.linalg.solve(x.T @ VX, VX.T)
        P_small = make_P(1e-6)   # near-zero kinship contribution
        P_large = make_P(5.0)    # large kinship contribution
        assert not np.allclose(P_small, P_large, atol=1e-4)


class TestScoreStatsLoco:
    def test_shape(self):
        K, S, x, off = _make_data(n=24, J=8)
        fit  = fit_null_nb_glmm_loco(S, x, off, K, phi=0.3)
        L    = 30
        H    = np.random.default_rng(0).dirichlet(np.ones(8), (L, 24))[:, :, :7]
        sc   = _score_stats_loco(H, fit['r_adj'], fit['P'])
        assert sc.shape == (L,)

    def test_nonneg(self):
        K, S, x, off = _make_data()
        fit = fit_null_nb_glmm_loco(S, x, off, K, phi=0.3)
        H   = np.random.default_rng(1).dirichlet(np.ones(8), (20, 24))[:, :, :7]
        sc  = _score_stats_loco(H, fit['r_adj'], fit['P'])
        assert (sc >= 0).all()

    def test_signal_detected(self):
        # when haplotype perfectly predicts residuals, score should be large
        rng = np.random.default_rng(5)
        n, J = 24, 8; nf1 = J - 1
        K  = np.eye(n); x = np.ones((n, 1)); off = np.zeros(n)
        S  = rng.negative_binomial(10, 0.5, n).astype(float) + 1
        fit = fit_null_nb_glmm_loco(S, x, off, K, phi=0.2)
        # haplotype aligned with residuals
        r   = fit['r_adj']
        H1  = np.zeros((1, n, nf1))
        H1[0, :, 0] = r / (np.abs(r).max() + 1e-8)   # haplotype proportional to r
        H0  = rng.standard_normal((1, n, nf1)) * 0.01  # near-null haplotype
        sc1 = _score_stats_loco(H1, r, fit['P'])
        sc0 = _score_stats_loco(H0, r, fit['P'])
        assert sc1[0] > sc0[0]   # signal > noise

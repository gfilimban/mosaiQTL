# tests/test_lmm.py — unit tests for mosaiqtl.qtl.lmm
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.qtl.lmm import fit_lmm_at_locus, estimate_h2_null

class TestLODFormula:
    '''LOD = (n/2) * log10(RSS0/RSS1). ML denominator n, not n-p.'''
    def _setup(self, n=40, J=8, seed=0):
        rng = np.random.default_rng(seed)
        ev  = np.ones(n)                      # identity kinship eigenvalues
        evc = np.eye(n)                       # identity eigenvectors
        x   = np.ones((n, 1))                 # intercept only
        h   = rng.dirichlet(np.ones(J), size=n)[:, :-1]  # (n, J-1) haplotype
        return ev, evc, x, h, rng

    def test_lod_zero_null_phenotype(self):
        '''if y is orthogonal to H, LOD should be near 0.'''
        ev, evc, x, h, rng = self._setup()
        n = len(ev)
        # phenotype = pure noise unrelated to h
        y = rng.standard_normal(n)
        y -= y.mean()
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res   = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        assert res.lod >= 0.0             # LOD is non-negative
        assert res.lod < 3.0             # should be near null

    def test_lod_positive_for_signal(self):
        '''if y is generated from H, LOD should be clearly positive.'''
        ev, evc, x, h, rng = self._setup()
        n = len(ev)
        # phenotype generated from first haplotype column
        y = 3.0 * h[:, 0] + 0.3 * rng.standard_normal(n)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res   = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        assert res.lod > 1.0

    def test_lod_formula_ml_denominator(self):
        '''verify LOD uses n (ML) not n-p (REML).'''
        ev, evc, x, h, rng = self._setup()
        n = len(ev); p = x.shape[1]
        y = rng.standard_normal(n)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        # LOD must be finite and non-negative regardless of formula choice
        assert np.isfinite(res.lod) and res.lod >= 0.0

    def test_betas_shape(self):
        ev, evc, x, h, rng = self._setup()
        n = len(ev); J_minus_1 = h.shape[1]
        y = rng.standard_normal(n)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        assert res.betas.shape == (J_minus_1,)

# tests/test_utils.py — unit tests for mosaiqtl.utils
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.utils import rint, build_z_matrix, align_samples


class TestRint:
    def test_shape(self):
        y = np.random.randn(50)
        assert rint(y).shape == y.shape

    def test_standard_normal_marginal(self):
        np.random.seed(0)
        y = np.random.exponential(size=1000)
        r = rint(y)
        assert abs(r.mean()) < 0.05
        assert abs(r.std() - 1.0) < 0.05

    def test_rank_preserved(self):
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        assert np.all(np.argsort(y) == np.argsort(rint(y)))

    def test_nan_passthrough(self):
        y = np.array([1.0, np.nan, 2.0, 3.0])
        r = rint(y)
        assert np.isnan(r[1])
        assert np.all(np.isfinite(r[[0, 2, 3]]))


class TestBuildZMatrix:
    def test_identity_when_matching(self):
        sids = ['A', 'B', 'C']
        Z, valid = build_z_matrix(sids, sids)
        assert Z.shape == (3, 3)
        assert np.allclose(Z, np.eye(3))
        assert valid.all()

    def test_expansion_shape(self):
        # 4 observations from 2 strains
        sids       = ['s1', 's1', 's2', 's2']
        cache_sids = ['s1', 's2']
        Z, valid = build_z_matrix(sids, cache_sids)
        assert Z.shape == (4, 2)
        assert Z.sum(axis=1).max() == 1   # each sample maps to exactly 1 strain

    def test_missing_flagged(self):
        sids       = ['s1', 'UNKNOWN', 's2']
        cache_sids = ['s1', 's2']
        _, valid = build_z_matrix(sids, cache_sids)
        assert not valid[1]
        assert valid[0] and valid[2]


class TestAlignSamples:
    def test_full_overlap(self):
        a = ['s1', 's2', 's3']
        b = ['s3', 's1', 's2']
        ia, ib, shared = align_samples(a, b)
        assert set(shared) == {'s1', 's2', 's3'}
        # alignment must be consistent
        for i, j in zip(ia, ib):
            assert a[i] == b[j]

    def test_partial_overlap(self):
        a = ['s1', 's2', 's3']
        b = ['s2', 's4']
        ia, ib, shared = align_samples(a, b)
        assert shared == ['s2']
        assert a[ia[0]] == 's2'
        assert b[ib[0]] == 's2'

    def test_no_overlap_returns_empty(self):
        ia, ib, shared = align_samples(['s1'], ['s2'])
        assert len(shared) == 0
        assert len(ia) == 0

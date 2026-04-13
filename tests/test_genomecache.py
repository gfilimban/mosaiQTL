# tests/test_genomecache.py — unit tests for genomecache.GenomeCache
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.genomecache import GenomeCache


def _make_cache(n_samples=20, n_loci=10, n_founders=8, chroms=('1', '2')):
    '''build minimal in-memory GenomeCache without disk I/O.'''
    rng = np.random.default_rng(0)
    data = {}
    for c in chroms:
        lids  = np.array([f'{c}_L{i}' for i in range(n_loci)])
        pos   = np.linspace(1.0, 100.0, n_loci)
        probs = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_samples))
        data[c] = {'locus_ids': lids, 'positions': pos, 'probs': probs}
    founders   = [f'F{i}' for i in range(n_founders)]
    sample_ids = [f's{i}' for i in range(n_samples)]
    return GenomeCache(data=data, sample_ids=sample_ids, founders=founders)


class TestGenomeCacheAttributes:
    def test_chromosomes_sorted(self):
        c = _make_cache(chroms=('2', '1'))
        assert c.chromosomes == ['1', '2']

    def test_sample_ids_length(self):
        c = _make_cache(n_samples=30)
        assert len(c.sample_ids) == 30

    def test_n_loci(self):
        c = _make_cache(n_loci=15)
        assert c.n_loci('1') == 15


class TestGenomeCacheGetChromosome:
    def setup_method(self):
        self.c = _make_cache(n_samples=20, n_loci=10, n_founders=8)

    def test_shapes(self):
        lids, pos, probs = self.c.get_chromosome('1')
        assert lids.shape  == (10,)
        assert pos.shape   == (10,)
        assert probs.shape == (10, 20, 8)

    def test_positions_ascending(self):
        _, pos, _ = self.c.get_chromosome('1')
        assert np.all(np.diff(pos) > 0)

    def test_probs_nonnegative(self):
        _, _, probs = self.c.get_chromosome('1')
        assert np.all(probs >= 0)

    def test_probs_rowsum_one(self):
        _, _, probs = self.c.get_chromosome('1')
        np.testing.assert_allclose(probs.sum(axis=2), 1.0, atol=1e-10)

    def test_missing_chrom_raises(self):
        with pytest.raises((KeyError, ValueError)):
            self.c.get_chromosome('99')


class TestGenomeCacheGetLocus:
    def setup_method(self):
        self.c = _make_cache(n_samples=20, n_loci=10, n_founders=8)

    def test_shape(self):
        h = self.c.get_locus('1', 0)
        assert h.shape == (20, 8)

    def test_probs_finite(self):
        h = self.c.get_locus('1', 5)
        assert np.all(np.isfinite(h))

    def test_consistent_with_get_chromosome(self):
        _, _, probs = self.c.get_chromosome('1')
        h = self.c.get_locus('1', 3)
        np.testing.assert_array_equal(h, probs[3])

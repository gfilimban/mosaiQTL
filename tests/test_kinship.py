# tests/test_kinship.py — unit tests for mosaiqtl.genotype.kinship
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship


def _make_cache(n_strains=20, n_loci=10, n_founders=8, chroms=('1', '2', '3')):
    rng  = np.random.default_rng(1)
    data = {}
    for c in chroms:
        probs = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_strains))
        data[c] = {'locus_ids': np.array([f'{c}_{i}' for i in range(n_loci)]),
                   'positions': np.linspace(1.0, 100.0, n_loci),
                   'probs': probs}
    return GenomeCache(data=data,
                       sample_ids=[f's{i}' for i in range(n_strains)],
                       founders=[f'F{i}' for i in range(n_founders)])


class TestCalcKinship:
    def setup_method(self):
        self.cache = _make_cache()
        self.n = 20

    def test_overall_shape(self):
        k = calc_kinship(self.cache, method='overall')
        assert k['overall'].shape == (self.n, self.n)

    def test_overall_symmetric(self):
        k = calc_kinship(self.cache, method='overall')['overall']
        np.testing.assert_allclose(k, k.T, atol=1e-10)

    def test_loco_keys_equal_chroms(self):
        k = calc_kinship(self.cache, method='loco')
        assert set(k.keys()) == set(self.cache.chromosomes)

    def test_loco_excludes_self_chr(self):
        '''loco kinship for chr1 must differ from overall — chr1 contribution excluded.'''
        k_loco    = calc_kinship(self.cache, method='loco')
        k_overall = calc_kinship(self.cache, method='overall')['overall']
        assert not np.allclose(k_loco['1'], k_overall)

    def test_chr_method_shape(self):
        k = calc_kinship(self.cache, method='chr')
        assert all(v.shape == (self.n, self.n) for v in k.values())

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            calc_kinship(self.cache, method='bad')


class TestDecompKinship:
    def setup_method(self):
        cache = _make_cache()
        self.k_dict = calc_kinship(cache, method='loco')

    def test_returns_values_and_vectors(self):
        decomp = decomp_kinship(self.k_dict)
        for v in decomp.values():
            assert 'values' in v and 'vectors' in v

    def test_eigenvalues_nonnegative(self):
        '''floor applied to negative noise eigenvalues.'''
        decomp = decomp_kinship(self.k_dict)
        for v in decomp.values():
            assert np.all(v['values'] >= 0)

    def test_eigenvalues_descending(self):
        decomp = decomp_kinship(self.k_dict)
        for v in decomp.values():
            assert np.all(np.diff(v['values']) <= 1e-10)

    def test_reconstruction(self):
        '''K ≈ U diag(λ) U^T for the first chromosome.'''
        cache  = _make_cache(chroms=('1',))
        k_dict = calc_kinship(cache, method='overall')
        decomp = decomp_kinship(k_dict)
        d      = decomp['overall']
        k_rec  = d['vectors'] @ np.diag(d['values']) @ d['vectors'].T
        np.testing.assert_allclose(k_rec, k_dict['overall'], atol=1e-8)

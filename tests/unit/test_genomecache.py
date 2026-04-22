"""Unit tests for genotype/genomecache.py — GenomeCache in-memory API."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.genotype.genomecache import (
    GenomeCache, CC_FOUNDERS, pheno_df, _save_group_probs, _save_kinship,
)
from mosaiqtl.genotype.founders import FOUNDER_ORDER

N_FOUNDERS = 8
N_LOCI     = 15
N_STRAINS  = 10
CHROMS     = ['1', '2']


# ── construction ──────────────────────────────────────────────────────────────

class TestGenomeCacheConstruction:
    def test_chromosomes_sorted(self, tiny_cache):
        assert tiny_cache.chromosomes == sorted(CHROMS)

    def test_sample_ids_shape(self, tiny_cache, strain_ids):
        assert len(tiny_cache.sample_ids) == N_STRAINS
        np.testing.assert_array_equal(tiny_cache.sample_ids, strain_ids)

    def test_founders_correct(self, tiny_cache):
        assert tiny_cache.founders == FOUNDER_ORDER

    def test_n_founders(self, tiny_cache):
        assert tiny_cache.n_founders == N_FOUNDERS

    def test_empty_data(self, strain_ids):
        gc = GenomeCache(strain_ids, FOUNDER_ORDER, data=None)
        assert gc.chromosomes == []

    def test_data_dict_stored(self, tiny_cache):
        assert len(tiny_cache._data) == len(CHROMS)


# ── get_chromosome ────────────────────────────────────────────────────────────

class TestGetChromosome:
    def test_returns_three_arrays(self, tiny_cache):
        result = tiny_cache.get_chromosome('1')
        assert len(result) == 3

    def test_locus_ids_shape(self, tiny_cache):
        lids, pos, probs = tiny_cache.get_chromosome('1')
        assert lids.shape == (N_LOCI,)

    def test_positions_shape(self, tiny_cache):
        lids, pos, probs = tiny_cache.get_chromosome('1')
        assert pos.shape == (N_LOCI,)

    def test_probs_shape(self, tiny_cache):
        lids, pos, probs = tiny_cache.get_chromosome('1')
        assert probs.shape == (N_LOCI, N_STRAINS, N_FOUNDERS)

    def test_positions_ordered(self, tiny_cache):
        _, pos, _ = tiny_cache.get_chromosome('1')
        assert np.all(np.diff(pos) >= 0)

    def test_probs_sum_to_one(self, tiny_cache):
        _, _, probs = tiny_cache.get_chromosome('1')
        np.testing.assert_allclose(probs.sum(axis=2), 1.0, atol=1e-6)

    def test_probs_nonnegative(self, tiny_cache):
        _, _, probs = tiny_cache.get_chromosome('1')
        assert np.all(probs >= -1e-10)


# ── get_locus ─────────────────────────────────────────────────────────────────

class TestGetLocus:
    def test_shape(self, tiny_cache):
        p = tiny_cache.get_locus('1', 0)
        assert p.shape == (N_STRAINS, N_FOUNDERS)

    def test_sums_to_one(self, tiny_cache):
        p = tiny_cache.get_locus('1', 3)
        np.testing.assert_allclose(p.sum(axis=1), 1.0, atol=1e-6)

    def test_last_locus(self, tiny_cache):
        p = tiny_cache.get_locus('1', N_LOCI - 1)
        assert p.shape == (N_STRAINS, N_FOUNDERS)


# ── n_loci ────────────────────────────────────────────────────────────────────

class TestNLoci:
    def test_correct_count(self, tiny_cache):
        assert tiny_cache.n_loci('1') == N_LOCI

    def test_both_chroms(self, tiny_cache):
        for c in CHROMS:
            assert tiny_cache.n_loci(c) == N_LOCI


# ── locus IDs ─────────────────────────────────────────────────────────────────

class TestLocusIds:
    def test_locus_id_format(self, tiny_cache):
        lids, _, _ = tiny_cache.get_chromosome('1')
        for lid in lids:
            assert lid.startswith('1_')

    def test_unique_locus_ids(self, tiny_cache):
        for c in CHROMS:
            lids, _, _ = tiny_cache.get_chromosome(c)
            assert len(set(lids)) == len(lids)


# ── two-chromosome consistency ────────────────────────────────────────────────

class TestMultiChrom:
    def test_probs_independent_per_chrom(self, tiny_cache):
        _, _, p1 = tiny_cache.get_chromosome('1')
        _, _, p2 = tiny_cache.get_chromosome('2')
        # should not be identical (seeded differently)
        assert not np.allclose(p1, p2)

    def test_sample_ids_consistent(self, tiny_cache):
        for c in CHROMS:
            _, _, probs = tiny_cache.get_chromosome(c)
            assert probs.shape[1] == len(tiny_cache.sample_ids)


# ── HDF5 roundtrip ────────────────────────────────────────────────────────────

class TestHdf5Roundtrip:
    def test_to_and_from_hdf5(self, tiny_cache, tmp_path):
        h5_path = str(tmp_path / 'test_cache.h5')
        tiny_cache.to_hdf5(h5_path)
        loaded = GenomeCache.from_hdf5(h5_path)

        assert loaded.chromosomes == tiny_cache.chromosomes
        assert list(loaded.founders) == list(tiny_cache.founders)
        np.testing.assert_array_equal(loaded.sample_ids, tiny_cache.sample_ids)

    def test_probs_preserved(self, tiny_cache, tmp_path):
        h5_path = str(tmp_path / 'cache2.h5')
        tiny_cache.to_hdf5(h5_path)
        loaded = GenomeCache.from_hdf5(h5_path)
        for c in CHROMS:
            lids_orig, pos_orig, probs_orig = tiny_cache.get_chromosome(c)
            lids_load, pos_load, probs_load = loaded.get_chromosome(c)
            np.testing.assert_array_equal(lids_orig, lids_load)
            np.testing.assert_allclose(pos_orig, pos_load, atol=1e-8)
            np.testing.assert_allclose(probs_orig, probs_load, atol=1e-8)


# ── from_qtl2_probs ───────────────────────────────────────────────────────────

class TestFromQtl2Probs:
    def test_builds_cache(self, strain_ids):
        rng = np.random.default_rng(99)
        # qtl2 convention: (n_subjects, n_loci, n_founders)
        probs_dict = {
            '1': rng.dirichlet(np.ones(N_FOUNDERS),
                               size=(N_STRAINS, N_LOCI)).astype(np.float64),
            '2': rng.dirichlet(np.ones(N_FOUNDERS),
                               size=(N_STRAINS, N_LOCI)).astype(np.float64),
        }
        map_dict = {
            '1': {'locus_ids': [f'1_{i}' for i in range(N_LOCI)],
                  'positions': np.linspace(1, 100, N_LOCI)},
            '2': {'locus_ids': [f'2_{i}' for i in range(N_LOCI)],
                  'positions': np.linspace(1, 100, N_LOCI)},
        }
        gc = GenomeCache.from_qtl2_probs(probs_dict, map_dict, strain_ids)
        assert gc.chromosomes == ['1', '2']
        assert len(gc.sample_ids) == N_STRAINS

    def test_probs_transposed_correctly(self, strain_ids):
        rng = np.random.default_rng(0)
        qtl2_probs = rng.dirichlet(np.ones(N_FOUNDERS),
                                    size=(N_STRAINS, N_LOCI)).astype(np.float64)
        map_dict = {'1': {'locus_ids': [f'1_{i}' for i in range(N_LOCI)],
                          'positions': np.linspace(1, 100, N_LOCI)}}
        gc = GenomeCache.from_qtl2_probs({'1': qtl2_probs}, map_dict, strain_ids)
        _, _, probs = gc.get_chromosome('1')
        # after transposition: (n_loci, n_subjects, n_founders)
        assert probs.shape == (N_LOCI, N_STRAINS, N_FOUNDERS)

    def test_uses_cc_founders_default(self, strain_ids):
        rng = np.random.default_rng(1)
        probs_dict = {'1': rng.dirichlet(np.ones(N_FOUNDERS),
                                          size=(N_STRAINS, N_LOCI))}
        map_dict = {'1': {'locus_ids': [f'1_{i}' for i in range(N_LOCI)],
                          'positions': np.linspace(1, 100, N_LOCI)}}
        gc = GenomeCache.from_qtl2_probs(probs_dict, map_dict, strain_ids,
                                          founders=None)
        assert gc.founders == CC_FOUNDERS


# ── collapse ──────────────────────────────────────────────────────────────────

class TestCollapse:
    def test_returns_genomecache(self, tiny_cache):
        collapsed = tiny_cache.collapse(tol=0.01)
        assert isinstance(collapsed, GenomeCache)

    def test_same_founders_and_samples(self, tiny_cache):
        collapsed = tiny_cache.collapse(tol=0.01)
        assert list(collapsed.founders) == list(tiny_cache.founders)
        np.testing.assert_array_equal(collapsed.sample_ids, tiny_cache.sample_ids)

    def test_fewer_or_equal_loci(self, tiny_cache):
        collapsed = tiny_cache.collapse(tol=0.5)
        for c in tiny_cache.chromosomes:
            assert collapsed.n_loci(c) <= tiny_cache.n_loci(c)

    def test_probs_sum_to_one(self, tiny_cache):
        collapsed = tiny_cache.collapse(tol=0.1)
        for c in collapsed.chromosomes:
            _, _, probs = collapsed.get_chromosome(c)
            np.testing.assert_allclose(probs.sum(axis=2), 1.0, atol=1e-5)

    def test_euclidean_criterion(self, tiny_cache):
        collapsed = tiny_cache.collapse(tol=0.05, criterion='euclidean')
        assert isinstance(collapsed, GenomeCache)

    def test_zero_tol_preserves_all(self, tiny_cache):
        collapsed = tiny_cache.collapse(tol=0.0)
        for c in tiny_cache.chromosomes:
            assert collapsed.n_loci(c) == tiny_cache.n_loci(c)


# ── _save_group_probs ─────────────────────────────────────────────────────────

class TestSaveGroupProbs:
    def test_creates_h5_file(self, tiny_cache, tmp_path):
        filepath = str(tmp_path / 'group.h5')
        sample_list = list(tiny_cache.sample_ids[:5])
        _save_group_probs(tiny_cache, sample_list, filepath)
        assert (tmp_path / 'group.h5').exists()

    def test_h5_has_chromosomes(self, tiny_cache, tmp_path):
        import h5py
        filepath = str(tmp_path / 'group2.h5')
        _save_group_probs(tiny_cache, list(tiny_cache.sample_ids), filepath)
        with h5py.File(filepath, 'r') as f:
            for c in tiny_cache.chromosomes:
                assert f'chr{c}' in f

    def test_no_op_on_unknown_samples(self, tiny_cache, tmp_path, caplog):
        import logging
        filepath = str(tmp_path / 'empty.h5')
        with caplog.at_level(logging.WARNING):
            _save_group_probs(tiny_cache, ['NOT_A_STRAIN'], filepath)
        assert not (tmp_path / 'empty.h5').exists()


# ── _save_kinship ─────────────────────────────────────────────────────────────

class TestSaveKinship:
    def test_creates_h5_file(self, tiny_kinship, tmp_path):
        import h5py
        filepath = str(tmp_path / 'kinship.h5')
        sample_ids = [f'CC{i:03d}' for i in range(10)]
        _save_kinship(tiny_kinship, sample_ids, filepath)
        assert (tmp_path / 'kinship.h5').exists()
        with h5py.File(filepath, 'r') as f:
            for c in tiny_kinship:
                assert f'chr{c}' in f


# ── pheno_df ──────────────────────────────────────────────────────────────────

class TestPhenoDf:
    def _make_h5ad(self, tmp_path, n_obs=20, n_genes=5, seed=0):
        """Write a minimal pseudobulk AnnData to tmp_path and return the path."""
        import anndata as ad
        import pandas as pd
        rng = np.random.default_rng(seed)
        X = rng.poisson(10, size=(n_obs, n_genes)).astype(np.float32)
        strains = [f'CC{(i % 5) + 1:03d}' for i in range(n_obs)]
        obs = pd.DataFrame({'strain': strains}, index=[f's{i}' for i in range(n_obs)])
        adata = ad.AnnData(X=X, obs=obs)
        path = str(tmp_path / 'pb.h5ad')
        adata.write_h5ad(path)
        return path

    def test_returns_dataframe(self, tmp_path):
        h5ad = self._make_h5ad(tmp_path)
        df = pheno_df(h5ad)
        assert hasattr(df, 'columns')

    def test_shape(self, tmp_path):
        h5ad = self._make_h5ad(tmp_path, n_obs=20, n_genes=5)
        df = pheno_df(h5ad, min_cells=1)
        assert df.shape[1] == 5

    def test_out_path(self, tmp_path):
        h5ad = self._make_h5ad(tmp_path)
        out_csv = str(tmp_path / 'out.csv')
        pheno_df(h5ad, min_cells=1, out_path=out_csv)
        assert (tmp_path / 'out.csv').exists()

    def test_min_cells_filters(self, tmp_path):
        h5ad = self._make_h5ad(tmp_path, n_obs=5)
        # only 1 obs per strain → filtered out with min_cells=3
        df = pheno_df(h5ad, min_cells=3)
        assert df.isnull().all(axis=1).any()

    def test_missing_strain_col_raises(self, tmp_path):
        import anndata as ad
        import pandas as pd
        X = np.ones((5, 3), dtype=np.float32)
        obs = pd.DataFrame({'sample': [f's{i}' for i in range(5)]},
                           index=[f's{i}' for i in range(5)])
        adata = ad.AnnData(X=X, obs=obs)
        path = str(tmp_path / 'bad.h5ad')
        adata.write_h5ad(path)
        with pytest.raises(ValueError, match='strain_col'):
            pheno_df(path)

"""Regression tests for defects found in the production LMM path."""
import h5py
import numpy as np
import pandas as pd
import pytest
from scipy import stats

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.io.covariates import build_covariate_matrix
from mosaiqtl.io.results_io import build_eqtl_finemapping_df
from mosaiqtl.qtl import lmm as lmm_module


class TestSurvivalFunctionPvalues:
    def test_strong_signal_pvalue_is_positive_and_matches_sf(self):
        rng = np.random.default_rng(0)
        n = 60
        d = rng.standard_normal(n)
        y = 8.0 * d + 0.01 * rng.standard_normal(n)
        x = np.ones((n, 1))
        res = lmm_module.fit_lmm_at_locus(
            y, d.reshape(-1, 1), x, np.ones(n), 0.0
        )
        assert res.p_value > 0.0
        np.testing.assert_allclose(
            res.p_value, stats.f.sf(res.f_stat, 1, n - 2), rtol=1e-9
        )


class TestDosageStandardError:
    def test_se_is_consistent_with_the_f_statistic(self):
        rng = np.random.default_rng(1)
        n = 40
        d = rng.standard_normal(n)
        y = 0.4 * d + rng.standard_normal(n)
        eigvals = np.linspace(0.2, 2.0, n)
        res = lmm_module.fit_lmm_at_locus(
            y, d.reshape(-1, 1), np.ones((n, 1)), eigvals, 0.3
        )
        np.testing.assert_allclose(
            res.se_betas[0], abs(res.betas[0]) / np.sqrt(res.f_stat), rtol=1e-8
        )


class TestSingularCovariateDesign:
    def test_locus_fwl_raises_instead_of_returning_zero_effects(self):
        from mosaiqtl.qtl.effects import _locus_fwl

        rng = np.random.default_rng(2)
        n = 12
        col = rng.standard_normal((n, 1))
        x = np.hstack([col, col])
        h = rng.dirichlet(np.ones(4), size=n)
        with pytest.raises(np.linalg.LinAlgError):
            _locus_fwl(rng.standard_normal(n), x, h, np.ones(n), np.eye(n), 0.2)


class TestMixedTypeCategoricalEncoding:
    def test_mixed_type_column_does_not_raise(self):
        meta = pd.DataFrame(
            {
                'sample_id': [f's{i}' for i in range(6)],
                'grp': pd.Series(['a', 1, 'a', 1, 'a', 1], dtype=object),
            }
        )
        out = build_covariate_matrix(meta, covariates=['grp'])
        assert len(out) == 6


class TestGenomeCacheLoadChecks:
    def _write(self, tiny_cache, tmp_path):
        path = str(tmp_path / 'cache.h5')
        tiny_cache.to_hdf5(path)
        return path

    def test_sample_id_mismatch_between_chromosomes_raises(self, tiny_cache, tmp_path):
        path = self._write(tiny_cache, tmp_path)
        with h5py.File(path, 'r+') as f:
            keys = [k for k in f.keys() if k.startswith('chr')]
            grp = f[keys[-1]]
            ids = grp['sample_ids'][:]
            del grp['sample_ids']
            grp.create_dataset('sample_ids', data=ids[::-1])
        with pytest.raises(ValueError, match='sample_ids'):
            GenomeCache.from_hdf5(path)

    def test_matching_cache_loads(self, tiny_cache, tmp_path):
        loaded = GenomeCache.from_hdf5(self._write(tiny_cache, tmp_path))
        np.testing.assert_array_equal(loaded.sample_ids, tiny_cache.sample_ids)


class TestEmptyCredibleSet:
    def test_no_credible_set_members_gives_empty_frame(self):
        fm = pd.DataFrame(
            {
                'snp': ['a', 'b'],
                'pos_mb': [1.0, 2.0],
                'lod': [0.1, 0.2],
                'z': [0.1, 0.2],
                'log_abf': [0.0, 0.0],
                'pip': [0.5, 0.5],
                'in_cs': [False, False],
            }
        )
        df = build_eqtl_finemapping_df(
            fm, 'ENSMUSG0001', '8', 100_200_000, 'UNC1234', 5.2
        )
        assert df.empty

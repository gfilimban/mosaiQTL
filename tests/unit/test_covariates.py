# tests/unit/test_covariates.py — unit tests for io/covariates.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))
from mosaiqtl.io.covariates import MissingCovariateError, build_covariate_matrix


def _meta(n=6, with_sex=True, with_batch=True):
    data = {'sample_id': [f'CC{i:03d}' for i in range(n)]}
    if with_sex:
        data['sex'] = (['M', 'F'] * n)[:n]
    if with_batch:
        data['batch'] = ([1, 1, 2, 2, 3, 3] * n)[:n]
    return pd.DataFrame(data)


def _pseudobulk(meta, n_genes=50):
    rng = np.random.default_rng(0)
    counts = rng.negative_binomial(5, 0.3, size=(len(meta), n_genes)).astype(float)
    return pd.DataFrame(counts, index=meta['sample_id'].values,
                        columns=[f'gene{i}' for i in range(n_genes)])


def _kinship(meta):
    rng = np.random.default_rng(1)
    n = len(meta)
    a = rng.standard_normal((n, 8))
    k = a @ a.T / 8
    return k, meta['sample_id'].tolist()


class TestMandatoryColumn:
    def test_missing_sample_id_raises(self):
        df = pd.DataFrame({'sex': ['M', 'F']})
        with pytest.raises(ValueError, match='sample_id'):
            build_covariate_matrix(df)

    def test_sample_id_only_returns_empty(self):
        df = pd.DataFrame({'sample_id': ['CC001', 'CC002']})
        result = build_covariate_matrix(df)
        assert result.shape == (2, 0)
        assert list(result.index) == ['CC001', 'CC002']


class TestCovariates:
    def test_covariates_present(self):
        meta = _meta()
        result = build_covariate_matrix(meta, covariates=['sex', 'batch'])
        assert any(c.startswith('sex') for c in result.columns)
        assert any(c.startswith('batch') for c in result.columns)

    def test_missing_requested_col_raises(self):
        meta = _meta(with_sex=False)
        with pytest.raises(ValueError, match='sex'):
            build_covariate_matrix(meta, covariates=['sex'])

    def test_output_all_float64(self):
        meta = _meta()
        result = build_covariate_matrix(meta, covariates=['sex', 'batch'])
        assert all(result[c].dtype == np.float64 for c in result.columns)

    def test_output_no_nan(self):
        meta = _meta()
        result = build_covariate_matrix(meta, covariates=['sex', 'batch'])
        assert not result.isna().any().any()

    def test_sample_order_deterministic(self):
        meta = _meta()
        r1 = build_covariate_matrix(meta, covariates=['sex'])
        r2 = build_covariate_matrix(meta.sample(frac=1, random_state=7),
                                     covariates=['sex'])
        pd.testing.assert_frame_equal(r1, r2)

    def test_numeric_covariate_included(self):
        meta = _meta()
        meta['age'] = np.linspace(10, 20, len(meta))
        result = build_covariate_matrix(meta, covariates=['age'])
        assert 'age' in result.columns

    def test_categorical_covariate_onehot(self):
        meta = _meta()
        meta['tissue'] = ['gastroc', 'heart'] * (len(meta) // 2)
        result = build_covariate_matrix(meta, covariates=['tissue'])
        tissue_cols = [c for c in result.columns if c.startswith('tissue')]
        assert len(tissue_cols) == 1


class TestCategoricalEncoding:
    def test_categorical_onehot(self):
        meta = _meta()
        result = build_covariate_matrix(meta, covariates=['sex'])
        # sex has 2 levels → drop_first → 1 column
        sex_cols = [c for c in result.columns if c.startswith('sex')]
        assert len(sex_cols) == 1

    def test_numeric_passthrough(self):
        meta = _meta()
        result = build_covariate_matrix(meta, covariates=['batch'])
        assert 'batch' in result.columns


class TestNanHandling:
    def test_nan_raises_missing_covariate_error(self):
        meta = _meta()
        meta.loc[2, 'batch'] = np.nan
        with pytest.raises(MissingCovariateError, match='batch'):
            build_covariate_matrix(meta, covariates=['batch'])

    def test_nan_error_names_samples(self):
        meta = _meta()
        meta.loc[1, 'batch'] = np.nan
        with pytest.raises(MissingCovariateError, match='CC001'):
            build_covariate_matrix(meta, covariates=['batch'])

    def test_missing_covariate_error_is_value_error(self):
        meta = _meta()
        meta.loc[0, 'batch'] = np.nan
        with pytest.raises(ValueError):
            build_covariate_matrix(meta, covariates=['batch'])


class TestExpressionPCs:
    def test_expr_pcs_shape(self):
        meta = _meta()
        pb = _pseudobulk(meta)
        result = build_covariate_matrix(meta, n_expression_pcs=3,
                                         pseudobulk_counts=pb)
        assert result.shape == (6, 3)
        assert list(result.columns) == ['expr_pc1', 'expr_pc2', 'expr_pc3']

    def test_expr_pcs_no_pseudobulk_raises(self):
        meta = _meta()
        with pytest.raises(ValueError, match='pseudobulk_counts'):
            build_covariate_matrix(meta, n_expression_pcs=3)

    def test_expr_pcs_combined_with_covariates(self):
        meta = _meta()
        pb = _pseudobulk(meta)
        result = build_covariate_matrix(meta, covariates=['batch'],
                                         n_expression_pcs=2,
                                         pseudobulk_counts=pb)
        assert 'batch' in result.columns
        assert 'expr_pc1' in result.columns
        assert result.shape[1] == 3  # batch + 2 PCs


class TestGenotypePCs:
    def test_geno_pcs_shape(self):
        meta = _meta()
        k, ids = _kinship(meta)
        result = build_covariate_matrix(meta, n_genotype_pcs=3,
                                         kinship_matrix=k,
                                         kinship_sample_ids=ids)
        assert result.shape == (6, 3)
        assert list(result.columns) == ['geno_pc1', 'geno_pc2', 'geno_pc3']

    def test_geno_pcs_no_kinship_raises(self):
        meta = _meta()
        with pytest.raises(ValueError, match='kinship_matrix'):
            build_covariate_matrix(meta, n_genotype_pcs=3)

    def test_geno_pcs_cc_population_raises(self):
        meta = _meta()
        k, ids = _kinship(meta)
        with pytest.raises(ValueError, match='not informative'):
            build_covariate_matrix(meta, n_genotype_pcs=3,
                                    kinship_matrix=k,
                                    kinship_sample_ids=ids,
                                    population='cc')

    def test_geno_pcs_cc_error_mentions_loco(self):
        meta = _meta()
        k, ids = _kinship(meta)
        with pytest.raises(ValueError, match='LOCO'):
            build_covariate_matrix(meta, n_genotype_pcs=3,
                                    kinship_matrix=k,
                                    kinship_sample_ids=ids,
                                    population='cc')

    def test_geno_pcs_do_population_allowed(self):
        meta = _meta()
        k, ids = _kinship(meta)
        result = build_covariate_matrix(meta, n_genotype_pcs=2,
                                         kinship_matrix=k,
                                         kinship_sample_ids=ids,
                                         population='do')
        assert result.shape == (6, 2)

    def test_geno_pcs_combined_with_expr_pcs(self):
        meta = _meta()
        pb = _pseudobulk(meta)
        k, ids = _kinship(meta)
        result = build_covariate_matrix(meta, n_expression_pcs=2,
                                         n_genotype_pcs=2,
                                         pseudobulk_counts=pb,
                                         kinship_matrix=k,
                                         kinship_sample_ids=ids,
                                         population='do')
        assert result.shape == (6, 4)
        assert 'expr_pc1' in result.columns
        assert 'geno_pc1' in result.columns

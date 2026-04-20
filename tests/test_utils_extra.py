# tests/test_utils_extra.py — tests for expand_kinship, expand_probs, make_design_matrix
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.utils import expand_kinship, expand_probs, make_design_matrix


class TestExpandKinship:
    def test_output_shape(self):
        # m=5 genotype strains, n=12 samples with replication
        rng = np.random.default_rng(0)
        m, n = 5, 12
        k = rng.random((m, m)); k = k @ k.T   # PSD
        z = np.zeros((n, m))
        for i in range(n):
            z[i, i % m] = 1.0               # each sample maps to one strain
        k_n = expand_kinship(k, z)
        assert k_n.shape == (n, n)

    def test_symmetric(self):
        rng = np.random.default_rng(1)
        m, n = 4, 8
        k = rng.random((m, m)); k = (k + k.T) / 2
        z = np.eye(m)[np.arange(n) % m]
        k_n = expand_kinship(k, z)
        np.testing.assert_allclose(k_n, k_n.T, atol=1e-12)

    def test_identity_z_recovers_k(self):
        '''when z=I and m=n, result equals k.'''
        rng = np.random.default_rng(2)
        m = 6
        k = rng.random((m, m)); k = k @ k.T
        k_n = expand_kinship(k, np.eye(m))
        np.testing.assert_allclose(k_n, k, atol=1e-12)


class TestExpandProbs:
    def test_output_shape(self):
        m, n, J = 5, 12, 8
        h = np.random.dirichlet(np.ones(J), size=m)   # (m, J)
        z = np.zeros((n, m))
        for i in range(n):
            z[i, i % m] = 1.0
        h_n = expand_probs(h, z)
        assert h_n.shape == (n, J)

    def test_rowsums_preserved(self):
        '''Z is row-stochastic in this usage — rowsums of h must be preserved.'''
        m, n, J = 4, 8, 8
        h = np.random.dirichlet(np.ones(J), size=m)
        z = np.zeros((n, m))
        for i in range(n):
            z[i, i % m] = 1.0
        h_n = expand_probs(h, z)
        np.testing.assert_allclose(h_n.sum(axis=1), 1.0, atol=1e-10)

    def test_identity_z(self):
        m, J = 5, 8
        h = np.random.dirichlet(np.ones(J), size=m)
        h_n = expand_probs(h, np.eye(m))
        np.testing.assert_array_equal(h_n, h)


class TestMakeDesignMatrix:
    def test_intercept_added(self):
        df = pd.DataFrame({'age': [1.0, 2.0, 3.0]})
        x, names = make_design_matrix(df)
        assert 'intercept' in names
        np.testing.assert_array_equal(x[:, names.index('intercept')], 1.0)

    def test_numeric_col_included(self):
        df = pd.DataFrame({'age': [10.0, 20.0, 30.0], 'bmi': [22.0, 25.0, 28.0]})
        x, names = make_design_matrix(df)
        assert 'age' in names and 'bmi' in names
        assert x.shape == (3, 3)   # intercept + age + bmi

    def test_categorical_one_hot(self):
        df = pd.DataFrame({'sex': ['M', 'F', 'M', 'F']})
        x, names = make_design_matrix(df)
        # intercept + 1 dummy (drop_first=True)
        assert x.shape == (4, 2)

    def test_formula_cols_subset(self):
        df = pd.DataFrame({'age': [1.0, 2.0], 'batch': [3.0, 4.0], 'extra': [5.0, 6.0]})
        x, names = make_design_matrix(df, formula_cols=['age'])
        assert 'extra' not in names

    def test_output_float64(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        x, _ = make_design_matrix(df)
        assert x.dtype == np.float64

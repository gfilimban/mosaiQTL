"""Unit tests for mosaiqtl/utils.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.utils import (
    rint,
    align_samples,
    build_z_matrix,
    expand_kinship,
    expand_probs,
    make_design_matrix,
)


class TestRint:
    def test_output_shape(self):
        x = np.array([3.0, 1.0, 2.0])
        assert rint(x).shape == (3,)

    def test_nan_preserved(self):
        x = np.array([1.0, np.nan, 2.0])
        out = rint(x)
        assert np.isnan(out[1])

    def test_order_preserved(self):
        x = np.array([1.0, 2.0, 3.0])
        out = rint(x)
        assert out[0] < out[1] < out[2]

    def test_mean_near_zero(self):
        rng = np.random.default_rng(0)
        x = rng.standard_normal(500)
        assert abs(rint(x).mean()) < 0.1

    def test_ties_handled(self):
        x = np.array([1.0, 1.0, 2.0])
        out = rint(x)
        assert np.isfinite(out).all()
        assert out[0] == pytest.approx(out[1])


class TestAlignSamples:
    def test_shared_subset(self):
        ids_a = ["s1", "s2", "s3"]
        ids_b = ["s2", "s3", "s4"]
        ia, ib, shared = align_samples(ids_a, ids_b)
        assert shared == ["s2", "s3"]
        assert list(ia) == [1, 2]
        assert list(ib) == [0, 1]

    def test_no_overlap(self):
        ia, ib, shared = align_samples(["a"], ["b"])
        assert shared == []
        assert len(ia) == 0

    def test_full_overlap_same_order(self):
        ids = ["x", "y", "z"]
        ia, ib, shared = align_samples(ids, ids)
        assert shared == ids
        assert list(ia) == list(ib) == [0, 1, 2]


class TestBuildZMatrix:
    """Z-matrix maps n_samples (replicates) → n_strains (unique genotypes).

    Real use case: CC001_M1, CC001_M2 are two replicates of strain CC001.
    Z[CC001_M1, CC001] = Z[CC001_M2, CC001] = 1 — both rows point to same strain.
    NB Stage 2 uses Z to expand strain-level kinship to sample level: K_s = Z K_m Z'.
    LMM uses Z the same way.
    """

    def test_shape_with_replicates(self):
        # 3 replicates across 2 strains → Z is (3, 2)
        strain_ids = np.array(["CC001", "CC002"])
        sample_strain = np.array(["CC001", "CC001", "CC002"])
        Z, _ = build_z_matrix(sample_strain, strain_ids)
        assert Z.shape == (3, 2)

    def test_two_replicates_same_strain_same_column(self):
        # CC001_M1 and CC001_M2 should both activate the CC001 column
        strain_ids = np.array(["CC001", "CC002"])
        sample_strain = np.array(["CC001", "CC001", "CC002"])
        Z, _ = build_z_matrix(sample_strain, strain_ids)
        assert Z[0, 0] == 1.0  # CC001_M1 → CC001
        assert Z[1, 0] == 1.0  # CC001_M2 → CC001
        assert Z[2, 1] == 1.0  # CC002_M1 → CC002

    def test_rows_sum_to_one_for_known_samples(self):
        strain_ids = np.array(["CC001", "CC002"])
        sample_strain = np.array(["CC001", "CC001", "CC002"])
        Z, valid = build_z_matrix(sample_strain, strain_ids)
        np.testing.assert_allclose(Z[valid].sum(axis=1), 1.0)

    def test_identity_when_one_sample_per_strain(self):
        strain_ids = np.array(["CC001", "CC002", "CC003"])
        Z, _ = build_z_matrix(strain_ids, strain_ids)
        np.testing.assert_array_equal(Z, np.eye(3))

    def test_valid_mask_excludes_unknown_sample(self):
        strain_ids = np.array(["CC001", "CC002"])
        sample_strain = np.array(["CC001", "CC999"])  # CC999 not in cache
        _, valid = build_z_matrix(sample_strain, strain_ids)
        assert valid[0] and not valid[1]

    def test_kinship_expansion_shape(self):
        # NB / LMM use case: K_samples = Z @ K_strains @ Z.T
        strain_ids = np.array(["CC001", "CC002"])
        sample_strain = np.array(["CC001", "CC001", "CC002", "CC002"])
        Z, _ = build_z_matrix(sample_strain, strain_ids)
        K_strains = np.eye(2)
        K_samples = Z @ K_strains @ Z.T
        assert K_samples.shape == (4, 4)


class TestExpandKinship:
    def test_shape(self):
        rng = np.random.default_rng(0)
        m = 4
        n = 8
        K = rng.random((m, m))
        Z = np.zeros((n, m))
        Z[:, 0] = 1  # all samples map to strain 0
        K_n = expand_kinship(K, Z)
        assert K_n.shape == (n, n)

    def test_identity_passthrough(self):
        m = 3
        K = np.eye(m)
        Z = np.eye(m)
        K_n = expand_kinship(K, Z)
        np.testing.assert_allclose(K_n, np.eye(m))

    def test_replicated_samples_share_kinship(self):
        # two samples per strain, two strains → block diagonal
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        Z = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        K_n = expand_kinship(K, Z)
        # diagonal pairs (0,1) and (2,3) share kinship=1.0
        assert K_n[0, 1] == pytest.approx(1.0)
        assert K_n[2, 3] == pytest.approx(1.0)


class TestExpandProbs:
    def test_shape(self):
        rng = np.random.default_rng(1)
        m = 4
        n = 8
        J = 8
        H = rng.dirichlet(np.ones(J), size=m)
        Z = np.zeros((n, m))
        Z[np.arange(n), np.arange(n) % m] = 1
        H_n = expand_probs(H, Z)
        assert H_n.shape == (n, J)

    def test_replicated_same_probs(self):
        rng = np.random.default_rng(2)
        J = 8
        m = 3
        H = rng.dirichlet(np.ones(J), size=m)
        Z = np.array([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
        H_n = expand_probs(H, Z)
        np.testing.assert_allclose(H_n[0], H[0])
        np.testing.assert_allclose(H_n[1], H[0])  # same strain → same probs


class TestMakeDesignMatrix:
    def test_intercept_added(self):
        import pandas as pd

        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        X, names = make_design_matrix(df)
        assert "intercept" in names
        np.testing.assert_allclose(X[:, 0], 1.0)

    def test_numeric_columns_included(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        X, names = make_design_matrix(df)
        assert "a" in names and "b" in names
        assert X.shape == (2, 3)  # intercept + a + b

    def test_categorical_one_hot(self):
        import pandas as pd

        df = pd.DataFrame({"cat": ["A", "B", "A", "C"]})
        X, names = make_design_matrix(df)
        # drop_first=True → J-1 dummies; 3 levels → 2 dummies + intercept
        assert X.shape == (4, 3)

    def test_formula_cols_subset(self):
        import pandas as pd

        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
        X, names = make_design_matrix(df, formula_cols=["a"])
        assert "b" not in names and "c" not in names

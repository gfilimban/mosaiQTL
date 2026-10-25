"""Unit tests for genotype/kinship.py."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.genotype.kinship import calc_kinship


class TestCalcKinship:
    def test_loco_returns_one_matrix_per_chrom(self, tiny_cache):
        K = calc_kinship(tiny_cache, method="loco")
        assert set(K.keys()) == set(tiny_cache.chromosomes)

    def test_overall_returns_single_matrix(self, tiny_cache):
        K = calc_kinship(tiny_cache, method="overall")
        assert "overall" in K

    def test_shape_loco(self, tiny_cache):
        K = calc_kinship(tiny_cache, method="loco")
        n = len(tiny_cache.sample_ids)
        for mat in K.values():
            assert mat.shape == (n, n)

    def test_symmetric(self, tiny_cache):
        K = calc_kinship(tiny_cache, method="loco")
        for mat in K.values():
            np.testing.assert_allclose(mat, mat.T, atol=1e-10)

    def test_positive_semidefinite(self, tiny_cache):
        K = calc_kinship(tiny_cache, method="loco")
        for mat in K.values():
            eigvals = np.linalg.eigvalsh(mat)
            assert eigvals.min() >= -1e-8

    def test_loco_excludes_chromosome(self, tiny_cache):
        K_loco = calc_kinship(tiny_cache, method="loco")
        K_all = calc_kinship(tiny_cache, method="overall")["overall"]
        # LOCO matrix should differ from overall
        for mat in K_loco.values():
            assert not np.allclose(mat, K_all)

    def test_diagonal_positive(self, tiny_cache):
        K = calc_kinship(tiny_cache, method="overall")["overall"]
        assert np.all(np.diag(K) > 0)

    def test_offdiag_nonnegative(self, tiny_cache):
        # K_ij = normalized dot product of non-negative prob vectors → always >= 0
        K = calc_kinship(tiny_cache, method="overall")["overall"]
        n = len(K)
        off = K[~np.eye(n, dtype=bool)]
        assert off.min() >= -1e-10

    def test_identical_strains_have_max_offdiag(self):
        # two strains with identical haplotype probs → K_ij == K_ii
        from mosaiqtl.genotype.genomecache import GenomeCache
        from mosaiqtl.genotype.cc import FOUNDER_ORDER

        rng = np.random.default_rng(7)
        probs = rng.dirichlet(np.ones(8), size=(5, 1)).astype(np.float64)
        probs = np.tile(probs, (1, 2, 1))  # two identical strains
        data = {
            "1": {
                "locus_ids": np.array(["1_0"] * 5),
                "positions": np.ones(5),
                "probs": probs,
            }
        }
        cache = GenomeCache(np.array(["CC001", "CC002"]), FOUNDER_ORDER, data)
        K = calc_kinship(cache, method="overall")["overall"]
        np.testing.assert_allclose(K[0, 1], K[0, 0], rtol=1e-8)

    def test_loco_equals_kinship_from_remaining_chroms(self, tiny_cache):
        # K_loco[c] must equal kinship computed using only the other chromosomes
        from mosaiqtl.genotype.genomecache import GenomeCache
        from mosaiqtl.genotype.cc import FOUNDER_ORDER

        K_loco = calc_kinship(tiny_cache, method="loco")
        for held_out in tiny_cache.chromosomes:
            remaining = {
                c: tiny_cache._data[c] for c in tiny_cache.chromosomes if c != held_out
            }
            sub_cache = GenomeCache(tiny_cache.sample_ids, FOUNDER_ORDER, remaining)
            K_ref = calc_kinship(sub_cache, method="overall")["overall"]
            np.testing.assert_allclose(K_loco[held_out], K_ref, rtol=1e-8)

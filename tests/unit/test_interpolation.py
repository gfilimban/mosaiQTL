"""Unit tests for genotype/interpolation.py."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.genotype.interpolation import (
    interpolate_haplotype_probs,
    collapse_to_dosage,
    snp_allele_stats,
    _linear_interp_1d,
)

# ── minimal fake cache ─────────────────────────────────────────────────────


def _make_cache(markers_mb: np.ndarray, probs: np.ndarray, chrom: str = "1"):
    """Return a minimal GenomeCache-like object for testing."""

    class FakeCache:
        chromosomes = [chrom]

        def get_chromosome(self, c):
            return None, markers_mb.copy(), probs.copy()

    return FakeCache()


def _ramp_probs(n_loci: int, n_strains: int = 4, n_founders: int = 8) -> np.ndarray:
    """Generate valid probability arrays that vary linearly across loci."""
    rng = np.random.default_rng(42)
    raw = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_strains))
    return raw  # (n_loci, n_strains, n_founders)


# ── _linear_interp_1d ─────────────────────────────────────────────────────


class TestLinearInterp1d:
    def test_at_knots(self):
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([10.0, 20.0, 30.0])
        out = _linear_interp_1d(x, y, x)
        np.testing.assert_allclose(out, y)

    def test_midpoint(self):
        x = np.array([0.0, 2.0])
        y = np.array([0.0, 4.0])
        out = _linear_interp_1d(x, y, np.array([1.0]))
        assert out[0] == pytest.approx(2.0)

    def test_before_grid_clamped(self):
        x = np.array([1.0, 2.0])
        y = np.array([5.0, 10.0])
        out = _linear_interp_1d(x, y, np.array([0.0]))
        assert out[0] == pytest.approx(5.0)

    def test_after_grid_clamped(self):
        x = np.array([1.0, 2.0])
        y = np.array([5.0, 10.0])
        out = _linear_interp_1d(x, y, np.array([3.0]))
        assert out[0] == pytest.approx(10.0)

    def test_coincident_knots_no_nan(self):
        x = np.array([0.0, 1.0, 1.0, 2.0])
        y = np.array([0.0, 5.0, 5.0, 10.0])
        out = _linear_interp_1d(x, y, np.array([1.0]))
        assert np.isfinite(out[0])


# ── interpolate_haplotype_probs ───────────────────────────────────────────


class TestInterpolateHaplotypeProbsBoundary:
    """SNPs at/beyond marker boundaries are clamped to the nearest marker."""

    @pytest.fixture
    def three_marker_cache(self):
        mb = np.array([1.0, 2.0, 3.0])
        probs = np.zeros((3, 2, 2))
        probs[0] = [[1.0, 0.0], [0.8, 0.2]]
        probs[1] = [[0.5, 0.5], [0.4, 0.6]]
        probs[2] = [[0.0, 1.0], [0.2, 0.8]]
        return _make_cache(mb, probs), mb, probs

    def test_at_left_marker(self, three_marker_cache):
        cache, _, probs = three_marker_cache
        out = interpolate_haplotype_probs(cache, np.array([1_000_000]), "1")
        np.testing.assert_allclose(out[0], probs[0], atol=1e-12)

    def test_at_right_marker(self, three_marker_cache):
        cache, _, probs = three_marker_cache
        out = interpolate_haplotype_probs(cache, np.array([3_000_000]), "1")
        np.testing.assert_allclose(out[0], probs[2], atol=1e-12)

    def test_before_grid_clamped_to_first(self, three_marker_cache):
        cache, _, probs = three_marker_cache
        out = interpolate_haplotype_probs(cache, np.array([500_000]), "1")
        np.testing.assert_allclose(out[0], probs[0], atol=1e-12)

    def test_after_grid_clamped_to_last(self, three_marker_cache):
        cache, _, probs = three_marker_cache
        out = interpolate_haplotype_probs(cache, np.array([4_000_000]), "1")
        np.testing.assert_allclose(out[0], probs[2], atol=1e-12)

    def test_midpoint_is_mean(self, three_marker_cache):
        cache, _, probs = three_marker_cache
        out = interpolate_haplotype_probs(cache, np.array([1_500_000]), "1")
        expected = 0.5 * probs[0] + 0.5 * probs[1]
        np.testing.assert_allclose(out[0], expected, atol=1e-12)

    def test_quarter_point(self, three_marker_cache):
        cache, _, probs = three_marker_cache
        # 1.25 Mb → t = 0.25
        out = interpolate_haplotype_probs(cache, np.array([1_250_000]), "1")
        expected = 0.75 * probs[0] + 0.25 * probs[1]
        np.testing.assert_allclose(out[0], expected, atol=1e-12)


class TestInterpolateOutputShape:
    def test_shape(self):
        n_loci, n_strains, n_founders, n_snps = 10, 4, 8, 25
        mb = np.linspace(1.0, 10.0, n_loci)
        probs = _ramp_probs(n_loci, n_strains, n_founders)
        cache = _make_cache(mb, probs)
        snp_bp = np.random.default_rng(0).integers(1_000_000, 10_000_000, n_snps)
        out = interpolate_haplotype_probs(cache, snp_bp, "1")
        assert out.shape == (n_snps, n_strains, n_founders)

    def test_dtype_float64(self):
        mb = np.array([1.0, 2.0])
        probs = np.ones((2, 2, 2)) * 0.5
        cache = _make_cache(mb, probs)
        out = interpolate_haplotype_probs(cache, np.array([1_500_000]), "1")
        assert out.dtype == np.float64


class TestInterpolateSimplexPreserved:
    """Interpolated probabilities must stay on the probability simplex."""

    def test_simplex_sum(self):
        n_loci, n_strains, n_founders = 20, 10, 8
        mb = np.linspace(0.5, 20.0, n_loci)
        probs = _ramp_probs(n_loci, n_strains, n_founders)
        cache = _make_cache(mb, probs)
        rng = np.random.default_rng(7)
        snp_bp = rng.integers(500_000, 20_000_000, 200)
        out = interpolate_haplotype_probs(cache, snp_bp, "1")
        np.testing.assert_allclose(out.sum(axis=2), 1.0, atol=1e-10)

    def test_simplex_non_negative(self):
        n_loci, n_strains, n_founders = 20, 10, 8
        mb = np.linspace(0.5, 20.0, n_loci)
        probs = _ramp_probs(n_loci, n_strains, n_founders)
        cache = _make_cache(mb, probs)
        rng = np.random.default_rng(13)
        snp_bp = rng.integers(500_000, 20_000_000, 200)
        out = interpolate_haplotype_probs(cache, snp_bp, "1")
        assert np.all(out >= -1e-14)


class TestInterpolateGeneticMap:
    """gmap= path: interpolation in cM space."""

    def test_gmap_shape_mismatch_raises(self):
        mb = np.array([1.0, 2.0, 3.0])
        probs = np.ones((3, 2, 2)) * 0.5
        cache = _make_cache(mb, probs)
        bad_gmap = np.array([[1.0, 2.0], [0.0, 5.0]])  # only 2 columns, need 3
        with pytest.raises(ValueError, match="gmap shape"):
            interpolate_haplotype_probs(
                cache, np.array([1_500_000]), "1", gmap=bad_gmap
            )

    def test_gmap_midpoint_matches_physical_for_uniform_map(self):
        # uniform genetic map (equal spacing) → gmap result equals physical
        mb = np.array([1.0, 2.0, 3.0])
        probs = np.zeros((3, 2, 2))
        probs[0] = [[1.0, 0.0], [0.9, 0.1]]
        probs[1] = [[0.5, 0.5], [0.5, 0.5]]
        probs[2] = [[0.0, 1.0], [0.1, 0.9]]
        cache = _make_cache(mb, probs)
        # genetic map with same relative spacing as physical
        gmap = np.array([mb, np.array([0.0, 1.0, 2.0])])  # (2, 3)
        out_phys = interpolate_haplotype_probs(cache, np.array([1_500_000]), "1")
        out_gmap = interpolate_haplotype_probs(
            cache, np.array([1_500_000]), "1", gmap=gmap
        )
        np.testing.assert_allclose(out_phys, out_gmap, atol=1e-10)


class TestInterpolateErrors:
    def test_unknown_chrom_raises(self):
        mb = np.array([1.0, 2.0])
        probs = np.ones((2, 2, 2)) * 0.5
        cache = _make_cache(mb, probs, chrom="1")
        with pytest.raises(ValueError, match="not found in cache"):
            interpolate_haplotype_probs(cache, np.array([1_500_000]), "X")


# ── collapse_to_dosage ────────────────────────────────────────────────────────


def _simplex_probs(n_snps: int, n_strains: int, n_founders: int) -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.dirichlet(np.ones(n_founders), size=(n_snps, n_strains))


class TestCollapseToDosage:
    def test_shape(self):
        probs = _simplex_probs(10, 4, 3)
        alt_mask = np.zeros((10, 3))
        alt_mask[:, 0] = 1.0
        out = collapse_to_dosage(probs, alt_mask)
        assert out.shape == (10, 4)

    def test_single_alt_founder_equals_prob_slice(self):
        probs = _simplex_probs(5, 6, 3)
        alt_mask = np.zeros((5, 3))
        alt_mask[:, 2] = 1.0
        out = collapse_to_dosage(probs, alt_mask)
        np.testing.assert_allclose(out, probs[:, :, 2], atol=1e-14)

    def test_all_alt_founders_gives_one(self):
        probs = _simplex_probs(4, 5, 8)
        alt_mask = np.ones((4, 8))
        out = collapse_to_dosage(probs, alt_mask)
        np.testing.assert_allclose(out, 1.0, atol=1e-14)

    def test_no_alt_founders_gives_zero(self):
        probs = _simplex_probs(4, 5, 8)
        alt_mask = np.zeros((4, 8))
        out = collapse_to_dosage(probs, alt_mask)
        np.testing.assert_allclose(out, 0.0, atol=1e-14)

    def test_dosage_in_unit_interval(self):
        probs = _simplex_probs(20, 10, 8)
        rng = np.random.default_rng(3)
        alt_mask = rng.integers(0, 2, size=(20, 8)).astype(np.float64)
        out = collapse_to_dosage(probs, alt_mask)
        assert np.all(out >= -1e-14) and np.all(out <= 1.0 + 1e-14)

    def test_shape_mismatch_raises(self):
        probs = _simplex_probs(5, 4, 3)
        bad_mask = np.zeros((5, 4))  # wrong n_founders
        with pytest.raises(ValueError, match="alt_mask shape"):
            collapse_to_dosage(probs, bad_mask)

    def test_snp_mismatch_raises(self):
        probs = _simplex_probs(5, 4, 3)
        bad_mask = np.zeros((6, 3))  # wrong n_snps
        with pytest.raises(ValueError, match="alt_mask shape"):
            collapse_to_dosage(probs, bad_mask)


# ── snp_allele_stats ──────────────────────────────────────────────────────────


class TestSnpAlleleStats:
    def test_uniform_dosage(self):
        dosage = np.full((5, 10), 0.4)
        af, maf, mac = snp_allele_stats(dosage, n_samples=100)
        np.testing.assert_allclose(af, 0.4, atol=1e-14)
        np.testing.assert_allclose(maf, 0.4, atol=1e-14)
        assert mac[0] == 40

    def test_maf_is_minor(self):
        # af > 0.5 → maf = 1 - af
        dosage = np.full((3, 8), 0.8)
        af, maf, _ = snp_allele_stats(dosage)
        np.testing.assert_allclose(maf, 0.2, atol=1e-14)

    def test_n_samples_default_is_n_strains(self):
        dosage = np.full((2, 10), 0.5)
        _, _, mac = snp_allele_stats(dosage)  # n_samples defaults to 10
        assert mac[0] == 5

    def test_output_shapes(self):
        dosage = _simplex_probs(7, 12, 4).sum(axis=2) / 4  # (7, 12) in [0,1]
        af, maf, mac = snp_allele_stats(dosage, n_samples=200)
        assert af.shape == maf.shape == mac.shape == (7,)

    def test_mac_dtype_is_int(self):
        dosage = np.full((3, 6), 0.3)
        _, _, mac = snp_allele_stats(dosage, n_samples=50)
        assert mac.dtype == np.int64

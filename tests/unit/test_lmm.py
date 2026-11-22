"""Unit tests for qtl/lmm.py — Gaussian LMM validated port."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.lmm import (
    estimate_h2_null,
    h2_at_boundary,
    precompute_chromosome,
    scan_chromosome,
)
from mosaiqtl.qtl.scan import scan_h2lmm


class TestEstimateH2Null:
    def test_returns_float_in_unit_interval(
        self, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        K = tiny_kinship["1"]
        eigvals, eigvecs = np.linalg.eigh(K)
        y_rot = eigvecs.T @ tiny_strain_phenotype
        x_rot = eigvecs.T @ tiny_strain_covariates
        h2, _ = estimate_h2_null(y_rot, x_rot, eigvals)
        assert 0.0 <= h2 <= 1.0

    def test_reml_and_ml_both_return(
        self, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        K = tiny_kinship["1"]
        eigvals, eigvecs = np.linalg.eigh(K)
        y_rot = eigvecs.T @ tiny_strain_phenotype
        x_rot = eigvecs.T @ tiny_strain_covariates
        h2_reml, ll_reml = estimate_h2_null(y_rot, x_rot, eigvals, method="reml")
        h2_ml, ll_ml = estimate_h2_null(y_rot, x_rot, eigvals, method="ml")
        assert np.isfinite(h2_reml) and np.isfinite(h2_ml)
        assert np.isfinite(ll_reml) and np.isfinite(ll_ml)


class TestScanChromosome:
    def test_returns_dict_with_required_keys(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        for key in ("locus_ids", "positions", "lod", "p_value", "betas"):
            assert key in result

    def test_lod_shape(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        n_loci = len(tiny_cache._data[chrom]["locus_ids"])
        assert result["lod"].shape == (n_loci,)

    def test_lod_nonnegative(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        assert np.all(result["lod"] >= -1e-8)

    def test_lod_finite(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        # LOD = (n/2) log10(RSS0/RSS1) — all values must be finite
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        assert np.all(np.isfinite(result["lod"]))

    def test_p_values_in_unit_interval(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs)
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        pv = result["p_value"]
        assert np.all(pv >= 0.0) and np.all(pv <= 1.0 + 1e-8)


class TestScanChromosomeFull:
    def test_full_betas_shape(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs, model="full")
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        n_loci = len(tiny_cache._data[chrom]["locus_ids"])
        J = tiny_cache.n_founders
        nf_full = J * (J + 1) // 2
        assert result["betas"].shape == (n_loci, nf_full)

    def test_full_lod_nonnegative(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs, model="full")
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        assert np.all(result["lod"] >= -1e-8)

    def test_full_lod_finite(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        chrom = tiny_cache.chromosomes[0]
        K = tiny_kinship[chrom]
        eigvals, eigvecs = np.linalg.eigh(K)
        h2, _ = estimate_h2_null(
            eigvecs.T @ tiny_strain_phenotype,
            eigvecs.T @ tiny_strain_covariates,
            eigvals,
        )
        cd = precompute_chromosome(tiny_cache, chrom, eigvals, eigvecs, model="full")
        result = scan_chromosome(
            tiny_strain_phenotype,
            cd,
            tiny_strain_covariates,
            eigvals,
            eigvecs,
            h2,
        )
        assert np.all(np.isfinite(result["lod"]))


class TestReplicatePath:
    """scan_h2lmm with replicated samples uses the sample-level Z K Z^T model."""

    def test_replicate_path_uses_sample_level_model(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_phenotype,
        sample_strain_map,
    ):
        """Replicate path (n=40, Z K Z^T) gives different LODs than the strain-mean
        path (m=10), confirming the n-dimensional model is active.  LODs must be
        finite, non-negative, and rank-consistent with the strain-mean path."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.utils import build_z_matrix

        kd = decomp_kinship(tiny_kinship)

        # replicate path: n=40 sample-level model with K_n = Z K_m Z^T
        res_rep = scan_h2lmm(
            tiny_phenotype,
            tiny_cache,
            kinship_decomp=kd,
            sample_ids=list(sample_strain_map),
            use_rint=False,
            scan_window="genome",
        )

        # strain-mean path: m=10 model (no replication)
        Z, _ = build_z_matrix(list(sample_strain_map), list(tiny_cache.sample_ids))
        r = Z.sum(axis=0)
        y_bar = (Z.T @ tiny_phenotype) / r
        res_strain = scan_h2lmm(
            y_bar,
            tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window="genome",
        )

        # models are intentionally different — LODs must not be equal
        assert not np.allclose(
            res_rep.lod, res_strain.lod
        ), "replicate and strain-mean LODs must differ (Z K Z^T vs K_m models)"
        # sanity: replicate-path LODs are finite and non-negative
        assert np.all(np.isfinite(res_rep.lod))
        assert np.all(res_rep.lod >= 0.0)
        # locus ranking must be consistent across the two models
        rank_rep = np.argsort(np.argsort(res_rep.lod))
        rank_strain = np.argsort(np.argsort(res_strain.lod))
        rho = np.corrcoef(rank_rep, rank_strain)[0, 1]
        assert rho > 0.7, f"rank correlation between models too low: {rho:.3f}"

    def test_replicate_path_lod_finite(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_phenotype,
        sample_strain_map,
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        res = scan_h2lmm(
            tiny_phenotype,
            tiny_cache,
            kinship_decomp=kd,
            sample_ids=list(sample_strain_map),
            use_rint=False,
            scan_window="genome",
        )
        assert np.all(np.isfinite(res.lod))
        assert np.all(res.lod >= 0.0)


class TestEstimateH2NullGrid:
    """Search range of the profile-ML optimiser (docs/math/03_lmm_scan.typ, section 2)."""

    @staticmethod
    def _rotated(h2_true: float, n: int = 2000, seed: int = 0):
        rng = np.random.default_rng(seed)
        eigvals = np.geomspace(1e-3, 200.0, n)
        w = h2_true * eigvals + (1.0 - h2_true)
        y_rot = np.sqrt(w) * rng.standard_normal(n)
        x_rot = np.ones((n, 1))
        return y_rot, x_rot, eigvals

    def test_interior_optimum_below_grid_start_is_recovered(self):
        y_rot, x_rot, eigvals = self._rotated(h2_true=0.002)
        h2, ll = estimate_h2_null(y_rot, x_rot, eigvals)
        assert 1e-6 < h2 < 0.0099
        assert ll >= _profile_ml(y_rot, x_rot, eigvals, 1e-6) - 1e-9
        assert ll >= _profile_ml(y_rot, x_rot, eigvals, 0.01) - 1e-9

    def test_optimum_beats_every_linear_grid_point(self):
        y_rot, x_rot, eigvals = self._rotated(h2_true=0.002)
        h2, ll = estimate_h2_null(y_rot, x_rot, eigvals)
        for h in np.linspace(0.01, 0.99, 20):
            assert ll >= _profile_ml(y_rot, x_rot, eigvals, h) - 1e-9

    def test_boundary_flag_marks_floor_and_ceiling_only(self):
        assert h2_at_boundary(1e-6)
        assert h2_at_boundary(1.0 - 1e-6)
        assert not h2_at_boundary(0.99)
        assert not h2_at_boundary(0.002)
        assert not h2_at_boundary(0.3)

    def test_boundary_flag_follows_the_estimate(self):
        n = 500
        eigvals = np.geomspace(0.05, 20.0, n)
        y_rot = np.sqrt(eigvals) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        h2_ceiling, _ = estimate_h2_null(y_rot, np.ones((n, 1)), eigvals)
        assert h2_at_boundary(h2_ceiling)
        y_rot, x_rot, eigvals = self._rotated(h2_true=0.3)
        h2_interior, _ = estimate_h2_null(y_rot, x_rot, eigvals)
        assert not h2_at_boundary(h2_interior)

    def test_grid_is_log_spaced_toward_both_ends(self):
        from mosaiqtl.qtl.lmm import _H2_GRID

        assert _H2_GRID.min() == pytest.approx(1e-6)
        assert _H2_GRID.max() == pytest.approx(1.0 - 1e-6)
        assert np.all(np.diff(_H2_GRID) > 0)
        np.testing.assert_allclose(1.0 - _H2_GRID[-4:][::-1], _H2_GRID[:4], rtol=1e-6)

    def test_ceiling_optimum_is_reported_at_the_ceiling(self):
        n = 500
        eigvals = np.geomspace(0.05, 20.0, n)
        y_rot = np.sqrt(eigvals) * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        x_rot = np.ones((n, 1))
        h2, _ = estimate_h2_null(y_rot, x_rot, eigvals)
        assert h2 > 0.9

    def test_result_stays_in_admissible_interval(self):
        for h2_true in (0.0, 0.002, 0.3, 0.9):
            y_rot, x_rot, eigvals = self._rotated(h2_true, n=300, seed=3)
            h2, _ = estimate_h2_null(y_rot, x_rot, eigvals)
            assert 1e-6 <= h2 <= 1.0 - 1e-6


def _profile_ml(y_rot, x_rot, eigvals, h2):
    """Profile ML log-likelihood at a fixed $h^2$ (reference for the grid tests)."""
    w = h2 * eigvals + (1.0 - h2)
    wi = 1.0 / w
    xtwx = x_rot.T @ (x_rot * wi[:, None])
    alpha = np.linalg.solve(xtwx, x_rot.T @ (wi * y_rot))
    r = y_rot - x_rot @ alpha
    n = len(y_rot)
    sigma2 = float(np.sum(wi * r * r) / n)
    return -0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0) - 0.5 * float(np.sum(np.log(w)))

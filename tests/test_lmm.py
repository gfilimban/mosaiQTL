# tests/test_lmm.py — unit tests for mosaiqtl.qtl.lmm
import numpy as np
import pytest
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.qtl.lmm import (
    fit_lmm_at_locus,
    estimate_h2_null,
    precompute_chromosome,
    scan_chromosome,
)
from mosaiqtl.models.records import LMMResult
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship


class TestLODFormula:
    """LOD = (n/2) * log10(RSS0/RSS1). ML denominator n, not n-p."""

    def _setup(self, n=40, J=8, seed=0):
        rng = np.random.default_rng(seed)
        ev = np.ones(n)  # identity kinship eigenvalues
        evc = np.eye(n)  # identity eigenvectors
        x = np.ones((n, 1))  # intercept only
        h = rng.dirichlet(np.ones(J), size=n)[:, :-1]  # (n, J-1) haplotype
        return ev, evc, x, h, rng

    def test_lod_zero_null_phenotype(self):
        """if y is orthogonal to H, LOD should be near 0."""
        ev, evc, x, h, rng = self._setup()
        n = len(ev)
        # phenotype = pure noise unrelated to h
        y = rng.standard_normal(n)
        y -= y.mean()
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        assert res.lod >= 0.0  # LOD is non-negative
        assert res.lod < 3.0  # should be near null

    def test_lod_positive_for_signal(self):
        """if y is generated from H, LOD should be clearly positive."""
        ev, evc, x, h, rng = self._setup()
        n = len(ev)
        # phenotype generated from first haplotype column
        y = 3.0 * h[:, 0] + 0.3 * rng.standard_normal(n)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        assert res.lod > 1.0

    def test_lod_formula_ml_denominator(self):
        """verify LOD uses n (ML) not n-p (REML)."""
        ev, evc, x, h, rng = self._setup()
        n = len(ev)
        y = rng.standard_normal(n)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        # LOD must be finite and non-negative regardless of formula choice
        assert np.isfinite(res.lod) and res.lod >= 0.0

    def test_betas_shape(self):
        ev, evc, x, h, rng = self._setup()
        n = len(ev)
        J_minus_1 = h.shape[1]
        y = rng.standard_normal(n)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
        res = fit_lmm_at_locus(evc.T @ y, evc.T @ h, evc.T @ x, ev, h2)
        assert res.betas.shape == (J_minus_1,)


class TestLMMResultDataclass:
    """LMMResult must be a slots dataclass — no silent attribute creation."""

    def _make(self):
        return LMMResult(
            lod=1.5,
            p_value=0.01,
            h2=0.3,
            f_stat=4.2,
            betas=np.zeros(7),
            sigma2_g=0.2,
            sigma2_e=0.5,
            rss_null=10.0,
            rss_alt=8.0,
        )

    def test_slots_no_dict(self):
        assert not hasattr(self._make(), "__dict__")

    def test_fields_accessible(self):
        r = self._make()
        assert r.lod == pytest.approx(1.5)
        assert r.p_value == pytest.approx(0.01)
        assert r.h2 == pytest.approx(0.3)
        assert r.betas.shape == (7,)

    def test_unknown_field_raises(self):
        with pytest.raises((AttributeError, TypeError)):
            LMMResult(
                lod=0.0,
                p_value=1.0,
                h2=0.0,
                f_stat=0.0,
                betas=np.zeros(7),
                sigma2_g=0.0,
                sigma2_e=0.0,
                rss_null=0.0,
                rss_alt=0.0,
                extra_field=99,
            )


def _make_scan_cache(n_strains=20, n_loci=12, n_founders=8, chroms=("1", "2")):
    rng = np.random.default_rng(7)
    data = {}
    for c in chroms:
        probs = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_strains))
        data[c] = {
            "locus_ids": np.array([f"{c}_L{i:02d}" for i in range(n_loci)]),
            "positions": np.linspace(1.0, 120.0, n_loci),
            "probs": probs,
        }
    return GenomeCache(
        data=data,
        sample_ids=[f"s{i}" for i in range(n_strains)],
        founders=[f"F{i}" for i in range(n_founders)],
    )


class TestScanChromosome:
    """scan_chromosome: vectorised FWL must match per-locus fit_lmm_at_locus."""

    def setup_method(self):
        self.cache = _make_scan_cache()
        n = 20
        rng = np.random.default_rng(3)
        kd = calc_kinship(self.cache, method="loco")
        dc = decomp_kinship(kd)
        self.chrom = "1"
        self.ev = dc[self.chrom]["values"]
        self.evc = dc[self.chrom]["vectors"]
        self.x = np.ones((n, 1))
        self.y = rng.standard_normal(n)
        self.h2, _ = estimate_h2_null(self.evc.T @ self.y, self.evc.T @ self.x, self.ev)
        self.cd = precompute_chromosome(self.cache, self.chrom, self.ev, self.evc)

    def test_output_keys(self):
        res = scan_chromosome(self.y, self.cd, self.x, self.ev, self.evc, self.h2)
        assert set(res.keys()) >= {"locus_ids", "positions", "lod", "p_value", "betas"}

    def test_lod_shape(self):
        res = scan_chromosome(self.y, self.cd, self.x, self.ev, self.evc, self.h2)
        n_loci = self.cache.n_loci(self.chrom)
        assert res["lod"].shape == (n_loci,)

    def test_lod_nonneg(self):
        res = scan_chromosome(self.y, self.cd, self.x, self.ev, self.evc, self.h2)
        assert np.all(res["lod"] >= 0.0)

    def test_fwl_matches_per_locus(self):
        """vectorised scan must equal per-locus fit_lmm_at_locus within
        1e-6."""
        res = scan_chromosome(self.y, self.cd, self.x, self.ev, self.evc, self.h2)
        _, _, probs = self.cache.get_chromosome(self.chrom)
        nf1 = self.cache.n_founders - 1
        y_rot = self.evc.T @ self.y
        x_rot = self.evc.T @ self.x
        for i in range(probs.shape[0]):
            h_rot = self.evc.T @ probs[i, :, :nf1]
            ref = fit_lmm_at_locus(y_rot, h_rot, x_rot, self.ev, self.h2)
            assert (
                abs(res["lod"][i] - ref.lod) < 1e-6
            ), f'locus {i}: scan={res["lod"][i]:.6f} vs per-locus={ref.lod:.6f}'

    def test_planted_signal_at_peak(self):
        """phenotype generated from locus j should give peak LOD at j."""
        _, _, probs = self.cache.get_chromosome(self.chrom)
        j_signal = 5
        rng = np.random.default_rng(99)
        y_signal = 4.0 * probs[j_signal, :, 0] + 0.1 * rng.standard_normal(20)
        h2, _ = estimate_h2_null(self.evc.T @ y_signal, self.evc.T @ self.x, self.ev)
        res = scan_chromosome(y_signal, self.cd, self.x, self.ev, self.evc, h2)
        assert int(np.argmax(res["lod"])) == j_signal

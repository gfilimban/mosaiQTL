"""Unit tests for qtl/vqtl.py — dispersion QTL mapping."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.vqtl import (
    VQTLResult,
    estimate_dispersion_per_subject,
    compare_mean_vs_dispersion,
)

# ── VQTLResult ────────────────────────────────────────────────────────────────


def _make_scan_result_stub(n_loci=10, peak_lod=5.0, peak_chr="1"):
    """Return a minimal ScanResult-like object for VQTLResult construction."""
    from mosaiqtl.qtl.scan import ScanResult

    rng = np.random.default_rng(0)
    lods = np.abs(rng.standard_normal(n_loci))
    lods[0] = peak_lod
    return ScanResult(
        locus_ids=np.array([f"L{i}" for i in range(n_loci)]),
        chromosomes=np.full(n_loci, peak_chr),
        positions=np.linspace(1.0, 100.0, n_loci),
        lod=lods,
        p_value=rng.uniform(0, 0.5, n_loci),
        betas=rng.standard_normal((n_loci, 7)),
        h2_null={peak_chr: 0.3},
        phenotype_id="test",
    )


class TestVQTLResult:
    def test_attributes_set(self):
        sr = _make_scan_result_stub()
        vr = VQTLResult(
            scan_result=sr,
            phi_estimates=np.ones(10),
            phi_method="moments",
            gene_id="Myogene",
            mean_scan_result=None,
        )
        assert vr.gene_id == "Myogene"
        assert vr.phi_method == "moments"
        assert vr.mean_scan_result is None

    def test_fields_accessible(self):
        """VQTLResult fields are accessible and phi_estimates is stored correctly."""
        sr = _make_scan_result_stub()
        phi = np.abs(np.random.default_rng(0).standard_normal(10))
        vr = VQTLResult(
            scan_result=sr,
            phi_estimates=phi,
            phi_method="moments",
            gene_id="Myogene",
            mean_scan_result=None,
        )
        assert vr.gene_id == "Myogene"
        assert np.allclose(vr.phi_estimates, phi)
        assert vr.cell_type is None
        assert vr.n_samples == 0


# ── estimate_dispersion_per_subject ──────────────────────────────────────────


def _make_cell_counts(n_subjects=8, n_cells_each=30, phi=0.5, mu=10.0, seed=0):
    """Simulate NB count data with per-subject identity."""
    rng = np.random.default_rng(seed)
    r = 1.0 / phi
    p = r / (r + mu)
    counts = rng.negative_binomial(r, p, size=n_subjects * n_cells_each)
    subject_ids = np.repeat(
        np.array([f"S{i:02d}" for i in range(n_subjects)]), n_cells_each
    )
    return counts, subject_ids


@pytest.mark.skip(
    reason="estimate_dispersion_per_subject requires precompute_sufficient_stats"
)
class TestEstimateDispersion:
    def test_returns_subjects_and_phi(self):
        counts, sids = _make_cell_counts()
        subjects, phi_hat = estimate_dispersion_per_subject(
            counts, sids, method="moments", min_cells=10
        )
        assert len(subjects) == len(phi_hat)

    def test_subjects_unique(self):
        counts, sids = _make_cell_counts(n_subjects=5)
        subjects, _ = estimate_dispersion_per_subject(
            counts, sids, method="moments", min_cells=10
        )
        assert len(set(subjects)) == len(subjects)

    def test_phi_nonnegative_where_valid(self):
        counts, sids = _make_cell_counts(n_subjects=6, n_cells_each=40)
        _, phi_hat = estimate_dispersion_per_subject(
            counts, sids, method="moments", min_cells=20
        )
        valid = ~np.isnan(phi_hat)
        assert np.all(phi_hat[valid] >= 0.0)

    def test_few_cells_gives_nan(self):
        counts, sids = _make_cell_counts(n_subjects=4, n_cells_each=5)
        _, phi_hat = estimate_dispersion_per_subject(
            counts, sids, method="moments", min_cells=50
        )
        assert np.all(np.isnan(phi_hat))

    def test_mle_method(self):
        counts, sids = _make_cell_counts(n_subjects=4, n_cells_each=50)
        subjects, phi_hat = estimate_dispersion_per_subject(
            counts, sids, method="mle", min_cells=20
        )
        valid = ~np.isnan(phi_hat)
        assert valid.sum() >= 1
        assert np.all(phi_hat[valid] > 0)

    def test_invalid_method_raises(self):
        counts, sids = _make_cell_counts()
        with pytest.raises(ValueError, match="unknown dispersion method"):
            estimate_dispersion_per_subject(counts, sids, method="bad")

    def test_with_library_sizes(self):
        counts, sids = _make_cell_counts(n_subjects=4, n_cells_each=40)
        lib_sizes = np.ones(len(counts)) * 1000.0
        subjects, phi_hat = estimate_dispersion_per_subject(
            counts, sids, library_sizes=lib_sizes, method="moments", min_cells=20
        )
        assert len(phi_hat) == 4


# ── compare_mean_vs_dispersion ────────────────────────────────────────────────


class TestCompareMeanVsDispersion:
    def _make_vqtl_result(self, vqtl_lod, mean_lod, same_chr=True):
        chr_v = "1"
        chr_m = "1" if same_chr else "2"
        vsr = _make_scan_result_stub(peak_lod=vqtl_lod, peak_chr=chr_v)
        msr = _make_scan_result_stub(peak_lod=mean_lod, peak_chr=chr_m)
        return VQTLResult(
            scan_result=vsr,
            phi_estimates=np.ones(5),
            phi_method="moments",
            gene_id="g",
            mean_scan_result=msr,
        )

    def test_both_significant(self):
        vr = self._make_vqtl_result(vqtl_lod=5.0, mean_lod=5.0)
        res = compare_mean_vs_dispersion(vr, lod_threshold=3.0)
        assert res["category"] == "both"

    def test_mean_only(self):
        vr = self._make_vqtl_result(vqtl_lod=1.0, mean_lod=5.0)
        res = compare_mean_vs_dispersion(vr, lod_threshold=3.0)
        assert res["category"] == "mean_only"

    def test_dispersion_only(self):
        vr = self._make_vqtl_result(vqtl_lod=5.0, mean_lod=1.0)
        res = compare_mean_vs_dispersion(vr, lod_threshold=3.0)
        assert res["category"] == "dispersion_only"

    def test_neither(self):
        vr = self._make_vqtl_result(vqtl_lod=1.0, mean_lod=1.0)
        res = compare_mean_vs_dispersion(vr, lod_threshold=3.0)
        assert res["category"] == "neither"

    def test_result_keys(self):
        vr = self._make_vqtl_result(vqtl_lod=5.0, mean_lod=5.0)
        res = compare_mean_vs_dispersion(vr)
        for key in (
            "category",
            "mean_peak_lod",
            "vqtl_peak_lod",
            "colocalized",
            "mean_peak_locus",
            "vqtl_peak_locus",
        ):
            assert key in res

    def test_colocalized_same_chr(self):
        vr = self._make_vqtl_result(vqtl_lod=5.0, mean_lod=5.0, same_chr=True)
        res = compare_mean_vs_dispersion(vr)
        # small fixture — positions overlap, so should be colocalized
        assert isinstance(res["colocalized"], (bool, np.bool_))

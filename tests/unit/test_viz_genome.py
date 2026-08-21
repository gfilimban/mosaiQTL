"""Unit tests for viz/genome.py — LOD profiles, Manhattan, scan summaries."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.viz.genome import (
    lod_profile,
    manhattan,
    lod_interval,
    pvalue_qq,
    lod_histogram,
    gene_manhattan,
    cis_distance_histogram,
    nb_vs_lmm_lod,
    library_size_qc,
    chromosome_lod_profile,
    peak_locus_zoom,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


N_LOCI = 30
CHROMS = ["1", "2"]


def _lod_array(seed=0):
    rng = np.random.default_rng(seed)
    return np.abs(rng.standard_normal(N_LOCI))


def _pos_array():
    return np.linspace(1.0, 150.0, N_LOCI)


# ── lod_profile ───────────────────────────────────────────────────────────────


class TestLodProfile:
    def test_returns_axes(self):
        ax = lod_profile(_pos_array(), _lod_array(), chrom="1")
        assert isinstance(ax, plt.Axes)

    def test_accepts_existing_ax(self):
        _, ax_in = plt.subplots()
        ax_out = lod_profile(_pos_array(), _lod_array(), ax=ax_in)
        assert ax_out is ax_in

    def test_with_threshold(self):
        ax = lod_profile(_pos_array(), _lod_array(), threshold=3.5)
        assert isinstance(ax, plt.Axes)

    def test_with_peak_pos(self):
        ax = lod_profile(_pos_array(), _lod_array(), peak_pos=75.0)
        assert isinstance(ax, plt.Axes)

    def test_ylim_from_zero(self):
        ax = lod_profile(_pos_array(), _lod_array())
        assert ax.get_ylim()[0] <= 0.0


# ── manhattan ─────────────────────────────────────────────────────────────────


class TestManhattan:
    def _make_scans(self):
        rng = np.random.default_rng(1)
        return {
            "1": (np.linspace(1, 100, N_LOCI), np.abs(rng.standard_normal(N_LOCI))),
            "2": (np.linspace(1, 100, N_LOCI), np.abs(rng.standard_normal(N_LOCI))),
        }

    def test_returns_figure(self):
        import matplotlib.figure

        fig = manhattan(self._make_scans())
        assert isinstance(fig, matplotlib.figure.Figure)

    def test_with_threshold(self):
        import matplotlib.figure

        fig = manhattan(self._make_scans(), threshold=4.0)
        assert isinstance(fig, matplotlib.figure.Figure)


# ── lod_interval ──────────────────────────────────────────────────────────────


class TestLodInterval:
    def test_returns_axes(self):
        pos = _pos_array()
        lod = _lod_array()
        ax = lod_interval(pos, lod, lod_drop=1.5)
        assert isinstance(ax, plt.Axes)


# ── pvalue_qq ─────────────────────────────────────────────────────────────────


class TestPvalueQq:
    def test_returns_axes(self):
        rng = np.random.default_rng(2)
        pvals = rng.uniform(0, 1, 100)
        ax = pvalue_qq(pvals)
        assert isinstance(ax, plt.Axes)


# ── lod_histogram ─────────────────────────────────────────────────────────────


class TestLodHistogram:
    def test_returns_axes(self):
        lod = np.abs(np.random.default_rng(3).standard_normal(200))
        ax = lod_histogram(lod)
        assert isinstance(ax, plt.Axes)

    def test_with_threshold(self):
        lod = np.abs(np.random.default_rng(4).standard_normal(200))
        ax = lod_histogram(lod, threshold=4.0)
        assert isinstance(ax, plt.Axes)


# ── gene_manhattan ────────────────────────────────────────────────────────────


class TestGeneManhattan:
    def test_returns_figure(self):
        import matplotlib.figure

        rng = np.random.default_rng(5)
        n = 50
        gene_pos = rng.uniform(1, 100, n)
        gene_chrs = np.array(["1"] * 25 + ["2"] * 25)
        peak_lod = np.abs(rng.standard_normal(n))
        fig = gene_manhattan(gene_pos, gene_chrs, peak_lod)
        assert isinstance(fig, matplotlib.figure.Figure)


# ── cis_distance_histogram ────────────────────────────────────────────────────


class TestCisDistanceHistogram:
    def test_returns_axes(self):
        rng = np.random.default_rng(6)
        tss_pos = rng.uniform(1, 100, 100)
        peak_pos = tss_pos + rng.uniform(-10, 10, 100)
        ax = cis_distance_histogram(tss_pos, peak_pos)
        assert isinstance(ax, plt.Axes)


# ── nb_vs_lmm_lod ─────────────────────────────────────────────────────────────


class TestNbVsLmmLod:
    def test_returns_axes(self):
        rng = np.random.default_rng(7)
        nb_lod = np.abs(rng.standard_normal(50))
        lmm_lod = nb_lod * 0.9 + rng.standard_normal(50) * 0.1
        ax = nb_vs_lmm_lod(nb_lod, lmm_lod)
        assert isinstance(ax, plt.Axes)


# ── library_size_qc ───────────────────────────────────────────────────────────


class TestLibrarySizeQc:
    def test_returns_axes(self, library_sizes_cells):
        ax = library_size_qc(library_sizes_cells)
        assert isinstance(ax, plt.Axes)


# ── chromosome_lod_profile ────────────────────────────────────────────────────


class TestChromosomeLodProfile:
    def test_returns_axes(self):
        ax = chromosome_lod_profile(_pos_array(), _lod_array(), chrom="8")
        assert isinstance(ax, plt.Axes)

    def test_ylim_from_zero(self):
        ax = chromosome_lod_profile(_pos_array(), _lod_array())
        assert ax.get_ylim()[0] == 0.0

    def test_with_threshold(self):
        ax = chromosome_lod_profile(_pos_array(), _lod_array(), threshold=3.5)
        # KW_THRESHOLD uses ':' (dotted); check that at least one threshold line is present
        # (>= 1 because matplotlib versions may produce extra dashed lines internally)
        lines = [ln for ln in ax.get_lines() if ln.get_linestyle() in ("--", ":")]
        assert len(lines) >= 1

    def test_n_loci_annotation(self):
        pos = _pos_array()
        ax = chromosome_lod_profile(pos, _lod_array(), annotate_n_loci=True)
        texts = [t.get_text() for t in ax.texts]
        assert any(str(len(pos)) in t for t in texts)

    def test_no_annotation_when_disabled(self):
        ax = chromosome_lod_profile(_pos_array(), _lod_array(), annotate_n_loci=False)
        assert len(ax.texts) == 0


# ── peak_locus_zoom ───────────────────────────────────────────────────────────


class TestPeakLocusZoom:
    def test_returns_axes(self):
        pos = _pos_array()
        lod = _lod_array()
        peak = float(pos[np.argmax(lod)])
        ax = peak_locus_zoom(pos, lod, peak, chrom="8")
        assert isinstance(ax, plt.Axes)

    def test_xlim_within_window(self):
        pos = _pos_array()
        lod = _lod_array()
        peak = 75.0
        window = 5.0
        ax = peak_locus_zoom(pos, lod, peak, window_mb=window)
        lo, hi = ax.get_xlim()
        assert lo == pytest.approx(peak - window, abs=1e-3)
        assert hi == pytest.approx(peak + window, abs=1e-3)

    def test_ylim_from_zero(self):
        pos = _pos_array()
        lod = _lod_array()
        peak = float(pos[np.argmax(lod)])
        ax = peak_locus_zoom(pos, lod, peak)
        assert ax.get_ylim()[0] == 0.0

    def test_ci_shading_added(self):
        pos = _pos_array()
        lod = _lod_array()
        peak = 75.0
        ax = peak_locus_zoom(
            pos, lod, peak, window_mb=20.0, ci_lo_mb=65.0, ci_hi_mb=85.0
        )
        shaded = [
            p for p in ax.patches if p.get_alpha() is not None and p.get_alpha() < 0.5
        ]
        assert len(shaded) >= 1

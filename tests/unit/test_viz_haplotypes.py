"""Unit tests for viz/haplotypes.py — heatmaps, pxg, kinship, epistasis grids."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.viz.haplotypes import (
    prob_heatmap,
    pxg,
    kinship_heatmap,
    epistasis_heatmap,
)
from mosaiqtl.genotype.founders import FOUNDER_ORDER  # noqa: F401

N_FOUNDERS = 8
N_SAMPLES  = 10


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def _hap_probs(seed=0):
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet(np.ones(N_FOUNDERS), size=N_SAMPLES)
    return raw  # (N_SAMPLES, N_FOUNDERS), sums to 1 per row


# ── prob_heatmap ──────────────────────────────────────────────────────────────

class TestProbHeatmap:
    def test_returns_axes(self):
        ax = prob_heatmap(_hap_probs())
        assert isinstance(ax, plt.Axes)

    def test_with_phenotype(self):
        rng = np.random.default_rng(1)
        pheno = rng.standard_normal(N_SAMPLES)
        ax = prob_heatmap(_hap_probs(), phenotype=pheno)
        assert isinstance(ax, plt.Axes)

    def test_with_sample_labels(self):
        labels = [f'S{i}' for i in range(N_SAMPLES)]
        ax = prob_heatmap(_hap_probs(), sample_labels=labels)
        assert isinstance(ax, plt.Axes)

    def test_with_chrom_and_pos(self):
        ax = prob_heatmap(_hap_probs(), chrom='8', position_mb=55.3)
        assert isinstance(ax, plt.Axes)

    def test_accepts_existing_ax(self):
        _, ax_in = plt.subplots()
        ax_out = prob_heatmap(_hap_probs(), ax=ax_in)
        assert ax_out is ax_in


# ── pxg ───────────────────────────────────────────────────────────────────────

class TestPxg:
    def test_returns_axes(self):
        rng = np.random.default_rng(2)
        hap   = _hap_probs()
        pheno = rng.standard_normal(N_SAMPLES)
        ax = pxg(pheno, hap)
        assert isinstance(ax, plt.Axes)

    def test_with_chrom_pos(self):
        rng = np.random.default_rng(3)
        hap   = _hap_probs()
        pheno = rng.standard_normal(N_SAMPLES)
        ax = pxg(pheno, hap, chrom='8', position_mb=55.3)
        assert isinstance(ax, plt.Axes)


# ── kinship_heatmap ───────────────────────────────────────────────────────────

class TestKinshipHeatmap:
    def test_returns_axes(self, tiny_kinship):
        K = tiny_kinship['1']
        ax = kinship_heatmap(K)
        assert isinstance(ax, plt.Axes)

    def test_with_sample_labels(self, tiny_kinship, strain_ids):
        K = tiny_kinship['1']
        ax = kinship_heatmap(K, sample_labels=list(strain_ids))
        assert isinstance(ax, plt.Axes)


# ── epistasis_heatmap ─────────────────────────────────────────────────────────

class TestEpistasisHeatmap:
    def test_returns_axes(self):
        rng = np.random.default_rng(4)
        # J×J interaction coefficient matrix
        int_mat = rng.standard_normal((N_FOUNDERS, N_FOUNDERS))
        ax = epistasis_heatmap(int_mat)
        assert isinstance(ax, plt.Axes)

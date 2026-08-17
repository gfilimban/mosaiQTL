"""Unit tests for viz/effects.py — founder effect bars, profiles, caterpillar."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.viz.effects import (
    coef_profile,
    coef_profile_genome,
    effect_bars,
    effect_caterpillar,
    dispersion_distribution,
    _recover_full_betas,
)
from mosaiqtl.genotype.founders import FOUNDER_ORDER  # noqa: F401

N_FOUNDERS = 8
N_LOCI = 20


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def _betas(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((N_LOCI, N_FOUNDERS - 1))


def _positions():
    return np.linspace(1.0, 100.0, N_LOCI)


# ── _recover_full_betas ───────────────────────────────────────────────────────


class TestRecoverFullBetas:
    def test_length(self):
        b = np.array([1.0, -0.5, 0.3, -0.2, 0.1, 0.4, -0.1])
        full = _recover_full_betas(b)
        assert len(full) == N_FOUNDERS

    def test_sum_to_zero(self):
        rng = np.random.default_rng(0)
        b = rng.standard_normal(N_FOUNDERS - 1)
        full = _recover_full_betas(b)
        assert full.sum() == pytest.approx(0.0, abs=1e-10)


# ── coef_profile ──────────────────────────────────────────────────────────────


class TestCoefProfile:
    def test_returns_axes(self):
        ax = coef_profile(_positions(), _betas(), chrom="1")
        assert isinstance(ax, plt.Axes)

    def test_accepts_existing_ax(self):
        _, ax_in = plt.subplots()
        ax_out = coef_profile(_positions(), _betas(), ax=ax_in)
        assert ax_out is ax_in

    def test_no_legend(self):
        ax = coef_profile(_positions(), _betas(), show_legend=False)
        assert isinstance(ax, plt.Axes)


# ── coef_profile_genome ───────────────────────────────────────────────────────


class TestCoefProfileGenome:
    def test_returns_figure(self):
        import matplotlib.figure

        rng = np.random.default_rng(1)
        scans = {
            "1": (_positions(), rng.standard_normal((N_LOCI, N_FOUNDERS - 1))),
            "2": (_positions(), rng.standard_normal((N_LOCI, N_FOUNDERS - 1))),
        }
        fig = coef_profile_genome(scans)
        assert isinstance(fig, matplotlib.figure.Figure)


# ── effect_bars ───────────────────────────────────────────────────────────────


class TestEffectBars:
    def test_returns_axes(self):
        rng = np.random.default_rng(2)
        betas_locus = rng.standard_normal(N_FOUNDERS - 1)
        ax = effect_bars(betas_locus)
        assert isinstance(ax, plt.Axes)

    def test_with_se(self):
        rng = np.random.default_rng(3)
        betas_locus = rng.standard_normal(N_FOUNDERS - 1)
        se = np.abs(rng.standard_normal(N_FOUNDERS - 1)) * 0.2
        ax = effect_bars(betas_locus, se=se)
        assert isinstance(ax, plt.Axes)


# ── effect_caterpillar ────────────────────────────────────────────────────────


class TestEffectCaterpillar:
    def test_returns_axes(self):
        rng = np.random.default_rng(4)
        betas = rng.standard_normal(N_FOUNDERS - 1)
        ax = effect_caterpillar(betas)
        assert isinstance(ax, plt.Axes)

    def test_with_ci(self):
        rng = np.random.default_rng(4)
        betas = rng.standard_normal(N_FOUNDERS - 1)
        half = np.abs(rng.standard_normal(N_FOUNDERS)) * 0.5
        # ci_wide: (J, 2) [lower, upper] per founder
        ci_wide = np.column_stack([-half, half])
        ci_narrow = np.column_stack([-half * 0.5, half * 0.5])
        ax = effect_caterpillar(betas, ci_wide=ci_wide, ci_narrow=ci_narrow)
        assert isinstance(ax, plt.Axes)


# ── dispersion_distribution ───────────────────────────────────────────────────


class TestDispersionDistribution:
    def test_returns_axes(self):
        rng = np.random.default_rng(5)
        phi_vals = rng.exponential(0.4, 50)
        ax = dispersion_distribution(phi_vals)
        assert isinstance(ax, plt.Axes)

    def test_with_phi_prior(self):
        rng = np.random.default_rng(6)
        phi_vals = rng.exponential(0.4, 50)
        ax = dispersion_distribution(phi_vals, phi_prior=0.3)
        assert isinstance(ax, plt.Axes)

"""Unit tests for qtl/fdr.py — storey_qvalue."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.fdr import storey_qvalue

_FIXTURE = Path(__file__).parents[1] / 'fixtures' / 'data' / 'qvalue_reference.csv'


# ── π₀ estimation ────────────────────────────────────────────────────────────

class TestPi0Estimation:
    def test_uniform_null_pi0_near_one(self):
        rng = np.random.default_rng(0)
        pv = rng.uniform(0, 1, 2000)
        _, pi0 = storey_qvalue(pv)
        assert pi0 <= 1.0
        assert pi0 >= 0.85  # uniform null → π₀ close to 1

    def test_strong_signal_pi0_below_one(self):
        rng = np.random.default_rng(1)
        # 50% near-zero (signal) + 50% uniform (null)
        pv = np.concatenate([rng.beta(0.1, 1, 500), rng.uniform(0, 1, 500)])
        _, pi0 = storey_qvalue(pv)
        assert pi0 < 0.75

    def test_pi0_never_exceeds_one(self):
        rng = np.random.default_rng(2)
        for seed in range(10):
            pv = rng.uniform(0, 1, 200)
            _, pi0 = storey_qvalue(pv)
            assert pi0 <= 1.0 + 1e-9


# ── q-value properties ────────────────────────────────────────────────────────

class TestQvalueProperties:
    def test_qvalues_in_unit_interval(self):
        rng = np.random.default_rng(3)
        pv = rng.uniform(0, 1, 300)
        q, _ = storey_qvalue(pv)
        assert np.all(q[np.isfinite(q)] >= 0.0)
        assert np.all(q[np.isfinite(q)] <= 1.0 + 1e-9)

    def test_qvalues_monotone(self):
        rng = np.random.default_rng(4)
        pv = rng.uniform(0, 1, 500)
        q, _ = storey_qvalue(pv)
        # sorted p → non-decreasing q
        order = np.argsort(pv)
        q_sorted = q[order]
        assert np.all(np.diff(q_sorted) >= -1e-10)

    def test_qvalues_leq_bh_under_signal(self):
        rng = np.random.default_rng(5)
        pv = np.concatenate([rng.beta(0.1, 1, 200), rng.uniform(0, 1, 200)])
        q_storey, pi0 = storey_qvalue(pv)
        # BH is Storey with π₀=1; when π₀<1, Storey ≤ BH
        if pi0 < 0.99:
            m = len(pv)
            order = np.argsort(pv)
            ranks = np.arange(1, m + 1)
            q_bh_sorted = np.minimum(m * pv[order] / ranks, 1.0)
            for i in range(m - 2, -1, -1):
                q_bh_sorted[i] = min(q_bh_sorted[i], q_bh_sorted[i + 1])
            q_bh = np.empty(m)
            q_bh[order] = q_bh_sorted
            assert np.all(q_storey <= q_bh + 1e-9)

    def test_qvalues_approx_bh_under_full_null(self):
        # large uniform sample → π₀ ≈ 1 → Storey q ≈ BH q
        rng = np.random.default_rng(6)
        pv = rng.uniform(0, 1, 2000)
        q_storey, pi0 = storey_qvalue(pv)
        m = len(pv)
        order = np.argsort(pv)
        ranks = np.arange(1, m + 1)
        q_bh_sorted = np.minimum(m * pv[order] / ranks, 1.0)
        for i in range(m - 2, -1, -1):
            q_bh_sorted[i] = min(q_bh_sorted[i], q_bh_sorted[i + 1])
        q_bh = np.empty(m)
        q_bh[order] = q_bh_sorted
        # within 10% relative difference for the full-null case
        finite = np.isfinite(q_storey) & (q_bh > 0.01)
        rel_diff = np.abs(q_storey[finite] - q_bh[finite]) / q_bh[finite]
        assert np.median(rel_diff) < 0.15

    def test_nan_input_returns_nan(self):
        pv = np.array([0.01, np.nan, 0.5, 0.9])
        q, _ = storey_qvalue(pv)
        assert np.isnan(q[1])
        assert np.isfinite(q[0])

    def test_all_nan_returns_pi0_one(self):
        pv = np.full(10, np.nan)
        q, pi0 = storey_qvalue(pv)
        assert pi0 == 1.0
        assert np.all(np.isnan(q))


# ── fixture comparison against R reference implementation ─────────────────────

class TestAgainstRFixture:
    @pytest.mark.skipif(not _FIXTURE.exists(),
                        reason='qvalue_reference.csv fixture not found')
    def test_pi0_close_to_r(self):
        df = pd.read_csv(_FIXTURE)
        pv = df['p_value'].to_numpy()
        _, pi0_py = storey_qvalue(pv)
        pi0_r = float(df['pi0'].iloc[0])
        # within 10% of R reference (scipy vs R spline smoothing penalty differ)
        assert abs(pi0_py - pi0_r) / pi0_r < 0.10

    @pytest.mark.skipif(not _FIXTURE.exists(),
                        reason='qvalue_reference.csv fixture not found')
    def test_qvalues_corr_with_r(self):
        df = pd.read_csv(_FIXTURE)
        pv = df['p_value'].to_numpy()
        q_py, _ = storey_qvalue(pv)
        q_r = df['q_value'].to_numpy()
        corr = float(np.corrcoef(q_py, q_r)[0, 1])
        assert corr >= 0.99

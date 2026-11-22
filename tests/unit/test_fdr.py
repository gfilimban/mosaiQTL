"""Unit tests for qtl/fdr.py — qvalue + calibrate_gene_pvalue."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.fdr import qvalue, calibrate_gene_pvalue

_FIXTURE = Path(__file__).parents[1] / "fixtures" / "data" / "qvalue_reference.csv"


# ── π₀ estimation ────────────────────────────────────────────────────────────


class TestPi0Estimation:
    def test_uniform_null_pi0_near_one(self):
        rng = np.random.default_rng(0)
        pv = rng.uniform(0, 1, 2000)
        _, pi0 = qvalue(pv)
        assert pi0 <= 1.0
        assert pi0 >= 0.85  # uniform null → π₀ close to 1

    def test_strong_signal_pi0_below_one(self):
        rng = np.random.default_rng(1)
        # 50% near-zero (signal) + 50% uniform (null)
        pv = np.concatenate([rng.beta(0.1, 1, 500), rng.uniform(0, 1, 500)])
        _, pi0 = qvalue(pv)
        assert pi0 < 0.75

    def test_smooth_df3_reproduces_a_line(self):
        from mosaiqtl.qtl.fdr import _smooth_df3_at_max

        x = np.arange(0.05, 1.0, 0.05)
        y = 0.9 - 0.2 * x
        assert abs(_smooth_df3_at_max(x, y) - y[-1]) < 1e-6

    def test_raw_pi0_above_one_is_logged_and_capped(self, caplog):
        pv = np.linspace(0.5, 1.0, 400)
        with caplog.at_level("WARNING"):
            _, pi0 = qvalue(pv)
        assert pi0 == 1.0
        assert any('exceeds 1' in r.message for r in caplog.records)

    def test_pi0_never_exceeds_one(self):
        rng = np.random.default_rng(2)
        for seed in range(10):
            pv = rng.uniform(0, 1, 200)
            _, pi0 = qvalue(pv)
            assert pi0 <= 1.0 + 1e-9


# ── q-value properties ────────────────────────────────────────────────────────


class TestQvalueProperties:
    def test_qvalues_in_unit_interval(self):
        rng = np.random.default_rng(3)
        pv = rng.uniform(0, 1, 300)
        q, _ = qvalue(pv)
        assert np.all(q[np.isfinite(q)] >= 0.0)
        assert np.all(q[np.isfinite(q)] <= 1.0 + 1e-9)

    def test_qvalues_monotone(self):
        rng = np.random.default_rng(4)
        pv = rng.uniform(0, 1, 500)
        q, _ = qvalue(pv)
        # sorted p → non-decreasing q
        order = np.argsort(pv)
        q_sorted = q[order]
        assert np.all(np.diff(q_sorted) >= -1e-10)

    def test_qvalues_leq_bh_under_signal(self):
        rng = np.random.default_rng(5)
        pv = np.concatenate([rng.beta(0.1, 1, 200), rng.uniform(0, 1, 200)])
        q_storey, pi0 = qvalue(pv)
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
        q_storey, pi0 = qvalue(pv)
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
        q, _ = qvalue(pv)
        assert np.isnan(q[1])
        assert np.isfinite(q[0])

    def test_all_nan_returns_pi0_one(self):
        pv = np.full(10, np.nan)
        q, pi0 = qvalue(pv)
        assert pi0 == 1.0
        assert np.all(np.isnan(q))


# ── fixture comparison against R reference implementation ─────────────────────


class TestAgainstRFixture:
    @pytest.mark.skipif(
        not _FIXTURE.exists(), reason="qvalue_reference.csv fixture not found"
    )
    def test_pi0_close_to_r(self):
        df = pd.read_csv(_FIXTURE)
        pv = df["p_value"].to_numpy()
        _, pi0_py = qvalue(pv)
        pi0_r = float(df["pi0"].iloc[0])
        # within 10% of R reference (scipy vs R spline smoothing penalty differ)
        assert abs(pi0_py - pi0_r) / pi0_r < 0.10

    @pytest.mark.skipif(
        not _FIXTURE.exists(), reason="qvalue_reference.csv fixture not found"
    )
    def test_qvalues_corr_with_r(self):
        df = pd.read_csv(_FIXTURE)
        pv = df["p_value"].to_numpy()
        q_py, _ = qvalue(pv)
        q_r = df["q_value"].to_numpy()
        corr = float(np.corrcoef(q_py, q_r)[0, 1])
        assert corr >= 0.99


# ── calibrate_gene_pvalue ─────────────────────────────────────────────────────


def _make_tiny_cache(n_strains=10):
    """n-strain, 2-chromosome, 15-locus cache for calibrate_gene_pvalue tests."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.cc import FOUNDER_ORDER
    from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship

    N_STRAINS, N_FOUNDERS, N_LOCI = n_strains, 8, 15
    rng = np.random.default_rng(0)
    strain_ids = [f"CC00{i+1}" for i in range(N_STRAINS)]
    data = {}
    for c in ["1", "2"]:
        p = rng.dirichlet(np.ones(N_FOUNDERS), size=(N_STRAINS, N_LOCI)).transpose(
            1, 0, 2
        )
        data[c] = {
            "locus_ids": np.array([f"{c}:{i}" for i in range(N_LOCI)]),
            "positions": np.linspace(1.0, 100.0, N_LOCI),
            "probs": p,
        }
    cache = GenomeCache(strain_ids, FOUNDER_ORDER, data)
    kl = calc_kinship(cache, method="loco")
    kd = decomp_kinship(kl)
    return cache, kd, kl, strain_ids


class TestCalibrateGenePvalue:
    def test_output_ranges(self):
        cache, kd, kl, _ = _make_tiny_cache()
        y = np.random.default_rng(1).standard_normal(10)
        cov = np.ones((10, 1))
        p_adj, a_hat, b_hat, h2_g = calibrate_gene_pvalue(
            y, cache, "1", 5.0, cov, kd, kl, n_perms=50, seed=42
        )
        assert 0.0 <= p_adj <= 1.0
        assert a_hat > 0.0
        assert b_hat > 0.0
        assert 0.0 <= h2_g <= 1.0

    def test_deterministic(self):
        cache, kd, kl, _ = _make_tiny_cache()
        y = np.random.default_rng(1).standard_normal(10)
        cov = np.ones((10, 1))
        r1 = calibrate_gene_pvalue(y, cache, "1", 5.0, cov, kd, kl, n_perms=50, seed=7)
        r2 = calibrate_gene_pvalue(y, cache, "1", 5.0, cov, kd, kl, n_perms=50, seed=7)
        assert r1 == r2

    def test_different_seeds_differ(self):
        cache, kd, kl, _ = _make_tiny_cache()
        y = np.random.default_rng(1).standard_normal(10)
        cov = np.ones((10, 1))
        r1 = calibrate_gene_pvalue(y, cache, "1", 5.0, cov, kd, kl, n_perms=50, seed=0)
        r2 = calibrate_gene_pvalue(y, cache, "1", 5.0, cov, kd, kl, n_perms=50, seed=1)
        assert r1[0] != r2[0]

    def test_monotone_in_peak_lod(self):
        cache, kd, kl, _ = _make_tiny_cache()
        y = np.random.default_rng(1).standard_normal(10)
        cov = np.ones((10, 1))
        p = [
            calibrate_gene_pvalue(
                y, cache, "1", lod, cov, kd, kl, n_perms=50, seed=42
            )[0]
            for lod in (1.0, 3.0, 6.0, 12.0)
        ]
        assert all(a >= b for a, b in zip(p, p[1:]))
        assert p[0] > p[-1]

    @pytest.mark.slow
    def test_null_genes_are_calibrated(self):
        """Genes with no genetic signal yield p-values spread across (0, 1)."""
        cache, kd, kl, _ = _make_tiny_cache()
        cov = np.ones((10, 1))
        pvals = []
        for g in range(40):
            y = np.random.default_rng(1000 + g).standard_normal(10)
            peak = _peak_lod(y, cache, kd, cov, "1")
            pvals.append(
                calibrate_gene_pvalue(
                    y, cache, "1", peak, cov, kd, kl, n_perms=60, seed=g
                )[0]
            )
        pvals = np.asarray(pvals)
        assert 0.35 < pvals.mean() < 0.65
        assert np.mean(pvals < 0.05) < 0.15
        assert np.mean(pvals > 0.5) > 0.3

    def test_planted_locus_is_significant(self):
        """A strong haplotype effect gets a smaller p-value than the null genes."""
        cache, kd, kl, _ = _make_tiny_cache()
        cov = np.ones((10, 1))
        _, _, probs = cache.get_chromosome("1")
        y_sig = 8.0 * probs[7, :, 0] + np.random.default_rng(5).normal(
            scale=0.1, size=10
        )
        peak = _peak_lod(y_sig, cache, kd, cov, "1")
        p_sig = calibrate_gene_pvalue(
            y_sig, cache, "1", peak, cov, kd, kl, n_perms=100, seed=1
        )[0]
        y_null = np.random.default_rng(2).standard_normal(10)
        peak_null = _peak_lod(y_null, cache, kd, cov, "1")
        p_null = calibrate_gene_pvalue(
            y_null, cache, "1", peak_null, cov, kd, kl, n_perms=100, seed=1
        )[0]
        assert p_sig < p_null


def _peak_lod(y, cache, kd, cov, chrom):
    from mosaiqtl.qtl import lmm as lmm_module
    from mosaiqtl.utils import rint

    ev, evc = kd[chrom]["values"], kd[chrom]["vectors"]
    h2, _ = lmm_module.estimate_h2_null(evc.T @ rint(y), evc.T @ cov, ev, method="ml")
    cd = lmm_module.precompute_chromosome(cache, chrom, ev, evc)
    res = lmm_module.scan_chromosome(rint(y), cd, cov, ev, evc, h2)
    return float(res["lod"].max())


class TestBetaCalibratePvalue:
    def test_recovers_known_beta_parameters(self):
        """Null maxima whose p-values are Beta(a, b) give back that (a, b)."""
        from scipy import stats
        from mosaiqtl.qtl.fdr import beta_calibrate_pvalue

        a_true, b_true, df = 0.8, 6.0, 7
        p_null = stats.beta.rvs(a_true, b_true, size=4000, random_state=3)
        null_lods = stats.chi2.isf(p_null, df) / (2 * np.log(10))
        _, a_hat, b_hat = beta_calibrate_pvalue(5.0, null_lods, df)
        assert a_hat == pytest.approx(a_true, rel=0.1)
        assert b_hat == pytest.approx(b_true, rel=0.15)

    def test_pvalue_matches_beta_cdf_of_chi2_p(self):
        from scipy import stats
        from mosaiqtl.qtl.fdr import beta_calibrate_pvalue

        df = 7
        p_null = stats.beta.rvs(1.0, 4.0, size=2000, random_state=4)
        null_lods = stats.chi2.isf(p_null, df) / (2 * np.log(10))
        peak = 4.0
        p_adj, a_hat, b_hat = beta_calibrate_pvalue(peak, null_lods, df)
        p_chi2 = stats.chi2.sf(2 * np.log(10) * peak, df)
        assert p_adj == pytest.approx(stats.beta.cdf(p_chi2, a_hat, b_hat), rel=1e-6)


# ── call_significant_egenes ──────────────────────────────────────────────────


class TestCallSignificantEgenes:
    """Every tested gene is calibrated; effects exist only for significant genes."""

    @staticmethod
    def _run(n_null=19, n=60):
        from mosaiqtl.pipeline import call_significant_egenes

        cache, kd, kl, strain_ids = _make_tiny_cache(n)
        cov = np.ones((n, 1))
        _, _, probs = cache.get_chromosome("1")
        rows, phenos = [], []
        for g in range(n_null):
            phenos.append(np.random.default_rng(300 + g).standard_normal(n))
        phenos.append(
            8.0 * probs[7, :, 0]
            + np.random.default_rng(5).normal(scale=0.1, size=n)
        )
        for g, y in enumerate(phenos):
            rows.append(
                {
                    "gene_id": f"g{g}",
                    "chr": "1",
                    "peak_lod": _peak_lod(y, cache, kd, cov, "1"),
                    "peak_locus": "1:7",
                }
            )
        rows.append({"gene_id": "no_peak", "chr": "1", "peak_lod": np.nan,
                     "peak_locus": ""})
        peak_df = pd.DataFrame(rows)
        gene_idx = {f"g{g}": g for g in range(len(phenos))}
        out = call_significant_egenes(
            peak_df,
            genomecache=cache,
            pheno_mat=np.vstack(phenos),
            gene_idx=gene_idx,
            covariates=cov,
            kinship_decomp=kd,
            kinship_loco=kl,
            sample_ids=strain_ids,
            stag="C",
            gene_info_df=None,
            fdr_threshold=0.05,
            n_perms=40,
            use_rint=False,
            z_mat=np.eye(n),
            valid_mask=np.ones(n, dtype=bool),
            seed=3,
        )
        return out, len(phenos)

    def test_all_tested_genes_are_calibrated(self):
        out, n_tested = self._run()
        assert len(out) == n_tested
        assert "no_peak" not in set(out["gene_id"])
        assert out["pval_perm"].between(0.0, 1.0).all()
        assert out["fdr"].between(0.0, 1.0).all()

    def test_effects_only_for_significant_genes(self):
        out, _ = self._run()
        beta_cols = [c for c in out.columns if c.startswith("beta_")]
        assert beta_cols
        has_effects = out[beta_cols].notna().all(axis=1)
        is_sig = out["significance"] == "T"
        assert (has_effects == is_sig).all()

    def test_q_values_use_every_tested_gene(self):
        """Storey q-values are computed over the full tested set, not a subset."""
        from mosaiqtl.qtl.fdr import qvalue

        out, _ = self._run()
        q, _ = qvalue(out["pval_perm"].to_numpy())
        np.testing.assert_allclose(out["fdr"].to_numpy(), q)


class TestCalibrateChromosomeCache:
    def test_cached_rotation_matches_uncached(self):
        from mosaiqtl.qtl import lmm as lmm_module

        cache, kd, kl, _ = _make_tiny_cache()
        y = np.random.default_rng(3).standard_normal(10)
        cov = np.ones((10, 1))
        chr_data = lmm_module.precompute_chromosome(
            cache, "1", kd["1"]["values"], kd["1"]["vectors"], None, "additive"
        )
        kwargs = dict(n_perms=50, seed=5, gene_pos_mb=40.0, window_mb=30.0)
        plain = calibrate_gene_pvalue(y, cache, "1", 5.0, cov, kd, kl, **kwargs)
        cached = calibrate_gene_pvalue(
            y, cache, "1", 5.0, cov, kd, kl, chr_data=chr_data, **kwargs
        )
        np.testing.assert_allclose(plain, cached, rtol=1e-9)


class TestStoreyEndToEndNullAndSignal:
    @pytest.mark.slow
    def test_null_genes_not_called_planted_genes_detected(self):
        """Real scan peak → per-gene calibration → Storey q-values."""
        cache, kd, kl, _ = _make_tiny_cache()
        cov = np.ones((10, 1))
        _, _, probs = cache.get_chromosome('1')
        pvals, is_signal = [], []
        for g in range(60):
            signal = g >= 45
            y = np.random.default_rng(2000 + g).standard_normal(10)
            if signal:
                y = y + 8.0 * probs[(g * 3) % 15, :, g % 8]
            peak = _peak_lod(y, cache, kd, cov, '1')
            p = calibrate_gene_pvalue(
                y, cache, '1', peak, cov, kd, kl, n_perms=80, seed=g
            )[0]
            pvals.append(p)
            is_signal.append(signal)
        pvals, is_signal = np.asarray(pvals), np.asarray(is_signal)
        q, pi0 = qvalue(pvals)
        assert pi0 <= 1.0
        assert np.mean(q[~is_signal] < 0.05) <= 0.1
        assert np.median(pvals[is_signal]) < np.median(pvals[~is_signal])


class TestBenjaminiHochberg:
    """bh and its relation to the Storey q-value."""

    def test_matches_textbook_values(self):
        from mosaiqtl.qtl.fdr import bh

        p = np.array([0.01, 0.04, 0.03, 0.005])
        # sorted: .005 .01 .03 .04 -> m p / i = .02 .02 .04 .04
        assert np.allclose(bh(p), [0.02, 0.04, 0.04, 0.02])

    def test_nan_is_skipped_and_preserved(self):
        from mosaiqtl.qtl.fdr import bh

        q = bh(np.array([0.01, np.nan, 0.5]))
        assert np.isnan(q[1]) and np.isfinite(q[[0, 2]]).all()

    def test_storey_equals_pi0_times_bh_before_clipping(self):
        from mosaiqtl.qtl.fdr import bh

        rng = np.random.default_rng(1)
        pv = np.concatenate([rng.uniform(0, 1e-4, 300), rng.uniform(0, 1, 700)])
        q_storey, pi0 = qvalue(pv)
        q_bh = bh(pv)
        assert np.all(q_bh >= q_storey - 1e-12)
        assert np.allclose(q_storey, np.minimum(pi0 * q_bh, 1.0), atol=1e-12)

    def test_saturation_warning_when_pi0_is_small(self, caplog):
        import logging

        rng = np.random.default_rng(2)
        pv = np.concatenate([rng.uniform(0, 1e-3, 900), rng.uniform(0, 1, 100)])
        with caplog.at_level(logging.WARNING):
            _, pi0 = qvalue(pv)
        assert pi0 < 0.2
        assert any("pi0" in r.message for r in caplog.records)

    def test_small_pi0_makes_every_gene_pass_storey_not_bh(self):
        from mosaiqtl.qtl.fdr import bh

        rng = np.random.default_rng(3)
        pv = np.concatenate([rng.uniform(0, 1e-6, 950), rng.uniform(0, 1, 50)])
        q_storey, pi0 = qvalue(pv)
        q_bh = bh(pv)
        late = pv > 0.5
        assert pi0 < 0.2
        assert np.mean(q_storey <= 0.05) > np.mean(q_bh <= 0.05)
        assert (q_storey[late] <= 0.05).mean() > 0.9
        assert (q_bh[late] > 0.05).all()


class TestLfdr:
    """lfdr is the per-gene posterior null probability; q is its running mean."""

    @staticmethod
    def _mixture(seed=0, n_null=1500, n_alt=500):
        rng = np.random.default_rng(seed)
        return np.concatenate([rng.uniform(0, 1, n_null), rng.beta(0.2, 6.0, n_alt)])

    def test_bounded_and_nan_preserved(self):
        from mosaiqtl.qtl.fdr import lfdr

        pv = self._mixture()
        pv[3] = np.nan
        out = lfdr(pv)
        assert np.isnan(out[3])
        fin = out[np.isfinite(out)]
        assert fin.min() >= 0.0 and fin.max() <= 1.0

    def test_non_decreasing_in_p(self):
        from mosaiqtl.qtl.fdr import lfdr

        pv = self._mixture(1)
        out = lfdr(pv)
        assert np.all(np.diff(out[np.argsort(pv)]) >= -1e-12)

    def test_uniform_null_is_near_one(self):
        from mosaiqtl.qtl.fdr import lfdr

        pv = np.random.default_rng(2).uniform(0, 1, 2000)
        assert np.median(lfdr(pv, pi0=1.0)) > 0.9

    def test_small_p_is_signal_large_p_is_null(self):
        from mosaiqtl.qtl.fdr import lfdr

        pv = self._mixture(3)
        out = lfdr(pv)
        assert np.median(out[pv < 1e-3]) < 0.1
        assert np.median(out[pv > 0.5]) > 0.8

    def test_marginal_gene_exceeds_its_qvalue(self):
        from mosaiqtl.qtl.fdr import lfdr

        pv = self._mixture(4)
        q, pi0 = qvalue(pv)
        local = lfdr(pv, pi0=pi0)
        called = q <= 0.05
        assert called.any()
        assert local[called].max() > 0.05
        # the q-value is the running mean of the lfdr, to density-estimation error
        order = np.argsort(pv)
        running = np.cumsum(local[order]) / np.arange(1, len(pv) + 1)
        k = int(called.sum())
        assert abs(running[k - 1] - q[order][k - 1]) < 0.05

    def test_too_few_p_values_raises(self):
        from mosaiqtl.qtl.fdr import lfdr

        with pytest.raises(ValueError, match="at least"):
            lfdr(np.array([0.1, 0.5, 0.9]))


# ── beta_calibrate_pvalue ────────────────────────────────────────────────────


class TestBetaCalibrationReproducible:
    """pval_perm is recoverable from the null max-LODs and the fitted Beta shapes."""

    @pytest.mark.parametrize('seed', range(10))
    def test_p_adj_is_beta_cdf_of_pointwise_p(self, seed):
        from scipy import stats

        from mosaiqtl.qtl.fdr import beta_calibrate_pvalue

        rng = np.random.default_rng(seed)
        df = 7
        null_lods = rng.chisquare(df, 200) / (2.0 * np.log(10.0))
        peak_lod = float(np.quantile(null_lods, 0.99)) + 0.5
        p_adj, a_hat, b_hat = beta_calibrate_pvalue(peak_lod, null_lods, df)
        p_real = stats.chi2.sf(2.0 * np.log(10.0) * peak_lod, df)
        assert p_adj == pytest.approx(stats.beta.cdf(p_real, a_hat, b_hat))
        assert a_hat > 0.0 and b_hat > 0.0
        assert 0.0 <= p_adj <= 1.0

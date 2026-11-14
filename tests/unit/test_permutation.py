"""Unit tests for qtl/permutation.py — permutation thresholds."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.permutation import (
    generate_null_outcomes,
    run_threshold_scans,
    KinshipSingularError,
    _draw_kinship_re,
    _null_fit_params,
)
from mosaiqtl.qtl.thresholds import get_gev_thresholds

# ── generate_null_outcomes ────────────────────────────────────────────────────


class TestPickRepGene:
    """Tests for pick_rep_gene — median-h² representative gene selection."""

    def _make_peak_df(self, h2_values: list, chroms: list | None = None) -> 'pd.DataFrame':
        import pandas as pd
        n = len(h2_values)
        return pd.DataFrame({
            'gene_id':   [f'ENSMUSG{i:011d}.1' for i in range(n)],
            'gene_symbol': [f'Gene{i}' for i in range(n)],
            'chr':       chroms if chroms else ['1'] * n,
            'start_mb':  [float(i) for i in range(n)],
            'h2':        h2_values,
        })

    def test_returns_median_h2_gene(self):
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.1, 0.3, 0.5, 0.7, 0.9])
        gene_id, chrom, start_mb, h2_med = pick_rep_gene(df)
        assert h2_med == pytest.approx(0.5)
        assert gene_id == 'ENSMUSG00000000002.1'  # h2=0.5 is index 2

    def test_chrom_filter_restricts_candidates(self):
        from mosaiqtl.qtl.permutation import pick_rep_gene
        # genes on chr1: h2=[0.1, 0.9]; genes on chr2: h2=[0.4, 0.6]
        df = self._make_peak_df([0.1, 0.4, 0.6, 0.9], chroms=['1', '2', '2', '1'])
        gene_id, chrom, start_mb, h2_med = pick_rep_gene(df, chrom='2')
        assert chrom == '2'
        assert h2_med == pytest.approx(0.5)  # median of [0.4, 0.6]

    def test_nan_h2_excluded(self):
        import numpy as np
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.1, float('nan'), 0.9])
        gene_id, chrom, start_mb, h2_med = pick_rep_gene(df)
        # median of [0.1, 0.9] = 0.5; nearest is either 0.1 or 0.9
        assert h2_med == pytest.approx(0.5)
        assert gene_id in ('ENSMUSG00000000000.1', 'ENSMUSG00000000002.1')

    def test_missing_columns_raises(self):
        import pandas as pd
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = pd.DataFrame({'gene_id': ['A'], 'chr': ['1'], 'h2': [0.5]})  # missing start_mb
        with pytest.raises(ValueError, match='missing columns'):
            pick_rep_gene(df)

    def test_all_nan_raises(self):
        import numpy as np
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([float('nan'), float('nan')])
        with pytest.raises(ValueError):
            pick_rep_gene(df)

    def test_chrom_filter_no_genes_raises(self):
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.3, 0.7], chroms=['1', '1'])
        with pytest.raises(ValueError):
            pick_rep_gene(df, chrom='99')

    def test_returns_correct_start_mb(self):
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.2, 0.5, 0.8])
        gene_id, chrom, start_mb, h2_med = pick_rep_gene(df)
        assert h2_med == pytest.approx(0.5)
        assert start_mb == pytest.approx(1.0)  # index 1 has start_mb=1.0

    def test_gene_tss_df_strand_aware_plus(self):
        """+ strand gene: TSS = start."""
        import pandas as pd
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.5])
        tss = pd.DataFrame({
            'gene_id': ['ENSMUSG00000000000.1'],
            'chr': ['1'], 'start': [2_000_000], 'end': [5_000_000], 'strand': ['+'],
        })
        _, _, tss_mb, _ = pick_rep_gene(df, gene_tss_df=tss)
        assert tss_mb == pytest.approx(2.0)

    def test_gene_tss_df_strand_aware_minus(self):
        """− strand gene: TSS = end."""
        import pandas as pd
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.5])
        tss = pd.DataFrame({
            'gene_id': ['ENSMUSG00000000000.1'],
            'chr': ['1'], 'start': [2_000_000], 'end': [5_000_000], 'strand': ['-'],
        })
        _, _, tss_mb, _ = pick_rep_gene(df, gene_tss_df=tss)
        assert tss_mb == pytest.approx(5.0)

    def test_gene_tss_df_precomputed_tss_mb(self):
        """Pre-computed tss_mb column is used directly (no re-derivation)."""
        import pandas as pd
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.5])
        tss = pd.DataFrame({
            'gene_id': ['ENSMUSG00000000000.1'],
            'chr': ['1'], 'start': [2_000_000], 'end': [5_000_000],
            'strand': ['-'], 'tss_mb': [3.7],
        })
        _, _, tss_mb, _ = pick_rep_gene(df, gene_tss_df=tss)
        assert tss_mb == pytest.approx(3.7)

    def test_gene_tss_df_missing_gene_falls_back_to_start_mb(self):
        """Gene absent from gene_tss_df falls back to start_mb."""
        import pandas as pd
        from mosaiqtl.qtl.permutation import pick_rep_gene
        df = self._make_peak_df([0.5])
        tss = pd.DataFrame({'gene_id': ['ENSMUSG99999999999.1'], 'chr': ['1'],
                             'start': [1_000], 'end': [2_000], 'strand': ['+']})
        _, _, tss_mb, _ = pick_rep_gene(df, gene_tss_df=tss)
        assert tss_mb == pytest.approx(0.0)  # start_mb of index 0

    def test_global_median_differs_from_chrom_restricted(self):
        """chrom=None picks from all chromosomes; chrom='19' restricts to that chr.

        9 genes: 5 on chr1 with h2 in [0.40, 0.60], 4 on chr19 with h2 in
        [0.00, 0.09].  Global median (5th of 9 sorted values) = 0.40, which
        is a chr1 gene.  chr19-restricted median = (0.03+0.06)/2 = 0.045,
        nearest chr19 gene is NOT the same gene.  This cannot be distinguished
        on a single-chromosome run (the parity test), where chrom=None and
        chrom='19' draw from the same pool.
        """
        import pandas as pd
        from mosaiqtl.qtl.permutation import pick_rep_gene

        h2 = [0.40, 0.45, 0.50, 0.55, 0.60,   # chr1 — all above 0.25
              0.00, 0.03, 0.06, 0.09]            # chr19 — all below 0.25
        df = pd.DataFrame({
            'gene_id':    [f'ENSMUSG{i:011d}.1' for i in range(9)],
            'gene_symbol': [f'Gene{i}' for i in range(9)],
            'chr':        ['1'] * 5 + ['19'] * 4,
            'start_mb':   [float(i) for i in range(9)],
            'h2':         h2,
        })

        global_id, global_chr, _, _ = pick_rep_gene(df, chrom=None)
        restricted_id, restricted_chr, _, _ = pick_rep_gene(df, chrom='19')

        # sorted h2: [0.00,0.03,0.06,0.09,0.40,0.45,0.50,0.55,0.60]
        # global median = 0.40 → gene index 0 on chr1 (exact match)
        assert global_chr == '1', f'global pick should land on chr1, got chr{global_chr}'
        assert restricted_chr == '19', f'chr19-restricted pick should stay on chr19'
        assert global_id != restricted_id, 'global and restricted must select different genes'


class TestGenerateNullOutcomes:
    def test_permutation_shape(self):
        rng = np.random.default_rng(0)
        y = rng.standard_normal(20)
        null = generate_null_outcomes(y, n_perms=50, method="permutation")
        assert null.shape == (20, 50)

    def test_bootstrap_shape(self):
        rng = np.random.default_rng(1)
        y = rng.standard_normal(15)
        null = generate_null_outcomes(y, n_perms=30, method="bootstrap")
        assert null.shape == (15, 30)

    def test_permutation_preserves_values(self):
        y = np.arange(10, dtype=float)
        null = generate_null_outcomes(y, n_perms=20, method="permutation")
        for col in range(null.shape[1]):
            assert set(null[:, col]) == set(y)

    def test_bootstrap_centred(self):
        rng = np.random.default_rng(2)
        y = rng.standard_normal(50) + 10.0  # large non-zero mean
        null = generate_null_outcomes(y, n_perms=100, method="bootstrap")
        # each column is drawn from centred distribution
        assert null.mean() == pytest.approx(0.0, abs=2.0)

    def test_invalid_method_raises(self):
        y = np.ones(10)
        with pytest.raises(ValueError, match="method"):
            generate_null_outcomes(y, n_perms=5, method="invalid")

    def test_reproducible_with_seed(self):
        rng = np.random.default_rng(3)
        y = rng.standard_normal(20)
        a = generate_null_outcomes(y, n_perms=10, seed=42)
        b = generate_null_outcomes(y, n_perms=10, seed=42)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        rng = np.random.default_rng(4)
        y = rng.standard_normal(20)
        a = generate_null_outcomes(y, n_perms=10, seed=1)
        b = generate_null_outcomes(y, n_perms=10, seed=2)
        assert not np.array_equal(a, b)


# ── get_gev_thresholds ────────────────────────────────────────────────────────


class TestGetGevThresholds:
    def test_returns_float_and_dict(self):
        rng = np.random.default_rng(10)
        max_lods = rng.exponential(2.0, 200)
        thresh, params = get_gev_thresholds(max_lods)
        assert isinstance(thresh, float)
        assert isinstance(params, dict)

    def test_threshold_above_median(self):
        rng = np.random.default_rng(11)
        max_lods = rng.exponential(2.0, 200)
        thresh, _ = get_gev_thresholds(max_lods, percentile=0.95)
        assert thresh > float(np.median(max_lods))

    def test_params_keys(self):
        rng = np.random.default_rng(12)
        max_lods = rng.exponential(2.0, 100)
        _, params = get_gev_thresholds(max_lods)
        for key in (
            "c",
            "loc",
            "scale",
            "threshold",
            "percentile",
            "n_perms",
            "empirical_quantile",
        ):
            assert key in params

    def test_too_few_values_returns_none(self):
        thr, params = get_gev_thresholds(np.array([1.0, 2.0, 3.0]))
        assert thr is None and params is None

    def test_higher_percentile_higher_threshold(self):
        rng = np.random.default_rng(13)
        max_lods = rng.exponential(2.0, 300)
        t90, _ = get_gev_thresholds(max_lods, percentile=0.90)
        t95, _ = get_gev_thresholds(max_lods, percentile=0.95)
        assert t95 >= t90

    def test_ignores_inf(self):
        rng = np.random.default_rng(14)
        max_lods = rng.exponential(2.0, 100)
        max_lods_inf = np.append(max_lods, [np.inf, np.nan])
        thresh, params = get_gev_thresholds(max_lods_inf)
        assert np.isfinite(thresh)
        assert params["n_perms"] == 100


# ── run_threshold_scans ───────────────────────────────────────────────────────


class TestRunThresholdScans:
    def test_returns_array(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        n_perms = 5
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=n_perms, seed=42)
        max_lods, chr_nulls = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            use_rint=False,
        )
        assert max_lods.shape == (n_perms,)
        assert isinstance(chr_nulls, dict)

    def test_max_lods_nonnegative(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=3, seed=1)
        max_lods, _ = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            use_rint=False,
        )
        assert np.all(max_lods >= -1e-8)

    def test_single_chrom_mode(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=3, seed=2)
        max_lods, chr_nulls = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            chrom="1",
            use_rint=False,
        )
        assert max_lods.shape == (3,)
        assert list(chr_nulls.keys()) == ["1"]

    def test_no_phenotype_raises(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=2, seed=4)
        with pytest.raises(ValueError, match="phenotype required"):
            run_threshold_scans(
                null,
                tiny_cache,
                kinship_decomp=kd,
                kinship_loco=tiny_kinship,
                phenotype=None,
            )

    # ── local-window (gene_pos_mb/window_mb) wiring ──────────────────────

    def test_default_path_unaffected_by_new_params(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """Explicitly passing gene_pos_mb=None/window_mb=None (the new
        defaults) must reproduce byte-identical output to the pre-existing
        call signature that never mentioned them — i.e. adding the local-
        window option did not alter default (genome-wide) behavior."""
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=5, seed=42)

        before = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            use_rint=False,
        )
        after = run_threshold_scans(
            null,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            use_rint=False,
            gene_pos_mb=None,
            window_mb=None,
        )
        np.testing.assert_array_equal(before[0], after[0])
        assert before[1].keys() == after[1].keys()
        for c in before[1]:
            np.testing.assert_array_equal(before[1][c], after[1][c])

    def test_gene_pos_mb_without_chrom_raises(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        """gene_pos_mb/window_mb anchor a window to one chromosome — without
        chrom there is no chromosome to anchor to, so this must fail loudly
        rather than silently scanning genome-wide."""
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=2, seed=5)
        with pytest.raises(ValueError, match="chrom"):
            run_threshold_scans(
                null,
                tiny_cache,
                phenotype=tiny_strain_phenotype,
                kinship_decomp=kd,
                kinship_loco=tiny_kinship,
                chrom=None,
                gene_pos_mb=1.0,
                window_mb=0.1,
                use_rint=False,
            )

    def test_local_window_restricts_loci_vs_genome_wide(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        """A local window covering exactly one of tiny_cache's 15 loci on
        chr '1' must scan strictly fewer loci than the whole-chromosome null
        — proving loci_mask is actually reaching scan_chromosome_batch, not
        silently ignored. Positions are np.linspace(1.0, 100.0, 15), so a
        window of gene_pos_mb=1.0 +/- 0.1 admits only the first locus."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from unittest.mock import patch
        from mosaiqtl.qtl import lmm as lmm_module

        kd = decomp_kinship(tiny_kinship)
        null = generate_null_outcomes(tiny_strain_phenotype, n_perms=3, seed=6)

        seen_masks = []
        real_scan = lmm_module.scan_chromosome_batch

        def _spy(*args, **kwargs):
            seen_masks.append(kwargs.get("loci_mask"))
            return real_scan(*args, **kwargs)

        with patch.object(lmm_module, "scan_chromosome_batch", side_effect=_spy):
            run_threshold_scans(
                null,
                tiny_cache,
                phenotype=tiny_strain_phenotype,
                kinship_decomp=kd,
                kinship_loco=tiny_kinship,
                chrom="1",
                gene_pos_mb=1.0,
                window_mb=0.1,
                use_rint=False,
            )

        assert len(seen_masks) >= 1
        mask = seen_masks[0]
        assert mask is not None
        assert mask.sum() == 1  # only the locus at position 1.0


# ── parametric permutation helpers ───────────────────────────────────────────


class TestDrawParametricNullBatchRankMatching:
    def test_output_is_rank_matched_to_null_draw(self):
        from mosaiqtl.qtl.permutation import draw_parametric_null_batch

        n = 40
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n) ** 3
        x = np.ones((n, 1))
        K, ev, evc = np.eye(n), np.ones(n), np.eye(n)
        out = draw_parametric_null_batch(
            y, x, K, evc, ev, 0.5, 5, np.random.default_rng(1)
        )
        assert out.shape == (n, 5)
        for j in range(5):
            assert np.allclose(np.sort(out[:, j]), np.sort(y))
        assert not np.allclose(out[:, 0], out[:, 1])

    def test_rank_order_follows_null_draw(self):
        from mosaiqtl.qtl.permutation import draw_parametric_null_batch

        n = 200
        y = np.arange(n, dtype=float)
        x = np.ones((n, 1))
        K, ev, evc = np.eye(n), np.ones(n), np.eye(n)
        # h2 = 1 with a strong shared component: re-running with the same seed
        # must give the identical ordering, and the output stays a permutation of y
        a = draw_parametric_null_batch(y, x, K, evc, ev, 0.9, 3, np.random.default_rng(7))
        b = draw_parametric_null_batch(y, x, K, evc, ev, 0.9, 3, np.random.default_rng(7))
        assert np.array_equal(a, b)
        assert sorted(a[:, 0]) == sorted(y)


class TestParametricHelpers:
    """unit tests for _null_fit_params, _draw_kinship_re, _parametric_null_col."""

    def _make_psd_kinship(self, n, seed=0):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        K = A @ A.T / n + np.eye(n) * 0.1
        return K

    def test_null_fit_params_shape(self):
        rng = np.random.default_rng(0)
        n, p = 20, 2
        y_rot = rng.standard_normal(n)
        x_rot = np.column_stack([np.ones(n), rng.standard_normal(n)])
        eigvals = np.abs(rng.standard_normal(n)) + 0.1
        beta, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, eigvals, h2=0.3)
        assert beta.shape == (p,)
        assert sigma2_g > 0
        assert sigma2_e > 0

    def test_null_fit_params_variance_partition(self):
        rng = np.random.default_rng(1)
        n = 30
        y_rot = rng.standard_normal(n)
        x_rot = np.ones((n, 1))
        eigvals = np.ones(n)
        h2 = 0.4
        _, sigma2_g, sigma2_e = _null_fit_params(y_rot, x_rot, eigvals, h2=h2)
        # sigma2_g / (sigma2_g + sigma2_e) should equal h2
        assert abs(sigma2_g / (sigma2_g + sigma2_e) - h2) < 1e-10

    def test_draw_kinship_re_shape(self):
        rng = np.random.default_rng(42)
        n = 15
        K = self._make_psd_kinship(n)
        u = _draw_kinship_re(K, sigma2_g=0.5, rng=rng)
        assert u.shape == (n,)

    def test_draw_kinship_re_singular_raises(self):
        rng = np.random.default_rng(0)
        # genuinely non-PSD: shift eigenvalues negative by subtracting 0.5I
        # min_eig ≈ -0.4 >> psd_tol (≈ max_eig * n² * eps << 1)
        n = 10
        A = rng.standard_normal((n, n))
        K_bad = A @ A.T / n - 0.5 * np.eye(n)
        with pytest.raises(KinshipSingularError):
            _draw_kinship_re(K_bad, sigma2_g=1.0, rng=rng)

    def test_run_threshold_scans_parametric_shape(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """parametric mode returns (n_perms,) max-LOD array."""
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        n_perms = 5
        # null_outcomes used only for n_perms in parametric mode
        dummy = np.empty((len(tiny_strain_phenotype), n_perms))
        c0 = list(tiny_cache.chromosomes)[0]
        max_lods, chr_nulls = run_threshold_scans(
            dummy,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            chrom=c0,
            permutation_method="parametric",
            use_rint=False,
            seed=99,
        )
        assert max_lods.shape == (n_perms,)
        assert np.all(max_lods >= 0)
        assert c0 in chr_nulls and chr_nulls[c0].shape == (n_perms,)

    def test_run_threshold_scans_parametric_without_kinship_loco(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        n_perms = 3
        dummy = np.empty((len(tiny_strain_phenotype), n_perms))
        max_lods, _ = run_threshold_scans(
            dummy,
            tiny_cache,
            phenotype=tiny_strain_phenotype,
            kinship_decomp=kd,
            permutation_method="parametric",
        )
        assert max_lods.shape == (n_perms,)
        assert np.all(max_lods >= 0)


# ── calibrate_gene_pvalue ─────────────────────────────────────────────────────


class TestRunThresholdScansReferenceAndSampleLevel:
    def test_ref_chrom_selects_reference_draw(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        """The reference null is drawn from ref_chrom's LOCO kinship, so
        changing ref_chrom changes the draw while a repeated call is exact."""
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        chrs = list(tiny_cache.chromosomes)
        dummy = np.empty((len(tiny_strain_phenotype), 8))

        def _run(ref):
            return run_threshold_scans(
                dummy,
                tiny_cache,
                phenotype=tiny_strain_phenotype,
                kinship_decomp=kd,
                kinship_loco=tiny_kinship,
                ref_chrom=ref,
                permutation_method="parametric",
                use_rint=False,
                seed=5,
            )[0]

        a1, a2, b = _run(chrs[0]), _run(chrs[0]), _run(chrs[1])
        np.testing.assert_array_equal(a1, a2)
        assert not np.allclose(a1, b)

    def test_z_mat_scans_at_sample_level(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        """With replicate samples and z_mat the null is drawn and scanned at
        n samples (n = 2m), not collapsed to m strain means."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.utils import build_z_matrix

        kd = decomp_kinship(tiny_kinship)
        strains = list(tiny_cache.sample_ids)
        Z, _ = build_z_matrix(strains + strains, strains)
        n = 2 * len(strains)
        rng = np.random.default_rng(0)
        y = np.tile(tiny_strain_phenotype, 2) + rng.normal(0, 0.3, n)
        max_lods, chr_nulls = run_threshold_scans(
            np.empty((n, 6)),
            tiny_cache,
            phenotype=y,
            covariates=np.ones((n, 1)),
            kinship_decomp=kd,
            kinship_loco=tiny_kinship,
            z_mat=Z,
            permutation_method="parametric",
            use_rint=False,
            seed=3,
        )
        assert max_lods.shape == (6,)
        assert np.all(np.isfinite(max_lods)) and np.all(max_lods >= 0)
        assert set(chr_nulls) == set(tiny_cache.chromosomes)


class TestCalibrateGenePvalue:
    """calibrate_gene_pvalue: per-gene Beta-calibrated p-value (SWE-09)."""

    def test_returns_tuple_of_four(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """Returns (p_adj, a_hat, b_hat, h2) — all finite scalars."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.fdr import calibrate_gene_pvalue

        kd = decomp_kinship(tiny_kinship)
        c0 = list(tiny_cache.chromosomes)[0]
        result = calibrate_gene_pvalue(
            tiny_strain_phenotype,
            tiny_cache,
            c0,
            2.0,
            tiny_strain_covariates,
            kd,
            tiny_kinship,
            n_perms=20,
            use_rint=False,
            seed=7,
        )
        assert len(result) == 4
        p_adj, a_hat, b_hat, h2_g = result
        for val in (p_adj, a_hat, b_hat, h2_g):
            assert np.isfinite(val), f"expected finite, got {val}"

    def test_p_adj_in_unit_interval(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """p_adj must be in [0, 1]."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.fdr import calibrate_gene_pvalue

        kd = decomp_kinship(tiny_kinship)
        c0 = list(tiny_cache.chromosomes)[0]
        p_adj, *_ = calibrate_gene_pvalue(
            tiny_strain_phenotype,
            tiny_cache,
            c0,
            1.5,
            tiny_strain_covariates,
            kd,
            tiny_kinship,
            n_perms=20,
            use_rint=False,
            seed=11,
        )
        assert 0.0 <= p_adj <= 1.0

    def test_beta_calibrate_pvalue_monotone(self):
        """beta_calibrate_pvalue (the inner function) gives smaller p for higher LOD.

        The full calibrate_gene_pvalue monotone property cannot be reliably
        demonstrated on the 10-strain tiny fixture — with such small n, the
        parametric null easily produces LODs as high as the observed value by
        chance, saturating p_adj at 1.0.  Instead we test the Beta-calibration
        step directly with fabricated null_max_lods drawn from a controlled
        distribution where the ordering is guaranteed.
        """
        from mosaiqtl.qtl.fdr import beta_calibrate_pvalue

        rng = np.random.default_rng(99)
        null_lods = rng.gamma(shape=2.0, scale=0.8, size=200)
        p_low, *_ = beta_calibrate_pvalue(0.5, null_lods, df=7)
        p_high, *_ = beta_calibrate_pvalue(8.0, null_lods, df=7)
        assert p_high < p_low, (
            f"higher LOD should give smaller p: p(LOD=8)={p_high:.4f} "
            f"vs p(LOD=0.5)={p_low:.4f}"
        )

    def test_reproducible_with_seed(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """Same seed → identical p_adj."""
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.fdr import calibrate_gene_pvalue

        kd = decomp_kinship(tiny_kinship)
        c0 = list(tiny_cache.chromosomes)[0]
        args = (tiny_strain_phenotype, tiny_cache, c0, 2.0, tiny_strain_covariates, kd, tiny_kinship)
        kwargs = dict(n_perms=15, use_rint=False, seed=42)
        p1, *_ = calibrate_gene_pvalue(*args, **kwargs)
        p2, *_ = calibrate_gene_pvalue(*args, **kwargs)
        assert p1 == p2

"""Unit tests for viz/validation.py — Phase 2 QC plots."""

import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.viz.validation import (
    set_style,
    plot_lod_profile,
    plot_kinship_heatmap,
    plot_founder_effects,
    plot_pvalue_qq,
    plot_dispersion_vs_truth,
    plot_lod_correlation,
    plot_lod_difference,
    plot_founder_effects_comparison,
    plot_validation_panel,
    plot_permutation_comparison,
    plot_mi_scan_comparison,
    plot_mediation_comparison,
    plot_instability_comparison,
    plot_phi_distribution,
    plot_nb_vs_lmm_comparison,
)
from mosaiqtl.qtl.scan import ScanResult

N_LOCI     = 20
N_FOUNDERS = 8


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close('all')


def _make_scan_result(n_loci=N_LOCI, n_founders=N_FOUNDERS):
    rng = np.random.default_rng(42)
    n_half = n_loci // 2
    lods = np.abs(rng.standard_normal(n_loci))
    return ScanResult(
        locus_ids   = np.array([f'{c}_{i}' for c in ('1', '2')
                                for i in range(n_half)]),
        chromosomes = np.array(['1'] * n_half + ['2'] * (n_loci - n_half)),
        positions   = np.linspace(1e6, 100e6, n_loci),
        lod         = lods,
        p_value     = np.clip(rng.uniform(0.001, 0.5, n_loci), 1e-10, 1),
        betas       = rng.standard_normal((n_loci, n_founders - 1)),
        h2_null     = {'1': 0.3, '2': 0.2},
        phenotype_id = 'test_gene',
    )


# ── set_style ─────────────────────────────────────────────────────────────────

class TestSetStyle:
    def test_runs_without_error(self):
        set_style()

    def test_activates_seaborn_style(self):
        set_style()
        # seaborn whitegrid adds axes.grid=True to rcParams
        assert plt.rcParams.get('axes.grid', False) is True


# ── plot_lod_profile ──────────────────────────────────────────────────────────

class TestPlotLodProfile:
    def test_returns_fig_ax(self):
        fig, ax = plot_lod_profile(_make_scan_result())
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_with_true_params(self):
        true_params = {'causal_chrom': '1', 'causal_locus_idx': 3}
        fig, ax = plot_lod_profile(_make_scan_result(), true_params=true_params)
        assert isinstance(fig, plt.Figure)

    def test_with_save_path(self, tmp_path):
        save_path = str(tmp_path / 'lod_profile.png')
        fig, ax = plot_lod_profile(_make_scan_result(), save_path=save_path)
        assert Path(save_path).exists()


# ── plot_kinship_heatmap ──────────────────────────────────────────────────────

class TestPlotKinshipHeatmap:
    def test_returns_fig_ax(self):
        rng = np.random.default_rng(0)
        K = rng.random((10, 10))
        K = (K + K.T) / 2
        fig, ax = plot_kinship_heatmap(K)
        assert isinstance(fig, plt.Figure)

    def test_with_sample_ids(self):
        rng = np.random.default_rng(1)
        K = rng.random((8, 8))
        # include a few founder names to trigger block line drawing
        from mosaiqtl.genotype.founders import FOUNDER_ORDER
        sample_ids = FOUNDER_ORDER[:5] + ['CC001', 'CC002', 'CC003']
        fig, ax = plot_kinship_heatmap(K, sample_ids=sample_ids)
        assert isinstance(fig, plt.Figure)

    def test_uses_seaborn_heatmap(self):
        rng = np.random.default_rng(2)
        K = rng.random((6, 6))
        fig, ax = plot_kinship_heatmap(K)
        # sns.heatmap renders a QuadMesh (or AxesImage) on the axes
        assert len(ax.collections) > 0 or len(ax.images) > 0


# ── plot_founder_effects ──────────────────────────────────────────────────────

class TestPlotFounderEffects:
    def test_returns_fig_ax(self, tiny_cache, tiny_strain_phenotype,
                             tiny_kinship):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        kd = decomp_kinship(tiny_kinship)
        scan_result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
        )
        fig, ax = plot_founder_effects(scan_result, tiny_cache)
        assert isinstance(fig, plt.Figure)

    def test_with_peak_chrom_idx(self, tiny_cache, tiny_strain_phenotype,
                                  tiny_kinship):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        kd = decomp_kinship(tiny_kinship)
        scan_result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
        )
        fig, ax = plot_founder_effects(
            scan_result, tiny_cache, peak_chrom='1', peak_idx=0,
        )
        assert isinstance(fig, plt.Figure)


# ── plot_pvalue_qq ────────────────────────────────────────────────────────────

class TestPlotPvalueQq:
    def test_returns_fig_ax(self):
        fig, ax = plot_pvalue_qq(_make_scan_result())
        assert isinstance(fig, plt.Figure)


# ── plot_dispersion_vs_truth ──────────────────────────────────────────────────

class TestPlotDispersionVsTruth:
    def test_returns_fig_ax(self):
        rng = np.random.default_rng(0)
        n = 40
        phi_true = rng.exponential(0.3, n)
        phi_est  = phi_true + rng.standard_normal(n) * 0.05
        fig, ax = plot_dispersion_vs_truth(phi_est, phi_true)
        assert isinstance(fig, plt.Figure)

    def test_handles_nan(self):
        rng = np.random.default_rng(1)
        n = 20
        phi_true = rng.exponential(0.3, n)
        phi_est  = phi_true.copy()
        phi_est[0] = np.nan
        phi_true[1] = np.nan
        fig, ax = plot_dispersion_vs_truth(phi_est, phi_true)
        assert isinstance(fig, plt.Figure)


# ── plot_lod_correlation ──────────────────────────────────────────────────────

class TestPlotLodCorrelation:
    def test_returns_fig_ax_with_matching_csv(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        # build a fake R CSV with matching locus IDs and lod column
        r_df = pd.DataFrame({
            'locus': sr.locus_ids,
            'lod':   np.abs(np.random.default_rng(5).standard_normal(N_LOCI)),
        })
        r_csv = str(tmp_path / 'r_scan.csv')
        r_df.to_csv(r_csv, index=False)
        fig, ax = plot_lod_correlation(sr, r_csv)
        assert isinstance(fig, plt.Figure)

    def test_returns_none_on_no_match(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        r_df = pd.DataFrame({'locus': ['no_match'], 'lod': [0.5]})
        r_csv = str(tmp_path / 'r_empty.csv')
        r_df.to_csv(r_csv, index=False)
        fig, ax = plot_lod_correlation(sr, r_csv)
        assert fig is None and ax is None


# ── plot_lod_difference ───────────────────────────────────────────────────────

class TestPlotLodDifference:
    def test_returns_fig_ax(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        # omit 'chromosome' — py_df already has it; including it would rename to
        # 'chromosome_py' and break merged['chromosome'] in the source function
        r_df = pd.DataFrame({'locus': sr.locus_ids, 'lod': sr.lod * 0.9})
        r_csv = str(tmp_path / 'r_scan2.csv')
        r_df.to_csv(r_csv, index=False)
        fig, ax = plot_lod_difference(sr, r_csv)
        assert isinstance(fig, plt.Figure)

    def test_returns_none_no_match(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        r_df = pd.DataFrame({'locus': ['none'], 'lod': [0.0]})
        r_csv = str(tmp_path / 'r_empty2.csv')
        r_df.to_csv(r_csv, index=False)
        fig, ax = plot_lod_difference(sr, r_csv)
        assert fig is None and ax is None


# ── plot_founder_effects_comparison ──────────────────────────────────────────

class TestPlotFounderEffectsComparison:
    def test_python_only(self, tiny_cache, tiny_strain_phenotype, tiny_kinship,
                          tmp_path):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
        )
        r_csv = str(tmp_path / 'nonexistent.csv')  # file doesn't exist
        fig, ax = plot_founder_effects_comparison(sr, tiny_cache, r_csv)
        assert isinstance(fig, plt.Figure)

    def test_with_r_effects_csv(self, tiny_cache, tiny_strain_phenotype,
                                 tiny_kinship, tmp_path):
        import pandas as pd
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        from mosaiqtl.genotype.founders import FOUNDER_ORDER
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
        )
        peak = sr.peak_locus()
        r_df = pd.DataFrame({
            'locus':   [peak['locus_id']] * 8,
            'founder': FOUNDER_ORDER,
            'effect':  np.random.default_rng(0).standard_normal(8),
        })
        r_csv = str(tmp_path / 'r_effects.csv')
        r_df.to_csv(r_csv, index=False)
        fig, ax = plot_founder_effects_comparison(sr, tiny_cache, r_csv)
        assert isinstance(fig, plt.Figure)


# ── plot_validation_panel ─────────────────────────────────────────────────────

class TestPlotValidationPanel:
    def test_internal_plots_only(self, tiny_cache, tiny_strain_phenotype,
                                  tiny_kinship, tmp_path):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
        )
        K = tiny_kinship['1']
        save_dir = str(tmp_path / 'val_plots')
        plot_validation_panel(sr, K, tiny_cache, save_dir=save_dir)
        assert Path(save_dir).exists()

    def test_with_dispersion_arrays(self, tiny_cache, tiny_strain_phenotype,
                                     tiny_kinship, tmp_path):
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.scan import scan_h2lmm
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd, use_rint=False,
        )
        K = tiny_kinship['1']
        rng = np.random.default_rng(0)
        phi_t = rng.exponential(0.3, 10)
        phi_e = phi_t + rng.standard_normal(10) * 0.05
        save_dir = str(tmp_path / 'val_plots2')
        plot_validation_panel(sr, K, tiny_cache,
                              phi_estimated=phi_e, phi_true=phi_t,
                              save_dir=save_dir)
        assert Path(save_dir).exists()


# ── plot_permutation_comparison ───────────────────────────────────────────────

class TestPlotPermutationComparison:
    def test_returns_fig_ax(self, tmp_path):
        import pandas as pd
        csv_path = str(tmp_path / 'r_thresh.csv')
        pd.DataFrame({'alpha': [0.05, 0.10], 'threshold': [3.5, 2.8]}).to_csv(
            csv_path, index=False)
        fig, ax = plot_permutation_comparison(
            {0.05: 3.6, 0.10: 2.9}, csv_path, alpha=0.05)
        assert fig is not None and ax is not None

    def test_saves_file(self, tmp_path):
        import pandas as pd
        csv_path = str(tmp_path / 'r_thresh2.csv')
        pd.DataFrame({'alpha': [0.05], 'threshold': [3.5]}).to_csv(
            csv_path, index=False)
        out_path = str(tmp_path / 'perm_cmp.png')
        plot_permutation_comparison({0.05: 3.6}, csv_path,
                                     alpha=0.05, save_path=out_path)
        assert Path(out_path).exists()

    def test_unmatched_alpha_still_runs(self, tmp_path):
        import pandas as pd
        csv_path = str(tmp_path / 'r_thresh3.csv')
        pd.DataFrame({'alpha': [0.10], 'threshold': [2.8]}).to_csv(
            csv_path, index=False)
        fig, ax = plot_permutation_comparison({0.05: 3.6}, csv_path, alpha=0.05)
        assert fig is not None


# ── plot_mi_scan_comparison ───────────────────────────────────────────────────

class TestPlotMiScanComparison:
    def test_returns_none_on_no_shared_loci(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        csv_path = str(tmp_path / 'r_mi.csv')
        pd.DataFrame({'locus': ['X_99'], 'lod': [1.0]}).to_csv(csv_path, index=False)
        fig, ax = plot_mi_scan_comparison(sr, csv_path)
        assert fig is None and ax is None

    def test_returns_fig_ax_with_shared_loci(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        lids = list(sr.locus_ids[:5])
        csv_path = str(tmp_path / 'r_mi2.csv')
        pd.DataFrame({'locus': lids, 'lod': np.abs(sr.lod[:5]) + 0.1}).to_csv(
            csv_path, index=False)
        fig, ax = plot_mi_scan_comparison(sr, csv_path)
        assert fig is not None and ax is not None

    def test_saves_file(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        lids = list(sr.locus_ids)
        csv_path = str(tmp_path / 'r_mi3.csv')
        pd.DataFrame({'locus': lids, 'lod': np.abs(sr.lod) + 0.1}).to_csv(
            csv_path, index=False)
        out_path = str(tmp_path / 'mi_cmp.png')
        plot_mi_scan_comparison(sr, csv_path, save_path=out_path)
        assert Path(out_path).exists()


# ── plot_mediation_comparison ─────────────────────────────────────────────────

class TestPlotMediationComparison:
    def _make_mediation_df(self, sr):
        import pandas as pd
        return pd.DataFrame({
            'locus_id': sr.locus_ids,
            'mediator_lod': np.abs(sr.lod) + 0.1,
        })

    def test_returns_none_on_no_shared_loci(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        py_df = self._make_mediation_df(sr)
        csv_path = str(tmp_path / 'r_med.csv')
        pd.DataFrame({'locus': ['X_999'], 'mediator_lod': [1.0]}).to_csv(
            csv_path, index=False)
        fig, ax = plot_mediation_comparison(py_df, csv_path)
        assert fig is None and ax is None

    def test_returns_fig_ax_with_shared_loci(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        py_df = self._make_mediation_df(sr)
        csv_path = str(tmp_path / 'r_med2.csv')
        pd.DataFrame({
            'locus': list(sr.locus_ids),
            'mediator_lod': np.abs(sr.lod) + 0.2,
        }).to_csv(csv_path, index=False)
        fig, ax = plot_mediation_comparison(py_df, csv_path)
        assert fig is not None and ax is not None

    def test_saves_file(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        py_df = self._make_mediation_df(sr)
        csv_path = str(tmp_path / 'r_med3.csv')
        pd.DataFrame({
            'locus': list(sr.locus_ids),
            'mediator_lod': np.abs(sr.lod) + 0.2,
        }).to_csv(csv_path, index=False)
        out_path = str(tmp_path / 'med_cmp.png')
        plot_mediation_comparison(py_df, csv_path, save_path=out_path)
        assert Path(out_path).exists()


# ── plot_instability_comparison ───────────────────────────────────────────────

class TestPlotInstabilityComparison:
    def test_returns_none_on_no_shared_loci(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        csv_path = str(tmp_path / 'r_inst.csv')
        pd.DataFrame({'locus': ['Z_999'], 'lod': [0.5]}).to_csv(csv_path, index=False)
        fig, ax = plot_instability_comparison(sr, csv_path)
        assert fig is None and ax is None

    def test_returns_fig_ax_with_shared_loci(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        csv_path = str(tmp_path / 'r_inst2.csv')
        pd.DataFrame({
            'locus': list(sr.locus_ids),
            'lod': np.abs(sr.lod) + 0.1,
        }).to_csv(csv_path, index=False)
        fig, ax = plot_instability_comparison(sr, csv_path)
        assert fig is not None and ax is not None

    def test_saves_file(self, tmp_path):
        import pandas as pd
        sr = _make_scan_result()
        csv_path = str(tmp_path / 'r_inst3.csv')
        pd.DataFrame({
            'locus': list(sr.locus_ids),
            'lod': np.abs(sr.lod) + 0.1,
        }).to_csv(csv_path, index=False)
        out_path = str(tmp_path / 'inst_cmp.png')
        plot_instability_comparison(sr, csv_path, save_path=out_path)
        assert Path(out_path).exists()


# ── plot_phi_distribution ─────────────────────────────────────────────────────

class TestPlotPhiDistribution:
    def test_returns_fig_ax_shrunk_only(self):
        rng = np.random.default_rng(0)
        phi = rng.exponential(0.3, 100)
        fig, ax = plot_phi_distribution(phi)
        assert fig is not None and ax is not None

    def test_returns_fig_ax_with_raw(self):
        rng = np.random.default_rng(1)
        phi_s = rng.exponential(0.3, 50)
        phi_r = rng.exponential(0.5, 50)
        fig, ax = plot_phi_distribution(phi_s, phi_raw=phi_r)
        assert fig is not None and ax is not None

    def test_nan_values_handled(self):
        rng = np.random.default_rng(2)
        phi = rng.exponential(0.3, 50)
        phi[::5] = np.nan
        fig, ax = plot_phi_distribution(phi)
        assert fig is not None

    def test_saves_file(self, tmp_path):
        rng = np.random.default_rng(3)
        phi = rng.exponential(0.3, 30)
        out_path = str(tmp_path / 'phi_dist.png')
        plot_phi_distribution(phi, save_path=out_path)
        assert Path(out_path).exists()


# ── plot_nb_vs_lmm_comparison ─────────────────────────────────────────────────

class TestPlotNbVsLmmComparison:
    def test_returns_none_on_no_shared_loci(self):
        sr1 = _make_scan_result()
        rng = np.random.default_rng(99)
        sr2 = ScanResult(
            locus_ids   = np.array(['Z_999']),
            chromosomes = np.array(['99']),
            positions   = np.array([50.0]),
            lod         = np.array([1.0]),
            p_value     = np.array([0.1]),
            betas       = rng.standard_normal((1, N_FOUNDERS - 1)),
            h2_null     = {'99': 0.3},
            phenotype_id = 'other',
        )
        fig, ax = plot_nb_vs_lmm_comparison(sr1, sr2)
        assert fig is None and ax is None

    def test_returns_fig_ax_with_shared_loci(self):
        rng = np.random.default_rng(5)
        sr1 = _make_scan_result()
        sr2 = ScanResult(
            locus_ids   = sr1.locus_ids,
            chromosomes = sr1.chromosomes,
            positions   = sr1.positions,
            lod         = np.abs(rng.standard_normal(N_LOCI)) + 0.1,
            p_value     = rng.uniform(0, 1, N_LOCI),
            betas       = rng.standard_normal((N_LOCI, N_FOUNDERS - 1)),
            h2_null     = sr1.h2_null,
            phenotype_id = 'lmm',
        )
        fig, ax = plot_nb_vs_lmm_comparison(sr1, sr2)
        assert fig is not None and ax is not None

    def test_saves_file(self, tmp_path):
        rng = np.random.default_rng(6)
        sr1 = _make_scan_result()
        sr2 = ScanResult(
            locus_ids   = sr1.locus_ids,
            chromosomes = sr1.chromosomes,
            positions   = sr1.positions,
            lod         = np.abs(rng.standard_normal(N_LOCI)) + 0.1,
            p_value     = rng.uniform(0, 1, N_LOCI),
            betas       = rng.standard_normal((N_LOCI, N_FOUNDERS - 1)),
            h2_null     = sr1.h2_null,
            phenotype_id = 'lmm',
        )
        out_path = str(tmp_path / 'nb_lmm.png')
        plot_nb_vs_lmm_comparison(sr1, sr2, save_path=out_path)
        assert Path(out_path).exists()

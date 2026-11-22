"""Tests for qtl/ci.py: contiguous LOD support interval and scan-window containment."""

import numpy as np
import pytest

from mosaiqtl.genotype.kinship import decomp_kinship
from mosaiqtl.qtl.ci import CI_COLS, ci_scan_multi_gene, lod_support_interval


class TestLodSupportIntervalContiguity:
    def test_secondary_peak_does_not_widen_interval(self):
        pos = np.arange(10, dtype=float)
        lod = np.array([0, 5.0, 6.0, 5.0, 0, 0, 5.9, 5.95, 5.9, 0])
        assert lod_support_interval(pos, lod, lod_drop=1.5) == (1.0, 3.0)

    def test_peak_at_first_locus(self):
        pos = np.arange(5, dtype=float)
        lod = np.array([6.0, 5.5, 5.0, 0.0, 0.0])
        assert lod_support_interval(pos, lod, lod_drop=1.5) == (0.0, 2.0)

    def test_peak_at_last_locus(self):
        pos = np.arange(5, dtype=float)
        lod = np.array([0.0, 0.0, 5.0, 5.5, 6.0])
        assert lod_support_interval(pos, lod, lod_drop=1.5) == (2.0, 4.0)

    def test_flat_profile_spans_all_loci(self):
        pos = np.arange(6, dtype=float)
        lod = np.full(6, 3.0)
        assert lod_support_interval(pos, lod, lod_drop=1.5) == (0.0, 5.0)

    def test_isolated_peak_is_a_point(self):
        pos = np.arange(5, dtype=float)
        lod = np.array([0.0, 0.0, 6.0, 0.0, 0.0])
        assert lod_support_interval(pos, lod, lod_drop=1.5) == (2.0, 2.0)

    def test_nan_loci_end_the_run(self):
        pos = np.arange(6, dtype=float)
        lod = np.array([5.0, np.nan, 6.0, 5.5, 5.0, 0.0])
        assert lod_support_interval(pos, lod, lod_drop=1.5) == (2.0, 4.0)

    def test_interval_always_contains_peak(self):
        rng = np.random.default_rng(3)
        for _ in range(50):
            pos = np.sort(rng.uniform(0, 100, 40))
            lod = rng.uniform(0, 8, 40)
            lo, hi = lod_support_interval(pos, lod, lod_drop=1.5)
            peak = pos[int(np.nanargmax(lod))]
            assert lo <= peak <= hi


class TestCiScanWindow:
    @staticmethod
    def _run(tiny_cache, tiny_kinship, y, x, **kwargs):
        return ci_scan_multi_gene(
            {'g1': y},
            {'g1': '1'},
            {'g1': 50.0},
            tiny_cache,
            covariates=x,
            kinship_decomp=decomp_kinship(tiny_kinship),
            use_rint=False,
            **kwargs,
        )

    def test_interval_inside_scan_window_and_contains_peak(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        tss, w = 50.0, 20.0
        df = self._run(
            tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates,
            gene_pos_mb_map={'g1': tss}, window_mb=w,
        )
        row = df.iloc[0]
        assert tss - w <= row['ci_lo'] <= row['peak_mb'] <= row['ci_hi'] <= tss + w

    def test_peak_matches_the_haplotype_scan_window(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        tss, w = 50.0, 20.0
        df = self._run(
            tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates,
            gene_pos_mb_map={'g1': tss}, window_mb=w,
        )
        pos = tiny_cache.get_chromosome('1')[1]
        in_window = pos[np.abs(pos - tss) <= w]
        assert df.iloc[0]['peak_mb'] in in_window

    def test_window_with_one_locus_is_truncated(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        df = self._run(
            tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates,
            gene_pos_mb_map={'g1': 50.0}, window_mb=1.0,
        )
        row = df.iloc[0]
        assert row['ci_lo'] == row['ci_hi'] == row['peak_mb']
        assert bool(row['ci_truncated'])

    def test_ci_truncated_is_an_output_column(self):
        assert 'ci_truncated' in CI_COLS

    def test_position_map_without_window_is_rejected(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        with pytest.raises(ValueError, match='window_mb'):
            self._run(
                tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates,
                gene_pos_mb_map={'g1': 50.0},
            )


    def test_chromosome_without_kinship_is_skipped(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        kd = decomp_kinship(tiny_kinship)
        kd.pop('1')
        df = ci_scan_multi_gene(
            {'g1': tiny_strain_phenotype},
            {'g1': '1'},
            {'g1': 50.0},
            tiny_cache,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            use_rint=False,
            gene_pos_mb_map={'g1': 50.0},
            window_mb=20.0,
        )
        assert df.empty

    def test_strains_without_samples_add_no_rows(
        self, tiny_cache, tiny_kinship, tiny_phenotype, tiny_covariates, sample_strain_map
    ):
        from mosaiqtl.qtl import lmm as lmm_module
        from mosaiqtl.utils import build_z_matrix

        kd = decomp_kinship(tiny_kinship)
        sid = list(sample_strain_map)[4:]
        y, x = tiny_phenotype[4:], tiny_covariates[4:]
        df = ci_scan_multi_gene(
            {'g1': y}, {'g1': '1'}, {'g1': 50.0}, tiny_cache,
            covariates=x, kinship_decomp=kd, sample_ids=sid, use_rint=False,
        )

        z, _ = build_z_matrix(sid, list(tiny_cache.sample_ids))
        assert z.shape[0] == len(y) and (z.sum(axis=0) == 0).any()
        ev, evc = kd['1']['values'], kd['1']['vectors']
        ev_n, evc_n = np.linalg.eigh(z @ ((evc * ev) @ evc.T) @ z.T)
        cd = lmm_module.precompute_chromosome(tiny_cache, '1', ev_n, evc_n, z)
        h2, _ = lmm_module.estimate_h2_null(evc_n.T @ y, evc_n.T @ x, ev_n, method='ml')
        ref = lmm_module.scan_chromosome(y, cd, x, ev_n, evc_n, h2)
        assert df.iloc[0]['ci_peak_lod'] == pytest.approx(float(np.nanmax(ref['lod'])))

    def test_gene_without_anchor_is_skipped(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        df = self._run(
            tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates,
            gene_pos_mb_map={'other': 50.0}, window_mb=20.0,
        )
        assert df.empty

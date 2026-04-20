# tests/test_vqtl.py — tests for VQTLResult, compare_mean_vs_dispersion
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.qtl.scan import ScanResult
from mosaiqtl.qtl.vqtl import VQTLResult, compare_mean_vs_dispersion


def _make_scan(peak_lod, peak_chr='1', peak_pos=50.0, n=20):
    '''minimal ScanResult with a single controlled peak.'''
    lod = np.ones(n) * 0.1
    lod[n // 2] = peak_lod
    return ScanResult(
        locus_ids   = np.array([f'L{i}' for i in range(n)]),
        chromosomes = np.array([peak_chr] * n),
        positions   = np.linspace(1.0, 100.0, n),
        lod         = lod,
        p_value     = np.ones(n) * 0.5,
        betas       = np.zeros((n, 7)),
        h2_null     = {peak_chr: 0.2},
        phenotype_id= 'gene_z',
    )


def _make_vqtl(disp_lod, mean_lod, disp_chr='1', mean_chr='1',
               disp_pos=50.0, mean_pos=50.0):
    sr_disp = _make_scan(disp_lod, peak_chr=disp_chr, peak_pos=disp_pos)
    sr_mean = _make_scan(mean_lod, peak_chr=mean_chr, peak_pos=mean_pos)
    return VQTLResult(
        scan_result      = sr_disp,
        mean_scan_result = sr_mean,
        phi_estimates    = np.random.rand(10),
        phi_method       = 'moments',
        gene_id          = 'Piezo1',
    )


class TestCompareMeanVsDispersion:
    def test_both(self):
        r = compare_mean_vs_dispersion(_make_vqtl(5.0, 5.0), lod_threshold=3.0)
        assert r['category'] == 'both'

    def test_mean_only(self):
        r = compare_mean_vs_dispersion(_make_vqtl(1.0, 5.0), lod_threshold=3.0)
        assert r['category'] == 'mean_only'

    def test_dispersion_only(self):
        r = compare_mean_vs_dispersion(_make_vqtl(5.0, 1.0), lod_threshold=3.0)
        assert r['category'] == 'dispersion_only'

    def test_neither(self):
        r = compare_mean_vs_dispersion(_make_vqtl(1.0, 1.0), lod_threshold=3.0)
        assert r['category'] == 'neither'

    def test_colocalized_same_chr_close(self):
        r = compare_mean_vs_dispersion(
            _make_vqtl(5.0, 5.0, disp_chr='1', mean_chr='1',
                       disp_pos=50.0, mean_pos=52.0))
        assert r['colocalized'] == True  # noqa: E712 — np.bool_ needs ==

    def test_not_colocalized_diff_chr(self):
        r = compare_mean_vs_dispersion(
            _make_vqtl(5.0, 5.0, disp_chr='1', mean_chr='2'))
        assert r['colocalized'] is False

    def test_lod_values_returned(self):
        r = compare_mean_vs_dispersion(_make_vqtl(4.5, 6.0), lod_threshold=3.0)
        assert r['vqtl_peak_lod'] == pytest.approx(4.5, abs=0.01)
        assert r['mean_peak_lod'] == pytest.approx(6.0, abs=0.01)

    def test_custom_threshold(self):
        r = compare_mean_vs_dispersion(_make_vqtl(2.0, 2.0), lod_threshold=1.5)
        assert r['category'] == 'both'

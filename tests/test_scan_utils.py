# tests/test_scan_utils.py — tests for ScanResult, grab_locus_from_scan, lod_drop_ci
import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.scan import ScanResult, grab_locus_from_scan, lod_drop_ci


def _make_result(n=30, J=8, chrom='1', seed=0):
    '''synthetic ScanResult with controllable peak position.'''
    rng = np.random.default_rng(seed)
    lod = rng.random(n) * 2.0
    lod[n // 2] = 10.0          # force peak at midpoint
    return ScanResult(
        locus_ids   = np.array([f'L{i}' for i in range(n)]),
        chromosomes = np.array([chrom] * n),
        positions   = np.linspace(1.0, 100.0, n),
        lod         = lod,
        p_value     = np.ones(n) * 0.01,
        betas       = rng.random((n, J - 1)),
        h2_null     = {chrom: 0.3},
        phenotype_id= 'gene_x',
    )


class TestScanResult:
    def test_peak_locus_returns_max(self):
        sr = _make_result(n=30)
        pk = sr.peak_locus()
        assert pk['lod'] == sr.lod.max()

    def test_peak_locus_locus_id(self):
        sr = _make_result(n=30)
        pk = sr.peak_locus()
        assert pk['locus_id'] == sr.locus_ids[np.argmax(sr.lod)]

    def test_to_dataframe_shape(self):
        sr = _make_result(n=30, J=8)
        df = sr.to_dataframe()
        assert df.shape == (30, 5 + 7)   # locus_id,chr,pos,lod,pval + 7 betas

    def test_to_dataframe_lod_col(self):
        sr = _make_result(n=30)
        df = sr.to_dataframe()
        np.testing.assert_array_equal(df['lod'].values, sr.lod)


class TestGrabLocusFromScan:
    def test_returns_peak_id(self):
        sr = _make_result(n=30)
        lid = grab_locus_from_scan(sr)
        assert lid == sr.locus_ids[np.argmax(sr.lod)]

    def test_type_is_str(self):
        sr = _make_result(n=30)
        assert isinstance(grab_locus_from_scan(sr), (str, np.str_))


class TestLodDropCI:
    def setup_method(self):
        # isolated peak at index 25; shoulders only to the right for width test
        n = 50
        pos  = np.linspace(1.0, 100.0, n)
        lod  = np.ones(n) * 0.5          # baseline well below drop
        lod[25] = 8.0                     # sharp peak
        lod[26:36] = 7.0                  # right shoulder within 1.5-drop
        self.sr = ScanResult(
            locus_ids   = np.array([f'L{i}' for i in range(n)]),
            chromosomes = np.array(['1'] * n),
            positions   = pos,
            lod         = lod,
            p_value     = np.ones(n),
            betas       = np.zeros((n, 7)),
            h2_null     = {'1': 0.2},
            phenotype_id= 'gene_y',
        )

    def test_peak_mb_correct(self):
        lo, hi, peak = lod_drop_ci(self.sr, '1', drop=1.5)
        # peak is at lod[25] (n//2); positions are linspace(1,100,50)
        assert peak == pytest.approx(self.sr.positions[25], abs=1e-6)

    def test_ci_brackets_peak(self):
        lo, hi, peak = lod_drop_ci(self.sr, '1', drop=1.5)
        assert lo <= peak <= hi

    def test_missing_chrom_returns_nan(self):
        lo, hi, peak = lod_drop_ci(self.sr, '99', drop=1.5)
        assert np.isnan(lo) and np.isnan(hi) and np.isnan(peak)

    def test_wider_drop_gives_wider_ci(self):
        lo1, hi1, _ = lod_drop_ci(self.sr, '1', drop=1.0)
        lo2, hi2, _ = lod_drop_ci(self.sr, '1', drop=3.0)
        assert (hi2 - lo2) >= (hi1 - lo1)

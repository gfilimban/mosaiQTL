# tests/unit/test_records.py — unit tests for models/records.py
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))
from mosaiqtl.models.records import FounderEffect, QTLResult, ScanResult, VQTLResult


def _scan_result(chr='1'):
    return ScanResult(
        lod=np.zeros(5),
        loci=['L0', 'L1', 'L2', 'L3', 'L4'],
        pos_mb=np.linspace(1.0, 5.0, 5),
        chr=chr,
    )


def _founder_effect(founder='AJ'):
    return FounderEffect(founder=founder, estimate=0.5, se=0.1, ci_lo=0.3, ci_hi=0.7)


def _qtl_result(**kwargs):
    defaults = dict(
        gene_id='ENSMUSG001', gene_name='Piezo1', cell_type='type2x',
        gene_chr='8', gene_tss_mb=100.0,
        peak_locus='UNC2373664', peak_chr='8', peak_pos_mb=101.5,
        lod=21.3, lod_threshold=7.5, p_value=1e-6, q_value=0.01,
        n_perms=1000, locus_type='cis', distance_to_tss_mb=1.5,
        cis_window_mb=1.0, founder_effects={'AJ': _founder_effect()},
        variance_explained=0.42, model='nb-glmm', n_samples=418,
    )
    defaults.update(kwargs)
    return QTLResult(**defaults)


class TestScanResult:
    def test_chr_is_str(self):
        r = _scan_result(chr='1')
        assert isinstance(r.chr, str)

    def test_slots_no_dict(self):
        r = _scan_result()
        assert not hasattr(r, '__dict__')

    def test_lod_shape(self):
        r = _scan_result()
        assert r.lod.shape == (5,)


class TestFounderEffect:
    def test_slots_no_dict(self):
        fe = _founder_effect()
        assert not hasattr(fe, '__dict__')

    def test_ci_ordered(self):
        fe = _founder_effect()
        assert fe.ci_lo < fe.ci_hi


class TestVQTLResult:
    def test_field_names_accessible(self):
        vr = VQTLResult(
            scan_result=_scan_result(),
            mean_scan_result=_scan_result(),
            phi_estimates=np.ones(10),
            phi_method='moments',
            gene_id='ENSMUSG001',
        )
        assert vr.scan_result is not None
        assert vr.mean_scan_result is not None

    def test_wrong_field_raises(self):
        vr = VQTLResult(
            scan_result=_scan_result(),
            mean_scan_result=_scan_result(),
            phi_estimates=np.ones(10),
            phi_method='moments',
            gene_id='ENSMUSG001',
        )
        with pytest.raises(AttributeError):
            _ = vr.vqtl_scan
        with pytest.raises(AttributeError):
            _ = vr.mean_scan

    def test_slots_no_dict(self):
        vr = VQTLResult(
            scan_result=_scan_result(),
            mean_scan_result=_scan_result(),
            phi_estimates=np.ones(10),
            phi_method='moments',
            gene_id='ENSMUSG001',
        )
        assert not hasattr(vr, '__dict__')


class TestQTLResult:
    def test_peak_locus_present(self):
        r = _qtl_result()
        assert r.peak_locus == 'UNC2373664'

    def test_gene_chr_and_peak_chr_separate(self):
        r = _qtl_result(gene_chr='8', peak_chr='4', locus_type='trans',
                        distance_to_tss_mb=float('nan'))
        assert r.gene_chr == '8'
        assert r.peak_chr == '4'
        assert r.locus_type == 'trans'
        assert math.isnan(r.distance_to_tss_mb)

    def test_chr_fields_are_str(self):
        r = _qtl_result()
        assert isinstance(r.gene_chr, str)
        assert isinstance(r.peak_chr, str)

    def test_locus_type_values(self):
        for lt in ('cis', 'local', 'trans'):
            r = _qtl_result(locus_type=lt)
            assert r.locus_type == lt

    def test_slots_no_dict(self):
        r = _qtl_result()
        assert not hasattr(r, '__dict__')

    def test_n_perms_present(self):
        r = _qtl_result(n_perms=5000)
        assert r.n_perms == 5000

    def test_model_field(self):
        r = _qtl_result(model='lmm')
        assert r.model == 'lmm'

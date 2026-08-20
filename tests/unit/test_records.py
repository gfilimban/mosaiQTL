# tests/unit/test_records.py — unit tests for models/records.py
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from mosaiqtl.models.records import (
    FounderEffect,
    ProvenanceRecord,
    QTLResult,
    ScanResult,
    VQTLResult,
)


def _scan_result(chr="1", gene_id="ENSMUSG001"):
    return ScanResult(
        gene_id=gene_id,
        lod=np.zeros(5),
        loci=["L0", "L1", "L2", "L3", "L4"],
        pos_mb=np.linspace(1.0, 5.0, 5),
        chr=chr,
    )


def _founder_effect(founder="AJ"):
    return FounderEffect(
        founder=founder,
        estimate=0.5,
        se=0.1,
        z_score=5.0,
        p_value=1e-6,
        ci_lo=0.3,
        ci_hi=0.7,
    )


def _provenance():
    return ProvenanceRecord(
        git_hash="abc1234",
        seed=42,
        timestamp="2026-04-19T22:00:00",
        scan_window="cis",
        phenotype_input="raw",
        offset="none",
        covariates="sex,batch",
        n_perms=1000,
        cis_window_mb=1.0,
    )


def _qtl_result(**kwargs):
    defaults = dict(
        gene_id="ENSMUSG001",
        gene_symbol="Piezo1",
        cell_type="type2x",
        data_granularity="pseudobulk",
        population="cc",
        model="nb-glmm",
        test_statistic="score",
        haplotype="additive",
        gene_chr="8",
        gene_tss_mb=100.0,
        peak_locus="UNC2373664",
        peak_chr="8",
        peak_pos_mb=101.5,
        lod=21.3,
        lod_threshold=7.5,
        p_value=1e-6,
        q_value=0.01,
        n_loci_tested=3200,
        n_perms=1000,
        n_samples=418,
        locus_type="cis",
        distance_to_tss_mb=1.5,
        cis_window_mb=1.0,
        founder_effects={"AJ": _founder_effect()},
        variance_explained=0.42,
        provenance=_provenance(),
    )
    defaults.update(kwargs)
    return QTLResult(**defaults)


class TestScanResult:
    def test_gene_id_present(self):
        r = _scan_result(gene_id="ENSMUSG001")
        assert r.gene_id == "ENSMUSG001"

    def test_chr_is_str(self):
        assert isinstance(_scan_result(chr="1").chr, str)

    def test_slots_no_dict(self):
        assert not hasattr(_scan_result(), "__dict__")

    def test_lod_shape(self):
        assert _scan_result().lod.shape == (5,)


class TestFounderEffect:
    def test_z_score_and_pvalue_present(self):
        fe = _founder_effect()
        assert fe.z_score == pytest.approx(5.0)
        assert fe.p_value == pytest.approx(1e-6)

    def test_ci_ordered(self):
        fe = _founder_effect()
        assert fe.ci_lo < fe.ci_hi

    def test_slots_no_dict(self):
        assert not hasattr(_founder_effect(), "__dict__")


class TestVQTLResult:
    def _make(self, cell_type="type2x", granularity="pseudobulk"):
        return VQTLResult(
            scan_result=_scan_result(),
            mean_scan_result=_scan_result(),
            phi_estimates=np.ones(10),
            phi_method="moments",
            gene_id="ENSMUSG001",
            cell_type=cell_type,
            data_granularity=granularity,
            model="nb-glmm",
            n_samples=418,
        )

    def test_field_names_accessible(self):
        vr = self._make()
        assert vr.scan_result is not None
        assert vr.mean_scan_result is not None

    def test_wrong_field_raises(self):
        vr = self._make()
        with pytest.raises(AttributeError):
            _ = vr.vqtl_scan
        with pytest.raises(AttributeError):
            _ = vr.mean_scan

    def test_cell_type_present(self):
        assert self._make().cell_type == "type2x"

    def test_provenance_fields_present(self):
        vr = self._make()
        assert vr.model == "nb-glmm"
        assert vr.n_samples == 418

    def test_bulk_cell_type_none(self):
        vr = self._make(cell_type=None, granularity="bulk")
        assert vr.cell_type is None
        assert vr.data_granularity == "bulk"

    def test_pseudobulk_has_cell_type(self):
        vr = self._make()
        assert vr.cell_type == "type2x"
        assert vr.data_granularity == "pseudobulk"

    def test_slots_no_dict(self):
        assert not hasattr(self._make(), "__dict__")


class TestProvenanceRecord:
    def test_slots_no_dict(self):
        assert not hasattr(_provenance(), "__dict__")

    def test_fields(self):
        p = _provenance()
        assert p.git_hash == "abc1234"
        assert p.seed == 42
        assert p.scan_window == "cis"

    def test_tier2_axes_present(self):
        p = _provenance()
        assert p.phenotype_input == "raw"
        assert p.offset == "none"
        assert p.covariates == "sex,batch"


class TestQTLResult:
    def test_peak_locus_present(self):
        assert _qtl_result().peak_locus == "UNC2373664"

    def test_gene_chr_and_peak_chr_separate(self):
        r = _qtl_result(
            gene_chr="8",
            peak_chr="4",
            locus_type="trans",
            distance_to_tss_mb=float("nan"),
        )
        assert r.gene_chr == "8"
        assert r.peak_chr == "4"
        assert r.locus_type == "trans"
        assert math.isnan(r.distance_to_tss_mb)

    def test_chr_fields_are_str(self):
        r = _qtl_result()
        assert isinstance(r.gene_chr, str)
        assert isinstance(r.peak_chr, str)

    def test_locus_type_values(self):
        for lt in ("cis", "local", "trans"):
            assert _qtl_result(locus_type=lt).locus_type == lt

    def test_n_loci_tested_present(self):
        assert _qtl_result(n_loci_tested=5000).n_loci_tested == 5000

    def test_n_perms_present(self):
        assert _qtl_result(n_perms=5000).n_perms == 5000

    def test_model_field(self):
        assert _qtl_result(model="lmm").model == "lmm"

    def test_tier1_axes_present(self):
        r = _qtl_result()
        assert r.population == "cc"
        assert r.test_statistic == "score"
        assert r.haplotype == "additive"

    def test_population_is_str(self):
        r = _qtl_result(population="magic")
        assert isinstance(r.population, str)
        assert r.population == "magic"

    def test_bulk_cell_type_none(self):
        r = _qtl_result(cell_type=None, data_granularity="bulk")
        assert r.cell_type is None
        assert r.data_granularity == "bulk"

    def test_pseudobulk_has_cell_type(self):
        r = _qtl_result()
        assert r.cell_type == "type2x"
        assert r.data_granularity == "pseudobulk"

    def test_provenance_attached(self):
        r = _qtl_result()
        assert isinstance(r.provenance, ProvenanceRecord)
        assert r.provenance.git_hash == "abc1234"

    def test_slots_no_dict(self):
        assert not hasattr(_qtl_result(), "__dict__")

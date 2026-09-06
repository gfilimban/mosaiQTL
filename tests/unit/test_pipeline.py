"""Unit tests for pipeline.py — Layer 1 / 2 / 3 API.

Covers:
    - DiscoveryResult, CharacterizationResult, ResolutionResult dataclasses
    - discover(model='lmm') and discover(model='nb')
    - characterize()
    - resolve() stubs and conditional_scan integration
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.pipeline import (
    DiscoveryResult,
    CharacterizationResult,
    ResolutionResult,
    discover,
    characterize,
    resolve,
)
from mosaiqtl.qtl.scan import ScanResult

# match conftest constants
N_STRAINS = 10
N_REPS_PER_STR = 4
N_CELLS_PER_SAMPLE = 8
N_GENES = 4
N_CELLS = N_STRAINS * N_REPS_PER_STR * N_CELLS_PER_SAMPLE  # 320


# ── dataclasses ───────────────────────────────────────────────────────────────


class TestDiscoveryResult:
    def test_fields_present(self):
        dr = DiscoveryResult(scan={}, model="lmm", genomecache=None)
        assert dr.scan == {}
        assert dr.model == "lmm"
        assert dr.genomecache is None
        assert dr.kinship_decomp is None
        assert dr.dispersion is None

    def test_nb_model_stored(self):
        dr = DiscoveryResult(scan={"g": object()}, model="nb", genomecache=None)
        assert dr.model == "nb"


class TestCharacterizationResult:
    def test_all_fields_optional(self):
        cr = CharacterizationResult()
        assert cr.allele_effects is None
        assert cr.vqtl is None
        assert cr.mediation is None
        assert cr.discovery is None


class TestResolutionResult:
    def test_modules_run_starts_empty(self):
        rr = ResolutionResult()
        assert rr.modules_run == []

    def test_all_resolution_fields_optional(self):
        rr = ResolutionResult()
        assert rr.snp_catalogue is None
        assert rr.conditional_scan is None
        assert rr.ase is None
        assert rr.sqtl is None


# ── discover — model='lmm' ────────────────────────────────────────────────────


class TestDiscoverLmm:
    def test_returns_discovery_result(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        pheno_mat = tiny_strain_phenotype[:, None]
        dr = discover(pheno_mat, ["gene_a"], tiny_cache, model="lmm", kinship_decomp=kd)
        assert isinstance(dr, DiscoveryResult)

    def test_model_field_is_lmm(self, tiny_cache, tiny_strain_phenotype):
        pheno_mat = tiny_strain_phenotype[:, None]
        dr = discover(pheno_mat, ["gene_a"], tiny_cache, model="lmm")
        assert dr.model == "lmm"

    def test_scan_keys_match_gene_ids(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(5)
        pheno_mat = np.column_stack(
            [
                tiny_strain_phenotype,
                rng.standard_normal(len(tiny_strain_phenotype)),
            ]
        )
        dr = discover(
            pheno_mat, ["g1", "g2"], tiny_cache, model="lmm", kinship_decomp=kd
        )
        assert set(dr.scan.keys()) == {"g1", "g2"}

    def test_each_scan_result_type(self, tiny_cache, tiny_strain_phenotype):
        pheno_mat = tiny_strain_phenotype[:, None]
        dr = discover(pheno_mat, ["g1"], tiny_cache, model="lmm")
        assert isinstance(dr.scan["g1"], ScanResult)

    def test_kinship_decomp_stored(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        pheno_mat = tiny_strain_phenotype[:, None]
        dr = discover(pheno_mat, ["g1"], tiny_cache, model="lmm", kinship_decomp=kd)
        assert dr.kinship_decomp is not None

    def test_genomecache_stored(self, tiny_cache, tiny_strain_phenotype):
        pheno_mat = tiny_strain_phenotype[:, None]
        dr = discover(pheno_mat, ["g1"], tiny_cache, model="lmm")
        assert dr.genomecache is tiny_cache

    # ── windowed scanning (scan_window/gene_chr_map/gene_pos_mb_map) ──────
    # scan_multi_gene already implements windowed scanning natively (see
    # qtl/CLAUDE.md); these tests cover discover()'s forwarding of it.

    def test_default_scan_window_forwarded_as_genome(
        self, tiny_cache, tiny_strain_phenotype
    ):
        """Omitting the new windowing kwargs (the pre-existing call shape)
        must still reach scan_multi_gene as scan_window='genome' with no
        window args — i.e. adding lmm windowing support did not change
        default behavior."""
        from unittest.mock import patch
        from mosaiqtl import pipeline as pipeline_module
        from mosaiqtl.qtl.scan import scan_multi_gene as real_scan_multi_gene

        pheno_mat = tiny_strain_phenotype[:, None]
        with patch.object(
            pipeline_module,
            "scan_multi_gene",
            side_effect=real_scan_multi_gene,
        ) as mock_scan:
            discover(pheno_mat, ["g1"], tiny_cache, model="lmm")
        _, kwargs = mock_scan.call_args
        assert kwargs["scan_window"] == "genome"
        assert kwargs["gene_chr_map"] is None
        assert kwargs["gene_pos_mb_map"] is None

    def test_local_scan_window_forwarded(self, tiny_cache, tiny_strain_phenotype):
        """scan_window='local' plus gene_chr_map/gene_pos_mb_map/window_mb
        must reach scan_multi_gene unchanged — this is the forwarding the
        cli.py `scan` and `permute` local-window flags rely on."""
        from unittest.mock import patch
        from mosaiqtl import pipeline as pipeline_module
        from mosaiqtl.qtl.scan import scan_multi_gene as real_scan_multi_gene

        pheno_mat = tiny_strain_phenotype[:, None]
        gene_chr_map = {"g1": "1"}
        gene_pos_mb_map = {"g1": 1.0}
        with patch.object(
            pipeline_module,
            "scan_multi_gene",
            side_effect=real_scan_multi_gene,
        ) as mock_scan:
            discover(
                pheno_mat,
                ["g1"],
                tiny_cache,
                model="lmm",
                scan_window="local",
                gene_chr_map=gene_chr_map,
                gene_pos_mb_map=gene_pos_mb_map,
                window_mb=0.1,
            )
        _, kwargs = mock_scan.call_args
        assert kwargs["scan_window"] == "local"
        assert kwargs["gene_chr_map"] == gene_chr_map
        assert kwargs["gene_pos_mb_map"] == gene_pos_mb_map
        assert kwargs["window_mb"] == 0.1

    def test_local_scan_window_restricts_loci(self, tiny_cache, tiny_strain_phenotype):
        """End-to-end (no mocking): a local window covering exactly one of
        tiny_cache's 15 loci on chr '1' (positions are
        np.linspace(1.0, 100.0, 15); gene_pos_mb=1.0 +/- 0.1 admits only the
        first) must scan strictly fewer loci than the genome-wide default,
        proving the restriction actually reaches the real lmm scan engine —
        not just that the kwargs were passed along."""
        pheno_mat = tiny_strain_phenotype[:, None]
        dr_genome = discover(pheno_mat, ["g1"], tiny_cache, model="lmm")
        dr_local = discover(
            pheno_mat,
            ["g1"],
            tiny_cache,
            model="lmm",
            scan_window="local",
            gene_chr_map={"g1": "1"},
            gene_pos_mb_map={"g1": 1.0},
            window_mb=0.1,
        )
        assert len(dr_local.scan["g1"].lod) == 1
        assert len(dr_local.scan["g1"].lod) < len(dr_genome.scan["g1"].lod)


# ── discover — model='nb' ─────────────────────────────────────────────────────


class TestDiscoverNb:
    @staticmethod
    def _strain_ids_per_cell(strain_ids):
        """(N_CELLS,) strain per cell — each strain repeated N_REPS×N_CELLS times."""
        return np.repeat(strain_ids, N_REPS_PER_STR * N_CELLS_PER_SAMPLE)

    def test_returns_discovery_result(self, tiny_cache, strain_ids, counts_cells):
        sc_ids = self._strain_ids_per_cell(strain_ids)
        # use 1 gene for speed
        dr = discover(
            counts_cells[:1], ["g0"], tiny_cache, model="nb", sample_ids=sc_ids
        )
        assert isinstance(dr, DiscoveryResult)

    def test_model_field_is_nb(self, tiny_cache, strain_ids, counts_cells):
        sc_ids = self._strain_ids_per_cell(strain_ids)
        dr = discover(
            counts_cells[:1], ["g0"], tiny_cache, model="nb", sample_ids=sc_ids
        )
        assert dr.model == "nb"

    def test_scan_is_dict(self, tiny_cache, strain_ids, counts_cells):
        sc_ids = self._strain_ids_per_cell(strain_ids)
        dr = discover(
            counts_cells[:1], ["g0"], tiny_cache, model="nb", sample_ids=sc_ids
        )
        assert isinstance(dr.scan, dict)

    def test_each_scan_result_type(self, tiny_cache, strain_ids, counts_cells):
        sc_ids = self._strain_ids_per_cell(strain_ids)
        gene_ids = [f"g{i}" for i in range(N_GENES)]
        dr = discover(counts_cells, gene_ids, tiny_cache, model="nb", sample_ids=sc_ids)
        for sr in dr.scan.values():
            assert isinstance(sr, ScanResult)

    def test_scan_lod_nonneg(self, tiny_cache, strain_ids, counts_cells):
        sc_ids = self._strain_ids_per_cell(strain_ids)
        dr = discover(
            counts_cells[:1], ["g0"], tiny_cache, model="nb", sample_ids=sc_ids
        )
        for sr in dr.scan.values():
            assert np.all(sr.lod >= -1e-8)

    def test_genomecache_stored(self, tiny_cache, strain_ids, counts_cells):
        sc_ids = self._strain_ids_per_cell(strain_ids)
        dr = discover(
            counts_cells[:1], ["g0"], tiny_cache, model="nb", sample_ids=sc_ids
        )
        assert dr.genomecache is tiny_cache


# ── discover — bad model ──────────────────────────────────────────────────────


class TestDiscoverBadModel:
    def test_raises_value_error(self, tiny_cache, tiny_strain_phenotype):
        with pytest.raises(ValueError, match="model must be"):
            discover(tiny_strain_phenotype[:, None], ["g"], tiny_cache, model="glm")


# ── characterize ──────────────────────────────────────────────────────────────


class TestCharacterize:
    @staticmethod
    def _lmm_disc(tiny_cache, tiny_strain_phenotype):
        pheno_mat = tiny_strain_phenotype[:, None]
        return discover(pheno_mat, ["g1"], tiny_cache, model="lmm")

    def test_returns_characterization_result(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=("allele_effects",))
        assert isinstance(cr, CharacterizationResult)

    def test_allele_effects_dict(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=("allele_effects",))
        assert isinstance(cr.allele_effects, dict)

    def test_allele_effects_keys_match_genes(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=("allele_effects",))
        assert set(cr.allele_effects.keys()) == set(dr.scan.keys())

    def test_vqtl_stub_empty(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=("vqtl",))
        assert cr.vqtl == {}

    def test_mediation_stub_empty(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=("mediation",))
        assert cr.mediation == {}

    def test_discovery_stored_on_result(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=())
        assert cr.discovery is dr

    def test_empty_modules_returns_result(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        cr = characterize(dr, modules=())
        assert isinstance(cr, CharacterizationResult)


# ── resolve ───────────────────────────────────────────────────────────────────


class TestResolve:
    @staticmethod
    def _lmm_disc(tiny_cache, tiny_strain_phenotype, tiny_kinship=None):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship) if tiny_kinship is not None else None
        pheno_mat = tiny_strain_phenotype[:, None]
        return discover(pheno_mat, ["g1"], tiny_cache, model="lmm", kinship_decomp=kd)

    def test_returns_resolution_result(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        rr = resolve(dr, modules=())
        assert isinstance(rr, ResolutionResult)

    def test_snp_catalogue_no_vcf_returns_empty(
        self, tiny_cache, tiny_strain_phenotype
    ):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        rr = resolve(dr, modules=("snp_catalogue",))
        assert rr.snp_catalogue == {}

    def test_ase_no_args_returns_empty(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        rr = resolve(dr, modules=("ase",))
        assert rr.ase == {}

    def test_sqtl_no_args_returns_empty(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        rr = resolve(dr, modules=("sqtl",))
        assert rr.sqtl == {}

    def test_unknown_module_raises(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        with pytest.raises(ValueError, match="unknown modules"):
            resolve(dr, modules=("bad_module",))

    def test_modules_run_recorded(self, tiny_cache, tiny_strain_phenotype):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype)
        rr = resolve(dr, modules=("snp_catalogue", "ase"))
        assert "snp_catalogue" in rr.modules_run
        assert "ase" in rr.modules_run

    def test_conditional_scan_returns_dict(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype, tiny_kinship)
        rr = resolve(dr, modules=("conditional_scan",))
        assert isinstance(rr.conditional_scan, dict)

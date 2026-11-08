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
        rr = resolve(dr, modules=("snp_catalogue",))
        assert "snp_catalogue" in rr.modules_run

    def test_conditional_scan_returns_dict(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        dr = self._lmm_disc(tiny_cache, tiny_strain_phenotype, tiny_kinship)
        rr = resolve(dr, modules=("conditional_scan",))
        assert isinstance(rr.conditional_scan, dict)


# ── calibrate_egenes_storey ───────────────────────────────────────────────────


class TestCalibrateEgenesStorey:
    """calibrate_egenes_storey: Storey q-value significance block (SWE-09)."""

    @staticmethod
    def _egenes_df(n_genes=8):
        """Fabricate a minimal egenes DataFrame with gene_id and peak_lod."""
        import pandas as pd

        return pd.DataFrame(
            {
                "gene_id": [f"g{i}" for i in range(n_genes)],
                "peak_lod": np.linspace(1.0, 5.0, n_genes),
            }
        )

    def test_sig_cols_present(self):
        """Output contains pval_perm, fdr, fdr_method, significance columns."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        df = self._egenes_df()
        pval_perm = {f"g{i}": 0.01 * (i + 1) for i in range(len(df))}
        out = calibrate_egenes_storey(df, pval_perm)
        for col in ("pval_perm", "fdr", "fdr_method", "significance"):
            assert col in out.columns, f"missing {col!r}"

    def test_fdr_method_is_qvalue(self):
        """fdr_method column is always the string 'qvalue'."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        df = self._egenes_df()
        pval_perm = {f"g{i}": 0.05 for i in range(len(df))}
        out = calibrate_egenes_storey(df, pval_perm)
        assert (out["fdr_method"] == "qvalue").all()

    def test_fdr_in_unit_interval(self):
        """All q-values are in [0, 1]."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        df = self._egenes_df(n_genes=20)
        rng = np.random.default_rng(0)
        pval_perm = {f"g{i}": float(p) for i, p in enumerate(rng.uniform(0, 1, 20))}
        out = calibrate_egenes_storey(df, pval_perm)
        assert out["fdr"].between(0.0, 1.0).all()

    def test_significance_t_f_only(self):
        """significance column contains only 'T' or 'F'."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        df = self._egenes_df()
        pval_perm = {f"g{i}": 0.01 if i < 3 else 0.8 for i in range(len(df))}
        out = calibrate_egenes_storey(df, pval_perm)
        assert set(out["significance"].unique()).issubset({"T", "F"})

    def test_low_pval_becomes_significant(self):
        """Genes with very small p-values are called significant (fdr ≤ 0.05)."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        n = 50
        df = self._egenes_df(n_genes=n)
        # first 10 genes have tiny p-values; rest are near-null
        pval_perm = {f"g{i}": 1e-6 if i < 10 else 0.9 for i in range(n)}
        out = calibrate_egenes_storey(df, pval_perm, fdr_threshold=0.05)
        n_sig = (out["significance"] == "T").sum()
        assert n_sig >= 5, f"expected ≥5 significant genes, got {n_sig}"

    def test_high_pval_not_significant(self):
        """Genes with p ≈ 1 are not called significant."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        df = self._egenes_df(n_genes=10)
        pval_perm = {f"g{i}": 0.99 for i in range(10)}
        out = calibrate_egenes_storey(df, pval_perm)
        assert (out["significance"] == "F").all()

    def test_row_count_preserved(self):
        """Output has exactly the same number of rows as input."""
        from mosaiqtl.pipeline import calibrate_egenes_storey

        n = 12
        df = self._egenes_df(n_genes=n)
        pval_perm = {f"g{i}": 0.1 for i in range(n)}
        out = calibrate_egenes_storey(df, pval_perm)
        assert len(out) == n


# ── Storey FDR end-to-end integration (SWE-09) ───────────────────────────────


class TestStoreyFdrEndToEnd:
    """Full chain: calibrate_gene_pvalue → calibrate_egenes_storey (SWE-09).

    Uses tiny_cache + tiny_kinship fixtures so no external data is needed.
    Runs calibrate_gene_pvalue on a small set of fabricated genes, pools
    their Beta-calibrated p-values, and feeds them into calibrate_egenes_storey
    to get final q-value significance calls.
    """

    def test_full_chain_runs_and_gives_valid_output(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """End-to-end: calibrate_gene_pvalue → calibrate_egenes_storey.

        Two fabricated genes: one with a genuinely high peak LOD (signal),
        one with a near-zero LOD (noise).  After going through the full
        chain the chain must at minimum produce finite q-values in [0,1]
        and not crash.
        """
        import pandas as pd
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.fdr import calibrate_gene_pvalue
        from mosaiqtl.pipeline import calibrate_egenes_storey

        kd = decomp_kinship(tiny_kinship)
        c0 = list(tiny_cache.chromosomes)[0]

        # two genes: strong signal (LOD=8) and near-null (LOD=0.1)
        genes = [
            ("gene_signal", 8.0),
            ("gene_null", 0.1),
        ]

        pval_perm = {}
        for gid, peak_lod in genes:
            p_adj, *_ = calibrate_gene_pvalue(
                tiny_strain_phenotype,
                tiny_cache,
                chrom=c0,
                peak_lod=peak_lod,
                covariates=tiny_strain_covariates,
                kinship_decomp=kd,
                kinship_loco=tiny_kinship,
                n_perms=25,
                use_rint=False,
                seed=17,
            )
            pval_perm[gid] = p_adj

        # assemble a minimal egenes_df (both genes cleared the Stage-1 gate)
        egenes_df = pd.DataFrame(
            {
                "gene_id": [g for g, _ in genes],
                "peak_lod": [lod for _, lod in genes],
            }
        )

        out = calibrate_egenes_storey(egenes_df, pval_perm)

        # structural checks
        assert len(out) == len(genes)
        assert out["fdr"].between(0.0, 1.0).all()
        assert set(out["significance"].unique()).issubset({"T", "F"})
        # the near-null gene should have a larger q-value than the signal gene
        fdr_signal = float(out.loc[out["gene_id"] == "gene_signal", "fdr"].iloc[0])
        fdr_null = float(out.loc[out["gene_id"] == "gene_null", "fdr"].iloc[0])
        assert fdr_signal <= fdr_null, (
            f"signal gene q-value ({fdr_signal:.3f}) should be ≤ "
            f"null gene q-value ({fdr_null:.3f})"
        )

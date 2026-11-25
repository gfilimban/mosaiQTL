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

import pandas as pd

from mosaiqtl.pipeline import (
    DiscoveryResult,
    CharacterizationResult,
    ResolutionResult,
    discover,
    characterize,
    resolve,
)
from mosaiqtl import pipeline
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


# ── calibrate_egenes ───────────────────────────────────────────────────


class TestCalibrateEgenesStorey:
    """calibrate_egenes: Storey q-value significance block (SWE-09)."""

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
        from mosaiqtl.pipeline import calibrate_egenes

        df = self._egenes_df()
        pval_perm = {f"g{i}": 0.01 * (i + 1) for i in range(len(df))}
        out = calibrate_egenes(df, pval_perm)
        for col in ("pval_perm", "fdr", "fdr_method", "significance"):
            assert col in out.columns, f"missing {col!r}"

    def test_default_rule_is_bh(self):
        """The default call rule is Benjamini-Hochberg."""
        from mosaiqtl.pipeline import calibrate_egenes

        df = self._egenes_df()
        pval_perm = {f"g{i}": 0.05 for i in range(len(df))}
        out = calibrate_egenes(df, pval_perm)
        assert (out["fdr_method"] == "BH").all()
        np.testing.assert_allclose(out["fdr"], out["qval_bh"])

    def test_fdr_in_unit_interval(self):
        """All q-values are in [0, 1]."""
        from mosaiqtl.pipeline import calibrate_egenes

        df = self._egenes_df(n_genes=20)
        rng = np.random.default_rng(0)
        pval_perm = {f"g{i}": float(p) for i, p in enumerate(rng.uniform(0, 1, 20))}
        out = calibrate_egenes(df, pval_perm)
        assert out["fdr"].between(0.0, 1.0).all()

    def test_significance_t_f_only(self):
        """significance column contains only 'T' or 'F'."""
        from mosaiqtl.pipeline import calibrate_egenes

        df = self._egenes_df()
        pval_perm = {f"g{i}": 0.01 if i < 3 else 0.8 for i in range(len(df))}
        out = calibrate_egenes(df, pval_perm)
        assert set(out["significance"].unique()).issubset({"T", "F"})

    def test_low_pval_becomes_significant(self):
        """Genes with very small p-values are called significant (fdr ≤ 0.05)."""
        from mosaiqtl.pipeline import calibrate_egenes

        n = 50
        df = self._egenes_df(n_genes=n)
        # first 10 genes have tiny p-values; rest are near-null
        pval_perm = {f"g{i}": 1e-6 if i < 10 else 0.9 for i in range(n)}
        out = calibrate_egenes(df, pval_perm, fdr_threshold=0.05)
        n_sig = (out["significance"] == "T").sum()
        assert n_sig >= 5, f"expected ≥5 significant genes, got {n_sig}"

    def test_high_pval_not_significant(self):
        """Genes with p ≈ 1 are not called significant."""
        from mosaiqtl.pipeline import calibrate_egenes

        df = self._egenes_df(n_genes=10)
        pval_perm = {f"g{i}": 0.99 for i in range(10)}
        out = calibrate_egenes(df, pval_perm)
        assert (out["significance"] == "F").all()

    def test_row_count_preserved(self):
        """Output has exactly the same number of rows as input."""
        from mosaiqtl.pipeline import calibrate_egenes

        n = 12
        df = self._egenes_df(n_genes=n)
        pval_perm = {f"g{i}": 0.1 for i in range(n)}
        out = calibrate_egenes(df, pval_perm)
        assert len(out) == n

    def test_perm_n_records_null_draws(self):
        """perm_n carries the per-gene null draw count on every row."""
        from mosaiqtl.pipeline import calibrate_egenes

        df = self._egenes_df(n_genes=6)
        pval_perm = {f"g{i}": 0.1 for i in range(6)}
        out = calibrate_egenes(df, pval_perm, n_perms=777)
        assert (out["perm_n"] == 777).all()


# ── Storey FDR end-to-end integration (SWE-09) ───────────────────────────────


class TestStoreyFdrEndToEnd:
    """Full chain: calibrate_gene_pvalue → calibrate_egenes (SWE-09).

    Uses tiny_cache + tiny_kinship fixtures so no external data is needed.
    Runs calibrate_gene_pvalue on a small set of fabricated genes, pools
    their Beta-calibrated p-values, and feeds them into calibrate_egenes
    to get final q-value significance calls.
    """

    def test_full_chain_runs_and_gives_valid_output(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        """End-to-end: calibrate_gene_pvalue → calibrate_egenes.

        Two fabricated genes: one with a genuinely high peak LOD (signal),
        one with a near-zero LOD (noise).  After going through the full
        chain the chain must at minimum produce finite q-values in [0,1]
        and not crash.
        """
        import pandas as pd
        from mosaiqtl.genotype.kinship import decomp_kinship
        from mosaiqtl.qtl.fdr import calibrate_gene_pvalue
        from mosaiqtl.pipeline import calibrate_egenes

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

        out = calibrate_egenes(egenes_df, pval_perm)

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


class TestImputation:
    def test_concat_with_array_attrs_does_not_raise(self):
        """Regression: pd.concat raises ValueError when DataFrame.attrs contains
        numpy arrays — impute_egenes must clear attrs before concat."""
        rng = np.random.default_rng(0)
        frames = []
        for _ in range(3):
            df = pd.DataFrame({'a': rng.random(5)})
            df.attrs['_ld'] = {'r2_matrix': rng.random((4, 4)), 'rep_patterns': []}
            df.attrs['_snp_patterns'] = rng.random((5, 4))
            # simulate what impute_egenes does: extract, clear, then append
            _ = df.attrs.pop('_ld', None)
            _ = df.attrs.pop('_snp_patterns', None)
            df.attrs.clear()
            frames.append(df)
        # must not raise
        result = pd.concat(frames, ignore_index=True)
        assert len(result) == 15


class TestCallRule:
    """calibrate_egenes: call_rule selects fdr and significance."""

    @staticmethod
    def _inputs():
        import pandas as pd

        rng = np.random.default_rng(4)
        pv = {f"g{i}": float(rng.uniform(0, 1e-6)) for i in range(950)}
        pv.update({f"h{i}": float(rng.uniform(0.6, 1.0)) for i in range(50)})
        df = pd.DataFrame({"gene_id": list(pv), "peak_lod": np.linspace(1.0, 5.0, len(pv))})
        return df, pv

    def test_both_adjusted_values_and_pi0_are_always_written(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        out = calibrate_egenes(df, pv)
        for col in ("qval_storey", "qval_bh", "lfdr", "pi0"):
            assert col in out.columns

    def test_lfdr_rule_thresholds_the_local_rate(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        out = calibrate_egenes(df, pv, call_rule="lfdr")
        assert (out["fdr_method"] == "lfdr").all()
        assert np.allclose(out["fdr"], out["lfdr"])
        assert ((out["lfdr"] <= 0.05) == (out["significance"] == "T")).all()

    def test_lfdr_rule_needs_enough_genes(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        small = df.iloc[:5].copy()
        with pytest.raises(ValueError, match="lfdr"):
            calibrate_egenes(small, pv, call_rule="lfdr")
        out = calibrate_egenes(small, pv)
        assert out["lfdr"].isna().all()

    def test_bh_rule_does_not_call_the_late_genes(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        # pi0 is about 0.14 here, so a 0.2 cutoff lies above the largest Storey
        # q-value (pi0 * p_(m)) and every gene passes it; BH keeps the late genes out.
        storey = calibrate_egenes(df, pv, fdr_threshold=0.2, call_rule="qvalue")
        bh = calibrate_egenes(df, pv, fdr_threshold=0.2, call_rule="bh")
        late = bh["gene_id"].str.startswith("h")
        assert storey["pi0"].iloc[0] < 0.2
        assert (storey["significance"] == "T").all()
        assert (bh.loc[late, "significance"] == "F").all()
        assert (bh["fdr_method"] == "BH").all()
        assert (storey["fdr_method"] == "qvalue").all()

    def test_unknown_rule_raises(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        with pytest.raises(ValueError, match="call_rule"):
            calibrate_egenes(df, pv, call_rule="bonferroni")

    def test_lod_threshold_gates_significance_but_not_the_qvalues(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        free = calibrate_egenes(df, pv)
        gated = calibrate_egenes(df, pv, lod_threshold=3.0)
        np.testing.assert_array_equal(gated["qval_bh"], free["qval_bh"])
        np.testing.assert_array_equal(gated["qval_storey"], free["qval_storey"])
        passes_fdr = gated["fdr"] <= 0.05
        above = gated["peak_lod"] > 3.0
        assert ((gated["significance"] == "T") == (passes_fdr & above)).all()
        assert (gated["significance"] == "T").sum() < (free["significance"] == "T").sum()

    def test_lod_equal_to_threshold_is_not_significant(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        thr = float(df["peak_lod"].iloc[10])
        out = calibrate_egenes(df, pv, lod_threshold=thr)
        assert out["significance"].iloc[10] == "F"

    def test_everything_below_the_threshold_is_not_called(self):
        from mosaiqtl.pipeline import calibrate_egenes

        df, pv = self._inputs()
        out = calibrate_egenes(df, pv, lod_threshold=1e6)
        assert (out["significance"] == "F").all()


class TestRintThreading:
    """use_rint is a required keyword on the fine-mapping chain and is forwarded."""

    @pytest.mark.parametrize("name", ["impute_egene", "impute_egenes", "finemap_egenes"])
    def test_use_rint_is_required_keyword(self, name):
        import inspect

        from mosaiqtl import pipeline

        p = inspect.signature(getattr(pipeline, name)).parameters["use_rint"]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY
        assert p.default is inspect.Parameter.empty

    @pytest.mark.parametrize("flag", [True, False])
    def test_impute_egene_forwards_flag(self, monkeypatch, flag):
        from mosaiqtl import pipeline
        from mosaiqtl.qtl import snp_scan

        seen = {}

        def fake_scan(*args, **kwargs):
            seen["use_rint"] = kwargs["use_rint"]
            return None

        monkeypatch.setattr(snp_scan, "imputed_snp_scan_h2lmm", fake_scan)
        monkeypatch.setattr(
            "mosaiqtl.genotype.snp_db.make_alt_mask",
            lambda window, order, ref_founder: np.zeros((len(window), len(order)), dtype=bool),
        )
        alleles = pd.DataFrame({"marker": ["m1"], "chr": ["1"], "pos_bp": [500], "pos_mb": [5e-4]})

        class _GC:
            founders = ["A", "B"]

        pipeline.impute_egene(
            "g1", "1", np.zeros(4), _GC(), alleles, 500, "m1", 3.0, 0, 1000,
            founder_order=["A", "B"], use_rint=flag,
        )
        assert seen["use_rint"] is flag


class TestEnrichPeakLodTss:
    @staticmethod
    def _gene_info():
        return pd.DataFrame({
            "gene_id": ["ENSMUSG1.5", "ENSMUSG2.1"],
            "gene_symbol": ["Aaa", "Bbb"],
            "start": [1_000_000, 5_000_000],
            "end": [1_100_000, 5_100_000],
            "strand": ["-", "+"],
        })

    @staticmethod
    def _peaks(ids):
        return pd.DataFrame({"gene_id": ids, "peak_mb": [1.09, 5.02]})

    def test_minus_strand_uses_end_as_tss(self):
        out = pipeline.enrich_peak_lod(self._peaks(["ENSMUSG1", "ENSMUSG2"]), self._gene_info())
        assert out["start_mb"].tolist() == pytest.approx([1.1, 5.0])
        assert out["cis_dist_mb"].tolist() == pytest.approx([0.01, 0.02])

    def test_cis_dist_negative_means_upstream_on_both_strands(self):
        peaks = pd.DataFrame({"gene_id": ["ENSMUSG1", "ENSMUSG2"], "peak_mb": [1.12, 4.98]})
        out = pipeline.enrich_peak_lod(peaks, self._gene_info())
        assert out["cis_dist_mb"].tolist() == pytest.approx([-0.02, -0.02])

    def test_matches_with_and_without_versions(self):
        for ids in (["ENSMUSG1.5", "ENSMUSG2.1"], ["ENSMUSG1", "ENSMUSG2"], ["ENSMUSG1.9", "ENSMUSG2"]):
            out = pipeline.enrich_peak_lod(self._peaks(ids), self._gene_info())
            assert out["gene_symbol"].tolist() == ["Aaa", "Bbb"]
            assert out["cis_dist_mb"].notna().all()
            assert out["gene_id"].tolist() == ids

    def test_gene_tss_takes_priority_over_gene_info(self):
        tss = pd.DataFrame({
            "gene_id": ["ENSMUSG1.5", "ENSMUSG1.5"],
            "start": [1_000_000, 1_000_000],
            "end": [1_095_000, 1_050_000],
            "strand": ["-", "-"],
        })
        out = pipeline.enrich_peak_lod(self._peaks(["ENSMUSG1", "ENSMUSG2"]), self._gene_info(), tss)
        assert out["near_tss_mb"].iloc[0] == pytest.approx(1.095)
        assert out["cis_dist_mb"].iloc[0] == pytest.approx(0.005)
        assert out["start_mb"].iloc[0] == pytest.approx(1.1)

    def test_fixed_gene_tss_distance_ignores_which_transcript_is_nearest(self):
        tss = pd.DataFrame({
            "gene_id": ["ENSMUSG1.5", "ENSMUSG1.5", "ENSMUSG2.1", "ENSMUSG2.1"],
            "start": [1_000_000, 1_000_000, 5_000_000, 5_030_000],
            "end": [1_095_000, 1_050_000, 5_100_000, 5_100_000],
            "strand": ["-", "-", "+", "+"],
        })
        out = pipeline.enrich_peak_lod(self._peaks(["ENSMUSG1", "ENSMUSG2"]), self._gene_info(), tss)
        assert out["near_tss_mb"].tolist() == pytest.approx([1.095, 5.03])
        assert out["cis_dist_gene_tss_mb"].tolist() == pytest.approx([0.005, 0.02])
        moved = self._peaks(["ENSMUSG1", "ENSMUSG2"]).assign(peak_mb=[1.04, 5.0])
        shifted = pipeline.enrich_peak_lod(moved, self._gene_info(), tss)
        assert shifted["cis_dist_gene_tss_mb"].tolist() == pytest.approx([0.055, 0.0])

    def test_fixed_gene_tss_distance_absent_without_transcript_table(self):
        out = pipeline.enrich_peak_lod(self._peaks(["ENSMUSG1", "ENSMUSG2"]), self._gene_info())
        assert "cis_dist_gene_tss_mb" not in out.columns

    def test_gene_absent_from_gene_tss_falls_back_to_gene_info(self):
        tss = pd.DataFrame({
            "gene_id": ["ENSMUSG1"], "start": [1_000_000], "end": [1_100_000], "strand": ["-"],
        })
        out = pipeline.enrich_peak_lod(self._peaks(["ENSMUSG1", "ENSMUSG2"]), self._gene_info(), tss)
        assert np.isnan(out["near_tss_mb"].iloc[1])
        assert out["cis_dist_mb"].iloc[1] == pytest.approx(0.02)


class TestScannedChromosomes:
    @staticmethod
    def _write(directory, chroms, skip=()):
        for c in chroms:
            for name in (f"lod_chr{c}.csv", f"peak_stats_chr{c}.csv", f"timing_chr{c}.json"):
                if name not in skip:
                    (directory / name).write_text("x\n")

    def test_biological_order(self, tmp_path):
        self._write(tmp_path, ["10", "2", "X", "1"])
        assert pipeline.scanned_chromosomes(tmp_path) == ["1", "2", "10", "X"]

    def test_missing_peak_file_names_it(self, tmp_path):
        self._write(tmp_path, ["1", "2"], skip={"peak_stats_chr2.csv"})
        with pytest.raises(ValueError, match="peak_stats_chr2.csv"):
            pipeline.scanned_chromosomes(tmp_path)

    def test_missing_timing_file_names_it(self, tmp_path):
        self._write(tmp_path, ["1"], skip={"timing_chr1.json"})
        with pytest.raises(ValueError, match="timing_chr1.json"):
            pipeline.scanned_chromosomes(tmp_path)

    def test_no_lod_files(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            pipeline.scanned_chromosomes(tmp_path)


class TestValidateFinemapFrames:
    @staticmethod
    def _fm(rows):
        return pd.DataFrame(rows, columns=["gene_id", "variant"])

    def test_valid_frames_pass(self):
        fm = self._fm([("g1", "a"), ("g1", "b"), ("g2", "a")])
        peak = pd.DataFrame({"gene_id": ["g1", "g2"]})
        pipeline.validate_finemap_frames(fm, peak, {"g1", "g2", "g3"})

    def test_gene_without_output_is_allowed(self):
        fm = self._fm([("g1", "a")])
        pipeline.validate_finemap_frames(fm, pd.DataFrame({"gene_id": ["g1"]}), {"g1", "g2"})

    def test_empty_frames_pass(self):
        pipeline.validate_finemap_frames(
            pd.DataFrame(columns=["gene_id", "variant"]),
            pd.DataFrame(columns=["gene_id"]),
            {"g1"},
        )

    def test_output_gene_not_in_input(self):
        with pytest.raises(ValueError, match="not in the input"):
            pipeline.validate_finemap_frames(
                self._fm([("g9", "a")]), pd.DataFrame({"gene_id": ["g9"]}), {"g1"},
            )

    def test_duplicate_variant_rows(self):
        with pytest.raises(ValueError, match="duplicate"):
            pipeline.validate_finemap_frames(
                self._fm([("g1", "a"), ("g1", "a")]), pd.DataFrame({"gene_id": ["g1"]}), {"g1"},
            )

    def test_peak_with_two_rows_per_gene(self):
        with pytest.raises(ValueError, match="several rows"):
            pipeline.validate_finemap_frames(
                self._fm([("g1", "a")]), pd.DataFrame({"gene_id": ["g1", "g1"]}), {"g1"},
            )

    def test_peak_gene_absent_from_variant_table(self):
        with pytest.raises(ValueError, match="absent from the per-variant"):
            pipeline.validate_finemap_frames(
                self._fm([("g1", "a")]), pd.DataFrame({"gene_id": ["g1", "g2"]}), {"g1", "g2"},
            )


class TestSupportIntervalAndSubmission:
    @staticmethod
    def _summary():
        return pd.DataFrame({
            "gene_id": ["g1", "g2", "g3"], "chr": ["chr1", "chr1", "chr2"],
            "peak_lod": [5.0, 4.0, 6.0], "fdr": [0.01, 0.2, 0.04],
        })

    @staticmethod
    def _ci():
        return pd.DataFrame({
            "gene_id": ["g1", "g3"], "ci_peak_lod": [5.0, 6.0],
            "ci_lo": [1.0, 2.0], "ci_hi": [2.0, 3.5], "ci_width_mb": [1.0, 1.5],
            "ci_truncated": [False, True],
        })

    def test_join_adds_interval_and_leaves_missing_gene_nan(self):
        out = pipeline.attach_support_interval(self._summary(), self._ci())
        assert len(out) == 3
        row = out.set_index("gene_id")
        assert row.loc["g1", "ci_lo"] == 1.0 and row.loc["g3", "ci_hi"] == 3.5
        assert np.isnan(row.loc["g2", "ci_lo"]) and pd.isna(row.loc["g2", "ci_truncated"])
        assert str(out["ci_truncated"].dtype) == "boolean"

    def test_columns_follow_summary_order(self):
        s = self._summary().assign(
            **{c: 0.0 for c in pipeline.EGENE_SUMMARY_COLS if c not in self._summary()}
        )
        out = pipeline.attach_support_interval(s, self._ci())
        assert list(out.columns) == [c for c in pipeline.EGENE_SUMMARY_COLS if c in out.columns]

    def test_replaces_existing_interval_columns(self):
        s = self._summary().assign(ci_lo=-9.0)
        out = pipeline.attach_support_interval(s, self._ci()).set_index("gene_id")
        assert out.loc["g1", "ci_lo"] == 1.0

    def test_header_only_inputs(self):
        out = pipeline.attach_support_interval(self._summary().iloc[0:0], self._ci().iloc[0:0])
        assert len(out) == 0 and "ci_lo" in out.columns

    def test_select_submission_genes_threshold_and_nan(self):
        peak = pd.DataFrame(
            {"gene_id": list("abcd"), "qval_bh": [0.1, 0.1000001, np.nan, 0.0]}
        )
        assert pipeline.select_submission_genes(peak, 0.1)["gene_id"].tolist() == ["a", "d"]

    def test_select_submission_genes_ignores_the_call_rule_column(self):
        peak = pd.DataFrame(
            {"gene_id": list("ab"), "qval_bh": [0.05, 0.5], "fdr": [0.9, 0.01]}
        )
        assert pipeline.select_submission_genes(peak, 0.1)["gene_id"].tolist() == ["a"]



"""Unit tests for qtl/mediation.py — QR-based mediation scan."""

import sys
from pathlib import Path

import numpy as np
import pytest  # noqa: F401

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.qtl.mediation import (
    mediation_scan_qr,
    fixef_mediation_scan,
)

# ── fixef_mediation_scan ──────────────────────────────────────────────────────


class TestFixefMediationScan:
    def test_returns_dataframe(
        self, tiny_cache, tiny_strain_phenotype, tiny_strain_covariates
    ):
        rng = np.random.default_rng(5)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            covariates=tiny_strain_covariates,
        )
        assert hasattr(df, "columns")

    def test_required_columns(
        self, tiny_cache, tiny_strain_phenotype, tiny_strain_covariates
    ):
        rng = np.random.default_rng(6)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
        )
        for col in (
            "outcome_id",
            "mediator_id",
            "locus",
            "chr",
            "pos_mb",
            "lod_base",
            "lod_mediated",
            "lod_drop",
            "prop_mediated",
            "p_value",
            "n",
        ):
            assert col in df.columns

    def test_lod_drop_nonnegative(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(7)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
        )
        assert (df["lod_drop"] >= 0).all()

    def test_row_count(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(8)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert len(df) == total_loci

    def test_just_these_loci_filter(self, tiny_cache, tiny_strain_phenotype):
        lids_ch1 = tiny_cache.get_chromosome("1")[0]
        target = set([lids_ch1[0], lids_ch1[1]])
        rng = np.random.default_rng(9)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            just_these_loci=target,
        )
        assert set(df["locus"]) == target

    def test_outcome_mediator_ids(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(10)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            outcome_id="gene_A",
            mediator_id="gene_B",
        )
        assert (df["outcome_id"] == "gene_A").all()
        assert (df["mediator_id"] == "gene_B").all()

    def test_lod_base_finite(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(11)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
        )
        assert np.all(np.isfinite(df["lod_base"]))

    def test_p_values_in_unit_interval(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(12)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
        )
        assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1.0 + 1e-8).all()

    def test_prop_mediated_bounded(self, tiny_cache, tiny_strain_phenotype):
        rng = np.random.default_rng(13)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = fixef_mediation_scan(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
        )
        assert (df["prop_mediated"] >= 0).all()
        assert (df["prop_mediated"] <= 1.0 + 1e-8).all()


# ── mediation_scan_qr ─────────────────────────────────────────────────────────


class TestMediationScanQr:
    def test_returns_dataframe(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype, tiny_strain_covariates
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(20)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
        )
        assert hasattr(df, "columns")

    def test_required_columns(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(21)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            kinship_decomp=kd,
        )
        for col in (
            "outcome_id",
            "mediator_id",
            "locus",
            "chr",
            "pos_mb",
            "lod_base",
            "lod_mediated",
            "lod_drop",
            "prop_mediated",
            "p_value",
            "h2",
            "n",
        ):
            assert col in df.columns

    def test_lod_drop_nonnegative(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(22)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            kinship_decomp=kd,
        )
        assert (df["lod_drop"] >= 0).all()

    def test_dict_mediators(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        """passing a dict of mediators returns one row-block per mediator."""
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(23)
        n = len(tiny_strain_phenotype)
        mediators = {"m1": rng.standard_normal(n), "m2": rng.standard_normal(n)}
        df = mediation_scan_qr(
            tiny_strain_phenotype,
            mediators,
            tiny_cache,
            kinship_decomp=kd,
        )
        assert set(df["mediator_id"]) == {"m1", "m2"}

    def test_p_values_in_unit_interval(
        self, tiny_cache, tiny_kinship, tiny_strain_phenotype
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(24)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            kinship_decomp=kd,
        )
        assert (df["p_value"] >= 0).all() and (df["p_value"] <= 1.0 + 1e-8).all()

    def test_h2_in_unit_interval(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(25)
        mediator = rng.standard_normal(len(tiny_strain_phenotype))
        df = mediation_scan_qr(
            tiny_strain_phenotype,
            mediator,
            tiny_cache,
            kinship_decomp=kd,
        )
        assert (df["h2"] >= 0).all() and (df["h2"] <= 1.0 + 1e-8).all()

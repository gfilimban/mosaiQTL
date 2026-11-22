"""Unit tests for qtl/effects.py — founder haplotype effect estimation."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

from mosaiqtl.models.records import FounderEffect
from mosaiqtl.qtl.effects import get_allele_effects


def _setup(tiny_cache, tiny_kinship):
    from mosaiqtl.genotype.kinship import decomp_kinship

    kd = decomp_kinship(tiny_kinship)
    chrom = tiny_cache.chromosomes[0]
    locus_id = str(tiny_cache._data[chrom]["locus_ids"][0])
    return kd, chrom, locus_id


class TestGetAlleleEffectsFixef:
    def test_returns_list_of_founder_effects(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        assert isinstance(effects, list)
        assert all(isinstance(e, FounderEffect) for e in effects)

    def test_length_equals_n_founders(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        assert len(effects) == tiny_cache.n_founders

    def test_se_positive(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        assert all(e.se > 0 for e in effects)

    def test_ci_spans_estimate(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        for e in effects:
            assert e.ci_lo <= e.estimate <= e.ci_hi

    def test_betas_sum_near_zero(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        # minimum-norm pinv solution ⊥ null(G) → sum ≈ 0
        betas = np.array([e.estimate for e in effects])
        assert abs(betas.sum()) < 1e-8

    def test_p_values_in_unit_interval(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        for e in effects:
            assert 0.0 <= e.p_value <= 1.0

    def test_invalid_locus_raises(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, _ = _setup(tiny_cache, tiny_kinship)
        with pytest.raises(ValueError, match="not found"):
            get_allele_effects(
                tiny_strain_phenotype,
                tiny_cache,
                "nonexistent_locus",
                chrom,
                kd,
                use_rint=False,
                method="fixef",
            )

    def test_invalid_method_raises(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        with pytest.raises(ValueError, match="method must be"):
            get_allele_effects(
                tiny_strain_phenotype,
                tiny_cache,
                locus_id,
                chrom,
                kd,
                use_rint=False,
                method="bad",
            )


class TestGetAlleleEffectsRanef:
    def test_returns_list_of_founder_effects(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="ranef",
        )
        assert isinstance(effects, list)
        assert all(isinstance(e, FounderEffect) for e in effects)

    def test_length_equals_n_founders(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="ranef",
        )
        assert len(effects) == tiny_cache.n_founders

    def test_se_positive(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="ranef",
        )
        assert all(e.se > 0 for e in effects)

    def test_ranef_shrinks_toward_zero(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        """Ridge shrinkage: ranef norm ≤ fixef norm."""
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        fx = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        rn = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="ranef",
        )
        b_fx = np.array([e.estimate for e in fx])
        b_rn = np.array([e.estimate for e in rn])
        assert np.linalg.norm(b_rn) <= np.linalg.norm(b_fx) + 1e-10

    def test_ranef_betas_differ_from_fixef(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        """ranef applies actual regularization — output must differ from fixef."""
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        fx = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="fixef",
        )
        rn = get_allele_effects(
            tiny_strain_phenotype,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            use_rint=False,
            method="ranef",
        )
        b_fx = np.array([e.estimate for e in fx])
        b_rn = np.array([e.estimate for e in rn])
        # ridge shrinks toward zero; exact match would indicate λ≈0 (degenerate)
        assert not np.allclose(b_rn, b_fx, atol=1e-6)


class TestRanefInference:
    def test_ranef_has_no_p_value(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype, tiny_cache, locus_id, chrom, kd,
            use_rint=False, method="ranef",
        )
        assert all(np.isnan(e.p_value) for e in effects)

    def test_fixef_keeps_test_and_interval(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_strain_phenotype,
    ):
        kd, chrom, locus_id = _setup(tiny_cache, tiny_kinship)
        effects = get_allele_effects(
            tiny_strain_phenotype, tiny_cache, locus_id, chrom, kd,
            use_rint=False, method="fixef",
        )
        assert all(np.isfinite(e.p_value) for e in effects)
        assert all(e.ci_lo < e.estimate < e.ci_hi for e in effects)


class TestGetAlleleEffectsZeroReplicateStrains:
    """regression: strains with no valid replicates for a gene must not cause SVD failure.

    when sample_ids is provided (replicate-aware mode) and all samples for a strain
    are NaN in the phenotype, that strain has r_counts==0 after the Z-matrix collapse.
    dividing by zero produces NaN in y/x → NaN in G → np.linalg.matrix_rank SVD failure.
    the fix drops those strains, re-eigendecomposes the kinship submatrix, and subsets h.
    """

    def test_zero_replicate_strain_does_not_raise(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_phenotype,
        sample_strain_map,
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        chrom = tiny_cache.chromosomes[0]
        locus_id = str(tiny_cache._data[chrom]['locus_ids'][0])

        # NaN out the first 4 samples (all replicates for strain CC001)
        # so that strain 0 has r_counts==0 after Z-matrix collapse
        pheno = tiny_phenotype.copy().astype(float)
        pheno[:4] = np.nan

        effects = get_allele_effects(
            pheno,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            sample_ids=list(sample_strain_map),
            use_rint=False,
            method='fixef',
        )
        assert isinstance(effects, list)
        assert len(effects) == tiny_cache.n_founders
        assert all(isinstance(e, FounderEffect) for e in effects)

    def test_zero_replicate_strain_effects_finite(
        self,
        tiny_cache,
        tiny_kinship,
        tiny_phenotype,
        sample_strain_map,
    ):
        from mosaiqtl.genotype.kinship import decomp_kinship

        kd = decomp_kinship(tiny_kinship)
        chrom = tiny_cache.chromosomes[0]
        locus_id = str(tiny_cache._data[chrom]['locus_ids'][0])

        pheno = tiny_phenotype.copy().astype(float)
        pheno[:4] = np.nan

        effects = get_allele_effects(
            pheno,
            tiny_cache,
            locus_id,
            chrom,
            kd,
            sample_ids=list(sample_strain_map),
            use_rint=False,
            method='fixef',
        )
        assert not np.allclose([e.estimate for e in effects], 0.0)
        for e in effects:
            assert np.isfinite(e.estimate), f'non-finite estimate for {e.founder}'
            assert np.isfinite(e.se), f'non-finite SE for {e.founder}'

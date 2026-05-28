"""Unit tests for qtl/scan.py — ScanResult and scan_h2lmm."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.scan import (
    ScanResult, scan_h2lmm, scan_multi_gene,
    grab_locus_from_scan, lod_drop_ci, find_peaks,
    fixef_eqtl_scan, instability_lm_scan,
)


# ── ScanResult ────────────────────────────────────────────────────────────────

def _make_scan_result(n_loci=20, n_founders=8):
    """Helper: build a minimal ScanResult with known values."""
    rng = np.random.default_rng(0)
    lods = np.abs(rng.standard_normal(n_loci))
    return ScanResult(
        locus_ids   = np.array([f'L{i}' for i in range(n_loci)]),
        chromosomes = np.array(['1'] * (n_loci // 2) + ['2'] * (n_loci - n_loci // 2)),
        positions   = np.linspace(1.0, 100.0, n_loci),
        lod         = lods,
        p_value     = np.clip(rng.uniform(0, 0.5, n_loci), 0, 1),
        betas       = rng.standard_normal((n_loci, n_founders - 1)),
        h2_null     = {'1': 0.3, '2': 0.4},
        phenotype_id = 'test_gene',
    )


class TestScanResult:
    def test_peak_locus_keys(self):
        sr = _make_scan_result()
        peak = sr.peak_locus()
        for key in ('locus_id', 'chromosome', 'position', 'lod', 'p_value', 'betas'):
            assert key in peak

    def test_peak_locus_is_max(self):
        sr = _make_scan_result()
        peak = sr.peak_locus()
        assert peak['lod'] == pytest.approx(sr.lod.max())

    def test_to_dataframe_shape(self):
        n = 10
        sr = _make_scan_result(n_loci=n)
        df = sr.to_dataframe()
        assert len(df) == n

    def test_to_dataframe_columns(self):
        sr = _make_scan_result()
        df = sr.to_dataframe()
        for col in ('locus_id', 'chromosome', 'position', 'lod', 'p_value'):
            assert col in df.columns

    def test_to_dataframe_beta_columns(self):
        sr = _make_scan_result(n_founders=8)
        df = sr.to_dataframe()
        for fi in range(7):
            assert f'beta_f{fi}' in df.columns

    def test_attributes_accessible(self):
        sr = _make_scan_result()
        assert hasattr(sr, 'lod')
        assert hasattr(sr, 'chromosomes')
        assert hasattr(sr, 'h2_null')
        assert hasattr(sr, 'phenotype_id')


# ── scan_h2lmm ────────────────────────────────────────────────────────────────

class TestScanH2Lmm:
    def test_returns_scan_result(self, tiny_cache, tiny_kinship,
                                  tiny_strain_phenotype, tiny_strain_covariates):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            covariates=tiny_strain_covariates,
            kinship_decomp=kd,
            use_rint=False,
            phenotype_id='test',
            scan_window='genome',
        )
        assert isinstance(result, ScanResult)

    def test_lod_shape(self, tiny_cache, tiny_kinship,
                       tiny_strain_phenotype, tiny_strain_covariates):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert result.lod.shape == (total_loci,)

    def test_lod_finite(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )
        assert np.all(np.isfinite(result.lod))

    def test_lod_nonnegative(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )
        assert np.all(result.lod >= -1e-8)

    def test_p_values_in_unit_interval(self, tiny_cache, tiny_kinship,
                                        tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )
        assert np.all(result.p_value >= 0.0)
        assert np.all(result.p_value <= 1.0 + 1e-8)

    def test_h2_null_per_chrom(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )
        for c in tiny_cache.chromosomes:
            assert c in result.h2_null
            assert 0.0 <= result.h2_null[c] <= 1.0

    def test_phenotype_id_stored(self, tiny_cache, tiny_kinship,
                                  tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            phenotype_id='my_gene',
            scan_window='genome',
        )
        assert result.phenotype_id == 'my_gene'

    def test_covariates_affect_lod(self, tiny_cache, tiny_kinship,
                                    tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        rng = np.random.default_rng(99)
        # use a non-trivial covariate (intercept + random noise) so it differs
        # from the default intercept-only model
        x_extra = np.column_stack([
            np.ones(len(tiny_strain_phenotype)),
            rng.standard_normal(len(tiny_strain_phenotype)),
        ])
        r_no_cov   = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                  kinship_decomp=kd, use_rint=False,
                                  scan_window='genome')
        r_with_cov = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                  covariates=x_extra,
                                  kinship_decomp=kd, use_rint=False,
                                  scan_window='genome')
        # LOD profiles differ when informative covariates are included
        assert not np.allclose(r_no_cov.lod, r_with_cov.lod)

    def test_betas_shape(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=kd,
            use_rint=False,
            scan_window='genome',
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert result.betas.shape == (total_loci, tiny_cache.n_founders)

    def test_no_kinship_auto_computes(self, tiny_cache, tiny_strain_phenotype):
        # kinship_decomp=None → computed internally from genomecache
        result = scan_h2lmm(
            tiny_strain_phenotype, tiny_cache,
            kinship_decomp=None,
            use_rint=False,
            scan_window='genome',
        )
        assert isinstance(result, ScanResult)

    def test_rint_changes_lod(self, tiny_cache, tiny_kinship,
                               tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        r_no_rint   = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                   kinship_decomp=kd, use_rint=False,
                                   scan_window='genome')
        r_with_rint = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                                   kinship_decomp=kd, use_rint=True,
                                   scan_window='genome')
        assert not np.allclose(r_no_rint.lod, r_with_rint.lod)


# ── grab_locus_from_scan ──────────────────────────────────────────────────────

class TestGrabLocus:
    def test_returns_locus_with_max_lod(self):
        sr = _make_scan_result()
        lid = grab_locus_from_scan(sr)
        assert lid == sr.locus_ids[np.argmax(sr.lod)]


# ── lod_drop_ci ───────────────────────────────────────────────────────────────

class TestLodDropCi:
    def test_returns_three_floats(self, tiny_cache, tiny_kinship,
                                   tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                         kinship_decomp=kd, use_rint=False,
                         scan_window='genome')
        lo, hi, pk = lod_drop_ci(sr, chrom='1', drop=1.5)
        assert np.isfinite(lo) and np.isfinite(hi) and np.isfinite(pk)

    def test_lo_le_peak_le_hi(self, tiny_cache, tiny_kinship,
                                tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        sr = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                         kinship_decomp=kd, use_rint=False,
                         scan_window='genome')
        lo, hi, pk = lod_drop_ci(sr, chrom='1', drop=1.5)
        assert lo <= pk <= hi

    def test_empty_chrom_returns_nan(self):
        sr = _make_scan_result()
        lo, hi, pk = lod_drop_ci(sr, chrom='99')
        assert np.isnan(lo) and np.isnan(hi) and np.isnan(pk)


# ── fixef_eqtl_scan ───────────────────────────────────────────────────────────

class TestFixefEqtlScan:
    def test_returns_scan_result(self, tiny_cache, tiny_strain_phenotype):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache, use_rint=False)
        assert isinstance(sr, ScanResult)

    def test_lod_nonneg(self, tiny_cache, tiny_strain_phenotype):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache, use_rint=False)
        assert (sr.lod >= -1e-8).all()

    def test_just_these_loci(self, tiny_cache, tiny_strain_phenotype):
        lids, _, _ = tiny_cache.get_chromosome('1')
        subset = {lids[0], lids[1]}
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache,
                              use_rint=False, just_these_loci=subset)
        assert set(sr.locus_ids) == subset

    def test_with_covariates(self, tiny_cache, tiny_strain_phenotype,
                              tiny_strain_covariates):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache,
                              covariates=tiny_strain_covariates,
                              use_rint=False)
        assert isinstance(sr, ScanResult)

    def test_phenotype_id_stored(self, tiny_cache, tiny_strain_phenotype):
        sr = fixef_eqtl_scan(tiny_strain_phenotype, tiny_cache,
                              phenotype_id='gene_x', use_rint=False)
        assert sr.phenotype_id == 'gene_x'


# ── scan_multi_gene ───────────────────────────────────────────────────────────

class TestScanMultiGene:
    def test_returns_dict(self, tiny_cache, tiny_kinship, tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        n = len(tiny_strain_phenotype)
        pheno_mat = np.column_stack([
            tiny_strain_phenotype,
            np.random.default_rng(1).standard_normal(n),
        ])
        results = scan_multi_gene(
            pheno_mat, ['gene_a', 'gene_b'], tiny_cache,
            kinship_decomp=kd,
        )
        assert isinstance(results, dict)
        assert set(results.keys()) == {'gene_a', 'gene_b'}

    def test_each_value_is_scan_result(self, tiny_cache, tiny_kinship,
                                        tiny_strain_phenotype):
        from mosaiqtl.genotype.kinship import decomp_kinship
        kd = decomp_kinship(tiny_kinship)
        pheno_mat = tiny_strain_phenotype[:, None]
        results = scan_multi_gene(
            pheno_mat, ['gene_a'], tiny_cache, kinship_decomp=kd,
        )
        assert isinstance(results['gene_a'], ScanResult)


# ── instability_lm_scan ───────────────────────────────────────────────────────

class TestInstabilityLmScan:
    def test_returns_scan_result(self, tiny_cache, tiny_strain_phenotype):
        sr = instability_lm_scan(
            tiny_strain_phenotype, tiny_cache, use_rint=False,
            n_subsamples=5, seed=0,
        )
        assert isinstance(sr, ScanResult)

    def test_lod_shape(self, tiny_cache, tiny_strain_phenotype):
        sr = instability_lm_scan(
            tiny_strain_phenotype, tiny_cache, use_rint=False,
            n_subsamples=3, seed=0,
        )
        total_loci = sum(tiny_cache.n_loci(c) for c in tiny_cache.chromosomes)
        assert sr.lod.shape[0] == total_loci

    def test_p_value_in_unit_interval(self, tiny_cache, tiny_strain_phenotype):
        sr = instability_lm_scan(
            tiny_strain_phenotype, tiny_cache, use_rint=False,
            n_subsamples=3, seed=1,
        )
        assert ((sr.p_value >= 0) & (sr.p_value <= 1 + 1e-8)).all()


# ── find_peaks ────────────────────────────────────────────────────────────────

def _make_scan_with_peak(peak_chr='1', peak_pos=50.0, peak_lod=8.0,
                          n_loci_per_chr=20, n_chr=2):
    """Build a ScanResult with one clear planted peak."""
    rng = np.random.default_rng(7)
    n = n_loci_per_chr * n_chr
    chrs = np.array([str(c + 1) for c in range(n_chr)
                     for _ in range(n_loci_per_chr)])
    pos  = np.concatenate([np.linspace(1.0, 100.0, n_loci_per_chr)
                            for _ in range(n_chr)])
    lod  = rng.uniform(0, 2, n)
    # plant a clear peak on peak_chr at pos nearest to peak_pos
    mask = chrs == peak_chr
    idx_chr = np.where(mask)[0]
    nearest = idx_chr[np.argmin(np.abs(pos[idx_chr] - peak_pos))]
    lod[nearest] = peak_lod
    lids = np.array([f'chr{chrs[i]}_L{i}' for i in range(n)])
    return ScanResult(
        locus_ids=lids, chromosomes=chrs, positions=pos,
        lod=lod, p_value=np.ones(n) * 0.01,
        betas=rng.standard_normal((n, 8)),
        h2_null={str(c + 1): 0.3 for c in range(n_chr)},
        phenotype_id='test_gene',
    )


class TestFindPeaks:
    def test_returns_dataframe(self):
        sr = _make_scan_with_peak()
        df = find_peaks(sr, threshold=5.0)
        assert isinstance(df, pd.DataFrame)

    def test_required_columns(self):
        sr = _make_scan_with_peak()
        df = find_peaks(sr, threshold=5.0)
        for col in ('phenotype_id', 'chr', 'locus_id', 'pos_mb',
                    'lod', 'ci_lo_mb', 'ci_hi_mb'):
            assert col in df.columns

    def test_peak_above_threshold(self):
        sr = _make_scan_with_peak(peak_lod=8.0)
        df = find_peaks(sr, threshold=5.0)
        assert len(df) >= 1
        assert (df['lod'] >= 5.0).all()

    def test_below_threshold_returns_empty(self):
        sr = _make_scan_with_peak(peak_lod=4.0)
        df = find_peaks(sr, threshold=5.0)
        assert len(df) == 0
        # columns still present even when empty
        assert 'lod' in df.columns

    def test_ci_contains_peak(self):
        sr = _make_scan_with_peak(peak_lod=8.0)
        df = find_peaks(sr, threshold=5.0)
        assert len(df) >= 1
        assert (df['ci_lo_mb'] <= df['pos_mb']).all()
        assert (df['pos_mb'] <= df['ci_hi_mb']).all()

    def test_one_per_chrom_by_default(self):
        # two chromosomes, each with a peak above threshold
        sr = _make_scan_with_peak(peak_chr='1', peak_lod=8.0, n_chr=2)
        # add a second peak on chr2
        sr2 = _make_scan_with_peak(peak_chr='2', peak_lod=8.0, n_chr=2)
        import numpy as _np
        sr.lod = _np.maximum(sr.lod, sr2.lod)
        df = find_peaks(sr, threshold=5.0)
        for chrom in df['chr'].unique():
            assert (df['chr'] == chrom).sum() == 1

    def test_phenotype_id_propagated(self):
        sr = _make_scan_with_peak(peak_lod=8.0)
        df = find_peaks(sr, threshold=5.0)
        assert (df['phenotype_id'] == 'test_gene').all()

    def test_planted_peak_recovered(self):
        # plant a wide LOD bump so the CI spans across the planted position
        rng = np.random.default_rng(7)
        n = 50
        pos = np.linspace(1.0, 100.0, n)
        # Gaussian bump centred at 50 Mb, height 10, width ~15 Mb
        lod = 10.0 * np.exp(-0.5 * ((pos - 50.0) / 15.0) ** 2)
        lod += rng.uniform(0, 0.1, n)
        sr = ScanResult(
            locus_ids=np.array([f'L{i}' for i in range(n)]),
            chromosomes=np.array(['1'] * n),
            positions=pos, lod=lod,
            p_value=np.ones(n) * 0.01,
            betas=rng.standard_normal((n, 8)),
            h2_null={'1': 0.3},
            phenotype_id='test_gene',
        )
        df = find_peaks(sr, threshold=5.0)
        assert len(df) == 1
        row = df.iloc[0]
        # planted peak at 50 Mb is within the 1.5-LOD-drop CI
        assert row['ci_lo_mb'] <= 50.0 <= row['ci_hi_mb']

    def test_allow_multiple_finds_secondary(self):
        # plant two well-separated peaks on the same chromosome
        rng = np.random.default_rng(42)
        n = 40
        pos = np.linspace(1.0, 200.0, n)
        lod = rng.uniform(0, 1, n)
        lod[5]  = 9.0   # primary   ~26 Mb
        lod[35] = 8.0   # secondary ~180 Mb — well separated
        sr = ScanResult(
            locus_ids=np.array([f'L{i}' for i in range(n)]),
            chromosomes=np.array(['1'] * n),
            positions=pos, lod=lod,
            p_value=np.ones(n) * 0.01,
            betas=rng.standard_normal((n, 8)),
            h2_null={'1': 0.3},
            phenotype_id='multi',
        )
        df = find_peaks(sr, threshold=5.0, allow_multiple=True, min_lod_gap=1.5)
        assert len(df) == 2

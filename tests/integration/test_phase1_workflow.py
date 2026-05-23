"""Integration tests for Phase 1 workflow — minimal miQTL mirror.

Mirrors the exact call sequence in validate_sim.py using tiny conftest
fixtures. No sparcc_cache required — these run in CI without real data.

Tests verify:
  - Each step of the Phase 1 workflow runs without error
  - Output types and shapes are correct
  - The three output streams (Python, comparison structure) are producible
  - Workflow is deterministic given the same seed
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

from mosaiqtl.qtl.scan import ScanResult, scan_h2lmm, scan_h2lmm_mi
from mosaiqtl.qtl.permutation import (
    generate_null_outcomes,
    get_gev_thresholds,
    run_threshold_scans,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _scan_to_manhattan_dict(scan):
    """build {chrom: (positions, lod)} for manhattan()."""
    out = {}
    for chrom in np.unique(scan.chromosomes):
        mask = scan.chromosomes == chrom
        out[str(chrom)] = (scan.positions[mask], scan.lod[mask])
    return out


# ── P1-S1: ROP scan ───────────────────────────────────────────────────────────

class TestP1ROPScan:
    def test_returns_scan_result(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                            phenotype_id='test', scan_window='genome')
        assert isinstance(result, ScanResult)

    def test_lod_shape_matches_loci(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        n_loci = sum(
            len(tiny_cache.get_chromosome(c)[0])
            for c in tiny_cache.chromosomes
        )
        assert result.lod.shape == (n_loci,)

    def test_lod_nonnegative(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        assert np.all(result.lod >= 0)

    def test_chromosomes_are_strings(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        assert all(isinstance(c, str) for c in result.chromosomes)

    def test_peak_locus_in_result(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        peak = result.peak_locus()
        assert peak['lod'] == pytest.approx(float(result.lod.max()))

    def test_manhattan_dict_structure(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        d = _scan_to_manhattan_dict(result)
        assert set(d.keys()) == set(str(c) for c in tiny_cache.chromosomes)
        for pos, lod in d.values():
            assert pos.shape == lod.shape
            assert np.all(lod >= 0)

    def test_deterministic(self, tiny_strain_phenotype, tiny_cache):
        r1 = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        r2 = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        np.testing.assert_array_equal(r1.lod, r2.lod)


# ── P1-S2: MI scan ────────────────────────────────────────────────────────────

class TestP1MIScan:
    def test_returns_scan_result(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm_mi(tiny_strain_phenotype, tiny_cache,
                               n_imp=3, seed=1, scan_window='genome')
        assert isinstance(result, ScanResult)

    def test_lod_shape_matches_rop(self, tiny_strain_phenotype, tiny_cache):
        rop = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        mi  = scan_h2lmm_mi(tiny_strain_phenotype, tiny_cache, n_imp=3, seed=1, scan_window='genome')
        assert mi.lod.shape == rop.lod.shape

    def test_lod_nonnegative(self, tiny_strain_phenotype, tiny_cache):
        result = scan_h2lmm_mi(tiny_strain_phenotype, tiny_cache, n_imp=3, seed=1, scan_window='genome')
        assert np.all(result.lod >= 0)

    def test_locus_ids_match_rop(self, tiny_strain_phenotype, tiny_cache):
        rop = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        mi  = scan_h2lmm_mi(tiny_strain_phenotype, tiny_cache, n_imp=3, seed=1, scan_window='genome')
        np.testing.assert_array_equal(rop.locus_ids, mi.locus_ids)

    def test_seed_reproducible(self, tiny_strain_phenotype, tiny_cache):
        mi1 = scan_h2lmm_mi(tiny_strain_phenotype, tiny_cache, n_imp=3, seed=7, scan_window='genome')
        mi2 = scan_h2lmm_mi(tiny_strain_phenotype, tiny_cache, n_imp=3, seed=7, scan_window='genome')
        np.testing.assert_array_equal(mi1.lod, mi2.lod)


# ── P1-S3: permutation + GEV threshold ───────────────────────────────────────

class TestP1Permutation:
    def test_null_outcomes_shape(self, tiny_strain_phenotype):
        null = generate_null_outcomes(tiny_strain_phenotype,
                                      n_perms=5, method='permutation', seed=1)
        assert null.shape == (len(tiny_strain_phenotype), 5)

    def test_run_threshold_scans_returns_max_lods(
            self, tiny_strain_phenotype, tiny_cache):
        null = generate_null_outcomes(tiny_strain_phenotype,
                                      n_perms=5, method='permutation', seed=1)
        # Phase 1 uses plain shuffle (no LOCO kinship — OLS baseline)
        max_lods, _ = run_threshold_scans(null, tiny_cache,
                                          phenotype=tiny_strain_phenotype,
                                          fix_h2=True, use_rint=True,
                                          permutation_method='plain')
        assert max_lods.shape == (5,)
        assert np.all(max_lods >= 0)

    def test_gev_threshold_is_scalar(self, tiny_strain_phenotype, tiny_cache):
        null = generate_null_outcomes(tiny_strain_phenotype,
                                      n_perms=15, method='permutation', seed=1)
        max_lods, _ = run_threshold_scans(null, tiny_cache,
                                          phenotype=tiny_strain_phenotype,
                                          fix_h2=True, use_rint=True,
                                          permutation_method='plain')
        threshold, params = get_gev_thresholds(max_lods, percentile=0.95)
        assert np.isfinite(threshold)
        assert threshold > 0
        assert 'empirical_quantile' in params

    def test_full_workflow_deterministic(self, tiny_strain_phenotype, tiny_cache):
        """full Phase 1 permutation workflow runs twice and gives same threshold."""
        def _run():
            null = generate_null_outcomes(tiny_strain_phenotype,
                                          n_perms=15, method='permutation', seed=42)
            max_lods, _ = run_threshold_scans(null, tiny_cache,
                                              phenotype=tiny_strain_phenotype,
                                              fix_h2=True, use_rint=True,
                                              permutation_method='plain')
            t, _ = get_gev_thresholds(max_lods, percentile=0.95)
            return t

        assert _run() == pytest.approx(_run())

    def test_threshold_exceeds_typical_lod(
            self, tiny_strain_phenotype, tiny_cache):
        """95th percentile threshold should exceed near-null LOD scores."""
        null = generate_null_outcomes(tiny_strain_phenotype,
                                      n_perms=20, method='permutation', seed=1)
        max_lods, _ = run_threshold_scans(null, tiny_cache,
                                          phenotype=tiny_strain_phenotype,
                                          fix_h2=True, use_rint=True,
                                          permutation_method='plain')
        threshold, _ = get_gev_thresholds(max_lods, percentile=0.95)
        rop = scan_h2lmm(tiny_strain_phenotype, tiny_cache, scan_window='genome')
        # with tiny random data median LOD should be below threshold
        assert float(np.median(rop.lod)) < threshold


# ── end-to-end Phase 1 ────────────────────────────────────────────────────────

class TestP1EndToEnd:
    def test_rop_scan_then_permutation(
            self, tiny_strain_phenotype, tiny_cache):
        """full Phase 1 pipeline: ROP scan → permutation → plot dict — no errors."""
        rop = scan_h2lmm(tiny_strain_phenotype, tiny_cache,
                         phenotype_id='sim.y.1', scan_window='genome')
        assert isinstance(rop, ScanResult)

        null     = generate_null_outcomes(tiny_strain_phenotype,
                                          n_perms=15, method='permutation', seed=1)
        max_lods, _ = run_threshold_scans(null, tiny_cache,
                                          phenotype=tiny_strain_phenotype,
                                          fix_h2=True, use_rint=True,
                                          permutation_method='plain')
        threshold, _ = get_gev_thresholds(max_lods, percentile=0.95)

        d = _scan_to_manhattan_dict(rop)
        assert len(d) == len(tiny_cache.chromosomes)
        assert np.isfinite(threshold)

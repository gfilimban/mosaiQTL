# tests/benchmarks/test_memory.py
# memory footprint checks for MEM-02 through MEM-05 (GOALS.md).
#
# production targets (from GOALS.md):
#   MEM-02  scan_chromosome peak RAM per worker   < 4 GB
#   MEM-03  genome cache in-RAM footprint         < 500 MB
#   MEM-04  kinship LOCO dict in RAM              < 600 MB
#   MEM-05  peak LOD array never materialised     architecture check
#
# CI fixture is intentionally smaller than production (n=40, 500 loci, 3 chr)
# so per-fixture scaled thresholds are used here; production thresholds
# are enforced by the full-scale baseline run (mosaiqtl.bench.baseline).
#
# run:
#   pytest tests/benchmarks/test_memory.py -v
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import tracemalloc
import numpy as np
import pytest

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.lmm import estimate_h2_null, scan_chromosome

# ── fixture sizes (matches test_scan_speed.py) ────────────────────────────────
N_SAMPLES  = 40
N_LOCI     = 500
N_FOUNDERS = 8
N_CHROM    = 3

# per-fixture thresholds scaled from production targets
# production: MEM-02 4 GB at (418, 74_000, 20chr); test: (40, 500, 3chr)
# scan arrays scale as O(n_samples * n_loci * n_founders)
# 4 GB * (40/418) * (500/74_000) * (3/20) ≈ 0.5 MB — use 32 MB as generous CI bound
_SCAN_RAM_LIMIT_MB    = 32    # MEM-02 (scaled)
# production: MEM-03 500 MB at (418, 74_000, 20chr)
# probs array: 40 * 500 * 8 * 8 bytes * 3 chrom ≈ 3.8 MB — use 64 MB
_CACHE_RAM_LIMIT_MB   = 64    # MEM-03 (scaled)
# production: MEM-04 600 MB at (418×418 × 20chr)
# test: 40×40 × 3 ≈ 38 KB — use 8 MB
_KINSHIP_RAM_LIMIT_MB = 8     # MEM-04 (scaled)


def _make_cache(n_samples=N_SAMPLES, n_loci=N_LOCI,
                n_founders=N_FOUNDERS, n_chrom=N_CHROM):
    rng  = np.random.default_rng(42)
    data = {}
    for i in range(n_chrom):
        c = str(i + 1)
        probs = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_samples))
        data[c] = {
            'locus_ids': np.array([f'{c}_L{j:04d}' for j in range(n_loci)]),
            'positions': np.linspace(1.0, 200.0, n_loci),
            'probs':     probs,
        }
    return GenomeCache(
        data=data,
        sample_ids=[f'CC{i:03d}' for i in range(n_samples)],
        founders=[f'F{i}' for i in range(n_founders)],
    )


@pytest.fixture(scope='module')
def cache():
    return _make_cache()


@pytest.fixture(scope='module')
def kinship_dict(cache):
    return calc_kinship(cache, method='loco')


@pytest.fixture(scope='module')
def kd(kinship_dict):
    return decomp_kinship(kinship_dict)


# ── MEM-02: scan_chromosome peak allocation ───────────────────────────────────

class TestScanMemory:
    """MEM-02 — scan_chromosome peak RAM stays below scaled threshold."""

    def test_scan_peak_within_limit(self, cache, kd):
        rng   = np.random.default_rng(7)
        chrom = '1'
        ev    = kd[chrom]['values']
        evc   = kd[chrom]['vectors']
        x     = np.ones((N_SAMPLES, 1))
        y     = rng.standard_normal(N_SAMPLES)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)

        tracemalloc.start()
        scan_chromosome(y, cache, chrom, x, ev, evc, h2)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / 1024 ** 2
        assert peak_mb < _SCAN_RAM_LIMIT_MB, (
            f'scan_chromosome peak {peak_mb:.1f} MB exceeds {_SCAN_RAM_LIMIT_MB} MB limit'
        )

    def test_scan_no_lod_matrix_retained(self, cache, kd):
        """MEM-05 — scan returns per-chr dict, not a materialised (genes×loci) LOD matrix."""
        rng   = np.random.default_rng(7)
        chrom = '1'
        ev    = kd[chrom]['values']
        evc   = kd[chrom]['vectors']
        x     = np.ones((N_SAMPLES, 1))
        y     = rng.standard_normal(N_SAMPLES)
        h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)

        result = scan_chromosome(y, cache, chrom, x, ev, evc, h2)

        # result is a flat per-chr dict, not a 2-D (n_genes × n_loci) array
        assert isinstance(result, dict), 'scan_chromosome must return a dict'
        assert 'lod' in result
        lod = result['lod']
        assert lod.ndim == 1, (
            f'LOD array must be 1-D (n_loci,), got shape {lod.shape} — '
            'materialising a 2-D matrix violates MEM-05'
        )


# ── MEM-03: genome cache footprint ───────────────────────────────────────────

class TestCacheMemory:
    """MEM-03 — GenomeCache in-RAM footprint stays below scaled threshold."""

    def test_cache_peak_within_limit(self):
        tracemalloc.start()
        _cache = _make_cache()  # kept alive during measurement
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del _cache

        peak_mb = peak / 1024 ** 2
        assert peak_mb < _CACHE_RAM_LIMIT_MB, (
            f'GenomeCache construction peak {peak_mb:.1f} MB exceeds {_CACHE_RAM_LIMIT_MB} MB limit'
        )

    def test_cache_probs_shape(self, cache):
        """probs stored as (n_loci, n_samples, n_founders) — no silent transposition."""
        _, _, probs = cache.get_chromosome('1')
        n_loci, n_samples, n_founders = probs.shape
        assert n_loci    == N_LOCI
        assert n_samples == N_SAMPLES
        assert n_founders == N_FOUNDERS


# ── MEM-04: kinship LOCO dict footprint ──────────────────────────────────────

class TestKinshipMemory:
    """MEM-04 — LOCO kinship dict stays below scaled threshold."""

    def test_kinship_peak_within_limit(self, cache):
        tracemalloc.start()
        _kd_local = calc_kinship(cache, method='loco')  # kept alive during measurement
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        del _kd_local

        peak_mb = peak / 1024 ** 2
        assert peak_mb < _KINSHIP_RAM_LIMIT_MB, (
            f'calc_kinship peak {peak_mb:.1f} MB exceeds {_KINSHIP_RAM_LIMIT_MB} MB limit'
        )

    def test_kinship_shape(self, kinship_dict):
        for c, k in kinship_dict.items():
            assert k.shape == (N_SAMPLES, N_SAMPLES), (
                f'kinship chr{c} shape {k.shape} != ({N_SAMPLES},{N_SAMPLES})'
            )

    def test_kinship_symmetric(self, kinship_dict):
        for c, k in kinship_dict.items():
            assert np.allclose(k, k.T, atol=1e-10), f'kinship chr{c} not symmetric'

    def test_decomp_vectors_orthonormal(self, kd):
        for c, d in kd.items():
            evc = d['vectors']
            assert np.allclose(evc.T @ evc, np.eye(N_SAMPLES), atol=1e-10), (
                f'eigenvectors chr{c} not orthonormal'
            )

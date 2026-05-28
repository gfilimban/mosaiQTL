# tests/benchmarks/test_scan_speed.py
# hot-path benchmarks for scan_chromosome, kinship, and h2 estimation.
#
# dataset: synthetic CC-like data (n=40, n_loci=500/chr, n_founders=8).
# sizes chosen to match sparc_cc40 per-chromosome scale (3700 loci/chr)
# at a 7x reduced loci count — fast enough for CI, still exercises the
# vectorized FWL and einsum paths.
#
# run:
#   pytest tests/benchmarks/ --benchmark-autosave \
#          --benchmark-json=benchmarks/baseline/latest.json
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pytest

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.lmm import estimate_h2_null, scan_chromosome

# ── fixture sizes ──────────────────────────────────────────────────────────────
N_STRAINS  = 40      # matches sparc_cc40 strain count
N_LOCI     = 500     # ~7x reduction from 3700/chr — CI-friendly
N_FOUNDERS = 8
N_CHROM    = 3       # 3 representative chromosomes


def _make_cache(n_strains=N_STRAINS, n_loci=N_LOCI,
                n_founders=N_FOUNDERS, n_chrom=N_CHROM):
    rng  = np.random.default_rng(42)
    data = {}
    for i in range(n_chrom):
        c = str(i + 1)
        probs = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_strains))
        data[c] = {
            'locus_ids': np.array([f'{c}_L{j:04d}' for j in range(n_loci)]),
            'positions': np.linspace(1.0, 200.0, n_loci),
            'probs':     probs,
        }
    return GenomeCache(
        data=data,
        sample_ids=[f'CC{i:03d}' for i in range(n_strains)],
        founders=[f'F{i}' for i in range(n_founders)],
    )


@pytest.fixture(scope='module')
def cache():
    return _make_cache()


@pytest.fixture(scope='module')
def kinship_decomp(cache):
    kd = calc_kinship(cache, method='loco')
    return decomp_kinship(kd)


@pytest.fixture(scope='module')
def scan_inputs(cache, kinship_decomp):
    """pre-compute shared scan inputs so benchmark measures scan only."""
    rng   = np.random.default_rng(7)
    chrom = '1'
    ev    = kinship_decomp[chrom]['values']
    evc   = kinship_decomp[chrom]['vectors']
    x     = np.ones((N_SAMPLES, 1))
    y     = rng.standard_normal(N_SAMPLES)
    h2, _ = estimate_h2_null(evc.T @ y, evc.T @ x, ev)
    return {'y': y, 'x': x, 'ev': ev, 'evc': evc, 'h2': h2, 'chrom': chrom}


# ── benchmarks ─────────────────────────────────────────────────────────────────

def test_bench_calc_kinship_loco(benchmark, cache):
    """LOCO kinship over N_CHROM chromosomes × N_LOCI loci × N_SAMPLES samples."""
    benchmark(calc_kinship, cache, method='loco')


def test_bench_decomp_kinship(benchmark, cache):
    """eigendecomposition of LOCO kinship dict (N_CHROM matrices, N_SAMPLES×N_SAMPLES)."""
    kd = calc_kinship(cache, method='loco')
    benchmark(decomp_kinship, kd)


def test_bench_estimate_h2_null(benchmark, scan_inputs):
    """REML h² estimation: 20-point grid + Brent refinement."""
    ev  = scan_inputs['ev']
    evc = scan_inputs['evc']
    x   = scan_inputs['x']
    y   = scan_inputs['y']
    benchmark(estimate_h2_null, evc.T @ y, evc.T @ x, ev)


def test_bench_scan_chromosome(benchmark, cache, scan_inputs):
    """vectorised FWL scan over N_LOCI loci — the primary hot path."""
    benchmark(
        scan_chromosome,
        scan_inputs['y'], cache, scan_inputs['chrom'],
        scan_inputs['x'], scan_inputs['ev'], scan_inputs['evc'],
        scan_inputs['h2'],
    )

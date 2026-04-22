"""Shared pytest fixtures for all test suites.

Biological hierarchy (CC population):
  strain   — unique CC genotype (haplotype probs live here; n_strains unique)
  sample   — individual mouse / replicate (multiple per strain; n_samples total)
  cell     — single nucleus (multiple per sample; n_cells total)

GenomeCache is indexed by strain. Z-matrix (n_samples × n_strains) expands
kinship from strain level to sample level for LMM and NB Stage 2.
NB Stage 1 operates at cell level: (n_genes, n_cells) with (n_cells,) sample_ids.
Pseudobulk aggregates to per-sample level (not per-strain).

Sample ID convention:
  CC001_M1, CC001_M2, CC001_F1, CC001_F2, CC002_M1, ...
  (N_MALES_PER_STR males then N_FEMALES_PER_STR females per strain)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.founders import FOUNDER_ORDER

# ── dimensions ───────────────────────────────────────────────────────────────
N_STRAINS          = 10   # unique CC genotypes: CC001–CC010
N_MALES_PER_STR    = 2    # male replicates per strain
N_FEMALES_PER_STR  = 2    # female replicates per strain
N_REPS_PER_STR     = N_MALES_PER_STR + N_FEMALES_PER_STR  # 4
N_SAMPLES          = N_STRAINS * N_REPS_PER_STR            # 40
N_CELLS_PER_SAMPLE = 8    # cells per sample → n_cells = 320
N_CELLS            = N_SAMPLES * N_CELLS_PER_SAMPLE
N_FOUNDERS         = 8
N_LOCI             = 15
N_GENES            = 4
CHROMS             = ['1', '2']


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_probs(n_loci, n_strains, n_founders, rng):
    """Return (n_loci, n_strains, n_founders) Dirichlet probs summing to 1 per row."""
    return rng.dirichlet(np.ones(n_founders), size=(n_loci, n_strains)).astype(np.float64)


def _make_sample_ids(strain_ids, n_males, n_females):
    """Build sample IDs: CC001_M1, CC001_M2, CC001_F1, CC001_F2, CC002_M1, ..."""
    ids = []
    for s in strain_ids:
        for i in range(1, n_males + 1):
            ids.append(f'{s}_M{i}')
        for i in range(1, n_females + 1):
            ids.append(f'{s}_F{i}')
    return np.array(ids)


def _make_sex_covariate(strain_ids, n_males, n_females):
    """(n_samples,) sex covariate: M=0, F=1."""
    vals = []
    for _ in strain_ids:
        vals.extend([0] * n_males)
        vals.extend([1] * n_females)
    return np.array(vals, dtype=np.float64)


# ── strain-level fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def strain_ids():
    """(n_strains,) unique CC genotype identifiers: CC001–CC010."""
    return np.array([f'CC{i:03d}' for i in range(1, N_STRAINS + 1)])


@pytest.fixture(scope='session')
def tiny_cache(strain_ids):
    """GenomeCache indexed by strain: 2 chromosomes, 15 loci, 10 strains, 8 founders."""
    rng  = np.random.default_rng(0)
    data = {}
    for c in CHROMS:
        data[c] = {
            'locus_ids': np.array([f'{c}_{i}' for i in range(N_LOCI)]),
            'positions': np.linspace(1.0, 100.0, N_LOCI),
            'probs':     _make_probs(N_LOCI, N_STRAINS, N_FOUNDERS, rng),
        }
    return GenomeCache(strain_ids, FOUNDER_ORDER, data)


@pytest.fixture(scope='session')
def tiny_kinship(tiny_cache):
    """LOCO kinship dict {chrom: (n_strains, n_strains)} from tiny_cache."""
    from mosaiqtl.genotype.kinship import calc_kinship
    return calc_kinship(tiny_cache, method='loco')


# ── sample-level fixtures ─────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def sample_ids(strain_ids):
    """(n_samples,) replicate IDs: CC001_M1, CC001_M2, CC001_F1, CC001_F2, ..."""
    return _make_sample_ids(strain_ids, N_MALES_PER_STR, N_FEMALES_PER_STR)


@pytest.fixture(scope='session')
def sample_strain_map(strain_ids):
    """(n_samples,) strain label for each sample — for Z-matrix construction."""
    return np.repeat(strain_ids, N_REPS_PER_STR)


@pytest.fixture(scope='session')
def sample_sex(strain_ids):
    """(n_samples,) sex covariate: M=0, F=1."""
    return _make_sex_covariate(strain_ids, N_MALES_PER_STR, N_FEMALES_PER_STR)


@pytest.fixture(scope='session')
def z_matrix(sample_strain_map, strain_ids):
    """(n_samples, n_strains) Z-matrix mapping samples to strains."""
    from mosaiqtl.utils import build_z_matrix
    return build_z_matrix(sample_strain_map, strain_ids)


@pytest.fixture(scope='session')
def kinship_samples(tiny_kinship, z_matrix):
    """LOCO kinship expanded to sample level: {chrom: (n_samples, n_samples)}."""
    Z, _ = z_matrix
    return {c: Z @ K @ Z.T for c, K in tiny_kinship.items()}


@pytest.fixture(scope='session')
def tiny_phenotype():
    """(n_samples,) continuous phenotype for LMM — one observation per replicate."""
    rng = np.random.default_rng(1)
    return rng.standard_normal(N_SAMPLES)


@pytest.fixture(scope='session')
def tiny_covariates(sample_sex):
    """(n_samples, 2) covariate matrix: [intercept, sex]."""
    return np.column_stack([np.ones(N_SAMPLES), sample_sex])


@pytest.fixture(scope='session')
def tiny_strain_phenotype():
    """(n_strains,) continuous phenotype — one observation per strain (LMM z_mat=None)."""
    rng = np.random.default_rng(1)
    return rng.standard_normal(N_STRAINS)


@pytest.fixture(scope='session')
def tiny_strain_covariates():
    """(n_strains, 1) covariate matrix: [intercept] — strain-level LMM."""
    return np.ones((N_STRAINS, 1))


# ── cell-level fixtures ───────────────────────────────────────────────────────

@pytest.fixture(scope='session')
def cell_sample_ids(sample_ids):
    """(n_cells,) sample label per cell — N_CELLS_PER_SAMPLE cells per replicate."""
    return np.repeat(sample_ids, N_CELLS_PER_SAMPLE)


@pytest.fixture(scope='session')
def counts_cells():
    """(n_genes, n_cells) raw int32 NB counts for NB Stage 1 tests.

    Drawn from NB(mu=10, phi=0.5) — realistic overdispersion without zero inflation.
    """
    rng = np.random.default_rng(2)
    mu  = 10.0
    phi = 0.5
    r   = 1.0 / phi
    p   = r / (r + mu)
    return rng.negative_binomial(r, p, size=(N_GENES, N_CELLS)).astype(np.int32)


@pytest.fixture(scope='session')
def counts_pb(counts_cells, cell_sample_ids):
    """(n_genes, n_samples) pseudobulk per-sample sums."""
    from mosaiqtl.qtl.nb import pseudobulk_sum
    pb, _, _ = pseudobulk_sum(counts_cells, cell_sample_ids)
    return pb


@pytest.fixture(scope='session')
def library_sizes_cells(counts_cells):
    """(n_cells,) total UMI per cell — NB offset."""
    return counts_cells.sum(axis=0).astype(np.float64)

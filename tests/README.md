# tests/

Unit, integration, and benchmark tests. Run with pytest from the repo root.

```bash
pytest tests/unit/ -x -q                          # fast gate (~7 s)
pytest tests/ --cov=src/mosaiqtl -q               # full suite + coverage
pytest tests/benchmarks/ --benchmark-autosave     # speed tracking
```

## Layout

| Path | Content |
|---|---|
| `conftest.py` | Shared fixtures (see below) |
| `fixtures/` | Static test data files |
| `unit/` | Per-module unit tests |
| `integration/` | End-to-end pipeline integration tests |
| `benchmarks/` | `pytest-benchmark` speed tests; baseline JSON in `benchmarks/baseline/` |
| `test_*.py` | Top-level unit tests (older; new tests go in `unit/`) |

## Fixture hierarchy

Biological hierarchy: **strain → sample/replicate → cell**.

| Fixture | Shape | Description |
|---|---|---|
| `strain_ids` | (10,) | CC001–CC010 |
| `tiny_cache` | GenomeCache | 2 chr, 15 loci, 10 strains, 8 founders |
| `tiny_kinship` | dict[str,(10,10)] | LOCO kinship at strain level |
| `sample_ids` | (40,) | CC001_M1 … CC010_F2 (4 replicates per strain) |
| `sample_strain_map` | (40,) | strain per sample |
| `z_matrix` | tuple[(40,10), mask] | Z-matrix + valid mask |
| `kinship_samples` | dict[str,(40,40)] | LOCO kinship expanded via Z K Zᵀ |
| `tiny_phenotype` | (40,) | sample-level continuous phenotype |
| `tiny_covariates` | (40,2) | [intercept, sex] |
| `counts_cells` | (4,320) int32 | NB(μ=10,φ=0.5) counts |

LMM tests use `tiny_strain_*` fixtures (strain level, 10×10 kinship).

## Rules

- Never remove existing tests — always append new ones.
- Coverage threshold: ≥ 90% (CI gate via `--cov-fail-under`).
- All tests deterministic given a fixed seed.

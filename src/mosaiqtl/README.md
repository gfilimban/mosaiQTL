# src/mosaiqtl/

Python package root. Install with `pip install -e '.[dev]'`.

## Modules

| Module | Purpose |
|---|---|
| `pipeline.py` | End-to-end scan orchestration: eGene discovery, BH-FDR, output CSVs |
| `utils.py` | RINT, safe cell-type naming, shared helpers |
| `genotype/` | Genome cache I/O, kinship, HMM, founder constants |
| `io/` | AnnData I/O, covariate building, results serialisation |
| `models/` | Data records: `ScanResult`, `ProvenanceRecord`, `ScanConfig` |
| `qtl/` | Scan engines: LMM, NB-GLMM, permutation, effects, mediation |
| `viz/` | Publication-quality plots: LOD profiles, effects, haplotypes |

## Entry point

One registered entry point (see `pyproject.toml`):

```
build-cache  →  mosaiqtl.genotype.genomecache:build
```

Full CLI subcommands (`fit-null`, `scan`, `permute`, `effects`, `report`) are planned
but not yet implemented — see `GOALS.md` DESIGN-02/DESIGN-08.

## Key design rules

- All public column schemas live in `pipeline.py` as `LOD_COLS`, `PEAK_LOD_COLS`, etc.
- Domain constants (founders, colors) imported from `genotype/founders.py` — never redefined inline.
- `adata.raw.X` for counts; `adata.X` is normalised and must not be used for QTL input.
- All public signatures type-annotated (PEP 484); Google-style docstrings on public API.

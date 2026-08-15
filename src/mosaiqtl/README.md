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

`cli.py` (Click) — not listed above because it is auto-generated at install time via
`pyproject.toml` entry point `mosaiqtl = mosaiqtl.cli:cli`.

## Key design rules

- All public column schemas live in `pipeline.py` as `LOD_COLS`, `PEAK_LOD_COLS`, etc.
- Domain constants (founders, colors) imported from `genotype/founders.py` — never redefined inline.
- `adata.raw.X` for counts; `adata.X` is normalised and must not be used for QTL input.
- All public signatures type-annotated (PEP 484); Google-style docstrings on public API.

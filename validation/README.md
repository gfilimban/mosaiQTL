# validation/

End-to-end validation scripts. Each phase mirrors a specific R reference and must pass
before the next phase is expanded.

## Validation phases

| Script | Phase | Data | Description |
|---|---|---|---|
| `validate_sim.py` | 1 | `sparcc_cache` + sim phenos | Minimal miQTL workflow mirror — ground truth |
| `validate_phase2.py` | 2 | `sparcc_cache` + `sim.CC.data` | Full function port, real LOCO kinship, S1–S11 |
| `validate_phase3.py` | 3 | sim data | Speed and memory benchmarking vs miQTL / R/qtl2 |
| `validate_port.py` | 2 gastroc | real gastroc data | Real-data port, S1–S11 |
| `validate_phase4.py` | 4 | simulated NB data | NB vs LMM comparison |
| `validate_gastroc.py` | gastroc | gastroc snRNA-seq | Full gastroc benchmark; R comparison optional |
| `validate_tissue.py` | tissue | any tissue h5ad | Tissue-agnostic wrapper over gastroc logic |

Run Phase 1 first. Phase 2 and beyond require Phase 1 green.

## Subdirectories

| Directory | Content |
|---|---|
| `r_reference/` | R scripts that generate ground truth CSVs (do not modify — R is ground truth) |
| `tools/` | Shared utility scripts: convert, plot, simulation helpers |
| `hpc/` | SLURM array scripts for running scans on HPC |
| `input/` | Static input files (gene info, TSS tables) |
| `py_ref/` | Shared Python outputs (genomecache, kinship) reused across scripts |
| `sparcc_cache/` | Simulated CC genome cache (from miQTL `sim.CC.data`) |
| `*_data/` | Per-tissue intermediate outputs (gitignored) |
| `phase*/` | Per-phase intermediate outputs (gitignored) |

## Key conventions

- R scripts call miQTL exactly as miQTL defines — no patches, no defensive branching.
- Python validation scripts are thin callers; plot logic lives in `src/mosaiqtl/viz/`.
- All section keys are descriptive strings (not `S*` codes) in `validate_gastroc.py`.
- `conda run -n bioinfo` prefix required for all Python commands.

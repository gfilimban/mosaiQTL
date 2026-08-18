# workflow/

Nextflow DSL2 pipeline for genome-wide LMM QTL scans on HPC (SLURM) or locally.

## Quick start

```bash
# Stub run — validate DAG, no compute
nextflow run workflow/main.nf -stub-run -params-file config/default_params.yaml

# Single cell type (gastroc)
nextflow run workflow/main.nf -profile slurm \
    -params-file config/gastroc_params.yaml -resume

# Multiple cell types (ctxhipp celltypes_dir)
nextflow run workflow/main.nf -profile slurm \
    -params-file config/ctxhipp_params.yaml -resume
```

## Layout

| Path | Content |
|---|---|
| `main.nf` | Workflow entry: channel construction, process wiring |
| `scripts/` | Python helper scripts called inside process `script:` blocks |

## Process DAG

```
KINSHIP_LOCO (once) ─┐
                      ├─► LMM_SCAN (per celltype × chr) ─► MERGE_RESULTS (per celltype)
CT channel ──────────┘
```

See `modules/` for process definitions and resource labels.
See `config/` for parameter files and profile selection.

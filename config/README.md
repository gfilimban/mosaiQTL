# config/

Nextflow parameter files. Pass to any workflow run with `-params-file`.

| File | Purpose |
|---|---|
| `default_params.yaml` | All parameters with defaults and inline docs — start here |
| `gastroc_params.yaml` | Gastroc single-cell-type scan (HPC paths, single-CT mode) |
| `ctxhipp_params.yaml` | Ctxhipp multi-cell-type scan (HPC paths, `celltypes_dir` mode) |

## Cell-type input modes

**Multi-CT** (`celltypes_dir`): each subdirectory is a cell type containing `pseudobulk.h5`.
Kinship is built once and reused across all cell types. Output: `{output_dir}/{celltype}/`.

```yaml
celltypes_dir: /path/to/{tissue}_data/celltypes
celltypes:     'all'   # or 'glutamatergic_neuron,oligodendrocyte'
gene_info:     /path/to/exp_gene_info_{tissue}.csv
```

**Single-CT** (`phenotypes`): one phenotype file, one cell type label.

```yaml
phenotypes: /path/to/pseudobulk.h5
celltype:   'gastroc'
gene_info:  /path/to/gene_info.csv
```

## Adding a new tissue

Copy the closest existing params file, update `genomecache`, `celltypes_dir` (or `phenotypes`),
`gene_info`, and `output_dir`. `kinship_loco: null` on first run; set it to the published path
for subsequent `-resume` runs.

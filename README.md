# mosaiQTL

**Haplotype-based QTL mapping for multiparent populations.**  
Maps molecular phenotypes to 8 founder haplotypes using a Gaussian LMM or NB-GLMM with LOCO
kinship correction. Designed for CC/DO mice (mm39); extensible to other MPPs.

> **Current status:** The **Gaussian LMM** (`--model lmm`) is the validated, production-ready
> model — Phase 1–3 gates pass, full gastroc scan complete, Sanger fine-mapping wired.
> The **NB-GLMM** (`--model nb`) is the intended long-term primary model for raw pseudobulk
> counts but has not yet passed end-to-end biological validation (Phase 4). Use `--model lmm`
> for any real analysis until Phase 4 is complete.

## Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Quickstart](#quickstart)
- [Before You Start](#before-you-start)
- [Usage](#usage)
- [Inputs](#inputs)
- [Outputs](#outputs)
- [Models](#models)
- [Documentation](#documentation)
- [License](#license)

---

## Introduction

mosaiQTL performs genome-wide haplotype-based QTL scans for single-cell modalities. It resolves
associations to 8 founder haplotypes using a score test under a NB-GLMM or Gaussian LMM with
LOCO (leave-one-chromosome-out) kinship correction.

**Supported modalities:** snRNA-seq eQTL · ATAC-seq caQTL (planned) · splicing QTL (planned)

| | mosaiQTL | miQTL (R) | tensorQTL |
|---|---|---|---|
| **Primary model** | NB-GLMM (counts) | Gaussian LMM | Gaussian linear |
| **Haplotype scan** | ✓ (8 founders) | ✓ (8 founders) | ✗ |
| **LOCO kinship** | ✓ mandatory | ✓ | ✗ |
| **Raw count input** | ✓ | ✗ | ✗ |
| **Peak RAM (43k genes, 20 chr)** | 897 MB | — | — |

---

## Installation

```bash
conda create -n mosaiqtl python=3.13
conda activate mosaiqtl
pip install -e '.[dev]'
```

Core deps (scan engine): `numpy scipy pandas h5py anndata`  
Required for CLI and plots: `pydantic click seaborn matplotlib`  
Optional: `rpy2` (convert existing R-format genome caches to HDF5)

---

## Quickstart

The fastest path to a result is a provided expression HDF5 + genome cache — no external
data download required if a collaborator shares these files with you.

### Whole-tissue or single-cell-type scan

Given `counts_lognorm.h5` (pseudobulk expression, samples × genes), `exp_gene_info.csv`,
`sample_meta.csv`, and `cache.h5`:

```bash
conda activate mosaiqtl

# S1–S6: genome cache → haplotypes → kinship → LMM scan → permutation → eGene calling
python validation/validate_tissue.py \
    --tissue <label> \
    --data   <directory-containing-the-four-files> \
    --scan

# S7: multiple-imputation scan on the called eGenes (runs on top of a completed S1–S6)
python validation/validate_tissue.py \
    --tissue <label> \
    --data   <directory> \
    --mi
```

Outputs land in `<data>/<label>/core/`:

| File | Description |
|---|---|
| `peak_lod_<label>.csv` | Per-gene peak LOD |
| `allele_effects_<label>.csv` | Founder haplotype effects at each gene's peak locus |
| `threshold_<label>.csv` | GEV-fitted genome-wide LOD threshold |
| `eqtl_summary.csv` | eGene table — all genes exceeding the LOD threshold |

### Cell-type scan

```bash
python validation/validate_tissue.py \
    --tissue ctxhipp \
    --data   validation/ctxhipp_data \
    --celltype 'glutamatergic neuron' \
    --scan
```

All sections are checkpointed — interrupted runs resume from the last completed section.
Pass `--force` to ignore checkpoints and rerun from scratch.

---

## Before You Start

mosaiQTL requires four inputs that you prepare once and reuse across all scans.

**Genome cache (`--cache cache.h5`)**  
An HDF5 file containing haplotype probabilities (samples × 8 CC founders × loci) and a loci
map covering all GigaMUGA marker positions. mosaiQTL loads one chromosome at a time for
constant-memory scanning.

Build a cache from public GigaMUGA reference data (CC mice, mm39):

```bash
mosaiqtl build-cache --population cc \
    --data-dir /path/to/GM/ \
    --cc-cols cc_cols.txt \
    --output cache.h5
```

All required input files are freely available from the
[JAX Diversity Outbred Reference Data](https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data)
page. See `docs/DATA_SOURCES.md` for the full provenance chain and file descriptions.

Once built, load in Python with:

```python
from mosaiqtl.genotype.genomecache import GenomeCache
cache = GenomeCache.from_hdf5('cache.h5')
```

**Pseudobulk counts (`--counts data.h5ad`)**  
A genes × samples count matrix aggregated from your single-cell data. mosaiQTL does not perform
aggregation — generate pseudobulk counts upstream using Scanpy or a similar tool, then pass the
result as an AnnData `.h5ad` file. Counts must be in `.raw.X` (raw, non-normalised integers).

**Gene info (`--gene-info gene_info.csv`)**  
A CSV with columns: `gene_id` (Ensembl ID), `gene_symbol`, `chr`, `start`, `end`. Used to define
the scan window around each gene's TSS and to annotate output files.

**Sample metadata (`--metadata meta.csv`)**  
A CSV with one row per sample. Required column: `sample_id`. Strongly recommended: `sex`
(0 = male, 1 = female) and `batch` (sequencing run identifier). Both are included as covariates
automatically when present.

---

## Usage

### CLI

**LMM** (Gaussian LMM — current working model):

LMM requires four stages. The null heritability (h²) is estimated inside `scan` — no separate
null fit is needed.

```bash
# 1. Genome-wide scan → per-gene peak LOD
mosaiqtl scan --model lmm \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --null-fit null_fit.csv --output peak_lod.csv

# 2. Permutation thresholds → GEV (generalized extreme value) fit on max-LOD distribution
mosaiqtl permute \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --null-fit null_fit.csv --output perm/ --n-perms 1000

# 3. Haplotype effect estimates at each gene's peak locus
mosaiqtl estimate \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --scan-result peak_lod.csv --output allele_effects.csv

# 4. eGene summary table + genome-wide Manhattan
mosaiqtl report \
    --scan-result peak_lod.csv --perm-dir perm/ --output-dir results/ \
    --allele-effects allele_effects.csv --pval-perm pval_perm.csv
```

`--allele-effects`/`--pval-perm` are optional: omit either to still get `eqtl_peak.csv`, just
with the corresponding columns (founder betas / Storey significance) left blank.

> **`--model` defaults to `'nb'`** (NB-GLMM). Pass `--model lmm` for the validated Gaussian path.

**NB-GLMM** (primary model for raw pseudobulk counts; adds a null fit stage before scanning):

```bash
# 0. NB null fit — estimates overdispersion φ and variance σ² per gene
mosaiqtl fit-null \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --output null_fit.csv

# 1–4. Same as LMM — pass the null fit to scan and permute
mosaiqtl scan    --cache cache.h5 --counts data.h5ad --gene-info gene_info.csv --metadata meta.csv --null-fit null_fit.csv --output peak_lod.csv
mosaiqtl permute --cache cache.h5 --counts data.h5ad --gene-info gene_info.csv --metadata meta.csv --null-fit null_fit.csv --output perm/ --n-perms 1000
mosaiqtl estimate --cache cache.h5 --counts data.h5ad --gene-info gene_info.csv --metadata meta.csv --scan-result peak_lod.csv --output allele_effects.csv
mosaiqtl report   --scan-result peak_lod.csv --perm-dir perm/ --output-dir results/
```

Run `mosaiqtl <command> --help` for the full flag list per command.

### Python API

The Python API mirrors the full `validate_tissue.py` pipeline — useful for custom covariates,
single-chromosome experiments, or building new downstream analyses on top of mosaiQTL internals.

```python
import numpy as np
import pandas as pd
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.io.covariates import build_covariate_matrix
from mosaiqtl.qtl.scan import scan_multi_gene
from mosaiqtl.qtl.permutation import run_threshold_scans, get_gev_thresholds, calibrate_gene_pvalue
from mosaiqtl.qtl.effects import estimate_founder_effects
from mosaiqtl.pipeline import call_egenes, calibrate_egenes_storey, finemap_egenes

# ── 1. Load cache + build LOCO kinship (once for the whole run) ────────────────
cache   = GenomeCache.from_hdf5('data/cache.h5')
kinship_decomp = decomp_kinship(calc_kinship(cache, method='loco'))

# ── 2. Covariates ──────────────────────────────────────────────────────────────
meta = pd.read_csv('gene_info.csv')            # gene_id, gene_symbol, chr, start, end
sample_meta = pd.read_csv('metadata.csv')      # sample_id required; sex, batch auto-included
cov_df = build_covariate_matrix(sample_meta, covariates=['sex', 'batch'])
X = cov_df.values                              # (n_samples, p) float64
sample_ids = list(cov_df.index)                # genotype-level IDs matching cache

# ── 3. Load expression + scan all chromosomes (streams one chr at a time) ─────
import anndata
adata = anndata.read_h5ad('counts.h5ad', backed='r')
Y = np.array(adata.raw.X.toarray(), dtype=np.float64)  # (n_genes, n_samples)
gene_ids = list(adata.var_names)

scan_results = scan_multi_gene(
    phenotype_matrix=Y,
    gene_ids=gene_ids,
    genomecache=cache,
    gene_chr_map=dict(zip(meta['gene_id'], meta['chr'].astype(str))),
    covariates=X,
    kinship_decomp=kinship_decomp,
    sample_ids=sample_ids,
    use_rint=True,                             # rank-normal transform; matches LMM reference
    scan_window='cis',                         # restrict each gene to its own chromosome
    gene_pos_mb_map=dict(zip(meta['gene_id'], meta['start'] / 1e6)),
    output_dir=None,                           # pass a Path to checkpoint per-chr CSVs
    model='additive',
)
# scan_results: dict[gene_id → ScanResult]

# ── 4. Permutation thresholds + GEV fit ───────────────────────────────────────
# run_threshold_scans is lower-level; use the CLI `mosaiqtl permute` for production.
# For exploration: get_gev_thresholds(max_lods_chr) returns (threshold, gev_params | None).
max_lods = pd.read_csv('perm/perm_max_lods.csv')['max_lod'].values
threshold, _ = get_gev_thresholds(max_lods)   # returns (None, None) on failure → use quantile

# ── 5. Stage-1 coarse gate ─────────────────────────────────────────────────────
from mosaiqtl.io.results_io import scan_result_to_df
peak_rows = [scan_result_to_df(res, gid) for gid, res in scan_results.items()]
peak_lod_df = pd.concat(peak_rows, ignore_index=True)

egenes_stage1 = call_egenes(peak_lod_df, threshold=threshold)

# ── 6. Stage-2: per-gene Beta calibration + Storey q-value ────────────────────
# calibrate_gene_pvalue draws a per-gene parametric null at the gene's own h²;
# run it only on genes that cleared Stage-1 (expensive — ~100 perms per gene).
pval_perm = {
    gene_id: calibrate_gene_pvalue(
        scan_results[gene_id], Y[gene_ids.index(gene_id)],
        cache, kinship_decomp, covariates=X,
    )
    for gene_id in egenes_stage1['gene_id']
}
egenes = calibrate_egenes_storey(egenes_stage1, pval_perm, fdr_threshold=0.05)

# ── 7. Founder effects at each eGene's peak ───────────────────────────────────
# estimate_founder_effects is also wrapped by `mosaiqtl estimate` in the CLI.
effects_df = estimate_founder_effects(
    egenes_df=egenes,
    phenotype_matrix=Y,
    gene_ids=gene_ids,
    genomecache=cache,
    covariates=X,
    kinship_decomp=kinship_decomp,
)

# ── 8. Sanger fine-mapping (requires alleles HDF5 built from MGP VCF) ─────────
# Build once: build_founder_allele_mgp(vcf_path, cache, 'data/sanger_alleles.h5')
finemap_egenes(
    egene_df=egenes,
    phenotype_matrix=Y,
    gene_ids=gene_ids,
    genomecache=cache,
    alleles_h5='data/sanger_alleles.h5',
    out_dir='results/',
    covariates=X,
    kinship_decomp=kinship_decomp,
    tag='gastroc_Lw1',
)
# Writes results/eqtl_finemapping_gastroc_Lw1.csv + eqtl_finemapping_peak_gastroc_Lw1.csv
```

For production runs with many genes, prefer `mosaiqtl scan` / `mosaiqtl permute` (they
checkpoint per-chromosome and resume after failures). Use the API for exploration,
custom covariate matrices, or single-chromosome deep-dives.

### HPC (Nextflow)

```bash
# Validate DAG — no compute
nextflow run workflow/main.nf -stub-run -params-file config/default_params.yaml

# Single cell type
nextflow run workflow/main.nf -profile slurm \
    -params-file config/gastroc_params.yaml -resume

# Multiple cell types (celltypes_dir × all chromosomes)
nextflow run workflow/main.nf -profile slurm \
    -params-file config/ctxhipp_params.yaml -resume
```

See [`workflow/README.md`](workflow/README.md) for process DAG, resource labels, and SLURM
configuration including Nextflow parameters (`scan_window`, `chroms`, `n_perms`, etc.).

---

## Inputs

### File Inputs

| Input | CLI flag | Format | Description |
|---|---|---|---|
| Genome cache | `--cache` | HDF5 | Haplotype probabilities + loci map; see [Before You Start](#before-you-start) |
| Gene info | `--gene-info` | CSV | `gene_id`, `gene_symbol`, `chr`, `start`, `end` |
| Pseudobulk counts | `--counts` | AnnData `.h5ad` | Genes × samples raw counts in `.raw.X` |
| Sample metadata | `--metadata` | CSV | `sample_id` required; `sex` + `batch` auto-included as covariates |
| NB null fit | `--null-fit` | CSV | Output of `mosaiqtl fit-null`; required for NB-GLMM scan |
| LOCO kinship | `--kinship` | HDF5 | Optional — built from `--cache` if not provided |

### Cell-type Inputs (Nextflow only)

Set one mode. `celltypes_dir` takes precedence if both are set.

**Multi-CT** — one scan job per cell type × chromosome:

| Nextflow param | Description |
|---|---|
| `celltypes_dir` | Directory where each subdirectory is named after a cell type and contains `pseudobulk.h5`; per-CT `gene_info.csv` used if present, otherwise falls back to global `gene_info` |
| `celltypes` | Cell types to include: `'all'` or comma-separated (e.g. `'glutamatergic_neuron,oligodendrocyte'`) |

**Single-CT** — used when `celltypes_dir` is not set:

| Nextflow param | Description |
|---|---|
| `phenotypes` | Pseudobulk HDF5 for a single cell type |
| `celltype` | Cell type label for output paths (default: `'default'`) |

### Nextflow Scan Parameters

Configured in `config/*.yaml` or overridden on the command line with `--param value`.

| Parameter | Default | Description |
|---|---|---|
| `scan_window` | `'cis'` | `'cis'` (±`window_mb` around TSS), `'genome'`, or `'local'` |
| `window_mb` | `1.0` | Half-window size in Mb; applies to `'cis'` and `'local'` modes |
| `chroms` | `'all'` | `'all'` (chr1–19 + X) or comma-separated (e.g. `'1,2,8'`) |
| `n_perms` | `1000` | Permutations per gene for GEV threshold estimation |
| `output_dir` | `'results'` | Root output directory; per-CT results at `{output_dir}/{celltype}/` |
| `seed` | `42` | Random seed for permutation reproducibility |

---

## Outputs

### Intermediate files

| File | Description |
|---|---|
| `null_fit.csv` | Per-gene NB-GLMM null fit — φ (overdispersion), σ² (variance), convergence flag (NB only) |
| `allele_effects.csv` | Per-founder effect estimates + SE at each gene's peak locus |
| `perm/perm_max_lods.csv` | Max-LOD distribution across permutations (one row per permutation) |

### Final outputs

Named `{tissue}_{celltype}_{suffix}`.

| File | Status | Description |
|---|---|---|
| `eqtl_peak.csv` | ✅ | Peak locus per eGene — haplotype effects, significance, run metadata. Biallelic-contrast columns (`variant`, `ref_founders`, `af`, `maf`, `mac`, …) are written but left blank — see 🟡 below. |
| `eqtl_stats.csv` | ✅ | Run-level summary — tissue, cell type, LOD threshold, GEV parameters, gene counts |
| `genome_summary.png` | ✅ | Genome-wide Manhattan and QC figures |
| `eqtl_summary.csv` | ✅ | eGene summary table (pre-IGVF-schema; kept alongside the files above) |
| `eqtl_scan.csv` | ✅ | Haplotype-level scan — one row per gene × GigaMUGA locus; `peak_locus` flag marks each gene's peak. Enable with `mosaiqtl report --eqtl-scan`. |
| `eqtl_finemapping.csv` | ✅ | Sanger-resolution fine-mapping — PIP, ABF credible set membership, CI bounds per variant. Requires `data/sanger_alleles.h5` (built once from MGP REL2021 VCF via `build_founder_allele_mgp`). |
| `eqtl_finemapping_peak.csv` | ✅ | Lead variant per eGene — highest-PIP Sanger SNP with haplotype effects and significance |

---

## Models

| Model | Function | When to use |
|---|---|---|
| **NB-GLMM + LOCO** | `scan_nb_gene()` | Raw pseudobulk counts — primary for snRNA-seq eQTL |
| **Gaussian LMM** | `scan_h2lmm()` | Normalized expression; exact miQTL/qtl2 port |
| **LMM + MI** | `scan_h2lmm_mi()` | Averages over haplotype-assignment uncertainty |
| **Dispersion QTL** | `scan_vqtl()` | NB dispersion as phenotype (transcriptional noise) |
| **Mediation** | `mediation_scan_qr()` | Candidate mediator LOD via QR Woodbury update |
| **Epistasis** | `scan_epistasis()` | Pairwise founder × founder interaction scan |

**Choosing NB-GLMM vs LMM:**

| | NB-GLMM (`--model nb`) | LMM (`--model lmm`) |
|---|---|---|
| **Input** | Raw integer pseudobulk counts | Log-normalised / continuous phenotype |
| **Overdispersion** | Modelled explicitly (NB φ per gene) | Not modelled — inflated false positives for low-expr genes |
| **Stage 0** | Required — `fit-null` estimates φ and σ² per gene | Not needed — h² estimated inside `scan` |
| **Speed** | Slower (PQL null per gene × chr) | Faster (h² once per chr, FWL vectorised) |
| **Use when** | snRNA-seq / ATAC-seq / count data | Normalized bulk RNA-seq or fast exploration |

---

## Examples (in preparation)

End-to-end worked examples showing expected inputs, commands, outputs, and figures for each
pipeline stage. Each example is a complete record of a successful run.

| Example | Description |
|---|---|
| `examples/gastroc_eqtl/` | Gastrocnemius snRNA-seq eQTL — single cell type, LMM, chr8 |
| `examples/ctxhipp_eqtl/` | Cortex/hippocampus multi-cell-type eQTL — Nextflow, all chromosomes |
| `examples/sim_cc/` | Simulated CC data — LMM scan with known planted QTL signal |

---

## Documentation

| Resource | Content |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, model derivations, inference choices |
| [`docs/math/`](docs/math/) | Step-by-step derivations for PQL, LOCO kinship, GEV thresholds |
| [`workflow/README.md`](workflow/README.md) | Nextflow DAG, resource labels, SLURM configuration |
| [`config/`](config/) | Annotated parameter files for gastroc and ctxhipp reference runs |

---

## License

GPL-3.0

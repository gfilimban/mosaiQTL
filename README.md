# mosaiQTL

**Haplotype-based QTL mapping for multiparent populations.**  
Maps molecular phenotypes to 8 founder haplotypes using a Gaussian LMM or NB-GLMM with LOCO
kinship correction. Designed for CC/DO mice (mm39); extensible to other MPPs.

> **Current status (v0.6.7):** The **Gaussian LMM** (`--model lmm`) is the validated,
> production-ready model — Phase 1–3 gates pass, full gastroc scan complete, Sanger
> fine-mapping with IGVF-schema outputs wired end-to-end. The **NB-GLMM** (`--model nb`) is
> the intended long-term primary model for raw pseudobulk counts but has not yet passed
> end-to-end biological validation (Phase 4). Use `--model lmm` for any real analysis until
> Phase 4 is complete.

---

## Introduction

mosaiQTL performs genome-wide haplotype-based QTL scans for single-cell modalities. It resolves
associations to 8 founder haplotypes using a score test under a NB-GLMM or Gaussian LMM with
LOCO (leave-one-chromosome-out) kinship correction. The pipeline streams one chromosome at a
time (peak RAM < 1 GB for 43 k genes across 20 chromosomes) and runs either as a local CLI or
a Nextflow workflow for HPC clusters.

**Supported modalities:** snRNA-seq eQTL · ATAC-seq caQTL (planned) · splicing QTL (planned)

| | mosaiQTL | miQTL (R) | tensorQTL |
|---|---|---|---|
| **Primary model** | NB-GLMM (counts) | Gaussian LMM | Gaussian linear |
| **Haplotype scan** | ✓ (8 founders) | ✓ (8 founders) | ✗ |
| **LOCO kinship** | ✓ mandatory | ✓ | ✗ |
| **Raw count input** | ✓ | ✗ | ✗ |
| **Sanger fine-mapping** | ✓ ABF credible sets | ✗ | ✗ |
| **Peak RAM (43k genes, 20 chr)** | 897 MB | — | — |

---

## Installation

```bash
conda create -n mosaiqtl python=3.13
conda activate mosaiqtl
pip install -e '.[dev]'
```

Core deps (scan engine): `numpy scipy pandas h5py anndata natsort`  
Required for CLI and plots: `pydantic click seaborn matplotlib`  
Optional: `rpy2` (convert existing R-format genome caches to HDF5), `pysam` (build Sanger
alleles HDF5 from MGP VCF)

---

## Quick start

```bash
# 1. Build a genome cache from GigaMUGA probs (JAX DO reference data)
mosaiqtl build-cache --gigamuga GM_ccf1fo_mm39.csv --probs cc_probs.rds --out cache.h5

# 2. Genome-wide LMM scan → per-gene peak LOD
mosaiqtl scan --model lmm \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --output peak_lod.csv

# 3. Permutation thresholds
mosaiqtl permute \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --output perm/ --n-perms 1000

# 4. Allele effects + eGene calling + IGVF outputs
mosaiqtl report \
    --scan-result peak_lod.csv --perm-dir perm/ --output-dir results/ \
    --allele-effects allele_effects.csv --pval-perm pval_perm.json

# 5. Sanger fine-mapping (requires sanger_alleles.h5)
mosaiqtl finemap \
    --egenes results/eqtl_summary.csv \
    --cache cache.h5 --counts data.h5ad \
    --metadata meta.csv \
    --sanger-alleles sanger_alleles.h5 \
    --output results/finemap/
```

For parallel per-chromosome HPC runs see **Nextflow pipeline** below.

---

## CLI reference

### Subcommands

| Subcommand | Description |
|---|---|
| `build-cache` | GigaMUGA CSV → genomecache HDF5 |
| `scan` | Per-chromosome LMM/NB scan → `peak_lod.csv` |
| `permute` | Permutation null distribution → GEV fit |
| `call-egenes` | Storey FDR eGene discovery |
| `finemap` | Sanger SNP fine-mapping → credible-set CSVs |
| `report` | Summary tables + QC plots |

Run `mosaiqtl <subcommand> --help` for the full flag list.

### Model flag

`--model lmm` (Gaussian LMM, validated) or `--model nb` (NB-GLMM, Phase 4 pending).  
The default is `nb`; always pass `--model lmm` for real analyses until Phase 4 completes.

### Scan window

`--scan-type L` (local ±`window_mb` around TSS, recommended for cis-eQTL),
`C` (whole chromosome), or `G` (genome-wide). The permutation null must use the same
`--scan-type` and `--window-mb` as the scan.

---

## Nextflow pipeline (HPC)

The Nextflow DSL2 workflow parallelises per-chromosome scanning across SLURM array jobs with
a shared LOCO kinship object.

```bash
# dry-run (validates DAG and stubs all processes)
nextflow run workflow/main.nf -profile local -stub-run \
    -params-file config/default_params.yaml

# production HPC run (SLURM)
nextflow run workflow/main.nf -profile slurm \
    -params-file config/default_params.yaml -resume
```

Key params in `config/default_params.yaml`:

| Param | Description |
|---|---|
| `celltypes_dir` | Directory of per-cell-type counts HDF5 files |
| `sanger_alleles` | Path to `sanger_alleles.h5` (optional — fine-mapping skipped if absent) |
| `scan_type` | `L`, `C`, or `G` |
| `window_mb` | Half-window in Mb for `scan_type=L` |
| `use_rint` | `1` (default) for rank-normalised expression; `0` for raw Gaussian phenotypes |
| `n_perms` | Number of permutations for GEV null fit |

---

## Python API

Use the API for custom workflows, single-chromosome exploration, or downstream analyses.
For full runs, prefer the CLI — it checkpoints per-chromosome and resumes after failures.

```python
import numpy as np
import pandas as pd
import anndata
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.utils import build_z_matrix, reduce_to_strain_means
from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
from mosaiqtl.io.results_io import scan_result_to_df
from mosaiqtl.qtl.scan import scan_multi_gene_lmm
from mosaiqtl.qtl.thresholds import get_gev_thresholds
from mosaiqtl.qtl.fdr import calibrate_gene_pvalue
from mosaiqtl.qtl.effects import get_allele_effects
from mosaiqtl.pipeline import (
    call_egenes, calibrate_egenes_storey, finalize_egene_summary,
    finemap_egenes, prepare_egenes_for_finemap,
)

# 1. Cache + LOCO kinship (once per run)
cache          = GenomeCache.from_hdf5('ref/cache.h5')
kinship_decomp = decomp_kinship(calc_kinship(cache, method='loco'))

# 2. Metadata, covariates, expression
meta       = pd.read_csv('metadata.csv')   # required: sample_id; opt: sex, batch
geno_map   = build_sample_geno_map(meta)
sample_ids = list(meta['sample_id'])
geno_ids   = [geno_map.get(s, s) for s in sample_ids]
cov_arr    = build_cov_array(meta, sample_ids, ['sex', 'batch'])  # (n_samples, p)

adata    = anndata.read_h5ad('counts.h5ad', backed='r')
Y        = np.array(adata.raw.X.toarray(), dtype=np.float64)  # (n_genes, n_samples)
gene_ids = list(adata.var_names)
gene_info = pd.read_csv('gene_info.csv')   # gene_id, gene_symbol, chr, start, end

# 3. Genome-wide scan (streams one chromosome at a time)
peak_rows = scan_multi_gene_lmm(
    phenotype_matrix=Y,
    gene_ids=gene_ids,
    genomecache=cache,
    gene_chr_map=dict(zip(gene_info['gene_id'], gene_info['chr'].astype(str))),
    covariates=cov_arr,
    kinship_decomp=kinship_decomp,
    sample_ids=geno_ids,
    use_rint=True,
    scan_type='L',
    window_mb=1.0,
    gene_pos_mb_map=dict(zip(gene_info['gene_id'], gene_info['start'] / 1e6)),
)
peak_lod_df = pd.DataFrame(peak_rows)

# 4. GEV threshold from permutation max-LOD null
max_lods     = pd.read_csv('perm/perm_max_lods.csv')['max_lod'].values
threshold, _ = get_gev_thresholds(max_lods)   # (None, None) on failure

# 5. Stage-1 coarse gate
egenes_stage1 = call_egenes(peak_lod_df, threshold=threshold)

# 6. Stage-2 FDR: per-gene Beta calibration + Storey q-value
# calibrate_gene_pvalue operates at strain level — reduce via Z-matrix
Z, valid = build_z_matrix(geno_ids, list(cache.sample_ids))
pval_perm: dict[str, float] = {}
for _, row in egenes_stage1.iterrows():
    gid = str(row['gene_id'])
    idx = gene_ids.index(gid)
    y_strain = reduce_to_strain_means(Y[idx], Z, valid)
    p, *_    = calibrate_gene_pvalue(
        y_gene=y_strain, genomecache=cache, chrom=str(row['chr']),
        peak_lod=float(row['peak_lod']), covariates=None,
        kinship_decomp=kinship_decomp, n_perms=200,
    )
    pval_perm[gid] = p

egenes = calibrate_egenes_storey(egenes_stage1, pval_perm, fdr_threshold=0.05)

# 7. Sanger fine-mapping → IGVF-schema outputs
egenes_fm = prepare_egenes_for_finemap(egenes)
fm_df, fm_peak_df = finemap_egenes(
    egene_df=egenes_fm, phenotype_matrix=Y, gene_ids=gene_ids,
    genomecache=cache, alleles_h5='ref/sanger_alleles.h5',
    out_dir='results/', covariates=cov_arr, kinship_decomp=kinship_decomp,
    sample_ids=geno_ids,
)
```

---

## Inputs

| Input | CLI flag | Format | Description |
|---|---|---|---|
| Genome cache | `--cache` | HDF5 | Haplotype probabilities + loci map. Build with `mosaiqtl build-cache`; source GigaMUGA probs from [JAX DO Reference Data](https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data). |
| Gene info | `--gene-info` | CSV | `gene_id`, `gene_symbol`, `chr`, `start`, `end` |
| Pseudobulk counts | `--counts` | AnnData `.h5ad` | Genes × samples; raw integer counts in `.raw.X` for NB-GLMM, log-normalised for LMM |
| Sample metadata | `--metadata` | CSV | `sample_id` required; `sex` (0=M, 1=F) and `batch` auto-included as covariates when present |
| Sanger alleles | `--sanger-alleles` | HDF5 | Built once from MGP REL2021 VCF via `build_founder_allele_mgp()`; optional (fine-mapping skipped if absent) |
| LOCO kinship | `--kinship` | HDF5 | Optional — built from `--cache` if not provided |

---

## Outputs

All output files are named `{tissue}_{celltype}_{scan_tag}_{filename}`.

### Scan outputs

| File | Status | Description |
|---|---|---|
| `peak_lod.csv` | ✅ | Per-gene peak LOD, peak locus, cis distance, null model params |
| `allele_effects.csv` | ✅ | Per-founder effect estimates at each gene's peak locus |
| `eqtl_summary.csv` | ✅ | eGene table with `pval_perm`, `fdr` (Storey q-value), `significance` |
| `eqtl_peak.csv` | ✅ | Peak locus per eGene — haplotype effects, run metadata, CI bounds |
| `eqtl_stats.csv` | ✅ | Run-level summary — LOD threshold, GEV params, gene counts, timing |
| `eqtl_scan.csv` | ✅ | Full haplotype scan — one row per gene × locus; `peak_locus` flag. Enable with `--eqtl-scan`. |
| `genome_summary.png` | ✅ | Genome-wide Manhattan and QC figures |

### Fine-mapping outputs (require `--sanger-alleles`)

IGVF variant-first column schema: `variant → rsid → chr → pos → ref → alt → effect_allele →
other_allele → gene_id → gene_symbol → pip → ...`

| File | Status | Description |
|---|---|---|
| `eqtl_finemapping.csv` | ✅ | All credible-set SNPs per eGene — PIP, ABF, CI bounds, biallelic contrast (effect/other allele, ref/alt founders, dosage, SE) |
| `eqtl_finemapping_peak.csv` | ✅ | Highest-PIP SNP per eGene with founder haplotype effects |

`dosage='T'` flags haplotype-probability-imputed calls (vs `'F'` for hard GigaMUGA genotype
calls). `se_effect_size` is derived via Wald inversion from the imputed SNP scan LOD.

---

## Models

| Model | Function | When to use |
|---|---|---|
| **NB-GLMM + LOCO** | `scan_nb_gene()` | Raw pseudobulk counts — primary for snRNA-seq eQTL |
| **Gaussian LMM** | `scan_h2lmm()` | Rank-normalised expression or strain-level phenotypes |
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
| **Validation** | Phase 4 (biological accuracy) pending | Phase 1–3 ✅ — use for all current analyses |

---

## Documentation

| Resource | Content |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, model derivations, inference choices |
| [`docs/math/`](docs/math/) | Step-by-step derivations for PQL, LOCO kinship, GEV thresholds, Storey q-value |
| [`config/`](config/) | Annotated parameter files for gastroc and ctxhipp reference runs |

---

## License

GPL-3.0

# mosaiQTL

**Haplotype-based QTL mapping for multiparent populations.**  
Maps molecular phenotypes to 8 founder haplotypes using a Gaussian LMM or NB-GLMM with LOCO
kinship correction. Designed for CC/DO mice (mm39); extensible to other MPPs.

> **Current status:** The **Gaussian LMM** (`--model lmm`) is the validated, production-ready
> model — Phase 1–3 gates pass, full gastroc scan complete, Sanger fine-mapping wired.
> The **NB-GLMM** (`--model nb`) is the intended long-term primary model for raw pseudobulk
> counts but has not yet passed end-to-end biological validation (Phase 4). Use `--model lmm`
> for any real analysis until Phase 4 is complete.

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

## Usage

### CLI

**LMM** (Gaussian LMM — current working model):

```bash
# 1. Genome-wide scan → per-gene peak LOD
mosaiqtl scan --model lmm \
    --cache cache.h5 --counts data.h5ad \
    --gene-info gene_info.csv --metadata meta.csv \
    --null-fit null_fit.csv --output peak_lod.csv

# 2. Permutation thresholds → GEV fit on max-LOD distribution
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

Use the API for custom covariate matrices, single-chromosome exploration, or downstream
analyses. For full runs, prefer the CLI — it checkpoints per-chromosome and resumes after
failures.

```python
import numpy as np
import pandas as pd
import anndata
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.utils import build_z_matrix
from mosaiqtl.io.covariates import build_covariate_matrix
from mosaiqtl.io.results_io import scan_result_to_df
from mosaiqtl.qtl.scan import scan_multi_gene
from mosaiqtl.qtl.thresholds import get_gev_thresholds
from mosaiqtl.qtl.fdr import calibrate_gene_pvalue
from mosaiqtl.qtl.ci import lod_support_interval, ci_scan_multi_gene
from mosaiqtl.qtl.effects import estimate_founder_effects
from mosaiqtl.pipeline import call_egenes, calibrate_egenes_storey, finemap_egenes

# 1. Cache + LOCO kinship (once per run)
cache = GenomeCache.from_hdf5('data/cache.h5')
kinship_loco   = calc_kinship(cache, method='loco')
kinship_decomp = decomp_kinship(kinship_loco)

# 2. Covariates + expression
meta        = pd.read_csv('gene_info.csv')   # columns: gene_id, gene_symbol, chr, start, end
sample_meta = pd.read_csv('metadata.csv')    # required: sample_id; auto-included: sex, batch
cov_df      = build_covariate_matrix(sample_meta, covariates=['sex', 'batch'])
X           = cov_df.values                  # (n_samples, p)
sample_ids  = list(cov_df.index)             # genotype-level IDs matching cache.sample_ids

adata    = anndata.read_h5ad('counts.h5ad', backed='r')
Y        = np.array(adata.raw.X.toarray(), dtype=np.float64)  # (n_genes, n_samples)
gene_ids = list(adata.var_names)

# 3. Genome-wide scan (streams one chromosome at a time)
scan_results = scan_multi_gene(
    phenotype_matrix=Y,
    gene_ids=gene_ids,
    genomecache=cache,
    gene_chr_map=dict(zip(meta['gene_id'], meta['chr'].astype(str))),
    covariates=X,
    kinship_decomp=kinship_decomp,
    sample_ids=sample_ids,
    use_rint=True,
    scan_window='cis',
    gene_pos_mb_map=dict(zip(meta['gene_id'], meta['start'] / 1e6)),
)

# 4. GEV threshold from permutation max-LOD distribution
max_lods  = pd.read_csv('perm/perm_max_lods.csv')['max_lod'].values
threshold, _ = get_gev_thresholds(max_lods)  # (None, None) on failure — fall back to quantile

# 5. Stage-1 coarse gate
peak_lod_df  = pd.concat(
    [scan_result_to_df(res, gid) for gid, res in scan_results.items()],
    ignore_index=True,
)
egenes_stage1 = call_egenes(peak_lod_df, threshold=threshold)

# 6. Stage-2 FDR: per-gene Beta calibration + Storey q-value
# calibrate_gene_pvalue requires strain-level phenotype — aggregate samples → strains first
Z, _ = build_z_matrix(sample_ids, list(cache.sample_ids))
r    = Z.sum(axis=0)                         # replicates per strain
X_g  = (Z.T @ X) / r[:, None]               # covariates at strain level

pval_perm = {}
for row in egenes_stage1.itertuples(index=False):
    gene_id = row.gene_id
    y_s     = Y[gene_ids.index(gene_id)]
    y_g     = (Z.T @ y_s) / r               # strain means
    p, *_   = calibrate_gene_pvalue(
        y_gene=y_g, genomecache=cache, chrom=str(row.chr),
        peak_lod=float(row.peak_lod), covariates=X_g,
        kinship_decomp=kinship_decomp, kinship_loco=kinship_loco,
        n_perms=200,
    )
    pval_perm[gene_id] = p

egenes = calibrate_egenes_storey(egenes_stage1, pval_perm, fdr_threshold=0.05)
# egenes gains columns: pval_perm, fdr, fdr_method, significance ('T'/'F')

# 7. Founder effects at each eGene's peak locus
effects_df = estimate_founder_effects(
    egenes_df=egenes, phenotype_matrix=Y, gene_ids=gene_ids,
    genomecache=cache, covariates=X, kinship_decomp=kinship_decomp,
)

# 8. Sanger fine-mapping (build alleles HDF5 once: build_founder_allele_mgp(vcf, cache, out))
finemap_egenes(
    egene_df=egenes, phenotype_matrix=Y, gene_ids=gene_ids,
    genomecache=cache, alleles_h5='data/sanger_alleles.h5',
    out_dir='results/', covariates=X, kinship_decomp=kinship_decomp,
    tag='gastroc_Lw1',
)
```

---

## Inputs

| Input | CLI flag | Format | Description |
|---|---|---|---|
| Genome cache | `--cache` | HDF5 | Haplotype probabilities + loci map. Build with `mosaiqtl build-cache`; source files from [JAX Diversity Outbred Reference Data](https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data). |
| Gene info | `--gene-info` | CSV | `gene_id`, `gene_symbol`, `chr`, `start`, `end` |
| Pseudobulk counts | `--counts` | AnnData `.h5ad` | Genes × samples raw counts in `.raw.X` (non-normalised integers) |
| Sample metadata | `--metadata` | CSV | `sample_id` required; `sex` (0=M, 1=F) and `batch` auto-included as covariates when present |
| NB null fit | `--null-fit` | CSV | Output of `mosaiqtl fit-null`; required for NB-GLMM scan |
| LOCO kinship | `--kinship` | HDF5 | Optional — built from `--cache` if not provided |

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
| `eqtl_peak.csv` | ✅ | Peak locus per eGene — haplotype effects, significance, run metadata |
| `eqtl_stats.csv` | ✅ | Run-level summary — tissue, cell type, LOD threshold, GEV parameters, gene counts |
| `genome_summary.png` | ✅ | Genome-wide Manhattan and QC figures |
| `eqtl_summary.csv` | ✅ | eGene summary table — includes `pval_perm`, `fdr` (Storey q-value), `fdr_method`, `significance` columns from Stage-2 FDR |
| `eqtl_scan.csv` | ✅ | Haplotype-level scan — one row per gene × GigaMUGA locus; `peak_locus` flag marks each gene's peak. Enable with `mosaiqtl report --eqtl-scan`. |
| `eqtl_finemapping.csv` | ✅ | Sanger-resolution fine-mapping — PIP, ABF credible set membership, CI bounds per variant. Requires `data/sanger_alleles.h5` (built once from MGP REL2021 VCF via `build_founder_allele_mgp`). |
| `eqtl_finemapping_peak.csv` | ✅ | Lead variant per eGene — highest-PIP Sanger SNP with haplotype effects and significance |

---

## Models

| Model | Function | When to use |
|---|---|---|
| **NB-GLMM + LOCO** | `scan_nb_gene()` | Raw pseudobulk counts — primary for snRNA-seq eQTL |
| **Gaussian LMM** | `scan_h2lmm()` | Normalized expression or strain-level phenotypes |
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

## Documentation

| Resource | Content |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | System design, model derivations, inference choices |
| [`docs/math/`](docs/math/) | Step-by-step derivations for PQL, LOCO kinship, GEV thresholds |
| [`config/`](config/) | Annotated parameter files for gastroc and ctxhipp reference runs |

---

## License

GPL-3.0

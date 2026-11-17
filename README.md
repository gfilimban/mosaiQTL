# mosaiQTL

**Haplotype-based QTL mapping for multiparent populations.**  
Maps molecular phenotypes to 8 founder haplotypes using a Gaussian LMM (production) or an
experimental NB-GLMM, with LOCO kinship correction. Designed for CC/DO mice (mm39); extensible
to other MPPs.

> **Current status (v0.7.0):** The **Gaussian LMM** is the validated, production model and the
> only model run by the CLI and the Nextflow workflow — Phase 1–3 gates pass, full gastroc
> scan complete, Sanger fine-mapping with IGVF-schema outputs wired end-to-end. The
> **NB-GLMM** (`qtl/nb.py`) is the intended long-term model for raw pseudobulk counts but is
> experimental: it is not wired into the CLI or Nextflow (Python API only, e.g.
> `pipeline.discover(model='nb')`) and has not yet passed end-to-end biological validation
> (Phase 4).

---

## Introduction

mosaiQTL performs genome-wide haplotype-based QTL scans for single-cell modalities. It resolves
associations to 8 founder haplotypes using a likelihood-ratio LOD under a Gaussian LMM
(production) with LOCO (leave-one-chromosome-out) kinship correction; an NB-GLMM score test is
available as an experimental Python API path. The pipeline streams one chromosome at a
time (peak RAM < 1 GB for 43 k genes across 20 chromosomes) and runs either as a local CLI or
a Nextflow workflow for HPC clusters.

**Supported modalities:** snRNA-seq eQTL · ATAC-seq caQTL (planned) · splicing QTL (planned)

| | mosaiQTL | miQTL (R) | tensorQTL |
|---|---|---|---|
| **Primary model** | Gaussian LMM (NB-GLMM experimental) | Gaussian LMM | Gaussian linear |
| **Haplotype scan** | ✓ (8 founders) | ✓ (8 founders) | ✗ |
| **LOCO kinship** | ✓ mandatory | ✓ | ✗ |
| **Raw count input** | NB-GLMM API only (not in CLI/Nextflow) | ✗ | ✗ |
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
# 1. Build a genome cache from GigaMUGA genotypes (separate `build-cache` console script)
build-cache --gigamuga GM_ccf1fo_mm39.csv --gmap-dir gmap/ --pmap-dir pmap/ \
    --cc-cols cc_cols.json --founder-cols founders.json --out-dir ref/   # → ref/genomecache.h5

# 2. LOCO kinship eigendecomposition (once per cache)
mosaiqtl build-kinship --cache ref/genomecache.h5 --output ref/kinship_decomp.h5

# 3. Per-chromosome LMM scan (repeat for every chromosome), then merge
mosaiqtl scan-chr --cache ref/genomecache.h5 --kinship ref/kinship_decomp.h5 \
    --phenotypes counts_lognorm.h5 --gene-info gene_info.csv --metadata meta.csv \
    --chr 1 --scan-type L --window-mb 1.0 --output-dir scan/
mosaiqtl merge-scan --scan-dir scan/ --metadata meta.csv --gene-tss gene_tss.csv \
    --scan-type L --suffix ct_Lw1 --output-dir results/   # → lod_/allele_effects_/peak_lod_ct_Lw1.csv

# 4. Permutation null → threshold.csv, perm_max_lods*.csv
mosaiqtl permute --cache ref/genomecache.h5 --kinship ref/kinship_decomp.h5 \
    --phenotypes counts_lognorm.h5 --metadata meta.csv --gene-info gene_info.csv \
    --allele-effects results/allele_effects_ct_Lw1.csv --output perm/ --n-perms 1000

# 5. Per-gene calibrated p-values, then eGene calling (Storey q-value)
mosaiqtl calibrate --cache ref/genomecache.h5 --kinship ref/kinship_decomp.h5 \
    --peak-lod results/peak_lod_ct_Lw1.csv --perm-dir perm/ \
    --phenotypes counts_lognorm.h5 --metadata meta.csv --output results/pval_perm.csv
mosaiqtl call-egenes --peak-lod results/peak_lod_ct_Lw1.csv --perm-dir perm/ \
    --pval-perm results/pval_perm.csv --output results/eqtl_summary_ct_Lw1.csv

# 6. Support intervals → submission tables (eqtl_peak + eqtl_stats)
mosaiqtl characterise --egenes results/eqtl_summary_ct_Lw1.csv --phenotypes counts_lognorm.h5 \
    --kinship ref/kinship_decomp.h5 --cache ref/genomecache.h5 --metadata meta.csv \
    --suffix ct_Lw1 --output-dir results/
mosaiqtl report --egenes results/eqtl_summary_ct_Lw1.csv \
    --allele-effects results/allele_effects_ct_Lw1.csv --ci results/eqtl_peak_ci_ct_Lw1.csv \
    --perm-dir perm/ --cache ref/genomecache.h5 --peak-lod results/peak_lod_ct_Lw1.csv \
    --suffix ct_Lw1 --output-dir results/

# 7. Optional Sanger fine-mapping: `impute` (imputed SNP scan) then `finemap` (ABF credible sets)
```

Known issue: `mosaiqtl impute` currently crashes because `pipeline.impute_egenes` returns a
`(DataFrame, ld_by_gene)` tuple; use the Nextflow workflow or `pipeline.finemap_egenes` for
fine-mapping.

For parallel per-chromosome HPC runs see **Nextflow pipeline** below.

---

## CLI reference

### Subcommands

| Subcommand | Description |
|---|---|
| `build-cache` | Separate console script (`mosaiqtl.genotype.cache_builder:build`): GigaMUGA CSV → `genomecache.h5` (+ per-group probs, raw LOCO kinship) |
| `build-kinship` | LOCO kinship eigendecomposition → kinship HDF5 (`chr{c}/values`, `chr{c}/vectors`) |
| `inspect-cache` | Cache / kinship QC tables |
| `scan-chr` | Replicate-aware LMM scan of one chromosome → `lod_chr{c}.csv`, `effects_chr{c}.csv` |
| `merge-scan` | Per-chromosome outputs → `lod_`, `allele_effects_`, `peak_lod_{suffix}.csv` |
| `permute` | Permutation null → `perm_max_lods*.csv` + GEV `threshold.csv` |
| `calibrate` | Per-gene Beta-calibrated permutation p-values → `pval_perm.csv` |
| `call-egenes` | LOD gate + Storey q-values → `eqtl_summary` |
| `characterise` | 1.5-LOD support intervals → `ci_`, `eqtl_peak_ci_{suffix}.csv` |
| `impute` | Imputed SNP scan inside each interval (optional) |
| `finemap` | ABF credible sets from the imputed scan (optional) |
| `report` | Submission tables `eqtl_peak` + `eqtl_stats` (+ fine-mapping, genome summary plot) |
| `scan-mi`, `mediate`, `assess-stability` | Follow-up scans on eGenes |

Run `mosaiqtl <subcommand> --help` for the full flag list.

Known issue: CLI `calibrate`/`call-egenes` compute Storey q-values only over genes that pass
the genome-wide LOD gate (the library documents this as invalid); the Nextflow path calibrates
every tested gene.

### Model flag

There is no model flag: the CLI and the Nextflow workflow run the Gaussian LMM only. The
`model` parameter in `nextflow.config` is not read by any process. The NB-GLMM is reachable
only from the Python API (`pipeline.discover(model='nb')`, the default there, or
`qtl.nb.scan_nb`).

### Scan window

`--scan-type L` (local ±`window_mb` around the strand-aware TSS, default `--window-mb 1.0`;
recommended for cis-eQTL), `C` (whole chromosome), or `G` (genome-wide). The permutation null
must use the same `--scan-type` and `--window-mb` as the scan.

---

## Nextflow pipeline (HPC)

The Nextflow DSL2 workflow (`workflow/main.nf` → `workflow/scripts/run_*.py`) is the production
path: it runs one LMM scan job per (cell type × chromosome) on SLURM with a shared LOCO kinship
file (`ref/igvf_kinship_loco.h5`, built only when absent).

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
| `covariates` | Comma-separated metadata columns used as fixed effects (default `sex,dataset`) |
| `use_rint` | `true` (default) for rank-normalised expression; `false` for raw Gaussian phenotypes |
| `n_perms` | Number of permutations for the genome-wide GEV null (QC reference threshold) |
| `n_perms_cal` | Per-gene parametric-null draws for the Beta-calibrated p-values that feed Storey FDR |

---

## Python API

Use the API for custom workflows, single-chromosome exploration, or downstream analyses.
For full runs, prefer the Nextflow workflow — it runs per-chromosome jobs and `-resume`
replays only changed stages.

```python
import pandas as pd
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.utils import build_tss_lookup, build_z_matrix
from mosaiqtl.io.anndata_io import load_expression_h5
from mosaiqtl.io.covariates import build_cov_array, build_sample_geno_map
from mosaiqtl.qtl.scan import scan_multi_gene_lmm
from mosaiqtl.qtl.thresholds import get_gev_thresholds
from mosaiqtl.qtl.ci import ci_scan_multi_gene
from mosaiqtl.pipeline import (
    call_significant_egenes, finemap_egenes, prepare_egenes_for_finemap,
)

# 1. Cache + LOCO kinship (once per run)
cache          = GenomeCache.from_hdf5('ref/igvf_cache.h5')
kinship_loco   = calc_kinship(cache, method='loco')   # {chrom: (m, m)}
kinship_decomp = decomp_kinship(kinship_loco)         # {chrom: {'values', 'vectors'}}

# 2. Metadata, covariates, expression
meta      = pd.read_csv('metadata.csv')    # required: sample_id, genotype; covariates e.g. sex, dataset
gene_info = pd.read_csv('gene_info.csv')   # gene_id, gene_symbol, chr, start, end, strand
# counts_lognorm.h5 holds datasets counts (genes × samples), genes, samples
Y, gene_ids, sample_ids = load_expression_h5('counts_lognorm.h5', gene_info['gene_id'].tolist())
geno_map = build_sample_geno_map(meta)                  # sample_id → genotype (strain)
geno_ids = [geno_map.get(s, s) for s in sample_ids]
cov_arr  = build_cov_array(meta, sample_ids, ['sex', 'dataset'])  # (n_samples, p)

# 3. Local scan ±1 Mb of the TSS (streams one chromosome at a time)
peak_lod_df = scan_multi_gene_lmm(
    expr_cache={g: Y[i] for i, g in enumerate(gene_ids)},
    gene_chr_map=dict(zip(gene_info['gene_id'], gene_info['chr'].astype(str))),
    genomecache=cache,
    covariates=cov_arr,
    kinship_decomp=kinship_decomp,
    sample_ids=geno_ids,
    use_rint=True,
    scan_window='local',
    gene_pos_mb_map=build_tss_lookup(gene_info),   # strand-aware TSS (Mb)
    window_mb=1.0,
    gene_meta=gene_info[['gene_id', 'gene_symbol']],
)   # DataFrame, one row per gene: peak_locus, peak_mb, peak_lod, h2, ..., beta_{founder}

# 4. Reference GEV threshold from a permutation max-LOD null (QC only)
max_lods     = pd.read_csv('perm/perm_max_lods.csv')['max_lod'].values
threshold, _ = get_gev_thresholds(max_lods)   # (None, None) on failure

# 5. eGene calling over ALL tested genes: per-gene Beta-calibrated p-value
#    → Storey q-value → founder effects for significant genes
Z, valid = build_z_matrix(geno_ids, list(cache.sample_ids))
called = call_significant_egenes(
    peak_lod_df, genomecache=cache, pheno_mat=Y,
    gene_idx={g: i for i, g in enumerate(gene_ids)},
    covariates=cov_arr, kinship_decomp=kinship_decomp, kinship_loco=kinship_loco,
    sample_ids=geno_ids, stag='Lw1', gene_info_df=gene_info,
    fdr_threshold=0.05, n_perms=1000, use_rint=True,
    z_mat=Z, valid_mask=valid, seed=42,
)
egenes = prepare_egenes_for_finemap(called[called['significance'] == 'T'])

# 6. 1.5-LOD support interval per eGene, in bp
ci = ci_scan_multi_gene(
    {g: Y[gene_ids.index(g)] for g in egenes['gene_id']},
    dict(zip(egenes['gene_id'], egenes['chr'])),
    dict(zip(egenes['gene_id'], egenes['peak_mb'])),
    cache, covariates=cov_arr, kinship_decomp=kinship_decomp, sample_ids=geno_ids,
)
ci_bp = ci.assign(ci_lo_bp=(ci['ci_lo'] * 1e6).round().astype('Int64'),
                  ci_hi_bp=(ci['ci_hi'] * 1e6).round().astype('Int64'))
egenes_fm = egenes.assign(
    pos=(egenes['peak_mb'] * 1e6).round().astype('Int64'),
    lod=egenes['peak_lod'], variant=egenes['peak_locus'],
).merge(ci_bp[['gene_id', 'ci_lo_bp', 'ci_hi_bp']], on='gene_id', how='left')

# 7. Sanger fine-mapping → IGVF-schema outputs (eqtl_finemapping[_peak].csv)
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
| Genome cache | `--cache` | HDF5 | Haplotype probabilities + loci map. Build with the `build-cache` console script; source GigaMUGA probs from [JAX DO Reference Data](https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data). |
| Gene info | `--gene-info` | CSV | `gene_id`, `chr`, `start` required; `end`, `strand` (strand-aware TSS), `gene_symbol` optional |
| Pseudobulk expression | `--phenotypes` | HDF5 (`counts_lognorm.h5`) | Datasets `counts` (genes × samples, log-normalised), `genes`/`gene_ids`, `samples`/`sample_ids` |
| Sample metadata | `--metadata` | CSV | `sample_id` and `genotype` (strain) required; covariates are the columns named by `--covariates` (default `sex,dataset`; numeric columns pass through, string columns are one-hot encoded) |
| Sanger alleles | `--alleles` (`impute`); Nextflow `sanger_alleles` | HDF5 | Built once from MGP REL2021 VCF via `build_founder_allele_mgp()`; optional (Nextflow writes header-only fine-mapping files if absent) |
| LOCO kinship | `--kinship` | HDF5 | Required by the CLI scan commands; build with `mosaiqtl build-kinship` (Nextflow builds `ref/igvf_kinship_loco.h5` when absent) |

---

## Outputs

Nextflow output files are named `{tissue}_{celltype}_{scan_tag}_{filename}` (tissue omitted
when empty; e.g. scan_tag `Lw1`); CLI outputs are named `{filename}_{suffix}.csv`.

### Scan outputs

| File | Status | Description |
|---|---|---|
| `peak_lod.csv` | ✅ | Per-gene peak LOD, peak locus, `h2`, `cis_dist_mb`, null-model diagnostics (`PEAK_LOD_COLS`); Nextflow copy also carries `beta_{founder}` and, after eGene calling, `is_egene` |
| `allele_effects.csv` | ✅ | CLI `merge-scan` only: per-founder `beta_{founder}` at each gene's peak locus |
| `eqtl_summary.csv` | ✅ | eGene table with `pval_perm`, `fdr` (Storey q-value), `significance` (Nextflow: significant eGenes only) |
| `eqtl_peak.csv` | ✅ | CLI `report` only: peak locus per eGene — haplotype effects (`effect_{founder}`), significance columns |
| `eqtl_stats.csv` | ✅ | Per-chromosome + genome rows — LOD threshold, GEV params, `n_perms`, eGene counts, FDR threshold, scan window, git hash |
| `eqtl_scan.csv` | ❌ | Full haplotype scan (`EQTL_SCAN_COLS`) — writer `io.results_io.write_eqtl_scan` exists but is not wired into the CLI or Nextflow |
| `genome_summary.png` | ✅ | Genome-wide Manhattan and QC figures (CLI `report --peak-lod`; Nextflow REPORT also writes further plots) |

### Fine-mapping outputs (require a Sanger alleles HDF5)

IGVF variant-first column schema: `variant → rsid → chr → pos → ref → alt → gene_id →
gene_symbol → cis_dist_mb → pip → CS → lead_snp → pip_method → ...`

| File | Status | Description |
|---|---|---|
| `eqtl_finemapping.csv` | ✅ | All credible-set SNPs per eGene — PIP, CS, CI bounds, biallelic contrast (effect/other allele, ref/alt founders, dosage, SE) |
| `eqtl_finemapping_peak.csv` | ✅ | Highest-PIP SNP per eGene with founder haplotype effects |

`dosage='T'` flags haplotype-probability-imputed calls (vs `'F'` for hard GigaMUGA genotype
calls). `se_effect_size` is derived via Wald inversion from the imputed SNP scan LOD.

---

## Models

| Model | Function | When to use |
|---|---|---|
| **NB-GLMM + LOCO** | `scan_nb_gene()` | Raw pseudobulk counts — experimental (Python API only) |
| **Gaussian LMM** | `scan_h2lmm()`, `scan_multi_gene_lmm()` | Rank-normalised expression or strain-level phenotypes — production (CLI + Nextflow) |
| **LMM + MI** | `scan_h2lmm_mi()` | Averages over haplotype-assignment uncertainty |
| **Dispersion QTL** | `scan_vqtl()` | NB dispersion as phenotype (transcriptional noise) |
| **Mediation** | `mediation_scan_qr()` | Candidate mediator LOD via QR Woodbury update |
| **Epistasis** | `scan_epistasis()` | Pairwise founder × founder interaction scan |

**Choosing NB-GLMM vs LMM:**

| | NB-GLMM (Python API, experimental) | LMM (CLI / Nextflow) |
|---|---|---|
| **Input** | Raw integer pseudobulk counts | Log-normalised / continuous phenotype |
| **Overdispersion** | Modelled explicitly (NB $\phi$ per gene) | Not modelled — inflated false positives for low-expr genes |
| **Stage 0** | Required — null fit (`fit_null_nb_glmm_loco` / `fit_null_sc`) estimates $\phi$ and $\sigma^2$ per gene | Not needed — $h^2$ estimated (ML) inside the scan |
| **Speed** | Slower (PQL null per gene × chr) | Faster ($h^2$ once per gene per chr, FWL vectorised over loci) |
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

GPL-3.0-or-later

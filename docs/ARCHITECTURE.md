# mosaiQTL — Architecture, Scope & Intention
# Human-readable package overview
# Read this file before touching any code.
# Companion to: CLAUDE.md (Claude Code instructions) | docs/REFERENCES.md (bibliography)

---

## 1. WHAT IS mosaiQTL?

mosaiQTL is a python package providing **multiomic, haplotype-resolved quantitative trait locus (QTL) mapping
framework** for multiparent populations (MPPs). mosaiQTL is designed as a generalizable, multi-model hub unified by a shared genotype probability backbone. It maps genetic variants controlling gene
expression, chromatin accessibility, splicing, and RNA modifications at **cell-type
resolution** from single-nucleus sequencing data from $n$ repliacate observations based on scan configuration space built using user input.


The central biological question: *which genomic regions — and which of founder haplotypes
carried in those regions — causally regulate molecular phenotypes across cell types?*


The package is parametric on population structure: the user specifies
$J$ founder strains, the set of derived lines, and the number of
replicates. Any multiparent design is supported — CC ($J = 8$),
DO ($J = 8$, outbred), MAGIC ($J = 4$, $8$, or $19$), heterogeneous
stocks, or custom crosses. The Collaborative Cross 8-founder population
serves as the proof-of-concept application.

**Current primary application**: eQTL mapping in mouse tissue using
single-nucleus RNA-seq (snRNA-seq) across the MPP, Collaborative Cross (CC), *33* strains +
*8* founders + *7* F1 hybrids = 48 genotypes, ~418 samples post-QC.
Genotyping: GigaMUGA array (~143k markers). Reference genome: mm39 (GRCm39).
Project: IGVF (Impact of Genomic Variation on Function), Seyedam Lab, UCI.

---

## 2. SCOPE

### In scope (implemented or planned)
| Modality        | Trait/Condition                  | QTL Code   | Omics type         | Module                  | Status            |
| --------------- | -------------------------------- | ---------- | ------------------ | ----------------------- | ----------------- |
| RNA-seq/DNA-seq | Gene Expression                  | cis-eQTL   | Transcriptomics    | nb.py (primary NB-GLMM) | 🔴 planned        |
| RNA-seq/DNA-seq | Distal Regulation                | trans-eQTL | Transcriptomics    | trans_scan.py           | 🔴 planned        |
| RNA-seq/DNA-seq | Dispersion Variance              | vQTL       | Transcriptomics    | vqtl.py                 | 🔴 planned        |
| RNA-seq/DNA-seq | Epistasis / conditional QTL      |            | Transcriptomics    | epistasis.py            | 🔴 planned        |
| RNA-seq/DNA-seq | Haplotype mediation              |            | Transcriptomics    | mediation.py            | 🔴 planned        |
| RNA-seq/DNA-seq | Allele-specific expression (ASE) |            | Transcriptomics    | ase.py                  | 🟡 scaffold       |
| LR-RNA-seq      | Splicing                         | sQTL       | Transcriptomics    | sqtl.py                 | 🔴 planned        |
| RNA-seq/DNA-seq | SNP fine-mapping                 |            | Genomics           | snp_scan.py             | 🟡 marginal model |
| ATAC-seq        | Chromatin accessibility          | caQTL      | Genomics           | atac.py                 | 🔴 planned        |
| dRNA-seq        | RNA modification                 | rmodQTL    | Epigenomics        | mod.py                  | 🔴 planned        |
| Fiber-seq       | DNA modification                 | dmodQTL    | Epigenomics        | mod.py                  | 🔴 planned        |
|                 | Permutation-based thresholds     |            | All modalities     | permutation.py          | 🟡 GEV unstable   |
|                 | Sex × genotype interaction (GxE) |            | All modalities     | gxe.py                  | 🔴 planned        |
|                 | Multi-tissue / multi-cell-type   |            | All modalities     | pipeline.py layer       | 🔴 planned        |
|                 | Streamlit data browser           |            | Public data access | app/                    | 🔴 planned        |

All modalities share nb.py as the primary scan engine (NB-GLMM + LOCO via Woodbury).
New modalities require only a new Stage 1 input handler — the scan is unchanged.

### Out of scope (explicit boundaries)
- **Pseudobulk aggregation**: mosaiQTL takes pre-aggregated pseudobulk counts and raw
  count matrices as inputs. Aggregation (sum UMI per sample × cell-type) and
  normalization are performed upstream by the user's pipeline.
- FASTQ alignment (here, kallisto)
- Cell-type clustering and annotation (here, leiden-based clustering was done and annotated using a marker-oriented manual method;  provide annotated AnnData)
- Doublet detection and ambient RNA removal (here, cellsweep was used; feed clean AnnData)
- Proteomics / mass spectrometry pQTL
- GWAS / common-variant association with individual SNPs as primary model
- Human data (mouse mm39 only; generalization planned post-publication)

---

## 3. MODULE STRUCTURE (src/mosaiqtl/)

```
src/mosaiqtl/
├── __init__.py
├── genotype/               # haplotype data, HMM, kinship, founders
│   ├── CLAUDE.md
│   ├── genomecache.py      # read/write HAPPY genome cache; from_rds, from_gigamuga
│   ├── hmm.py              # forward-backward HMM (n_gen=1 for CC)
│   ├── kinship.py          # LOCO, eigendecomp cache, Woodbury identity
│   └── founders.py         # FOUNDER_ORDER, FOUNDER_COLORS, F1_COLORS
├── qtl/                    # QTL scan engine — all modalities
│   ├── CLAUDE.md
│   ├── nb.py               # NB-GLMM Stage 1 and 2 (primary model, transcriptomics)
│   ├── lmm.py              # Gaussian LMM engine (qtl2/miqtl-port); scan_chromosome_batch with return_peak_positions; scan_chromosome_mi vectorised over imputations
│   ├── scan.py             # Z-matrix n=418, MI (vectorised), fixef, instability (strain-level subsampling); scan_window='cis'|'genome'|'local'
│   ├── permutation.py      # fix_h2=True, GEV thresholds; run_positional_scans vectorised via scan_chromosome_batch
│   ├── effects.py          # founder haplotype effect estimation
│   ├── mediation.py        # QR-based mediation; mediation_scan_qr_batch vectorises all mediators per outcome via Woodbury rank-1
│   ├── epistasis.py        # pairwise GxG, conditional scanning
│   ├── vqtl.py             # dispersion QTL, VQTLResult
│   ├── snp_scan.py         # marginal SNP fine-mapping
│   └── ase.py              # allele-specific expression (scaffold)
├── models/                 # shared dataclasses and Pydantic schemas
│   ├── records.py          # ScanResult, VQTLResult, QTLResult, FounderEffect
│   └── config.py           # ScanConfig, PipelineConfig (Pydantic v2)
├── io/                     # all file I/O and covariate construction
│   ├── CLAUDE.md
│   ├── covariates.py       # build_covariate_matrix(); encode_sex() utility; user covariate input
│   ├── anndata_io.py       # AnnData read helpers (backed=True always)
│   ├── qtl2_io.py          # R/qtl2 format I/O (rpy2, optional extra)
│   └── results_io.py       # result table serialisation with provenance header
├── viz/                    # all visualisation
│   ├── CLAUDE.md
│   ├── genome.py           # lod_profile (fill=), loci_per_chr, perm_distribution, egene_lod_distribution, mediation_profile, ci_strip
│   ├── effects.py          # founder effect bar plots (Phase 4 biological)
│   ├── haplotypes.py       # prob_heatmap, kinship_distribution, epistasis grids (Phase 4)
│   ├── validation.py       # Phase 2 R vs Python QC plots ONLY (S1–S11)
│   └── celltypes.py        # UMAP, dotplots (cell-type summaries)
├── pipeline.py             # discover, characterize, resolve
├── cli.py                  # Click CLI (mosaiqtl fit-null/scan/permute/effects/report)
└── utils.py                # RINT, Z-matrix, align_samples, build_z_matrix
```

### edgePython NB-engine inspiration
mosaiQTL NB-GLMM model is inspired by[edgePython](https://github.com/pachterlab/edgePython)(Pachter 2026) to implement the negative
binomial fitting, dispersion estimation, and NEBULA-LN likelihood


**Scope boundary enforced in structure:**
- `preprocessing/`, `annotation/` — do not exist in mosaiQTL; these are upstream
- `pseudobulk/aggregate.py`, `pseudobulk/normalize.py` — do not exist; aggregation is upstream
- `io/covariates.py` — the only pseudobulk-related code: reads user-supplied metadata, builds covariate matrix

**Validation plot assignment:**
- `viz/validation.py` — Phase 2 only (S1–S11 R vs Python QC). Never imported from installed package; tests/ only.
- `viz/genome.py` + `viz/effects.py` + `viz/haplotypes.py` — Phase 4 biological plots (P1–P4) and production figures

**Planned modality modules** (not yet created):
```
qtl/atac.py       # ATAC-seq Stage 1 — chromatin accessibility QTL
qtl/sqtl.py       # splicing ratios Stage 1 — sQTL
qtl/rnamod.py     # m6A / RNA modification counts Stage 1
qtl/gxe.py        # sex × genotype interaction
qtl/trans_scan.py # trans-eQTL with distance filter
```


---

## 4. DESIGN INTENTION

### Why this tool exists
miQTL (R) is the canonical haplotype QTL package for CC/DO mice. It is correct but
slow, R-only, bulk-data-based, and lacks replicate-awareness and cell-type resolution from single-nucleus data. 
mosaiQTL ports and extends it to Python with:
1. A **two-stage NB-GLMM model** that operates on raw counts without ad-hoc normalization
2. **Cell-type resolution** via per-cell-type pseudobulk inputs (aggregated upstream)
3. **Novel extensions** (vQTL, epistasis, mediation, ASE) absent from miQTL
4. **HPC scalability** via Nextflow SLURM pipeline
5. **Multiomics generalization** — same backbone for ATAC, splicing, RNA/DNA modifications


### USER INPUT 

scan configuration space is defined by the user orthogonal choice axes (except test axis model-conditional strains) at scan time: 


| Axis                  | CLI flag     | Options                                                       | Default                            |
| --------------------- | ------------ | ------------------------------------------------------------- | ---------------------------------- |
| **Population**        | --pop        | `cc`/`do`/`mpp`/`f2`/`structured`                             | user-specified                     |
| **Statistical Model** | --model      | `nb-glmm`/`lmm`                                               | `nb-glmm`                          |
| **Test Statistic**    | --test       | nb-glmm: `score`/`lrt` ; lmm: `lrt`/`f_test`                  | nb-glmm: `score` ; lmm: `lrt`      |
| **Phenotype input**   | --input      | `raw`/`normalized`/`averaged` counts                          | nb-glmm: `raw` ; lmm: `normalized` |
| **Granularity level** | --granular   | `single-cell`/`single-nuclei`/`pseudobulk`/`bulk`             | user-specified                     |
| **Scan Window**       | --window     | `cis`/`chr`/`genome`/`loci=set`                               | `chr`                              |
| **Offset**            | --set-offset | `from_libsize`/`from_file`/`none`                             | `none`                             |
| **Covariates**        | --covar      | `none`/`user`/`expr_pcs`/`geno_pcs`/ or a mix between any two | `none`                             |
| **Haplotype**         | --haplotype  | `additive`/`full`/`mi`                                        | `additive`                                   |


### Design philosophy
- **Algorithmic innovation over hardware**: score test fits null once per gene, then
  vectorises over all loci — skips IRLS re-fit per marker (analogous to kallisto
  pseudoalignment skipping full alignment). Numba JIT for IRLS hot path when profiling shows need.
- **Composable single-purpose CLI**: `mosaiqtl fit-null`, `scan`, `permute`, `effects`,
  `report` — each subcommand reads/writes files; no hidden state between stages
- **Constant-memory streaming**: never materialize the full (genes × loci) LOD matrix;
  stream per-chromosome, merge on completion
- **Proper data management**: create and save analysis-friendly raw and intermediate data; ensures ability to go back and pick up from any step
- **Minimal core deps**: `pip install mosaiqtl` requires only numpy+pandas+scipy+anndata+h5py
  (DESIGN-01); GPU and R extras are opt-in
- **Math companion before code**: every algorithm in `docs/math/` has a derivation
  before implementation. PRs require derivation → pseudocode → code order (DESIGN-03/05).
- **Reproducibility baked in**: `ProvenanceRecord` on every scan; seed always written;
  git commit hash embedded in all output files (DESIGN-06/07)
- **Measure before optimising**: scalene + pytest-benchmark before any perf PR;
  benchmark results committed to repo (DESIGN-04)
- **Moment stats before MLE**: moment-based dispersion default; MLE via flag

### Why NB-GLMM (not pure LMM)?
Single-nucleus RNA-seq counts are overdispersed integers. Log-normalising and treating
as continuous (LMM) loses statistical power and can inflate false positives for
lowly-expressed genes. The NB-GLM captures count overdispersion explicitly.
nb.py implements both stages: Stage 1 (NB-GLMM cell-level null) and Stage 2 (PQL NB-GLMM score scan with LOCO kinship integrated via Woodbury identity, all matrix ops at m×m cost). lmm.py is the Gaussian LMM validated port. See: He et al. 2021 (NEBULA, DOI:
10.1038/s42003-021-02146-6) and Yazar et al. 2022 (DOI: 10.1126/science.abf3041).

### Why LOCO kinship?
The LOCO (leave-one-chromosome-out) kinship matrix is recomputed excluding markers
on the chromosome being tested. This prevents the QTL signal from inflating the
random-effect kinship estimate — a bias known as proximal contamination.
See: Listgarten et al. 2012 (DOI: 10.1038/nmeth.2037).

---

## 5. Kinship


The realized kinship between subjects $i$ and $j$ from allele-sharing
across $L$ loci:

$$K_{ij} = \frac{1}{L} \sum_{\ell=1}^{L} \sum_{f=1}^{J} H_{i\ell f} \, H_{j\ell f}$$

**LOCO:** For chromosome $c$, exclude all loci on $c$:

$$K_{ij}^{(-c)} = \frac{1}{L_{-c}} \sum_{\ell \notin c} \sum_{f=1}^{J} H_{i\ell f} \, H_{j\ell f}$$

**Expansion:** Kinship is computed in 48-genotype space (unique strains).
For replicated designs ($n = 384$ or $418$ animals), expand via index
replication:

$$\tilde{K}_{ij} = K_{s(i), \, s(j)}$$

where $s(i)$ maps animal $i$ to its strain genotype.

---

## 6. DATA FLOW

```
[UPSTREAM — not in mosaiQTL, performed by user]
Raw FASTQ (per strain × batch)
  → Kallisto
  → scanpy QC: joint filters, doublet detection, 
  → cellsweep: ambient RNA removal
  → Leiden clustering + cell-type annotation
  → Pseudo-bulk aggregation: sum raw UMI per unit × category
  → TMM / DESeq2 normalization
  → Output: per-cell-type AnnData (adata.raw = raw counts) + sample metadata CSV

[mosaiQTL BOUNDARY — inputs below here]
mosaiQTL accepts:
  (A) Pseudobulk count matrix  (n_samples × n_genes) per cell type — raw integer counts
  (B) AnnData with adata.raw preserved — for NB Stage 1 cell-level fit
  (C) Sample metadata CSV — with mandatory cols: sample_id, sex, batch
      plus any user-supplied covariates (age, body_weight_g, etc.)
  (D) Genome cache (HDF5) — haplotype probs

**Expression loading is the caller's responsibility — no HDF5 inside engine functions.**

All scan functions (`scan_h2lmm`, `scan_h2lmm_mi`, `mediation_scan_qr_batch`,
`instability_lm_scan`, `imputed_snp_scan_h2lmm`, etc.) accept pre-loaded `np.ndarray`
inputs. They perform no file I/O. The caller must load gene expression vectors before
invoking any engine function.

**Antipattern — per-gene HDF5 open/scan/close inside a loop:**
```python
for gene in egenes:
    y, sids, _ = _load_expression(h5_path, [gene_id])  # ← BAD
    scan_h2lmm(y, ...)
```
At gastroc scale (43,650 genes × 415 samples, chunks=(1000, 50)), each call opens the
file, reads all gene ID strings, and decompresses ~9 column chunks to read one row.
With 50+ outcomes × 90 mediators, this adds 10+ minutes of I/O overhead (measured).

**Canonical pattern — single batch preload before the loop:**
```python
all_ids = [gene_name_to_id.get(g, g) for g in needed_genes]
counts, loaded_ids, sids = _load_expression(h5_path, all_ids)  # ← ONE call
expr = {gid: counts[i] for i, gid in enumerate(loaded_ids)}
for gene in egenes:
    y = expr[gene_id].astype(float)
    scan_h2lmm(y, ...)
```
`sids` (the sample list) is identical for all genes — the HDF5 sample axis is fixed.
Memory cost: (n_needed_genes × n_samples) × 4 bytes — typically < 1 MB for per-section
subsets of eGenes. See `validate_gastroc.py` S7–S11 for the reference implementation.

  → Covariate matrix construction        [io/covariates.py]
      - mandatory: sample_id, sex, batch
      - user-supplied: any numeric/categorical column from sample metadata
      - expression PCs (n_expression_pcs=3 default; PEER-equivalent)
      - categorical user covariates one-hot encoded automatically
      - NaN in any column: raises MissingCovariateError before scan
          - Highlight which column

  → Genome cache (HDF5)                  [genotype/genomecache.py]
      - haplotype probs shape (n_samples, 8, n_loci)
      - allele encoding uses CC strain calls via qtl2convert
  → LOCO kinship matrices                [genotype/kinship.py]
      - one (n_samples × n_samples) matrix per chromosome
      - Woodbury identity; eigendecomp cached

  → Stage 1+2: NB-GLMM scan            [qtl/nb.py — primary]
      Stage 1 (fit_null_sc_native): NEBULA-LN cell null per gene
        - output: phi_shrunk (EB dispersion), sigma2_sample — phi fixed for all Stage 2
      Stage 2 (scan_nb_gene + fit_null_nb_glmm_loco): PQL NB-GLMM with LOCO kinship
        - Model: S_i ~ NB(mu_i, phi), log mu_i = x^T beta + u_i + offset
                 u ~ N(0, tau K_LOCO). phi held fixed from Stage 1.
        - PQL each iteration: W = mu/(1+phi*mu)  [NB variance fn, phi-dependent]
                              z = (log mu - offset) + (S-mu)/mu  [NB working response]
                              Gaussian solver estimates beta, u_hat, tau from (W,z)
                              mu = exp(x@beta + u_hat + offset)  [back to NB]
          W and z rederived from NB mu every iteration — model stays NB throughout.
          Converged solution: PQL approximation to true NB-GLMM (Chen et al. 2016).
        - Woodbury: V=diag(1/W)+tau*Z K_m Z^T — ops at m×m, never n×n
        - Score stat: Q_l ~ chi2(J-1); LOD = Q_l/(2 ln 10)
  → LMM validated port                  [qtl/lmm.py, qtl/scan.py]
      - Gaussian eigendecomp LMM (Kang 2008 / qtl2) for Phase 2 R comparison
      - called by: scan.py, permutation.py, mediation.py, epistasis.py

  → Permutation thresholds               [qtl/permutation.py]
      - fix_h2=True; GEV fit on max-LOD distribution
      - minimum 1000 permutations; 5000 preferred
      - expression vector permuted (not genotype)

  → FDR correction                       [pipeline.py]
      - BH-FDR (anti-conservative without permutation; Storey q-value planned)
      - eGene discovery

  → Haplotype effects                    [qtl/effects.py]
      - 8 per-founder effect estimates at peak locus
      - standard errors + CIs

  → Optional downstream                  [qtl/mediation.py, epistasis.py, vqtl.py]
      - QR-based mediation: mediation_scan_qr_batch vectorises all mediators of one outcome per chromosome via Woodbury rank-1 update (H⊥X computed once; G_inv shared; per-mediator LOD via Sherman-Morrison). h2 fixed from base model (drift ≤ 0.03 empirically). ~100–250× faster than per-mediator loop at 127k pairs.
      - pairwise epistasis scan
      - dispersion QTL (vQTL)
      - conditional multi-signal scan

Final outputs:
  - eGene tables per cell type (gene_id, chr, pos_mb, LOD, FDR, founder effects)
  - LOD profile CSVs
  - QC plots (PNG, dpi=300, font Carlito)
  - HTML Nextflow run report
```

---

## 7. COVARIATE HANDLING

Users may supply covariates via the `fixed_covariates` and `user_covariates` parameters in
`build_covariate_matrix()` (in `io/covariates.py`) and by passing a metadata CSV to the
Nextflow pipeline. mosaiQTL does not generate covariates — it consumes them from
user-supplied sample metadata alongside the mandatory defaults.

```python
# covariates (user must supply these columns in sample metadata)
mandatory = ['sex', 'batch']
# sex:           X-linked dosage; hormone-driven expression differences
# batch:         sequencing batch technical variation
# User-supplied via sample_metadata DataFrame

# Expression PCs (PEER-equivalent confounders)
# n_expression_pcs=3 default; set 0 to disable
```

In the Nextflow pipeline, pass a metadata CSV with any additional columns:
```yaml
# config/default_params.yaml
sample_metadata: 'config/samplesheet.csv'
fixed_covariates: ['sex', 'batch', 'age', 'body_weight_g', 'tissue_weight_mg']
n_expression_pcs: 3
```

The model equation including user covariates:

$$y = H\alpha + C_{\text{default}}\beta_1 + C_{\text{user}}\beta_2 + Zu + \varepsilon$$

where $C_{\text{user}}$ is the user-supplied covariate matrix (numeric, no NaN).
Both $\beta_1$ and $\beta_2$ are included as fixed effects in the NB-GLMM score scan.


---

## 7. DEPENDENCIES

### Core (always installed)
```
python >= 3.13
numpy >= 1.26        # array ops, memmap
pandas >= 2.2        # result tables, covariate matrices
scipy >= 1.13        # stats, sparse
h5py >= 3.11         # HDF5 genome cache
anndata >= 0.10      # AnnData input format
seaborn >= 0.13      # all plotting
matplotlib >= 3.9    # figure backend
pydantic >= 2.7      # config and parameter validation
click >= 8.1         # CLI entry points
```

### Optional
```
scanpy >= 1.10       # pip install mosaiqtl[sc]  — UMAP/dotplot viz only
rpy2 >= 3.5          # pip install mosaiqtl[r]   — R/qtl2 cross-validation only
cupy / torch         # pip install mosaiqtl[gpu] — GPU scan acceleration
nextflow             # HPC pipeline orchestration (external; see docs/HPC.md)
```

### Dev
```
pytest >= 8.2        pytest-cov >= 5.0    pytest-benchmark >= 4.0
hypothesis >= 6.100  black >= 24.4        flake8 >= 7.0
mypy >= 1.10         interrogate >= 1.7   bandit >= 1.7
memory-profiler      scalene >= 1.5
```

### R (validated port — not installed by default)
```
R >= 4.3.3   qtl2   qtl2convert   miqtl   edgeR   ggplot2
```

---

## 8. STREAMLIT DATA BROWSER (app/)

A public-facing Streamlit application for interactive navigation of CC eQTL results.
Separate from the analysis package — reads pre-computed outputs only, never re-runs scans.

### Scope
- Browse eGene tables per cell type (filter by LOD, FDR, chromosome, gene name)
- Visualise LOD profiles for any gene × cell-type combination
- Display per-founder haplotype effect plots at peak loci
- Show vQTL dispersion results alongside mean eQTL
- Founder color palette and strain ordering consistent with mosaiQTL viz/ conventions

### Architecture
```
app/
├── main.py              <- Streamlit entry point
├── pages/
│   ├── 01_Browse.py     <- eGene table with filters
│   ├── 02_Locus.py      <- LOD profile + founder effects for one gene
│   ├── 03_Dispersion.py <- vQTL browser
│   └── 04_About.py      <- dataset description + citation
├── data/                <- pre-computed result CSVs and HDF5 (.claudeignored)
│   ├── egenes_*.csv     <- eGene tables per cell type
│   ├── lod_*.h5         <- LOD profiles (read-only)
│   └── effects_*.csv    <- founder effects at peak loci
└── utils/
    └── plot.py          <- imports from mosaiqtl.viz (same palette + conventions)
```

### Rules
- app/ reads pre-computed files only — never imports qtl/, nb.py, or scan functions
- All colors imported from `mosaiqtl.genotype.founders` — never redefined in app/
- Streamlit deployment: separate conda env or Docker container; not part of `pip install mosaiqtl`
- Data files in app/data/ are .claudeignored — never auto-read by Claude Code

### Dependencies (app/ only — not in core mosaiqtl)
```
streamlit >= 1.35
plotly >= 5.20     # interactive LOD profiles
```


---

## 9. VALIDATION APPROACH

mosaiQTL uses a staged validation pipeline. All validation uses **simulated data with
known ground truth** — no real experimental data required for any validation phase.
This makes validation fully reproducible and decoupled from HPC data access.

---

### DATA: sparc simulated phenotypes (Phase 2 and Phase 3)

Phase 2 and Phase 4 validation both use phenotypes simulated by **sparc**
(Keele 2021, github.com/gkeele/sparcc) on the **sparcc_cache** genome cache
(github.com/gkeele/sparcc_cache), which contains real CC haplotype probabilities
for the CC lines. This is identical to the miQTL README.md example workflow,
enabling direct comparison against miQTL ground truth.

```r
# R — generate sparc phenotypes (run once; save output)
library(sparcc)
download.file('https://github.com/gkeele/sparcc_cache/archive/master.zip',
              destfile = '~/sparcc_cache.zip')
unzip('~/sparcc_cache.zip')
genomecache <- '~/sparcc_cache-master'

phenotypes <- sim.CC.data(
    genomecache     = genomecache,
    num.lines       = 40,          # 40 CC lines (matches miQTL example)
    num.sim         = 10,          # 10 simulated phenotypes
    num.replicates  = 2,           # 2 replicates per line (strain means used for scan)
    qtl.effect.size = 0.8,         # large effect — ensures detectable QTL signal
    seed            = 42
)
# sparc outputs: phenotypes$data (data.frame with SUBJECT.NAME + sim.y.1..10)
#                phenotypes$locus (planted QTL locus per simulation)
#                phenotypes$properties (effect sizes, allelic series)
```

The planted QTL locus (`phenotypes$locus`) provides ground truth for:
- Peak position recovery
- LOD score magnitude
- Per-founder effect direction

---

### Phase 2 (R vs Python, S1–S11) — sparc + miQTL ground truth

Script: `validation/validate_port.py`
R reference: `validation/r_reference/port_validation.R`
Data: sparc phenotypes on sparcc_cache genome cache

**Setup**: R script runs miQTL on sparc phenotypes → exports CSV reference outputs.
Python script runs mosaiQTL on the same sparcc_cache genome cache + phenotypes →
compares against R CSVs. LOD concordance is verifiable against known ground truth
(planted locus position and effect size).

| Stage | What is tested |
|---|---|
| S1 | HDF5 genome cache structure vs R HAPPY format (sparcc_cache) |
| S2 | HMM haplotype probabilities vs R qtl2 (sparcc_cache strains) |
| S3 | LOCO kinship matrices vs R |
| S4 | LMM LOD scores vs miQTL — sparc phenotypes; planted locus recovered |
| S5 | Cache collapse (CC strain compression) |
| S6 | Permutation thresholds — GEV fit |
| S7 | Multiple imputation scan |
| S8 | SNP scan (marginal vs diplotype model difference documented) |
| S9 | Mediation LOD scores |
| S10 | Positional confidence intervals — CI overlaps planted locus |
| S11 | Instability scan |

Pass criterion for S4: planted locus recovered as peak in ≥8/10 simulated phenotypes.

#### S4 diagnostic comparisons

Two secondary comparisons are generated by `validate_sim_p2.py` alongside the main LOD r² gate.
They do not gate validation but characterise the scan engine.

**fixef vs ranef allele effects** (`compare/s4_allele_effects_ranef_vs_fixef.csv`)

Two estimators of per-founder haplotype effects at the peak locus for sim.y.1:

| Estimator | Method | Behaviour |
|---|---|---|
| fixef | FWL pinv on GLS-rotated H — all J founders, sum-to-zero constraint | Maximum likelihood; no shrinkage; inflated when per-founder n is small |
| ranef | Kinship-BLUP: solve (H^T V^{-1} H + λI)^{-1} H^T V^{-1} y | Shrinkage toward zero; lower variance; biased for large effects |

With ~5 CC strains per founder, fixef has high variance and ranef can differ substantially.
For the primary scan (LOD) both methods yield identical LOD scores — only the reported β magnitudes
differ. fixef is the value stored in `ScanResult.betas`; ranef is available via `qtl/effects.py`.

**Allele probability scale note** (fixef betas vs miQTL allele effects):
Python's cache stores haplotype probabilities normalised to sum 1 per strain (values 0–1).
R's `scan.h2lmm` reads raw HAPPY dosage values (0–2 for diploids; 2 = both alleles are founder f).
As a result Python fixef betas are exactly 2× miQTL's reported allele effects (per-individual scale
vs per-allele scale). LOD and predicted phenotype values are identical regardless of convention;
the factor of 2 is purely in beta reporting. For heterozygous populations (DO) both conventions
give identical predicted phenotypes: `H[f] * beta_py = (2*H[f]) * (beta_py/2)`.

**full vs additive LOD** (`compare/s4_full_vs_additive_lod.csv`)

Compares the additive haplotype model (J=8 parameters per locus) against the full diplotype model
(J*(J+1)/2 = 36 parameters per locus, constructed as the outer-product P(fi/fj)=P(fi)*P(fj)):

| Model | Parameters per locus | Valid for |
|---|---|---|
| additive | 8 founder columns | CC inbred + DO outbred |
| full (diplotype) | 36 outer-product columns | CC inbred only (homozygous → outer product exact) |

LOD_full > LOD_additive signals non-additive (dominance) structure at that locus.
For a purely additive planted QTL the two models should be near-equal; large differences indicate
the additive model is under-fitting. In practice the full model has more df and is not always better
even when it fits better per-df (the LOD is reported as ML, not corrected for df).

---

### Phase 3 (NB model consistency, P1–P4) — sparc-derived single-cell simulation

Script: `validation/validate_phase4.py`
Data: simulated single-cell counts derived from sparc phenotypes (no real data needed)

**Strategy**: sparc provides strain-level continuous phenotypes with known QTL signal.
Phase 4 converts these into single-cell-adjacent NB count data by:

1. **Strain means → NB parameters**: for each sparc simulation, treat the strain mean
   phenotype as the log-mean of a NB distribution. Per-strain counts are drawn as:
   ```python
   # For each strain i, gene g:
   mu_i = exp(sparc_mean_i * effect_scale + baseline)
   counts_ig = rng.negative_binomial(1/phi, 1/(1 + phi*mu_i), size=n_cells_per_strain)
   # phi ~ 0.3 (typical skeletal muscle overdispersion from gastroc data)
   # n_cells_per_strain ~ Poisson(200) (realistic per-strain cell count)
   ```
2. **Planted signal preserved**: strains with high sparc phenotype value get higher
   NB mean → QTL signal is preserved in the NB count layer
3. **Known ground truth**: planted locus from sparc is recoverable if NB-GLMM scan
   is correctly implemented

```python
# Python — generate Phase 3 NB counts from sparc output
from validation.sim_sc import sparc_to_nb_counts

counts_cells, strain_ids, true_locus = sparc_to_nb_counts(
    phenotypes_h5    = 'validation/sparcc_data/phenotypes.h5',  # no rpy2
    sim_index        = 0,                  # which of the 10 simulations to use
    phi              = 0.3,                # NB dispersion
    n_cells_mean     = 200,               # avg cells per strain
    seed             = 42,
)
# counts_cells: (n_genes=1, n_cells) raw UMI counts
# strain_ids:   (n_cells,) strain per cell
# true_locus:   str — planted locus from sparc
```

A utility `validation/sim_sc.py` implements `sparc_to_nb_counts()`.

| Section | Gate | Ground truth check |
|---|---|---|
| P1 scan_nb | planted locus in top 5% by LOD | sparc true_locus recovered |
| P2 scan_vqtl | ≥1 result | dispersion heterogeneity present |
| P3 scan_epistasis | lod_additive > 0 | additive signal detectable |
| P4 scan_conditional | signal1_lod > 0 | primary signal recoverable |

Biological anchors (*Piezo1*, *Ccdc171*, *Pmis2*) remain as production checks
on real gastroc data (HPC only) but are not required for CI validation.

---

## 10. PERMUTATION THRESHOLD DESIGN — `run_threshold_scans`

### What the permutation threshold answers

For a genome-wide FWE (family-wise error) threshold at α = 0.05 we need:

> P(max LOD genome-wide > threshold | no QTL anywhere) = 0.05

We estimate this by simulating phenotypes from the null distribution and scanning each one
genome-wide. The 95th percentile of the resulting max-LOD distribution is the threshold.

---

### Parametric null phenotype generation

A plain label shuffle (`rng.permutation(y)`) is only valid when h² ≈ 0 (no kinship
structure, OLS null). When real LOCO kinship is used, the null phenotype must honour
the genetic correlation structure, or the LMM scan will see spurious signal.

The correct parametric null draws:

```
u ~ N(0, K·τ²)          # polygenic background
e ~ N(0, diag(1/w)·σ²)  # residual
ŷ = Xβ + u + e           # parametric null sample
y_perm[i] = y[rank(ŷ)]  # rank-permute original y by null order → preserves marginal distribution
```

`_draw_parametric_batch` (permutation.py) implements this using variance components
(σ²_g, σ²_e) estimated from the real phenotype at a fixed h².

---

### Why a single null draw (ref_chrom) for the genome-wide threshold

`run_threshold_scans` draws null phenotypes **once** from a single reference chromosome's
LOCO kinship (`ref_chrom`), then scans all chromosomes with that same null matrix.

**Why one draw, not one per chromosome:**
For a genome-wide max-LOD, each permutation i must ask: "if this one null phenotype were
observed, what is the highest LOD the genome-wide scan would produce?" That requires scanning
the **same** phenotype vector on all chromosomes and taking the max.

Drawing a fresh null for each chromosome instead would compute:
`max(scan_chr1(null₁), scan_chr2(null₂), ..., scan_chr20(null₂₀))`
where null₁ ... null₂₀ are independent draws. This is the maximum over 20 independent
experiments — it inflates the threshold relative to the correct genome-wide max over 20
scans of one experiment.

**Why ref_chrom's kinship is valid for all chromosomes:**
LOCO kinship matrices K_loco[c] differ only by the ~5% of markers on chromosome c.
All K_loco[c] are close approximations to the whole-genome K. The h² and variance
components (σ²_g, σ²_e) estimated from any one chromosome's LOCO kinship vary by <1%
across chromosomes for typical CC datasets. The null phenotype just needs the right
**polygenic correlation structure** — the small chromosome-to-chromosome differences
in K_loco[c] are negligible for this purpose.

**What peak_chrom is:**
`peak_chrom` (or `ref_chrom` in the API) is the chromosome carrying the highest LOD peak
for the phenotype — i.e. the chromosome on which the base genome scan was run before
calling `run_threshold_scans`. It is the natural reference because the variance components
(h², σ²_g, σ²_e) used for null draws come from that scan's null fit, and the null
phenotype's kinship structure should match what was estimated from the data.

If `ref_chrom=None`, `run_threshold_scans` falls back to the first chromosome in the scan
list whose kinship matrix passes the positive semi-definiteness check — acceptable when the
peak chromosome is unknown, since all LOCO kinship matrices are similar.

---

### Why fix_h2=True and the single vectorised path

`fix_h2=True` is a hard rule (see CLAUDE.md). Under the null (no QTL), h² reflects
the polygenic architecture — a nuisance parameter that is stable and should not change
across permutations. miQTL's `run.threshold.scans` extracts h² from the scan object's
null fit and holds it fixed; Python does the same.

With h² fixed, the GLS eigendecomposition (ev, evc) is identical for every permutation.
This means the entire null matrix (n × n_perms) can be passed through
`scan_chromosome_batch` as a single vectorised matrix multiply — one batch covers all
n_perms simultaneously rather than looping. This is the only execution path that exists
in practice (~100–500× faster than a per-permutation loop on large locus counts).

```
h² estimated once from phenotype → fixed for all permutations
null matrix drawn once (n × n_perms)
for each chromosome c:
    scan_chromosome_batch(null_mat, cache, c, x, ev_c, evc_c, h2_c)
    → (n_perms,) max-LOD per permutation on chromosome c
    max_lods = np.maximum(max_lods, chr_max)   # accumulate genome-wide max
```

`scan_chromosome_batch` also supports `return_peak_positions=True`, which returns
`(max_lods, peak_positions_mb)` — used by `run_positional_scans` to vectorise bootstrap
CI computation. Instead of N sequential `scan_chromosome` calls (one per bootstrap
replicate), `run_positional_scans` builds the full `(n_strains, n_boot)` matrix once and
calls `scan_chromosome_batch` once per chromosome, recovering both the peak LOD and its
genomic position for all replicates in a single pass.

A `fix_h2=False` slow path (re-estimate h² per permutation, loop over n_perms × n_chr)
exists in the code as a legacy fallback but is dead code: the `fix_h2=True` rule means it
is never reached. It is a candidate for removal.

---

### Multiple-imputation scan — `scan_chromosome_mi`

`scan_h2lmm_mi` implements Rubin (1987) multiple imputation: sample `n_imp` hard
diplotype assignments from the soft haplotype probability matrix, compute the
complete-data LOD for each, and return the mean. This integrates out uncertainty in
founder assignments at each locus, recovering power at loci with ambiguous probabilities.

**Old sequential path (removed):**
- For each of `n_imp` imputations: sample one-hot `H_imp` per chromosome, wrap in a new
  `GenomeCache`, call `scan_chromosome` — full FWL recomputed from scratch each time.
- Cost: `n_imp × n_chr` `scan_chromosome` calls + `n_imp × n_chr` GenomeCache allocations.

**New vectorised path (`scan_chromosome_mi`):**
- For each chromosome: generate all `n_imp` one-hot draws at once as an
  `(n_imp, n_loci, n_strains, J)` tensor; compute H_perp, G, G_pinv in a single batched
  einsum pass over the imputation axis in loci mini-batches; accumulate LOD sum.
- h² is fixed from the real (soft) cache — the `fix_h2=True` rule applies here too.
- Cost: 1 pass per chromosome; `n_imp × n_loci × J` work done in a single vectorised step.
- Null model (e0, rss0) computed once per chromosome from the real phenotype — identical
  across all imputations since only the genotype varies.

```
for each chromosome c:
    draw imp_h: (n_imp, n_loci, n_s, J) one-hot from probs
    for each loci batch b:
        H_t  = einsum('nk,ibnj->ibkj', eigvecs, imp_h[:,b]) * w_sqrt   # rotate + weight
        proj = einsum('pk,ibkj->ibpj', xt_pinv, H_t)                    # FWL projection
        H_perp = H_t - einsum('kp,ibpj->ibkj', xt, proj)
        G    = einsum('ibkj,ibkl->ibjl', H_perp, H_perp)                # (n_imp,b,J,J)
        c    = einsum('ibkj,k->ibj', H_perp, e0)                        # (n_imp,b,J)
        lod_sum[b] += ((n/2)*log10(rss0/(rss0 - c·G_pinv·c))).sum(axis=0)
mean_lod = lod_sum / n_imp
```

`scan_h2lmm_mi` then calls `scan_chromosome` once per chromosome on the soft probs
to populate the reference `p_value` and `betas` fields, and uses `mean_lod` as `.lod`.
Verified: with `n_imp=1` and the same random draw, `scan_chromosome_mi` matches
`scan_chromosome` on the equivalent one-hot cache to floating-point precision (≤ 1e-15).

---

### Kinship positive semi-definiteness (PSD) guard

Drawing `u ~ N(0, K·σ²_g)` via `multivariate_normal` requires K to be positive
semi-definite — all eigenvalues ≥ 0 — because the function internally computes the
Cholesky factorisation of K, which only exists when no eigenvalue is negative.

K_loco[c] is PSD by construction: it is a sum of outer products of haplotype probability
vectors, so all eigenvalues are non-negative in exact arithmetic. However, floating-point
accumulation across ~17,000 loci can push the smallest eigenvalue slightly negative
(e.g. −10⁻¹⁶), which is numerical noise rather than genuine rank-deficiency. A strict
zero tolerance would incorrectly reject valid kinship matrices.

The guard uses a scaled tolerance so that noise-level negativity passes and structural
singularity does not:

```python
min_eig = np.linalg.eigvalsh(K).min()
psd_tol = max_eig * n² * machine_eps   # O(n²) accounts for Cholesky float accumulation
if min_eig < -psd_tol:
    raise KinshipSingularError(...)
```

A minimum eigenvalue of −3×10⁻¹⁶ passes; −10⁻³ raises `KinshipSingularError` and the
caller skips that chromosome. Near-structural singularity in practice arises on
chromosomes with low founder diversity (few loci, locally fixed founder haplotypes, or
the X chromosome in sex-structured data).

---

### Per-chromosome null distributions and Storey q-values

`run_threshold_scans` always returns a second value: a `dict {chrom: np.ndarray (n_perms,)}`
containing the per-chromosome max-LOD null distributions. These are computed as a byproduct
of the genome-wide scan — every chromosome's per-permutation max-LOD is already available
before the genome-wide max is taken. Returning them adds zero extra computation.

**Why per-chromosome nulls matter for FDR**

The genome-wide FWER threshold answers: *what is the 95th percentile of the maximum LOD
across the entire genome under the null?* For eGene discovery at FDR rather than FWER,
we need per-gene p-values. A gene whose peak LOD is 4.5 on chromosome 8 should be compared
against the null LOD distribution for chromosome 8 specifically — not the genome-wide max,
which is dominated by the chromosome with the most loci.

Per-gene empirical p-value:

```python
p_g = (chr_null_lods[gene_chr] >= peak_lod_g).mean()
```

This calibrates each gene against the null for its own chromosome, avoiding the conservatism
of genome-wide FWER thresholding.

**Storey q-value (qtl/fdr.py)**

Given a vector of per-gene empirical p-values, `storey_qvalue(pvalues)` applies the Storey
(2002) q-value estimator:

1. Estimate π₀ (proportion of true null genes) from the p-value distribution shape using a
   natural cubic spline smoother over a λ grid (0.05 to 0.95).
2. Compute q-values as BH-adjusted p-values scaled by π₀:
   `q(i) = π₀ × m × p(i) / rank(p(i))`, with monotone enforcement.

Under complete null (no eGenes), π₀ → 1 and q ≈ BH q. When real signal is present, π₀ < 1
and q-values are less conservative than BH, recovering power lost by the BH assumption that
all m genes are null.

**eGene summary output** (`eqtl_summary.csv`)

| Column | Description |
|---|---|
| `gene` | gene name |
| `chr` | chromosome of peak locus |
| `peak_locus` | peak locus identifier |
| `peak_mb` | peak locus position (Mb) |
| `peak_lod` | peak LOD score (Python scan) |
| `p_permtest_chr` | per-gene empirical p-value vs chromosome-specific permutation null |
| `q_fdr_storey` | Storey (2002) q-value across all tested genes |
| `pi0` | estimated proportion of true null genes (π₀) |
| `perm_n` | number of permutations used |
| `chi2_haplotype` | LRT χ²(J−1) for 1-way haplotype effect — derived from scan LOD at zero extra compute |
| `p_haplotype` | p-value for 1-way haplotype effect (LMM-correct, kinship-corrected) |
| `lod_gxe_sex` | LOD_full − LOD_additive for haplotype × sex interaction (LMM-LRT) |
| `p_gxe_sex` | p-value for haplotype × sex interaction, χ²(J−1) (LMM-correct) |
| `in_r_egenes` | whether gene appears in R miQTL eGene list (when `r_data/egenes.csv` present) |

**LMM-based GxE characterization (`qtl_gxe_lrt`)**

At each eGene's peak locus, two statistics are computed using `qtl_gxe_lrt` from
`qtl/effects.py`. Both are kinship-corrected via the LOCO eigendecomposition.

**1-way haplotype effect** — derived from the existing scan LOD at zero extra compute:
```
chi2_haplotype = 2 ln(10) × LOD_additive
p_haplotype    = 1 − chi2.cdf(chi2_haplotype, df=J−1)
```
This is LMM-correct because the scan LOD itself comes from the LOCO LMM.

**Haplotype × sex interaction** — single-locus LMM-LRT equivalent to
`scan.h2lmm(intcovar=sex)` evaluated at one locus:
```
h_full    = [H, H ⊙ sex]          # 2J columns: additive + interaction
LOD_full  = (n/2) log₁₀(RSS₀ / RSS₁)   # FWL-rotated, kinship-whitened
lod_gxe_sex = LOD_full − LOD_additive
chi2_gxe    = 2 ln(10) × max(lod_gxe_sex, 0)
p_gxe_sex   = 1 − chi2.cdf(chi2_gxe, df=J−1)
```
Kinship correction: data rotated by LOCO eigenvectors before the LRT.
Covariates: dataset dummies + sex as main effects (sex appears as both covariate and interaction partner).

**Visualisation (`viz/effects.py::haplotype_sex_profile`)**

Per-founder mean ± SEM split by sex, with `lod_gxe_sex` and `p_gxe_sex` annotated as a
text box (LMM-corrected). The plot is descriptive (raw group means); significance comes
from `qtl_gxe_lrt`. Companion plot: sex-faceted PxG (`viz/haplotypes.py::pxg_sex_facet`).

A significant `p_gxe_sex` flags sex-specific eQTLs where the haplotype effect magnitude
or direction differs between males and females.

---

## 11. MPP GENERALIZATION — POPULATION BOUNDARY

The codebase is parametric on population structure. All population-specific
configuration lives in a `Population` instance (`models/config.py`); the scan
engine, kinship, and HMM are population-agnostic.

### Two categories of CC-specific code

**Type A — by design, belongs in Population config**

These are not bugs. They are parameters that differ across populations and must
be supplied by the user as part of their `Population` instance:

| Parameter | CC value | Encoded in |
|---|---|---|
| `founders` list and order | 8 CC founders | `Population.founders` |
| per-founder hex colors | `FOUNDER_COLORS` dict | `Population.colors` |
| `genotyping_error_prob` | 0.002 (GigaMUGA) | `Population.genotyping_error_prob` |
| `n_gen` | 1 (consensus map) | `Population.n_gen` |
| `inbred` | True | `Population.inbred` |
| `chromosome_names` | mm39 autosomes + chrX | `Population.chromosome_names` |

**Type B — analysis engine leakage (actual bugs)**

These are hardcoded assumptions in the engine that break non-CC use silently.
They are fixed as encountered; do not introduce new ones:

| Location | Old code | Fixed code | Status |
|---|---|---|---|
| `genotype/founders.py:cc_strain_palette` | `len(probs) != 8` | `len(FOUNDER_ORDER)` | ✅ fixed |
| `io/covariates.py:_LOCO_POPULATIONS` | `frozenset({'cc'})` | `population.loco_sufficient` | ✅ fixed |

### Population as API boundary

`Population` (frozen Pydantic model) is the single object that crosses the
boundary between user configuration and the engine. It is:
- **frozen**: hashable, safe as a dict key or function arg; no mutation after construction
- **validated**: `colors` must cover all `founders`; enforced at construction time
- **serialisable**: `.model_dump()` / `.model_validate()` for config files

`CCPopulation` in `genotype/founders.py` is the built-in default instance.
Users building DO, MAGIC, or custom MPP pipelines construct their own
`Population` and pass it to `ScanConfig.population` (as a name string) or
directly to engine functions that accept `Population`.

### founders.py structural refactor (post-Phase-2)

The current `founders.py` contains a mix of CC constants and utility functions
that other modules import directly. After Phase 2 validation passes, these will
be migrated so that viz and engine modules pull colors and founder order from
`cache.population` (the Population on the GenomeCache) rather than importing
from `founders.py` directly. This is tracked as a goal in GOALS.md.
Until then, `founders.py` remains the single source of truth for all CC hex
colors — never redefine a hex string anywhere else.

---

## 12. NAMING TAXONOMY

### Dimension names

Every dimension that appears in array shapes, function signatures, and CLI flags
follows a fixed vocabulary. Using the wrong name is a type error in the API contract.

| Name | Meaning | Typical range | Where it appears |
|---|---|---|---|
| `n_cells` | individual nuclei/cells before pseudobulk aggregation | 100–10,000 per strain per cell type | `counts_cells`, AnnData `.obs` |
| `n_samples` | individual mice — one row in `metadata.csv`; replicates included | 80–418 for gastroc | phenotype vectors, covariate matrix, `sample_ids` |
| `n_strains` | unique CC/DO genotype labels (CC001, etc.) — what GenomeCache indexes | 40–418 | `GenomeCache.sample_ids`, haplotype prob arrays, kinship matrices |
| `n_replicates` | mice per strain in a simulation or data generation step | 2–4 | `sim.CC.data(num.replicates=...)`, fixture builders only |
| `n_cell_types` | distinct cell-type labels after clustering | 5–30 | pipeline-level loops, `scan_multi_gene` outer wrapper |
| `n_founders` | haplotype founder strains (J) | 8 for CC/DO | prob array axis, `GenomeCache.n_founders`, `Population.n_founders` |
| `n_loci` | markers in genome cache (pseudomarkers excluded from scan) | ~74k total, ~854 per chr | `GenomeCache.n_loci(chrom)`, LOD array length |
| `n_perms` | permutation or bootstrap replicates in null distribution | 100–5000 | `generate_null_outcomes(n_perms=...)`, `run_threshold_scans` |
| `n_subsamples` | subsampling iterations in instability scan | 100 | `instability_lm_scan(n_subsamples=...)` |
| `n_imp` | multiple imputation draws | 11 | `scan_h2lmm_mi(n_imp=...)` |

**Key distinction — `n_samples` vs `n_strains`:**
A single CC001 mouse is one sample (`n_samples += 1`). CC001 as a genotype appears
once in the genome cache regardless of how many mice were sequenced (`n_strains += 1`).
All scan functions operate at the strain level via the Z-matrix
(`n × m` mapping samples to strains). The genome cache is always indexed by `n_strains`.

---

### Array shape conventions

All arrays follow these axis-order conventions. Deviating from them silently produces
wrong results because numpy broadcasting treats mismatched axes identically.

| Array | Shape | Convention |
|---|---|---|
| Haplotype probabilities | `(n_loci, n_strains, n_founders)` | loci first; founders last |
| Kinship matrix | `(n_strains, n_strains)` | strain × strain |
| Kinship (sample-expanded) | `(n_samples, n_samples)` | sample × sample via Z-matrix |
| Phenotype | `(n_samples,)` or `(n_strains,)` | flat; axis=0 always the individual |
| Covariate matrix | `(n_samples, p)` | samples × predictors |
| LOD array | `(n_loci,)` | matches locus order in `GenomeCache` |
| Betas (allele effects) | `(n_loci, n_founders)` | sum ≈ 0 per locus (J founders, not J-1) |
| Null outcome matrix | `(n_strains, n_perms)` | strains × permutation columns |

---

### `scan_window` defaults

| Function | Default | Rationale |
|---|---|---|
| `scan_h2lmm` | `'cis'` | discovery scan; one gene, test its own chromosome; `gene_chr` required |
| `scan_h2lmm_mi` | `'cis'` | MI is a refinement step after finding a gene; no reason to scan genome-wide |
| `scan_multi_gene` | `'genome'` | pipeline-level batch scan across all genes; chromosome scope resolved per call |
| `run_threshold_scans` | all chromosomes (`chrom=None`) | calibration needs the genome-wide max-LOD distribution for FWE |
| `instability_lm_scan` | all chromosomes | instability score covers the whole scan region |

`scan_window='cis'` requires `gene_chr: str`.
`scan_window='local'` requires both `gene_chr` and `gene_pos_mb`.
`scan_window='genome'` requires neither — scans all chromosomes in `kinship_decomp`.

---

### CLI flag naming (target)

When mosaiQTL CLI flags are defined, they must use the same vocabulary:

```
--n-perms       permutation count (not --n-samples, not --permutations)
--n-imp         multiple imputation draws
--scan-window   cis | genome | local
--gene-chr      chromosome of the gene (required for cis/local)
--gene-pos-mb   TSS position in Mb (required for local)
--window-mb     half-window for local scan (default 1.0)
```

Never use `--samples` to mean permutations, and never use `--perms` to mean
biological observations. The biological sample count is not a CLI parameter —
it is inferred from the input data.

---

## 13. KNOWN OPEN ISSUES

| Issue | Location | Priority |
|---|---|---|
| S2 needs `haplotype_vals_r.csv` rerun | port_validation.R | MED |
| S11 chr-type bug fixed but not rerun | validate_port.py | MED |
| S8 marginal vs diplotype model difference documented, not resolved | snp_scan.py | LOW |
| Storey q-value implemented in qtl/fdr.py; integrated in validate_gastroc.py eGene summary; pipeline.py BH-FDR placeholder still needs replacing (SWE-09) | pipeline.py | MED |
| No baseline benchmarks measured yet | benchmarks/ | MED |
| `fix_h2=False` slow path in `run_threshold_scans` not reachable from any current caller — reference implementation only | permutation.py | LOW |
| `instability_lm_scan` previously subsampled at sample level (out-of-bounds when n_samples > n_strains) — fixed: now pre-reduces to strain means via Z-matrix before subsampling | scan.py | ✅ fixed |
| `run_positional_scans` previously called `scan_chromosome` sequentially per bootstrap replicate — fixed: vectorised via `scan_chromosome_batch(return_peak_positions=True)` | permutation.py | ✅ fixed |
| `mediation_scan_qr` recomputed base scan for every (outcome, mediator) pair — fixed: `mediation_scan_qr_batch` processes all mediators of one outcome in a single chromosome pass using Woodbury rank-1 (h2 fixed from base model; empirical drift ≤ 0.03, r(exact,fixed) ≥ 0.9999). validate_gastroc.py S9 updated to call batch variant. | mediation.py | ✅ fixed |
| `scan_h2lmm_mi` looped n_imp × n_chr `scan_chromosome` calls + GenomeCache allocations per imputation — fixed: `scan_chromosome_mi` generates all draws at once and vectorises over the imputation axis in a single batched einsum pass | lmm.py + scan.py | ✅ fixed |
| S11 gastroc comparison: R exports `neg_log10_p` (median −log10 F-test p across subsamples); Python exports `lod` (mean bootstrap LOD). Incompatible on linear scale — fixed: comparison switched to Spearman rank correlation (monotone with instability signal) | validate_gastroc.py | ✅ fixed |
| S9 gastroc comparison: miQTL's `mediation.scan.qr()` may export `LOD` (uppercase) and `pos.Mb` rather than `lod`/`pos_mb` — fixed: column normalization applied after `pd.read_csv` before `_pos_merge` | validate_gastroc.py | ✅ fixed |
| gastroc benchmark S4–S11 not yet run — pending HPC run; rsync r_data/ then run validate_gastroc.py --compare | gastroc_data/ | MED |
| `estimate_h2_null` emitted `RuntimeWarning: invalid value encountered in scalar subtract` from Brent's parabolic interpolation on every permutation. Root cause: degenerate permuted phenotype (constant after eigenrotation) caused `wrss ≈ 0` → `sigma2 → 0` → `log(sigma2) = -inf` → `neg_loglik` returned `-inf` → `−inf − (−inf) = NaN` in Brent's step arithmetic. Also removed dead `w <= 0` guard (w = delta*eigvals + 1 ≥ 1 always) and redundant `isfinite` check. Fixed: `if wrss <= 0: return 1e10` before the log call; large finite sentinel preserves Brent's arithmetic. Normal permutations unaffected. | lmm.py::estimate_h2_null | ✅ fixed |
| `scan_h2lmm` and `scan_h2lmm_mi` applied RINT **after** Z-matrix reduction (RINT of strain means), diverging from miQTL's canonical behaviour. miQTL's `scan.h2lmm` applies RINT to all n sample-level values in `pheno_df` first, then reduces to strain means internally (`mean(RINT(samples))` per strain). The wrong order inflated LOD for sparse genes: a gene with one high-expressing replicate in strain S gets `RINT(mean) ≈ Φ⁻¹(rank_S / m)` (extreme), whereas the correct order gives `mean(RINT) = RINT(high_val) / r_S` (diluted by within-strain averaging). For `Insl3` in the gastroc benchmark this produced LOD 64.75 (wrong) vs the expected ~3.5. Fixed: RINT moved before Z-matrix reduction in both functions. Averaging RINT values preserves normality (CLT); the reduced variance ~1/r is absorbed by σ² estimation in the LMM. | scan.py::scan_h2lmm, scan_h2lmm_mi | ✅ fixed |
| `scan_h2lmm` reduced replicated data to strain means before scanning, implementing an m-dimensional model (m×m kinship, LOD = (m/2)·log₁₀(RSS₀/RSS₁)). miQTL's canonical replicated design uses `pheno.id='sample_id'`, which expands kinship to K_n = Z K_m Zᵀ (n×n) and scans in n-dimensional space — giving LODs on the n-sample scale. The strain-mean shortcut is statistically valid only for n_reps=1 (Phase 1/2 simulated data) but underestimates power for replicated data: within-strain variance is absorbed by the residual σ²_e I_n rather than averaged away. Fixed: `scan_h2lmm` now retains y and x at sample level; per-chromosome loop reconstructs K_m from its eigendecomp, expands to K_n = Z K_m Zᵀ, and eigendecomposes K_n (O(n³), n≤416, negligible). `scan_chromosome` already accepted `z_mat` for H expansion; the two-step expand-then-rotate (`Z H` then `U'(ZH)`) is fused to `(U'Z)H` via precomputed z_rot = eigvecs.T @ z_mat (n×m), reducing per-call cost from O(n_loci·n²·J) to O(n²·m + n_loci·n·m·J) — approximately 1.5–2× the m-dimensional cost rather than 75×. R's S5 permutation was also fixed: the miQTL helper functions (`generate.simple.permutation.index.matrix`, `run.simple.permutation.threshold.scans`) crash when `fit0$K` is the n×n expanded matrix; replaced with a direct loop over `scan.h2lmm` calls at sample level (`pheno.id='sample_id'`, `formula = gene ~ sex + dataset`), which is exactly what the helpers do internally — bypassing the cached-K bug. | scan.py, lmm.py, gastroc_miqtl_scan.R | ✅ fixed |

---
*Last updated: see git log. Canonical source: github.com/gfilimban/mosaiqtl/docs/ARCHITECTURE.md*

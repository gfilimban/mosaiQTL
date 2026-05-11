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
│   ├── lmm.py              # Gaussian LMM engine (qtl2/miqtl-port)
│   ├── scan.py             # Z-matrix n=418, MI, fixef, instability
│   ├── permutation.py      # fix_h2=True, GEV thresholds
│   ├── effects.py          # founder haplotype effect estimation
│   ├── mediation.py        # QR-based mediation
│   ├── epistasis.py        # pairwise GxG, conditional scanning
│   ├── vqtl.py             # dispersion QTL, VQTLResult
│   ├── snp_scan.py         # marginal SNP fine-mapping
│   └── ase.py              # allele-specific expression (scaffold)
├── models/                 # shared dataclasses and Pydantic schemas
│   ├── records.py          # ScanResult, VQTLResult, QTLResult, FounderEffect
│   └── config.py           # ScanConfig, PipelineConfig (Pydantic v2)
├── io/                     # all file I/O and covariate construction
│   ├── CLAUDE.md
│   ├── covariates.py       # build_covariate_matrix(); user covariate input
│   ├── anndata_io.py       # AnnData read helpers (backed=True always)
│   ├── qtl2_io.py          # R/qtl2 format I/O (rpy2, optional extra)
│   └── results_io.py       # result table serialisation with provenance header
├── viz/                    # all visualisation
│   ├── CLAUDE.md
│   ├── genome.py           # LOD profiles, Manhattan plots (Phase 3 + production)
│   ├── effects.py          # founder effect bar plots (Phase 3 biological)
│   ├── haplotypes.py       # haplotype heatmaps, epistasis grids (Phase 3)
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
- `viz/genome.py` + `viz/effects.py` + `viz/haplotypes.py` — Phase 3 biological plots (P1–P4) and production figures

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
      - QR-based mediation analysis
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

Phase 2 and Phase 3 validation both use phenotypes simulated by **sparc**
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

---

### Phase 3 (NB model consistency, P1–P4) — sparc-derived single-cell simulation

Script: `validation/validate_phase3.py`
Data: simulated single-cell counts derived from sparc phenotypes (no real data needed)

**Strategy**: sparc provides strain-level continuous phenotypes with known QTL signal.
Phase 3 converts these into single-cell-adjacent NB count data by:

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

## 10. PERMUTATION DESIGN GAP — KINSHIP SAMPLING

miQTL's `generate.sample.outcomes.matrix` with `method='permutation'` is **not** a
simple phenotype shuffle. It draws from the kinship-informed null distribution:

```r
# sampling.processes.R lines 110–137
u ~ N(0, K·τ²)          # kinship random effect — requires chol(K·τ²)
e ~ N(0, diag(1/w)·σ²)  # residual
ŷ = Xβ + u + e           # parametric null sample
y_perm[i] = y[rank(ŷ)]  # rank-permute original y using ŷ's order
```

Python's `generate_null_outcomes` (permutation.py) does a plain shuffle:
```python
rng.permutation(y)        # no kinship sampling; τ² not used
```

**Consequence:** Python and R permutation semantics diverge whenever τ² > 0.
Phase 1 validation used identity kinship (τ²=0, OLS) specifically to avoid this gap.
Phase 2 validation (real LOCO kinship) will expose the divergence in S6 threshold gate.

**Design requirement for Python `generate_null_outcomes`:**
To match miQTL, the parametric permutation path must:
1. Draw `u ~ N(0, K_m·τ²)` using `np.random.multivariate_normal` (calls Cholesky on K_m)
2. Draw `e ~ N(0, diag(1/w)·σ²)` — scalar normal if w uniform
3. Rank-permute original y using ŷ's order

**Cholesky requirement on LOCO kinship:**
`np.random.multivariate_normal(zeros(m), K_m·τ²)` requires K_m to be PSD.
Near-zero eigenvalues occur on chromosomes with low founder diversity (few loci,
locally fixed founders, or chrX in sex-structured data). This is the same failure
mode as `mnormt::rmnorm` in R, patched in `gastroc_miqtl_scan.R` S5 with a
pre-check `min(eigvalsh(K_loco[ch])) < -1e-8` before calling `generate.sample.outcomes.matrix`.
Threshold is negative to allow floating-point noise around zero (-3e-16 is valid PSD).

**Required guard in Python (permutation.py):**
```python
def _draw_kinship_re(K: np.ndarray, tau2: float,
                     rng: np.random.Generator) -> np.ndarray:
    min_eig = np.linalg.eigvalsh(K).min()
    if min_eig < -1e-8:
        raise KinshipSingularError(f'LOCO kinship not PSD (min_eig={min_eig:.2e})')
    return rng.multivariate_normal(np.zeros(len(K)), K * tau2)
```

Caller skips the chromosome and logs a warning (mirrors R behaviour).

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

## 12. KNOWN OPEN ISSUES (when validation was based on actual data)

| Issue | Location | Priority |
|---|---|---|
| Permutation gap: Python uses plain shuffle; miQTL uses kinship-informed rank-permutation | permutation.py | HIGH |
| Cholesky guard needed: `_draw_kinship_re` must check PSD before sampling | permutation.py | HIGH |
| S6 GEV instability — `mean_delta=0.619 > 0.5` (partly caused by permutation gap above) | permutation.py | HIGH |
| S2 needs `haplotype_vals_r.csv` rerun | port_validation.R | MED |
| S11 chr-type bug fixed but not rerun | validate_port.py | MED |
| S8 marginal vs diplotype model difference documented, not resolved | snp_scan.py | LOW |
| Storey q-value not yet implemented (BH-FDR anti-conservative) | pipeline.py | HIGH |
| No baseline benchmarks measured yet | benchmarks/ | MED |

---
*Last updated: see git log. Canonical source: github.com/gfilimban/mosaiqtl/docs/ARCHITECTURE.md*

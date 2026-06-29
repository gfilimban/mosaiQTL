# mosaiQTL — Architecture, Scope & Intention
# Human-readable package overview
# Read this file before touching any code.
# Companion to: CLAUDE.md (Claude Code instructions) | docs/REFERENCES.md (bibliography)

---

## 1. WHAT IS mosaiQTL?

mosaiQTL is a python package providing **multiomic, haplotype-resolved quantitative trait locus (QTL) mapping
framework** for multiparent populations (MPPs). mosaiQTL is designed as a generalizable, multi-model hub unified by a shared genotype probability backbone. It maps genetic variants controlling gene
expression, chromatin accessibility, splicing, and RNA modifications at **cell-type
resolution** from single-nucleus sequencing data from $n$ replicate observations based on scan configuration space built using user input.


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
| RNA-seq/DNA-seq | Gene Expression                  | cis-eQTL   | Transcriptomics    | nb.py (primary NB-GLMM) | 🟡 implemented    |
| RNA-seq/DNA-seq | Distal Regulation                | trans-eQTL | Transcriptomics    | trans_scan.py           | 🔴 planned        |
| RNA-seq/DNA-seq | Dispersion Variance              | vQTL       | Transcriptomics    | vqtl.py                 | 🟡 implemented    |
| RNA-seq/DNA-seq | Epistasis / conditional QTL      |            | Transcriptomics    | epistasis.py            | 🟡 implemented    |
| RNA-seq/DNA-seq | Within-transcriptomics mediation |            | Transcriptomics    | mediation.py            | 🟡 implemented    |
| multi-omics     | Cross-omics mediation (e.g. ATAC → expression) | | Multi-omics | mediation.py       | 🔴 planned        |
| RNA-seq/DNA-seq | Allele-specific expression (ASE) |            | Transcriptomics    | ase.py                  | 🟡 scaffold       |
| LR-RNA-seq      | Splicing                         | sQTL       | Transcriptomics    | sqtl.py                 | 🔴 planned        |
| RNA-seq/DNA-seq | SNP fine-mapping                 |            | Genomics           | snp_scan.py             | 🟡 marginal model |
| ATAC-seq        | Chromatin accessibility          | caQTL      | Genomics           | atac.py                 | 🔴 planned        |
| dRNA-seq        | RNA modification                 | rmodQTL    | Epigenomics        | mod.py                  | 🔴 planned        |
| Fiber-seq       | DNA modification                 | dmodQTL    | Epigenomics        | mod.py                  | 🔴 planned        |
|                 | Permutation-based thresholds     |            | All modalities     | permutation.py          | 🟢 P2-S6 passing  |
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
│   ├── lmm.py              # Gaussian LMM engine (qtl2/miqtl-port); scan_chromosome_batch with return_peak_positions; scan_chromosome_mi vectorised over imputations; precompute_chromosome + scan_chromosome_fast for per-gene multi-gene loops
│   ├── scan.py             # Z-matrix n=418, MI (vectorised), fixef, instability; scan_multi_gene_mi + instability_scan_multi_gene (streaming multi-gene wrappers)
│   ├── permutation.py      # fix_h2=True, GEV thresholds; run_positional_scans vectorised via scan_chromosome_batch; ci_scan_multi_gene (streaming CI wrapper)
│   ├── effects.py          # founder haplotype effect estimation
│   ├── mediation.py        # QR-based mediation; mediation_scan_qr_batch vectorises all mediators per outcome; mediation_scan_multi_outcome (streaming multi-outcome wrapper)
│   ├── epistasis.py        # pairwise GxG, conditional scanning
│   ├── vqtl.py             # dispersion QTL, VQTLResult
│   ├── snp_scan.py         # marginal SNP fine-mapping; snp_scan_multi_gene (streaming multi-gene wrapper)
│   └── ase.py              # allele-specific expression (scaffold)
├── models/                 # shared dataclasses and Pydantic schemas
│   ├── records.py          # ScanResult, VQTLResult, QTLResult, FounderEffect
│   └── config.py           # Population (frozen, MPP boundary), ScanConfig, PipelineConfig
├── io/                     # all file I/O and covariate construction
│   ├── CLAUDE.md
│   ├── covariates.py       # build_covariate_matrix(); encode_sex()
│   ├── anndata_io.py       # AnnData read helpers — backed=True, adata.raw.X  (planned)
│   ├── qtl2_io.py          # R/qtl2 format I/O (rpy2, optional extra)            (planned)
│   ├── results_io.py       # result serialisation with provenance header          (planned)
│   └── offsets.py          # compute_offsets() — from_libsize / from_file / none (planned)
├── viz/                    # all visualisation
│   ├── CLAUDE.md
│   ├── genome.py           # lod_profile, Manhattan, permutation distribution, CI strip
│   ├── effects.py          # coef_profile (per-founder β traces along chr), effect_bars, effect_caterpillar
│   ├── haplotypes.py       # prob_heatmap, kinship heatmap, epistasis grid
│   ├── validation.py       # Phase 2 QC plots only (S1–S11, not shipped)
│   └── celltypes.py        # cell-type dotplots, per-gene summaries               (planned)
├── pipeline.py             # discover, characterize, resolve
├── cli.py                  # Click CLI: fit-null / scan / permute / effects / report (planned)
└── utils.py                # rint(), build_z_matrix(), align_samples()
```

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
The canonical R haplotype QTL implementation for CC/DO mice is correct but Gaussian-only — raw UMI counts require log-normalisation before analysis, inflating false positives for lowly-expressed genes. mosaiQTL ports the validated scan engine to Python and extends it with:
1. A **two-stage NB-GLMM model** that operates on raw counts without ad-hoc normalization
2. **Cell-type resolution** via per-cell-type pseudobulk inputs (aggregated upstream)
3. **Novel extensions**: vQTL, epistasis, mediation, ASE
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

## 5b. LMM MODEL AND LOD SCORE

### Model equations

At each locus $\ell$, two nested models are compared:

**Null (H₀ — no QTL at locus ℓ):**

$$y = X\beta + Zu + \varepsilon, \quad u \sim \mathcal{N}(0,\, \tau^2 K^{(-c)}), \quad \varepsilon \sim \mathcal{N}(0,\, \sigma^2 I)$$

**Alternative (H₁ — QTL present at locus ℓ):**

$$y = H_\ell\,\alpha + X\beta + Zu + \varepsilon$$

| Symbol | Dimension | Meaning |
|---|---|---|
| $y$ | $n \times 1$ | RINT-transformed phenotype (mean 0, SD 1 by construction) |
| $H_\ell$ | $n \times J$ | Founder haplotype probabilities at locus $\ell$ |
| $\alpha$ | $J \times 1$ | Per-founder allele effects (target of inference) |
| $X$ | $n \times p$ | Fixed covariate matrix (intercept, sex, batch, …) |
| $Z$ | $n \times m$ | Sample-to-strain design matrix |
| $u$ | $m \times 1$ | Polygenic random effects |
| $K^{(-c)}$ | $m \times m$ | LOCO kinship for chromosome $c$ |
| $\tau^2, \sigma^2$ | scalars | Genetic and residual variance components |

The only structural difference between H₀ and H₁ is the presence of $H_\ell\alpha$.

---

### Kinship rotation (eigendecomposition trick)

The covariance $V = \sigma^2 I + \tau^2 Z K^{(-c)} Z^T$ is dense and non-diagonal.
Eigendecomposing $K^{(-c)} = U \Lambda U^T$ and rotating data into eigenvector space
converts the LMM to weighted least squares:

$$\tilde{y} = U^T y, \quad \tilde{H} = U^T H_\ell, \quad \tilde{X} = U^T X$$

Weights: $w_i = \bigl(\sigma^2 + \tau^2\lambda_i\bigr)^{-1}$, where $\lambda_i$ are eigenvalues of $K^{(-c)}$.
All matrix operations are then $O(n^2 J)$ per locus rather than $O(n^3)$.

**Precomputation savings:** $U^T Z H$ (the rotated haplotype tensor) depends only on the
genome cache and kinship eigenvectors — not on $y$. `precompute_chromosome` computes
it once per chromosome; `scan_chromosome` reuses it for every gene in the loop,
reducing per-gene cost from $O(L \cdot n^2 \cdot J)$ to $O(L \cdot n \cdot J)$.

---

### Multi-gene scan amortization

**Per-gene overhead when the gene loop sits outside the scan package.** Each call accepts one phenotype vector:

```r
for (gene in genes) result[[gene]] <- scan_one_gene(pheno[, gene], genomecache, ...)
```

Every call re-executes the full per-chromosome setup: read one locus matrix per locus from disk,
apply the kinship rotation to it ($M \cdot H_\ell$, $O(n^2 \times J)$ per locus),
then project and score. This rotation depends only on the genome cache and kinship eigenvectors —
**not on $y$** — yet it is redone per locus per gene call. Over $L$ loci the total cost is
$O(L \cdot n^2 \cdot J)$ per gene.
For $G=2{,}000$ genes on a chromosome this repeats $\approx 4 \times 10^{10}$ FLOPs of
kinship rotation 2,000 times.

**mosaiQTL's fix.** The gene loop lives inside the package. `precompute_chromosome` runs once
per chromosome and caches chromosome-constant quantities:

```python
chr_data = precompute_chromosome(genomecache, chrom, eigvals, eigvecs, z_mat)
# H_t_raw : (L, n, J)   rotated tensor  (U'Z) H  — fused einsum, no explicit K_n
# xt      : (p, n)       rotated covariates U' X
# xt_pinv : (n, p)       pseudoinverse of xt
```

`scan_chromosome(y, chr_data, ...)` then runs only the $O(n)$ weight scaling
$w_i = (h^2\lambda_i + (1-h^2))^{-1}$, the $O(L \cdot n \cdot J)$ FWL projection
$H_\perp = H_t\sqrt{w} - X_t(X_t^T H_t\sqrt{w})$, and the $O(L \cdot n \cdot J)$ RSS
ratio per gene. No $O(n^2)$ work reappears per gene.

| Term | miQTL (per gene) | mosaiQTL (per gene) |
|---|---|---|
| Kinship rotation of H, $O(L n^2 J)$ | ✗ repeated | ✓ precomputed once |
| FWL + RSS $O(L n J)$ | ✗ repeated | ✗ repeated (unavoidable) |

At $G=2{,}000$, $L=3{,}700$, $n=416$, $J=8$: theoretical ~2,000× reduction in dominant cost.
Measured speedup is 20× on the gastroc benchmark — the gap closes due to I/O, BLAS threading,
and per-gene Python overhead vs compiled BLAS. The speedup is an
**API boundary decision**: moving the gene loop inside the package
exposes the chromosome as the natural amortization unit.

---

### LOD score

After rotation, both models produce a weighted residual sum of squares (RSS):

$$\text{LOD}_\ell = \frac{n}{2} \log_{10}\!\left(\frac{\text{RSS}_0}{\text{RSS}_1}\right)$$

- **RSS₀**: variance in $\tilde{y}$ unexplained by covariates + kinship alone (H₀ fit)
- **RSS₁**: variance unexplained after adding $\tilde{H}_\ell\alpha$ (H₁ fit)
- **RSS₀ − RSS₁**: variance *explained* by the haplotype at locus $\ell$
- Denominator is $n$ (ML convention), not $n-p$ (REML) — matches miQTL/qtl2

| LOD | Likelihood interpretation |
|---|---|
| 1 | H₁ is 10× more likely than H₀ |
| 3 | H₁ is 1,000× more likely (approximate genome-wide significance) |
| 0 | no improvement — haplotype adds nothing beyond polygenic background |

---

### What the outputs tell you

| Output | Source | Meaning |
|---|---|---|
| $\hat{\alpha}$ (`betas`) | H₁ fit | Per-founder effect in RINT units — which haplotype raises/lowers expression and by how many SDs |
| $\hat{h}^2 = \tau^2/(\tau^2+\sigma^2)$ | Null fit | Fraction of phenotypic variance explained by polygenic background |
| LOD | RSS₀/RSS₁ | Evidence that haplotype at locus $\ell$ explains expression *beyond* background relatedness |

**RINT baseline:** $y$ is standard normal (mean = 0, SD = 1), so $\hat{\alpha}_f = -0.5$ means strains
carrying founder $f$'s haplotype express ~0.5 SDs *below* the population mean at that locus
(roughly the 31st percentile). To report biological effect sizes, re-run `get_allele_effects`
on the un-RINTed log-normalized phenotype — betas will then be in log₂ CPM units.

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
      - haplotype probs shape (n_loci, n_strains, n_founders)
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
      - Gaussian eigendecomp LMM for Phase 2 R comparison
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
      - mediation_scan_multi_outcome: streams one outcome at a time; del + gc.collect() per gene; output_path writes CSV incrementally — peak RAM bounded to one gene regardless of dataset size.
      - pairwise epistasis scan
      - dispersion QTL (vQTL)
      - conditional multi-signal scan

  → Streaming multi-gene wrappers (all scan types)
      Each wrapper owns the gene loop, writes results incrementally via output_path: Path | None,
      and del + gc.collect() after each gene. Callers get constant-memory behaviour automatically.
      scan.py:       scan_multi_gene_mi        — MI scan across a gene set
      scan.py:       instability_scan_multi_gene — instability scan across a gene set
      permutation.py: ci_scan_multi_gene       — bootstrap CI across a gene set (accepts peak_mb_map)
      snp_scan.py:   snp_scan_multi_gene       — SNP fine-mapping across a gene set
      mediation.py:  mediation_scan_multi_outcome — mediation across all (outcome, mediator) pairs

Final outputs:
  - eGene tables per cell type (gene_id, chr, pos_mb, LOD, FDR, founder effects)
  - LOD profile CSVs
  - QC plots (PNG, dpi=300, font Carlito)
  - HTML Nextflow run report
```

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

mosaiQTL uses a staged validation pipeline. Phases 1, 2, 3, and 4 use simulated data with
known ground truth, making them fully reproducible and decoupled from HPC data access.
The gastroc track (below) additionally validates against real CC pseudobulk data.

---

### DATA: sparc simulated phenotypes (Phase 2 and Phase 3)

Phase 2 and Phase 4 validation both use phenotypes simulated by **sparc**
(Keele 2021, github.com/gkeele/sparcc) on the **sparcc_cache** genome cache
(github.com/gkeele/sparcc_cache), which contains real CC haplotype probabilities
for the CC lines. This is identical to the miQTL README example workflow,
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

### Phase 2 (R vs Python, S1–S11) — sparc simulation

Script: `validation/validate_phase2.py`
R reference: `validation/r_reference/phase2_validation.R`
Data: sparc phenotypes on sparcc_cache genome cache

**Setup**: R reference script runs miQTL on sparc phenotypes → exports CSV reference outputs.
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

Two secondary comparisons are generated by `validate_phase2.py` alongside the main LOD r² gate.
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

### Phase 4 (NB model consistency, P1–P4) — sparc-derived single-cell simulation

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

### Gastroc real-data concordance — mosaiQTL vs miQTL (2026-06-21)

**Dataset**: gastroc pseudobulk, 1,215 cis-eGene candidates on chr19 (all genes with a
cis locus on chr19 scanned by miQTL on real CC data, n ≈ 416 samples, 33 CC strains + founders + F1s).

**Prerequisite**: miQTL K-ordering bug fixed in `gastroc_miqtl_scan.R` — `pheno_df` sorted
by genotype before any scan call so `make.processed.data` first-appearance order = alphabetical
= `model.matrix` Z column order = K_mpd row order. Without the fix the scrambled K_expanded
produced LOD inflated by ~5× (max eigenvalue 60.31 scrambled vs 72.92 correct on chr8).

**Result** (Python `validate_gastroc.py` S4 peak LOD vs R `gastroc_miqtl_scan.R` chr19 output):

| Metric | Value | Notes |
|---|---|---|
| Genes compared | 1,215 | inner join on gene name |
| **LOD r²** | **0.999880** | well above Phase 2 threshold of ≥ 0.990 |
| MAE | 0.0023 LOD | sub-0.01 absolute error |
| Mean bias (Python − R) | −0.0002 LOD | no systematic offset |
| Top gene delta (Gm31356, LOD ≈ 9.19) | 0.0001 LOD | top genes agree to 4+ decimal places |

**Interpretation**: mosaiQTL and miQTL are concordant on real replicated CC pseudobulk data.
Residual discrepancies (max ~0.22 LOD at low-LOD genes) are consistent with floating-point
differences in the h² solver and carry no systematic bias. The K-ordering bug that previously
inflated LOD by ~5× is confirmed resolved.

**Scripts used**:
```bash
conda run -n bioinfo Rscript validation/r_reference/gastroc_miqtl_scan.R 1,2,3,4 19
conda run -n bioinfo python validation/validate_gastroc.py --sections 4
```

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
across permutations. h² is extracted from the real phenotype's null fit and held fixed
for all permutations.

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

### Per-chromosome null distributions and Storey q-values — implemented machinery, not yet in pipeline output

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

Per-gene empirical p-value (design intent):

```python
p_g = (chr_null_lods[gene_chr] >= peak_lod_g).mean()
```

This calibrates each gene against the null for its own chromosome, avoiding the conservatism
of genome-wide FWER thresholding. **Current limitation**: S5 derives its permutation null
from one representative gene's variance components shared across all genes — valid for a
single genome-wide FWER threshold, but invalid as a per-gene calibrated null (π₀ climbed
past 1.0 to 2.58 rather than plateauing). Per-gene empirical p-values and Storey q-values
are not included in `eqtl_summary.csv` until a per-gene-calibrated null is available.

**Storey q-value (`qtl/fdr.py::storey_qvalue`) — implemented but not yet wired to pipeline**

Given a vector of per-gene empirical p-values, `storey_qvalue(pvalues)` applies the Storey
(2002) q-value estimator:

1. Estimate π₀ (proportion of true null genes) from the p-value distribution shape using a
   natural cubic spline smoother over a λ grid (0.05 to 0.95).
2. Compute q-values as BH-adjusted p-values scaled by π₀:
   `q(i) = π₀ × m × p(i) / rank(p(i))`, with monotone enforcement.

Under complete null (no eGenes), π₀ → 1 and q ≈ BH q. When real signal is present, π₀ < 1
and q-values are less conservative than BH, recovering power lost by the BH assumption that
all m genes are null.

**eGene summary output** (`eqtl_summary.csv`) — canonical columns

| Column | Description |
|---|---|
| `gene` | gene name |
| `chr` | chromosome of peak locus |
| `tss_mb` | TSS position (Mb) |
| `peak_locus` | peak locus identifier |
| `peak_mb` | peak locus position (Mb) |
| `cis_dist_mb` | distance from TSS to peak locus (Mb) |
| `peak_lod` | peak LOD score |
| `h2` | null h² (fraction of variance explained by polygenic background) |
| `perm_n` | number of permutations used |
| `perm_chr` | chromosome whose LOCO kinship was used for permutation null |
| `chi2_haplotype` | LRT χ²(J−1) for haplotype effect — derived from scan LOD at zero extra compute |
| `p_haplotype` | p-value for haplotype effect, χ²(J−1) |
| `in_r_egenes` | whether gene appears in the miQTL eGene list (when `r_data/egenes.csv` present) |

**LMM-based GxE characterization (`qtl_gxe_lrt`) — planned, not yet in pipeline output**

`qtl_gxe_lrt` in `qtl/effects.py` computes the following statistics at a single locus.
Neither is currently included in `eqtl_summary.csv`; both are kinship-corrected via the LOCO eigendecomposition.

**1-way haplotype effect** — derived from the existing scan LOD at zero extra compute:
```
chi2_haplotype = 2 ln(10) × LOD_additive
p_haplotype    = 1 − chi2.cdf(chi2_haplotype, df=J−1)
```
This is LMM-correct because the scan LOD itself comes from the LOCO LMM.

**Haplotype × sex interaction** — single-locus LMM-LRT for haplotype × sex at a fixed locus:
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
| pipeline.py BH-FDR placeholder needs replacing with Storey q-value (SWE-09) | pipeline.py | MED |
| gastroc S4 full genome and S5–S11 pending HPC run; rsync r_data/ then `validate_gastroc.py --compare` | gastroc_data/ | MED |

---

## 14. GLOSSARY / ABBREVIATIONS

All terms, acronyms, and single-letter symbols used across the codebase, docs, and comments.
When a symbol appears in code it follows this definition exactly.

### Statistical / Mathematical terms

| Term | Full form | Meaning |
|---|---|---|
| LOD | Logarithm of Odds | `(n/2) × log₁₀(RSS₀/RSS₁)` — primary QTL evidence statistic; higher = stronger signal |
| LRT | Likelihood Ratio Test | Compares nested models via `2(ℓ₁ − ℓ₀)`; LOD = LRT / (2 ln 10) |
| RSS | Residual Sum of Squares | Unexplained variance under a model fit |
| RSS₀ / RSS₁ | null / alternative RSS | RSS under H₀ (no QTL) vs H₁ (QTL present) |
| h² | Heritability (narrow-sense) | `τ²/(τ²+σ²)` — fraction of phenotypic variance from polygenic background; estimated per gene per chromosome; fixed for permutations (`fix_h2=True`) |
| τ² | Genetic variance component | Variance of polygenic random effect `u`; scales kinship K |
| σ² | Residual variance component | Variance of i.i.d. residual ε |
| φ (phi) | NB overdispersion parameter | Variance = μ + φμ²; estimated per gene in Stage 1; held fixed in Stage 2 |
| π₀ | Proportion of true nulls | Storey (2002) estimate of fraction of genes with no real QTL; used to calibrate q-values |
| λ (lambda) | Eigenvalue | Eigenvalue of the LOCO kinship matrix K; used to compute per-sample weights `w_i` |
| w_i | Per-sample weight | `(σ² + τ²λ_i)⁻¹` after kinship rotation; converts LMM to weighted least squares |
| J | Number of founders | 8 for CC/DO; varies by population |
| m | Number of strains (genotypes) | Unique MPP lines in the dataset (e.g. CC strains, DO individuals, MAGIC lines); kinship and Z-matrix operate at this scale |
| n | Number of samples (observations) | Individual biological units (animals, plants, humans, …); ≥ m because strains/lines can have replicates |
| L / n_loci | Number of loci | Markers in the genome cache on a given chromosome (pseudomarkers excluded from scan) |
| p | Number of covariates | Columns of the fixed-effect covariate matrix X |
| FWL | Frisch-Waugh-Lovell | Theorem: partial regression of y on H after projecting out X yields same β as full model; mosaiQTL uses this to vectorise over loci without refitting X per locus |
| fixef | Fixed-effects estimator | Allele-effect estimator using FWL + `np.linalg.pinv(G)` on GLS-rotated data; no kinship shrinkage; maximum-likelihood point estimates; high variance when per-founder n is small. Stored in `ScanResult.betas`. See `qtl/effects.py::get_allele_effects(method='fixef')`. |
| ranef | Random-effects estimator | Allele-effect estimator using kinship-BLUP: solve `(H^T V⁻¹ H + λI)⁻¹ H^T V⁻¹ y`; shrinks toward zero; lower variance than fixef but biased for large effects. See `qtl/effects.py::get_allele_effects(method='ranef')`. |
| GLS | Generalised Least Squares | Least squares accounting for non-i.i.d. errors; reduces to WLS after kinship rotation |
| WLS | Weighted Least Squares | GLS after eigendecomposition; weights `w_i` absorb kinship structure |
| Woodbury | Woodbury matrix identity | `(A + UCV)⁻¹ = A⁻¹ − A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹`; used in mediation and NB-GLMM to invert the n×n covariance V via m×m ops (m = n_strains ≪ n); also the rank-1 update that shares `H_perp` and `G_inv` across all mediators on a chromosome |
| BLUP | Best Linear Unbiased Predictor | Optimal random-effect estimate `û = τ²KZ^T V⁻¹(y − Xβ̂)`; used in `ranef` allele-effect estimator |
| REML | Restricted Maximum Likelihood | Variance component estimation that accounts for fixed-effect df; mosaiQTL uses ML (denominator = n, not n−p) to match miQTL/qtl2 |
| ML | Maximum Likelihood | LOD denominator convention: `n/2 × log₁₀(RSS₀/RSS₁)` with n not n−p |
| PQL | Penalised Quasi-Likelihood | Iterative Gaussian approximation to NB-GLMM; each iteration updates working response z and weights W, then solves a Gaussian LMM |
| IRLS | Iteratively Reweighted Least Squares | Algorithmic core of PQL; re-weights and re-fits at each iteration |
| GEV | Generalised Extreme Value | Distribution fitted to the empirical max-LOD null to extrapolate permutation thresholds beyond the observed quantile range |
| FWER | Family-Wise Error Rate | P(≥1 false positive genome-wide); controlled at α=0.05 by permutation threshold |
| FDR | False Discovery Rate | Expected proportion of false positives among discoveries; controlled by BH or Storey q-value |
| BH | Benjamini-Hochberg | Step-up FDR procedure assuming π₀=1 (all genes null); conservative when real signal is present |
| q-value | Storey q-value | FDR analogue of p-value calibrated to the estimated π₀; implemented in `qtl/fdr.py::storey_qvalue` |
| RINT | Rank-based Inverse Normal Transform | Maps phenotype ranks to standard normal quantiles; applied before LMM scan to satisfy Gaussian residual assumption. Controlled by `use_rint` parameter in primary entry-point functions (`scan_h2lmm`, `scan_h2lmm_mi`, `imputed_snp_scan_h2lmm`); downstream functions (`mediation_scan_qr`, `scan_epistasis`) receive pre-transformed input and do not apply RINT internally. |
| CI | Confidence Interval | Genomic interval around the peak locus (LOD-drop or Bayes) — not to be confused with CI = continuous integration |
| CI (infra) | Continuous Integration | Automated test + lint pipeline run on every PR (`.github/workflows/ci.yml`) |
| lod_drop_ci | LOD-drop confidence interval | Genomic interval where LOD ≥ peak_LOD − 1.5; implemented in `qtl/scan.py::lod_drop_ci`; standard qtl2-compatible CI method |
| parametric null | Parametric null phenotype | Null draw from `N(0, τ²K + σ²I)` matching the real phenotype's LOCO variance components; used by `run_threshold_scans` instead of a plain label shuffle, which would be anti-conservative under non-zero kinship |
| strain means | Strain-mean phenotype | Sample-level observations collapsed to m strain means via Z-matrix: `ȳ = (Z^T Z)⁻¹ Z^T y`; required before calling `scan_h2lmm` or `scan_h2lmm_mi` when kinship is at strain level (m×m) |
| n_perms | Permutation count | Number of null phenotype draws used to build the max-LOD null distribution; minimum 100 for stable GEV fit, 1000 in production |
| n_imp | Imputation count | Number of hard diplotype draws averaged in the MI scan (`scan_h2lmm_mi`); default 11, matching miQTL |

### Model matrices and arrays

| Symbol | Shape | Meaning |
|---|---|---|
| H / H_ℓ | `(n, J)` | Founder haplotype probability matrix at locus ℓ; rows sum to 1 |
| H_t | `(n, J)` | Kinship-rotated H: `U^T Z H`; precomputed once per chromosome in `precompute_chromosome` |
| H_perp | `(n, J)` | FWL residual of H_t after projecting out rotated covariates X_t; shared across mediators via Woodbury rank-1 update |
| G | `(J, J)` | Gram matrix of FWL-residualised haplotypes: `G = H_perp^T H_perp`; its pseudoinverse gives fixef betas |
| G_inv | `(J, J)` | `pinv(G, rcond=1e-10)`; used directly for fixef beta estimation and as the shared inverse in Woodbury mediation update |
| e0 | `(n,)` | Null residual vector: `ỹ − X_t β̂` at H₀ (no haplotype); `rss0 = e0 · diag(w) · e0`; reused for all loci on the chromosome |
| X | `(n, p)` | Fixed-effect covariate matrix (intercept, sex, batch, …) |
| X_t / xt | `(p, n)` | Kinship-rotated X: `U^T X`; transposed for einsum efficiency |
| Z | `(n, m)` | Sample-to-strain design matrix; one 1 per row in the column for that sample's strain |
| K / K_loco | `(m, m)` | LOCO kinship matrix; `K_{ij} = (1/L) Σ_ℓ Σ_f H_{iℓf} H_{jℓf}` excluding test chromosome |
| K_mpd | `(m, m)` | Mean Parental Diversity kinship from miQTL (`calc.kinship.from.genomecache.with.DOQTL`) |
| K_n | `(n, n)` | Sample-expanded kinship: `Z K Z^T`; never materialised in hot path (Woodbury avoids it) |
| U / eigvecs | `(m, m)` | Eigenvectors of K from `K = U Λ U^T`; used to rotate data into diagonal-covariance space |
| Λ / eigvals | `(m,)` | Eigenvalues of K; λ_i enters per-sample weight formula |
| u | `(m,)` | Polygenic random effect vector; `u ~ N(0, τ²K)` |
| y / ỹ | `(n,)` | Phenotype vector / kinship-rotated phenotype `U^T y` |
| W | `(n,)` | NB PQL working weights: `W_i = μ_i / (1 + φμ_i)` |
| z (PQL) | `(n,)` | NB PQL working response: `(log μ − offset) + (S − μ)/μ` |
| α | `(J,)` | Per-founder allele effects at a locus; estimated by FWL+pinv (fixef) or BLUP (ranef) |
| β | `(p,)` | Fixed-effect covariate coefficients |

### Biology / Genomics terms

| Term | Full form | Meaning |
|---|---|---|
| QTL | Quantitative Trait Locus | Genomic region statistically associated with variation in a quantitative trait |
| eQTL | Expression QTL | QTL for gene expression levels |
| cis-eQTL | cis-acting eQTL | eQTL where peak locus is on the same chromosome as the gene (≤ a defined window, default ±10 Mb in local mode — matches typical CC haplotype block size) |
| trans-eQTL | trans-acting eQTL | eQTL where peak locus is on a different chromosome from the gene |
| vQTL | Variance QTL | QTL for phenotypic dispersion (φ in NB; variance in LMM) rather than mean |
| caQTL | Chromatin Accessibility QTL | QTL for ATAC-seq peak height |
| sQTL | Splicing QTL | QTL for alternative splicing ratios |
| ASE | Allele-Specific Expression | Within-individual difference in expression from two alleles; requires phased reads |
| GxE | Genotype × Environment interaction | Statistical interaction between haplotype and an environmental/biological covariate (e.g. sex, treatment) |
| GxG / epistasis | Genotype × Genotype interaction | Pairwise interaction between two genomic loci |
| eGene | eQTL gene | Gene with a significant QTL after multiple-testing correction |
| TSS | Transcription Start Site | 5′ start of a gene's primary transcript; used as the gene position reference for cis-distance |
| UMI | Unique Molecular Identifier | Barcode attached to each mRNA molecule before amplification; count of unique UMIs = RNA molecule count, removing PCR duplicates |
| CPM | Counts Per Million | `count / total_counts × 1e6`; normalises for library size |
| pseudobulk | — | Aggregated per-(sample × cell-type) count matrix summed from single-cell data; mosaiQTL's primary input |
| snRNA-seq | Single-nucleus RNA sequencing | RNA-seq from isolated nuclei rather than whole cells; used for tissues that don't dissociate well (e.g. muscle) |
| HMM | Hidden Markov Model | Used in `genotype/hmm.py` to compute founder haplotype probabilities from genotyping array calls |
| GigaMUGA | Giga Mouse Universal Genotyping Array | ~143k SNP array used to genotype CC/DO mice; mosaiQTL genome = mm39 |
| mm39 | Mouse genome assembly 39 | GRCm39; reference genome for all position coordinates in mosaiQTL |
| IBD | Identity By Descent | Shared haplotype segments inherited from a common ancestor; basis for kinship |
| haplotype | — | Single founder-strain chromosome strand; at any locus a CC strain carries two identical copies (inbred → homozygous), so haplotype probability = diplotype probability for CC |
| diplotype | — | Pair of founder haplotypes at a locus; for CC inbred strains the outer product `P(fi) × P(fj)` is used in the full (non-additive) model; for heterozygous DO strains diplotype probabilities must be tracked explicitly |
| allelic series | — | Pattern of per-founder effects at a QTL locus; describes which founders carry the high-effect vs low-effect allele; visualised via `viz/effects.py::coef_profile` |
| founder haplotype | — | The genomic segment at a locus inherited from one of the J CC/DO founders; the J-column probability vector H_{i,ℓ} gives the probability that individual i carries each founder's haplotype at locus ℓ |
| dosage | — | In SNP scan: allele dosage at a biallelic SNP, typically 0/1/2 for ref/het/alt; the `beta_dosage` column in SNP scan output gives the effect per unit dosage |
| planted QTL / planted locus | — | A simulated QTL inserted at a known position by `sparc::sim.CC.data`; used as ground truth in Phase 1–2 validation to verify that the scan recovers the correct peak position and effect direction |
| pseudomarker | — | Interpolated marker inserted between typed markers to densify the genetic map; present in the genome cache but excluded from the scan (`loci_max_delta=1237` reflects the number removed in sparcc_cache) |
| proximal contamination | — | Bias in h² estimation when markers on the test chromosome are included in the kinship matrix; the test-locus haplotype signal "leaks" into the random effect, inflating h² and suppressing LOD; prevented by LOCO kinship |
| kinship rotation | — | Diagonalising the LMM covariance by `K = UΛU^T` and rotating all data into eigenvector space; converts the LMM to WLS with per-sample weights `w_i = (σ² + τ²λ_i)⁻¹`, eliminating all O(n³) operations after the one-time O(m³) eigendecomposition |
| collapse_genomecache | — | Locus-collapsing step that merges redundant markers with nearly identical haplotype probability vectors (L2 greedy forward scan); reduces scan cost proportionally; implemented in `genotype/genomecache.py::collapse_genomecache`. **Not yet validated** — P2-S5 gate requires an uncollapsed cache; validation pending (see GOALS.md P2-S5). |
| amortization | — | Precomputing the chromosome-constant kinship rotation of H (`H_t`) once per chromosome in `precompute_chromosome`, then reusing it for every gene in the gene loop; the key algorithmic difference from miQTL's per-gene loop |
| Brent's method | — | Scalar optimizer used in `estimate_h2_null` to find the ML or REML optimum of h² ∈ [0, 1); bracket search on the log-likelihood; standard `scipy.optimize.brent` |

### Populations / experimental designs

| Term | Full form | Meaning |
|---|---|---|
| CC | Collaborative Cross | Panel of ~60 inbred mouse strains derived from 8 founders; each strain fixed as homozygous mosaic of 8 haplotypes |
| DO | Diversity Outbred | Outbred population derived from same 8 CC founders; heterozygous, requires diplotype model |
| MAGIC | Multiparent Advanced Generation Inter-Cross | General term for MPP; specifically also a plant (Arabidopsis / wheat) multiparent population |
| MPP | Multiparent Population | Any population derived from >2 founder strains (CC, DO, MAGIC, heterogeneous stock) |
| F2 | Second-generation intercross | Two-founder cross; not CC/DO but supported by Population config |
| CC founders | — | The 8 inbred strains: A/J, C57BL/6J, 129S1/SvImJ, NOD/ShiLtJ, NZO/HlLtJ, CAST/EiJ, PWK/PhJ, WSB/EiJ |
| ROP | Random Observed Phenotype | Standard LMM scan using actual phenotype values; baseline scan mode |
| MI | Multiple Imputation | Scan where `n_imp` hard diplotype draws are averaged to integrate founder-assignment uncertainty |

### Tools / infrastructure

| Term | Full form | Meaning |
|---|---|---|
| miQTL | — | R package by Greg Keele; mosaiQTL's primary algorithmic reference and validation ground truth |
| qtl2 | R/qtl2 | Broman lab R package for QTL mapping; mosaiQTL API mirrors its scan conventions |
| HAPPY | — | R haplotype probability format used by qtl2/miQTL; mosaiQTL's `GenomeCache.from_rds` reads this |
| sparcc | — | R package (Keele) for simulating CC phenotypes with planted QTL; used in Phase 1–2 validation |
| sparcc_cache | — | Companion genome cache (HDF5) containing real CC haplotype probabilities; used by sparcc and mosaiQTL validation |
| NEBULA | — | `NB-mixed model for Ultra-Large single-cell Analysis`; Stage 1 null model in nb.py follows NEBULA-LN approach (He et al. 2021) |
| LOCO | Leave-One-Chromosome-Out | Kinship recomputed excluding test chromosome; prevents proximal contamination of kinship estimate |
| HPC | High-Performance Computing | Cluster compute environment (UCI CRSP / SLURM); used for full gastroc genome-wide scan |
| SLURM | Simple Linux Utility for Resource Management | Job scheduler on HPC clusters; mosaiQTL submits per-chromosome jobs as a SLURM array |
| HDF5 | Hierarchical Data Format v5 | Binary file format used for genome cache (`.h5`) and expression data; chunked, supports lazy I/O |
| MPS | Metal Performance Shaders | Apple Silicon GPU backend for PyTorch; used in SPEED-07 GPU path on M-series Macs |
| CUDA | Compute Unified Device Architecture | NVIDIA GPU compute platform; HPC GPU path in SPEED-07 |
| RDS | R Data Serialisation | R's native binary format (`.rds`); `GenomeCache.from_rds` converts this to HDF5 |
| PEER | Probabilistic Estimation of Expression Residuals | Method for computing expression latent factors as covariates; expression PCs (SWE-10) are a simpler equivalent |

### Validation / pipeline shorthand

| Term | Meaning |
|---|---|
| S1–S11 | The 11 comparison sections in Phase 2 validation (`validate_phase2.py`); numbered to match `phase2_validation.R` output |
| P1–P7 | Phase 4 biological gate IDs (`validate_phase4.py`) |
| Phase 1 | Minimal miQTL workflow mirror — ROP LOD + MI LOD + GEV threshold (`validate_sim.py`) |
| Phase 2 | Full 11-section R vs Python comparison on sparc simulated data (`validate_phase2.py`) |
| Phase 3 | Speed and memory benchmarking (`validate_phase3.py`) |
| Phase 4 | NB vs LMM comparison on simulated single-cell data (`validate_phase4.py`) |
| gastroc | Gastrocnemius (skeletal muscle) — the real CC pseudobulk dataset used for biological validation |
| r_data/ | Subdirectory holding R reference outputs; Python validation reads these for comparison |
| py_data/ | Subdirectory holding Python scan outputs |
| compare/ | Subdirectory holding merged R-vs-Python comparison tables and plots |
| fix_h2=True | Hardcoded behavior in `run_threshold_scans`: h² is always fixed from the real phenotype and never re-estimated per permutation. The `fix_h2` parameter was removed — this is now the only mode. Required for the vectorised `scan_chromosome_batch` path (h² fixed → single eigendecomposition shared across all permutations). |
| ref_chrom / peak_chrom | The chromosome whose LOCO kinship is used to draw null phenotypes for permutation; should be the chromosome with the highest observed LOD |
| scan_window | Controls the genomic scope of a scan call. Three options: **G** (`'genome'`) — scans all chromosomes in `kinship_decomp`; **C** (`'chr'`) — scans the gene's own chromosome only (requires `gene_chr`); **L** (`'local'`) — scans loci within ±`window_mb` of `gene_pos_mb` on `gene_chr` (requires both `gene_chr` and `gene_pos_mb`). Default is `'cis'` in `scan_h2lmm`/`scan_h2lmm_mi` (equivalent to `'chr'`) and `'genome'` in `scan_multi_gene`. |
| window_mb | Half-window in Mb for `scan_window='local'`; default 10.0 (matches typical CC haplotype block size). Only active in local mode — ignored for `'chr'` and `'genome'`. |

### Mediation output columns

Columns in the DataFrame returned by `mediation_scan_qr` and `fixef_mediation_scan`.

| Column | Meaning |
|---|---|
| `lod_base` | LOD at the locus from the base model `y ~ H \| X` (no mediator); the haplotype QTL evidence before conditioning on the mediator |
| `lod_mediated` | LOD at the same locus from the mediated model `y ~ H \| X, M`; haplotype signal remaining after accounting for mediator M |
| `lod_drop` | `lod_base − lod_mediated`; reduction in LOD explained by the mediator; the primary mediation signal. Always ≥ 0 (clamped). |
| `prop_mediated` | `lod_drop / max(lod_base, ε)`; proportion of the haplotype effect explained by the mediator; bounded [0, 1]. A value near 1 means the mediator fully explains the QTL. |
| `p_value` | chi-squared(df=1) p-value for `lod_drop`; tests whether mediation is statistically significant |
| `h2` | LOCO null h² from the base scan on the outcome; fixed for all mediators on the same chromosome |
| `n` | Effective sample size after NaN filtering and Z-matrix reduction to strain means |

### Data objects

| Object | Module | Meaning |
|---|---|---|
| `ScanResult` | `models/records.py` | Primary output of `scan_h2lmm` / `scan_h2lmm_mi`; fields: `lod (n_loci,)`, `betas (n_loci, J)`, `p_value (n_loci,)`, `h2`, `chr`, `phenotype_id`, `locus_ids`, `positions` |
| `LMMResult` | `qtl/lmm.py` | Per-locus result from `fit_lmm_at_locus`; fields: `lod`, `p_value`, `f_stat`, `h2`, `betas (J,)`, `rss_null`, `rss_alt` |
| `VQTLResult` | `models/records.py` | Dispersion QTL result; fields: `scan_result` (ScanResult for dispersion), `mean_scan_result` (ScanResult for mean). **Note**: field name is `scan_result` not `vqtl_scan`; `mean_scan_result` not `mean_scan` — common bug source. |
| `QTLResult` | `models/records.py` | eGene record written to `eqtl_summary.csv`; one row per significant gene after permutation thresholding and FDR correction |
| `FounderEffect` | `models/records.py` | Per-founder allele effect at the peak locus; fields: `gene`, `locus`, `founder`, `beta`, `se`, `ci_lo`, `ci_hi` |
| `GenomeCache` | `genotype/genomecache.py` | HDF5-backed haplotype probability store; lazy per-chromosome loading via `get_chromosome(chrom)`; holds `sample_ids`, `chromosomes`, `n_founders`, `population` |
| `Population` | `models/config.py` | Frozen Pydantic model encoding all population-specific config: `founders` list, `colors` dict, `n_gen`, `inbred`, `chromosome_names`, `genotyping_error_prob`. `CCPopulation` is the built-in CC instance. |
| `ScanConfig` | `models/config.py` | Scan-level configuration: `population`, `scan_window`, `cis_window_mb` (default 10.0), `n_perms`, `seed`, `offset`, `permutation_method` |
| `kinship_decomp` | `genotype/kinship.py` | Dict `{chrom: {'values': eigvals (m,), 'vectors': eigvecs (m,m)}}` returned by `decomp_kinship`; passed to all scan functions that require LOCO kinship |
| `KinshipSingularError` | `qtl/permutation.py` | Raised when a LOCO kinship matrix has a minimum eigenvalue below the PSD tolerance; chromosome is skipped in permutation |

### API parameters

Common parameters shared across scan entry-point functions.

| Parameter | Type | Meaning |
|---|---|---|
| `phenotype_id` | str | String label attached to the output `ScanResult.phenotype_id`; used as gene name in multi-gene wrappers and output CSVs |
| `sample_ids` | list[str] | Genotype-level strain IDs (one per row of `phenotype` / `covariates`); must match `GenomeCache.sample_ids`. **Not** raw sample labels — use `meta_geno_map` to convert before calling. |
| `gene_chr` | str | Chromosome of the gene being scanned; required for `scan_window='cis'` and `'local'`; must be a string (not int) |
| `gene_pos_mb` | float | TSS position in Mb; required for `scan_window='local'` |
| `kinship_decomp` | dict | See Data objects above; when `None`, computed internally from `genomecache` (slow — pass explicitly in loops) |
| `covariates` | np.ndarray (n, p) | Fixed-effect covariate matrix including intercept; when `None` an intercept-only matrix is constructed |
| `use_rint` | bool | Apply RINT to the phenotype before scanning (default `True`); set `False` when the caller has already normalized or when matching R reference behavior that RINTs at a different stage |
| `base_lod` | dict | Pre-computed base LOD scores `{chrom: ndarray (n_loci,)}`; passed to `mediation_scan_qr` to skip re-running the outcome scan for multiple mediators |

### Key output file columns

**`lod_genome.csv` / `rop_lod.csv` / `mi_lod.csv`** — per-gene LOD profiles

| Column | Type | Meaning |
|---|---|---|
| `gene` | str | Gene name (symbol) |
| `locus` | str | Locus identifier from genome cache (e.g. `rs12345678` or `chr8:45.23`) |
| `chr` | str | Chromosome (always `.astype(str)` — rule in INVIOLABLE RULES) |
| `pos_mb` | float | Physical position in megabases (Mb) on the reference genome |
| `lod` | float | LOD score at this locus for this gene |

**`perm_max_lods.csv` / `perm_max_lods_chr{N}.csv`** — permutation null distribution

| Column | Type | Meaning |
|---|---|---|
| `perm_index` | int | Permutation replicate index (0-based) |
| `max_lod` | float | Genome-wide (or per-chromosome) maximum LOD under the null for this replicate |
| `peak_mb` | float | Position of that max-LOD locus (present when `return_peak_positions=True`) |
| `chr` | str | Chromosome of max-LOD locus (per-chromosome files only) |

**`perm_threshold.csv`** — GEV-fitted significance thresholds

| Column | Type | Meaning |
|---|---|---|
| `chr` | str | Chromosome (or `'genome'` for genome-wide threshold) |
| `threshold_lod` | float | LOD threshold at α=0.05 from GEV fit |
| `gev_shape` | float | GEV shape parameter ξ |
| `gev_loc` | float | GEV location parameter μ |
| `gev_scale` | float | GEV scale parameter σ |
| `n_perms` | int | Number of permutation replicates used for the GEV fit |

**`allele_effects_chr{N}.csv`** — per-founder haplotype effects at peak loci

| Column | Type | Meaning |
|---|---|---|
| `gene` | str | Gene name |
| `locus` | str | Peak locus identifier |
| `pos_mb` | float | Peak locus position (Mb) |
| `founder` | str | Founder strain name (e.g. `A/J`, `B6/J`) |
| `beta` | float | Allele effect in RINT units (fixef estimator); see §9 note on 2× scale vs miQTL |
| `se` | float | Standard error of beta |

**`eqtl_summary.csv`** — one row per significant eGene (pipeline output)

| Column | Type | Meaning |
|---|---|---|
| `gene` | str | Gene name |
| `chr` | str | Chromosome of peak locus |
| `tss_mb` | float | TSS position (Mb) |
| `peak_locus` | str | Peak locus identifier |
| `peak_mb` | float | Peak locus position (Mb) |
| `cis_dist_mb` | float | Signed distance from TSS to peak (positive = downstream) |
| `peak_lod` | float | Peak LOD score |
| `h2` | float | Null h² (polygenic background fraction) |
| `perm_n` | int | Number of permutation replicates |
| `perm_chr` | str | Chromosome used for permutation null draws |
| `chi2_haplotype` | float | LRT χ²(J−1) for haplotype effect |
| `p_haplotype` | float | p-value for haplotype effect, χ²(J−1) |
| `in_r_egenes` | bool | Present in miQTL eGene list (when `r_data/egenes.csv` available) |

**`scan_stats.csv`** — per-chromosome scan timing and memory (written by `validate_gastroc.py`)

| Column | Type | Meaning |
|---|---|---|
| `chr` | str | Chromosome |
| `n_genes` | int | Number of genes scanned on that chromosome |
| `n_loci` | int | Number of loci (pseudomarkers excluded) |
| `wall_sec` | float | Wall-clock time in seconds |
| `rss_mb` | float | Peak RSS memory in MB (from `tracemalloc` or `resource`) |
| `genes_per_min` | float | Throughput: `n_genes / (wall_sec / 60)` |

**`bootstrap_ci.csv`** — positional confidence intervals from `run_positional_scans`

| Column | Type | Meaning |
|---|---|---|
| `gene` | str | Gene name |
| `chr` | str | Chromosome |
| `ci_lo_mb` | float | Lower bound of LOD-drop or bootstrap CI (Mb) |
| `ci_hi_mb` | float | Upper bound of CI (Mb) |
| `peak_mb` | float | Bootstrap mean peak position (Mb) |
| `n_boot` | int | Number of bootstrap replicates |

**`snp_lod.csv`** — SNP fine-mapping output from `imputed_snp_scan_h2lmm`

| Column | Type | Meaning |
|---|---|---|
| `gene` | str | Gene name (from `phenotype_id`) |
| `snp` | str | SNP identifier |
| `chr` | str | Chromosome |
| `pos_mb` | float | SNP position (Mb) |
| `lod` | float | LOD score under the marginal SNP model |
| `p_value` | float | chi-squared p-value derived from LOD |
| `f_stat` | float | F-statistic from `fit_lmm_at_locus` |
| `h2` | float | LOCO null h² |
| `n` | int | Effective sample size |
| `rss0` | float | Null RSS (no SNP effect) |
| `rss1` | float | Alternative RSS (SNP effect included) |
| `beta_dosage` | float | Estimated effect per unit allele dosage (0/1/2) |

**`mediation_lod.csv`** — mediation scan output from `mediation_scan_qr`

| Column | Type | Meaning |
|---|---|---|
| `outcome_id` | str | Outcome gene name |
| `mediator_id` | str | Mediator gene name |
| `chr` | str | Chromosome |
| `locus` | str | Locus identifier |
| `pos_mb` | float | Locus position (Mb) |
| `lod_base` | float | Haplotype LOD without mediator covariate |
| `lod_mediated` | float | Haplotype LOD after conditioning on mediator |
| `lod_drop` | float | `lod_base − lod_mediated`; the mediation signal |
| `prop_mediated` | float | `lod_drop / max(lod_base, ε)`; proportion explained |
| `p_value` | float | chi-squared(1) p-value for `lod_drop` |
| `h2` | float | Outcome null h² (from base scan) |
| `n` | int | Effective sample size |

---
*Last updated: see git log. Canonical source: github.com/gfilimban/mosaiqtl/docs/ARCHITECTURE.md*

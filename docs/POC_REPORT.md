# mosaiQTL — Proof-of-Concept Report

**Repo:** `github.com/gfilimban/mosaiqtl` (private)  
**Version:** 0.2.0  
**Latest commit:** `see git log`  
**Last updated:** April 2026

---

## Overview

mosaiQTL maps genetic variants that regulate gene expression at founder-haplotype
resolution in multiparent populations. The primary application is single-nucleus
RNA-seq from Collaborative Cross (CC) gastrocnemius muscle (type1, type2a, type2b,
type2x myonuclei; 300,000+ nuclei; ~48 CC strains + 8 founders + 7 F1 hybrids).

The package is structured in three progressive analysis layers:

```
Layer 1 — QTL Discovery           pipeline.discover()
Layer 2 — QTL Characterization    pipeline.characterize()
Layer 3 — Mechanistic Resolution  pipeline.resolve(modules=[...])
```

Each layer is independent. Layer 1 alone produces a genome-wide eGene catalogue.
Layers 2 and 3 add characterisation and mechanistic resolution progressively.

---

## Shared design principles

**Z-matrix formulation.** All models support $n \geq m$ — $n$ phenotype
observations (cells or pseudobulk replicates) mapping to $m$ unique genotyped
strains. The mapping matrix $Z \in \{0,1\}^{n \times m}$ is built at scan time
from `sample_ids` vs `genome_cache.sample_ids`. When $n = m$ and IDs match,
$Z = I_m$ and all formulas reduce to the strain-mean case. No module hardcodes
$n$, $m$, or $J$.

**Model hierarchy.**

| Model | Module | Phenotype | Kinship | Primary use |
|---|---|---|---|---|
| NB GLM (pseudobulk) | `nb.py` | raw UMI counts | none (u_i random effect) | **primary eQTL** |
| Gaussian LMM | `lmm.py` + `scan.py` | RINT expression | LOCO | miQTL port / validation |
| Dispersion scan (vQTL) | `vqtl.py` | log(φ̂) per strain | LOCO | cell-count heterogeneity |
| Epistasis | `epistasis.py` | RINT expression | LOCO | pairwise GxG at haplotype level |

**LOD formula.** All models use the ML denominator:
$\text{LOD} = \frac{n}{2} \log_{10}\!\frac{\text{RSS}_0}{\text{RSS}_1}$,
matching miQTL. REML is used for $h^2$ estimation under the null; LOD scoring
uses ML to ensure comparability across loci.

---

## Layer 1 — QTL Discovery

### 1a. NB GLM scan (primary model)

Two-stage design in `nb.py`:

**Stage 1 — cell-level null model (run once per cell type):**

$$Y_{ij} \sim \text{NB}(\mu_{ij},\, \phi_g), \quad
\log \mu_{ij} = X_j \beta_g + \log s_j + u_i, \quad
u_i \sim \mathcal{N}(0, \sigma^2_g)$$

$Y_{ij}$: UMI count for gene $g$ in cell $j$ of strain $i$.
$s_j$: TMM library size offset.
$u_i$: unstructured per-subject random effect (handles within-strain cell
variance). Not a LOCO-kinship random effect — that is a planned extension.

Implemented via `edgePython.glm_sc_fit()` (Pachter 2026, GPL-3.0) + empirical
Bayes dispersion shrinkage via `shrink_sc_disp()`.

**Stage 2 — NB-GLMM with LOCO kinship + score test (recommended):**

For each chromosome $c$, fits a null NB-GLMM using PQL (Breslow & Clayton 1993;
Chen et al. 2016 AJHG) with the LOCO kinship $K^{(-c)}$ as structured covariance:

$$S_i \sim \text{NB}(\mu_i, \phi_g), \quad
\log \mu_i = \mathbf{x}_i^\top \boldsymbol{\beta} + u_i + \log(n_i \bar{s}_i),
\quad \mathbf{u} \sim \mathcal{N}(\mathbf{0}, \tau K^{(-c)})$$

PQL converges in 3–5 iterations. The projection matrix $P = V^{-1} - V^{-1}X(X^\top V^{-1}X)^{-1}X^\top V^{-1}$
(where $V = \operatorname{diag}(1/W) + \tau K^{(-c)}$) is computed once per chromosome per gene,
then vectorised over all $L$ loci:

$$T_\ell = \mathbf{U}_\ell^\top \mathbf{I}_\ell^{-1} \mathbf{U}_\ell \sim \chi^2_{J-1}, \quad
\mathbf{U}_\ell = H_\ell^\top (P\mathbf{z}), \quad
\mathbf{I}_\ell = H_\ell^\top P H_\ell$$

Without kinship (fallback when `kinship_loco=None`): $\sigma_\text{sample}$ from
Stage 1 inflates the diagonal information matrix.

Entry point: `nb.scan_nb(counts_cells, genome_cache, strain_ids_cells, kinship_loco=kinship_loco, ...)`.
Returns `dict {gene_id: ScanResult}`.

### 1b. Gaussian LMM scan (miQTL port, validation only)

Eigendecomposition LMM (Kang et al. 2008):

$$\mathbf{y} = \mathbf{H}_\ell \boldsymbol{\gamma} + \mathbf{X} \boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}, \quad
\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \sigma^2_g \mathbf{K}), \quad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2_e \mathbf{I})$$

$h^2$ estimated once under the null via REML grid search (fix-par convention,
matching miQTL `use.fix.par=TRUE`). Genome-wide scan vectorised via
Frisch-Waugh-Lovell (FWL) — mathematically identical to per-locus WLS but
~100× faster. LOCO kinship used throughout.

Entry point: `scan.scan_h2lmm(phenotype, genome_cache, covariates, kinship_decomp, ...)`.
Returns `ScanResult`.

---

## Layer 2 — QTL Characterisation

### 2a. Dispersion QTL scan (vQTL)

For each gene, estimate per-strain overdispersion $\hat{\phi}_i$ from cell-level
counts using method-of-moments or MLE, then scan $\log \hat{\phi}$ as a
phenotype through the Gaussian LMM.

A significant vQTL (independent of or co-localised with the mean eQTL) implies
the founder haplotype affects not just mean expression but expression noise —
mechanistically distinct from a mean eQTL and only detectable at cell resolution.

Implementation: `vqtl.scan_vqtl(counts, subject_ids, genome_cache, ...)`.
Returns `VQTLResult` with fields `scan_result`, `mean_scan_result`, `phi_estimates`.

### 2b. Mediation analysis

QR-based mediation scan tests whether including a candidate mediator gene as a
covariate reduces the QTL LOD, quantifying mediation of the haplotype effect
through a second gene's expression.

$\text{LOD\_drop} = \text{LOD\_base} - \text{LOD\_mediated}$

Negative LOD drop (amplification) is not a bug — it indicates the mediator is
part of the same pathway and conditioning on it sharpens the genetic signal.

Implementation: `mediation.mediation_scan_qr(outcome, mediator, genome_cache, ...)`.

---

## Layer 3 — Mechanistic Resolution

### 3a. Epistasis (pairwise GxG)

Tests whether the combination of founder haplotypes at two loci produces
expression not predicted by their additive sum:

$$\text{additive:} \quad y = H^a \alpha + H^b \beta + X \xi + u$$
$$\text{interaction:} \quad y = H^a \alpha + H^b \beta + (H^a \otimes H^b) \gamma + X \xi + u$$
$$\text{LOD\_epistasis} = \text{LOD\_interaction} - \text{LOD\_additive}$$

The interaction term $(H^a \otimes H^b) \in \mathbb{R}^{(J-1)^2}$ is the
Khatri-Rao product of the two haplotype design matrices.

Implementation: `epistasis.scan_epistasis(phenotype, genome_cache, locus_a, locus_b, ...)`.

Companion: `epistasis.test_haplotype_combinations()` produces the $J \times J$
founder-pair expression heatmap — the empirical counterpart to the interaction
test.

### 3b. Conditional multi-signal decomposition

Forward-stepwise: at each iteration, compute the BLUP of the haplotype effect at
the peak locus, subtract it from the phenotype residual, and rescan. Continues
until the residual peak falls below `lod_drop_threshold`.

Implementation: `epistasis.scan_conditional_with_y(phenotype, scan_result, genome_cache, ...)`.
Returns `list[ScanResult]`, one per detected signal.

### 3c. SNP catalogue (`snp_scan.py`)

Overlays Sanger MGP r9 founder sequences onto haplotype-level QTL intervals to
build a ranked candidate SNP list. Founder haplotype probabilities define allele
dosage at each variant; imputed SNP-level LOD is computed via the LMM.

### 3d. ASE (`ase.py`)

Allele-specific expression estimation via kallisto on F1 hybrid FASTQ reads,
aligned to personalised founder-pair references.

---

## Phase 2 validation status (S1–S11, genome-wide)

Port equivalence with miQTL R package (`gkeele/miqtl`) established via
`validation/validate_port.py` against `validation/r_reference/port_validation.R`.

| S# | Section | Result | Status |
|---|---|---|---|
| S1 | Cache info | `founders_match=True  samples_match=True  loci_exact_match=True` | ✅ |
| S2 | Haplotype probs | `max_dev=0.00`; actual prob_r pending R rerun | ⚠️ |
| S3 | Kinship LOCO | `mean_r=0.9998  max\|ΔK\|=1.98e-02` | ✅ |
| S4 | LMM scan | `60.2% pass  median_r=0.9730` | ✅ |
| S5 | Cache collapse | `py=91.4%  lod_preservation=0.9999` | ✅ |
| S6 | Permutation | `mean_delta=0.619` — GEV instability undiagnosed | ❌ |
| S7 | MI scan | `r=0.8191` | ✅ |
| S8 | SNP scan | `r=0.5509` — documented marginal vs diplotype model diff | ❌ doc |
| S9 | Mediation | `r=0.9983` | ✅ |
| S10 | Positional CI | `overlap=True` | ✅ |
| S11 | Instability | chr-type bug fixed; rerun pending | ⚠️ |

**Genuine passes: S1, S3, S4, S5, S7, S9, S10 (7/11).** S6 and S8 are documented
failures, not regressions. S2 and S11 require one targeted rerun each.

**S4 fail-rate note.** The 39.8% fail rate concentrates at genes with peak LMM
LOD < 1 — the LMM-vs-LMM comparison is noisy at low signal. A LOD-stratified
breakdown is now included in `s4_gene_tiers.png`.

**S8 model note.** Python uses marginal haplotype probabilities (forward-backward
HMM, 8-state); R uses HAPPY diplotype probabilities (36-state). The r=0.55 SNP
scan correlation is expected and documented, not a bug. Moving to diplotype HMM
is a future extension.

**S6 permutation note.** `mean_delta=0.619` indicates GEV tail instability when
pooling across chromosomes with very different cis-loci counts. Per-chromosome
threshold deltas are now plotted in `s6_permutation.png` (panel 3) for diagnosis.

---

## Phase 3 validation (P1–P4, real gastroc data)

Implemented in `validation/validate_phase3.py`. Runs against
`gastroc_all_processed2.h5ad` (HPC only). No R reference — validated via internal
consistency and concordance with known LMM top hits.

| Section | Entry point | Pass gate |
|---|---|---|
| P1 scan_nb | `nb.scan_nb()` | LOD right-skewed + Piezo1 LOD > 3 |
| P2 scan_vqtl | `vqtl.scan_vqtl()` | ≥1 gene returns result |
| P3 scan_epistasis | `epistasis.scan_epistasis()` | lod_additive > 0 |
| P4 scan_conditional | `epistasis.scan_conditional_with_y()` | signal1_lod > 0 |

**Known top LMM candidates (anchors for phase 3 sanity checks):**

| Gene | Chr | Approx peak | LMM LOD | Founders | Biology |
|---|---|---|---|---|---|
| Piezo1 | 8 | ~122 Mb | ~21.8 | CAST/J↑, PWK/J↑ | Mechanosensitive ion channel; muscle fiber integrity; GTEx v8 skeletal muscle eGene |
| Ccdc171 | 4 | ~84 Mb | ~8.0 | NOD/J specific | No prior muscle annotation |
| Pmis2 | 7 | ~85 Mb | ~6.0 | Cell-type restricted | Only visible in pseudobulked fiber subtypes |

---

## Genome cache construction

The HAPPY genome cache is built by `validation/r_reference/build_genomecache.R` (replaces the
former `founders_f1_genomecache.ipynb`). The script is a single executable R file
covering all steps from raw GigaMUGA matrix to HAPPY cache and LOCO kinship:

```
Rscript validation/r_reference/build_genomecache.R [base_dir] [n_cores]
```

**Key filtering step (step 3).** `qtl2convert::find_unique_geno()` is applied to
the **8 founder columns** — the canonical qtl2 workflow
(kbroman.org/qtl2/pages/prep_do_data.html). Markers where founders are monomorphic
or fully missing across all 8 lines are dropped; these carry zero haplotype-
discrimination power for the HAPPY HMM. `encode_geno()` is then applied to founders
first, then CC strains using the same founder-derived allele codes.
Python's `GenomeCache.from_gigamuga()` replicates this exactly.

**Pseudomarker policy.** `insert_pseudomarkers` → `calc_genoprob` is the standard
qtl2 workflow for `scan1()` (qtl2-native scanning at inter-marker positions).
Neither pipeline uses it here because:

- **R side** (`port_validation.R`) drives `scan.h2lmm`, which reads the HAPPY cache
  directly and performs its own internal haplotype interpolation via the multiple-
  imputation model (`use.multi.impute=TRUE`). Pseudomarkers in the HAPPY cache would
  be redundant and inconsistent with what `scan.h2lmm` actually queries.
- **Python side** (`scan_chromosome`) operates on the same genotyped-marker positions
  as R for validation parity. If Python added pseudomarker positions and R did not,
  S4 LOD comparisons would be position-mismatched and uninterpretable.
- **Kinship** (`genoprob_to_alleleprob` → `calc_kinship`) requires accurate
  allele-probability estimates at genotyped positions; pseudomarker interpolations
  are not needed and would add noise. The qtl2 DO user guide itself uses
  `calc_genoprob` without pseudomarkers for kinship computation.

The two workflows are **fully independent**: R builds the HAPPY cache and kinship
once from `build_genomecache.R`; Python rebuilds both independently from the same
raw GigaMUGA inputs via `from_gigamuga`. S1 validates they produce identical marker
sets; S2–S11 validate agreement on derived quantities. No shared intermediate
objects pass between pipelines during validation.

**Strain order.** Canonical order in all cache outputs: founders → F1s → CC strains.

## Validation workflow

```
# Phase 2 — S1-S11 port validation
Rscript validation/r_reference/port_validation.R all Inf   # R reference (SLURM array)
Rscript validation/merge_chr_outputs.R                      # merge per-chr outputs
python validation/validate_port.py                          # Python comparison + plots

# Targeted reruns (section-level checkpointing)
python validation/validate_port.py --sections 2,11
python validation/validate_port.py --force                  # ignore checkpoints

# Phase 3 — novel extensions (HPC only)
python validation/validate_phase3.py --cell_type type2x --n_genes 500
python validation/validate_phase3.py --force                # ignore checkpoints

# Unit tests
python -m pytest tests/ -v

# Quick syntax check
make check
```

**Output directories:**

```
validation/phase2/
  r_data/     ← R reference CSVs (from port_validation.R + merge)
  py_data/    ← Python comparison CSVs + checkpoint .done files
  plots/      ← QC plots (S1–S11)
validation/phase3/
  py_data/    ← NB scan, vQTL, epistasis, conditional CSVs
  plots/      ← Phase 3 QC + comparison plots
```

---


---

## Assessment audit log

### Session: find_unique_geno & pseudomarker review (commit `7aeea93`)

**Trigger:** Observed `loci_max_delta=1237` in S1 (max delta on single worst chromosome,
not a per-chromosome average as previously mis-stated). Two independent bugs identified
in `build_genomecache.R`.

**Bug 1 — `find_unique_geno` applied to CC strains (wrong).** The function is a
legitimate `qtl2convert` export (`find_unique_geno`: find unique genotypes per marker
row, returning NA if ≠ 2 unique alleles). However the prior script applied it to CC
strain calls to derive allele codes. The canonical qtl2 workflow
(kbroman.org/qtl2/pages/prep_do_data.html) applies `find_unique_geno` to **founder**
genotypes to identify the two alleles, then `encode_geno` encodes both founders and
study samples using those founder-derived codes. Applying it to CC strains: (a) admits
markers monomorphic across all 8 founders but polymorphic by chance in the 33-strain
CC panel — these have zero haplotype-discrimination power for the HAPPY HMM; (b)
conflates study-sample missingness with true monomorphism.

**Bug 2 — `insert_pseudomarkers(step=0.2)` passed to `calc_genoprob`.** This added
synthetic inter-marker positions to `probs_combined`, which `convert.qtl2.to.HAPPY`
then wrote into the HAPPY cache. Python's `from_gigamuga` reads only genotyped markers
from `gmap.csv`/`pmap.csv` → loci_max_delta >> 0 in S1 (confirmed root cause of 1237
delta on densest chromosome).

**Pseudomarker policy established.** `insert_pseudomarkers` is a `scan1()` interpolation
device. Neither R (`scan.h2lmm` uses internal MI interpolation) nor Python (scans
genotyped positions for validation parity) should include pseudomarkers in the cache.
If Python ever adds pseudomarker-density scanning for production use (not validation),
that is a scan-time concern internal to `scan_chromosome`, not a cache-building concern.
The HAPPY cache and `from_gigamuga` always use genotyped markers only.

**Pipeline independence confirmed.** R and Python pipelines are fully parallel:
- R: `build_genomecache.R` (GigaMUGA → HAPPY cache + kinship) → `port_validation.R`
  (reads cache, runs `scan.h2lmm`, exports reference CSVs)
- Python: `from_gigamuga` (same GigaMUGA → HMM probs independently) → `validate_port.py`
  (compares against R CSVs)
No intermediate objects are shared. S1 is the entry-point consistency check.

**Files changed:** `validation/r_reference/build_genomecache.R`, `mosaiqtl/genome_cache.py`,
`validation/validate_port.py` (S1 gate hardened), `docs/POC_REPORT.md`.
111/111 tests pass post-fix.

## Known limitations and planned extensions

| Issue | Status |
|---|---|
| S2: HMM validation uses actual per-locus per-founder probs (haplotype_vals_r.csv) | ✅ resolved |
| S6: GEV permutation threshold unstable across chr | ❌ undiagnosed |
| NB-LMM: full LOCO kinship in NB model | 🔲 planned |
| Permutation FDR: genome-wide permutation thresholds | 🔲 planned |
| Fine-mapping: GBRS imputation → causal SNP resolution | 🔲 planned |
| Multi-tissue extension | 🔲 planned |
| PyPI publication | 🔲 deferred until NB pipeline fully validated |

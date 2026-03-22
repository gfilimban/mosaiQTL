# mosaiQTL — Haplotype-Based QTL Mapping for Multiparent Populations

## Overview

mosaiQTL is a Python package for QTL mapping in multiparent populations,
designed as a generalizable, model-agnostic hub unified by a shared
genotype probability backbone.

The user controls three orthogonal choices at scan time:

| Choice | Options | Default |
|--------|---------|---------|
| **Statistical model** | NB GLM, LMM (Gaussian), NB-LMM (mixed) | NB GLM |
| **Phenotype input** | raw counts, normalized expression, collapsed genotype means | raw counts (replicate-aware) |
| **Aggregation level** | lineage pseudobulk, subtype pseudobulk, cell-level | user-specified |

The **negative binomial GLM on raw pseudobulk counts** is the primary
mode: it respects the count nature of the data, uses all $n$ replicate
observations (not collapsed genotype means), includes biological
covariates (sex, batch), and models library size as an offset. The
**LMM on RINT-transformed expression** exists as the miQTL-equivalent
port for validation and backward compatibility, not as the recommended
default. The **NB-LMM** (NEBULA-LN) operates at cell-level resolution
for questions that pseudobulking cannot answer.

The package is parametric on population structure: the user specifies
$J$ founder strains, the set of derived lines, and the number of
replicates. Any multiparent design is supported — CC ($J = 8$),
DO ($J = 8$, outbred), MAGIC ($J = 4$, $8$, or $19$), heterogeneous
stocks, or custom crosses. The Collaborative Cross 8-founder population
serves as the proof-of-concept application.

For single-cell data, mosaiQTL supports three complementary scan
resolutions as first-class modes, each answering a different biological
question from the same dataset:

| Mode | Resolution | Question |
|------|-----------|----------|
| **Pseudobulk lineage** | Aggregate related cell types (e.g., all myonuclei) | Does haplotype affect expression in this cell program? |
| **Pseudobulk subtype** | Single cell type (e.g., type2b only) | Does haplotype affect expression specifically in this subtype? |
| **Cell-level** (NEBULA-LN, vQTL) | Individual cells | Does haplotype affect expression *noise* within this cell type? |

Pseudobulking at a chosen cell-identity level is not a compromise — it
is the highest-powered scan for lineage-level effects and produces the
primary eGene catalog. Cell-level modeling answers questions that
pseudobulking structurally cannot (dispersion QTL, within-subject
heterogeneity).

---

## Package Structure

```
mosaiqtl/
  __init__.py
  genome_cache.py     # read/write HAPPY genome cache (RDS, HDF5)
  kinship.py          # kinship: overall, LOCO, eigen decomposition
  nb_glm.py           # NB GLM scan: wraps edgePython dispersion + IRLS
  lmm.py              # LMM engine: eigendecomposition trick (miQTL port)
  nebula_ln.py        # cell-level: wraps edgePython glm_sc_fit + our scan
  scan.py             # genome-wide scan orchestrator (model-agnostic dispatch)
  vqtl.py             # dispersion vQTL: two-stage and joint approaches
  gxe.py              # genotype × environment interaction QTL (planned)
  trans.py            # trans-eQTL genome-wide scan (planned)
  sqtl.py             # splicing QTL via PSI phenotypes (planned)
  merge.py            # allelic series / merge analysis (planned)
  snp_scan.py         # imputed SNP association scan (planned)
  permutation.py      # genome-wide permutation thresholds (planned)
  utils.py            # RINT, data alignment, design matrices
  io.py               # read h5ad, PLINK, covariate files (planned)

tests/                # unit tests with small simulated datasets
validation/           # cross-language validation against R miQTL/qtl2
```

### Hybrid Dependency Architecture

mosaiQTL uses [edgePython](https://github.com/pachterlab/edgePython)
(Pachter 2026) as its NB engine, rather than reimplementing the negative
binomial fitting, dispersion estimation, and NEBULA-LN likelihood from
scratch. edgePython provides:

| Component | edgePython function | Used by mosaiQTL in |
|-----------|--------------------|--------------------|
| TMM normalization | `calc_norm_factors()` | pseudobulk preprocessing |
| Common + trended + tagwise $\phi$ | `estimate_disp()` | NB GLM null model (§I Model A) |
| Empirical Bayes shrinkage (WLEB) | `estimate_tagwise_disp()` | per-gene $\phi$ stabilization |
| NB GLM fitting (IRLS) | `glm_fit()` | null model per gene |
| Quasi-likelihood F-test | `glm_ql_ftest()` | alternative to score test |
| NEBULA-LN mixed model | `glm_sc_fit()` | cell-level eQTL (§I Model C) |
| Cell-level dispersion shrinkage | `shrink_sc_disp()` | cell-level $\phi$ estimation |

mosaiQTL provides what edgePython does *not*:

| Component | mosaiQTL module | Novel contribution |
|-----------|----------------|-------------------|
| $J$-founder haplotype design matrix | `genome_cache.py` | multiparent population structure |
| LOCO kinship + eigendecomposition | `kinship.py` | population structure correction |
| Haplotype score test at each locus | `nb_glm.py`, `scan.py` | $\mathbf{H}_\ell^{(-J)}$ plugged into NB/LMM |
| Dispersion vQTL (two-stage) | `vqtl.py` | per-subject $\hat{\phi}_i$ → LMM scan |
| GxE interaction scan | `gxe.py` | $\mathbf{H}_\ell \times$ environment |
| Trans-eQTL genome-wide scan | `trans.py` | all-chr × all-gene with LOCO |
| Splicing QTL (PSI phenotype) | `sqtl.py` | junction-level haplotype test |
| Allelic series / merge analysis | `merge.py` | founder grouping contrasts |
| LMM port of miQTL | `lmm.py` | validation against R reference |

The boundary is clean: edgePython handles *how to fit an NB model*;
mosaiQTL handles *what to test* (which haplotype, at which locus, under
which genetic model, with which kinship correction).

**Validation against miQTL remains necessary** for all haplotype-specific
code — the kinship, eigendecomposition, per-locus design matrix
construction, LOD computation, and founder effect estimation are
mosaiQTL-native and must match R miQTL output numerically.

---

## I. Statistical Models

All models in mosaiQTL share the same haplotype probability backbone
$\mathbf{H}_\ell \in [0, 1]^{n \times J}$, the same kinship
infrastructure, and the same scan loop. They differ only in the
likelihood function and link. The user selects the model at scan time
via `model='nb'`, `model='lmm'`, or `model='nb_lmm'`.

### Model A — Negative Binomial GLM (primary, `nb_glm.py`)

For pseudobulked raw counts with $n$ replicate-level observations (not
collapsed genotype means). This is the recommended default for
count-based phenotypes.

$$Y_i \sim \text{NB}(\mu_i, \; \phi)$$

$$\log \mu_i = \mathbf{h}_{i(\ell)}^\top \boldsymbol{\gamma} + \mathbf{x}_i^\top \boldsymbol{\alpha} + \log L_i$$

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $Y_i$ | scalar | Raw pseudobulk UMI count for sample $i$ |
| $\mu_i$ | scalar | Expected count |
| $\phi$ | scalar | Overdispersion ($\text{Var} = \mu + \phi \mu^2$) |
| $\mathbf{h}_{i(\ell)}$ | $(J-1) \times 1$ | Founder haplotype probs for sample $i$ at locus $\ell$ |
| $\boldsymbol{\gamma}$ | $(J-1) \times 1$ | Founder effects on expression (log-fold change) |
| $\mathbf{x}_i$ | $p \times 1$ | Covariates (intercept, sex, batch, expression PCs) |
| $L_i$ | scalar | Library size (total UMI for sample $i$), used as offset |

The NB GLM operates on all $n$ replicate samples directly. For the CC
proof-of-concept, $n \approx 418$ (48 genotypes $\times$ ~8.7 replicates),
sex is a covariate, and the kinship structure is handled by including
expression PCs or by an explicit random effect (see Model C).

**Test:** Score test (efficient) or likelihood ratio test (more powerful):

The **score test** evaluates the gradient of the log-likelihood at the
null ($\boldsymbol{\gamma} = \mathbf{0}$). For the NB log-likelihood
$\ell(\boldsymbol{\gamma})$:

$$T_\ell^{\text{score}} = \mathbf{U}(\hat{\boldsymbol{\theta}}_0)^\top \mathcal{I}(\hat{\boldsymbol{\theta}}_0)^{-1} \mathbf{U}(\hat{\boldsymbol{\theta}}_0) \sim \chi^2(J-1)$$

where $\mathbf{U}$ is the score vector and $\mathcal{I}$ is the Fisher
information, both evaluated at the null MLE $\hat{\boldsymbol{\theta}}_0$.
The null model is fit once per gene; per-locus evaluation requires only
matrix products against the haplotype design matrix.

The **LRT** fits both null and alternative models:

$$\Lambda_\ell = 2 \left[ \ell(\hat{\boldsymbol{\theta}}_1) - \ell(\hat{\boldsymbol{\theta}}_0) \right] \sim \chi^2(J-1)$$

More powerful than the score test but requires per-locus IRLS convergence.

**Dispersion estimation:** Gene-wise $\phi$ estimated once under the null
model using edgePython's three-stage empirical Bayes pipeline:

1. **Common** $\phi$: single value across all genes (starting point)
2. **Trended** $\tilde{\phi}_g$: smooth function of mean expression
   (captures the mean-variance relationship)
3. **Tagwise** $\hat{\phi}_g^{\text{shrunk}}$: per-gene estimate shrunk
   toward the trend via weighted likelihood empirical Bayes (WLEB):

$$\hat{\phi}_g^{\text{shrunk}} = \frac{d_0 \tilde{\phi}_g + d_g \hat{\phi}_g^{\text{raw}}}{d_0 + d_g}$$

where $d_0$ is the prior degrees of freedom and $d_g$ is the residual
df for gene $g$. This is edgePython's `estimate_disp()` — identical to
edgeR's Cox-Reid adjusted profile likelihood approach.

The shrunk $\hat{\phi}_g$ is held fixed during the per-locus genome scan.
This stabilizes p-values for low-count genes without introducing bias
for the haplotype effect test.

**Note on vQTL:** This gene-level shrinkage operates across *genes*
(borrowing information from genes with similar mean expression), not
across *subjects*. It does not compress the between-subject dispersion
variation that the vQTL scan is designed to detect.

**Phenotype input:** Raw pseudobulk UMI counts (integers). No RINT, no
log-CPM transformation. Library size $L_i$ enters as a known offset.

### Model B — Linear Mixed Model (miQTL port, `lmm.py`)

For RINT-transformed or normalized expression. This is the direct port
of miQTL `scan.h2lmm()` / qtl2 `scan1()`, provided for validation and
backward compatibility.

$$\mathbf{y} = \mathbf{G}_\ell \boldsymbol{\gamma} + \mathbf{X} \boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

$$\mathbf{u} \sim \mathcal{N}(\mathbf{0},\; \sigma^2_g \mathbf{K}), \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\; \sigma^2_e \mathbf{I}_n)$$

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $\mathbf{y}$ | $n \times 1$ | Phenotype vector (RINT or normalized expression) |
| $\mathbf{G}_\ell$ | $n \times q$ | Genetic predictor at locus $\ell$ (see §II) |
| $\boldsymbol{\gamma}$ | $q \times 1$ | Genetic effects to be tested |
| $\mathbf{X}$ | $n \times p$ | Covariate design matrix |
| $\mathbf{K}$ | $n \times n$ | Kinship matrix (LOCO) |
| $\sigma^2_g$ | scalar | Polygenic genetic variance |
| $\sigma^2_e$ | scalar | Residual variance |

**Phenotype input:** RINT-transformed log-CPM (continuous), or collapsed
genotype means for validation against R miQTL output. The LMM can operate
on either $n$ replicates (with expanded $\tilde{\mathbf{K}}_{n \times n}$)
or 48 genotype means (with the native $\mathbf{K}_{48 \times 48}$).

**Eigendecomposition acceleration (Kang et al. 2008):** specific to the
LMM. Precompute the spectral decomposition $\mathbf{K} = \mathbf{U}
\mathbf{D} \mathbf{U}^\top$ once per LOCO chromosome. Rotating all
terms by $\mathbf{U}^\top$ diagonalizes the covariance, reducing each
per-locus fit from $O(n^3)$ to $O(nq^2)$ weighted least squares. The
null $h^2$ is estimated once per chromosome via REML.

**Test:** $F$-test on RSS ratio:

$$\text{LOD}_\ell = \frac{n - p}{2} \log_{10} \left( \frac{\text{RSS}_0}{\text{RSS}_1} \right), \qquad F_\ell = \frac{(\text{RSS}_0 - \text{RSS}_1) / q}{\text{RSS}_1 / (n - p - q)}$$

### Model C — NB-LMM / NEBULA-LN (cell-level, `nebula_ln.py`)

For cell-level counts without pseudobulking. Uses edgePython's
`glm_sc_fit()` + `shrink_sc_disp()` for the NB-gamma mixed model
fitting and cell-level dispersion shrinkage, with mosaiQTL providing
the haplotype design matrix $\mathbf{H}_\ell^{(-J)}$ at each locus.

See §III for full model specification.

---

## II. Genetic Predictor Frameworks

All three models share the same haplotype probability backbone. The
frameworks below define how $\mathbf{G}_\ell$ (or equivalently
$\mathbf{h}_{i(\ell)}$ in the NB GLM) is constructed at each locus.
They are model-agnostic — any framework can be used with any of
Models A, B, or C.

### A. Additive Haplotype (implemented ✓)

The default framework, matching miQTL `scan.h2lmm()`. At each locus
$\ell$, the genetic predictor is the $(J - 1)$-column founder haplotype
probability matrix (dropping one founder for identifiability):

$$\mathbf{G}_\ell = \mathbf{H}_\ell^{(-J)} \in \mathbb{R}^{n \times (J-1)}$$

where $\mathbf{H}_\ell \in [0, 1]^{n \times J}$ contains the
probability that subject $i$ carries founder $f$'s haplotype at locus
$\ell$, with $\sum_{f=1}^{J} H_{i\ell f} = 1$. The number of founders
$J$ and which founder to drop are user-specified at scan time.

**Test:** $F(J - 1, \; n - p - (J - 1))$ — tests whether *any* founder
has a differential effect.

**Cost:** $O(n \cdot (J-1)^2)$ per locus.

**Proof-of-concept:** For CC/DO mice, $J = 8$, $q = 7$, drop WSB/EiJ
by convention. For a 4-way MAGIC, $J = 4$, $q = 3$.

### B. Biallelic SNP Association (design matrix swap)

Impute SNP genotypes from the founder haplotypes. If founders are
partitioned into reference ($r = 0$) and alternative ($a = 1$) alleles
at a biallelic SNP, the dosage for subject $i$ is:

$$g_{i\ell} = \sum_{f \in \mathcal{A}} H_{i\ell f}$$

where $\mathcal{A}$ is the set of founders carrying the alternative
allele. The genetic predictor becomes scalar:

$$\mathbf{G}_\ell = \mathbf{g}_\ell \in \mathbb{R}^{n \times 1}, \qquad q = 1$$

**Test:** $F(1, \; n - p - 1)$ — standard single-SNP association.

**Proximity:** Trivial. Replace $\mathbf{H}_\ell^{(-J)}$ with
$\mathbf{g}_\ell$ in the existing `fit_lmm_at_locus()` call. No
changes to the eigendecomposition or scan loop.

**Use case:** Fine-mapping within a QTL region using founder-imputed
SNPs (matching qtl2 SNP association scans, Broman et al. 2019).

### C. Full Diplotype Model (design matrix swap)

For heterozygous populations (DO mice, F$_1$ hybrids), model all
$\binom{J}{2} + J$ unordered diplotype states:

$$\mathbf{G}_\ell = \mathbf{P}_\ell^{(-1)} \in \mathbb{R}^{n \times (d-1)}$$

where $d = \binom{J}{2} + J$ and $P_{i\ell d}$ is the probability that
subject $i$ carries diplotype $d$ (founder pair $f_1 / f_2$) at locus
$\ell$.

**Test:** $F(d - 1, \; n - p - (d - 1))$ — captures dominance and
epistatic interactions between founder alleles.

**Proof-of-concept:** For CC/DO with $J = 8$: $d = 36$, $q = 35$.
For a 4-way cross: $d = 10$, $q = 9$.

### D. Allelic Series / Merge Analysis (contrast matrix)

Test whether subsets of founders share effects. Instead of the full
$(J - 1)$ df haplotype test, define a grouping
$\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$ that partitions founders
into $k$ functional classes, then:

$$\mathbf{G}_\ell = \mathbf{H}_\ell \mathbf{C} \in \mathbb{R}^{n \times (k-1)}$$

where $\mathbf{C} \in \{0, 1\}^{J \times (k-1)}$ is the contrast
matrix mapping founders to groups (dropping one group for
identifiability), and $q = k - 1$.

**Example (CC, $J = 8$):** Group wild-derived founders (CAST, PWK, WSB)
vs. classical lab strains (A/J, B6, 129, NOD, NZO) → $k = 2$, $q = 1$:

$$g_{i\ell} = H_{i\ell,\text{CAST}} + H_{i\ell,\text{PWK}} + H_{i\ell,\text{WSB}}$$

**Proximity:** Matrix multiply $\mathbf{H}_\ell \mathbf{C}$ before
passing to `fit_lmm_at_locus()`. Enumerate candidate groupings
programmatically using the Stirling partition numbers
$S(J, k)$ for $k = 2, \ldots, J - 1$.

**Use case:** Increase power when the true allelic series has fewer
than 8 distinct effect levels, which is biologically common
(Keele et al. 2020).

### E. Haley-Knott Regression (simplification)

Drop the kinship term entirely ($\sigma^2_g = 0$):

$$\mathbf{y} = \mathbf{G}_\ell \boldsymbol{\gamma} + \mathbf{X} \boldsymbol{\alpha} + \boldsymbol{\varepsilon}, \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \; \sigma^2 \mathbf{I}_n)$$

**Proximity:** Set $h^2 = 0$ in the existing LMM code (all weights
become 1). No eigendecomposition needed.

**Use case:** Populations with minimal relatedness, or as a fast
first-pass scan before LMM refinement. Matches qtl2 `scan1()` without
the `kinship` argument.

### F. Score Test (alternative statistic)

Instead of fitting the full alternative model at every locus, evaluate
the gradient of the log-likelihood under the null:

$$T_\ell^{\text{score}} = \frac{\left[ \boldsymbol{\gamma}^\top \mathbf{G}_\ell^{*\top} \mathbf{W}^{-1} \hat{\boldsymbol{\varepsilon}}_0 \right]^2}{\boldsymbol{\gamma}^\top \mathbf{G}_\ell^{*\top} \mathbf{W}^{-1} \mathbf{G}_\ell^* \boldsymbol{\gamma}}$$

where $\hat{\boldsymbol{\varepsilon}}_0 = \mathbf{y}^* - \mathbf{X}^* \hat{\boldsymbol{\alpha}}_0$ are null residuals and
$\mathbf{W} = \text{diag}(\delta d_i + 1)$.

**Proximity:** Moderate. Requires computing the null residuals once per
chromosome (already done during $h^2$ estimation) and then a single
matrix-vector product per locus — no per-locus least squares solve.

**Use case:** Order-of-magnitude speedup for genome-wide scans with
many loci (e.g., trans-eQTL with $\sim 10^5$ tests per gene). The
score test is asymptotically equivalent to the F-test.

---

## III. Cell-Level Negative Binomial Model (NEBULA-LN)

For single-cell eQTL mapping *without* pseudobulking. For cell $j$ in
subject $i$, gene $g$:

$$Y_{ij} \mid u_i \sim \text{NB}(\mu_{ij}, \; \phi)$$

$$\log \mu_{ij} = \mathbf{x}_{ij}^\top \boldsymbol{\beta} + \mathbf{h}_{i(\ell)}^\top \boldsymbol{\gamma} + u_i + \log L_{ij}$$

$$\exp(u_i) \sim \text{Gamma}\!\left(\frac{1}{\sigma^2}, \; \frac{1}{\sigma^2}\right)$$

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $Y_{ij}$ | scalar | Raw UMI count, cell $j$, subject $i$ |
| $\mu_{ij}$ | scalar | Expected count |
| $\phi$ | scalar | Cell-level overdispersion (NB) |
| $\sigma^2$ | scalar | Subject-level overdispersion (random effect variance) |
| $\mathbf{x}_{ij}$ | $p \times 1$ | Cell-level + subject-level covariates |
| $\mathbf{h}_{i(\ell)}$ | $(J-1) \times 1$ | Founder haplotype probs for subject $i$ at locus $\ell$ |
| $\boldsymbol{\gamma}$ | $(J-1) \times 1$ | Founder effects on mean expression |
| $L_{ij}$ | scalar | Library size (total UMI for cell $j$) |
| $u_i$ | scalar | Subject-level random effect |

The NB variance decomposes into technical and biological components:

$$\text{Var}(Y_{ij} \mid u_i) = \mu_{ij} + \phi \, \mu_{ij}^2$$

The NEBULA-LN approximation leverages large cells-per-subject (CPS):
when CPS is large, the marginal log-likelihood per subject can be
analytically approximated via the law of large numbers, avoiding the
Laplace approximation (He et al. 2021).

### Sufficient Statistics (precomputed once per gene)

For each subject $i$:

$$S_i = \sum_{j=1}^{n_i} Y_{ij}, \qquad T_i = \sum_{j=1}^{n_i} \log \Gamma(Y_{ij} + 1), \qquad n_i = \text{cells per subject}$$

These collapse cell-level data to subject-level summaries. The per-locus
optimization updates only $\boldsymbol{\gamma}$ (haplotype effects) and
re-estimates $\sigma^2$ and $\phi$.

---

## IV. Dispersion vQTL

Maps genetic effects on expression *noise* rather than mean expression.
This is the key extension requiring single-cell data — pseudobulking
destroys within-subject variance information.

### Two-Stage Approach (implemented ✓)

**Stage 1:** Estimate per-subject overdispersion $\hat{\phi}_i$ via
method-of-moments:

$$\hat{\phi}_i = \frac{\text{Var}_j(Y_{ij}) - \bar{Y}_i}{\bar{Y}_i^2}$$

or via per-subject NB maximum likelihood.

**Stage 2:** Scan $\log \hat{\phi}_i$ as a continuous phenotype through
the standard LMM haplotype scan:

$$\log \hat{\phi}_i = \mathbf{H}_{i(\ell)}^\top \boldsymbol{\beta}_v + \mathbf{x}_i^\top \boldsymbol{\alpha}_v + u_{v,i} + \varepsilon_{v,i}$$

A significant hit means founder haplotype at locus $\ell$ controls the
*variability* of gene expression across cells within a subject.

### Joint Approach (future)

Extend NEBULA to model the overdispersion itself as a function of
haplotype:

$$\log \phi_i = \mathbf{h}_{i(\ell)}^\top \boldsymbol{\delta}$$

and test $H_0: \boldsymbol{\delta} = \mathbf{0}$. This is a double
GLMM — computationally heavier but avoids two-stage estimation error.

---

## V. Planned Extensions

### GxE Interaction QTL (`gxe.py`)

Add founder haplotype × environment interaction terms:

$$\mathbf{y} = \mathbf{H}_\ell^{(-J)} \boldsymbol{\beta} + \left(\mathbf{H}_\ell^{(-J)} \odot \mathbf{e}\right) \boldsymbol{\delta} + \mathbf{X} \boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

where $\mathbf{e}$ is the environment vector (e.g., sex coded as 0/1)
and $\odot$ denotes element-wise product broadcast across columns.

**Test:** $F(J - 1, \; n - p - 2(J - 1))$ on
$\boldsymbol{\delta} = \mathbf{0}$.

**Proximity:** Append $\mathbf{H}_\ell^{(-J)} \odot \mathbf{e}$ to the
design matrix. The eigendecomposition trick applies unchanged since
$\mathbf{K}$ is unaffected.

### Trans-eQTL (`trans.py`)

Scan all chromosomes for a gene, not just the cis-window. For a gene on
chromosome $c$, when testing a locus on chromosome $c'$, use the LOCO
kinship excluding $c'$ (not $c$):

$$\mathbf{K}^{(-c')} = \frac{1}{L_{-c'}} \sum_{\ell \notin c'} \mathbf{H}_\ell \mathbf{H}_\ell^\top$$

**Proximity:** Structurally identical to cis scan — iterate over all
LOCO chromosomes instead of one. Multiple testing correction via
Bonferroni across $\sim 19{,}000$ loci per gene, or permutation.

### Splicing QTL (`sqtl.py`)

Phenotype is percent-spliced-in (PSI) per junction per subject:

$$\psi_{ig} = \frac{\text{inclusion}_{ig}}{\text{inclusion}_{ig} + \text{exclusion}_{ig}}$$

After RINT, $\psi$ plugs directly into `scan_h2lmm()` as a continuous
phenotype. No changes to the LMM engine.

### Conditional / Multi-QTL Scans

After identifying a primary QTL at locus $\ell_1$, add it as a fixed
covariate and re-scan:

$$\mathbf{y} = \mathbf{H}_{\ell_1}^{(-J)} \boldsymbol{\gamma}_1 + \mathbf{H}_\ell^{(-J)} \boldsymbol{\gamma}_2 + \mathbf{X} \boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

**Proximity:** Append $\mathbf{H}_{\ell_1}$ to $\mathbf{X}$ and re-run
the genome scan. Matches miQTL `condition.out.locus.for.scan()`.

### Deferred Extensions

| Extension | Model Class | Proximity | Notes |
|-----------|-------------|-----------|-------|
| Fine-mapping (SuSiE) | Bayesian credible sets | Moderate | Operates on scan output, not the LMM engine |
| Colocalization | Bayesian hypothesis test | Moderate | Compares two scan result objects |
| Allele-specific expression | Binomial test | New model | Leverages F$_1$ heterozygosity |
| Pseudotime QTL | LMM with pseudotime covariate | Low | $\mathbf{G}_\ell \times t$ interaction |
| Composition QTL | Dirichlet regression | New model | Cell-type proportions as multivariate phenotype |
| Multivariate joint model | MANOVA-LMM | Moderate | Simultaneous mapping across cell types |
| Bayesian haplotype effects | BayesB/C shrinkage | New model | Hierarchical prior on $\boldsymbol{\gamma}$ |

---

## VI. Kinship

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

## VII. Data Flow

```
h5ad (cells × genes, with cell-type annotations)
  ↓
Cell-identity resolution choice:
  ├─ lineage level (e.g., all myonuclei = type1 + type2a + type2b + type2x)
  ├─ subtype level (e.g., type2b only)
  └─ cell level (no aggregation — NEBULA-LN / vQTL)
  ↓
┌───────────────────────────────────────────────────────────────┐
│ PSEUDOBULK PATH (lineage or subtype)                         │
│   aggregate counts per sample × chosen cell identity         │
│   → TMM normalize → log-CPM → RINT per gene                 │
│   → scan.scan_h2lmm()  [highest power for mean eQTL]        │
└───────────────────────────────────────────────────────────────┘
┌───────────────────────────────────────────────────────────────┐
│ CELL-LEVEL PATH (no pseudobulking)                           │
│   nebula_ln.precompute_sufficient_stats()                    │
│   → nebula_ln.fit_nebula_ln()  [mean eQTL on raw counts]    │
│   → vqtl.scan_vqtl()  [dispersion eQTL — requires cells]   │
└───────────────────────────────────────────────────────────────┘
  ↓
genome_cache.load()                  → H_ℓ ∈ ℝ^{n × J} per locus
  ↓
kinship.calc_kinship_loco()          → {chr: K^{(-c)}} + eigendecompositions
  ↓
scan.scan_h2lmm()       → cis-eQTL:  LOD, p, γ (J−1 founder effects)
vqtl.scan_vqtl()        → vQTL:      LOD for dispersion effect
gxe.scan_gxe()          → GxE:       LOD for haplotype × environment
trans.scan_trans()       → trans:     genome-wide LOD
sqtl.scan_sqtl()        → sQTL:      LOD for splicing effect
snp_scan.scan_snp()     → SNP assoc: LOD per imputed variant
merge.scan_merge()      → allelic:   LOD per candidate founder grouping
```

---

## VIII. Compute Cost Estimates

Benchmarked at $n = 48$ (CC proof-of-concept, $J = 8$). Costs scale as
$O(n (J-1)^2)$ per locus for the additive haplotype model, and
$O(n q^2)$ in general for any design matrix with $q$ genetic parameters.

| Step | Per gene ($n = 48$) | 20k genes | Notes |
|------|---------------------|-----------|-------|
| REML $h^2$ estimation (19 chr) | 8.9 ms | 3 min | once per chr per gene |
| Additive haplotype scan (1k loci/chr) | 0.57 s | 3.2 hr | $q = J - 1$ |
| SNP association scan (10k SNPs/chr) | 0.19 s | 1.1 hr | $q = 1$ |
| Two-stage vQTL | 0.57 s | 3.2 hr | same LMM engine |
| NEBULA-LN (cell-level, $\sim 200$ CPS) | ~38 s | ~210 hr | SLURM parallel |
| PGMM (Poisson-gamma, closed-form) | ~1.9 s | ~10.5 hr | no cell-level $\phi$ |

NEBULA-LN is ~20× slower than LMM per locus. The two-stage vQTL uses
the cheap LMM scan on pre-estimated $\hat{\phi}$, making it the
pragmatic first implementation. Full NEBULA-LN is parallelizable across
genes via SLURM array jobs.

---

## IX. Validation Strategy

For every module, the validation script:

1. Generates a simulated CC dataset (8 founders, 7 F$_1$s, 33 CC strains,
   8 replicates per strain = 384 samples)
2. Runs the computation in both Python (mosaiQTL) and R (miQTL/qtl2)
3. Compares outputs numerically ($r > 0.999$, $\max |\Delta| < 10^{-6}$)

| Module | Status | Reference |
|--------|--------|-----------|
| Kinship (48×48 LOCO) | ✓ validated | `qtl2::calc_kinship()` |
| K expansion (48→384) | ✓ validated | R: `k_chr[strain_idx, strain_idx]` |
| Null $h^2$ estimation | ✓ validated | `qtl2::scan1()` null fit |
| LOD scores per locus | ✓ validated | `miQTL::scan.h2lmm()` |
| Dispersion $\hat{\phi}$ estimation | ✓ validated | moment estimator |
| vQTL scan | ✓ validated | two-stage on simulated data |
| NB GLM dispersion (edgePython) | planned | `edgePython::estimate_disp()` vs `edgeR` |
| NB GLM score test | planned | `edgePython::glm_fit()` + custom score |
| NEBULA-LN $\sigma^2$, $\phi$ | planned | `edgePython::glm_sc_fit()` vs `nebula::nebula()` |
| Permutation $p$-values | planned | `miQTL` permutation scan |
| SNP association | planned | `qtl2::scan1snps()` |
| Merge analysis | planned | manual contrast validation |

---

## X. References

- Pachter, L. (2026). Differential analysis of genomics count data with
  edge*. *bioRxiv*.
  [doi:10.64898/2026.02.16.706223](https://doi.org/10.64898/2026.02.16.706223)
- He, L. et al. (2021). NEBULA: fast negative binomial mixed model for
  multi-subject single-cell data. *Commun. Biol.* 4:629.
  [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)
- Kang, H.M. et al. (2008). Efficient control of population structure in
  model organism association mapping. *Genetics* 178:1709–1723.
  [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101)
- Keele, G.R. et al. (2020). Determinants of QTL mapping power in the
  realized Collaborative Cross. *G3* 9:1707–1727.
  [doi:10.1534/g3.119.400194](https://doi.org/10.1534/g3.119.400194)
- Keele, G.R. et al. (2020). Genetic fine-mapping and identification of
  candidate genes and variants for adiposity traits in outbred rats.
  *PLOS Genet.* 16(1):e1008537.
  [doi:10.1371/journal.pgen.1008537](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. et al. (2019). R/qtl2: software for mapping quantitative
  trait loci with high-dimensional data and multiparent populations.
  *Genetics* 211:495–502.
  [doi:10.1534/genetics.118.301595](https://doi.org/10.1534/genetics.118.301595)
- Haley, C.S. & Knott, S.A. (1992). A simple regression method for
  mapping quantitative trait loci in line crosses using flanking markers.
  *Heredity* 69:315–324.
  [doi:10.1038/hdy.1992.131](https://doi.org/10.1038/hdy.1992.131)
- King, E.G. et al. (2014). Bayesian modeling of haplotype effects in
  multiparent populations. *Genetics* 198:139–156.
  [doi:10.1534/genetics.114.166249](https://doi.org/10.1534/genetics.114.166249)
- Robinson, M.D. & Smyth, G.K. (2008). Small-sample estimation of
  negative binomial dispersion, with applications to SAGE data.
  *Biostatistics* 9:321–332.
  [doi:10.1093/biostatistics/kxm030](https://doi.org/10.1093/biostatistics/kxm030)
- McCarthy, D.J., Chen, Y. & Smyth, G.K. (2012). Differential expression
  analysis of multifactor RNA-Seq experiments with respect to biological
  variation. *Nucleic Acids Res.* 40:4288–4297.
  [doi:10.1093/nar/gks042](https://doi.org/10.1093/nar/gks042)

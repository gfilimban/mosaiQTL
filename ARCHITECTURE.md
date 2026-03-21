# mosaiQTL — Haplotype-Based QTL Mapping for Multiparent Populations

## Overview

mosaiQTL is a Python package for QTL mapping in multiparent populations
(CC, DO, MAGIC, heterogeneous stocks), designed as a hub for multiple
mapping frameworks unified by a shared genotype probability backbone.

The package supports haplotype-based, SNP-based, and cell-level models
under a common eigendecomposition-accelerated mixed model engine, with
extensions for dispersion QTL, genotype–environment interaction, trans
mapping, and splicing QTL.

---

## Package Structure

```
mosaiqtl/
  __init__.py
  genome_cache.py     # read/write HAPPY genome cache (RDS, HDF5)
  kinship.py          # kinship: overall, LOCO, eigen decomposition
  lmm.py              # fast LMM engine (eigendecomposition trick)
  nebula_ln.py        # NEBULA-LN: NB-gamma mixed model for cell-level data
  scan.py             # genome-wide scan orchestrator (≈ scan.h2lmm)
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

---

## I. Core Linear Mixed Model

All mapping frameworks in mosaiQTL share a common LMM engine. The
general model for $n$ subjects, phenotype $\mathbf{y}$, and genetic
predictor $\mathbf{G}_\ell$ at locus $\ell$ is:

$$\mathbf{y} = \mathbf{G}_\ell \boldsymbol{\gamma} + \mathbf{X} \boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

$$\mathbf{u} \sim \mathcal{N}(\mathbf{0},\; \sigma^2_g \mathbf{K}), \qquad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\; \sigma^2_e \mathbf{I}_n)$$

| Symbol | Dimension | Description |
|--------|-----------|-------------|
| $\mathbf{y}$ | $n \times 1$ | Phenotype vector (one gene, RINT-transformed) |
| $\mathbf{G}_\ell$ | $n \times q$ | Genetic predictor at locus $\ell$ (depends on framework; see §II) |
| $\boldsymbol{\gamma}$ | $q \times 1$ | Genetic effects to be tested |
| $\mathbf{X}$ | $n \times p$ | Covariate design matrix (intercept, sex, batch, PCs) |
| $\boldsymbol{\alpha}$ | $p \times 1$ | Covariate effects (nuisance) |
| $\mathbf{K}$ | $n \times n$ | Kinship matrix (LOCO) |
| $\sigma^2_g$ | scalar | Polygenic genetic variance |
| $\sigma^2_e$ | scalar | Residual variance |

Defining heritability $h^2 = \sigma^2_g / (\sigma^2_g + \sigma^2_e)$
and ratio $\delta = h^2 / (1 - h^2)$, the covariance of $\mathbf{y}$
under the null ($\boldsymbol{\gamma} = \mathbf{0}$) is:

$$\text{Var}(\mathbf{y}) = \sigma^2_e \left( \delta \mathbf{K} + \mathbf{I}_n \right)$$

### Eigendecomposition Acceleration (Kang et al. 2008)

Precompute the spectral decomposition of the kinship matrix once per
LOCO chromosome:

$$\mathbf{K} = \mathbf{U} \mathbf{D} \mathbf{U}^\top$$

where $\mathbf{D} = \text{diag}(d_1, \ldots, d_n)$ are eigenvalues and
$\mathbf{U}$ is the orthogonal eigenvector matrix. Rotating all terms
by $\mathbf{U}^\top$:

$$\mathbf{y}^* = \mathbf{U}^\top \mathbf{y}, \quad \mathbf{G}^*_\ell = \mathbf{U}^\top \mathbf{G}_\ell, \quad \mathbf{X}^* = \mathbf{U}^\top \mathbf{X}$$

The covariance becomes diagonal:

$$\text{Var}(\mathbf{y}^*) = \sigma^2_e \, \text{diag}(\delta d_1 + 1, \ldots, \delta d_n + 1)$$

This reduces each per-locus fit from $O(n^3)$ matrix inversion to
$O(n q^2)$ weighted least squares. The null $h^2$ is estimated once per
chromosome via REML profile likelihood optimization over $h^2 \in (0, 1)$.

### Test Statistics

The **LOD score** compares the alternative (with $\mathbf{G}_\ell$) to
the null (without) via the ratio of weighted residual sums of squares:

$$\text{LOD}_\ell = \frac{n - p}{2} \log_{10} \left( \frac{\text{RSS}_0}{\text{RSS}_1} \right)$$

The **$F$-test** provides a $p$-value under the null:

$$F_\ell = \frac{(\text{RSS}_0 - \text{RSS}_1) / q}{\text{RSS}_1 / (n - p - q)} \sim F(q, \; n - p - q)$$

where $q$ is the number of genetic parameters being tested (varies by
framework — see §II).

---

## II. Mapping Frameworks

Each framework plugs into the same LMM engine by defining
$\mathbf{G}_\ell$ and $q$ differently. They are ordered below by
proximity to the current implementation — trivial swaps first, larger
extensions last.

### A. Additive Haplotype (implemented ✓)

The default framework, matching miQTL `scan.h2lmm()`. At each locus
$\ell$, the genetic predictor is the $J - 1$ founder haplotype
probability matrix (dropping one founder for identifiability):

$$\mathbf{G}_\ell = \mathbf{H}_\ell^{(-J)} \in \mathbb{R}^{n \times (J-1)}$$

where $\mathbf{H}_\ell \in [0, 1]^{n \times J}$ contains the
probability that subject $i$ carries founder $f$'s haplotype at locus
$\ell$, with $\sum_{f=1}^{J} H_{i\ell f} = 1$. For CC/DO mice,
$J = 8$ and $q = 7$.

**Test:** $F(7, \; n - p - 7)$ — tests whether *any* founder has
a differential effect.

**Cost:** $O(n \cdot 7^2)$ per locus.

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
$\binom{J}{2} + J = 36$ unordered diplotype states:

$$\mathbf{G}_\ell = \mathbf{P}_\ell^{(-1)} \in \mathbb{R}^{n \times 35}$$

where $P_{i\ell d}$ is the probability that subject $i$ carries
diplotype $d$ (founder pair $f_1 / f_2$) at locus $\ell$.

**Test:** $F(35, \; n - p - 35)$ — captures dominance and epistatic
interactions between founder alleles.

**Proximity:** Design matrix swap only. The LMM engine handles
arbitrary $q$. For CC inbred lines this is unnecessary (homozygous),
but becomes essential for DO or F$_2$ populations.

### D. Allelic Series / Merge Analysis (contrast matrix)

Test whether subsets of founders share effects. Instead of the full
$J - 1 = 7$ df haplotype test, define a grouping
$\mathcal{S} = \{S_1, S_2, \ldots, S_k\}$ that partitions founders
into $k$ functional classes, then:

$$\mathbf{G}_\ell = \mathbf{H}_\ell \mathbf{C} \in \mathbb{R}^{n \times (k-1)}$$

where $\mathbf{C} \in \{0, 1\}^{J \times (k-1)}$ is the contrast
matrix mapping founders to groups (dropping one group for
identifiability), and $q = k - 1$.

**Example:** Group wild-derived founders (CAST, PWK, WSB) vs. classical
lab strains (A/J, B6, 129, NOD, NZO) → $k = 2$, $q = 1$:

$$g_{i\ell} = H_{i\ell,\text{CAST}} + H_{i\ell,\text{PWK}} + H_{i\ell,\text{WSB}}$$

**Proximity:** Matrix multiply $\mathbf{H}_\ell \mathbf{C}$ before
passing to `fit_lmm_at_locus()`. Enumerate candidate groupings
programmatically using the Stirling partition numbers
$S(8, k)$ for $k = 2, \ldots, 7$.

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

**Test:** $F(J - 1, \; n - p - 2(J-1))$ on
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
h5ad (cells × genes)
  ↓
io.load_sc_data()                    → count matrix + subject IDs + covariates
  ↓
nebula_ln.precompute_sufficient_stats() → S_i, T_i, n_i per subject
  ↓
genome_cache.load()                  → H_ℓ ∈ ℝ^{48 × 8} per locus
  ↓
kinship.calc_kinship_loco()          → {chr: K^{(-c)}} + eigendecompositions
  ↓
scan.scan_h2lmm()       → cis-eQTL:  LOD, p, γ (founder effects)
vqtl.scan_vqtl()        → vQTL:      LOD for dispersion effect
gxe.scan_gxe()          → GxE:       LOD for haplotype × sex interaction
trans.scan_trans()       → trans:     genome-wide LOD
sqtl.scan_sqtl()        → sQTL:      LOD for splicing effect
snp_scan.scan_snp()     → SNP assoc: LOD per imputed variant
merge.scan_merge()      → allelic:   LOD per candidate grouping
```

---

## VIII. Compute Cost Estimates (type2b myonuclei, ~480 subjects)

| Step | Per gene | 20k genes |
|------|----------|-----------|
| Precompute sufficient stats ($S_i$, $T_i$, $n_i$) | ~0.5 s | ~2.8 hr |
| Kinship eigendecomposition (19 LOCO) | one-time | ~30 s |
| Additive haplotype scan (1000 loci/chr, $q=7$) | ~0.1 s/locus | ~5.5 hr |
| SNP association scan (10k SNPs/chr, $q=1$) | ~0.01 s/locus | ~5.5 hr |
| Score test scan (1000 loci/chr) | ~0.02 s/locus | ~1.1 hr |
| NEBULA-LN scan (1000 loci/chr) | ~2 s/locus | ~110 hr |
| Two-stage vQTL (LMM on $\log \hat{\phi}$) | ~0.1 s/locus | ~5.5 hr |

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
| NEBULA-LN $\sigma^2$, $\phi$ | planned | `nebula::nebula()` |
| Permutation $p$-values | planned | `miQTL` permutation scan |
| SNP association | planned | `qtl2::scan1snps()` |
| Merge analysis | planned | manual contrast validation |

---

## X. References

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

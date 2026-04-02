# mosaiQTL — Proof-of-Concept Report

**Repo:** `github.com/gfilimban/mosaiqtl` (private)  
**Scope:** Novel extensions beyond the miQTL validation port  
**Status:** Scaffolded / design-complete; pending end-to-end validation  
**Last updated:** April 2026 | commit `04eaeea`

---

## Overview

The miQTL validation port (S1–S11) establishes mathematical equivalence with the
R reference. This report documents the novel mosaiQTL extensions that go beyond
miQTL: a negative binomial primary model, genome-wide dispersion scan, and planned
epistasis module. Each section states the scientific motivation, the statistical
model, the implementation design, and the planned validation criteria.

---

## Shared Architectural Note — Z-Matrix

All models below operate parametrically on $n$ samples and $m$ genotyped strains
($n \geq m$). The Z-matrix $Z \in \{0,1\}^{n \times m}$ ($Z_{ik} = \mathbf{1}[\text{sample } i \text{ belongs to strain } k]$)
is built at scan time from `sample_ids` vs `genome_cache.sample_ids`. When $n = m$
and IDs match, $Z = I_m$ and all formulas reduce to the strain-mean case exactly.
No module hardcodes $n$ or $m$.

---

## P1 — NB GLM Scan (`scan_nb`)

### Purpose

The LMM assumes $y \sim \mathcal{N}$ after RINT. Pseudobulk snRNA-seq counts are
non-negative integers with overdispersion — RINT discards this structure. The NB
GLM is the natural primary model: it handles zero-inflation, library-size offsets,
and overdispersion natively, with no normality assumption.

---

### Model

**Null model** (covariates only):

$$Y_i \sim \text{NB}(\mu_i^0,\, \phi), \quad \log \mu_i^0 = X_i\,\beta_0 + \log s_i$$

where $X_i \in \mathbb{R}^p$ = intercept + batch + PCs, $s_i$ = library size offset,
$\phi$ = dispersion ($\text{Var}[Y_i] = \mu_i + \phi\mu_i^2$).

**Alternative model** at locus $\ell$:

$$\log \mu_i^\ell = H_i^\ell\,\gamma + X_i\,\beta + \log s_i$$

where $H_i^\ell = (Z H_m^\ell)_i \in \mathbb{R}^{J-1}$ are founder haplotype
dosages after Z-expansion (last founder dropped, linear constraint).

**NB log-likelihood:**

$$\ell(y \mid \mu, \phi) = \sum_i \Bigl[\ln\Gamma\!\bigl(y_i + \phi^{-1}\bigr) - \ln\Gamma\!\bigl(\phi^{-1}\bigr) + \phi^{-1}\ln\!\tfrac{\phi^{-1}}{\phi^{-1}+\mu_i} + y_i\ln\!\tfrac{\mu_i}{\phi^{-1}+\mu_i} - \ln\Gamma(y_i+1)\Bigr]$$

---

### Score Test (default, vectorised)

Null is fitted once per gene. Score statistic at locus $\ell$:

$$Q_\ell = U_\ell^\top I_\ell^{-1} U_\ell \;\sim\; \chi^2(J-1) \quad \text{under } H_0$$

$$U_\ell = H_\ell^\top r, \quad r_i = \frac{y_i - \hat\mu_i^0}{1 + \phi\hat\mu_i^0}$$

$$I_\ell = H_\ell^\top \operatorname{diag}(w)\, H_\ell, \quad w_i = \frac{\hat\mu_i^0}{1 + \phi\hat\mu_i^0}$$

$$\text{LOD}_\ell = \frac{Q_\ell}{2\ln 10}$$

All loci on a chromosome are evaluated simultaneously via `einsum` on
$H_\text{all} \in \mathbb{R}^{L \times n \times (J-1)}$. Complexity per gene:
$O(L \times n \times J^2)$ — analogous to FWL vectorisation in `scan_chromosome`.

**Difference from LRT:** $Q_\ell$ is asymptotically equivalent to the LRT statistic
but does not require fitting the alternative model. Exact when $n \to \infty$;
conservative for small $n$.

---

### LRT (optional, exact betas)

$$\text{LOD}_\ell = \frac{\ell_\text{alt}^\ell - \ell_\text{null}}{\ \ln 10\ }, \quad \text{LRT} = 2(\ell_\text{alt}^\ell - \ell_\text{null}) \;\sim\; \chi^2(J-1)$$

Alt model fitted per locus via `edgepython.glm_fit`. Returns log-scale founder
haplotype effects $\hat\gamma \in \mathbb{R}^{J-1}$. Complexity $O(L \times \text{IRLS} \times n)$
— use at peak loci only after score-test pre-screening.

---

### Dispersion Estimation

Estimated once per gene from the null model via edgePython's three-stage EB pipeline:

1. Common $\hat\phi_c$ across all genes (moment estimator)
2. Trended $\hat\phi_t(g)$ as function of mean expression
3. Tagwise $\hat\phi_g$ shrunk toward trend (WLEB)

$\hat\phi_g$ is fixed across all loci for gene $g$, matching the `glmQLFit`
convention in edgeR and avoiding $L$ separate dispersion fits.

---

### Implementation

| Function | Location | Status |
|---|---|---|
| `scan_nb()` | `mosaiqtl/nb.py` | ✅ scaffolded |
| `scan_nb_multi_gene()` | `mosaiqtl/nb.py` | ✅ scaffolded |
| `_nb_loglik()` | `mosaiqtl/nb.py` | ✅ implemented |
| `_score_stat_vectorised()` | `mosaiqtl/nb.py` | ✅ implemented |
| `estimate_gene_dispersion()` | `mosaiqtl/nb.py` | ✅ implemented |

**Pending:** end-to-end run on real pseudobulk data; comparison of NB LOD profiles
against LMM LOD profiles for Piezo1, Ccdc171, Pmis2.

---

### Planned Validation Criteria

| Check | Threshold |
|---|---|
| Score test $\approx$ LRT at peak loci (large $n$) | $|\Delta\text{LOD}| < 0.5$ |
| NB LOD rank-order agrees with LMM LOD rank-order across genes | Spearman $\rho > 0.7$ |
| $p$-value calibration under permuted null | $\lambda_\text{GC} \in [0.95, 1.10]$ |
| Dispersion estimates $\hat\phi$ match edgeR output | $r > 0.99$ |

---

## P2 — Dispersion QTL Scan (`vqtl.py`)

### Purpose

Test whether genetic variation at a locus affects not just the mean expression level
($\mu$-QTL) but the *variance* of expression across replicates. A dispersion QTL
(vQTL) at locus $\ell$ indicates that founder haplotype identity modulates
transcriptional noise.

---

### Model

For each strain $k$, observe replicate counts $\{Y_{ik}\}_{i=1}^{n_k}$.  
Per-strain dispersion: $\hat\phi_k$ from moment estimator or NB MLE.

**Null:** $\hat\phi_k \sim \log\mathcal{N}(\mu_\phi, \sigma^2_\phi)$ — common log-dispersion.

**Alternative at locus $\ell$:** $\log\hat\phi_k = H_k^\ell\,\delta + \epsilon_k$,
$\epsilon_k \sim \mathcal{N}(0, \sigma^2)$ — haplotype effects on log-dispersion.

$$F_\ell = \frac{(RSS_0 - RSS_1)/( J-1)}{RSS_1/(m - p - J + 1)} \;\sim\; F(J-1,\, m-p-J+1)$$

$$\text{LOD}_\ell = \frac{m}{2}\log_{10}\!\frac{RSS_0}{RSS_1}$$

**Difference from $\mu$-QTL:** response variable is $\log\hat\phi_k$ (per-strain
dispersion), not $Y_i$ (counts). Operates in strain space ($m$ observations), not
sample space ($n$).

---

### Implementation

| Function | Location | Status |
|---|---|---|
| `scan_vqtl()` | `mosaiqtl/vqtl.py` | ✅ scaffolded |

---

## P3 — Epistasis Module (`epistasis.py`) — Planned

### Purpose

Test whether combinations of founder haplotypes at two loci produce expression
levels not predictable from the sum of their individual effects — polygenic epistasis
in the Collaborative Cross.

---

### Planned Functions

**`scan_epistasis(locus_a, locus_b)`** — pairwise $G \times G$ interaction:

$$Y_i = H_i^a\,\alpha + H_i^b\,\beta + (H_i^a \otimes H_i^b)\,\gamma + X_i\,\xi + \epsilon_i$$

Interaction term $(J-1)^2$-dimensional; LOD from LRT vs additive null.

**`test_haplotype_combinations()`** — $J \times J$ founder-pair expression heatmap:
for each pair $(f, f')$, subset strains carrying $(f, f')$ at two peak loci and
report mean expression — visualises emergent polygenic combinations.

**`scan_conditional(locus_fixed)`** — multi-QTL residual scanning:
regress out effect at `locus_fixed`, scan residuals genome-wide for secondary QTL.

---

### Implementation

| Function | Location | Status |
|---|---|---|
| `scan_epistasis()` | `mosaiqtl/epistasis.py` | 🔲 not yet created |
| `test_haplotype_combinations()` | `mosaiqtl/epistasis.py` | 🔲 not yet created |
| `scan_conditional()` | `mosaiqtl/epistasis.py` | 🔲 not yet created |

---

## Open Items (all POC modules)

| # | Item | Blocking? |
|---|---|---|
| 1 | Run `scan_nb` on real pseudobulk (Piezo1 chr8 proof of concept) | yes — primary validation |
| 2 | Compare NB LOD vs LMM LOD for same gene set | yes — model justification |
| 3 | Confirm edgePython `fit['fitted.values']` key matches API | yes — `scan_nb` correctness |
| 4 | Permutation null for NB LOD thresholds | yes — significance calling |
| 5 | K-expansion (n > m) end-to-end test with real data | yes — Z-matrix validation |
| 6 | Implement `epistasis.py` | no — post-validation extension |

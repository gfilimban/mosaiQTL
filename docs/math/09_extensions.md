# Extensions: Mediation, Epistasis, and vQTL

**Modules:** `qtl/mediation.py`, `qtl/epistasis.py`, `qtl/vqtl.py`  
**miQTL parallel:** `mediation.scan.qr()`, `scan.h2lmm()` (conditional), `instability.lm.scan()`

---

## 1. Mediation scan (LOD-drop)

**Question:** Does the haplotype effect on outcome gene $y$ operate through
intermediate gene $m$? If haplotype → $m$ → $y$, then conditioning on $m$
should reduce (or eliminate) the haplotype LOD for $y$.

**Statistic:**

$$\text{LOD}_{\text{drop},\ell} = \text{LOD}_{\text{base},\ell} - \text{LOD}_{\text{mediated},\ell}$$

where:
- $\text{LOD}_{\text{base},\ell}$ = haplotype LOD for $y$ with covariates $X$ only.
- $\text{LOD}_{\text{mediated},\ell}$ = haplotype LOD for $y$ with $[m_{\text{std}},\, X]$ as covariates (mediator added).

$\text{LOD}_{\text{drop}} > 0$: the haplotype effect shrinks when $m$ is controlled —
mediation detected.  
$\text{LOD}_{\text{drop}} < 0$: amplification — a shared upstream regulator drives
both $y$ and $m$ in the same direction, so conditioning on $m$ removes shared signal.

**Two implementations:**

`mediation_scan_qr` — LMM with LOCO kinship. Estimates $h^2$ separately for
both the base model and mediated model (conditioning on $m$ changes the phenotype,
hence the heritability). Uses the full FWL batch scan per chromosome. Slower but
accounts for polygenic background.

`fixef_mediation_scan` — OLS, no kinship. Same FWL batch scan without eigendecomp.
Used for fast screening or when kinship is not available. `just_these_loci` filter
enables cis-window restriction.

**Implementation detail:** The mediator $m$ is z-score standardized before being
added to the covariate matrix to put it on a comparable scale to the intercept and
other covariates. This prevents numerical issues when $m$ has a very different
magnitude than $y$.

---

## 2. Epistasis scan

**Module:** `qtl/epistasis.py`

**Question:** Do two loci $\ell_1$ and $\ell_2$ have an interaction effect on $y$
beyond their additive contributions?

**Model:**

$$y = H_{\ell_1}\gamma_1 + H_{\ell_2}\gamma_2 + (H_{\ell_1} \otimes H_{\ell_2})\Gamma_{12} + X\alpha + u + \varepsilon$$

where $H_{\ell_1} \otimes H_{\ell_2} \in \mathbb{R}^{m \times J^2}$ is the Khatri-Rao
(column-wise Kronecker) product of the two haplotype matrices, encoding all
pairwise founder combinations.

**Pairwise scan:** for a primary locus $\ell_1$ (peak locus from main scan),
sweep all secondary loci $\ell_2$ across the genome. LOD for the interaction term
$\Gamma_{12}$ is reported after marginalizing out the additive effects of both loci.

**Conditional scan (`scan_conditional_with_y`):** given an established QTL at
$\ell_1$, test for additional independent signals by conditioning on $H_{\ell_1}$
as a covariate and re-scanning. A secondary peak in the conditional scan indicates
a distinct signal. This is the standard multi-signal detection approach (analogous
to stepwise regression in GWAS fine-mapping).

---

## 3. Variance QTL (vQTL)

**Module:** `qtl/vqtl.py`

**Question:** Does the haplotype at locus $\ell$ affect the *variance* (dispersion)
of $y$ across strains, beyond any effect on the mean?

The vQTL tests whether founder ancestry predicts the residual variance of $y$ after
removing the mean QTL effect.

**Correct Stage 2 design (planned — VQTL-01):**  
Stage 2 should run a NB-GLMM **dispersion score test**: the null is $\phi$ constant
across haplotypes; the alternative is $\log\phi_{ig} = H_{i\ell}^\top\delta + \ldots$.
The score test statistic for dispersion effects uses the NB score gradient with
respect to $\phi$, not with respect to the mean $\mu$.

**Current implementation caveat (VQTL-01 open):**  
The current `scan_vqtl` incorrectly calls `scan_h2lmm` (Gaussian LMM) on
$\log\hat\phi$ per strain. This treats the estimated log-dispersion as a
continuous phenotype and applies the LMM scan to it. While this is a reasonable
approximation when $m$ is large and $\hat\phi$ estimates are stable, it is not
the proper model: (i) $\hat\phi$ estimates are noisy for low-count genes, (ii)
the LMM does not account for the asymmetric distribution of $\log\hat\phi$, and
(iii) the scan stat is no longer a proper likelihood ratio test.

Until VQTL-01 is resolved, interpret vQTL results with caution.

---

## 4. LOD-drop confidence interval (Manichaikul)

**Module:** `qtl/scan.py::lod_drop_ci`  
**miQTL parallel:** `calc.LOD.drop()`

The LOD-drop CI is the genomic interval where:

$$\text{LOD}(\ell) \geq \text{LOD}_{\text{peak}} - \Delta$$

Common choices: $\Delta = 1.5$ (≈ 95% Bayes CI under Haley-Knott), $\Delta = 1.8$.
Faster than the bootstrap positional CI (section 6 of `06_permutation.md`) but
does not propagate phenotypic sampling uncertainty. Use bootstrap CI for publications;
LOD-drop CI for quick exploratory analysis.

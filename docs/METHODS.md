# mosaiQTL — Mathematical Methods Reference

**Scope:** Formal definitions of every statistical concept used in mosaiQTL,
ordered from inputs to outputs. Intended as the authoritative mathematical
reference for the package — each concept links to its implementing module.

---

## 1. Haplotype probability inference — forward-backward HMM

**Module:** `hmm.py`  
**What it is:** Given observed genotype data at a set of markers along a
chromosome, estimate the posterior probability that each genomic position was
inherited from each of the $J$ founder strains.

**Model.** At locus $\ell$, individual $i$ inherits a haplotype from founder
$f$ with probability $H_{if\ell} \in [0,1]$, $\sum_f H_{if\ell} = 1$.
The hidden Markov model treats founder ancestry as a latent Markov chain along
the chromosome. Emission probabilities encode how likely the observed genotype
call is given descent from each founder. Transition probabilities encode
recombination rates (derived from the Haldane map function applied to genetic
distances in cM).

**Forward-backward algorithm.** For a chain with $J$ states and $L$ loci:

$$\alpha_{if\ell} = P(\text{obs}_{1:\ell} \mid z_{i\ell}=f) \cdot \alpha_{if,\ell-1} \cdot T_{ff',\ell-1\to\ell}$$

$$\beta_{if\ell} = \sum_{f'} T_{ff',\ell\to\ell+1} \cdot P(\text{obs}_{\ell+1} \mid z_{i,\ell+1}=f') \cdot \beta_{if',\ell+1}$$

$$H_{if\ell} = \frac{\alpha_{if\ell} \cdot \beta_{if\ell}}{\sum_{f'} \alpha_{if'\ell} \cdot \beta_{if'\ell}}$$

**miQTL difference.** R's HAPPY HMM estimates joint diplotype probabilities
$P(f \times f' \mid i, \ell) \in \mathbb{R}^{J(J+1)/2}$ (36 states for $J=8$).
mosaiQTL estimates marginal haplotype probabilities by summing over the diplotype
partner: $H_{if\ell} = \sum_{f'} P(f \times f' \mid i, \ell)$. This is exact
under the inbred-founder assumption (CC strains are nearly fully inbred, so
off-diagonal diplotypes $f \neq f'$ have near-zero probability). The marginal
representation is sufficient for all LMM and NB scan steps; the 36-state
representation only adds information in the SNP scan (S8), which accounts for
the documented r=0.55 discrepancy in that section.

---

## 2. van Raden kinship matrix

**Module:** `kinship.py`  
**What it is:** A genomic relationship matrix $K \in \mathbb{R}^{m \times m}$
quantifying the fraction of the genome shared identical-by-descent between each
pair of strains. Used as the covariance structure for the polygenic random
effect in the LMM.

**Formula** (van Raden 2008):

$$K = \frac{1}{L} \sum_{\ell=1}^{L} \tilde{H}_\ell \tilde{H}_\ell^\top$$

where $\tilde{H}_\ell = H_\ell - \bar{H}_\ell$ is the mean-centred haplotype
matrix at locus $\ell$, $H_\ell \in \mathbb{R}^{m \times J}$ with rows summing
to 1, and $L = |\mathcal{M}|$ is the number of QC-passing loci.

Diagonal entries $K_{ii} \approx 1$ (self-relatedness); off-diagonals reflect
shared haplotype ancestry. For fully inbred strains (CC founders) $K_{ii} = 1$
exactly; for partially inbred CC strains $K_{ii}$ slightly exceeds 1.

**LOCO kinship.** To avoid confounding the genetic signal at a locus with the
polygenic background from that same chromosome, the kinship matrix is recomputed
leaving out chromosome $c$:

$$K^{(-c)} = \frac{1}{L^{(-c)}} \sum_{\ell \notin c} \tilde{H}_\ell \tilde{H}_\ell^\top$$

where $L^{(-c)} = |\mathcal{M} \setminus c|$. One LOCO matrix per chromosome
is precomputed and cached. This is the standard approach in animal genetics
(Yang et al. 2014) and is implemented identically in miQTL and mosaiQTL.

---

## 3. h² estimation — REML with eigendecomposition

**Module:** `lmm.py → estimate_h2_null()`  
**What it is:** Under the null model (no QTL effect), heritability $h^2$
controls the balance between polygenic and residual variance. Estimating it
once per gene under the null and holding it fixed across all loci (the
"fix-par" convention) dramatically reduces computation.

**LMM variance structure:**

$$\mathbf{y} = \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$
$$\mathbf{u} \sim \mathcal{N}(\mathbf{0},\, \sigma_g^2 \mathbf{K}), \quad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\, \sigma_e^2 \mathbf{I})$$
$$\mathbf{V} = \sigma_g^2 \mathbf{K} + \sigma_e^2 \mathbf{I} = \sigma_e^2 (h^2 \mathbf{K} + (1-h^2)\mathbf{I})$$

where $h^2 = \sigma_g^2 / (\sigma_g^2 + \sigma_e^2)$.

**Eigendecomposition trick** (Kang et al. 2008). Compute $\mathbf{K} = \mathbf{V}_K \boldsymbol{\Lambda}_K \mathbf{V}_K^\top$ once. Then:

$$\mathbf{V}^{-1} = \mathbf{V}_K \operatorname{diag}\!\left(\frac{1}{h^2\lambda_j + (1-h^2)}\right) \mathbf{V}_K^\top$$

Rotating the data: $\tilde{\mathbf{y}} = \mathbf{V}_K^\top \mathbf{y}$,
$\tilde{\mathbf{X}} = \mathbf{V}_K^\top \mathbf{X}$, the REML log-likelihood
becomes a 1D function of $h^2$ alone:

$$\ell_\text{REML}(h^2) = -\frac{1}{2}\left[\sum_j \log(h^2\lambda_j + 1-h^2) + \tilde{\mathbf{y}}^\top \mathbf{V}^{-1} M_{\tilde{X}} \mathbf{V}^{-1} \tilde{\mathbf{y}}\right]$$

where $M_{\tilde{X}} = I - \tilde{X}(\tilde{X}^\top \tilde{X})^{-1}\tilde{X}^\top$
is the annihilator of the rotated covariate space. This 1D optimisation is fast
(Brent's method or a grid search). The eigendecomposition itself is cached per
chromosome: computed once, reused for all genes on that chromosome.

---

## 4. LMM scan — LOD score

**Module:** `lmm.py → fit_lmm_at_locus()`  
**What it is:** At each genomic locus $\ell$, test whether including the
haplotype design matrix $H_\ell$ as a fixed effect significantly reduces
residual sum of squares under the fixed-$h^2$ LMM.

**Model:**

$$H_0: \mathbf{y} = \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$
$$H_1: \mathbf{y} = \mathbf{H}_\ell \boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

With $h^2$ fixed from the null, this reduces to weighted least squares in the
rotated space. The LOD score uses the maximum-likelihood denominator $n$
(not $n-p$, which would be REML):

$$\text{LOD}_\ell = \frac{n}{2} \log_{10}\!\frac{\text{RSS}_0}{\text{RSS}_1}$$

where $\text{RSS}_k = \tilde{\mathbf{y}}^\top \mathbf{D}^{-1} M_k \tilde{\mathbf{y}}$,
$\mathbf{D} = \operatorname{diag}(h^2\lambda_j + 1-h^2)$, and $M_k$ is the
annihilator of the appropriate design matrix. This formula matches miQTL exactly
and is positive by construction since $H_1$ is a superset of $H_0$.

---

## 5. Frisch-Waugh-Lovell (FWL) vectorization

**Module:** `lmm.py → scan_chromosome()`  
**What it is:** The key computational speedup in mosaiQTL. Instead of running a
separate WLS regression per locus (miQTL's approach, $O(L \cdot n \cdot J^2)$
sequential), FWL allows the covariate projection to be computed once and the
haplotype regressions to be vectorized across all $L$ loci simultaneously.

**Theorem (Frisch-Waugh-Lovell).** In OLS (and WLS), the coefficient on a
set of regressors $\mathbf{H}$ in the full model
$\mathbf{y} = \mathbf{H}\boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \boldsymbol{\varepsilon}$
is identical to the coefficient obtained by first projecting out $\mathbf{X}$:

$$\hat{\boldsymbol{\gamma}}_\text{FWL} = (\tilde{\mathbf{H}}^\top \tilde{\mathbf{H}})^{-1} \tilde{\mathbf{H}}^\top \tilde{\mathbf{y}}$$

where $\tilde{\mathbf{y}} = M_X \mathbf{y}$ and $\tilde{\mathbf{H}} = M_X \mathbf{H}$
are the residuals after projecting out covariates $\mathbf{X}$.

**Application in mosaiQTL.** In the rotated (eigendecomposed) space:
1. Project out covariates once: $\tilde{\mathbf{y}}_\text{rot} = M_{X_\text{rot}} \mathbf{y}_\text{rot}$. Cost: $O(n \cdot p^2)$.
2. For each locus, project haplotypes: $\tilde{\mathbf{H}}_{\ell,\text{rot}} = M_{X_\text{rot}} \mathbf{H}_{\ell,\text{rot}}$.
3. Compute RSS via $\text{RSS}_1 = \tilde{\mathbf{y}}^\top \tilde{\mathbf{y}} - \tilde{\mathbf{y}}^\top \tilde{\mathbf{H}}(\tilde{\mathbf{H}}^\top \tilde{\mathbf{H}})^{-1} \tilde{\mathbf{H}}^\top \tilde{\mathbf{y}}$.

Steps 2 and 3 are batched across all $L$ loci using `numpy.einsum`, giving
effectively $O(L \cdot n \cdot J)$ cost rather than $O(L \cdot n \cdot J^2)$.
The per-locus $(J-1) \times (J-1)$ solve is replaced by a batch matrix operation.
**Result: ~100× speedup over miQTL's sequential per-locus WLS loop** for
typical chromosome sizes ($L \sim 4000$ loci).

This is mathematically identical to per-locus WLS — no approximation is made.
The S4 validation (median LOD correlation r=0.973 against miQTL) confirms
numerical equivalence.

---

## 6. RINT — rank-based inverse normal transformation

**Module:** `utils.py → rint()`  
**What it is:** Applied to phenotypes before LMM fitting to remove distributional
assumptions. Converts any continuous phenotype to approximate normality while
preserving rank order.

**Blom's formula:**

$$y_i^\text{rint} = \Phi^{-1}\!\left(\frac{r_i - 3/8}{n + 1/4}\right)$$

where $r_i = \text{rank}(y_i)$ (average ranks for ties) and $\Phi^{-1}$ is the
standard normal quantile function. The constants $-3/8$ and $+1/4$ minimise
bias in the expected order statistics. Not applied in the NB GLM (which models
counts directly) or in dispersion estimation.

---

## 7. NB GLM Stage 1 — NEBULA-LN cell-level null model

**Module:** `nb.py → fit_null_sc()`  
**What it is:** A negative binomial generalised linear mixed model fit at the
cell level, capturing within-strain cell-to-cell variability and between-strain
overdispersion simultaneously. Run once per cell type before the genome-wide scan.

**Model:**

$$Y_{ij} \sim \text{NB}(\mu_{ij},\, \phi_g), \quad
\log \mu_{ij} = \mathbf{x}_j^\top \boldsymbol{\beta}_g + \log s_j + u_i$$
$$u_i \sim \mathcal{N}(0,\, \sigma^2_g)$$

$Y_{ij}$: raw UMI count for gene $g$ in cell $j$ of strain $i$.  
$s_j$: TMM-normalised library size offset.  
$\mathbf{x}_j$: cell-level covariates (intercept, batch dummies, expression PCs).  
$u_i$: unstructured per-subject random effect — absorbs between-strain variance
not explained by fixed covariates. This is not a LOCO-kinship random effect;
it does not capture genetic relatedness structure. A full NB-LMM with LOCO
kinship is a planned future extension.  
$\phi_g$: NB overdispersion parameter for gene $g$ (shared across cells).

**NB parameterisation:** $\text{Var}(Y_{ij}) = \mu_{ij} + \phi_g \mu_{ij}^2$.
As $\phi_g \to 0$ the model approaches Poisson. As $\phi_g$ increases,
overdispersion increases. Estimation is performed via NEBULA-LN (He et al. 2021)
as implemented in `edgePython` (Pachter 2026, GPL-3.0).

**Empirical Bayes dispersion shrinkage.** Per-gene MLE dispersion estimates are
noisy for lowly expressed genes. After fitting, `shrink_sc_disp()` applies
weighted likelihood empirical Bayes (WLEB) shrinkage toward a trended prior
$\tilde{\phi}(\bar{\mu}_g)$ — a smooth function of mean expression. The shrinkage
weight $w_g$ increases with sample size; genes with many cells are shrunk less:

$$\hat{\phi}_g^\text{shrunk} = w_g \hat{\phi}_g^\text{MLE} + (1-w_g)\tilde{\phi}(\bar{\mu}_g)$$

This is the same framework as edgeR's tagwise dispersion estimation (Robinson &
Smyth 2007). The shrunk dispersion $\hat{\phi}_g^\text{shrunk}$ is what gets
passed to Stage 2.

---

## 8. NB GLM Stage 2 — pseudobulk score test

**Module:** `nb.py → scan_nb_gene()`, `_score_stats_chromosome()`  
**What it is:** For each gene, aggregate cell-level counts to per-strain
pseudobulk sums, then test for association between haplotype and pseudobulk
count at each genomic locus using a score test. Vectorized across all loci
simultaneously.

**Pseudobulk aggregation:**

$$S_{ig} = \sum_{j \in \text{strain } i} Y_{ij}$$

$S_{ig}$ is the total UMI count for gene $g$ in strain $i$. Under the null
(no haplotype effect), $\log \mathbb{E}[S_{ig}] = \mathbf{x}_i^\top\boldsymbol{\beta}_0 + \log(n_i \bar{s}_i)$
where $n_i$ is the cell count for strain $i$ and $\bar{s}_i$ is the mean
library size.

**Score test.** The score test evaluates the gradient of the log-likelihood at
the null parameter estimate, avoiding the need to fit the alternative model at
every locus. For the NB model, the score statistic at locus $\ell$ is:

$$Q_\ell = \mathbf{U}_\ell^\top \mathbf{I}_\ell^{-1} \mathbf{U}_\ell \sim \chi^2(J-1) \text{ under } H_0$$

where $\mathbf{U}_\ell = \mathbf{H}_\ell^\top \mathbf{r}$ is the score vector
(weighted residuals projected onto the haplotype design matrix),
$\mathbf{r} = \mathbf{S} - \hat{\boldsymbol{\mu}}_0$ are the null residuals,
and $\mathbf{I}_\ell = \mathbf{H}_\ell^\top \mathbf{W} \mathbf{H}_\ell$ is the
Fisher information with weight matrix $\mathbf{W} = \operatorname{diag}(\hat{\mu}_{i0}^2 / \hat{\phi}_g + \hat{\mu}_{i0} + \sigma_\text{sample}^2 \hat{\mu}_{i0}^2)$.

The $\sigma_\text{sample}^2$ term from Stage 1 inflates the denominator to
account for residual subject-level variance not captured by fixed covariates —
making the test conservative in the right direction.

**Vectorization.** Computing $Q_\ell$ for all $L$ loci simultaneously:

$$\mathbf{U} = \operatorname{einsum}(H_\text{all},\, \mathbf{r}) \in \mathbb{R}^{L \times (J-1)}$$
$$\mathbf{I} = \operatorname{einsum}(H_\text{all},\, \mathbf{W},\, H_\text{all}) \in \mathbb{R}^{L \times (J-1) \times (J-1)}$$

where $H_\text{all} \in \mathbb{R}^{L \times n \times (J-1)}$ is the full
haplotype tensor. The per-locus solve $\mathbf{I}_\ell^{-1}$ is the dominant
cost: $O(L \cdot (J-1)^3)$. For $J=8$, $(J-1)^3=343$ — negligible per locus.

The score statistic $Q_\ell$ is converted to a LOD score:
$\text{LOD}_\ell = Q_\ell / (2\ln 10)$ (chi-squared to LOD equivalence).

---

## 9. vQTL — dispersion QTL scan

**Module:** `vqtl.py → scan_vqtl()`  
**What it is:** Tests whether haplotype affects not mean expression but
expression variance across cells — a biologically distinct signal that is
only detectable at single-cell resolution. A mean eQTL affects how much a
gene is expressed on average; a vQTL affects how consistently it is expressed
cell-to-cell within a strain.

**Two-stage design.**

*Stage 1 — per-strain dispersion estimation.*
For each strain $i$, the NB overdispersion $\hat{\phi}_i$ is estimated from
its own cells using method-of-moments:

$$\hat{\phi}_i = \frac{\hat{\sigma}^2_i - \hat{\mu}_i}{\hat{\mu}_i^2}$$

where $\hat{\mu}_i$ and $\hat{\sigma}^2_i$ are the sample mean and variance of
counts across cells in strain $i$ (adjusted for library size). Strains with
fewer than `min_cells` cells are excluded.

*Stage 2 — LMM scan on log dispersion.*
$\log\hat{\phi}_i$ is treated as a strain-level phenotype and passed through
the standard LMM scan (`scan_h2lmm`) with LOCO kinship:

$$\log\hat{\phi}_i = \mathbf{H}_{i\ell}\boldsymbol{\gamma} + \mathbf{x}_i^\top\boldsymbol{\alpha} + u_i + \varepsilon_i$$

A significant vQTL LOD means the founder haplotype at locus $\ell$ explains
variance in expression dispersion across strains. Co-localisation of a vQTL
with a mean eQTL implies the haplotype affects both; an independent vQTL
implies the haplotype specifically modulates expression noise.

---

## 10. Mediation — QR LOD-drop

**Module:** `mediation.py → mediation_scan_qr()`  
**What it is:** Quantifies how much of a QTL's effect on an outcome gene
flows through a mediator gene's expression. If the mediator fully explains
the QTL, the LOD drops to near zero when the mediator is included as a
covariate.

**Model:**

$$\text{base scan: }  y_\text{outcome} = \mathbf{H}_\ell\boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + u + \varepsilon$$
$$\text{mediated scan: } y_\text{outcome} = \mathbf{H}_\ell\boldsymbol{\gamma}' + y_\text{mediator}\delta + \mathbf{X}\boldsymbol{\alpha}' + u + \varepsilon$$
$$\text{LOD\_drop}_\ell = \text{LOD\_base}_\ell - \text{LOD\_mediated}_\ell$$

A large positive LOD\_drop indicates mediation: including the mediator
absorbs the haplotype effect. A negative LOD\_drop (amplification) is not a
bug — it occurs when conditioning on the mediator sharpens the residual
signal by removing mediator-correlated noise.

**QR implementation.** Rather than solving two separate LMM scans, the
mediated scan uses a QR decomposition to efficiently augment the design matrix
with the mediator, avoiding redundant kinship computations.

---

## 11. Epistasis — Khatri-Rao product

**Module:** `epistasis.py → scan_epistasis()`  
**What it is:** Tests whether the combination of founder haplotypes at two
distinct loci produces an expression outcome not predictable from their additive
sum — polygenic epistasis at haplotype resolution.

**Models:**

$$\text{additive: } y = \mathbf{H}^a\boldsymbol{\alpha} + \mathbf{H}^b\boldsymbol{\beta} + \mathbf{X}\boldsymbol{\xi} + u + \varepsilon$$

$$\text{interaction: } y = \mathbf{H}^a\boldsymbol{\alpha} + \mathbf{H}^b\boldsymbol{\beta} + (\mathbf{H}^a \odot \mathbf{H}^b)\boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\xi} + u + \varepsilon$$

**Khatri-Rao product.** $\mathbf{H}^a \odot \mathbf{H}^b \in \mathbb{R}^{n \times (J-1)^2}$
is the column-wise Kronecker product of the two haplotype design matrices (each
with last founder dropped for identifiability). For $J=8$: $(J-1)^2 = 49$
interaction terms. Each term $H^a_{if} \cdot H^b_{if'}$ represents the
probability that strain $i$ carries haplotype $f$ at locus $a$ AND haplotype
$f'$ at locus $b$ simultaneously.

**Epistasis LOD:**

$$\text{LOD\_epistasis} = \text{LOD\_interaction} - \text{LOD\_additive}$$

A positive LOD\_epistasis means the two-locus combination explains expression
beyond the sum of individual founder effects. The companion function
`test_haplotype_combinations()` produces the $J \times J$ founder-pair mean
expression heatmap as an empirical visualisation of the interaction structure.

---

## 12. Conditional multi-signal decomposition — BLUP conditioning

**Module:** `epistasis.py → scan_conditional_with_y()`  
**What it is:** Forward-stepwise detection of multiple independent QTL signals
for a single gene. At each step, the haplotype effect at the current peak
locus is estimated and subtracted, and the genome is rescanned for residual
signal.

**BLUP (Best Linear Unbiased Prediction).** At each iteration $k$:

1. Find peak locus $\ell^*$ in the current scan result.
2. Fit the LMM at $\ell^*$ to obtain the allele effect vector $\hat{\boldsymbol{\gamma}}_{\ell^*}$.
3. Compute the BLUP of the haplotype contribution:
   $$\hat{\mathbf{y}}_{\text{hap}} = \mathbf{H}_{\ell^*}^\top \hat{\boldsymbol{\gamma}}_{\ell^*}$$
4. Condition out: $\mathbf{y}^{(k+1)} = \mathbf{y}^{(k)} - \hat{\mathbf{y}}_\text{hap}$
5. Rescan $\mathbf{y}^{(k+1)}$ genome-wide.

Iteration stops when the residual peak LOD falls below `lod_drop_threshold`
(default 1.5). The method is conservative: BLUP prediction shrinks the
estimated effect toward zero, so residual signals are not over-subtracted.

**Why BLUP, not OLS?** OLS at the peak locus would perfectly fit the observed
haplotype pattern including noise, leading to over-subtraction and spurious
signals. BLUP uses the LMM's kinship structure to regularise the prediction,
conditioning only on the systematic genetic component.

---

## 13. GEV permutation thresholds

**Module:** `permutation.py → run_threshold_scans()`, `get_gev_thresholds()`  
**What it is:** Derives genome-wide significance thresholds by permuting the
phenotype (breaking the genotype-phenotype link) and fitting a Generalised
Extreme Value (GEV) distribution to the null distribution of maximum LOD scores.

**Procedure.**
For each permutation $b = 1, \ldots, B$:
1. Randomly permute $\mathbf{y}$ across strains.
2. Run the full genome scan on the permuted phenotype.
3. Record $M_b = \max_\ell \text{LOD}_\ell^{(b)}$.

The empirical distribution $\{M_b\}$ follows a GEV distribution
(by extreme value theory: the maximum of many correlated chi-squared statistics
converges to GEV). Fit:

$$F(m; \mu, \sigma, \xi) = \exp\!\left\{-\left[1 + \xi\!\left(\frac{m-\mu}{\sigma}\right)\right]^{-1/\xi}\right\}$$

The 95th percentile of this fitted distribution is the genome-wide threshold.
GEV fitting requires far fewer permutations ($B \sim 100$) than the empirical
approach ($B \sim 1000$), making it tractable.

**fix_h2 optimisation.** mosaiQTL re-uses $\hat{h}^2$ from the real
phenotype rather than re-estimating it per permutation. This is valid when
$h^2$ is estimated under the null (which it is) and reduces permutation cost
by ~10×. miQTL re-estimates $h^2$ per permutation, which is more conservative
but computationally prohibitive at genome-wide scale.

**Known issue.** When pooling across chromosomes with very different cis-loci
counts, the GEV fit is unstable — short chromosomes with few loci produce heavy
tails that inflate the pooled threshold. The current `mean_delta=0.619` failure
in S6 is attributed to this effect. Per-chromosome thresholds are now visualised
in `s6_permutation.png` panel 3 for diagnosis.

---

## 14. Z-matrix expansion

**Module:** `utils.py → build_z_matrix()`  
**What it is:** mosaiQTL's design supports $n \geq m$ — more phenotype
observations than unique genotyped strains. The Z-matrix maps samples to
strains at scan time, avoiding pre-collapsing to strain means and preserving
within-strain biological replication.

$$Z \in \{0,1\}^{n \times m}, \quad Z_{ik} = \mathbf{1}[\text{sample } i \text{ belongs to strain } k]$$

At scan time:
$$\mathbf{H}_n = Z\mathbf{H}_m \quad (\text{expand haplotype probs to } n \text{ samples})$$
$$\mathbf{K}_n = Z\mathbf{K}_m Z^\top \quad (\text{expand kinship})$$

When $n = m$ and IDs match, $Z = I_m$ and all formulas reduce to the
strain-mean case exactly. No module hardcodes $n$, $m$, or $J$.

---

## Summary table

| Concept | Module | Key formula | Key property |
|---|---|---|---|
| HMM forward-backward | `hmm.py` | $H_{if\ell} \propto \alpha_{if\ell}\beta_{if\ell}$ | marginal over diplotypes |
| van Raden kinship | `kinship.py` | $K = \frac{1}{L}\sum_\ell \tilde{H}_\ell\tilde{H}_\ell^\top$ | shared-ancestry covariance |
| LOCO kinship | `kinship.py` | $K^{(-c)}$ excludes chr $c$ | avoids QTL confounding |
| REML h² | `lmm.py` | eigendecomp trick, 1D grid | computed once per gene |
| LMM scan LOD | `lmm.py` | $\frac{n}{2}\log_{10}\frac{\text{RSS}_0}{\text{RSS}_1}$ | ML denominator |
| FWL vectorization | `lmm.py` | project covariates once, batch loci | ~100× faster, exact |
| RINT | `utils.py` | Blom's formula | rank-preserving normalisation |
| NB Stage 1 (NEBULA-LN) | `nb.py` | $Y_{ij} \sim \text{NB}(\mu_{ij},\phi_g)$, $u_i \sim \mathcal{N}$ | within-strain RE |
| EB dispersion shrinkage | `nb.py` | WLEB toward trended prior | stabilises low-expr genes |
| NB Stage 2 score test | `nb.py` | $Q_\ell = U_\ell^\top I_\ell^{-1} U_\ell \sim \chi^2_{J-1}$ | no alt-model fit needed |
| vQTL | `vqtl.py` | scan $\log\hat{\phi}_i$ as phenotype | cell-count heterogeneity |
| Mediation LOD-drop | `mediation.py` | LOD\_base $-$ LOD\_mediated | negative drop = amplification |
| Khatri-Rao epistasis | `epistasis.py` | $H^a \odot H^b \in \mathbb{R}^{(J-1)^2}$ | pairwise haplotype interaction |
| BLUP conditioning | `epistasis.py` | subtract $H\hat{\gamma}$, rescan | regularised multi-signal |
| GEV permutation | `permutation.py` | fit GEV to $\{M_b\}$ | extreme value theory |
| Z-matrix | `utils.py` | $Z \in \{0,1\}^{n\times m}$ | $n \geq m$ replication support |

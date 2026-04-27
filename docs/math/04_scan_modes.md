# Scan Modes: NB-GLMM, Additive LMM, Full Diplotype, and Multiple Imputation

**Modules:** `qtl/scan.py`, `qtl/lmm.py`, `qtl/nb.py`  
**miQTL parallel:** `scan.h2lmm(use.multi.impute=FALSE/TRUE, model='additive'/'full')`

---

## 1. Two scan engines — different observation granularities

mosaiQTL has two independent scan engines with fundamentally different input spaces:

| | NB-GLMM (`nb.py`) | LMM (`lmm.py` + `scan.py`) |
|---|---|---|
| **Input** | Raw single-cell / pseudobulk counts | Continuous phenotype |
| **Observation level** | Cell (Stage 1) → pseudobulk (Stage 2) | Sample or strain |
| **Strain reduction** | Never — NB aggregates to pseudobulk sum $S_{ig}$ internally | Strain-mean via Z-matrix when replicates present |
| **Kinship integration** | PQL + Woodbury (Stage 2) | Eigendecomp WLS |
| **Primary use** | snRNA-seq eQTL | Gaussian phenotypes, validation |
| **Status** | Primary model | Validated port (Phase 1/2 ground truth) |

---

## 2. NB-GLMM scan (primary)

**Full model description:** `08_nb_glmm.md`

The NB-GLMM scan is the primary model for count-based omics data. Observations
are raw UMI counts; no log-normalization, RINT, or strain-mean reduction is applied.
RINT is an LMM-specific preprocessing step that converts a continuous phenotype
to approximate normality — it has no meaning for discrete counts and is never
called in the NB path.

**Stage 1** (`nb.py::fit_null_sc_native`): Fits a NB-gamma null model at
the **cell level** — each cell is one observation. Outputs per-gene:
- $\hat\phi_g$ — EB-shrunk cell-level NB dispersion
- $\hat\sigma^2_g$ — subject-level overdispersion (between-strain variance)

**Stage 2** (`nb.py::scan_nb_gene`): Scans at **pseudobulk level** using
$S_{ig} = \sum_{j \in \text{cells of strain } i} Y_{ijg}$ (sum, not mean).
The NB score test at locus $\ell$ is:

$$Q_\ell = U_\ell^\top \mathcal{I}_\ell^{-1} U_\ell \sim \chi^2(J-1),
\qquad \text{LOD}_\ell = \frac{Q_\ell}{2 \ln 10}$$

where $U_\ell$ is the score vector and $\mathcal{I}_\ell$ is the Fisher information
at the null. The null is fitted **once per gene per chromosome**; the score is then
vectorized over all $L$ loci.

No Z-matrix or strain-mean transformation is applied in the NB path. The pseudobulk
sum $S_{ig}$ is one observation per strain per gene — the NB variance function
$\operatorname{Var}(S) = \mu + \phi \mu^2$ accounts for both Poisson and overdispersion
components without first converting to a continuous phenotype.

---

## 3. LMM additive (ROP) scan

**miQTL name:** `scan.h2lmm(use.multi.impute=FALSE, model='additive')`

The LMM can handle replicates in two ways:

**Path A — Strain-mean reduction (current mosaiQTL implementation):**  
Reduce $n$ sample phenotypes to $m$ strain means before scanning:

$$\bar{y}_i = \frac{1}{r_i}\sum_{j:\,s(j)=i} y_j, \qquad
  \bar{X}_i = \frac{1}{r_i}\sum_{j:\,s(j)=i} x_j$$

via $\bar{y} = (Z^\top y)/r$. The scan then operates entirely in strain space
($m \times m$ kinship, $m$-row haplotype matrix). This is `scan_h2lmm` with
`sample_ids` provided.

**Path B — Replicate-aware LMM with expanded kinship (miQTL `lmmbygls.replicates()`):**  
Keep all $n$ sample observations. Expand the strain-level kinship to sample
space via the Z-matrix:

$$\tilde{K} = Z K Z^\top \in \mathbb{R}^{n \times n}, \qquad \tilde{K}_{ij} = K_{s(i),s(j)}$$

The LMM then fits $y = H_\ell \gamma + X\alpha + u + \varepsilon$ at sample level,
where $u \sim \mathcal{N}(0, \sigma^2_g \tilde{K})$. This correctly models the
covariance between replicate samples of the same strain (they share the same
genetic background, so $\tilde{K}_{ij} = K_{ii} > 0$ for $s(i) = s(j)$).

**Which to use:** Path A discards within-strain phenotype variance and treats
all replicates as a single data point. Path B retains replicate-level information
and is statistically more powerful when phenotypes vary substantially within a
strain. For Gaussian phenotypes with low within-strain variance, both paths give
nearly identical LOD profiles. Path B corresponds to `LMM-EXT-01` (planned in
mosaiQTL; currently in strain-mean-only state).

The additive scan uses $H_\ell \in \mathbb{R}^{m \times J}$ (Path A) or
$\tilde{H}_\ell = Z H_\ell \in \mathbb{R}^{n \times J}$ (Path B) as the
haplotype design matrix. Betas are per-founder allele effects in either case.

---

## 4. LMM full diplotype scan

**miQTL name:** `scan.h2lmm(use.multi.impute=FALSE, model='full')`

The full model constructs the outer-product diplotype design matrix:

$$P(f_i / f_j \mid \text{strain},\ell) \approx H_{if\ell} \cdot H_{jf\ell}, \quad i \leq j$$

This gives $J(J+1)/2 = 36$ diplotype columns for $J = 8$.

The outer-product approximation is exact for inbred CC strains because
off-diagonal heterozygous diplotype probabilities $P(f \neq f')$ are near zero —
the marginal product is numerically equivalent to the true joint diplotype probability
in that regime.

**Implementation** (`lmm.py::scan_chromosome`, `model='full'`):

```python
ii, jj = np.triu_indices(J)
probs  = probs[:, :, ii] * probs[:, :, jj]    # (L, m, 36)
```

Betas shape: `(n_loci, 36)`. Effective df: 35 (one sum-to-one constraint remains).
Use full diplotype when testing dominance or parent-of-origin effects; otherwise
the additive model uses 7 df and has higher power for purely additive traits.

---

## 5. Multiple imputation (MI) scan

**miQTL name:** `scan.h2lmm(use.multi.impute=TRUE, num.imp=K)`  
**Implementation:** `scan.py::scan_h2lmm_mi`  
**Applies to LMM only** — not implemented for NB-GLMM.

The MI scan converts haplotype uncertainty (soft probabilities) into $K$ discrete
hard-call draws and averages the resulting LOD scores.

**Algorithm:**

1. **Pre-estimate h²** once per chromosome from the soft haplotype probs.
   This h² is held fixed across all $K$ draws.

2. **For each draw $k = 1, \ldots, K$:**
   - At each locus $\ell$ and strain $i$, sample a hard founder assignment:
     $z_{i\ell}^{(k)} \sim \text{Categorical}(h_{i\ell})$,
     using inverse CDF (cumulative $h_{i\ell}$, compare uniform draw).
   - Build binary hard-call matrix $H_\ell^{(k)} \in \{0,1\}^{m \times J}$.
   - Run `scan_h2lmm` with fixed h² → LOD$^{(k)}_\ell$.

3. **Average:** $\text{LOD}^{\text{MI}}_\ell = K^{-1}\sum_k \text{LOD}^{(k)}_\ell$.

**Why MI over soft additive?** The soft additive model assigns fractional weights
to all founders at uncertain loci, implicitly assuming the trait effect is linear
in founder dosage. MI makes no such assumption — it samples the discrete ancestry
state and averages the discrete LOD. For DO outbred mice with high locus uncertainty,
MI gives better-calibrated LOD scores. For fully inbred CC strains, soft additive ≈ MI
because ancestries are already near-deterministic.

**Choosing $K$:** The variance of the averaged LOD decreases as $1/K$, so
there are diminishing returns beyond a moderate number of draws. $K = 3$–$5$
is adequate for screening; $K = 11$ is the miQTL reference default (used in
mosaiQTL validation for parity); $K = 50$–$100$ is recommended for
publication-quality results at loci with high haplotype uncertainty (e.g., DO
mice where multiple founders share roughly equal probability). For fully inbred
CC strains, $K = 11$ is already stable because per-locus ancestry is
near-deterministic. There is no statistical argument for exactly 11 — it is
a convention, not a requirement.

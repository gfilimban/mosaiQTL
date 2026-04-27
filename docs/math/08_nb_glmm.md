# NB-GLMM: Two-Stage Negative Binomial Scan

**Module:** `qtl/nb.py`  
**miQTL parallel:** none — mosaiQTL-native; design inspired by NEBULA (He et al. 2021)
and edgePython (Pachter 2026)

---

## 1. Why NB-GLMM over LMM

Single-nucleus RNA-seq counts are overdispersed integers. Log-normalizing and
treating as continuous (LMM) can inflate false positives for lowly-expressed
genes and loses information about the count distribution. The NB-GLMM models
the count-generating process explicitly:

- **NB variance function** $\operatorname{Var}(Y) = \mu + \phi\mu^2$ captures
  both Poisson variance ($\mu$) and overdispersion ($\phi\mu^2$).
- **Offset** $\log(n_i \bar{s}_i)$ accounts for library size differences between
  strains without normalization.
- **Random effect** $u_i$ absorbs polygenic background without requiring a
  separate normalization step.

No RINT, no log-normalization, no strain-mean reduction — raw counts in, LOD out.

---

## 2. Cell-level model (Stage 1)

For gene $g$, cell $j$ belonging to strain $i$:

$$Y_{ijg} \sim \text{NB}(\mu_{ijg},\, \phi_g), \qquad
  \log\mu_{ijg} = x_j^\top \beta_g + \log s_j + u_i$$

$$u_i \sim \mathcal{N}(0,\, \sigma^2_g)$$

where:
- $\phi_g$ — cell-level NB dispersion (inverse of size parameter; $\operatorname{Var} = \mu + \phi\mu^2$)
- $s_j$ — library size of cell $j$ (total UMI count)
- $u_i$ — subject/strain random effect, absorbing within-strain cell-to-cell correlation
- $\sigma^2_g$ — subject-level overdispersion variance

This is the **NEBULA-LN** model (He et al. 2021): the NB-gamma mixture is approximated
via a log-normal (LN) for the random effect, enabling closed-form pseudobulk
likelihood via the LN-NB marginal.

---

## 3. Stage 1 fitting (`nb.py::fit_null_sc_native`)

For each gene passing expression filters (mean CPC $> 0.005$, expressed in $\geq 5$ cells):

**Step 1 — Pseudobulk per subject:**
$S_i = \sum_{j \in i} Y_{ijg}$ (sum, not mean).

**Step 2 — Alternate IRLS + Newton-Raphson:**

- IRLS estimates $\hat\beta$ (fixed effects) holding $k = 1/\sigma^2$ fixed,
  using NB working weights $W = \mu_s/(1 + \mu_s/k)$ and working response
  $z = \eta + (S - \mu_s)/\mu_s$.
- Newton-Raphson updates $k$ (hence $\sigma^2$) using the pseudobulk marginal
  NB gradient and Hessian via digamma/polygamma functions.
- Alternated for 8 iterations until convergence.

**Step 3 — Cell-level $\phi$ via method of moments:**
After obtaining $\hat\mu_s$ per strain, the scale factor $v_i = S_i / \hat\mu_i$
is propagated to cells. The cell-level dispersion is:

$$\hat\phi_g = \text{mean}_j\!\left[\frac{(Y_{ijg} - \hat\mu_{ijg})^2 - \hat\mu_{ijg}}{\hat\mu_{ijg}^2}\right]$$

---

## 4. Empirical Bayes dispersion shrinkage (`nb.py::_wleb_shrink`)

Per-gene MOM estimates $\hat\phi_g$ are noisy, especially for low-count genes.
EB shrinkage pulls each $\hat\phi_g$ toward a trended prior that depends on mean
expression — following the weighted likelihood empirical Bayes (WLEB) framework
of Robinson & Smyth 2007 (tagwise dispersion in edgeR).

**Algorithm:**
1. Fit a piecewise bin-mean prior $\tilde\phi(\mu)$ by binning genes into 10 expression
   quantile bins and computing the median $\hat\phi$ per bin.
2. Posterior: $\hat\phi^{\text{shrunk}}_g = \frac{d_0 \tilde\phi(\bar\mu_g) + \hat\phi_g}{d_0 + 1}$
   where $d_0 = 10$ (prior df).

The shrunk dispersion $\hat\phi^{\text{shrunk}}_g$ is used in all Stage 2 scans.
This prevents high-variance per-gene estimates from producing unstable score tests.

---

## 5. Pseudobulk scan model (Stage 2)

For **sample** $i$ (individual mouse replicate, $i = 1,\ldots,n$) and locus $\ell$:

$$S_{ig} \sim \text{NB}(\mu_{ig},\, \hat\phi_g^{\text{shrunk}})$$

$$H_0:\; \log\mu_{ig} = x_i^\top\beta_0 + \log(n_i \bar{s}_i)$$

$$H_1:\; \log\mu_{ig} = H_{i\ell}^\top\gamma + x_i^\top\beta + \log(n_i \bar{s}_i)$$

where $n_i$ is the cell count for sample $i$ and $\bar{s}_i$ is its mean library size.

**Observation level:** pseudobulk $S_{ig}$ is aggregated per sample (individual
mouse), not per strain. $n$ observations index individual animals; multiple samples
share the same genotype. The haplotype matrix $H_{i\ell}$ for sample $i$ is
the strain-level probability row of $i$'s strain $s(i)$:

$$H_{i\ell} = H_{s(i),\ell} \in \mathbb{R}^J$$

This is the **opposite** of the LMM path, which reduces $n$ sample phenotypes to
$m$ strain means before scanning. NB-GLMM keeps all $n$ pseudobulk observations
and expands the $m$-row haplotype matrix to $n$ rows via the strain-to-sample
mapping.

**LOCO kinship random effect via PQL + Woodbury:**
A kinship random effect $u \sim \mathcal{N}(0, \tau K^{(-c)})$ can be integrated
into Stage 2 via Penalized Quasi-Likelihood (PQL, Breslow & Clayton 1993).
The PQL iteration linearizes the NB likelihood into a working Gaussian response,
then solves the resulting LMM using the Woodbury identity to keep all operations
at $m \times m$ cost:

$$V^{-1} = W^{-1} - W^{-1}Z(Z^\top W^{-1}Z + \tau^{-1}K^{(-c)-1})^{-1}Z^\top W^{-1}$$

where $W = \text{diag}(\mu/(1+\hat\phi\mu))$ are the NB working weights.

---

## 6. Score test

The NB score test at locus $\ell$ avoids re-fitting the model under $H_1$ at
every locus — only the null $\hat\mu_i$ is needed:

$$U_\ell = H_\ell^\top (S - \hat\mu),  \qquad
  \mathcal{I}_\ell = H_\ell^\top W H_\ell - H_\ell^\top W X (X^\top W X)^{-1} X^\top W H_\ell$$

$$Q_\ell = U_\ell^\top \mathcal{I}_\ell^{-1} U_\ell \sim \chi^2(J-1)$$

$$\text{LOD}_\ell = \frac{Q_\ell}{2 \ln 10}$$

The $\chi^2(J-1)$ approximation holds under standard regularity conditions.
The null $\hat\mu$ is computed once per gene per chromosome; the score is
then vectorized over all $L$ loci via einsum — $O(LmJ^2)$ total per chromosome.

**Key property:** the score test only requires the null fit. No IRLS re-convergence
per locus — this is the analogue of the FWL trick in `lmm.py` that gives
$O(1)$ per-locus cost beyond the null.

# Permutation Thresholds and Positional Confidence Intervals

**Module:** `qtl/permutation.py`  
**miQTL parallel:** `generate.sample.outcomes.matrix()`, `run.threshold.scans()`,
`get.gev.thresholds()`, `run.positional.scans()`

---

## 1. Why permutation thresholds

Genome-wide LOD significance thresholds derived from $\chi^2$ or $F$ approximations
are anti-conservative for multiparent populations because:
- Multiple testing across $L$ correlated loci inflates Type I error.
- The correlation structure between loci (LD / haplotype blocks) is
  population-specific and cannot be captured by Bonferroni correction.

Permutation breaks the genotype–phenotype link while preserving the LD and
kinship structure of the genome. The result is an empirical null distribution
of genome-wide max-LOD from which a threshold at any $\alpha$ level is read directly.

The two scan engines require different permutation strategies:

| | NB-GLMM | LMM |
|---|---|---|
| What is permuted | Strain labels on pseudobulk count matrix | Phenotype vector $y$ |
| RINT applied | **No** — raw counts only | Yes (default) |
| Fixed parameter | Dispersion $\hat\phi$ from Stage 1 | $\hat{h}^2$ from null REML |
| Null score statistic | $Q_\ell \sim \chi^2(J-1)$ | LOD from RSS ratio |

---

## 2. Generating null outcomes

**Permutation** (default): shuffle the mapping from strains to phenotype values,
breaking geno-pheno association while preserving marginal distributions.

- **LMM:** shuffle the continuous phenotype vector $y$. Each permuted $y^{(b)}$
  is re-RINT-transformed and optionally reduced to strain means via Z-matrix
  before scanning.

- **NB-GLMM:** shuffle strain labels on the pseudobulk count matrix — i.e.,
  permute the row order of $S \in \mathbb{Z}^{m \times G}$ so that each strain's
  count profile is assigned to a different genotype row. RINT is never applied.
  The dispersion $\hat\phi_g$ (estimated from real cell-level data in Stage 1) is
  held fixed; only the strain-to-genotype assignment is permuted.

**Bootstrap** (used for positional CI, section 5): resample with replacement to
produce phenotype vectors with shifted peaks for CI estimation. For LMM: resample
from centered $y - \bar{y}$. For NB-GLMM: resample strain rows from $S$ with
replacement (count-preserving bootstrap).

---

## 3. fix\_h² (LMM) and fix\_φ (NB-GLMM)

Re-estimating model parameters for every permuted phenotype is statistically
correct but expensive.

**LMM — fix\_h²:** Pre-estimate $\hat{h}^2_c$ once from the original phenotype
per chromosome via REML, then reuse across all $B$ permutations.

$$\hat{h}^2_c = \arg\max_{h^2} \ell_{\text{REML}}(h^2 \mid y_{\text{orig}}, K^{(-c)})$$

Under $H_0$ the permuted phenotype has approximately zero heritability;
using $\hat{h}^2 > 0$ from the original data is slightly conservative.
miQTL uses `fix.h2=TRUE` by default; mosaiQTL matches this.

**NB-GLMM — fix\_φ:** The Stage 1 dispersion $\hat\phi_g$ is always fixed — it
is never re-estimated during permutation scans because Stage 1 runs on real
cell-level data, which is unaffected by the permutation of strain labels. The NB
score test at each locus conditions on $\hat\phi_g$ throughout, so no additional
fix step is needed. Stage 2 permutation scans simply run the score test with the
permuted strain assignment and the same fixed $\hat\phi_g$.

---

## 4. Threshold scan loop

**LMM path** (`run_threshold_scans`):

For each permuted $y^{(b)}$:
1. Apply RINT transform.
2. Reduce to strain means via Z-matrix if replicates present.
3. For each chromosome $c$: run `scan_chromosome` with fixed $\hat{h}^2_c$.
4. Record $M^{(b)} = \max_c \max_\ell \text{LOD}_\ell^{(b)}$.

**NB-GLMM path:**

For each permuted strain-label assignment $\pi^{(b)}$:
1. Reorder pseudobulk rows: $S^{(b)}_{\pi^{(b)}(i)} \leftarrow S_i$.
2. For each chromosome $c$: run NB score scan (fixed $\hat\phi_g$) with permuted
   count matrix against original haplotype probs.
3. Record $M^{(b)} = \max_c \max_\ell Q_\ell^{(b)} / (2\ln 10)$.

Both paths produce $\{M^{(b)}\}_{b=1}^B$ — the empirical null distribution of
genome-wide max-LOD.

---

## 5. GEV threshold fitting

The distribution of genome-wide maximum statistics converges to the
**Generalised Extreme Value (GEV)** distribution (Fisher–Tippett–Gnedenko theorem).
Fitting GEV to $\{M^{(b)}\}$ is more stable than empirical quantiles when $B < 500$.

**GEV density:**

$$f(x;\,\mu,\sigma,\xi) = \frac{1}{\sigma}\!\left(1+\xi\frac{x-\mu}{\sigma}\right)^{-1/\xi-1}
  \!\exp\!\left[-\!\left(1+\xi\frac{x-\mu}{\sigma}\right)^{-1/\xi}\right]$$

Three families: Gumbel ($\xi=0$), Fréchet ($\xi>0$), Weibull ($\xi<0$).

**Implementation** (`permutation.py::get_gev_thresholds`):

```python
finite = max_lods[np.isfinite(max_lods)]
c, loc, scale = genextreme.fit(finite)
threshold = genextreme.ppf(percentile, c, loc=loc, scale=scale)
```

Default `percentile = 0.95` → genome-wide 5% significance threshold.  
The empirical quantile is also returned for comparison. With $B \geq 1000$ both converge.

---

## 6. Positional confidence interval via parametric bootstrap

**LMM path:** Bootstrap resamples with replacement from centered $y - \bar{y}$
(continuous). For each replicate: apply RINT, run `scan_chromosome` on the peak
chromosome with fixed $\hat{h}^2$, record peak position (Mb). CI is the empirical
$[\alpha/2, 1-\alpha/2]$ quantile of the bootstrap peak distribution.

**NB-GLMM path:** Bootstrap resamples strain rows from $S$ with replacement
(count-preserving). For each replicate: run NB score scan on the peak chromosome,
record peak position. CI is computed identically.

**Versus LOD-drop CI:** The LOD-drop method (Manichaikul et al. 2006) returns
the genomic interval where LOD drops by 1.5 below the peak (≈ 95% Bayes CI under
Haley–Knott approximation). It is faster but ignores phenotypic sampling noise.
The bootstrap CI explicitly propagates phenotype uncertainty into positional
uncertainty and is the preferred method for publication.

Default $\alpha = 0.05$ → 95% CI. P2-S10 gate: CI contains planted locus in
$\geq 80\%$ of bootstrap replicates at moderate effect size.

---

## 7. PSD projection of the sample-level LOCO kinship

**Context.** The parametric null draw in `run_threshold_scans` requires a
positive semi-definite (PSD) kinship matrix $K_n$ at sample level.  When a
$Z$-matrix is supplied (replicate design), $K_n$ is constructed as

$$
K_n = Z \, K_m \, Z^\top
$$

where $K_m$ is the $m \times m$ strain-level LOCO kinship reconstructed from
its eigendecomposition `(evc_m * ev_m) @ evc_m.T`.

**The problem.** The strain-level LOCO kinship $K_m$ is theoretically PSD but
accumulates floating-point error of $\mathcal{O}(10^{-7})$ in its near-zero
eigenvalues (observed: $-3.09 \times 10^{-7}$).  The previous PSD check in
`_draw_parametric_batch` recomputed `eigvalsh(K)` with a machine-epsilon
tolerance that is orders of magnitude tighter than this noise, triggering a
false `KinshipSingularError` and skipping every chromosome — exhausting the
null draw and collapsing the GEV threshold to 0.0 so every gene is called
as an eGene.

**The fix.** `_draw_parametric_batch` already receives `ev` (the
eigenvalues of $K$) and `evc` (its eigenvectors).  Rather than recomputing
`eigvalsh(K)`, we use `ev` directly, clamp near-zero negatives to zero, and
draw $U$ from the eigendecomposition:

$$
U \;=\; V \,\operatorname{diag}\!\bigl(\sqrt{\max(\lambda_i,0)\,\sigma^2_g}\bigr)\, Z,
\quad Z \sim \mathcal{N}(0, I)
$$

This is mathematically equivalent to $U \sim \mathcal{N}(0,\,K\sigma^2_g)$
when $K$ is exactly PSD, and draws from the nearest PSD projection otherwise.
`KinshipSingularError` is raised only when $\min(\lambda_i) < -10^{-4}$, which
signals a genuine defect rather than float noise.

**Implementation** (`permutation.py::_draw_parametric_batch`):

```python
if float(ev.min()) < -1e-4:
    raise KinshipSingularError(...)
sqrt_ev = np.sqrt(np.maximum(ev, 0.0) * sigma2_g)
U_mat = evc @ (sqrt_ev[:, None] * rng.standard_normal((len(ev), n_perms)))
```

**Why not Tikhonov regularisation?** Adding $\varepsilon I$ shifts *all*
eigenvalues uniformly and changes the kinship structure for the $m$ positive
eigenvalues that are correct.  Clamping only corrects the $n - m$ eigenvalues
that are theoretically zero — the kinship signal is preserved exactly.

---

## References

[1] Higham, N.J. (1988). Computing a nearest symmetric positive semidefinite
matrix. *Linear Algebra and its Applications*, 103, 103–118.
<https://doi.org/10.1016/0024-3795(88)90223-6>

[2] Manichaikul, A. et al. (2006). Poor performance of bootstrap confidence
intervals for the location of a quantitative trait locus. *Genetics*, 174(1),
481–489. <https://doi.org/10.1534/genetics.106.061549>

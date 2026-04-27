# LMM Scan: h² Estimation and Vectorized Chromosome Scan

**Modules:** `qtl/lmm.py`  
**miQTL parallel:** `scan.h2lmm()` (ROP/additive) → `lmmbygls()` inner loop

---

## 1. Model

$$y = H_\ell \gamma + X \alpha + u + \varepsilon$$

$$u \sim \mathcal{N}(0,\, \sigma^2_g K^{(-c)}), \quad \varepsilon \sim \mathcal{N}(0,\, \sigma^2_e I)$$

- $y \in \mathbb{R}^m$ — strain-mean phenotype (RINT-transformed)
- $H_\ell \in \mathbb{R}^{m \times J}$ — haplotype probabilities at locus $\ell$
- $X \in \mathbb{R}^{m \times p}$ — covariates including intercept
- $K^{(-c)}$ — LOCO kinship for chromosome $c$

The variance of $y$ is $V = \sigma^2_e (\delta K + I)$ where $\delta = \sigma^2_g / \sigma^2_e = h^2/(1-h^2)$.

---

## 2. Profile REML h² estimation

After eigendecomposing $K = U D U^\top$, rotate: $\tilde{y} = U^\top y$, $\tilde{X} = U^\top X$.
The rotated variance is diagonal: $\operatorname{Var}(\tilde{y}_k) = \sigma^2_e(\delta\lambda_k + 1)$.

**Profile REML log-likelihood** as a function of $h^2$ only:

$$\ell_{\text{REML}}(h^2) = -\frac{n-p}{2}\Bigl[\log(2\pi\hat\sigma^2_e) + 1\Bigr]
  - \frac{1}{2}\sum_k \log(\delta\lambda_k+1)
  - \frac{1}{2}\log\bigl|(\tilde{X}^\top W \tilde{X})\bigr|$$

where weights $w_k = 1/(\delta\lambda_k+1)$, $W = \text{diag}(w_k)$, and
$\hat\sigma^2_e = \tilde{r}^\top W \tilde{r} / (n-p)$ with $\tilde{r} = \tilde{y} - \tilde{X}\hat\alpha$.

**Three-step algorithm** (`lmm.py::estimate_h2_null`):

1. **Grid search** — evaluate $-\ell_{\text{REML}}$ at 20 equally spaced points on $[0.01, 0.99]$; identify the best grid interval $[h^2_{\text{lo}}, h^2_{\text{hi}}]$.

2. **Brent refinement** — run `scipy.optimize.minimize_scalar` with `method='bounded'` inside $[h^2_{\text{lo}}, h^2_{\text{hi}}]$, tolerance $10^{-8}$; produces $\hat{h}^2_{\text{opt}}$.

3. **Zero safeguard** — evaluate REML at $h^2 = 10^{-6}$ (pure OLS); if this exceeds $\hat{h}^2_{\text{opt}}$ on the likelihood scale, return $h^2 = 0$. Guards against spurious positive heritability from a flat profile.

h² is estimated **once under the null model** (no QTL term) per chromosome per phenotype. It is held fixed for all loci during the scan (see section 4).

---

## 3. LOD formula

The genome scan tests $H_0: \gamma = 0$ vs $H_1: \gamma \ne 0$ using the likelihood ratio statistic converted to LOD:

$$\text{LOD}_\ell = \frac{n}{2} \log_{10}\!\left(\frac{\text{RSS}_0}{\text{RSS}_1(\ell)}\right)$$

- Denominator is $n$ (ML), **not** $n-p$ — matches miQTL / qtl2 convention.
- REML is used only for h² estimation; the LOD itself uses the ML formula.
- $\text{RSS}_0$ = null model (no haplotype term), computed once.
- $\text{RSS}_1(\ell)$ = alternative model with $H_\ell$, computed per locus.

---

## 4. FWL vectorized batch scan

Frisch–Waugh–Lovell theorem: the coefficient $\hat\gamma$ and RSS reduction from
adding $H_\ell$ to a model already containing $X$ equals running OLS of $M_X H_\ell$
on $M_X y$, where $M_X = I - X(X^\top W X)^{-1} X^\top W$ is the weighted residual
maker.

**Key insight:** $M_X$ depends only on $X$ and the weight vector $w$, both of which
are identical for every locus on a chromosome (same $h^2$, same $X$). This allows
computing the null model **once** and replacing the per-locus loop with tensor ops.

**Algorithm** (`lmm.py::scan_chromosome`):

```
# Shared across all loci on one chromosome:
w_sqrt = 1 / sqrt(δ λ + 1)           # (n,)  weight square-roots
yt     = (U'y) * w_sqrt               # weighted rotated phenotype
xt     = (U'X) * w_sqrt[:,None]       # weighted rotated covariates
e0     = yt - xt @ solve(xt'xt, xt'yt)   # null residual (n,)
rss0   = e0 @ e0                      # scalar null RSS

# Batched over L loci:
H_t    = einsum('nk,lnj->lkj', U, H) * w_sqrt   # (L, n, J) rotated+weighted
H_perp = H_t - xt @ (solve(xt'xt, xt') @ H_t)   # (L, n, J) FWL residual
c      = einsum('lnj,n->lj', H_perp, e0)          # (L, J)   cross-term
G      = einsum('lnj,lnk->ljk', H_perp, H_perp)  # (L, J, J) Gram matrices
β̂      = pinv(G) @ c                              # (L, J)   betas (batched)
rss1   = rss0 - einsum('lj,lj->l', c, β̂)          # (L,)     alt RSS per locus
LOD    = (n/2) * log10(rss0 / rss1)               # (L,)
```

The result is equivalent to `qtl2::scan_pg_onechr` (Rcpp), now in NumPy.

---

## 5. All-J founders with pseudoinverse

All $J = 8$ founders are included in $H_\ell$, making the Gram matrix $G$ rank-deficient:
the sum-to-one constraint $H_\ell \mathbf{1}_J = \mathbf{1}_n$ means $\mathbf{1}_J$
lies in the null space of $G$ after FWL residualization (since $\mathbf{1}_n \in \text{col}(X)$
via the intercept column).

$G$ is rank $J-1$ at most loci; rank $J-2$ at loci where one founder has near-zero
probability. `np.linalg.pinv(G, rcond=1e-10)` handles both cases, matching R's
column-pivoting `lm.fit()` exactly. Using $J-1$ columns with ridge regression fails
because R does not always drop the last founder — it drops whichever column is
most collinear (often CAST or PWK at a given locus).

The minimum-norm pinv solution satisfies $\sum_f \hat\gamma_f \approx 0$ because
$\mathbf{1}_J \in \text{null}(G)$ and the minimum-norm solution is orthogonal to
the null space.

---

## 6. Per-locus F-test (secondary statistic)

The F-statistic is reported alongside LOD for completeness:

$$F_\ell = \frac{(\text{RSS}_0 - \text{RSS}_1(\ell)) / q_\ell}{\text{RSS}_1(\ell) / (n - p - q_\ell)}$$

where $q_\ell = \text{rank}(G_\ell)$ is the effective degrees of freedom
(normally $J-1 = 7$; $J-2$ when a founder is absent). The $p$-value uses
$F(q_\ell,\, n-p-q_\ell)$. LOD is the primary statistic for threshold comparison;
F/p are supplementary.

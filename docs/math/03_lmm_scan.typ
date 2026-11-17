#import "_setup.typ": *
#show: mathdoc

= LMM Scan: #mi(`h^2`); Estimation and Vectorized Chromosome Scan
#label("lmm-scan-h²-estimation-and-vectorized-chromosome-scan")
#strong[Modules:] `qtl/lmm.py` \ #strong[miQTL parallel:] `scan.h2lmm()` (ROP/additive) → `lmmbygls()` inner loop

#divider()

== 1. Model
<1-model>
#mitex(`y = H_\ell \gamma + X \alpha + u + \varepsilon`)

#mitex(`u \sim \mathcal{N}(0,\, \sigma^2_g K^{(-c)}), \quad \varepsilon \sim \mathcal{N}(0,\, \sigma^2_e I)`)

- #mi(`y \in \mathbb{R}^n`); --- per-sample phenotype, RINT-transformed at sample level (#mi(`n = m`); strains when there is one sample per strain)
- #mi(`H_\ell \in \mathbb{R}^{n \times J}`); --- haplotype probabilities at locus #mi(`\ell`);; with replicates, the strain-level #mi(`P_\ell \in \mathbb{R}^{m \times J}`); is expanded to #mi(`H_\ell = Z P_\ell`);
- #mi(`X \in \mathbb{R}^{n \times p}`); --- covariates including intercept (an intercept column is prepended when the first column is not all ones)
- #mi(`K^{(-c)}`); --- LOCO kinship for chromosome #mi(`c`);; with replicates, the sample-level #mi(`Z K^{(-c)} Z^\top`); is used and re-eigendecomposed

The variance of #mi(`y`); is #mi(`V = \sigma^2_g K + \sigma^2_e I = \sigma^2\,[h^2 K + (1-h^2) I]`); with #mi(`\sigma^2 = \sigma^2_g + \sigma^2_e`);, #mi(`h^2 = \sigma^2_g/\sigma^2`);; equivalently #mi(`V = \sigma^2_e (\delta K + I)`); with #mi(`\delta = \sigma^2_g / \sigma^2_e = h^2/(1-h^2)`);.

#divider()

== 2. Profile ML #mi(`h^2`); estimation
#label("2-profile-ml-h²-estimation")
After eigendecomposing #mi(`K = U \Lambda U^\top`); (eigenvalues #mi(`\lambda_k`);), rotate: #mi(`\tilde{y} = U^\top y`);, #mi(`\tilde{X} = U^\top X`);.
The rotated variance is diagonal: #mi(`\operatorname{Var}(\tilde{y}_k) = \sigma^2 w_k`); with #mi(`w_k = h^2\lambda_k + (1-h^2)`);.

#strong[Profile ML log-likelihood] as a function of #mi(`h^2`); only:

#mitex(`\ell_{\text{ML}}(h^2) = -\frac{n}{2}\Bigl[\log(2\pi\hat\sigma^2) + 1\Bigr]
  - \frac{1}{2}\sum_k \log w_k`)

where #mi(`W = \text{diag}(w_k^{-1})`);, #mi(`\hat\alpha = (\tilde{X}^\top W \tilde{X})^{-1}\tilde{X}^\top W \tilde{y}`);, and
#mi(`\hat\sigma^2 = \tilde{r}^\top W \tilde{r} / n`); with #mi(`\tilde{r} = \tilde{y} - \tilde{X}\hat\alpha`);.
ML is used because the LOD is a likelihood-ratio test between models with different fixed effects, for which REML likelihoods are not comparable.

#strong[History:] before 2026-05-31 (commit `11a4b68e`) `estimate_h2_null` defaulted to profile REML,
#mi(`\ell_{\text{REML}}(h^2) = -\frac{n-p}{2}[\log(2\pi\hat\sigma^2) + 1] - \frac{1}{2}\sum_k \log w_k - \frac{1}{2}\log|\tilde{X}^\top W \tilde{X}|`);
with #mi(`\hat\sigma^2 = \tilde{r}^\top W \tilde{r}/(n-p)`);. REML is still available as `method='reml'`, but no scan path calls it.

#strong[Three-step algorithm] (`lmm.py::estimate_h2_null`, default `method='ml'`):

+ #strong[Grid search] --- evaluate #mi(`-\ell_{\text{ML}}`); at 20 equally spaced points on #mi(`[0.01, 0.99]`);; the best grid point and its two neighbours give the bracket #mi(`[h^2_{\text{lo}}, h^2_{\text{hi}}]`);.

+ #strong[Brent refinement] --- run `scipy.optimize.minimize_scalar` with `method='bounded'` inside #mi(`[h^2_{\text{lo}}, h^2_{\text{hi}}]`);, `xatol` #mi(`10^{-8}`);; produces #mi(`\hat{h}^2_{\text{opt}}`);.

+ #strong[Zero safeguard] --- evaluate #mi(`\ell_{\text{ML}}`); at #mi(`h^2 = 10^{-6}`); (effectively OLS); if it exceeds #mi(`\ell_{\text{ML}}(\hat{h}^2_{\text{opt}})`);, return #mi(`h^2 = 10^{-6}`);. Guards against spurious positive heritability from a flat profile. Inside the likelihood #mi(`h^2`); is clipped to #mi(`[10^{-6},\, 1-10^{-6}]`);.

#mi(`h^2`); is estimated #strong[once under the null model] (no QTL term) per chromosome per phenotype. It is held fixed for all loci during the scan (see section 4).

#divider()

== 3. LOD formula
<3-lod-formula>
The genome scan tests #mi(`H_0: \gamma = 0`); vs #mi(`H_1: \gamma \ne 0`); using the likelihood ratio statistic converted to LOD:

#mitex(`\text{LOD}_\ell = \frac{n}{2} \log_{10}\!\left(\frac{\text{RSS}_0}{\text{RSS}_1(\ell)}\right)`)

- Denominator is #mi(`n`); (ML), #strong[not] #mi(`n-p`); --- matches miQTL / qtl2 convention.
- #mi(`h^2`); is also estimated by ML (section 2) and held fixed, so both RSS terms use the same weights.
- #mi(`\text{RSS}_0`); = weighted null-model RSS (no haplotype term), computed once.
- #mi(`\text{RSS}_1(\ell)`); = weighted alternative-model RSS with #mi(`H_\ell`);, computed per locus and floored at #mi(`10^{-30}`);; #mi(`\text{LOD}_\ell = 0`); when #mi(`\text{RSS}_1(\ell) \ge \text{RSS}_0`);.

#divider()

== 4. FWL vectorized batch scan
<4-fwl-vectorized-batch-scan>
Frisch--Waugh--Lovell theorem: the coefficient #mi(`\hat\gamma`); and RSS reduction from
adding #mi(`H_\ell`); to a model already containing #mi(`X`); equals running OLS of #mi(`M_X H_\ell`);
on #mi(`M_X y`);, where #mi(`M_X = I - X(X^\top W X)^{-1} X^\top W`); is the weighted residual
maker.

#strong[Key insight:] #mi(`M_X`); depends only on #mi(`X`); and the weight vector #mi(`w`);, both of which
are identical for every locus on a chromosome (same #mi(`h^2`);, same #mi(`X`);). This allows
computing the null model #strong[once] and replacing the per-locus loop with tensor ops.

#strong[Algorithm] (`lmm.py::precompute_chromosome` once per chromosome, then
`lmm.py::scan_chromosome` → `_scan_chromosome_core` once per phenotype):

```
# Once per chromosome, shared by all genes (precompute_chromosome):
H_t_raw = einsum('nk,lnj->lkj', U, H)            # (L, n, J) rotated, unweighted
#   replicates: einsum('ks,lsj->lkj', U'Z, P)    # Z expansion fused into the rotation

# Once per phenotype (_scan_chromosome_core):
w_sqrt = 1 / sqrt(h2 * lam + (1 - h2))           # (n,)  weight square-roots
yt     = (U'y) * w_sqrt                          # weighted rotated phenotype
xt     = (U'X) * w_sqrt[:,None]                  # weighted rotated covariates
e0     = yt - xt @ solve(xt'xt, xt'yt)           # null residual (n,)
rss0   = e0 @ e0                                 # scalar null RSS

# Batched over L loci:
H_t    = H_t_raw * w_sqrt[None,:,None]           # (L, n, J) rotated+weighted
H_perp = H_t - xt @ (solve(xt'xt, xt') @ H_t)    # (L, n, J) FWL residual
c      = einsum('lnj,n->lj', H_perp, e0)         # (L, J)   cross-term
G      = einsum('lnj,lnk->ljk', H_perp, H_perp)  # (L, J, J) Gram matrices
beta_h = pinv(G, rcond=1e-10) @ c                # (L, J)   betas (batched)
rss1   = max(rss0 - einsum('lj,lj->l', c, beta_h), 1e-30)   # (L,) alt RSS
LOD    = where(rss1 < rss0, (n/2) * log10(rss0 / rss1), 0)  # (L,)
```

The result is equivalent to `qtl2::scan_pg_onechr` (Rcpp), now in NumPy.

#divider()

== 5. All-J founders with pseudoinverse
<5-all-j-founders-with-pseudoinverse>
All #mi(`J = 8`); founders are included in #mi(`H_\ell`);, making the Gram matrix #mi(`G`); rank-deficient:
the sum-to-one constraint #mi(`H_\ell \mathbf{1}_J = \mathbf{1}_n`); means #mi(`\mathbf{1}_J`);
lies in the null space of #mi(`G`); after FWL residualization (since #mi(`\mathbf{1}_n \in \text{col}(X)`);
via the intercept column).

#mi(`G`); is rank #mi(`J-1`); at most loci; rank #mi(`J-2`); at loci where one founder has near-zero
probability. `np.linalg.pinv(G, rcond=1e-10)` handles both cases, matching R\'s
column-pivoting `lm.fit()` exactly. Using #mi(`J-1`); columns with ridge regression fails
because R does not always drop the last founder --- it drops whichever column is
most collinear (often CAST or PWK at a given locus).

The minimum-norm pinv solution satisfies #mi(`\sum_f \hat\gamma_f \approx 0`); because
#mi(`\mathbf{1}_J \in \text{null}(G)`); and the minimum-norm solution is orthogonal to
the null space.

#divider()

== 6. Per-locus F-test (secondary statistic)
<6-per-locus-f-test-secondary-statistic>
The F-statistic is reported alongside LOD for completeness:

#mitex(`F_\ell = \frac{(\text{RSS}_0 - \text{RSS}_1(\ell)) / q_\ell}{\text{RSS}_1(\ell) / (n - p - q_\ell)}`)

where #mi(`q_\ell = \text{rank}(G_\ell)`); is the effective degrees of freedom
(normally #mi(`J-1 = 7`);; #mi(`J-2`); when a founder is absent). The #mi(`p`);-value uses
#mi(`F(q_\ell,\, \max(n-p-q_\ell,\, 1))`); and is stored as `ScanResult.p_value`. LOD is the primary statistic for threshold comparison;
F/p are supplementary.

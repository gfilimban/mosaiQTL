# Allele Effect Estimation at a QTL Locus

**Module:** `qtl/effects.py`  
**miQTL parallel:** `get.allele.effects.from.fixef()`, `get.allele.effects.from.ranef()`

> **Scope:** The fixef and ranef estimators described here are **LMM-only**.
> They operate on the FWL gram matrix derived from the LMM eigendecomp.
> For NB-GLMM, allele effects at a peak locus are estimated from the NB
> score gradient (first derivative of the NB log-likelihood with respect to
> $\gamma$ at the null), which uses the NB working weights $W = \mu/(1+\phi\mu)$
> in place of the LMM eigenweights. The NB path is not yet implemented in
> `effects.py` (planned extension).

---

## 1. Setup: FWL gram matrix at one locus

After fitting the genome scan and identifying a peak locus $\ell^*$ on chromosome $c$,
the goal is to estimate the per-founder effect vector $\hat\gamma \in \mathbb{R}^J$
with standard errors.

Using the same FWL construction as the scan (see `03_lmm_scan.md`), at the single
locus of interest:

$$G = H_\perp^\top H_\perp \in \mathbb{R}^{J \times J}, \qquad
  c = H_\perp^\top e_0 \in \mathbb{R}^J$$

where $H_\perp = M_X^W H$ is the haplotype matrix residualized on covariates under
the LMM weight $W = \text{diag}(1/(\delta\lambda_k+1))$, and $e_0 = \tilde{y} - \tilde{X}\hat\alpha_0$
is the null model weighted residual. Both quantities are computed from the
LOCO eigendecomp at chromosome $c$.

$G$ is rank $J-1$ (sum-to-one constraint); its null space is $\text{span}(\mathbf{1}_J)$.
The solution $\hat\gamma$ satisfying $G\hat\gamma = c$ is not unique — the two
methods below differ in which particular solution they return.

---

## 2. fixef — pseudoinverse (matches scan betas exactly)

$$\hat\gamma^{\text{fixef}} = G^+ c$$

where $G^+ = \text{pinv}(G,\, \text{rcond}=10^{-10})$ is the Moore-Penrose pseudoinverse.

The minimum-norm solution is orthogonal to $\text{null}(G) = \text{span}(\mathbf{1}_J)$,
so $\sum_f \hat\gamma_f^{\text{fixef}} \approx 0$. This is identically what
`scan_chromosome` computes at every locus; `fixef` simply re-runs the same
computation at the selected peak.

**Covariance:**

$$\hat\Sigma_{\text{fixef}} = \hat\sigma^2_e \, G^+, \qquad
  \hat\sigma^2_e = \frac{\text{RSS}_1}{\max(n - p - q, 1)}$$

where $q = \text{rank}(G)$ is the effective df and
$\text{RSS}_1 = \max(\text{RSS}_0 - c^\top\hat\gamma, 10^{-30})$.

**Use when:** you want effect estimates that exactly reproduce the scan betas, or
when comparing effects across loci (consistent estimator).

---

## 3. ranef — ridge / BLUP (empirical-Bayes shrinkage)

$$\hat\gamma^{\text{ranef}} = (G + \lambda I)^{-1} c$$

where $\lambda = \hat\sigma^2_e / \hat\sigma^2_a$ is the signal-to-noise ratio
estimated from the data (empirical Bayes).

**Estimating $\lambda$:**

1. Compute the fixef solution $\hat\gamma^{\text{fixef}} = G^+ c$ and its residual
   $\text{RSS}_1 = \text{RSS}_0 - c^\top \hat\gamma^{\text{fixef}}$.
2. $\hat\sigma^2_e = \text{RSS}_1 / \max(n-p-q, 1)$
3. Signal: $\hat\sigma^2_a = \max(\text{RSS}_0 - \text{RSS}_1, 0) / (\max(q,1) \cdot n)$,
   floored at $\hat\sigma^2_e \times 10^{-4}$ to prevent $\lambda \to \infty$.
4. $\lambda = \hat\sigma^2_e / \hat\sigma^2_a$

**Ridge covariance** (sandwich form):

$$\hat\Sigma_{\text{ranef}} = \hat\sigma^2_e \, (G + \lambda I)^{-1} G \, (G + \lambda I)^{-1}$$

This is the standard ridge BLUP covariance — it accounts for the bias introduced
by the shrinkage penalty.

**Use when:** the QTL signal is modest or the number of strains $m$ is small
relative to $J$. Ridge shrinks all founder effects toward zero, reducing variance
at the cost of bias. For large effects with $m \gg J$ the two methods converge.

---

## 4. Returned object

Both methods return a `list[FounderEffect]` with one entry per founder in
`FOUNDER_ORDER`:

```python
@dataclass
class FounderEffect:
    founder:  str    # founder name (e.g., 'A/J')
    estimate: float  # β̂_f
    se:       float  # sqrt(diag(Σ̂))[f]
    z_score:  float  # β̂_f / se
    p_value:  float  # two-sided t-test, df = n - p - q
    ci_lo:    float  # β̂_f − 1.96 × se
    ci_hi:    float  # β̂_f + 1.96 × se
```

The $p$-values use a $t_{n-p-q}$ distribution. For the typical CC scan
($m = 40$, $p = 1$, $q = 7$) df = 32.

---

## 5. Comparison to Diploffect.INLA

Both fixef and ranef treat the haplotype probabilities as fixed (plug-in $H_\ell$).
This ignores uncertainty in the ancestry assignment itself.

Diploffect (Zhang et al. 2014; Keele's INLA implementation) propagates haplotype
uncertainty into the posterior distribution of $\gamma$ by treating $H_\ell$ as a
latent variable and integrating over it. This gives wider, more honest credible
intervals when ancestry probabilities are diffuse, at the cost of substantially
higher computational expense. A `method='diploffect'` path in `get_allele_effects`
is a natural future extension (see `09_extensions.md`).

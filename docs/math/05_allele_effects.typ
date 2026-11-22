#import "_setup.typ": *
#show: mathdoc

= Allele Effect Estimation at a QTL Locus
<allele-effect-estimation-at-a-qtl-locus>
#strong[Module:] `qtl/effects.py` \ #strong[miQTL parallel:] `get.allele.effects.from.fixef()`, `get.allele.effects.from.ranef()`

#quote(block: true)[
#strong[Scope:] The fixef and ranef estimators described here are #strong[LMM-only].
They operate on the FWL gram matrix derived from the LMM eigendecomp.
For NB-GLMM, allele effects at a peak locus are estimated from the NB
score gradient (first derivative of the NB log-likelihood with respect to
#mi(`\gamma`); at the null), which uses the NB working weights #mi(`W = \mu/(1+\phi\mu)`);
in place of the LMM eigenweights. The NB path is not yet implemented in
`effects.py` (planned extension).
]

#divider()

== 1. Setup: FWL gram matrix at one locus
<1-setup-fwl-gram-matrix-at-one-locus>
After fitting the genome scan and identifying a peak locus #mi(`\ell^*`); on chromosome #mi(`c`);,
the goal is to estimate the per-founder effect vector #mi(`\hat\gamma \in \mathbb{R}^J`);
with standard errors.

Using the same FWL construction as the scan (see `03_lmm_scan.typ`), at the single
locus of interest:

#mitex(`G = H_\perp^\top H_\perp \in \mathbb{R}^{J \times J}, \qquad
  c = H_\perp^\top e_0 \in \mathbb{R}^J`)

where #mi(`H_\perp = M_X^W H`); is the haplotype matrix residualized on covariates under
the LMM weight #mi(`W = \text{diag}(1/(\delta\lambda_k+1))`);, and #mi(`e_0 = \tilde{y} - \tilde{X}\hat\alpha_0`);
is the null model weighted residual. Both quantities are computed from the
LOCO eigendecomp at chromosome #mi(`c`);.

`get_allele_effects` fits in #strong[strain-mean space]: with replicates, phenotype (RINT at
sample level) and covariates are averaged to strain means via #mi(`Z`);, strains with no
valid sample are dropped (kinship re-eigendecomposed on the remaining strains), and
#mi(`h^2`); is re-estimated by ML (`estimate_h2_null`) unless supplied.

#mi(`G`); is rank #mi(`J-1`); (sum-to-one constraint); its null space is #mi(`\text{span}(\mathbf{1}_J)`);.
The solution #mi(`\hat\gamma`); satisfying #mi(`G\hat\gamma = c`); is not unique --- the two
methods below differ in which particular solution they return.

#divider()

== 2. fixef --- pseudoinverse (matches scan betas when there are no replicates)
<2-fixef--pseudoinverse-matches-scan-betas-exactly>
#mitex(`\hat\gamma^{\text{fixef}} = G^+ c`)

where #mi(`G^+ = \text{pinv}(G,\, \text{rcond}=10^{-10})`); is the Moore-Penrose pseudoinverse.

The minimum-norm solution is orthogonal to #mi(`\text{null}(G) = \text{span}(\mathbf{1}_J)`);,
so #mi(`\sum_f \hat\gamma_f^{\text{fixef}} \approx 0`);. With one sample per strain and the
same #mi(`h^2`); this is identically what `scan_chromosome` computes at every locus; `fixef`
simply re-runs the same computation at the selected peak. Known issue: with replicates the
scan fits sample-level data with #mi(`Z K Z^\top`); (`04_scan_modes.typ`, Path B) while `fixef`
fits strain means, so the two sets of betas differ.

#strong[Covariance:]

#mitex(`\hat\Sigma_{\text{fixef}} = \hat\sigma^2_e \, G^+, \qquad
  \hat\sigma^2_e = \frac{\text{RSS}_1}{\max(n - p - q, 1)}`)

where #mi(`q = \text{rank}(G)`); (`np.linalg.matrix_rank`, absolute `tol=1e-10`) is the effective df and
#mi(`\text{RSS}_1 = \max(\text{RSS}_0 - c^\top\hat\gamma, 10^{-30})`);.

#strong[Use when:] you want effect estimates that reproduce the scan betas (one sample per strain), or
when comparing effects across loci (consistent estimator).

#divider()

== 3. ranef --- ridge / BLUP (empirical-Bayes shrinkage)
<3-ranef--ridge--blup-empirical-bayes-shrinkage>
#mitex(`\hat\gamma^{\text{ranef}} = (G + \lambda I)^{-1} c`)

where #mi(`\lambda = \hat\sigma^2_e / \hat\sigma^2_a`); is the signal-to-noise ratio
estimated from the data (empirical Bayes).

#strong[Estimating #mi(`\lambda`);:]

+ Compute the fixef solution #mi(`\hat\gamma^{\text{fixef}} = G^+ c`); and its residual
  #mi(`\text{RSS}_1 = \text{RSS}_0 - c^\top \hat\gamma^{\text{fixef}}`);.
+ #mi(`\hat\sigma^2_e = \text{RSS}_1 / \max(n-p-q, 1)`);
+ Signal: #mi(`\hat\sigma^2_a = \max(\text{RSS}_0 - \text{RSS}_1, 0) / (\max(q,1) \cdot n)`);,
  floored at #mi(`\hat\sigma^2_e \times 10^{-4}`); to prevent #mi(`\lambda \to \infty`);.
+ #mi(`\lambda = \hat\sigma^2_e / \hat\sigma^2_a`);

Known issue: dividing the explained sum of squares by #mi(`q \cdot n`); rather than by #mi(`\operatorname{tr}(G)`); (moment estimator #mi(`(\text{RSS}_0 - \text{RSS}_1 - q\hat\sigma^2_e)/\operatorname{tr}(G)`);) mis-scales #mi(`\hat\sigma^2_a`);.

#mi(`G + \lambda I`); is inverted directly, with a pseudoinverse fallback if it is singular.

#strong[Ridge covariance] (sandwich form):

#mitex(`\hat\Sigma_{\text{ranef}} = \hat\sigma^2_e \, (G + \lambda I)^{-1} G \, (G + \lambda I)^{-1}`)

This is the frequentist sampling covariance of the ridge estimator; it does #strong[not] account
for the bias introduced by the shrinkage penalty. Known issue: `ranef` p-values and CIs
nevertheless use it in the same #mi(`t`);-test as `fixef`, as though the estimate were unbiased.

#strong[Use when:] the QTL signal is modest or the number of strains #mi(`m`); is small
relative to #mi(`J`);. Ridge shrinks all founder effects toward zero, reducing variance
at the cost of bias. For large effects with #mi(`m \gg J`); the two methods converge.

#divider()

== 4. Returned object
<4-returned-object>
Both methods return a `list[FounderEffect]` (`models/records.py`) with one entry per founder
in `founder_names[:J]` (default: `genomecache.founders`):

```python
@dataclass
class FounderEffect:
    founder:  str    # founder name
    estimate: float  # effect estimate (see below)
    se:       float  # standard error
    z_score:  float  # estimate / se
    p_value:  float  # two-sided t-test
    ci_lo:    float  # lower 95% bound
    ci_hi:    float  # upper 95% bound
```

with, for founder #mi(`f`); and #mi(`\nu = \max(n-p-q,\, 1)`);:

#mitex(`\text{se}_f = \max\bigl(\sqrt{\hat\Sigma_{ff}},\, 10^{-15}\bigr), \qquad
  z_f = \hat\gamma_f / \text{se}_f, \qquad
  \text{CI}_f = \hat\gamma_f \pm t_{0.975,\nu}\, \text{se}_f`)

The #mi(`p`);-values use a #mi(`t_{\nu}`); distribution, #mi(`p_f = 2\,P(t_\nu > |z_f|)`);. For the typical CC scan
(#mi(`n = m = 40`);, #mi(`p = 1`);, #mi(`q = 7`);) #mi(`\nu = 32`);.

#divider()

== 5. Comparison to Diploffect.INLA
<5-comparison-to-diploffectinla>
Both fixef and ranef treat the haplotype probabilities as fixed (plug-in #mi(`H_\ell`);).
This ignores uncertainty in the ancestry assignment itself.

Diploffect (Zhang et al. 2014; Keele\'s INLA implementation) propagates haplotype
uncertainty into the posterior distribution of #mi(`\gamma`); by treating #mi(`H_\ell`); as a
latent variable and integrating over it. This gives wider, more honest credible
intervals when ancestry probabilities are diffuse, at the cost of substantially
higher computational expense. A `method='diploffect'` path in `get_allele_effects`
is a natural future extension (see `09_extensions.typ`).

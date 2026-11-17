#import "_setup.typ": *
#show: mathdoc

= NB-GLMM: Two-Stage Negative Binomial Scan
<nb-glmm-two-stage-negative-binomial-scan>
#strong[Module:] `qtl/nb.py` \ #strong[miQTL parallel:] none --- mosaiQTL-native; design inspired by NEBULA (He et al. 2021)
and edgePython (Pachter 2026) \
#strong[Status:] experimental. The NB-GLMM is not wired into the Nextflow workflow (`params.model` in
`nextflow.config` is declared but never read under `workflow/`) or the CLI; production mapping uses the
Gaussian LMM. It is reachable only from Python: `scan_nb`, `scan_nb_gene`, or `pipeline.discover(model='nb')`,
which passes no kinship and therefore always takes the no-kinship path of section 6.

#divider()

== 1. Why NB-GLMM over LMM
<1-why-nb-glmm-over-lmm>
Single-nucleus RNA-seq counts are overdispersed integers. Log-normalizing and
treating as continuous (LMM) can inflate false positives for lowly-expressed
genes and loses information about the count distribution. The NB-GLMM models
the count-generating process explicitly:

- #strong[NB variance function] #mi(`\operatorname{Var}(Y) = \mu + \phi\mu^2`); captures
  both Poisson variance (#mi(`\mu`);) and overdispersion (#mi(`\phi\mu^2`);).
- #strong[Offset] accounts for library size differences between samples. Stage 1 uses
  #mi(`\log(\text{subject library size}) + \log(\text{TMM factor})`);; Stage 2 uses #mi(`\log s_i`);
  with #mi(`s_i`); the caller-supplied `lib_sizes` (offset 0 when `None`).
- #strong[Random effect] #mi(`u_i`); absorbs polygenic background without requiring a
  separate normalization step.

No RINT, no log-normalization, no strain-mean reduction --- raw counts in, LOD out.

#divider()

== 2. Cell-level model (Stage 1)
<2-cell-level-model-stage-1>
For gene #mi(`g`);, cell #mi(`j`); belonging to strain #mi(`i`);:

#mitex(`Y_{ijg} \sim \text{NB}(\mu_{ijg},\, \phi_g), \qquad
  \log\mu_{ijg} = x_j^\top \beta_g + \log s_j + u_i`)

#mitex(`u_i \sim \mathcal{N}(0,\, \sigma^2_g)`)

where:

- #mi(`\phi_g`); --- cell-level NB dispersion (inverse of size parameter; #mi(`\operatorname{Var} = \mu + \phi\mu^2`);)
- #mi(`s_j`); --- library size of cell #mi(`j`); (total UMI count)
- #mi(`u_i`); --- subject/strain random effect, absorbing within-strain cell-to-cell correlation
- #mi(`\sigma^2_g`); --- subject-level overdispersion variance

This is the #strong[NEBULA-LN] model (He et al. 2021): the NB-gamma mixture is approximated
via a log-normal (LN) for the random effect, enabling closed-form pseudobulk
likelihood via the LN-NB marginal.

As implemented, #mi(`\sigma^2_g = 1/k`); is estimated from the pseudobulk marginal
#mi(`S_i \sim \text{NB}(\mu_i,\ \text{size}=k)`); (section 3), i.e. a gamma subject effect with
variance #mi(`\sigma^2_g`);; no log-normal integral is computed. It is returned as `sigma_sample`.

#divider()

== 3. Stage 1 fitting (`nb.py::fit_null_sc_native`)
<3-stage-1-fitting-nbpyfit_null_sc_native>
For each gene passing expression filters (mean CPC #mi(`> 0.005`);, expressed in #mi(`\geq 5`); cells):

#strong[Step 1 --- Pseudobulk per subject:]
#mi(`S_i = \sum_{j \in i} Y_{ijg}`); (sum, not mean).

#strong[Step 2 --- Alternate IRLS + Newton-Raphson:]

- IRLS estimates #mi(`\hat\beta`); (fixed effects) holding #mi(`k = 1/\sigma^2`); fixed,
  using NB working weights #mi(`W = \mu_s/(1 + \mu_s/k)`); and working response
  #mi(`z = \eta + (S - \mu_s)/\mu_s`);, with subject offset
  #mi(`\log\!\big(\sum_{j \in i}(s_j + 1)\big) + \log(\text{TMM}_i)`);.
- Newton-Raphson updates #mi(`k`); (hence #mi(`\sigma^2`);) using the pseudobulk marginal
  NB gradient and Hessian via digamma/polygamma functions, starting from the pseudobulk
  moment estimate #mi(`k_0 = 1/\max\{(\operatorname{var}(S) - \bar S)/\bar S^2,\ 10^{-3}\}`);.
- Alternated for a fixed 8 rounds; there is no outer convergence test.

#strong[Step 3 --- Cell-level #mi(`\phi`); via method of moments:]
After obtaining #mi(`\hat\mu_i`); per subject, the scale factor #mi(`v_i = S_i / \hat\mu_i`);
is propagated to cells as #mi(`\hat\mu_{ijg} = v_i \hat\mu_i / n_i = S_i / n_i`); (#mi(`n_i`); = cells in
subject #mi(`i`);). The cell-level dispersion is:

#mitex(`\hat\phi_g = \max\!\left\{\text{mean}_j\!\left[\frac{(Y_{ijg} - \hat\mu_{ijg})^2 - \hat\mu_{ijg}}{\hat\mu_{ijg}^2}\right],\ 10^{-6}\right\}`)

Known issue: because #mi(`\hat\mu_i`); cancels, per-cell library size #mi(`s_j`); never enters
#mi(`\hat\mu_{ijg}`);, so within-subject depth variation is counted as overdispersion (ledger NB01).

#divider()

== 4. Empirical Bayes dispersion shrinkage (`nb.py::_wleb_shrink`)
<4-empirical-bayes-dispersion-shrinkage-nbpy_wleb_shrink>
Per-gene MOM estimates #mi(`\hat\phi_g`); are noisy, especially for low-count genes.
EB shrinkage pulls each #mi(`\hat\phi_g`); toward a trended prior that depends on mean
expression --- following the weighted likelihood empirical Bayes (WLEB) framework
of Robinson & Smyth 2007 (tagwise dispersion in edgeR).

#strong[Algorithm:]

+ Fit a piecewise-constant bin-median prior #mi(`\tilde\phi(\mu)`); by binning genes into
  #mi(`\min(10, \lfloor G_{\text{ok}}/5 \rfloor)`); quantile bins of #mi(`\log\bar\mu_g`); (#mi(`\bar\mu_g`); = mean
  fitted pseudobulk count) and taking the median #mi(`\hat\phi`); per bin; bins with fewer than 3 genes
  keep the global median.
+ Posterior: #mi(`\hat\phi^{\text{shrunk}}_g = \frac{d_0 \tilde\phi(\bar\mu_g) + \hat\phi_g}{d_0 + 1}`);
  where #mi(`d_0 = 10`); (prior df), floored at #mi(`10^{-6}`);.

Known issue: `_wleb_shrink` raises a shape error whenever any gene fails its validity filter
(#mi(`\hat\phi_g \le 0`);, non-finite, or #mi(`\bar\mu_g = 0`);) (ledger NB05).

The shrunk dispersion #mi(`\hat\phi^{\text{shrunk}}_g`); is used in all Stage 2 scans.
This prevents high-variance per-gene estimates from producing unstable score tests.

#divider()

== 5. Pseudobulk scan model (Stage 2)
<5-pseudobulk-scan-model-stage-2>
For #strong[sample] #mi(`i`); (individual mouse replicate, #mi(`i = 1,\ldots,n`);) and locus #mi(`\ell`);:

#mitex(`S_{ig} \sim \text{NB}(\mu_{ig},\, \hat\phi_g^{\text{shrunk}})`)

#mitex(`H_0:\; \log\mu_{ig} = x_i^\top\beta_0 + \log s_i`)

#mitex(`H_1:\; \log\mu_{ig} = H_{i\ell}^\top\gamma + x_i^\top\beta + \log s_i`)

where #mi(`s_i`); is the caller-supplied `lib_sizes` entry for sample #mi(`i`); (#mi(`s_i = 1`);, offset 0,
when `lib_sizes` is `None`).

Known issue: `scan_nb` never computes `lib_sizes`, so by default Stage 2 runs with no library-size
offset (ledger NB04); the docstrings call `lib_sizes` a log offset but the code takes its log again
(ledger NB03). Stage 2 also uses the cell-level #mi(`\hat\phi_g^{\text{shrunk}}`); as the pseudobulk
dispersion (ledger NB02).

#strong[Observation level:] `scan_nb_gene` accepts one pseudobulk row per sample, each labelled
with its genotype (strain) ID; IDs may repeat, so #mi(`n`); rows can share one genotype. Rows are
mapped to genome-cache strains by exact ID match (`utils.build_z_matrix`), and the haplotype row of
sample #mi(`i`); is the probability row of its strain #mi(`s(i)`); with the last founder dropped:

#mitex(`H_{i\ell} = H_{s(i),\ell} \in \mathbb{R}^{J-1}`)

`scan_nb` itself sums cells by the per-cell label it is given and uses the unique labels as genotype
IDs, so through `scan_nb` there is one row per label, and the labels must equal genome-cache strain IDs.

This is the #strong[opposite] of the LMM path, which reduces #mi(`n`); sample phenotypes to
#mi(`m`); strain means before scanning. NB-GLMM keeps all #mi(`n`); pseudobulk observations
and expands the #mi(`m`);-row haplotype matrix to #mi(`n`); rows via the strain-to-sample
mapping.

=== 5.1 Why PQL: the marginal likelihood has no closed form
<51-why-pql-the-marginal-likelihood-has-no-closed-form>
In any GLMM, the random effect #mi(`u`); is a latent variable. The marginal
likelihood requires integrating it out:

#mitex(`p(y \mid \beta, \tau) = \int p(y \mid u,\, \beta)\ p(u \mid \tau)\ du`)

The prior on #mi(`u`); is always Gaussian: #mi(`u \sim \mathcal{N}(0, \tau K_m)`);.
This is true in both LMM and NB-GLMM --- kinship #emph[is] the random effect covariance
in both cases.

#strong[LMM:] #mi(`p(y_i \mid u_i) = \mathcal{N}(x_i^\top\beta + u_i,\; \sigma^2)`);.
Gaussian #mi(`\times`); Gaussian integrates exactly. The marginal is
#mi(`y \sim \mathcal{N}(X\beta,\; \sigma^2 I + \tau ZKZ^\top)`); --- a single matrix solve,
no iteration.

#strong[NB-GLMM:] #mi(`p(y_i \mid u_i) = \text{NB}(\exp(x_i^\top\beta + u_i),\; \phi)`);.
The link is logarithmic, not identity, and the NB distribution is not conjugate
with the Gaussian prior on #mi(`u`);. The integral:

#mitex(`\int \text{NB}\!\left(y_i;\; e^{x_i^\top\beta + u_i},\; \phi\right)\cdot
  \mathcal{N}(u_i;\; 0,\; \tau K_m)\; du_i`)

has no closed form. PQL (Breslow & Clayton 1993) bypasses this by never computing
the integral directly --- instead it approximates the NB log-likelihood locally
as Gaussian at each step and solves that Gaussian system exactly.

=== 5.2 PQL working linearization
<52-pql-working-linearization>
At iteration #mi(`t`);, given current estimates #mi(`\hat\mu^{(t)}`);, construct:

#mitex(`z_i = \underbrace{(\log\hat\mu_i - \text{offset}) +
  \frac{y_i - \hat\mu_i}{\hat\mu_i}}_{\text{NB working response}}
\qquad
W_i = \frac{\hat\mu_i}{1 + \hat\phi\, \hat\mu_i}`)

#mi(`z`); is a first-order Taylor expansion of the NB log-likelihood around #mi(`\hat\mu`);.
#mi(`W`); is the NB variance weight --- it encodes how much information each observation
carries at its current mean. Both change at every iteration because #mi(`\hat\mu`);
changes.

Now treat #mi(`z`); and #mi(`W`); as fixed and solve the weighted Gaussian GLMM:

#mitex(`z = X\beta + Zu + \varepsilon, \qquad
  \varepsilon \sim \mathcal{N}(0,\, W^{-1}), \qquad
  u \sim \mathcal{N}(0,\, \tau K_m)`)

In `fit_null_nb_glmm_loco`, #mi(`\tau`); is re-estimated at every PQL step by REML on this working
model (bounded Brent search over #mi(`\log\tau`); within #mi(`\pm 6`); of the previous value); given
#mi(`\tau`);, #mi(`\hat\beta`); (GLS) and #mi(`\hat{u}`); (BLUP) are closed-form. The updated #mi(`\hat\beta`);,
#mi(`\hat{u}`); give new #mi(`\hat\mu = \exp(X\hat\beta + Z\hat{u} + \text{offset})`);, which recomputes #mi(`z`);
and #mi(`W`);. Iteration starts from #mi(`\hat\mu = \max(S, 0.5)`);, #mi(`\tau = 0.1`); and stops after at most
5 steps or, from the second step on, when the maximum relative change in #mi(`\hat\mu`); falls below
#mi(`10^{-4}`);.

=== 5.3 Kinship enters through the random effect covariance
<53-kinship-enters-through-the-random-effect-covariance>
At each PQL step the marginal covariance of the working response is:

#mitex(`V = W^{-1} + \tau\, Z K_m^{(-c)} Z^\top \qquad (n \times n)`)

where #mi(`K_m^{(-c)}`); is the #mi(`m \times m`); LOCO kinship for chromosome #mi(`c`); at strain
level, and #mi(`Z`); is the #mi(`n \times m`); sample-to-strain mapping matrix.

Kinship does not change the #emph[reason] PQL is needed --- that is a property of
the NB likelihood alone. Kinship determines the #emph[structure] of #mi(`V`); that
must be inverted at each PQL step.

=== 5.4 Woodbury identity at each PQL step
<54-woodbury-identity-at-each-pql-step>
Naively inverting the #mi(`n \times n`); matrix #mi(`V`); at every iteration costs
#mi(`O(n^3)`);. With #mi(`n = 418`); samples, many loci, and many genes this is
prohibitive.

The Woodbury identity rewrites the inverse in terms of the #mi(`m \times m`);
strain-level system (#mi(`m = 48`);):

#mitex(`V^{-1} = W - WZ\left(\tau^{-1}K_m^{-1} + Z^\top WZ\right)^{-1}Z^\top W`)

The inner matrix is #mi(`m \times m`);, so its inversion costs #mi(`O(m^3)`);.
Because #mi(`m \ll n`);, this is the dominant saving per iteration.

#strong[Dependency chain:]

+ NB likelihood non-conjugate with Gaussian prior on #mi(`u`); #mi(`\Rightarrow`); cannot integrate
  out #mi(`u`); in closed form.
+ #mi(`\Rightarrow`); PQL: iterate, constructing a Gaussian working model at each step.
+ Each step requires #mi(`V^{-1}`); where #mi(`V = W^{-1} + \tau Z K_m Z^\top`);; #mi(`W`); changes every
  step (depends on the current #mi(`\hat\mu`);).
+ #mi(`\Rightarrow`); cannot pre-eigendecompose #mi(`V`); once (as the LMM does).
+ #mi(`\Rightarrow`); Woodbury: reduce the #mi(`O(n^3)`); inversion to #mi(`O(m^3)`); per iteration.

PQL is a mathematical necessity (NB non-conjugate). Woodbury is an efficiency
tool that makes the kinship inversion inside each PQL step tractable.
Both are needed; they solve orthogonal problems.

#divider()

== 6. Score test
<6-score-test>
The NB score test at locus #mi(`\ell`); avoids re-fitting the model under #mi(`H_1`); at
every locus --- only the null #mi(`\hat\mu_i`); is needed:

#strong[With LOCO kinship] (`_score_stats_loco`), using the final PQL working response #mi(`z`); and
#mi(`P = V^{-1} - V^{-1}X(X^\top V^{-1}X)^{-1}X^\top V^{-1}`);:

#mitex(`U_\ell = H_\ell^\top P z,  \qquad
  \mathcal{I}_\ell = H_\ell^\top V^{-1} H_\ell - H_\ell^\top V^{-1} X (X^\top V^{-1} X)^{-1} X^\top V^{-1} H_\ell`)

#strong[Without kinship] (`_score_stats_chromosome`), from an NB GLM null fitted by IRLS, with
#mi(`c = \max(1 + \sigma^2_g, 1)`); (`sigma_sample` used only here):

#mitex(`U_\ell = H_\ell^\top r,\quad r_i = \frac{S_i - \hat\mu_i}{(1 + \phi\hat\mu_i)\,c}, \qquad
  \mathcal{I}_\ell = H_\ell^\top \operatorname{diag}(w) H_\ell,\quad w_i = \frac{\hat\mu_i}{(1 + \phi\hat\mu_i)\,c}`)

Known issue: the no-kinship information omits the covariate projection
#mi(`-H_\ell^\top W X (X^\top W X)^{-1} X^\top W H_\ell`); (ledger NB16).

In both paths #mi(`H_\ell \in \mathbb{R}^{n \times (J-1)}`); (last founder dropped) and

#mitex(`Q_\ell = U_\ell^\top \mathcal{I}_\ell^{-1} U_\ell \sim \chi^2(J-1)`)

#mitex(`\text{LOD}_\ell = \frac{Q_\ell}{2 \ln 10}, \qquad p_\ell = 1 - F_{\chi^2_{J-1}}(Q_\ell)`)

The #mi(`\chi^2(J-1)`); approximation holds under standard regularity conditions.
With kinship, the null #mi(`\hat\mu`); is fitted once per gene per chromosome (LOCO); without
kinship, once per gene (`scan_nb_gene`) or once per gene per chromosome (chromosome-parallel path,
`n_cores > 1`). The score is then vectorized over all #mi(`L`); loci via einsum ---
#mi(`O(L\,n\,J(J+m))`); per chromosome with the Woodbury path, #mi(`O(L\,n\,J^2)`); without kinship.
The returned `ScanResult` carries zero `betas` #mi(`(L, J)`); and an empty `h2_null`; the score test
does not estimate founder effects.

#strong[Key property:] the score test only requires the null fit. No IRLS re-convergence
per locus --- this is the analogue of the FWL trick in `lmm.py` that gives
#mi(`O(1)`); per-locus cost beyond the null.

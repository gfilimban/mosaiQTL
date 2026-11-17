#import "math/_setup.typ": *
#show: mathdoc

= mosaiQTL --- Mathematical Methods Reference
<mosaiqtl--mathematical-methods-reference>
#strong[Scope:] Formal definitions of every statistical concept used in mosaiQTL,
ordered from inputs to outputs. Intended as the authoritative mathematical
reference for the package --- each concept links to its implementing module.

#divider()

== 1. Haplotype probability inference --- forward-backward HMM
<1-haplotype-probability-inference--forward-backward-hmm>
#strong[Module:] `genotype/hmm.py → calc_genoprob_cc()`, `forward_backward()` \ #strong[What it is:] Given observed genotype data at a set of markers along a
chromosome, estimate the posterior probability that each genomic position was
inherited from each of the #mi(`J`); founder strains.

#strong[Model.] At locus #mi(`\ell`);, individual #mi(`i`); inherits a haplotype from founder
#mi(`f`); with probability #mi(`H_{if\ell} \in [0,1]`);, #mi(`\sum_f H_{if\ell} = 1`);.
The hidden Markov model treats the unordered diplotype #mi(`s = \{f, f'\}`); as a latent
Markov chain along the chromosome (#mi(`J(J+1)/2 = 36`); states for #mi(`J=8`);). Emission
probabilities encode how likely the observed A/H/B call is given the diplotype:
#mi(`1-\varepsilon`); for the expected call and #mi(`\varepsilon/2`); for each other call
(#mi(`\varepsilon = 0.002`);; uniform #mi(`1/3`); when a founder call is missing). Transition
probabilities encode recombination rates: the Haldane map function
#mi(`r = \tfrac{1}{2}(1 - e^{-2d})`); with #mi(`d = \Delta_\text{cM}/100`); the inter-marker distance in Morgans
gives a haplotype kernel with #mi(`P(\text{stay}) = 1 - r(J-1)/J`); and
#mi(`P(f \to g) = r/J`);, combined over the two haplotypes into a diplotype kernel. The
default initial distribution (`cc_inbred_init`) puts mass #mi(`1/J`); on each homozygous state.
Known issue: `transition_matrix()` halves every heterozygous-target entry before row renormalisation (ledger HMM-01).

#strong[Forward-backward algorithm.] For a chain with #mi(`S`); diplotype states and #mi(`L`); markers
(#mi(`e_\ell(s)`); the emission of the observed call; #mi(`\alpha`);, #mi(`\beta`); renormalised to sum 1 at each marker):

#mitex(`\alpha_{i\ell}(s) \propto e_\ell(s) \sum_{s'} \alpha_{i,\ell-1}(s') \, T_{s's,\ell-1\to\ell}`)

#mitex(`\beta_{i\ell}(s) \propto \sum_{s'} T_{ss',\ell\to\ell+1} \, e_{\ell+1}(s') \, \beta_{i,\ell+1}(s')`)

#mitex(`\gamma_{i\ell}(s) = \frac{\alpha_{i\ell}(s) \, \beta_{i\ell}(s)}{\sum_{s'} \alpha_{i\ell}(s') \, \beta_{i\ell}(s')}`)

#strong[miQTL difference.] R\'s HAPPY HMM estimates joint diplotype probabilities
#mi(`P(f \times f' \mid i, \ell) \in \mathbb{R}^{J(J+1)/2}`); (36 states for #mi(`J=8`);).
mosaiQTL runs the same 36-state diplotype HMM and then collapses the posterior to
additive founder probabilities (`diplotype_to_allele_probs()`):
#mi(`H_{if\ell} = \gamma_{i\ell}(\{f,f\}) + \tfrac{1}{2}\sum_{f' \neq f} \gamma_{i\ell}(\{f,f'\})`);. Little is lost
under the inbred-founder assumption (CC strains are nearly fully inbred, so
heterozygous diplotypes #mi(`f \neq f'`); have near-zero probability). The genome cache
stores this additive representation, and every scan consumes it --- LMM, NB and the
SNP scan (which collapses it to ALT-allele dosage via `collapse_to_dosage()`).

#divider()

== 2. Kinship matrix --- shared founder probability
<2-kinship-matrix--shared-founder-probability>
#strong[Module:] `genotype/kinship.py → calc_kinship()`, `decomp_kinship()` \ #strong[What it is:] A genomic relationship matrix #mi(`K \in \mathbb{R}^{m \times m}`);
quantifying the fraction of the genome shared identical-by-descent between each
pair of strains. Used as the covariance structure for the polygenic random
effect in the LMM.

#strong[Formula] (uncentred founder-probability sharing, not mean-centred van Raden):

#mitex(`K' = \frac{1}{L} \sum_{\ell=1}^{L} H_\ell H_\ell^\top, \qquad K = \frac{K'}{\operatorname{mean}_i K'_{ii}}`)

where #mi(`H_\ell \in \mathbb{R}^{m \times J}`); is the founder-probability matrix at locus
#mi(`\ell`); with rows summing to 1, and #mi(`L`); is the number of loci in the genome cache.

Before scaling, a fully homozygous strain has #mi(`K'_{ii} = 1`); and heterozygous segments
lower it (#mi(`K'_{ii} = 0.5`); for an F1); off-diagonals reflect shared haplotype
ancestry. The final #mi(`K`); is rescaled so that its mean diagonal entry is exactly 1, so
the more inbred strains have #mi(`K_{ii}`); slightly above 1.

#strong[LOCO kinship.] To avoid confounding the genetic signal at a locus with the
polygenic background from that same chromosome, the kinship matrix is recomputed
leaving out chromosome #mi(`c`); (then rescaled to mean diagonal 1 as above):

#mitex(`K'^{(-c)} = \frac{1}{L - L_c} \sum_{\ell \notin c} H_\ell H_\ell^\top`)

where #mi(`L_c`); is the number of loci on chromosome #mi(`c`);. One LOCO matrix per chromosome
is precomputed; `decomp_kinship()` eigendecomposes each (eigenvalues sorted descending,
negative eigenvalues floored at 0) and `save_kinship_decomp()` caches the result to HDF5. This is the standard approach in animal genetics
(Yang et al. 2014) and is implemented identically in miQTL and mosaiQTL.

#divider()

== 3. #mi(`h^2`); estimation --- ML with eigendecomposition
#label("3-h²-estimation--ml-with-eigendecomposition")
#strong[Module:] `qtl/lmm.py → estimate_h2_null()` \ #strong[What it is:] Under the null model (no QTL effect), heritability #mi(`h^2`);
controls the balance between polygenic and residual variance. Estimating it
once per gene under the null for each scanned chromosome (each has its own LOCO
kinship) and holding it fixed across all loci on that chromosome (the
\"fix-par\" convention) dramatically reduces computation.

#strong[LMM variance structure:]

#mitex(`\mathbf{y} = \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}`)

#mitex(`\mathbf{u} \sim \mathcal{N}(\mathbf{0},\, \sigma_g^2 \mathbf{K}), \quad
\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0},\, \sigma_e^2 \mathbf{I})`)

#mitex(`\mathbf{V} = \sigma_g^2 \mathbf{K} + \sigma_e^2 \mathbf{I} = \sigma_e^2 (h^2 \mathbf{K} + (1-h^2)\mathbf{I})`)

where #mi(`h^2 = \sigma_g^2 / (\sigma_g^2 + \sigma_e^2)`);.

#strong[Eigendecomposition trick] (Kang et al. 2008). Compute #mi(`\mathbf{K} = \mathbf{V}_K \boldsymbol{\Lambda}_K \mathbf{V}_K^\top`); once. Then:

#mitex(`\mathbf{V}^{-1} = \mathbf{V}_K \operatorname{diag}\!\left(\frac{1}{h^2\lambda_j + (1-h^2)}\right) \mathbf{V}_K^\top`)

Rotating the data: #mi(`\tilde{\mathbf{y}} = \mathbf{V}_K^\top \mathbf{y}`);,
#mi(`\tilde{\mathbf{X}} = \mathbf{V}_K^\top \mathbf{X}`);, and writing #mi(`w_j = h^2\lambda_j + (1-h^2)`);, the
profile maximum-likelihood (ML) log-likelihood becomes a 1D function of #mi(`h^2`); alone:

#mitex(`\ell_\text{ML}(h^2) = -\frac{1}{2}\left[n\left(\log(2\pi\hat{\sigma}^2) + 1\right) + \sum_j \log w_j\right], \quad \hat{\sigma}^2 = \frac{1}{n}\sum_j \frac{(\tilde{y}_j - \tilde{\mathbf{x}}_j^\top\hat{\boldsymbol{\alpha}})^2}{w_j}`)

where #mi(`\hat{\boldsymbol{\alpha}}`); is the weighted (GLS) estimate at the current #mi(`h^2`);. ML rather
than REML is used because the LOD is a likelihood-ratio test between models with
different fixed effects; REML remains available as `method='reml'`. The 1D
optimisation is: a 20-point grid on #mi(`[0.01, 0.99]`);, bounded Brent refinement
(`minimize_scalar(method='bounded')`) between the grid neighbours of the best point,
then a check against #mi(`h^2 \approx 0`); (#mi(`10^{-6}`);), keeping whichever has the higher
log-likelihood. Before commit 11a4b68e (2026-05-31), #mi(`h^2`); was estimated by REML.
The eigendecomposition itself is cached per chromosome: computed once, reused for
all genes on that chromosome (with a Z-matrix, `scan_h2lmm()` re-decomposes the
expanded #mi(`K_n = Z K_m Z^\top`); per chromosome on each call).

#divider()

== 4. LMM scan --- LOD score
<4-lmm-scan--lod-score>
#strong[Module:] `qtl/lmm.py → fit_lmm_at_locus()` \ #strong[What it is:] At each genomic locus #mi(`\ell`);, test whether including the
haplotype design matrix #mi(`H_\ell`); as a fixed effect significantly reduces
residual sum of squares under the fixed-#mi(`h^2`); LMM. #mi(`H_\ell`); carries all #mi(`J`); founder
columns; least squares (`lstsq`) absorbs the rank deficiency (rows sum to 1).

#strong[Model:]

#mitex(`H_0: \mathbf{y} = \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}`)

#mitex(`H_1: \mathbf{y} = \mathbf{H}_\ell \boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}`)

With #mi(`h^2`); fixed from the null, this reduces to weighted least squares in the
rotated space. The LOD score uses the maximum-likelihood denominator #mi(`n`);
(not #mi(`n-p`);, which would be REML):

#mitex(`\text{LOD}_\ell = \frac{n}{2} \log_{10}\!\frac{\text{RSS}_0}{\text{RSS}_1}`)

where #mi(`\text{RSS}_k = \tilde{\mathbf{y}}^\top \mathbf{D}^{-1} M_k \tilde{\mathbf{y}}`);,
#mi(`\mathbf{D} = \operatorname{diag}(h^2\lambda_j + 1-h^2)`);, and #mi(`M_k`); is the
annihilator of the appropriate design matrix. This formula matches miQTL exactly
and is positive by construction since #mi(`H_1`); is a superset of #mi(`H_0`);.

#divider()

== 5. Frisch-Waugh-Lovell (FWL) vectorization
<5-frisch-waugh-lovell-fwl-vectorization>
#strong[Module:] `qtl/lmm.py → precompute_chromosome()`, `scan_chromosome()` \ #strong[What it is:] The key computational speedup in mosaiQTL. Instead of running a
separate WLS regression per locus (miQTL\'s approach, #mi(`O(L \cdot n \cdot J^2)`);
sequential), FWL allows the covariate projection to be computed once and the
haplotype regressions to be vectorized across all #mi(`L`); loci simultaneously.

#strong[Theorem (Frisch-Waugh-Lovell).] In OLS (and WLS), the coefficient on a
set of regressors #mi(`\mathbf{H}`); in the full model
#mi(`\mathbf{y} = \mathbf{H}\boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \boldsymbol{\varepsilon}`);
is identical to the coefficient obtained by first projecting out #mi(`\mathbf{X}`);:

#mitex(`\hat{\boldsymbol{\gamma}}_\text{FWL} = (\tilde{\mathbf{H}}^\top \tilde{\mathbf{H}})^{-1} \tilde{\mathbf{H}}^\top \tilde{\mathbf{y}}`)

where #mi(`\tilde{\mathbf{y}} = M_X \mathbf{y}`); and #mi(`\tilde{\mathbf{H}} = M_X \mathbf{H}`);
are the residuals after projecting out covariates #mi(`\mathbf{X}`);.

#strong[Application in mosaiQTL.] In the rotated (eigendecomposed) space:

+ Project out covariates once: #mi(`\tilde{\mathbf{y}}_\text{rot} = M_{X_\text{rot}} \mathbf{y}_\text{rot}`);. Cost: #mi(`O(n \cdot p^2)`);.
+ For each locus, project haplotypes: #mi(`\tilde{\mathbf{H}}_{\ell,\text{rot}} = M_{X_\text{rot}} \mathbf{H}_{\ell,\text{rot}}`);.
+ Compute RSS via #mi(`\text{RSS}_1 = \tilde{\mathbf{y}}^\top \tilde{\mathbf{y}} - \tilde{\mathbf{y}}^\top \tilde{\mathbf{H}}(\tilde{\mathbf{H}}^\top \tilde{\mathbf{H}})^{+} \tilde{\mathbf{H}}^\top \tilde{\mathbf{y}}`);, with #mi(`(\cdot)^{+}`); the Moore--Penrose pseudo-inverse.

Steps 2 and 3 are batched across all #mi(`L`); loci using `numpy.einsum`. The
phenotype-independent rotation #mi(`\mathbf{V}_K^\top \mathbf{H}_\ell`); (#mi(`O(L \cdot n \cdot m \cdot J)`);) is paid once
per chromosome by `precompute_chromosome()`; per gene only the
#mi(`O(L \cdot n \cdot J)`); weighting, the covariate projection and the #mi(`O(L \cdot n \cdot J^2)`); Gram
products remain. All #mi(`J`); founder columns are kept, and the per-locus #mi(`J \times J`);
solve is a batched pseudo-inverse (`np.linalg.pinv`, `rcond=1e-10`) over all loci.
#strong[Result: #mi(`\sim 100\times`); speedup over miQTL\'s sequential per-locus WLS loop] for
typical chromosome sizes (#mi(`L \sim 4000`); loci).

This is mathematically identical to per-locus WLS --- no approximation is made.
The S4 validation (median LOD correlation #mi(`r = 0.973`); against miQTL) confirms
numerical equivalence.

#divider()

== 6. RINT --- rank-based inverse normal transformation
<6-rint--rank-based-inverse-normal-transformation>
#strong[Module:] `utils.py → rint()` \ #strong[What it is:] Applied to phenotypes before LMM fitting to remove distributional
assumptions. Converts any continuous phenotype to approximate normality while
preserving rank order.

#strong[Blom\'s formula:]

#mitex(`y_i^\text{rint} = \Phi^{-1}\!\left(\frac{r_i - 3/8}{n + 1/4}\right)`)

where #mi(`r_i = \text{rank}(y_i)`); (average ranks for ties) and #mi(`\Phi^{-1}`); is the
standard normal quantile function. The constants #mi(`-3/8`); and #mi(`+1/4`); minimise
bias in the expected order statistics. Not applied in the NB GLM (which models
counts directly) or in dispersion estimation.

#divider()

== 7. NB GLM Stage 1 --- null model (native NB-gamma; NEBULA-LN legacy)
<7-nb-glm-stage-1--nebula-ln-cell-level-null-model>
#strong[Module:] `qtl/nb.py → fit_null_sc()` (default `use_native=True` → `fit_null_sc_native()`) \ #strong[What it is:] A null (no-genotype) negative binomial model per gene that estimates a
cell-level dispersion and a subject-level overdispersion, separating within-strain
cell-to-cell variability from between-strain variability. Run once per cell type before
the genome-wide scan. #strong[Status:] the NB path (Sections 7, 8, 17) is experimental; it is
reachable from the Python API (`scan_nb()`, `pipeline.discover(model='nb')`) but is not
wired into the Nextflow workflow or the CLI.

#strong[Model] (native path). For gene #mi(`g`);, the subject (strain) pseudobulk sum #mi(`S_i = \sum_{j \in i} Y_{ij}`); is fit as

#mitex(`S_i \sim \text{NB}(\mu_i,\, \sigma^2_g), \quad
\log \mu_i = \mathbf{x}_i^\top \boldsymbol{\beta}_g + \log L_i + \log f_i`)

#mi(`Y_{ij}`);: raw UMI count for gene #mi(`g`); in cell #mi(`j`); of strain #mi(`i`);. \ #mi(`L_i`);: subject library size (sum of per-cell totals, each #mi(`+1`);); #mi(`f_i`);: TMM normalisation factor computed on the subject pseudobulk matrix. \ #mi(`\mathbf{x}_i`);: intercept plus cell-level covariates averaged within subject. \ #mi(`\sigma^2_g = 1/k_g`);: subject-level (NB-gamma) overdispersion, stored as `sigma_sample`; #mi(`\boldsymbol{\beta}_g`); and #mi(`k_g`); are fit by 8 alternations of IRLS (`_irls_nb_pseudobulk`) and Newton--Raphson (`_nr_k`). `sigma_sample` is propagated to Stage 2 as a variance-inflation term. \ #mi(`\phi_g`);: cell-level NB dispersion, a method-of-moments estimate over cells with #mi(`\hat{\mu}_{ij} = S_i / n_i`); (#mi(`n_i`); cells in strain #mi(`i`);, #mi(`N`); cells in total):

#mitex(`\hat{\phi}_g = \frac{1}{N}\sum_{j} \frac{(Y_{ij} - \hat{\mu}_{ij})^2 - \hat{\mu}_{ij}}{\hat{\mu}_{ij}^2}`)

Genes pass the filter when their mean count per cell exceeds 0.005 and they are detected in at least 5 cells.

#strong[NB parameterisation:] #mi(`\text{Var}(Y_{ij}) = \mu_{ij} + \phi_g \mu_{ij}^2`);.
The legacy path (`use_native=False`) instead fits the cell-level NEBULA-LN model
(He et al. 2021) with the optional `edgepython` package (`glm_sc_fit`, then
`shrink_sc_disp`). #mi(`\phi_g`); is held fixed for Stage 2.

#strong[Empirical Bayes dispersion shrinkage.] Per-gene dispersion estimates are
noisy for lowly expressed genes. After fitting, `_wleb_shrink()` shrinks toward a
trended prior #mi(`\tilde{\phi}(\bar{\mu}_g)`); --- the median #mi(`\hat{\phi}`); within up to 10
quantile bins of log mean pseudobulk count. The weight is fixed (#mi(`d_0 = 10`);, `df_prior`), so
every gene is shrunk by the same proportion regardless of its cell count:

#mitex(`\hat{\phi}_g^\text{shrunk} = \frac{d_0\,\tilde{\phi}(\bar{\mu}_g) + \hat{\phi}_g}{d_0 + 1}`)

This is analogous to edgeR\'s tagwise dispersion estimation (Robinson &
Smyth 2007) but with a fixed prior weight. The shrunk dispersion #mi(`\hat{\phi}_g^\text{shrunk}`);
(`dispersion_use`) is what gets passed to Stage 2.
Known issue: `_wleb_shrink()` raises a shape-mismatch error whenever any gene fails its finite/positive filter (ledger NB05).

#divider()

#divider()

== 8. NB GLM Stage 2 --- pseudobulk NB-GLMM with LOCO kinship
<8-nb-glm-stage-2--pseudobulk-nb-glmm-with-loco-kinship>
#strong[Module:] `qtl/nb.py → fit_null_nb_glmm_loco(), _score_stats_loco()` \ #strong[Literature:] Breslow & Clayton 1993 (PQL); Chen et al. 2016 AJHG (GMMAT score test) \ #strong[What it is:] For each gene, for each chromosome, fit a null negative binomial
GLMM at pseudobulk level with LOCO kinship as the structured random effect covariance.
Extract kinship-adjusted working residuals and use them in the score test at all loci
on that chromosome. This is the recommended path when `kinship_loco` is provided
to `scan_nb()` or `scan_nb_gene()`.
Known issue: `pipeline.discover(model='nb')` never passes `kinship_loco`, so this path is reachable only by calling `scan_nb()`/`scan_nb_gene()` directly (ledger NB15).

=== 8a. Model
<8a-model>
#mitex(`S_i \sim \text{NB}(\mu_i, \phi_g), \quad \log \mu_i = \mathbf{x}_i^\top \boldsymbol{\beta} + u_i + \text{offset}_i`)

#mitex(`\mathbf{u} \sim \mathcal{N}(\mathbf{0},\, \tau Z K^{(-c)} Z^\top)`)

#mi(`S_i = \sum_j Y_{ij}`);: pseudobulk sum for gene #mi(`g`); in sample #mi(`i`);. \ #mi(`\text{offset}_i = \log s_i`);: log of the caller-supplied library size `lib_sizes` (zero offset when it is `None`). \ #mi(`\phi_g`);: dispersion held fixed from Stage 1 EB shrinkage. \ #mi(`\tau`);: kinship variance component estimated per gene per chromosome via REML. \ #mi(`K^{(-c)}`);: strain-level LOCO kinship for chromosome #mi(`c`); (precomputed by `genotype/kinship.py`); #mi(`Z`);: sample-to-strain matrix (Section 14).
Known issue: `scan_nb()` passes `lib_sizes` through unchanged (default `None`), so Stage 2 runs with zero offset unless the caller supplies library sizes (ledger NB04).

=== 8b. PQL fitting (Breslow & Clayton 1993)
<8b-pql-fitting-breslow--clayton-1993>
PQL iteratively linearises the NB-GLMM around the current iterate, producing a
#strong[working linear mixed model] at each step:

#mitex(`\mathbf{z}^{(t)} \sim \mathcal{N}\!\left(\mathbf{X}\boldsymbol{\beta} + \mathbf{u},\; \mathbf{D}^{(t)}\right), \quad \mathbf{D}^{(t)} = \operatorname{diag}\!\left(\frac{1}{W_i^{(t)}}\right)`)

#strong[Working weights] (NB link):

#mitex(`W_i^{(t)} = \frac{\hat{\mu}_i^{(t)}}{1 + \phi_g \hat{\mu}_i^{(t)}}`)

#strong[Working response] (log link, offset removed):

#mitex(`z_i^{(t)} = \log \hat{\mu}_i^{(t)} - \text{offset}_i + \frac{S_i - \hat{\mu}_i^{(t)}}{\hat{\mu}_i^{(t)}}`)

At each iteration:

+ #strong[REML for #mi(`\tau`);] --- 1D minimisation of the restricted log-likelihood of the working LMM
  #mi(`\ell_\text{REML}(\tau) = -\frac{1}{2}[\log|V| + \log|\mathbf{X}^\top V^{-1}\mathbf{X}| + \mathbf{z}^\top P_V \mathbf{z}]`);
  (minimised over #mi(`\log\tau`); within #mi(`\pm 6`); of the previous iterate) where #mi(`V = D + \tau Z K^{(-c)} Z^\top`); and #mi(`P_V = V^{-1} - V^{-1}\mathbf{X}(\mathbf{X}^\top V^{-1}\mathbf{X})^{-1}\mathbf{X}^\top V^{-1}`);.
  At #mi(`n = 418`); samples with #mi(`m = 48`); unique strains, #mi(`V \in \mathbb{R}^{n\times n}`); is too large to invert
  directly per gene per chromosome. The Woodbury identity reduces all operations to #mi(`m\times m`); cost.

+ #strong[GLS update] --- #mi(`\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top V^{-1}\mathbf{X})^{-1}\mathbf{X}^\top V^{-1}\mathbf{z}`);

+ #strong[BLUP update] --- #mi(`\hat{\mathbf{u}} = \tau Z K^{(-c)} Z^\top V^{-1}(\mathbf{z} - \mathbf{X}\hat{\boldsymbol{\beta}})`);

+ #strong[Mean update] --- #mi(`\hat{\mu}_i = \exp(\mathbf{x}_i^\top \hat{\boldsymbol{\beta}} + \hat{u}_i + \text{offset}_i)`);

Iterate until #mi(`\max_i |\hat{\mu}_i^{(t+1)} - \hat{\mu}_i^{(t)}| / |\hat{\mu}_i^{(t)}| < 10^{-4}`); (`tol`; checked from the second iteration) or `max_iter = 5` iterations,
starting from #mi(`\hat{\mu}_i^{(0)} = \max(S_i, 0.5)`); and #mi(`\tau^{(0)} = 0.1`);.

#strong[PQL accuracy.] PQL is an approximation --- it is biased for binary outcomes with
small group sizes (Breslow & Lin 1995). For pseudobulk counts #mi(`S_i \gg 1`); (which
holds at our cell numbers per strain), the NB working response variance #mi(`1/W_i`);
is small relative to the signal, and PQL approaches exact GLMM likelihood. GMMAT
(Chen et al. 2016) uses the same PQL approach for GWAS with large cohorts.

=== 8c. Kinship-corrected score test
<8c-kinship-corrected-score-test>
After PQL convergence, the projection matrix and kinship-adjusted residuals are:

#mi(`P`); is never stored (#mi(`n\times n`); at #mi(`n=418`); is 1.4 MB per gene --- infeasible at scale).
Computed implicitly via Woodbury:

#mitex(`r_\text{adj} = V^{-1}z - V^{-1}X(X^\top V^{-1}X)^{-1}X^\top V^{-1}z \quad (n \text{ vector, cheap})`)

#mi(`P`); satisfies #mi(`P\mathbf{X} = 0`); (oblique projection in #mi(`V`);-metric; #mi(`P^2 = P`); only when #mi(`V=I`);).

Score test at locus #mi(`\ell`); across all #mi(`L`); loci (vectorised):

#mitex(`\mathbf{U}_\ell = \mathbf{H}_\ell^\top \mathbf{r}_{\text{adj}} \quad (J-1 \text{ vector})`)

#mitex(`\mathbf{I}_\ell = \mathbf{H}_\ell^\top P \mathbf{H}_\ell \quad (J-1) \times (J-1)`)

#mitex(`T_\ell = \mathbf{U}_\ell^\top \mathbf{I}_\ell^{-1} \mathbf{U}_\ell \sim \chi^2_{J-1} \text{ under } H_0`)

#strong[Woodbury vectorisation] (no P stored):
#mi(`V^{-1}H_\text{all}`); computed via `_wb_batch` using three einsums at #mi(`O(LnmJ)`); cost.
Then #mi(`I_\ell = H_\ell^\top V^{-1}H_\ell - (H_\ell^\top V^{-1}X)(X^\top V^{-1}X)^{-1}(X^\top V^{-1}H_\ell)`);
via batched einsums. At #mi(`L=4000`);, #mi(`n=418`);, #mi(`m=48`);, #mi(`J=8`);: #mi(`\approx 7 \times 10^{8}`); ops per chromosome,
#mi(`\sim 0.7`); s --- acceptable for a genome-wide scan.

#strong[LOD conversion:] #mi(`\text{LOD}_\ell = T_\ell / (2\ln 10)`);, identical to the LMM convention.

=== 8d. Comparison to original score test (without kinship)
<8d-comparison-to-original-score-test-without-kinship>
#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Property], [Without kinship], [With LOCO kinship],),
    table.hline(),
    [Random effect], [none (NB GLM null); `sigma_sample` enters only as variance inflation #mi(`1+\sigma^2_\text{sample}`);], [structured #mi(`\mathbf{u} \sim \mathcal{N}(0, \tau Z K^{(-c)} Z^\top)`);],
    [Score denominator], [#mi(`\mathbf{H}^\top W \mathbf{H}`); (diagonal weights)], [#mi(`\mathbf{H}^\top P \mathbf{H}`); (kinship-corrected information)],
    [Null fit], [once genome-wide], [once per chromosome per gene],
    [Population structure], [partially via #mi(`\sigma^2_\text{sample}`); inflation], [fully via #mi(`K^{(-c)}`); covariance (`sigma_sample` unused)],
    [API], [`scan_nb(..., kinship_loco=None)`], [`scan_nb(..., kinship_loco=kinship_loco)`],
    [P stored], [no #mi(`P`); (diagonal weights only)], [never (implicit Woodbury)],
  )]
  , kind: table
  )

== 17. NB GLM Stage 2 --- pseudobulk score test (without kinship)
<17-nb-glm-stage-2--pseudobulk-score-test-without-kinship>
#strong[Module:] `qtl/nb.py → scan_nb_gene()`, `_score_stats_chromosome()` \ #strong[What it is:] For each gene, aggregate cell-level counts to per-strain
pseudobulk sums, then test for association between haplotype and pseudobulk
count at each genomic locus using a score test. Vectorized across all loci
simultaneously.

#strong[Pseudobulk aggregation:]

#mitex(`S_{ig} = \sum_{j \in \text{strain } i} Y_{ij}`)

#mi(`S_{ig}`); is the total UMI count for gene #mi(`g`); in strain #mi(`i`);. Under the null
(no haplotype effect), #mi(`\log \mathbb{E}[S_{ig}] = \mathbf{x}_i^\top\boldsymbol{\beta}_0 + \log s_i`);
where #mi(`s_i`); is the caller-supplied library size (`lib_sizes`; zero offset when `None`),
fit once genome-wide by NB IRLS with #mi(`\hat{\phi}_g`); fixed.

#strong[Score test.] The score test evaluates the gradient of the log-likelihood at
the null parameter estimate, avoiding the need to fit the alternative model at
every locus. For the NB model, the score statistic at locus #mi(`\ell`); is:

#mitex(`Q_\ell = \mathbf{U}_\ell^\top \mathbf{I}_\ell^{-1} \mathbf{U}_\ell \sim \chi^2(J-1) \text{ under } H_0`)

where #mi(`\mathbf{U}_\ell = \mathbf{H}_\ell^\top \mathbf{r}`); is the score vector
(weighted residuals projected onto the haplotype design matrix),
#mi(`r_i = (S_i - \hat{\mu}_{i0}) / [(1 + \hat{\phi}_g\hat{\mu}_{i0})(1 + \sigma_\text{sample}^2)]`); are the
null working-score residuals, #mi(`\mathbf{H}_\ell`); holds the first #mi(`J-1`); founder columns,
and #mi(`\mathbf{I}_\ell = \mathbf{H}_\ell^\top \mathbf{W} \mathbf{H}_\ell`); is the
Fisher information with weight matrix #mi(`\mathbf{W} = \operatorname{diag}\!\left(\hat{\mu}_{i0} / [(1 + \hat{\phi}_g\hat{\mu}_{i0})(1 + \sigma_\text{sample}^2)]\right)`);.

The #mi(`1 + \sigma_\text{sample}^2`); factor from Stage 1 scales #mi(`Q_\ell`); by #mi(`1/(1 + \sigma_\text{sample}^2)`); to
account for residual subject-level variance not captured by fixed covariates ---
making the test conservative in the right direction.
Known issue: the information omits the covariate projection (it uses #mi(`\mathbf{H}^\top W \mathbf{H}`); rather than the Schur complement #mi(`\mathbf{H}^\top W \mathbf{H} - \mathbf{H}^\top W \mathbf{X}(\mathbf{X}^\top W \mathbf{X})^{-1}\mathbf{X}^\top W \mathbf{H}`);), which deflates #mi(`Q_\ell`); (ledger NB16).

#strong[Vectorization.] Computing #mi(`Q_\ell`); for all #mi(`L`); loci simultaneously:

#mitex(`\mathbf{U} = \operatorname{einsum}(H_\text{all},\, \mathbf{r}) \in \mathbb{R}^{L \times (J-1)}`)

#mitex(`\mathbf{I} = \operatorname{einsum}(H_\text{all},\, \mathbf{W},\, H_\text{all}) \in \mathbb{R}^{L \times (J-1) \times (J-1)}`)

where #mi(`H_\text{all} \in \mathbb{R}^{L \times n \times (J-1)}`); is the full
haplotype tensor. The per-locus solve #mi(`\mathbf{I}_\ell^{-1}`); is the dominant
cost: #mi(`O(L \cdot (J-1)^3)`);. For #mi(`J=8`);, #mi(`(J-1)^3=343`); --- negligible per locus.

The score statistic #mi(`Q_\ell`); is converted to a LOD score:
#mi(`\text{LOD}_\ell = Q_\ell / (2\ln 10)`); (chi-squared to LOD equivalence).

#divider()

== 9. vQTL --- dispersion QTL scan
<9-vqtl--dispersion-qtl-scan>
#strong[Module:] `qtl/vqtl.py → estimate_dispersion_per_subject()`, `scan_vqtl()` \ #strong[What it is:] Tests whether haplotype affects not mean expression but
expression variance across cells --- a biologically distinct signal that is
only detectable at single-cell resolution. A mean eQTL affects how much a
gene is expressed on average; a vQTL affects how consistently it is expressed
cell-to-cell within a strain.

#strong[Two-stage design.]

#emph[Stage 1 --- per-strain dispersion estimation.]
For each strain #mi(`i`);, the NB overdispersion #mi(`\hat{\phi}_i`); is estimated from
its own cells using method-of-moments:

#mitex(`\hat{\phi}_i = \frac{\hat{\sigma}^2_i - \hat{\mu}_i}{\hat{\mu}_i^2}`)

where #mi(`\hat{\mu}_i`); and #mi(`\hat{\sigma}^2_i`); are the sample mean and (divisor-#mi(`n`);) variance of
raw counts across cells in strain #mi(`i`); (`library_sizes` is accepted but not used). Strains with
fewer than `min_cells` (default 20) cells or with #mi(`\hat{\mu}_i \le 0.01`); are excluded, and
#mi(`\hat{\phi}_i`); is floored at #mi(`10^{-6}`);. `phi_method='mle'` fits a per-strain NB likelihood instead.

#emph[Stage 2 --- LMM scan on log dispersion.]
#mi(`\log\hat{\phi}_i`); is treated as a strain-level phenotype (RINT-transformed by default,
`use_rint=True`) and passed through the standard LMM scan (`scan_h2lmm`, genome-wide) with LOCO kinship:

#mitex(`\log\hat{\phi}_i = \mathbf{H}_{i\ell}\boldsymbol{\gamma} + \mathbf{x}_i^\top\boldsymbol{\alpha} + u_i + \varepsilon_i`)

A significant vQTL LOD means the founder haplotype at locus #mi(`\ell`); explains
variance in expression dispersion across strains. Co-localisation of a vQTL
with a mean eQTL implies the haplotype affects both; an independent vQTL
implies the haplotype specifically modulates expression noise.
Known issue: `scan_vqtl()` calls `scan_h2lmm()` without `sample_ids`, so the phenotype is aligned to the full genome-cache sample list rather than the shared subjects (ledger VQ01).

#divider()

== 10. Mediation --- QR LOD-drop
<10-mediation--qr-lod-drop>
#strong[Module:] `qtl/mediation.py → mediation_scan_qr()` \ #strong[What it is:] Quantifies how much of a QTL\'s effect on an outcome gene
flows through a mediator gene\'s expression. If the mediator fully explains
the QTL, the LOD drops to near zero when the mediator is included as a
covariate.

#strong[Model:]

#mitex(`\text{base scan: }  y_\text{outcome} = \mathbf{H}_\ell\boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + u + \varepsilon`)

#mitex(`\text{mediated scan: } y_\text{outcome} = \mathbf{H}_\ell\boldsymbol{\gamma}' + y_\text{mediator}\delta + \mathbf{X}\boldsymbol{\alpha}' + u + \varepsilon`)

#mitex(`\text{LOD\_drop}_\ell = \max\!\left(\text{LOD\_base}_\ell - \text{LOD\_mediated}_\ell,\; 0\right)`)

A large positive LOD\_drop indicates mediation: including the mediator
absorbs the haplotype effect. The drop is clamped at 0, so amplification
(a mediated LOD above the base LOD) is reported as 0 rather than as a negative drop.
The output also carries `prop_mediated` #mi(`= \text{LOD\_drop}/\text{LOD\_base}`); and a
#mi(`\chi^2_1`); p-value of #mi(`2\ln(10)\,\text{LOD\_drop}`);.

#strong[Rank-1 update implementation.] Despite the function name, no QR decomposition
is used. Rather than solving two separate LMM scans, #mi(`h^2`); is fixed from the base null
model #mi(`y \sim X`); for every mediator on a chromosome, the FWL quantities of the base scan
(#mi(`\tilde{\mathbf{H}}_\ell`);, #mi(`(\tilde{\mathbf{H}}_\ell^\top\tilde{\mathbf{H}}_\ell)^{+}`);) are computed once per chromosome,
and each mediator\'s LOD is obtained by a Woodbury (Sherman--Morrison) rank-1 update.

#divider()

== 11. Epistasis --- Khatri-Rao product
<11-epistasis--khatri-rao-product>
#strong[Module:] `qtl/epistasis.py → scan_epistasis()` \ #strong[What it is:] Tests whether the combination of founder haplotypes at two
distinct loci produces an expression outcome not predictable from their additive
sum --- polygenic epistasis at haplotype resolution.

#strong[Models:]

#mitex(`\text{additive: } y = \mathbf{H}^a\boldsymbol{\alpha} + \mathbf{H}^b\boldsymbol{\beta} + \mathbf{X}\boldsymbol{\xi} + u + \varepsilon`)

#mitex(`\text{interaction: } y = \mathbf{H}^a\boldsymbol{\alpha} + \mathbf{H}^b\boldsymbol{\beta} + (\mathbf{H}^a \odot \mathbf{H}^b)\boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\xi} + u + \varepsilon`)

#strong[Khatri-Rao product.] #mi(`\mathbf{H}^a \odot \mathbf{H}^b \in \mathbb{R}^{n \times (J-1)^2}`);
is the row-wise Kronecker (face-splitting) product of the two haplotype design matrices,
row #mi(`i`); being #mi(`\mathbf{h}^a_i \otimes \mathbf{h}^b_i`); (each matrix with the last founder dropped for identifiability). For #mi(`J=8`);: #mi(`(J-1)^2 = 49`);
interaction terms. Each term #mi(`H^a_{if} \cdot H^b_{if'}`); represents the
probability that strain #mi(`i`); carries haplotype #mi(`f`); at locus #mi(`a`); AND haplotype
#mi(`f'`); at locus #mi(`b`); simultaneously.

#strong[Epistasis LOD:]

#mitex(`\text{LOD\_epistasis} = \max\!\left(\text{LOD\_interaction} - \text{LOD\_additive},\; 0\right)`)

Both models are fit by weighted least squares at a fixed #mi(`h^2`); with locus #mi(`a`);\'s LOCO
kinship; the p-value refers #mi(`2\ln(10)\,\text{LOD\_epistasis}`); to #mi(`\chi^2_{(J-1)^2}`);.
A positive LOD\_epistasis means the two-locus combination explains expression
beyond the sum of individual founder effects. The companion function
`test_haplotype_combinations()` produces the #mi(`J \times J`); founder-pair mean
expression heatmap as an empirical visualisation of the interaction structure.
Known issue: the interaction design has #mi(`2(J-1) + (J-1)^2 + p`); columns (#mi(`63 + p`); for #mi(`J=8`);), more than the strains in a 48-strain panel, so the interaction fit is saturated (ledger EPI-03).

#divider()

== 12. Conditional multi-signal decomposition --- BLUP conditioning
<12-conditional-multi-signal-decomposition--blup-conditioning>
#strong[Module:] `qtl/epistasis.py → scan_conditional_with_y()` \ #strong[What it is:] Forward-stepwise detection of multiple independent QTL signals
for a single gene. At each step, the haplotype effect at the current peak
locus is estimated and subtracted, and the genome is rescanned for residual
signal.

#strong[Peak-effect conditioning (called \"BLUP\" in the code).] At each iteration #mi(`k`);:

+ Find peak locus #mi(`\ell^*`); in the current scan result.
+ Re-estimate #mi(`h^2`); on the current residual phenotype with the peak chromosome\'s LOCO kinship, then fit the LMM at #mi(`\ell^*`); (`fit_lmm_at_locus()`, first #mi(`J-1`); founder columns) to obtain the allele effect vector #mi(`\hat{\boldsymbol{\gamma}}_{\ell^*}`);.
+ Compute the fitted haplotype contribution:

#mitex(`\hat{\mathbf{y}}_{\text{hap}} = \mathbf{H}_{\ell^*} \hat{\boldsymbol{\gamma}}_{\ell^*}`)

#block[
#set enum(numbering: "1.", start: 4)
+ Condition out: #mi(`\mathbf{y}^{(k+1)} = \mathbf{y}^{(k)} - \hat{\mathbf{y}}_\text{hap}`);
+ Rescan #mi(`\mathbf{y}^{(k+1)}`); genome-wide (`scan_h2lmm`, no RINT).
]

Iteration stops after `n_signals` (default 3) signals or when the rescan\'s peak LOD
falls below `lod_drop_threshold` (default 1.5), applied as an absolute LOD floor.

#strong[BLUP vs fixed effect.] Despite the name, #mi(`\hat{\boldsymbol{\gamma}}_{\ell^*}`); is the unshrunk
weighted least-squares (GLS) fixed-effect estimate at the fixed #mi(`h^2`);, not a shrunken
BLUP, so the subtracted effect is not regularised toward zero.
Known issue: the pipeline (`_run_conditional_scan`) calls `scan_conditional()` rather than this function, and `scan_conditional()` only re-reads the original LOD array, performing no conditioning (ledger EPI-01).

#divider()

== 13. GEV permutation thresholds
<13-gev-permutation-thresholds>
#strong[Module:] `qtl/permutation.py → run_threshold_scans()`; `qtl/thresholds.py → get_gev_thresholds()` \ #strong[What it is:] Derives genome-wide significance thresholds by permuting the
phenotype (breaking the genotype-phenotype link) and fitting a Generalised
Extreme Value (GEV) distribution to the null distribution of maximum LOD scores.
In production this threshold is logged for QC only; eGene calls use the per-gene
calibrated p-value (`qtl/fdr.py → calibrate_gene_pvalue()`) followed by `storey_qvalue()`
over all tested genes.

#strong[Procedure.]
For each permutation #mi(`b = 1, \ldots, B`);:

+ Draw a null phenotype. Default (`permutation_method='parametric'`): simulate
  #mi(`\mathbf{y}^\ast = \mathbf{X}\hat{\boldsymbol{\alpha}} + \mathbf{u} + \boldsymbol{\varepsilon}`);, #mi(`\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \hat{\sigma}_g^2 K)`);,
  #mi(`\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \hat{\sigma}_e^2 I)`);, and reorder the observed #mi(`\mathbf{y}`); by the ranks of #mi(`\mathbf{y}^\ast`);;
  `'plain'` shuffles #mi(`\mathbf{y}`); across strains.
+ Run the scan on the null phenotype (all chromosomes with one null matrix, or the cis window).
+ Record #mi(`M_b = \max_\ell \text{LOD}_\ell^{(b)}`);.

The empirical distribution #mi(`\{M_b\}`); follows a GEV distribution
(by extreme value theory: the maximum of many correlated chi-squared statistics
converges to GEV). Fit:

#mitex(`F(m; \mu, \sigma, \xi) = \exp\!\left\{-\left[1 + \xi\!\left(\frac{m-\mu}{\sigma}\right)\right]^{-1/\xi}\right\}`)

The fit is `scipy.stats.genextreme.fit` (maximum likelihood; SciPy\'s shape is #mi(`c = -\xi`);),
and the 95th percentile (`percentile=0.95`) of the fitted distribution is the genome-wide threshold;
`get_gev_thresholds()` returns `None` when fewer than 10 finite maxima exist or the fit fails
(the eGene step then falls back to the empirical 95th percentile).
GEV fitting requires far fewer permutations (#mi(`B \sim 100`);) than the empirical
approach (#mi(`B \sim 1000`);), making it tractable.

#strong[fix\_h2 optimisation.] mosaiQTL always re-uses the ML #mi(`\hat{h}^2`); from the real
phenotype rather than re-estimating it per permutation. This is valid when
#mi(`h^2`); is estimated under the null (which it is) and reduces permutation cost
by #mi(`\sim 10\times`);. miQTL re-estimates #mi(`h^2`); per permutation, which is more conservative
but computationally prohibitive at genome-wide scale.

#strong[Known issue (historical).] In an earlier phase-2 validation run, pooling across
chromosomes with very different cis-loci counts made the GEV fit unstable --- short
chromosomes with few loci produce heavy tails that inflate the pooled threshold --- and an
S6 failure (`mean_delta=0.619`) was attributed to this effect. Per-chromosome thresholds were
visualised in `s6_permutation.png` (written by `validation/tools/plot_phase2.py`) for diagnosis.

#divider()

== 14. Z-matrix expansion
<14-z-matrix-expansion>
#strong[Module:] `utils.py → build_z_matrix()` \ #strong[What it is:] mosaiQTL\'s design supports #mi(`n \geq m`); --- more phenotype
observations than unique genotyped strains. The Z-matrix maps samples to
strains at scan time, avoiding pre-collapsing to strain means and preserving
within-strain biological replication.

#mitex(`Z \in \{0,1\}^{n \times m}, \quad Z_{ik} = \mathbf{1}[\text{sample } i \text{ belongs to strain } k]`)

At scan time:

#mitex(`\mathbf{H}_n = Z\mathbf{H}_m \quad (\text{expand haplotype probs to } n \text{ samples})`)

#mitex(`\mathbf{K}_n = Z\mathbf{K}_m Z^\top \quad (\text{expand kinship})`)

When #mi(`n = m`); and IDs match, #mi(`Z = I_m`); and all formulas reduce to the
strain-mean case exactly. No module hardcodes #mi(`n`);, #mi(`m`);, or #mi(`J`);.

#divider()

== Summary table
<summary-table>
#figure(
  align(center)[#table(
    columns: 4,
    align: (auto,auto,auto,auto,),
    table.header([Concept], [Module], [Key formula], [Key property],),
    table.hline(),
    [HMM forward-backward], [`genotype/hmm.py`], [#mi(`\gamma_{i\ell}(s) \propto \alpha_{i\ell}(s)\beta_{i\ell}(s)`);], [36 diplotypes collapsed to additive],
    [Kinship], [`genotype/kinship.py`], [#mi(`K \propto \frac{1}{L}\sum_\ell H_\ell H_\ell^\top`);, mean diag 1], [shared-ancestry covariance],
    [LOCO kinship], [`genotype/kinship.py`], [#mi(`K^{(-c)}`); excludes chr #mi(`c`);], [avoids QTL confounding],
    [ML #mi(`h^2`);], [`qtl/lmm.py`], [eigendecomp trick, grid + bounded Brent], [once per gene per chromosome],
    [LMM scan LOD], [`qtl/lmm.py`], [#mi(`\frac{n}{2}\log_{10}\frac{\text{RSS}_0}{\text{RSS}_1}`);], [ML denominator],
    [FWL vectorization], [`qtl/lmm.py`], [project covariates once, batch loci], [#mi(`\sim 100\times`); faster, exact],
    [RINT], [`utils.py`], [Blom\'s formula], [rank-preserving normalisation],
    [NB Stage 1 (native; NEBULA-LN legacy)], [`qtl/nb.py`], [NB-gamma IRLS + Newton on subject sums; moment #mi(`\hat{\phi}`);], [no edgepython needed],
    [EB dispersion shrinkage], [`qtl/nb.py`], [fixed-weight blend toward binned-median trend], [stabilises low-expr genes],
    [NB Stage 2 (no kinship)], [`qtl/nb.py`], [#mi(`Q_\ell = U_\ell^\top (H^\top W H)^{-1} U_\ell`);], [no RE; #mi(`\sigma^2_\text{sample}`); inflation],
    [NB Stage 2 (LOCO kinship)], [`qtl/nb.py`], [#mi(`T_\ell = U_\ell^\top (H^\top P H)^{-1} U_\ell \sim \chi^2_{J-1}`);], [PQL, Chen et al. 2016],
    [vQTL], [`qtl/vqtl.py`], [scan #mi(`\log\hat{\phi}_i`); as phenotype], [per-strain dispersion (expression noise)],
    [Mediation LOD-drop], [`qtl/mediation.py`], [#mi(`\max(\text{LOD\_base} - \text{LOD\_mediated}, 0)`);], [rank-1 update; drop clamped at 0],
    [Khatri-Rao epistasis], [`qtl/epistasis.py`], [#mi(`H^a \odot H^b \in \mathbb{R}^{n \times (J-1)^2}`);], [pairwise haplotype interaction],
    [Peak conditioning (\"BLUP\")], [`qtl/epistasis.py`], [subtract #mi(`H\hat{\gamma}`);, rescan], [unshrunk GLS peak effect],
    [GEV permutation], [`qtl/permutation.py`, `qtl/thresholds.py`], [fit GEV to #mi(`\{M_b\}`);], [extreme value theory; QC only],
    [Z-matrix], [`utils.py`], [#mi(`Z \in \{0,1\}^{n\times m}`);], [#mi(`n \geq m`); replication support],
  )]
  , kind: table
  )

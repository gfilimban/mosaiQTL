#import "_setup.typ": *
#show: mathdoc

= Scan Modes: NB-GLMM, Additive LMM, Full Diplotype, and Multiple Imputation
<scan-modes-nb-glmm-additive-lmm-full-diplotype-and-multiple-imputation>
#strong[Modules:] `qtl/scan.py`, `qtl/lmm.py`, `qtl/mi.py`, `qtl/nb.py` \ #strong[miQTL parallel:] `scan.h2lmm(use.multi.impute=FALSE/TRUE, model='additive'/'full')`

#divider()

== 1. Two scan engines --- different observation granularities
<1-two-scan-engines--different-observation-granularities>
mosaiQTL has two independent scan engines with fundamentally different input spaces:

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([], [NB-GLMM (`nb.py`)], [LMM (`lmm.py` + `scan.py`)],),
    table.hline(),
    [#strong[Input]], [Raw single-cell / pseudobulk counts], [Continuous phenotype],
    [#strong[Observation level]], [Cell (Stage 1) → pseudobulk per sample (Stage 2)], [Sample (strain means only in the MI path)],
    [#strong[Strain reduction]], [Never --- one pseudobulk sum #mi(`S_{ig}`); per sample; Z expands haplotypes to sample rows], [None: Z expands kinship (#mi(`Z K Z^\top`);) and haplotypes to sample level; MI scan reduces to strain means],
    [#strong[Kinship integration]], [PQL + Woodbury (Stage 2), when `kinship_loco` is given], [Eigendecomp WLS],
    [#strong[Primary use]], [snRNA-seq eQTL (experimental)], [Production eQTL scans on pseudobulk (Nextflow); validation],
    [#strong[Status]], [Experimental; Python API only (`qtl/nb.py`, `pipeline.discover`), not wired into Nextflow or the CLI], [Production model (Nextflow `run_lmm_scan_chr.py` → `scan_multi_gene_lmm`)],
  )]
  , kind: table
  )

#divider()

== 2. NB-GLMM scan (experimental)
<2-nb-glmm-scan-primary>
#strong[Full model description:] `08_nb_glmm.typ`

The NB-GLMM scan is an experimental model for count-based omics data; production
scans use the Gaussian LMM (section 3). Observations
are raw UMI counts; no log-normalization, RINT, or strain-mean reduction is applied.
RINT is an LMM-specific preprocessing step that converts a continuous phenotype
to approximate normality --- it has no meaning for discrete counts and is never
called in the NB path.

#strong[Stage 1] (`nb.py::fit_null_sc_native`): Fits a NB-gamma null model at
the #strong[cell level] --- each cell is one observation. Outputs per-gene:

- #mi(`\hat\phi_g`); --- EB-shrunk cell-level NB dispersion
- #mi(`\hat\sigma^2_g`); --- subject-level overdispersion (between-strain variance)

#strong[Stage 2] (`nb.py::scan_nb_gene`): Scans at #strong[pseudobulk level] using
#mi(`S_{ig} = \sum_{j \in \text{cells of sample } i} Y_{ijg}`); (sum, not mean).
The NB score test at locus #mi(`\ell`); is:

#mitex(`Q_\ell = U_\ell^\top \mathcal{I}_\ell^{-1} U_\ell \sim \chi^2(J-1),
\qquad \text{LOD}_\ell = \frac{Q_\ell}{2 \ln 10}`)

where #mi(`U_\ell`); is the score vector and #mi(`\mathcal{I}_\ell`); is the Fisher information
at the null. With LOCO kinship the null is fitted #strong[once per gene per chromosome]
(without kinship, one global NB GLM null per gene); the score is then
vectorized over all #mi(`L`); loci.

No strain-mean transformation is applied in the NB path; when samples and cache strains differ,
the Z-matrix only expands the strain-level haplotype probabilities to sample rows. The pseudobulk
sum #mi(`S_{ig}`); is one observation per sample per gene --- the NB variance function
#mi(`\operatorname{Var}(S) = \mu + \phi \mu^2`); accounts for both Poisson and overdispersion
components without first converting to a continuous phenotype.

#divider()

== 3. LMM additive (ROP) scan
<3-lmm-additive-rop-scan>
#strong[miQTL name:] `scan.h2lmm(use.multi.impute=FALSE, model='additive')`

The LMM can handle replicates in two ways:

#strong[Path A --- Strain-mean reduction (MI scan and allele effects only):] \ Reduce #mi(`n`); sample phenotypes to #mi(`m`); strain means before scanning:

#mitex(`\bar{y}_i = \frac{1}{r_i}\sum_{j:\,s(j)=i} y_j, \qquad
  \bar{X}_i = \frac{1}{r_i}\sum_{j:\,s(j)=i} x_j`)

via #mi(`\bar{y} = (Z^\top y)/r`);. The scan then operates entirely in strain space
(#mi(`m \times m`); kinship, #mi(`m`);-row haplotype matrix). This is what `mi.py::scan_h2lmm_mi`
and `effects.py::get_allele_effects` do when `sample_ids` differ from the cache strains;
before 2026-05-28 (commit `3cc3c527`) `scan_h2lmm` did this too.

#strong[Path B --- Replicate-aware LMM with expanded kinship (current `scan_h2lmm`, `scan_multi_gene`, `scan_multi_gene_lmm`; miQTL `lmmbygls.replicates()`):] \ Keep all #mi(`n`); sample observations. Expand the strain-level kinship to sample
space via the Z-matrix and re-eigendecompose it per chromosome:

#mitex(`\tilde{K} = Z K Z^\top \in \mathbb{R}^{n \times n}, \qquad \tilde{K}_{ij} = K_{s(i),s(j)}`)

The LMM then fits #mi(`y = H_\ell \gamma + X\alpha + u + \varepsilon`); at sample level,
where #mi(`u \sim \mathcal{N}(0, \sigma^2_g \tilde{K})`);. This correctly models the
covariance between replicate samples of the same strain (they share the same
genetic background, so #mi(`\tilde{K}_{ij} = K_{s(i),s(i)} > 0`); for #mi(`s(i) = s(j)`);).

#strong[Which to use:] Path A discards within-strain phenotype variance and treats
all replicates as a single data point. Path B retains replicate-level information
and is statistically more powerful when phenotypes vary substantially within a
strain. For Gaussian phenotypes with low within-strain variance, both paths give
nearly identical LOD profiles. Path B is the replicate path of the main LMM scan
(RINT is applied to the sample-level phenotype before the scan).
Known issue: the MI scan (Path A) and the main scan (Path B) therefore fit different models on replicated data, and their LODs are not directly comparable.

The additive scan uses #mi(`H_\ell \in \mathbb{R}^{m \times J}`); (Path A) or
#mi(`\tilde{H}_\ell = Z H_\ell \in \mathbb{R}^{n \times J}`); (Path B, with #mi(`Z`); fused into the
eigenvector rotation in `lmm.py::precompute_chromosome`) as the
haplotype design matrix. Betas are per-founder allele effects in either case.

#divider()

== 4. LMM full diplotype scan
<4-lmm-full-diplotype-scan>
#strong[miQTL name:] `scan.h2lmm(use.multi.impute=FALSE, model='full')`

The full model constructs the outer-product diplotype design matrix:

#mitex(`P(f_i / f_j \mid s,\ell) \approx H_{s i \ell} \cdot H_{s j \ell}, \quad i \leq j`)

for strain #mi(`s`); and founders #mi(`i, j`);.

This gives #mi(`J(J+1)/2 = 36`); diplotype columns for #mi(`J = 8`);.

The outer-product approximation is exact for inbred CC strains because
off-diagonal heterozygous diplotype probabilities #mi(`P(f \neq f')`); are near zero ---
the marginal product is numerically equivalent to the true joint diplotype probability
in that regime.

#strong[Implementation] (`lmm.py::precompute_chromosome` and `scan_chromosome_batch`, `model='full'`;
reachable from `scan_h2lmm`, `scan_multi_gene` and `scan_multi_gene_lmm`, not from the MI scan):

```python
_ii, _jj = np.triu_indices(J)
probs  = probs[:, :, _ii] * probs[:, :, _jj]    # (L, m, 36)
```

Betas shape: `(n_loci, 36)`. Effective df: #mi(`q_\ell = \text{rank}(G_\ell)`); per locus, at most 35
(one linear constraint remains: #mi(`\sum_i H_{si}^2 + 2\sum_{i<j} H_{si}H_{sj} = 1`);).
Use full diplotype when testing dominance or parent-of-origin effects; otherwise
the additive model uses 7 df and has higher power for purely additive traits.

#divider()

== 5. Multiple imputation (MI) scan
<5-multiple-imputation-mi-scan>
#strong[miQTL name:] `scan.h2lmm(use.multi.impute=TRUE, num.imp=K)` \ #strong[Implementation:] `mi.py::scan_h2lmm_mi` → `lmm.py::scan_chromosome_mi` (gene loop: `mi.py::scan_multi_gene_mi`) \ #strong[Applies to the additive LMM only] --- not implemented for NB-GLMM or `model='full'`.

The MI scan converts haplotype uncertainty (soft probabilities) into #mi(`K`); discrete
hard-call draws (`n_imp`, default 11; `seed` default 1) and averages the resulting LOD scores.
Replicated phenotypes are reduced to strain means (Path A, section 3).

#strong[Algorithm:]

+ #strong[Pre-estimate #mi(`h^2`);] once per chromosome under the null model (kinship and covariates only, no haplotype term).
  This #mi(`h^2`); is held fixed across all #mi(`K`); draws.

+ #strong[For each draw #mi(`k = 1, \ldots, K`);] (all draws vectorised in one pass, batched over loci):

  - At each locus #mi(`\ell`); and strain #mi(`i`);, sample a hard founder assignment:
    #mi(`z_{i\ell}^{(k)} \sim \text{Categorical}(h_{i\ell})`);,
    using inverse CDF (cumulative #mi(`h_{i\ell}`);, compare uniform draw).
  - Build binary hard-call matrix #mi(`H_\ell^{(k)} \in \{0,1\}^{m \times J}`);.
  - Run the FWL scan of `03_lmm_scan.typ` with fixed #mi(`h^2`); → #mi(`\text{LOD}^{(k)}_\ell`);.

+ #strong[Average:] #mi(`\text{LOD}^{\text{MI}}_\ell = K^{-1}\sum_k \text{LOD}^{(k)}_\ell`);.

The returned `ScanResult.lod` is this average; its `p_value` and `betas` come from a separate
scan on the soft probabilities. Known issue: `scan_window='local'` is validated but not applied, so the MI scan always covers the whole chromosome.

#strong[Why MI over soft additive?] The soft additive model assigns fractional weights
to all founders at uncertain loci, implicitly assuming the trait effect is linear
in founder dosage. MI makes no such assumption --- it samples the discrete ancestry
state and averages the discrete LOD. For DO outbred mice with high locus uncertainty,
MI gives better-calibrated LOD scores. For fully inbred CC strains, soft additive #mi(`\approx`); MI
because ancestries are already near-deterministic.

#strong[Choosing #mi(`K`);:] The variance of the averaged LOD decreases as #mi(`1/K`);, so
there are diminishing returns beyond a moderate number of draws. #mi(`K = 3`);--#mi(`5`);
is adequate for screening; #mi(`K = 11`); is the miQTL reference default (used in
mosaiQTL validation for parity); #mi(`K = 50`);--#mi(`100`); is recommended for
publication-quality results at loci with high haplotype uncertainty (e.g., DO
mice where multiple founders share roughly equal probability). For fully inbred
CC strains, #mi(`K = 11`); is already stable because per-locus ancestry is
near-deterministic. There is no statistical argument for exactly 11 --- it is
a convention, not a requirement.

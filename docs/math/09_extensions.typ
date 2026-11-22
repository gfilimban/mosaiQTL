#import "_setup.typ": *
#show: mathdoc

= Extensions: Mediation, Epistasis, and vQTL
<extensions-mediation-epistasis-and-vqtl>
#strong[Modules:] `qtl/mediation.py`, `qtl/epistasis.py`, `qtl/vqtl.py` \ #strong[miQTL parallel:] `mediation.scan.qr()`, `scan.h2lmm()` (conditional), `instability.lm.scan()`

#divider()

== 1. Mediation scan (LOD-drop)
<1-mediation-scan-lod-drop>
#strong[Question:] Does the haplotype effect on outcome gene #mi(`y`); operate through
intermediate gene #mi(`m`);? If #mi(`\text{haplotype} \to m \to y`);, then conditioning on #mi(`m`);
should reduce (or eliminate) the haplotype LOD for #mi(`y`);.

#strong[Statistic:]

#mitex(`\text{LOD}_{\text{drop},\ell} = \max\!\left(\text{LOD}_{\text{base},\ell} - \text{LOD}_{\text{mediated},\ell},\ 0\right)`)

where:

- #mi(`\text{LOD}_{\text{base},\ell}`); = haplotype LOD for #mi(`y`); with covariates #mi(`X`); only.
- #mi(`\text{LOD}_{\text{mediated},\ell}`); = haplotype LOD for #mi(`y`); with #mi(`[m_{\text{std}},\, X]`); as covariates (mediator added).

#mi(`\text{LOD}_{\text{drop}} > 0`);: the haplotype effect shrinks when #mi(`m`); is controlled ---
mediation detected. \ A negative difference (amplification --- a shared upstream regulator drives
both #mi(`y`); and #mi(`m`); in the same direction, so conditioning on #mi(`m`); removes shared signal) is
clamped to 0 in both implementations, so amplification is never reported.

#strong[Two implementations:]

`mediation_scan_qr` --- LMM with LOCO kinship (computed if `kinship_decomp` is not supplied).
Estimates #mi(`h^2`); once per chromosome from the base null model #mi(`y \sim X`); and reuses it
for the mediated model and for every mediator. Uses the full FWL batch scan per chromosome;
each mediator's LOD is a Woodbury rank-1 update of the base quantities, so many mediators are
tested per call. `scan_window` is `'genome'` (default), `'cis'` or `'local'`. `p_value` is
#mi(`\chi^2_1`); applied to #mi(`2\ln 10 \cdot \text{LOD}_{\text{drop}}`);. Slower but accounts for
polygenic background. Known issue: #mi(`\text{LOD}_{\text{drop}}`); is a difference of two LODs from
different model comparisons, not a likelihood-ratio statistic, so this #mi(`\chi^2_1`); p-value has no
valid null (ledger MED-02).

`fixef_mediation_scan` --- OLS, no kinship, one mediator per call. Same FWL batch scan without
eigendecomp. Used for fast screening or when kinship is not available. `just_these_loci` filter
enables cis-window restriction. Its `p_value` is the F-test of the haplotype term in the
mediated model, not a test of mediation.

#strong[Implementation detail:] The mediator #mi(`m`); is z-score standardized before being
added to the covariate matrix to put it on a comparable scale to the intercept and
other covariates. This prevents numerical issues when #mi(`m`); has a very different
magnitude than #mi(`y`);.

#divider()

== 2. Epistasis scan
<2-epistasis-scan>
#strong[Module:] `qtl/epistasis.py`

#strong[Question:] Do two loci #mi(`\ell_1`); and #mi(`\ell_2`); have an interaction effect on #mi(`y`);
beyond their additive contributions?

#strong[Model:]

#mitex(`y = H_{\ell_1}\gamma_1 + H_{\ell_2}\gamma_2 + (H_{\ell_1} \otimes H_{\ell_2})\Gamma_{12} + X\alpha + u + \varepsilon`)

where #mi(`H_{\ell_1}, H_{\ell_2} \in \mathbb{R}^{n \times (J-1)}`); (last founder dropped) and
#mi(`H_{\ell_1} \otimes H_{\ell_2} \in \mathbb{R}^{n \times (J-1)^2}`); is their row-wise Kronecker
(transposed Khatri-Rao) product --- row #mi(`i`); is #mi(`h_{\ell_1,i} \otimes h_{\ell_2,i}`); --- encoding all
pairwise founder combinations.

#strong[Pairwise test (`scan_epistasis`):] tests one specified pair #mi(`(\ell_1, \ell_2)`); (`locus_a`,
`locus_b`); there is no genome-wide sweep over #mi(`\ell_2`);. #mi(`h^2`); is estimated once by ML on the
null model with the LOCO kinship of #mi(`\ell_1`);'s chromosome and held fixed for both fits. The
interaction LOD is
#mi(`\text{LOD}_{\text{epi}} = \max(\text{LOD}_{\text{int}} - \text{LOD}_{\text{add}},\ 0)`);, with
#mi(`p = 1 - F_{\chi^2_{(J-1)^2}}(2 \ln 10 \cdot \text{LOD}_{\text{epi}})`);.
Known issue: for #mi(`J = 8`); the interaction design has #mi(`2(J-1) + (J-1)^2 + p = 63 + p`); columns, so
with fewer strains than that the fit is saturated and the interaction LOD is not meaningful (ledger EPI-03).

#strong[Conditional scan (`scan_conditional_with_y`):] given an established QTL at
#mi(`\ell_1`); (the current peak), test for additional independent signals by subtracting the
fitted peak haplotype effect #mi(`H_{\ell_1}\hat\gamma_1`); (GLS estimate at the LOCO #mi(`h^2`);) from
#mi(`y`); and re-scanning the residual genome-wide with `scan_h2lmm` (no RINT); repeated up to
`n_signals` (default 3) and stopped when the rescan's maximum LOD is below 1.5. A secondary peak in
the conditional scan indicates a distinct signal. This is the standard multi-signal detection
approach (analogous to stepwise regression in GWAS fine-mapping). `scan_conditional_with_y` has no
caller in `src/`; `pipeline.characterize` runs `scan_conditional`, which does no conditioning:
it takes the next-highest LOD on the original profile outside an exclusion window. Known issue: that
window is written as #mi(`5 \times 10^6`); while positions are in Mb, so every locus is excluded and
`scan_conditional` never returns more than one signal (ledger EPI-02).

#divider()

== 3. Variance QTL (vQTL)
<3-variance-qtl-vqtl>
#strong[Module:] `qtl/vqtl.py`

#strong[Question:] Does the haplotype at locus #mi(`\ell`); affect the #emph[variance] (dispersion)
of #mi(`y`); between cells of the same subject, beyond any effect on the mean?

The vQTL tests whether founder ancestry predicts the per-subject cell-to-cell overdispersion
#mi(`\phi`); of #mi(`y`);. The mean QTL effect is not removed; with `also_scan_mean=True` (default) a
separate mean scan of per-subject mean counts is run for comparison.

#strong[Correct Stage 2 design (planned --- VQTL-01):] \ Stage 2 should run a NB-GLMM #strong[dispersion score test]: the null is #mi(`\phi`); constant
across haplotypes; the alternative is #mi(`\log\phi_{ig} = H_{i\ell}^\top\delta + \ldots`);.
The score test statistic for dispersion effects uses the NB score gradient with
respect to #mi(`\phi`);, not with respect to the mean #mi(`\mu`);.

#strong[Current implementation caveat (VQTL-01 open):] \ The current `scan_vqtl` incorrectly calls `scan_h2lmm` (Gaussian LMM) on
#mi(`\log\hat\phi`); per subject. #mi(`\hat\phi`); is by default the moment estimate
#mi(`\max\{(\operatorname{var} - \text{mean})/\text{mean}^2,\ 10^{-6}\}`); of the subject's raw cell counts
(subjects with at least 20 cells and mean above 0.01; `phi_method='mle'` is the alternative), RINT is
applied by default, and the scan is genome-wide. This treats the estimated log-dispersion as a
continuous phenotype and applies the LMM scan to it. While this is a reasonable
approximation when #mi(`m`); is large and #mi(`\hat\phi`); estimates are stable, it is not
the proper model: (i) #mi(`\hat\phi`); estimates are noisy for low-count genes, (ii)
the LMM does not account for the asymmetric distribution of #mi(`\log\hat\phi`);, and
(iii) the scan stat is no longer a proper likelihood ratio test.

Until VQTL-01 is resolved, interpret vQTL results with caution. Known issue: `scan_vqtl` does not
pass `sample_ids` to `scan_h2lmm`, so #mi(`\log\hat\phi`); (in sorted subject order) is assumed to be
in genome-cache order and of cache length (ledger VQ01).

#divider()

== 4. LOD-drop confidence interval (Manichaikul)
<4-lod-drop-confidence-interval-manichaikul>
#strong[Module:] `qtl/peaks.py::lod_drop_ci` (engine `ScanResult`, exported as `mosaiqtl.lod_drop_ci`;
used by `find_peaks`) and `qtl/ci.py::lod_support_interval` (used by `ci_scan_multi_gene`, the CLI
`characterise` CI step; `ci.lod_drop_ci` wraps it for `models.records.ScanResult`) \
#strong[miQTL parallel:] `calc.LOD.drop()`

The LOD-drop CI is the genomic interval, on the peak chromosome, from the first to the last locus where:

#mitex(`\text{LOD}(\ell) \geq \text{LOD}_{\text{peak}} - \Delta`)

Common choices: #mi(`\Delta = 1.5`); (#mi(`\approx 95\%`); Bayes CI under Haley-Knott; the code default),
#mi(`\Delta = 1.8`);.
It does not propagate phenotypic sampling uncertainty. The current code has no bootstrap
positional CI (the one described in section 6 of `06_permutation.typ`); the LOD support interval is
the only positional CI the package computes.

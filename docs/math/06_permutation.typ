#import "_setup.typ": *
#show: mathdoc

= Permutation Thresholds and Positional Confidence Intervals
<permutation-thresholds-and-positional-confidence-intervals>
#strong[Module:] `qtl/permutation.py` (null draws, threshold scans), `qtl/thresholds.py` (GEV fit),
`qtl/ci.py` (positional CI) \ #strong[miQTL parallel:] `generate.sample.outcomes.matrix()`, `run.threshold.scans()`,
`get.gev.thresholds()`

#strong[Role in production:] the Nextflow `PERMUTE` process (`workflow/scripts/run_permute.py`) builds this
null for the LMM scan, and `CALL_EGENES` logs the resulting GEV threshold for QC only. eGenes are
called from per-gene calibrated p-values and Storey q-values (`11_storey_qvalue.typ`). The CLI
`mosaiqtl call-egenes` (`cli.py::call_egenes_cmd`) differs: it keeps only genes whose peak LOD
exceeds this threshold before Storey. Known issue: that LOD-selected q-value does not control FDR (ledger ORC-02).

#divider()

== 1. Why permutation thresholds
<1-why-permutation-thresholds>
Genome-wide LOD significance thresholds derived from #mi(`\chi^2`); or #mi(`F`); approximations
are anti-conservative for multiparent populations because:

- Multiple testing across #mi(`L`); correlated loci inflates Type I error.
- The correlation structure between loci (LD / haplotype blocks) is
  population-specific and cannot be captured by Bonferroni correction.

Permutation breaks the genotype--phenotype link while preserving the LD and
kinship structure of the genome. The result is an empirical null distribution
of genome-wide max-LOD from which a threshold at any #mi(`\alpha`); level is read directly.

The two scan engines would require different permutation strategies. Only the LMM path is
implemented: `run_threshold_scans` scans with the LMM only, and the NB-GLMM (`qtl/nb.py`) is
experimental and has no permutation path. The NB-GLMM column is a design sketch.

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([], [NB-GLMM (not implemented)], [LMM],),
    table.hline(),
    [What is permuted], [Strain labels on pseudobulk count matrix], [Phenotype vector #mi(`y`); (kinship-informed rank permutation by default, §2)],
    [RINT applied], [#strong[No] --- raw counts only], [Yes (default)],
    [Fixed parameter], [Dispersion #mi(`\hat\phi`); from Stage 1], [#mi(`\hat{h}^2`); from the ML null fit],
    [Null score statistic], [#mi(`Q_\ell \sim \chi^2(J-1)`);], [LOD from RSS ratio],
  )]
  , kind: table
  )

#divider()

== 2. Generating null outcomes
<2-generating-null-outcomes>
`run_threshold_scans(permutation_method=...)` supports two LMM null generators.

- #strong[Parametric] (default, `draw_parametric_null_batch`): a kinship-informed rank
  permutation. At the fixed #mi(`\hat{h}^2`); the null fit gives #mi(`\hat\beta`);,
  #mi(`\sigma^2_g = \hat{h}^2 \sigma^2`); and #mi(`\sigma^2_e = (1-\hat{h}^2)\sigma^2`);, where
  #mi(`\sigma^2 = \mathrm{WRSS}/n`); and the weights are #mi(`w_i = \hat{h}^2\lambda_i + (1-\hat{h}^2)`);. Each column
  draws

  #mitex(`y^{\ast(b)} = X\hat\beta + u^{(b)} + e^{(b)}, \qquad u^{(b)} \sim \mathcal{N}(0, \sigma^2_g K), \quad e^{(b)} \sim \mathcal{N}(0, \sigma^2_e I)`)

  and the null phenotype #mi(`y^{(b)}`); assigns the real (RINT-transformed) values of #mi(`y`); in the
  rank order of #mi(`y^{\ast(b)}`);. The marginal distribution of #mi(`y`); is preserved exactly and
  the kinship-induced correlation is kept. The contents of `null_outcomes` are ignored in
  this mode; only its column count sets #mi(`B`);.

- #strong[Plain:] shuffle the phenotype vector #mi(`y`); (`generate_null_outcomes(method='permutation')`).
  Each permuted #mi(`y^{(b)}`); is RINT-transformed and, when `sample_ids` is given without
  `z_mat`, reduced to strain means via the Z-matrix before scanning. This is valid only when
  #mi(`\tau^2 \approx 0`); (identity kinship / OLS) and is used by Phase 1 validation only.

- #strong[NB-GLMM] (not implemented): the design would shuffle strain labels on the pseudobulk
  count matrix --- i.e., permute the row order of #mi(`S \in \mathbb{Z}^{m \times G}`); so that each
  strain\'s count profile is assigned to a different genotype row --- with no RINT and the
  Stage 1 dispersion #mi(`\hat\phi_g`); held fixed.

#strong[Bootstrap] (`generate_null_outcomes(method='bootstrap')`): resample with replacement from
the centered #mi(`y - \bar{y}`);. No production step uses it; positional CIs are LOD support
intervals (§6).

#divider()

== 3. Fixed #mi(`h^2`); (LMM) and fixed #mi(`\phi`); (NB-GLMM)
#label("3-fix_h²-lmm-and-fix_φ-nb-glmm")
Re-estimating model parameters for every permuted phenotype is statistically
correct but expensive.

#strong[LMM --- fixed #mi(`h^2`);:] Pre-estimate #mi(`\hat{h}^2_c`); once from the original (RINT-transformed)
phenotype per chromosome by maximum likelihood (`lmm.estimate_h2_null(..., method='ml')`), then
reuse it across all #mi(`B`); permutations. When the `h2` argument is given, that value is used
for every chromosome instead.

#mitex(`\hat{h}^2_c = \arg\max_{h^2} \ell_{\text{ML}}(h^2 \mid y_{\text{orig}}, K^{(-c)})`)

In `run_threshold_scans` #mi(`h^2`); is always fixed; there is no `fix_h2` switch. The
parametric null is drawn once at the reference chromosome\'s #mi(`\hat{h}^2`); (§4), so it carries
that heritability by construction. Under the plain shuffle the permuted phenotype has
approximately zero heritability, and using #mi(`\hat{h}^2 > 0`); from the original data is
slightly conservative. miQTL uses `fix.h2=TRUE` by default.

#strong[NB-GLMM --- fixed #mi(`\phi`);] (not implemented): the design would hold the Stage 1 dispersion
#mi(`\hat\phi_g`); fixed, since Stage 1 runs on real cell-level data that the permutation of
strain labels does not change, and would run the NB score test on the permuted strain
assignment with that same #mi(`\hat\phi_g`);.

#divider()

== 4. Threshold scan loop
<4-threshold-scan-loop>
#strong[LMM path] (`run_threshold_scans`):

+ Apply RINT to the real phenotype (default `use_rint=True`). With `sample_ids` and no `z_mat`,
  reduce phenotype and covariates to strain means. With `z_mat` (the production path), scan at
  sample level with #mi(`K_n = Z K_m Z^\top`); (§7).
+ For each chromosome #mi(`c`);: estimate #mi(`\hat{h}^2_c`); (§3).
+ Build the #mi(`n \times B`); null matrix once. Parametric mode draws it from the reference
  chromosome\'s kinship and #mi(`\hat{h}^2`); (`ref_chrom`, else the first chromosome whose
  kinship passes the PSD check) and reuses it on every chromosome. Plain mode uses the
  RINT-transformed shuffles.
+ For each chromosome #mi(`c`);: run `lmm.scan_chromosome_batch` with fixed #mi(`\hat{h}^2_c`);, vectorised
  over the #mi(`B`); columns. With `chrom`, `gene_pos_mb` and `window_mb`, the scan is restricted to
  loci within #mi(`\pm`); `window_mb` of `gene_pos_mb` on `chrom`.
+ Record #mi(`M^{(b)} = \max_c \max_\ell \text{LOD}_\ell^{(b)}`);. The per-chromosome maxima are also
  returned (`chr_null_lods`). A chromosome that raises `KinshipSingularError` contributes zeros.

This produces #mi(`\{M^{(b)}\}_{b=1}^B`); --- the empirical null distribution of max-LOD over the
scanned loci. In Nextflow (`run_permute.py`) the phenotype is that of the representative gene
chosen by `pick_rep_gene` (the gene whose #mi(`h^2`); is nearest the median). The scanned loci depend
on `scan_type`: `L` uses that gene\'s chromosome and #mi(`\pm`); `window_mb` around its TSS (strand-aware when a
gene-TSS table is supplied, else `start_mb`), `C` uses that gene\'s whole chromosome, and `G` uses all chromosomes. The outputs are
`{tag}_perm_max_lods.csv`, `{tag}_perm_max_lods_chr{c}.csv` and `{tag}_perm_gev_params.json`.

#strong[NB-GLMM path] (not implemented): the design would reorder pseudobulk rows,
#mi(`S^{(b)}_{\pi^{(b)}(i)} \leftarrow S_i`);, run the NB score scan (fixed #mi(`\hat\phi_g`);) against the
original haplotype probs, and record #mi(`M^{(b)} = \max_c \max_\ell Q_\ell^{(b)} / (2\ln 10)`);.

#divider()

== 5. GEV threshold fitting
<5-gev-threshold-fitting>
The distribution of genome-wide maximum statistics converges to the
#strong[Generalised Extreme Value (GEV)] distribution (Fisher--Tippett--Gnedenko theorem).
Fitting GEV to #mi(`\{M^{(b)}\}`); is more stable than empirical quantiles when #mi(`B < 500`);.

#strong[GEV density:]

#mitex(`f(x;\,\mu,\sigma,\xi) = \frac{1}{\sigma}\!\left(1+\xi\frac{x-\mu}{\sigma}\right)^{-1/\xi-1}
  \!\exp\!\left[-\!\left(1+\xi\frac{x-\mu}{\sigma}\right)^{-1/\xi}\right]`)

Three families: Gumbel (#mi(`\xi=0`);), Fréchet (#mi(`\xi>0`);), Weibull (#mi(`\xi<0`);).

#strong[Implementation] (`thresholds.py::get_gev_thresholds`):

```python
finite = max_lods[np.isfinite(max_lods)]
c, loc, scale = genextreme.fit(finite)
threshold = genextreme.ppf(percentile, c, loc=loc, scale=scale)
```

Default `percentile = 0.95` gives the 5% family-wise threshold over the scanned loci. \ The empirical quantile is also returned
for comparison (`empirical_quantile` in the parameter dict). With #mi(`B \geq 1000`); both converge. The function returns
`(None, None)` when fewer than 10 finite values remain or the fit fails, and `run_call_egenes.py`
then falls back to the empirical 0.95 quantile. SciPy\'s shape parameter is #mi(`c = -\xi`);;
`build_gev_row` stores `c` unchanged as `gev_xi`. `load_per_chr_thresholds` applies the same fit to
each `perm_max_lods_chr*.csv`.

In the Nextflow pipeline this threshold is QC only. `CALL_EGENES` (`run_call_egenes.py`) logs it as a
reference, and eGene calls come from the per-gene Beta-calibrated p-values and Storey q-values of
`11_storey_qvalue.typ`. The CLI `call-egenes` still uses it as a Stage-1 filter (see the header).

#divider()

== 6. Positional confidence interval
<6-positional-confidence-interval-via-parametric-bootstrap>
#strong[Current method --- LOD support interval] (`qtl/ci.py`). Let #mi(`t_g`); be the strand-aware TSS of
gene #mi(`g`); and #mi(`w`); the scan half-window (`window_mb`). The haplotype scan evaluates the loci
#mi(`\mathcal{L}_g = \{\ell : |x_\ell - t_g| \le w\}`);, where #mi(`x_\ell`); is the position of locus #mi(`\ell`);.
`ci_scan_multi_gene` scans exactly #mi(`\mathcal{L}_g`); (ML #mi(`\hat{h}^2`);, the same `loci_mask` as the
haplotype scan) and reads the support interval from that profile. With peak locus
#mi(`\hat\ell = \arg\max_{\ell \in \mathcal{L}_g} \text{LOD}_\ell`);, the interval is the maximal run of
consecutive loci in #mi(`\mathcal{L}_g`); that contains #mi(`\hat\ell`); and satisfies
#mi(`\text{LOD}_\ell \geq \text{LOD}_{\hat\ell} - 1.5`); (`lod_drop=1.5`). Hence

#mitex(`\hat\ell \in [\mathrm{ci}_{\mathrm{lo}}, \mathrm{ci}_{\mathrm{hi}}] \subseteq [\,t_g - w,\; t_g + w\,]`)

so the interval is never wider than the scan window and always contains the peak. The flag `ci_truncated`
is true when the run reaches the first or last locus of #mi(`\mathcal{L}_g`);: the LOD profile did not fall
by 1.5 inside the window, so the interval is bounded by the window and not by the data.

#strong[Why these choices.] A LOD support region is approximately a confidence region for the location of
#emph[one] QTL, and its width scales with the sample size (Dupuis & Siegmund 1999); likelihood support
intervals and approximate Bayes credible intervals have appropriate coverage where the bootstrap does not
(Manichaikul et al. 2006). A union of disjoint segments, as produced by taking the minimum and maximum of all
loci above the drop threshold, is not an interval for one location: a secondary peak elsewhere on the
chromosome widened it (ledger CI-01). Restricting to the contiguous run fixes that. Restricting the scan to
#mi(`\mathcal{L}_g`); keeps the interval inside the region in which the gene was tested and calibrated
(§1--3 of `11_storey_qvalue.typ`). Posterior inclusion probabilities and credible sets are conditional on the
region analysed (Maller et al. 2012; Schaid et al. 2018), so the fine-mapping region must be fixed
beforehand and must not exceed the tested region. cis-eQTL positions concentrate around the TSS
(Veyrieras et al. 2008), so one TSS anchor defines the scan window, the support-interval scan, the permutation
null and the fine-mapping window. Nextflow calls `ci_scan_multi_gene` from `run_finemap.py`.

#strong[History --- bootstrap CI:] before 2026-10 (`run_positional_scans`, removed in commit 09208c41) the package resampled #mi(`y - \bar{y}`); with replacement,
applied RINT, ran `scan_chromosome` on the peak chromosome with fixed #mi(`\hat{h}^2`);, and took the
empirical #mi(`[\alpha/2, 1-\alpha/2]`); quantiles of the bootstrap peak positions (default
#mi(`\alpha = 0.05`);; validation gate P2-S10 checked that the CI contained the planted locus). The
code no longer contains this path. The `ci.py` module docstring explains why: the phenotypic
bootstrap breaks the genotype--phenotype pairing and gives near-chromosome-wide intervals
(see also Manichaikul et al. 2006). A count-preserving NB-GLMM bootstrap was never implemented.

#divider()

== 7. PSD projection of the sample-level LOCO kinship
<7-psd-projection-of-the-sample-level-loco-kinship>
#strong[Context.] The parametric null draw (`draw_parametric_null_batch`, used by
`run_threshold_scans` and by the per-gene `fdr.calibrate_gene_pvalue`) requires a
positive semi-definite (PSD) kinship matrix #mi(`K_n`); at sample level. When a
#mi(`Z`);-matrix is supplied (replicate design), #mi(`K_n`); is constructed as

#mitex(`K_n = Z \, K_m \, Z^\top`)

where #mi(`K_m`); is the #mi(`m \times m`); strain-level LOCO kinship reconstructed from
its eigendecomposition `(evc_m * ev_m) @ evc_m.T`. In `run_threshold_scans` the eigenvalues of
#mi(`K_n`); are also clamped at zero when it is built. `pipeline._calibrate_pvalues_loop` passes the
unclamped `eigh(K_n)` and relies on the clamp below.

#strong[The problem (history).] The strain-level LOCO kinship #mi(`K_m`); is theoretically PSD but
accumulates floating-point error of #mi(`\mathcal{O}(10^{-7})`); in its near-zero
eigenvalues (observed: #mi(`-3.09 \times 10^{-7}`);). The earlier PSD check in
`_draw_parametric_batch` (the former name of `draw_parametric_null_batch`) recomputed `eigvalsh(K)` with a machine-epsilon
tolerance that is orders of magnitude tighter than this noise. It raised a
false `KinshipSingularError` and skipped every chromosome, which exhausted the
null draw and collapsed the GEV threshold to 0.0. The threshold then gated eGene calls, so every
gene was called as an eGene.

#strong[The fix.] `draw_parametric_null_batch` already receives `ev` (the
eigenvalues of #mi(`K`);) and `evc` (its eigenvectors). Rather than recomputing
`eigvalsh(K)`, we use `ev` directly, clamp near-zero negatives to zero, and
draw #mi(`U`); from the eigendecomposition:

#mitex(`U \;=\; V \,\operatorname{diag}\!\bigl(\sqrt{\max(\lambda_i,0)\,\sigma^2_g}\bigr)\, Z,
\quad Z \sim \mathcal{N}(0, I)`)

This is mathematically equivalent to #mi(`U \sim \mathcal{N}(0,\,K\sigma^2_g)`);
when #mi(`K`); is exactly PSD, and draws from the nearest PSD projection otherwise.
`KinshipSingularError` is raised only when #mi(`\min(\lambda_i) < -10^{-4}`);, which
signals a genuine defect rather than float noise.

#strong[Implementation] (`permutation.py::draw_parametric_null_batch`):

```python
if float(ev.min()) < -1e-4:
    raise KinshipSingularError(...)
sqrt_ev = np.sqrt(np.maximum(ev, 0.0) * sigma2_g)
U_mat = evc @ (sqrt_ev[:, None] * rng.standard_normal((len(ev), n_perms)))
```

#strong[Why not Tikhonov regularisation?] Adding #mi(`\varepsilon I`); shifts #emph[all]
eigenvalues uniformly and changes the kinship structure for the #mi(`m`); positive
eigenvalues that are correct. Clamping only corrects the #mi(`n - m`); eigenvalues
that are theoretically zero --- the kinship signal is preserved exactly.

#divider()

== References
<references>
\[1\] Higham, N.J. (1988). Computing a nearest symmetric positive semidefinite
matrix. #emph[Linear Algebra and its Applications], 103, 103--118.
#link("https://doi.org/10.1016/0024-3795(88)90223-6")

\[2\] Manichaikul, A. et al. (2006). Poor performance of bootstrap confidence
intervals for the location of a quantitative trait locus. #emph[Genetics], 174(1),
481--489. #link("https://doi.org/10.1534/genetics.106.061549")

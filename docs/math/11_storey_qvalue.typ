#import "_setup.typ": *
#show: mathdoc

= Per-Gene Beta-Approximated Permutation p-values and Storey q-values
<per-gene-beta-approximated-permutation-p-values-and-storey-q-values>
#strong[Module:] `qtl/fdr.py` (`calibrate_gene_pvalue`, `beta_calibrate_pvalue`, `qvalue`, `bh`),
`qtl/permutation.py` (`draw_parametric_null_batch`), `pipeline.py` (`calibrate_egenes_storey`,
`call_significant_egenes`); Nextflow `CALL_EGENES` → `workflow/scripts/run_call_egenes.py`
#strong[miQTL parallel:] none --- miQTL uses BH-FDR on genome-wide max-LOD; this design follows the
FastQTL/tensorQTL/GTEx precedent instead (see §0).
#strong[Replaced (history):] the BH-FDR placeholder in `pipeline.py::_SIG_COLS` (`fdr_method = 'BH'`), flagged
`SWE-09` since scaffolding. `fdr_method` is now the constant `'qvalue'`.

#divider()

== 0. Why this document exists
<0-why-this-document-exists>
Two unrelated multiple-testing problems are stacked in every eQTL scan:

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Problem], [Tested units], [Current mechanism],),
    table.hline(),
    [Multiple #strong[loci] within one gene\'s scan], [#mi(`L`); correlated loci (LD/haplotype blocks)], [Per-gene parametric null of the window max-LOD + Beta fit (§1--3); the GEV threshold of `06_permutation.typ` is QC only],
    [Multiple #strong[genes] scanned genome-wide], [#mi(`G`); genes (#mi(`10^4`);--#mi(`10^5`);)], [Storey q-value over all tested genes (§4); BH placeholder before 2026-05],
  )]
  , kind: table
  )

#strong[History.] Before per-gene calibration, `pipeline.py` defined `pval_perm`/`fdr` fields in
`EGENE_SUMMARY_COLS` but no function populated them --- `finalize_egene_summary()` never touched
them, and `call_egenes()` applied a single scalar LOD threshold (`get_gev_thresholds`) genome-wide
instead. A prior attempt to wire Storey q-values through this scalar-threshold p-value produced
#mi(`\hat\pi_0 = 2.58`); --- mathematically impossible for a proportion --- because the p-value passed
to Storey was computed against #strong[one representative gene\'s] permutation null. Genes whose
#mi(`\hat{h}^2`); differs from that one gene\'s value are scored against the wrong null, so their
p-values are not #mi(`\mathrm{Uniform}(0,1)`); under #mi(`H_0`); and Storey\'s flat-tail assumption (§4) is violated
systemically.

The fix is #strong[per-gene calibration]: every gene\'s p-value must come from a null built from that
gene\'s own fixed nuisance parameter. Full per-gene permutation (#mi(`\geq 1000`); perms #mi(`\times G`); genes) would
violate the project\'s `SPEED-01` runtime goal (\<4h genome-wide). The standard
field solution --- used by FastQTL (Ongen et al. 2016, #emph[Bioinformatics] 10.1093/bioinformatics/btv722,
listed in `docs/REFERENCES.md`) and
adopted by tensorQTL and the GTEx pipeline --- is to run a #strong[small] number of per-gene permutations
and fit a #strong[Beta distribution] to the resulting null: the Beta family absorbs most of the sampling
noise a raw empirical tail would carry at low #mi(`B`);, so #mi(`B \approx 100`);--#mi(`1000`); per gene suffices.

#divider()

== 1. Per-gene max statistic and its null
<1-per-gene-max-statistic-and-its-null>
For gene #mi(`g`); with cis-window loci #mi(`\ell = 1,\dots,L_g`);, define the per-gene test statistic as the
best (minimum) nominal p-value across the window:

#mitex(`p_{\min,g} = \min_{\ell=1,\dots,L_g} p_{\ell,g} = 1 - F_{\chi^2_{J-1}}\!\left(2\ln(10)\,\text{LOD}_{\text{peak},g}\right)`)

where #mi(`p_{\ell,g}`); is the locus-level nominal p-value implied by the scan\'s LOD through the
likelihood-ratio relation #mi(`2\ln(10)\,\text{LOD} \sim \chi^2_{J-1}`);, with #mi(`J`); = `n_founders`. The relation
is monotone, so the code converts only the peak LOD (`beta_calibrate_pvalue`, p-values clipped to
#mi(`[10^{-300}, 1]`);). #mi(`p_{\min,g}`); is the eQTL-calling analogue
of the max-LOD in §4 of `06_permutation.typ`, restricted to one gene\'s own window.

For each gene, `calibrate_gene_pvalue` generates #mi(`B`); null phenotypes #strong[using that gene\'s own fixed nuisance parameter]:

+ RINT the gene\'s phenotype (`use_rint=True`) and estimate #mi(`\hat{h}^2_g`); by ML on its chromosome\'s LOCO
  kinship. This repeats the scan\'s estimate rather than reusing it (ledger FD-07).
+ Draw #mi(`B`); kinship-informed rank-permuted nulls at #mi(`\hat{h}^2_g`); with `draw_parametric_null_batch`
  (`06_permutation.typ` §2).
+ Scan them with `lmm.scan_chromosome_batch` over the same locus window as the real scan, and record
  the per-permutation max LOD, which gives #mi(`\{p_{\min,g}^{(b)}\}_{b=1}^B`); after the #mi(`\chi^2_{J-1}`); conversion.

In the pipeline (`pipeline._calibrate_pvalues_loop`) the scan is at sample level with
#mi(`K_n = Z K_m Z^\top`);, eigendecomposed once per chromosome. The window comes from the scan tag: `Lw<N>`
means #mi(`\pm N`); Mb around the gene TSS, and `C` means the whole chromosome. Each gene gets its own seed
stream derived from the base seed and its `gene_id`. A failed gene gets `NaN`. The library default is
#mi(`B = 200`);, and the Nextflow configs set `n_perms_cal: 1000`, the same as the genome-wide `n_perms`.

#divider()

== 2. Engine-agnosticism of §1 (LMM and NB-GLMM)
<2-engine-agnosticism-of-1-lmm-and-nb-glmm>
§1\'s null-generation loop consumes only a scalar max-statistic per permutation --- the Beta and
Storey steps (§3--4) have no dependence on which engine produced it. Only the LMM null is
implemented: `calibrate_gene_pvalue` is LMM-only, and the NB-GLMM (`qtl/nb.py`) is experimental and
not wired into calibration, the CLI or Nextflow. The NB-GLMM column is a design sketch.

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([], [LMM], [NB-GLMM (not implemented)],),
    table.hline(),
    [Fixed nuisance param (never refit per permutation)], [#mi(`\hat h^2_g`); (ML on real #mi(`y`);)], [#mi(`\hat\phi_g`); (Stage 1, real cell-level data)],
    [Per-permutation cost], [`scan_chromosome_batch` --- fixed #mi(`\hat h^2_g`);, rotated genotypes precomputed once per chromosome (`chr_data`), vectorized over permutations and the #mi(`L_g`); loci], [`_score_stats_chromosome` / `_score_stats_loco` --- same shape: null fit once, score stat vectorized over all loci],
    [Statistic #mi(`\to`); LOD], [RSS-ratio LOD], [#mi(`Q_\ell\sim\chi^2(J{-}1) \Rightarrow \text{LOD}=Q_\ell/(2\ln10)`);],
  )]
  , kind: table
  )

Fitting the null once and vectorising over all loci is the property that makes per-gene (not
just genome-wide) permutation tractable. In the NB-GLMM design, #mi(`\hat\phi_g`); would never be
re-estimated during permutation (`06_permutation.typ` §3), so its marginal per-permutation cost
would be score-stat evaluation only.

#divider()

== 3. Beta approximation to the per-gene null
<3-beta-approximation-to-the-per-gene-null>
Under #mi(`H_0`);, order statistics of #mi(`L_g`); correlated p-values do not have a simple closed form, but
their minimum is well approximated by a #mi(`\mathrm{Beta}(a, b)`); distribution --- the shape a minimum
of correlated Uniform-ish variables takes when #mi(`L_g`); is roughly stable across the window (Ongen et
al. 2016). Two parameters absorb the #emph[effective] number of independent tests without computing it
explicitly.

#strong[Method-of-moments starting point], from the empirical mean #mi(`\hat\mu`); and variance #mi(`\hat\sigma^2`);
of #mi(`\{p_{\min,g}^{(b)}\}`);:

#mitex(`\hat{a}_0 = \hat\mu\left(\frac{\hat\mu(1-\hat\mu)}{\hat\sigma^2} - 1\right), \qquad
  \hat{b}_0 = (1-\hat\mu)\left(\frac{\hat\mu(1-\hat\mu)}{\hat\sigma^2} - 1\right)`)

In code the bracket is floored at #mi(`10^{-3}`);, as are #mi(`\hat{a}_0`); and #mi(`\hat{b}_0`);. The start is
#mi(`(1, 1)`); when #mi(`\hat\sigma^2 = 0`); or #mi(`\hat\mu \notin (0,1)`);.

#strong[MLE refinement] (starting from #mi(`\hat{a}_0,\hat{b}_0`);): maximize the Beta log-likelihood over
#mi(`\{p_{\min,g}^{(b)}\}`); with `scipy.optimize.minimize(method='Nelder-Mead')`. The code falls back to
#mi(`(\hat{a}_0,\hat{b}_0)`); when the fit does not converge. MLE is more accurate in the tail than MoM
alone, which is what makes a small #mi(`B`); sufficient.

#strong[Beta-adjusted per-gene p-value], applied to the real (unpermuted) minimum nominal p-value
#mi(`p_{\min,g}^{\text{obs}}`);:

#mitex(`p_{\text{adj},g} = F_{\mathrm{Beta}(\hat{a},\hat{b})}\!\left(p_{\min,g}^{\text{obs}}\right)`)

#mi(`p_{\text{adj},g}`); is calibrated #mi(`\mathrm{Uniform}(0,1)`); under #mi(`H_0`); #strong[for that gene specifically], because
it was built from that gene\'s own #mi(`\hat{h}^2_g`); null. This value populates `pval_perm`
in `EGENE_SUMMARY_COLS`. `run_call_egenes.py` also writes it to `{tag}_pval_perm.json` for every tested gene.

#divider()

== 4. Storey q-value across genes
<4-storey-q-value-across-genes>
With #mi(`\{p_{\text{adj},g}\}_{g=1}^G`); now genuinely #mi(`\mathrm{Uniform}(0,1)`); under each gene\'s own null, Storey\'s
construction (Storey & Tibshirani 2003, PNAS 10.1073/pnas.1530509100; Storey 2002, JRSS-B
10.1111/1467-9868.00346 --- both already in `docs/REFERENCES.md`) applies validly. The q-value step
(`fdr.qvalue`) operates purely on the p-value vector, with no engine dependence at all. `NaN`
p-values are skipped, and #mi(`G`); is the number of finite p-values. #mi(`G`); covers #strong[all tested genes]:
`call_significant_egenes` passes every gene with a non-missing `gene_id`, `chr` and `peak_lod`.

#strong[Step 1 --- estimate #mi(`\hat\pi_0`);] (proportion of true nulls) from the flat right tail of the
p-value histogram, over the grid #mi(`\lambda \in \{0.05, 0.10, \dots, 0.95\}`);:

#mitex(`\hat\pi_0(\lambda) = \frac{\#\{p_{\text{adj},g} > \lambda\}}{G(1-\lambda)}`)

Fit a cubic smoothing spline with 3 effective degrees of freedom to #mi(`\hat\pi_0(\lambda)`); across
#mi(`\lambda`);. The penalty is found by `brentq` so that the smoother-matrix trace equals 3
(`_smooth_df3_at_max`). Evaluate the spline at #mi(`\max\lambda = 0.95`); for the final #mi(`\hat\pi_0`);. This
needs at least 5 finite, positive #mi(`\hat\pi_0(\lambda)`);. Otherwise, or if the fit fails, the code
uses #mi(`\min_\lambda \hat\pi_0(\lambda)`);. Known issue: this fallback is the most anti-conservative grid
point (ledger FD-01). The result is clipped to #mi(`[10^{-6}, 1]`);, and a warning is logged when the raw
value exceeds 1. The #mi(`\hat\pi_0 = 2.58`); failure mode from the mis-calibrated single-gene null (§0)
should not recur once §1--3\'s per-gene calibration is in place.

#strong[Step 2 --- q-values], from p-values sorted ascending #mi(`p_{(1)} \le \cdots \le p_{(G)}`);:

#mitex(`\hat{q}(p_{(G)}) = \min\!\left(\hat\pi_0 \, p_{(G)},\, 1\right), \qquad
  \hat{q}(p_{(i)}) = \min\!\left(\hat\pi_0 \frac{G \, p_{(i)}}{i},\; 1,\; \hat{q}(p_{(i+1)})\right)
  \quad \text{for } i = G-1,\dots,1`)

the monotone (step-down) minimum enforces #mi(`\hat{q}(p_{(i)})`); non-decreasing in #mi(`i`);. An eGene is
called at #mi(`\hat{q}(p_{\text{adj},g}) \le 0.05`); when the call rule is `qvalue` (`fdr_threshold`, default 0.05;
`SWE-09`\'s BIO-A5 gate target, unchanged). The call rule is set by `call_rule` (`--call-rule`, Nextflow
`params.call_rule`); the selected adjusted value is written to `fdr`, with `fdr_method` equal to `'qvalue'` or `'BH'`
and `significance = 'T'` iff #mi(`\text{fdr} \le`); `fdr_threshold`. Both adjusted values are always written, as
`qval_storey` and `qval_bh`, together with #mi(`\hat\pi_0`); as `pi0`.

#strong[Versus BH-FDR.] The Benjamini--Hochberg adjusted value is the #mi(`\hat\pi_0 = 1`); case of the same
formula,

#mitex(`q^{\mathrm{BH}}_{(i)} = \min_{j \ge i}\ \min\!\left(1,\ \frac{G\,p_{(j)}}{j}\right),
  \qquad q^{\mathrm{Storey}}_{(i)} = \min\!\left(1,\ \hat\pi_0\, q^{\mathrm{BH}}_{(i)}\right)`)

so, with #mi(`\hat\pi_0 \le 1`);, the Storey q-value is never larger than the BH value: BH is the stricter rule
and Storey recovers power only through #mi(`\hat\pi_0 < 1`);. BH controls FDR for independent or positively
dependent tests (Benjamini and Yekutieli 2001). The Storey bound requires weak dependence and
#mi(`\mathrm{Uniform}(0,1)`);-under-#mi(`H_0`); p-values (Storey, Taylor and Siegmund 2004), which is what §1--3
provide for LMM-scanned genes; a mis-specified null distorts #mi(`\hat\pi_0`); (Efron 2004).

#strong[Saturation.] The largest q-value is #mi(`\hat q(p_{(G)}) = \hat\pi_0\, p_{(G)} \le \hat\pi_0`);, so if
#mi(`\hat\pi_0 \le`); `fdr_threshold` every tested gene has #mi(`\hat q \le`); `fdr_threshold` whatever its
p-value, and `significance` then describes the table and not the gene. `qvalue` logs a warning when
#mi(`\hat\pi_0 < 0.2`);. In the ctxhipp cell-type tables the share of tested genes with #mi(`\hat q \le 0.05`); was
100% for glutamatergic neurons (72% by BH) and 91% for astrocytes (50% by BH). Whether a small #mi(`\hat\pi_0`);
is real signal or a miscalibrated null is decided by checking the null p-value distribution against simulation,
not by the q-value itself. Estimator edge case: when every p-value is below every #mi(`\lambda`); on the grid,
all #mi(`\hat\pi_0(\lambda)`); are zero and the code returns #mi(`\hat\pi_0 = 1`); instead of the floor.

#strong[Across cell types and tissues.] Each cell-type table is a separate family, so the call above controls
FDR within a table and not over the study. Over families selected on the data, the error rate must be controlled
on the selected families (Benjamini and Bogomolov 2014): with #mi(`R`); of #mi(`F`); families selected, each is
tested at level #mi(`q\,R/F`);. The same gene recurs across cell types of the same strains, so tests are
positively dependent, which is the setting of the BH guarantee but is an assumption of the family-level
adjustment, to be checked by simulation. TreeQTL (Peterson et al. 2016) is the hierarchical design for a gene
level above a condition level; its released form is single-tissue, so a tissue level is an extension.
#emph[Documented, not implemented.] Sharing of effect estimates across cell types is a separate problem
(Urbut et al. 2019) and does not control the error rate. All citations are in `docs/REFERENCES.md`.

#divider()

== 5. Two-stage design: what happens to the genome-wide GEV threshold
<5-two-stage-design-what-happens-to-the-genome-wide-gev-threshold>
#strong[Current pipeline (since 2026-11-11):] there is no Stage-1 filter. `run_call_egenes.py` →
`call_significant_egenes` calibrates #strong[every] tested gene and estimates the q-value over all of
them. Selecting genes by peak LOD first would concentrate the surviving null p-values near zero
and bias #mi(`\hat\pi_0`); downward (`docs/REFERENCES.md`, Bourgon et al. 2010). The GEV threshold
(`06_permutation.typ` §4--5, `run_threshold_scans` #mi(`\to`); `get_gev_thresholds`) is computed and logged as a
QC reference only. It is also the source of `gev_mu`/`gev_sigma`/`gev_xi`/`lod_threshold` in
`EQTL_STATS_COLS`. `call_egenes()` remains in `pipeline.py` and is used only by the CLI.

#strong[History --- the two-stage design below] was the original plan. The CLI `call-egenes`
(`cli.py::call_egenes_cmd`) still implements it. Known issue: its q-values are computed on the
LOD-selected subset and do not control FDR (ledger ORC-02).

Per-gene Beta-calibration (§1--3) is what makes Storey valid, but running it for every gene in a
#mi(`10^4`);--#mi(`10^5`);-gene scan is still expensive relative to a single global permutation run. In
that design the genome-wide/per-chromosome GEV threshold is #strong[not discarded] --- it is
repurposed as an adaptive two-stage filter, matching FastQTL\'s own adaptive-permutation strategy:

#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Stage], [Mechanism], [Role],),
    table.hline(),
    [1 --- screen], [Genome-wide/per-chromosome GEV threshold (existing, cheap: one global permutation run)], [Coarse pre-filter --- only genes with `peak_lod` clearing the chromosome-level GEV threshold are promoted to Stage 2. Also remains the Manhattan-plot significance line and keeps `gev_mu`/`gev_sigma`/`gev_xi`/`lod_threshold` in `EQTL_STATS_COLS` meaningful.],
    [2 --- calibrate], [Per-gene Beta-approximated permutation + Storey q-value (§1--4, new)], [Determines actual eGene calling: `significance = 'T'` iff #mi(`\hat{q} \le 0.05`);, not the raw Stage-1 LOD cutoff.],
  )]
  , kind: table
  )

In that plan, `call_egenes(threshold=...)`\'s raw scalar-threshold comparison stops being the final
eGene-calling criterion --- its output becomes the Stage-1 promotion set. The aim was to keep
`SPEED-01`\'s \<4h budget (Stage 2 only on the minority of genes clearing Stage 1). The
current pipeline drops this saving to keep the q-value valid.

#divider()

== 6. Summary of what changes downstream
<6-summary-of-what-changes-downstream>
- `qtl/fdr.py::calibrate_gene_pvalue`: the per-gene entry point. It reuses
  `permutation.draw_parametric_null_batch` and `lmm.scan_chromosome_batch` directly, not
  `run_threshold_scans`. It is LMM-only and returns #mi(`(p_{\text{adj},g}, \hat{a}, \hat{b}, \hat{h}^2_g)`);.
  `beta_calibrate_pvalue` returns #mi(`(p_{\text{adj},g}, \hat{a}, \hat{b})`);.
- `pipeline.py::_SIG_COLS`: `pval_perm` populated from §3\'s output; `fdr` from §4\'s q-value;
  `fdr_method` is `'qvalue'` (no longer hardcoded `'BH'`); `significance` gated on #mi(`\hat q \le`); `fdr_threshold`
  instead of the raw `peak_lod` threshold (`calibrate_egenes_storey`).
- `pipeline.py::call_significant_egenes`: calibrates all tested genes, applies Storey, then computes
  founder effects for significant genes only. `run_call_egenes.py` writes `{tag}_peak_lod.csv` (with
  `is_egene`), `{tag}_pval_perm.json` and `{tag}_eqtl_summary.csv` (significant eGenes only).
- `call_egenes()`: the Stage-1 promotion gate of §5, used only by the CLI `call-egenes`.
- Validation: `BIO-A5` (observed FDR #mi(`\le 0.05`); on `sim_cc48_n200qtl`) must use this Storey path,
  not BH --- per `GOALS.md` `SWE-09`. NB-GLMM-scanned genes would flow through the identical §3--4
  machinery once a per-gene NB null exists (§2).

History: this document was written under the `DESIGN-03`/`DESIGN-05` order, before the code change
to `pipeline.py` and `qtl/fdr.py`. The bullets above now describe the implemented state.

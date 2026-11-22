#import "_setup.typ": *
#show: mathdoc

= Bootstrap Instability Scan
<bootstrap-instability-scan>
#strong[Module:] `qtl/instability.py::instability_lm_scan` (multi-gene wrapper:
`instability_scan_multi_gene`; CLI: `mosaiqtl assess-stability`) \ #strong[miQTL parallel:] `instability.lm.scan(n.samples, subsample.prop=0.63)`

#divider()

== 1. Purpose
<1-purpose>
The instability scan asks: #emph[how reproducible is the LOD profile across
subsamples of the data?] A locus with a genuine QTL signal produces a
consistently high LOD regardless of which strains are included. An artifactual
peak driven by one or two outlier strains collapses when those strains are
removed.

The output is a mean-LOD profile across bootstrap subsamples, plus how often each locus was
the subsample peak. The engine `ScanResult` stores the frequency as `p_value` #mi(`= 1 - \text{peak frequency}`);. High mean-LOD
at a locus indicates a stable, strain-robust signal. The instability scan is
a QC layer after the main scan, not a primary significance test, and is not part of the
Nextflow pipeline. Known issue: the `p_value` field is not a p-value (ledger INS-04).

#divider()

== 2. Algorithm
<2-algorithm>
#strong[Input:] phenotype #mi(`y`);, genome cache, covariates #mi(`X`);, #mi(`n_{\text{subsamples}}`); subsamples
(`n_subsamples`, default 100), `subsample_prop` #mi(`p_{\text{sub}} = 0.63`);.

+ #strong[Reduce:] when `sample_ids` is given, average #mi(`y`); and #mi(`X`); to strain means once, up
  front. Load the haplotype probabilities of every requested locus once into one
  #mi(`(L, m, J)`); array.

+ #strong[Subsample:] draw #mi(`k = \max(\lfloor m \cdot p_{\text{sub}} \rfloor, 2)`); strains without
  replacement from #mi(`\{1,\ldots,m\}`); where #mi(`p_{\text{sub}} = 0.63`); (the 63.2%
  bootstrap inclusion rate from the #mi(`1 - e^{-1}`); limit of sampling with replacement).
  Index the preloaded array at those strains. A `_SubCache` view class is still defined
  but never used (ledger INS-06).

+ #strong[Scan:] apply RINT to the subsample (`use_rint=True`), then run a vectorised fixed-effect
  OLS scan with no kinship. Partial #mi(`X`); out of #mi(`y`); and of the haplotype block #mi(`H_\ell`); and set
  #mi(`\mathrm{RSS}_1 = \mathrm{RSS}_0 - c_\ell^\top G_\ell^{+} c_\ell`);, with #mi(`G_\ell = H_{\ell\perp}^\top H_{\ell\perp}`);, #mi(`c_\ell = H_{\ell\perp}^\top e_0`);
  and #mi(`\text{LOD}_\ell = (k/2)\log_{10}(\mathrm{RSS}_0/\mathrm{RSS}_1)`);. This is the same model as
  `fixef_eqtl_scan`, which is not called here. A fixed-effect OLS scan (not LMM) is used because:

  - LOCO kinship re-computation per subsample would be prohibitively expensive.
  - The instability scan is a screening tool; conservative OLS LOD suffices
    to detect whether the signal persists.
  - miQTL\'s `instability.lm.scan` also uses the fixed-effect (QR-based) scan
    for the same reason.

+ #strong[Accumulate:] add the LOD profile for this subsample to a running sum, and add one
  to the count of the locus with the highest LOD.

+ #strong[Repeat] steps 2--4 for #mi(`n_{\text{subsamples}}`); subsamples.

+ #strong[Mean LOD:] divide the accumulated sum by #mi(`n_{\text{subsamples}}`);. The peak frequency is
  count #mi(`/\,n_{\text{subsamples}}`);. The returned founder effects come from the first subsample
  only (ledger INS-03).

#divider()

== 3. Why 63%?
<3-why-63>
The #mi(`1 - 1/e \approx 63.2\%`); inclusion rate is the expected fraction of
unique observations in a bootstrap sample (sampling with replacement from #mi(`m`);).
Subsampling at exactly this rate mimics one bootstrap draw\'s unique sample set,
making the instability scan\'s null distribution comparable to standard bootstrap
analyses. miQTL uses `subsample.prop = 0.63` as the default; mosaiQTL matches.

#divider()

== 4. Interpretation
<4-interpretation>
#figure(
  align(center)[#table(
    columns: 2,
    align: (auto,auto,),
    table.header([Mean LOD pattern], [Interpretation],),
    table.hline(),
    [High and narrow at peak chromosome], [Stable QTL --- signal robust to strain removal],
    [Broad elevated region], [Possible QTL block; imprecise localization],
    [High at one locus, collapses in most subsamples], [Likely outlier-driven; treat peak with caution],
    [Low across genome], [Weak or absent signal regardless of strains included],
  )]
  , kind: table
  )

The instability LOD profile is plotted alongside the main scan LOD profile
via `viz/genome.py::instability_lod_overlay`. A stable QTL has mean-LOD profile that peaks near the
main scan peak with roughly #mi(`63\% \times \text{LOD}_{\text{main}}`); magnitude
(since only 63% of strains are used per subsample).

#divider()

== 5. Scope note
<5-scope-note>
The instability scan is implemented for the fixed-effect OLS scan only. There is no LMM or
kinship variant. For NB-GLMM,
a count-level instability scan would subsample strains from the pseudobulk
count matrix and re-run the NB score scan --- this path is not yet implemented
but would follow the same accumulate-and-average logic.

`instability_scan_multi_gene` restricts each gene to its own chromosome. With `peak_mb_map`,
it would restrict each gene to loci within #mi(`\pm`); `window_mb` (default 10 Mb) of the peak. It
writes the peak-locus row (`peak_mb`, `peak_lod`) or all loci (`lod`), but not the peak
frequency. Known issue: with `peak_mb_map` the window filter unpacks `get_chromosome` in the wrong order. Every gene
that has a peak position then fails, the failure is logged, and the gene is skipped. `assess-stability`
builds this map whenever the eGene table has `peak_mb` (ledger INS-01).

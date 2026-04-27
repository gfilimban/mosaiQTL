# Bootstrap Instability Scan

**Module:** `qtl/scan.py::instability_lm_scan`  
**miQTL parallel:** `instability.lm.scan(n.samples, subsample.prop=0.63)`

---

## 1. Purpose

The instability scan asks: *how reproducible is the LOD profile across
subsamples of the data?* A locus with a genuine QTL signal produces a
consistently high LOD regardless of which strains are included. An artifactual
peak driven by one or two outlier strains collapses when those strains are
removed.

The output is a mean-LOD profile across bootstrap subsamples. High mean-LOD
at a locus indicates a stable, strain-robust signal. The instability scan is
used as a QC layer after the main scan, not as a primary significance test.

---

## 2. Algorithm

**Input:** phenotype $y$, genome cache, covariates $X$, $n_{\text{samples}}$ subsamples.

1. **Subsample:** draw $\lfloor m \cdot p_{\text{sub}} \rfloor$ strains without
   replacement from $\{1,\ldots,m\}$ where $p_{\text{sub}} = 0.63$ (the 63.2%
   bootstrap inclusion rate from the $1 - e^{-1}$ limit of sampling with replacement).

2. **Build subcache:** construct a `_SubCache` object that wraps the full
   `GenomeCache` but returns only the selected rows from each chromosome's
   `probs` array. This avoids copying the full haplotype tensor.

3. **Scan:** run `fixef_eqtl_scan` (OLS, no kinship) on the subsampled strains.
   A fixed-effect OLS scan (not LMM) is used because:
   - LOCO kinship re-computation per subsample would be prohibitively expensive.
   - The instability scan is a screening tool; conservative OLS LOD suffices
     to detect whether the signal persists.
   - miQTL's `instability.lm.scan` also uses the fixed-effect (QR-based) scan
     for the same reason.

4. **Accumulate:** add the LOD profile for this subsample to a running sum.

5. **Repeat** steps 1–4 for $n_{\text{samples}}$ subsamples.

6. **Mean LOD:** divide the accumulated sum by $n_{\text{samples}}$.

---

## 3. Why 63%?

The $1 - 1/e \approx 63.2\%$ inclusion rate is the expected fraction of
unique observations in a bootstrap sample (sampling with replacement from $m$).
Subsampling at exactly this rate mimics one bootstrap draw's unique sample set,
making the instability scan's null distribution comparable to standard bootstrap
analyses. miQTL uses `subsample.prop = 0.63` as the default; mosaiQTL matches.

---

## 4. Interpretation

| Mean LOD pattern | Interpretation |
|---|---|
| High and narrow at peak chromosome | Stable QTL — signal robust to strain removal |
| Broad elevated region | Possible QTL block; imprecise localization |
| High at one locus, collapses in most subsamples | Likely outlier-driven; treat peak with caution |
| Low across genome | Weak or absent signal regardless of strains included |

The instability LOD profile is plotted alongside the main scan LOD profile
via `viz/genome.py`. A stable QTL has mean-LOD profile that peaks near the
main scan peak with roughly $63\% \times \text{LOD}_{\text{main}}$ magnitude
(since only 63% of strains are used per subsample).

---

## 5. Scope note

The instability scan is implemented for LMM / fixef OLS only. For NB-GLMM,
a count-level instability scan would subsample strains from the pseudobulk
count matrix and re-run the NB score scan — this path is not yet implemented
but would follow the same accumulate-and-average logic.

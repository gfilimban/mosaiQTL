# eQTL summary table

One row per eGene, meaning one gene whose gene-level significance passes the FDR threshold (`significance` = `T`).
The table joins the haplotype peak of each eGene to its calibrated p-value and its false discovery rate (FDR)
estimates, so it is the file to read for the list of significant genes. Every gene is tested and calibrated before
this selection, so the q-values and $\hat\pi_0$ reflect all tested genes. See the [output schema](../output_schema.md)
for the full conventions.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `gene_id` | str | – | Unversioned gene ID; join key |
| `gene_symbol` | str | – | Display symbol; not unique |
| `chr` | str | – | Chromosome, written as `chr{N}` |
| `start_mb` | float | Mb | Strand-aware TSS of the gene; anchor of the scan window |
| `near_tss_mb` | float | Mb | Strand-aware transcript TSS nearest the peak |
| `peak_locus` | str | – | Marker ID of the highest-LOD locus |
| `peak_mb` | float | Mb | Position of the highest-LOD locus |
| `cis_dist_mb` | float | Mb | `peak_mb` − `near_tss_mb` (or − `start_mb` when no transcript TSS is available); signed by the gene's strand: negative is upstream and positive downstream of the TSS on both strands |
| `peak_lod` | float | LOD | LOD at the peak locus against the covariate-only null |
| `h2` | float | [0, 1] | Null heritability fitted for this gene on its LOCO kinship |
| `perm_n` | int | – | Null draws per gene behind `pval_perm`; the smallest attainable p-value is about 1/`perm_n` |
| `chi2_haplotype` | float | $\chi^2$ | $2\ln(10)\,\mathrm{peak\_lod}$ |
| `p_haplotype` | float | p | Upper tail of `chi2_haplotype` on (founders − 1) degrees of freedom; nominal, not multiplicity-adjusted |
| `pval_perm` | float | p | Per-gene p-value of the window's peak LOD, calibrated against null draws simulated at the gene's own $h^2$ over its own window |
| `qval_storey` | float | q | Storey q-value across all tested genes |
| `qval_bh` | float | q | Benjamini–Hochberg adjusted `pval_perm` across all tested genes |
| `lfdr` | float | [0, 1] | Local false discovery rate: posterior probability that the gene is null; NaN when fewer than 10 genes were tested |
| `pi0` | float | [0, 1] | Estimated null fraction used by `qval_storey` and `lfdr`; one value per table |
| `fdr` | float | [0, 1] | The value selected by the call rule (`qval_storey`, `qval_bh` or `lfdr`) |
| `fdr_method` | str | – | Call rule used: `qvalue`, `BH` or `lfdr` |
| `significance` | str | – | `T` when `fdr` is at or below the FDR threshold, otherwise `F` |
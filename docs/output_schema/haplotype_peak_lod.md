# Haplotype peak LOD table

One row per tested gene. For each gene, the table reports the strongest signal of the haplotype scan within the
gene's cis window (the gene's strand-aware TSS ±1 Mb for the default `Lw1` window), where the window is the same for
every gene. The LOD compares a model with the haplotype effects at each locus against a model with covariates only.
Significance is not decided here; the calibration and FDR columns are added later (see the
[output schema](../output_schema.md)), and they also add the per-founder effects `beta_{founder}` and the
`is_egene` flag.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `gene_id` | str | – | Unversioned gene ID; join key |
| `gene_symbol` | str | – | Display symbol; not unique |
| `chr` | str | – | Chromosome, written as `chr{N}` |
| `start_mb` | float | Mb | Strand-aware TSS of the gene (`start` on +, `end` on −); anchor of the scan window |
| `near_tss_mb` | float | Mb | Strand-aware transcript TSS nearest the peak |
| `peak_locus` | str | – | Marker ID of the highest-LOD locus |
| `peak_mb` | float | Mb | Position of the highest-LOD locus |
| `cis_dist_mb` | float | Mb | `peak_mb` − `near_tss_mb` (or − `start_mb` when no transcript TSS is available); signed by the gene's strand: negative is upstream and positive downstream of the TSS on both strands |
| `peak_lod` | float | LOD | LOD at the peak locus, $\mathrm{LOD}=\tfrac{n}{2}\log_{10}(\mathrm{RSS}_0/\mathrm{RSS}_1)$ |
| `peak_at_edge` | bool | – | `True` when the maximum is the first or last locus of the window; the true maximum may lie outside the window, so the peak position and support interval are truncated |
| `h2` | float | [0, 1] | Null heritability fitted for this gene on its leave-one-chromosome-out (LOCO) kinship |
| `n_valid` | int | – | Strains with a finite phenotype |
| `rss0` | float | – | Residual sum of squares of the covariate-only null model |
| `rss1` | float | – | Residual sum of squares of the haplotype model at the peak locus |
| `r2_peak` | float | [0, 1] | Proportion of null residual sum of squares explained at the peak, $(\mathrm{RSS}_0-\mathrm{RSS}_1)/\mathrm{RSS}_0$ |
| `sigma2_e` | float | – | Residual variance, $\mathrm{RSS}_0(1-h^2)/(n-p)$, with $p$ the number of covariate columns |
| `tau2` | float | – | Polygenic variance, $\mathrm{RSS}_0\,h^2/(n-p)$ |

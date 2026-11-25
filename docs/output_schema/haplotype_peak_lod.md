# Haplotype peak LOD table

`{tissue}_{celltype}_{scan_tag}_haplotype_peak_lod.csv`, written by `CALL_EGENES` (`run_call_egenes.py`).

One row per gene whose Benjamini-Hochberg q-value (`qval_bh`) is at or below `params.submission_fdr` (default 0.1).
This is a filtered copy of `{tag}_peak_lod_all.csv`, which holds every tested gene. Calibration and the q-values are
computed over all tested genes before the filter, so `pi0`, `qval_*` and `fdr` do not depend on it. The file
therefore holds the called eGenes (`is_egene` = `True`: `fdr` ≤ `params.fdr_threshold` and `peak_lod` above the
permutation LOD threshold) and the genes that fall short of either condition (`is_egene` = `False`). The called eGenes, with support intervals, are in the
[eQTL summary table](eqtl_summary.md).

For each gene the row reports the strongest signal of the haplotype scan within the gene's cis window: the
strand-aware TSS ±1 Mb for the default `Lw1` window, the same width for every gene. The LOD compares a model with
the founder haplotype effects at each locus against a model with covariates only, both at the gene's null $h^2$.
See the [output schema](../output_schema.md) for the shared conventions.

## Scan block

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `gene_id` | str | – | Unversioned gene ID; join key |
| `gene_symbol` | str | – | Display symbol; not unique |
| `chr` | str | – | Chromosome, written as `chr{N}` |
| `start_mb` | float | Mb | Strand-aware gene TSS (`start` on +, `end` on −); anchor of the scan window |
| `near_tss_mb` | float | Mb | Strand-aware transcript TSS nearest the peak |
| `peak_locus` | str | – | Marker ID of the highest-LOD locus |
| `peak_mb` | float | Mb | Position of the highest-LOD locus |
| `cis_dist_mb` | float | Mb | `peak_mb` − `near_tss_mb` (− `start_mb` when no transcript TSS is available), signed by strand: negative is upstream and positive downstream of the TSS on both strands; within ±`window_mb` |
| `cis_dist_gene_tss_mb` | float | Mb | `peak_mb` minus the gene's most upstream transcript TSS (fixed per gene, same sign convention); NaN without a `gene_tss` entry |
| `peak_lod` | float | LOD | LOD at the peak locus, $\mathrm{LOD}=\tfrac{n}{2}\log_{10}(\mathrm{RSS}_0/\mathrm{RSS}_1)$ with $n$ = `n_valid` |
| `peak_at_edge` | bool | – | `True` when the maximum is the first or last locus of the window; the true maximum may lie outside it, so the peak position and support interval are truncated |
| `h2` | float | [10⁻⁶, 1 − 10⁻⁶] | Null heritability, fitted by maximum likelihood (ML) on the gene's leave-one-chromosome-out (LOCO) kinship; fixed for both models |
| `h2_at_boundary` | bool | – | `True` when `h2` sits on an end of the search grid ($h^2 \le 10^{-5}$, or within $10^{-4}$ of the top grid point), so the estimate is censored there |
| `n_valid` | int | samples | Pseudobulk samples in the scan (samples whose strain is in the genome cache); constant within a cell type |
| `rss0` | float | – | Residual sum of squares of the covariate-only model, in the rotated and $h^2$-weighted (generalized least squares) space |
| `rss1` | float | – | Residual sum of squares of the haplotype model at the peak locus, same space |
| `r2_peak` | float | [0, 1] | Share of the null residual sum of squares explained at the peak, $(\mathrm{RSS}_0-\mathrm{RSS}_1)/\mathrm{RSS}_0$ |
| `sigma2_e` | float | – | Residual variance, $\mathrm{RSS}_0(1-h^2)/(n-p)$, with $p$ the covariate columns including the intercept |
| `tau2` | float | – | Polygenic variance, $\mathrm{RSS}_0\,h^2/(n-p)$ |
| `beta_{founder}` | float | phenotype units | Effect of each founder haplotype at the peak locus (`beta_AJ`, `beta_B6J`, `beta_129S1J`, `beta_NODJ`, `beta_NZOJ`, `beta_CASTJ`, `beta_PWKJ`, `beta_WSBJ`); minimum-norm solution after the covariates are projected out, so the eight values sum to 0 and each is a deviation from the founder mean. On the rank-inverse-normal scale when `use_rint` is on |

## Calibration and FDR block

Added by `CALL_EGENES`.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `is_egene` | bool | – | `True` when `fdr` ≤ `params.fdr_threshold`; these genes form the eQTL summary table |
| `pval_perm` | float | [0, 1] | Calibrated gene-level p-value. The nominal p of `peak_lod` (χ² with J − 1 degrees of freedom on 2 ln 10 · LOD) is read off a Beta distribution fitted to the gene's own parametric null, drawn at its ML $h^2$ over the same window |
| `beta_a` | float | – | First shape parameter of the fitted Beta null |
| `beta_b` | float | – | Second shape parameter of the fitted Beta null |
| `perm_n` | int | draws | Null draws per gene (`params.n_perms_cal`) |
| `qval_storey` | float | [0, 1] | Storey q-value over all tested genes |
| `qval_bh` | float | [0, 1] | Benjamini–Hochberg adjusted p-value over all tested genes |
| `lfdr` | float | [0, 1] | Local FDR; NaN when fewer than 10 genes have a calibrated p-value |
| `pi0` | float | [0, 1] | Estimated share of true nulls $\hat\pi_0$; one value per cell type |
| `fdr` | float | [0, 1] | The FDR used for calling: `qval_bh` (default), `qval_storey` or `lfdr`, chosen by `call_rule` |
| `fdr_method` | str | – | Name of the rule behind `fdr` (`qvalue`, `BH` or `lfdr`) |

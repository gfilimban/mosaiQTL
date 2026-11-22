# Fine-mapping peak table

One row per resolved eGene: an eGene whose lead variant has a posterior inclusion probability (`pip`) above 0.5, and
that row is the lead variant. Such a gene has a single variant that dominates its credible set. Every other eGene
stays in the per-variant and LD-group fine-mapping tables. Variant-level columns describe that lead variant; the
gene-level significance columns repeat the values from the [eQTL summary](eqtl_summary.md); the `effect_{founder}`
columns are the per-founder haplotype effects at the haplotype peak, one column per founder. The
[output schema](../output_schema.md) gives the full conventions.

| Column | Type | Unit | Meaning |
|---|---|---|---|
| `variant` | str | – | Variant ID (SPDI) |
| `rsid` | str | – | dbSNP rsID; empty if absent |
| `chr` | str | – | Chromosome, written as `chr{N}` |
| `pos` | int | bp | Variant position |
| `ref` | str | – | Reference allele |
| `alt` | str | – | Alternative allele |
| `gene_id` | str | – | Unversioned gene ID; join key |
| `gene_symbol` | str | – | Display symbol; not unique |
| `cis_dist_mb` | float | Mb | Distance of the gene's peak from its TSS, as defined for the [eQTL summary](eqtl_summary.md) |
| `pip` | float | [0, 1] | Posterior inclusion probability of the variant |
| `log_abf` | float | natural log | Natural log of the approximate Bayes factor, association over null; positive favours association; NaN when no valid Wald statistic |
| `CS` | str | – | `T` for credible-set members |
| `pip_method` | str | – | Fine-mapping method (`ABF`) |
| `lod` | float | LOD | LOD of the variant |
| `pval_nominal` | float | p | One-degree-of-freedom $\chi^2$ p-value of the variant LOD |
| `log10p_nominal` | float | – | $-\log_{10}$ `pval_nominal` |
| `pval_perm` | float | p | Per-gene p-value of the window's peak LOD |
| `qval_storey` | float | q | Storey q-value across all tested genes |
| `qval_bh` | float | q | Benjamini–Hochberg adjusted `pval_perm` across all tested genes |
| `lfdr` | float | [0, 1] | Local false discovery rate of the gene; NaN when fewer than 10 genes were tested |
| `pi0` | float | [0, 1] | Estimated null fraction used by `qval_storey` and `lfdr` |
| `fdr` | float | [0, 1] | The value selected by the call rule |
| `fdr_method` | str | – | Call rule used: `qvalue`, `BH` or `lfdr` |
| `significance` | str | – | `T` when `fdr` is at or below the FDR threshold, otherwise `F` |
| `effect_allele` | str | – | Allele coded as the dosage predictor |
| `other_allele` | str | – | Complement of `effect_allele` |
| `ref_founders` | str | – | Comma-joined founders carrying the reference allele |
| `alt_founders` | str | – | Comma-joined founders carrying the alternative allele |
| `dosage` | bool | – | `T` if the genotype is imputed from haplotype probabilities, `F` for hard calls |
| `af` | float | [0, 1] | Mean imputed dosage of the effect allele over the gene's scanned strains; replicates are counted individually |
| `maf` | float | [0, 1] | Minor allele frequency, $\min(\mathrm{af},1-\mathrm{af})$ |
| `mac` | int | – | Minor allele count, $\mathrm{round}(\mathrm{maf}\times n_{\mathrm{sample}})$ |
| `ma_sample` | int | – | Same value as `mac` |
| `effect_size` | float | phenotype units | Dosage regression slope |
| `se_effect_size` | float | phenotype units | Standard error of `effect_size` |
| `hap_peak_variant` | str | – | Marker ID of the haplotype-scan peak the interval was built around |
| `hap_peak_pos` | int | bp | Position of that haplotype-scan peak |
| `hap_peak_lod` | float | LOD | LOD of that haplotype-scan peak |
| `ci_lo_bp` | int | bp | Lower bound of the fine-mapping region: contiguous loci within 1.5 LOD of the peak, clamped to the scan window |
| `ci_hi_bp` | int | bp | Upper bound of the fine-mapping region, as for `ci_lo_bp` |
| `h2_null_chr` | float | [0, 1] | Leave-one-chromosome-out null heritability on the gene's chromosome |
| `sd_phenotype` | float | phenotype units | $\sqrt{\mathrm{RSS}_0/(n-1)}$ |
| `n_sample` | int | – | Strains in the null fit |
| `effect_AJ`, `effect_B6J`, `effect_129S1J`, `effect_NODJ`, `effect_NZOJ`, `effect_CASTJ`, `effect_PWKJ`, `effect_WSBJ` | float | phenotype units | Per-founder haplotype effect at the haplotype peak, one column per founder (the eight CC founders shown); sums to about 0 |

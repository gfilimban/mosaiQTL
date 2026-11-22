# Output schema

What each Nextflow output file contains and what every column means. The column lists live in code
(`mosaiqtl.pipeline`); `tests/unit/test_output_schema.py` fails when a written column has no row in the
[column reference](#column-reference).

## Conventions

- `gene_id` is the unversioned gene ID (e.g. Ensembl ENSMUSG…) and the only join key; `gene_symbol` is display-only.
- `chr` is a `chr{N}` string in every written file.
- `pos`, `*_bp` are integer bp; `*_mb` are Mb. All coordinates are on the reference assembly of the genotype cache.
- Scan windows are identical across scans: ±`window_mb` of the strand-aware TSS (e.g. `Lw1` = ±1 Mb).
- Files are in `{output_dir}/{celltype}/{scan_tag}/core/`; `{tag}` is `{tissue}_{celltype}_{scan_tag}`
  (tissue omitted when empty).
- Every file is written atomically: complete or absent.
- Significance is estimated over all tested genes; no gene is selected by LOD first.
- "Founder" columns have one entry per founder of the population (e.g. the eight CC/DO founders AJ, B6J, …),
  in the founder order of the genotype cache.

## Files

| File | One row per | Columns | Written by |
|---|---|---|---|
| `{tag}_lod_genome.csv` | gene × locus in the window | `LOD_COLS` | `run_merge.py` |
| `{tag}_peak_lod.csv` | tested gene | `PEAK_LOD_COLS`, `beta_{founder}`; after calling also `is_egene` and the calibration block (`pval_perm`, `beta_a`, `beta_b`, `perm_n`, `qval_storey`, `qval_bh`, `pi0`, `fdr`, `fdr_method`) | `run_merge.py`, then `run_call_egenes.py` |
| `{tag}_scan_stats.csv` | chromosome | `chr`, `n_genes`, `elapsed_s` | `run_merge.py` |
| `{tag}_pval_perm.json` | tested gene | `gene_id` → `pval_perm` | `run_call_egenes.py` |
| `{tag}_eqtl_summary.csv` | eGene (`significance == 'T'`) | `EGENE_SUMMARY_COLS` | `run_call_egenes.py` |
| `{tag}_eqtl_ci.csv` | eGene | `CI_COLS` | `run_finemap.py`, merged by `run_finemap_merge.py` |
| `{tag}_eqtl_finemapping.csv` | credible-set member × eGene | `EQTL_FINEMAPPING_COLS` | `run_finemap.py`, merged by `run_finemap_merge.py` |
| `{tag}_eqtl_finemapping_peak.csv` | eGene whose lead variant has `pip` > `PEAK_PIP_THRESHOLD` (0.5), as its lead variant | `EQTL_FINEMAPPING_PEAK_COLS` | `run_finemap.py`, merged by `run_finemap_merge.py` |
| `{tag}_eqtl_finemapping_groups.csv` | LD group × eGene (all groups, including those outside the credible set) | `EQTL_FINEMAPPING_GROUP_COLS` | `run_finemap.py`, merged by `run_finemap_merge.py` |

`in_r_egenes` is in `EGENE_SUMMARY_COLS` but is filled only by validation scripts.

## Column reference

Origin: **ci** = support-interval scan; **scan** = LMM scan; **annot** = gene annotation join; **call** = calibration and FDR; **fm** = fine-mapping.

| Column | Meaning | Unit / type | Origin |
|---|---|---|---|
| `gene_id` | Unversioned gene ID; join key | str | scan |
| `gene_symbol` | Display symbol; not unique | str | annot |
| `chr` | Chromosome | `chr{N}` | scan |
| `locus` | Marker ID of a scanned locus | str | scan |
| `pos_mb` | Locus position | Mb | scan |
| `lod` | Haplotype-model LOD against the covariate-only null, $\mathrm{LOD}=\tfrac n2\log_{10}(\mathrm{RSS}_0/\mathrm{RSS}_1)$; in the fine-mapping files, the LOD of the variant | LOD | scan, fm |
| `start_mb` | Strand-aware TSS of the gene (`start` on +, `end` on −); anchor of the scan window | Mb | annot |
| `near_tss_mb` | Strand-aware transcript TSS nearest the peak | Mb | annot |
| `peak_locus` | Marker ID of the highest-LOD locus | str | scan |
| `peak_mb` | Position of the highest-LOD locus | Mb | scan |
| `cis_dist_mb` | `peak_mb − near_tss_mb` (else `peak_mb − start_mb`). Signed by strand: negative is upstream and positive downstream of the TSS on both strands. In the fine-mapping files, the same quantity for the gene | Mb | annot |
| `peak_lod` | LOD at the peak locus | LOD | scan |
| `peak_at_edge` | The maximum is the first or last locus of the scan window, so the true maximum may lie outside it and the peak position, support interval and fine-mapping window are truncated. The haplotype LOD profile is smooth over megabases, so a short window often puts its maximum at an end. Written for local windows; `pval_perm` is unaffected because its null is scanned over the same window | bool | scan |
| `h2` | Null heritability fitted for this gene on its LOCO kinship | [0, 1] | scan |
| `h2_at_boundary` | True when the ML $h^2$ sits at either end of the search grid, so the estimate is censored | bool | scan |
| `n_valid` | Samples with a finite phenotype | int | scan |
| `rss0` | Residual sum of squares of the covariate-only null | – | scan |
| `rss1` | Residual sum of squares of the haplotype model at the peak | – | scan |
| `r2_peak` | $(\mathrm{RSS}_0-\mathrm{RSS}_1)/\mathrm{RSS}_0$ at the peak | [0, 1] | scan |
| `sigma2_e` | Residual variance $\mathrm{RSS}_0(1-h^2)/(n-p)$, $p$ = covariate columns | – | scan |
| `tau2` | Polygenic variance $\mathrm{RSS}_0\,h^2/(n-p)$ | – | scan |
| `beta_{founder}` | Per-founder haplotype effect at the peak locus; sums to about 0 | phenotype units | scan |
| `is_egene` | `True` when `significance == 'T'` | bool | call |
| `chi2_haplotype` | $2\ln(10)\,\mathrm{peak\_lod}$ | $\chi^2$ | call |
| `p_haplotype` | Upper tail of `chi2_haplotype` on (founders − 1) df; nominal, not multiplicity-adjusted | p | call |
| `pval_perm` | Per-gene p-value of the window's peak LOD. Each null draw's peak LOD is converted to a pointwise $\chi^2$ upper-tail p on (founders − 1) df, with $\chi^2=2\ln(10)\,\mathrm{LOD}$; a Beta($a$, $b$) is fitted to those null p-values by maximum likelihood, and `pval_perm` is that Beta CDF at the observed peak's pointwise p. The null is drawn at the gene's own $h^2$ over its own window | p | call |
| `beta_a`, `beta_b` | Fitted Beta shape parameters behind `pval_perm`. $a\approx1$ means the window behaves as one independent test, so `pval_perm` tracks the pointwise p; larger $a$ is more multiplicity absorbed | > 0; NaN when calibration failed | call |
| `perm_n` | Null draws per gene behind `pval_perm`; the smallest attainable p is about 1/`perm_n`, and smaller values are extrapolations | int | call |
| `qval_storey` | Storey q-value across all tested genes: $\hat\pi_0$ × BH-adjusted `pval_perm` | q | call |
| `qval_bh` | Benjamini–Hochberg adjusted `pval_perm` across all tested genes | q | call |
| `lfdr` | Local false discovery rate: posterior probability the gene is null, $\hat\pi_0 f_0(p)/\hat f(p)$ from a probit-scale density of `pval_perm` over all tested genes; NaN when fewer than 10 genes were tested | [0, 1] | call |
| `pi0` | Estimated null fraction used by `qval_storey` and `lfdr`; one value per table | [0, 1] | call |
| `fdr` | The value selected by the call rule (`qval_storey`, `qval_bh` or `lfdr`) | [0, 1] | call |
| `fdr_method` | Call rule used: `qvalue`, `BH` or `lfdr` | str | call |
| `significance` | `T` when `fdr` ≤ the FDR threshold, else `F` | `T`/`F` | call |
| `in_r_egenes` | Gene also called in an external reference (validation only) | bool | – |
| `variant` | Variant ID (SPDI) | str | fm |
| `rsid` | dbSNP rsID; empty if absent | str | fm |
| `pos` | Variant position | bp | fm |
| `ref`, `alt` | Reference and alternative allele | str | fm |
| `pip` | Posterior inclusion probability of the variant: a probability, not a log value, so "PIP > 0.5" reads as usual | [0, 1] | fm |
| `log_abf` | Natural log of the approximate Bayes factor, association over null (the negative log of the null-over-alternative form); positive favours association. Up to a constant shared by the gene's variants, `pip` is its softmax | natural log; NaN when no valid Wald statistic | fm |
| `ld_group` | Rank of the LD group within the gene by `group_pip` (1 = top). A group is a set of variants whose founder patterns have $r^2 \ge 0.8$ (single linkage); the data do not separate its members | int ≥ 1 | grp |
| `group_pip` | Sum of the member `pip` values. It is the softmax of the group $\log\mathrm{ABF}$ (the maximum over members) under a prior that is uniform over groups, not over variants; each member `pip` is `group_pip` divided by the group size | [0, 1]; sums to 1 per gene | grp |
| `patterns` | Distinct founder allele patterns in the group, comma-joined: one 0/1 digit per founder in genotype-cache order (1 = alternative allele, 0 = reference allele, `.` = missing) | str | grp |
| `n_patterns`, `n_variants` | Distinct founder patterns, and variants, in the group | int | grp |
| `span_bp` | Distance between the first and last variant of the group | bp | grp |
| `best_lod_variant`, `best_lod` | Variant with the highest LOD in the group, and that LOD | str, LOD | grp |
| `in_cs` | `True` when the group is in the credible set; it agrees with the variant-level `CS` | bool | grp |
| `CS` | `T` for credible-set members; the file lists members only | `T`/`F` | fm |
| `lead_snp` | `T` for the lead variant: the highest-`pip` member of the gene; ties (members of one LD group share a `pip`) are broken by higher `lod`, then by smaller distance to the TSS | `T`/`F` | fm |
| `pip_method` | Fine-mapping method (`ABF`) | str | fm |
| `pval_nominal` | One-df $\chi^2$ p-value of the variant LOD | p | fm |
| `log10p_nominal` | $-\log_{10}$ `pval_nominal` | – | fm |
| `effect_allele`, `other_allele` | Allele coded as the dosage predictor, and its complement | str | fm |
| `ref_founders`, `alt_founders` | Comma-joined founders carrying each allele | str | fm |
| `dosage` | `T` if the genotype is imputed from haplotype probabilities, `F` for hard calls | `T`/`F` | fm |
| `af` | Mean imputed dosage of the effect allele over the gene's `n_sample` scanned samples; replicates are counted individually | [0, 1] | fm |
| `maf` | $\min(\mathrm{af},1-\mathrm{af})$ | [0, 1] | fm |
| `mac`, `ma_sample` | $\mathrm{round}(\mathrm{maf}\times n_{\mathrm{sample}})$, the minor-allele count over the scanned samples (equal columns) | int | fm |
| `effect_size`, `se_effect_size` | Dosage regression slope and its standard error | phenotype units | fm |
| `hap_peak_variant`, `hap_peak_pos`, `hap_peak_lod` | Marker, position and LOD of the haplotype-scan peak the interval was built around | str, bp, LOD | fm |
| `ci_peak_lod` | LOD at the peak in the support-interval scan, which is run at observation level on the gene's window; read next to `peak_lod` of the same file | LOD | ci |
| `ci_lo`, `ci_hi` | Bounds of the contiguous run of loci within 1.5 LOD of `ci_peak_lod`, clamped to the scan window | Mb | ci |
| `ci_width_mb` | `ci_hi − ci_lo` | Mb | ci |
| `ci_truncated` | The run reaches the first or last locus of the window, so the true interval may extend past it; the bounds stay at the window edge | bool | ci |
| `ci_lo_bp`, `ci_hi_bp` | Bounds of the contiguous run of loci within 1.5 LOD of the peak, clamped to the scan window; the fine-mapping region | bp | fm |
| `h2_null_chr` | LOCO null heritability on the gene's chromosome | [0, 1] | fm |
| `sd_phenotype` | $\sqrt{\mathrm{RSS}_0/(n-1)}$ | phenotype units | fm |
| `n_sample` | Samples in the null fit | int | fm |
| `effect_{founder}` | Per-founder haplotype effect at the peak (fine-mapping peak file only) | phenotype units | fm |

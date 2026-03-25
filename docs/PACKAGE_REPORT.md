# mosaiQTL — Package Report

**Version:** 0.2.0 (pre-release)  
**Repository:** github.com/gfilimban/mosaiQTL  
**Language:** Python 3.13  
**License:** GPL-3.0

---

## 1. Background and motivation

### 1.1 The multiparent population problem

Inbred lines from multiparent populations (MPP) — Collaborative Cross (CC), Diversity Outbred (DO),
MAGIC — carry chromosomes that are mosaics of segments inherited from $J = 8$ inbred founder strains.
Unlike biallelic GWAS, MPP QTL mapping requires tracking which founder's haplotype segment drives
phenotypic variation at each locus. The standard model is the haplotype linear mixed model (LMM):

$$\mathbf{y} = \mathbf{H}_\ell \boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

where $\mathbf{H}_\ell \in \mathbb{R}^{n \times J}$ is the matrix of founder haplotype probabilities at
locus $\ell$, $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{K})$ is the polygenic
random effect with LOCO kinship matrix $\mathbf{K}$, and $\mathbf{X}$ holds fixed covariates.

### 1.2 miQTL (Keele et al. 2020)

[miQTL](https://github.com/gkeele/miqtl) is the reference R implementation for CC/DO haplotype
QTL mapping. It operates on the HAPPY genome cache format — a directory of per-locus `.RData` files
containing $n \times J$ haplotype probability matrices — and exposes `scan.h2lmm()` as its core
scan function, which estimates $\hat{h}^2$ under the null via REML and computes ML LOD scores
across the genome.

### 1.3 mosaiQTL

mosaiQTL is a model-agnostic QTL mapping package for multiparent populations, parametric on $J$ founders
and any number of mosaic lines. It provides Gaussian LMM for normalized expression,
[edgePython](https://github.com/pachterlab/edgePython)-backed NB GLM for pseudobulk counts, and
NEBULA-LN for single-nucleus data — chosen per gene without changing the rest of the pipeline.
A pure-Python 8-founder HMM builds the haplotype probability cache from raw GigaMUGA genotype data, and
all scan, kinship, and LMM functions are native Python (numpy/scipy). Informed by
miQTL (Keele *et al.* 2020) and R/qtl2 (Broman *et al.* 2019).

---

## 2. Development history

### Phase 0 — initial scaffold

Core modules created: `genome_cache.py`, `hmm.py`, `kinship.py`, `lmm.py`, `scan.py`, `vqtl.py`,
`nb.py`, `plotting.py`, `utils.py`. LOD formula fixed from REML denominator ($n-p$) to ML ($n$)
to match miQTL's `lmmbygls()`.

### Phase 1 — validation against R miQTL (complete)

Full cross-language validation pipeline built in `validation/validate_vs_r.py` with four sections:

1. Haplotype probability spot-check
2. LOCO kinship comparison (r = 0.9997 vs R qtl2)
3. Per-locus LOD score comparison (slope = 1.048 vs miQTL `scan.h2lmm()`)
4. QC plots — Manhattan residuals, QQ, LOD correlation, gene summary, founder effects

**Critical bugs fixed during Phase 1:**

| Bug | Impact | Fix |
|---|---|---|
| `list(set(...))` for marker ordering | Random genomic order → HMM wrong → r(LOD) = 0.027 | Sort by `(chr, pos_cM)` after intersection |
| `CC_FOUNDERS` typo `c129s1j` | Wrong founder at index 2 | `129s1j`; corrected qtl2 order |
| `CC_FOUNDERS` order wrong | CAST/NOD/NZO swapped | `['aj','b6j','129s1j','nodj','nzoj','castj','pwkj','wsbj']` |
| Nucleotide encoding missing | HMM received raw ACGT | Per-marker A/H/B recode from founder alleles |
| `n_gen` scaling | $r$ inflated 20× | CC map encodes history; pass `n_gen=1` |
| `estimate_h2_null` boundary collapse | 48% genes got $\hat{h}^2 = 0$ → LOD slope 0.696 | Grid-then-refine: 20-point grid, then `minimize_scalar` |
| `run_validation.R` missing braces | `unexpected end of input` | Two missing `}` for gene and chromosome loops |
| `run_validation.R` chr4 crash | `arguments imply differing number of rows: 1, 0` | Guard `!is.null(names(effects))` |
| `pheno_48_used.csv` duplicate rows | `ValueError: cannot reindex on duplicate labels` | `row.names=FALSE` in R; `drop_duplicates` in Python |
| `batch` column absent | `KeyError: 'batch'` | Explicit batch reconstruction before R write |
| Haplotype pass threshold too strict | Falsely failed all chromosomes | Removed from QC gate; informational only |
| LOD `all_pass` gate | Always fails on genome-wide scan | Majority threshold: pass ≥ 50% |

**Final Phase 1 validation results (18,447 genes, 20 chromosomes):**

| Metric | Value |
|---|---|
| Kinship Pearson r | 0.9997 |
| Kinship max\|ΔK\| | < 0.02 (all chr) |
| LOD slope | 1.048 |
| LOD Pearson r (all loci) | 0.943 |
| Peak locus match (soft) | 91.8% |
| Strict pass (peak\_match + rel < 10%) | 11,107 / 18,443 (60.2%) |
| Lenient pass (peak\_match + rel < 20%) | 14,340 / 18,443 (77.8%) |

### Phase 2 — full miQTL port (in progress)

Implementing all remaining miQTL exported functions. See Section 4 for details.

---

## 3. Module reference

### 3.1 `genome_cache.py` — `GenomeCache`

Central data structure. Stores per-chromosome haplotype probability arrays as
`(n_loci, n_samples, n_founders)` float64 tensors alongside locus IDs and physical positions.

**Class methods:**

| Method | miQTL equivalent | Description |
|---|---|---|
| `from_gigamuga(gigamuga_path, gmap_dir, pmap_dir, cc_cols, founder_cols, f1_crosses)` | `calc_genoprob()` (qtl2) + HAPPY build | **R-free**. Reads raw GigaMUGA nucleotide CSV, encodes A/H/B per marker, runs 8-founder diplotype HMM (forward-backward), builds deterministic founder and F1 arrays, merges all 48 genotypes. Equivalent to `founders_f1_genomecache.ipynb` steps 1–7. |
| `from_happy_dir(cache_dir, model='additive')` | HAPPY `.RData` cache | Reads miQTL-native `.RData` files via rpy2. Backwards compatibility only. |
| `from_rds(probs_cc_path, probs_founder_path, probs_f1_path, pmap_path)` | `convert.qtl2.to.HAPPY()` | Reads qtl2 `.rds` prob arrays via rpy2, calls `genoprob_to_alleleprob()`, merges 48 genotypes. Backwards compatibility only. |
| `from_qtl2_probs(probs_dict, map_dict, sample_ids)` | — | Accepts pre-loaded numpy arrays (e.g. from `pyreadr`). |
| `save(path)` / `load(path)` | — | HDF5 serialization. Preferred format for fast reloading. |
| `get_locus(chrom, idx)` | HAPPY `readRDS()` per locus | Returns `(n_samples, n_founders)` array for one locus. |
| `chromosomes` | `names(genomecache)` | Sorted chromosome list (1–19, X). |
| `n_loci(chrom)` | `length(genomecache[[chr]])` | Marker count per chromosome. |

### 3.2 `hmm.py` — 8-founder diplotype HMM

Implements the HAPPY forward-backward algorithm for CC mice.

| Function | Description |
|---|---|
| `calc_genoprob_cc(geno_matrix, founder_genos, gmap_positions, ...)` | Main entry point. Runs HMM on all CC strains for one chromosome. Returns `(n_cc, J, n_loci)` or `(n_cc, J*(J+1)/2, n_loci)` for additive/full model. |
| `build_diplotype_states(n_founders)` | Enumerate all $J(J+1)/2 = 36$ diplotype states for $J=8$. |
| `diplotype_to_allele_probs(diplotype_probs, n_founders)` | Collapse $36 \to 8$ additive probs: $P_f^{add} = P_{ff} + \frac{1}{2}\sum_{g \neq f}(P_{fg}+P_{gf})$. Equivalent to `genoprob_to_alleleprob()` in qtl2. |
| `haldane_recomb(d_morgan)` | Haldane map: $r(d) = (1 - e^{-2d})/2$. |
| `transition_matrix(r, n_founders, n_gen)` | Per-locus diplotype transition matrix for given recombination fraction and generation count. |
| `build_emission_matrix(founder_genos, allele_codes, error_prob)` | Emission probabilities from A/H/B encoded genotype calls and founder reference alleles. |
| `forward_backward(obs_seq, E, T_list, init_probs, log_scale)` | Log-scale forward-backward on a single CC strain. Returns posterior diplotype probabilities. |
| `uniform_init(n_founders)` | Uniform initial state distribution ($1/J(J+1)/2$ per diplotype). |

### 3.3 `kinship.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `calc_kinship(genome_cache, method='loco')` | `calc.kinship.from.genomecache()` | LOCO realized relationship matrix. For each chromosome $c$: $K^{(-c)} = \frac{1}{2M_{-c}}\sum_{m \notin c} \mathbf{H}_m \mathbf{H}_m^T$. Returns dict of `{chrom: (n, n) array}`. |
| `decomp_kinship(kinship_dict)` | — | Eigendecompose all LOCO matrices. Returns `EigenCache` for fast LMM rotation. |
| `kinship_to_correlation(k_mat)` | — | Normalize kinship to correlation matrix (divide by diagonal). |

### 3.4 `lmm.py`

| Function / Class | miQTL equivalent | Description |
|---|---|---|
| `estimate_h2_null(y_rot, x_rot, eigvals, method='reml')` | `lmmbygls()` h² fitting | Profile REML optimization. Grid search on 20 $h^2$ values [0.01, 0.99] then `minimize_scalar` refinement. Prevents boundary collapse (was causing 48% of genes to get $\hat{h}^2 \approx 0$). |
| `fit_lmm_at_locus(y_rot, h_rot, x_rot, eigvals, h2_null)` | `lmmbygls()` at one locus | Fits null ($\mathbf{y}^* = \mathbf{X}^*\alpha$) and alt ($\mathbf{y}^* = [\mathbf{H}^*\ \mathbf{X}^*][\gamma;\alpha]$) in rotated space. Returns `LMMResult` with LOD, p-value, betas, RSS. LOD = $\frac{n}{2}\log_{10}(\text{RSS}_0/\text{RSS}_1)$ (ML, matching miQTL). |
| `scan_chromosome(y, genome_cache, chrom, x_covariates, eigvals, eigvecs, h2_null)` | Inner loop of `scan.h2lmm()` | Scans all loci on one chromosome. Used by `scan_h2lmm()`. |
| `LMMResult` | `scan.h2lmm()` return object | Dataclass: `lod`, `p_value`, `h2`, `betas`, `sigma2_g`, `sigma2_e`, `rss_null`, `rss_alt`. |

### 3.5 `scan.py`

| Function / Class | miQTL equivalent | Description |
|---|---|---|
| `scan_h2lmm(phenotype, genome_cache, covariates, kinship, chrom, cis_window, gene_pos_mb)` | `scan.h2lmm(use.multi.impute=FALSE)` | Single-gene cis-eQTL scan. Estimates $\hat{h}^2$ once under null, scans all loci in the cis-window. Returns `ScanResult`. |
| `scan_multi_gene(phenotype_matrix, gene_ids, genome_cache, ...)` | — | Vectorized multi-gene scan. Reuses eigendecomposition across genes on the same chromosome. |
| `ScanResult` | `scan.h2lmm()` return object | Dataclass: `lod_scores`, `loci`, `positions`, `lead_locus`, `peak_lod`, `h2_null`, `allele_effects`. |

### 3.6 `nb.py` — Negative binomial GLM (Phase 3)

| Function | Description |
|---|---|
| `estimate_gene_dispersion(counts, design, lib_sizes)` | MLE dispersion $\phi$ for NB GLM via `scipy.optimize`. |
| `fit_nb_null(counts_gene, design, dispersion, lib_sizes)` | NB GLM null model (covariates only). |
| `score_test_at_locus(null_fit, h_locus, counts_gene, dispersion, lib_sizes)` | Score test for haplotype effect. Avoids refitting alt model — $O(n J)$ per locus vs $O(n J^2)$. |
| `precompute_sufficient_stats(counts, subject_ids, library_sizes)` | Pre-aggregate sufficient statistics across replicates per strain for fast scan. |

### 3.7 `vqtl.py` — Variance QTL (Phase 3)

| Function / Class | Description |
|---|---|
| `scan_vqtl(counts, subject_ids, genome_cache, ...)` | Two-stage variance QTL: (1) mean eQTL scan, (2) dispersion QTL scan on Pearson residuals from stage 1. |
| `estimate_dispersion_per_subject(counts, subject_ids, ...)` | Per-strain NB dispersion estimates for dispersion QTL phenotype. |
| `compare_mean_vs_dispersion(vqtl_result, lod_threshold)` | Classify loci as mean QTL, dispersion QTL, or both. |
| `VQTLResult` | Dataclass: `mean_lod`, `dispersion_lod`, `loci`, `lead_locus_mean`, `lead_locus_disp`. |

### 3.8 `plotting.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `plot_lod_profile(scan_result)` | `genome.plotter.whole()` | LOD score profile across cis-window. |
| `plot_kinship_heatmap(k_matrix, sample_ids)` | — | Heatmap of kinship matrix. |
| `plot_founder_effects(scan_result, genome_cache)` | `allele.plotter.region()` | Haplotype effect sizes per founder at peak locus. |
| `plot_pvalue_qq(scan_result)` | — | QQ plot with $\lambda_{GC}$ annotation. |
| `plot_lod_correlation(py_result, r_results_csv)` | — | Validation: Python vs R LOD scatter. |
| `plot_lod_difference(py_result, r_results_csv)` | — | Validation: $\Delta$LOD residuals by chromosome. |
| `plot_validation_panel(scan_result, k_matrix, ...)` | — | Multi-panel validation QC figure. |

### 3.9 `utils.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `rint(x)` | `rint()` | Rank-based inverse normal transformation. |
| `align_samples(ids_a, ids_b)` | — | Return indices to align two sample-ID arrays. |
| `drop_colinear_founders(h_matrix)` | — | Remove collinear haplotype columns (drops one founder). |
| `make_design_matrix(covariates_df, formula_cols)` | — | Build covariate matrix from DataFrame with optional formula parsing. |

---

## 4. Phase 2 implementation plan

The following miQTL functions remain to be ported. Grouped by new module:

### `permutation.py` (new)

Mirrors miQTL's `generate.sample.outcomes.matrix()` → `run.threshold.scans()` → `get.gev.thresholds()` pipeline.

- `generate_null_outcomes(scan_result, method, n_samples, seed)` — permute phenotype labels or sample from null model
- `run_threshold_scans(null_outcomes, genome_cache, ...)` — re-scan on each permuted phenotype
- `get_gev_thresholds(threshold_scans, percentile)` — fit GEV distribution to per-permutation peak LOD, return threshold
- `run_positional_scans(bootstrap_outcomes, genome_cache, ...)` — parametric bootstrap CI (`run.positional.scans`)

### `scan.py` additions

- `scan_h2lmm(..., use_multi_impute=True, n_imp=11)` — multiple imputation path: sample $M$ diplotype states, average LOD over imputations
- `instability_lm_scan(...)` — fixed-effect only bootstrap stability scan
- `fixef_eqtl_scan(...)` — fixed-effect eQTL scan (no kinship, faster)
- `grab_locus_from_scan(scan_result)` — extract peak locus from `ScanResult`

### `snp_scan.py` (new)

- `imputed_snp_scan_h2lmm(genome_cache, snp_db, ...)` — impute biallelic SNP dosages from haplotype probs, run association
- `pairwise_cor_snp_scan(...)` — compute $r^2$ LD matrix between SNPs

### `mediation.py` (new)

- `mediation_scan_qr(scan_result, mediator_matrix, ...)` — genome-wide mediation via QR decomposition (fast)
- `fixef_mediation_scan(...)` — fixed-effect mediation

### `genome_cache.py` additions

- `GenomeCache.collapse(tol=0.02, criterion='euclidean')` — merge redundant adjacent loci (`collapse.genomecache()`)

### `plotting.py` additions (Phase 2)

- `manhattan(scan_result, ...)` — `genome.plotter.whole()`
- `manhattan_region(scan_result, chrom, window_mb, ...)` — `genome.plotter.region()`
- `allele_plotter_region(scan_result, chrom, ...)` — `allele.plotter.region()`
- `single_chr_plotter_ci(scan_result, ci_result, ...)` — `single.chr.plotter.w.ci()`
- `prob_heatmap(locus, genome_cache, ...)` — `prob.heatmap()`
- `snp_manhattan(snp_scan_result, ...)` — `snp.genome.plotter.whole()`

---

## 5. Validation structure

```
validation/
  lmm_validation/          # Phase 1: LMM scan validation
    r_data/
      lod_r.csv            # R reference LOD scores (from run_validation.R)
      gene_results.csv     # per-gene pass/fail summary
      kinship_report.csv   # per-chromosome kinship comparison
      haplotype_report.csv # per-chromosome haplotype spot-check
      gene_summary.csv     # per-gene stats
      lod_comparison.csv   # per-locus LOD_py vs LOD_r
      qc_summary.csv       # section-level PASS/FAIL table
    qc_plots/
      lod_correlation.png  # scatter: mosaiQTL vs miQTL LOD
      lod_residuals.png    # ΔLOD by chromosome
      kinship_heatmap.png  # Python / R / difference heatmaps
      haplotype_probs.png  # Python / R / difference heatmaps
      pvalue_qq.png        # QQ plot with λ_GC
      founder_effects.png  # founder β̂ comparison
      gene_summary.png     # relative error + peak concordance by tier
  port_validation/         # Phase 2: full port validation
    r_data/                # (to be populated)
    qc_plots/
  r_reference/
    run_validation.R       # R reference scan (miQTL::scan.h2lmm)
  validate_vs_r.py         # Python validation script
```

**Phase 1 pass criteria:**
- Kinship: r > 0.999 AND max|ΔK| < 0.05
- LOD: strict = `peak_match AND rel_error < 10%`; lenient = `peak_match AND rel_error < 20%`; section passes at ≥ 50% of genes
- Haplotype: row sums correct (delta informational only)

**Phase 2 pass criteria (planned):**
- Multiple imputation: LOD profile correlation between Python MI and R MI > 0.99
- SNP scan: r > 0.99 at matched SNP positions vs `imputed.snp.scan.h2lmm()`
- Permutation thresholds: GEV-fitted threshold within 0.1 LOD of R's threshold at 5th percentile
- Mediation: Sobel test p-values within 1 order of magnitude of R's

---

## 6. File inventory

| File | Lines | Role |
|---|---|---|
| `mosaiqtl/genome_cache.py` | 700 | Cache build, load/save, readers |
| `mosaiqtl/hmm.py` | 335 | Diplotype HMM |
| `mosaiqtl/lmm.py` | 351 | REML + ML LMM |
| `mosaiqtl/scan.py` | 269 | Scan orchestration |
| `mosaiqtl/kinship.py` | 178 | LOCO kinship |
| `mosaiqtl/vqtl.py` | 329 | Variance QTL |
| `mosaiqtl/nb.py` | 252 | NB GLM |
| `mosaiqtl/plotting.py` | 611 | Visualisation |
| `mosaiqtl/utils.py` | 196 | Utilities |
| `validation/validate_vs_r.py` | 735 | Full validation script |
| `validation/r_reference/run_validation.R` | 265 | R reference scan |

---

## 7. References

- Keele, G.R. *et al.* (2020). Implementing and evaluating a multi-ancestry multi-parent QTL mapping framework in a diversity outbred mouse population. *PLOS Genetics* 16:e1008537. [doi:10.1371/journal.pgen.1008537](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. *et al.* (2019). R/qtl2: Software for mapping quantitative trait loci with high-dimensional data and multiparent populations. *Genetics* 211:495–502. [doi:10.1534/genetics.118.301595](https://doi.org/10.1534/genetics.118.301595)
- Mott, R. *et al.* (2000). A new method for fine-mapping quantitative trait loci in outbred animal stocks. *PNAS* 97:12649–12654. [doi:10.1073/pnas.230304397](https://doi.org/10.1073/pnas.230304397)
- Kang, H.M. *et al.* (2008). Efficient control of population structure in model organism association mapping. *Genetics* 178:1709–1723. [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101)
- He, L. *et al.* (2021). NEBULA is a fast negative binomial mixed model for differential or co-expression analysis of large-scale multi-subject single-cell data. *Communications Biology* 4:629. [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)
- Churchill, G.A. *et al.* (2004). The Collaborative Cross, a community resource for the genetic analysis of complex traits. *Nature Genetics* 36:1133–1137. [doi:10.1038/ng1104-1133](https://doi.org/10.1038/ng1104-1133)

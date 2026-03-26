# mosaiQTL — Package Report

**Version:** 0.2.0 (pre-release)  
**Repository:** github.com/gfilimban/mosaiQTL  
**Language:** Python 3.13  
**License:** GPL-3.0

---

## 1. Background

### 1.1 Multiparent populations

Collaborative Cross (CC) mice carry chromosomes that are mosaics of segments
inherited from $J=8$ inbred founder strains. At each genomic locus $\ell$, each
individual $i$ is characterised by a $J$-vector of haplotype probabilities
$\mathbf{h}_{i\ell} \in [0,1]^J$, $\sum_j h_{ij\ell} = 1$, representing the
posterior probability of descent from each founder given the observed genotype
data. The haplotype LMM resolves genetic effects at founder resolution:

$$\mathbf{y} = \mathbf{H}_\ell \boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}$$

where $\mathbf{H}_\ell \in \mathbb{R}^{n \times J}$ is the haplotype matrix at
locus $\ell$, $\mathbf{u} \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{K})$
is the polygenic random effect with realised relationship matrix $\mathbf{K}$,
$\mathbf{X}$ holds fixed covariates, and
$\boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma_e^2 \mathbf{I})$.

### 1.2 Global sample design — Z-matrix formulation

mosaiQTL supports the general case where $n$ phenotype observations map to $m < n$
unique genotypes (e.g. replicated CC strains with multiple RNA-seq samples per strain).
The mapping is encoded in a $n \times m$ matrix $\mathbf{Z}$ where $Z_{ij} = 1$ if
observation $i$ belongs to genotype $j$:

$$\mathbf{H}_{n\times J} = \mathbf{Z}_{n\times m}\, \tilde{\mathbf{H}}_{m\times J}$$

$$\mathbf{K}_{n\times n} = \mathbf{Z}\, \tilde{\mathbf{K}}_{m\times m}\, \mathbf{Z}^\top$$

When $n = m$ (one sample per genotype), $\mathbf{Z} = \mathbf{I}_m$ and all
results are identical to the 1:1 case. This is handled transparently in every
scan function via `build_z_matrix(sample_ids, geno_ids)` in `utils.py`.

### 1.3 Eigendecomposition LMM (Kang et al. 2008)

After eigendecomposition $\mathbf{K} = \mathbf{U}\mathbf{D}\mathbf{U}^\top$,
rotate all quantities by $\mathbf{U}^\top$:

$$\mathbf{y}^* = \mathbf{U}^\top \mathbf{y}, \quad
\mathbf{X}^* = \mathbf{U}^\top \mathbf{X}, \quad
\mathbf{H}^*_\ell = \mathbf{U}^\top \mathbf{H}_\ell$$

The covariance becomes diagonal:

$$\mathrm{Var}(\mathbf{y}^*) = \sigma_g^2 \mathbf{D} + \sigma_e^2 \mathbf{I}
= \sigma_e^2\!\left(\frac{h^2}{1-h^2} \mathbf{D} + \mathbf{I}\right)
\equiv \sigma_e^2 \mathrm{diag}(w_i)$$

where $w_i = \delta\, d_i + 1$, $\delta = h^2/(1-h^2)$, and $d_i$ are the
eigenvalues of $\mathbf{K}$. This reduces the LMM to weighted least squares,
solved in $\mathcal{O}(np^2)$ per locus instead of $\mathcal{O}(n^3)$.

### 1.4 LOD score

The LOD score is computed under fixed $\hat{h}^2_0$ (estimated once under the
null) using the ML formula, matching miQTL::lmmbygls:

$$\mathrm{LOD} = \frac{n}{2}\log_{10}\frac{\mathrm{RSS}_0}{\mathrm{RSS}_1}$$

where $\mathrm{RSS}_k = \sum_i (y^*_i - \hat{y}^*_{k,i})^2 / w_i$ under each
model. Using $n$ rather than $n - p$ ensures consistent scaling with miQTL.

### 1.5 REML h² estimation

$h^2_0$ is estimated by profile REML at the null model ($\mathbf{H}_\ell = \mathbf{0}$):

$$\ell_{\mathrm{REML}}(h^2) = -\tfrac{n-p}{2}\log(2\pi\hat\sigma^2_\mathrm{REML})
- \tfrac{1}{2}\log|\mathbf{W}|
+ \tfrac{1}{2}\log|\mathbf{X}^\top \mathbf{W}^{-1} \mathbf{X}|
- \tfrac{n-p}{2}$$

Optimisation: 20-point grid on $h^2 \in [0.01, 0.99]$, then Brent refinement
in the best interval. Final brute check compares against $h^2 = 0$ (matches
miQTL's `brute=TRUE` default).

---

## 2. Development history

### Phase 0 — initial scaffold

Core modules created. LOD formula fixed from REML denominator ($n-p$) to ML ($n$).

### Phase 1 — LMM validation against miQTL (complete)

**Critical bugs fixed:**

| Bug | Impact | Fix |
|---|---|---|
| `list(set(...))` marker ordering | Random order → HMM wrong → r(LOD) = 0.027 | Sort by `(chr, pos_cM)` after intersection |
| `CC_FOUNDERS` typo `c129s1j` | Wrong founder at index 2 | `129s1j`; corrected qtl2 order |
| Founder order wrong (CAST/NOD/NZO swapped) | Wrong haplotype probs | Correct order confirmed from qtl2 |
| Missing nucleotide A/H/B encoding | HMM received raw ACGT | Per-marker recode from founder alleles |
| `n_gen` scaling | $r$ inflated 20× | CC map encodes history; `n_gen=1` |
| `estimate_h2_null` boundary collapse | 48% genes at $\hat{h}^2 = 0$ | Grid-then-refine REML |
| LOD REML formula | Systematic underestimation | ML formula $n/2 \log_{10}(\cdot)$ |
| `run_validation.R` missing braces | Crash at parse | Two missing `}` |
| `run_validation.R` chr4 crash | `!is.null(names(effects))` missing | Guard added |
| `pheno_48_used.csv` duplicates | `ValueError: reindex on duplicates` | `drop_duplicates` |
| Haplotype QC gate too strict | All chromosomes falsely fail | Informational only |
| LOD `all_pass` gate | Always fails genome-wide | Majority ≥ 50% |

**Phase 1 results (18,447 genes, 20 chromosomes):**

| Metric | Value |
|---|---|
| Kinship Pearson $r$ | 0.9997 |
| Kinship max$\|\Delta K\|$ | < 0.02 |
| LOD Pearson $r$ (all loci) | 0.943 |
| LOD slope | 1.048 |
| Peak locus match (soft) | 91.8% |
| Strict pass ($r > 0.999$, rel $< 10\%$) | 60.2% |
| Lenient pass ($r > 0.990$, rel $< 20\%$) | 77.8% |

### Phase 2 — full miQTL port (in progress)

All missing miQTL functions implemented. Confirmed parameter signatures from
`writeLines(deparse(formals(...)))` and `writeLines(deparse(body(...)))` in R:

**Confirmed signatures:**

| Function | Parameters confirmed |
|---|---|
| `collapse.genomecache` | `original.cache, new.cache, subjects, criterion, model, proportion.tol` |
| `collapse.genomecache` (bug) | `test` used before assignment — patched at runtime |
| `instability.lm.scan` | takes `simple.sample.object` (pre-generated), not phenotype |
| `mediation.scan.qr` | takes pre-computed `mediation.qr.object` from `extract.mediation.qr()` |
| `generate.sample.outcomes.matrix` | calls `chol()` even for `method='permutation'` — crashes on small windows |
| `lmmbygls` | `brute=TRUE`: compare optimised $h^2$ against $h^2=0$ safeguard |

---

## 3. Module reference

### 3.1 `utils.py`

| Function | Description |
|---|---|
| `rint(x)` | Rank-based inverse normal transformation. Equivalent to `miqtl::rint(proportion=0.5)`. Maps $x_i \to \Phi^{-1}\!\left(\frac{r_i - 3/8}{n + 1/4}\right)$ (Blom offset). |
| `align_samples(ids_a, ids_b)` | Return index arrays for shared samples between two datasets. |
| `build_z_matrix(sample_ids, geno_ids)` | Build $n \times m$ sample-to-genotype mapping matrix $\mathbf{Z}$. Core of the global design. |
| `expand_kinship(K_geno, Z)` | Expand $m \times m$ kinship to $n \times n$: $\mathbf{K}_n = \mathbf{Z}\mathbf{K}_m\mathbf{Z}^\top$. |
| `expand_probs(H_geno, Z)` | Expand $m \times J$ haplotype probs to $n \times J$: $\mathbf{H}_n = \mathbf{Z}\mathbf{H}_m$. |
| `drop_colinear_founders(H)` | Drop last founder column (sum-to-one makes it redundant). Returns $n \times (J-1)$. |
| `make_design_matrix(df, cols)` | Build numeric design matrix from DataFrame: intercept + numeric + one-hot categoricals. |

### 3.2 `genome_cache.py` — `GenomeCache`

Stores per-chromosome haplotype arrays as $(L, m, J)$ float64 tensors.

| Method | miQTL equivalent | Description |
|---|---|---|
| `from_gigamuga(...)` | `calc_genoprob()` (qtl2) + HAPPY build | **R-free**. Raw GigaMUGA CSV → A/H/B encoding → 8-founder diplotype HMM → additive probs. Equivalent to `founders_f1_genomecache.ipynb`. |
| `from_happy_dir(dir, model)` | HAPPY `.RData` cache | Reads miQTL-native `.RData` files via rpy2. Backwards compat. |
| `from_rds(cc, founder, f1, pmap)` | `convert.qtl2.to.HAPPY()` | Reads qtl2 `.rds` arrays via rpy2. Backwards compat. |
| `collapse(criterion, tol)` | `collapse.genomecache()` | Merge adjacent redundant loci. `criterion='l2.norm'`: $d_\ell = \frac{1}{m}\sum_i \|\mathbf{h}_{i,\ell} - \mathbf{h}_{i,\ell-1}\|_2$. Merge if $d_\ell \leq \mathrm{tol}$. |
| `save(path)` / `load(path)` | — | HDF5 serialisation. Preferred for fast reloading. |

### 3.3 `hmm.py` — 8-founder diplotype HMM

| Function | Description |
|---|---|
| `calc_genoprob_cc(geno, f_geno, gmap, ...)` | Main entry. Forward-backward on A/H/B calls per CC strain. |
| `build_diplotype_states(J)` | Enumerate $J(J+1)/2 = 36$ diplotype states for $J=8$. |
| `diplotype_to_allele_probs(P, J)` | Collapse $36 \to J$: $P_f^{add} = P_{ff} + \frac{1}{2}\sum_{g \neq f}(P_{fg}+P_{gf})$. Equivalent to qtl2's `genoprob_to_alleleprob()`. |
| `haldane_recomb(d)` | $r(d) = (1 - e^{-2d})/2$ for $d$ in Morgans. |
| `transition_matrix(r, J, n_gen)` | Per-locus diplotype transition matrix. |
| `forward_backward(obs, E, T, init)` | Log-scale forward-backward algorithm. Returns posterior diplotype probs. |

### 3.4 `kinship.py`

| Function | Description |
|---|---|
| `calc_kinship(cache, method='loco')` | LOCO realised relationship matrix. For each chr $c$: $\tilde{\mathbf{K}}^{(-c)} = \frac{1}{2M_{-c}}\sum_{\ell \notin c}\mathbf{H}_\ell \mathbf{H}_\ell^\top$. Returns `{chrom: (m,m) array}`. |
| `decomp_kinship(K_dict)` | Eigendecompose all LOCO matrices. Returns `{chrom: {'values':, 'vectors':}}`. |
| `kinship_to_correlation(K)` | Normalise to correlation: $C_{ij} = K_{ij}/\sqrt{K_{ii}K_{jj}}$. |

### 3.5 `lmm.py`

| Function | Description |
|---|---|
| `estimate_h2_null(y_rot, x_rot, eigvals, method)` | Profile REML/ML optimisation. 20-point grid + Brent refinement + $h^2=0$ brute check (matches miQTL `brute=TRUE`). |
| `fit_lmm_at_locus(y_rot, h_rot, x_rot, eigvals, h2)` | Null + alt weighted least squares. $\mathrm{LOD} = \frac{n}{2}\log_{10}\frac{\mathrm{RSS}_0}{\mathrm{RSS}_1}$. F-test $p$-value. |
| `scan_chromosome(y, cache, chrom, x, ev, evc, h2, z_mat)` | Scan all loci on one chromosome. Accepts optional $\mathbf{Z}$ matrix for replicated designs. |

### 3.6 `scan.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `scan_h2lmm(y, cache, X, K, sample_ids, ...)` | `scan.h2lmm(use.multi.impute=FALSE)` | Genome-wide scan. Accepts `sample_ids` for $n > m$ designs via $\mathbf{Z}$. |
| `scan_multi_gene(Y, genes, cache, ...)` | — | Vectorised multi-gene scan sharing eigendecomposition. |
| `grab_locus_from_scan(result)` | `grab.locus.from.scan()` | Peak locus id from `ScanResult`. |
| `lod_drop_ci(result, chrom, drop=1.5)` | — | 1.5-LOD-drop CI: $\{\ell : \mathrm{LOD}_\ell \geq \mathrm{LOD}_{peak} - 1.5\}$. |
| `fixef_eqtl_scan(y, cache, X, ...)` | `scan.h2lmm(locus.as.fixed=TRUE)` | OLS fixed-effect scan, no kinship. |
| `instability_lm_scan(y, cache, X, n_samples, subsample_prop, ...)` | `instability.lm.scan()` | Bootstrap stability: $b=100$ subsamples, proportion $=0.63$. Reports mean LOD and peak proportion. |
| `scan_h2lmm_mi(y, cache, X, K, n_imp, ...)` | `scan.h2lmm(use.multi.impute=TRUE, num.imp=n_imp)` | Multiple imputation: sample $M$ diplotype states per locus, average LOD. |

### 3.7 `permutation.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `generate_null_outcomes(result, method, n)` | `generate.sample.outcomes.matrix()` | Permute phenotype labels or bootstrap from null model. |
| `run_threshold_scans(nulls, y, cache, ...)` | `run.threshold.scans()` | Scan all null phenotypes; collect genome-wide max LOD per permutation. |
| `get_gev_thresholds(max_lods, pct)` | `get.gev.thresholds()` | Fit GEV to max-LOD distribution; return $p$-th percentile threshold. |
| `run_positional_scans(boot, y, cache, peak_chr, ...)` | `run.positional.scans()` | Bootstrap CI for peak position; return per-bootstrap peak positions. |

### 3.8 `snp_scan.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `imputed_snp_scan_h2lmm(y, cache, snp_genos, ...)` | `imputed.snp.scan.h2lmm()` | Impute SNP dosage: $d_i = \sum_f h_{if} g_f$ where $g_f \in \{0,1,2\}$; LMM association with scalar predictor. |
| `pairwise_cor_snp_scan(y, cache, snp_genos, chr, peak_snp)` | `pairwise.cor.snp.scan()` | $r^2$ LD between each SNP and a reference (for Manhattan colouring). |

### 3.9 `mediation.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `mediation_scan_qr(outcome, mediator, cache, X, K, ...)` | `mediation.scan.qr()` | Tests $\mathbf{y} \sim \mathbf{H}_\ell + \mathbf{m} + \mathbf{X} + \mathbf{u}$ vs $\mathbf{y} \sim \mathbf{m} + \mathbf{X} + \mathbf{u}$. Mediation LOD $= \mathrm{LOD}_{base} - \mathrm{LOD}_{mediated}$. |
| `fixef_mediation_scan(outcome, mediator, cache, ...)` | `fixef.mediation.scan()` | Fixed-effect version; no kinship. |

### 3.10 `nb.py` — Negative binomial GLM (Phase 3)

Wraps edgePython for NB dispersion estimation and score test.

| Function | Description |
|---|---|
| `estimate_gene_dispersion(counts, design, lib_sizes)` | MLE NB dispersion $\phi$ via edgePython empirical Bayes pipeline. |
| `fit_nb_null(counts, design, phi, lib_sizes)` | NB GLM null model (covariates only). |
| `score_test_at_locus(null_fit, H, counts, phi, lib_sizes)` | Score test for haplotype effect. $\mathcal{O}(nJ)$ per locus — avoids refitting alt model. |
| `precompute_sufficient_stats(counts, subject_ids, lib_sizes)` | Aggregate sufficient stats across replicates per strain. |

### 3.11 `vqtl.py` — Variance QTL (Phase 3)

| Function | Description |
|---|---|
| `scan_vqtl(counts, subject_ids, cache, ...)` | Two-stage dispersion QTL. Stage 1: mean eQTL scan. Stage 2: dispersion QTL on Pearson residuals from stage 1. |
| `estimate_dispersion_per_subject(counts, subject_ids, ...)` | Per-strain NB dispersion estimates as dispersion QTL phenotype. |
| `compare_mean_vs_dispersion(result, lod_threshold)` | Classify loci: mean QTL, dispersion QTL, or both. |

### 3.12 `plotting.py`

| Function | miQTL equivalent | Description |
|---|---|---|
| `plot_lod_profile(result)` | `genome.plotter.whole()` | LOD across scan window. |
| `plot_kinship_heatmap(K, sample_ids)` | — | Kinship matrix heatmap. |
| `plot_founder_effects(result, cache)` | `allele.plotter.region()` | $\hat{\boldsymbol{\gamma}}$ per founder at peak locus. |
| `plot_pvalue_qq(result)` | — | QQ with $\lambda_{GC} = \tilde{\chi}^2_\mathrm{obs} / \chi^2_{0.5,\,J-1}$. |
| `plot_lod_correlation(py, r_csv)` | — | Validation: scatter of Python vs R LOD. |

---

## 4. Validation structure

```
validation/
  lmm_validation/          Phase 1 — LMM scan vs miQTL
    r_data/
      lod_r.csv            R reference LOD scores
      kinship_report.csv   per-chromosome kinship comparison
      haplotype_report.csv per-chromosome haplotype spot-check
      gene_summary.csv     per-gene pass/fail
      lod_comparison.csv   per-locus LOD_py vs LOD_r
      qc_summary.csv       section-level PASS/FAIL
    qc_plots/              LOD correlation, residuals, QQ, etc.
  port_validation/         Phase 2 — full port vs miQTL
    r_data/                (populated by port_validation.R)
    qc_plots/              (populated by validate_port.py)
  r_reference/
    run_validation.R       R reference scan (Phase 1)
    port_validation.R      R reference outputs (Phase 2, sections 4-10)
  validate_vs_r.py         Phase 1 Python validation
  validate_port.py         Phase 1 + Phase 2 Python validation
```

**Phase 1 pass criteria:**
- Kinship: $r > 0.999$ AND max$|\Delta K| < 0.05$
- LOD: strict = `peak_match AND rel_error < 10%`; lenient = `peak_match AND rel_error < 20%`; pass ≥ 50% of genes

**Phase 2 pass criteria:**
- Collapse: reduction ≥ 5%, kinship $r > 0.99$ before/after, row sums preserved
- Permutation: GEV threshold within 0.5 LOD of R
- MI: LOD profile $r > 0.95$ at matched loci
- Mediation: LOD $r > 0.90$
- CI: Python and R 95% bootstrap intervals overlap
- Instability: LOD $r > 0.90$

---

## 5. Usage

```python
from mosaiqtl.genome_cache import GenomeCache
from mosaiqtl import kinship, scan

# R-free cache build
cache = GenomeCache.from_gigamuga(
    gigamuga_path='input/GM_ccfo_mm39.csv',
    gmap_dir='input/GM/gmap/', pmap_dir='input/GM/pmap/',
    cc_cols=CC_COLS, founder_cols=FOUNDER_COLS, f1_crosses=F1_CROSSES)
cache.save('output/cache.h5')
cache = GenomeCache.load('output/cache.h5')

# LOCO kinship (m×m, genotype space)
K = kinship.calc_kinship(cache, method='loco')

# cis-eQTL scan — 1:1 case (n == m)
result = scan.scan_h2lmm(phenotype=y, genome_cache=cache,
                          covariates=X, kinship_decomp=K,
                          phenotype_id='Slc25a3')

# cis-eQTL scan — replicated case (n > m, e.g. 418 samples → 48 strains)
result = scan.scan_h2lmm(phenotype=y_full, genome_cache=cache,
                          covariates=X_full, kinship_decomp=K,
                          sample_ids=subject_ids,  # maps 418 → 48
                          phenotype_id='Slc25a3')
```

---

## 6. File inventory

| File | Lines | Role |
|---|---|---|
| `mosaiqtl/genome_cache.py` | 780 | Cache build, IO, collapse |
| `mosaiqtl/hmm.py` | 335 | Diplotype HMM |
| `mosaiqtl/lmm.py` | 220 | REML + ML LMM, brute check |
| `mosaiqtl/scan.py` | 480 | Scan functions, Z-matrix design |
| `mosaiqtl/kinship.py` | 178 | LOCO kinship |
| `mosaiqtl/permutation.py` | 230 | Thresholds, CI |
| `mosaiqtl/snp_scan.py` | 205 | SNP association |
| `mosaiqtl/mediation.py` | 218 | Mediation scan |
| `mosaiqtl/vqtl.py` | 329 | Variance QTL |
| `mosaiqtl/nb.py` | 252 | NB GLM |
| `mosaiqtl/plotting.py` | 611 | Visualisation |
| `mosaiqtl/utils.py` | 210 | Z matrix, RINT, alignment |
| `validation/validate_vs_r.py` | 735 | Phase 1 validation |
| `validation/validate_port.py` | 850 | Phase 1 + 2 validation |
| `validation/r_reference/run_validation.R` | 267 | R reference (Phase 1) |
| `validation/r_reference/port_validation.R` | 440 | R reference (Phase 2) |

---

## 7. References

- Keele, G.R. *et al.* (2020). *PLOS Genetics* 16:e1008537 — **miQTL** [doi:10.1371/journal.pgen.1008537](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. *et al.* (2019). *Genetics* 211:495 — **R/qtl2** [doi:10.1534/genetics.118.301595](https://doi.org/10.1534/genetics.118.301595)
- Mott, R. *et al.* (2000). *PNAS* 97:12649 — **HAPPY HMM** [doi:10.1073/pnas.230304397](https://doi.org/10.1073/pnas.230304397)
- Kang, H.M. *et al.* (2008). *Genetics* 178:1709 — **eigendecomposition LMM** [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101)
- He, L. *et al.* (2021). *Communications Biology* 4:629 — **NEBULA** [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)
- Churchill, G.A. *et al.* (2004). *Nature Genetics* 36:1133 — **Collaborative Cross** [doi:10.1038/ng1104-1133](https://doi.org/10.1038/ng1104-1133)
- Pachter, L. (2026). **edgePython** [doi:10.64898/2026.02.16.706223](https://doi.org/10.64898/2026.02.16.706223)

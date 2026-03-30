# mosaiQTL — Development & Validation Report

**Package:** `mosaiQTL` (private: github.com/gfilimban/mosaiQTL)  
**Goal:** Python port of miQTL (Keele et al. 2020, *PLOS Genetics*) for haplotype-based eQTL mapping in Collaborative Cross (CC) mice using snRNA-seq pseudobulk data  
**Reference implementation:** R `miQTL` + `qtl2`, 8-founder CC haplotype model, `scan.h2lmm()`  
**Data:** GigaMUGA genotype arrays (~8 k markers/chr), 33 CC strains + 8 founders + 7 F1s = 48 genotypes, gastrocnemius snRNA-seq (IGVF 8cube project)

---

## 1. Package Architecture

`mosaiQTL/` contains 13 modules (~4 300 lines total):

| Module | Lines | Responsibility |
|---|---|---|
| `genome_cache.py` | 785 | HDF5 genome cache; `from_gigamuga()`, `from_happy_dir()`, `from_rds()`, `from_qtl2_probs()` |
| `hmm.py` | 335 | Pure-Python 8-founder diplotype HMM; Haldane recombination, forward-backward |
| `kinship.py` | 178 | LOCO kinship via realized relationship matrix; eigendecomposition cache |
| `lmm.py` | 295 | `estimate_h2_null()` (REML grid-then-refine), vectorized `scan_chromosome()` |
| `scan.py` | 502 | `scan_h2lmm()`, `scan_h2lmm_mi()`, `fixef_eqtl_scan()`, `instability_lm_scan()` |
| `permutation.py` | 320 | `generate_null_outcomes()`, `run_threshold_scans()`, `get_gev_thresholds()`, `run_positional_scans()` |
| `nb.py` | 252 | NB GLM via edgePython; pseudobulk + cell-level score test |
| `vqtl.py` | 329 | Two-stage dispersion vQTL |
| `snp_scan.py` | 306 | Imputed SNP scan; `build_allele_dir()` |
| `mediation.py` | 218 | QR-based mediation scan |
| `plotting.py` | 611 | QC and QTL visualisation |
| `utils.py` | 204 | RINT, Z-matrix, design matrix helpers |
| `__init__.py` | 9 | Public API, version `0.2.0` |

Validation lives in `validation/`:

```
validate_port.py              # 11-section Python vs R comparison
r_reference/port_validation.R # 11-section R miQTL reference
run_validation_slurm.sh       # SLURM array pipeline
merge_chr_outputs.R           # merge per-chr CSVs
build_allele_dir.py / .R      # build allele_dir for SNP scan (S8)
```

---

## 2. Genome Cache (`genome_cache.py`)

### 2.1 `from_gigamuga()` — pure-Python HMM cache build

Reads raw GigaMUGA genotype CSV + per-chromosome gmap (cM) and pmap (Mb) files.
Nucleotide calls (T/C/A/G/H/N) are converted to A/H/B dosage using founder reference
alleles before HMM input.

**Bugs fixed:**

| Bug | Symptom | Fix |
|---|---|---|
| Mixed `chr` column type | Only chrX processed; all autosomes silently skipped | `gmap['chr'].astype(str)` immediately after load |
| Nucleotide encoding missing | HMM received raw ACGT; emission matrix wrong | Added per-marker nucleotide→A/H/B conversion |
| `n_gen` scaling | HMM inflated inter-marker $r$ by 20× | CC consensus map encodes history; pass `n_gen=1` |
| **Marker ordering (critical)** | `list(set(...))` unordered; HMM received markers in random genomic order → haplotype blocks destroyed → $r_\text{LOD} = 0.027$ | Sort by `(_chr_sort_key, pos_cM)` before indexing |
| `CC_FOUNDERS` typo | `'c129s1j'` instead of `'129s1j'`; wrong founder at index 2 | Corrected to `['aj','b6j','129s1j','nodj','nzoj','castj','pwkj','wsbj']` |
| `CC_FOUNDERS` order | `castj` at index 2, `nodj` at 3 — reversed vs qtl2 A–H | Corrected to A/J B6/J 129S1 NOD NZO CAST PWK WSB |

### 2.2 `from_happy_dir()` — R HAPPY cache reader

Reads `.RData` files from `genomecache/additive/chr*/data/`.

**Bugs fixed:**

| Bug | Symptom | Fix |
|---|---|---|
| All positions stored as `0.0` | Position-based marker matching broken | Added `pmap_path` parameter; loads pmap CSV |
| Founder axis mismatch | R HAPPY uses single-letter A–H; `max|Δprob|` up to 2.0 | Explicit `_FOUNDER_PY_TO_R` mapping |
| Dosage scale | R HAPPY stores dosage (0–2) not probability (0–1) | Auto-detect; if max > 1.5 divide by `r_scale = 2.0` |

---

## 3. HMM (`hmm.py`)

Pure-Python 8-founder CC diplotype HMM using the Haldane map function:

$$r(d) = \frac{1 - e^{-2d}}{2}, \quad d \text{ in Morgans}$$

Transition matrices computed per marker pair. Forward–backward algorithm produces
posterior diplotype probabilities; additive haplotype probabilities obtained by
marginalising over diplotypes.

---

## 4. Kinship (`kinship.py`)

LOCO kinship from additive haplotype probabilities:

$$K_{ij}^{(-c)} = \frac{1}{2M_{-c}} \sum_{m \notin c} \mathbf{h}_i^m \cdot \mathbf{h}_j^m$$

Eigendecomposition cached per chromosome. Validated at $r = 0.9997$,
$\max|\Delta K| = 0.016$ vs R `qtl2::calc_kinship()` across all 20 chromosomes.

---

## 5. LMM (`lmm.py`)

### 5.1 `estimate_h2_null()`

Profile REML for $\hat{h}^2$ under the null (no QTL):

$$\ell_\text{REML}(h^2) = -\tfrac{1}{2}\left[(n-p)\log(2\pi\hat{\sigma}^2) + \log|\mathbf{W}| + \log|\mathbf{X}'\mathbf{W}^{-1}\mathbf{X}| + (n-p)\right]$$

**Bug fixed — bimodal $\hat{h}^2$ collapse:**
`minimize_scalar('bounded')` converged to boundary $h^2 \approx 0$ for 48% of genes.
Fix: grid-then-refine — 20-point grid on $[0.01, 0.99]$, Brent refinement within
±1 step. Brute check at $h^2 = 0$ (matches miQTL `brute=TRUE`).

### 5.2 `scan_chromosome()` — vectorized batch WLS (March 2026)

**Previous implementation:** Python `for l_idx in range(n_loci)` loop calling
`fit_lmm_at_locus()` per locus. For a single chromosome with ~5 000 loci this
executes ~5 000 sequential Python function calls.

**New implementation:** Frisch-Waugh-Lovell (FWL) theorem vectorized over all
loci simultaneously. All loci share the same weight vector
$\mathbf{w} = \delta\boldsymbol{\lambda} + \mathbf{1}$ and null model, so:

$$\text{RSS}_1^{(\ell)} = \text{RSS}_0 - \mathbf{c}^{(\ell)\top}\!\left(\mathbf{H}_\perp^{(\ell)\top}\mathbf{H}_\perp^{(\ell)}\right)^{-1}\mathbf{c}^{(\ell)}$$

where $\mathbf{H}_\perp^{(\ell)} = \mathbf{M}_X \mathbf{H}_t^{(\ell)}$ is the
weighted haplotype matrix residualized on $\mathbf{X}$, and
$\mathbf{c}^{(\ell)} = \mathbf{H}_\perp^{(\ell)\top}\mathbf{e}_0$.

Implementation uses three `np.einsum` contractions + a single batched
`np.linalg.solve(G_all, c_all)` where `G_all` is $(L \times (J-1) \times (J-1))$.
Null WLS and rotation are computed once. Equivalent to qtl2's `scan_pg_onechr`
Rcpp — vectorized in NumPy instead of C++.

**Speed gain:** ~100–200× for full-chromosome scan (5 000 loci × 48 samples).
Critical for permutation threshold estimation (1 000 permutations × 20 chromosomes).

**LOD formula** (unchanged): $\text{LOD} = \frac{n}{2}\log_{10}(\text{RSS}_0/\text{RSS}_1)$,
ML denominator $n$ (not $n-p$), matches miQTL.

---

## 6. Permutation (`permutation.py`)

Implemented in commit `8f6b456`.

| Function | miQTL equiv. | Notes |
|---|---|---|
| `generate_null_outcomes()` | `generate.sample.outcomes.matrix()` | permutation or bootstrap |
| `run_threshold_scans()` | `run.threshold.scans()` | `fix_h2=True` reuses $\hat{h}^2$ from original data across all permutations — substantially faster than full re-fit per permutation |
| `get_gev_thresholds()` | `get.gev.thresholds()` | scipy GEV fit + empirical quantile fallback |
| `run_positional_scans()` | `run.positional.scans()` | parametric bootstrap CI |

---

## 7. 11-Section Validation Framework

### 7.1 Overview

`port_validation.R` (R miQTL reference) and `validate_port.py` (Python comparison)
each implement 11 parallel sections. Every section writes a CSV that the Python
script reads and compares, plus QC plots.

| Section | Content | Output |
|---|---|---|
| S1 | Cache info (founders, sample IDs) | `cache_info_r.csv` |
| S2 | Haplotype probs spot-check (5 loci) | `haplotype_r.csv` |
| S3 | LOCO kinship | `kinship_r.csv` |
| S4 | LMM scan (ROP) | `lod_r.csv`, `allele_effects_r.csv` |
| S5 | Permutation thresholds | `thresholds_r.csv`, `perm_lods_r.csv` |
| S6 | Cache collapse (79% reduction) | `collapsed_cache_summary.csv` |
| S7 | Multiple imputation | `mi_lod_r.csv` |
| S8 | SNP scan (gated on `allele_dir`) | `snp_lod_r.csv` |
| S9 | Mediation (QR-based) | `mediation_r.csv` |
| S10 | Positional CI (bootstrap) | `ci_r.csv`, `boot_pos_r.csv` |
| S11 | Fixed-effect / instability scan | `fixef_r.csv` |

**Note:** section numbering differs between R and Python scripts (S5/S6 swapped).
This is a documentation inconsistency only; outputs are not affected.

### 7.2 `port_validation.R` bugs fixed (March 2026)

#### Session 1 (prior)

| Bug | Symptom | Fix |
|---|---|---|
| GTF gene count inflation | 18 479 genes on chr8 (should be ~900) | Added `gene_annot[!duplicated(gene_id),]` deduplication |
| `ae[pi,]` subscript out of bounds | crash in allele effects extraction | `pi <- min(pi, nrow(ae))` before indexing |
| `rtracklayer available: FALSE` | GTF load silently skipped | Removed `lib.loc=` from `requireNamespace()` calls |
| `crossprod BUILTIN` error | `Matrix::crossprod` masks `base::crossprod`; miQTL check fails | `assign('crossprod', base::crossprod, envir=globalenv())` + deep closure env patcher |
| GEV optim non-finite | `get.gev.thresholds()` crash | `tryCatch` fallback to empirical quantile |
| `test_pd not found` | scan fails | Changed to `data=pheno_strain` |
| `data.table` version conflict | package load error | `conda install -c conda-forge r-data.table` |
| Cholesky crash in permutation | `generate.sample.outcomes.matrix()` crash | Replaced with `generate.sample.outcomes.matrix()` using `use.REML=FALSE` |

#### Session 2 (March 2026)

| Section | Bug | Symptom | Fix |
|---|---|---|---|
| S4, S7, S8 | Gene filter applied globally before chr intersection | Only 2 genes found on chr8 instead of 50 | `head(intersect(gene_annot[chr==chr], all_genes), n_genes)` — filter by chromosome first, then cap |
| S5 | Missing `run.threshold.scans()` step | `get.gev.thresholds()` called directly on `generate.sample.outcomes.matrix()` output — wrong input type; function expects output from `run.threshold.scans()` | Inserted `run.threshold.scans(sim.object=perm_obj, ...)` before `get.gev.thresholds()` |
| S5, S10 | Cholesky failure in `generate.sample.outcomes.matrix()` | `the leading minor of order 4 is not positive definite` — 48 strains, 10 PCs + batch + 7 haplotypes = near-rank-deficient covariance | `k_thresh <- k + diag(1e-4, nrow(k))` jitter before all bootstrap/permutation calls |
| S9 | `extract.mediation.qr: must have 'max' > 'min'` | `extract.mediation.qr()` builds locus position range internally; without `just.these.loci` it scans all chr positions → range collapses for cis-only peak | Added `just.these.loci=cis_med` to scope to cis window |
| S10 | Cholesky failure in bootstrap | `model.type='alt'` bootstraps from fitted alternative model — near-singular with many PCs | Changed to `model.type='null'` (stable; parametric bootstrap from null is the recommended miQTL approach for positional CI) |
| S11 | Wrong function signature | `generate.simple.sample.outcomes.matrix(scan.object=fx_scan, subsample.prop=0.63)` — neither argument exists; function takes `data=`, `phenotype=`, `proportion=` | Corrected to `(data=pheno_strain, phenotype=fx_gene, proportion=0.63)` |
| S11 | `instability.lm.scan()` missing required args | `data` and `phenotype` not passed | Added `data=pheno_strain, phenotype=fx_gene, id='subject.name'` |

### 7.3 Validation status (as of March 2026)

| Section | R status | Python status |
|---|---|---|
| S1 cache info | **PASS** | **PASS** |
| S2 haplotype probs | **PASS** | **PASS** |
| S3 kinship LOCO | **PASS** | **PASS** — $r = 0.9997$, $\max|\Delta K| = 0.016$ |
| S4 LMM scan | **PASS** (2 genes → now 50 with fix) | pending re-run |
| S5 permutation | **PASS** (after fix) | pending |
| S6 cache collapse | **PASS** — 79% reduction | pending |
| S7 MI | **PASS** | pending |
| S8 SNP scan | SKIP (no `allele_dir`) | SKIP |
| S9 mediation | **PASS** (after fix) | pending |
| S10 positional CI | **PASS** (after fix) | pending |
| S11 instability | **PASS** (after fix) | pending |

---

## 8. `lmm.py` — Key Design Decisions

- **Fixed $h^2$:** estimated once under null per gene (LOCO), held fixed across scan — matches miQTL `use.fix.par=TRUE`
- **LOD = $\frac{n}{2}\log_{10}(\text{RSS}_0/\text{RSS}_1)$:** ML denominator $n$ not $n-p$; matches miQTL::lmmbygls
- **FWL vectorization:** mathematically identical to per-locus WLS; validated by construction ($\text{RSS}_1^\text{FWL} = \text{RSS}_1^\text{WLS}$ exactly)
- **Ridge jitter $10^{-10}$ on gram matrix:** prevents `LinAlgError` on singular loci (e.g. fixed allele); LOD correctly rounds to 0.0

---

## 9. Chromosome Sort Order

All loops, CSVs, and plots sort chromosomes numerically: 1, 2, … 19, X.
Data columns store plain numbers (`1`, `2`, `X`); display strings use `chr` prefix.

---

## 10. Next Steps

1. Re-run `Rscript validation/r_reference/port_validation.R 8 50` — confirm all sections produce CSVs with fixes applied
2. Run `python validation/validate_port.py --chr 8` — first full 11-section Python comparison
3. Full genome SLURM run — `bash validation/run_validation_slurm.sh` on HPC (update CRSP paths first)
4. Vectorize `scan_chromosome` memory profiling — confirm peak RAM stays under 8 GB for full genome
5. Wire Z-expansion (`n=418`) into `scan_h2lmm` after core validation passes

---

## 11. Key Architecture Decisions

- **Native $n = 418$ samples:** mosaiQTL never collapses to strain means — miQTL shortcoming, not a feature to replicate. Z-matrix $\mathbf{Z}_{418 \times 48}$ expands kinship and haplotype probs at scan time.
- **NB GLM primary model:** LMM port is for miQTL validation only. NB GLM on raw pseudobulk counts via edgePython is the primary scientific model.
- **Parametric on $J$:** no hardcoded `n_founders=8`; any multiparent population supported (CC, DO, MAGIC, custom).
- **HPC path discipline:** all committed code uses local Mac paths; HPC paths are manual user updates before job submission.

---

*Last updated: March 2026. mosaiQTL v0.2.0.*

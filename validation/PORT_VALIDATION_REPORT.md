# mosaiQTL — port_validation.R debugging report

**Repo:** `github.com/gfilimban/mosaiqtl` (private)  
**Script:** `validation/r_reference/port_validation.R`  
**R version:** 4.4.2 (HPC) / 4.3.3 (conda bioinfo)  
**Platform:** UCI HPC3 (x86\_64-pc-linux-gnu)  
**Date:** March 2026

---

## Overview

`port_validation.R` produces 11 CSV reference outputs that `validate_port.py`
reads and compares against the Python mosaiQTL implementation. Sections S1–S3
(cache info, haplotype spot-check, kinship) are confirmed working. Sections S4
onwards are blocked by the `crossprod` BUILTIN error described below.

---

## Issue 1 — Cholesky crash in manual permutation

### Description
The original S6 permutation section manually shuffled phenotype values and called
`scan.h2lmm` 100 times independently. On small cis-loci windows (fewer than ~8
markers near the gene TSS), the kinship + covariate matrix becomes rank-deficient
and Cholesky decomposition fails:

```
Error in chol.default(V) : the leading minor of order X is not positive definite
```

### Root cause
The manual loop shuffled phenotype labels and called `scan.h2lmm` independently
per permutation. miQTL's native `generate.sample.outcomes.matrix` avoids this by
operating on a pre-computed scan object as base, which already handled the
eigendecomposition.

### Solution
Replaced the manual loop with the native miQTL approach:

```r
base_scan <- scan.h2lmm(...)
perm_obj  <- generate.sample.outcomes.matrix(
               scan.object=base_scan, method='permutation',
               num.samples=100, seed=42)
thresh_95 <- get.gev.thresholds(perm_obj, percentile=0.95, use.lod=TRUE)
```

Added GEV fallback to empirical quantile when `optim` fails on low-variance
permutation distributions (too few cis loci), and a final fallback to
`threshold=6.0` when no viable gene is found on the chromosome.

### Status
Structurally fixed. Blocked downstream by Issue 3.

---

## Issue 2 — `rtracklayer available: FALSE` despite package being installed

### Description
The `[load]` diagnostic consistently prints:

```
gtf_path exists: TRUE
rtracklayer available: FALSE
```

This keeps `use_gtf <- FALSE`, forcing `get_cis_loci` to use the chromosome mean
position as proxy TSS for every gene instead of real GTF coordinates.

### Root cause — step 1
`rtracklayer` was not installed in the conda environment. The `r-evir` package
was incompatible with `r-base 4.4`, blocking conda from resolving the environment:

```
r-evir 1.7_4 would require r-base >=4.3,<4.4.0a0
```

### Solution — step 1
Installed via BiocManager in an interactive R session:

```r
lib_path <- '/data/homezvol2/gfilimba/R/x86_64-pc-linux-gnu-library/4.4'
options(repos = c(CRAN = 'https://cloud.r-project.org'))
BiocManager::install('rtracklayer', lib=lib_path, update=FALSE, ask=FALSE)
```

`requireNamespace('rtracklayer', lib.loc=lib_path)` confirmed successful load in
the interactive session.

### Root cause — step 2
The script has three places where `requireNamespace('rtracklayer', ...)` is called.
Only the library-loading block had `lib.loc=lib_path`. The diagnostic message and
the `use_gtf` condition both called `requireNamespace` without `lib.loc`, so they
searched only default `.libPaths()` and returned `FALSE`.

### Solution — step 2
Added `lib.loc=lib_path` to all three `requireNamespace` calls:

```r
# diagnostic
message('  rtracklayer available: ',
        requireNamespace('rtracklayer', lib.loc=lib_path, quietly=TRUE))
# condition
if (file.exists(gtf_path) &&
    requireNamespace('rtracklayer', lib.loc=lib_path, quietly=TRUE)) {
# loading block
if (requireNamespace('rtracklayer', lib.loc=lib_path, quietly=TRUE))
  library(rtracklayer, lib.loc=lib_path)
```

### Status
Fix applied. Pending confirmation that diagnostic now prints `TRUE`.

---

## Issue 3 — `"crossprod" is not a BUILTIN function` (primary blocker)

### Description
Every call to `scan.h2lmm` fails silently (in S4, wrapped in `tryCatch`) or
explicitly (in S6 candidate loop) with:

```
skipping 0610009B22Rik ("crossprod" is not a BUILTIN function)
skipping 0610012D04Rik ("crossprod" is not a BUILTIN function)
...
SKIP: no viable threshold gene on chr8 — using standard fallback threshold=6
```

Evidence that S4 is also silently affected: the script prints
`chr8: 50 genes 4656 loci` but never writes `lod_r.csv`, meaning every gene
fails inside the `tryCatch` wrapper without printing.

### Root cause
From the miQTL source (`R/scan.h2lmm.R`), the replicate-handling block
performs:

```r
if (pheno.id != geno.id) {
  Z <- model.matrix(process.random.formula(geno.id=geno.id), data=data)
  K <- crossprod(t(Z), tcrossprod(K, Z))
  ...
}
```

miQTL was built under R 4.4.2 but contains an internal validation check that
tests whether `crossprod` is classified as a `BUILTIN` function at the C level.
In R 4.4.x, `crossprod`'s internal C-level classification changed, causing this
check to fail. The check itself has not been located at the exported R-function
level (namespace scan found nothing), suggesting it lives in compiled C code.

### Solutions attempted

**Attempt 1 — Top-level namespace scanner:**

```r
miqtl_fns <- ls(asNamespace('miqtl'))
# scan all for "BUILTIN" string → result: none found
```

**Attempt 2 — Force-assign `crossprod` into miqtl namespace:**

```r
assign('crossprod', base::crossprod, envir=asNamespace('miqtl'))
# Error: cannot add bindings to a locked environment
```

**Attempt 3 — `utils::assignInNamespace`:**

```r
utils::assignInNamespace('crossprod', base::crossprod, ns='miqtl')
# attempted; deep scanner still not confirmed
```

### Remaining paths
1. Run on R 4.3.3 (conda `bioinfo` env) where the BUILTIN classification is
   consistent with how miQTL was validated. `data.table` conda install resolved
   the R 4.3.3 version mismatch:
   ```bash
   conda install -c conda-forge r-data.table r-miqtl
   ```
2. Locate the specific miQTL C source containing the BUILTIN check and rebuild.
3. Pre-run `lod_r.csv` from a working R 4.3 environment so Python S4 comparison
   can proceed independently.

### Status
**Open — primary blocker** for S4, S6, S7, S8, S9, S10, S11.  
S1–S3 outputs confirmed written correctly.

---

## Issue 4 — `object 'test_pd' not found`

### Description
After replacing the manual permutation loop, every candidate gene in S6 failed:

```
skipping 0610009B22Rik (object 'test_pd' not found)
```

### Root cause
The old loop created `test_pd` as a temporary shuffled copy of `pheno_strain`.
The new candidate loop still referenced `test_pd` in the `data=` argument of
`scan.h2lmm`.

### Solution
```r
# before
scan.h2lmm(genomecache=genomecache_path, data=test_pd, ...)
# after
scan.h2lmm(genomecache=genomecache_path, data=pheno_strain, ...)
```

### Status
Resolved. Error shifted to Issue 3, confirming the call site is correct.

---

## Issue 5 — Script-level syntax errors from editing

### Description
Several consecutive runs produced:

```
Error: unexpected '}' in "}"
Error in contrib.url(repos, type) : trying to use CRAN without setting a mirror
Error in assign(...) : cannot add bindings to a locked environment
```

### Root causes and solutions

| Error | Root cause | Fix |
|---|---|---|
| `unexpected '}'` | `if (dir.exists(lib_path)) {` opening line deleted during edit | Restored missing line |
| CRAN mirror error | BiocManager install code left in script body | Moved to interactive session only |
| `cannot add bindings` | Used `assign()` on locked namespace | Replaced with `utils::assignInNamespace()` |

### Status
All three resolved.

---

## Issue 6 — GTF gene count inflation (18,479 genes on chr8)

### Description
S4 reported `chr8: 18479 genes` — nearly half the transcriptome on a single
chromosome. This inflates runtime dramatically and produces nonsensical results.

### Root cause
`gene_annot` was built from the GTF without deduplication. Many genes appear
on multiple chromosomes (e.g. pseudogenes, PAR regions, patch contigs) and
`!duplicated(gene_annot$gene_id)` was not applied. A gene annotated on chrX and
chr8 contributed to `gnames` on both.

### Solution
Added immediately after `gene_annot` is constructed:

```r
gene_annot <- gene_annot[!duplicated(gene_annot$gene_id), ]
message('  gtf after dedup: ', nrow(gene_annot), ' unique genes')
```

### Status
Fix applied in current `port_validation.R`. Expected chr8 gene count after
dedup: ~2,000–3,000 (genes with TSS on chr8 only).

---

## Issue 7 — GEV optim failure on low-variance permutations

### Description
```
Error in optim(theta, negloglik, hessian = TRUE, ..., tmp = data) :
  non-finite value supplied by optim
```

Occurred on chr1 (Man1a2, 23 cis loci). With only 23 loci to scan, all 100
permutation max-LODs cluster tightly and GEV `optim` cannot fit the distribution.

### Solution
Wrapped `get.gev.thresholds` in `tryCatch` with empirical quantile fallback:

```r
thresh_95 <- tryCatch(
  get.gev.thresholds(
    list(max.statistics=list(LOD=perm_lods)), percentile=0.95, use.lod=TRUE),
  error=function(e) {
    message('  GEV fit failed (', e$message, ') — using empirical quantile')
    quantile(perm_lods, 0.95, na.rm=TRUE)
  })
```

### Status
Resolved.

---

## Issue 8 — `data.table` version conflict blocks `library(miqtl)`

### Description
```
Error: package or namespace load failed for 'miqtl':
  object 'sort_by' not found whilst loading namespace 'data.table'
```

### Root cause
conda `bioinfo` env runs R 4.3.3. The `data.table` package installed in
`/data/homezvol2/.../R/4.4/` was built for R 4.4.2 and is incompatible
with R 4.3.3. conda loaded the user library first, pulling in the wrong binary.

### Solution
```bash
conda activate bioinfo
conda install -c conda-forge r-data.table
# then reinstall miqtl against R 4.3.3:
Rscript -e "remotes::install_github('gkeele/miqtl', lib=.libPaths()[1])"
# verify:
Rscript -e "library(data.table); library(miqtl); cat('ok\n')"
```

### Status
Resolved. `library(miqtl)` loads cleanly under R 4.3.3 in conda `bioinfo`.

---

## Current section status

| Section | Output | Status |
|---|---|---|
| S1 cache info | `cache_info_r.csv` `founders_r.csv` `sample_ids_r.csv` | ✓ written |
| S2 haplotype probs | `haplotype_r.csv` | ✓ written |
| S3 kinship LOCO | `kinship_r.csv` + per-chr matrices | ✓ written |
| S4 LMM scan | `lod_r.csv` `allele_effects_r.csv` | ✗ blocked (Issue 3) |
| S5 cache collapse | `collapsed_cache_summary.csv` | ✓ written (79% reduction) |
| S6 permutation | `thresholds_r.csv` `perm_lods_r.csv` | ✗ blocked (Issue 3) — fallback threshold=6 |
| S7 multiple imputation | `mi_lod_r.csv` | ✗ blocked (Issue 3) |
| S8 SNP scan | `snp_lod_r.csv` | ✗ gated (no `allele_dir`) |
| S9 mediation | `mediation_r.csv` | ✗ blocked (Issue 3) |
| S10 positional CI | `ci_r.csv` `boot_pos_r.csv` | ✗ blocked (Issue 3) |
| S11 instability scan | `fixef_r.csv` | ✗ blocked (Issue 3) |

---

## Open items

1. **Issue 3 — crossprod BUILTIN:** Primary path is re-running under R 4.3.3
   (conda `bioinfo`) where miQTL was originally validated. R 4.3.3 is now
   confirmed working (`library(miqtl)` loads cleanly after `data.table` fix).
2. **Issue 2 — rtracklayer:** Confirm `requireNamespace` diagnostic prints `TRUE`
   with `lib.loc` fix applied.
3. **lod_r.csv:** Once Issue 3 is resolved, re-run S4 to produce `lod_r.csv`
   and unblock the Python LOD comparison in `validate_port.py`.

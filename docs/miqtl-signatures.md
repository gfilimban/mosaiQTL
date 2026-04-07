---
name: miqtl-signatures
description: >
  Exact parameter names and rules for the 10 miQTL functions most commonly
  called in port_validation.R. Use this skill any time you write or debug a
  call to scan.h2lmm, generate.sample.outcomes.matrix, run.threshold.scans,
  run.positional.scans, generate.simple.sample.outcomes.matrix,
  instability.lm.scan, imputed.snp.scan.h2lmm, grab.locus.from.scan,
  get.gev.thresholds, or collapse.genomecache.
---

# miQTL Function Signatures

## Cardinal rules (apply to every function below)

1. **Stored-parameter rule**: `generate.sample.outcomes.matrix` stores `formula`,
   `model`, `K`, `pheno.id`, `geno.id`, and `weights` inside its return object.
   `run.threshold.scans` and `run.positional.scans` extract these internally and
   forward them to `scan.h2lmm`. **Never pass them again through `...`** — doing
   so duplicates arguments and crashes with `formal argument X matched by multiple
   actual arguments`.

2. **`...` is not free**: every extra arg you add to `run.threshold.scans` or
   `run.positional.scans` is forwarded to `scan.h2lmm`. If `scan.h2lmm` already
   received that arg from the stored object, it crashes.

3. **Subject alignment**: `run.threshold.scans` replaces outcome columns in `data`
   row-by-row from `perm_obj$y.matrix`. If `nrow(data) != nrow(perm_obj$y.matrix)`
   you get `not a multiple of replacement length`. Always filter `data` to
   `rownames(perm_obj$y.matrix)` before calling `run.threshold.scans`.

4. **`SUBJECT.NAME` (uppercase)**: `imputed.snp.scan.h2lmm` has no `pheno.id`/
   `geno.id` params — it always joins on a column literally named `SUBJECT.NAME`.

5. **`model.type='alt'` requires `just.these.loci`**: `generate.sample.outcomes.matrix`
   with `model.type='alt'` needs `scan.object$fit1` to exist, which only happens
   when the scan was run with `just.these.loci` pointing at a single locus.

---

## scan.h2lmm

```r
scan.h2lmm(
  genomecache,          # path to HAPPY cache dir
  data,                 # data.frame, must have pheno.id column
  formula,              # two-sided lm formula: outcome ~ covariates
  K          = NULL,    # LOCO kinship matrix (rownames = pheno.id values)
  model      = c('additive', 'full'),
  pheno.id   = 'SUBJECT.NAME',
  geno.id    = 'SUBJECT.NAME',
  use.multi.impute = TRUE,
  num.imp    = 11,
  chr        = 'all',
  just.these.loci  = NULL,   # ← set to single locus for model.type='alt' bootstrap
  return.allele.effects = FALSE,
  use.fix.par = TRUE,
  brute       = TRUE,
  seed        = 1,
  weights     = NULL,
  ...
)
# returns: list with $LOD, $p.value, $loci, $formula, $model.type,
#          $impute.map, $fit0, $fit1 (only if just.these.loci single locus),
#          $allele.effects (only if return.allele.effects=TRUE)
```

---

## generate.sample.outcomes.matrix

```r
generate.sample.outcomes.matrix(
  scan.object,          # scan.h2lmm() output
  model.type = c('null', 'alt'),  # 'alt' requires scan.object$fit1 (single locus scan)
  method     = c('bootstrap', 'permutation', 'subsample'),
  subsample.prop = 0.63,
  use.REML   = TRUE,
  use.BLUP   = FALSE,
  num.samples,          # required, no default
  seed       = 1
)
# returns: list with $y.matrix (nrow=n_subjects, ncol=num.samples),
#          $formula, $model, $K, $weights, $impute.map, $locus
# NOTE: $y.matrix rownames = subjects scan.h2lmm actually used (may be < nrow(data))
```

---

## run.threshold.scans

```r
run.threshold.scans(
  sim.threshold.object, # from generate.sample.outcomes.matrix()
  genomecache,
  data,                 # must be filtered to rownames(sim.threshold.object$y.matrix)
  use.multi.impute = TRUE,
  num.imp    = 11,
  chr        = 'all',
  just.these.loci = NULL,
  scan.seed  = 1,
  outcome.type = c('outcome', 'index'),
  keep.full.scans = TRUE,
  scan.index = NULL
  # NO formula, K, model, pheno.id, geno.id — stored in sim.threshold.object
)
# returns: list with $full.results (matrix loci×samples or list of scan objects),
#          $max.statistics (list with $LOD, $p.value)
```

---

## run.positional.scans

```r
run.positional.scans(
  sim.object,           # from generate.sample.outcomes.matrix(model.type='alt')
  genomecache,
  data,                 # filtered to rownames(sim.object$y.matrix)
  model      = c('additive', 'full'),  # ← OK to pass (explicit named param)
  use.multi.impute = TRUE,
  num.imp    = 11,
  use.par    = 'h2',
  brute      = TRUE,
  use.fix.par = FALSE,
  scan.seed  = 1,
  keep.full.scans = TRUE,
  alpha      = 0.05,
  do.augment = FALSE
  # NO formula, K, pheno.id, geno.id, chr — extracted from sim.object internally
  # chr is derived from sim.object$locus (the peak locus chromosome)
)
# returns: list with $full.results (list of scan objects, one per bootstrap),
#          $max.statistics
# IMPORTANT: iterate over $full.results, not top-level object
# each element: list with $loci, $LOD, $p.value
```

---

## generate.simple.sample.outcomes.matrix

```r
generate.simple.sample.outcomes.matrix(
  formula,              # lm formula (two-sided: outcome ~ covariates)
  data,                 # data.frame
  pheno.id   = 'SUBJECT.NAME',
  method     = c('bootstrap', 'permutation'),
  use.REML   = TRUE,
  num.samples,
  seed       = 1
  # NO scan.object, NO proportion, NO subsample.prop
)
# returns: same structure as generate.sample.outcomes.matrix output
```

---

## instability.lm.scan

```r
instability.lm.scan(
  simple.sample.object, # from generate.simple.sample.outcomes.matrix()
  genomecache,
  model      = c('additive', 'full'),
  seed       = 1,
  chr        = 'all',
  just.these.loci = NULL,
  use.ROP    = TRUE,
  num.imp    = 11,
  print.locus.fit = FALSE
  # NO data, NO phenotype, NO id — stored in simple.sample.object
)
# returns: list with $loci, $LOD (may be length 0 if no stable peak found)
# GUARD: check length(inst_res$loci) > 0 before write.csv
```

---

## imputed.snp.scan.h2lmm

```r
imputed.snp.scan.h2lmm(
  data,                 # MUST have column 'SUBJECT.NAME' (uppercase, no pheno.id param)
  formula,
  K,
  allele.dir  = NULL,   # path to dir containing chrN.alleles flat text files (see format below)
  genomecache,
  model       = c('additive', 'full'),
  chr         = 'all',
  just.these.loci = NULL,
  exclusion.freq  = .Machine$double.eps,
  use.fix.par = FALSE,
  brute       = TRUE,
  print.locus.fit = FALSE,
  use.progress.bar = TRUE,
  X.list      = NULL,   # pre-computed design matrices (reuse across scans)
  return.X.list = FALSE
  # NO pheno.id, NO geno.id
)
# allele file format: allele_dir/chrN.alleles — one file per chromosome
# 4 lines per SNP, parsed via: grep -w -A 3 'SNP_ID' chrN.alleles
#   line 1 : SNP_ID
#   line 2 : ref_allele\tf1\tf2\t...\tf8   (0/1 indicator per founder)
#   line 3 : alt_allele\tf1\tf2\t...\tf8
#   line 4 : NA\t0\t0\t...\t0             (filtered out internally)
# miQTL does: matrix(unlist(strsplit(raw[-1], '\t')), nrow=3, byrow=TRUE)[,-1]
#   → 3×8 matrix; [,-1] drops allele label col, leaving 3×8 numeric
#   → which.max per founder column assigns allele
#   → mapping.matrix (36×8, from straineff.mapping.matrix()) maps to diplotypes
# IMPORTANT: do NOT include SNP_ID on data lines — grep -w matches it in data lines
#   causing matrix parse failure (too many fields → non-conformable arguments)
# founder order: A=AJ B=B6J C=129S1J D=NODJ E=NZOJ F=CASTJ G=PWKJ H=WSBJ
# macOS grep -A adds leading spaces to context lines → strip with trimws() if patching
```

---

## grab.locus.from.scan

```r
grab.locus.from.scan(
  scan.object,          # scan.h2lmm() output
  chr = NULL            # if NULL, genome-wide peak; if chr string, that chr's peak
)
# returns: character(1) locus ID
# returns NA if all LOD are non-finite — always check !is.na(result)
```

---

## get.gev.thresholds

```r
get.gev.thresholds(
  threshold.scans,      # output of run.threshold.scans() — NOT generate.sample.outcomes.matrix()
  use.lod    = FALSE,   # TRUE = use LOD column; FALSE = use p-value column
  percentile = 0.95,
  type       = c('min', 'max')
)
# returns: scalar threshold value
```

---

## collapse.genomecache

```r
# Not called directly — patched via miqtl:::collapse.genomecache
collapse.genomecache(
  original.cache,
  new.cache,
  subjects       = NULL,
  criterion      = c('l2.norm', 'max.category'),
  model          = 'additive',
  proportion.tol = 0.02
)
# patch required before calling (see port_validation.R S6):
# fn_lines <- gsub('check.l2.norm(', 'miqtl:::check.l2.norm(', fn_lines)
# fn_lines <- append(fn_lines, 'test <- FALSE', after=target-1)
```

---

## Common crash → root cause → fix

| Error | Root cause | Fix |
|---|---|---|
| `formal argument "X" matched by multiple actual arguments` | Passing `formula`/`K`/`model`/`pheno.id`/`geno.id` to `run.threshold.scans` or `run.positional.scans` when already stored in sim object | Remove those args — only pass explicitly named params listed above |
| `number of items to replace is not a multiple of replacement length` | `nrow(data) != nrow(perm_obj$y.matrix)` | Filter: `data <- data[rownames(perm_obj$y.matrix), ]` |
| `Scan object does not include the alternative model` | `generate.sample.outcomes.matrix(model.type='alt')` on multi-locus scan | Re-run `scan.h2lmm` with single `just.these.loci=peak_locus` first |
| `$ operator is invalid for atomic vectors` | Iterating `boot_scans` directly (hits `$max.statistics` as atomic vector) | Use `boot_scans$full.results` for per-scan iteration |
| `'data' must be of a vector type, was 'NULL'` | `imputed.snp.scan.h2lmm` can't find `SUBJECT.NAME` column | Rename: `data$SUBJECT.NAME <- data$subject.name` |
| `arguments imply differing number of rows: 1, 0` | `instability.lm.scan` returned empty `$loci` | Guard: `if (!is.null(inst_res) && length(inst_res$loci) > 0)` |

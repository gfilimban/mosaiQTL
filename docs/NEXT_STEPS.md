# NEXT_STEPS.md — mosaiQTL
# Instructions for Claude Code. Work through these in order.
# Each task is self-contained and completable in one session.
# Latest commit: b1288ac | Repo: github.com/gfilimban/mosaiqtl

---

## TASK 0a — SPARC DATA SETUP (do before any validation)

Download sparcc_cache and generate sparc phenotypes. Required for both Phase 2 and Phase 3 validation.
No real experimental data needed — entirely reproducible from public repos.

```r
# install sparcc if not already installed
# remotes::install_github("gkeele/sparcc")

library(sparcc)

# download sparcc_cache genome cache (CC haplotype probabilities)
download.file(
    url     = 'https://github.com/gkeele/sparcc_cache/archive/master.zip',
    destfile = 'validation/sparcc_cache.zip'
)
unzip('validation/sparcc_cache.zip', exdir = 'validation/')
# rename: validation/sparcc_cache-master/ -> validation/sparcc_cache/

genomecache <- 'validation/sparcc_cache'

# simulate phenotypes — matches miQTL README.md example exactly
phenotypes <- sim.CC.data(
    genomecache     = genomecache,
    num.lines       = 40,
    num.sim         = 10,
    num.replicates  = 2,
    qtl.effect.size = 0.8,
    seed            = 42
)
saveRDS(phenotypes, 'validation/sparcc_data/phenotypes.rds')
```

Then convert to Python fixtures:
```bash
python validation/sim_sc.py \
    --sparc_h5 validation/sparcc_data/phenotypes.h5 \
    --sparcc_cache validation/sparcc_cache/ \
    --output tests/fixtures/data/sparc_cc40/ \
    --seed 42
```

Add to .claudeignore:
```
validation/sparcc_cache/
validation/sparcc_data/
tests/fixtures/data/sparc_cc40/
```

---

## TASK 0 — REPO MIGRATION (do first, unlocks everything else)

Migrate from flat layout to src/ layout. Run on HPC or local machine.

### 0a and 0b was already done, but kept for reference

```bash
# 0a. create directory structure
mkdir -p src/mosaiqtl/{genotype,qtl,models,io,viz}
mkdir -p goals/ docs/math/ config/ workflow/scripts modules/local benchmarks/baseline

# 0b. move all modules into src/
git mv mosaiqtl/genomecache.py src/mosaiqtl/genotype/
git mv mosaiqtl/hmm.py          src/mosaiqtl/genotype/
git mv mosaiqtl/kinship.py      src/mosaiqtl/genotype/
git mv mosaiqtl/nb.py           src/mosaiqtl/qtl/
git mv mosaiqtl/lmm.py          src/mosaiqtl/qtl/
git mv mosaiqtl/scan.py         src/mosaiqtl/qtl/
git mv mosaiqtl/permutation.py  src/mosaiqtl/qtl/
git mv mosaiqtl/mediation.py    src/mosaiqtl/qtl/
git mv mosaiqtl/epistasis.py    src/mosaiqtl/qtl/
git mv mosaiqtl/vqtl.py         src/mosaiqtl/qtl/
git mv mosaiqtl/snp_scan.py     src/mosaiqtl/qtl/
git mv mosaiqtl/ase.py          src/mosaiqtl/qtl/
git mv mosaiqtl/pipeline.py     src/mosaiqtl/
git mv mosaiqtl/plotting.py     src/mosaiqtl/viz/
git mv mosaiqtl/utils.py        src/mosaiqtl/
git mv mosaiqtl/__init__.py     src/mosaiqtl/

# 0c. create founders.py (extract from utils.py or create new)
# founders.py must define: FOUNDER_ORDER, FOUNDER_COLORS, F1_COLORS, STRAIN_ORDER
# and a function cc_strain_palette() that generates ancestry-aware CC colors

# 0d. create stub __init__.py files for each subpackage
touch src/mosaiqtl/genotype/__init__.py
touch src/mosaiqtl/qtl/__init__.py
touch src/mosaiqtl/models/__init__.py
touch src/mosaiqtl/io/__init__.py
touch src/mosaiqtl/viz/__init__.py

# 0e. update pyproject.toml
# [tool.setuptools.packages.find]
# where = ['src']

# 0f. verify
pip install -e '.[dev]'
python -c 'import mosaiqtl; print(mosaiqtl.__version__)'
pytest tests/unit/ -x -q

# 0g. place CLAUDE.md files (from docs/ in repo)
# Root: CLAUDE.md -> mosaiqtl/CLAUDE.md
# Package: src_mosaiqtl_CLAUDE.md -> src/mosaiqtl/CLAUDE.md
# Submodules: genotype_CLAUDE.md -> src/mosaiqtl/genotype/CLAUDE.md etc.

# 0h. commit
git add -A
git commit -m 'refactor: src/ layout + submodule structure + CLAUDE.md hierarchy'
git config user.email 'gfilimba@uci.edu'
make bundle
```

---

## TASK 1 — CREATE genotype/founders.py

Create `src/mosaiqtl/genotype/founders.py` with all strain constants.

Requirements:
- `FOUNDER_ORDER`: list of 8 founder strain names, canonical order
- `FOUNDER_COLORS`: dict mapping strain name to hex
- `F1_COLORS`: dict mapping F1 name to hex
- `STRAIN_ORDER`: founders + F1s + CC strains (alphanumeric for CC)
- `cc_strain_palette(cc_strains: list[str]) -> dict[str, str]`: returns ancestry-aware
  colors for CC strains based on predominant founder haplotype. Use a perceptually
  uniform sequential blend between the two dominant founder colors.
- All constants match exactly the values in root CLAUDE.md
- No other module defines any hex color string — they all import from here
- Tests: `tests/unit/test_founders.py`
  - test_founder_order_length — len == 8
  - test_founder_colors_keys_match_order
  - test_f1_colors_complete — 7 F1s
  - test_no_inline_hex_elsewhere — grep src/ for hardcoded hex outside founders.py
  - test_cc_palette_returns_dict
  - test_strain_order_founders_first

---

## TASK 2 — CREATE models/records.py

Create `src/mosaiqtl/models/records.py` with all result dataclasses.

Requirements (use `@dataclass` with `__slots__` for memory efficiency):
```python
@dataclass(slots=True)
class ScanResult:
    lod: np.ndarray      # (n_loci,)
    loci: list[str]
    pos_mb: np.ndarray   # (n_loci,)
    chr: str             # always str, never int

@dataclass(slots=True)
class VQTLResult:
    scan_result: ScanResult        # dispersion scan  <- NOT .vqtl_scan
    mean_scan_result: ScanResult   # mean eQTL scan   <- NOT .mean_scan
    phi_estimates: np.ndarray      # (n_subjects,)
    phi_method: str                # 'moments' or 'mle'
    gene_id: str

@dataclass(slots=True)
class FounderEffect:
    founder: str
    estimate: float
    se: float
    ci_lo: float
    ci_hi: float

@dataclass(slots=True)
class QTLResult:
    gene_id: str
    gene_name: str
    cell_type: str
    chr: str
    pos_mb: float
    lod: float
    lod_threshold: float
    p_value: float
    q_value: float
    is_cis: bool
    founder_effects: dict[str, FounderEffect]
    variance_explained: float
```
- Tests: `tests/unit/test_records.py`
  - test_vqtl_field_names — res.scan_result accessible, res.vqtl_scan raises AttributeError
  - test_scan_result_chr_is_str
  - test_slots_no_dict — assert not hasattr(ScanResult(...), '__dict__')

---

## TASK 3 — CREATE io/covariates.py

Create `src/mosaiqtl/io/covariates.py`.

Requirements:
```python
def build_covariate_matrix(
    sample_metadata: pd.DataFrame,   # user-supplied; must have sample_id, sex, batch, n_cells
    fixed_covariates: list[str] = ['sex', 'batch'],
    include_n_cells: bool = True,    # adds log10(n_cells) covariate
    n_expression_pcs: int = 3,       # 0 to disable; computed from pseudobulk if provided
    user_covariates: Optional[list[str]] = None,  # any extra numeric/categorical columns
    pseudobulk_counts: Optional[pd.DataFrame] = None,  # needed if n_expression_pcs > 0
) -> pd.DataFrame:
```
Rules:
- Categorical columns in user_covariates → one-hot encoded automatically
- log10(n_cells): computed from `n_cells` column in sample_metadata
- n_expression_pcs computed from pseudobulk_counts via PCA if provided; if None and
  n_expression_pcs > 0, raise ValueError with clear message
- Return DataFrame: index=sample_id (sorted), columns=covariate names, all float64, no NaN
- Tests: `tests/unit/test_covariates.py`
  - test_mandatory_covariates_present
  - test_user_covariate_included
  - test_categorical_onehot
  - test_nan_raises_missing_covariate_error
  - test_n_cells_log10_transformed
  - test_output_all_float64
  - test_output_no_nan
  - test_sample_order_deterministic

---

## TASK 4 — UNIT TESTS FOR EXISTING MODULES

Create missing unit tests. These modules have no test files yet:

### tests/unit/test_genomecache.py
- test_loci_count_matches_r
- test_hdf5_seek_per_locus — O(log L) access time
- test_allele_encoding_cc_calls_only — not founder calls
- test_loci_max_delta_expected_1237 — pseudomarker exclusion is correct
- test_no_pseudomarkers_in_cache

### tests/unit/test_hmm.py
- test_forward_backward_normalization — posterior sums to 1 per locus
- test_n_gen_1_for_cc — assert n_gen=1 default for CC
- test_ibs_resolved_probabilistically — IBS loci have non-trivial posteriors

### tests/unit/test_lmm.py
- test_fwl_equivalence — FWL result matches per-locus WLS within 1e-6
- test_lod_formula — LOD = (n/2)*log10(RSS0/RSS1) with ML denominator n
- test_z_matrix_shape — (418, 8) not (48, 8)

### tests/unit/test_nb.py
- test_pearson_residuals_shape — (n_strains,) not (n_cells,)
- test_stage1_uses_raw_counts — integer input required
- test_dispersion_shrinkage_reduces_variance

### tests/unit/test_kinship.py
- test_loco_excludes_chromosome
- test_kinship_symmetric
- test_kinship_psd
- test_woodbury_identity_accuracy — compare to direct inverse

### tests/unit/test_permutation.py
- test_permutes_expression_not_genotype
- test_fix_h2_matches_miqtl
- test_gev_fit_returns_scalar

### tests/unit/test_mediation.py
- test_qr_decomp_accuracy — mediation LOD r > 0.99 vs R reference on chr1

### tests/integration/test_pipeline.py
- test_discover_on_sparc_data — planted locus recovered in ≥8/10 sparc simulations
- test_covariate_passthrough — user covariate affects LOD profile
- test_results_deterministic — same seed produces identical LOD scores (DESIGN-07)
- test_lod_concordance_vs_miqtl — r² ≥ 0.999 vs miQTL R output on same sparc phenotype

### validation/sim_sc.py (NEW — sparc→NB count bridge)
- sparc_to_nb_counts(phenotypes_h5, sim_index, phi, n_cells_mean, seed) → (counts, strain_ids, true_locus)
- test_sparc_to_nb_signal_preserved — high-phenotype strains have higher NB mean
- test_sparc_to_nb_counts_integer — output dtype is int32
- test_true_locus_matches_sparc — returned locus == true_locus from phenotypes.h5

### tests/unit/test_design_constraints.py (DESIGN goals enforcement)
- test_no_inline_hex_in_src — grep src/ for hex patterns outside founders.py (DESIGN-01 / SWE-03)
- test_core_install_minimal_deps — import mosaiqtl with only numpy+pandas+scipy+anndata+h5py (DESIGN-01)
- test_derivation_files_exist — assert each file in docs/wiki/03_Algorithms/ exists for
  each qtl/ module (DESIGN-03): NB_GLM_Stage1.md, PQL_Woodbury.md, LOCO_Kinship.md,
  GEV_Permutation.md, HMM_Haplotypes.md
- test_provenance_header_in_all_outputs — every saved CSV has git_hash header (DESIGN-06)
- test_benchmark_baseline_committed — benchmarks/baseline/latest.json exists in git (DESIGN-04)

---

## TASK 5 — BENCHMARK BASELINE

Establish the first speed measurement. Run AFTER Task 0 (migration) is complete.

```bash
# Generate sparc validation data (R — run once)
Rscript validation/r_reference/generate_sparc_data.R
# outputs: validation/sparcc_data/phenotypes.rds + sparcc_cache/ (genome cache)

# Convert sparc output to Python fixtures
python validation/sim_sc.py \
    --sparc_h5 validation/sparcc_data/phenotypes.h5 \
    --sparcc_cache validation/sparcc_cache/ \
    --output tests/fixtures/data/sparc_cc40/ \
    --seed 42

# Establish baseline (commit this JSON — never overwrite)
pytest tests/benchmarks/ --benchmark-autosave \
    --benchmark-json=benchmarks/baseline/v0.2.0_$(date +%Y%m%d).json

# Symlink latest
ln -sf v0.2.0_$(date +%Y%m%d).json benchmarks/baseline/latest.json

# Run comparison vs all reference tools (SPEED-02 through SPEED-04)
# Note: tensorQTL and limix do not use haplotype probs — speed+memory only, not LOD r²
python -m mosaiqtl.bench.compare_tools \
    --tools mosaiqtl miqtl tensorqtl limix \
    --dataset tests/fixtures/data/sim_cc48/ \
    --chr 8 --n_genes 50 \
    --output benchmarks/results/comparison_$(date +%Y%m%d).json

# Update GOALS.md Current column for all SPEED and MEM goals
python goals/update_goals.py \
    --benchmark benchmarks/results/comparison_$(date +%Y%m%d).json

# Memory profile specifically for genome cache (MEM-03)
python -c "
import tracemalloc, mosaiqtl
tracemalloc.start()
cache = mosaiqtl.genotype.genomecache.GenomeCache(cache_path)
_, peak = tracemalloc.get_traced_memory()
print(f\'genome cache peak RSS: {peak/1e6:.1f} MB (target <500 MB)\')
"
```

Fill in GOALS.md SPEED-01 through SPEED-06 and MEM-01 through MEM-05 Current column.
Do NOT fill in a number until you have a measured result.

---

## TASK 5b — REWRITE port_validation.R AND validate_port.py ON SPARC DATA

Rewrite both validation scripts to use sparcc_cache + sparc phenotypes as data source,
and ensure ALL 10 miQTL functions from miqtl-signatures-SKILL.md are tested natively.

### Current coverage gaps (port_validation.R)
Source: R/ directory in miQTL GitHub (not miqtl-signatures-SKILL.md).

#### based on validation using gastroc data. need to be revalidated using sparc simulated data

**Core scan (R/scan.h2lmm.R + R/grab.locus.from.scan.R)**
| Function | Current status | Target |
|---|---|---|
| scan.h2lmm | ✅ tested (S4, S7, S9, S10, S11) | keep |
| grab.locus.from.scan | ✅ tested (S5, S10) | keep |

**Sampling / permutation (R/sampling.processes.R)**
| Function | Current status | Target |
|---|---|---|
| generate.sample.outcomes.matrix | ✅ tested (S5, S10) | keep |
| run.threshold.scans | ⚠️ BYPASSED — manual loop (RINT dimension bug) | test natively on sparc |
| run.positional.scans | ✅ tested (S10) | keep |
| generate.simple.sample.outcomes.matrix | ⚠️ BYPASSED — never called | add to S11 |
| instability.lm.scan | ⚠️ BYPASSED — manual workaround | test natively on sparc |
| get.gev.thresholds | ✅ tested (S5) | keep |

**SNP scan (R/imputed.snp.scan.h2lmm.R)**

| Function | Current status | Target |
|---|---|---|
| imputed.snp.scan.h2lmm | ✅ tested (S8) | keep |

**Cache ops (R/collapse.genomecache.R, R/convert.*.R)**
| Function | Current status | Target |
|---|---|---|
| collapse.genomecache | ✅ tested (S6) | keep |
| convert.qtl2.to.HAPPY | ❌ not tested | out of scope — sparcc_cache already in HAPPY format |
| convert.DOQTL.to.HAPPY | ❌ not tested | out of scope — mosaiQTL uses qtl2 path only |

**Fixed-effect QR scan (R/fixef.eQTL.scan.R)**

| Function | Current status | Target |
|---|---|---|
| scan.qr / pull.qr.from.compact | ❌ not tested | low priority — no kinship, used sparc-internally only |

**Kinship (R/calc.kinship.from.genomecache.with.DOQTL.R)**
| Function | Current status | Target |
|---|---|---|
| calc.kinship.from.genomecache.with.DOQTL | ❌ not tested | out of scope — requires DOQTL dep |

**Plotting (R/genome.plotter.R, R/allele.plotter.R, R/CC_plot.R)**
| Function | Python target | Status |
|---|---|---|
| genome.plotter.whole | viz/genome.py plot_lod_profile | ⚠️ partial — single scan only; needs multi-scan + hard_thresholds + mark_locus |
| genome.plotter.chr | viz/genome.py plot_lod_chr | ❌ MISSING |
| genome.plotter.region | viz/genome.py plot_lod_region | ❌ MISSING |
| allele.plotter.whole | viz/effects.py plot_founder_effects_genome | ❌ MISSING — genome-wide line plot per founder |
| genome.plotter.sparcc | viz/genome.py plot_lod_sparc | ❌ MISSING — validation viz: planted locus marked |
| single.sim.plot | viz/genome.py plot_lod_sparc | ❌ MISSING — wraps plot_lod_sparc per sim index |

These are captured in **TASK 5d** below, not in TASK 5b.

Priority for Task 5b: the 3 BYPASSED functions + the section-to-function map below.

### Pipeline isolation rule — no rpy2, no cross-language calls, ever

R and Python run as **fully independent parallel pipelines** sharing only files.
rpy2 is never imported in any validation script. No R objects are ever passed to Python
in-memory. The boundary is always a file on disk.

```
R pipeline:                         Python pipeline:
  sparcc_cache (HAPPY format)         sparcc_cache (same HAPPY format)
  + sparc phenotypes.rds              + sparc_phenotypes.h5
       ↓                                   ↓
  port_validation.R                   validate_port.py
  (miQTL functions)                   (mosaiQTL functions)
       ↓                                   ↓
  r_data/*.csv   ────────────────→   validate_port.py reads CSVs
```

**sparc phenotypes format conversion (Task 0a, R → HDF5 once):**

The sparc output is `phenotypes.rds` (R binary). Python never calls readRDS.
Instead, Task 0a includes a one-time R export to HDF5:

```r
# R — run once in Task 0a after sim.CC.data()
# converts phenotypes.rds to a format Python reads natively
library(rhdf5)
h5_out <- 'validation/sparcc_data/phenotypes.h5'
h5createFile(h5_out)

# phenotype matrix: (n_ind, n_sim) — sim.y.1..10
sim_cols <- grep('^sim\.y\.', names(phenotypes$data), value = TRUE)
h5write(as.matrix(phenotypes$data[, sim_cols]),
        h5_out, 'sim_y')                   # (n_ind, n_sim) float64

# subject IDs per simulation: (n_ind, n_sim)
id_cols  <- grep('^SUBJECT\.NAME\.', names(phenotypes$data), value = TRUE)
h5write(as.matrix(phenotypes$data[, id_cols]),
        h5_out, 'subject_ids')             # (n_ind, n_sim) string

# planted locus per simulation: (n_sim,)
h5write(phenotypes$locus,    h5_out, 'true_locus')    # (n_sim,) string
h5write(phenotypes$locus.pos$Mb, h5_out, 'true_pos_mb') # (n_sim,) float64

# simulation properties
h5write(phenotypes$properties$qtl.effect.size, h5_out, 'qtl_effect_size')
h5write(phenotypes$properties$num.lines,        h5_out, 'num_lines')
```

```python
# Python — reads phenotypes.h5 directly, no rpy2
import h5py, numpy as np

with h5py.File('validation/sparcc_data/phenotypes.h5', 'r') as f:
    sim_y        = f['sim_y'][:]           # (n_ind, n_sim) float64
    subject_ids  = f['subject_ids'][:].astype(str)  # (n_ind, n_sim)
    true_locus   = f['true_locus'][:].astype(str)   # (n_sim,)
    true_pos_mb  = f['true_pos_mb'][:]     # (n_sim,)
```

**CSV exchange for all S1–S11 comparison data — unchanged:**
R writes `r_data/*.csv` → Python reads with `pd.read_csv`. No format change.
The CSV precision concern for haplotype probs (S2) is irrelevant here because S2
compares Python HMM output against Python-loaded sparcc_cache directly, not R-exported
probs. R and Python both read the same sparcc_cache HAPPY files independently.

### R script rewrite: validation/r_reference/port_validation.R

Replace real data path with sparcc_cache. Test all bypassed functions natively:

```r
# data source — replace at top of script
genomecache_path <- 'validation/sparcc_cache'
# R still reads its own phenotypes.rds internally — this is R-only
pheno_df   <- phenotypes$data   # data.frame with SUBJECT.NAME.1..10 + sim.y.1..10
true_locus <- phenotypes$locus[1]  # planted QTL locus for sim 1

# S5 — test run.threshold.scans NATIVELY (not manual loop)
# The bypass existed because of a dimension bug with RINT phenotypes.
# sparc phenotypes are continuous — no RINT step — so native path should work.
perm_obj <- generate.sample.outcomes.matrix(
    scan.object = base_scan, method = 'permutation', num.samples = 100, seed = 1)
thresh_scans <- run.threshold.scans(
    sim.threshold.object = perm_obj,
    genomecache = genomecache_path,
    data = pheno_df[rownames(perm_obj$y.matrix), ],   # aligned per miqtl-signatures rule
    scan.seed = 1)
threshold <- get.gev.thresholds(thresh_scans, use.lod = TRUE, percentile = 0.95)

# S11 — test instability.lm.scan NATIVELY + generate.simple.sample.outcomes.matrix
# sparc phenotypes are strain means — OLS instability scan should converge
simple_obj <- generate.simple.sample.outcomes.matrix(
    formula = sim.y.1 ~ 1,
    data = pheno_df,
    pheno.id = 'SUBJECT.NAME.1',
    method = 'bootstrap',
    num.samples = 100,
    seed = 1)
inst_res <- instability.lm.scan(
    simple.sample.object = simple_obj,
    genomecache = genomecache_path,
    model = 'additive',
    seed = 1)
# guard: length(inst_res$loci) > 0 before writing CSV
```

### Python rewrite: validation/validate_port.py

Replace real data loading with sparc fixture loading. Add planted locus recovery check:

```python
# at top of script — reads phenotypes.h5, no rpy2
import h5py, numpy as np

with h5py.File('validation/sparcc_data/phenotypes.h5', 'r') as f:
    true_locus  = f['true_locus'][:].astype(str)   # (n_sim,) planted locus per sim
    true_pos_mb = f['true_pos_mb'][:]              # (n_sim,) position in Mb
    sim_y       = f['sim_y'][:]                    # (n_ind, n_sim) phenotype values
    subject_ids = f['subject_ids'][:].astype(str)  # (n_ind, n_sim)

# S4 pass criterion: peak locus within 5 Mb of true_locus in >=8/10 sims
# S6 pass criterion: mean_delta < 0.5 (now using native run.threshold.scans)
# S10 pass criterion: CI overlaps true_locus position
# S11 pass criterion: instability peak near true_locus (native instability.lm.scan)
```

### Section-to-function mapping after rewrite
Source: R/ directory in miQTL GitHub.

| Section | R source file | miQTL function(s) | Pass criterion |
|---|---|---|---|
| S1 | -- | -- (cache structure) | loci_max_delta == 1237 |
| S2 | -- | qtl2::calc_genoprob (not miQTL) | haplotype r > 0.90; HDF5 exchange |
| S3 | calc.kinship...DOQTL.R | mosaiqtl.calc_kinship (loco) | kinship r > 0.999; HDF5 exchange |
| S4 | scan.h2lmm.R | scan.h2lmm (ROP, use.multi.impute=FALSE) | LOD r2 >= 0.999; planted locus >=8/10 sims |
| S5 | sampling.processes.R | generate.sample.outcomes.matrix + run.threshold.scans (native) + get.gev.thresholds | mean_delta < 0.5 |
| S6 | collapse.genomecache.R | collapse.genomecache | LOD preservation r > 0.999 |
| S7 | scan.h2lmm.R | scan.h2lmm (MI, num.imp=11) | MI LOD r > 0.75 |
| S8 | imputed.snp.scan.h2lmm.R | imputed.snp.scan.h2lmm | SNP LOD r > 0.50 |
| S9 | scan.h2lmm.R | scan.h2lmm (mediation, two-scan) | mediation LOD r > 0.90 |
| S10 | sampling.processes.R | generate.sample.outcomes.matrix (alt) + run.positional.scans + grab.locus.from.scan | CI overlaps planted locus |
| S11 | sampling.processes.R + fixef.eQTL.scan.R | generate.simple.sample.outcomes.matrix + instability.lm.scan (native) | instability r > 0.90 |

### Dependency order
- Requires TASK 0a (sparc data) complete first
- Requires TASK 0 (migration) for correct import paths
- Does NOT require TASK 1–4 (independent of package restructure)
- Blocks TASK 6 (S6 fix) — fix S6 after native run.threshold.scans works

---

## TASK 5c -- CREATE src/mosaiqtl/cli.py
Latest bundle commit: b1288ac. Click CLI; no logic in cli.py -- each subcommand
delegates to exactly one primary function in nb.py or qtl/.

### make some for validation

```python
# src/mosaiqtl/cli.py
import click, logging
from pathlib import Path
logger = logging.getLogger('mosaiqtl')

@click.group()
@click.option('--verbose', is_flag=True, default=False)
def cli(verbose):
    'mosaiQTL: haplotype-based QTL mapping for multiparent populations.'
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(name)s %(levelname)s %(message)s')

@cli.command('fit-null')
@click.option('--counts',    required=True, type=click.Path(exists=True))
@click.option('--cell-type', required=True)
@click.option('--output',    required=True, type=click.Path())
@click.option('--shrink/--no-shrink', default=True)
@click.option('--seed', default=42, type=int)
def fit_null(counts, cell_type, output, shrink, seed):
    'Stage 1: NEBULA-LN null fit -> phi_shrunk, sigma2 per gene.'
    from .nb import fit_null_sc
    from .io.anndata_io import load_anndata
    ...

@cli.command('scan')
@click.option('--null-fit',    required=True, type=click.Path(exists=True))
@click.option('--genomecache', required=True, type=click.Path(exists=True))
@click.option('--covariates',  required=True, type=click.Path(exists=True))
@click.option('--output',      required=True, type=click.Path())
@click.option('--n-cores', default=1, type=int)
@click.option('--cis-only/--genome-wide', default=True)
def scan(null_fit, genomecache, covariates, output, n_cores, cis_only):
    'Stage 2: PQL NB-GLMM score scan -> ScanResult per gene.'
    from .nb import scan_nb
    ...

@cli.command('permute')
@click.option('--scan-result', required=True, type=click.Path(exists=True))
@click.option('--genomecache', required=True, type=click.Path(exists=True))
@click.option('--n-perms', default=1000, type=int)
@click.option('--output',  required=True, type=click.Path())
@click.option('--seed',    default=42, type=int)
def permute(scan_result, genomecache, n_perms, output, seed):
    'Permutation thresholds -> GEV fit on max-LOD distribution.'
    from .permutation import compute_permutation_threshold
    ...

@cli.command('effects')
@click.option('--scan-result', required=True, type=click.Path(exists=True))
@click.option('--genomecache', required=True, type=click.Path(exists=True))
@click.option('--output',      required=True, type=click.Path())
def effects(scan_result, genomecache, output):
    'Founder haplotype effects at significant loci (LRT mode).'
    from .nb import scan_nb_gene
    ...

@cli.command('report')
@click.option('--scan-result', required=True, type=click.Path(exists=True))
@click.option('--threshold',   required=True, type=float)
@click.option('--output-dir',  required=True, type=click.Path())
def report(scan_result, threshold, output_dir):
    'Summary tables + QC plots -> output_dir/.'
    from .viz import genome
    ...

def main():
    cli()
```

Add to pyproject.toml:
```toml
[project.scripts]
mosaiqtl = 'mosaiqtl.cli:main'
```

### Tests
- test_cli_fit_null_invokes_nb -- mock fit_null_sc; verify called with correct args
- test_cli_scan_invokes_scan_nb -- mock scan_nb
- test_cli_permute_invokes_threshold -- mock compute_permutation_threshold
- test_cli_subcommand_isolated -- each subcommand runnable with --help alone
- test_cli_no_argparse -- assert argparse not imported in cli.py (INVIOLABLE RULE)

### Dependencies
- Requires TASK 0 (migration) for correct import paths
- Requires TASK 1 (founders.py) for color constants in report
- Independent of TASK 5b

---

## TASK 5d -- PORT R/genome.plotter.R + R/allele.plotter.R TO viz/

Python equivalents of the miQTL plotting functions, using seaborn + founder palette.
All functions take ScanResult objects (not R scan.h2lmm lists). mosaiQTL conventions
enforced: FOUNDER_COLORS from genotype/founders.py, dpi=300, Carlito font, axes from zero.

### viz/genome.py additions

```python
def plot_lod_profile(
    scan_results: dict[str, 'ScanResult'],  # {label: ScanResult} — multi-scan overlay
    hard_thresholds: list[float] | None = None,  # horizontal lines (GEV thresholds)
    mark_locus: str | None = None,           # vertical line at planted/peak locus
    use_lod: bool = True,                    # False -> -log10(p)
    scale: str = 'mb',                       # 'mb' or 'cm'
    just_these_chr: list[str] | None = None,
    ax: 'Axes | None' = None,
) -> 'Axes': ...
# R equiv: genome.plotter.whole(scan.list, hard.thresholds, mark.locus, use.lod, scale)
# Key: scan_results is a dict not a list — labels become legend entries
# hard_thresholds drawn as horizontal dashed red lines (matches R default)
# mark_locus draws vertical dotted grey line at that locus position

def plot_lod_chr(
    scan_result: 'ScanResult',
    chrom: str,
    hard_thresholds: list[float] | None = None,
    mark_locus: str | None = None,
    use_lod: bool = True,
    ax: 'Axes | None' = None,
) -> 'Axes': ...
# R equiv: genome.plotter.chr(scan.object, chr, hard.thresholds, mark.locus)
# zoomed single-chromosome view; used in CI and instability validation

def plot_lod_sparc(
    scan_result: 'ScanResult',
    true_locus: str,               # planted locus from sparc phenotypes$locus
    sim_index: int,                # which simulation (for title)
    threshold: float | None = None,
    ax: 'Axes | None' = None,
) -> 'Axes': ...
# R equiv: single.sim.plot / genome.plotter.sparcc
# marks true_locus with vertical red dashed line
# used in Task 5b validation to confirm planted locus recovery
```

### viz/effects.py additions

```python
def plot_founder_effects_genome(
    scan_result: 'ScanResult',     # must have non-zero betas (use_lrt=True)
    scale: str = 'mb',
    just_these_chr: list[str] | None = None,
    ax: 'Axes | None' = None,
) -> 'Axes': ...
# R equiv: allele.plotter.whole(scan.object)
# 8 lines across genome — one per founder — colored by FOUNDER_COLORS
# y-axis = per-founder haplotype effect (beta); x-axis = genomic position
# requires scan run with use_lrt=True or return_allele_effects=True
# note: NB score test returns zero betas — must use LRT mode for this plot
```

### Tests
- test_plot_lod_profile_multi_scan -- two ScanResults in dict -> two lines on ax
- test_plot_lod_profile_hard_thresholds -- horizontal line at specified y
- test_plot_lod_profile_mark_locus -- vertical line at locus position
- test_plot_lod_chr_subset -- only chrom-specific loci plotted
- test_plot_lod_sparc_marks_true_locus -- vertical line at true_locus x-position
- test_plot_founder_effects_genome_8_lines -- exactly 8 lines, colored by FOUNDER_COLORS
- test_no_inline_hex -- all colors imported from genotype/founders.py (INVIOLABLE RULE)

### Dependencies
- Requires TASK 1 (founders.py) for FOUNDER_COLORS
- Requires TASK 0 (migration) for ScanResult import path
- Independent of TASK 5b, 5c

---

## TASK 6 — S6 PERMUTATION FIX (gastroc data note to be checked in sparc)

Fix `mean_delta=0.619 > 0.5` in Phase 2 S6 (permutation GEV instability).

Diagnosis steps:
1. Run `python validation/validate_port.py --sections 6 --verbose` and inspect
   per-chr threshold delta — which chromosomes are worst?
2. Check if GEV fit is failing on small per-chr permutation counts — likely cause
3. Try pooled max-LOD GEV (pool all chromosomes) vs per-chr GEV
4. Compare to R's `get.gev.thresholds(use.lod=FALSE)` output
5. If pooled fixes it: update `permutation.py` to pool by default; add flag
   `per_chr=False` as default, `per_chr=True` for legacy R comparison mode
6. Re-run: `python validation/validate_port.py --sections 6`
7. Pass gate: `mean_delta < 0.5`

---

## TASK 7 — S2 + S11 RERUN (gastroc data note to be checked in sparc)

Quick reruns for two validation sections with known fixes already in place.

```bash
# S2: R needs to export haplotype_vals_r.csv first
Rscript validation/r_reference/port_validation.R all Inf 2
python validation/validate_port.py --sections 2
# pass gate: r > 0.90 on actual haplotype probability values

# S11: chr-type bug already fixed, just needs rerun
python validation/validate_port.py --sections 11
# pass gate: r > 0.90
```

---

## TASK 8 — OBSIDIAN VAULT SCAFFOLD

Create `docs/wiki/` as an Obsidian vault with wikilink structure.

```
docs/wiki/
├── .obsidian/              <- Obsidian config (gitignore the workspace.json)
├── 00_Overview/
│   ├── README.md           <- vault index with [[wikilinks]] to all sections
│   └── DESIGN_AND_REFERENCES.md  <- your existing reference document
├── 01_Scope_and_Intention/
│   └── Scope.md            <- extract from ARCHITECTURE.md §1-3
├── 02_Data_Flow/
│   └── Pipeline.md         <- extract from ARCHITECTURE.md §4
├── 03_Algorithms/
│   ├── NB_GLM_Stage1.md    <- math derivation: NB-GLM null fit + score test
│   ├── PQL_Woodbury.md     <- math derivation: PQL iteration + Woodbury identity
│   ├── FWL_LMM_Stage2.md   <- math derivation: FWL vectorization (lmm.py validation ref only)
│   ├── LOCO_Kinship.md     <- math derivation: Woodbury identity
│   ├── GEV_Permutation.md  <- math derivation: GEV threshold fitting
│   └── HMM_Haplotypes.md   <- forward-backward algorithm for CC
├── 04_Validation/
│   ├── Phase2_S1_S11.md    <- current status table + per-section notes
│   └── Phase3_P1_P4.md     <- biological anchors + pass criteria
├── 05_Goals/
│   └── Goals_Tracker.md    <- mirrors goals/GOALS.md; links to GitHub milestones
├── 06_References/
│   └── Bibliography.md     <- annotated references from DESIGN_AND_REFERENCES.md
└── 07_HPC/
    └── HPC_Setup.md        <- cluster setup, Singularity, SLURM account
```

Key wikilinks to create:
- `[[NB_GLM_Stage1]]` ↔ `[[LOCO_Kinship]]` (LOCO kinship feeds Stage 2 PQL solver)
- `[[NB_GLM_Stage1]]` ↔ `[[PQL_Woodbury]]` (PQL mechanism + Woodbury identity)
- `[[PQL_Woodbury]]` ↔ `[[LOCO_Kinship]]` (Woodbury operates on LOCO K_m)
- `[[Phase2_S1_S11]]` ↔ `[[NB_GLM_Stage1]]` (validation context)
- `[[Goals_Tracker]]` ↔ `[[Phase2_S1_S11]]`, `[[Phase3_P1_P4]]`
Note: FWL belongs to lmm.py (Gaussian validation reference only) — not a Stage 2 wikilink.
      Replace FWL_LMM_Stage2.md with PQL_Woodbury.md in the algorithm docs.

`.gitignore` additions:
```
docs/wiki/.obsidian/workspace.json
docs/wiki/.obsidian/workspace-mobile.json
```

Note: Obsidian vault is FOR HUMANS. Do not import it into any CLAUDE.md.
Algorithm derivation files in `docs/wiki/03_Algorithms/` can be @-imported into
module CLAUDE.md files when Claude Code needs the math to implement correctly.
Example in qtl/CLAUDE.md:
```
## ALGORITHM SPEC
@docs/wiki/03_Algorithms/NB_GLM_Stage1.md
@docs/wiki/03_Algorithms/PQL_Woodbury.md
```

---

## TASK 9 — GITHUB ACTIONS CI

Create `.github/workflows/ci.yml`.

Requirements:
- trigger: push to main + all PRs
- jobs: quality (lint + tests + coverage) + workflow-validation (nextflow stub-run)
- coverage gate: `--cov-fail-under=90`  # corrected from GOALS.md SWE-01
- benchmark comparison: `--benchmark-compare=benchmarks/baseline/latest.json --benchmark-fail-max-delta=10%`
- upload benchmark artifact on every push to main
- on regression: open GitHub Issue with label `regression`
- HPC scaling job: manual trigger only (`workflow_dispatch`)

See `codepack_ref_guide_fixed.md` §CI/CD PIPELINE TEMPLATE for the full YAML.
Adapt: replace `[package]` with `mosaiqtl`, `[env_name]` with `bioinfo`.

---

## TASK ORDER SUMMARY

```
TASK 0 (migration)  → unlocks all other tasks
TASK 1 (founders)   → unlocks TASK 4 viz tests
TASK 2 (records)    → unlocks TASK 4 records tests
TASK 3 (covariates) → unlocks TASK 4 covariate tests
TASK 4 (tests)      → needed before TASK 5 benchmark baseline
TASK 5 (benchmark)  → fill GOALS.md speed columns
TASK 5b (port_validation rewrite) → native miQTL coverage on sparc; blocks TASK 6
TASK 5c (cli.py) → Click CLI entry points; independent of 5b
TASK 5d (viz/) → port genome.plotter.R + allele.plotter.R to Python
TASK 6 (S6 fix)     → VAL-06 goal (after 5b — needs native run.threshold.scans)
TASK 7 (S2+S11)     → VAL-S2, VAL-S10 goals
TASK 8 (obsidian)   → human knowledge base; parallel with any task
TASK 9 (CI)         → last; needs tasks 0-4 complete first
```

---

## NOTES FOR CLAUDE CODE

- Always run `git config user.email 'gfilimba@uci.edu'` before any commit
- Never commit HPC paths — set manually at top of each script
- Before /compact: run 'Update relevant CLAUDE.md with decisions from this session'
- After any architecture decision: update root CLAUDE.md first, then the module CLAUDE.md
- Speed regression detected? Check Amdahl bottleneck before optimizing anything

# mosaiQTL — Project Wiki

**Version:** 0.2.0  
**Repo:** `github.com/gfilimban/mosaiqtl` (private)  
**HPC path:** `/share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl`  
**Last updated:** April 2026 | commit `a9ad082`

This document is the single source of truth for the project. Read it at the start of every session before touching any code.

---

## 1. Scientific context

**Goal:** Map genetic variants that regulate gene expression at founder-haplotype resolution in Collaborative Cross (CC) mice.

**Data:**
- Phenotype: single-nucleus RNA-seq, gastrocnemius skeletal muscle, ~300,000 nuclei
- Cell types: type1, type2a, type2b, type2x myonuclei (pseudobulked per strain)
- Genotype: GigaMUGA array, HAPPY format via R/qtl2
- Design: 33 CC strains + 8 founders + 7 F1 hybrids (B6J as dam) = 48 unique genotypes, ~415 pseudobulked samples after QC
- Lab: Seyedam lab, UCI — IGVF project

**Top QTL candidates (LMM genome-wide scan):**

| Gene | Chr | Peak (Mb) | LOD | Founders | Biology |
|---|---|---|---|---|---|
| Piezo1 | 8 | ~122 | ~21.8 | CAST/J↑, PWK/J↑ | Mechanosensitive ion channel; GTEx v8 skeletal muscle eGene |
| Ccdc171 | 4 | ~84 | ~8.0 | NOD/J-specific | No prior muscle annotation |
| Pmis2 | 7 | ~85 | ~6.0 | Cell-type restricted | Only in pseudobulked fiber subtypes |

---

## 2. Package structure

```
mosaiqtl/              16 modules, ~5800 lines
  __init__.py          exports: scan_nb, scan_nb_gene, fit_null_nb_glmm_loco,
                                fit_null_sc, pseudobulk_sum, scan_h2lmm,
                                scan_multi_gene, ScanResult, build_snp_catalogue,
                                discover, characterize, resolve,
                                scan_epistasis, test_haplotype_combinations,
                                scan_conditional, scan_conditional_with_y
  genome_cache.py      read/write HAPPY genome cache (RDS, HDF5) — 813 lines
  nb.py                NB-GLMM primary model — 677 lines
  plotting.py          QC and results visualisation — 611 lines
  scan.py              Gaussian LMM scan engine — 526 lines
  pipeline.py          three-layer user API — 422 lines
  epistasis.py         pairwise GxG, conditional multi-signal — 411 lines
  snp_scan.py          SNP catalogue and imputed SNP scan — 393 lines
  hmm.py               forward-backward HMM — 335 lines
  vqtl.py              dispersion QTL scan — 329 lines
  kinship.py           van Raden kinship + LOCO + eigendecomp — ~200 lines
  lmm.py               LMM engine (eigendecomp, FWL) — ~200 lines
  mediation.py         QR LOD-drop mediation — ~150 lines
  utils.py             RINT, Z-matrix, align_samples — ~130 lines
  permutation.py       GEV permutation thresholds — ~120 lines
  ase.py               allele-specific expression (kallisto) — scaffolded
  pipeline.py          Layer 1/2/3 user API

docs/
  WIKI.md              ← this file
  METHODS.md           formal math for all 17 concepts (561 lines)
  POC_REPORT.md        proof-of-concept report, validation status
  PORT_VALIDATION_REPORT.md  S1–S11 plot inventory and inference (995 lines)
  ARCHITECTURE.md      package design and API reference
  QTL_LITERATURE_OVERVIEW.md  background literature

tests/
  test_utils.py        RINT, Z-matrix, align_samples (10 tests)
  test_lmm.py          LOD formula, betas shape (4 tests)
  test_nb.py           PQL tau REML, PQL null fit, score stats (9 tests)
                       23 total, all passing

validation/
  r_reference/
    port_validation.R  S1–S11 R reference generator (SLURM array)
    merge_chr_outputs.R  merge per-chr outputs after SLURM array
  phase2/              S1–S11 output dirs (r_data/, py_data/, plots/, logs/)
  phase3/              P1–P4 output dirs (py_data/, plots/, logs/)
  validate_port.py     Phase 2 runner — S1–S11 comparison
  validate_phase3.py   Phase 3 runner — P1–P4 novel extensions
  build_allele_dir.py  build miQTL allele_dir from GigaMUGA CSV

README.md  Makefile  setup.py  .gitignore
```

---

## 3. Three-layer analysis design

```
Layer 1 — QTL Discovery           pipeline.discover()
Layer 2 — QTL Characterisation    pipeline.characterize()
Layer 3 — Mechanistic Resolution  pipeline.resolve(modules=[...])
```

**Layer 1 models (choose one):**

| Model | Entry point | Kinship | Status |
|---|---|---|---|
| NB-GLMM + LOCO (primary) | `scan_nb(..., kinship_loco=K)` | PQL + Woodbury, per chr | ✅ implemented |
| NB GLM (fallback) | `scan_nb(..., kinship_loco=None)` | σ² inflation | ✅ implemented |
| Gaussian LMM | `scan_h2lmm(...)` | LOCO, FWL | ✅ implemented (miQTL port) |

**Layer 2:** `scan_vqtl()` (dispersion QTL), `mediation_scan_qr()` — ✅ implemented  
**Layer 3:** `scan_epistasis()`, `scan_conditional_with_y()`, `build_snp_catalogue()`, `ase.py` — ✅/scaffolded

---

## 4. Primary model — NB-GLMM with LOCO kinship

Implemented in `nb.py`. Two-stage design:

**Stage 1 — cell-level null (once per cell type):**
```python
fit_null = fit_null_sc(counts_cells, strain_ids, covariates_cells)
# → dispersion_use (EB shrunk), sigma_sample per gene
```
Uses native NEBULA-LN (`fit_null_sc_native`, `use_native=True` by default) —
no edgePython required. Design follows He et al. 2021 + Pachter 2026 edgePython.
Legacy edgePython path still available via `use_native=False`. Not a kinship RE. `sigma_sample` propagates to Stage 2 denominator inflation (fallback path only).

**Stage 2 — per-chromosome PQL null + score test:**
```python
kinship_loco = kinship.calc_kinship(genome_cache, method='loco')
results = scan_nb(counts_cells, genome_cache, strain_ids,
                  kinship_loco=kinship_loco, ...)
```

PQL per chromosome (Breslow & Clayton 1993; Chen et al. 2016 AJHG):
- Working weights: $W_i = \mu_i/(1+\phi\mu_i)$
- Working response: $z_i = (\log\hat\mu_i - \text{offset}_i) + (S_i-\hat\mu_i)/\hat\mu_i$
- REML 1D minimisation for $\tau$ (kinship variance component)
- GLS for $\hat\beta$, BLUP for $\hat u$, update $\hat\mu$. 3–5 iterations.
- Output: projection matrix $P = V^{-1} - V^{-1}X(X^\top V^{-1}X)^{-1}X^\top V^{-1}$

Score test at all $L$ loci vectorised:
$T_\ell = (H_\ell^\top r_\text{adj})^\top (H_\ell^\top P H_\ell)^{-1} (H_\ell^\top r_\text{adj}) \sim \chi^2_{J-1}$

**Three correctness checks (verify without reading Chen 2016):**
1. $W_i = \mu_i/(1+\phi\mu_i)$ matches NB variance function ✓
2. $P\mathbf{X} = 0$ — tested in `test_nb.py::TestWoodbuyExactness`
3. Woodbury $V^{-1}v$ matches explicit inversion at small $n$ to 1e-15 — tested in `test_nb.py`

**n=418, m=48.** $V \in \mathbb{R}^{418\times418}$ — never formed. Woodbury reduces all
linear algebra to $48\times48$ Cholesky. P is never stored; applied implicitly per locus.

**Note on $P$:** $P$ is an oblique projection in the $V$-metric. $P^2 = P$ only when $V = I$. The correct idempotency property is $P\mathbf{X} = 0$, not $P^2 = P$.

---

## 5. Gaussian LMM (miQTL port)

Entry point: `scan_h2lmm()` in `scan.py`. Used for validation and as the reference model in all phase 2 sections.

Key design points:
- Eigendecomposition of $K^{(-c)}$ cached once per chromosome, reused for all genes
- $h^2$ estimated once per gene under null via REML grid search (fix-par convention, matching miQTL `use.fix.par=TRUE`)
- **FWL vectorisation** (~100× speedup): project covariates once, batch all $L$ loci via `np.einsum`. Mathematically identical to per-locus WLS.
- LOD formula: $\text{LOD} = (n/2)\log_{10}(\text{RSS}_0/\text{RSS}_1)$, ML denominator $n$ (not $n-p$)
- **HMM difference from R:** Python uses marginal 8-state probabilities (forward-backward); R uses HAPPY 36-state diplotype probabilities. This causes S8 SNP scan r=0.55 — documented, not a bug.

---

## 6. Phase 2 validation — current status

Run with `python validation/validate_port.py` (uses checkpoint `.done` files — `--force` to rerun all).

| S# | Section | Status | Key metric |
|---|---|---|---|
| S1 | Cache info | ✅ | `founders_match=True  loci_max_delta=1237` |
| S2 | Haplotype probs | ⚠️ rerun | `prob_r=NA` — needs `haplotype_vals_r.csv` from R S2 |
| S3 | Kinship LOCO | ✅ | `mean_r=0.9998  max\|ΔK\|=1.98e-02` |
| S4 | LMM scan | ✅ | `60.2% pass  median_r=0.9730` |
| S5 | Cache collapse | ✅ | `py=91.4%  lod_preservation=0.9999` |
| S6 | Permutation | ❌ | `mean_delta=0.619` — GEV instability, per-chr variation |
| S7 | MI scan | ✅ | `r=0.8191` (threshold 0.75 for n_imp=11, independent seeds) |
| S8 | SNP scan | ❌ doc | `r=0.5509` — marginal vs diplotype model, expected |
| S9 | Mediation | ✅ | `r=0.9983` |
| S10 | Positional CI | ✅ | `overlap=True` |
| S11 | Instability | ⚠️ rerun | chr-type bug fixed; rerun pending |

**Genuine passes: 7/11.** S6 and S8 are documented failures, not regressions.

**Pending reruns:**
```bash
# generate R S2 reference (HPC):
Rscript validation/r_reference/port_validation.R all Inf 2
# then rerun Python S2 and S11:
python validation/validate_port.py --sections 2,11
```

**S4 note:** 39.8% fail rate concentrates at LOD < 1. See `s4_gene_tiers.png` — fail rate drops to ~0 for LOD > 3.

**S6 note:** `mean_delta=0.619` — GEV tail instability when pooling chromosomes with different cis-loci counts. Inspect `s6_ecdf.png` and `s6_threshold_scatter.png` for per-chromosome diagnosis.

**S8 note:** r=0.55 is the expected consequence of marginal (Python) vs diplotype (R) haplotype model. `s8_maf_concordance.png` tests this mechanistically — |ΔLOD| should increase at low MAF if the explanation is correct.

---

## 7. Phase 3 validation — current status

Run with `python validation/validate_phase3.py --cell_type type2x --n_genes 500` (HPC only — requires `gastroc_all_processed2.h5ad`).

| Section | Entry point | Pass gate |
|---|---|---|
| P1 scan_nb | `nb.scan_nb(..., kinship_loco=kinship_py)` | LOD right-skewed + Piezo1 LOD > 3 |
| P2 scan_vqtl | `vqtl.scan_vqtl()` | ≥1 gene returns result |
| P3 scan_epistasis | `epistasis.scan_epistasis()` | lod_additive > 0 |
| P4 scan_conditional | `epistasis.scan_conditional_with_y()` | signal1_lod > 0 |

**P1 now uses LOCO kinship** (`kinship_loco=kinship_py` wired at validate_phase3.py line 375).

---

## 8. Validation plot inventory (phase 2)

Each section produces separate files. Multipanel only when plots must be read together.

| Section | Plots | Key new additions |
|---|---|---|
| S1 | `s1_cache_info.png` | — |
| S2 | `s2_haplotype_heatmaps.png`, `s2_haplotype_diff_grid.png`, `s2_prob_comparison.png`, `s2_bland_altman.png`, `s2_calibration.png` | Bland-Altman, reliability diagram |
| S3 | `s3_kinship.png`, `s3_offdiag_scatter.png`, `s3_relatedness_distributions.png` | Off-diagonal scatter with CCC, diagonal/off-diagonal histograms |
| S4 | `s4_lod_scatter.png`, `s4_lod_residuals.png`, `s4_gene_tiers.png`, `s4_gene_summary.png`, `s4_pvalue_qq.png` (with λ_GC), `s4_founder_effects.png`, `s4_discrepancy_manhattan.png`, `s4_allele_effect_concordance.png` | Manhattan, allele effect CCC distribution |
| S5 | `s5_cache_collapse.png`, `s5_loci_spacing.png` | Spacing distribution |
| S6 | `s6_permutation.png`, `s6_ecdf.png`, `s6_threshold_scatter.png` | ECDF, threshold scatter |
| S7 | `s7_mi_scan.png`, `s7_lod_diff_violin.png` | Bias violin |
| S8 | `s8_snp_scan.png`, `s8_lod_profile_chr8.png`, `s8_maf_concordance.png` | LOD profile, MAF concordance |
| S9 | `s9_mediation.png`, `s9_lod_drop_distribution.png` | LOD-drop histogram |
| S10 | `s10_positional_ci.png`, `s10_width_scatter.png` | Width concordance scatter |
| S11 | `s11_instability.png`, `s11_peak_proportion.png` | Bootstrap calibration |

**Global plot standards:** `dpi=300, transparent=True`. Font Carlito/Calibri. All plot axes start from zero. Founder palette fixed (see `userPreferences`).

**Global helper functions** (in `validate_port.py`):
- `scatter_qc(ax, x, y, ...)` — scatter with identity line and Pearson r
- `ccc(x, y)` — concordance correlation coefficient (Lin 1989)
- `bland_altman(ax, x, y, ...)` — Bland & Altman 1986 method-agreement plot
- `ecdf_plot(ax, data_list, ...)` — empirical CDF for tail comparison

---

## 9. Known bugs and recurring patterns

### Recurring bug: variable scoping across checkpointed sections

Variables defined inside `if section(n, ...)` blocks are undefined when that section is checkpointed. **Fix:** hoist all cross-section variables before `section(1)`:

```python
# load R CSVs unconditionally — cheap, survives checkpointing
r_lod = load_r('lod_r.csv')      # used by S4, S5 (cis-loci), S6
r_ae  = load_r('allele_effects_r.csv')  # used by S4 allele concordance

if section(1, 'CACHE INFO'):
    ...
```

**Also:** initialize variables before their conditional definition block:
```python
lf = []; lc = []; r_fc = np.nan   # used in plot code outside the if block
if pheno_strain is not None and r_lod is not None:
    lf = [...]; lc = [...]; r_fc = ...
```

### Recurring bug: typo propagation in related variable names

When fixing a variable name (e.g. `_r_ae` → `r_ae`), grep all occurrences before declaring done:
```bash
grep -n '_r_ae' validation/validate_port.py
```
The original `_r_ae` bug had two occurrences on the same filter expression — fixing only one left a second crash.

### S2 row sum always 0.00 — expected

`probs.sum(axis=1) == 1.0` by HMM normalisation. The `max_dev=0.00` is a self-consistency check (Python produces normalised probs), NOT a comparison to R. The real S2 validation is `prob_r` from `haplotype_vals_r.csv`. Do not treat 0.00 as suspicious.

### P is an oblique projection, not orthogonal

The NB-GLMM projection matrix $P = V^{-1} - V^{-1}X(X^\top V^{-1}X)^{-1}X^\top V^{-1}$ satisfies $P\mathbf{X} = 0$ (annihilates covariate space) but $P^2 \neq P$ when $V \neq I$. This is correct. Do not add a test for $P^2 = P$.

---

## 10. Key design decisions (permanent record)

| Decision | Rationale |
|---|---|
| PQL NB-GLMM for kinship correction | Preserves NB count structure via working weights; $P^2=P$ only requires when $V=I$ — not needed here |
| Per-chromosome null fit (not genome-wide) | LOCO kinship varies by chromosome; correct null must exclude each chromosome |
| Direct Cholesky at $n=48$ | Eigendecomp trick unnecessary at this $n$; direct solve is faster and simpler |
| $\phi$ held fixed from Stage 1 EB shrinkage | Only $\tau$ estimated in Stage 2; reduces per-chr computation; stabilises low-expr genes |
| `kinship_loco=None` fallback | Preserves backward compatibility; σ² inflation still captures within-strain variance |
| LOD formula uses ML denominator $n$ | Matches miQTL; REML used only for $h^2$ and $\tau$ estimation |
| FWL vectorisation in LMM | ~100× speedup; mathematically identical to per-locus WLS — not an approximation |
| Marginal 8-state HMM (Python) vs HAPPY 36-state (R) | Marginal is exact under inbred-founder assumption; causes S8 r=0.55 (documented) |
| Separate phase2/ and phase3/ output dirs | Prevents validate_phase3.py from overwriting phase2 CSVs via same `_load_py()` |
| Cross-section R CSV pre-loading | Survives section checkpointing; `load_r()` is cheap (returns None if missing) |

---

## 11. HPC workflow

### Git sync (bundle-based)

```bash
# on local machine after any commit:
make bundle   # → mosaiqtl.bundle

# first time on HPC:
git clone mosaiqtl.bundle mosaiqtl --branch main

# updates:
git stash
git pull /path/to/mosaiqtl.bundle main --rebase
git stash pop   # rule: bundle wins for validate_port.py and port_validation.R;
                #       HPC wins for run_validation_slurm.sh
```

HPC-local files (gitignored, never committed):
- `validation/run_validation_slurm.sh`
- `validation/rerun_failed_chrs.sh`
- `validation/phase2/py_data/*.done` (checkpoint files)
- `validation/phase3/py_data/*.done`
- `mosaiqtl.bundle`

### Before pulling — protect R outputs

```bash
# R outputs are gitignored but the directory rename (port_validation/ → phase2/)
# means scripts now look in phase2/r_data/. Move outputs after first pull:
cp -r validation/port_validation/r_data/*.csv validation/phase2/r_data/
```

### Makefile targets

```bash
make bundle               # create mosaiqtl.bundle
make check                # syntax check all Python files
make test                 # run pytest (23 tests)
make validate-phase2      # run S1–S11
make validate-chr8        # quick test: chr8, 50 genes
make validate-section SECTIONS=2,11   # specific sections
make validate-force       # ignore checkpoints
make validate-phase3      # P1–P4 (HPC, requires .h5ad)
make validate-phase3-quick  # P1+P2 only, 100 genes
make clean                # clear phase2 outputs
make clean-phase3         # clear phase3 outputs
```

---

## 12. Environment and dependencies

**HPC conda env:** `bioinfo` (Python 3.13)  
**HPC path:** `/share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl`

**Core dependencies** (`setup.py`):

| Package | Min version | Used for |
|---|---|---|
| numpy | 1.21 | all numerical ops |
| scipy | 1.7 | eigendecomp, chi2 CDF, minimize_scalar |
| pandas | 1.3 | CSV I/O, data wrangling |
| matplotlib | 3.5 | all plots |
| seaborn | 0.12 | plot styling |
| statsmodels | 0.13 | multipletests (BH-FDR) in phase3 |
| edgepython | 0.1 | NB GLM (GPL-3.0, Pachter lab 2026) |

**Optional:**
- `h5py`, `scanpy`, `anndata` — for reading `.h5ad` files
- `rpy2` — for `genome_cache.from_rds()` and all validation scripts

---

## 13. Documentation map

| Question | Where to look |
|---|---|
| What does function X do mathematically? | `docs/METHODS.md` |
| What is the current validation status? | `docs/POC_REPORT.md` §Phase 2/3 |
| What does each plot show and why? | `docs/PORT_VALIDATION_REPORT.md` §Plot inventory |
| What is the overall architecture? | `docs/ARCHITECTURE.md` |
| What literature supports a design choice? | `docs/QTL_LITERATURE_OVERVIEW.md` |
| How do I sync to HPC? | §11 of this file, `docs/WIKI.md` |
| What bugs keep recurring? | §9 of this file |
| What decisions were made and why? | §10 of this file |

---

## 14. Planned extensions (not yet implemented)

| Extension | Module | Blocker |
|---|---|---|
| Permutation-based FDR (replace BH-FDR) | `permutation.py` | S6 GEV instability unresolved |
| Cell-level scan (no pseudobulk, SAIGE-QTL style) | new module | insufficient cell depth in current data |
| Multi-tissue extension | `pipeline.py` | data not yet available |
| sQTL | new module | junction matrix not yet built |
| Full PyPI publication | — | NB-GLMM phase 3 validation pending |

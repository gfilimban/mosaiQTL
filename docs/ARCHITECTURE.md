# docs/ARCHITECTURE.md — mosaiQTL v0.7.0
# Multiomics haplotype-based QTL mapping for CC/DO mice (mm39)
# Statistical model: LMM scan (primary) | NB-GLMM (planned Stage-2)
# Full citations: docs/references.md

---

## 1. WHAT IS MOSAIQTL

mosaiQTL maps expression QTL (eQTL) in Multiparent Populations (MPP) — specifically the
Collaborative Cross (CC) mouse panel — using pseudobulk single-cell RNA-seq data.
Genotypes are represented as continuous haplotype probability vectors at each locus,
one per founder strain (J=8 for CC/DO), computed by HMM forward-backward over
the GigaMUGA array [Morgan 2016].

The scan tests each locus using a Linear Mixed Model with LOCO kinship (see §4), producing
a LOD score per gene × locus. Significance is established by parametric permutation [Churchill
1994] with GEV tail extrapolation; FDR across eGenes uses Storey q-values [Storey 2002].
Sanger SNP fine-mapping collapses the haplotype signal to a credible set of imputed SNPs
per eGene via approximate Bayes factors [Wakefield 2009].

**Scope:** pseudobulk snRNA-seq eQTL in CC mice on mm39.
**Out of scope:** pseudobulk aggregation, proteomics, human GWAS, single-cell-level testing.

**Primary model:** LMM haplotype scan, validated against miQTL [Keele 2019] and R/qtl2
[Broman 2019]. NB-GLMM (nb.py) is fully implemented but not yet wired into the scan
pipeline; LMM runs in production today.

---

## 2. PACKAGE STRUCTURE

```
src/mosaiqtl/
  genotype/
    genomecache.py     HDF5-backed haplotype prob store; lazy per-chr I/O
    hmm.py             Forward-backward HMM (n_gen=1 for CC inbred)
    kinship.py         LOCO kinship build + eigendecomposition
    founders.py        ALL color/strain constants (CCPopulation singleton)
    snp_db.py          Sanger MGP VCF → alleles HDF5 (build_founder_allele_mgp)
    interpolation.py   Linear interpolation of haplotype probs to SNP positions
  qtl/
    lmm.py             Gaussian LMM — scan engine (validated port of miQTL)
    nb.py              NB-GLMM Stages 1+2 — PQL+Woodbury; primary future path
    scan.py            replicate-aware, MI, instability, fixef wrappers
    permutation.py     Parametric null + GEV threshold; fix_h2=True
    fdr.py             beta_calibrate_pvalue + storey_qvalue
    effects.py         Per-founder allele effects (fixef / ranef / BLUP)
    ci.py              LOD-drop support interval; ci_scan_multi_gene
    imputation.py      Haplotype → SNP dosage; ABF credible sets; SuSiE-RSS stub
    mediation.py       QR-based mediation scan; Woodbury rank-1 update
    epistasis.py       Pairwise GxG; conditional scan
    vqtl.py            Dispersion QTL (vQTL); VQTLResult
  io/
    anndata_io.py      AnnData read (backed=True; adata.raw.X for counts)
    covariates.py      build_covariate_matrix; build_cov_array; build_sample_geno_map
    results_io.py      CSV writers with provenance headers (ProvenanceRecord)
  models/
    records.py         ScanResult, VQTLResult, QTLResult, FounderEffect
    config.py          Population (frozen Pydantic), ScanConfig, PipelineConfig
  viz/
    genome.py          LOD profiles, Manhattan plots
    effects.py         Founder effect bars, coef_profile
    haplotypes.py      Haplotype heatmaps, epistasis grids
    celltypes.py       Cross-cell-type comparison plots
  pipeline.py          End-to-end: discover → characterize → resolve → call_egenes
                        → calibrate_egenes_storey → cap_egene_ci → finemap_egenes
  utils.py             RINT, Z-matrix, align_samples, chr_sort_key, build_tss_lookup
  cli.py               Click CLI (scan, merge, permute, call-egenes, finemap, report)
```

Key non-obvious locations:
- `ref/` — shared `igvf_cache.h5`, `igvf_kinship_loco.h5`, `sanger_alleles.h5`, `gene_tss.vm36.csv`
- `validation/validate_tissue.py` — tissue-agnostic Phase 2 validation script
- `validation/tools/` — convert_h5ad, build_sanger_alleles, plot_phase2, etc.
- `workflow/` — Nextflow DSL2 multi-cell-type pipeline; scripts in `workflow/scripts/`

---

## 3. STATISTICAL MODEL

### 3.1 LMM formulation

For each gene g and each locus ℓ, mosaiQTL tests:

**Null (H₀):**
```
  y = Xβ + u + ε
  u ~ N(0, τ²K_loco[c])
  ε ~ N(0, σ²I)
```

**Alternative (H₁):**
```
  y = Hα + Xβ + u + ε
```

Where:
- y is the RINT-transformed pseudobulk expression vector (n_samples × 1)
- H is the (n_samples × J) haplotype probability matrix at locus ℓ
- X is the covariate matrix (intercept + sex + batch + ...)
- K_loco[c] is the LOCO kinship for chromosome c (§4)
- τ², σ² are variance components estimated once per gene under H₀

The LOD score follows [Broman 2019]'s ML convention (denominator n, not n−p):
```
  LOD = (n/2) × log₁₀(RSS₀ / RSS₁)
```

This matches miQTL [Keele 2019] and R/qtl2 [Broman 2019]; deviating breaks the
Phase 1 validation gate (P1-S1 r²≥0.990).

### 3.2 Kinship rotation (spectral trick)

Eigendecomposition K = UΛUᵀ [Kang 2008] diagonalises the LMM covariance, converting it
to WLS with per-sample weights w_i = (σ² + τ²λᵢ)⁻¹ [Lippert 2011].
After rotating y, H, X by Uᵀ, all O(n³) inversions reduce to element-wise operations.
h² = τ²/(τ²+σ²) is estimated by Brent's method on the REML profile likelihood — once per
gene per chromosome, then fixed for permutations (§5).

### 3.3 FWL vectorization

Frisch-Waugh-Lovell theorem [FWL, in Lippert 2011]: partial regression of y on H after
projecting out X yields the same residual sum of squares as the full model. mosaiQTL
precomputes H_t = UᵀZH and the null residual e₀ = ỹ − X̃β̂ once per chromosome in
`precompute_chromosome`, then reuses them for every locus on that chromosome. This
amortizes the O(n×p²) covariate fit across all L loci — the key algorithmic advantage
over miQTL's per-gene loop [Keele 2019].

Resulting scan complexity: O(L × J²) per gene (Gram matrix G = H_perp^T H_perp + LOD),
compared to O(L × n²) in a naive per-locus LMM.

### 3.4 NB-GLMM (implemented; not yet in scan pipeline)

For overdispersed count data (snRNA-seq), the LMM Gaussian assumption inflates false
positives for lowly-expressed genes [Squair 2021]. `qtl/nb.py` implements a two-stage
approach informed by NEBULA-LN [He 2021]:

**Stage 1** (NEBULA-LN null): fit per-gene NB-GLMM null with per-subject random effects
using PQL [Breslow 1993]. Outputs Pearson residuals and per-gene dispersion φ.

**Stage 2** (PQL scan): scan haplotype effects on the working response using the
Woodbury identity to keep operations at the m×m kinship scale (m = n_strains ≪ n_cells).

nb.py uses ThreadPoolExecutor for n_cores>1; each thread opens its own HDF5 handle via
`GenomeCache._h5_path` to avoid handle sharing across threads.

---

## 4. LOCO KINSHIP

Kinship measures polygenic background — shared ancestry across the genome. Including
test-chromosome loci in the kinship estimate causes proximal contamination: the
test-locus haplotype signal leaks into the random effect, inflating τ² and suppressing
LOD [Listgarten 2012]. LOCO recomputes K excluding each chromosome under test.

```
K_loco[c]_{ij} = (1/L_{-c}) × Σ_{ℓ ∉ chr c} Σ_f H_{iℓf} H_{jℓf}
```

The eigendecomposition of K_loco[c] is computed once per chromosome in `decomp_kinship`
and stored as `{chrom: {'values': eigvals, 'vectors': eigvecs}}`. Every scan call
receives this precomputed decomposition — never recompute it inside a gene loop.

**PSD guard:** Floating-point accumulation over ~17,000 loci can push the smallest
eigenvalue slightly negative (e.g. −10⁻¹⁶). The guard uses a scaled tolerance
`psd_tol = max_eig × n² × machine_eps`; noise-level negativity passes, structural
singularity raises `KinshipSingularError` and the chromosome is skipped in permutation.

---

## 5. PERMUTATION THRESHOLD

### 5.1 Why parametric null, not label shuffle

Under non-zero kinship, shuffling phenotype labels between strains produces an
anti-conservative null: the shuffled phenotype still shares polygenic background with
the genotype [Churchill 1994]. mosaiQTL uses a parametric null:

```
  y_null ~ N(0, τ²K_loco[c] + σ²I)
```

where τ², σ² are the null variance components of the real phenotype. This generates null
draws that preserve the kinship structure while removing the QTL signal.

### 5.2 fix_h2=True — mandatory

h² is fixed at the real phenotype's null estimate for all permutations. Re-estimating
h² per permutation is statistically invalid under kinship (Abney 2015, cited in Keele
2019): the null h² estimate conflates polygenic signal with the scan signal. With
fix_h2=True, the same LOCO eigendecomposition is shared across all n_perms permutations,
enabling the vectorized `scan_chromosome_batch` path: one precomputation, all permutations
in a single pass.

### 5.3 Representative gene selection

Permutations use a single representative gene whose h² (null heritability) is closest to
the median across all scanned genes. This representative must be selected **globally**
(chrom=None), not chromosome-restricted: restricting to the scan chromosome biases the
h² median and makes the null non-representative for the multi-gene population. The
representative gene's chromosome naturally determines `perm_chrom` in L/C modes —
this is correct downstream, not a bug.

### 5.4 GEV tail fitting

With n_perms=1000, the max-LOD null distribution is well-approximated by a Generalised
Extreme Value (GEV) distribution. `get_gev_thresholds` fits GEV parameters (shape ξ,
location μ, scale σ) and extrapolates thresholds at α=0.05 beyond the observed quantile
range. Returns (None, None) on failure; callers fall back to empirical quantile.

Per-chromosome GEV params are written to `{tag}_perm_gev_params.json` by `run_permute.py`
and used in eqtl_stats.csv construction via `get_gev_thresholds` per chr file.

---

## 6. FDR — STOREY Q-VALUE

### 6.1 Two-stage eGene calling

**Stage 1 — `call_egenes`:** Coarse gate. Each gene's observed peak LOD is compared
against the genome-wide (or per-chromosome) GEV threshold. Genes above threshold proceed
to Stage 2.

**Stage 2 — `calibrate_egenes_storey`:** Per-gene p-values from `beta_calibrate_pvalue`
(per-gene Beta-approximated null, FastQTL-style) are fed into `storey_qvalue` [Storey
2002] to estimate π₀ (proportion of true null genes) and compute q-values:

```
  q(i) = π₀ × m × p(i) / rank(p(i))   (monotone enforced)
```

Under complete null, π₀→1 and q≈BH-q. When real signal is present, π₀<1 and q-values
are less conservative than BH [Benjamini 1995], recovering power. This fixes the earlier
invalid shared-null approach where π₀ climbed past 1 (observed: 2.58).

### 6.2 Per-gene calibration

`calibrate_gene_pvalue` runs a per-gene parametric permutation: each gene gets its own
h² estimate and its own cis-window null LOD distribution (via `scan_chromosome_batch`
with `loci_mask`). This is the statistically correct approach — a shared representative-gene
null is only valid for FWER thresholding, not FDR.

---

## 7. SANGER FINE-MAPPING

### 7.1 Stack overview

```
genotype/snp_db.py        build_founder_allele_mgp: Sanger MGP VCF → alleles HDF5
genotype/interpolation.py interpolate_haplotype_probs: haplotype probs at SNP positions
qtl/imputation.py         collapse_to_dosage: haplotype → biallelic dosage
                          fine_map_region: ABF credible sets; SuSiE-RSS stub
                          imputed_snp_scan_from_db: full-genome scan (opens HDF5 per call)
io/results_io.py          build_eqtl_finemapping_df / write_eqtl_finemapping
                          build_eqtl_finemapping_peak_df / write_eqtl_finemapping_peak
pipeline.py               finemap_egene / finemap_egenes
cli.py                    mosaiqtl finemap subcommand
```

### 7.2 ABF credible sets

For each eGene, imputed SNP dosages within the 1.5-LOD support interval are tested.
Approximate Bayes factors [Wakefield 2009]:

```
  log(ABF) = 0.5 × [log(σ²/(σ²+V)) + (z²V)/(σ²+V)]
```

where z is the dosage z-score, V is the prior variance (default 0.04 for cis-eQTL on
standardized phenotypes), σ² is the residual variance. Credible sets are formed by
including SNPs until their cumulative posterior inclusion probability (PIP) ≥ 0.95.

SuSiE-RSS [Wang 2020, Zou 2022] is wired but blocked on the optional `rss` dependency;
ABF is the production path.

### 7.3 Sanger VCF details

`sanger_alleles.h5` is built from `mgp_REL2021_snps_founders.rsID.vcf.gz` (Sanger MGP
REL2021, GRCm39-native, 7 non-reference founders, 33.5M SNPs). C57BL/6J is synthesised
as the all-REF column (absent from the VCF because it is the GRCm39 reference). The
`CC_VCF_SAMPLE_MAP` in `snp_db.py` translates VCF registry names (`A_J`, `NOD_ShiLtJ`,
...) to canonical founder names (`AJ`, `NODJ`, ...) matching `genotype/founders.py`.

Variant identifiers use SPDI notation [Holmes 2020]:
`{GRCm39_RefSeq_accession}:{pos_0based}:{ref}:{alt}`

---

## 8. GENOME CACHE

The genome cache (HDF5) stores haplotype probability arrays indexed by chromosome,
strain, and locus. `GenomeCache.from_hdf5` opens the file lazily; `get_chromosome(chrom)`
loads one chromosome's probabilities into RAM and returns them.

**Invariant (must never be violated):** Store `h5py.Dataset` references, never load
`grp['probs'][:]` at construction time. Full-genome eager loading violates the
constant-memory streaming requirement: at CC scale (n=418, 74k loci, 8 founders, float32)
the array is ~940 MB; loading it once per gene would exhaust RAM on any multi-gene run.

**Build pipeline:** GigaMUGA CSV → `from_gigamuga` (HMM forward-backward over n_gen=1
consensus map) → `to_hdf5`. Source: JAX GRCm39 reference files
(`gm_uwisc_v4.csv`, `GigaMUGA_genotypes.csv.gz`). See `docs/DATA_SOURCES.md` for the
full derivation chain.

**Locus collapsing:** `collapse_genomecache` merges redundant markers with nearly
identical probability vectors (L2 greedy scan). Not yet validated — P2-S5 gate requires
an uncollapsed cache.

---

## 9. PIPELINE DATA FLOW

```
Input
  ref/igvf_cache.h5          GigaMUGA haplotype probabilities (CC strains × loci × 8)
  ref/igvf_kinship_loco.h5   LOCO kinship eigendecompositions per chromosome
  ref/sanger_alleles.h5      Sanger MGP alleles per founder per SNP
  ref/gene_tss.vm36.csv      GENCODE vM36 strand-aware TSS positions
  {ct}/counts_lognorm.h5     Pseudobulk log-normalised expression (genes × samples)
  {ct}/exp_gene_info_*.csv   gene_id + gene_symbol + chr + start + end + strand
  metadata.csv               sample_id + strain + sex + dataset (+ optional covariates)

Stage 1 — scan_multi_gene (per chromosome)
  → lod_chr{c}.csv           (gene × locus LOD profiles; n_loci per chr)
  → peak_lod_chr{c}.csv      (peak LOD + allele effects + h2 per gene per chr)

Stage 2 — merge_results
  → peak_lod.csv             (genome-wide peak per gene; chr always str)

Stage 3 — permute (run_threshold_scans)
  → perm_max_lods.csv + perm_max_lods_chr{c}.csv
  → perm_gev_params.json

Stage 4 — call_egenes
  → eqtl_summary.csv         (eGenes: gene_id + gene_symbol + peak_mb + peak_lod + h2
                                + perm_n + perm_chr + chi2_haplotype + p_haplotype
                                + pval_perm + storey_q + fdr_method)

Stage 5 — bootstrap CI
  → bootstrap_ci.csv         (LOD-drop support interval per eGene: ci_lo_mb/ci_hi_mb)

Stage 5a — cap_egene_ci (L-mode only)
  → cap_egene_ci() clips ci_lo_bp/ci_hi_bp to ±window_mb from strand-aware TSS

Stage 6 — finemap_egenes
  → {tag}_eqtl_finemapping.csv       credible-set SNPs (EQTL_FINEMAPPING_COLS)
  → {tag}_eqtl_finemapping_peak.csv  lead SNP per eGene (EQTL_FINEMAPPING_PEAK_COLS)

Stage 7 — report (eqtl_peak + eqtl_stats)
  → {tag}_eqtl_peak.csv      EQTL_PEAK_COLS: haplotype peak per eGene (IGVF submission)
  → {tag}_eqtl_stats.csv     EQTL_STATS_COLS: GEV params + significance per eGene
```

**Output column schemas** (all defined in `pipeline.py`; never redefine inline):
`LOD_COLS`, `ALLELE_EFFECTS_BASE_COLS`, `PEAK_LOD_COLS`, `EGENE_SUMMARY_COLS`,
`MEDIATION_COLS`, `CI_COLS`, `EQTL_PEAK_COLS`, `EQTL_STATS_COLS`,
`EQTL_FINEMAPPING_COLS`, `EQTL_FINEMAPPING_PEAK_COLS`.

**Gene ID conventions:** `gene_id` is the Ensembl ID used as the join/loop key throughout.
`gene_symbol` is the human-readable display column. R validation scripts output `gene` —
rename immediately after read with `df.rename(columns={'gene': 'gene_id'})`.
Never patch the R scripts to use the Python naming.

---

## 10. NEXTFLOW PIPELINE

```
workflow/main.nf                     DSL2 entry point
modules/local/
  build_cache.nf                     mosaiqtl build-cache
  kinship_loco.nf  (chr array)       mosaiqtl estimate (LOCO eigendecomp per chr)
  lmm_scan_chr.nf  (chr array)       mosaiqtl scan (per-chr LMM haplotype scan)
  merge_results.nf                   mosaiqtl merge-results
  permute.nf                         mosaiqtl permute (parametric null + GEV)
  call_egenes.nf                     mosaiqtl call-egenes (Storey FDR)
  finemap_chr.nf   (chr array)       mosaiqtl finemap (Sanger ABF per chr)
  report.nf                          mosaiqtl report (tables + plots)
workflow/scripts/                    thin wrappers — all domain logic in package
config/
  default_params.yaml                production parameters
  parity_chr19_params.yaml           parity test: chr19, L-mode, type_ii_myonucleus
```

Design principle [Di Tommaso 2017, Ewels 2020]: each process is independently re-runnable;
`-resume` replays only changed stages. Nextflow's `celltypes_dir` → (CT × chr) grid
enables the full multi-cell-type scan without nested parallelism.

**Wrapper scripts must be thin.** Any domain logic (column renames, statistical choices,
scan_type routing) in `workflow/scripts/` creates divergence with `validate_tissue.py`
and is a maintenance hazard. Wrapper scripts call package functions; the package owns
the logic.

**Parity test:** `config/parity_chr19_params.yaml` — chr19, scan_type=L, window_mb=1.0,
type_ii_myonucleus, gastroc, next dataset. VT command and NF pipeline outputs must agree
on eqtl_summary.csv, permutation threshold, and bootstrap CI before any multi-chr run.

---

## 11. MPP GENERALIZATION

mosaiQTL is parametric on population structure. All population-specific configuration
lives in a `Population` instance (`models/config.py`); the scan engine, kinship builder,
and HMM are population-agnostic.

`Population` (frozen Pydantic model) encodes: `founders` list, `colors` dict (must cover
all founders), `n_gen`, `inbred`, `chromosome_names`, `genotyping_error_prob`.
`CCPopulation` in `genotype/founders.py` is the built-in CC default.

**Type A code (by design):** Population-specific constants like FOUNDER_COLORS, n_gen=1,
genotyping_error_prob=0.002 — these live in `CCPopulation` and must never be inlined
elsewhere in the engine.

**Type B code (leakage bugs):** Hardcoded CC assumptions in the engine — fixed on
encounter; never introduce new ones.

DO mice (heterozygous) require diplotype probabilities (outer product H_i ⊗ H_j); the
current scan assumes inbred (H_i = diplotype probability). DO support requires
`Population.inbred=False` and a diplotype-aware H construction, tracked as MPP-04/MPP-05.

---

## 12. VALIDATION FRAMEWORK

### Phase 1 (gate for all subsequent phases)
Minimal miQTL workflow mirror: ROP scan LOD r²≥0.990, MI scan LOD r≥0.75, GEV threshold
delta<0.5 LOD. Currently: **🟢 r²=0.9975 / r=0.9718 / delta=0.1543**.
`port_sim_validation.R` is ground truth — no tryCatch patches, no workarounds.

### Phase 2
Full 11-section R vs Python comparison on sparc simulated data. Currently: 🟢 9/9
runnable gates. P2-S5 (cache collapse) is N/A: sparcc_cache is pre-collapsed.

### Phase 4
NB vs LMM comparison on simulated single-cell data. All gates 🔴 not yet run.

### Tissue validation (validate_tissue.py)
Tissue-agnostic script replaces validate_gastroc.py. Outputs at
`out_dir/{scan_tag}/core/` for scan CSVs; `extended/`, `plots/core/`, `plots/extended/`.
No `s*_` file prefixes — stage numbers appear only in `.done` checkpoints.

---

## 13. DESIGN PRINCIPLES

| Principle | Implementation |
|---|---|
| Algorithmic innovation over hardware | Score test fits null once per gene, vectorises over all loci. FWL amortization eliminates per-locus X fit. |
| Constant-memory streaming [Melsted 2021] | Never materialise full (genes × loci) LOD matrix. Stream per-chromosome, merge on completion. |
| Lazy per-chromosome I/O | GenomeCache loads one chromosome at a time via `get_chromosome()`; `h5py.Dataset` refs, never eager `[:]`. |
| Composable CLI | Each subcommand reads/writes HDF5 or CSV; every stage independently re-runnable. |
| Math before code | Every algorithm has a derivation in `docs/math/` before implementation. |
| Reproducibility [Peng 2011] | `ProvenanceRecord` (git hash + seed + params) on every scan output. All results deterministic given same seed. |
| Measure before optimising | `scalene` + `pytest-benchmark` before any perf PR. Benchmark results committed to `benchmarks/`. |
| Minimal core deps | `numpy + pandas + scipy + anndata + h5py` only. GPU, R extras are opt-in. |
| Package-first | All domain logic in `src/mosaiqtl/`. Validation scripts and NF wrappers are thin callers. |

---

## 14. NAMING CONVENTIONS

**Dimension vocabulary** (deviating silently produces wrong results):

| Name | Meaning | Typical range |
|---|---|---|
| `n_cells` | Individual nuclei before pseudobulk aggregation | 100–10,000 per strain/CT |
| `n_samples` | Individual mice (rows of metadata.csv) | 80–418 |
| `n_strains` | Unique CC genotype labels (GenomeCache index) | 40–418 |
| `n_founders` | Haplotype founder strains (J) | 8 for CC/DO |
| `n_loci` | Markers in genome cache (pseudomarkers excluded) | ~74k total |
| `n_perms` | Permutation replicates | 100–5000 |

**Key distinction — `n_samples` vs `n_strains`:**
One CC001 mouse = one sample. CC001 as genotype appears once in the cache regardless of
how many mice were sequenced. Scan functions operate at strain level via the Z-matrix
(n_samples × n_strains). `sample_ids` passed to scan calls must be **strain-level IDs**;
use `meta_geno_map` (from `build_sample_geno_map`) to convert before every call.

**Array axis order** (mismatching causes silent wrong results):

| Array | Shape | Convention |
|---|---|---|
| Haplotype probs | `(n_loci, n_strains, n_founders)` | loci first; founders last |
| Kinship | `(n_strains, n_strains)` | strain × strain |
| Covariate matrix | `(n_samples, p)` | samples × predictors |
| LOD array | `(n_loci,)` | matches locus order in cache |
| Betas | `(n_loci, n_founders)` | sum ≈ 0 per locus |
| Null outcomes | `(n_strains, n_perms)` | strains × perms |

---

## 15. SCAN WINDOW MODES

Three scan window modes, controlled by `scan_window` parameter:

| Mode | Key | Requires | Use for |
|---|---|---|---|
| Genome-wide | `'G'` | nothing | Full discovery scan |
| Cis-chromosome | `'C'` | `gene_chr` | Gene's own chromosome only |
| Local | `'L'` | `gene_chr` + `gene_pos_mb` + `window_mb` | ±window_mb around TSS |

Default for `scan_h2lmm`/`scan_h2lmm_mi`: `'cis'` (synonym for `'C'`).
Default for `scan_multi_gene`: `'genome'` (`'G'`).

**L-mode CI cap:** In L-mode, the support interval is capped to ±window_mb of the
strand-aware TSS by `pipeline.cap_egene_ci()`. TSS source priority: `gene_tss_df`
(GENCODE vM36 strand-aware) → `gene_info_df` start/end/strand → `start_mb` column
→ `peak_mb` fallback.

---

## 16. KEY OUTPUT FILES

**`eqtl_summary.csv`** — one row per significant eGene

| Column | Meaning |
|---|---|
| `gene_id` | Ensembl gene ID (join key throughout) |
| `gene_symbol` | Human-readable gene symbol (display only) |
| `chr` | Chromosome of peak locus (always str) |
| `peak_mb` | Peak locus position (Mb) |
| `peak_lod` | Peak LOD score |
| `h2` | Null h² (polygenic background fraction) |
| `pval_perm` | Per-gene empirical p-value (Beta-calibrated) |
| `storey_q` | Storey q-value across eGenes |
| `cis_dist_mb` | Signed distance from TSS to peak (Mb; positive = downstream) |
| `near_tss_mb` | Physical distance to nearest TSS (unsigned Mb) |

**`{tag}_eqtl_finemapping_peak.csv`** — lead Sanger SNP per eGene (IGVF submission)

Key columns: `variant` (SPDI), `chr`, `pos` (integer bp, GRCm39), `ref_founders`,
`alt_founders`, `PIP`, `PIP_method`, `CS`, `effect_{founder}` × 8, `pval_perm`, `fdr`.

**`perm_max_lods.csv` / `perm_max_lods_chr{N}.csv`** — permutation null

Columns: `max_lod` per permutation replicate. Per-chromosome files used in
`calibrate_gene_pvalue` and GEV fitting in `eqtl_stats.csv`.

---

## 17. OPEN ISSUES

| Issue | Location | Priority |
|---|---|---|
| Phase 4 NB vs LMM gates not yet run | validate_phase4.py | HIGH |
| Full gastroc genome-wide scan pending HPC | validation/hpc/ | HIGH |
| TSS accuracy for −strand genes (tss_mb = end/1e6) | convert_h5ad.py, next run | MED |
| collapse_genomecache not validated (P2-S5 N/A) | genotype/genomecache.py | MED |
| SWE-11 EQTL_SCAN_COLS scan writer not yet wired | pipeline.py, cli.py | MED |
| DO diplotype model (MPP-04/05) | genotype/ | LOW |
| vQTL NB-GLMM dispersion score test (VQTL-01) | qtl/vqtl.py | LOW |

---

## 18. GLOSSARY

| Term | Meaning |
|---|---|
| LOD | `(n/2) × log₁₀(RSS₀/RSS₁)` — primary QTL evidence statistic |
| FWL | Frisch-Waugh-Lovell: partial regression vectorisation over loci |
| LOCO | Leave-One-Chromosome-Out kinship — prevents proximal contamination |
| PQL | Penalised Quasi-Likelihood — Gaussian approximation to NB-GLMM |
| GEV | Generalised Extreme Value — fitted to max-LOD null for threshold extrapolation |
| FWER | Family-Wise Error Rate — controlled by permutation threshold at α=0.05 |
| FDR | False Discovery Rate — controlled by Storey q-value [Storey 2002] |
| ABF | Approximate Bayes Factor [Wakefield 2009] — SNP fine-mapping credible sets |
| SPDI | Sequence Position Deletion Insertion — NCBI variant notation [Holmes 2020] |
| RINT | Rank-based Inverse Normal Transform — applied to phenotype before LMM |
| h² | Narrow-sense heritability: τ²/(τ²+σ²); fraction of variance from polygenic background |
| J | Number of founders (8 for CC/DO) |
| m | Number of strains (unique genotype labels; kinship is m×m) |
| n | Number of samples (observations; ≥m because strains can have replicates) |
| CC | Collaborative Cross — ~60 inbred mouse strains from 8 founders |
| DO | Diversity Outbred — outbred population from same 8 founders; requires diplotype model |
| MPP | Multiparent Population — any population from >2 founders (CC, DO, MAGIC, HS) |
| pseudobulk | Per-(sample × cell-type) count matrix summed from single-cell data |
| eGene | Gene with a significant QTL after multiple-testing correction |
| proximal contamination | Bias in h² when test-chromosome markers are in the kinship |
| kinship rotation | Diagonalising LMM covariance via K=UΛUᵀ → WLS with weights w_i=(σ²+τ²λᵢ)⁻¹ |
| amortization | Precomputing chromosome-constant H_t once; reusing for all genes on that chr |

---
*Citations: docs/references.md | Last updated: see git log*
*Canonical source: github.com/gfilimban/mosaiqtl/docs/ARCHITECTURE.md*

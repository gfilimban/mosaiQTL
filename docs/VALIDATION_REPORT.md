# mosaiQTL Validation Report

**Package**: mosaiQTL v0.2.0  
**Repo**: `github.com/gfilimban/mosaiqtl` (private)  
**Latest commit**: `fc96107`  
**Reference**: miQTL R package (`gkeele/miqtl`)  
**Data**: Gastrocnemius snRNA-seq, Collaborative Cross mouse panel (CC + founders + F1s, $n=48$ strains, $N=418$ nuclei), chr8 validation chromosome  
**Run command**: `Rscript validation/r_reference/port_validation.R 8 50` → `python validation/validate_port.py --chr 8`

---

## Validation overview

mosaiQTL is a full Python port and extension of the miQTL R pipeline for haplotype-based QTL mapping in multiparent populations. The 11-section validation framework cross-validates every major component of the Python implementation against miQTL R output, using chr8 pseudobulked snRNA-seq phenotypes and GigaMUGA genotype probabilities (HAPPY cache format).

**Pass criteria (global)**:
- Pearson $r > 0.999$ for continuous vectors (LOD, kinship)
- Maximum absolute difference $|\Delta| < 0.05$ for probability matrices
- Identical peak locus identity for all tested genes
- Relative error $< 10\%$ on all scalar summaries

---

## Module inventory

| Module | Lines | Status | Description |
|---|---|---|---|
| `genome_cache.py` | 785 | ✅ complete | Reads and indexes HAPPY-format haplotype probability caches. Supports HDF5 native format, `.rds` conversion via `from_rds`, and direct construction from GigaMUGA marker files via `from_gigamuga`. Core data interface for all scan modules. |
| `hmm.py` | 335 | ✅ complete | Pure Python forward-backward HMM for haplotype state inference. Computes posterior diplotype probabilities from GigaMUGA marker data given an 8-founder transition model. |
| `kinship.py` | 178 | ✅ complete | Computes LOCO (leave-one-chromosome-out) realized kinship matrices via $K = \frac{1}{2L}\sum_l H_l H_l^\top$, where $H_l$ is the $n \times J$ haplotype probability matrix at locus $l$. Caches eigendecompositions to avoid redundant computation across genes on the same chromosome. |
| `lmm.py` | ~300 | ✅ complete | Linear mixed model scan engine. Implements the Frisch-Waugh-Lovell (FWL) theorem vectorization: projects both $y$ and $H$ onto the column space orthogonal to covariates once per chromosome, eliminating the per-locus Python loop (~100× speedup). LOD formula: $\frac{n}{2}\log_{10}\!\left(\frac{\text{RSS}_0}{\text{RSS}_1}\right)$. |
| `scan.py` | 502 | ✅ complete | Top-level scan orchestration. Implements the Z-matrix architecture to natively expand haplotype matrices and kinship from genotype space ($n=48$ strains) to sample space ($N=418$ nuclei) at scan time — the key design advantage over miQTL. Also implements multiple imputation (MI), fixed-effect scan, and instability subsample scan. |
| `permutation.py` | 320 | ✅ impl, unvalidated | Permutation-based significance threshold estimation. Supports `fix_h2=True` mode (fixes polygenic variance component at null estimate, matching miQTL behavior) to reduce runtime. Fits GEV distribution to permutation max-LOD distribution for analytical threshold extrapolation. |
| `nb.py` | 252 | ✅ scaffold | Negative binomial GLM scan via edgePython (`glm_sc_fit()`). **Primary mosaiQTL model**: operates on raw pseudobulk counts rather than RINT-transformed values, avoiding the Gaussianity assumption of LMM. Full end-to-end NB GLM pipeline is the primary deliverable for PyPI publication. |
| `vqtl.py` | 329 | ✅ novel | Variance QTL (vQTL) scan. Tests whether founder haplotype affects expression *dispersion*, not just mean. Uses Levene-type statistic on squared residuals from the null LMM. Novel extension not present in miQTL. |
| `snp_scan.py` | 306 | ✅ impl, unvalidated | Imputed SNP scan. Reads founder-coded allele files (`chrN.alleles`, 9 lines/SNP) to construct per-SNP dosage vectors and tests each SNP for association within the haplotype-defined cis window. Refines haplotype-level eQTLs to credible causal variants. |
| `mediation.py` | 218 | ✅ impl, unvalidated | QR-based causal mediation analysis. For each candidate mediator gene, adds its expression as a covariate to the LMM and reports LOD drop ($\Delta\text{LOD} = \text{LOD}_\text{base} - \text{LOD}_\text{mediated}$). Positive drop = partial/full mediation; negative drop = confounding or amplification. |
| `plotting.py` | 611 | ✅ complete | Visualization suite. Generates LOD profiles, founder effect bar charts, haplotype dosage heatmaps, kinship heatmaps, and QC correlation/residual plots. Enforces project color palette (8 CC founders + 7 F1s) and font conventions. |
| `utils.py` | 204 | ✅ complete | Shared utilities: rank-normal inverse transform (RINT), Z-matrix construction (strain-to-sample expansion), covariate design matrix assembly, and gene annotation helpers. |

**Total**: ~4,300 lines Python

---

## 11-section validation status

| S# | Name | R output file(s) | R status | Python status |
|---|---|---|---|---|
| S1 | Cache info | `cache_info_r.csv` | ✅ pass | ✅ pass |
| S2 | Haplotype probs | `haplotype_r.csv` | ✅ pass | ⏳ pending |
| S3 | Kinship LOCO | `kinship_r.csv`, `kinship_mat_r_chr{N}.csv` | ✅ pass | ✅ pass ($r=0.9997$) |
| S4 | LMM scan | `lod_r.csv`, `allele_effects_r.csv` | ✅ pass | ⏳ pending |
| S5 | Permutation | `thresholds_r.csv`, `perm_lods_r.csv` | ⚠️ failing | ⏳ pending |
| S6 | Cache collapse | `collapsed_cache_summary.csv` | ✅ pass (79% reduction) | ⏳ pending |
| S7 | Multiple imputation | `mi_lod_r.csv` | ✅ pass | ⏳ pending |
| S8 | SNP scan | `snp_lod_r.csv` | ⚠️ needs rebuilt allele_dir | ⏳ pending |
| S9 | Mediation | `mediation_r.csv` | ✅ pass | ⏳ pending |
| S10 | Positional CI | `ci_r.csv`, `boot_pos_r.csv` | ⚠️ boot_pos all NaN | ⏳ pending |
| S11 | Instability | `fixef_r.csv` | ✅ pass (empty = correct) | ⏳ pending |

---

## Section descriptions

---

### S1 — Cache info

**What it validates**: The `genome_cache.py` reader correctly parses a HAPPY-format haplotype probability cache and reports the expected number of loci and positional range for a given chromosome.

**Scientific purpose**: The HAPPY cache is the central data structure for all downstream analyses. It stores 8-founder haplotype probability matrices $H \in [0,1]^{n \times J}$ at every GigaMUGA marker locus, where $J=8$ founders and $n=48$ strains. Verifying locus count and positional range confirms the cache was loaded without truncation, index offset errors, or chromosome misidentification.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `chr` | str | Chromosome (1–19, X) |
| `n_loci` | int | Number of marker loci in the HAPPY cache for this chromosome |
| `min_mb` | float | Physical position of first locus (Mb, mm39) |
| `max_mb` | float | Physical position of last locus (Mb, mm39) |

**Pass criterion**: chr8 → `n_loci ≈ 4848`, spanning 3.17–129.86 Mb.

---

### S2 — Haplotype probabilities

**What it validates**: The haplotype probability matrices loaded by `genome_cache.py` are numerically consistent with miQTL's R reader at both the marginal (summary statistics) and full-matrix levels.

**Scientific purpose**: Haplotype probabilities encode the probabilistic ancestry of each subject at each locus. They are the primary input to all association models. In the additive dosage parameterization, each subject's probability vector sums to 2.0 (representing two allele copies). Discrepancies here propagate into every downstream LOD score. The QC summary checks: (1) row sums equal 2.0, (2) maximum per-subject probability (haplotype certainty) is in a plausible range for GigaMUGA marker density, and (3) Python and R matrices agree within floating-point tolerance.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `chr` | str | Chromosome |
| `locus` | str | Locus identifier |
| `pos_mb` | float | Physical position (Mb) |
| `mean_rowsum` | float | Mean across subjects of $\sum_j H_{ij}$; should equal 2.0 |
| `min_rowsum` | float | Minimum row sum (should be 2.0) |
| `max_rowsum` | float | Maximum row sum (should be 2.0) |
| `mean_max_prob` | float | Mean of $\max_j H_{ij}$ per subject; indicates haplotype certainty |

**Scientific interpretation**:
- `mean_rowsum = 2.0`: required by the additive diploid dosage model
- `mean_max_prob ≈ 2.0`: near-certain haplotype assignment across subjects
- `mean_max_prob ≈ 0.25` ($= 2/8$): maximum uncertainty, all 8 founders equally likely
- Expected range for GigaMUGA: 1.4–1.9

**QC plot `haplotype_probs.png`**: Three-panel heatmap (Python / R / difference). X-axis: 8 founders (A–H); Y-axis: 48 subjects (founders → F1s → CC strains). Color encodes dosage 0–2. **Pass**: $\max|\Delta P| < 0.05$.

---

### S3 — Kinship LOCO

**What it validates**: The LOCO kinship matrix computed by `kinship.py` matches miQTL's R implementation, both at the summary statistic level and at the full 48×48 matrix level.

**Scientific purpose**: The LOCO kinship matrix $K_{-c}$ is the polygenic relatedness matrix computed from all chromosomes *except* chromosome $c$. It is used as the random effect covariance in the LMM to control for genome-wide population structure while preserving power to detect cis-eQTLs on chromosome $c$. The kinship matrix is computed as:

$$K_{-c} = \frac{1}{2L_{-c}} \sum_{l \notin c} H_l H_l^\top$$

where $L_{-c}$ is the number of loci outside chromosome $c$. Numerical agreement confirms that both implementations sum the same set of loci and apply the same normalization.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `chr` | str | Chromosome left out |
| `n` | int | Number of subjects (48) |
| `mean_diag` | float | Mean self-kinship (diagonal) |
| `mean_offdiag` | float | Mean pairwise kinship (off-diagonal) |
| `min_val` | float | Minimum kinship value |
| `max_val` | float | Maximum kinship value |

**Scientific interpretation**:
- Diagonal ≈ 0.91–0.99: expected for nearly fully inbred CC strains
- Off-diagonal ≈ 0.13: expected mean pairwise relatedness in an 8-way cross
- Founders: diagonal = 1.0, off-diagonal = 0.0 (fully inbred, unrelated by design)
- F1s: diagonal = 1.0; off-diagonal with B6J parent ≈ 0.5

**`kinship_mat_r_chr{N}.csv`**: Full 48×48 kinship matrix for leave-chr-$N$-out. Pass criterion: $r > 0.999$, $\max|\Delta K| < 0.05$ vs Python.

**QC plot `kinship_heatmap.png`**: Three-panel heatmap (Python / R / difference). Block structure: 8 founders form 8 isolated high-kinship blocks; CC strains show diffuse mosaic of shared ancestry.

---

### S4 — LMM scan

**What it validates**: The per-locus LOD scores and founder allele effects from `lmm.py` / `scan.py` match miQTL's `scan.h2lmm` R function.

**Scientific purpose**: The LMM association test is the core statistical engine. At each locus $l$, it fits:

$$y = \mu + H_l \beta + u + \epsilon, \quad u \sim \mathcal{N}(0, \sigma_g^2 K_{-c}), \quad \epsilon \sim \mathcal{N}(0, \sigma_e^2 I)$$

where $y$ is RINT-transformed pseudobulk expression, $H_l \in \mathbb{R}^{n \times J}$ is the haplotype dosage matrix, $\beta \in \mathbb{R}^J$ is the vector of founder allele effects, and $K_{-c}$ controls for polygenic background. The LOD score quantifies evidence for the alternative model ($\beta \neq 0$):

$$\text{LOD} = \frac{n}{2} \log_{10}\!\left(\frac{\text{RSS}_0}{\text{RSS}_1}\right)$$

using the ML denominator $n$ (not $n - p$), matching miQTL convention.

The **FWL vectorization** in mosaiQTL is the key performance contribution: both $y$ and $H_l$ are projected onto the null space of the covariate matrix $X$ once per chromosome, transforming the scan into a sequence of weighted regressions that can be batched without a Python loop (~100× speedup over naïve per-locus iteration).

**Key outputs**:

`lod_r.csv`:

| Column | Type | Meaning |
|---|---|---|
| `gene_id` | str | Gene name (from GTF `gene_name`) |
| `locus` | str | GigaMUGA marker ID |
| `chr` | str | Chromosome of locus |
| `pos_mb` | float | Physical position (Mb) |
| `lod` | float | LOD score at this locus |
| `p_value` | float | F-test $p$-value ($\text{df}_1 = J-1 = 7$, $\text{df}_2 = n - p - q$) |

`allele_effects_r.csv`:

| Column | Type | Meaning |
|---|---|---|
| `gene_id` | str | Gene name |
| `chr` | str | Chromosome |
| `locus` | str | Peak locus (argmax LOD) |
| `founder` | str | Founder letter A–H (single character; if locus names appear here, this is the `ae[pi,]` indexing bug) |
| `effect` | float | Regression coefficient $\hat{\beta}_j$: effect of one unit increase in founder-$j$ dosage on RINT expression |

**Scientific interpretation of allele effects**:
- $\hat{\beta}_j > 0$: more founder-$j$ haplotype → higher expression
- Sum of effects ≈ 0 (RINT centering)
- Large effect spread ($\max_j \hat{\beta}_j - \min_j \hat{\beta}_j > 0.5$) = strong allelic series
- Flat effects (all $|\hat{\beta}_j| \approx 0$) = weak or null QTL

---

### S5 — Permutation thresholds

**What it validates**: The permutation-based genome-wide significance threshold computed by `permutation.py` matches miQTL's `run.threshold.scans`.

**Scientific purpose**: A permutation threshold is generated by randomly shuffling the phenotype vector $y$ (breaking genotype-phenotype association) and recording the maximum LOD score across all cis loci for each permutation. The distribution of these max-LOD values is then fit to a generalized extreme value (GEV) distribution, which allows extrapolation to stringent significance levels (e.g., 95th or 99th percentile) from a manageable number of permutations ($n_\text{perm} = 100$–1000). The `fix_h2=True` mode fixes the polygenic variance component $\hat{h}^2$ at its null-model estimate across all permutations, reducing runtime without substantially biasing the threshold.

**Known issue**: miQTL's installed `run.threshold.scans` has a float-as-integer indexing bug (`nrow(data) != nrow(perm_obj$y.matrix)` after internal NA filtering). Current workaround: manual permutation loop replicating `run.threshold.scans` intent; fix is to align `data` rows to `rownames(perm_obj$y.matrix)` after `perm_obj` construction.

**Key outputs**:

`thresholds_r.csv`:

| Column | Type | Meaning |
|---|---|---|
| `percentile` | float | Significance level (0.95 = genome-wide 5%) |
| `lod_threshold` | float | LOD threshold at `percentile` |
| `gene` | str | Gene used for permutations |
| `chr` | str | Chromosome |
| `n_perms` | int | Number of valid permutations (`0` = fallback used) |
| `perm_mean` | float | Mean of permutation max-LOD distribution |
| `perm_sd` | float | SD of permutation max-LOD distribution |

**Sanity**: `n_perms = 0` → `lod_threshold = 6.0` is a hardcoded fallback, not a valid threshold. Real runs require `n_perms ≥ 100`.

`perm_lods_r.csv`:

| Column | Meaning |
|---|---|
| `perm_lod` | Per-permutation maximum LOD score across all cis loci |

**Scientific interpretation**: GEV fit to this distribution gives the analytical genome-wide threshold. Empirical 95th percentile is the fallback if GEV optimization fails.

---

### S6 — Cache collapse

**What it validates**: The `genome_cache.py` collapse utility (L2-norm redundancy removal) produces the expected locus reduction consistent with miQTL's `collapse.genomecache`.

**Scientific purpose**: Many adjacent GigaMUGA markers carry nearly identical haplotype probability vectors because they are in strong LD. Collapsing removes redundant loci where the L2 distance between haplotype vectors is below a tolerance ($\delta = 0.02$), retaining one representative locus per block. This reduces scan time proportionally to the collapse ratio without materially affecting QTL detection power, because redundant loci contribute no independent information to the likelihood. The collapsed cache is the preferred input for permutation scans and instability scans.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `chr` | str | Chromosome |
| `n_loci_orig` | int | Number of loci in original HAPPY cache |
| `n_loci_coll` | int | Number of loci after L2-norm collapse ($\delta = 0.02$) |

**Expected reduction**: ~79% genome-wide (94,251 → 19,756 loci for CC GigaMUGA data). Chr8 specifically: ~4848 → ~1020 loci.

---

### S7 — Multiple imputation

**What it validates**: The multiple imputation (MI) LOD scores from `scan.py` match miQTL's MI scan output.

**Scientific purpose**: The regression-on-probabilities (ROP) model uses continuous haplotype probabilities $H_l$ directly in the regression. Multiple imputation (MI) instead draws $m = 11$ random diplotype assignments per subject per locus from the posterior, runs a discrete-genotype regression for each draw, and averages the LOD scores. MI is more conservative than ROP because averaging over uncertainty shrinks LOD scores, but it is the theoretically correct Rubin's rules approach when haplotype probabilities are substantially uncertain. MI and ROP converge at high-confidence loci.

**Key outputs**: Same columns as `lod_r.csv`.

**Scientific interpretation**:
- MI LOD ≤ ROP LOD: expected (averaging over uncertainty is conservative)
- MI LOD constant within imputation blocks: expected (each imputation maps a genomic block to a single diplotype state, giving identical LODs within the block)
- MI preferred at low marker density; ROP preferred for speed when GigaMUGA confidence is high

---

### S8 — SNP scan

**What it validates**: The imputed SNP-level scan in `snp_scan.py` matches miQTL's `imputed.snp.scan.h2lmm` using the same allele directory.

**Scientific purpose**: The haplotype scan identifies QTL intervals at ~0.1 cM resolution, but does not pinpoint a causal variant. The SNP scan refines this by imputing per-SNP founder dosages from the GigaMUGA marker panel: for each SNP in the `allele_dir`, the 8-founder dosage encoding (0/1/2 per founder per SNP) is combined with haplotype probabilities to produce an imputed SNP dosage vector, which is then tested in the same LMM framework. A SNP with higher LOD than its flanking haplotype loci is a credible causal variant.

**Allele file format** (`chrN.alleles`): 9 lines per SNP — line 1 = SNP identifier, lines 2–9 = one founder dosage (0/1/2) per line, in A–H founder order.

**Known issue**: The previous `build_allele_dir.py` wrote all-zero dosages (encoding bug — nucleotide calls were not translated to 0/1/2 per founder). Rewritten version correctly maps nucleotide calls → per-founder dosages based on reference/alternate allele assignment. Re-run required before this section can be validated.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `gene_id` | str | Gene name |
| `snp` | str | SNP identifier (`rs*` or GigaMUGA marker ID) |
| `chr` | str | Chromosome |
| `lod` | float | Imputed SNP-level LOD |
| `p_value` | float | F-test $p$-value |

---

### S9 — Mediation

**What it validates**: The mediation LOD drop scores from `mediation.py` match miQTL's mediation scan output.

**Scientific purpose**: Mediation analysis asks whether the genetic effect of a QTL on gene $A$ is transmitted through gene $B$'s expression. The test is: fit the LMM for gene $A$ at its peak locus twice — once without any mediator, once with gene $B$'s RINT expression as an additional fixed-effect covariate. The LOD drop is:

$$\Delta\text{LOD} = \text{LOD}_\text{base} - \text{LOD}_\text{mediated}$$

A large positive drop indicates that controlling for gene $B$ explains most of the QTL signal → gene $B$ is a likely mediator of the genetic effect on gene $A$. This is the statistical basis for causal gene identification at an eQTL.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `gene_id` | str | Outcome gene (the one with the eQTL) |
| `mediator` | str | Candidate mediator gene |
| `chr` | str | Chromosome |
| `locus` | str | Cis locus |
| `pos_mb` | float | Locus position (Mb) |
| `lod_base` | float | LOD without mediator covariate |
| `lod_mediated` | float | LOD with mediator as fixed-effect covariate |
| `lod_drop` | float | $\Delta\text{LOD} = \text{LOD}_\text{base} - \text{LOD}_\text{mediated}$ |

**Scientific interpretation**:
- $\Delta\text{LOD} > 0$: partial or full mediation — genetic effect on `gene_id` is partly via `mediator` expression
- $\Delta\text{LOD} \approx \text{LOD}_\text{base}$: complete mediation — all genetic signal passes through `mediator`
- $\Delta\text{LOD} < 0$: adding the mediator *amplified* the LOD — `mediator` is correlated with the locus but acts as a confounder or reveals a suppressed signal; this is biology, not a bug

---

### S10 — Positional confidence interval

**What it validates**: The bootstrap-based positional CI from `scan.py` matches miQTL's `run.positional.scans` output.

**Scientific purpose**: The positional CI quantifies resolution of the eQTL localization. Bootstrap resampling of subjects ($B = 100$ replicates, sampling with replacement) produces a distribution of peak locus positions. The 2.5th–97.5th percentile interval across bootstrap peak positions gives the 95% CI for the true QTL position. Narrow CIs (< 1 Mb) indicate high LOD and informative haplotype block structure; wide CIs (> 5 Mb) indicate low LOD or ambiguous linkage.

**Key outputs**:

`ci_r.csv`:

| Column | Type | Meaning |
|---|---|---|
| `gene_id` | str | Gene |
| `chr` | str | Chromosome |
| `peak_locus` | str | Lead eQTL locus (argmax LOD on main scan) |
| `ci_lo_mb` | float | 2.5th percentile of bootstrap peak positions (Mb) |
| `ci_hi_mb` | float | 97.5th percentile of bootstrap peak positions (Mb) |
| `peak_lod` | float | LOD at peak locus |
| `n_boot` | int | Number of valid bootstrap samples |

**Sanity**: `ci_lo_mb = NaN` or `n_boot < 5` → bootstrap failed or produced too few valid scans.

`boot_pos_r.csv`:

| Column | Meaning |
|---|---|
| `boot_pos` | Peak position (Mb) per bootstrap replicate; NaN = scan failed for that replicate |

**Known issue**: `boot_pos` returning all NaN was traced to incorrect extraction from the `run.positional.scans` return structure. Must iterate `$full.results` list and extract `$LOD` matrix per replicate. Fix committed in `fc96107`; awaiting re-run.

---

### S11 — Instability scan

**What it validates**: The fixed-effect instability scan in `scan.py` matches miQTL's `instability.lm.scan`, which identifies QTL peaks that are reproducible across bootstrap subsamples of the data.

**Scientific purpose**: The instability scan addresses the concern that a QTL peak detected in the full dataset may be driven by a small number of influential subjects. It runs a fixed-effect LM scan (no kinship term, faster) on $B = 20$ bootstrap subsamples, each containing ~63% of subjects (sampling without replacement). The mean LOD across subsamples at each locus is reported. A locus with consistently high mean LOD across subsamples is considered *stable* (reproducible). The collapsed cache is used to reduce runtime.

**Key outputs**:

| Column | Type | Meaning |
|---|---|---|
| `gene_id` | str | Gene |
| `chr` | str | Chromosome |
| `locus` | str | Locus ID |
| `pos_mb` | float | Position (Mb) |
| `lod` | float | Mean LOD across $B$ bootstrap subsamples |

**Scientific interpretation**:
- High mean LOD = stable, reproducible signal — robust to subject-level influence
- Empty result (`loci` vector of length 0): no locus exceeded the instability threshold on chr8 for the tested gene. This is a valid biological outcome (chr8 has no stable peak for this gene), not an error. The scan itself ran correctly.

---

## Cross-section QC plots

| Plot | Description |
|---|---|
| `lod_correlation.png` | Scatter of Python vs R LOD for all gene×locus pairs. Line: $y = x$. **Pass**: $r > 0.999$. |
| `lod_residuals.png` | $\Delta\text{LOD} = \text{LOD}_\text{Python} - \text{LOD}_\text{R}$ vs locus index. **Pass**: residuals flat near 0. |
| `pvalue_qq.png` | Q-Q plot of $-\log_{10}(p)$ values. $\lambda_\text{GC}$ annotated. **Pass**: $\lambda_\text{GC} \approx 1.0$ for non-eGenes. |
| `gene_summary.png` | Left: histogram of per-gene Pearson $r$. Right: scatter of relative error by gene. Tiers: strict ($r > 0.999$, rel\_err $< 0.10$); lenient ($r > 0.99$, rel\_err $< 0.20$). |
| `haplotype_probs.png` | Python / R / difference heatmap. $n \times J$ dosage matrix. **Pass**: $\max|\Delta P| < 0.05$. |
| `kinship_heatmap.png` | Python / R / difference heatmap. $48 \times 48$ kinship matrix. Block structure: 8 founder blocks + CC mosaic. |
| `founder_effects.png` | Bar chart of $\hat{\beta}_j$ for each founder at peak locus per gene. Color: project founder palette. |

---

## Open bugs

| S# | Error | Root cause | Fix status |
|---|---|---|---|
| S5 | `not a multiple of replacement length` | `nrow(data) != nrow(perm_obj$y.matrix)` after internal NA filter in `perm_obj` | ⚠️ next fix: `data <- data[rownames(perm_obj$y.matrix), ]` after `perm_obj` creation |
| S8 | `subscript out of bounds` | Old `build_allele_dir.py` wrote all-zero dosages | ⚠️ new builder written; awaiting re-run with rebuilt allele_dir |
| S10 | `boot_pos` all NaN | `run.positional.scans` returns structured list; must iterate `$full.results` per replicate | ⚠️ fix committed in `fc96107`; awaiting re-run |
| S11 | Empty `inst_res$loci` | No stable peak on chr8 for tested gene | ✅ guarded; cosmetic — scan ran correctly |

---

## Key design decisions

| Decision | Rationale |
|---|---|
| **Native $N = 418$ samples** | Z-matrix expands kinship and haplotypes from genotype space (48 strains) to sample space (418 nuclei) at scan time. miQTL collapses to strain means, losing within-strain single-cell resolution. |
| **NB GLM as primary model** | LMM on RINT values is a validated port of miQTL; NB GLM on raw pseudobulk counts via edgePython is the mosaiQTL scientific model. NB respects the discrete, overdispersed nature of pseudobulk count data. |
| **FWL vectorization** | Frisch-Waugh-Lovell theorem: projecting $y$ and $H_l$ onto the covariate null space once per chromosome allows batch regression without a per-locus Python loop. Mathematically identical to per-locus WLS; ~100× faster. |
| **LOD formula uses $n$ (ML denominator)** | $\frac{n}{2}\log_{10}(\text{RSS}_0 / \text{RSS}_1)$, not $\frac{n-p}{2}$. Matches miQTL convention; slight inflation at small $n$ is accepted for cross-software compatibility. |
| **Parametric on $J$ founders** | No hardcoded `n_founders = 8`. Package generalizes to any multiparent population (DO, MAGIC, MPP). |
| **HPC paths never committed** | All committed code uses local Mac paths. HPC paths (`/share/crsp/lab/seyedam/gfilimba/`) are manual user updates before SLURM submission. |

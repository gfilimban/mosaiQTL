# mosaiQTL — Proof-of-Concept Report

**Repo:** `github.com/gfilimban/mosaiqtl` (private)  
**Scope:** Novel extensions beyond the miQTL validation port  
**Status:** Scaffolded / design-complete; pending end-to-end validation  
**Last updated:** April 2026 | commit `HEAD`

---

## Overview

The miQTL validation port (S1–S11) establishes mathematical equivalence with the
R reference. This report documents the novel mosaiQTL extensions beyond miQTL,
organized as three analysis layers. Users choose how far to go — Layer 1 alone
produces genome-wide eQTL maps; Layers 2 and 3 add characterization and mechanistic
resolution progressively.

```
Layer 1 — QTL Discovery           pipeline.discover()
Layer 2 — QTL Characterization    pipeline.characterize()
Layer 3 — Mechanistic Resolution  pipeline.resolve(modules=[...])
```

Each layer is implemented in `mosaiqtl/pipeline.py` as a standalone function.
Layer 3 is fully modular: users pass only the subsections their data support.
Each section below states the scientific motivation, the statistical model, the
implementation design, and the planned validation criteria.

---

## Shared Architectural Note — Z-Matrix

All models below operate parametrically on $n$ samples and $m$ genotyped strains
($n \geq m$). The Z-matrix $Z \in \{0,1\}^{n \times m}$ ($Z_{ik} = \mathbf{1}[\text{sample } i \text{ belongs to strain } k]$)
is built at scan time from `sample_ids` vs `genome_cache.sample_ids`. When $n = m$
and IDs match, $Z = I_m$ and all formulas reduce to the strain-mean case exactly.
No module hardcodes $n$ or $m$.

---

## Layer 1 — QTL Discovery

Implemented in `pipeline.discover()`. Users choose `model='nb'` (raw pseudobulk
counts, primary model) or `model='lmm'` (normalized expression, miQTL validation
port). Returns a `DiscoveryResult` with one `ScanResult` per gene.

See P1 and M-sections below for full model mathematics.

---

## Layer 2 — QTL Characterization

Implemented in `pipeline.characterize()`. Takes a `DiscoveryResult` and runs
selected modules: allele effect extraction at peak loci, dispersion QTL scan
(vQTL), and mediation analysis. Returns a `CharacterizationResult`.

See P2 below for vQTL model.

---

## Layer 3 — Mechanistic Resolution

Implemented in `pipeline.resolve(modules=[...])`. Users pass only the subsections
their data support. Each module is independent — none require the others.

```python
results = pipeline.resolve(
    disc,
    modules=['snp_catalogue', 'conditional_scan', 'ase', 'sqtl'],
    snp_catalogue_kwargs    = {'mgp_vcf': '/path/mgp.vcf.gz'},
    conditional_scan_kwargs = {'n_signals': 3},
    ase_kwargs              = {'f1_fastq_dirs': {...},
                               'founder_fastas': {...},
                               'gtf_path': '/path/gencode.gtf',
                               'tool': 'kallisto'},
    sqtl_kwargs             = {'psi_matrix': psi, 'junction_ids': jids},
)
```

| Module | Flag | Purpose | Min data required |
|---|---|---|---|
| SNP catalogue | `snp_catalogue` | Founder-divergent SNP candidates per QTL interval | MGP VCF |
| Conditional scan | `conditional_scan` | Multi-signal decomposition within 10 Mb blocks | Layer 1 result |
| ASE validation | `ase` | Allele-specific expression in F1 hybrids (kallisto) | F1 FASTQs + founder FASTAs |
| sQTL integration | `sqtl` | Splicing QTL scan on PSI phenotypes | PSI matrix (LeafCutter or SQANTI3) |

---

## P1 — NB GLM Scan (`scan_nb`)

### Purpose

The LMM assumes $y \sim \mathcal{N}$ after RINT. Pseudobulk snRNA-seq counts are
non-negative integers with overdispersion — RINT discards this structure. The NB
GLM is the natural primary model: it handles zero-inflation, library-size offsets,
and overdispersion natively, with no normality assumption.

---

### Model

**Null model** (covariates only):

$$Y_i \sim \text{NB}(\mu_i^0,\, \phi), \quad \log \mu_i^0 = X_i\,\beta_0 + \log s_i$$

where $X_i \in \mathbb{R}^p$ = intercept + batch + PCs, $s_i$ = library size offset,
$\phi$ = dispersion ($\text{Var}[Y_i] = \mu_i + \phi\mu_i^2$).

**Alternative model** at locus $\ell$:

$$\log \mu_i^\ell = H_i^\ell\,\gamma + X_i\,\beta + \log s_i$$

where $H_i^\ell = (Z H_m^\ell)_i \in \mathbb{R}^{J-1}$ are founder haplotype
dosages after Z-expansion (last founder dropped, linear constraint).

**NB log-likelihood:**

$$\ell(y \mid \mu, \phi) = \sum_i \Bigl[\ln\Gamma\!\bigl(y_i + \phi^{-1}\bigr) - \ln\Gamma\!\bigl(\phi^{-1}\bigr) + \phi^{-1}\ln\!\tfrac{\phi^{-1}}{\phi^{-1}+\mu_i} + y_i\ln\!\tfrac{\mu_i}{\phi^{-1}+\mu_i} - \ln\Gamma(y_i+1)\Bigr]$$

---

### Score Test (default, vectorised)

Null is fitted once per gene. Score statistic at locus $\ell$:

$$Q_\ell = U_\ell^\top I_\ell^{-1} U_\ell \;\sim\; \chi^2(J-1) \quad \text{under } H_0$$

$$U_\ell = H_\ell^\top r, \quad r_i = \frac{y_i - \hat\mu_i^0}{1 + \phi\hat\mu_i^0}$$

$$I_\ell = H_\ell^\top \operatorname{diag}(w)\, H_\ell, \quad w_i = \frac{\hat\mu_i^0}{1 + \phi\hat\mu_i^0}$$

$$\text{LOD}_\ell = \frac{Q_\ell}{2\ln 10}$$

All loci on a chromosome are evaluated simultaneously via `einsum` on
$H_\text{all} \in \mathbb{R}^{L \times n \times (J-1)}$. Complexity per gene:
$O(L \times n \times J^2)$ — analogous to FWL vectorisation in `scan_chromosome`.

**Difference from LRT:** $Q_\ell$ is asymptotically equivalent to the LRT statistic
but does not require fitting the alternative model. Exact when $n \to \infty$;
conservative for small $n$.

---

### LRT (optional, exact betas)

$$\text{LOD}_\ell = \frac{\ell_\text{alt}^\ell - \ell_\text{null}}{\ \ln 10\ }, \quad \text{LRT} = 2(\ell_\text{alt}^\ell - \ell_\text{null}) \;\sim\; \chi^2(J-1)$$

Alt model fitted per locus via `edgepython.glm_fit`. Returns log-scale founder
haplotype effects $\hat\gamma \in \mathbb{R}^{J-1}$. Complexity $O(L \times \text{IRLS} \times n)$
— use at peak loci only after score-test pre-screening.

---

### Dispersion Estimation

Estimated once per gene from the null model via edgePython's three-stage EB pipeline:

1. Common $\hat\phi_c$ across all genes (moment estimator)
2. Trended $\hat\phi_t(g)$ as function of mean expression
3. Tagwise $\hat\phi_g$ shrunk toward trend (WLEB)

$\hat\phi_g$ is fixed across all loci for gene $g$, matching the `glmQLFit`
convention in edgeR and avoiding $L$ separate dispersion fits.

---

### Implementation

| Function | Location | Status |
|---|---|---|
| `scan_nb()` | `mosaiqtl/nb.py` | ✅ scaffolded |
| `scan_nb_multi_gene()` | `mosaiqtl/nb.py` | ✅ scaffolded |
| `_nb_loglik()` | `mosaiqtl/nb.py` | ✅ implemented |
| `_score_stat_vectorised()` | `mosaiqtl/nb.py` | ✅ implemented |
| `estimate_gene_dispersion()` | `mosaiqtl/nb.py` | ✅ implemented |

**Pending:** end-to-end run on real pseudobulk data; comparison of NB LOD profiles
against LMM LOD profiles for Piezo1, Ccdc171, Pmis2.

---

### Planned Validation Criteria

| Check | Threshold |
|---|---|
| Score test $\approx$ LRT at peak loci (large $n$) | $|\Delta\text{LOD}| < 0.5$ |
| NB LOD rank-order agrees with LMM LOD rank-order across genes | Spearman $\rho > 0.7$ |
| $p$-value calibration under permuted null | $\lambda_\text{GC} \in [0.95, 1.10]$ |
| Dispersion estimates $\hat\phi$ match edgeR output | $r > 0.99$ |

---

## P2 — Dispersion QTL Scan (`vqtl.py`)

### Purpose

Test whether genetic variation at a locus affects not just the mean expression level
($\mu$-QTL) but the *variance* of expression across replicates. A dispersion QTL
(vQTL) at locus $\ell$ indicates that founder haplotype identity modulates
transcriptional noise.

---

### Model

For each strain $k$, observe replicate counts $\{Y_{ik}\}_{i=1}^{n_k}$.  
Per-strain dispersion: $\hat\phi_k$ from moment estimator or NB MLE.

**Null:** $\hat\phi_k \sim \log\mathcal{N}(\mu_\phi, \sigma^2_\phi)$ — common log-dispersion.

**Alternative at locus $\ell$:** $\log\hat\phi_k = H_k^\ell\,\delta + \epsilon_k$,
$\epsilon_k \sim \mathcal{N}(0, \sigma^2)$ — haplotype effects on log-dispersion.

$$F_\ell = \frac{(RSS_0 - RSS_1)/( J-1)}{RSS_1/(m - p - J + 1)} \;\sim\; F(J-1,\, m-p-J+1)$$

$$\text{LOD}_\ell = \frac{m}{2}\log_{10}\!\frac{RSS_0}{RSS_1}$$

**Difference from $\mu$-QTL:** response variable is $\log\hat\phi_k$ (per-strain
dispersion), not $Y_i$ (counts). Operates in strain space ($m$ observations), not
sample space ($n$).

---

### Implementation

| Function | Location | Status |
|---|---|---|
| `scan_vqtl()` | `mosaiqtl/vqtl.py` | ✅ scaffolded |

---

## P3 — Epistasis Module (`epistasis.py`)

### Purpose

Test whether combinations of founder haplotypes at two loci produce expression
levels not predictable from the sum of their individual effects — polygenic epistasis
in the Collaborative Cross.

---

### P3a — Pairwise Interaction Scan (`scan_epistasis`)

**Additive model** at loci $a$ and $b$:

$$Y_i = H_i^a\,\alpha + H_i^b\,\beta + X_i\,\xi + u_i + \varepsilon_i$$

**Interaction model**:

$$Y_i = H_i^a\,\alpha + H_i^b\,\beta + (H_i^a \otimes H_i^b)\,\gamma + X_i\,\xi + u_i + \varepsilon_i$$

where $H_i^a \otimes H_i^b \in \mathbb{R}^{(J-1)^2}$ is the row-wise Khatri-Rao product
of the two haplotype design matrices.

**Epistasis LOD:**

$$\text{LOD}_\text{epi} = \text{LOD}_\text{int} - \text{LOD}_\text{add}$$

Tested as $\chi^2((J-1)^2)$ via LRT approximation.

---

### P3b — Founder-Pair Heatmap (`test_haplotype_combinations`)

Hard-call founder assignment (argmax over $H_{if\ell}$) at two peak loci.
For each pair $(f_a, f_b)$, compute mean phenotype over strains assigned
to both. Returns a $J \times J$ DataFrame — visualises emergent polygenic
combinations that survive the additive decomposition.

---

### P3c — Multi-Signal Decomposition (`scan_conditional`, `scan_conditional_with_y`)

Two implementations with different input requirements:

**`scan_conditional`** (LOD-drop heuristic, no phenotype needed):
identifies non-overlapping peaks using a 5 Mb exclusion zone around each
already-found signal. Requires only a `ScanResult`.

**`scan_conditional_with_y`** (full BLUP conditioning, preferred):
at each step, fits the BLUP of the peak haplotype effect and subtracts it
from the phenotype residual before rescanning. Requires the original phenotype vector.

$$y_\text{res}^{(k)} = y_\text{res}^{(k-1)} - H_{\ell^*}^{(k)} \hat{\gamma}^{(k)}$$

where $\hat{\gamma}^{(k)} = \text{BLUP of haplotype effect at the }k\text{-th peak locus}$.

BLUP conditioning is more conservative than OLS residualization — it accounts
for uncertainty in $\hat{\gamma}$ and is the haplotype-space analogue of
SuSiE's iterative Bayesian stepwise selection.

---

### Implementation

| Function | Location | Status |
|---|---|---|
| `scan_epistasis()` | `mosaiqtl/epistasis.py` | ✅ implemented |
| `test_haplotype_combinations()` | `mosaiqtl/epistasis.py` | ✅ implemented |
| `scan_conditional()` | `mosaiqtl/epistasis.py` | ✅ implemented (LOD-drop) |
| `scan_conditional_with_y()` | `mosaiqtl/epistasis.py` | ✅ implemented (BLUP) |

---

## Layer 3 — Module Details

### 3a — SNP Catalogue (`snp_scan.py` + MGP overlay)

**Purpose:** Translate a haplotype QTL interval into a ranked list of candidate
causal SNPs. Given ~10 Mb CC haplotype blocks, this produces O(1000s) candidates
per interval — not single-variant resolution, but a filtered catalogue for
downstream functional prioritization.

**Method:** For each peak locus, query the Sanger MGP r9 VCF within `window_mb`
of the peak. For each SNP, compute imputed dosage:

$$d_i^{(s)} = \sum_{f=1}^{J} H_{if\ell^*} \cdot g_f^{(s)}, \quad g_f^{(s)} \in \{0,1\}$$

Filter to SNPs private to founders with the largest haplotype effects. Output:
ranked DataFrame of founder-divergent candidate SNPs per gene.

**Status:** `snp_scan.py` ✅ implemented; `build_snp_catalogue()` 🔲 planned.

---

### 3b — Conditional Scan (`scan_conditional`)

**Purpose:** Decompose multi-signal QTL intervals. CC blocks average ~10 Mb;
a single haplotype scan peak may contain several independent regulatory variants.
Forward stepwise conditioning separates them.

**Method:** Condition on the BLUP of the peak haplotype effect, rescan residuals
for secondary signal. Repeat up to `n_signals` times. BLUP conditioning (vs
hard peak-locus residual) reduces false secondary signals from LD within the block.
This is the haplotype-space analogue of SuSiE's iterative Bayesian stepwise
selection but operating on founder dosages rather than SNP dosages.

**Status:** `scan_conditional()` stub in `scan.py` 🔲 planned post-validation.

---

### 3c — ASE Validation (`ase.py`, kallisto|bustools)

**Purpose:** Confirm that a haplotype eQTL operates through cis-regulatory
allele-specific expression in the relevant cell type, using heterozygous F1
hybrid snRNA-seq as orthogonal evidence.

**Why F1 only:** CC strains are inbred (>98% homozygous) — no within-strain
heterozygosity. F1 crosses (e.g., B6CASTF1/J) carry exactly two haplotypes and
show informative ASE at every divergent locus between the parental founders.

**Method (kallisto|bustools):**

1. Build founder-specific transcriptomes from MGP pseudogenomes (SNPs/indels
   applied to mm39) using `kb ref` with the founder genome as d-list.
2. Concatenate dam + sire transcriptomes into a two-allele index, prefixing
   transcript names by allele of origin.
3. Run `kallisto bus` — pseudoalignment to the two-allele index.
4. `bustools sort` → `bustools capture` by allele prefix → `bustools count`
   per allele → two count matrices (dam / sire).
5. Compute allelic imbalance per gene:

$$	ext{AI}_g = rac{N_g^	ext{dam} - N_g^	ext{sire}}{N_g^	ext{dam} + N_g^	ext{sire}}$$

A gene with $|	ext{AI}| > 0$ at the haplotype QTL locus, in the cell types
where the eQTL was detected, is direct cis-regulatory evidence for the QTL.

**Why kallisto over CompMap:**
- Consistent with existing lab pipeline (F1 paper)
- Memory-frugal pseudoalignment vs full alignment scoring
- Native snRNA-seq support via `bustools nac` workflow
- Works natively for multi-sample F1 crosses via barcode demultiplexing

**Precision note:** k-mer discrimination is reliable for highly divergent founders
(CAST/J, PWK/J); may have lower allele assignment accuracy for classical founders
(e.g., 129S1/J vs B6/J) that share high sequence identity. Report allele
assignment rate alongside AI for quality control.

**Status:** `ase.py` ✅ scaffolded; end-to-end test pending F1 FASTQ access.

---

### 3d — sQTL Integration (`pipeline.resolve`, `sqtl` module)

**Purpose:** Map splicing QTLs using the same haplotype scan engine with PSI
(percent-spliced-in) as the phenotype. Non-overlapping with eQTL (~75% of sQTL
genes are missed by gene-level analysis). Long-read data enables isoform-resolved
sQTL assignment via SQANTI3.

**Method:** PSI is bounded $[0,1]$ — RINT + LMM is appropriate; NB GLM is not.
`pipeline.resolve(sqtl_kwargs={'psi_matrix': psi})` calls `scan_h2lmm` with PSI
phenotypes directly. No new statistical machinery — only a new phenotype input.

**Short-read sQTL:** LeafCutter junction PSI → same scan engine.  
**Long-read sQTL:** SQANTI3 isoform fractions → transcript-eQTL (teQTL) mapping.

**Status:** `_run_sqtl()` in `pipeline.py` ✅ scaffolded; requires PSI matrix.

---

## Open Items

| # | Item | Layer | Blocking? |
|---|---|---|---|
| 1 | Run `scan_nb` end-to-end on real pseudobulk (Piezo1 chr8) | 1 | yes |
| 2 | Compare NB LOD vs LMM LOD for same gene set | 1 | yes |
| 3 | Confirm edgePython `fit['fitted.values']` key | 1 | yes |
| 4 | Permutation null for NB LOD thresholds | 1 | yes |
| 5 | K-expansion (n > m) end-to-end test | 1 | yes |
| 6 | `scan_conditional()` + `scan_conditional_with_y()` | 3b | ✅ implemented |
| 7 | Implement `build_snp_catalogue()` (Layer 3a) | 3a | post-validation |
| 8 | End-to-end ASE test with F1 FASTQs (Layer 3c) | 3c | post-validation |
| 9 | LeafCutter PSI matrix → sQTL scan (Layer 3d) | 3d | post-validation |
| 10 | `epistasis.py`: scan_epistasis, test_haplotype_combinations | 3 | ✅ implemented |

---

## M — Model Comparison: miQTL LMM vs mosaiQTL LMM vs mosaiQTL NB GLM

### Purpose

Three distinct statistical models are active in this codebase. The miQTL LMM is the
R reference used to validate the port (S1–S11). The mosaiQTL LMM is the Python port —
architecturally extended but mathematically equivalent at $n = m$. The mosaiQTL NB
GLM is the novel primary model for pseudobulk snRNA-seq counts. This section defines
all three precisely and documents every point of similarity, divergence, and design
rationale.

---

### M1 — Phenotype and Input Data

| Property | miQTL LMM | mosaiQTL LMM | mosaiQTL NB GLM |
|---|---|---|---|
| Input data type | Continuous (RINT-transformed) | Continuous (RINT-transformed) | Raw integer counts |
| Normality assumption | $y \sim \mathcal{N}$ (after RINT) | $y \sim \mathcal{N}$ (after RINT) | None — $y \sim \text{NB}$ |
| Phenotype transform | RINT applied before fitting | RINT applied before fitting | None — counts used directly |
| Library size | Not applicable | Not applicable | $\log s_i$ offset per sample |
| Sample design | $n = m$ (strain means, $Z = I_m$) | $n \geq m$ (Z-matrix expansion) | $n \geq m$ (Z-matrix expansion) |
| Expression scale | Normalized (e.g. CPM, VST) | Normalized | Raw pseudobulk integer counts |

**RINT** is applied in the LMM to satisfy the Gaussian residual assumption:

$$y_i^\text{rint} = \Phi^{-1}\!\left(\frac{r_i - 3/8}{n + 1/4}\right)$$

where $r_i = \text{rank}(y_i)$, $\Phi^{-1}$ is the standard normal quantile. RINT
destroys count-specific information (zero mass, overdispersion) and is inappropriate
for raw counts — hence the NB GLM.

---

### M2 — Haplotype Design Matrix

At every locus $\ell$, the founder haplotype probability matrix is:

$$H_\ell \in \mathbb{R}^{m \times J}, \quad \sum_{f=1}^J H_{if\ell} = 1, \quad H_{if\ell} \in [0,1]$$

One column is dropped for identifiability (linear dependency $\sum_f H_{if\ell} = 1$):

$$\tilde{H}_\ell = H_\ell[\,,\, 1:J-1] \in \mathbb{R}^{m \times (J-1)}$$

**Z-matrix expansion (mosaiQTL LMM and NB GLM):**

$$\tilde{H}_{\ell,n} = Z\,\tilde{H}_\ell \in \mathbb{R}^{n \times (J-1)}, \quad Z \in \{0,1\}^{n \times m},\quad Z_{ik} = \mathbf{1}[\text{sample }i\text{ belongs to strain }k]$$

When $n = m$ and IDs match, $Z = I_m$ and $\tilde{H}_{\ell,n} = \tilde{H}_\ell$ exactly.

**miQTL LMM** operates on strain means — effectively $Z = I_m$ always, discarding
within-strain replication. **mosaiQTL LMM and NB GLM** retain all $n$ samples,
recovering within-strain information that contributes independent evidence at every locus.

**Scale convention:**

| | miQTL LMM | mosaiQTL LMM | mosaiQTL NB GLM |
|---|---|---|---|
| Haplotype range | $H_{if\ell} \in [0,2]$ (additive dosage) | $H_{if\ell} \in [0,1]$ (marginal) | $H_{if\ell} \in [0,1]$ (marginal) |
| Row sum | 2 | 1 | 1 |
| HAPPY source | Diplotype marginalised to additive | Direct from `GenomeCache` | Direct from `GenomeCache` |

---

### M3 — Confounding Correction

**miQTL LMM and mosaiQTL LMM — kinship random effect:**

Polygenic background is modelled as a Gaussian random effect with covariance
proportional to the LOCO kinship matrix $K^{(-c)}$:

$$\varepsilon \sim \mathcal{N}\!\left(0,\; \sigma^2\!\left[h^2 K^{(-c)} + (1 - h^2)I\right]\right)$$

This simultaneously accounts for:
- Population structure (correlated genetic backgrounds across strains)
- Residual polygenicity (all other loci acting on expression)
- The $h^2$ parameter partitions variance between polygenic and residual components

**mosaiQTL NB GLM — fixed covariates only:**

No kinship term. Confounding is controlled through:
$$\log \mu_i = H_i^\ell\,\gamma + X_i\,\beta + \log s_i$$

where $X_i$ includes intercept, batch, and top expression PCs. Library size $s_i$
enters as a log-offset, not a covariate — it is treated as known and exact.

**Implication:** The NB GLM is more conservative than the LMM in the presence of
population structure — uncorrected kinship is confounded with batch, cell-type
composition, and genotype effects on expression. For pseudobulk snRNA-seq from
Collaborative Cross strains with substantial relatedness, this is a known limitation.
A mixed NB model (NEBULA-LN style) would add the polygenic random effect but is
deferred to a future extension.

| Property | miQTL LMM | mosaiQTL LMM | mosaiQTL NB GLM |
|---|---|---|---|
| Polygenic correction | LOCO kinship random effect | LOCO kinship random effect | None (PCs as fixed proxy) |
| $h^2$ estimated | Yes, per gene per chromosome | Yes, per gene per chromosome | N/A |
| Batch correction | Fixed covariate in $X$ | Fixed covariate in $X$ | Fixed covariate in $X$ |
| Library size | Not applicable | Not applicable | Log-offset $\log s_i$ |

---

### M4 — Full Statistical Models

#### miQTL LMM

**Null ($H_0$):**
$$y^\text{rint} = X\beta_0 + u + \varepsilon_0$$
$$u \sim \mathcal{N}(0,\; \sigma^2 h^2 K^{(-c)}), \quad \varepsilon_0 \sim \mathcal{N}(0,\; \sigma^2(1-h^2)I)$$

**Alternative at locus $\ell$ ($H_1$):**
$$y^\text{rint} = \tilde{H}_\ell\,\alpha + X\beta + u + \varepsilon_1$$

Both null and alternative share the same $\hat{h}^2$ estimated under $H_0$ (use.fix.par=TRUE).

**Combined variance:**
$$V = h^2 K^{(-c)} + (1-h^2)I$$

Operates on $m$ strain means; $K^{(-c)} \in \mathbb{R}^{m \times m}$.

---

#### mosaiQTL LMM

**Null ($H_0$):**
$$y^\text{rint} = X\beta_0 + u_n + \varepsilon_0$$
$$u_n \sim \mathcal{N}(0,\; \sigma^2 h^2 K_n^{(-c)}), \quad \varepsilon_0 \sim \mathcal{N}(0,\; \sigma^2(1-h^2)I_n)$$

where $K_n^{(-c)} = Z K_m^{(-c)} Z^\top \in \mathbb{R}^{n \times n}$ is the Z-expanded kinship.

**Alternative at locus $\ell$ ($H_1$):**
$$y^\text{rint} = \tilde{H}_{\ell,n}\,\alpha + X\beta + u_n + \varepsilon_1$$

**At $n = m$, $Z = I_m$:** $K_n^{(-c)} = K_m^{(-c)}$ — full algebraic identity with miQTL LMM.

**Additional benefit at $n > m$:** Within-strain replication increases effective sample size.
The log-likelihood has $n$ contributing observations (not $m$), so:
$$\text{LOD}_\ell = \frac{n}{2}\log_{10}\frac{\text{RSS}_0}{\text{RSS}_1}$$
grows with $n$, improving power proportional to $\sqrt{n/m}$.

---

#### mosaiQTL NB GLM

**Null ($H_0$):**
$$Y_i \sim \text{NB}(\mu_i^0,\; \phi), \quad \log\mu_i^0 = X_i\beta_0 + \log s_i$$
$$\text{Var}[Y_i] = \mu_i^0 + \phi(\mu_i^0)^2$$

**Alternative at locus $\ell$ ($H_1$):**
$$Y_i \sim \text{NB}(\mu_i^\ell,\; \phi), \quad \log\mu_i^\ell = \tilde{H}_{i,\ell,n}\,\gamma + X_i\beta + \log s_i$$

$\phi$ is fixed across all loci for gene $g$ (estimated once from null model).  
$\gamma \in \mathbb{R}^{J-1}$ are log-scale founder haplotype effects on mean expression.

---

### M5 — Likelihoods and Test Statistics

#### LMM (both implementations)

**Gaussian log-likelihood** at a given $h^2$ and $\alpha$:

$$\ell(y^\text{rint} \mid \alpha, \beta, \sigma^2, h^2) = -\frac{n}{2}\ln(2\pi) - \frac{1}{2}\ln|V| - \frac{1}{2\sigma^2}\,e^\top V^{-1} e$$

where $e = y^\text{rint} - \tilde{H}_\ell\alpha - X\beta$.

**REML log-likelihood** (for $h^2$ estimation, excluding fixed-effect information):

$$\ell_\text{REML}(h^2) = -\frac{1}{2}\ln|V| - \frac{1}{2}\ln|X^\top V^{-1} X| - \frac{n-p}{2}\ln(y^{\top} V^{-1} M_X y)$$

where $M_X = I - X(X^\top V^{-1} X)^{-1} X^\top V^{-1}$ is the REML residual maker.

**Eigendecomposition acceleration** ($K = V_K\Lambda_K V_K^\top$):

$$V^{-1} = V_K\,\text{diag}\!\left(\frac{1}{h^2\lambda_j + (1-h^2)}\right)V_K^\top$$

Precomputed once; $h^2$ optimisation becomes a scalar loop over $[0,1]$.

**LOD score** (ML, fixed $\hat{h}^2$):

$$\text{LOD}_\ell = \frac{n}{2}\log_{10}\frac{\text{RSS}_0}{\text{RSS}_1}$$

where $\text{RSS}_k = e_k^\top V^{-1} e_k$ (generalised residual sum of squares under $H_k$).

This uses ML denominator $n$, not REML denominator $n-p$, matching miQTL exactly.

---

#### NB GLM

**NB log-likelihood** (size parameter $r = 1/\phi$):

$$\ell_\text{NB}(y \mid \mu, \phi) = \sum_{i=1}^n \Bigl[\ln\Gamma(y_i + r) - \ln\Gamma(r) - \ln\Gamma(y_i+1) + r\ln\tfrac{r}{r+\mu_i} + y_i\ln\tfrac{\mu_i}{r+\mu_i}\Bigr]$$

**Score test statistic** (default — null evaluated once):

$$U_\ell = \tilde{H}_{\ell,n}^\top r, \qquad r_i = \frac{y_i - \hat\mu_i^0}{1 + \phi\hat\mu_i^0}$$

$$I_\ell = \tilde{H}_{\ell,n}^\top \operatorname{diag}(w)\,\tilde{H}_{\ell,n}, \qquad w_i = \frac{\hat\mu_i^0}{1 + \phi\hat\mu_i^0}$$

$$Q_\ell = U_\ell^\top I_\ell^{-1} U_\ell \;\sim\; \chi^2(J-1) \quad \text{under } H_0$$

**LOD conversion:**

$$\text{LOD}_\ell = \frac{Q_\ell}{2\ln 10}$$

**LRT statistic** (alternative fitted at each locus):

$$\Lambda_\ell = 2\!\left[\ell_\text{NB}(y \mid \hat\mu^\ell, \phi) - \ell_\text{NB}(y \mid \hat\mu^0, \phi)\right] \;\sim\; \chi^2(J-1)$$

$$\text{LOD}_\ell = \frac{\Lambda_\ell}{2\ln 10}$$

**Asymptotic equivalence of score and LRT:**

As $n \to \infty$, $Q_\ell \to \Lambda_\ell$ by Le Cam's third lemma. For finite $n$,
$Q_\ell \leq \Lambda_\ell$ (score test is conservative) — use LRT at peak loci for exact betas.

---

### M6 — Effect Sizes and Biological Interpretation

| Property | miQTL LMM | mosaiQTL LMM | mosaiQTL NB GLM |
|---|---|---|---|
| Effect parameter | $\hat\alpha \in \mathbb{R}^{J-1}$ | $\hat\alpha \in \mathbb{R}^{J-1}$ | $\hat\gamma \in \mathbb{R}^{J-1}$ |
| Effect scale | RINT units ($\sigma_y \approx 1$) | RINT units ($\sigma_y \approx 1$) | Log fold-change (natural log) |
| Reference founder | Last founder (WSBJ, index $J$) | Last founder (WSBJ, index $J$) | Last founder (WSBJ, index $J$) |
| Interpretation | $\hat\alpha_f$: RINT $y$ shift for founder $f$ vs WSBJ | same | $e^{\hat\gamma_f}$: multiplicative mean count ratio for founder $f$ vs WSBJ |
| Total effect (sum over strains) | $\sum_f H_{if\ell}\hat\alpha_f$ | $\sum_f H_{if\ell}\hat\alpha_f$ | $\exp\!\left(\sum_f H_{if\ell}\hat\gamma_f\right)$ |
| Score test betas | Full betas from WLS | Full betas from FWL | Zeros (score test skips alt fit) |
| LRT betas | N/A (WLS always fits alt) | N/A (FWL always fits alt) | $\hat\gamma$ from `glm_fit` (LRT mode) |

**Contrast to reference:** because column $J$ (WSBJ) is dropped, all effects are
interpreted relative to WSBJ. To recover the $J$-th effect, apply the sum-to-zero
constraint: $\hat\alpha_J = -\sum_{f=1}^{J-1}\hat\alpha_f$.

---

### M7 — LOD Score Comparison

All three models produce a LOD score $\in [0, \infty)$ at every locus. The formulas
are superficially similar but derive from different likelihoods:

**LMM (both):**
$$\text{LOD}_\ell^\text{LMM} = \frac{n}{2}\log_{10}\frac{\text{RSS}_0}{\text{RSS}_1}, \quad \text{RSS}_k = (y - \hat y_k)^\top V^{-1}(y - \hat y_k)$$

This is the Gaussian ML LOD; $V$ encodes kinship and heritability.

**NB score test:**
$$\text{LOD}_\ell^\text{NB\text{-}score} = \frac{Q_\ell}{2\ln 10} = \frac{U_\ell^\top I_\ell^{-1} U_\ell}{2\ln 10}$$

**NB LRT:**
$$\text{LOD}_\ell^\text{NB\text{-}LRT} = \frac{\Lambda_\ell}{2\ln 10} = \frac{\ell_\text{alt}^\ell - \ell_\text{null}}{\ln 10}$$

**Calibration difference:** LMM LOD is calibrated to $\chi^2(J-1)/(2\ln 10)$ under $H_0$
after RINT, assuming Gaussian residuals. NB LOD is calibrated to the same $\chi^2(J-1)/(2\ln 10)$
distribution directly, without distributional assumptions on $y$. For count data with
strong overdispersion ($\phi \gg 0$), the NB LOD is expected to be larger than the
LMM LOD at the same locus because RINT compresses the dynamic range of extreme
counts, attenuating the signal available to the LMM.

**Not directly comparable:** LOD scores across models should not be thresholded
with the same cutoff. Each model requires its own permutation-calibrated threshold.

---

### M8 — Computational Complexity

Let $L$ = loci per chromosome, $n$ = samples, $J$ = founders, $p$ = covariates.

| Step | miQTL LMM | mosaiQTL LMM | mosaiQTL NB GLM |
|---|---|---|---|
| Kinship eigendecomp | $O(m^3)$ once per chr | $O(n^3)$ once per chr | N/A |
| $h^2$ estimation | $O(n^2 \cdot G_\text{grid})$ per gene | same | N/A |
| Null model fit | $O(n^2)$ per gene (WLS) | same (FWL) | $O(n \cdot p \cdot \text{IRLS})$ per gene |
| Per-locus scan | $O(L \cdot n \cdot J)$ WLS loop | $O(L \cdot n \cdot J^2)$ FWL vectorised | $O(L \cdot n \cdot J^2)$ score test |
| Alt model fit | Implicit in WLS | Implicit in FWL | $O(\text{IRLS} \cdot n \cdot J)$ per locus (LRT only) |
| Score test mode | N/A | N/A | Null fit once; all loci in batch |
| Bottleneck | $h^2$ estimation | eigendecomp for $n > m$ | dispersion estimation (EB pipeline) |

The mosaiQTL LMM FWL vectorisation and NB score test share the same
$O(L \cdot n \cdot J^2)$ per-chromosome structure, making them directly comparable
in wall time. The key distinction is the inner loop: FWL executes a batched QR
projection; score test executes a batched matrix-vector product and solve.

---

### M9 — Summary Comparison Table

| Property | miQTL LMM | mosaiQTL LMM | mosaiQTL NB GLM |
|---|---|---|---|
| Module | R `miqtl::scan.h2lmm` | `scan.py::scan_h2lmm` | `nb.py::scan_nb` |
| Distribution | $\mathcal{N}$ (after RINT) | $\mathcal{N}$ (after RINT) | NB ($\text{NB}(\mu, \phi)$) |
| Phenotype transform | RINT | RINT | None |
| Random effect | LOCO kinship $K^{(-c)}$ | LOCO kinship $K^{(-c)}_n = Z K_m^{(-c)} Z^\top$ | None |
| Fixed confounders | Batch + PCs | Batch + PCs | Batch + PCs |
| Library size | None | None | Log-offset $\log s_i$ |
| Overdispersion | Implicitly zero (Gaussian) | Implicitly zero (Gaussian) | Explicit $\phi$ per gene |
| Sample design | $n = m$ only | $n \geq m$ | $n \geq m$ |
| Effect scale | RINT units | RINT units | Log fold-change |
| Test statistic | $F$-test / LOD from $\text{RSS}_0/\text{RSS}_1$ | same via FWL | Score $Q_\ell$ or LRT $\Lambda_\ell$ |
| Degrees of freedom | $J-1$ | $J-1$ | $J-1$ |
| $h^2$ estimated | Yes | Yes | No |
| Primary use | Validation reference | Production LMM (strain means) | Primary model (raw counts) |
| Status | R reference (S1–S11) | ✅ validated ($r > 0.98$) | ✅ scaffolded, pending end-to-end |

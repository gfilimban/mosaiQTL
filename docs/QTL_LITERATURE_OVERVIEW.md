# mosaiQTL in Context: Classical and Contemporary QTL Mapping

## 1. Classical Foundations

### 1.1 Interval Mapping (Lander & Botstein 1989)
The first principled QTL mapping framework. Computed a LOD score at each genomic
position by comparing the likelihood of a QTL at that position against the null:

$$\text{LOD} = \log_{10} \frac{P(\mathbf{y} \mid \text{QTL at } \ell)}{P(\mathbf{y} \mid \text{no QTL})}$$

Assumed biparental crosses (F2 or backcross), Gaussian phenotypes, and a single
QTL. Permutation thresholds were introduced by Churchill & Doerge (1994) to
control genome-wide error rates — still the standard today and implemented in
mosaiQTL's `permutation.py`.

**Limitation relevant to us:** restricted to two alleles and two-parent crosses.
The 8-founder CC structure requires extending the model to $J=8$ alleles.

### 1.2 Composite Interval Mapping (Zeng 1994)
Added marker cofactors to the regression to control for background QTL, improving
power and resolution. Conceptually equivalent to mosaiQTL's `scan_conditional()`
(planned epistasis module) — conditioning on a known QTL while scanning for
additional signal.

### 1.3 Multiple QTL Models (Kao et al. 1999; Broman & Speed 2002)
R/qtl formalised multi-QTL models via stepwise model selection. The key insight —
that expression traits are polygenic and that a single-QTL scan is a marginal test
under a multi-QTL model — directly motivates the epistasis expansion planned for
mosaiQTL.

---

## 2. The Multiparent Population Revolution

### 2.1 Why Two-Parent Crosses Are Insufficient
Classical biparental QTL mapping suffers from:
- Low allelic diversity (only two alleles segregate)
- Wide LD blocks (~20–40 cM in F2), poor resolution
- Limited replication across genetic backgrounds

The Collaborative Cross (CC), Diversity Outbred (DO), and MAGIC populations were
designed to address all three by combining multiple founder strains.

### 2.2 The HAPPY Framework (Mott et al. 2000)
The foundational probabilistic model for multiparent QTL mapping. At each locus,
each individual carries a pair of founder haplotypes with probabilities inferred by
a hidden Markov model (HMM) from marker data:

$$P(\text{founder} = f \mid \text{markers}) \in [0,1], \quad \sum_f P_f = 1$$

The additive haplotype model then regresses on these probabilities:

$$\mathbf{y} = \sum_{f=1}^{J} \mathbf{h}_f \gamma_f + \mathbf{X}\boldsymbol{\alpha} + \boldsymbol{\varepsilon}$$

This is the core model in mosaiQTL. The genome cache stores these
$P(\text{founder})$ matrices; `from_gigamuga()` re-implements the HMM
from scratch in Python.

### 2.3 The Collaborative Cross (Churchill et al. 2004)
Eight inbred founder strains (A/J, C57BL/6J, 129S1/SvImJ, NOD/ShiLtJ, NZO/HlLtJ,
CAST/EiJ, PWK/PhJ, WSB/EiJ) chosen to maximise genetic diversity across the mouse
genome. After ~20 generations of intercrossing and inbreeding, each CC line is a
unique mosaic of the 8 founders with LD blocks of ~2–5 Mb — approximately 4× finer
resolution than an F2 cross.

The key property that enables our epistasis question: every CC line is a unique
haplotype combination. Unlike DO mice (which are also 8-founder but outbred),
CC lines are fully inbred and replicated — the same genotype can be phenotyped
multiple times across conditions and cell types.

### 2.4 R/qtl2 (Broman et al. 2019)
The current standard for multiparent QTL mapping in R. Extended R/qtl to handle
8-founder designs, introduced the `calc_genoprob()` + `scan1()` workflow, and
formalised the LOCO (Leave-One-Chromosome-Out) kinship correction. mosaiQTL's
`calc_kinship(method='loco')` directly mirrors R/qtl2's implementation.

**Key difference from mosaiQTL:** R/qtl2 stores haplotype probs as large
in-memory 3D arrays. mosaiQTL uses the HAPPY genome cache (one file per locus)
for memory efficiency on HPC, matching miQTL's design.

---

## 3. The LMM Framework for Population Structure

### 3.1 Mixed Model Association (Kang et al. 2008; Yu et al. 2006)
Controlling for population structure and polygenic background via a kinship-based
random effect became standard in GWAS and was then adopted for QTL mapping:

$$\mathbf{y} = \mathbf{H}_\ell \boldsymbol{\gamma} + \mathbf{X}\boldsymbol{\alpha} + \mathbf{u} + \boldsymbol{\varepsilon}, \quad \mathbf{u} \sim \mathcal{N}(\mathbf{0}, \sigma_g^2 \mathbf{K})$$

The eigendecomposition trick (Kang et al. 2008) reduces the $\mathcal{O}(n^3)$
matrix inversion to $\mathcal{O}(n^2)$ by rotating to the eigenbasis of
$\mathbf{K}$, making genome-wide scans tractable. This is the core of
mosaiQTL's `lmm.py`.

### 3.2 REML vs ML for Heritability Estimation
miQTL (and mosaiQTL) use REML to estimate $h^2$ under the null model, then
fix that estimate and compute ML LOD scores for the scan. This two-step approach
(confirmed from miQTL source via `use.par='h2.REML'`) is a pragmatic compromise:
REML gives unbiased variance component estimates; ML gives a LOD score on the
correct scale for comparison.

### 3.3 LOCO Kinship (Yang et al. 2014)
Using kinship computed from all chromosomes except the one being tested prevents
proximal contamination — the tested chromosome cannot contribute to its own
background correction. Standard in modern QTL mapping and implemented in
mosaiQTL's `calc_kinship(method='loco')`.

---

## 4. eQTL Mapping and Single-Cell Resolution

### 4.1 Bulk eQTL Mapping
Early eQTL studies (Brem et al. 2002 in yeast; Schadt et al. 2003 in mice)
treated gene expression as a quantitative trait and mapped it to genomic loci.
The distinction between *cis*-eQTL (locus within ~500 kb of the gene's TSS) and
*trans*-eQTL (distal regulatory effects) became the organising framework.
mosaiQTL's `cis_window_mb=0.5` implements the standard ±500 kb cis window.

### 4.2 The Cell-Type Problem
Bulk eQTL studies conflate expression across all cell types present in a tissue.
An eQTL may be cell-type-specific — active in type2b myonuclei but absent in
other muscle cell types. Single-nucleus RNA-seq enables cell-type-resolved
pseudobulk eQTL mapping, which is mosaiQTL's primary application.

**Contemporary standard (Bryois et al. 2022; Kim-Hellmuth et al. 2020):**
aggregate single-cell counts per cell type per individual (pseudobulk), then
apply a bulk eQTL pipeline. The NB GLM in mosaiQTL's `nb.py` handles the
count distribution directly rather than requiring normalisation before mapping.

### 4.3 NEBULA (He et al. 2021)
The current state-of-the-art for single-nucleus eQTL that bypasses pseudobulking.
Models each cell individually with a negative binomial-gamma mixed model
(NEBULA-LN approximation), preserving within-sample variance information that
pseudobulking discards. Planned for mosaiQTL's Phase 3.

**Why this matters for vQTL:** dispersion QTL require within-sample cell-level
variance. Pseudobulking averages away this variance, making vQTL detection
impossible. NEBULA-LN operates on single cells and can test both mean and
dispersion components simultaneously.

---

## 5. Fine Mapping and SNP Resolution

### 5.1 From Haplotype to Variant
Haplotype QTL mapping localises a signal to a founder haplotype and a genomic
region (~2–5 Mb in CC). Fine mapping to a causal variant requires SNP-level
resolution. In the CC, this is achieved by imputing SNP dosages from founder
haplotype probabilities:

$$d_i = \sum_{f=1}^{J} h_{if} \cdot g_f, \quad g_f \in \{0, 1, 2\}$$

where $g_f$ is the founder's genotype at the SNP. This is `imputed_snp_scan_h2lmm()`
in mosaiQTL — equivalent to miQTL's `imputed.snp.scan.h2lmm()`.

### 5.2 Credible Sets and Posterior Probabilities
Modern fine mapping (Benner et al. 2016 FINEMAP; Wang et al. 2020 SuSiE) computes
posterior inclusion probabilities for each SNP. SuSiE's sum-of-single-effects
model is particularly well suited to CC data because the LD structure between
imputed SNPs is determined by the founder haplotype mosaic — well-defined and
interpretable. Integration with SuSiE is a natural extension after SNP scan
validation.

---

## 6. Variance QTL and Dispersion Genetics

### 6.1 Beyond Mean Effects
Standard eQTL maps genetic effects on mean expression. vQTL (variance QTL)
maps genetic effects on expression variability — identifying loci where one
founder allele produces consistently higher or lower cell-to-cell variation
in expression (Rönnegård & Valdar 2011; Hulse & Bhatt 2021).

In the CC, vQTL are particularly meaningful: a founder haplotype that increases
expression variance across cells of the same type may tag a regulatory element
that is sensitive to stochastic activation (enhancer switching, chromatin
accessibility) rather than a fixed transcriptional effect.

### 6.2 Two-Stage vQTL (mosaiQTL's approach)
Rather than fitting a full dispersion model at every locus (computationally
prohibitive), mosaiQTL uses a two-stage scan:

1. **Stage 1:** standard mean eQTL scan → identify loci with mean effects
2. **Stage 2:** regress out mean effects, then scan squared Pearson residuals $r_i^2$
   for dispersion signal (Brown-Forsythe-type test)

$$\text{vQTL LOD} = \text{LOD}(\mathbf{e}_i^2 \sim \mathbf{H}_\ell) - \text{LOD}(\mathbf{e}_i^2 \sim \mathbf{X})$$

This is ~20× cheaper than NEBULA-LN per locus and detects loci where the
haplotype predicts variance above what the mean model explains.

---

## 7. The Epistasis Question in Context

### 7.1 Polygenic Expression Architecture
The majority of gene expression traits in mice are influenced by multiple QTL
(Broman et al. 2003; Rockman & Kruglyak 2006). In the CC, each line's expression
is determined by its unique combination of founder haplotypes across all
contributing QTL.

### 7.2 Emergent Expression Levels
The question — *do haplotype combinations at multiple QTL produce expression
levels not predictable from summing individual founder effects?* — is a direct
test of the **additive model's sufficiency**. In classical genetics terms, this
is epistasis: the deviation from additivity when both loci are considered jointly.

$$\text{LOD}_{epistasis} = \text{LOD}(\mathbf{y} \sim \mathbf{H}_{\ell_1} + \mathbf{H}_{\ell_2} + \mathbf{H}_{\ell_1} \otimes \mathbf{H}_{\ell_2}) - \text{LOD}(\mathbf{y} \sim \mathbf{H}_{\ell_1} + \mathbf{H}_{\ell_2})$$

The $8 \times 8$ haplotype combination heatmap visualises this directly: a purely
additive landscape is a rank-1 matrix (outer product of two founder effect vectors);
deviation from rank-1 structure indicates interaction.

**What makes CC uniquely powered for this:** the 8 founders span extreme
phenotypic diversity, and CC lines are replicated — each haplotype combination
is stable across replicates. This gives enough statistical power to detect
moderate epistatic effects that would be invisible in outbred populations.

---

## 8. mosaiQTL's Position

| Feature | R/qtl (2003) | R/qtl2 (2019) | miQTL (2020) | mosaiQTL |
|---|---|---|---|---|
| Founders | 2 | 2–8 | 2–8 | 2–8 |
| Phenotype | Gaussian | Gaussian | Gaussian | Gaussian + NB + NEBULA |
| Single-cell | ✗ | ✗ | ✗ | ✓ (pseudobulk + vQTL) |
| Haplotype probs | ✗ | ✓ | ✓ | ✓ (R-free HMM) |
| SNP imputation | ✗ | partial | ✓ | ✓ |
| Mediation | ✗ | ✗ | ✓ | ✓ |
| vQTL | ✗ | ✗ | ✗ | ✓ |
| Epistasis | partial | partial | ✗ | planned |
| Language | R | R | R | Python |
| Replicated design (Z) | ✗ | ✗ | ✗ | ✓ |

The primary contribution of mosaiQTL is bringing the HAPPY haplotype LMM
framework — previously R-only — into Python with single-cell awareness (NB/NEBULA),
replicated design support (Z-matrix), and the vQTL/epistasis extensions that
directly address the CC's unique genetic architecture.

---

## Key References

- Lander & Botstein (1989) *Genetics* 121:185 — interval mapping
- Churchill & Doerge (1994) *Genetics* 138:963 — permutation thresholds
- Mott et al. (2000) *PNAS* 97:12649 — HAPPY HMM
- Churchill et al. (2004) *Nat Genet* 36:1133 — Collaborative Cross
- Yu et al. (2006) *Nat Genet* 38:203 — mixed model association
- Kang et al. (2008) *Genetics* 178:1709 — eigendecomposition LMM
- Broman et al. (2019) *Genetics* 211:495 — R/qtl2
- Keele et al. (2020) *PLOS Genet* 16:e1008537 — miQTL
- He et al. (2021) *Commun Biol* 4:629 — NEBULA
- Kim-Hellmuth et al. (2020) *Science* 369:eaaz5900 — cell-type eQTL
- Rönnegård & Valdar (2011) *Genetics* 188:435 — variance QTL

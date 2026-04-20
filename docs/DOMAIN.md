# docs/DOMAIN.md — mosaiQTL Domain Entities
# Biological and computational domain primer for new collaborators.
# Never auto-loaded by Claude Code. Human reference only.

---

## POPULATION: COLLABORATIVE CROSS (CC)

The CC is a panel of inbred mouse strains derived by systematic intercrossing and
inbreeding of 8 genetically diverse founder strains. Each CC strain carries a mosaic
genome of segments from all 8 founders, resolved by inbreeding to homozygosity.

- **33 CC strains** in this study + 8 founders + 7 F1 hybrids (B6J as dam) = 48 genotypes
- **~418 pseudobulked samples** after QC
- Genotyping: **GigaMUGA array** (~143k markers; Morgan et al. 2016)
- Reference genome: **mm39** (GRCm39). Never mix mm10/mm39 coordinates.
- n_gen=1 is correct for the CC consensus genetic map (inbred, not outbred)

### Founder strains (canonical order A→H)
| Code | Strain | Color | Ancestry |
|---|---|---|---|
| A | A/J | #F4C245 | Classical |
| B | B6/J | #C0BFBF | Classical |
| C | 129S1/J | #DA9CC1 | Classical |
| D | NOD/J | #4F6EAF | Classical |
| E | NZO/J | #52A5DB | Classical |
| F | CAST/J | #55AF5B | Wild-derived |
| G | PWK/J | #D83026 | Wild-derived |
| H | WSB/J | #683C91 | Wild-derived |

### F1 hybrids (B6J as dam; 7 strains)
B6AF1/J · B6129S1F1/J · B6CASTF1/J · B6NODF1/J · B6NZOF1/J · B6PWKF1/J · B6WSBF1/J

---

## HAPLOTYPE PROBABILITY TENSOR

The core genotype representation. For each sample at each marker, an 8-vector
giving the posterior probability that the genomic interval derives from each founder.

```
shape: (n_samples, 8, n_loci)
axis 0: samples (CC strains, founders, F1s)
axis 1: 8 founders in FOUNDER_ORDER (A→H)
axis 2: genetic markers ordered by genomic position
values: probabilities in [0,1], summing to 1.0 along axis 1 (±1e-6)
```

Computed by HMM (forward-backward algorithm) from GigaMUGA SNP data using qtl2 (in R).
IBS (identical by state) positions are resolved probabilistically via flanking markers.
Not the same as individual SNP genotypes — never convert to biallelic for QTL mapping.

---

## LOCO KINSHIP

Leave-One-Chromosome-Out kinship matrix. For chromosome $c$, the kinship is computed
from all markers **not** on chromosome $c$. This prevents the QTL signal on chromosome
$c$ from inflating the random-effect variance component — a bias called proximal
contamination (Listgarten et al. 2012).

```
kinship_loco: dict {chr: np.ndarray (n_strains, n_strains)}
Each K_c is symmetric, positive semi-definite, entries in [0,1].
```

**Never use genome-wide kinship in the QTL scan.** LOCO is mandatory.

---

## CURRENT DATA: GASTROCNEMIUS snRNA-seq

- **Tissue**: gastrocnemius skeletal muscle (hind limb)
- **Technology**: Parse split-pool barcoding
- **Biological anchors** (preliminary eQTL signals, tissue-level):
  - *Piezo1* chr8 — LOD ~21 (strongest signal)
  - *Ccdc171* chr4
  - *Pmis2* chr7
- **AnnData file**: `gastroc_all_pb_bulk.h5ad` (HPC only)
- **Gene annotation**: GENCODE vM32 (mm39)

---

## MODEL ENTITIES

### NB-GLMM (primary)
$$S_i \sim \text{NB}(\mu_i, \phi), \quad \log\mu_i = x_i^\top\beta + u_i + \text{offset}_i$$
$$u \sim \mathcal{N}(0, \tau K_\text{LOCO})$$
$\phi$ fixed from Stage 1. PQL estimation via Woodbury (see qtl/CLAUDE.md).

### LOD score
$$\text{LOD} = \frac{Q_\ell}{2\ln 10}$$
where $Q_\ell = U_\ell^\top I_\ell^{-1} U_\ell \sim \chi^2(J-1)$ is the NB score statistic.
For lmm.py (validation): $\text{LOD} = \frac{n}{2}\log_{10}\frac{\text{RSS}_0}{\text{RSS}_1}$, $n$ = ML denominator.

### eGene
A gene with a significant QTL peak after permutation-based threshold and FDR correction.
Current count: 18,479 at 5% BH-FDR (anti-conservative without permutation correction).

### cis-eQTL
QTL peak within 1 Mb of gene TSS OR on the same chromosome. Both definitions reported.

---

## VALIDATION STRUCTURE

- **Phase 2 (S1–S11)**: Python vs R comparison. R reference: miQTL + qtl2 on sparc simulated phenotypes (matching miQTL example workflow).
  Scripts: `validation/validate_port.py`, `validation/r_reference/port_validation.R`
- **Phase 3 (P1–P4)**: Internal biological consistency. No R reference.
  Script: `validation/validate_phase3.py`
- **Passes (in gastrocnemius data)**: 7/11 Phase 2; all 4 Phase 3. See GOALS.md for current status.

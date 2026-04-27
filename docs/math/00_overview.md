# mosaiQTL — Mathematical Overview

This directory contains the derivations for every statistical method in mosaiQTL,
ordered by data flow. Each file maps directly to one or more source modules.

---

## Reading order

| File | Covers | Module(s) |
|---|---|---|
| `01_haplotypes.md` | HMM forward-backward; haplotype tensor; HAPPY vs HDF5 | `hmm.py`, `genomecache.py` |
| `02_kinship.md` | van Raden K; LOCO; Z-matrix expansion; eigendecomp caching | `kinship.py`, `utils.py` |
| `03_lmm_scan.md` | REML h² estimation; LMM scan; LOD formula; FWL vectorization | `lmm.py` |
| `04_scan_modes.md` | ROP vs Full diplotype; MI scan; additive vs full model | `scan.py` |
| `05_allele_effects.md` | fixef (pseudoinverse) vs ranef (ridge-BLUP) at peak locus | `effects.py` |
| `06_permutation.md` | GEV permutation threshold; fix_h2; bootstrap positional CI | `permutation.py` |
| `07_instability.md` | Bootstrap instability scan; mean-LOD interpretation | `scan.py` |
| `08_nb_glmm.md` | NB Stage 1 (NEBULA-LN); EB dispersion shrinkage; PQL Stage 2; Woodbury | `nb.py` |
| `09_extensions.md` | Mediation LOD-drop; Khatri-Rao epistasis; BLUP conditioning; vQTL | `mediation.py`, `epistasis.py`, `vqtl.py` |
| `10_snp_scan.md` | Haplotype-imputed SNP dosage; marginal model; r=0.55 gap | `snp_scan.py` |

---

## Notation used throughout

| Symbol | Meaning |
|---|---|
| $n$ | number of phenotype observations (samples / replicates) |
| $m$ | number of unique genotyped strains ($m \leq n$) |
| $J$ | number of founders ($J = 8$ for CC/DO) |
| $L$ | number of loci on a chromosome |
| $p$ | number of fixed-effect covariates (including intercept) |
| $H_\ell \in \mathbb{R}^{m \times J}$ | haplotype probability matrix at locus $\ell$ |
| $K \in \mathbb{R}^{m \times m}$ | realized kinship matrix |
| $K^{(-c)}$ | LOCO kinship (chromosome $c$ excluded) |
| $Z \in \{0,1\}^{n \times m}$ | replicate-to-strain mapping matrix |
| $h^2$ | narrow-sense heritability: $\sigma_g^2 / (\sigma_g^2 + \sigma_e^2)$ |
| $\text{LOD}$ | $\log_{10}$ likelihood ratio ($= Q / (2\ln 10)$ for score tests) |

Two design spaces coexist in mosaiQTL:

- **Strain space** ($m \times m$): kinship, eigendecomp, h² estimation, scan
- **Sample space** ($n \times n$): raw phenotype input, Z-matrix, strain-mean reduction

Everything at scan time operates in strain space. Z-expansion happens only at data
loading time.

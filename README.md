# mosaiQTL

Haplotype-based QTL mapping for multiparent populations (MPP) — parametric on $J$ founders,
currently designed for cell-type-resolved pseudobulk data from Collaborative Cross / DO / MAGIC
mice, with planned extension to single-cell resolution.

Maps molecular phenotypes to $J$ founder haplotypes across MPP-derived lines. Primary model:
NB-GLMM with LOCO kinship (raw pseudobulk counts). Validated Gaussian LMM port included for
normalized phenotypes.

---

## Models

| Model | Entry point | Use case |
|---|---|---|
| **NB-GLMM + LOCO** (primary) | `qtl/nb.py` | Raw pseudobulk counts — PQL null + LOCO kinship, score test |
| **Gaussian LMM** (validated port) | `scan_h2lmm()` | Normalized expression — LOCO kinship, FWL vectorised over loci |
| **LMM + multiple imputation** | `scan_h2lmm_mi()` | Integrates haplotype-assignment uncertainty via MI averaging |
| **Dispersion QTL** | `qtl/vqtl.py` | Transcriptional noise — NB dispersion as phenotype (Stage 2 under development) |
| **Mediation** | `mediation_scan_qr()` | QR-based mediation; Woodbury rank-1 update across mediators |
| **Epistasis** | `scan_epistasis()` | Pairwise GxG, conditional scan |

The LMM precomputes chromosome-constant quantities (kinship-rotated haplotype tensor) once per
chromosome and reuses them across all genes — amortizing the dominant O(n²L) cost to O(1) per
gene. The NB-GLMM fits a PQL null once per gene per chromosome then applies a score test across
all loci. Both engines are parametric on $J$ founders via `Population` config.

---

## Package structure

```
src/mosaiqtl/
  genotype/
    genomecache.py    GenomeCache — from_rds(), from_gigamuga(), HDF5 save/load, collapse_genomecache()
    hmm.py            forward-backward HMM, emission model
    kinship.py        calc_kinship() LOCO, decomp_kinship() eigendecomposition
    founders.py       FOUNDER_ORDER, FOUNDER_COLORS, F1_COLORS, CCPopulation
  qtl/
    lmm.py            estimate_h2_null() ML+Brent, scan_chromosome() FWL batch WLS
    scan.py           scan_h2lmm(), scan_h2lmm_mi(), instability_lm_scan(), grab_locus_from_scan()
    nb.py             NB-GLMM Stages 1+2 — PQL null + Woodbury score scan (primary)
    permutation.py    generate_null_outcomes(), run_threshold_scans(), get_gev_thresholds()
                      run_positional_scans() — parametric null, fix_h2=True (miQTL-matched)
    effects.py        get_allele_effects() — fixef (FWL+pinv) and ranef (BLUP) methods
    mediation.py      mediation_scan_qr(), fixef_mediation_scan()
    epistasis.py      scan_epistasis(), scan_conditional_with_y()
    vqtl.py           scan_vqtl(), VQTLResult
    snp_scan.py       imputed_snp_scan_h2lmm(), imputed_snp_scan_from_db()
    fdr.py            storey_qvalue()
  models/
    records.py        ScanResult, VQTLResult, QTLResult, FounderEffect
    config.py         Population (frozen Pydantic), ScanConfig, PipelineConfig
  io/
    covariates.py     build_covariate_matrix()
    anndata_io.py     AnnData read helpers (backed=True)
    results_io.py     serialisation with provenance header
  viz/
    genome.py         LOD profiles, Manhattan plots
    effects.py        founder effect bar plots, coef_profile()
    haplotypes.py     haplotype heatmaps, epistasis grids
  pipeline.py         discover(), characterize(), resolve()
  utils.py            rint(), build_z_matrix(), align_samples()
```

---

## Z-matrix design (n ≥ m)

mosaiQTL operates natively on $n$ samples mapping to $m \leq n$ genotyped strains.
The Z-matrix $Z \in \{0,1\}^{n \times m}$ ($Z_{ik} = 1$ if sample $i$ belongs to
strain $k$) expands haplotypes and kinship at scan time:

$$H_n = Z H_m \qquad K_n = Z K_m Z^\top$$

When $n = m$ and IDs match, $Z = I_m$ and results are identical to the strain-mean case.

---

## Installation

```bash
conda create -n mosaiqtl python=3.13
conda activate mosaiqtl
pip install -e '.[dev]'
```

**Required:** `numpy scipy pandas h5py anndata seaborn matplotlib pydantic click`  
**Optional:** `rpy2` (HAPPY `.RData` readers), `scanpy`

---

## Validation

Phases 1, 2, and 3 use simulated CC data from
[sparcc](https://github.com/gkeele/sparcc) /
[sparcc_cache](https://github.com/gkeele/sparcc_cache) — no real data required for CI.
Phase 1 is ground truth; all other phases gate on it passing first.

### Prerequisites

**Python environment** (see Installation above).

**R dependencies:**

```r
install.packages(c('remotes', 'BiocManager'))
BiocManager::install('edgeR')
remotes::install_github('gkeele/miqtl')
remotes::install_github('rqtl/qtl2')
install.packages('qtl2convert')
```

**sparcc + genome cache** (Phase 1 and 2 only):

```r
remotes::install_github('gkeele/sparcc')
```

```bash
git clone https://github.com/gkeele/sparcc_cache validation/sparcc_cache
```

### Running validation

```bash
# Phase 1 — minimal miQTL workflow mirror (ROP scan + MI + GEV threshold)
Rscript validation/r_reference/sim_validation.R
python validation/validate_sim.py

# Phase 2 sim — all 11 miQTL functions (S1–S11), real LOCO kinship
Rscript validation/r_reference/phase2_validation.R
python validation/validate_phase2.py
python validation/validate_phase2.py --sections 3,4,6   # selective
python validation/validate_phase2.py --force             # force rerun all sections

# Phase 3 — speed and memory benchmarking
Rscript validation/r_reference/phase3_validation.R
python validation/validate_phase3.py

# Phase 4 — NB-GLMM vs LMM biological consistency
python validation/validate_phase4.py

# Gastroc benchmark — miQTL S1–S11 on real CC pseudobulk data (requires HPC)
python validation/convert_gastroc_h5ad.py
Rscript validation/r_reference/gastroc_miqtl_scan.R all 8 C   # local single-chr test
bash validation/run_gastroc_slurm.sh                           # full HPC run
Rscript validation/r_reference/merge_chr_outputs.R             # after SLURM array
```

### Current status

| Phase | Gate | Threshold | Status |
|---|---|---|---|
| P1-S1 | ROP LOD r² | ≥ 0.990 | 🟢 0.9975 |
| P1-S2 | MI LOD r | ≥ 0.75 | 🟢 0.9718 |
| P1-S3 | GEV threshold delta | < 0.5 | 🟢 0.1543 |
| P2-S1..S11 | Phase 2 sim (all 11 sections) | various | 🟢 9/9 runnable |
| Phase 4 | NB-GLMM biological gates | various | 🔴 not yet run |

### Speed (M3 Mac)

| Benchmark | mosaiQTL | miQTL | Speedup | Data |
|---|---|---|---|---|
| Genome-wide scan (43k genes, 20 chr) | 2.79h | — | — | gastroc |
| LMM scan (50 genes genome-wide) | 19.8s | 400.8s | **20.2×** | simulated |
| Permutation (1000 perms, genome-wide) | 2.3s | 1555s | **678×** | simulated |
| Peak RSS (43k genes, 20 chr) | 897 MB | — | — | gastroc |

---

## Tests

```bash
pytest tests/unit/ -x -q                    # fast gate
pytest tests/ --cov=src/mosaiqtl -q         # full + coverage
make lint                                    # black + flake8 + mypy + interrogate
```

---

## References

- Kang, H.M. *et al.* (2008). *Genetics* 178:1709 — eigendecomposition LMM
- Broman, K.W. *et al.* (2019). *Genetics* 211:495 — R/qtl2
- Keele, G.R. *et al.* (2019). *G3* — miQTL
- Listgarten, J. *et al.* (2012). *Nat Methods* 9:525 — LOCO kinship
- He, Z. *et al.* (2021). *Commun Biol* 4:629 — NEBULA
- Churchill, G.A. *et al.* (2004). *Nat Genet* 36:1133 — Collaborative Cross

## License

GPL-3.0

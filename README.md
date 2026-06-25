# mosaiQTL

Haplotype-based QTL mapping for multiparent populations — parametric on $J$ founders,
designed for cell-type-resolved single-nucleus data from Collaborative Cross / DO / MAGIC mice.

Maps genetic effects to $J$ inbred founder haplotypes across mosaic recombinant lines.
Primary model: NB-GLMM with LOCO kinship (raw pseudobulk counts). Validated Gaussian LMM
port included for normalized phenotypes.

---

## Models

| Model | Function | Use case |
|---|---|---|
| **NB-GLMM + LOCO** (primary) | `qtl/nb.py` | Raw pseudobulk counts — PQL null + LOCO kinship, score test |
| **Gaussian LMM** (validated) | `qtl/scan.py::scan_h2lmm` | Normalized expression — LOCO kinship, FWL vectorised |
| **Multiple imputation** | `qtl/scan.py::scan_h2lmm_mi` | Haplotype uncertainty via MI averaging |
| **Dispersion QTL** | `qtl/vqtl.py` | Transcriptional noise — dispersion as phenotype |
| **Mediation** | `qtl/mediation.py` | QR-based mediation scan |
| **Epistasis** | `qtl/epistasis.py` | Pairwise GxG, conditional scan |

The LMM uses Frisch-Waugh-Lovell projection to vectorise across all loci simultaneously
(O(L)→O(1) per gene). The NB-GLMM fits a PQL null once per gene per chromosome then
applies a score test across all loci. Both are parametric on $J$ founders via
`Population` config.

---

## Package structure

```
src/mosaiqtl/
  genotype/
    genomecache.py    GenomeCache — from_rds(), from_gigamuga(), HDF5 save/load, collapse()
    hmm.py            forward-backward HMM, Haldane map, CC/DO emission model
    kinship.py        calc_kinship() LOCO, decomp_kinship() eigendecomposition
    founders.py       FOUNDER_ORDER, FOUNDER_COLORS, F1_COLORS, CCPopulation
  qtl/
    lmm.py            estimate_h2_null() REML+grid+Brent, scan_chromosome() FWL batch WLS
    scan.py           scan_h2lmm(), scan_h2lmm_mi(), instability_lm_scan(), grab_locus_from_scan()
    nb.py             NB-GLMM Stages 1+2 — PQL null + Woodbury score scan (primary)
    permutation.py    generate_null_outcomes(), run_threshold_scans(), get_gev_thresholds()
                      run_positional_scans() — parametric permutation default (miQTL-matched)
    effects.py        get_allele_effects() — fixef (pinv) and ranef (BLUP) methods
    mediation.py      mediation_scan_qr()
    epistasis.py      scan_epistasis(), scan_conditional_with_y()
    vqtl.py           scan_vqtl(), VQTLResult
    snp_scan.py       imputed_snp_scan_h2lmm()
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
    validation.py     Phase 2 QC plots (S1–S11, not shipped)
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
No hardcoded sample counts anywhere in the engine.

---

## Validation

Validation runs in three phases using sparcc simulated data (no real data required for
CI). Phase 1 must pass before Phase 2; Phase 2 passes ground truth to Phase 3.

```bash
conda activate bioinfo && pip install -e '.[dev]'

# Phase 1 — minimal miQTL workflow mirror (ROP scan + MI + GEV threshold)
Rscript validation/r_reference/port_sim_validation.R
python validation/validate_sim.py

# Phase 2 sim — all 11 miQTL functions (S1–S11), real LOCO kinship
Rscript validation/r_reference/port_sim_validation_p2.R
python validation/validate_sim_p2.py
python validation/validate_sim_p2.py --sections 3,4,6   # selective
python validation/validate_sim_p2.py --force             # ignore checkpoints

# Gastroc benchmark — miQTL S1–S11 on real pseudobulk data (HPC)
conda run -n bioinfo python validation/convert_gastroc_h5ad.py
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

### Speed (M3 Mac, simulated, same hardware)

| Benchmark | mosaiQTL | miQTL | Speedup |
|---|---|---|---|
| Genome-wide scan (43k genes, 20 chr) | 2.79h | — | — |
| LMM scan (50 genes genome-wide) | 19.8s | 400.8s | **20.2×** |
| Permutation (1000 perms, genome-wide) | 2.3s | 1555s | **678×** |
| Peak RSS (43k genes, 20 chr) | 897 MB | — | — |

---

## Installation

```bash
pip install -e '.[dev]'
```

**Required:** `numpy scipy pandas h5py anndata seaborn matplotlib pydantic click`  
**Optional:** `rpy2` (HAPPY `.RData` readers), `scanpy` (UMAP/dotplot viz)

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

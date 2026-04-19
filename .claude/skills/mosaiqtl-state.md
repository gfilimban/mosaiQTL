---
name: mosaiqtl-state
description: >
  Current mosaiQTL module inventory, validation section status, output file
  expectations, and known open issues. Use at the start of every session to
  orient without reading the bundle. Update whenever a section passes or a
  module changes substantially.
---

# mosaiQTL State — v0.2.0

**Latest commit**: `f846d81`  
**Primary language**: Python 3.12 (HPC conda `bioinfo`) | R 4.3.3 for miQTL validation only  
**HPC**: `/share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl`  
**Local**: `/Users/ghassan/.../igvf/analysis/qtl/mosaiqtl`  
**Bundle sync**: `make bundle` → `git pull /path/to/mosaiqtl.bundle main --rebase`

---

## Module inventory (16 modules, 5630 lines)

| Module | Lines | Status | Primary contribution |
|---|---|---|---|
| `genome_cache.py` | 813 | ✅ complete | HDF5, `from_rds`, `from_gigamuga` |
| `plotting.py` | 611 | ✅ complete | QC plots, founder palette |
| `scan.py` | 526 | ✅ complete | Z-matrix n=418, MI, fixef, instability |
| `nb.py` | 477 | ✅ two-stage | `glm_sc_fit` + pseudobulk score test (primary model) |
| `pipeline.py` | 422 | ✅ Layer 1/2/3 | `discover`, `characterize`, `resolve` |
| `epistasis.py` | 411 | ✅ novel | `scan_epistasis`, `test_haplotype_combinations`, `scan_conditional_with_y` |
| `snp_scan.py` | 393 | ✅ impl | `build_snp_catalogue`, allele dir builder |
| `hmm.py` | 335 | ✅ complete | Pure Python forward-backward |
| `vqtl.py` | 329 | ✅ novel | Dispersion QTL, `VQTLResult` |
| `permutation.py` | 315 | ✅ impl | `fix_h2=True`, GEV thresholds |
| `lmm.py` | 286 | ✅ complete | FWL-vectorized `scan_chromosome` |
| `mediation.py` | 266 | ✅ impl, S9 validated | QR-based mediation |
| `ase.py` | 183 | ✅ scaffold | kallisto allele-specific expression |
| `utils.py` | 145 | ✅ complete | RINT, Z-matrix, `align_samples` |
| `kinship.py` | 87 | ✅ complete | LOCO, eigendecomp cache |
| `__init__.py` | 31 | ✅ | |

---

## Phase 2 validation — genome-wide merged run (S1–S11)

```bash
# R: only S2 needs rerun this session (exports haplotype_vals_r.csv)
Rscript validation/r_reference/port_validation.R all Inf 2

# Python: all sections, or selective rerun
python validation/validate_port.py
python validation/validate_port.py --sections 2,4,6,7,9,10,11
```

| S# | Section | Genome-wide result | Status |
|---|---|---|---|
| S1 | Cache info | `founders_match=True  loci_exact_match=True  crossinfo=33/33` | ✅ |
| S2 | Haplotype probs | `max_dev=0.00`; prob_r pending `haplotype_vals_r.csv` rerun | ⚠️ |
| S3 | Kinship LOCO | `mean_r=0.9998  max|ΔK|=1.98e-02` | ✅ |
| S4 | LMM scan | `60.2% pass  median_r=0.9730` | ✅ |
| S5 | Cache collapse | `py=91.4%  lod_preservation=0.9999` | ✅ |
| S6 | Permutation | `mean_delta=0.619` — GEV instability, undiagnosed | ❌ |
| S7 | MI scan | `r=0.8191  n_genes=99` | ✅ |
| S8 | SNP scan | `r=0.5509` — documented marginal vs diplotype model diff | ❌ doc |
| S9 | Mediation | `r=0.9983` | ✅ |
| S10 | Positional CI | `overlap=True` | ✅ |
| S11 | Instability | chr type bug fixed; rerun needed to confirm r | ⚠️ |

Genuine passes: **S1, S3, S4, S5, S7, S9, S10** (7/11).

---

## Phase 3 validation status

Script: `validation/validate_phase3.py`  
Requires: `gastroc_all_processed2.h5ad` (HPC only)

```bash
python validation/validate_phase3.py --cell_type type2x --n_genes 500
make validate-phase3
make validate-phase3-quick   # n_genes=100, sections nb,vqtl
```

| Section | Entry point | Status |
|---|---|---|
| P1 scan_nb | `nb.scan_nb()` | ✅ implemented |
| P2 scan_vqtl | `vqtl.scan_vqtl()` | ✅ implemented |
| P3 scan_epistasis | `epistasis.scan_epistasis()` | ✅ implemented |
| P4 scan_conditional | `epistasis.scan_conditional_with_y()` | ✅ implemented |

---

## VQTLResult fields (common bug source)

```python
res = scan_vqtl(...)
res.scan_result        # ScanResult  vQTL scan      ← NOT .vqtl_scan
res.mean_scan_result   # ScanResult  mean eQTL      ← NOT .mean_scan
res.phi_estimates      # (n_subjects,)
res.phi_method         # 'moments' or 'mle'
```

---

## Key design decisions (never override)

- **NB GLM = primary model** — LMM is validation only
- **Native n=418** — Z-matrix expands at scan time; never pre-collapse
- **LOD** — `(n/2) * log10(RSS0/RSS1)`, ML denominator `n`
- **FWL vectorization** — `scan_chromosome` FWL-equivalent to per-locus WLS
- **Parametric on J** — no hardcoded `n_founders=8`
- **HPC paths** — never committed; set manually at top of each script

---

## File paths

```
miqtl_dir  = /share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc
adata_path = {miqtl_dir}/gastroc_all_processed2.h5ad
pkg_dir    = /share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl
r_data     = {pkg_dir}/validation/port_validation/r_data
py_data    = {pkg_dir}/validation/port_validation/py_data
qc_plots   = {pkg_dir}/validation/port_validation/qc_plots
```

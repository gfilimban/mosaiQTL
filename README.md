# mosaiQTL v0.6.0

**Haplotype-based QTL mapping for multiparent populations.**  
Maps molecular phenotypes to $J$ founder haplotypes — NB-GLMM primary model for raw pseudobulk
counts, validated Gaussian LMM port for normalized phenotypes. CC mice (mm39); extensible to
DO, MAGIC, and other MPPs.

---

## Why mosaiQTL?

| | mosaiQTL | miQTL (R) | tensorQTL |
|---|---|---|---|
| **Primary model** | NB-GLMM (counts) | Gaussian LMM | Gaussian linear |
| **Haplotype scan** | ✓ ($J$ founders) | ✓ ($J$ founders) | ✗ |
| **LOCO kinship** | ✓ mandatory | ✓ | ✗ |
| **Raw count input** | ✓ | ✗ | ✗ |
| **Speed vs miQTL** | **20× LMM · 678× permutation** | 1× | — |
| **Peak RAM (43k genes)** | 897 MB | — | — |

**NB-GLMM matters for counts.** Log-normalising pseudobulk counts and treating them as
Gaussian inflates false positives for lowly-expressed genes. mosaiQTL models overdispersion
explicitly — raw counts in, calibrated LOD scores out.

**LOCO kinship is mandatory.** Including the test chromosome in the kinship inflates the
random-effect estimate (proximal contamination). mosaiQTL recomputes $K$ excluding each
chromosome under test.

---

## Installation

```bash
conda create -n mosaiqtl python=3.13
conda activate mosaiqtl
pip install -e '.[dev]'
```

Core deps: `numpy scipy pandas h5py anndata seaborn matplotlib pydantic click`  
Optional: `rpy2` (HAPPY `.RData` genome cache readers)

---

## Quick start — CLI

Five composable subcommands; each reads/writes CSV or HDF5, no hidden state between stages.

```bash
# 1. NB null fit per gene (Stage 1; LMM skips this step)
mosaiqtl fit-null \
    --cache cache.h5 --counts data.h5ad --metadata meta.csv \
    --output null_fit.csv [--cell-type 'type2x']

# 2. Genome-wide QTL scan → per-gene peak LOD
mosaiqtl scan \
    --cache cache.h5 --counts data.h5ad --metadata meta.csv \
    --null-fit null_fit.csv --output peak_lod.csv
    # LMM: add --model lmm (no fit-null needed)

# 3. Permutation thresholds → GEV fit on max-LOD distribution
mosaiqtl permute \
    --cache cache.h5 --counts data.h5ad --metadata meta.csv \
    --null-fit null_fit.csv --output perm/ --n-perms 1000

# 4. Haplotype effect estimates at each gene's peak locus
mosaiqtl estimate \
    --cache cache.h5 --counts data.h5ad --metadata meta.csv \
    --scan-result peak_lod.csv --output allele_effects.csv

# 5. Summary table + genome-wide Manhattan
mosaiqtl report \
    --scan-result peak_lod.csv --perm-dir perm/ --output-dir results/
```

Outputs: `null_fit.csv` · `peak_lod.csv` · `perm/perm_max_lods.csv` ·
`allele_effects.csv` · `results/eqtl_summary.csv` · `results/genome_summary.png`

### NB-GLMM vs LMM — which model?

| | NB-GLMM (`--model nb`, default) | LMM (`--model lmm`) |
|---|---|---|
| **Input** | Raw integer pseudobulk counts | Log-normalised / continuous phenotype |
| **Overdispersion** | Modelled explicitly (NB φ per gene) | Not modelled — inflated FP for low-expr genes |
| **Stage 1 (`fit-null`)** | Required — estimates φ and σ² per gene | Not needed — h² estimated inside `scan` |
| **Speed** | Slower (PQL null per gene × chr) | Faster (h² once per chr, FWL vectorised) |
| **Use when** | snRNA-seq / ATAC-seq / any count data | Normalized bulk RNA-seq or fast exploration |

```bash
# NB-GLMM (default) — two stages
mosaiqtl fit-null --cache cache.h5 --counts data.h5ad --metadata meta.csv --output null_fit.csv
mosaiqtl scan     --cache cache.h5 --counts data.h5ad --metadata meta.csv --null-fit null_fit.csv --output peak_lod.csv

# LMM — skip fit-null entirely
mosaiqtl scan     --cache cache.h5 --counts data.h5ad --metadata meta.csv --null-fit null_fit.csv --output peak_lod.csv --model lmm
```

---

## Quick start — Python API

```python
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.scan import scan_h2lmm

gc     = GenomeCache.from_hdf5('cache.h5')
kd_raw = calc_kinship(gc, method='loco')          # one K per chromosome
kd     = decomp_kinship(kd_raw)

result = scan_h2lmm(
    phenotype=y,          # (n,) strain-mean expression
    genomecache=gc,
    kinship_decomp=kd,
    covariates=X,         # (n, p) including intercept; sex + batch minimum
)
# result.lod     — (n_loci,) LOD scores
# result.betas   — (n_loci, J) per-founder effects
# result.pos_mb  — (n_loci,) genomic positions
peak_idx = result.lod.argmax()
print(f'Peak: chr{result.chr} {result.pos_mb[peak_idx]:.2f} Mb  LOD={result.lod[peak_idx]:.2f}')
```

---

## Models

| Model | Entry point | When to use |
|---|---|---|
| **NB-GLMM + LOCO** | `qtl/nb.py` | Raw pseudobulk counts — primary for snRNA-seq eQTL |
| **Gaussian LMM** | `scan_h2lmm()` | Normalized expression; exact miQTL/qtl2 port |
| **LMM + MI** | `scan_h2lmm_mi()` | Averages over haplotype-assignment uncertainty |
| **Dispersion QTL** | `scan_vqtl()` | NB dispersion as phenotype (transcriptional noise) |
| **Mediation** | `mediation_scan_qr()` | Candidate mediator LOD via QR Woodbury update |
| **Epistasis** | `scan_epistasis()` | Pairwise GxG; conditional scan |

---

## Package structure

```
src/mosaiqtl/
  cli.py              CLI entry point — fit-null / scan / permute / estimate / report
  pipeline.py         discover(), characterize(), call_egenes(), finalize_egene_summary()
  utils.py            rint(), build_z_matrix(), align_samples()
  genotype/
    genomecache.py    HDF5 B-tree cache; lazy per-chromosome load
    kinship.py        LOCO kinship build + eigendecomposition
    founders.py       Founder/F1/CC color constants — canonical source
  qtl/
    nb.py             NB-GLMM Stages 1 + 2 (primary)
    lmm.py            Gaussian LMM engine
    scan.py           scan_h2lmm(), scan_h2lmm_mi(), instability, MI
    permutation.py    Null outcomes, GEV threshold, run_threshold_scans()
    effects.py        get_allele_effects() — fixef and ranef (ridge BLUP)
    mediation.py      mediation_scan_qr()
    epistasis.py      scan_epistasis(), scan_conditional_with_y()
    vqtl.py           scan_vqtl()
  models/
    records.py        ScanResult, VQTLResult, FounderEffect dataclasses
    config.py         Population, ScanConfig, PipelineConfig (Pydantic)
  io/
    covariates.py     build_covariate_matrix() from sample metadata CSV
    anndata_io.py     AnnData readers (backed=True; counts from .raw.X)
    results_io.py     CSV serialisation with provenance header
  viz/
    _theme.py         Package identity palette and style helpers
    genome.py         LOD profiles, Manhattan, genome summary
    effects.py        Founder effect bars, coef_profile()
    haplotypes.py     Haplotype heatmaps, epistasis grids
    compare.py        R-vs-Python and cross-source comparison figures
    celltypes.py      Cross-cell-type LOD and effect comparison
```

---

## Validation status

Phases 1–2 use simulated CC data (`sparcc_cache`) — no real data needed for CI.

| Phase | Description | Status |
|---|---|---|
| **Phase 1** | Minimal miQTL workflow mirror: ROP r²=0.9975, MI r=0.9756, GEV Δ=0.154 | 🟢 passing |
| **Phase 2** | Full miQTL port — 9/9 runnable sections (S1–S11) | 🟢 passing |
| **Phase 3** | Speed + memory vs miQTL and R/qtl2 | 🟢 benchmarked |
| **Phase 4** | NB-GLMM biological consistency (Piezo1 etc.) | 🔴 not yet run |
| **Gastroc** | Real CC pseudobulk — Python scan complete; R comparison pending HPC | 🟡 in progress |

### Running validation

```bash
# Phase 1
Rscript validation/r_reference/sim_validation.R
conda run -n bioinfo python validation/validate_sim.py

# Phase 2
Rscript validation/r_reference/phase2_validation.R
conda run -n bioinfo python validation/validate_phase2.py

# Phase 3
Rscript validation/r_reference/phase3_validation.R
conda run -n bioinfo python validation/validate_phase3.py

# Gastroc (real data; HPC for full run)
conda run -n bioinfo python validation/tools/convert_h5ad.py
conda run -n bioinfo Rscript validation/r_reference/gastroc_miqtl_scan.R all 8 C
bash validation/hpc/run_gastroc_slurm.sh
```

---

## Tests

```bash
pytest tests/unit/ -x -q                  # fast unit gate
pytest tests/ --cov=src/mosaiqtl -q       # full suite + coverage
make lint                                  # flake8 + mypy + interrogate
```

---

## References

- Kang *et al.* (2008) *Genetics* 178:1709 — eigendecomposition LMM
- Listgarten *et al.* (2012) *Nat Methods* 9:525 — LOCO kinship
- Broman *et al.* (2019) *Genetics* 211:495 — R/qtl2
- Keele *et al.* (2019) *G3* — miQTL
- He *et al.* (2021) *Commun Biol* 4:629 — NEBULA-LN
- Churchill *et al.* (2004) *Nat Genet* 36:1133 — Collaborative Cross

## License

GPL-3.0

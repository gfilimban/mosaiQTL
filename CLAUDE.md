# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Read `docs/WIKI.md` at the start of every session before touching any code.** It is the single source of truth for scientific context, validation status, known bugs, and design decisions.

---

## Commands

```bash
# Install
pip install -e .

# Syntax check
make check

# Unit tests (23 tests, all must pass)
make test
# Single test file:
python -m pytest tests/test_nb.py -v

# Phase 2 validation (port vs R miQTL reference, S1–S11)
make validate-phase2
make validate-chr8                       # quick: chr8, 50 genes
make validate-section SECTIONS=2,11     # specific sections only
make validate-force                      # ignore .done checkpoints

# Phase 3 validation (HPC only — requires gastroc_all_processed2.h5ad)
make validate-phase3
make validate-phase3-quick

# HPC sync
make bundle                              # create mosaiqtl.bundle for transfer
```

---

## Architecture

mosaiQTL is a three-layer QTL mapping package parametric on $J$ founder strains. The shared backbone is the genome cache (haplotype probability tensor + genetic map), from which all scan models read.

### Three-layer API (`pipeline.py`)

```
Layer 1 — QTL Discovery       discover()     → scan_nb() or scan_h2lmm()
Layer 2 — Characterisation    characterize() → scan_vqtl(), mediation_scan_qr()
Layer 3 — Mechanistic         resolve()      → scan_epistasis(), build_snp_catalogue(), ase, sqtl
```

The low-level entry points (`scan_nb`, `scan_h2lmm`, etc.) are also exported from `__init__.py` for direct use.

### Primary model: NB-GLMM with LOCO kinship (`nb.py`)

Two-stage design:
1. **Stage 1** — cell-level null via native NEBULA-LN (`fit_null_sc`, `use_native=True`). Produces EB-shrunk dispersion `dispersion_use` and `sigma_sample` per gene.
2. **Stage 2** — per-chromosome PQL null fit + vectorised score test across all loci. $V \in \mathbb{R}^{418\times418}$ is never formed — Woodbury reduces to $48\times48$ Cholesky. $\phi$ is held fixed from Stage 1; only $\tau$ (kinship variance component) is estimated per chromosome.

Score test: $T_\ell = (H_\ell^\top r_\text{adj})^\top (H_\ell^\top P H_\ell)^{-1} (H_\ell^\top r_\text{adj}) \sim \chi^2_{J-1}$

**Correctness invariant:** $P\mathbf{X} = 0$ (not $P^2 = P$ — $P$ is oblique when $V \neq I$). Tested in `test_nb.py::TestWoodbuyExactness`. Do not add a $P^2 = P$ test.

### Gaussian LMM (`scan.py`, `lmm.py`)

miQTL port, used for validation. Eigendecomposition of $K^{(-c)}$ cached once per chromosome; FWL vectorisation (`np.einsum`) for ~100× speedup. LOD formula uses ML denominator $n$, not $n-p$.

**Known difference from R:** Python uses marginal 8-state HMM probabilities; R uses HAPPY 36-state diplotype probabilities. This causes S8 SNP scan r=0.55 — documented, not a bug.

### Z-matrix design (`utils.py`)

$Z \in \{0,1\}^{n \times m}$ maps $n$ samples to $m$ genotyped strains. Haplotypes and kinship expand at scan time: $H_n = ZH_m$, $K_n = ZK_mZ^\top$. When $n=m$ and IDs match, $Z=I$. No hardcoded sample counts anywhere.

### Genome cache (`genome_cache.py`)

Stores haplotype probability tensor (loci × strains × founders), physical/genetic maps, and founder metadata. Built from GigaMUGA CSV or HAPPY `.RData` via `from_gigamuga()` / `from_happy_dir()`. Serialised to HDF5. `collapse()` reduces loci spacing for scan efficiency.

### Validation infrastructure

- **Phase 2** (`validation/validate_port.py`): S1–S11, compares Python output to R miQTL reference. Uses `.done` checkpoint files — pass `--force` to rerun. Checkpoint scoping bug: hoist cross-section variables (R CSVs) before `section(1)` to survive checkpointing (see WIKI §9).
- **Phase 3** (`validation/validate_phase3.py`): P1–P4 novel extensions. HPC-only; requires `.h5ad` file. Outputs go to `validation/phase3/` (separate from `phase2/` to prevent CSV collisions).

### Current validation status (phase 2)

Genuine passes: S1, S3, S4, S5, S7, S9, S10 (7/11). S6 (GEV permutation) and S8 (SNP scan) are documented failures with known causes, not regressions. S2 and S11 need reruns after recent fixes.

---

## Key invariants

- **LOD formula:** $(n/2)\log_{10}(\text{RSS}_0/\text{RSS}_1)$ — ML denominator $n$, matching miQTL.
- **$h^2$ fix-par convention:** `use.fix.par=TRUE` in miQTL, matched in Python REML grid search.
- **Founder column order:** canonical 8-founder order must be preserved in all CSVs and HDF5 tensors (recent fix: `haplotype_vals_r.csv` column reorder, commit `bbd8420`).
- **LOCO kinship for NB-GLMM:** `kinship_loco` must be passed to `scan_nb()` for kinship correction. `kinship_loco=None` falls back to σ² inflation only.
- **Phase 2 / phase 3 output dirs are separate** — never write phase3 outputs to `phase2/` paths.
- **Kinship ground truth is qtl2 van Raden** (`calc_kinship` in R). Python's `kinship.py` must replicate this formula — do not change R's kinship to match Python's dot-product formula.
- **`probs_cc.rds` must be built with crossinfo.** `build_genomecache.R` uses `cc_crossinfo.csv` for per-strain breeding funnels. If the file predates `cc_crossinfo.csv`, re-run `build_genomecache.R` to regenerate probs, kinship, and HAPPY cache with crossinfo-sharp founder assignments.

---

## Tooling conventions

**Python data analysis:** always use `conda run -n bioinfo python3 -c '...'` with pandas rather than shell one-liners (`awk`, `cut`, `sort -u`). Pandas gives richer analysis per call, avoids quoting issues, and keeps context use low.

```bash
# prefer this:
conda run -n bioinfo python3 -c "
import pandas as pd, numpy as np
df = pd.read_csv('path/to/file.csv')
...
"

# over this:
awk -F',' 'NR>1 {print $2}' file.csv | sort -u
```

**R scripts** (`build_genomecache.R`, `port_validation.R`) run locally via `Rscript` or on HPC via SLURM. Both live in `validation/r_reference/`. R outputs write to `validation/phase2/r_data/` (not the old `port_validation/r_data/`).

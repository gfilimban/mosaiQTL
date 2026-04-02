---
name: mosaiqtl-state
description: >
  Current mosaiQTL v0.2.0 module inventory, validation section status, output
  file expectations, and known open issues. Use at the start of every session
  to orient without reading the bundle. Update this file whenever a section
  passes validation or a module is substantially changed.
---

# mosaiQTL State — v0.2.0

**Repo**: `github.com/gfilimban/mosaiqtl` (private)  
**Latest commit**: `2333b69`  
**Primary language**: Python 3.13 | R 4.3.3 for miQTL validation only  
**Env**: conda `bioinfo`  
**Local**: `/Users/ghassan/.../igvf/analysis/qtl/mosaiqtl`  
**HPC**: `/share/crsp/lab/seyedam/gfilimba/packages/mosaiqtl`

---

## Module inventory

| Module | Lines | Status | Primary novel contribution |
|---|---|---|---|
| `genome_cache.py` | 785 | ✅ complete | HDF5, `from_rds`, `from_gigamuga` |
| `hmm.py` | 335 | ✅ complete | Pure Python forward-backward |
| `kinship.py` | 178 | ✅ complete | LOCO, eigendecomp cache |
| `lmm.py` | ~300 | ✅ complete | **Vectorized `scan_chromosome` via FWL** |
| `scan.py` | 502 | ✅ complete | Z-matrix n=418, MI, fixef, instability |
| `permutation.py` | 320 | ✅ impl, unvalidated | `fix_h2=True` mode |
| `nb.py` | 252 | ✅ scaffold | edgePython wrapping (primary model) |
| `vqtl.py` | 329 | ✅ novel | Dispersion QTL scan |
| `snp_scan.py` | 306 | ✅ impl, unvalidated | Allele dir builder |
| `mediation.py` | 218 | ✅ impl, unvalidated | QR-based mediation |
| `plotting.py` | 611 | ✅ complete | — |
| `utils.py` | 204 | ✅ complete | RINT, Z-matrix, design matrix |

**Total**: ~4,300 lines Python

---

## 11-section validation status

Run with: `Rscript validation/r_reference/port_validation.R 8 50`  
Then: `python validation/validate_port.py --chr 8`

| S# | Name | R output file | R status | Python status |
|---|---|---|---|---|
| S1 | Cache info | `cache_info_r.csv` | ✅ | ✅ |
| S2 | Haplotype probs | `haplotype_r.csv` | ✅ | pending |
| S3 | Kinship LOCO | `kinship_r.csv`, `kinship_mat_r_chr{N}.csv` | ✅ | ✅ r=0.9997 |
| S4 | LMM scan | `lod_r.csv`, `allele_effects_r.csv` | ✅ | pending |
| S5 | Permutation | `thresholds_r.csv`, `perm_lods_r.csv` | ✅ manual loop confirmed | pending |
| S6 | Cache collapse | `collapsed_cache_summary.csv` | ✅ 79% reduction | pending |
| S7 | Multiple imputation | `mi_lod_r.csv` | ✅ | pending |
| S8 | SNP scan | `snp_lod_r.csv` | ✅ fixed_extract patch applied | pending |
| S9 | Mediation | `mediation_r.csv` | ✅ | pending |
| S10 | Positional CI | `ci_r.csv`, `boot_pos_r.csv` | ✅ fr$pos$Mb extraction working | pending |
| S11 | Instability | `fixef_r.csv` | ✅ manual OLS bootstrap confirmed | pending |

---

## Expected output sanity checks

| File | Shape | Key check |
|---|---|---|
| `cache_info_r.csv` | 1×4 | chr=8, n_loci≈4848 |
| `haplotype_r.csv` | 5×7 | mean_rowsum=2.0 (additive dosage), mean_max_prob≥0.7 |
| `kinship_r.csv` | 1×6 | n=48, mean_diag≈0.91, mean_offdiag≈0.13 |
| `lod_r.csv` | n_genes×36×cols | LOD range 0–15, gene_id and locus present |
| `allele_effects_r.csv` | n_genes×8 rows | exactly 8 rows/gene, founder col = A–H (not locus names) |
| `mi_lod_r.csv` | same shape as lod_r | MI LOD < ROP LOD (conservative); constant within imputation blocks |
| `collapsed_cache_summary.csv` | 20×3 | n_loci_coll ≈ 21% of n_loci_orig |
| `thresholds_r.csv` | 1×7 | n_perms>0, lod_threshold≈5–8 |
| `mediation_r.csv` | 36×9 | lod_drop may be negative (amplification, not a bug) |
| `ci_r.csv` | 1×7 | n_boot>0, ci_lo/ci_hi not NaN |
| `boot_pos_r.csv` | n_boot×1 | majority non-NaN |
| `fixef_r.csv` | 36×5 | length(loci)>0, LOD range 0–6 |

---

## Open issues

| # | Description | Status |
|---|---|---|
| 1 | Global Python validation pending merged R SLURM outputs | 🔄 ready to resubmit after `2333b69` pull |
| 2 | K-expansion (48→418) not wired into `scan_h2lmm` | 🔲 planned |
| 3 | Permutation thresholds unimplemented as standalone module | 🔲 planned |
| 4 | `nb.py` scaffold only — no genome-wide `scan_nb()` | 🔲 primary model, next milestone |
| 5 | `epistasis.py` not yet created | 🔲 future extension |

---

## Key design decisions (never override)

- **Native n=418 samples** — Z-matrix expands kinship/haplotypes at scan time; never collapse to strain means
- **NB GLM = primary model** — LMM port is validation only; NB on raw pseudobulk counts via edgePython is the science
- **LOD formula** — `(n/2) * log10(RSS0/RSS1)`, ML denominator `n` not `n-p`, matches miQTL
- **FWL vectorization** — `scan_chromosome` now uses Frisch-Waugh-Lovell; mathematically identical to per-locus WLS but ~100× faster
- **Parametric on J** — no hardcoded `n_founders=8`
- **HPC paths** — never committed; user updates manually before SLURM submission

---

## SLURM pipeline (post-2333b69)

```bash
# on HPC, from pkg_dir after git pull:
bash validation/run_validation_slurm.sh

# step 0: setup job  → s1,2,3,6  (genome-level, one node)
# step 1: array jobs → s4,5,7,8,9,10,11  (20 nodes, one per chr)
# step 2: merge job  → merge_chr_outputs.R + validate_port.py
```

**Fixed in 2333b69:**
- `h`/`founders`/`sample_ids` moved to shared setup block — per-chr array jobs no longer crash on `h not found`
- S8 `fixed_extract` patch: `extract.imputed.design.matrix.from.doqtl.genotype` dim bug fixed (ref_count now length 8, conformable with mapping.matrix 36×8)
- S11 manual OLS bootstrap loop replaces inaccessible `instability.lm.scan` closure
- `scan_h2lmm_mi`: vectorized one-hot imputation, h² pre-estimated once from real cache

**Fixed in merge (this session):**
- `merge_chr_outputs.R` now includes S9/S10/S11 outputs (`mediation_r.csv`, `ci_r.csv`, `boot_pos_r.csv`, `fixef_r.csv`) — previously missing, causing `load_r()` to return None for these sections

---

## File paths

```
data_dir    = /Users/ghassan/.../igvf/analysis/qtl/miqtl
gm_path     = {data_dir}/input/GM_ccfo_mm39.csv
pmap_path   = {data_dir}/input/pmap.csv
allele_dir  = {data_dir}/output/allele_dir        # flat chrN.alleles files
genomecache = {data_dir}/output/genomecache
kinship_rds = {data_dir}/output/kinship_loco.rds
pheno_rds   = {data_dir}/output/pheno_strain.rds
collapsed   = {data_dir}/output/genomecache_collapsed
output_dir  = {pkg_dir}/validation/port_validation/r_data
```


# validate-port State

**Latest bundle commit**: `f846d81`  
**R script**: `validation/r_reference/port_validation.R`  
**Python script**: `validation/validate_port.py`  
**Genome-wide run**: SLURM array → `merge_chr_outputs.R` → `python validate_port.py`  
**Single-chr check**: `Rscript port_validation.R 8 50` → `python validate_port.py --chr 8 --n_genes 50`

---

## Status table — genome-wide merged run

| S# | Section | R status | Python status | Result |
|---|---|---|---|---|
| S1 | Cache info | ✅ | ✅ | `loci_exact_match=True  crossinfo=33/33` |
| S2 | Haplotype probs | ✅ + `haplotype_vals_r.csv` added | ⚠️ rerun needed | `max_dev=0.00`; actual prob_r TBD |
| S3 | Kinship LOCO | ✅ | ✅ | `mean_r=0.9998` |
| S4 | LMM scan | ✅ | ✅ | `60.2% pass  median_r=0.9730` |
| S5 | Cache collapse | ✅ | ✅ | `lod_preservation=0.9999` |
| S6 | Permutation | ✅ | ❌ | `mean_delta=0.619 > 0.5` |
| S7 | MI scan | ✅ | ✅ | `r=0.8191` |
| S8 | SNP scan | ✅ | ❌ doc | `r=0.5509` — model diff documented |
| S9 | Mediation | ✅ | ✅ | `r=0.9983` (all chr) |
| S10 | Positional CI | ✅ | ✅ | `overlap=True` (all chr) |
| S11 | Instability | ✅ | ⚠️ rerun needed | chr type bug fixed; was `r=-0.5408` |

---

## Key changes this session

- **S2**: R exports `haplotype_vals_r.csv` (actual per-locus per-founder probs). Python compares real HMM values, not row sums. Pass gate: `r > 0.90` on actual values.
- **S4**: Split into 3 separate plots. Tier bar order: strict → lenient → fail. LOD-stratified fail rate panel added. Residuals x-axis = pos_mb.
- **S6**: `perm_lods_py.csv` now has `gene_id, chr, pos_mb, perm_lod`.
- **S7/S9/S10/S11**: Profile plot x-axis = pos_mb throughout.
- **S9/S10**: Loop all (gene_id, chr) pairs from merged R data — no longer chr1 only.
- **S11**: Explicit `.astype(str)` on chr column (root cause of `r=-0.54`); loop all pairs; per-pair diagnostic printed.

---

## R output files — column schemas

| File | Key columns |
|---|---|
| `cache_info_r.csv` | `chr, n_loci, min_mb, max_mb` |
| `haplotype_r.csv` | `chr, locus, pos_mb, mean_rowsum, min_rowsum, max_rowsum, mean_max_prob` |
| `haplotype_vals_r.csv` | `chr, locus, pos_mb, sample_id, [AJ, B6J, 129S1J, NODJ, NZOJ, CASTJ, PWKJ, WSBJ]` |
| `kinship_r.csv` | `chr, n, mean_diag, mean_offdiag, min_val, max_val` |
| `kinship_mat_r_chr{N}.csv` | `sample_id, [48 subject columns]` |
| `lod_r.csv` | `gene_id, chr, locus, pos_mb, lod, p_value` |
| `allele_effects_r.csv` | `gene_id, chr, locus, founder, effect` (founder = A–H codes) |
| `thresholds_r.csv` | `gene_id, chr, n_perms, lod_threshold, perm_mean, perm_sd` |
| `perm_lods_r.csv` | `gene_id, chr, pos_mb, perm_lod` |
| `collapsed_cache_summary.csv` | `chr, n_loci_orig, n_loci_coll` |
| `mi_lod_r.csv` | same as `lod_r.csv` |
| `snp_lod_r.csv` | `gene_id, chr, snp, lod, p_value` |
| `mediation_r.csv` | `gene_id, chr, locus, pos_mb, mediator, lod_base, lod_mediated, lod_drop` |
| `ci_r.csv` | `gene_id, chr, locus, ci_lo_mb, ci_hi_mb, peak_lod, n_boot` |
| `boot_pos_r.csv` | `gene_id, chr, locus, boot_pos` |
| `fixef_r.csv` | `gene_id, chr, locus, pos_mb, lod` |

---

## Python output files — column schemas

| File | Key columns |
|---|---|
| `haplotype_val_comparison.csv` | `r_prob, py_prob` (R÷2 to match [0,1] scale) |
| `lod_comparison.csv` | `gene_id, chr, locus, pos_mb, lod_r, lod_py` |
| `gene_summary.csv` | `gene_id, chr, tier, r, rel_error, peak_match, peak_lod_r, peak_lod_py, h2_null, lod_bin` |
| `kinship_comparison.csv` | `chr, r, max_delta, pass` |
| `perm_lods_py.csv` | `gene_id, chr, pos_mb, perm_lod` |
| `threshold_comparison.csv` | `gene_id, chr, n_perms, threshold_r, threshold_py, delta` |
| `mi_lod_comparison.csv` | `gene_id, chr, locus, pos_mb, lod_r, lod_py` |
| `snp_lod_comparison.csv` | `gene_id, chr, snp, lod_r, lod_py` |
| `mediation_comparison.csv` | `gene_id, chr, locus, pos_mb, lod_r, lod_py` (all chr) |
| `ci_comparison.csv` | `gene_id, chr, locus, ci_lo_r, ci_lo_py, ci_hi_r, ci_hi_py, overlap` (all chr) |
| `fixef_comparison.csv` | `gene_id, chr, locus, pos_mb_r, lod_r, lod_py` (all chr) |

---

## QC plot inventory

| Plot | Content |
|---|---|
| `s1_cache_info.png` | loci per chr bar (Python vs R) + metadata table |
| `s2_haplotype_probs.png` | haplotype heatmap + actual prob scatter (R vs Python) + row-sum deviation |
| `s3_kinship.png` | 3-panel heatmap: Python / R / difference |
| `s4_lod_scatter.png` | LOD scatter Python vs R |
| `s4_lod_residuals.png` | ΔLOD vs position (Mb) |
| `s4_gene_tiers.png` | tier bar (strict→lenient→fail) + fail rate by LOD bin |
| `s4_founder_effects.png` | grouped bar Python vs R allele effects |
| `s5_cache_collapse.png` | collapse % + kinship stability + LOD preservation |
| `s6_permutation.png` | pooled histogram + QQ + per-chr delta bar |
| `s7_mi_scan.png` | scatter + profile (pos_mb x-axis) |
| `s8_snp_scan.png` | scatter |
| `s9_mediation.png` | scatter + profile (pos_mb x-axis, all chr) |
| `s10_positional_ci.png` | bootstrap histogram + interval comparison (all chr) |
| `s11_instability.png` | scatter + profile (pos_mb x-axis, all chr) |

Conventions: R on x-axis, Python on y-axis. R in `#D83026`, Python in `#4F6EAF`.

---

## Pass criteria

| Metric | Threshold |
|---|---|
| Kinship Pearson r | > 0.999 |
| Haplotype prob values r | > 0.90 |
| LMM LOD tier pass rate | ≥ 50% |
| MI LOD r | > 0.75 |
| SNP LOD r | > 0.50 (model diff) |
| Mediation LOD r | > 0.90 |
| Permutation threshold Δ | < 0.5 |
| Instability LOD r | > 0.90 |
| CI overlap | True |

---

## Open issues requiring rerun

- **S2**: `haplotype_vals_r.csv` not yet generated → `Rscript port_validation.R all Inf 2` then `python validate_port.py --sections 2`
- **S11**: chr type bug fixed but not rerun → `python validate_port.py --sections 11`
- **S6**: GEV instability undiagnosed — inspect per-chr threshold delta before next rerun

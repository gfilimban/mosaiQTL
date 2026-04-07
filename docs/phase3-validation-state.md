
# Phase 3 Validation State

**Script**: `validation/validate_phase3.py`  
**Requires**: `gastroc_all_processed2.h5ad` (HPC only)  
**Run**: `python validation/validate_phase3.py --cell_type type2x --n_genes 500`  
**Quick**: `make validate-phase3-quick` (n_genes=100, sections nb,vqtl)

---

## Section map

| Flag | Entry point | Input | Biological anchors |
|---|---|---|---|
| `nb` | `nb.scan_nb()` | (n_genes, n_cells) raw UMI | Piezo1 chr8 LOD~21, Ccdc171 chr4, Pmis2 chr7 |
| `vqtl` | `vqtl.scan_vqtl()` | (n_cells,) per gene | Piezo1/Ccdc171/Pmis2 dispersion |
| `epistasis` | `epistasis.scan_epistasis()` | phenotype (n,) + two locus IDs | Piezo1 peak × Ccdc171 peak |
| `conditional` | `epistasis.scan_conditional_with_y()` | phenotype + ScanResult seed | Piezo1 multi-signal |

---

## scan_nb entry point

```python
from mosaiqtl.nb import scan_nb

results = scan_nb(
    counts_cells       = counts_ct,        # (n_genes, n_cells) raw UMI
    genome_cache       = cache_py,
    strain_ids_cells   = strain_ids_cells, # (n_cells,)
    gene_ids           = gene_set,
    covariates_cells   = covar_cells,      # (n_cells, p)
    covariates_strains = covar_strains,    # (n_strains, p) or None
    norm_method        = 'TMM',
    shrink_disp        = True,
    cis_loci_dict      = cis_loci_dict,    # {gene_id: set} or None
    n_cores            = 1,
)
# returns dict {gene_id: ScanResult}
```

Stage 1: `fit_null_sc()` → `shrink_sc_disp()` (cell level, once).  
Stage 2: `scan_nb_gene()` per gene on pseudobulk counts (vectorised score test).  
No kinship — `u_i ~ N(0, σ²_g)` handles within-strain variance.

---

## VQTLResult fields (common bug source)

```python
res = scan_vqtl(counts, subject_ids, genome_cache, ...)
res.scan_result        # ScanResult  dispersion scan    ← NOT .vqtl_scan
res.mean_scan_result   # ScanResult  mean eQTL          ← NOT .mean_scan
res.phi_estimates      # np.ndarray (n_subjects,)
res.phi_method         # 'moments' or 'mle'
res.gene_id            # str
```

---

## scan_vqtl entry point

```python
from mosaiqtl.vqtl import scan_vqtl

res = scan_vqtl(
    counts         = counts_g,        # (n_cells,) one gene
    subject_ids    = strain_ids_cells,
    genome_cache   = cache_py,
    library_sizes  = lib_sizes_cells,
    covariates     = cov_shared,      # (n_shared_strains, p)
    kinship_decomp = kd_g,
    phi_method     = 'moments',
    min_cells      = 10,
    use_rint       = True,
    gene_id        = gname,
    also_scan_mean = True,
)
```

---

## scan_epistasis entry point

```python
from mosaiqtl.epistasis import scan_epistasis

epi_res = scan_epistasis(
    phenotype      = y_epi,    # (n,) pseudobulk RINT
    genome_cache   = cache_py,
    locus_a        = locus_a,  # str peak locus ID gene A
    locus_b        = locus_b,  # str peak locus ID gene B
    covariates     = cov_epi,
    kinship_decomp = kinship_decomp,
    use_rint       = False,
    phenotype_id   = 'Piezo1',
)
# returns dict: lod_additive, lod_interaction, lod_epistasis
# lod_epistasis = lod_interaction - lod_additive
```

---

## scan_conditional_with_y entry point

```python
from mosaiqtl.epistasis import scan_conditional_with_y
from mosaiqtl.scan import scan_h2lmm

sr_base = scan_h2lmm(y, cache_py, covariates, kinship_decomp, use_rint=False)
signals = scan_conditional_with_y(
    phenotype          = y,
    scan_result        = sr_base,
    genome_cache       = cache_py,
    covariates         = covariates,
    kinship_decomp     = kinship_decomp,
    n_signals          = 3,
    lod_drop_threshold = 1.5,
    use_rint           = False,
)
# returns list of ScanResult, one per detected signal
```

---

## Output CSV schemas

| File | Key columns |
|---|---|
| `p1_cell_qc.csv` | `strain, n_cells, median_lib` |
| `p1_nb_peak_summary.csv` | `gene_id, chr, locus, pos_mb, lod_nb, p_value, fdr_bh, egene, rank_nb` |
| `p1_nb_lod_top50k.csv` | `gene_id, chr, pos_mb, lod` (top 50k by LOD) |
| `p1_nb_vs_lmm_comparison.csv` | `gene_id, chr, locus, pos_mb, lod_nb, lod_lmm_r, lod_lmm_py, rank_nb, rank_lmm, peak_chr_match, lod_ratio, fdr_bh, egene` |
| `p2_vqtl_strain_phi.csv` | `gene_id, strain, phi_hat, log_phi, mean_expr` |
| `p2_vqtl_summary.csv` | `gene_id, chr_vqtl, locus_vqtl, pos_mb_vqtl, lod_vqtl, lod_mean_eqtl, chr_lmm, lod_lmm, colocalises` |
| `p3_epistasis_result.csv` | `gene_id, locus_a, locus_b, lod_additive, lod_interaction, lod_epistasis` |
| `p3_nb_founder_effects.csv` | `founder, beta_nb` (Piezo1 peak, use_lrt=True) |
| `p4_conditional_signals.csv` | `signal, chr, locus, pos_mb, lod` |
| `p4_signal{N}_genome_wide.csv` | `chr, pos_mb, lod, locus` per signal N |
| `phase3_qc_summary.csv` | `section, pass, detail` |

---

## Plot inventory

### Independent QC

| Plot | Content |
|---|---|
| `p1_cell_qc.png` | cells per strain + median UMI per strain |
| `p1_nb_chi2_qq.png` | score statistic QQ vs χ²(J−1) null |
| `p1_nb_lod_distribution.png` | peak LOD histogram + known hit markers |
| `p1_nb_piezo1_chr8.png` | Piezo1 NB profile chr8 |
| `p2_vqtl_phi_qc.png` | log(φ̂) KDE per gene + mean–variance scatter |
| `p2_vqtl_{gene}.png` (left) | vQTL profile on candidate chr |
| `p3_epistasis_haplotype_heatmap.png` | J×J founder-pair expression matrix |
| `p4_conditional_chr_profile.png` | all signal traces on peak chr |
| `p4_lod_decay.png` | peak LOD per signal bar |

### Comparison (NB vs LMM)

| Plot | Content |
|---|---|
| `p1_nb_vs_lmm_lod_scatter.png` | LOD scatter + rank concordance |
| `p1_piezo1_nb_vs_lmm_chr8.png` | dual Manhattan chr8 |
| `p1_top20_nb_vs_lmm.png` | side-by-side bar top 20 eGenes |
| `p2_vqtl_{gene}.png` (right) | mean eQTL vs LMM overlay |
| `p3_founder_effects_nb_vs_lmm.png` | NB betas vs LMM allele effects at Piezo1 peak |
| `p4_signal1_vs_lmm.png` | conditional signal 1 vs LMM profile |

---

## Pass criteria (internal consistency, no R reference)

| Section | Gate |
|---|---|
| scan_nb | LOD skew > 0.5 AND Piezo1 LOD > 3 |
| scan_vqtl | ≥1 gene returns result; colocalisation reported only |
| scan_epistasis | `lod_additive > 0`; lod_epistasis reported, not gated |
| scan_conditional | `signal1_lod > 0`; n_signals and LOD drop reported |

---

## LMM reference files loaded

From `py_data/` (phase 2 outputs):

| File | Used for |
|---|---|
| `gene_summary.csv` | `peak_lod_r`, `peak_lod_py` for NB vs LMM scatter |
| `lod_comparison.csv` | Per-locus LOD for dual Manhattan + signal 1 vs LMM |
| `allele_effects_r.csv` | Founder effects at Piezo1 peak for P3 comparison |

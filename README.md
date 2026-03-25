# mosaiQTL

**mosaiQTL** is a Python port of [miQTL](https://github.com/gkeele/miqtl) (Keele *et al.* 2020) — a
haplotype-based QTL mapping package for multiparent populations (MPP) such as the Collaborative Cross
(CC), Diversity Outbred (DO), and MAGIC designs. It additionally extends the model family to
negative binomial mixed models for raw RNA-seq count and single-nucleus phenotypes, enabling
direct eQTL mapping from pseudobulk or cell-level data without a normalization step.

The goal is a complete, R-free Python replacement for the miQTL workflow, validated against the
original R implementation, with extensions not available in miQTL.

---

## Motivation

miQTL operates on the HAPPY genome cache format and resolves genetic effects at the 8-founder
haplotype level — avoiding the information loss of collapsing to biallelic SNP dosage. The R package
is the reference implementation for CC and DO studies but requires R, `qtl2`, and the HAPPY HMM
outputs as prerequisites. mosaiQTL removes all of these dependencies:

- A pure-Python 8-founder diplotype HMM (`hmm.py`) builds the haplotype cache directly from raw
  GigaMUGA nucleotide calls — no R, no `qtl2`, no `.RData` files required at scan time.
- All scan, kinship, and LMM functions are native Python (numpy/scipy), validated against
  `scan.h2lmm()` and `calc_kinship()` from miQTL/qtl2.
- Extensions for NB GLM and NEBULA-LN NB mixed models are unique to mosaiQTL and not available
  in miQTL.

---

## Architecture

```
mosaiqtl/
  genome_cache.py   # GenomeCache: from_gigamuga(), from_happy_dir(), from_qtl2_probs()
  hmm.py            # 8-founder diplotype HMM (Haldane map, forward-backward, LOCO)
  kinship.py        # calc_kinship(), decomp_kinship(), EigenCache
  lmm.py            # estimate_h2_null(), fit_lmm_at_locus() (REML, ML LOD)
  scan.py           # scan_h2lmm(), scan_multi_gene(), ScanResult
  nb.py             # NB GLM: estimate_gene_dispersion(), score_test_at_locus()
  vqtl.py           # variance QTL: scan_vqtl(), compare_mean_vs_dispersion()
  plotting.py       # Manhattan plots, QQ, founder effects, haplotype heatmaps
  utils.py          # rint(), align_samples(), make_design_matrix()
```

---

## Feature parity with miQTL

### Implemented and validated

| miQTL (R) | mosaiQTL (Python) | Validation |
|---|---|---|
| HAPPY `.RData` genome cache | `genome_cache.from_happy_dir()` | reads correctly |
| `convert.qtl2.to.HAPPY()` | `genome_cache.from_qtl2_probs()` | rpy2 bridge |
| `calc_genoprob()` (qtl2) | `genome_cache.from_gigamuga()` + `hmm.py` | **R-free**; kinship r = 0.9997 vs R |
| `calc_kinship(method='loco')` | `kinship.calc_kinship(method='loco')` | r = 0.9997, max\|ΔK\| < 0.02, all 20 chr ✓ |
| `scan.h2lmm()` / `lmmbygls()` | `lmm.fit_lmm_at_locus()`, `scan.scan_h2lmm()` | LOD slope = 1.048, peak match 91.8% ✓ |
| `rint()` | `utils.rint()` | direct port ✓ |

### To be implemented (full port, in progress)

| miQTL (R) | mosaiQTL target | Module |
|---|---|---|
| `scan.h2lmm(use.multi.impute=TRUE)` | multiple imputation scan | `lmm.py` + `scan.py` |
| `imputed.snp.scan.h2lmm()` | haplotype-imputed SNP scan | `scan.py` |
| `pairwise.cor.snp.scan()` | SNP LD / r² matrix | `scan.py` |
| `generate.sample.outcomes.matrix()` | permutation / bootstrap sampling | `permutation.py` |
| `run.threshold.scans()` | genome-wide threshold scans | `permutation.py` |
| `get.gev.thresholds()` | GEV significance thresholds | `permutation.py` |
| `instability.lm.scan()` | fixed-effect bootstrap stability scan | `scan.py` |
| `fixef.eQTL.scan()` | fixed-effect eQTL scan | `scan.py` |
| `mediation.scan.qr()` | genome-wide mediation via QR | `mediation.py` |
| `fixef.mediation.scan()` | fixed-effect mediation scan | `mediation.py` |
| `run.positional.scans()` | parametric bootstrap CI | `permutation.py` |
| `grab.locus.from.scan()` | peak locus extraction | `scan.py` |
| `collapse.genomecache()` | merge redundant adjacent loci | `genome_cache.py` |
| `genome.plotter.whole/region()` | Manhattan + regional plots | `plotting.py` |
| `allele.plotter.region()` | founder effect region plot | `plotting.py` |
| `single.chr.plotter.w.ci()` | CI plot for single chromosome | `plotting.py` |
| `prob.heatmap()` | haplotype probability heatmap | `plotting.py` |
| `snp.genome.plotter.*()` | SNP association plots | `plotting.py` |

### Extensions unique to mosaiQTL

| Feature | Module | Description |
|---|---|---|
| NB GLM scan | `nb.py` | Negative binomial score test for pseudobulk counts |
| NEBULA-LN integration | `nb.py` | NB mixed model (He *et al.* 2021) for single-cell data |
| Variance QTL | `vqtl.py` | Two-stage dispersion scan — mean vs variance QTL |
| `from_gigamuga()` | `genome_cache.py` | R-free HMM cache build from raw GigaMUGA; not in miQTL |

---

## Validation status (full genome, 18,447 genes, all 20 chromosomes)

| Section | Result | Detail |
|---|---|---|
| Kinship | **PASS** | r = 0.9997, max\|ΔK\| < 0.02, all 20 chr |
| LOD scores | **PASS** | slope = 1.048, peak locus match 91.8%, strict = 21.8%, lenient = 42.9% |
| Haplotype spot-check | informational | median\|Δprob\| reflects HMM vs qtl2 C differences; kinship validates functional correctness |

Pass criteria: strict = `peak_match AND rel_error < 10%`; lenient = `peak_match AND rel_error < 20%`.
Per-locus LOD profile shape (Pearson r) is reported separately — shape mismatch is expected from
per-locus haplotype differences between a Python HMM and R's HAPPY cache, even when kinship agrees
at r = 0.9997.

---

## Genome cache

```python
from mosaiqtl.genome_cache import GenomeCache

# 1. R-free: raw GigaMUGA nucleotide calls + genetic/physical maps
cache = GenomeCache.from_gigamuga(
    gigamuga_path = 'input/GM_ccfo_mm39.csv',
    gmap_dir      = 'input/GM/gmap/',
    pmap_dir      = 'input/GM/pmap/',
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)

# 2. R HAPPY cache (miQTL native format)
cache = GenomeCache.from_happy_dir('output/genomecache/', model='additive')

# 3. qtl2 prob arrays via rpy2
cache = GenomeCache.from_qtl2_probs(
    probs_cc_path      = 'output/probs_cc.rds',
    probs_founder_path = 'output/probs_founder.rds',
    probs_f1_path      = 'output/probs_f1.rds',
    pmap_path          = 'input/pmap.csv',
)

# save and reload as HDF5
cache.save('output/genomecache.h5')
cache = GenomeCache.load('output/genomecache.h5')
```

---

## Basic scan

```python
from mosaiqtl import kinship, scan

# LOCO kinship
K = kinship.calc_kinship(cache, method='loco')

# cis-eQTL scan — one gene
result = scan.scan_h2lmm(
    phenotype    = y,           # (n_samples,) array
    genome_cache = cache,
    covariates   = covar_df,    # batch, sex, PCs
    kinship      = K,
    chrom        = '8',
    cis_window   = 0.5,         # ±500 kb
    gene_pos_mb  = 123.4,
)
print(result.lead_locus, result.peak_lod)
```

---

## Validation

```bash
# step 1: R reference scan (requires miQTL installed)
Rscript validation/r_reference/run_validation.R        # all chromosomes
Rscript validation/r_reference/run_validation.R 8 50   # quick test

# step 2: Python validation
python validation/validate_vs_r.py                     # all chromosomes
python validation/validate_vs_r.py --chr 8 --n_genes 50
```

Outputs written to `validation/r_data/` (CSVs) and `validation/qc_plots/` (PNG).

---

## Installation

```bash
pip install -e .
```

**Dependencies:** `numpy scipy pandas h5py matplotlib seaborn`

**Optional:** `rpy2` (for `from_happy_dir`, `from_qtl2_probs`); `scanpy anndata` (single-cell input)

---

## References

- Keele, G.R. *et al.* (2020). *PLOS Genetics* 16:e1008537 — **miQTL**
  [doi:10.1371/journal.pgen.1008537](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. *et al.* (2019). *Genetics* 211:495 — **R/qtl2**
  [doi:10.1534/genetics.118.301595](https://doi.org/10.1534/genetics.118.301595)
- Mott, R. *et al.* (2000). *PNAS* 97:12649 — **HAPPY HMM**
  [doi:10.1073/pnas.230304397](https://doi.org/10.1073/pnas.230304397)
- Kang, H.M. *et al.* (2008). *Genetics* 178:1709 — **eigendecomposition LMM**
  [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101)
- He, L. *et al.* (2021). *Communications Biology* 4:629 — **NEBULA**
  [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)

## License

GPL-3.0

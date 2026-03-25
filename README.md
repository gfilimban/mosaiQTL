# mosaiQTL

mosaiQTL is a model-agnostic QTL mapping package for multiparent populations: Gaussian LMM
for normalized expression, [edgePython](https://github.com/pachterlab/edgePython)-backed NB GLM
for pseudobulk counts, and NEBULA-LN for single-nucleus data — chosen per gene without changing
the rest of the pipeline.

It resolves genetic effects to $J$ inbred founder haplotypes across any number of mosaic lines,
parametric on both $J$ and population size. Designed for populations where each individual genome is
a unique recombinant mosaic of segments inherited from $J$ founders: Collaborative Cross (CC, $J=8$,
~33 lines), Diversity Outbred (DO), or any user-defined MAGIC design. Informed by
[miQTL](https://github.com/gkeele/miqtl) (Keele *et al.* 2020) and
[R/qtl2](https://kbroman.org/qtl2/) (Broman *et al.* 2019).

---

## Current phase — Phase 2: full miQTL port

**Phase 1 (complete):** Core LMM engine validated against R miQTL on a full genome scan
(18,447 genes, all 20 chromosomes). All three QC sections pass.

**Phase 2 (in progress):** Complete port of all remaining miQTL functions — multiple imputation,
SNP scan, permutation thresholds, mediation, positional CI, and publication-quality plots — to
eliminate R as a dependency entirely for analyses.

**Phase 3 (planned):** Novel extensions — NB GLM scan, NEBULA-LN NB mixed model for single-nucleus
data, variance QTL.

---

## Validation status (Phase 1 — full genome)

| Section | Status | Detail |
|---|---|---|
| Haplotype probs | **PASS** | row sums correct; spot-check delta informational (HMM vs qtl2 C impl.) |
| LOCO kinship | **PASS** | r = 0.9997, max\|ΔK\| < 0.02, all 20 chromosomes |
| LOD scores | **PASS** | slope = 1.048, peak locus match 91.8%, 77.7% pass (strict + lenient) |

Pass criteria: strict = `peak_match AND rel_error < 10%`; lenient = `peak_match AND rel_error < 20%`.

---

## Architecture

```
mosaiqtl/
  genome_cache.py   # GenomeCache — from_gigamuga(), from_happy_dir(), save/load HDF5
  hmm.py            # 8-founder diplotype HMM — Haldane map, forward-backward
  kinship.py        # calc_kinship() LOCO, decomp_kinship(), EigenCache
  lmm.py            # estimate_h2_null() REML, fit_lmm_at_locus() ML LOD
  scan.py           # scan_h2lmm(), scan_multi_gene(), ScanResult
  nb.py             # NB GLM: dispersion estimation, score test (Phase 3)
  vqtl.py           # variance QTL: scan_vqtl(), mean vs dispersion (Phase 3)
  plotting.py       # LOD profiles, kinship heatmap, founder effects, QQ
  utils.py          # rint(), align_samples(), make_design_matrix()

validation/
  lmm_validation/           # Phase 1 outputs (LMM scan vs miQTL)
    r_data/                 # CSVs: lod_r.csv, gene_results.csv, kinship_report.csv, ...
    qc_plots/               # PNGs: lod_correlation, kinship_heatmap, gene_summary, ...
  port_validation/          # Phase 2 outputs (full port vs miQTL)
    r_data/
    qc_plots/
  r_reference/
    run_validation.R        # R reference scan using miQTL::scan.h2lmm()
  validate_vs_r.py          # Python validation script (all 4 sections)
```

---

## Feature parity with miQTL — implementation plan

### Phase 1 — implemented and validated ✓

| miQTL (R) | mosaiQTL (Python) | Notes |
|---|---|---|
| `calc_genoprob()` (qtl2) + HAPPY build | `genome_cache.from_gigamuga()` | **R-free**; pure Python HMM |
| HAPPY `.RData` cache | `genome_cache.from_happy_dir()` | rpy2 bridge (compat only) |
| `convert.qtl2.to.HAPPY()` | `genome_cache.from_rds()` | rpy2 bridge (compat only) |
| `calc_kinship(loco)` | `kinship.calc_kinship('loco')` | validated r = 0.9997 |
| `scan.h2lmm()` / `lmmbygls()` | `scan.scan_h2lmm()` + `lmm.fit_lmm_at_locus()` | validated slope = 1.048 |
| `rint()` | `utils.rint()` | direct port |

### Phase 2 — in progress

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
| `single.chr.plotter.w.ci()` | CI plot with confidence interval | `plotting.py` |
| `prob.heatmap()` | haplotype probability heatmap | `plotting.py` |
| `snp.genome.plotter.*()` | SNP association plots | `plotting.py` |

### Phase 3 — extensions unique to mosaiQTL

| Feature | Module | Description |
|---|---|---|
| NB GLM scan | `nb.py` | Negative binomial score test for pseudobulk counts |
| NEBULA-LN | `nb.py` | NB mixed model (He *et al.* 2021) for single-cell data |
| Variance QTL | `vqtl.py` | Two-stage dispersion scan — mean vs variance QTL |

---

## Usage

```python
from mosaiqtl.genome_cache import GenomeCache
from mosaiqtl import kinship, scan

# build R-free cache from raw GigaMUGA data
cache = GenomeCache.from_gigamuga(
    gigamuga_path = 'input/GM_ccfo_mm39.csv',
    gmap_dir      = 'input/GM/gmap/',
    pmap_dir      = 'input/GM/pmap/',
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)
cache.save('output/cache.h5')          # save once
cache = GenomeCache.load('output/cache.h5')  # fast reload

K = kinship.calc_kinship(cache, method='loco')

result = scan.scan_h2lmm(
    phenotype    = y,
    genome_cache = cache,
    covariates   = covar_df,
    kinship      = K,
    chrom        = '8',
    cis_window   = 0.5,
    gene_pos_mb  = 123.4,
)
```

---

## Validation

```bash
# Phase 1 — LMM validation (R required for reference scan)
Rscript validation/r_reference/run_validation.R
python validation/validate_vs_r.py

# Phase 2 — full port validation (validate_port.py, built alongside the port)
python validation/validate_port.py
```

---

## Installation

```bash
pip install -e .
```

**Dependencies:** `numpy scipy pandas h5py matplotlib seaborn`  
**Optional:** `rpy2` (backwards-compat readers only)

---

## References

- Keele, G.R. *et al.* (2020). *PLOS Genetics* 16:e1008537 — **miQTL** [doi:10.1371/journal.pgen.1008537](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. *et al.* (2019). *Genetics* 211:495 — **R/qtl2** [doi:10.1534/genetics.118.301595](https://doi.org/10.1534/genetics.118.301595)
- Mott, R. *et al.* (2000). *PNAS* 97:12649 — **HAPPY HMM** [doi:10.1073/pnas.230304397](https://doi.org/10.1073/pnas.230304397)
- Kang, H.M. *et al.* (2008). *Genetics* 178:1709 — **eigendecomposition LMM** [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101)
- He, L. *et al.* (2021). *Communications Biology* 4:629 — **NEBULA** [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)

## License

GPL-3.0

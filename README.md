# mosaiQTL

**mosaic-QTL (mosaiQTL)** is a model-agnostic haplotype-based QTL mapping
package for multiparent populations (MPP) — such as the Collaborative Cross
(CC), Diversity Outbred (DO), and Multiparent Advanced Generation Inter-Cross
(MAGIC) — in which each individual carries a unique mosaic of alleles inherited
from *J* inbred founder strains. By mapping which founder's haplotype segment
at each locus drives variation in a phenotype, mosaiQTL resolves genetic
effects at the resolution of the original inbred lines rather than collapsing
to biallelic SNP dosage.

mosaiQTL is a Python port and extension of
[miQTL](https://github.com/gkeele/miqtl) (Keele *et al.* 2020), an R package
for haplotype-based QTL mapping in multiparent populations. miQTL's core scan
engine is `scan.h2lmm()` — a Gaussian LMM that uses the
[HAPPY](http://www.well.ox.ac.uk/~rdott/happy.html) genome cache format
(Mott *et al.* 2000) and supports both regression on probabilities (ROP) and
multiple imputation. mosaiQTL ports `scan.h2lmm()` to Python and extends it
with negative binomial and NB mixed models suited for raw RNA-seq count
phenotypes and single-nucleus data.

Generalizes miQTL / [R/qtl2](https://kbroman.org/qtl2/) to support negative
binomial, Gaussian LMM, and NB mixed models on raw counts, normalized
expression, or collapsed genotype means. Uses
[edgePython](https://github.com/pachterlab/edgePython) (Pachter 2026) for NB
fitting infrastructure.

## Relationship to miQTL

| miQTL (R) | mosaiQTL (Python) | Notes |
|-----------|-------------------|-------|
| `scan.h2lmm()` | `lmm.scan_h2lmm()` | Ported; cross-validated against R |
| `lmmbygls()` | `lmm.fit_lmm_at_locus()` | ML LOD formula; Kang 2008 eigendecomposition |
| `calc_kinship()` (qtl2) | `kinship.calc_kinship()` | LOCO kinship; validated vs R |
| `convert.qtl2.to.HAPPY()` | `genome_cache.from_rds()` | rpy2 bridge; reads probs_cc/founder/f1.rds |
| HAPPY `.RData` cache | `genome_cache.from_happy_dir()` | direct read |
| `calc_genoprob()` (qtl2) | `genome_cache.from_gigamuga()` + `hmm.py` | **R-free** diplotype HMM from raw GigaMUGA |
| `scan.h2lmm(model='additive')` | NB GLM (`nb.py`) | **Extension**: raw count phenotypes |
| permutation thresholds | *planned* | GEV fit |
| parametric bootstrap CI | *planned* | positional confidence |
| `genome.plotter.whole()` | *planned* | manhattan plot |
| multiple imputation | *not planned* | ROP only |

## Status

**Under active development — pre-release (v0.2.0).**

### Validated

| Module | Description | Status |
|--------|-------------|--------|
| `genome_cache.from_rds()` | merge CC + founder + F1 diplotype probs via rpy2, collapse to additive | ✓ runs |
| `genome_cache.from_gigamuga()` | R-free CC HMM from raw GigaMUGA nucleotide calls | ✓ runs — validation in progress |
| `genome_cache.from_happy_dir()` | read R HAPPY `.RData` cache | ✓ validated |
| `kinship.calc_kinship(loco)` | LOCO kinship from Python cache | cross-validating vs R |
| `lmm.estimate_h2_null()` | REML h² on rotated phenotype | ✓ validated |
| `lmm.fit_lmm_at_locus()` | ML LOD, F-test p-value, founder effects | ✓ validated; r≈1 vs miQTL |
| `scan.scan_h2lmm()` | genome-wide scan orchestrator | ✓ functional |

### In progress

| Module | Issue |
|--------|-------|
| `from_gigamuga()` | HMM probs being compared to R qtl2 probs |
| `kinship` | comparing Python LOCO K vs `kinship_loco.rds` |
| `nb.py` | edgePython NB GLM integration (scaffold exists) |

### Planned

| Module | Description |
|--------|-------------|
| `permutation` | genome-wide thresholds (GEV fit), mirroring miQTL's `run.threshold.scans()` |
| `bootstrap_ci` | positional confidence, mirroring miQTL's `run.positional.scans()` |
| `gxe` | haplotype × environment interaction |
| `trans` | genome-wide trans-eQTL scan |
| `sqtl` | splicing QTL via PSI phenotypes |
| `snp_scan` | founder-imputed biallelic SNP association |

## Genome cache loaders

mosaiQTL can build or read the haplotype probability cache three ways:

```python
from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS

# 1. R-free: raw GigaMUGA genotype CSV + genetic/physical maps
#    runs the 8-founder diplotype HMM (forward-backward) in pure Python
cache = GenomeCache.from_gigamuga(
    gigamuga_path = 'input/GM_ccfo_mm39.csv',
    gmap_dir      = 'input/GM/gmap/',   # GM_gmap1.csv, GM_gmap2.csv, ...
    pmap_dir      = 'input/GM/pmap/',
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)

# 2. rpy2 bridge: reads probs_cc.rds + probs_founder.rds + probs_f1.rds
#    mirrors founders_f1_genomecache.ipynb steps 5-7
cache = GenomeCache.from_rds(
    probs_cc_path      = 'output/probs_cc.rds',
    probs_founder_path = 'output/probs_founder.rds',
    probs_f1_path      = 'output/probs_f1.rds',
    pmap_path          = 'input/pmap.csv',
)

# 3. read R HAPPY cache (miQTL native format)
cache = GenomeCache.from_happy_dir('output/genomecache/', model='additive')
```

## Validation

Three-layer cross-language validation against R miQTL:

```bash
# prerequisite: run the R reference scan
Rscript validation/r_reference/run_validation.R 8 20

# full validation (all chromosomes, all genes)
python validation/validate_vs_r.py

# quick test
python validation/validate_vs_r.py --chr 8 --n_genes 20
```

Validation sections:
1. **Haplotype probs** — Python `from_gigamuga()` vs R HAPPY cache (`haplotype_report.csv`)
2. **Kinship** — Python LOCO K vs `kinship_loco.rds` (`kinship_report.csv`)
3. **LOD scores** — Python `fit_lmm_at_locus()` vs `scan.h2lmm()` (`gene_results.csv`)

Pass thresholds: strict (r > 0.999, max|ΔLOD| < 0.01) or lenient (r > 0.990,
relative error < 5%). With correct haplotype probs, the LMM scan achieves
strict PASS on the majority of chr 8 genes.

QC plots saved to `validation/qc_plots/`:
`haplotype_probs.png`, `kinship_heatmap.png`, `lod_correlation.png`,
`lod_residuals.png`, `pvalue_qq.png`, `founder_effects.png`, `gene_summary.png`

## Installation

```bash
pip install -e .
```

## Dependencies

- **numpy, scipy, pandas** — core
- **rpy2** — optional, for `from_rds()` and `from_happy_dir()`
- **edgePython** — optional, for NB GLM and NEBULA-LN
- **h5py, scanpy, anndata** — optional, for single-cell input

## References

- Keele, G.R. et al. (2020). *PLOS Genet.* 16:e1008537. **miQTL**
  [doi:10.1371/journal.pgen.1008537](https://doi.org/10.1371/journal.pgen.1008537)
- Mott, R. et al. (2000). *PNAS* 97:12649. **HAPPY HMM**
  [doi:10.1073/pnas.230304397](https://doi.org/10.1073/pnas.230304397)
- Broman, K.W. et al. (2019). *Genetics* 211:495. **R/qtl2**
  [doi:10.1534/genetics.118.301595](https://doi.org/10.1534/genetics.118.301595)
- Kang, H.M. et al. (2008). *Genetics* 178:1709. **Eigendecomposition LMM**
  [doi:10.1534/genetics.107.080101](https://doi.org/10.1534/genetics.107.080101)
- He, L. et al. (2021). *Commun. Biol.* 4:629. **NEBULA**
  [doi:10.1038/s42003-021-02146-6](https://doi.org/10.1038/s42003-021-02146-6)
- Pachter, L. (2026). **edgePython**
  [doi:10.64898/2026.02.16.706223](https://doi.org/10.64898/2026.02.16.706223)

## Acknowledgments

Core architecture, LMM engine, genome cache, HMM, and validation framework
developed with assistance from Claude (Anthropic).

## License

GPL-3.0 — see [LICENSE](LICENSE).

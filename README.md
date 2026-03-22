# mosaiQTL

Model-agnostic haplotype-based QTL mapping for multiparent populations
(CC, DO, MAGIC), with NB GLM as the primary engine for count-based
phenotypes and NEBULA-LN for cell-level single-cell modeling.

Generalizes [miQTL](https://github.com/gkeele/miqtl) /
[R/qtl2](https://kbroman.org/qtl2/) to support negative binomial,
Gaussian LMM, and NB mixed models on raw counts, normalized expression,
or collapsed genotype means. Uses
[edgePython](https://github.com/pachterlab/edgePython) (Pachter 2026)
for NB fitting infrastructure.

## Status

**Under active development — pre-release (v0.2.0).**

### Implemented and validated (on simulated CC data)

| Module | What it does | R reference | Notes |
|--------|-------------|-------------|-------|
| `genome_cache` | Read/write HAPPY genome cache (HDF5) | HAPPY / qtl2 probs | HDF5 path works; `.RData` loader needs fix (`load()` not `readRDS()`) |
| `kinship` | LOCO kinship + eigendecomposition | `qtl2::calc_kinship()` | Correct structure; pending cross-validation against R output |
| `lmm` | Eigendecomposition-accelerated LMM | `miQTL::lmmbygls()` | F-test per locus; pending covariate residualization optimization |
| `scan` | Genome-wide haplotype scan orchestrator | `miQTL::scan.h2lmm()` | LMM path only; model dispatch (`model='nb'`) not yet implemented |
| `vqtl` | Two-stage dispersion vQTL | *novel* | Moment estimator validated (r=0.84 with truth); no shrinkage yet |
| `utils` | RINT, sample alignment, design matrices | `miQTL::rint()` | Parametric on J founders |

### Implemented but needs fix

| Module | Issue | Plan |
|--------|-------|------|
| `nebula_ln` | Loglikelihood term1 approximation has ~3000% error | Replace with edgePython `glm_sc_fit()` wrapper |
| `genome_cache` | `from_happy_dir()` uses `readRDS()` but real cache uses `load()` | Fix to use rpy2 `load()` or native `.RData` reader |
| `scan` | K expansion (48 to n) validated but not wired into scan loop | Integrate for replicate-aware Path B |

### Planned

| Module | What it will do | Depends on |
|--------|----------------|------------|
| `nb_glm` | NB GLM score test wrapping edgePython `glm_fit()` + `estimate_disp()` | edgePython integration |
| `gxe` | Genotype x environment (haplotype x sex) interaction scan | `nb_glm` or `lmm` |
| `trans` | Genome-wide trans-eQTL scan with per-locus LOCO | `scan` + model dispatch |
| `sqtl` | Splicing QTL via PSI phenotypes | `scan` + junction quantification |
| `merge` | Allelic series / founder grouping contrasts | `scan` + Stirling enumeration |
| `snp_scan` | Founder-imputed biallelic SNP association | `scan` + SNP dosage from H |
| `permutation` | Genome-wide permutation thresholds (GEV fit) | `scan` |

### Planned optimizations

| Optimization | Where | Impact |
|-------------|-------|--------|
| Residualize covariates before per-locus scan | `lmm.py` | ~7x speedup per locus |
| Streaming kinship (don't load full 3D tensor) | `kinship.py` | Scales to DO (n > 500) |
| Saddlepoint approximation for NB score test p-values | `nb_glm.py` | Accurate tail p-values |

## Installation

```bash
pip install -e .
```

For full single-cell support:
```bash
pip install -e ".[full]"
```

## Quick start

```python
import mosaiqtl as mq

# load haplotype probabilities (n_genotypes x j founders per locus)
gc = mq.genome_cache.GenomeCache.from_hdf5('genomecache.h5')

# compute loco kinship + eigendecomposition
k_loco = mq.kinship.calc_kinship(gc, method='loco')
k_decomp = mq.kinship.decomp_kinship(k_loco)

# current: lmm on rint'd expression (miqtl-equivalent)
result = mq.scan.scan_h2lmm(
    phenotype=y_rint,          # rint'd log-cpm
    genome_cache=gc,
    covariates=x,              # intercept + pcs
    kinship_decomp=k_decomp,
    phenotype_id='my_gene',
)

# coming: nb glm on raw pseudobulk counts (primary mode)
# result = mq.scan.scan_h2lmm(
#     phenotype=counts,        # raw pseudobulk umi counts
#     genome_cache=gc,
#     covariates=x,            # intercept + sex + batch + pcs
#     kinship_decomp=k_decomp,
#     model='nb',              # negative binomial glm
#     phenotype_id='my_gene',
# )

peak = result.peak_locus()
print(f'peak: chr{peak["chromosome"]} lod={peak["lod"]:.2f}')
```

## Validation

Cross-language validation against R miQTL/qtl2 on simulated CC data
(48 genotypes: 8 founders + 7 F1s + 33 CC strains, 8 replicates each):

```bash
python validation/validate_vs_r.py
```

Current validation results (5/5 passing):
- kinship: LOCO matrices symmetric, PSD, correct founder/CC structure
- k_expansion: 48 to 384 index replication matches R `k[strain_idx, strain_idx]`
- scan: causal QTL localized to correct chromosome (LOD=14.36)
- dispersion: per-subject phi correlates 0.84 with truth
- vqtl: dispersion QTL detected on correct chromosome

To compare against R outputs (requires R with miQTL installed):
```bash
Rscript validation/r_reference/run_validation.R validation/r_data/
```

## Workflow

mosaiQTL's scan workflow mirrors
[edgePython](https://github.com/pachterlab/edgePython)'s differential
expression pipeline, extended with a per-locus haplotype scan loop:

```
edgePython                          mosaiQTL addition
---                                 ---
make_dgelist(counts)                genome_cache.load(haplotypes)
calc_norm_factors(tmm)              kinship.calc_kinship_loco()
estimate_disp(common+trend+tag)     kinship.decomp_kinship()
glm_fit(null model, per gene)       |
                                    +-- for each locus l:
                                    |     build H_l^{(-J)} design matrix
                                    |     score test or LRT vs null
                                    +-- collect LOD, p, founder effects
                                    |
top_tags(fdr correction)            permutation thresholds (GEV)
```

For cell-level scans, the equivalent flow uses edgePython's
`glm_sc_fit()` followed by `shrink_sc_disp()` followed by mosaiQTL's per-locus
haplotype test.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full mathematical
framework (LaTeX notation), all mapping models, and planned extensions.

## Dependencies

### edgePython (hybrid dependency)

mosaiQTL uses [edgePython](https://github.com/pachterlab/edgePython)
(Pachter 2026) for:
- NB GLM fitting (`glm_fit`, `glm_ql_fit`)
- Dispersion estimation with empirical Bayes shrinkage (`estimate_disp`, WLEB)
- TMM normalization (`calc_norm_factors`)
- NEBULA-LN mixed model (`glm_sc_fit`, `shrink_sc_disp`, `glm_sc_test`)

mosaiQTL provides what edgePython does not:
- J-founder haplotype design matrix and genome cache
- LOCO kinship with eigendecomposition acceleration
- Per-locus haplotype score test / F-test / LRT
- Dispersion vQTL (two-stage scan)
- GxE, trans, sQTL, allelic series extensions
- LMM port of miQTL for cross-language validation

### Other dependencies

- numpy, scipy, pandas (core)
- h5py, scanpy, anndata (optional, for single-cell data)
- rpy2 (optional, for reading R `.RData` genome cache files)

## References

- Pachter, L. (2026). Differential analysis of genomics count data
  with edge\*. *bioRxiv*.
  [doi:10.64898/2026.02.16.706223](https://doi.org/10.64898/2026.02.16.706223)
- He, L. et al. (2021). *Commun. Biol.* 4:629.
  [NEBULA](https://doi.org/10.1038/s42003-021-02146-6)
- Kang, H.M. et al. (2008). *Genetics* 178:1709.
  [Eigendecomposition LMM](https://doi.org/10.1534/genetics.107.080101)
- Keele, G.R. et al. (2020). *PLOS Genet.* 16:e1008537.
  [miQTL](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. et al. (2019). *Genetics* 211:495.
  [R/qtl2](https://doi.org/10.1534/genetics.118.301595)
- McCarthy, D.J. et al. (2012). *Nucleic Acids Res.* 40:4288.
  [edgeR GLM](https://doi.org/10.1093/nar/gks042)

## Acknowledgments

Core package architecture, LMM engine, validation framework, and
documentation developed with assistance from Claude (Anthropic).

## License

GPL-3.0 — see [LICENSE](LICENSE).

# mosaiQTL

Haplotype-based QTL mapping for multiparent populations (CC, DO, MAGIC),
with NEBULA-LN cell-level negative binomial modeling for single-cell eQTL,
dispersion vQTL, GxE interaction, trans, and splicing QTL mapping.

A Python port of [miQTL](https://github.com/gkeele/miqtl) /
[R/qtl2](https://kbroman.org/qtl2/) extended to single-cell resolution.

## Status

**Under active development — pre-release.**

| Module | Status | R equivalent |
|--------|--------|--------------|
| `genome_cache` | ✓ validated | HAPPY genome cache |
| `kinship` | ✓ validated | `qtl2::calc_kinship()` |
| `lmm` | ✓ validated | `miQTL::lmmbygls()` |
| `scan` | ✓ validated | `miQTL::scan.h2lmm()` |
| `vqtl` | ✓ validated | *novel — no R equivalent* |
| `nebula_ln` | in progress | `nebula::nebula()` |
| `gxe` | planned | *novel* |
| `trans` | planned | *novel* |
| `sqtl` | planned | *novel* |
| `permutation` | planned | `miQTL` permutation scan |

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

# load haplotype probabilities (48 genotypes × 8 founders per locus)
gc = mq.genome_cache.GenomeCache.from_hdf5('genomecache.h5')

# compute loco kinship + eigendecomposition
k_loco = mq.kinship.calc_kinship(gc, method='loco')
k_decomp = mq.kinship.decomp_kinship(k_loco)

# scan one gene
result = mq.scan.scan_h2lmm(
    phenotype=y,             # rint'd expression (48 strain means)
    genome_cache=gc,
    covariates=x,            # intercept + pcs
    kinship_decomp=k_decomp,
    phenotype_id='my_gene',
)

peak = result.peak_locus()
print(f'peak: chr{peak["chromosome"]} lod={peak["lod"]:.2f}')
```

## Validation

Cross-language validation against R miQTL/qtl2 on simulated CC data:

```bash
python validation/validate_vs_r.py
```

To compare against R outputs (requires R with miQTL installed):
```bash
Rscript validation/r_reference/run_validation.R validation/r_data/
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full mathematical
framework, all mapping models, and planned extensions.

## References

- He, L. et al. (2021). *Commun. Biol.* 4:629. [NEBULA](https://doi.org/10.1038/s42003-021-02146-6)
- Kang, H.M. et al. (2008). *Genetics* 178:1709. [eigendecomposition LMM](https://doi.org/10.1534/genetics.107.080101)
- Keele, G.R. et al. (2020). *PLOS Genet.* 16:e1008537. [miQTL](https://doi.org/10.1371/journal.pgen.1008537)
- Broman, K.W. et al. (2019). *Genetics* 211:495. [R/qtl2](https://doi.org/10.1534/genetics.118.301595)

## License

MIT

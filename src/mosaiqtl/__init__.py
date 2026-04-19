"""
mosaiqtl — model-agnostic haplotype-based QTL mapping for multiparent populations

three-layer API:
  Layer 1 — discovery        scan_nb(), scan_h2lmm(), pipeline.discover()
  Layer 2 — characterization pipeline.characterize()  (allele effects, vqtl, mediation)
  Layer 3 — resolution       pipeline.resolve()       (snp catalogue, conditional, ase, sqtl)

supports nb glm (primary), gaussian lmm, and cell-level nb via edgepython.
parametric on J founders — no hardcoded assumptions about population size.
"""

__version__ = '0.2.0'

# primary model — NB GLM (NEBULA-LN cell level + pseudobulk scan)
from .nb import (scan_nb, scan_nb_gene,                             # noqa: F401
                  fit_null_sc, fit_null_nb_glmm_loco,
                  pseudobulk_sum)

# gaussian lmm scan
from .scan import scan_h2lmm, scan_multi_gene, ScanResult  # noqa: F401

# snp catalogue (Layer 3a)
from .snp_scan import build_snp_catalogue           # noqa: F401

# three-layer user API
from .pipeline import discover, characterize, resolve       # noqa: F401

# epistasis and multi-signal decomposition
from .epistasis import (scan_epistasis,                     # noqa: F401
                         test_haplotype_combinations,
                         scan_conditional,
                         scan_conditional_with_y)

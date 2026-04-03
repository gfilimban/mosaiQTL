"""
mosaiqtl — model-agnostic haplotype-based qtl mapping for multiparent populations

supports nb glm (primary), gaussian lmm (miqtl port), and nb-lmm (nebula-ln)
on raw counts, normalized expression, or collapsed genotype means. user
specifies j founders, aggregation level, and statistical model at scan time.
"""

__version__ = '0.2.0'

# primary model — NB GLM on raw pseudobulk counts
from .nb import scan_nb, scan_nb_multi_gene  # noqa: F401

# validation port — Gaussian LMM (miQTL-equivalent)
from .scan import scan_h2lmm, scan_multi_gene, ScanResult  # noqa: F401

# three-layer user API
from .pipeline import discover, characterize, resolve  # noqa: F401

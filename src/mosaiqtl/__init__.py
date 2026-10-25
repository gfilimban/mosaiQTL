"""
mosaiqtl — model-agnostic haplotype-based QTL mapping for multiparent populations

three-layer API:
  Layer 1 — discovery        scan_nb(), scan_h2lmm(), pipeline.discover()
  Layer 2 — characterization pipeline.characterize()  (allele effects, vqtl, mediation)
  Layer 3 — resolution       pipeline.resolve()       (snp catalogue, conditional, ase, sqtl)

supports nb glm (primary), gaussian lmm, and cell-level nb via edgepython.
parametric on J founders — no hardcoded assumptions about population size.
"""

try:
    from importlib.metadata import version as _pkg_version

    __version__ = _pkg_version("mosaiqtl")
except Exception:  # package not installed (e.g. running from a raw checkout)
    __version__ = "0.0.0+unknown"

# population configuration — first-class public type
from .models.config import Population, ScanConfig, PipelineConfig  # noqa: F401
from .genotype.cc import CCPopulation  # noqa: F401

# genome cache and kinship
from .genotype.genomecache import GenomeCache  # noqa: F401
from .genotype.kinship import calc_kinship, decomp_kinship  # noqa: F401

# result types
from .models.records import (  # noqa: F401
    QTLResult,
    VQTLResult,
    FounderEffect,
    ProvenanceRecord,
)

# primary model — NB GLM (NEBULA-LN cell level + pseudobulk scan)
from .qtl.nb import (  # noqa: F401
    scan_nb,
    scan_nb_gene,
    fit_null_sc,
    fit_null_nb_glmm_loco,
    pseudobulk_sum,
)

# gaussian lmm scan
from .qtl.scan import scan_h2lmm, scan_multi_gene, scan_multi_gene_lmm, ScanResult  # noqa: F401
from .qtl.mi import scan_h2lmm_mi, scan_multi_gene_mi  # noqa: F401
from .qtl.instability import (  # noqa: F401
    fixef_eqtl_scan,
    instability_lm_scan,
    instability_scan_multi_gene,
)
from .qtl.peaks import grab_locus_from_scan, lod_drop_ci, find_peaks  # noqa: F401

# permutation engine
from .qtl.permutation import (  # noqa: F401
    KinshipSingularError,
    generate_null_outcomes,
    run_threshold_scans,
    draw_parametric_null_batch,
)

# significance thresholds (GEV fitting + per-chromosome loading)
from .qtl.thresholds import get_gev_thresholds, load_per_chr_thresholds  # noqa: F401

# positional CI (LOD support interval)
from .qtl.ci import CI_COLS, lod_support_interval, ci_scan_multi_gene  # noqa: F401

# per-gene FDR calibration
from .qtl.fdr import (  # noqa: F401
    beta_calibrate_pvalue,
    storey_qvalue,
    calibrate_gene_pvalue,
)

from .genotype.snp_db import build_snp_catalogue  # noqa: F401
from .qtl.snp_scan import snp_scan_multi_gene  # noqa: F401

# mediation
from .qtl.mediation import (  # noqa: F401
    mediation_scan_qr,
    fixef_mediation_scan,
    select_cis_mediation_pairs,
    mediation_scan_multi_outcome,
)

# three-layer user API
from .pipeline import discover, characterize, resolve  # noqa: F401

# epistasis and multi-signal decomposition
from .qtl.epistasis import (  # noqa: F401
    scan_epistasis,
    test_haplotype_combinations,
    scan_conditional,
    scan_conditional_with_y,
)

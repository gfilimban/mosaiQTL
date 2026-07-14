"""
mosaiqtl — model-agnostic haplotype-based QTL mapping for multiparent populations

three-layer API:
  Layer 1 — discovery        scan_nb(), scan_h2lmm(), pipeline.discover()
  Layer 2 — characterization pipeline.characterize()  (allele effects, vqtl, mediation)
  Layer 3 — resolution       pipeline.resolve()       (snp catalogue, conditional, ase, sqtl)

supports nb glm (primary), gaussian lmm, and cell-level nb via edgepython.
parametric on J founders — no hardcoded assumptions about population size.
"""

__version__ = '0.5.2'

# population configuration — first-class public type
from .models.config import Population, ScanConfig, PipelineConfig    # noqa: F401
from .genotype.founders import CCPopulation                           # noqa: F401

# genome cache and kinship
from .genotype.genomecache import GenomeCache                         # noqa: F401
from .genotype.kinship import calc_kinship, decomp_kinship            # noqa: F401

# result types
from .models.records import (                                         # noqa: F401
    QTLResult, VQTLResult, FounderEffect, ProvenanceRecord,
)

# primary model — NB GLM (NEBULA-LN cell level + pseudobulk scan)
from .qtl.nb import (scan_nb, scan_nb_gene,                         # noqa: F401
                      fit_null_sc, fit_null_nb_glmm_loco,
                      pseudobulk_sum)

# gaussian lmm scan
from .qtl.scan import (scan_h2lmm, scan_h2lmm_mi, scan_multi_gene,  # noqa: F401
                        ScanResult, grab_locus_from_scan, lod_drop_ci,
                        find_peaks, fixef_eqtl_scan, instability_lm_scan,
                        scan_multi_gene_mi, instability_scan_multi_gene)

# permutation thresholds
from .qtl.permutation import (KinshipSingularError,                  # noqa: F401
                                generate_null_outcomes,
                                run_threshold_scans,
                                get_gev_thresholds,
                                run_positional_scans,
                                ci_scan_multi_gene)

# snp catalogue (Layer 3a)
from .qtl.snp_scan import (build_snp_catalogue,                      # noqa: F401
                             snp_scan_multi_gene)

# mediation
from .qtl.mediation import (mediation_scan_qr, fixef_mediation_scan,    # noqa: F401
                              select_cis_mediation_pairs,
                              mediation_scan_multi_outcome)

# three-layer user API
from .pipeline import discover, characterize, resolve                # noqa: F401

# epistasis and multi-signal decomposition
from .qtl.epistasis import (scan_epistasis,                          # noqa: F401
                              test_haplotype_combinations,
                              scan_conditional,
                              scan_conditional_with_y)

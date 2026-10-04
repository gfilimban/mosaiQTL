"""
validate_tissue.py — mosaiQTL tissue characterization + optional miQTL comparison

Primary purpose: characterize the Python mosaiqtl scan output on real pseudobulk
data for any tissue (S1–S11 each run and produce output independently of R).

R comparison is optional: when R reference CSVs are present in {data}/r_data/,
each section reports a concordance metric; when absent the section still runs
and reports Python-only statistics.

Prerequisites (required):
  1. python validation/convert_h5ad.py --tissue <tissue> --data validation/<tissue>_data \
       --h5ad /path/to/tissue.h5ad
     → writes <data>/metadata.csv, exp_gene_info_{tissue}.csv
  2. Build py_data/cache.h5 (one-time, requires rpy2 + miqtl R package):
       python validation/validate_tissue.py --tissue gastroc --build-cache
       python validation/validate_tissue.py --tissue gastroc --build-cache \
         --genomecache /path/to/genomecache
     Cache is shared across tissues (same CC mice, same mm39 genome).
     All tissues share validation/py_ref/ — no extra flags needed for subsequent tissues.

Prerequisites (optional, for R comparison gates):
  Rscript validation/r_reference/gastroc_miqtl_scan.R [sections] [chr] [scan_type]
  (currently gastroc-specific; R comparison sections are skipped when r_data/ absent)

Outputs, per cell type ({data}/celltypes/{ct}/) or whole-tissue ({data}/whole/):
  cache_info.csv, gate_summary.csv, haplotype_probs.csv, kinship_summary.csv
              scan-type-invariant (genome cache + genotype set only) — written once
              at this root, shared across every scan_tag below
  {scan_tag}/ everything scan_type/window_mb-dependent, where scan_tag is 'C', 'G',
              or 'Lw<window_mb>' (see run()::_scan_tag) — so different
              --scan_type/--window_mb runs never overwrite each other:
    core/, extended/    scan CSVs, checkpoint .done files, run_summary.csv,
                         eqtl_summary.csv, scan_info.json, scan_stats.csv
                         (the 4 core scan CSVs — lod/peak_lod/allele_effects/threshold —
                         carry a '_{celltype_or_whole}_{scan_tag}' filename suffix, e.g.
                         'peak_lod_astrocyte_Lw1.csv', 'lod_whole_C.csv' — see run()::_sfx)
    plots/               descriptive + optional R-comparison figures; R-comparison
                         scatter PNGs carry the same suffix, e.g. 'mi_scatter_whole_G.png'
    compare/             py-vs-R merged CSVs (written only when R data present)

Usage (gastroc, existing default):
  python validation/validate_tissue.py --tissue gastroc --data validation/gastroc_data
  python validation/validate_tissue.py --tissue gastroc --full
  python validation/validate_tissue.py --tissue gastroc --scan
  python validation/validate_tissue.py --tissue gastroc --scan_type G
  python validation/validate_tissue.py --tissue gastroc --scan_type L --window_mb 2.0 --chr 8

Usage (ctxhipp, new tissue — shares validation/py_ref/ genomecache):
  python validation/validate_tissue.py \
    --tissue ctxhipp \
    --data validation/ctxhipp_data \
    --celltype 'glutamatergic neuron' \
    --scan

scan_type:
  C  — chromosome: each gene scanned against every locus on its own chromosome
  G  — genome-wide: all genes against every chromosome
  L  — local: cis genes, loci within ±window_mb of gene TSS (default ±1.0 Mb) (default)
"""

from __future__ import annotations

import argparse
import gc
import logging
import re
import resource
import sys
import time
import tracemalloc
from pathlib import Path

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Carlito'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as sp_stats

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mosaiqtl.genotype.founders import (
    COLOR_CC_FALLBACK, COLOR_NEUTRAL_DARK, COLOR_NEUTRAL_LIGHT,
    F1_COLORS, FOUNDER_CODE, FOUNDER_COLORS, FOUNDER_ORDER, STRAIN_ORDER,
)
from mosaiqtl.utils import chr_sort_key as _chr_sort_key

_FOUNDER_CODE_INV: dict[str, str] = {v: k for k, v in FOUNDER_CODE.items()}


def _display_founders(cache_founders: 'list[str]') -> list[str]:
    """Map cache founder codes (A-H from HAPPY format) to FOUNDER_ORDER display names."""
    return [_FOUNDER_CODE_INV.get(f, _FOUNDER_CODE_INV.get(f.upper(), f))
            for f in cache_founders]


def _get_rss_mb() -> float:
    """Current process RSS in MB (macOS returns bytes; Linux returns kilobytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return usage.ru_maxrss / 1024**2
    return usage.ru_maxrss / 1024


from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.viz.genome import (
    loci_per_chr as _viz_loci_per_chr,
    lod_profile as _viz_lod_profile,
    manhattan as _viz_manhattan,
    perm_distribution as _viz_perm_distribution,
    egene_lod_distribution as _viz_egene_lod_distribution,
    mediation_profile as _viz_mediation_profile,
    ci_strip as _viz_ci_strip,
    lod_profiles_per_chr as _viz_lod_profiles_per_chr,
    lod_panel_grid as _viz_lod_panel_grid,
    perm_distributions_per_chr as _viz_perm_distributions_per_chr,
    perm_panel_grid as _viz_perm_panel_grid,
    snp_profiles_per_chr as _viz_snp_profiles_per_chr,
    genome_summary as _viz_genome_summary,
    genome_diagnostic as _viz_genome_diagnostic,
    snp_lod_profile as _viz_snp_lod_profile,
    instability_lod_overlay as _viz_instability_overlay,
    mi_scan_overlay as _viz_mi_scan_overlay,
    perm_threshold_comparison as _viz_perm_threshold_comparison,
    ci_genomic_interval as _viz_ci_genomic_interval,
    scan_throughput as _viz_scan_throughput,
    mediation_summary_bar as _viz_mediation_summary_bar,
    timing_memory_summary as _viz_timing_memory_summary,
    cis_distance_histogram as _viz_cis_distance_histogram,
    cis_distance_panel as _viz_cis_distance_panel,
    _MI_COLOR,
    _INST_COLOR,
)
from mosaiqtl.viz.locus import (
    locus_panels_per_chr as _viz_locus_panels_per_chr,
    locus_panel_grid as _viz_locus_panel_grid,
)
from mosaiqtl.viz.haplotypes import (
    kinship_distribution as _viz_kinship_distribution,
    kinship_heatmap as _viz_kinship_heatmap,
    prob_heatmap as _viz_prob_heatmap,
)
from mosaiqtl.viz.effects import (
    effect_bars as _viz_effect_bars,
    genome_effects_grid as _viz_genome_effects_grid,
)
from mosaiqtl.viz.compare import (
    lod_scatter as _cmp_scatter,
    lod_density_comparison as _cmp_lod_density,
    allele_effects_comparison as _cmp_allele_effects,
    lod_profiles_comparison as _cmp_lod_profiles,
    perm_lod_comparison as _cmp_perm_lod,
)
from mosaiqtl.viz.celltypes import (
    safe_ct as _ct_safe,
    resolve_ct_scan_dir as _ct_resolve_dir,
    discover_cell_types as _ct_discover,
    build_ct_palette as _ct_palette,
    load_threshold as _ct_load_threshold,
    ct_vs_whole_lod_scatter as _ct_vs_whole_lod,
    ct_vs_whole_effects_scatter as _ct_vs_whole_effects,
    ct_lod_profile_overlay as _ct_lod_overlay,
    ct_pairwise_lod_scatter as _ct_pairwise_lod,
    ct_pairwise_effects_scatter as _ct_pairwise_effects,
    ct_lod_distributions as _ct_lod_distributions,
    ct_cis_distance_comparison as _ct_cis_distance,
    compare_scan_types as _compare_scan_types,
)
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.scan import scan_h2lmm, ScanResult
from mosaiqtl.qtl.peaks import grab_locus_from_scan
from mosaiqtl.qtl.mi import scan_h2lmm_mi, scan_multi_gene_mi
from mosaiqtl.qtl.instability import instability_lm_scan, instability_scan_multi_gene
from mosaiqtl.qtl.lmm import estimate_h2_null, precompute_chromosome, scan_chromosome
from mosaiqtl.qtl.ci import ci_scan_multi_gene
from mosaiqtl.qtl.fdr import calibrate_gene_pvalue
from mosaiqtl.qtl.permutation import generate_null_outcomes, run_threshold_scans
from mosaiqtl.qtl.thresholds import get_gev_thresholds, load_per_chr_thresholds
from mosaiqtl.qtl.mediation import (
    mediation_scan_qr,
    mediation_scan_qr,
    select_cis_mediation_pairs,
    mediation_scan_multi_outcome,
)
from mosaiqtl.io.covariates import encode_sex
from mosaiqtl.io.results_io import pos_merge as _pos_merge, scan_result_to_df as _scan_to_df
from mosaiqtl.io.anndata_io import load_expression_h5 as _load_expression_h5
from mosaiqtl.qtl.scan import scan_multi_gene_lmm as _scan_multi_gene_lmm
from mosaiqtl.pipeline import (
    calibrate_egenes_storey,
    call_egenes, finalize_egene_summary, finemap_egenes,
    LOD_COLS, ALLELE_EFFECTS_BASE_COLS, PEAK_LOD_COLS, EGENE_SUMMARY_COLS,
)
from mosaiqtl.utils import build_z_matrix, rint
from natsort import natsorted

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_tissue')


def _git_hash() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        return 'unknown'

PASS = '✅'
FAIL = '❌'
SKIP = '⚠️ '

# ── paths ─────────────────────────────────────────────────────────────────────

_MIQTL_LOCAL = (Path.home() / 'Documents/journey/learning/graduate'
                / 'lab/projects/igvf/analysis/qtl/miqtl')
_MIQTL_HPC   = Path('/share/crsp/lab/seyedam/gfilimba/projects'
                    '/igvf_8cube/short-read/data/gastroc')
_MIQTL_DIR   = _MIQTL_LOCAL if _MIQTL_LOCAL.exists() else _MIQTL_HPC
GENOMECACHE  = _MIQTL_DIR / 'output' / 'genomecache'   # default for --build-cache

# shared genomecache + kinship — tissue-agnostic (same CC mice, same mm39 genome).
# built by: conda run -n bioinfo python validation/tools/build_cache.py
_DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PY_REF = _DATA_DIR if _DATA_DIR.is_dir() else Path('validation/py_ref')


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────



def _check(label: str, value: float, threshold: float, op: str = '>=') -> bool:
    ok = (value >= threshold) if op == '>=' else (value < threshold)
    sym = PASS if ok else FAIL
    log.info('  %s %s = %.4f  (threshold %s %.4f)', sym, label, value, op, threshold)
    return ok


def _checkpoint(py_dir: Path, section: str, force: bool) -> bool:
    done = py_dir / f's{section}.done'
    if done.exists() and not force:
        log.info('  (checkpoint — skipping section %s; use --force to rerun)', section)
        return True
    return False


def _mark_done(py_dir: Path, section: str) -> None:
    (py_dir / f's{section}.done').write_text('done')



# ─────────────────────────────────────────────────────────────────────────────
# one-time cache builder — invoked only via --build-cache flag, never during run
# ─────────────────────────────────────────────────────────────────────────────

def _build_cache_from_happy(genomecache_dir: Path, out_h5: Path) -> None:
    """Read HAPPY genomecache via rpy2/miqtl DiploprobReader → save as HDF5.

    One-time step.  Run with: python validate_tissue.py --tissue gastroc --build-cache
    Subsequent runs load from HDF5 without touching R.
    """
    try:
        import rpy2.robjects as ro
    except ImportError:
        raise ImportError(
            'rpy2 required for --build-cache.  Install with: pip install rpy2'
        )

    gc_path = str(genomecache_dir)
    log.info('opening DiploprobReader: %s', gc_path)
    ro.r('library(miqtl); library(abind)')
    ro.r(f'rd <- DiploprobReader$new("{gc_path}")')

    subjects_r = list(ro.r('rd$getSubjects()'))
    founders_r = list(ro.r('rd$getFounders()'))
    chrs       = list(ro.r('rd$getChromList()'))
    log.info('  %d subjects  %d founders  %d chromosomes',
             len(subjects_r), len(founders_r), len(chrs))

    data: dict = {}
    for chrom in chrs:
        t0 = time.perf_counter()
        ro.r(f'loci_c <- rd$getLoci("{chrom}")')
        ro.r(f'pos_c  <- rd$getLocusStart(loci_c, scale="Mb")')
        ro.r('''
        plist_c <- lapply(loci_c, function(loc)
            rd$getLocusMatrix(loc, model="additive"))
        parr_c  <- abind(plist_c, along=3)
        ''')
        lids    = np.array(list(ro.r('loci_c')))
        pos_mb  = np.array(list(ro.r('pos_c')), dtype=np.float64)
        probs_r = np.array(ro.r('parr_c'))          # (n_subjects, n_founders, n_loci)
        probs   = np.transpose(probs_r, (2, 0, 1)) / 2.0   # → (n_loci, n_subjects, 8)
        data[chrom] = {
            'locus_ids': lids,
            'positions': pos_mb,
            'probs':     probs.astype(np.float64),
        }
        log.info('  chr%s: %d loci  %.1fs', chrom, len(lids),
                 time.perf_counter() - t0)

    cache = GenomeCache(
        sample_ids=np.array(subjects_r),
        founders=founders_r,
        data=data,
    )
    cache = cache.reorder_to_canonical()
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    cache.to_hdf5(str(out_h5))
    log.info('saved cache.h5 → %s  (%d strains, canonical order)', out_h5,
             len(cache.sample_ids))


# ─────────────────────────────────────────────────────────────────────────────
# expression + metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_expression(h5_path: Path, genes_wanted: list[str],
                     ) -> tuple[np.ndarray, list[str], list[str]]:
    """load only the requested genes from counts_lognorm.h5.

    No cross-loading: only the cis-gene rows for the chromosome being scanned
    are materialised from the HDF5 file.

    Returns
    -------
    counts : (n_genes, n_samples) float32
    genes  : gene IDs in order (subset of genes_wanted that were found)
    samples: sample IDs in order
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        all_genes   = np.array(f['genes']).astype(str).tolist()
        all_samples = np.array(f['samples']).astype(str).tolist()
        counts_ds   = f['counts']   # (n_genes_total, n_samples)
        wanted_set  = set(genes_wanted)
        idx = [i for i, g in enumerate(all_genes) if g in wanted_set]
        if not idx:
            return np.empty((0, len(all_samples))), [], all_samples
        counts = np.array(counts_ds[idx, :], dtype=np.float32)
    genes_out = [all_genes[i] for i in idx]
    return counts, genes_out, all_samples


def _build_covariates(meta: pd.DataFrame,
                      sample_order: list[str]) -> np.ndarray:
    """build covariate matrix aligned to sample_order — matches R formula '~ sex + dataset'.

    Columns (scan_h2lmm prepends intercept internally):
      sex     : 0=Female (reference), 1=Male  — mandatory; X-linked dosage + hormone effects
      dataset : n_cats-1 dummies; the 'founder*' category (case-insensitive) is the reference
                level (biological baseline); remaining categories sorted lexicographically.
    """
    meta_idx = meta.set_index('sample_id')
    rows     = meta_idx.loc[sample_order]

    sex_vec  = encode_sex(rows['sex']).values.astype(np.float64)

    # auto-detect reference level: first category whose name starts with 'founder'
    # (handles both 'Founder' in gastroc and 'founder' in ctxhipp)
    ds_vals     = rows['dataset'].values
    unique_cats = sorted(set(ds_vals), key=str.lower)
    ref         = next(
        (c for c in unique_cats if c.lower().startswith('founder')),
        unique_cats[0],
    )
    ordered = [ref] + [c for c in unique_cats if c != ref]
    cat     = pd.Categorical(ds_vals, categories=ordered)
    dummies = pd.get_dummies(cat, drop_first=True).values.astype(np.float64)

    return np.column_stack([sex_vec, dummies])


# ─────────────────────────────────────────────────────────────────────────────
# descriptive plots — qtl2charts-informed, no R data required
# ─────────────────────────────────────────────────────────────────────────────

# readable labels for the timing / memory summary plot
_SECTION_LABELS: dict[str, str] = {
    'cache':           'Cache structure',
    'haplotype_probs': 'Haplotype probabilities',
    'loco_kinship':    'LOCO kinship',
    'lmm_scan':        'LMM scan',
    'permutation':     'Permutation threshold',
    'egene_calling':   'eGene calling',
    'bootstrap_ci':    'Bootstrap CI',
    'sanger_finemap':  'Sanger fine-mapping',
    'mi_scan':         'Multiple imputation',
    'mediation':       'Mediation scan',
    'instability':     'Instability scan',
}
_SECTION_ORDER: list[str] = list(_SECTION_LABELS)


def _pick_top_gene(df_chr: 'pd.DataFrame', peak_chr: str,
                   lod_col: str = 'lod') -> str:
    """Return top gene by lod_col max."""
    by_gene = df_chr.groupby('gene_symbol')[lod_col].max()
    return by_gene.idxmax()


def _make_descriptive_plots(
    plt_dir: Path,
    py_dir:  Path,
    r_data:  Path,
    cache,
    kinship_decomp: dict,
    founders:       list[str],
    peak_chr:       str,
    scan_tag:       str,
    celltype:       str | None = None,
    priority_gene:  dict[str, str] | None = None,
    low_expr_genes: set[str] | None = None,
    gene_info:      'pd.DataFrame | None' = None,
    gtf_path:       'str | None' = None,
) -> None:
    """qtl2charts-inspired descriptive plots from Python outputs only.

    Figures produced (all in plt_dir/):
      loci_per_chr.png              — loci per chromosome bar
      haplotype_heatmap.png         — founder prob heatmap at sampled loci
      kinship_heatmap.png           — LOCO kinship matrix heatmap
      kinship_distribution.png      — kinship diagonal vs off-diagonal KDE
      lod_profile/                  — LOD profile for top gene (per chr)
      founder_effects/              — founder coef bar at peak locus (canonical colors)
      manhattan.png                 — peak LOD per gene across chr
      cis_distance.png              — cis-distance scatter + KDE (all scanned vs eGenes)
      cis_histogram.png             — histogram of peak − TSS distance for eGenes
      perm_distribution.png         — max-LOD histogram + GEV fit + threshold
      egene_lod_distribution.png    — peak LOD histogram; eGenes vs threshold
      extended/mi/                  — MI scan LOD profile for top gene
      extended/snp/                 — SNP scan LOD scatter for top gene
      mediation_by_chr.png          — mediation LOD drop summary by chromosome
      ci_strip.png                  — bootstrap CI strip per eGene
      extended/instability/         — instability LOD profile for top gene
      timing_memory_summary.png     — elapsed time per section bar chart
    """
    sns.set_style('ticks')
    f_colors = {f: FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in founders}
    core_dir = py_dir / 'core'
    # filename suffix (celltype_or_whole + scan_tag) matching what run() wrote
    # every core scan CSV out as — see _sfx()
    _fsfx = _sfx(celltype, scan_tag)
    # bind parameters to local names so the rest of the function body is unchanged
    PRIORITY_GENE = priority_gene or {}  # type: dict[str, str]
    low_expr_genes = low_expr_genes or set()  # type: set[str]

    # ── loci per chromosome bar ───────────────────────────────────────────
    cache_info_path = py_dir / 'cache_info.csv'
    if cache_info_path.exists():
        try:
            ci = pd.read_csv(cache_info_path)
            ci['chr'] = ci['chr'].astype(str)

            def _chr_key(c):
                try:
                    return (0, int(c))
                except ValueError:
                    return (1, c)
            ci = ci.sort_values('chr', key=lambda s: [_chr_key(x) for x in s])
            fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
            _viz_loci_per_chr(ci['chr'].tolist(), ci['n_loci'].values, ax=ax)
            ax.set_title(f'Loci per chromosome  (total={ci["n_loci"].sum():,})',
                         fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 'core' / 'loci_per_chr.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  core/loci_per_chr.png')
        except Exception as exc:
            log.warning('  loci_per_chr: %s', exc)

    # ── haplotype prob heatmap ────────────────────────────────────────────
    hap_probs_path = py_dir / 'haplotype_probs.csv'
    if hap_probs_path.exists():
        try:
            hp       = pd.read_csv(hap_probs_path)
            locus0   = hp['locus'].iloc[0]
            sub      = hp[hp['locus'] == locus0].copy()
            strain_order = list(cache.sample_ids)
            sub      = sub.set_index('genotype').reindex(strain_order).reset_index()
            mat      = sub[founders].values  # (n_strains, J) — founders = cache codes
            n_s      = len(strain_order)
            fig_h    = max(3.5, n_s * 0.18)
            fig, ax  = plt.subplots(figsize=(6, fig_h), dpi=300)
            _viz_prob_heatmap(mat, _display_founders(founders), ax=ax,
                              chrom=peak_chr, position_mb=None)
            ax.set_title(f'Haplotype probs  locus={locus0}  (n={n_s})',
                         fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 'core' / 'haplotype_heatmap.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  core/haplotype_heatmap.png  (%d strains)', n_s)
        except Exception as exc:
            log.warning('  haplotype_heatmap: %s', exc)

    # ── kinship heatmap + distribution ───────────────────────────────────
    if peak_chr in kinship_decomp:
        try:
            kd = kinship_decomp[peak_chr]
            K  = kd['vectors'] @ np.diag(kd['values']) @ kd['vectors'].T
            n  = K.shape[0]

            strains_k = list(cache.sample_ids)
            fo_set_k  = set(FOUNDER_ORDER)
            f1_set_k  = set(F1_COLORS.keys())
            n_fo_k    = sum(1 for s in strains_k if s in fo_set_k)
            n_f1_k    = sum(1 for s in strains_k if s in f1_set_k)

            side  = max(5.0, n * 0.06)
            fig1, ax1 = plt.subplots(figsize=(side, side), dpi=300)
            _viz_kinship_heatmap(
                K, ax=ax1,
                sample_labels=strains_k,
                block_boundaries=[n_fo_k, n_fo_k + n_f1_k],
                block_labels=['Founders', 'F1s', 'CC'],
                chrom=peak_chr,
                cmap='Blues', vmin=0,
            )
            ax1.set_title(f'LOCO kinship  chr{peak_chr}  (n={n})', fontsize=9)
            fig1.savefig(plt_dir / 'core' / 'kinship_heatmap.png', dpi=300,
                         bbox_inches='tight', transparent=True)
            plt.close(fig1)
            log.info('  core/kinship_heatmap.png  (%d strains)', n)

            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
            _viz_kinship_distribution(K, ax=ax2, chrom=peak_chr)
            ax2.set_title(f'Kinship distribution  chr{peak_chr}', fontsize=9)
            fig2.tight_layout()
            fig2.savefig(plt_dir / 'core' / 'kinship_distribution.png', dpi=300, transparent=True)
            plt.close(fig2)
            log.info('  core/kinship_distribution.png')
        except Exception as exc:
            log.warning('  kinship_heatmap: %s', exc)

    ext_dir = py_dir / 'extended'

    # ── read lod_df once (reused for profile, Manhattan) ─────────────────
    # lod.csv is the full genome-wide (genes × loci) LOD table — at gastroc
    # scale this is multi-GB / 100M+ rows. Loading it whole here would
    # re-materialise the matrix MEM-05 forbids and was the source of a severe
    # plotting-stage stall. Every consumer below only needs the LOD profile
    # for a handful of named genes (top gene per chromosome + priority genes),
    # so we determine that gene set first and stream-filter the CSV in
    # chunks, keeping only matching rows.
    lod_path = core_dir / f'lod_{_fsfx}.csv'
    pk_path  = core_dir / f'peak_lod_{_fsfx}.csv'
    eff_path = core_dir / f'allele_effects_{_fsfx}.csv'
    _lod_df: 'pd.DataFrame | None' = None
    if lod_path.exists() and pk_path.exists():
        try:
            _pk_for_filter = pd.read_csv(pk_path)
            _pk_for_filter['chr'] = _pk_for_filter['chr'].astype(str)
            _needed_genes: set[str] = set(
                _pk_for_filter.loc[_pk_for_filter.groupby('chr')['peak_lod'].idxmax(),
                                   'gene_symbol'].astype(str))
            _needed_genes.update(PRIORITY_GENE.values())
            if eff_path.exists():
                _eff_for_filter = pd.read_csv(eff_path, usecols=['chr', 'gene_symbol', 'peak_lod'])
                _eff_for_filter['chr'] = _eff_for_filter['chr'].astype(str)
                _needed_genes.update(
                    _eff_for_filter.loc[_eff_for_filter.groupby('chr')['peak_lod'].idxmax(),
                                        'gene_symbol'].astype(str))

            _chunks: 'list[pd.DataFrame]' = []
            for _chunk in pd.read_csv(lod_path, low_memory=False, chunksize=2_000_000):
                _hit = _chunk[_chunk['gene_symbol'].isin(_needed_genes)]
                if len(_hit) > 0:
                    _chunks.append(_hit.copy())
            _lod_df = (pd.concat(_chunks, ignore_index=True)
                       if _chunks else pd.DataFrame(columns=['gene_symbol', 'chr', 'locus', 'pos_mb', 'lod']))
            _lod_df['chr'] = _lod_df['chr'].astype(str)
            del _chunks
            gc.collect()
            log.info('  lod.csv filtered to %d genes  (%d rows)',
                     len(_needed_genes), len(_lod_df))
        except Exception as exc:
            log.warning('  lod.csv read failed: %s', exc)

    # ── threshold lookup (genome-wide scalar from threshold.csv) ──────────
    # Primary: core/threshold.csv (written by this script after permutation).
    # Fallback: r_data/thresholds_chr.csv (R reference, available earlier — passed in
    # directly since it lives outside py_dir's own tree, see run()'s r_data local).
    _thr_by_chr: 'dict[str, float | None]' = {}
    _gw_thr_scalar: float | None = None
    _thr_path = core_dir / f'threshold_{_fsfx}.csv'
    _thr_fallback = r_data / 'thresholds_chr.csv'
    _thr_source = _thr_path if _thr_path.exists() else (
        _thr_fallback if _thr_fallback.exists() else None
    )
    if _thr_source is not None:
        try:
            _td0 = pd.read_csv(_thr_source)
            _td0['chr'] = _td0['chr'].astype(str)
            if 'genome_wide' in _td0['chr'].values:
                _gw_thr_scalar = float(
                    _td0.loc[_td0['chr'] == 'genome_wide', 'threshold'].iloc[0])
                _thr_by_chr['genome_wide'] = _gw_thr_scalar
                _per_chr_rows = _td0[_td0['chr'] != 'genome_wide']
                if len(_per_chr_rows) > 0:
                    # per-chr GEV rows present — use them directly
                    for _, _r in _per_chr_rows.iterrows():
                        _thr_by_chr[str(_r['chr'])] = (
                            float(_r['threshold']) if pd.notna(_r['threshold'])
                            else _gw_thr_scalar)
                else:
                    # no per-chr rows yet — expand genome_wide to all standard chrs
                    for _tc in [str(i) for i in range(1, 20)] + ['X']:
                        _thr_by_chr[_tc] = _gw_thr_scalar
            else:
                for _tc, _tg in _td0.groupby('chr'):
                    _thr_by_chr[str(_tc)] = float(_tg['threshold'].iloc[0])
            log.info('  threshold loaded from %s (%d chrs)',
                     _thr_source.name, len(_thr_by_chr))
        except Exception:
            pass

    # override genome-wide-expanded values with true per-chr GEV thresholds
    # computed from perm_max_lods_chr*.csv (works with checkpointed data)
    _per_chr_thr = load_per_chr_thresholds(core_dir, percentile=0.95)
    if _per_chr_thr:
        _thr_by_chr.update(_per_chr_thr)
        log.info('  per-chr GEV thresholds loaded (%d chrs)', len(_per_chr_thr))

    # ── LOD profile — top gene per chromosome ────────────────────────────
    _lod_dir = plt_dir / 'core' / 'lod_profile'
    _lod_dir.mkdir(parents=True, exist_ok=True)
    if _lod_df is not None and pk_path.exists():
        try:
            pk_df = pd.read_csv(pk_path)
            pk_df['chr'] = pk_df['chr'].astype(str)
            for c in sorted(_lod_df['chr'].unique(), key=_chr_sort_key):
                pk_c = pk_df[pk_df['chr'] == c]
                if pk_c.empty:
                    continue
                top_gene = str(pk_c.loc[pk_c['peak_lod'].idxmax(), 'gene_symbol'])
                lod_top  = _lod_df[
                    (_lod_df['chr'] == c) & (_lod_df['gene_symbol'] == top_gene)
                ].sort_values('pos_mb')
                if lod_top.empty:
                    continue
                # genome-wide threshold for eGene calling (not per-chr GEV)
                thr = _gw_thr_scalar
                fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_lod_profile(lod_top['pos_mb'].values, lod_top['lod'].values,
                                 ax=ax, chrom=c, threshold=thr, fill=True,
                                 gene=top_gene)
                fig.tight_layout()
                fig.savefig(_lod_dir / f'lod_profile_chr{c}.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  lod_profile/lod_profile_chr%s.png  (gene=%s)', c, top_gene)
                _p1 = PRIORITY_GENE.get(c)
                if _p1 and _p1 != top_gene:
                    lod_p1 = _lod_df[
                        (_lod_df['chr'] == c) & (_lod_df['gene_symbol'] == _p1)
                    ].sort_values('pos_mb')
                    if not lod_p1.empty:
                        fig2, ax2 = plt.subplots(figsize=(12, 3.5), dpi=300)
                        _viz_lod_profile(lod_p1['pos_mb'].values, lod_p1['lod'].values,
                                         ax=ax2, chrom=c, threshold=thr, fill=True,
                                         gene=_p1)
                        fig2.tight_layout()
                        fig2.savefig(_lod_dir / f'lod_profile_{_p1}.png', dpi=300, transparent=True)
                        plt.close(fig2)
                        log.info('  lod_profile/lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  lod_profile: %s', exc)

    # ── founder coef bar — top gene per chromosome ───────────────────────
    _eff_dir = plt_dir / 'core' / 'founder_effects'
    _eff_dir.mkdir(parents=True, exist_ok=True)
    if eff_path.exists():
        try:
            eff_df    = pd.read_csv(eff_path)
            eff_df['chr'] = eff_df['chr'].astype(str)
            beta_cols = [bc for bc in eff_df.columns if bc.startswith('beta_')]
            for c in sorted(eff_df['chr'].unique(), key=_chr_sort_key):
                eff_c = eff_df[eff_df['chr'] == c]
                if eff_c.empty:
                    continue
                top_row = eff_c.loc[eff_c['peak_lod'].idxmax()]
                betas   = np.array([top_row[bc] for bc in beta_cols])
                fnames  = _display_founders([bc.replace('beta_', '') for bc in beta_cols])
                fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
                _viz_effect_bars(betas, fnames, ax=ax,
                                 gene=str(top_row['gene_symbol']),
                                 locus=str(top_row['peak_locus']))
                ax.text(0.98, 0.97, f'LOD = {top_row["peak_lod"]:.2f}',
                        transform=ax.transAxes, fontsize=10, va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=COLOR_NEUTRAL_DARK, alpha=0.85))
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(_eff_dir / f'founder_effects_chr{c}.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  founder_effects/founder_effects_chr%s.png  (gene=%s)', c, top_row['gene_symbol'])
                _p1 = PRIORITY_GENE.get(c)
                if _p1 and top_row['gene_symbol'] != _p1 and _p1 in eff_c['gene_symbol'].values:
                    p1_row   = eff_c[eff_c['gene_symbol'] == _p1].iloc[0]
                    betas_p1 = np.array([p1_row[bc] for bc in beta_cols])
                    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=300)
                    _viz_effect_bars(betas_p1, fnames, ax=ax2,
                                     gene=str(p1_row['gene_symbol']),
                                     locus=str(p1_row['peak_locus']))
                    ax2.text(0.98, 0.97, f'LOD = {p1_row["peak_lod"]:.2f}',
                             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor=COLOR_NEUTRAL_DARK, alpha=0.85))
                    sns.despine(ax=ax2)
                    fig2.tight_layout()
                    fig2.savefig(_eff_dir / f'founder_effects_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig2)
                    log.info('  founder_effects/founder_effects_%s.png', _p1)
        except Exception as exc:
            log.warning('  founder_effects: %s', exc)

    # ── founder effects grid + locus panel grids ─────────────────────────
    _lp_dir = plt_dir / 'core' / 'locus_panel'
    _lp_dir.mkdir(parents=True, exist_ok=True)
    if eff_path.exists() and _lod_df is not None:
        try:
            _eff     = pd.read_csv(eff_path)
            _eff['chr'] = _eff['chr'].astype(str)
            _beta_cols  = [bc for bc in _eff.columns if bc.startswith('beta_')]
            _fnames     = _display_founders([bc.replace('beta_', '') for bc in _beta_cols])
            fig_eg = _viz_genome_effects_grid(_eff, _fnames)
            fig_eg.savefig(plt_dir / 'core' / 'founder_effects_grid.png', dpi=300,
                           bbox_inches='tight', transparent=True)
            plt.close(fig_eg)
            log.info('  core/founder_effects_grid.png')
            fig_lpg = _viz_lod_panel_grid(
                _lod_df, _eff,
                thr_by_chr=_thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
            )
            fig_lpg.savefig(plt_dir / 'core' / 'lod_panel_grid.png', dpi=300,
                            bbox_inches='tight', transparent=True)
            plt.close(fig_lpg)
            log.info('  core/lod_panel_grid.png')
            fig_lcg = _viz_locus_panel_grid(
                _lod_df, _eff, _fnames,
                _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
            )
            fig_lcg.savefig(plt_dir / 'core' / 'locus_panel_grid.png', dpi=300,
                            bbox_inches='tight', transparent=True)
            plt.close(fig_lcg)
            log.info('  core/locus_panel_grid.png')
            _start_mb_map = None
            _start_col = 'near_tss_mb' if 'near_tss_mb' in _eff.columns else \
                ('start_mb' if 'start_mb' in _eff.columns else None)
            if _start_col is not None:
                _start_mb_map = {
                    str(g): float(t)
                    for g, t in zip(_eff['gene_symbol'], _eff[_start_col])
                    if pd.notna(t)
                }
            # build gene-body table for the gene track (start_mb / end_mb in Mb)
            _genes_df: 'pd.DataFrame | None' = None
            if gene_info is not None and 'start' in gene_info.columns:
                _gi = gene_info[['gene_symbol', 'chr', 'start', 'end', 'strand']].copy() \
                    if 'strand' in gene_info.columns \
                    else gene_info[['gene_symbol', 'chr', 'start', 'end']].copy()
                _gi['start_mb'] = _gi['start'] / 1e6
                _gi['end_mb']   = _gi['end']   / 1e6
                _genes_df = _gi.drop(columns=['start', 'end'])

            # near_tss_mb_map: TSS of the nearest gene to the QTL peak
            _near_tss_map: 'dict | None' = None
            if 'near_tss_mb' in _eff.columns:
                _near_tss_map = {
                    str(g): float(t)
                    for g, t in zip(_eff['gene_symbol'], _eff['near_tss_mb'])
                    if pd.notna(t)
                }

            # zoom panel margin: for a local scan (scan_tag == 'Lw<window_mb>'),
            # the scanned LOD data itself only spans ±window_mb around the
            # gene, so the zoom panel must use that same window — the 5.0 Mb
            # package default leaves most of the panel blank. C/G scans keep
            # the default, since their data covers the whole chromosome.
            _lw_match = re.match(r'^Lw([0-9.]+)$', scan_tag)
            _zoom_window_mb = float(_lw_match.group(1)) if _lw_match else 5.0
            _lp_figs = _viz_locus_panels_per_chr(
                _lod_df, _eff, _fnames,
                _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
                tss_mb_map=_start_mb_map,
                near_tss_mb_map=_near_tss_map,
                genes_df=_genes_df,
                zoom_window_mb=_zoom_window_mb,
                gtf_path=gtf_path,
            )
            for _lp_c, _lp_fig in _lp_figs.items():
                _lp_fig.savefig(_lp_dir / f'locus_panel_chr{_lp_c}.png', dpi=300, transparent=True)
                plt.close(_lp_fig)
            if _lp_figs:
                log.info('  locus_panel/locus_panel_chr*.png  (%d files)', len(_lp_figs))
        except Exception as exc:
            log.warning('  locus_panel_grid: %s', exc)

    # ── Manhattan — peak LOD per gene across all scanned chr ──────────────
    # peak_lod.csv already holds the genome-wide peak-per-gene reduction
    # computed during the streaming S4 scan — read it directly rather than
    # rederiving it from _lod_df, which (above) is now filtered to only the
    # per-chromosome top genes and would silently drop every other gene.
    peak_lod_path = core_dir / f'peak_lod_{_fsfx}.csv'
    if peak_lod_path.exists():
        try:
            _pk2     = pd.read_csv(peak_lod_path)
            _pk2['chr'] = _pk2['chr'].astype(str)
            peak_pos = _pk2[['gene_symbol', 'chr', 'peak_mb', 'peak_lod']].rename(
                columns={'peak_mb': 'pos_mb', 'peak_lod': 'lod'})
            chrs_sorted = sorted(peak_pos['chr'].unique(), key=_chr_sort_key)
            scans_dict  = {
                ch: (peak_pos[peak_pos['chr'] == ch]['pos_mb'].values,
                     peak_pos[peak_pos['chr'] == ch]['lod'].values)
                for ch in chrs_sorted
            }
            _lo_mask_dict = (
                {ch: peak_pos[peak_pos['chr'] == ch]['gene_symbol'].isin(low_expr_genes).values
                 for ch in chrs_sorted}
                if low_expr_genes else None
            )
            thr = _thr_by_chr.get(str(peak_chr))
            fig = _viz_manhattan(scans_dict, threshold=thr,
                                 low_expr_mask=_lo_mask_dict)
            _ax_m = fig.axes[0]
            _ax_m.text(0.01, 0.97, f'n = {len(peak_pos):,} genes',
                       transform=_ax_m.transAxes, fontsize=10, va='top', ha='left',
                       fontweight='bold', color=COLOR_NEUTRAL_DARK)
            fig.tight_layout()
            fig.savefig(plt_dir / 'core' / 'manhattan.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  core/manhattan.png')
        except Exception as exc:
            log.warning('  manhattan: %s', exc)

    # ── cis-distance scatter (full + ±1 Mb zoom) + KDE ───────────────────
    if peak_lod_path.exists():
        try:
            _cd = pd.read_csv(peak_lod_path)
            _cd_tss_col = 'near_tss_mb' if 'near_tss_mb' in _cd.columns else 'start_mb'
            if {_cd_tss_col, 'peak_mb', 'peak_lod'}.issubset(_cd.columns):
                _cd = _cd.dropna(subset=[_cd_tss_col, 'peak_mb', 'peak_lod'])
                _sum_cd = py_dir / 'eqtl_summary.csv'
                _eg_cd  = (set(pd.read_csv(_sum_cd)['gene_symbol'])
                           if _sum_cd.exists() else set())
                _is_eg_cd = _cd['gene_symbol'].isin(_eg_cd).values
                _is_lo_cd = _cd['gene_symbol'].isin(low_expr_genes).values
                fig_cd = _viz_cis_distance_panel(
                    _cd['peak_lod'].values,
                    _cd[_cd_tss_col].values,
                    _cd['peak_mb'].values,
                    is_egene=_is_eg_cd,
                    is_low_expr=_is_lo_cd if low_expr_genes else None,
                    zoom_mb=1.0,
                    threshold=_gw_thr_scalar,
                )
                fig_cd.savefig(plt_dir / 'core' / 'cis_distance.png', dpi=300,
                               transparent=True, bbox_inches='tight')
                plt.close(fig_cd)
                log.info('  core/cis_distance.png  (%d genes, %d eGenes) [tss_col=%s]',
                         len(_cd), int(_is_eg_cd.sum()), _cd_tss_col)
            else:
                log.warning('  cis_distance: %s/peak_mb missing — '
                            'rerun scan to populate', _cd_tss_col)
        except Exception as exc:
            log.warning('  cis_distance: %s', exc)

    # ── cis-distance histogram (eGenes only) ──────────────────────────────
    if peak_lod_path.exists():
        try:
            _ch = pd.read_csv(peak_lod_path)
            _ch_tss_col = 'near_tss_mb' if 'near_tss_mb' in _ch.columns else 'start_mb'
            if {_ch_tss_col, 'peak_mb'}.issubset(_ch.columns):
                _ch = _ch.dropna(subset=[_ch_tss_col, 'peak_mb'])
                _sum_ch = py_dir / 'eqtl_summary.csv'
                _eg_ch  = (set(pd.read_csv(_sum_ch)['gene_symbol'])
                           if _sum_ch.exists() else set())
                _ch_eg  = _ch[_ch['gene_symbol'].isin(_eg_ch)]
                if len(_ch_eg) > 0:
                    fig_ch, ax_ch = plt.subplots(figsize=(6, 3.8), dpi=300,
                                                 facecolor='white')
                    _viz_cis_distance_histogram(
                        _ch_eg[_ch_tss_col].values,
                        _ch_eg['peak_mb'].values,
                        ax=ax_ch,
                    )
                    fig_ch.tight_layout()
                    fig_ch.savefig(plt_dir / 'core' / 'cis_histogram.png', dpi=300,
                                   transparent=True, bbox_inches='tight')
                    plt.close(fig_ch)
                    log.info('  core/cis_histogram.png  (%d eGenes)', len(_ch_eg))
                else:
                    log.warning('  cis_histogram: no eGenes in eqtl_summary.csv')
        except Exception as exc:
            log.warning('  cis_histogram: %s', exc)

    # ── permutation max-LOD distribution + GEV fit + threshold ───────────
    perm_max_lods_path = core_dir / 'perm_max_lods.csv'
    thr_path           = core_dir / f'threshold_{_fsfx}.csv'
    if perm_max_lods_path.exists():
        try:
            max_lods = pd.read_csv(perm_max_lods_path)['max_lod'].values
            # genome-wide threshold — same value used by pipeline.call_egenes,
            # not a per-chr or arbitrary-first-row value from threshold.csv
            thr = _thr_by_chr.get('genome_wide')
            fig, ax  = plt.subplots(figsize=(8, 4.5), dpi=300)
            _viz_perm_distribution(max_lods, ax=ax, threshold=thr, chrom=peak_chr)
            ax.set_xlabel('Genome-wide max LOD (per permutation)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Count', fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=11)
            # n_perms + threshold are already drawn by _viz_perm_distribution itself
            # (bottom-left info box + top-right legend label) — adding a second,
            # separately-anchored info box here duplicated them and overlapped the legend.
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 'core' / 'perm_distribution.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  core/perm_distribution.png')
        except Exception as exc:
            log.warning('  perm_distribution: %s', exc)

    # ── per-chromosome permutation distributions ──────────────────────────
    _perm_dir = plt_dir / 'core' / 'perm'
    _perm_dir.mkdir(parents=True, exist_ok=True)
    # _per_chr_thr computed above from perm_max_lods_chr*.csv via package function
    if _per_chr_thr:
        try:
            _perm_null: dict[str, np.ndarray] = {}
            for _c_perm, _pf_path in [
                (c, core_dir / f'perm_max_lods_chr{c}.csv') for c in _per_chr_thr
            ]:
                if _pf_path.exists():
                    _perm_null[_c_perm] = pd.read_csv(_pf_path)['max_lod'].values
            _perm_thr_dict = _per_chr_thr
            if _perm_null:
                _perm_figs = _viz_perm_distributions_per_chr(_perm_null, _perm_thr_dict)
                for _pc, _pf in _perm_figs.items():
                    _pf.savefig(_perm_dir / f'perm_chr{_pc}.png', dpi=300, transparent=True)
                    plt.close(_pf)
                log.info('  perm/perm_chr*.png  (%d files)', len(_perm_figs))
                fig_pg = _viz_perm_panel_grid(_perm_null, _perm_thr_dict)
                fig_pg.savefig(plt_dir / 'core' / 'perm_panel_grid.png', dpi=300,
                               bbox_inches='tight', transparent=True)
                plt.close(fig_pg)
                log.info('  core/perm_panel_grid.png')
        except Exception as exc:
            log.warning('  perm_per_chr: %s', exc)

    # ── eGene calling — LOD distribution + threshold ─────────────────────
    peak_lod_path = core_dir / f'peak_lod_{_fsfx}.csv'
    if peak_lod_path.exists():
        try:
            pk         = pd.read_csv(peak_lod_path)
            # genome-wide threshold — matches the value pipeline.call_egenes
            # actually gated eGenes against (log line "genome-wide threshold=...")
            thr = _thr_by_chr.get('genome_wide')
            _sum_path  = py_dir / 'eqtl_summary.csv'
            egenes_set = (set(pd.read_csv(_sum_path)['gene_symbol'])
                          if _sum_path.exists() else set())
            lods       = pk['peak_lod'].dropna().values
            is_egene   = pk['gene_symbol'].isin(egenes_set).values
            fig, ax    = plt.subplots(figsize=(9, 4.5), dpi=300)
            _viz_egene_lod_distribution(lods, is_egene, ax=ax, threshold=thr)
            fig.tight_layout()
            fig.savefig(plt_dir / 'core' / 'egene_lod_distribution.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  core/egene_lod_distribution.png  (%d eGenes / %d genes)',
                     is_egene.sum(), len(pk))
        except Exception as exc:
            log.warning('  egene_lod_distribution: %s', exc)

    # ── genome-wide summary + diagnostic plots ────────────────────────────
    _pk_gs_path = core_dir / f'peak_lod_{_fsfx}.csv'
    if _pk_gs_path.exists():
        try:
            _pk_gs = pd.read_csv(_pk_gs_path)
            _pk_gs['chr'] = _pk_gs['chr'].astype(str)
            _sum_path = py_dir / 'eqtl_summary.csv'
            _eg_set = (set(pd.read_csv(_sum_path)['gene_symbol'])
                        if _sum_path.exists() else set())
            _is_eg = _pk_gs['gene_symbol'].isin(_eg_set).values
            _td_gs = pd.read_csv(thr_path) if thr_path.exists() else None
            _thr_gs: dict = {}
            if _td_gs is not None:
                _td_gs['chr'] = _td_gs['chr'].astype(str)
                _thr_gs = dict(zip(_td_gs['chr'],
                                 _td_gs['threshold'].where(_td_gs['threshold'].notna(), None)))
            _n_perms = (int(_td_gs['n_perms'].iloc[0])
                        if _td_gs is not None and 'n_perms' in _td_gs.columns
                        and len(_td_gs) > 0 else None)
            _scan_stats_gs: dict = {
                'n_genes': len(_pk_gs),
                'n_egenes': int(_is_eg.sum()),
                'peak_chr': str(peak_chr),
                'n_perms': _n_perms,
                'top_gene': (str(_pk_gs.loc[_pk_gs['peak_lod'].idxmax(), 'gene_symbol'])
                             if len(_pk_gs) > 0 else ''),
            }
            # _thr_gs has genome_wide + per-chr keys from threshold.csv (post-S5)
            # _thr_by_chr is the same data but already loaded earlier; prefer it
            _thr_plot = _thr_by_chr if _thr_by_chr else _thr_gs
            fig_gs = _viz_genome_summary(_pk_gs, _is_eg, _thr_plot or None,
                                         scan_stats=_scan_stats_gs)
            fig_gs.savefig(plt_dir / 'core' / 'genome_summary.png', dpi=300,
                           bbox_inches='tight', transparent=True)
            plt.close(fig_gs)
            log.info('  core/genome_summary.png')
            fig_gd = _viz_genome_diagnostic(_pk_gs, _thr_plot, scan_stats=_scan_stats_gs)
            fig_gd.savefig(plt_dir / 'core' / 'genome_diagnostic.png', dpi=300,
                           bbox_inches='tight', transparent=True)
            plt.close(fig_gd)
            log.info('  core/genome_diagnostic.png')
        except Exception as exc:
            log.warning('  genome_summary: %s', exc)

    # ── MI scan LOD profile ───────────────────────────────────────────────
    _mi_dir = plt_dir / 'extended' / 'mi'
    _mi_dir.mkdir(parents=True, exist_ok=True)
    mi_lod_path = ext_dir / 'mi_lod.csv'
    if mi_lod_path.exists():
        try:
            mi_df = pd.read_csv(mi_lod_path, low_memory=False)
            mi_df['chr'] = mi_df['chr'].astype(str)
            mi_pc = mi_df[mi_df['chr'] == str(peak_chr)]
            if len(mi_pc) > 0:
                top_gene = _pick_top_gene(mi_pc, peak_chr)
                mi_top   = mi_pc[mi_pc['gene_symbol'] == top_gene].sort_values('pos_mb')
                fig, ax  = plt.subplots(figsize=(10, 3), dpi=300)
                _viz_lod_profile(
                    mi_top['pos_mb'].values, mi_top['lod'].values,
                    ax=ax, chrom=peak_chr, fill=True,
                    color=FOUNDER_COLORS['WSBJ'],
                )
                ax.set_ylabel('LOD Score (MI)')
                ax.set_title(f'MI LOD Score Profile  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(_mi_dir / 'mi_lod_profile.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  extended/mi/mi_lod_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in mi_pc['gene_symbol'].values:
                    mi_p1 = mi_pc[mi_pc['gene_symbol'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        mi_p1['pos_mb'].values, mi_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, fill=True,
                        color=FOUNDER_COLORS['WSBJ'],
                    )
                    ax_p1.set_ylabel('LOD Score (MI)')
                    ax_p1.set_title(f'MI LOD Score Profile  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(_mi_dir / f'mi_lod_profile_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig_p1)
                    log.info('  extended/mi/mi_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  mi_lod_profile: %s', exc)

    # ── per-chromosome MI overlay (MI LOD vs scan LOD) ────────────────────
    if mi_lod_path.exists():
        try:
            _mi = pd.read_csv(mi_lod_path, low_memory=False)
            _mi['chr'] = _mi['chr'].astype(str)
            _pk = None
            if (core_dir / f'peak_lod_{_fsfx}.csv').exists():
                _pk = pd.read_csv(core_dir / f'peak_lod_{_fsfx}.csv')
                _pk['chr'] = _pk['chr'].astype(str)
            _scan = None
            if (core_dir / f'lod_{_fsfx}.csv').exists():
                _scan = pd.read_csv(core_dir / f'lod_{_fsfx}.csv', low_memory=False)
                _scan['chr'] = _scan['chr'].astype(str)
            _saved = 0
            for _c in sorted(_mi['chr'].unique(), key=_chr_sort_key):
                _mi_c = _mi[_mi['chr'] == _c]
                if _mi_c.empty:
                    continue
                _prio = PRIORITY_GENE.get(_c)
                if _prio and _prio in _mi_c['gene_symbol'].values:
                    _g = _prio
                elif _pk is not None:
                    _pk_c = _pk[_pk['chr'] == _c]
                    _g = (str(_pk_c.loc[_pk_c['peak_lod'].idxmax(), 'gene_symbol'])
                           if not _pk_c.empty
                           else str(_mi_c.groupby('gene_symbol')['lod'].max().idxmax()))
                else:
                    _g = str(_mi_c.groupby('gene_symbol')['lod'].max().idxmax())
                _mi_top = _mi_c[_mi_c['gene_symbol'] == _g].sort_values('pos_mb')
                if _mi_top.empty:
                    continue
                _thr = _thr_by_chr.get(_c)
                _common_pos = _mi_top['pos_mb'].values
                _mi_lod = _mi_top['lod'].values
                if (_scan is not None
                        and _g in _scan[_scan['chr'] == _c]['gene_symbol'].values):
                    _scan_top = _scan[(_scan['chr'] == _c)
                                      & (_scan['gene_symbol'] == _g)].sort_values('pos_mb')
                    _scan_lod = np.interp(_common_pos, _scan_top['pos_mb'].values,
                                          _scan_top['lod'].values)
                else:
                    _scan_lod = np.zeros_like(_mi_lod)
                fig_mi, ax_mi = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_mi_scan_overlay(
                    _common_pos, _scan_lod, _mi_lod,
                    ax=ax_mi, chrom=_c, gene=_g, threshold=_thr)
                fig_mi.tight_layout()
                fig_mi.savefig(_mi_dir / f'mi_chr{_c}.png', dpi=300, transparent=True)
                plt.close(fig_mi)
                _saved += 1
            if _saved:
                log.info('  extended/mi/mi_chr*.png  (%d files, MI vs scan overlay)', _saved)
        except Exception as exc:
            log.warning('  mi_per_chr: %s', exc)

    # ── SNP LOD profile for top eGene ────────────────────────────────────
    _snp_dir = plt_dir / 'extended' / 'snp'
    _snp_dir.mkdir(parents=True, exist_ok=True)
    snp_lod_path = ext_dir / 'snp_lod.csv'
    if snp_lod_path.exists():
        try:
            snp_df = pd.read_csv(snp_lod_path, low_memory=False)
            snp_df['chr'] = snp_df['chr'].astype(str)
            snp_pc = snp_df[snp_df['chr'] == str(peak_chr)]
            if len(snp_pc) > 0:
                top_gene = _pick_top_gene(snp_pc, peak_chr)
                snp_top  = snp_pc[snp_pc['gene_symbol'] == top_gene].sort_values('pos_mb')
                _snp_beta = (snp_top['beta_dosage'].values
                            if 'beta_dosage' in snp_top.columns
                            else np.zeros(len(snp_top)))
                _snp_peak_row = snp_top.loc[snp_top['lod'].idxmax()]
                fig, ax  = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_snp_lod_profile(
                    snp_top['pos_mb'].values, snp_top['lod'].values, _snp_beta,
                    chrom=str(peak_chr), gene=top_gene,
                    peak_pos=float(_snp_peak_row['pos_mb']),
                    ax=ax,
                )
                fig.tight_layout()
                fig.savefig(_snp_dir / 'snp_lod_profile.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  extended/snp/snp_lod_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in snp_pc['gene_symbol'].values:
                    snp_p1 = snp_pc[snp_pc['gene_symbol'] == _p1].sort_values('pos_mb')
                    _beta_p1 = (snp_p1['beta_dosage'].values
                                if 'beta_dosage' in snp_p1.columns
                                else np.zeros(len(snp_p1)))
                    _pk_p1 = snp_p1.loc[snp_p1['lod'].idxmax()]
                    fig_p1, ax_p1 = plt.subplots(figsize=(12, 3.5), dpi=300)
                    _viz_snp_lod_profile(
                        snp_p1['pos_mb'].values, snp_p1['lod'].values, _beta_p1,
                        chrom=str(peak_chr), gene=_p1,
                        peak_pos=float(_pk_p1['pos_mb']),
                        ax=ax_p1,
                    )
                    fig_p1.tight_layout()
                    fig_p1.savefig(_snp_dir / f'snp_lod_profile_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig_p1)
                    log.info('  extended/snp/snp_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  snp_lod_profile: %s', exc)

    # ── per-chromosome SNP fine-mapping figures ───────────────────────────
    if snp_lod_path.exists():
        try:
            _snp = pd.read_csv(snp_lod_path, low_memory=False)
            _snp['chr'] = _snp['chr'].astype(str)
            _pk = None
            if (core_dir / f'peak_lod_{_fsfx}.csv').exists():
                _pk = pd.read_csv(core_dir / f'peak_lod_{_fsfx}.csv')
                _pk['chr'] = _pk['chr'].astype(str)
            _scan = None
            if (core_dir / f'lod_{_fsfx}.csv').exists():
                _scan = pd.read_csv(core_dir / f'lod_{_fsfx}.csv', low_memory=False)
                _scan['chr'] = _scan['chr'].astype(str)
            _saved = 0
            for _c in sorted(_snp['chr'].unique(), key=_chr_sort_key):
                _snp_c = _snp[_snp['chr'] == _c]
                if _snp_c.empty:
                    continue
                _prio = PRIORITY_GENE.get(_c)
                if _prio and _prio in _snp_c['gene_symbol'].values:
                    _g = _prio
                elif _pk is not None:
                    _pk_c = _pk[_pk['chr'] == _c]
                    _candidates = set(_snp_c['gene_symbol'].values)
                    _pk_c_cand = _pk_c[_pk_c['gene_symbol'].isin(_candidates)]
                    _g = (str(_pk_c_cand.loc[_pk_c_cand['peak_lod'].idxmax(), 'gene_symbol'])
                           if not _pk_c_cand.empty
                           else str(_snp_c.groupby('gene_symbol')['lod'].max().idxmax()))
                else:
                    _g = str(_snp_c.groupby('gene_symbol')['lod'].max().idxmax())
                _snp_g = _snp_c[_snp_c['gene_symbol'] == _g].sort_values('pos_mb')
                if _snp_g.empty:
                    continue
                _snp_peak_row = _snp_g.loc[_snp_g['lod'].idxmax()]
                _snp_peak_mb = float(_snp_peak_row['pos_mb'])
                _haplo_pos, _haplo_lod = None, None
                if _scan is not None and _g in _scan[_scan['chr'] == _c]['gene_symbol'].values:
                    _scan_row = _scan[(_scan['chr'] == _c) & (_scan['gene_symbol'] == _g)].sort_values('pos_mb')
                    _haplo_pos = _scan_row['pos_mb'].values
                    _haplo_lod = _scan_row['lod'].values
                _beta = (_snp_g['beta_dosage'].values
                          if 'beta_dosage' in _snp_g.columns
                          else np.zeros(len(_snp_g)))
                fig_snp, ax_snp = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_snp_lod_profile(
                    _snp_g['pos_mb'].values, _snp_g['lod'].values, _beta,
                    haplo_positions=_haplo_pos, haplo_lod=_haplo_lod,
                    peak_pos=_snp_peak_mb, ax=ax_snp, chrom=_c, gene=_g)
                fig_snp.tight_layout()
                fig_snp.savefig(_snp_dir / f'snp_chr{_c}.png', dpi=300, transparent=True)
                plt.close(fig_snp)
                _saved += 1
            if _saved:
                log.info('  extended/snp/snp_chr*.png  (%d files)', _saved)
        except Exception as exc:
            log.warning('  snp_per_chr: %s', exc)

    # ── mediation lod_drop distribution per chromosome ────────────────────
    mediation_path = ext_dir / 'mediation.csv'
    if mediation_path.exists():
        try:
            med_df = pd.read_csv(mediation_path, low_memory=False)
            med_df['chr'] = med_df['chr'].astype(str)
            if 'lod_drop' not in med_df.columns:
                if 'lod_base' in med_df.columns and 'lod_mediated' in med_df.columns:
                    med_df['lod_drop'] = (med_df['lod_base'] - med_df['lod_mediated']).clip(lower=0)
                else:
                    raise KeyError('Column not found: lod_drop (and lod_base/lod_mediated absent)')
            _chr_sort_series = lambda s: s.map(lambda c: int(c) if c.isdigit() else 100)
            med_grp = (med_df.groupby('chr')['lod_drop']
                       .agg(mean_drop='mean', std_drop='std', n_pairs='count')
                       .reset_index()
                       .sort_values('chr', key=_chr_sort_series))
            n_outcomes_med = (len(med_df['gene_symbol'].unique())
                              if 'gene_symbol' in med_df.columns else 0)
            n_chrs = len(med_grp)
            fig, ax = plt.subplots(
                figsize=(max(8, n_chrs * 0.6), 4.5), dpi=300)
            _viz_mediation_summary_bar(
                med_grp['chr'].tolist(),
                med_grp['mean_drop'].values,
                std_drop=med_grp['std_drop'].fillna(0).values,
                n_pairs=med_grp['n_pairs'].values,
                n_outcomes=n_outcomes_med,
                ax=ax,
            )
            sns.despine(ax=ax)
            fig.tight_layout(pad=1.5)
            fig.savefig(plt_dir / 'extended' / 'mediation_by_chr.png', dpi=300,
                        bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.info('  extended/mediation_by_chr.png  (%d chr)', n_chrs)
        except Exception as exc:
            log.warning('  mediation_by_chr: %s', exc)

    # ── CI strip per eGene ───────────────────────────────────────────────
    ci_path = ext_dir / 'ci.csv'
    if ci_path.exists():
        try:
            ci_df = pd.read_csv(ci_path, low_memory=False)
            ci_df['ci_width'] = (ci_df['ci_hi_mb'] - ci_df['ci_lo_mb']
                                 if 'ci_hi_mb' in ci_df.columns
                                 else ci_df['ci_hi'] - ci_df['ci_lo'])
            if 'chr' not in ci_df.columns and 'gene_symbol' in ci_df.columns:
                _pk_path = core_dir / f'peak_lod_{_fsfx}.csv'
                if _pk_path.exists():
                    _pkdf = pd.read_csv(_pk_path)[['gene_symbol', 'chr']]
                    _pkdf['chr'] = _pkdf['chr'].astype(str)
                    ci_df = ci_df.merge(_pkdf, on='gene_symbol', how='left')
            ci_df['chr'] = ci_df.get('chr', pd.Series(dtype=str)).astype(str)
            if 'ci_lo_mb' not in ci_df.columns:
                ci_df['ci_lo_mb'] = ci_df.get('ci_lo', np.nan)
            if 'ci_hi_mb' not in ci_df.columns:
                ci_df['ci_hi_mb'] = ci_df.get('ci_hi', np.nan)
            _pk_path = core_dir / f'peak_lod_{_fsfx}.csv'
            if 'peak_lod' not in ci_df.columns and _pk_path.exists():
                _pk_lod_df = pd.read_csv(_pk_path)[['gene_symbol', 'peak_lod']]
                ci_df = ci_df.merge(_pk_lod_df, on='gene_symbol', how='left')
            _genome_thr = None
            if (core_dir / f'threshold_{_fsfx}.csv').exists():
                _t = pd.read_csv(core_dir / f'threshold_{_fsfx}.csv')
                _t['chr'] = _t['chr'].astype(str)
                _thr_wide = _t[_t['chr'] == 'genome']
                _col = 'threshold' if 'threshold' in _t.columns else 'threshold_lod'
                if len(_thr_wide) > 0:
                    _genome_thr = float(_thr_wide[_col].iloc[0])
            fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
            _ci_lo = (ci_df.get('ci_lo_mb', ci_df.get('ci_lo', np.full(len(ci_df), np.nan)))
                        .astype(float).values)
            _ci_hi = (ci_df.get('ci_hi_mb', ci_df.get('ci_hi', np.full(len(ci_df), np.nan)))
                        .astype(float).values)
            _ci_lod = (ci_df['peak_lod'].values if 'peak_lod' in ci_df.columns
                         else np.ones(len(ci_df)) * (_genome_thr or 3.0))
            _ci_mb = (ci_df['peak_mb'].values if 'peak_mb' in ci_df.columns
                        else np.zeros(len(ci_df)))
            _viz_ci_genomic_interval(
                ci_df['gene_symbol'].tolist() if 'gene_symbol' in ci_df.columns else list(ci_df.index.astype(str)),
                ci_df['chr'].astype(str).tolist() if 'chr' in ci_df.columns else ['?'] * len(ci_df),
                _ci_lo, _ci_hi, _ci_lod, _ci_mb,
                ax=ax,
            )
            fig.tight_layout(pad=1.5)
            fig.savefig(plt_dir / 'extended' / 'ci_strip.png', dpi=300,
                        bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.info('  extended/ci_strip.png  (%d genes)', len(ci_df))
        except Exception as exc:
            log.warning('  ci_strip: %s', exc)

    # ── instability LOD profile for top gene ─────────────────────────────
    _instability_dir = plt_dir / 'extended' / 'instability'
    _instability_dir.mkdir(parents=True, exist_ok=True)
    instability_path = ext_dir / 'instability.csv'
    if instability_path.exists():
        try:
            inst_df = pd.read_csv(instability_path, low_memory=False)
            inst_df['chr'] = inst_df['chr'].astype(str)
            inst_pc = inst_df[inst_df['chr'] == str(peak_chr)]
            if len(inst_pc) > 0:
                top_gene = _pick_top_gene(inst_pc, peak_chr)
                inst_top = inst_pc[inst_pc['gene_symbol'] == top_gene].sort_values('pos_mb')
                fig, ax  = plt.subplots(figsize=(10, 3), dpi=300)
                _viz_lod_profile(
                    inst_top['pos_mb'].values, inst_top['lod'].values,
                    ax=ax, chrom=peak_chr, fill=True,
                    color=FOUNDER_COLORS['CASTJ'],
                )
                ax.set_ylabel('Instability LOD Score')
                ax.set_title(f'Instability  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(_instability_dir / 'instability_profile.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  extended/instability/instability_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in inst_pc['gene_symbol'].values:
                    inst_p1 = inst_pc[inst_pc['gene_symbol'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        inst_p1['pos_mb'].values, inst_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, fill=True,
                        color=FOUNDER_COLORS['CASTJ'],
                    )
                    ax_p1.set_ylabel('Instability LOD Score')
                    ax_p1.set_title(f'Instability  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(_instability_dir / f'instability_profile_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig_p1)
                    log.info('  extended/instability/instability_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  instability_profile: %s', exc)

    # ── per-chromosome instability overlay (instability score vs scan LOD) ─
    if instability_path.exists():
        try:
            _inst = pd.read_csv(instability_path)
            _inst['chr'] = _inst['chr'].astype(str)
            _pk = None
            if (core_dir / f'peak_lod_{_fsfx}.csv').exists():
                _pk = pd.read_csv(core_dir / f'peak_lod_{_fsfx}.csv')
                _pk['chr'] = _pk['chr'].astype(str)
            _scan = None
            if (core_dir / f'lod_{_fsfx}.csv').exists():
                _scan = pd.read_csv(core_dir / f'lod_{_fsfx}.csv', low_memory=False)
                _scan['chr'] = _scan['chr'].astype(str)
            _saved = 0
            for _c in sorted(_inst['chr'].unique(), key=_chr_sort_key):
                _inst_c = _inst[_inst['chr'] == _c]
                if _inst_c.empty:
                    continue
                _prio = PRIORITY_GENE.get(_c)
                if _prio and _prio in _inst_c['gene_symbol'].values:
                    _g = _prio
                elif _pk is not None:
                    _pk_c = _pk[_pk['chr'] == _c]
                    _g = (str(_pk_c.loc[_pk_c['peak_lod'].idxmax(), 'gene_symbol'])
                            if not _pk_c.empty
                            else str(_inst_c.groupby('gene_symbol')['lod'].max().idxmax()))
                else:
                    _g = str(_inst_c.groupby('gene_symbol')['lod'].max().idxmax())
                _inst_top = _inst_c[_inst_c['gene_symbol'] == _g].sort_values('pos_mb')
                if _inst_top.empty:
                    continue
                _thr = _thr_by_chr.get(_c)
                _scan_lod: np.ndarray
                _scan_pos: np.ndarray
                if (_scan is not None
                        and _g in _scan[_scan['chr'] == _c]['gene_symbol'].values):
                    _scan_top = _scan[(_scan['chr'] == _c)
                                        & (_scan['gene_symbol'] == _g)].sort_values('pos_mb')
                    _scan_pos = _scan_top['pos_mb'].values
                    _scan_lod = _scan_top['lod'].values
                else:
                    _scan_pos = _inst_top['pos_mb'].values
                    _scan_lod = np.zeros(len(_scan_pos))
                _inst_col = ('peak_freq' if 'peak_freq' in _inst_top.columns else 'lod')
                _inst_raw = _inst_top[_inst_col].values
                _inst_scores = (_inst_raw / max(_inst_raw.max(), 1e-6)
                                  if _inst_raw.max() > 1.0 else _inst_raw)
                _common_pos = _inst_top['pos_mb'].values
                _scan_interp = np.interp(_common_pos, _scan_pos, _scan_lod)
                fig_inst, ax_inst = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_instability_overlay(
                    _common_pos, _scan_interp, _inst_scores,
                    ax=ax_inst, chrom=_c, gene=_g, threshold=_thr)
                fig_inst.tight_layout()
                fig_inst.savefig(_instability_dir / f'instability_chr{_c}.png', dpi=300, transparent=True)
                plt.close(fig_inst)
                _saved += 1
            if _saved:
                log.info('  extended/instability/instability_chr*.png  (%d files, overlay)', _saved)
        except Exception as exc:
            log.warning('  instability_per_chr: %s', exc)

    # ── timing / memory summary ───────────────────────────────────────────────
    # Excluded from plot (no meaningful compute or memory signal):
    #   S1 — I/O only: reads cache_info CSV + founder/genotype ID checks
    #   S2 — I/O only: reads a probe subset of haplotype prob arrays
    #   S3 — I/O only: reads kinship summary CSV + correlation stats
    #   S6 — dataframe filter: applies LOD threshold, no scan work
    # Shown sections are the compute-intensive scan steps.
    # Peak heap (tracemalloc) is used instead of RSS delta: ru_maxrss is a
    # process-lifetime high-water mark — once S4 allocates peak memory it stays
    # flat for all subsequent sections regardless of their actual footprint.
    _PLOT_SECTIONS = ['lmm_scan', 'permutation', 'mi_scan', 'mediation', 'bootstrap_ci', 'sanger_finemap', 'instability']
    scan_stats_path = py_dir / 'scan_stats.csv'
    if scan_stats_path.exists():
        try:
            ss_raw = pd.read_csv(scan_stats_path)
            ss_raw = ss_raw[
                ss_raw['section'].isin(_PLOT_SECTIONS)
                & ss_raw['elapsed_s'].gt(0)
            ].copy()
            ss_raw['_ord'] = ss_raw['section'].map(
                {s: i for i, s in enumerate(_PLOT_SECTIONS)})
            ss = ss_raw.sort_values('_ord').reset_index(drop=True)
            labels = [_SECTION_LABELS.get(s, s) for s in ss['section']]

            if len(ss) > 0:
                has_mem = ('peak_heap_mb' in ss.columns
                           and ss['peak_heap_mb'].gt(0).any())
                heap_vals = (ss['peak_heap_mb'].fillna(0).values
                             if has_mem else None)
                fig_tm = _viz_timing_memory_summary(
                    labels,
                    ss['elapsed_s'].values,
                    peak_heap_mb=heap_vals if has_mem else None,
                )
                fig_tm.savefig(plt_dir / 'timing_memory_summary.png',
                               dpi=300, transparent=True)
                plt.close(fig_tm)
                log.info('  timing_memory_summary.png')
        except Exception as exc:
            log.warning('  timing_memory_summary: %s', exc)

    # ── S4 throughput by chromosome (genes/min on real data) ─────────────────
    scan_stats_chr_path = py_dir / 'scan_stats.csv'
    if scan_stats_chr_path.exists():
        try:
            ss_all = pd.read_csv(scan_stats_chr_path)
            chr_rows = ss_all[ss_all['section'].str.startswith('lmm_scan_chr')].copy()
            if len(chr_rows) > 0:
                chr_rows['chrom'] = (chr_rows['section']
                                     .str.replace('lmm_scan_chr', '', regex=False))

                chr_rows = chr_rows.sort_values(
                    'chrom', key=lambda s: s.map(
                        lambda c: int(c) if c.isdigit() else 100))
                chr_rows['gpm'] = (chr_rows['n_items']
                                   / chr_rows['elapsed_s'].replace(0, float('nan')) * 60)
                _chrs_tp = chr_rows['chrom'].tolist()
                _gpm_tp = chr_rows['gpm'].fillna(0).values
                fig_tp, ax_tp = plt.subplots(
                    figsize=(max(9, len(chr_rows) * 0.55), 3.5), dpi=300)
                _viz_scan_throughput(_chrs_tp, _gpm_tp, ax=ax_tp)
                fig_tp.tight_layout()
                fig_tp.savefig(plt_dir / 'throughput_by_chr.png',
                               dpi=300, bbox_inches='tight', transparent=True)
                plt.close(fig_tp)
                med_gpm = chr_rows['gpm'].median()
                log.info('  throughput_by_chr.png  (median %.0f genes/min)', med_gpm)
        except Exception as exc:
            log.warning('  throughput_by_chr: %s', exc)


# ─────────────────────────────────────────────────────────────────────────────
# comparison plots  (9 figures covering S2–S5)
# ─────────────────────────────────────────────────────────────────────────────

def _make_comparison_plots(
    plt_dir: Path,
    cmp_dir: Path,
    py_dir:  Path,
    r_data:  Path,
    cache:   'GenomeCache',
    kinship_decomp: dict,
    founders: list[str],
    peak_chr: str,
    scan_tag: str,
    celltype: str | None = None,
    low_expr_genes: set[str] | None = None,
) -> None:
    """generate all R-vs-Python comparison figures; skip silently if inputs missing."""
    from scipy.stats import genextreme as gev_dist

    core_dir = py_dir / 'core'
    # filename suffix (celltype_or_whole + scan_tag) matching what run() wrote
    # every core scan CSV out as — see _sfx()
    _fsfx = _sfx(celltype, scan_tag)
    thr_path = core_dir / f'threshold_{_fsfx}.csv'

    display_f = _display_founders(founders)  # full names for labels/colors
    fcolors = [FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in display_f]

    # ── haplotype prob scatter ─────────────────────────────────────────────
    hap_path = r_data / 'haplotype_probs.csv'
    if hap_path.exists() and cache is not None:
        try:
            hap_r   = pd.read_csv(hap_path)
            fcols   = list(founders)
            lids8, _, probs8 = cache.get_chromosome('8')
            lidx    = {l: i for i, l in enumerate(lids8)}
            sidx    = {s: i for i, s in enumerate(cache.sample_ids)}
            py_vals, r_vals = [], []
            for _, grp in hap_r.groupby('locus'):
                loc = grp['locus'].iloc[0]
                if loc not in lidx:
                    continue
                li = lidx[loc]
                for _, rrow in grp.iterrows():
                    gt = rrow['genotype']
                    if gt not in sidx:
                        continue
                    si = sidx[gt]
                    py_vals.extend(probs8[li, si, :].tolist())
                    r_vals.extend((rrow[fcols].values.astype(float) / 2.0).tolist())
            if len(py_vals) >= 10:
                py_v, r_v = np.array(py_vals), np.array(r_vals)
                _hap_df = pd.DataFrame({'r_prob': r_v, 'py_prob': py_v})
                _hap_sparse = _hap_df['r_prob'] < 0.05
                fig, _ = _cmp_scatter(
                    _hap_df, 'r_prob', 'py_prob',
                    'R haplotype prob', 'Python haplotype prob',
                    cbar_label='mean probability',
                    sparse_mask=_hap_sparse,
                    save_path=plt_dir / 'haplotype_scatter.png',
                )
                plt.close(fig)
                log.info('  [plot] haplotype_scatter.png')
        except Exception as exc:
            log.warning('  [plot] haplotype_scatter: %s', exc)

    # ── kinship element scatter ───────────────────────────────────────────
    k8_path = r_data / f'kinship_chr{peak_chr}_elements.csv'
    if not k8_path.exists():
        k8_path = r_data / 'kinship_chr8_elements.csv'
    if k8_path.exists() and peak_chr in kinship_decomp:
        try:
            k_elem = pd.read_csv(k8_path)
            kd8    = kinship_decomp.get(str(peak_chr)) or kinship_decomp.get('8')
            K8_py  = kd8['vectors'] @ np.diag(kd8['values']) @ kd8['vectors'].T
            sids   = list(cache.sample_ids)
            sid_r  = k_elem['strain_i'].unique().tolist()
            common = [s for s in sids if s in set(sid_r)]
            cidx_py = {s: i for i, s in enumerate(sids)}
            cidx_r  = {s: i for i, s in enumerate(sid_r)}
            py_k, r_k = [], []
            for _, row in k_elem.iterrows():
                si, sj = row['strain_i'], row['strain_j']
                if si in cidx_py and sj in cidx_py:
                    py_k.append(float(K8_py[cidx_py[si], cidx_py[sj]]))
                    r_k.append(float(row['k_value']))
            if len(py_k) >= 10:
                _k_df = pd.DataFrame({'k_r': r_k, 'k_py': py_k})
                fig, _ = _cmp_scatter(
                    _k_df, 'k_r', 'k_py',
                    f'R kinship (LOCO chr{peak_chr})', f'Python kinship (LOCO chr{peak_chr})',
                    cbar_label='mean K',
                    save_path=plt_dir / 'kinship_scatter.png',
                )
                plt.close(fig)
                log.info('  [plot] kinship_scatter.png')
        except Exception as exc:
            log.warning('  [plot] kinship_scatter: %s', exc)

    # ── LOD density (peak LOD distribution) ───────────────────────────────
    # Read Python and R independently so each series plots whatever is available
    # without requiring both to be present in the merged comparison CSV.
    _py_lod_path = core_dir / f'peak_lod_{_fsfx}.csv'
    _r_lod_path  = r_data / 'peak_lod.csv'
    _density_sources = [
        (_py_lod_path, 'peak_lod', 'Python', '#4F6EAF'),
        (_r_lod_path,  'lod',      'R',      '#D83026'),
    ]
    _density_any = any(p.exists() for p, *_ in _density_sources)
    if _density_any:
        try:
            _lod_thr_d: float | None = None
            if thr_path.exists():
                _td_ld = pd.read_csv(thr_path)
                _td_ld['chr'] = _td_ld['chr'].astype(str)
                _tr_ld = _td_ld[_td_ld['chr'] == str(peak_chr)]
                _lod_thr_d = float(_tr_ld['threshold'].iloc[0]) if len(_tr_ld) > 0 else None
            _py_lods_d = (pd.read_csv(_py_lod_path, usecols=['peak_lod'])['peak_lod'].dropna().values
                          if _py_lod_path.exists() else None)
            _r_lod_col = 'lod' if _r_lod_path.exists() and 'lod' in pd.read_csv(
                _r_lod_path, nrows=0).columns else 'peak_lod'
            _r_lods_d  = (pd.read_csv(_r_lod_path, usecols=[_r_lod_col])[_r_lod_col].dropna().values
                          if _r_lod_path.exists() else None)
            fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
            _cmp_lod_density(_py_lods_d, _r_lods_d,
                             threshold=_lod_thr_d, ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 'lod_density.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  [plot] lod_density.png')
        except Exception as exc:
            log.warning('  [plot] lod_density: %s', exc)

    # ── peak LOD scatter (R vs Python) ────────────────────────────────────
    _cmp_lod_path = cmp_dir / 'peak_lod_comparison.csv'
    if _cmp_lod_path.exists():
        try:
            _cmp_lod = pd.read_csv(_cmp_lod_path)
            if len(_cmp_lod) >= 2 and 'peak_lod_py' in _cmp_lod.columns and 'peak_lod_r' in _cmp_lod.columns:
                _sparse_cmp = (
                    _cmp_lod['gene_symbol'].isin(low_expr_genes)
                    if low_expr_genes and 'gene_symbol' in _cmp_lod.columns else None
                )
                fig, _ = _cmp_scatter(
                    _cmp_lod, 'peak_lod_r', 'peak_lod_py',
                    'miQTL (R) peak LOD', 'mosaiQTL (Python) peak LOD',
                    priority_genes={'Cwc22': FOUNDER_COLORS['CASTJ']},
                    sparse_mask=_sparse_cmp,
                    save_path=plt_dir / 'peak_lod_scatter.png',
                )
                plt.close(fig)
                log.info('  [plot] peak_lod_scatter.png')
        except Exception as exc:
            log.warning('  [plot] peak_lod_scatter: %s', exc)

    # ── peak position scatter ──────────────────────────────────────────────
    eff_path = core_dir / f'allele_effects_{_fsfx}.csv'
    if eff_path.exists():
        try:
            eff_py = pd.read_csv(eff_path, usecols=['gene_symbol', 'peak_mb'])
            # Prefer r_data/peak_lod.csv (already one row per gene with pos_mb).
            # Fall back to per-chr lod.csv files only if peak_lod.csv absent.
            r_peak_lod_pos = r_data / 'peak_lod.csv'
            r_peak_rows = []
            if r_peak_lod_pos.exists():
                _rpp = pd.read_csv(r_peak_lod_pos, usecols=['gene', 'pos_mb'])
                _rpp = _rpp.rename(columns={'gene': 'gene_symbol'})
                r_peak_rows.append(_rpp.rename(columns={'pos_mb': 'peak_mb_r'}))
            else:
                for cand in sorted(
                    r_data.glob('chr*/lod.csv'),
                    key=lambda p: _chr_sort_key(p.parent.name.replace('chr', '')),
                ):
                    tmp = pd.read_csv(cand, usecols=['gene', 'pos_mb', 'lod'])
                    tmp = tmp.rename(columns={'gene': 'gene_symbol'})
                    r_peak_rows.append(
                        tmp.sort_values('lod', ascending=False)
                           .drop_duplicates('gene_symbol')[['gene_symbol', 'pos_mb']]
                           .rename(columns={'pos_mb': 'peak_mb_r'}))
            if r_peak_rows:
                r_pos = pd.concat(r_peak_rows).groupby('gene_symbol')['peak_mb_r'].first().reset_index()
                merged_pos = eff_py.merge(r_pos, on='gene_symbol', how='inner')
                if len(merged_pos) >= 5:
                    fig, _ = _cmp_scatter(
                        merged_pos, 'peak_mb_r', 'peak_mb',
                        'R peak position (Mb)', 'Python peak position (Mb)',
                        cbar_label='mean position (Mb)',
                        save_path=plt_dir / 'peak_pos_scatter.png',
                    )
                    plt.close(fig)
                    log.info('  [plot] peak_pos_scatter.png')
        except Exception as exc:
            log.warning('  [plot] peak_pos_scatter: %s', exc)

    # ── LOD profiles (top 3 genes on peak_chr) ────────────────────────────
    py_lod_path = core_dir / f'lod_{_fsfx}.csv'
    if py_lod_path.exists():
        try:
            lod_py_all = pd.read_csv(py_lod_path)
            lod_pc     = lod_py_all[lod_py_all['chr'].astype(str) == str(peak_chr)]
            if len(lod_pc) > 0:
                top_genes = (lod_pc.groupby('gene_symbol')['lod'].max()
                             .nlargest(3).index.tolist())
                r_lod_chr_path = r_data / f'chr{peak_chr}' / 'lod.csv'
                r_lod_chr = (pd.read_csv(r_lod_chr_path)
                             if r_lod_chr_path.exists() else None)
                _comp_thr_p: float | None = None
                if thr_path.exists():
                    _td_cp = pd.read_csv(thr_path)
                    _td_cp['chr'] = _td_cp['chr'].astype(str)
                    _tr_cp = _td_cp[_td_cp['chr'] == str(peak_chr)]
                    _comp_thr_p = float(_tr_cp['threshold'].iloc[0]) if len(_tr_cp) > 0 else None
                fig = _cmp_lod_profiles(
                    lod_pc, r_lod_chr, top_genes, str(peak_chr), thr=_comp_thr_p)
                fig.savefig(plt_dir / 'lod_profiles.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  [plot] lod_profiles.png  top %d genes on chr%s',
                         len(top_genes), peak_chr)
        except Exception as exc:
            log.warning('  [plot] lod_profiles: %s', exc)

    # ── allele effects scatter (Python betas vs R betas, 2× scale note) ───
    r_eff_path = r_data / 'allele_effects_chr8.csv'
    if eff_path.exists() and r_eff_path.exists():
        try:
            py_eff = pd.read_csv(eff_path)
            r_eff  = pd.read_csv(r_eff_path)
            if 'gene' in r_eff.columns:
                r_eff = r_eff.rename(columns={'gene': 'gene_symbol'})
            _disp_f = _display_founders(founders)
            py_beta_cols = [f'beta_{d}' for d in _disp_f if f'beta_{d}' in py_eff.columns]
            r_beta_cols  = [c for c in r_eff.columns if c.startswith('beta_')]
            merged_eff   = py_eff.merge(r_eff, on='gene_symbol', how='inner', suffixes=('_py', '_r'))
            if len(merged_eff) >= 5 and py_beta_cols and r_beta_cols:
                py_betas, r_betas = [], []
                for gene_row in merged_eff.itertuples():
                    for pcol, rcol in zip(py_beta_cols, r_beta_cols):
                        py_betas.append(getattr(gene_row, pcol + '_py', np.nan))
                        r_betas.append(getattr(gene_row, rcol + '_r',  np.nan))
                py_b = np.array(py_betas, dtype=float)
                r_b  = np.array(r_betas,  dtype=float)
                mask = np.isfinite(py_b) & np.isfinite(r_b)
                if mask.sum() >= 10:
                    fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
                    _cmp_allele_effects(py_b[mask], r_b[mask], _disp_f, ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 'allele_effects_scatter.png', dpi=300, transparent=True)
                    plt.close(fig)
                    log.info('  [plot] allele_effects_scatter.png')
        except Exception as exc:
            log.warning('  [plot] allele_effects_scatter: %s', exc)

    # ── permutation distribution (py vs R, fitted GEV) ────────────────────
    # Read each source independently — plot whichever series exists.
    py_perm_path = core_dir / 'perm_max_lods.csv'
    r_perm_path  = r_data / f'perm_max_lods_chr{peak_chr}.csv'
    if py_perm_path.exists() or r_perm_path.exists():
        try:
            py_perm = (pd.read_csv(py_perm_path)['max_lod'].dropna().values
                       if py_perm_path.exists() else np.array([]))
            r_perm  = (pd.read_csv(r_perm_path)['max_lod'].dropna().values
                       if r_perm_path.exists() else np.array([]))
            # derive canonical R n_perms from timing_sections.csv; fall back to CSV row count
            r_n_perms = len(r_perm)
            _r_timing_sec = r_data / 'timing_sections.csv'
            if _r_timing_sec.exists():
                _ts = pd.read_csv(_r_timing_sec)
                _perm_row = _ts[_ts['section'] == 'permutation']
                if len(_perm_row) > 0 and pd.notna(_perm_row['n_perms'].iloc[0]):
                    r_n_perms = int(_perm_row['n_perms'].iloc[0])
            def _safe_thr95(vals: np.ndarray) -> 'float | None':
                try:
                    return float(gev_dist.ppf(0.95, *gev_dist.fit(vals)))
                except Exception:
                    return None

            _has_plot = (len(py_perm) >= 5 or len(r_perm) >= 5)
            if _has_plot:
                fig, ax = plt.subplots(figsize=(8, 4.5), dpi=300)
                _cmp_perm_lod(
                    py_perm if len(py_perm) >= 5 else np.array([]),
                    r_perm  if len(r_perm)  >= 5 else np.array([]),
                    label_a=f'Python  n={len(py_perm)}',
                    label_b=f'R  n={r_n_perms}',
                    threshold_a=_safe_thr95(py_perm) if len(py_perm) >= 5 else None,
                    threshold_b=_safe_thr95(r_perm)  if len(r_perm)  >= 5 else None,
                    ax=ax,
                )
                fig.tight_layout()
                fig.savefig(plt_dir / 'perm_distribution.png', dpi=300, transparent=True)
                log.info('  [plot] perm_distribution.png')
                plt.close(fig)
        except Exception as exc:
            log.warning('  [plot] perm_distribution: %s', exc)

    # ── scan speed comparison (genes/min py vs R per chr) ─────────────────
    # Plot each series from its own source; show Python-only when R timing absent.
    scan_stats_path = py_dir / 'scan_stats.csv'
    r_timing_path   = r_data / 'timing_chr.csv'
    if scan_stats_path.exists() or r_timing_path.exists():
        try:
            py_scan_timing, r_scan_timing = pd.DataFrame(), None
            if scan_stats_path.exists():
                py_stats = pd.read_csv(scan_stats_path)
                py_scan_timing = py_stats[py_stats['section'].str.startswith('lmm_scan_chr')].copy()
                py_scan_timing['chr'] = py_scan_timing['section'].str.replace('lmm_scan_chr', '', regex=False)
                py_scan_timing['chr'] = py_scan_timing['chr'].astype(str)
                py_scan_timing['gpm'] = py_scan_timing['n_items'] / py_scan_timing['elapsed_s'].replace(0, float('nan')) * 60
            if r_timing_path.exists():
                r_timing = pd.read_csv(r_timing_path)
                if 'genes_per_min' in r_timing.columns:
                    r_scan_timing = r_timing[['chr', 'genes_per_min']].copy()
                    r_scan_timing['chr'] = r_scan_timing['chr'].astype(str)
                elif 'elapsed_s' in r_timing.columns and 'n_genes' in r_timing.columns:
                    r_scan_timing = r_timing.copy()
                    r_scan_timing['chr'] = r_scan_timing['chr'].astype(str)
                    r_scan_timing['genes_per_min'] = (r_scan_timing['n_genes']
                                             / r_scan_timing['elapsed_s'].replace(0, float('nan')) * 60)

            has_py = len(py_scan_timing) >= 2
            has_r  = r_scan_timing is not None and len(r_scan_timing) >= 2
            if has_py or has_r:
                chrs_sp = sorted(
                    (set(py_scan_timing['chr']) if has_py else set()) |
                    (set(r_scan_timing['chr']) if has_r else set()),
                    key=lambda c: int(c) if c.isdigit() else 100,
                )
                if len(chrs_sp) >= 2:
                    py_gpm = ({row['chr']: row['gpm'] for _, row in py_scan_timing.iterrows()}
                              if has_py else {})
                    r_gpm  = ({row['chr']: row['genes_per_min']
                               for _, row in r_scan_timing.iterrows()}
                              if has_r else {})
                    _py_arr_sp = np.array([py_gpm.get(c, 0.0) for c in chrs_sp])
                    _r_arr_sp  = (np.array([r_gpm.get(c, 0.0) for c in chrs_sp])
                                  if has_r else None)
                    fig, ax = plt.subplots(figsize=(max(10, len(chrs_sp) * 0.6), 4.5), dpi=300)
                    _viz_scan_throughput(chrs_sp, _py_arr_sp,
                                        r_genes_per_min=_r_arr_sp, ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 'scan_speed.png', dpi=300, transparent=True)
                    plt.close(fig)
                    log.info('  [plot] scan_speed.png  %d chr', len(chrs_sp))
        except Exception as exc:
            log.warning('  [plot] scan_speed: %s', exc)

    # ── per-chromosome FWER threshold comparison (Python vs R bars) ────────
    py_thr_path = core_dir / f'threshold_{_fsfx}.csv'
    r_thr_path  = r_data / 'thresholds_chr.csv'
    if py_thr_path.exists():
        try:
            _py_thr_df = pd.read_csv(py_thr_path)
            _py_thr_df['chr'] = _py_thr_df['chr'].astype(str)
            _col_thr = 'threshold' if 'threshold' in _py_thr_df.columns else 'threshold_lod'
            _py_thrs = dict(zip(_py_thr_df['chr'], _py_thr_df[_col_thr]))
            _genome_thr = _py_thrs.pop('genome', None)

            _r_thrs: dict[str, float] | None = None
            if r_thr_path.exists():
                _r_thr_df = pd.read_csv(r_thr_path)
                _r_thr_df['chr'] = _r_thr_df['chr'].astype(str)
                _r_col = 'threshold' if 'threshold' in _r_thr_df.columns else 'threshold_lod'
                _r_thrs = dict(zip(_r_thr_df['chr'], _r_thr_df[_r_col]))

            if _py_thrs:
                fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
                _viz_perm_threshold_comparison(
                    _py_thrs, _r_thrs,
                    genome_wide_thr=_genome_thr, ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 'perm_threshold_comparison.png',
                            dpi=300, transparent=True)
                plt.close(fig)
                log.info('  [plot] perm_threshold_comparison.png')
        except Exception as exc:
            log.warning('  [plot] perm_threshold_comparison: %s', exc)


# ─────────────────────────────────────────────────────────────────────────────
# main validation run
# ─────────────────────────────────────────────────────────────────────────────

def _scan_tag(scan_type: str, window_mb: float) -> str:
    """Directory-safe token identifying a scan configuration.

    Used to nest output under a scan-type-specific subdirectory so that runs
    with different ``scan_type`` (or, for local mode, different ``window_mb``)
    against the same data/out dir never overwrite each other's results.

    Parameters
    ----------
    scan_type :
        'C' (chromosome), 'G' (genome-wide), or 'L' (local ±window_mb).
    window_mb :
        Half-window size in Mb; only distinguishes the tag when scan_type=='L'.

    Returns
    -------
    str
        e.g. 'C', 'G', 'Lw1', 'Lw2.5'.
    """
    return scan_type if scan_type != 'L' else f'Lw{window_mb:g}'


def _sfx(celltype: str | None, scan_tag: str) -> str:
    """Filename suffix identifying a run's celltype (or whole-tissue) + scan config.

    Applied to every core scan CSV (lod/peak_lod/allele_effects/threshold) and plot
    PNG written by ``run()`` so a file keeps its run identity even when copied out
    of its directory nesting.

    Parameters
    ----------
    celltype :
        Cell-type label, or None for a whole-tissue run.
    scan_tag :
        Scan configuration token from ``_scan_tag()`` (e.g. 'C', 'G', 'Lw1').

    Returns
    -------
    str
        e.g. 'astrocyte_Lw1', 'whole_C'.
    """
    ct_token = (re.sub(r'[^a-z0-9]+', '_', celltype.lower()).strip('_')
                if celltype is not None else 'whole')
    return f'{ct_token}_{scan_tag}'


def run(data_dir: Path, out_dir: Path, sections: set[int],
        force: bool, n_genes: int | None,
        scan_type: str = 'L',
        window_mb: float = 1.0,
        scan_chr: list[str] | None = None,
        celltype: str | None = None,
        tissue: str = 'gastroc',
        priority_gene: dict[str, str] | None = None,
        gtf_path: str | None = None,
        sanger_alleles: str | None = None) -> dict:
    """run tissue characterization sections; return gate results dict.

    Parameters
    ----------
    data_dir         : tissue data root (contains metadata.csv, exp_gene_info_{tissue}.csv,
                       r_data/); in CT mode this is the CT dir with counts_lognorm.h5
    out_dir          : output root for this CT (or 'whole' dir for whole-tissue) — holds
                       the scan-type-invariant cache files (cache_info.csv,
                       haplotype_probs.csv, kinship_summary.csv, gate_summary.csv) plus
                       one {scan_tag}/ subdir per scan configuration — see _scan_tag()
    scan_type        : 'C' (chromosome — each gene vs every locus on its own chr),
                       'G' (genome-wide), or 'L' (local ±window_mb, default)
    window_mb        : half-window size in Mb for L mode (ignored for C/G)
    scan_chr         : chromosomes to scan; None = all chromosomes in kinship_decomp
    celltype         : cell-type label when running in CT mode; None = whole-tissue
    tissue           : tissue identifier used for file naming (e.g. 'gastroc', 'ctxhipp')
    priority_gene    : {chr: gene_name} to highlight in LOD/effect plots; empty = no priority
    """
    # in CT mode: outputs go flat into out_dir/{scan_tag}/ (no py_data/ subdir);
    # _tissue_root is the top-level tissue data dir (out_dir.parent.parent in CT mode,
    # data_dir otherwise) — used for shared files: metadata, r_data, celltypes/compare/
    _ct_mode     = celltype is not None
    _tissue_root = out_dir.parent.parent if _ct_mode else data_dir
    _stag        = _scan_tag(scan_type, window_mb)  # e.g. 'C', 'G', 'Lw1'
    # filename suffix (celltype_or_whole + scan_tag) appended to every core scan
    # CSV and plot PNG this run writes — see _sfx() — so a copied-out file keeps
    # its run identity (e.g. 'peak_lod_astrocyte_Lw1.csv', 'lod_whole_C.csv')
    _run_sfx     = _sfx(celltype, _stag)
    PRIORITY_GENE: dict[str, str] = priority_gene or {}
    # S8 allele H5 (GigaMUGA; superseded by Sanger fine-mapping — used only when present)
    _allele_h5_cand = data_dir / 'founder_alleles.h5'
    _allele_h5 = (_allele_h5_cand if _allele_h5_cand.exists()
                  else Path('validation/phase2/py_data/founder_alleles.h5'))
    # scan-type-scoped so a 'C'/'G'/'L' (or differing window_mb) rerun never
    # overwrites another scan configuration's output — see _scan_tag(). cache_dir is
    # NOT scan-type-scoped: cache_info/haplotype_probs/kinship_summary/gate_summary
    # depend only on the genome cache + genotype set, not on scan_type/window_mb, so
    # they live once at out_dir (shared across every scan_tag under it).
    r_data    = _tissue_root / _stag / 'r_data'   # R reference always from whole-tissue
    cache_dir = out_dir
    py_dir    = out_dir / _stag
    cmp_dir   = (_tissue_root / _stag / 'celltypes' / 'compare'
                 if _ct_mode else out_dir / _stag / 'compare')
    plt_dir   = py_dir / 'plots'
    ext_dir  = py_dir / 'extended'
    core_dir = py_dir / 'core'
    dirs_to_make = [cache_dir, py_dir, cmp_dir, plt_dir, core_dir, ext_dir,
                    plt_dir / 'core', plt_dir / 'extended']
    for d in dirs_to_make:
        d.mkdir(parents=True, exist_ok=True)

    results: dict[str, bool | None] = {}
    _scan_stats: list[dict] = []
    _t0 = time.perf_counter()

    def _record_stats(section: str, component: str, n_items: int,
                      elapsed_s: float, peak_heap_mb: float,
                      peak_rss_mb: float = 0.0) -> None:
        _scan_stats.append({
            'section':       section,
            'component':     component,
            'n_items':       n_items,
            'elapsed_s':     round(elapsed_s, 3),
            'peak_heap_mb':  round(peak_heap_mb, 1),
            'peak_rss_mb':   round(peak_rss_mb, 1),
        })

    # ── load metadata (always shared; CT mode reads same file as whole-tissue) ──
    meta_path = _tissue_root / 'metadata.csv'
    if not meta_path.exists():
        log.error('metadata.csv not found at %s — run convert_h5ad.py', meta_path)
        sys.exit(1)
    meta = pd.read_csv(meta_path)
    sample_ids_meta = meta['sample_id'].tolist()
    # map sample-level IDs (individual mice) to genotype IDs (strain names in cache)
    meta_geno_map: dict[str, str] = dict(zip(meta['sample_id'], meta['genotype']))
    log.info('metadata: %d samples  %d genotypes  datasets: %s',
             len(meta), meta['genotype'].nunique(),
             sorted(meta['dataset'].unique()))

    # ── gene info (chr annotation + strand-aware gene boundary for scan filter) ──
    # schema: gene_id=Ensembl (HDF5 key, identity/join column), gene_symbol=display
    # name, strand, chr, start, end. gene_id is the only stable identity key across
    # cohorts (var_names is cohort-dependent — see convert_h5ad.py module docstring).
    # start_mb = start/1e6 for + strand, end/1e6 for - strand (gene boundary, not TSS)
    if _ct_mode:
        _ct_safe_label = re.sub(r'[^a-z0-9]+', '_', celltype.lower()).strip('_')
        gene_info_path = data_dir / f'exp_gene_info_{_ct_safe_label}.csv'
    else:
        gene_info_path = data_dir / f'exp_gene_info_{tissue}.csv'
    gene_info = pd.read_csv(gene_info_path)
    gene_info['chr'] = gene_info['chr'].astype(str)
    log.info('gene_info: %d genes  (%s)', len(gene_info), gene_info_path.name)

    low_expr_genes: set[str] = set()
    if 'frac_samples_cpm_gt1' in gene_info.columns:
        low_expr_genes = set(
            gene_info.loc[gene_info['frac_samples_cpm_gt1'] < 0.5, 'gene_symbol'])
        log.info('low-expression genes (frac_cpm_gt1 < 0.5): %d / %d',
                 len(low_expr_genes), len(gene_info))

    # ── transcript TSS table (for near_tss_mb after peak calling) ───────────────
    # near_tss_mb = transcript TSS (strand-aware) nearest to peak_mb per gene
    # source: GENCODE vM36 transcript features; one row per transcript
    # gene_tss.vm36.csv is a raw derived-annotation file, excluded from the repo
    # (validation/input/ per .claudeignore policy); it lives in the sibling qtl/input/
    # directory alongside the raw h5ad — repo_root.parent, not a hardcoded home path
    _input_dirs = [Path(__file__).parent / 'input',
                  Path(__file__).parent.parent.parent / 'input']
    _gene_tss_path = next(
        (d / 'gene_tss.vm36.csv' for d in _input_dirs if (d / 'gene_tss.vm36.csv').exists()),
        None,
    )
    if _gene_tss_path is not None:
        _gene_tss = pd.read_csv(_gene_tss_path,
                                usecols=['gene_name', 'start', 'end', 'strand'])
        _gene_tss['tss_mb'] = np.where(
            _gene_tss['strand'] == '-',
            _gene_tss['end'] / 1e6,
            _gene_tss['start'] / 1e6,
        )
        log.info('gene_tss: %d transcripts  (%s)', len(_gene_tss), _gene_tss_path.name)
    else:
        _gene_tss = None
        log.warning('gene_tss.vm36.csv not found in %s — near_tss_mb will be omitted',
                    ' or '.join(str(d) for d in _input_dirs))

    if scan_type == 'L' and 'start' not in gene_info.columns:
        log.warning('%s has no "start" column — L scan mode will not filter loci',
                    gene_info_path.name)

    # ── load GenomeCache (must be pre-built as HDF5) ──────────────────────────
    hdf5_cache = PY_REF / 'cache.h5'
    if hdf5_cache.exists():
        log.info('loading GenomeCache from HDF5: %s', hdf5_cache)
        cache = GenomeCache.from_hdf5(str(hdf5_cache))
        cache = cache.reorder_to_canonical()  # fast no-op if already canonical
    else:
        log.error(
            'cache.h5 not found at %s\n'
            'Build it once with:\n'
            '  conda run -n bioinfo python validation/validate_tissue.py '
            '--tissue gastroc --build-cache\n'
            'or specify the genomecache path:\n'
            '  ... --build-cache --genomecache /path/to/genomecache',
            hdf5_cache)
        sys.exit(1)
    log.info('cache: %d chromosomes  %d strains  sample[0]=%s',
             len(cache.chromosomes), len(cache.sample_ids), cache.sample_ids[0])

    # ── load or compute LOCO kinship decomp ───────────────────────────────────
    kd_npz = PY_REF / 'kinship_decomp.npz'
    if kd_npz.exists():
        log.info('loading kinship decomp: %s', kd_npz)
        raw = np.load(kd_npz, allow_pickle=True)
        kinship_decomp = {k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
                          for k in cache.chromosomes if f'{k}_vals' in raw}
        del raw
    else:
        log.info('computing LOCO kinship from cache (first run — saves to %s)...', kd_npz)
        PY_REF.mkdir(parents=True, exist_ok=True)
        kin_raw = calc_kinship(cache, method='loco')
        kinship_decomp = decomp_kinship(kin_raw)
        save_kd: dict[str, np.ndarray] = {}
        for ch, kd in kinship_decomp.items():
            save_kd[f'{ch}_vals'] = kd['values']
            save_kd[f'{ch}_vecs'] = kd['vectors']
        np.savez(kd_npz, **save_kd)
        del kin_raw, save_kd; gc.collect()
        log.info('  saved to %s', kd_npz)

    # ── chromosomes to scan ───────────────────────────────────────────────────
    chrs_to_scan = [c for c in cache.chromosomes if c in kinship_decomp]
    if scan_chr is not None:
        chrs_to_scan = [c for c in chrs_to_scan if c in scan_chr]
        missing = set(scan_chr) - set(chrs_to_scan)
        if missing:
            log.warning('requested chr not in kinship_decomp: %s', sorted(missing))

    log.info('chromosomes to scan (%d): %s', len(chrs_to_scan), chrs_to_scan)

    # ── build covariates (sex + dataset dummies) + sex lookup for GxE ────────
    covariates_meta = _build_covariates(meta, sample_ids_meta)
    _ds_cats = sorted(set(meta['dataset']), key=str.lower)
    _ds_ref  = next((c for c in _ds_cats if c.lower().startswith('founder')), _ds_cats[0])
    log.info('covariates: shape %s  sex + dataset dummies  ref=%s  categories=%s',
             covariates_meta.shape, _ds_ref, _ds_cats)
    _sex_by_sid: dict[str, float] = (
        encode_sex(meta['sex']).set_axis(meta['sample_id']).to_dict()
        if 'sex' in meta.columns else {}
    )
    sex_meta = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sample_ids_meta])

    sec = 1
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\ncache: verifying loci count, founder IDs, and sample ordering against HDF5...')
            _rss_cache = _get_rss_mb()
            tracemalloc.start()
            _t_cache = time.perf_counter()
            cache_ok = True

            # always: characterize Python cache
            py_loci = {c: len(cache.get_chromosome(c)[0]) for c in cache.chromosomes}
            ci_py = pd.DataFrame([{'chr': c, 'n_loci': n}
                                  for c, n in py_loci.items()])
            ci_py.to_csv(cache_dir / 'cache_info.csv', index=False)
            log.info('  cache: %d chromosomes  total loci=%d  founders=%d  genotypes=%d',
                     len(cache.chromosomes), sum(py_loci.values()),
                     len(cache.founders), len(cache.sample_ids))
            log.info('  founders: %s', list(cache.founders))
            _n_items_cache = len(ci_py)

            # optional: compare loci counts to R
            cache_info_r = r_data / 'cache_info.csv'
            if cache_info_r.exists():
                ci_r = pd.read_csv(cache_info_r)
                ci_r['chr'] = ci_r['chr'].astype(str)
                merged = ci_py.merge(
                    ci_r[['chr', 'n_loci']].rename(columns={'n_loci': 'n_loci_r'}),
                    on='chr', how='outer')
                merged['delta'] = (merged['n_loci'] - merged['n_loci_r']).abs()
                merged.to_csv(cmp_dir / 'cache_info_cmp.csv', index=False)
                max_delta = int(merged['delta'].fillna(9999).max())
                loci_ok = _check('max loci delta', float(max_delta), 0.5, '<')
                cache_ok = cache_ok and loci_ok
                log.info('    vs R: loci delta max=%d', max_delta)

            # optional: compare founders/genotypes to R
            founders_r_path = r_data / 'founders.csv'
            if founders_r_path.exists():
                founders_r = pd.read_csv(founders_r_path)['founder'].tolist()
                py_founders = list(cache.founders)
                founders_ok = founders_r == py_founders
                sym = PASS if founders_ok else FAIL
                log.info('  %s vs R founders match=%s', sym, founders_ok)
                cache_ok = cache_ok and founders_ok

            genotype_ids_r_path = r_data / 'genotype_ids.csv'
            if genotype_ids_r_path.exists():
                gids_r  = pd.read_csv(genotype_ids_r_path)['genotype'].tolist()
                gids_py = list(cache.sample_ids)
                set_match = set(gids_r) == set(gids_py)
                order_match = gids_r == gids_py
                sym = PASS if (set_match and order_match) else (
                      FAIL if not set_match else '⚠️ ')
                log.info('  %s vs R genotypes: set_match=%s  order_match=%s',
                         sym, set_match, order_match)
                cache_ok = cache_ok and set_match

            _, _cache_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _cache_elapsed = time.perf_counter() - _t_cache
            _record_stats('cache', 'cache_info', _n_items_cache,
                          _cache_elapsed, _cache_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_cache))
            log.info('  [timing] cache: %.2fs  heap_peak=%.0fMB',
                     _cache_elapsed, _cache_peak / 1e6)

            results['cache_structure'] = cache_ok
            _mark_done(py_dir, str(sec))

    # haplotype probabilities
    sec = 2
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nhaplotype probs: exporting per-locus founder probability matrix sample for QC...')
            _rss_hap = _get_rss_mb()
            tracemalloc.start()
            _t_hap = time.perf_counter()
            founder_cols = list(cache.founders)
            probe_chr = '8' if '8' in cache.chromosomes else cache.chromosomes[0]
            lids_probe, _, probs_probe = cache.get_chromosome(probe_chr)
            locus_idx = {lid: i for i, lid in enumerate(lids_probe)}

            # always: sample 5 evenly-spaced loci and export prob table
            step     = max(1, len(lids_probe) // 5)
            sample_lids = [lids_probe[i] for i in range(0, len(lids_probe), step)][:5]
            hap_rows = []
            for lid in sample_lids:
                li = locus_idx[lid]
                for si, gtype in enumerate(cache.sample_ids):
                    row = {'chr': probe_chr, 'locus': lid, 'genotype': gtype}
                    row.update(dict(zip(founder_cols, probs_probe[li, si, :].tolist())))
                    hap_rows.append(row)
            hap_py = pd.DataFrame(hap_rows)
            hap_py.to_csv(cache_dir / 'haplotype_probs.csv', index=False)
            log.info('  sampled %d loci × %d genotypes on chr%s',
                     len(sample_lids), len(cache.sample_ids), probe_chr)
            _n_items_hap = len(hap_rows)

            # optional: compare argmax to R haplotype_probs.csv
            hap_path = r_data / 'haplotype_probs.csv'
            if hap_path.exists():
                hap_r = pd.read_csv(hap_path)
                sid_idx = {sid: i for i, sid in enumerate(cache.sample_ids)}
                rows = []
                for _, grp in hap_r.groupby('locus'):
                    loc = grp['locus'].iloc[0]
                    if loc not in locus_idx:
                        continue
                    li = locus_idx[loc]
                    for _, rrow in grp.iterrows():
                        gtype = rrow['genotype']
                        if gtype not in sid_idx:
                            continue
                        si = sid_idx[gtype]
                        py_probs = probs_probe[li, si, :]
                        r_probs  = rrow[founder_cols].values.astype(float) / 2.0
                        rows.append({
                            'locus': loc, 'genotype': gtype,
                            'py_argmax': int(np.argmax(py_probs)),
                            'r_argmax':  int(np.argmax(r_probs)),
                            'match': int(np.argmax(py_probs)) == int(np.argmax(r_probs)),
                        })
                if rows:
                    hap_cmp_df = pd.DataFrame(rows)
                    hap_cmp_df.to_csv(cmp_dir / 'haplotype_argmax_cmp.csv', index=False)
                    concordance = float(hap_cmp_df['match'].mean())
                    hap_ok = _check('haplotype argmax concordance', concordance, 0.95)
                    results['haplotype_probs'] = hap_ok
                    log.info('    vs R: concordance=%.4f  (%d/%d pairs)',
                             concordance, hap_cmp_df['match'].sum(), len(hap_cmp_df))
                else:
                    results['haplotype_probs'] = None
            else:
                results['haplotype_probs'] = None

            _, _hap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _hap_elapsed = time.perf_counter() - _t_hap
            _record_stats('haplotype_probs', 'haplotype_probe', _n_items_hap,
                          _hap_elapsed, _hap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_hap))
            log.info('  [timing] haplotype probs: %.2fs  heap_peak=%.0fMB',
                     _hap_elapsed, _hap_peak / 1e6)
            _mark_done(py_dir, str(sec))

    # LOCO kinship
    sec = 3
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nLOCO kinship: per-chromosome kinship excluding test chr, eigendecomposition for WLS rotation...')
            _rss_kinship = _get_rss_mb()
            tracemalloc.start()
            _t_kinship = time.perf_counter()

            # always: compute and export kinship summary across all chromosomes
            py_rows = []
            for ch, kd in kinship_decomp.items():
                K = kd['vectors'] @ np.diag(kd['values']) @ kd['vectors'].T
                n = K.shape[0]
                eigs = kd['values']
                py_rows.append({
                    'chr':          ch,
                    'n_genotypes':  n,
                    'mean_diag':    float(np.diag(K).mean()),
                    'mean_offdiag': float(K[np.tril_indices(n, -1)].mean()),
                    'min_eig':      float(eigs.min()),
                    'max_eig':      float(eigs.max()),
                })
            k_py = pd.DataFrame(py_rows)
            k_py.to_csv(cache_dir / 'kinship_summary.csv', index=False)
            _n_items_kinship = len(py_rows)
            log.info('  kinship: %d chr  mean_diag=%.4f  mean_offdiag=%.4f',
                     len(py_rows),
                     k_py['mean_diag'].mean(), k_py['mean_offdiag'].mean())

            # optional: compare to R kinship_summary.csv
            ksum_path = r_data / 'kinship_summary.csv'
            if ksum_path.exists():
                k_r = pd.read_csv(ksum_path)
                k_r['chr'] = k_r['chr'].astype(str)
                merged_k = k_py.merge(
                    k_r.rename(columns={'mean_diag': 'mean_diag_r',
                                        'mean_offdiag': 'mean_offdiag_r'}),
                    on='chr', how='inner')
                merged_k.to_csv(cmp_dir / 'kinship_cmp.csv', index=False)
                r_diag, _ = sp_stats.pearsonr(
                    merged_k['mean_diag'], merged_k['mean_diag_r'])
                r_off,  _ = sp_stats.pearsonr(
                    merged_k['mean_offdiag'], merged_k['mean_offdiag_r'])
                kinship_ok = (_check('kinship diag r',    r_diag, 0.999) and
                         _check('kinship offdiag r', r_off,  0.999))
                results['loco_kinship'] = kinship_ok
            else:
                results['loco_kinship'] = None

            _, _kinship_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _kinship_elapsed = time.perf_counter() - _t_kinship
            _record_stats('loco_kinship', 'kinship_summary', _n_items_kinship,
                          _kinship_elapsed, _kinship_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_kinship))
            log.info('  [timing] LOCO kinship: %.2fs  heap_peak=%.0fMB',
                     _kinship_elapsed, _kinship_peak / 1e6)

            if _scan_stats:
                _sec_ord = {s: i for i, s in enumerate(_SECTION_ORDER)}
                _sorted_stats = sorted(_scan_stats,
                                       key=lambda r: _sec_ord.get(r['section'], 999))
                pd.DataFrame(_sorted_stats).to_csv(
                    py_dir / 'scan_stats.csv', index=False)

            _mark_done(py_dir, str(sec))

    # LMM scan
    sec = 4
    py_lod_path  = core_dir / f'lod_{_run_sfx}.csv'
    peak_lod_path = core_dir / f'peak_lod_{_run_sfx}.csv'

    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nLMM scan: FWL-vectorized score test with LOCO kinship  scan_type=%s  chrs=%s...',
                     scan_type, chrs_to_scan)

            # ── sample breakdown — printed once before scan so users can
            # assess per-strain replicate depth before committing to a long run
            _n_total = len(meta)
            _n_strains = meta['genotype'].nunique()
            _rep_counts = (meta.groupby('genotype', sort=False)
                               .size()
                               .sort_values(ascending=False))
            _min_rep = int(_rep_counts.min())
            _max_rep = int(_rep_counts.max())
            _med_rep = float(_rep_counts.median())
            log.info('  samples: %d total  %d strains  replicates/strain '
                     'min=%d  median=%.1f  max=%d',
                     _n_total, _n_strains, _min_rep, _med_rep, _max_rep)
            if _min_rep == 1:
                _singles = sorted(_rep_counts[_rep_counts == 1].index.tolist())
                log.warning('  %d strain(s) with n=1 replicate (low-power): %s',
                            len(_singles), ', '.join(_singles))
            # canonical sort: founders (FOUNDER_ORDER) → F1s (F1_COLORS order)
            # → CC strains (natsorted); unknown strains append natsorted at end
            _strain_rank: dict[str, int] = {s: i for i, s in enumerate(STRAIN_ORDER)}
            _ns_keys = natsorted(_rep_counts.index.tolist())
            _ns_rank: dict[str, int] = {s: i for i, s in enumerate(_ns_keys)}

            def _canonical_key(strain: str) -> tuple:
                known = _strain_rank.get(strain)
                if known is not None:
                    return (0, known, 0)        # founders + F1s first
                return (1, 0, _ns_rank.get(strain, 0))  # CC strains natsorted

            _sorted_strains: list[tuple] = sorted(
                _rep_counts.items(), key=lambda x: _canonical_key(x[0]))

            # one line per strain: strain  n_samples  [dataset breakdown if >1 dataset]
            _multi_dataset = meta['dataset'].nunique() > 1 if 'dataset' in meta.columns else False
            for _strain, _n in _sorted_strains:
                if _multi_dataset:
                    _ds_counts = (meta[meta['genotype'] == _strain]
                                  .groupby('dataset').size()
                                  .to_dict())
                    _ds_str = '  ' + '  '.join(f'{d}:{n}' for d, n in sorted(_ds_counts.items()))
                else:
                    _ds_str = ''
                log.info('    %-20s  n=%d%s', _strain, _n, _ds_str)

            import json as _json
            from importlib.metadata import version as _pkg_version

            # ── sample_breakdown.csv — written alongside scan_info.json ──────
            # tidy per-strain replicate table; one row per sample so downstream
            # scripts can filter / pivot without re-parsing metadata.csv
            _breakdown_cols = ['strain', 'n_samples']
            if _multi_dataset:
                _breakdown_cols.append('dataset_breakdown')
            _breakdown_rows = []
            for _strain, _n in _sorted_strains:
                _row: dict = {'strain': _strain, 'n_samples': int(_n)}
                if _multi_dataset:
                    _ds_counts = (meta[meta['genotype'] == _strain]
                                  .groupby('dataset').size().to_dict())
                    _row['dataset_breakdown'] = ';'.join(
                        f'{d}:{n}' for d, n in sorted(_ds_counts.items()))
                _breakdown_rows.append(_row)
            _breakdown_df = pd.DataFrame(_breakdown_rows, columns=_breakdown_cols)
            _breakdown_df.to_csv(py_dir / 'sample_breakdown.csv', index=False)
            log.info('  sample_breakdown.csv written → %s', py_dir / 'sample_breakdown.csv')

            # ── scan_info.json — provenance + sample summary ──────────────────
            (py_dir / 'scan_info.json').write_text(_json.dumps({
                'mosaiqtl_version': _pkg_version('mosaiqtl'),
                'git_hash':         _git_hash(),
                'run_date':         time.strftime('%Y-%m-%d'),
                'celltype':         celltype if celltype else 'whole_tissue',
                'scan_type':        scan_type,
                'chr':              ','.join(chrs_to_scan) if chrs_to_scan else 'all',
                'n_genes':          n_genes,
                'samples': {
                    'n_total':   _n_total,
                    'n_strains': _n_strains,
                    'min_reps':  _min_rep,
                    'max_reps':  _max_rep,
                    'per_strain': {str(s): int(n) for s, n in _sorted_strains},
                },
            }, indent=2))

            h5_path = data_dir / 'counts_lognorm.h5'
            all_effect_rows: list[dict] = []
            _lod_header_written = False
            _rss_scan = _get_rss_mb()
            tracemalloc.start()
            t_scan = time.perf_counter()
            n_genes_total = 0
            # truncate output file so we start fresh on re-runs within a forced section
            py_lod_path.write_text('')

            for ch in chrs_to_scan:
                # select genes for this chromosome
                if scan_type == 'G':
                    ch_genes = gene_info
                else:  # C or L — cis genes only
                    ch_genes = gene_info[gene_info['chr'] == str(ch)]

                if len(ch_genes) == 0:
                    continue

                gene_ids_ch = ch_genes['gene_id'].tolist()   # HDF5 key = phenotype_id

                if n_genes is not None:
                    gene_ids_ch = gene_ids_ch[:n_genes]

                counts_ch, loaded_ids, sids_ch = _load_expression(h5_path, gene_ids_ch)
                if counts_ch.shape[0] == 0:
                    log.info('  chr%s: no genes found in HDF5 — skipping', ch)
                    continue

                # map sample-level IDs → genotype IDs (strain names in cache)
                geno_ids_ch = [meta_geno_map.get(s, s) for s in sids_ch]

                cov_ch = _build_covariates(meta, sids_ch)
                n_loci_ch = len(cache.get_chromosome(ch)[0])

                log.info('  chr%s: %d genes  %d loci (%s mode)',
                         ch, len(loaded_ids), n_loci_ch, scan_type)

                # build gene start lookup for all scan modes (needed for eff dict + L-mode filter)
                # strand-aware gene boundary: start for + strand, end for - strand (from gene_info)
                start_lookup: dict[str, float] = {}
                if 'start' in ch_genes.columns:
                    _has_strand = 'strand' in ch_genes.columns
                    for _, grow in ch_genes.iterrows():
                        if _has_strand and grow['strand'] == '-':
                            start_lookup[grow['gene_id']] = float(grow['end']) / 1e6
                        else:
                            start_lookup[grow['gene_id']] = float(grow['start']) / 1e6

                # gene_id is the HDF5 join key and phenotype_id — loaded_ids are already
                # real gene_id values. Look up the display gene_symbol from
                # exp_gene_info_{tissue}.csv, keyed by gene_id. Falls back to gene_id
                # itself when a CSV predates the gene_symbol column.
                gene_symbol_lookup: dict[str, str] = {}
                if 'gene_symbol' in ch_genes.columns:
                    gene_symbol_lookup = dict(zip(ch_genes['gene_id'], ch_genes['gene_symbol']))

                # map CLI scan_type to scan_window API
                _sw = {'C': 'cis', 'G': 'genome', 'L': 'local'}[scan_type]

                # ── precompute chromosome-constant data (cis / local only) ──────
                # K_n, evecs_ch, and H_t_raw depend only on the genotype cache and
                # kinship — not on the gene phenotype.  computing them once here
                # saves O(n_loci·n·m·J) rotation work per gene in the loop below.
                _chr_precomputed = False
                if _sw in ('cis', 'local'):
                    _z_ch, _vsid_ch = build_z_matrix(geno_ids_ch, list(cache.sample_ids))
                    _z_sub_ch    = _z_ch[_vsid_ch]                                # (n_valid, m)
                    _r_counts_ch = _z_sub_ch.sum(axis=0)                          # (m,)
                    _cov_valid   = cov_ch[_vsid_ch]                               # (n_valid, p)
                    _x_full_ch   = np.hstack(
                        [np.ones((_z_sub_ch.shape[0], 1)), _cov_valid])
                    # expand K_n = Z K_m Z^T (n×n, rank m); scan at sample level to match R
                    _ev_m  = kinship_decomp[ch]['values']
                    _evc_m = kinship_decomp[ch]['vectors']
                    _K_m   = (_evc_m * _ev_m) @ _evc_m.T
                    _K_n   = _z_sub_ch @ _K_m @ _z_sub_ch.T
                    del _K_m
                    _evals_ch, _evecs_ch = np.linalg.eigh(_K_n)
                    del _K_n
                    _chr_data_ch = precompute_chromosome(
                        cache, ch, _evals_ch, _evecs_ch, z_mat=_z_sub_ch)
                    _chr_precomputed = True
                    log.info('    chr%s: H precomputed (%d loci)',
                             ch, len(_chr_data_ch['locus_ids']))

                log.info('  running LMM scan on chr%s ...', ch)
                t_chr = time.perf_counter()
                sex_ch = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sids_ch])
                for gi, gene_id in enumerate(loaded_ids):
                    gene_symbol = gene_symbol_lookup.get(gene_id, gene_id)
                    if gi % 100 == 0 and gi > 0:
                        elapsed = time.perf_counter() - t_scan
                        log.info('    chr%s %d/%d  %.0fs elapsed',
                                 ch, gi + 1, len(loaded_ids), elapsed)
                    y_raw      = counts_ch[gi].astype(float)
                    gene_start_mb = start_lookup.get(gene_id)

                    if _chr_precomputed and not np.any(np.isnan(y_raw[_vsid_ch])):
                        # precomputed path: sample-level K_n scan (matches R's K_n = Z K_m Z^T)
                        y_sub  = rint(y_raw[_vsid_ch])                          # (n,) sample-level
                        h2, _  = estimate_h2_null(
                            _evecs_ch.T @ y_sub, _evecs_ch.T @ _x_full_ch, _evals_ch)
                        chr_res = scan_chromosome(
                            y_sub, _chr_data_ch, _x_full_ch, _evals_ch, _evecs_ch, h2)
                        if _sw == 'local' and gene_start_mb is not None:
                            _mask = np.abs(chr_res['positions'] - gene_start_mb) <= window_mb
                            if not _mask.any():
                                log.debug(
                                    '    %s: no loci within %.1fMb of %.3f on chr%s'
                                    ' — skipped', gene_id, window_mb, gene_start_mb, ch)
                                continue
                            chr_res = {
                                k: (v[_mask] if isinstance(v, np.ndarray)
                                    and len(v) == len(_mask) else v)
                                for k, v in chr_res.items()
                            }
                        n_loci_res = len(chr_res['locus_ids'])
                        scan = ScanResult(
                            locus_ids   = chr_res['locus_ids'],
                            chromosomes = np.full(n_loci_res, str(ch)),
                            positions   = chr_res['positions'],
                            lod         = chr_res['lod'],
                            p_value     = chr_res['p_value'],
                            betas       = chr_res['betas'],
                            h2_null     = {str(ch): h2},
                            phenotype_id = gene_id,
                            model        = 'additive',
                        )
                    else:
                        # fallback: NaN samples present or G mode — full scan_h2lmm path
                        scan = scan_h2lmm(
                            y_raw, cache,
                            covariates=cov_ch,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_ch,
                            use_rint=True,
                            phenotype_id=gene_id,
                            scan_window=_sw,
                            gene_chr=ch,
                            gene_pos_mb=gene_start_mb if _sw == 'local' else None,
                            window_mb=window_mb,
                        )
                    df = _scan_to_df(
                        scan,
                        gene_id,
                        gene_symbol=gene_symbol,
                    )
                    df.to_csv(py_lod_path,
                              mode='a',
                              header=not _lod_header_written,
                              index=False)
                    _lod_header_written = True

                    # collect peak-locus betas and h2 for allele effects + threshold gene selection
                    if hasattr(scan, 'betas') and scan.betas is not None:
                        peak_idx = int(np.argmax(scan.lod))
                        _h2_val  = (list(scan.h2_null.values())[0]
                                    if scan.h2_null else float('nan'))
                        _disp_founders = _display_founders(list(cache.founders))
                        # rss0/rss1 from precomputed path; NaN for scan_h2lmm fallback
                        _rss0_g = float(chr_res.get('rss0', float('nan'))
                                        if _chr_precomputed else float('nan'))
                        _rss1_g = float(chr_res['rss1'][peak_idx]
                                        if _chr_precomputed and 'rss1' in chr_res
                                        else float('nan'))
                        _nv     = int(_z_sub_ch.shape[0]) if _chr_precomputed else 0
                        _p      = int(_x_full_ch.shape[1]) if _chr_precomputed else 0
                        _df    = max(_nv - _p, 1)
                        _ok    = (_nv > _p
                                  and not np.isnan(_rss0_g)
                                  and not np.isnan(_rss1_g)
                                  and _rss0_g > 0)
                        _h2f   = float(_h2_val) if not np.isnan(float(_h2_val)) else 0.0
                        eff: dict = {
                            'gene_symbol':     gene_symbol,
                            'chr':        str(ch),
                            'start_mb':   round(float(gene_start_mb), 3) if gene_start_mb is not None else float('nan'),
                            'peak_locus': str(scan.locus_ids[peak_idx]),
                            'peak_mb':    float(scan.positions[peak_idx]),
                            'peak_lod':   float(scan.lod[peak_idx]),
                            'h2':         float(_h2_val),
                            'n_valid':    _nv,
                            'rss0':       _rss0_g,
                            'rss1':       _rss1_g,
                            'r2_peak':    round((_rss0_g - _rss1_g) / _rss0_g, 6) if _ok else float('nan'),
                            'sigma2_e':   round(_rss0_g * (1 - _h2f) / _df, 6)    if _ok else float('nan'),
                            'tau2':       round(_rss0_g * _h2f / _df, 6)           if _ok else float('nan'),
                        }
                        for fi, dname in enumerate(_disp_founders):
                            eff[f'beta_{dname}'] = float(scan.betas[peak_idx, fi])
                        all_effect_rows.append(eff)

                elapsed_chr = time.perf_counter() - t_chr
                gpm_chr     = len(loaded_ids) / elapsed_chr * 60 if elapsed_chr > 0 else 0
                _, _chr_heap_peak = tracemalloc.get_traced_memory()
                _record_stats(f'lmm_scan_chr{ch}', 'lmm_scan',
                              len(loaded_ids), elapsed_chr,
                              _chr_heap_peak / 1e6)
                log.info('    chr%s done: %d genes  %d loci  %.1fs  %.1f genes/min',
                         ch, len(loaded_ids), n_loci_ch, elapsed_chr, gpm_chr)

                n_genes_total += len(loaded_ids)
                del counts_ch
                if _chr_precomputed:
                    del _chr_data_ch, _evals_ch, _evecs_ch, _z_sub_ch, _r_counts_ch, _cov_valid
                gc.collect()

            elapsed_s = time.perf_counter() - t_scan
            _, _scan_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if _lod_header_written:
                genes_per_min = n_genes_total / elapsed_s * 60 if elapsed_s > 0 else 0
                log.info('  scan complete: %d genes  %.1fs  %.1f genes/min',
                         n_genes_total, elapsed_s, genes_per_min)
                _record_stats('lmm_scan', 'genome_total', n_genes_total, elapsed_s,
                              _scan_heap_peak / 1e6, max(0.0, _get_rss_mb() - _rss_scan))
                # read back only the columns needed for peak reduction — avoids
                # holding the full (genes × loci) LOD matrix in RAM twice
                log.info('  computing peak LOD per gene ...')
                # LOD_COLS (pipeline.py, single source of truth) is the schema on disk —
                # read it directly, no gene_name/gene_symbol translation shim.
                lod_df = pd.read_csv(py_lod_path, usecols=LOD_COLS,
                             dtype={'chr': str}, low_memory=False).drop(columns=['gene_id'])
            else:
                log.warning('  no LOD rows produced — check chr/gene_info alignment')
                lod_df = pd.DataFrame(columns=['gene_symbol', 'chr', 'locus', 'pos_mb', 'lod'])

            # save allele effects (peak-locus betas per gene)
            if all_effect_rows:
                eff_df = pd.DataFrame(all_effect_rows)
                if 'gene_id' in gene_info.columns:
                    _gid_map = gene_info.drop_duplicates('gene_symbol').set_index('gene_symbol')['gene_id']
                    eff_df.insert(1, 'gene_id', eff_df['gene_symbol'].map(_gid_map))
                _beta_cols = [c for c in eff_df.columns if c.startswith('beta_')]
                eff_df = eff_df.reindex(
                    columns=[c for c in ALLELE_EFFECTS_BASE_COLS + _beta_cols
                             if c in eff_df.columns])
                eff_df.to_csv(core_dir / f'allele_effects_{_run_sfx}.csv', index=False)
                log.info('  allele_effects.csv: %d genes', len(eff_df))

            # peak LOD per gene — include chr/locus/pos so GxE loop has position without effects file
            if len(lod_df) > 0:
                peak_py = (lod_df.loc[lod_df.groupby('gene_symbol')['lod'].idxmax()]
                           [['gene_symbol', 'chr', 'locus', 'pos_mb', 'lod']]
                           .rename(columns={'lod': 'peak_lod',
                                            'locus': 'peak_locus',
                                            'pos_mb': 'peak_mb'})
                           .reset_index(drop=True))
                del lod_df; gc.collect()
                if all_effect_rows:
                    _h2_map       = {r['gene_symbol']: r['h2']       for r in all_effect_rows}
                    _start_mb_map = {r['gene_symbol']: r['start_mb'] for r in all_effect_rows}
                    _nv_map       = {r['gene_symbol']: r['n_valid']  for r in all_effect_rows}
                    _rss0_map     = {r['gene_symbol']: r['rss0']     for r in all_effect_rows}
                    _rss1_map     = {r['gene_symbol']: r['rss1']     for r in all_effect_rows}
                    _r2_map       = {r['gene_symbol']: r['r2_peak']  for r in all_effect_rows}
                    _sig2e_map    = {r['gene_symbol']: r['sigma2_e'] for r in all_effect_rows}
                    _tau2_map     = {r['gene_symbol']: r['tau2']     for r in all_effect_rows}
                    peak_py['h2']       = peak_py['gene_symbol'].map(_h2_map)
                    peak_py['start_mb'] = peak_py['gene_symbol'].map(_start_mb_map)
                    peak_py['n_valid']  = peak_py['gene_symbol'].map(_nv_map)
                    peak_py['rss0']     = peak_py['gene_symbol'].map(_rss0_map)
                    peak_py['rss1']     = peak_py['gene_symbol'].map(_rss1_map)
                    peak_py['r2_peak']  = peak_py['gene_symbol'].map(_r2_map)
                    peak_py['sigma2_e'] = peak_py['gene_symbol'].map(_sig2e_map)
                    peak_py['tau2']     = peak_py['gene_symbol'].map(_tau2_map)
                # ── near_tss_mb: transcript TSS nearest to peak_mb (post-peak join) ──
                # cis_dist_mb = peak_mb − near_tss_mb; falls back to start_mb if NaN
                if _gene_tss is not None and 'peak_mb' in peak_py.columns:
                    _near: dict[str, float] = {}
                    for _, _pr in peak_py.iterrows():
                        _cands = _gene_tss[_gene_tss['gene_name'] == _pr['gene_symbol']]
                        if _cands.empty:
                            _near[_pr['gene_symbol']] = float('nan')
                        else:
                            _idx = (_cands['tss_mb'] - _pr['peak_mb']).abs().idxmin()
                            _near[_pr['gene_symbol']] = float(_cands.loc[_idx, 'tss_mb'])
                    peak_py['near_tss_mb'] = peak_py['gene_symbol'].map(_near)
                if 'peak_mb' in peak_py.columns:
                    _ref = peak_py['near_tss_mb'] if 'near_tss_mb' in peak_py.columns \
                        else peak_py.get('start_mb')
                    if _ref is not None:
                        peak_py['cis_dist_mb'] = (
                            peak_py['peak_mb'] - _ref).round(6)
                        if 'near_tss_mb' in peak_py.columns and 'start_mb' in peak_py.columns:
                            _mask_nan = peak_py['near_tss_mb'].isna()
                            peak_py.loc[_mask_nan, 'cis_dist_mb'] = (
                                peak_py.loc[_mask_nan, 'peak_mb']
                                - peak_py.loc[_mask_nan, 'start_mb']).round(6)
                peak_py = peak_py.reindex(
                    columns=[c for c in PEAK_LOD_COLS if c in peak_py.columns])
                if 'gene_id' in gene_info.columns:
                    _gid_map = gene_info.drop_duplicates('gene_symbol').set_index('gene_symbol')['gene_id']
                    peak_py.insert(1, 'gene_id', peak_py['gene_symbol'].map(_gid_map))
                peak_py.to_csv(peak_lod_path, index=False)

                # ── compare to R reference ────────────────────────────────────
                # Prefer r_data/peak_lod.csv (merged genome-wide, one row per gene).
                # Fall back to per-chr lod.csv files if peak_lod.csv not yet written.
                # R CSVs output 'gene' — rename to gene_symbol immediately after read,
                # matching the display-name convention on the Python side (LOD_COLS/
                # PEAK_LOD_COLS), so the comparison merge key lines up without a shim.
                r_peak_lod_path = r_data / 'peak_lod.csv'
                r_peak_rows = []
                if r_peak_lod_path.exists():
                    _rpl = pd.read_csv(r_peak_lod_path, usecols=['gene', 'lod'])
                    _rpl = _rpl.rename(columns={'gene': 'gene_symbol'})
                    r_peak_rows.append(
                        _rpl.rename(columns={'lod': 'peak_lod_r'}))
                    del _rpl
                else:
                    for ch in chrs_to_scan:
                        r_lod_path_ch = r_data / f'chr{ch}' / 'lod.csv'
                        if r_lod_path_ch.exists():
                            r_lod_ch = pd.read_csv(r_lod_path_ch,
                                                    usecols=['gene', 'lod'])
                            r_lod_ch = r_lod_ch.rename(columns={'gene': 'gene_symbol'})
                            pk_r = (r_lod_ch.groupby('gene_symbol')['lod'].max()
                                    .reset_index()
                                    .rename(columns={'lod': 'peak_lod_r'}))
                            r_peak_rows.append(pk_r)
                            del r_lod_ch; gc.collect()

                if r_peak_rows:
                    peak_r = pd.concat(r_peak_rows, ignore_index=True)
                    peak_r = (peak_r.groupby('gene_symbol')['peak_lod_r'].max()
                              .reset_index())
                    cmp = peak_py.merge(peak_r, on='gene_symbol', how='inner')
                    # add _py suffix only in the comparison CSV for side-by-side readability
                    cmp = cmp.rename(columns={'peak_lod': 'peak_lod_py'})
                    cmp.to_csv(cmp_dir / 'peak_lod_comparison.csv', index=False)
                    n_matched = len(cmp)
                    log.info('  matched %d / %d genes to R reference', n_matched, len(peak_py))

                    if n_matched >= 2:
                        r_val, _ = sp_stats.pearsonr(cmp['peak_lod_py'],
                                                      cmp['peak_lod_r'])
                        r2 = float(r_val ** 2)
                        slope, intercept, *_ = sp_stats.linregress(
                            cmp['peak_lod_r'], cmp['peak_lod_py'])
                        max_delta = float((cmp['peak_lod_py'] -
                                           cmp['peak_lod_r']).abs().max())
                        lod_ok = _check('peak LOD r²', r2, 0.975)
                        _check('slope', abs(slope - 1.0), 0.05, '<')
                        log.info('  r²=%.4f  slope=%.3f  intercept=%.3f  '
                                 'max|Δ|=%.3f  n=%d',
                                 r2, slope, intercept, max_delta, n_matched)

                        _sparse_mask = (
                            cmp['gene_symbol'].isin(low_expr_genes)
                            if low_expr_genes and 'gene_symbol' in cmp.columns else None
                        )
                        fig, _ax = _cmp_scatter(
                            cmp, 'peak_lod_r', 'peak_lod_py',
                            'miQTL (R) peak LOD', 'mosaiQTL (Python) peak LOD',
                            priority_genes={'Cwc22': FOUNDER_COLORS['CASTJ']},
                            sparse_mask=_sparse_mask,
                            save_path=plt_dir / f'peak_lod_scatter_{_run_sfx}.png',
                        )
                        plt.close(fig)

                        results['lmm_lod_r2'] = lod_ok
                        results['lmm_slope_ok'] = abs(slope - 1.0) < 0.05
                    else:
                        log.warning('  too few matched genes for r²')
                        results['lmm_lod_r2'] = False
                else:
                    log.info('  %s no R chr*/lod.csv found — r² comparison skipped', SKIP)
                    results['lmm_lod_r2'] = None
            else:
                results['lmm_lod_r2'] = None
            _mark_done(py_dir, str(sec))
    elif py_lod_path.exists():
        log.info('loading cached scan results (lod.csv from prior run)')
        # reload gate result and scan stats from their checkpoint CSVs so the
        # scan summary at the end of the run shows real values, not N/A
        _gate_csv = cache_dir / 'gate_summary.csv'
        if _gate_csv.exists():
            try:
                _gc = pd.read_csv(_gate_csv)
                _lod_gate_row = _gc[_gc['gate'] == 'lmm_lod_r2']
                if len(_lod_gate_row) > 0:
                    _rv = str(_lod_gate_row.iloc[0]['result'])
                    if _rv == 'True':
                        results['lmm_lod_r2'] = True
                    elif _rv == 'False':
                        results['lmm_lod_r2'] = False
                    # 'skip' or missing → leave absent (None via .get)
            except Exception as _exc:
                log.debug('  gate_summary reload failed: %s', _exc)
        _ss_csv = py_dir / 'scan_stats.csv'
        if _ss_csv.exists() and not _scan_stats:
            try:
                _ss_loaded = pd.read_csv(_ss_csv).to_dict('records')
                _scan_stats.extend(_ss_loaded)
            except Exception as _exc:
                log.debug('  scan_stats reload failed: %s', _exc)

    # ── determine peak chromosome (from S4 results) ───────────────────────────
    # used as reference for S5–S11 downstream sections
    peak_chr = chrs_to_scan[0] if chrs_to_scan else '8'
    if peak_lod_path.exists():
        _pk = pd.read_csv(peak_lod_path)
        _pk['chr'] = _pk['chr'].astype(str)
        if len(_pk) > 0 and 'peak_lod' in _pk.columns and 'chr' in _pk.columns:
            peak_chr = str(_pk.iloc[_pk['peak_lod'].argmax()]['chr'])

    log.info('[downstream sections] peak_chr=%s', peak_chr)

    egenes_df: 'pd.DataFrame | None' = None
    chr_null_lods: 'dict | None' = None
    perm_max_lods: 'np.ndarray | None' = None
    perm_n: int = 0

    # permutation threshold
    sec = 5
    r_perm_lod_path = r_data / f'perm_max_lods_chr{peak_chr}.csv'
    if sec in sections:
        log.info('\npermutation threshold + eGene calling: genome-wide GEV null from phenotype permutation → LOD threshold → eGene calls...')
        if not _checkpoint(py_dir, str(sec), force):
            _rss_perm = _get_rss_mb()
            tracemalloc.start()
            _t_perm = time.perf_counter()
            if not peak_lod_path.exists():
                log.info('  %s peak_lod.csv not found — run S4 first', SKIP)
                results['perm_threshold'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'

                # ── pick median-h² gene as representative for permutation ──────
                # Prefer h² from allele_effects.csv if the column exists
                # (written when S4 was run with the current code).  When the
                # on-disk file predates the h² column, estimate h² independently
                # using estimate_h2_null on a random gene sample — fast (Brent's
                # 1D optimisation, no scan) and makes S5 self-contained.
                _perm_gene: str
                _perm_id:   str
                _h2_med:    float

                eff_path_perm = core_dir / f'allele_effects_{_run_sfx}.csv'
                _h2_from_effects = False
                if eff_path_perm.exists():
                    _eff_cols = pd.read_csv(eff_path_perm, nrows=0).columns.tolist()
                    if 'h2' in _eff_cols:
                        _h2_from_effects = True

                if _h2_from_effects:
                    _eff_perm    = pd.read_csv(eff_path_perm, usecols=['gene_id', 'gene_symbol', 'h2'])
                    _h2_valid  = _eff_perm['h2'].dropna()
                    _h2_med    = float(_h2_valid.median())
                    _best_row  = _eff_perm.iloc[(_h2_valid - _h2_med).abs().values.argmin()]
                    _perm_gene = str(_best_row['gene_symbol'])
                    _perm_id   = str(_best_row['gene_id'])
                    # join peak_lod for chr + LOD of representative gene
                    _perm_gene_info = (pd.read_csv(peak_lod_path, usecols=['gene_symbol', 'chr', 'peak_lod'])
                                   if peak_lod_path.exists() else None)
                    _rep_info = (
                        _perm_gene_info[_perm_gene_info['gene_symbol'] == _perm_gene].iloc[0]
                        if _perm_gene_info is not None and (_perm_gene_info['gene_symbol'] == _perm_gene).any()
                        else None
                    )
                    if _rep_info is not None:
                        log.info('  h² from effects file  median=%.3f  gene: %s  '
                                 'chr%s  LOD=%.2f  h²=%.3f',
                                 _h2_med, _perm_gene,
                                 str(_rep_info['chr']), float(_rep_info['peak_lod']),
                                 float(_best_row['h2']))
                    else:
                        log.info('  h² from effects file  median=%.3f  gene: %s (h²=%.3f)',
                                 _h2_med, _perm_gene, float(_best_row['h2']))
                else:
                    # estimate h² on-the-fly from a random sample of peak_lod genes
                    log.info('  estimating h² for representative gene selection ...')
                    _peak_df_perm     = pd.read_csv(peak_lod_path)
                    _perm_chr_genes    = _peak_df_perm[_peak_df_perm['chr'].astype(str) == str(peak_chr)]
                    _perm_chr_sample   = _perm_chr_genes.sample(
                        min(500, len(_perm_chr_genes)), random_state=1)
                    _samp_ids     = _perm_chr_sample['gene_id'].tolist()
                    _samp_symbols = dict(zip(_perm_chr_sample['gene_id'],
                                              _perm_chr_sample['gene_symbol']))
                    _perm_counts, _perm_loaded, _perm_sids = _load_expression(h5_path, _samp_ids)
                    _perm_gids   = [meta_geno_map.get(s, s) for s in _perm_sids]
                    _kd_ref    = kinship_decomp[peak_chr]
                    _ev_ref    = _kd_ref['values']
                    _evc_ref   = _kd_ref['vectors']
                    # restrict to samples present in cache
                    _perm_valid_idx   = np.array([i for i, g in enumerate(_perm_gids)
                                           if g in cache.sample_ids])
                    _h2_pairs: list[tuple[str, float]] = []
                    for _pgi, _pgn in enumerate(_perm_loaded):
                        _py = rint(_perm_counts[_pgi].astype(float))
                        if np.any(np.isnan(_py[_perm_valid_idx])):
                            continue
                        try:
                            _pcov = _build_covariates(meta, [_perm_sids[j]
                                                              for j in _perm_valid_idx])
                            _px   = np.hstack([np.ones((_pcov.shape[0], 1)), _pcov])
                            _ph2, _ = estimate_h2_null(
                                _evc_ref.T @ _py[_perm_valid_idx],
                                _evc_ref.T @ _px,
                                _ev_ref)
                            _h2_pairs.append((_pgn, float(_ph2)))
                        except Exception:
                            pass
                    del _perm_counts; gc.collect()
                    if not _h2_pairs:
                        log.warning('  h² estimation failed for all sampled genes; '
                                    'falling back to highest-LOD gene on chr%s', peak_chr)
                        _fallback_idx = _perm_chr_genes['peak_lod'].idxmax()
                        _perm_id   = str(_perm_chr_genes.loc[_fallback_idx, 'gene_id'])
                        _perm_gene = str(_perm_chr_genes.loc[_fallback_idx, 'gene_symbol'])
                        _h2_med    = float('nan')
                    else:
                        _h2_arr    = np.array([h for _, h in _h2_pairs])
                        _h2_med    = float(np.median(_h2_arr))
                        _best_idx  = int(np.argmin(np.abs(_h2_arr - _h2_med)))
                        _perm_id   = _h2_pairs[_best_idx][0]
                        _perm_gene = _samp_symbols.get(_perm_id, _perm_id)
                    log.info('  median h²=%.3f  representative gene: %s', _h2_med, _perm_gene)

                _cts, _lds, _sids = _load_expression(h5_path, [_perm_id])
                if _cts.shape[0] == 0:
                    log.info('  %s expression for %s not found', SKIP, _perm_id)
                    results['perm_threshold'] = None
                else:
                    _gids = [meta_geno_map.get(s, s) for s in _sids]
                    _y    = _cts[0].astype(float)

                    # L (local) mode: the real per-gene scan never leaves a
                    # ±window_mb window around the gene's own TSS, so a null
                    # drawn from a genome-wide (or even whole-chromosome) max
                    # tests far more loci than the real scan ever did —
                    # inflating the threshold and understating power. Restrict
                    # the representative gene's null scan to its own
                    # chr/window so the multiple-testing burden matches.
                    # C/G modes keep the genome-wide null: 'C' still tests
                    # every locus on some chromosome per gene (the max can
                    # land on any of them), and 'G' is genome-wide already.
                    _perm_chrom = None
                    _perm_gene_pos_mb = None
                    if scan_type == 'L':
                        _perm_row = gene_info[gene_info['gene_id'] == _perm_id]
                        if not _perm_row.empty:
                            _prow = _perm_row.iloc[0]
                            _perm_chrom = str(_prow['chr'])
                            if 'strand' in _perm_row.columns and _prow['strand'] == '-':
                                _perm_gene_pos_mb = float(_prow['end']) / 1e6
                            else:
                                _perm_gene_pos_mb = float(_prow['start']) / 1e6
                            log.info('  local-window null: chr%s  TSS=%.3fMb  ±%.1fMb',
                                     _perm_chrom, _perm_gene_pos_mb, window_mb)
                        else:
                            log.warning('  %s not found in gene_info — falling back to '
                                        'genome-wide null for representative gene %s',
                                        _perm_id, _perm_gene)

                    _null = generate_null_outcomes(_y, n_perms=1000, seed=1)
                    _t_perm_scan = time.perf_counter()
                    _gw_max, _chr_nulls = run_threshold_scans(
                        _null, cache,
                        phenotype=_y,
                        covariates=None,
                        kinship_decomp=kinship_decomp,
                        chrom=_perm_chrom,
                        sample_ids=_gids,
                        gene_pos_mb=_perm_gene_pos_mb,
                        window_mb=window_mb if _perm_chrom is not None else None,
                    )
                    _perm_scan_elapsed = time.perf_counter() - _t_perm_scan
                    chr_null_lods = _chr_nulls
                    perm_max_lods = _gw_max
                    perm_n        = int(np.isfinite(_gw_max).sum())
                    _perms_per_hr = round(perm_n / _perm_scan_elapsed * 3600, 1)

                    _gw_thr, _ = get_gev_thresholds(_gw_max, percentile=0.95)
                    if _gw_thr is not None and _gw_thr == 0.0:
                        log.error(
                            '  permutation null is all-zeros — kinship singular on every '
                            'chromosome; threshold=0.0 is invalid.  Re-run with a wider '
                            'scan window or check kinship_decomp for this cell type.')
                        _gw_thr = None
                    log.info('  genome-wide GEV threshold=%s  (n_perms=%d)',
                             f'{_gw_thr:.3f}' if _gw_thr is not None else 'n/a', perm_n)
                    log.info('  scan loop: %.0fs for %d perms  (%.1f perms/hr)',
                             _perm_scan_elapsed, perm_n, _perms_per_hr)

                    # save genome-wide null distribution (single file)
                    pd.DataFrame({'max_lod': _gw_max}).to_csv(
                        core_dir / 'perm_max_lods.csv', index=False)

                    # save per-chromosome nulls for distribution plots
                    for _c5, _c5_null in _chr_nulls.items():
                        pd.DataFrame({'max_lod': _c5_null}).to_csv(
                            core_dir / f'perm_max_lods_chr{_c5}.csv', index=False)

                    # threshold.csv: genome_wide row (used for eGene calling)
                    # + per-chr rows (GEV on each chr's null, for comparison only)
                    _thr_rows_out: list[dict] = [{
                        'chr':            'genome_wide',
                        'threshold':      _gw_thr,
                        'perm_gene':      _perm_gene,
                        'h2_median':      _h2_med,
                        'n_perms':        perm_n,
                        'scan_elapsed_s': round(_perm_scan_elapsed, 1),
                        'perms_per_hr':   _perms_per_hr,
                    }]
                    for _c5, _c5_null in _chr_nulls.items():
                        _c5_thr, _ = get_gev_thresholds(_c5_null, percentile=0.95)
                        _thr_rows_out.append({
                            'chr':            str(_c5),
                            'threshold':      _c5_thr,
                            'perm_gene':      _perm_gene,
                            'h2_median':      None,
                            'n_perms':        int(np.isfinite(_c5_null).sum()),
                            'scan_elapsed_s': None,
                            'perms_per_hr':   None,
                        })
                    pd.DataFrame(_thr_rows_out).to_csv(
                        core_dir / f'threshold_{_run_sfx}.csv', index=False)

                    # compare to R peak-chr permutation (approximate — different scope)
                    if r_perm_lod_path.exists():
                        r_max_lods = pd.read_csv(r_perm_lod_path)['max_lod'].values
                        r_thr, _   = get_gev_thresholds(r_max_lods, percentile=0.95)
                        if r_thr is not None and _gw_thr is not None:
                            log.info('  R chr%s GEV=%.3f  delta vs genome-wide=%.3f'
                                     '  (expected: gw > per-chr)', peak_chr, r_thr,
                                     _gw_thr - r_thr)

                    results['perm_threshold'] = True
                    n_genes_perm_total = len(_eff_perm)

            _, _perm_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('permutation', 'gev_threshold', n_genes_perm_total if 'n_genes_perm_total' in locals() else 0,
                          time.perf_counter() - _t_perm, _perm_heap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_perm))
            _mark_done(py_dir, str(sec))

    # ── eGene calling (after S5 — genome-wide threshold from threshold.csv)
    sec = 6
    eqtl_summary_path = py_dir / 'eqtl_summary.csv'
    if peak_lod_path.exists() and not _checkpoint(py_dir, str(sec), force):
        log.info('\n── eGene calling ────────────────────────────────────────────')
        peak_py_loaded = pd.read_csv(peak_lod_path)
        py_thr_path = core_dir / f'threshold_{_run_sfx}.csv'
        thr: 'float | None' = None
        if py_thr_path.exists():
            _thr_df  = pd.read_csv(py_thr_path)
            _thr_map = {str(r['chr']): float(r['threshold']) for _, r in _thr_df.iterrows()}
            # genome-wide sentinel: one threshold applied to all chromosomes
            _gw_thr_val = _thr_map.get('genome_wide')
            thr = _gw_thr_val if _gw_thr_val is not None else _thr_map.get(str(peak_chr))
            if thr is not None:
                eff_path = core_dir / f'allele_effects_{_run_sfx}.csv'
                eff_tmp: 'pd.DataFrame | None' = None
                if eff_path.exists():
                    eff_tmp = pd.read_csv(
                        eff_path,
                        usecols=['gene_symbol', 'chr', 'peak_locus', 'peak_mb', 'peak_lod', 'h2'])
                _frames: 'list[pd.DataFrame]' = []
                for _ch, _ch_df in peak_py_loaded.groupby(
                        peak_py_loaded['chr'].astype(str)):
                    _ch_eff = (eff_tmp[eff_tmp['chr'].astype(str) == str(_ch)]
                               if eff_tmp is not None else None)
                    # R uses strict `lod > thr`; pre-filter to match before call_egenes (>=)
                    _ch_df_strict = _ch_df[_ch_df['peak_lod'] > thr]
                    _frames.append(call_egenes(_ch_df_strict, thr, effects_df=_ch_eff))
                egenes_df = (pd.concat(_frames, ignore_index=True)
                             if _frames else pd.DataFrame())
                log.info('[eGenes] genome-wide threshold=%.3f → %d / %d genes called',
                         thr, len(egenes_df), len(peak_py_loaded))

            # ── reload perm nulls from CSV when running from checkpoint ─────────
            if perm_max_lods is None:
                _pml_path = core_dir / 'perm_max_lods.csv'
                if _pml_path.exists():
                    perm_max_lods = pd.read_csv(_pml_path)['max_lod'].values
                    perm_n = len(perm_max_lods)
                    # load all per-chromosome nulls saved during S5 run
                    chr_null_lods = {}
                    for _f in sorted(
                        core_dir.glob('perm_max_lods_chr*.csv'),
                        key=lambda p: _chr_sort_key(
                            p.stem.replace('perm_max_lods_chr', '')
                        ),
                    ):
                        _ch_key = _f.stem.replace('perm_max_lods_chr', '')
                        chr_null_lods[_ch_key] = pd.read_csv(_f)['max_lod'].values
                    if not chr_null_lods:
                        chr_null_lods = {peak_chr: perm_max_lods}

            # ── Storey q-value: per-gene Beta calibration ─────────────────────
            # Each eGene that cleared the Stage-1 FWER gate gets its own
            # parametric null drawn at its own h² (calibrate_gene_pvalue),
            # then Storey's method estimates pi0 and q-values across all eGenes.
            # kinship_loco is recomputed here because only the decomposed form is
            # persisted to the .npz checkpoint; calc_kinship is fast.
            if egenes_df is not None and len(egenes_df) > 0 and 'gene_id' in egenes_df.columns:
                log.info('[Storey] calibrating per-gene p-values for %d eGenes...', len(egenes_df))
                _kin_raw = calc_kinship(cache, method='loco')
                _h5_path_st = data_dir / 'counts_lognorm.h5'
                _pval_perm: dict = {}
                for _, _eg in egenes_df.iterrows():
                    _gid     = str(_eg['gene_id'])
                    _chrom   = str(_eg['chr'])
                    _plod    = float(_eg['peak_lod'])
                    _pos_mb  = float(_eg['peak_mb']) if pd.notna(_eg.get('peak_mb')) else None
                    _cts, _lds, _sids = _load_expression(_h5_path_st, [_gid])
                    if not _lds:
                        log.debug('[Storey] skip %s — not in h5', _gid)
                        continue
                    _y   = _cts[0].astype(float)
                    _gds = [meta_geno_map.get(s, s) for s in _sids]
                    _cov = _build_covariates(meta, _sids)
                    _z, _ = build_z_matrix(_gds, list(cache.sample_ids))
                    _r = _z.sum(axis=0)
                    _y_g = (_z.T @ _y) / _r
                    _cov_g = (_z.T @ _cov) / _r[:, None]
                    _p, _, _, _ = calibrate_gene_pvalue(
                        y_gene=_y_g,
                        genomecache=cache,
                        chrom=_chrom,
                        peak_lod=_plod,
                        covariates=_cov_g,
                        kinship_decomp=kinship_decomp,
                        kinship_loco=_kin_raw,
                        n_perms=200,
                        gene_pos_mb=_pos_mb,
                        window_mb=window_mb if scan_type == 'L' else None,
                        use_rint=True,
                    )
                    _pval_perm[_gid] = _p
                    log.debug('[Storey] %s  chr%s  peak_lod=%.3f  p_adj=%.4g',
                              _gid, _chrom, _plod, _p)
                del _kin_raw
                import gc as _gc; _gc.collect()
                if _pval_perm:
                    egenes_df = calibrate_egenes_storey(egenes_df, _pval_perm, fdr_threshold=0.05)
                    log.info('[Storey] done  significant(q<=0.05)=%d / %d',
                             int((egenes_df['significance'] == 'T').sum()), len(egenes_df))

            if perm_max_lods is not None:
                egenes_df['perm_n']   = perm_n
                egenes_df['perm_chr'] = peak_chr

            # GxE/PxG disabled — not accurate until per-gene permutation null
            # is available (S5 currently shares one representative gene's h²
            # across all eGenes, making per-gene interaction p-values unreliable)

            # ── eqtl_summary CSV ───────────────────────────────────────────────
            eqtl_sum = egenes_df.copy()
            # Derive R eGenes from peak_lod.csv + thresholds_genome.csv so the
            # comparison always uses the most recent R scan output, not a stale
            # egenes.csv from a previous run.
            r_peak_lod_path   = r_data / 'peak_lod.csv'
            r_thr_genome_path = r_data / 'thresholds_genome.csv'
            egenes_r = None
            if r_peak_lod_path.exists() and r_thr_genome_path.exists():
                _r_pk  = pd.read_csv(r_peak_lod_path)
                if 'gene' in _r_pk.columns:
                    _r_pk = _r_pk.rename(columns={'gene': 'gene_symbol'})
                _r_thr = pd.read_csv(r_thr_genome_path)
                # column may be threshold_gev or threshold
                _thr_col = next((c for c in ('threshold_gev', 'threshold')
                                 if c in _r_thr.columns), None)
                if _thr_col is not None:
                    _r_gev = float(_r_thr[_thr_col].iloc[0])
                    # peak_lod.csv column may be lod or peak_lod
                    _lod_col = 'peak_lod' if 'peak_lod' in _r_pk.columns else 'lod'
                    egenes_r = _r_pk[_r_pk[_lod_col] >= _r_gev][['gene_symbol', _lod_col]].rename(
                        columns={_lod_col: 'lod'})
                    log.info('[eGene R]  %d R eGenes at threshold=%.3f (from peak_lod.csv)',
                             len(egenes_r), _r_gev)
            if egenes_r is not None:
                eqtl_sum['in_r_egenes'] = eqtl_sum['gene_symbol'].isin(egenes_r['gene_symbol'])
                # ── R vs Python eGene LOD comparison ──────────────────────────
                _cmp = egenes_r.merge(
                    eqtl_sum[['gene_symbol', 'peak_lod']].rename(columns={'peak_lod': 'lod_py'}),
                    on='gene_symbol', how='inner',
                ).rename(columns={'lod': 'lod_r'})
                if len(_cmp) > 0:
                    _ratio       = _cmp['lod_r'] / _cmp['lod_py'].replace(0, np.nan)
                    _med_ratio   = float(_ratio.median())
                    _thr3        = (_ratio > 3).sum()
                    _thr2        = (_ratio > 2).sum()
                    _thr_half    = (_ratio < 0.5).sum()
                    _n_shared    = len(_cmp)
                    _n_only_r    = len(egenes_r) - _n_shared
                    _n_only_py   = len(eqtl_sum) - _n_shared
                    log.info(
                        '[eGene R/Py]  shared=%d  R-only=%d  Py-only=%d',
                        _n_shared, _n_only_r, _n_only_py,
                    )
                    log.info(
                        '[eGene R/Py]  median LOD ratio R/Py=%.3f  '
                        'outliers(>3×)=%d  elevated(>2×)=%d  suppressed(<0.5×)=%d',
                        _med_ratio, _thr3, _thr2, _thr_half,
                    )
                    _out = _cmp[_ratio > 3].sort_values('lod_r', ascending=False)
                    for _, _row in _out.iterrows():
                        log.info(
                            '[eGene R/Py]    outlier  %s  R=%.2f  Py=%.2f  ratio=%.1f×',
                            _row['gene_symbol'], _row['lod_r'], _row['lod_py'],
                            _row['lod_r'] / _row['lod_py'],
                        )
            # h2, start_mb, near_tss_mb, cis_dist_mb already present from peak_lod.csv
            # gene_id join + chi2/p_haplotype + canonical col_order via package function
            if 'gene_id' not in eqtl_sum.columns and 'gene_id' in gene_info.columns:
                _gid_map = gene_info.drop_duplicates('gene_name').set_index('gene_name')['gene_id']
                eqtl_sum.insert(1, 'gene_id', eqtl_sum['gene_symbol'].map(_gid_map))
            eqtl_sum = finalize_egene_summary(eqtl_sum)
            eqtl_sum.to_csv(eqtl_summary_path, index=False)
            log.info('[eqtl_summary] %d eGenes → eqtl_summary.csv', len(eqtl_sum))
            _mark_done(py_dir, str(sec))
        else:
            log.info('[eGenes] no threshold available — run S5 first')
    elif eqtl_summary_path.exists():
        egenes_df = pd.read_csv(eqtl_summary_path)
        log.info('[eGenes] loaded %d eGenes from checkpoint (%s)',
                 len(egenes_df), eqtl_summary_path.name)

    # gene_meta passed to all streaming scan functions (S7–S11)
    _gene_meta: 'pd.DataFrame | None' = None
    _peak_mb_map: 'dict[str, float]' = {}
    if egenes_df is not None and len(egenes_df) > 0:
        _gm_cols = [c for c in ['gene_id', 'gene_symbol', 'near_tss_mb', 'cis_dist_mb', 'peak_lod', 'peak_mb']
                    if c in egenes_df.columns]
        _gene_meta = (egenes_df[_gm_cols]
                      .drop_duplicates('gene_id')
                      .rename(columns={'peak_lod': 'peak_lod_lmm', 'peak_mb': 'peak_mb_lmm'})
                      .copy())
        _peak_mb_map = {
            str(r['gene_id']): float(r['peak_mb_lmm'])
            for _, r in _gene_meta.iterrows()
            if pd.notna(r.get('peak_mb_lmm'))
        }

    # MI scan
    sec = 7
    r_mi_path = r_data / 'mi_lod_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_mi = _get_rss_mb()
            tracemalloc.start()
            _t_mi = time.perf_counter()
            _n_imp_mi = 11
            log.info('\nMI scan: LOD averaged over %d hard founder imputations to integrate haplotype assignment uncertainty...',
                     _n_imp_mi)
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['mi_scan'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _mi_ids = [r['gene_id']
                           for _, r in egenes_df.iterrows()]
                _mi_counts, _mi_loaded, _mi_sids = _load_expression(h5_path, _mi_ids)
                _mi_expr = {gid: _mi_counts[i].astype(float)
                            for i, gid in enumerate(_mi_loaded)}
                _mi_chr_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene_id']
                    _mi_chr_map[gid] = str(erow.get('chr', peak_chr))
                geno_ids_mi = [meta_geno_map.get(s, s) for s in _mi_sids]
                cov_mi = _build_covariates(meta, _mi_sids)
                _mi_out_path = ext_dir / 'mi_lod.csv'
                log.info('  scanning %d eGenes via MI (streaming)...', len(_mi_expr))
                mi_df = scan_multi_gene_mi(
                    _mi_expr, _mi_chr_map, cache,
                    covariates=cov_mi,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_mi,
                    n_imp=_n_imp_mi, seed=1,
                    peak_mb_map=_peak_mb_map,
                    output_path=_mi_out_path,
                    gene_meta=_gene_meta,
                )
                del _mi_expr, _mi_counts
                if not mi_df.empty:
                    log.info('  MI scan complete: %d genes', mi_df['gene_symbol'].nunique())
                    results['mi_scan'] = True
                    if r_mi_path.exists():
                        r_mi = pd.read_csv(r_mi_path)
                        if 'gene' in r_mi.columns:
                            r_mi = r_mi.rename(columns={'gene': 'gene_symbol'})
                        # peak_only=True → one row per gene; compare peak LOD per gene
                        r_mi_peak = (r_mi.groupby('gene_symbol')['lod'].max()
                                     .reset_index().rename(columns={'lod': 'r'}))
                        merged_mi = (mi_df[['gene_symbol', 'peak_lod']]
                                     .rename(columns={'peak_lod': 'py'})
                                     .merge(r_mi_peak, on='gene_symbol', how='inner'))
                        if len(merged_mi) >= 10:
                            r_mi_r, _ = sp_stats.pearsonr(merged_mi['py'],
                                                           merged_mi['r'])
                            merged_mi.to_csv(cmp_dir / 'mi_cmp.csv', index=False)
                            _check('MI LOD r', abs(r_mi_r), 0.75)
                            log.info('  vs R: r=%.4f  n=%d', r_mi_r, len(merged_mi))
                            fig, _ax = _cmp_scatter(
                                merged_mi, 'r', 'py',
                                'MI LOD (R)', 'MI LOD (Python)',
                                save_path=plt_dir / f'mi_scatter_{_run_sfx}.png',
                            )
                            plt.close(fig)
                else:
                    results['mi_scan'] = None
            _, _mi_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('mi_scan', 'rubin_average', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_mi, _mi_heap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_mi))
            _mark_done(py_dir, str(sec))

    # SNP scan
    # mediation scan
    sec = 9
    r_med_path = r_data / 'mediation_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_med = _get_rss_mb()
            tracemalloc.start()
            _t_med = time.perf_counter()
            log.info('\nmediation scan: QR-based test whether cis-eGenes mediate each other\'s haplotype QTL signal...')
            if egenes_df is None or len(egenes_df) < 2:
                log.info('  %s need >=2 eGenes — run S4+S5 first', SKIP)
                results['mediation_r'] = None
            elif 'chr' not in egenes_df.columns or 'peak_mb' not in egenes_df.columns:
                log.info('  %s egenes_df missing chr/peak_mb — run S4 first to build effects', SKIP)
                results['mediation_r'] = None
            else:
                h5_path  = data_dir / 'counts_lognorm.h5'
                # restrict mediator pool to eGenes: a valid mediator must itself be
                # haplotype-regulated — non-eGenes have no haplotype effect to mediate
                egene_ids_set = set(egenes_df['gene_id'].dropna())
                gene_info_egenes = gene_info[gene_info['gene_id'].isin(egene_ids_set)]
                pairs    = select_cis_mediation_pairs(
                    egenes_df, gene_info_egenes,
                    window_mb=10.0,
                    gene_col='gene_id',
                    info_gene_col='gene_id',
                )
                log.info('  %d (outcome, mediator) pairs from eGene proximity filter', len(pairs))
                # group by (outcome, chr) — select_cis_mediation_pairs guarantees
                # all mediators for one outcome share the same chr
                _pairs_by_outcome: dict[str, list[tuple[str, str]]] = {}
                for _on, _mn, _ch in pairs:
                    _pairs_by_outcome.setdefault(_on, []).append((_mn, _ch))

                # Batch-load ALL outcome + mediator expressions in ONE HDF5 call.
                # Without this, per-mediator _load_expression calls (up to 90+ per
                # outcome × N outcomes) each open the file, scan 43k gene IDs, and
                # decompress 9 column chunks — adding 10–20 min to S9 total runtime.
                _med_all_names: set[str] = set(_pairs_by_outcome.keys())
                for _ml in _pairs_by_outcome.values():
                    _med_all_names.update(_mn for _mn, _ in _ml)
                _med_all_ids = list(dict.fromkeys(_med_all_names))
                _med_counts, _med_loaded, _med_sids = _load_expression(h5_path, _med_all_ids)
                _med_expr = {gid: _med_counts[i].astype(float)
                             for i, gid in enumerate(_med_loaded)}
                del _med_counts
                log.info('  pre-loaded %d gene expressions (%d unique IDs) in 1 HDF5 call',
                         len(_med_loaded), len(_med_all_ids))

                geno_ids_med = [meta_geno_map.get(s, s) for s in _med_sids]
                cov_med = _build_covariates(meta, _med_sids)
                _med_out_path = ext_dir / 'mediation.csv'
                log.info('  running mediation for %d outcomes (streaming)...',
                         len(_pairs_by_outcome))
                med_df = mediation_scan_multi_outcome(
                    _pairs_by_outcome, _med_expr, cache,
                    covariates=cov_med,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_med,
                    output_path=_med_out_path,
                    gene_meta=_gene_meta,
                )
                del _med_expr
                if not med_df.empty:
                    log.info('  mediation complete: %d genes', med_df['gene_symbol'].nunique())
                    results['mediation_r'] = True
                    if r_med_path.exists():
                        r_med = pd.read_csv(r_med_path)
                        if 'gene' in r_med.columns:
                            r_med = r_med.rename(columns={'gene': 'gene_symbol'})
                        # miQTL mediation.scan.qr may export 'LOD' (uppercase) and
                        # 'pos.Mb' — normalize to lowercase snake_case for _pos_merge
                        if 'LOD' in r_med.columns and 'lod' not in r_med.columns:
                            r_med = r_med.rename(columns={'LOD': 'lod'})
                        if 'pos.Mb' in r_med.columns and 'pos_mb' not in r_med.columns:
                            r_med = r_med.rename(columns={'pos.Mb': 'pos_mb'})
                        if 'lod_mediated' in r_med.columns and 'lod' not in r_med.columns:
                            r_med = r_med.rename(columns={'lod_mediated': 'lod'})
                        merged_med = _pos_merge(med_df, r_med, 'lod_mediated', 'lod',
                                                left_label='py', right_label='r')
                        if len(merged_med) >= 10:
                            r_m, _ = sp_stats.pearsonr(merged_med['py'],
                                                        merged_med['r'])
                            merged_med.to_csv(cmp_dir / 'mediation_cmp.csv',
                                              index=False)
                            _check('mediation LOD r', abs(r_m), 0.90)
                            log.info('  vs R: r=%.4f  n=%d', r_m, len(merged_med))
                            fig, _ax = _cmp_scatter(
                                merged_med, 'r', 'py',
                                'Mediated LOD (R)', 'Mediated LOD (Python)',
                                save_path=plt_dir / f'mediation_scatter_{_run_sfx}.png',
                            )
                            plt.close(fig)
                else:
                    results['mediation_r'] = None
            _, _med_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('mediation', 'qr_mediation', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_med, _med_heap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_med))
            _mark_done(py_dir, str(sec))

    # bootstrap CI
    sec = 10
    r_ci_path = r_data / 'ci_summary_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_ci = _get_rss_mb()
            tracemalloc.start()
            _t_ci = time.perf_counter()
            log.info('\nbootstrap CI: phenotype-resampling positional confidence intervals around eGene peak loci...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['bootstrap_ci'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _ci_ids = [r['gene_id']
                           for _, r in egenes_df.iterrows()]
                _ci_counts, _ci_loaded, _ci_sids = _load_expression(h5_path, _ci_ids)
                _ci_expr = {gid: _ci_counts[i].astype(float)
                            for i, gid in enumerate(_ci_loaded)}
                del _ci_counts
                _ci_chr_map = {}
                _ci_peak_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene_id']
                    _ci_chr_map[gid] = str(erow.get('chr', peak_chr))
                    _ci_peak_map[gid] = float(erow.get('peak_mb', np.nan))
                geno_ids_ci = [meta_geno_map.get(s, s) for s in _ci_sids]
                cov_ci = _build_covariates(meta, _ci_sids)
                _ci_out_path = ext_dir / 'ci.csv'
                log.info('  1.5-LOD support CI for %d eGenes (streaming)...', len(_ci_expr))
                ci_df = ci_scan_multi_gene(
                    _ci_expr, _ci_chr_map, _ci_peak_map, cache,
                    covariates=cov_ci,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_ci,
                    output_path=_ci_out_path,
                    gene_meta=_gene_meta,
                )
                del _ci_expr
                if not ci_df.empty:
                    log.info('  CI complete: %d genes  median width=%.2f Mb',
                             len(ci_df),
                             float(np.median(ci_df['ci_width_mb'])))
                    results['bootstrap_ci'] = True
                    if r_ci_path.exists():
                        r_ci = pd.read_csv(r_ci_path)
                        if 'gene' in r_ci.columns:
                            r_ci = r_ci.rename(columns={'gene': 'gene_symbol'})
                        merged_ci = ci_df.merge(r_ci, on='gene_symbol', how='inner',
                                                suffixes=('_py', '_r'))
                        if len(merged_ci) > 0:
                            # Python cols: ci_lo, ci_hi; R cols: ci_lo_mb, ci_hi_mb
                            # No duplicate names → no suffix applied; reference directly
                            overlap = (
                                (merged_ci['ci_lo'] <= merged_ci['ci_hi_mb']) &
                                (merged_ci['ci_hi'] >= merged_ci['ci_lo_mb'])
                            ).mean()
                            merged_ci.to_csv(cmp_dir / 'ci_cmp.csv', index=False)
                            log.info('  vs R: CI overlap=%.2f  n=%d',
                                     overlap, len(merged_ci))
                else:
                    results['bootstrap_ci'] = None
            _, _ci_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('bootstrap_ci', 'positional_resample', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_ci, _ci_heap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_ci))
            _mark_done(py_dir, str(sec))

    # Sanger fine-mapping (requires bootstrap CI outputs + --sanger-alleles HDF5)
    sec = 12
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_fm = _get_rss_mb()
            tracemalloc.start()
            _t_fm = time.perf_counter()
            log.info('\nSanger fine-mapping: ABF credible sets on CI-window SNPs...')
            _eqtl_peak_csv = py_dir / f'eqtl_peak_{_run_sfx}.csv'
            _sanger_h5 = Path(sanger_alleles) if sanger_alleles else None
            if _sanger_h5 is None or not _sanger_h5.exists():
                log.info('  %s --sanger-alleles not set or file missing', SKIP)
                results['sanger_finemap'] = None
            elif not _eqtl_peak_csv.exists():
                log.info('  %s eqtl_peak_%s.csv not found — run report first', SKIP, _run_sfx)
                results['sanger_finemap'] = None
            elif egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes', SKIP)
                results['sanger_finemap'] = None
            else:
                egene_peak_df = pd.read_csv(_eqtl_peak_csv, dtype={'chr': str})
                _required = {'gene_id', 'chr', 'pos', 'lod', 'variant', 'ci_lo_bp', 'ci_hi_bp'}
                if not _required.issubset(egene_peak_df.columns):
                    log.warning('  %s eqtl_peak CSV missing columns: %s', SKIP,
                                _required - set(egene_peak_df.columns))
                    results['sanger_finemap'] = None
                else:
                    h5_path = data_dir / 'counts_lognorm.h5'
                    _fm_ids = list(egene_peak_df['gene_id'])
                    _fm_counts, _fm_loaded, _fm_sids = _load_expression(h5_path, _fm_ids)
                    geno_ids_fm = [meta_geno_map.get(s, s) for s in _fm_sids]
                    cov_fm = _build_covariates(meta, _fm_sids)
                    fm_dir = py_dir / 'finemap'
                    fm_df, fm_peak_df = finemap_egenes(
                        egene_df=egene_peak_df,
                        phenotype_matrix=_fm_counts,
                        gene_ids=list(_fm_loaded),
                        genomecache=cache,
                        alleles_h5=str(_sanger_h5),
                        out_dir=fm_dir,
                        covariates=cov_fm,
                        kinship_decomp=kinship_decomp,
                        tag=_run_sfx,
                    )
                    del _fm_counts
                    results['sanger_finemap'] = (
                        fm_df is not None and len(fm_df) > 0
                    )
                    if results['sanger_finemap']:
                        log.info('  fine-mapped %d eGenes → %s',
                                 fm_peak_df['gene_id'].nunique() if fm_peak_df is not None else 0,
                                 fm_dir)
            _, _fm_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('sanger_finemap', 'abf_credible_sets',
                          len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_fm, _fm_heap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_fm))
            _mark_done(py_dir, str(sec))

    # instability scan
    sec = 11
    r_inst_path = r_data / 'instability_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_inst = _get_rss_mb()
            tracemalloc.start()
            _t_inst = time.perf_counter()
            log.info('\ninstability scan: subsampling stability of eGene peak loci across bootstrap resamples...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['instability'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _inst_ids = [r['gene_id']
                             for _, r in egenes_df.iterrows()]
                _inst_counts, _inst_loaded, _inst_sids = _load_expression(h5_path, _inst_ids)
                _inst_expr = {gid: _inst_counts[i].astype(float)
                              for i, gid in enumerate(_inst_loaded)}
                del _inst_counts
                _inst_chr_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene_id']
                    _inst_chr_map[gid] = str(erow.get('chr', peak_chr))
                geno_ids_inst = [meta_geno_map.get(s, s) for s in _inst_sids]
                cov_inst = _build_covariates(meta, _inst_sids)
                _inst_out_path = ext_dir / 'instability.csv'
                log.info('  instability scan for %d eGenes (streaming)...', len(_inst_expr))
                inst_df = instability_scan_multi_gene(
                    _inst_expr, _inst_chr_map, cache,
                    covariates=cov_inst,
                    sample_ids=geno_ids_inst,
                    n_subsamples=100, seed=1,
                    peak_mb_map=_peak_mb_map,
                    output_path=_inst_out_path,
                    gene_meta=_gene_meta,
                )
                del _inst_expr
                if not inst_df.empty:
                    log.info('  instability complete: %d genes', inst_df['gene_symbol'].nunique())
                    results['instability'] = True
                    if r_inst_path.exists():
                        r_inst = pd.read_csv(r_inst_path)
                        if 'gene' in r_inst.columns:
                            r_inst = r_inst.rename(columns={'gene': 'gene_symbol'})
                        # peak_only=True → one row per gene; compare peak instability per gene
                        # R: neg_log10_p (median -log10 F-test p); Python: peak_lod
                        # Both monotone with instability signal — use Spearman (scale-invariant)
                        r_inst_peak = (r_inst.groupby('gene_symbol')['neg_log10_p'].max()
                                       .reset_index().rename(columns={'neg_log10_p': 'r'}))
                        merged_inst = (inst_df[['gene_symbol', 'peak_lod']]
                                       .rename(columns={'peak_lod': 'py'})
                                       .merge(r_inst_peak, on='gene_symbol', how='inner'))
                        if len(merged_inst) >= 10:
                            r_i, _ = sp_stats.spearmanr(merged_inst['py'],
                                                          merged_inst['r'])
                            merged_inst.to_csv(cmp_dir / 'instability_cmp.csv',
                                               index=False)
                            _check('instability rho', abs(r_i), 0.90)
                            log.info('  vs R (Spearman): rho=%.4f  n=%d'
                                     '  [py=peak_lod, r=peak neg_log10_p — rank correlation]',
                                     r_i, len(merged_inst))
                            fig, _ax = _cmp_scatter(
                                merged_inst, 'r', 'py',
                                'Instability −log₁₀p (R)', 'Instability LOD (Python)',
                                identity_line=False,
                                save_path=plt_dir / f'instability_scatter_{_run_sfx}.png',
                            )
                            plt.close(fig)
                else:
                    results['instability'] = None
            _, _inst_heap_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('instability', 'subsample_scan', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_inst, _inst_heap_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_inst))
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # gate summary + run_summary.csv
    # ─────────────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - _t0
    log.info('\n── GATE SUMMARY ──────────────────────────────────────────────')

    gate_rows = []
    gate_labels = {
        'cache_structure':          'cache characterization',
        'haplotype_probs':      'haplotype probabilities',
        'loco_kinship':        'LOCO kinship',
        'lmm_lod_r2':         f'{scan_type} peak LOD r²  (PRIMARY GATE)',
        'perm_threshold': 'permutation threshold',
        'mi_scan':        'MI scan',
        'sanger_finemap':  'Sanger fine-mapping',
        'mediation_r':      'mediation scan',
        'bootstrap_ci':            'bootstrap CI',
        'instability':   'instability scan',
    }
    for key, label in gate_labels.items():
        val = results.get(key)
        if val is None:
            sym = SKIP
        elif val:
            sym = PASS
        else:
            sym = FAIL
        log.info('  %s %s', sym, label)
        gate_rows.append({'gate': key, 'label': label,
                          'result': str(val) if val is not None else 'skip'})

    pd.DataFrame(gate_rows).to_csv(cache_dir / 'gate_summary.csv', index=False)

    n_egenes = len(egenes_df) if egenes_df is not None else 0
    n_genes_scanned = (len(pd.read_csv(peak_lod_path))
                       if peak_lod_path.exists() else 0)
    run_summary = {
        'n_genes_scanned':  n_genes_scanned,
        'n_egenes':         n_egenes,
        'scan_type':        scan_type,
        'scan_chr':         ','.join(chrs_to_scan),
        'peak_chr':         peak_chr,
        'window_mb':        window_mb if scan_type == 'L' else None,
        'total_elapsed_s':  round(total_elapsed, 1),
        'lmm_lod_r2':       results.get('lmm_lod_r2'),
        'timestamp':        pd.Timestamp.now().isoformat(),
    }
    pd.DataFrame([run_summary]).to_csv(py_dir / 'run_summary.csv', index=False)
    log.info('  run_summary.csv written (%d eGenes / %d scanned)',
             n_egenes, n_genes_scanned)

    # flush any accumulated section stats (S1-S4) — idempotent if already written after S3
    if _scan_stats:
        _sec_ord = {s: i for i, s in enumerate(_SECTION_ORDER)}
        _sorted_stats = sorted(_scan_stats,
                               key=lambda r: _sec_ord.get(r['section'], 999))
        pd.DataFrame(_sorted_stats).to_csv(py_dir / 'scan_stats.csv', index=False)

    # ── descriptive plots (always, no R needed) ───────────────────────────────
    log.info('\n── DESCRIPTIVE PLOTS ─────────────────────────────────────────')
    try:
        _make_descriptive_plots(
            plt_dir=plt_dir, py_dir=py_dir, r_data=r_data,
            cache=cache, kinship_decomp=kinship_decomp,
            founders=list(cache.founders), peak_chr=peak_chr,
            scan_tag=_stag, celltype=celltype,
            priority_gene=PRIORITY_GENE, low_expr_genes=low_expr_genes,
            gene_info=gene_info, gtf_path=gtf_path,
        )
    except Exception as exc:
        log.warning('  descriptive plots raised: %s', exc)

    # ── comparison plots (R reference present) ───────────────────────────────────
    if any((r_data / f).exists() for f in
           ['cache_info.csv', 'haplotype_probs.csv', 'kinship_summary.csv']):
        log.info('\n── COMPARISON PLOTS ──────────────────────────────────────────')
        try:
            _make_comparison_plots(
                plt_dir=plt_dir, cmp_dir=cmp_dir, py_dir=py_dir, r_data=r_data,
                cache=cache, kinship_decomp=kinship_decomp,
                founders=list(cache.founders), peak_chr=peak_chr,
                scan_tag=_stag, celltype=celltype,
                low_expr_genes=low_expr_genes,
            )
        except Exception as exc:
            log.warning('  comparison plots raised: %s', exc)

    # ── detailed scan summary ─────────────────────────────────────────────────
    n_loci_total = sum(
        len(cache.get_chromosome(c)[0])
        for c in chrs_to_scan
    ) if chrs_to_scan else 0
    lmm_lod_r2  = results.get('lmm_lod_r2')
    gpm_val = None
    if _scan_stats:
        scan_stat_row = next((r for r in _scan_stats if r['section'] == 'lmm_scan'), None)
        if scan_stat_row and scan_stat_row['elapsed_s'] > 0:
            gpm_val = scan_stat_row['n_items'] / scan_stat_row['elapsed_s'] * 60
    log.info('\n── SCAN SUMMARY ──────────────────────────────────────────────')
    log.info('  genes scanned    : %d',  n_genes_scanned)
    log.info('  loci evaluated   : %d',  n_loci_total)
    log.info('  eGenes called    : %d',  n_egenes)
    log.info('  scan type        : %s',  scan_type)
    log.info('  peak chr         : %s',  peak_chr)
    log.info('  LMM scan LOD r²  : %s',  lmm_lod_r2 if lmm_lod_r2 is not None else 'N/A')
    log.info('  throughput (py)  : %s',
             f'{gpm_val:.1f} genes/min' if gpm_val else 'N/A')
    r_timing = r_data / 'timing_chr.csv'
    if r_timing.exists():
        rt  = pd.read_csv(r_timing)
        if 'genes_per_min' in rt.columns:
            r_gpm = rt['genes_per_min'].mean()
        elif 'elapsed_s' in rt.columns and 'n_genes' in rt.columns:
            r_gpm = (rt['n_genes'].sum() / rt['elapsed_s'].sum() * 60
                     if rt['elapsed_s'].sum() > 0 else None)
        else:
            r_gpm = None
        if r_gpm is not None:
            speedup = gpm_val / r_gpm if (gpm_val and r_gpm > 0) else None
            log.info('  throughput (R)   : %.1f genes/min', r_gpm)
            if speedup is not None:
                log.info('  speedup vs R     : %.1f×', speedup)

    log.info('── total elapsed: %.0fs ─────────────────────────────────────',
             total_elapsed)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description='validate mosaiqtl on real pseudobulk data for any tissue')
    ap.add_argument('--tissue', required=True,
                    help='tissue identifier used for file naming, e.g. gastroc, ctxhipp')
    ap.add_argument('--data', default=None,
                    help='tissue data root directory containing metadata.csv, '
                         'exp_gene_info_{tissue}.csv, counts_lognorm.h5 (or celltypes/); '
                         'defaults to validation/{tissue}_data, or '
                         'validation/{tissue}_data/{dataset} if --dataset is set; '
                         'an explicit --data overrides --dataset')
    ap.add_argument('--dataset', default=None, choices=['next', 'nova'],
                    help='dataset generation subdirectory under validation/{tissue}_data/ '
                         '(default: none — reads/writes directly under {tissue}_data); '
                         'ignored if --data is also given')
    ap.add_argument('--out', default=None,
                    help='whole-tissue mode only (no --celltype): output root — holds '
                         'cache_info/gate_summary/haplotype_probs/kinship_summary.csv plus '
                         'one {scan_tag}/ subdir per scan config — defaults to --data/whole. '
                         'Lets input (--data) and output (--out) diverge, e.g. rerunning a '
                         'scan into a versioned subdirectory without touching the source '
                         'CSVs/HDF5. Ignored in per-cell-type mode, where output already '
                         'lives under --data/celltypes/<ct>.')
    ap.add_argument('--build-cache', action='store_true',
                    help='one-time: read HAPPY genomecache via rpy2 → save '
                         'validation/py_ref/cache.h5, then exit (requires rpy2)')
    ap.add_argument('--genomecache', default=None,
                    help='path to HAPPY genomecache dir for --build-cache '
                         f'(default: {GENOMECACHE})')
    ap.add_argument('--full', action='store_true',
                    help='run all sections (cache, haplotypes, kinship, scan, permutation, '
                         'eGene calling, LOD support CI, Sanger fine-mapping, MI, mediation, instability)')
    ap.add_argument('--scan', action='store_true',
                    help='run core scan sections (cache, haplotypes, kinship, LMM scan, '
                         'permutation, eGene calling; default when no flag given)')
    ap.add_argument('--mi', action='store_true',
                    help='run multiple-imputation scan (requires scan outputs)')
    ap.add_argument('--mediation', action='store_true',
                    help='run mediation scan (requires scan outputs)')
    ap.add_argument('--ci', action='store_true',
                    help='compute 1.5-LOD support intervals for all eGenes (requires scan outputs)')
    ap.add_argument('--finemap', action='store_true',
                    help='run Sanger fine-mapping (requires CI outputs + --sanger-alleles)')
    ap.add_argument('--sanger-alleles', default=None, metavar='PATH',
                    help='path to Sanger alleles HDF5 built by build_founder_allele_mgp()')
    ap.add_argument('--instability', action='store_true',
                    help='run instability scan (requires scan outputs)')
    ap.add_argument('--plots-only', action='store_true',
                    help='skip all scans, regenerate descriptive plots from existing scan output')
    ap.add_argument('--force',    action='store_true',
                    help='ignore checkpoints and rerun all sections')
    ap.add_argument('--n_genes',  type=int, default=None,
                    help='limit cis genes per chr for S4 (testing)')
    ap.add_argument('--scan_type', default='L', choices=['C', 'G', 'L'],
                    help='C = chromosome (each gene vs every locus on its own chr); '
                         'G = genome-wide (all genes vs all chr); '
                         'L = local (cis genes, loci within ±window_mb of TSS, default)')
    ap.add_argument('--window_mb', type=float, default=1.0,
                    help='half-window in Mb for --scan_type L (default 1.0)')
    ap.add_argument('--chr', default='all',
                    help='chromosome(s) to scan: "all" or comma-separated like "8" or '
                         '"1,8,11" (default: all)')
    ap.add_argument('--celltype', type=str, default=None,
                    help='cell type label (e.g. myonuclei); runs scan in '
                         'celltypes/{ct}/ instead of whole-tissue {data}/')
    ap.add_argument('--all-celltypes', action='store_true',
                    help='discover all cell types under {data}/celltypes/, skip '
                         'low-quality variants, and run the scan for each in '
                         'natsorted order; mutually exclusive with --celltype')
    ap.add_argument('--compare-celltypes', action='store_true',
                    help='cross-cell-type comparison: reads per-CT scan outputs and '
                         'produces comparison plots; exits after completion')
    ap.add_argument('--celltypes', nargs='+', default=None, metavar='CT',
                    help='cell-type labels for --compare-celltypes (auto-discovered '
                         'from celltypes/<name>/<scan_tag>/core/peak_lod_<name>_<scan_tag>.csv '
                         'when omitted) or entities for --compare-scan-types (whole-tissue '
                         'when omitted); under --data, or --data/--dataset when '
                         '--dataset is set; scan_tag from --scan_type/--window_mb')
    ap.add_argument('--rep-gene', default='Piezo1',
                    help='representative gene for LOD profile overlay in '
                         '--compare-celltypes / --compare-scan-types (default: Piezo1)')
    ap.add_argument('--compare-scan-types', action='store_true',
                    help='same-entity, cross-scan-config comparison: pairwise compares '
                         'the scan tags in --scan-tags for one whole-tissue run or each '
                         'cell type in --celltypes; exits after completion. Output '
                         'filenames are prefixed {celltype_or_whole}_{tag_x}_vs_{tag_y}_ '
                         'so files stay identifiable outside their originating '
                         'compare_scan_types/ directory')
    ap.add_argument('--scan-tags', default=None, metavar='TAG1,TAG2[,...]',
                    help='comma-separated scan tags to compare with --compare-scan-types '
                         '(e.g. "C,Lw1"); required by --compare-scan-types')
    ap.add_argument('--gtf', default=None, metavar='PATH',
                    help='path to .gtf or .gtf.gz annotation file (e.g. GENCODE vM36); '
                         'when provided, locus panel gene tracks show real '
                         'exon/intron/CDS structure instead of plain gene-body bars')
    args = ap.parse_args()

    # ── resolve tissue data root ──────────────────────────────────────────────
    tissue = args.tissue
    if args.data:
        data_dir = Path(args.data)
    else:
        data_dir = Path(f'validation/{tissue}_data')
        if args.dataset:
            data_dir = data_dir / args.dataset

    # parse scan_type (handle "L:2.5" format) — needed early for --compare-celltypes'
    # scan_tag-scoped path resolution, in addition to run() below
    _scan_type_raw = args.scan_type.upper()
    window_mb      = args.window_mb
    if _scan_type_raw.startswith('L'):
        _parts = _scan_type_raw.split(':')
        if len(_parts) >= 2:
            window_mb = float(_parts[1])
        scan_type = 'L'
    else:
        scan_type = _scan_type_raw   # 'C' or 'G'
    _stag = _scan_tag(scan_type, window_mb)

    # ── --compare-celltypes: cross-CT comparison, exits when done ────────────
    if args.compare_celltypes:
        _celltypes_dir = data_dir / 'celltypes'
        if args.celltypes:
            _cts = [_ct_safe(c) for c in args.celltypes]
        else:
            _cts = _ct_discover(_celltypes_dir, _stag)
        if not _cts:
            log.error('no cell types found under %s (scan_tag=%s)', _celltypes_dir, _stag)
            sys.exit(1)
        log.info('--compare-celltypes  cell_types=%s  rep_gene=%s  scan_tag=%s',
                 _cts, args.rep_gene, _stag)
        _ct_roots      = {ct: _ct_resolve_dir(data_dir, ct, _stag) for ct in _cts}
        # scan CSVs (peak_lod_<ct>_<scan_tag>.csv / allele_effects_<ct>_<scan_tag>.csv /
        # lod_<ct>_<scan_tag>.csv / threshold_<ct>_<scan_tag>.csv — see run()::_sfx) live
        # under <scan_tag>/core/, sibling to plots/ — see resolve_ct_scan_dir()
        _ct_scan_dirs  = {ct: _ct_roots[ct] / 'core' for ct in _cts}
        _ct_plot_dirs  = {ct: _ct_roots[ct] / 'plots' for ct in _cts}
        _compare_out   = data_dir / 'celltypes' / 'compare' / _stag
        # whole-tissue output lives under --out when the scan that produced it was
        # redirected there (--out diverges from --data); defaults to data_dir/'whole'
        _whole_out_dir = Path(args.out) if args.out else (data_dir / 'whole')
        _whole_scan_dir = _whole_out_dir / _stag / 'core'
        for d in list(_ct_plot_dirs.values()) + [_compare_out]:
            d.mkdir(parents=True, exist_ok=True)
        _palette   = _ct_palette(_cts)
        _threshold = _ct_load_threshold(_whole_scan_dir / f'threshold_{_sfx(None, _stag)}.csv')
        _ct_vs_whole_lod(_cts, _ct_scan_dirs, _whole_scan_dir,
                         _ct_plot_dirs, _compare_out, _palette,
                         _threshold, _stag, rep_gene=args.rep_gene)
        _ct_vs_whole_effects(_cts, _ct_scan_dirs, _whole_scan_dir, _ct_plot_dirs, _stag)
        _ct_lod_overlay(_cts, _ct_scan_dirs, _whole_scan_dir,
                        _compare_out, _palette, _threshold, args.rep_gene, _stag)
        _ct_pairwise_lod(_cts, _ct_scan_dirs, _ct_plot_dirs,
                         _compare_out, _threshold, _stag, rep_gene=args.rep_gene)
        _ct_pairwise_effects(_cts, _ct_scan_dirs, _ct_plot_dirs, _stag)
        _ct_lod_distributions(_cts, _ct_scan_dirs, _whole_scan_dir,
                               _compare_out, _palette, _threshold, _stag)
        _ct_cis_distance(_cts, _ct_scan_dirs, _whole_scan_dir, _compare_out, _palette, _stag)
        log.info('--compare-celltypes done  outputs → %s', _compare_out)
        sys.exit(0)

    # ── --compare-scan-types: same entity, different scan configs, exits when done ──
    if args.compare_scan_types:
        if not args.scan_tags:
            log.error('--compare-scan-types requires --scan-tags TAG1,TAG2[,...]')
            sys.exit(1)
        _tags = [t.strip() for t in args.scan_tags.split(',') if t.strip()]
        if len(_tags) < 2:
            log.error('--scan-tags needs >=2 comma-separated tags, got: %r', args.scan_tags)
            sys.exit(1)
        _whole_out_dir = Path(args.out) if args.out else (data_dir / 'whole')
        if args.celltypes:
            _entities = [_ct_safe(c) for c in args.celltypes]
            _entity_scan_dirs = {
                ent: {tag: _ct_resolve_dir(data_dir, ent, tag) / 'core' for tag in _tags}
                for ent in _entities
            }
            _entity_out_dirs = {
                ent: _ct_resolve_dir(data_dir, ent, _tags[0]).parent / 'compare_scan_types'
                for ent in _entities
            }
        else:
            _entities = ['whole']
            _entity_scan_dirs = {
                'whole': {tag: _whole_out_dir / tag / 'core' for tag in _tags}
            }
            _entity_out_dirs = {'whole': _whole_out_dir / 'compare_scan_types'}
        log.info('--compare-scan-types  entities=%s  scan_tags=%s  rep_gene=%s',
                 _entities, _tags, args.rep_gene)
        for ent in _entities:
            _out = _entity_out_dirs[ent]
            _out.mkdir(parents=True, exist_ok=True)
            _dirs = _entity_scan_dirs[ent]
            _threshold = _ct_load_threshold(_dirs[_tags[0]] / 'threshold.csv')
            _compare_scan_types(ent, _tags, _dirs, _out,
                                 threshold=_threshold, rep_gene=args.rep_gene)
            log.info('--compare-scan-types %s done  outputs → %s', ent, _out)
        sys.exit(0)

    # ── --build-cache: one-time rpy2 step, exits when done ────────────────────
    if args.build_cache:
        gc_dir  = Path(args.genomecache) if args.genomecache else GENOMECACHE
        out_h5  = PY_REF / 'cache.h5'
        if not gc_dir.exists():
            log.error('genomecache not found: %s', gc_dir)
            sys.exit(1)
        PY_REF.mkdir(parents=True, exist_ok=True)
        log.info('--build-cache  genomecache=%s  → %s', gc_dir, out_h5)
        _build_cache_from_happy(gc_dir, out_h5)
        log.info('done — run validate_tissue.py --tissue %s without --build-cache to validate',
                 tissue)
        sys.exit(0)

    if args.plots_only:
        sections: set[int] = set()
    elif not any([args.full, args.scan, args.mi,
                  args.mediation, args.ci, args.finemap, args.instability]):
        sections = set(range(1, 7))    # default: --scan (cache → egene calling)
    else:
        sections = set()
        if args.full:
            sections = set(range(1, 13))
        else:
            if args.scan:
                sections.update(range(1, 7))
            if args.mi:
                sections.add(7)
            if args.mediation:
                sections.add(9)
            if args.ci:
                sections.add(10)
            if args.finemap:
                sections.add(12)
            if args.instability:
                sections.add(11)

    # scan_type/window_mb (and _stag) already parsed above, before --compare-celltypes

    # parse --chr
    if args.chr.lower() == 'all':
        scan_chr = None   # all chromosomes in kinship_decomp
    else:
        scan_chr = [c.strip() for c in args.chr.split(',')]

    # ── mutual-exclusion guard ────────────────────────────────────────────────
    if args.all_celltypes and args.celltype:
        ap.error('--all-celltypes and --celltype are mutually exclusive')

    scan_desc = scan_type if scan_type != 'L' else f'L ±{window_mb:.1f} Mb'

    # ── helper: validate and return (run_data_dir, run_out_dir) for one CT ───
    def _ct_dirs(ct_label: str) -> 'tuple[Path, Path] | None':
        """Return (data_dir, out_dir) for *ct_label*, or None to skip."""
        _ct = re.sub(r'[^a-z0-9]+', '_', ct_label.lower()).strip('_')
        if 'low' in _ct and 'qual' in _ct:
            log.warning('skipping low-quality cell type %r', ct_label)
            return None
        _ct_dir    = data_dir / 'celltypes' / _ct
        _ct_exp_gi = _ct_dir / f'exp_gene_info_{_ct}.csv'
        if not _ct_dir.exists():
            log.error('cell-type directory not found: %s\n'
                      'Run first: conda run -n bioinfo python validation/tools/convert_h5ad.py'
                      ' --tissue %s --celltype %s', _ct_dir, tissue, ct_label)
            return None
        if not _ct_exp_gi.exists():
            log.error('exp_gene_info_%s.csv not found — rerun convert_h5ad.py --celltype %s',
                      _ct, ct_label)
            return None
        return _ct_dir, _ct_dir

    # ── all-celltypes mode ────────────────────────────────────────────────────
    if args.all_celltypes:
        _cts_root = data_dir / 'celltypes'
        if not _cts_root.is_dir():
            log.error('celltypes/ directory not found under %s — run convert_h5ad.py first',
                      data_dir)
            sys.exit(1)
        _all_cts = natsorted(d.name for d in _cts_root.iterdir() if d.is_dir())
        log.info('--all-celltypes: discovered %d cell types under %s', len(_all_cts), _cts_root)
        log.info('  py_ref    = %s', PY_REF)
        _n_ok = _n_skip = _n_err = 0
        for _ct_label in _all_cts:
            _dirs = _ct_dirs(_ct_label)
            if _dirs is None:
                _n_skip += 1
                continue
            _run_data_dir, _run_out_dir = _dirs
            log.info('=== cell type %r ===', _ct_label)
            log.info('  data_dir  = %s', _run_data_dir)
            log.info('  out_dir   = %s', _run_out_dir)
            log.info('  scan_tag  = %s', _stag)
            try:
                run(_run_data_dir, _run_out_dir, sections, args.force, args.n_genes,
                    scan_type=scan_type, window_mb=window_mb, scan_chr=scan_chr,
                    celltype=_ct_label, tissue=tissue, gtf_path=args.gtf,
                    sanger_alleles=args.sanger_alleles)
                _n_ok += 1
            except Exception:
                log.exception('cell type %r failed — continuing with remaining CTs', _ct_label)
                _n_err += 1
        log.info('--all-celltypes done: %d ok  %d skipped  %d errors',
                 _n_ok, _n_skip, _n_err)
        if _n_err:
            sys.exit(1)
        return

    # ── single cell-type mode ─────────────────────────────────────────────────
    _ct_label = args.celltype
    if _ct_label is not None:
        _dirs = _ct_dirs(_ct_label)
        if _dirs is None:
            sys.exit(0)
        _run_data_dir, _run_out_dir = _dirs
    else:
        _ct_label = None
        _run_data_dir = data_dir
        # whole-tissue output nests under its own 'whole/' dir, sibling to 'celltypes/'
        # (an explicit --out always wins — e.g. HPC merge redirects)
        _run_out_dir  = Path(args.out) if args.out else (data_dir / 'whole')

    log.info('validate_tissue.py  tissue=%s  sections=%s  force=%s  n_genes=%s  '
             'scan_type=%s  chr=%s  celltype=%s',
             tissue, sorted(sections), args.force, args.n_genes, scan_desc, args.chr,
             _ct_label or 'whole_tissue')
    log.info('  data_dir  = %s', _run_data_dir)
    log.info('  out_dir   = %s', _run_out_dir)
    log.info('  scan_tag  = %s  (output nests under this — see run()::_scan_tag)', _stag)
    log.info('  py_ref    = %s', PY_REF)

    run(_run_data_dir, _run_out_dir, sections, args.force, args.n_genes,
        scan_type=scan_type, window_mb=window_mb, scan_chr=scan_chr,
        celltype=_ct_label,
        tissue=tissue, gtf_path=args.gtf,
        sanger_alleles=args.sanger_alleles)


if __name__ == '__main__':
    main()

"""
validate_gastroc.py — mosaiQTL gastroc characterization + optional miQTL comparison

Primary purpose: characterize the Python mosaiqtl scan output on real gastroc
pseudobulk data (S1–S11 each run and produce output independently of R).

R comparison (gastroc_miqtl_scan.R outputs in validation/gastroc_data/) is
optional: when R reference CSVs are present, each section reports a concordance
metric; when absent the section still runs and reports Python-only statistics.

Prerequisites (required):
  1. python validation/convert_gastroc_h5ad.py
     → writes gastroc_data/counts_lognorm.h5, metadata.csv, gene_info.csv
  2. Build py_data/cache.h5 (one-time, requires rpy2 + miqtl R package):
       python validation/validate_gastroc.py --build-cache
       python validation/validate_gastroc.py --build-cache --genomecache /path/to/genomecache

Prerequisites (optional, for R comparison gates):
  Rscript validation/r_reference/gastroc_miqtl_scan.R [sections] [chr] [scan_type]

Outputs (all in validation/gastroc_data/):
  py_data/    scan CSVs, checkpoint .done files, run_summary.csv, eqtl_summary.csv
  plots/      descriptive + optional R-comparison figures
  compare/    py-vs-R merged CSVs (written only when R data present)

Usage:
  python validation/validate_gastroc.py
  python validation/validate_gastroc.py --sections 1,2,3,4
  python validation/validate_gastroc.py --force
  python validation/validate_gastroc.py --n_genes 50   # fast test run
  python validation/validate_gastroc.py --scan_type C --chr all
  python validation/validate_gastroc.py --scan_type G
  python validation/validate_gastroc.py --scan_type L --window_mb 2.0 --chr 8

scan_type:
  C  — cis: each gene scanned only against its own chromosome (default)
  G  — genome-wide: all genes against every chromosome
  L  — local: cis genes, loci within ±window_mb of gene TSS (default ±1.0 Mb)
"""

from __future__ import annotations

import argparse
import gc
import logging
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
    F1_COLORS, FOUNDER_CODE, FOUNDER_COLORS, FOUNDER_ORDER,
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
    locus_panels_per_chr as _viz_locus_panels_per_chr,
    locus_panel_grid as _viz_locus_panel_grid,
    perm_distributions_per_chr as _viz_perm_distributions_per_chr,
    perm_panel_grid as _viz_perm_panel_grid,
    snp_profiles_per_chr as _viz_snp_profiles_per_chr,
    genome_summary as _viz_genome_summary,
    genome_diagnostic as _viz_genome_diagnostic,
    snp_lod_profile as _viz_snp_lod_profile,
    instability_lod_overlay as _viz_instability_overlay,
    mi_rop_overlay as _viz_mi_rop_overlay,
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
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.scan import (
    grab_locus_from_scan,
    instability_lm_scan,
    scan_h2lmm,
    scan_h2lmm_mi,
    scan_multi_gene_mi,
    instability_scan_multi_gene,
    ScanResult,
)
from mosaiqtl.qtl.lmm import estimate_h2_null, precompute_chromosome, scan_chromosome
from mosaiqtl.qtl.permutation import (
    generate_null_outcomes,
    get_gev_thresholds,
    run_positional_scans,
    run_threshold_scans,
    ci_scan_multi_gene,
)
from mosaiqtl.qtl.mediation import (
    mediation_scan_qr,
    mediation_scan_qr,
    select_cis_mediation_pairs,
    mediation_scan_multi_outcome,
)
from mosaiqtl.qtl.snp_scan import imputed_snp_scan_from_db, snp_scan_multi_gene
from mosaiqtl.io.covariates import encode_sex
from mosaiqtl.io.results_io import pos_merge as _pos_merge
from mosaiqtl.pipeline import call_egenes
from mosaiqtl.utils import build_z_matrix, rint

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_gastroc')


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

_ALLELE_H5_LOCAL = Path('validation/gastroc_data/founder_alleles.h5')
_ALLELE_H5_PHASE2 = Path('validation/phase2/py_data/founder_alleles.h5')
ALLELE_H5    = (_ALLELE_H5_LOCAL if _ALLELE_H5_LOCAL.exists()
                else _ALLELE_H5_PHASE2)

GASTROC_DATA = Path('validation/gastroc_data')
OUT_DIR      = GASTROC_DATA   # py_data/, plots/, compare/ all live under gastroc_data/


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
        log.info('  (checkpoint — skipping S%s; use --force to rerun)', section)
        return True
    return False


def _mark_done(py_dir: Path, section: str) -> None:
    (py_dir / f's{section}.done').write_text('done')


def _scan_to_df(scan, gene_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        'gene':   gene_id,
        'chr':    scan.chromosomes,
        'locus':  scan.locus_ids,
        'pos_mb': scan.positions,
        'lod':    scan.lod,
    })


# ─────────────────────────────────────────────────────────────────────────────
# one-time cache builder — invoked only via --build-cache flag, never during run
# ─────────────────────────────────────────────────────────────────────────────

def _build_cache_from_happy(genomecache_dir: Path, out_h5: Path) -> None:
    """Read HAPPY genomecache via rpy2/miqtl DiploprobReader → save as HDF5.

    One-time step.  Run with: python validate_gastroc.py --build-cache
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
      dataset : 4 dummies (CC1, CC2, CC3, F1); Founder = biological reference (matches R relevel)
    Total: 5 columns.
    """
    meta_idx = meta.set_index('sample_id')
    rows     = meta_idx.loc[sample_order]

    sex_vec  = encode_sex(rows['sex']).values.astype(np.float64)          # (n,)

    cat      = pd.Categorical(rows['dataset'].values,
                              categories=['Founder', 'CC1', 'CC2', 'CC3', 'F1'])
    dummies  = pd.get_dummies(cat, drop_first=True).values.astype(np.float64)  # (n, 4)

    return np.column_stack([sex_vec, dummies])   # (n_samples, 5)


# ─────────────────────────────────────────────────────────────────────────────
# descriptive plots — qtl2charts-informed, no R data required
# ─────────────────────────────────────────────────────────────────────────────

# priority gene to show in LOD/allele-effect plots, keyed by chromosome
PRIORITY_GENE: dict[str, str] = {'8': 'Piezo1'}

# readable labels for the timing / memory summary plot
_SECTION_LABELS: dict[str, str] = {
    'S1':  'Cache structure',
    'S2':  'Haplotype probabilities',
    'S3':  'LOCO kinship',
    'S4':  'LMM scan',
    'S5':  'Permutation threshold',
    'S6':  'eGene calling',
    'S7':  'Multiple imputation',
    'S8':  'SNP fine-mapping scan',
    'S9':  'Mediation scan',
    'S10': 'Bootstrap CI',
    'S11': 'Instability scan',
}
_SECTION_ORDER: list[str] = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6',
                              'S7', 'S8', 'S9', 'S10', 'S11']


def _pick_top_gene(df_chr: 'pd.DataFrame', peak_chr: str,
                   lod_col: str = 'lod') -> str:
    """Return top gene by lod_col max."""
    by_gene = df_chr.groupby('gene')[lod_col].max()
    return by_gene.idxmax()


def _make_descriptive_plots(
    plt_dir: Path,
    py_dir:  Path,
    cache,
    kinship_decomp: dict,
    founders:       list[str],
    peak_chr:       str,
) -> None:
    """qtl2charts-inspired descriptive plots from Python outputs only.

    Figures produced (all in plt_dir/):
      s1_loci_per_chr.png            — loci per chromosome bar
      s2_haplotype_heatmap.png       — founder prob heatmap at sampled loci
      s3_kinship_heatmap.png         — LOCO kinship matrix heatmap
      s3_kinship_distribution.png    — kinship diagonal vs off-diagonal KDE
      s4_lod_profile.png             — LOD profile for top gene
      s4_founder_effects.png         — founder coef bar at peak locus (canonical colors)
      s4_manhattan.png               — peak LOD per gene across chr
      s4_cis_distance.png            — cis-distance scatter + KDE (all scanned vs eGenes)
      s4_cis_histogram.png           — histogram of peak − TSS distance for eGenes
      s5_perm_distribution.png       — max-LOD histogram + GEV fit + threshold
      s6_egene_lod_distribution.png  — peak LOD histogram; eGenes vs threshold
      s7_mi_lod_profile.png          — MI scan LOD profile for top gene
      s8_snp_lod_profile.png         — SNP scan LOD scatter for top gene
      s9_mediation_profile.png       — mediation LOD base/mediated/drop
      s10_ci_strip.png               — bootstrap CI strip per eGene
      s11_instability_profile.png    — instability LOD profile for top gene
      timing_memory_summary.png      — elapsed time per section bar chart
    """
    sns.set_style('ticks')
    f_colors = {f: FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in founders}

    # ── S1: loci per chromosome bar ───────────────────────────────────────────
    s1_path = py_dir / 's1_cache_info.csv'
    if s1_path.exists():
        try:
            ci = pd.read_csv(s1_path)
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
            fig.savefig(plt_dir / 's1_loci_per_chr.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  s1_loci_per_chr.png')
        except Exception as exc:
            log.warning('  s1_loci_per_chr: %s', exc)

    # ── S2: haplotype prob heatmap ────────────────────────────────────────────
    s2_path = py_dir / 's2_haplotype_probs.csv'
    if s2_path.exists():
        try:
            hp       = pd.read_csv(s2_path)
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
            fig.savefig(plt_dir / 's2_haplotype_heatmap.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  s2_haplotype_heatmap.png  (%d strains)', n_s)
        except Exception as exc:
            log.warning('  s2_haplotype_heatmap: %s', exc)

    # ── S3: kinship heatmap + distribution ───────────────────────────────────
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
            fig1.savefig(plt_dir / 's3_kinship_heatmap.png', dpi=300,
                         bbox_inches='tight', transparent=True)
            plt.close(fig1)
            log.info('  s3_kinship_heatmap.png  (%d strains)', n)

            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
            _viz_kinship_distribution(K, ax=ax2, chrom=peak_chr)
            ax2.set_title(f'Kinship distribution  chr{peak_chr}', fontsize=9)
            fig2.tight_layout()
            fig2.savefig(plt_dir / 's3_kinship_distribution.png', dpi=300, transparent=True)
            plt.close(fig2)
            log.info('  s3_kinship_distribution.png')
        except Exception as exc:
            log.warning('  s3_kinship_heatmap: %s', exc)

    # ── S4: read lod_df once (reused for profile, Manhattan) ─────────────────
    # s4_lod.csv is the full genome-wide (genes × loci) LOD table — at gastroc
    # scale this is multi-GB / 100M+ rows. Loading it whole here would
    # re-materialise the matrix MEM-05 forbids and was the source of a severe
    # plotting-stage stall. Every consumer below only needs the LOD profile
    # for a handful of named genes (top gene per chromosome + priority genes),
    # so we determine that gene set first and stream-filter the CSV in
    # chunks, keeping only matching rows.
    s4_lod_path = py_dir / 's4_lod.csv'
    s4_pk_path  = py_dir / 's4_peak_lod.csv'
    s4_eff_path = py_dir / 's4_allele_effects.csv'
    _lod_df: 'pd.DataFrame | None' = None
    if s4_lod_path.exists() and s4_pk_path.exists():
        try:
            _pk_for_filter = pd.read_csv(s4_pk_path)
            _pk_for_filter['chr'] = _pk_for_filter['chr'].astype(str)
            _needed_genes: set[str] = set(
                _pk_for_filter.loc[_pk_for_filter.groupby('chr')['peak_lod'].idxmax(),
                                   'gene'].astype(str))
            _needed_genes.update(PRIORITY_GENE.values())
            if s4_eff_path.exists():
                _eff_for_filter = pd.read_csv(s4_eff_path, usecols=['chr', 'gene', 'peak_lod'])
                _eff_for_filter['chr'] = _eff_for_filter['chr'].astype(str)
                _needed_genes.update(
                    _eff_for_filter.loc[_eff_for_filter.groupby('chr')['peak_lod'].idxmax(),
                                        'gene'].astype(str))

            _chunks: 'list[pd.DataFrame]' = []
            for _chunk in pd.read_csv(s4_lod_path, low_memory=False, chunksize=2_000_000):
                _hit = _chunk[_chunk['gene'].isin(_needed_genes)]
                if len(_hit) > 0:
                    _chunks.append(_hit.copy())
            _lod_df = (pd.concat(_chunks, ignore_index=True)
                       if _chunks else pd.DataFrame(columns=['gene', 'chr', 'locus', 'pos_mb', 'lod']))
            _lod_df['chr'] = _lod_df['chr'].astype(str)
            del _chunks
            gc.collect()
            log.info('  s4_lod.csv filtered to %d genes  (%d rows)',
                     len(_needed_genes), len(_lod_df))
        except Exception as exc:
            log.warning('  s4_lod.csv read failed: %s', exc)

    # ── threshold lookup (genome-wide scalar from s5_threshold.csv) ──────────
    _thr_by_chr: 'dict[str, float | None]' = {}
    _gw_thr_scalar: float | None = None
    _s4_thr_path = py_dir / 's5_threshold.csv'
    if _s4_thr_path.exists():
        try:
            _td0 = pd.read_csv(_s4_thr_path)
            _td0['chr'] = _td0['chr'].astype(str)
            if 'genome_wide' in _td0['chr'].values:
                _gw_thr_scalar = float(
                    _td0.loc[_td0['chr'] == 'genome_wide', 'threshold'].iloc[0])
                for _tc in chrs_to_scan:
                    _thr_by_chr[str(_tc)] = _gw_thr_scalar
            else:
                for _tc, _tg in _td0.groupby('chr'):
                    _thr_by_chr[str(_tc)] = float(_tg['threshold'].iloc[0])
        except Exception:
            pass

    # ── S4: LOD profile — top gene per chromosome ────────────────────────────
    _s4_lod_dir = plt_dir / 's4_lod_profile'
    _s4_lod_dir.mkdir(exist_ok=True)
    if _lod_df is not None and s4_pk_path.exists():
        try:
            pk_df = pd.read_csv(s4_pk_path)
            pk_df['chr'] = pk_df['chr'].astype(str)
            for c in sorted(_lod_df['chr'].unique(), key=_chr_sort_key):
                pk_c = pk_df[pk_df['chr'] == c]
                if pk_c.empty:
                    continue
                top_gene = str(pk_c.loc[pk_c['peak_lod'].idxmax(), 'gene'])
                lod_top  = _lod_df[
                    (_lod_df['chr'] == c) & (_lod_df['gene'] == top_gene)
                ].sort_values('pos_mb')
                if lod_top.empty:
                    continue
                thr = _thr_by_chr.get(c)
                fig, ax = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_lod_profile(lod_top['pos_mb'].values, lod_top['lod'].values,
                                 ax=ax, chrom=c, threshold=thr, fill=True,
                                 gene=top_gene)
                fig.tight_layout()
                fig.savefig(_s4_lod_dir / f's4_lod_profile_chr{c}.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  s4_lod_profile/s4_lod_profile_chr%s.png  (gene=%s)', c, top_gene)
                _p1 = PRIORITY_GENE.get(c)
                if _p1 and _p1 != top_gene:
                    lod_p1 = _lod_df[
                        (_lod_df['chr'] == c) & (_lod_df['gene'] == _p1)
                    ].sort_values('pos_mb')
                    if not lod_p1.empty:
                        fig2, ax2 = plt.subplots(figsize=(12, 3.5), dpi=300)
                        _viz_lod_profile(lod_p1['pos_mb'].values, lod_p1['lod'].values,
                                         ax=ax2, chrom=c, threshold=thr, fill=True,
                                         gene=_p1)
                        fig2.tight_layout()
                        fig2.savefig(_s4_lod_dir / f's4_lod_profile_{_p1}.png', dpi=300, transparent=True)
                        plt.close(fig2)
                        log.info('  s4_lod_profile/s4_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s4_lod_profile: %s', exc)

    # ── S4: founder coef bar — top gene per chromosome ───────────────────────
    _s4_eff_dir = plt_dir / 's4_founder_effects'
    _s4_eff_dir.mkdir(exist_ok=True)
    if s4_eff_path.exists():
        try:
            eff_df    = pd.read_csv(s4_eff_path)
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
                                 gene=str(top_row['gene']),
                                 locus=str(top_row['peak_locus']))
                ax.text(0.98, 0.97, f'LOD = {top_row["peak_lod"]:.2f}',
                        transform=ax.transAxes, fontsize=10, va='top', ha='right',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                  edgecolor=COLOR_NEUTRAL_DARK, alpha=0.85))
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(_s4_eff_dir / f's4_founder_effects_chr{c}.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  s4_founder_effects/s4_founder_effects_chr%s.png  (gene=%s)', c, top_row['gene'])
                _p1 = PRIORITY_GENE.get(c)
                if _p1 and top_row['gene'] != _p1 and _p1 in eff_c['gene'].values:
                    p1_row   = eff_c[eff_c['gene'] == _p1].iloc[0]
                    betas_p1 = np.array([p1_row[bc] for bc in beta_cols])
                    fig2, ax2 = plt.subplots(figsize=(8, 4), dpi=300)
                    _viz_effect_bars(betas_p1, fnames, ax=ax2,
                                     gene=str(p1_row['gene']),
                                     locus=str(p1_row['peak_locus']))
                    ax2.text(0.98, 0.97, f'LOD = {p1_row["peak_lod"]:.2f}',
                             transform=ax2.transAxes, fontsize=10, va='top', ha='right',
                             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                                       edgecolor=COLOR_NEUTRAL_DARK, alpha=0.85))
                    sns.despine(ax=ax2)
                    fig2.tight_layout()
                    fig2.savefig(_s4_eff_dir / f's4_founder_effects_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig2)
                    log.info('  s4_founder_effects/s4_founder_effects_%s.png', _p1)
        except Exception as exc:
            log.warning('  s4_founder_effects: %s', exc)

    # ── S4: founder effects grid + locus panel grids ─────────────────────────
    _s4_lp_dir = plt_dir / 's4_locus_panel'
    _s4_lp_dir.mkdir(exist_ok=True)
    if s4_eff_path.exists() and _lod_df is not None:
        try:
            _eff4     = pd.read_csv(s4_eff_path)
            _eff4['chr'] = _eff4['chr'].astype(str)
            _beta_cols4  = [bc for bc in _eff4.columns if bc.startswith('beta_')]
            _fnames4     = _display_founders([bc.replace('beta_', '') for bc in _beta_cols4])
            fig_eg = _viz_genome_effects_grid(_eff4, _fnames4)
            fig_eg.savefig(plt_dir / 's4_founder_effects_grid.png', dpi=300,
                           bbox_inches='tight', transparent=True)
            plt.close(fig_eg)
            log.info('  s4_founder_effects_grid.png')
            fig_lpg = _viz_lod_panel_grid(
                _lod_df, _eff4,
                thr_by_chr=_thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
            )
            fig_lpg.savefig(plt_dir / 's4_lod_panel_grid.png', dpi=300,
                            bbox_inches='tight', transparent=True)
            plt.close(fig_lpg)
            log.info('  s4_lod_panel_grid.png')
            fig_lcg = _viz_locus_panel_grid(
                _lod_df, _eff4, _fnames4,
                _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
            )
            fig_lcg.savefig(plt_dir / 's4_locus_panel_grid.png', dpi=300,
                            bbox_inches='tight', transparent=True)
            plt.close(fig_lcg)
            log.info('  s4_locus_panel_grid.png')
            _start_mb_map4 = None
            _start_col4 = 'near_tss_mb' if 'near_tss_mb' in _eff4.columns else \
                ('start_mb' if 'start_mb' in _eff4.columns else None)
            if _start_col4 is not None:
                _start_mb_map4 = {
                    str(g): float(t)
                    for g, t in zip(_eff4['gene'], _eff4[_start_col4])
                    if pd.notna(t)
                }
            _lp_figs = _viz_locus_panels_per_chr(
                _lod_df, _eff4, _fnames4,
                _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
                tss_mb_map=_start_mb_map4,
            )
            for _lp_c, _lp_fig in _lp_figs.items():
                _lp_fig.savefig(_s4_lp_dir / f's4_locus_panel_chr{_lp_c}.png', dpi=300, transparent=True)
                plt.close(_lp_fig)
            if _lp_figs:
                log.info('  s4_locus_panel/s4_locus_panel_chr*.png  (%d files)', len(_lp_figs))
        except Exception as exc:
            log.warning('  s4_grid: %s', exc)

    # ── S4: Manhattan — peak LOD per gene across all scanned chr ──────────────
    # s4_peak_lod.csv already holds the genome-wide peak-per-gene reduction
    # computed during the streaming S4 scan — read it directly rather than
    # rederiving it from _lod_df, which (above) is now filtered to only the
    # per-chromosome top genes and would silently drop every other gene.
    peak_lod_path = py_dir / 's4_peak_lod.csv'
    if peak_lod_path.exists():
        try:
            _pk2     = pd.read_csv(peak_lod_path)
            _pk2['chr'] = _pk2['chr'].astype(str)
            peak_pos = _pk2[['gene', 'chr', 'peak_mb', 'peak_lod']].rename(
                columns={'peak_mb': 'pos_mb', 'peak_lod': 'lod'})
            chrs_sorted = sorted(peak_pos['chr'].unique(), key=_chr_sort_key)
            scans_dict  = {
                ch: (peak_pos[peak_pos['chr'] == ch]['pos_mb'].values,
                     peak_pos[peak_pos['chr'] == ch]['lod'].values)
                for ch in chrs_sorted
            }
            _lo_mask_dict = (
                {ch: peak_pos[peak_pos['chr'] == ch]['gene'].isin(low_expr_genes).values
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
            fig.savefig(plt_dir / 's4_manhattan.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  s4_manhattan.png')
        except Exception as exc:
            log.warning('  s4_manhattan: %s', exc)

    # ── S4: cis-distance scatter (full + ±1 Mb zoom) + KDE ───────────────────
    if peak_lod_path.exists():
        try:
            _cd = pd.read_csv(peak_lod_path)
            _cd_tss_col = 'near_tss_mb' if 'near_tss_mb' in _cd.columns else 'start_mb'
            if {_cd_tss_col, 'peak_mb', 'peak_lod'}.issubset(_cd.columns):
                _cd = _cd.dropna(subset=[_cd_tss_col, 'peak_mb', 'peak_lod'])
                _sum_cd = py_dir / 'eqtl_summary.csv'
                _eg_cd  = (set(pd.read_csv(_sum_cd)['gene'])
                           if _sum_cd.exists() else set())
                _is_eg_cd = _cd['gene'].isin(_eg_cd).values
                _is_lo_cd = _cd['gene'].isin(low_expr_genes).values
                fig_cd = _viz_cis_distance_panel(
                    _cd['peak_lod'].values,
                    _cd[_cd_tss_col].values,
                    _cd['peak_mb'].values,
                    is_egene=_is_eg_cd,
                    is_low_expr=_is_lo_cd if low_expr_genes else None,
                    zoom_mb=1.0,
                    threshold=_gw_thr_scalar,
                )
                fig_cd.savefig(plt_dir / 's4_cis_distance.png', dpi=300,
                               transparent=True, bbox_inches='tight')
                plt.close(fig_cd)
                log.info('  s4_cis_distance.png  (%d genes, %d eGenes) [tss_col=%s]',
                         len(_cd), int(_is_eg_cd.sum()), _cd_tss_col)
            else:
                log.warning('  s4_cis_distance: %s/peak_mb missing — '
                            'rerun scan to populate', _cd_tss_col)
        except Exception as exc:
            log.warning('  s4_cis_distance: %s', exc)

    # ── S4: cis-distance histogram (eGenes only) ──────────────────────────────
    if peak_lod_path.exists():
        try:
            _ch = pd.read_csv(peak_lod_path)
            _ch_tss_col = 'near_tss_mb' if 'near_tss_mb' in _ch.columns else 'start_mb'
            if {_ch_tss_col, 'peak_mb'}.issubset(_ch.columns):
                _ch = _ch.dropna(subset=[_ch_tss_col, 'peak_mb'])
                _sum_ch = py_dir / 'eqtl_summary.csv'
                _eg_ch  = (set(pd.read_csv(_sum_ch)['gene'])
                           if _sum_ch.exists() else set())
                _ch_eg  = _ch[_ch['gene'].isin(_eg_ch)]
                if len(_ch_eg) > 0:
                    fig_ch, ax_ch = plt.subplots(figsize=(6, 3.8), dpi=300,
                                                 facecolor='white')
                    _viz_cis_distance_histogram(
                        _ch_eg[_ch_tss_col].values,
                        _ch_eg['peak_mb'].values,
                        ax=ax_ch,
                    )
                    fig_ch.tight_layout()
                    fig_ch.savefig(plt_dir / 's4_cis_histogram.png', dpi=300,
                                   transparent=True, bbox_inches='tight')
                    plt.close(fig_ch)
                    log.info('  s4_cis_histogram.png  (%d eGenes)', len(_ch_eg))
                else:
                    log.warning('  s4_cis_histogram: no eGenes in eqtl_summary.csv')
        except Exception as exc:
            log.warning('  s4_cis_histogram: %s', exc)

    # ── S5: permutation max-LOD distribution + GEV fit + threshold ───────────
    s5_perm_path = py_dir / 's5_perm_max_lods.csv'
    s5_thr_path  = py_dir / 's5_threshold.csv'
    if s5_perm_path.exists():
        try:
            max_lods = pd.read_csv(s5_perm_path)['max_lod'].values
            thr = None
            if s5_thr_path.exists():
                _td = pd.read_csv(s5_thr_path); _td['chr'] = _td['chr'].astype(str)
                _tr = _td[_td['chr'] == str(peak_chr)]
                thr = float(_tr['threshold'].iloc[0]) if len(_tr) > 0 else float(_td['threshold'].iloc[0])
            fig, ax  = plt.subplots(figsize=(8, 4.5), dpi=300)
            _viz_perm_distribution(max_lods, ax=ax, threshold=thr, chrom=peak_chr)
            ax.set_xlabel('Genome-wide max LOD (per permutation)', fontsize=13, fontweight='bold')
            ax.set_ylabel('Count', fontsize=13, fontweight='bold')
            ax.tick_params(labelsize=11)
            _perm_info = [f'n perms = {np.isfinite(max_lods).sum():,}']
            if thr:
                _perm_info.append(f'FWER thr = {thr:.2f}')
            ax.text(0.98, 0.97, '\n'.join(_perm_info),
                    transform=ax.transAxes, fontsize=10, va='top', ha='right',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                              edgecolor=COLOR_NEUTRAL_DARK, alpha=0.85))
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 's5_perm_distribution.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  s5_perm_distribution.png')
        except Exception as exc:
            log.warning('  s5_perm_distribution: %s', exc)

    # ── S5: per-chromosome permutation distributions ──────────────────────────
    _s5_perm_dir = plt_dir / 's5_perm'
    _s5_perm_dir.mkdir(exist_ok=True)
    if s5_thr_path.exists():
        try:
            _thr5 = pd.read_csv(s5_thr_path)
            _thr5['chr'] = _thr5['chr'].astype(str)
            _thr5_dict = dict(
                zip(_thr5['chr'], _thr5['threshold'].where(_thr5['threshold'].notna(), None))
            )
            _perm5: dict[str, np.ndarray] = {}
            for _c5 in _thr5_dict:
                _p5f = py_dir / f's5_perm_max_lods_chr{_c5}.csv'
                if _p5f.exists():
                    _perm5[_c5] = pd.read_csv(_p5f)['max_lod'].values
            if _perm5:
                _perm_figs = _viz_perm_distributions_per_chr(_perm5, _thr5_dict)
                for _pc5, _pf5 in _perm_figs.items():
                    _pf5.savefig(_s5_perm_dir / f's5_perm_chr{_pc5}.png', dpi=300, transparent=True)
                    plt.close(_pf5)
                log.info('  s5_perm/s5_perm_chr*.png  (%d files)', len(_perm_figs))
                fig_pg = _viz_perm_panel_grid(_perm5, _thr5_dict)
                fig_pg.savefig(plt_dir / 's5_perm_panel_grid.png', dpi=300,
                               bbox_inches='tight', transparent=True)
                plt.close(fig_pg)
                log.info('  s5_perm_panel_grid.png')
        except Exception as exc:
            log.warning('  s5_per_chr: %s', exc)

    # ── S6: eGene calling — LOD distribution + threshold ─────────────────────
    peak_lod_path = py_dir / 's4_peak_lod.csv'
    if peak_lod_path.exists():
        try:
            pk         = pd.read_csv(peak_lod_path)
            thr_path   = py_dir / 's5_threshold.csv'
            thr = None
            if thr_path.exists():
                _td = pd.read_csv(thr_path); _td['chr'] = _td['chr'].astype(str)
                _tr = _td[_td['chr'] == str(peak_chr)]
                thr = float(_tr['threshold'].iloc[0]) if len(_tr) > 0 else float(_td['threshold'].iloc[0])
            _sum_path  = py_dir / 'eqtl_summary.csv'
            egenes_set = (set(pd.read_csv(_sum_path)['gene'])
                          if _sum_path.exists() else set())
            lods       = pk['peak_lod'].dropna().values
            is_egene   = pk['gene'].isin(egenes_set).values
            fig, ax    = plt.subplots(figsize=(9, 4.5), dpi=300)
            _viz_egene_lod_distribution(lods, is_egene, ax=ax, threshold=thr)
            fig.tight_layout()
            fig.savefig(plt_dir / 's6_egene_lod_distribution.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  s6_egene_lod_distribution.png  (%d eGenes / %d genes)',
                     is_egene.sum(), len(pk))
        except Exception as exc:
            log.warning('  s6_egene_lod_distribution: %s', exc)

    # ── S6: genome-wide summary + diagnostic plots ────────────────────────────
    _pk6_path = py_dir / 's4_peak_lod.csv'
    if _pk6_path.exists():
        try:
            _pk6 = pd.read_csv(_pk6_path)
            _pk6['chr'] = _pk6['chr'].astype(str)
            _sum6_path = py_dir / 'eqtl_summary.csv'
            _eg6_set = (set(pd.read_csv(_sum6_path)['gene'])
                        if _sum6_path.exists() else set())
            _is_eg6 = _pk6['gene'].isin(_eg6_set).values
            _thr6: dict = {}
            if s5_thr_path.exists():
                _td6 = pd.read_csv(s5_thr_path)
                _td6['chr'] = _td6['chr'].astype(str)
                _thr6 = dict(zip(_td6['chr'],
                                 _td6['threshold'].where(_td6['threshold'].notna(), None)))
            _n_perms6 = 0
            if s5_thr_path.exists():
                _td_np6 = pd.read_csv(s5_thr_path)
                if 'n_perms' in _td_np6.columns and len(_td_np6) > 0:
                    _n_perms6 = int(_td_np6['n_perms'].iloc[0])
            _scan_stats6: dict = {
                'n_genes': len(_pk6),
                'n_egenes': int(_is_eg6.sum()),
                'peak_chr': str(peak_chr),
                'n_perms': _n_perms6,
                'top_gene': (str(_pk6.loc[_pk6['peak_lod'].idxmax(), 'gene'])
                             if len(_pk6) > 0 else ''),
            }
            fig_gs = _viz_genome_summary(_pk6, _is_eg6, _thr6 or None,
                                         scan_stats=_scan_stats6)
            fig_gs.savefig(plt_dir / 's6_genome_summary.png', dpi=300,
                           bbox_inches='tight', transparent=True)
            plt.close(fig_gs)
            log.info('  s6_genome_summary.png')
            fig_gd = _viz_genome_diagnostic(_pk6, _thr6, scan_stats=_scan_stats6)
            fig_gd.savefig(plt_dir / 's6_genome_diagnostic.png', dpi=300,
                           bbox_inches='tight', transparent=True)
            plt.close(fig_gd)
            log.info('  s6_genome_diagnostic.png')
        except Exception as exc:
            log.warning('  s6_genome_summary: %s', exc)

    # ── S8: SNP LOD profile for top eGene ────────────────────────────────────
    _s8_snp_dir = plt_dir / 's8_snp'
    _s8_snp_dir.mkdir(exist_ok=True)
    s8_path = py_dir / 's8_snp_lod.csv'
    if s8_path.exists():
        try:
            snp_df = pd.read_csv(s8_path, low_memory=False)
            snp_df['chr'] = snp_df['chr'].astype(str)
            snp_pc = snp_df[snp_df['chr'] == str(peak_chr)]
            if len(snp_pc) > 0:
                top_gene = _pick_top_gene(snp_pc, peak_chr)
                snp_top  = snp_pc[snp_pc['gene'] == top_gene].sort_values('pos_mb')
                _beta_s8 = (snp_top['beta_dosage'].values
                            if 'beta_dosage' in snp_top.columns
                            else np.zeros(len(snp_top)))
                _pk_row_s8 = snp_top.loc[snp_top['lod'].idxmax()]
                fig, ax  = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_snp_lod_profile(
                    snp_top['pos_mb'].values, snp_top['lod'].values, _beta_s8,
                    chrom=str(peak_chr), gene=top_gene,
                    peak_pos=float(_pk_row_s8['pos_mb']),
                    ax=ax,
                )
                fig.tight_layout()
                fig.savefig(_s8_snp_dir / 's8_snp_lod_profile.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  s8_snp/s8_snp_lod_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in snp_pc['gene'].values:
                    snp_p1 = snp_pc[snp_pc['gene'] == _p1].sort_values('pos_mb')
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
                    fig_p1.savefig(_s8_snp_dir / f's8_snp_lod_profile_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig_p1)
                    log.info('  s8_snp/s8_snp_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s8_snp_lod_profile: %s', exc)

    # ── S8: per-chromosome SNP fine-mapping figures ───────────────────────────
    if s8_path.exists():
        try:
            _snp8 = pd.read_csv(s8_path, low_memory=False)
            _snp8['chr'] = _snp8['chr'].astype(str)
            _pk8 = None
            if (py_dir / 's4_peak_lod.csv').exists():
                _pk8 = pd.read_csv(py_dir / 's4_peak_lod.csv')
                _pk8['chr'] = _pk8['chr'].astype(str)
            _rop8 = None
            if (py_dir / 's4_lod.csv').exists():
                _rop8 = pd.read_csv(py_dir / 's4_lod.csv', low_memory=False)
                _rop8['chr'] = _rop8['chr'].astype(str)
            _saved8 = 0
            for _c8 in sorted(_snp8['chr'].unique(), key=_chr_sort_key):
                _snp8c = _snp8[_snp8['chr'] == _c8]
                if _snp8c.empty:
                    continue
                _prio8 = PRIORITY_GENE.get(_c8)
                if _prio8 and _prio8 in _snp8c['gene'].values:
                    _g8 = _prio8
                elif _pk8 is not None:
                    _pk8c = _pk8[_pk8['chr'] == _c8]
                    _candidates8 = set(_snp8c['gene'].values)
                    _pk8c2 = _pk8c[_pk8c['gene'].isin(_candidates8)]
                    _g8 = (str(_pk8c2.loc[_pk8c2['peak_lod'].idxmax(), 'gene'])
                           if not _pk8c2.empty
                           else str(_snp8c.groupby('gene')['lod'].max().idxmax()))
                else:
                    _g8 = str(_snp8c.groupby('gene')['lod'].max().idxmax())
                _snp_g8 = _snp8c[_snp8c['gene'] == _g8].sort_values('pos_mb')
                if _snp_g8.empty:
                    continue
                _snp_peak_row8 = _snp_g8.loc[_snp_g8['lod'].idxmax()]
                _snp_peak_mb8 = float(_snp_peak_row8['pos_mb'])
                # optional haplo background
                _haplo_pos8, _haplo_lod8 = None, None
                if _rop8 is not None and _g8 in _rop8[_rop8['chr'] == _c8]['gene'].values:
                    _r8 = _rop8[(_rop8['chr'] == _c8) & (_rop8['gene'] == _g8)].sort_values('pos_mb')
                    _haplo_pos8 = _r8['pos_mb'].values
                    _haplo_lod8 = _r8['lod'].values
                _beta8 = (_snp_g8['beta_dosage'].values
                          if 'beta_dosage' in _snp_g8.columns
                          else np.zeros(len(_snp_g8)))
                fig8, ax8 = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_snp_lod_profile(
                    _snp_g8['pos_mb'].values, _snp_g8['lod'].values, _beta8,
                    haplo_positions=_haplo_pos8, haplo_lod=_haplo_lod8,
                    peak_pos=_snp_peak_mb8, ax=ax8, chrom=_c8, gene=_g8)
                fig8.tight_layout()
                fig8.savefig(_s8_snp_dir / f's8_snp_chr{_c8}.png', dpi=300, transparent=True)
                plt.close(fig8)
                _saved8 += 1
            if _saved8:
                log.info('  s8_snp/s8_snp_chr*.png  (%d files)', _saved8)
        except Exception as exc:
            log.warning('  s8_snp_per_chr: %s', exc)

    # ── S10: CI strip per eGene ───────────────────────────────────────────────
    s10_path = py_dir / 's10_ci.csv'
    if s10_path.exists():
        try:
            ci_df = pd.read_csv(s10_path, low_memory=False)
            ci_df['ci_width'] = (ci_df['ci_hi_mb'] - ci_df['ci_lo_mb']
                                 if 'ci_hi_mb' in ci_df.columns
                                 else ci_df['ci_hi'] - ci_df['ci_lo'])
            # merge chr from peak_lod if not already present
            if 'chr' not in ci_df.columns and 'gene' in ci_df.columns:
                _pk10 = py_dir / 's4_peak_lod.csv'
                if _pk10.exists():
                    _pkdf10 = pd.read_csv(_pk10)[['gene', 'chr']]
                    _pkdf10['chr'] = _pkdf10['chr'].astype(str)
                    ci_df = ci_df.merge(_pkdf10, on='gene', how='left')
            ci_df['chr'] = ci_df.get('chr', pd.Series(dtype=str)).astype(str)
            # normalise column names for ci_genomic_interval
            if 'ci_lo_mb' not in ci_df.columns:
                ci_df['ci_lo_mb'] = ci_df.get('ci_lo', np.nan)
            if 'ci_hi_mb' not in ci_df.columns:
                ci_df['ci_hi_mb'] = ci_df.get('ci_hi', np.nan)
            _pk10_path = py_dir / 's4_peak_lod.csv'
            if 'peak_lod' not in ci_df.columns and _pk10_path.exists():
                _pk10df = pd.read_csv(_pk10_path)[['gene', 'peak_lod']]
                ci_df = ci_df.merge(_pk10df, on='gene', how='left')
            _genome_thr10 = None
            if (py_dir / 's5_threshold.csv').exists():
                _t10 = pd.read_csv(py_dir / 's5_threshold.csv')
                _t10['chr'] = _t10['chr'].astype(str)
                _g10 = _t10[_t10['chr'] == 'genome']
                _col10 = 'threshold' if 'threshold' in _t10.columns else 'threshold_lod'
                if len(_g10) > 0:
                    _genome_thr10 = float(_g10[_col10].iloc[0])
            fig, ax = plt.subplots(figsize=(16, 5), dpi=300)
            _ci10_lo = (ci_df.get('ci_lo_mb', ci_df.get('ci_lo', np.full(len(ci_df), np.nan)))
                        .astype(float).values)
            _ci10_hi = (ci_df.get('ci_hi_mb', ci_df.get('ci_hi', np.full(len(ci_df), np.nan)))
                        .astype(float).values)
            _ci10_lod = (ci_df['peak_lod'].values if 'peak_lod' in ci_df.columns
                         else np.ones(len(ci_df)) * (_genome_thr10 or 3.0))
            _ci10_mb = (ci_df['peak_mb'].values if 'peak_mb' in ci_df.columns
                        else np.zeros(len(ci_df)))
            _viz_ci_genomic_interval(
                ci_df['gene'].tolist() if 'gene' in ci_df.columns else list(ci_df.index.astype(str)),
                ci_df['chr'].astype(str).tolist() if 'chr' in ci_df.columns else ['?'] * len(ci_df),
                _ci10_lo, _ci10_hi, _ci10_lod, _ci10_mb,
                ax=ax,
            )
            fig.tight_layout(pad=1.5)
            fig.savefig(plt_dir / 's10_ci_strip.png', dpi=300,
                        bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.info('  s10_ci_strip.png  (%d genes)', len(ci_df))
        except Exception as exc:
            log.warning('  s10_ci_strip: %s', exc)

    # ── S11: instability LOD profile for top gene ─────────────────────────────
    _s11_dir = plt_dir / 's11_instability'
    _s11_dir.mkdir(exist_ok=True)
    s11_path = py_dir / 's11_instability.csv'
    if s11_path.exists():
        try:
            inst_df = pd.read_csv(s11_path, low_memory=False)
            inst_df['chr'] = inst_df['chr'].astype(str)
            inst_pc = inst_df[inst_df['chr'] == str(peak_chr)]
            if len(inst_pc) > 0:
                top_gene = _pick_top_gene(inst_pc, peak_chr)
                inst_top = inst_pc[inst_pc['gene'] == top_gene].sort_values('pos_mb')
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
                fig.savefig(_s11_dir / 's11_instability_profile.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  s11_instability/s11_instability_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in inst_pc['gene'].values:
                    inst_p1 = inst_pc[inst_pc['gene'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        inst_p1['pos_mb'].values, inst_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, fill=True,
                        color=FOUNDER_COLORS['CASTJ'],
                    )
                    ax_p1.set_ylabel('Instability LOD Score')
                    ax_p1.set_title(f'Instability  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(_s11_dir / f's11_instability_profile_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig_p1)
                    log.info('  s11_instability/s11_instability_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s11_instability_profile: %s', exc)

    # ── S11: per-chromosome instability overlay (instability score vs ROP LOD) ─
    if s11_path.exists():
        try:
            _inst11 = pd.read_csv(s11_path)
            _inst11['chr'] = _inst11['chr'].astype(str)
            _pk11 = None
            if (py_dir / 's4_peak_lod.csv').exists():
                _pk11 = pd.read_csv(py_dir / 's4_peak_lod.csv')
                _pk11['chr'] = _pk11['chr'].astype(str)
            _rop11 = None
            if (py_dir / 's4_lod.csv').exists():
                _rop11 = pd.read_csv(py_dir / 's4_lod.csv', low_memory=False)
                _rop11['chr'] = _rop11['chr'].astype(str)
            _saved11 = 0
            for _c11 in sorted(_inst11['chr'].unique(), key=_chr_sort_key):
                _inst11c = _inst11[_inst11['chr'] == _c11]
                if _inst11c.empty:
                    continue
                _prio11 = PRIORITY_GENE.get(_c11)
                if _prio11 and _prio11 in _inst11c['gene'].values:
                    _g11 = _prio11
                elif _pk11 is not None:
                    _pk11c = _pk11[_pk11['chr'] == _c11]
                    _g11 = (str(_pk11c.loc[_pk11c['peak_lod'].idxmax(), 'gene'])
                            if not _pk11c.empty
                            else str(_inst11c.groupby('gene')['lod'].max().idxmax()))
                else:
                    _g11 = str(_inst11c.groupby('gene')['lod'].max().idxmax())
                _inst_top11 = _inst11c[_inst11c['gene'] == _g11].sort_values('pos_mb')
                if _inst_top11.empty:
                    continue
                _thr11 = _thr_by_chr.get(_c11)
                _rop_lod11: np.ndarray
                _rop_pos11: np.ndarray
                if (_rop11 is not None
                        and _g11 in _rop11[_rop11['chr'] == _c11]['gene'].values):
                    _rop_top11 = _rop11[(_rop11['chr'] == _c11)
                                        & (_rop11['gene'] == _g11)].sort_values('pos_mb')
                    _rop_pos11 = _rop_top11['pos_mb'].values
                    _rop_lod11 = _rop_top11['lod'].values
                else:
                    _rop_pos11 = _inst_top11['pos_mb'].values
                    _rop_lod11 = np.zeros(len(_rop_pos11))
                # inst scores (0–1); use peak_freq if available, else normalise lod
                _inst_col = ('peak_freq' if 'peak_freq' in _inst_top11.columns else 'lod')
                _inst_raw = _inst_top11[_inst_col].values
                _inst_scores11 = (_inst_raw / max(_inst_raw.max(), 1e-6)
                                  if _inst_raw.max() > 1.0 else _inst_raw)
                # align positions: use union grid
                _common_pos = _inst_top11['pos_mb'].values
                _rop_interp = np.interp(_common_pos, _rop_pos11, _rop_lod11)
                fig11, ax11 = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_instability_overlay(
                    _common_pos, _rop_interp, _inst_scores11,
                    ax=ax11, chrom=_c11, gene=_g11, threshold=_thr11)
                fig11.tight_layout()
                fig11.savefig(_s11_dir / f's11_instability_chr{_c11}.png', dpi=300, transparent=True)
                plt.close(fig11)
                _saved11 += 1
            if _saved11:
                log.info('  s11_instability/s11_instability_chr*.png  (%d files, overlay)', _saved11)
        except Exception as exc:
            log.warning('  s11_per_chr: %s', exc)

    # ── S7: MI scan LOD profile ───────────────────────────────────────────────
    _s7_mi_dir = plt_dir / 's7_mi'
    _s7_mi_dir.mkdir(exist_ok=True)
    s7_path = py_dir / 's7_mi_lod.csv'
    if s7_path.exists():
        try:
            mi_df = pd.read_csv(s7_path, low_memory=False)
            mi_df['chr'] = mi_df['chr'].astype(str)
            mi_pc = mi_df[mi_df['chr'] == str(peak_chr)]
            if len(mi_pc) > 0:
                top_gene = _pick_top_gene(mi_pc, peak_chr)
                mi_top   = mi_pc[mi_pc['gene'] == top_gene].sort_values('pos_mb')
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
                fig.savefig(_s7_mi_dir / 's7_mi_lod_profile.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  s7_mi/s7_mi_lod_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in mi_pc['gene'].values:
                    mi_p1 = mi_pc[mi_pc['gene'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        mi_p1['pos_mb'].values, mi_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, fill=True,
                        color=FOUNDER_COLORS['WSBJ'],
                    )
                    ax_p1.set_ylabel('LOD Score (MI)')
                    ax_p1.set_title(f'MI LOD Score Profile  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(_s7_mi_dir / f's7_mi_lod_profile_{_p1}.png', dpi=300, transparent=True)
                    plt.close(fig_p1)
                    log.info('  s7_mi/s7_mi_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s7_mi_lod_profile: %s', exc)

    # ── S7: per-chromosome MI overlay (MI LOD vs ROP LOD) ─────────────────────
    if s7_path.exists():
        try:
            _mi7 = pd.read_csv(s7_path, low_memory=False)
            _mi7['chr'] = _mi7['chr'].astype(str)
            _pk7 = None
            if (py_dir / 's4_peak_lod.csv').exists():
                _pk7 = pd.read_csv(py_dir / 's4_peak_lod.csv')
                _pk7['chr'] = _pk7['chr'].astype(str)
            _rop7 = None
            if (py_dir / 's4_lod.csv').exists():
                _rop7 = pd.read_csv(py_dir / 's4_lod.csv', low_memory=False)
                _rop7['chr'] = _rop7['chr'].astype(str)
            _saved7 = 0
            for _c7 in sorted(_mi7['chr'].unique(), key=_chr_sort_key):
                _mi7c = _mi7[_mi7['chr'] == _c7]
                if _mi7c.empty:
                    continue
                _prio7 = PRIORITY_GENE.get(_c7)
                if _prio7 and _prio7 in _mi7c['gene'].values:
                    _g7 = _prio7
                elif _pk7 is not None:
                    _pk7c = _pk7[_pk7['chr'] == _c7]
                    _g7 = (str(_pk7c.loc[_pk7c['peak_lod'].idxmax(), 'gene'])
                           if not _pk7c.empty
                           else str(_mi7c.groupby('gene')['lod'].max().idxmax()))
                else:
                    _g7 = str(_mi7c.groupby('gene')['lod'].max().idxmax())
                _mi_top7 = _mi7c[_mi7c['gene'] == _g7].sort_values('pos_mb')
                if _mi_top7.empty:
                    continue
                _thr7 = _thr_by_chr.get(_c7)
                _common_pos7 = _mi_top7['pos_mb'].values
                _mi_lod7 = _mi_top7['lod'].values
                if (_rop7 is not None
                        and _g7 in _rop7[_rop7['chr'] == _c7]['gene'].values):
                    _rop_top7 = _rop7[(_rop7['chr'] == _c7)
                                      & (_rop7['gene'] == _g7)].sort_values('pos_mb')
                    _rop_lod7 = np.interp(_common_pos7, _rop_top7['pos_mb'].values,
                                          _rop_top7['lod'].values)
                else:
                    _rop_lod7 = np.zeros_like(_mi_lod7)
                fig7, ax7 = plt.subplots(figsize=(12, 3.5), dpi=300)
                _viz_mi_rop_overlay(
                    _common_pos7, _rop_lod7, _mi_lod7,
                    ax=ax7, chrom=_c7, gene=_g7, threshold=_thr7)
                fig7.tight_layout()
                fig7.savefig(_s7_mi_dir / f's7_mi_chr{_c7}.png', dpi=300, transparent=True)
                plt.close(fig7)
                _saved7 += 1
            if _saved7:
                log.info('  s7_mi/s7_mi_chr*.png  (%d files, MI vs ROP overlay)', _saved7)
        except Exception as exc:
            log.warning('  s7_per_chr: %s', exc)

    # ── S9: mediation LOD profile (base vs mediated) ──────────────────────────
    # ── S9: lod_drop distribution per chromosome ──────────────────────────────
    s9_path = py_dir / 's9_mediation.csv'
    if s9_path.exists():
        try:
            med_df = pd.read_csv(s9_path, low_memory=False)
            med_df['chr'] = med_df['chr'].astype(str)
            # derive lod_drop if not present (lod_base - lod_mediated)
            if 'lod_drop' not in med_df.columns:
                if 'lod_base' in med_df.columns and 'lod_mediated' in med_df.columns:
                    med_df['lod_drop'] = (med_df['lod_base'] - med_df['lod_mediated']).clip(lower=0)
                else:
                    raise KeyError('Column not found: lod_drop (and lod_base/lod_mediated absent)')
            _chr_sort_series = lambda s: s.map(lambda c: int(c) if c.isdigit() else 100)
            s9_grp = (med_df.groupby('chr')['lod_drop']
                      .agg(mean_drop='mean', std_drop='std', n_pairs='count')
                      .reset_index()
                      .sort_values('chr', key=_chr_sort_series))
            n_outcomes_s9 = (len(med_df['gene'].unique())
                             if 'gene' in med_df.columns else 0)
            n_chrs = len(s9_grp)
            fig, ax = plt.subplots(
                figsize=(max(8, n_chrs * 0.6), 4.5), dpi=300)
            _viz_mediation_summary_bar(
                s9_grp['chr'].tolist(),
                s9_grp['mean_drop'].values,
                std_drop=s9_grp['std_drop'].fillna(0).values,
                n_pairs=s9_grp['n_pairs'].values,
                n_outcomes=n_outcomes_s9,
                ax=ax,
            )
            sns.despine(ax=ax)
            fig.tight_layout(pad=1.5)
            fig.savefig(plt_dir / 's9_mediation_by_chr.png', dpi=300,
                        bbox_inches='tight', transparent=True)
            plt.close(fig)
            log.info('  s9_mediation_by_chr.png  (%d chr)', n_chrs)
        except Exception as exc:
            log.warning('  s9_mediation_by_chr: %s', exc)

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
    _PLOT_SECTIONS = ['S4', 'S5', 'S7', 'S8', 'S9', 'S10', 'S11']
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
            chr_rows = ss_all[ss_all['section'].str.startswith('S4_chr')].copy()
            if len(chr_rows) > 0:
                chr_rows['chrom'] = (chr_rows['section']
                                     .str.replace('S4_chr', '', regex=False))

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
) -> None:
    """generate all R-vs-Python comparison figures; skip silently if inputs missing."""
    from scipy.stats import genextreme as gev_dist

    s5_thr_path = py_dir / 's5_threshold.csv'

    display_f = _display_founders(founders)  # full names for labels/colors
    fcolors = [FOUNDER_COLORS.get(f, COLOR_CC_FALLBACK) for f in display_f]

    # ── S2: haplotype prob scatter ─────────────────────────────────────────────
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
                    save_path=plt_dir / 's2_haplotype_scatter.png',
                )
                plt.close(fig)
                log.info('  [plot] s2_haplotype_scatter.png')
        except Exception as exc:
            log.warning('  [plot] s2_haplotype_scatter: %s', exc)

    # ── S3: kinship element scatter (chr8) ────────────────────────────────────
    k8_path = r_data / 'kinship_chr8_elements.csv'
    if k8_path.exists() and peak_chr in kinship_decomp:
        try:
            k_elem = pd.read_csv(k8_path)
            kd8    = kinship_decomp.get('8') or kinship_decomp.get(peak_chr)
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
                    'R kinship (LOCO chr8)', 'Python kinship (LOCO chr8)',
                    cbar_label='mean K',
                    save_path=plt_dir / 's3_kinship_scatter.png',
                )
                plt.close(fig)
                log.info('  [plot] s3_kinship_scatter.png')
        except Exception as exc:
            log.warning('  [plot] s3_kinship_scatter: %s', exc)

    # ── S4: LOD density (peak LOD distribution) ───────────────────────────────
    # Read Python and R independently so each series plots whatever is available
    # without requiring both to be present in the merged comparison CSV.
    _py_lod_path = py_dir / 's4_peak_lod.csv'
    _r_lod_path  = r_data / 'peak_lod.csv'
    _density_sources = [
        (_py_lod_path, 'peak_lod', 'Python', '#4F6EAF'),
        (_r_lod_path,  'lod',      'R',      '#D83026'),
    ]
    _density_any = any(p.exists() for p, *_ in _density_sources)
    if _density_any:
        try:
            _lod_thr_d: float | None = None
            if s5_thr_path.exists():
                _td_ld = pd.read_csv(s5_thr_path)
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
            fig.savefig(plt_dir / 's4_lod_density.png', dpi=300, transparent=True)
            plt.close(fig)
            log.info('  [plot] s4_lod_density.png')
        except Exception as exc:
            log.warning('  [plot] s4_lod_density: %s', exc)

    # ── S4: peak LOD scatter (R vs Python) ────────────────────────────────────
    _cmp_lod_path = cmp_dir / 's4_peak_lod_comparison.csv'
    if _cmp_lod_path.exists():
        try:
            _cmp_lod = pd.read_csv(_cmp_lod_path)
            if len(_cmp_lod) >= 2 and 'peak_lod_py' in _cmp_lod.columns and 'peak_lod_r' in _cmp_lod.columns:
                _sparse_cmp = (
                    _cmp_lod['gene'].isin(low_expr_genes)
                    if low_expr_genes and 'gene' in _cmp_lod.columns else None
                )
                fig, _ = _cmp_scatter(
                    _cmp_lod, 'peak_lod_r', 'peak_lod_py',
                    'miQTL (R) peak LOD', 'mosaiQTL (Python) peak LOD',
                    priority_genes={'Cwc22': FOUNDER_COLORS['CASTJ']},
                    sparse_mask=_sparse_cmp,
                    save_path=plt_dir / 's4_peak_lod_scatter.png',
                )
                plt.close(fig)
                log.info('  [plot] s4_peak_lod_scatter.png')
        except Exception as exc:
            log.warning('  [plot] s4_peak_lod_scatter: %s', exc)

    # ── S4: peak position scatter ──────────────────────────────────────────────
    eff_path = py_dir / 's4_allele_effects.csv'
    if eff_path.exists():
        try:
            eff_py = pd.read_csv(eff_path, usecols=['gene', 'peak_mb'])
            # Prefer r_data/peak_lod.csv (already one row per gene with pos_mb).
            # Fall back to per-chr lod.csv files only if peak_lod.csv absent.
            r_peak_lod_pos = r_data / 'peak_lod.csv'
            r_peak_rows = []
            if r_peak_lod_pos.exists():
                _rpp = pd.read_csv(r_peak_lod_pos, usecols=['gene', 'pos_mb'])
                r_peak_rows.append(_rpp.rename(columns={'pos_mb': 'peak_mb_r'}))
            else:
                for cand in sorted(
                    r_data.glob('chr*/lod.csv'),
                    key=lambda p: _chr_sort_key(p.parent.name.replace('chr', '')),
                ):
                    tmp = pd.read_csv(cand, usecols=['gene', 'pos_mb', 'lod'])
                    r_peak_rows.append(
                        tmp.sort_values('lod', ascending=False)
                           .drop_duplicates('gene')[['gene', 'pos_mb']]
                           .rename(columns={'pos_mb': 'peak_mb_r'}))
            if r_peak_rows:
                r_pos = pd.concat(r_peak_rows).groupby('gene')['peak_mb_r'].first().reset_index()
                merged_pos = eff_py.merge(r_pos, on='gene', how='inner')
                if len(merged_pos) >= 5:
                    fig, _ = _cmp_scatter(
                        merged_pos, 'peak_mb_r', 'peak_mb',
                        'R peak position (Mb)', 'Python peak position (Mb)',
                        cbar_label='mean position (Mb)',
                        save_path=plt_dir / 's4_peak_pos_scatter.png',
                    )
                    plt.close(fig)
                    log.info('  [plot] s4_peak_pos_scatter.png')
        except Exception as exc:
            log.warning('  [plot] s4_peak_pos_scatter: %s', exc)

    # ── S4: LOD profiles (top 3 genes on peak_chr) ────────────────────────────
    py_lod_path = py_dir / 's4_lod.csv'
    if py_lod_path.exists():
        try:
            lod_py_all = pd.read_csv(py_lod_path)
            lod_pc     = lod_py_all[lod_py_all['chr'].astype(str) == str(peak_chr)]
            if len(lod_pc) > 0:
                top_genes = (lod_pc.groupby('gene')['lod'].max()
                             .nlargest(3).index.tolist())
                r_lod_chr_path = r_data / f'chr{peak_chr}' / 'lod.csv'
                r_lod_chr = (pd.read_csv(r_lod_chr_path)
                             if r_lod_chr_path.exists() else None)
                _comp_thr_p: float | None = None
                if s5_thr_path.exists():
                    _td_cp = pd.read_csv(s5_thr_path)
                    _td_cp['chr'] = _td_cp['chr'].astype(str)
                    _tr_cp = _td_cp[_td_cp['chr'] == str(peak_chr)]
                    _comp_thr_p = float(_tr_cp['threshold'].iloc[0]) if len(_tr_cp) > 0 else None
                fig = _cmp_lod_profiles(
                    lod_pc, r_lod_chr, top_genes, str(peak_chr), thr=_comp_thr_p)
                fig.savefig(plt_dir / 's4_lod_profiles.png', dpi=300, transparent=True)
                plt.close(fig)
                log.info('  [plot] s4_lod_profiles.png  top %d genes on chr%s',
                         len(top_genes), peak_chr)
        except Exception as exc:
            log.warning('  [plot] s4_lod_profiles: %s', exc)

    # ── S4: allele effects scatter (Python betas vs R betas, 2× scale note) ───
    r_eff_path = r_data / 'allele_effects_chr8.csv'
    if eff_path.exists() and r_eff_path.exists():
        try:
            py_eff = pd.read_csv(eff_path)
            r_eff  = pd.read_csv(r_eff_path)
            _disp_f = _display_founders(founders)
            py_beta_cols = [f'beta_{d}' for d in _disp_f if f'beta_{d}' in py_eff.columns]
            r_beta_cols  = [c for c in r_eff.columns if c.startswith('beta_')]
            merged_eff   = py_eff.merge(r_eff, on='gene', how='inner', suffixes=('_py', '_r'))
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
                    fig.savefig(plt_dir / 's4_allele_effects_scatter.png', dpi=300, transparent=True)
                    plt.close(fig)
                    log.info('  [plot] s4_allele_effects_scatter.png')
        except Exception as exc:
            log.warning('  [plot] s4_allele_effects_scatter: %s', exc)

    # ── S5: permutation distribution (py vs R, fitted GEV) ────────────────────
    # Read each source independently — plot whichever series exists.
    py_perm_path = py_dir / 's5_perm_max_lods.csv'
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
                _s5_row = _ts[_ts['section'] == 'S5_permutation']
                if len(_s5_row) > 0 and pd.notna(_s5_row['n_perms'].iloc[0]):
                    r_n_perms = int(_s5_row['n_perms'].iloc[0])
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
                fig.savefig(plt_dir / 's5_perm_distribution.png', dpi=300, transparent=True)
                log.info('  [plot] s5_perm_distribution.png')
                plt.close(fig)
        except Exception as exc:
            log.warning('  [plot] s5_perm_distribution: %s', exc)

    # ── S4: scan speed comparison (genes/min py vs R per chr) ─────────────────
    # Plot each series from its own source; show Python-only when R timing absent.
    scan_stats_path = py_dir / 'scan_stats.csv'
    r_timing_path   = r_data / 'timing_chr.csv'
    if scan_stats_path.exists() or r_timing_path.exists():
        try:
            py_s4, r_s4 = pd.DataFrame(), None
            if scan_stats_path.exists():
                py_stats = pd.read_csv(scan_stats_path)
                py_s4 = py_stats[py_stats['section'].str.startswith('S4_chr')].copy()
                py_s4['chr'] = py_s4['section'].str.replace('S4_chr', '', regex=False)
                py_s4['chr'] = py_s4['chr'].astype(str)
                py_s4['gpm'] = py_s4['n_items'] / py_s4['elapsed_s'].replace(0, float('nan')) * 60
            if r_timing_path.exists():
                r_timing = pd.read_csv(r_timing_path)
                if 'genes_per_min' in r_timing.columns:
                    r_s4 = r_timing[['chr', 'genes_per_min']].copy()
                    r_s4['chr'] = r_s4['chr'].astype(str)
                elif 'elapsed_s' in r_timing.columns and 'n_genes' in r_timing.columns:
                    r_s4 = r_timing.copy()
                    r_s4['chr'] = r_s4['chr'].astype(str)
                    r_s4['genes_per_min'] = (r_s4['n_genes']
                                             / r_s4['elapsed_s'].replace(0, float('nan')) * 60)

            has_py = len(py_s4) >= 2
            has_r  = r_s4 is not None and len(r_s4) >= 2
            if has_py or has_r:
                chrs_sp = sorted(
                    (set(py_s4['chr']) if has_py else set()) |
                    (set(r_s4['chr']) if has_r else set()),
                    key=lambda c: int(c) if c.isdigit() else 100,
                )
                if len(chrs_sp) >= 2:
                    py_gpm = ({row['chr']: row['gpm'] for _, row in py_s4.iterrows()}
                              if has_py else {})
                    r_gpm  = ({row['chr']: row['genes_per_min']
                               for _, row in r_s4.iterrows()}
                              if has_r else {})
                    _py_arr_sp = np.array([py_gpm.get(c, 0.0) for c in chrs_sp])
                    _r_arr_sp  = (np.array([r_gpm.get(c, 0.0) for c in chrs_sp])
                                  if has_r else None)
                    fig, ax = plt.subplots(figsize=(max(10, len(chrs_sp) * 0.6), 4.5), dpi=300)
                    _viz_scan_throughput(chrs_sp, _py_arr_sp,
                                        r_genes_per_min=_r_arr_sp, ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's4_scan_speed.png', dpi=300, transparent=True)
                    plt.close(fig)
                    log.info('  [plot] s4_scan_speed.png  %d chr', len(chrs_sp))
        except Exception as exc:
            log.warning('  [plot] s4_scan_speed: %s', exc)

    # ── S5: per-chromosome FWER threshold comparison (Python vs R bars) ────────
    py_thr_path = py_dir / 's5_threshold.csv'
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
                fig.savefig(plt_dir / 's5_perm_threshold_comparison.png',
                            dpi=300, transparent=True)
                plt.close(fig)
                log.info('  [plot] s5_perm_threshold_comparison.png')
        except Exception as exc:
            log.warning('  [plot] s5_perm_threshold_comparison: %s', exc)


# ─────────────────────────────────────────────────────────────────────────────
# main validation run
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: Path, out_dir: Path, sections: set[int],
        force: bool, n_genes: int | None,
        scan_type: str = 'C',
        window_mb: float = 1.0,
        scan_chr: list[str] | None = None,
        celltype: str | None = None,
        plt_dir_override: Path | None = None) -> dict:
    """run gastroc characterization sections; return gate results dict.

    Parameters
    ----------
    data_dir         : root of gastroc_data/ — contains metadata.csv and r_data/;
                       in CT mode this is also the CT dir with counts_lognorm.h5
    out_dir          : directory where py_data/ (or CT-mode flat outputs) are written
    scan_type        : 'C' (cis), 'G' (genome-wide), or 'L' (local ±window_mb of TSS)
    window_mb        : half-window size in Mb for L mode (ignored for C/G)
    scan_chr         : chromosomes to scan; None = all chromosomes in kinship_decomp
    celltype         : cell-type label when running in CT mode; None = whole-tissue
    plt_dir_override : plot output directory override (CT mode points here)
    """
    # in CT mode: outputs go flat into out_dir (no py_data/ subdir);
    # cmp_dir points to a shared celltypes/compare/ for CT-vs-CT comparison CSVs
    _ct_mode = celltype is not None
    r_data  = GASTROC_DATA / 'r_data'   # R reference always from whole-tissue
    py_dir  = out_dir if _ct_mode else out_dir / 'py_data'
    cmp_dir = (GASTROC_DATA / 'celltypes' / 'compare'
               if _ct_mode else out_dir / 'compare')
    plt_dir = plt_dir_override if plt_dir_override is not None else out_dir / 'plots'
    dirs_to_make = [py_dir, cmp_dir, plt_dir]
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
    meta_path = GASTROC_DATA / 'metadata.csv'
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
    # schema: gene_name=symbol (HDF5 key), gene_id=Ensembl, strand, chr, start, end
    # start_mb = start/1e6 for + strand, end/1e6 for - strand (gene boundary, not TSS)
    if _ct_mode:
        import re as _re_gi
        _ct_safe_label = _re_gi.sub(r'[^a-z0-9]+', '_', celltype.lower()).strip('_')
        gene_info_path = data_dir / f'exp_gene_info_{_ct_safe_label}.csv'
    else:
        gene_info_path = data_dir / 'exp_gene_info_gastroc.csv'
    gene_info = pd.read_csv(gene_info_path)
    gene_info['chr'] = gene_info['chr'].astype(str)
    log.info('gene_info: %d genes  (%s)', len(gene_info), gene_info_path.name)

    low_expr_genes: set[str] = set()
    if 'frac_samples_cpm_gt1' in gene_info.columns:
        low_expr_genes = set(
            gene_info.loc[gene_info['frac_samples_cpm_gt1'] < 0.5, 'gene_name'])
        log.info('low-expression genes (frac_cpm_gt1 < 0.5): %d / %d',
                 len(low_expr_genes), len(gene_info))

    # ── transcript TSS table (for near_tss_mb after peak calling) ───────────────
    # near_tss_mb = transcript TSS (strand-aware) nearest to peak_mb per gene
    # source: GENCODE vM32 transcript features for this run; one row per transcript
    _gene_tss_path = pathlib.Path(__file__).parent / 'input' / 'gene_tss.vm32.csv'
    if _gene_tss_path.exists():
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
        log.warning('gene_tss.vm32.csv not found — near_tss_mb will be omitted')

    if scan_type == 'L' and 'start' not in gene_info.columns:
        log.warning('%s has no "start" column — L scan mode will not filter loci',
                    gene_info_path.name)

    # ── load GenomeCache (must be pre-built as HDF5) ──────────────────────────
    hdf5_cache = GASTROC_DATA / 'py_data' / 'cache.h5'
    if hdf5_cache.exists():
        log.info('loading GenomeCache from HDF5: %s', hdf5_cache)
        cache = GenomeCache.from_hdf5(str(hdf5_cache))
        cache = cache.reorder_to_canonical()  # fast no-op if already canonical
    else:
        log.error(
            'cache.h5 not found at %s\n'
            'Build it once with:\n'
            '  conda run -n bioinfo python validation/validate_gastroc.py --build-cache\n'
            'or specify the genomecache path:\n'
            '  ... --build-cache --genomecache /path/to/genomecache',
            hdf5_cache)
        sys.exit(1)
    log.info('cache: %d chromosomes  %d strains  sample[0]=%s',
             len(cache.chromosomes), len(cache.sample_ids), cache.sample_ids[0])

    # ── load or compute LOCO kinship decomp ───────────────────────────────────
    kd_npz = GASTROC_DATA / 'py_data' / 'kinship_decomp.npz'
    if kd_npz.exists():
        log.info('loading kinship decomp: %s', kd_npz)
        raw = np.load(kd_npz, allow_pickle=True)
        kinship_decomp = {k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
                          for k in cache.chromosomes if f'{k}_vals' in raw}
        del raw
    else:
        log.info('computing LOCO kinship from cache (first run — saves to %s)...', kd_npz)
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
    log.info('covariates: shape %s  columns = [sex, CC1, CC2, CC3, F1]  ref=Founder',
             covariates_meta.shape)
    _sex_by_sid: dict[str, float] = (
        encode_sex(meta['sex']).set_axis(meta['sample_id']).to_dict()
        if 'sex' in meta.columns else {}
    )
    sex_meta = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sample_ids_meta])

    # ─────────────────────────────────────────────────────────────────────────
    # S1 — cache characterization
    # always characterizes Python cache; compares to R cache_info.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 1
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS1: cache characterization...')
            _rss_s1 = _get_rss_mb()
            tracemalloc.start()
            _t_s1 = time.perf_counter()
            s1_ok = True

            # always: characterize Python cache
            py_loci = {c: len(cache.get_chromosome(c)[0]) for c in cache.chromosomes}
            ci_py = pd.DataFrame([{'chr': c, 'n_loci': n}
                                  for c, n in py_loci.items()])
            ci_py.to_csv(py_dir / 's1_cache_info.csv', index=False)
            log.info('  cache: %d chromosomes  total loci=%d  founders=%d  genotypes=%d',
                     len(cache.chromosomes), sum(py_loci.values()),
                     len(cache.founders), len(cache.sample_ids))
            log.info('  founders: %s', list(cache.founders))
            _s1_n_items = len(ci_py)

            # optional: compare loci counts to R
            cache_info_r = r_data / 'cache_info.csv'
            if cache_info_r.exists():
                ci_r = pd.read_csv(cache_info_r)
                ci_r['chr'] = ci_r['chr'].astype(str)
                merged = ci_py.merge(
                    ci_r[['chr', 'n_loci']].rename(columns={'n_loci': 'n_loci_r'}),
                    on='chr', how='outer')
                merged['delta'] = (merged['n_loci'] - merged['n_loci_r']).abs()
                merged.to_csv(cmp_dir / 's1_cache_info_cmp.csv', index=False)
                max_delta = int(merged['delta'].fillna(9999).max())
                loci_ok = _check('max loci delta', float(max_delta), 0.5, '<')
                s1_ok = s1_ok and loci_ok
                log.info('    vs R: loci delta max=%d', max_delta)

            # optional: compare founders/genotypes to R
            founders_r_path = r_data / 'founders.csv'
            if founders_r_path.exists():
                founders_r = pd.read_csv(founders_r_path)['founder'].tolist()
                py_founders = list(cache.founders)
                founders_ok = founders_r == py_founders
                sym = PASS if founders_ok else FAIL
                log.info('  %s vs R founders match=%s', sym, founders_ok)
                s1_ok = s1_ok and founders_ok

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
                s1_ok = s1_ok and set_match

            _, _s1_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _s1_elapsed = time.perf_counter() - _t_s1
            _record_stats('S1', 'cache_info', _s1_n_items,
                          _s1_elapsed, _s1_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s1))
            log.info('  [timing] S1: %.2fs  heap_peak=%.0fMB',
                     _s1_elapsed, _s1_peak / 1e6)

            results['S1_cache'] = s1_ok
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S2 — haplotype probabilities
    # always exports haplotype prob sample; compares to R if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 2
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS2: haplotype probabilities...')
            _rss_s2 = _get_rss_mb()
            tracemalloc.start()
            _t_s2 = time.perf_counter()
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
            hap_py.to_csv(py_dir / 's2_haplotype_probs.csv', index=False)
            log.info('  sampled %d loci × %d genotypes on chr%s',
                     len(sample_lids), len(cache.sample_ids), probe_chr)
            _s2_n = len(hap_rows)

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
                    df_s2 = pd.DataFrame(rows)
                    df_s2.to_csv(cmp_dir / 's2_haplotype_argmax_cmp.csv', index=False)
                    concordance = float(df_s2['match'].mean())
                    s2_ok = _check('haplotype argmax concordance', concordance, 0.95)
                    results['S2_haplotype'] = s2_ok
                    log.info('    vs R: concordance=%.4f  (%d/%d pairs)',
                             concordance, df_s2['match'].sum(), len(df_s2))
                else:
                    results['S2_haplotype'] = None
            else:
                results['S2_haplotype'] = None

            _, _s2_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _s2_elapsed = time.perf_counter() - _t_s2
            _record_stats('S2', 'haplotype_probe', _s2_n,
                          _s2_elapsed, _s2_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s2))
            log.info('  [timing] S2: %.2fs  heap_peak=%.0fMB',
                     _s2_elapsed, _s2_peak / 1e6)
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S3 — LOCO kinship
    # always exports kinship summary; compares to R if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 3
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS3: LOCO kinship...')
            _rss_s3 = _get_rss_mb()
            tracemalloc.start()
            _t_s3 = time.perf_counter()

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
            k_py.to_csv(py_dir / 's3_kinship_summary.csv', index=False)
            _s3_n = len(py_rows)
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
                merged_k.to_csv(cmp_dir / 's3_kinship_cmp.csv', index=False)
                r_diag, _ = sp_stats.pearsonr(
                    merged_k['mean_diag'], merged_k['mean_diag_r'])
                r_off,  _ = sp_stats.pearsonr(
                    merged_k['mean_offdiag'], merged_k['mean_offdiag_r'])
                s3_ok = (_check('kinship diag r',    r_diag, 0.999) and
                         _check('kinship offdiag r', r_off,  0.999))
                results['S3_kinship'] = s3_ok
            else:
                results['S3_kinship'] = None

            _, _s3_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _s3_elapsed = time.perf_counter() - _t_s3
            _record_stats('S3', 'kinship_summary', _s3_n,
                          _s3_elapsed, _s3_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s3))
            log.info('  [timing] S3: %.2fs  heap_peak=%.0fMB',
                     _s3_elapsed, _s3_peak / 1e6)

            if _scan_stats:
                _sec_ord = {s: i for i, s in enumerate(_SECTION_ORDER)}
                _sorted_stats = sorted(_scan_stats,
                                       key=lambda r: _sec_ord.get(r['section'], 999))
                pd.DataFrame(_sorted_stats).to_csv(
                    py_dir / 'scan_stats.csv', index=False)

            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S4 — LMM scan  (PRIMARY GATE)
    # Modes: C (cis, each gene vs its own chr), G (genome-wide), L (local ±Mb)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 4
    py_lod_path  = py_dir / 's4_lod.csv'
    peak_lod_path = py_dir / 's4_peak_lod.csv'

    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS4: LMM scan  scan_type=%s  chrs=%s...',
                     scan_type, chrs_to_scan)

            import json as _json
            from importlib.metadata import version as _pkg_version
            (py_dir / 'scan_info.json').write_text(_json.dumps({
                'mosaiqtl_version': _pkg_version('mosaiqtl'),
                'git_hash':         _git_hash(),
                'run_date':         time.strftime('%Y-%m-%d'),
                'celltype':         celltype if celltype else 'whole_tissue',
                'scan_type':        scan_type,
                'chr':              ','.join(chrs_to_scan) if chrs_to_scan else 'all',
                'n_genes':          n_genes,
            }, indent=2))

            h5_path = data_dir / 'counts_lognorm.h5'
            all_effect_rows: list[dict] = []
            _s4_lod_header_written = False
            _rss_s4 = _get_rss_mb()
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

                gene_names_ch = ch_genes['gene_name'].tolist()   # symbol = HDF5 key = phenotype_id

                if n_genes is not None:
                    gene_names_ch = gene_names_ch[:n_genes]

                counts_ch, loaded_ids, sids_ch = _load_expression(h5_path, gene_names_ch)
                if counts_ch.shape[0] == 0:
                    log.info('  chr%s: no genes found in HDF5 — skipping', ch)
                    continue

                # map sample-level IDs → genotype IDs (strain names in cache)
                geno_ids_ch = [meta_geno_map.get(s, s) for s in sids_ch]

                loaded_names = loaded_ids  # symbol = phenotype_id; no transformation needed
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
                            start_lookup[grow['gene_name']] = float(grow['end']) / 1e6
                        else:
                            start_lookup[grow['gene_name']] = float(grow['start']) / 1e6

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
                    log.info('    chr%s: H precomputed (%d loci  %d samples)',
                             ch, len(_chr_data_ch['locus_ids']), _z_sub_ch.shape[0])

                log.info('  running LMM scan on chr%s ...', ch)
                t_chr = time.perf_counter()
                sex_ch = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sids_ch])
                for gi, (gene_id, gene_name) in enumerate(zip(loaded_ids, loaded_names)):
                    if gi % 100 == 0 and gi > 0:
                        elapsed = time.perf_counter() - t_scan
                        log.info('    chr%s %d/%d  %.0fs elapsed',
                                 ch, gi + 1, len(loaded_ids), elapsed)
                    y_raw      = counts_ch[gi].astype(float)
                    gene_start_mb = start_lookup.get(gene_name)

                    if _chr_precomputed and not np.any(np.isnan(y_raw[_vsid_ch])):
                        # precomputed path: sample-level K_n scan (matches R's K_n = Z K_m Z^T)
                        y_sub  = rint(y_raw[_vsid_ch])                          # (n,) sample-level
                        h2, _  = estimate_h2_null(
                            _evecs_ch.T @ y_sub, _evecs_ch.T @ _x_full_ch, _evals_ch)
                        chr_res = scan_chromosome(
                            y_sub, _chr_data_ch, _x_full_ch, _evals_ch, _evecs_ch, h2)
                        if _sw == 'local' and gene_start_mb is not None:
                            _mask = np.abs(chr_res['positions'] - gene_start_mb) <= window_mb
                            chr_res = {
                                k: (v[_mask] if isinstance(v, np.ndarray)
                                    and len(v) == len(_mask) else v)
                                for k, v in chr_res.items()
                            }
                            chr_res['locus_ids'] = np.asarray(chr_res['locus_ids'])[_mask]
                        n_loci_res = len(chr_res['locus_ids'])
                        scan = ScanResult(
                            locus_ids   = chr_res['locus_ids'],
                            chromosomes = np.full(n_loci_res, str(ch)),
                            positions   = chr_res['positions'],
                            lod         = chr_res['lod'],
                            p_value     = chr_res['p_value'],
                            betas       = chr_res['betas'],
                            h2_null     = {str(ch): h2},
                            phenotype_id = gene_name,
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
                            phenotype_id=gene_name,
                            scan_window=_sw,
                            gene_chr=ch,
                            gene_pos_mb=gene_start_mb if _sw == 'local' else None,
                            window_mb=window_mb,
                        )
                    df = _scan_to_df(scan, gene_name)
                    df.to_csv(py_lod_path,
                              mode='a',
                              header=not _s4_lod_header_written,
                              index=False)
                    _s4_lod_header_written = True

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
                            'gene':       gene_name,
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
                _, _s4_chr_peak = tracemalloc.get_traced_memory()
                _record_stats(f'S4_chr{ch}', f'lmm_scan_chr{ch}',
                              len(loaded_ids), elapsed_chr,
                              _s4_chr_peak / 1e6)
                log.info('    chr%s done: %d genes  %d loci  %.1fs  %.1f genes/min',
                         ch, len(loaded_ids), n_loci_ch, elapsed_chr, gpm_chr)

                n_genes_total += len(loaded_ids)
                del counts_ch
                if _chr_precomputed:
                    del _chr_data_ch, _evals_ch, _evecs_ch, _z_sub_ch, _r_counts_ch, _cov_valid
                gc.collect()

            elapsed_s = time.perf_counter() - t_scan
            _, _s4_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            if _s4_lod_header_written:
                genes_per_min = n_genes_total / elapsed_s * 60 if elapsed_s > 0 else 0
                log.info('  scan complete: %d genes  %.1fs  %.1f genes/min',
                         n_genes_total, elapsed_s, genes_per_min)
                _record_stats('S4', 'lmm_scan', n_genes_total, elapsed_s,
                              _s4_peak / 1e6, max(0.0, _get_rss_mb() - _rss_s4))
                # read back only the columns needed for peak reduction — avoids
                # holding the full (genes × loci) LOD matrix in RAM twice
                log.info('  computing peak LOD per gene ...')
                lod_df = pd.read_csv(py_lod_path, usecols=['gene', 'chr', 'locus', 'pos_mb', 'lod'])
            else:
                log.warning('  no LOD rows produced — check chr/gene_info alignment')
                lod_df = pd.DataFrame(columns=['gene', 'chr', 'locus', 'pos_mb', 'lod'])

            # save allele effects (peak-locus betas per gene)
            if all_effect_rows:
                eff_df = pd.DataFrame(all_effect_rows)
                eff_df.to_csv(py_dir / 's4_allele_effects.csv', index=False)
                log.info('  s4_allele_effects.csv: %d genes', len(eff_df))

            # peak LOD per gene — include chr/locus/pos so GxE loop has position without effects file
            if len(lod_df) > 0:
                peak_py = (lod_df.loc[lod_df.groupby('gene')['lod'].idxmax()]
                           [['gene', 'chr', 'locus', 'pos_mb', 'lod']]
                           .rename(columns={'lod': 'peak_lod',
                                            'locus': 'peak_locus',
                                            'pos_mb': 'peak_mb'})
                           .reset_index(drop=True))
                del lod_df; gc.collect()
                if all_effect_rows:
                    _h2_map       = {r['gene']: r['h2']       for r in all_effect_rows}
                    _start_mb_map = {r['gene']: r['start_mb'] for r in all_effect_rows}
                    _nv_map       = {r['gene']: r['n_valid']  for r in all_effect_rows}
                    _rss0_map     = {r['gene']: r['rss0']     for r in all_effect_rows}
                    _rss1_map     = {r['gene']: r['rss1']     for r in all_effect_rows}
                    _r2_map       = {r['gene']: r['r2_peak']  for r in all_effect_rows}
                    _sig2e_map    = {r['gene']: r['sigma2_e'] for r in all_effect_rows}
                    _tau2_map     = {r['gene']: r['tau2']     for r in all_effect_rows}
                    peak_py['h2']       = peak_py['gene'].map(_h2_map)
                    peak_py['start_mb'] = peak_py['gene'].map(_start_mb_map)
                    peak_py['n_valid']  = peak_py['gene'].map(_nv_map)
                    peak_py['rss0']     = peak_py['gene'].map(_rss0_map)
                    peak_py['rss1']     = peak_py['gene'].map(_rss1_map)
                    peak_py['r2_peak']  = peak_py['gene'].map(_r2_map)
                    peak_py['sigma2_e'] = peak_py['gene'].map(_sig2e_map)
                    peak_py['tau2']     = peak_py['gene'].map(_tau2_map)
                # ── near_tss_mb: transcript TSS nearest to peak_mb (post-peak join) ──
                # cis_dist_mb = peak_mb − near_tss_mb; falls back to start_mb if NaN
                if _gene_tss is not None and 'peak_mb' in peak_py.columns:
                    _near: dict[str, float] = {}
                    for _, _pr in peak_py.iterrows():
                        _cands = _gene_tss[_gene_tss['gene_name'] == _pr['gene']]
                        if _cands.empty:
                            _near[_pr['gene']] = float('nan')
                        else:
                            _idx = (_cands['tss_mb'] - _pr['peak_mb']).abs().idxmin()
                            _near[_pr['gene']] = float(_cands.loc[_idx, 'tss_mb'])
                    peak_py['near_tss_mb'] = peak_py['gene'].map(_near)
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
                _pk_cols = [
                    'gene', 'chr', 'start_mb', 'near_tss_mb', 'peak_locus', 'peak_mb',
                    'cis_dist_mb', 'peak_lod', 'h2', 'n_valid',
                    'rss0', 'rss1', 'r2_peak', 'sigma2_e', 'tau2',
                ]
                peak_py = peak_py.reindex(
                    columns=[c for c in _pk_cols if c in peak_py.columns])
                peak_py.to_csv(peak_lod_path, index=False)

                # ── compare to R reference ────────────────────────────────────
                # Prefer r_data/peak_lod.csv (merged genome-wide, one row per gene).
                # Fall back to per-chr lod.csv files if peak_lod.csv not yet written.
                r_peak_lod_path = r_data / 'peak_lod.csv'
                r_peak_rows = []
                if r_peak_lod_path.exists():
                    _rpl = pd.read_csv(r_peak_lod_path, usecols=['gene', 'lod'])
                    r_peak_rows.append(
                        _rpl.rename(columns={'lod': 'peak_lod_r'}))
                    del _rpl
                else:
                    for ch in chrs_to_scan:
                        r_lod_path_ch = r_data / f'chr{ch}' / 'lod.csv'
                        if r_lod_path_ch.exists():
                            r_lod_ch = pd.read_csv(r_lod_path_ch,
                                                    usecols=['gene', 'lod'])
                            pk_r = (r_lod_ch.groupby('gene')['lod'].max()
                                    .reset_index()
                                    .rename(columns={'lod': 'peak_lod_r'}))
                            r_peak_rows.append(pk_r)
                            del r_lod_ch; gc.collect()

                if r_peak_rows:
                    peak_r = pd.concat(r_peak_rows, ignore_index=True)
                    peak_r = (peak_r.groupby('gene')['peak_lod_r'].max()
                              .reset_index())
                    cmp = peak_py.merge(peak_r, on='gene', how='inner')
                    # add _py suffix only in the comparison CSV for side-by-side readability
                    cmp = cmp.rename(columns={'peak_lod': 'peak_lod_py'})
                    cmp.to_csv(cmp_dir / 's4_peak_lod_comparison.csv', index=False)
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
                        s4_ok = _check('peak LOD r²', r2, 0.975)
                        _check('slope', abs(slope - 1.0), 0.05, '<')
                        log.info('  r²=%.4f  slope=%.3f  intercept=%.3f  '
                                 'max|Δ|=%.3f  n=%d',
                                 r2, slope, intercept, max_delta, n_matched)

                        _sparse_s4 = (
                            cmp['gene'].isin(low_expr_genes)
                            if low_expr_genes and 'gene' in cmp.columns else None
                        )
                        fig, _ax = _cmp_scatter(
                            cmp, 'peak_lod_r', 'peak_lod_py',
                            'miQTL (R) peak LOD', 'mosaiQTL (Python) peak LOD',
                            priority_genes={'Cwc22': FOUNDER_COLORS['CASTJ']},
                            sparse_mask=_sparse_s4,
                            save_path=plt_dir / 's4_peak_lod_scatter.png',
                        )
                        plt.close(fig)

                        results['S4_lod_r2'] = s4_ok
                        results['S4_slope_ok'] = abs(slope - 1.0) < 0.05
                    else:
                        log.warning('  too few matched genes for r²')
                        results['S4_lod_r2'] = False
                else:
                    log.info('  %s no R chr*/lod.csv found — r² comparison skipped', SKIP)
                    results['S4_lod_r2'] = None
            else:
                results['S4_lod_r2'] = None
            _mark_done(py_dir, str(sec))
    elif py_lod_path.exists():
        log.info('[S4] loading cached s4_lod.csv')
        # reload gate result and scan stats from their checkpoint CSVs so the
        # scan summary at the end of the run shows real values, not N/A
        _gate_csv = py_dir / 'gate_summary.csv'
        if _gate_csv.exists():
            try:
                _gc = pd.read_csv(_gate_csv)
                _s4_row = _gc[_gc['gate'] == 'S4_lod_r2']
                if len(_s4_row) > 0:
                    _rv = str(_s4_row.iloc[0]['result'])
                    if _rv == 'True':
                        results['S4_lod_r2'] = True
                    elif _rv == 'False':
                        results['S4_lod_r2'] = False
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
    chr_null_lods_s5: 'dict | None' = None
    perm_max_lods_s5: 'np.ndarray | None' = None
    perm_n_s5: int = 0

    # ─────────────────────────────────────────────────────────────────────────
    # S5 — genome-wide permutation threshold (Churchill & Doerge 1994)
    # Representative gene: median-h² gene across all scanned genes.
    # Rationale: fix_h2=True makes the null max-LOD distribution approximately
    # gene-independent given h²; median-h² gives a threshold accurate on average
    # across genes rather than calibrated only for the extreme (max-LOD) gene.
    # A single genome-wide max-LOD vector (all chromosomes, one permuted phenotype)
    # → one GEV threshold applied uniformly across all chromosomes.
    # ─────────────────────────────────────────────────────────────────────────
    sec = 5
    r_perm_lod_path = r_data / f'perm_max_lods_chr{peak_chr}.csv'
    if sec in sections:
        log.info('\nS5: genome-wide permutation threshold (1000 perms, median-h² gene)...')
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s5 = _get_rss_mb()
            tracemalloc.start()
            _t_s5 = time.perf_counter()
            if not peak_lod_path.exists():
                log.info('  %s s4_peak_lod.csv not found — run S4 first', SKIP)
                results['S5_perm_threshold'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'

                # ── pick median-h² gene as representative for permutation ──────
                # Prefer h² from s4_allele_effects.csv if the column exists
                # (written when S4 was run with the current code).  When the
                # on-disk file predates the h² column, estimate h² independently
                # using estimate_h2_null on a random gene sample — fast (Brent's
                # 1D optimisation, no scan) and makes S5 self-contained.
                _perm_gene: str
                _perm_id:   str
                _h2_med:    float

                eff_path_s5 = py_dir / 's4_allele_effects.csv'
                _h2_from_effects = False
                if eff_path_s5.exists():
                    _eff_cols = pd.read_csv(eff_path_s5, nrows=0).columns.tolist()
                    if 'h2' in _eff_cols:
                        _h2_from_effects = True

                if _h2_from_effects:
                    _eff_s5    = pd.read_csv(eff_path_s5, usecols=['gene', 'h2'])
                    _h2_valid  = _eff_s5['h2'].dropna()
                    _h2_med    = float(_h2_valid.median())
                    _best_row  = _eff_s5.iloc[(_h2_valid - _h2_med).abs().values.argmin()]
                    _perm_gene = str(_best_row['gene'])
                    _perm_id   = _perm_gene
                    # join peak_lod for chr + LOD of representative gene
                    _pk_s5_info = (pd.read_csv(peak_lod_path, usecols=['gene', 'chr', 'peak_lod'])
                                   if peak_lod_path.exists() else None)
                    _rep_info = (
                        _pk_s5_info[_pk_s5_info['gene'] == _perm_gene].iloc[0]
                        if _pk_s5_info is not None and (_pk_s5_info['gene'] == _perm_gene).any()
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
                    _pk_s5     = pd.read_csv(peak_lod_path)
                    _sub_s5    = _pk_s5[_pk_s5['chr'].astype(str) == str(peak_chr)]
                    _samp_s5   = (_sub_s5['gene']
                                  .sample(min(500, len(_sub_s5)), random_state=1)
                                  .tolist())
                    _samp_ids  = _samp_s5
                    _cts_s5, _lds_s5, _sids_s5 = _load_expression(h5_path, _samp_ids)
                    _gids_s5   = [meta_geno_map.get(s, s) for s in _sids_s5]
                    _kd_ref    = kinship_decomp[peak_chr]
                    _ev_ref    = _kd_ref['values']
                    _evc_ref   = _kd_ref['vectors']
                    # restrict to samples present in cache
                    _vsid_s5   = np.array([i for i, g in enumerate(_gids_s5)
                                           if g in cache.sample_ids])
                    _h2_pairs: list[tuple[str, float]] = []
                    for _gi5, _gn5 in enumerate(_lds_s5):
                        _y5 = rint(_cts_s5[_gi5].astype(float))
                        if np.any(np.isnan(_y5[_vsid_s5])):
                            continue
                        try:
                            _cov5 = _build_covariates(meta, [_sids_s5[j]
                                                              for j in _vsid_s5])
                            _x5   = np.hstack([np.ones((_cov5.shape[0], 1)), _cov5])
                            _h5, _ = estimate_h2_null(
                                _evc_ref.T @ _y5[_vsid_s5],
                                _evc_ref.T @ _x5,
                                _ev_ref)
                            _h2_pairs.append((_gn5, float(_h5)))
                        except Exception:
                            pass
                    del _cts_s5; gc.collect()
                    if not _h2_pairs:
                        log.warning('  h² estimation failed for all sampled genes; '
                                    'falling back to highest-LOD gene on chr%s', peak_chr)
                        _perm_gene = str(
                            _sub_s5.loc[_sub_s5['peak_lod'].idxmax(), 'gene'])
                        _h2_med    = float('nan')
                    else:
                        _h2_arr    = np.array([h for _, h in _h2_pairs])
                        _h2_med    = float(np.median(_h2_arr))
                        _best_idx  = int(np.argmin(np.abs(_h2_arr - _h2_med)))
                        _perm_gene = _h2_pairs[_best_idx][0]
                    _perm_id = _perm_gene
                    log.info('  median h²=%.3f  representative gene: %s', _h2_med, _perm_gene)

                _cts, _lds, _sids = _load_expression(h5_path, [_perm_id])
                if _cts.shape[0] == 0:
                    log.info('  %s expression for %s not found', SKIP, _perm_id)
                    results['S5_perm_threshold'] = None
                else:
                    _gids = [meta_geno_map.get(s, s) for s in _sids]
                    _y    = _cts[0].astype(float)

                    # genome-wide scan: pass full kinship_decomp so max is taken
                    # across ALL chromosomes per permutation replicate
                    _null = generate_null_outcomes(_y, n_perms=1000, seed=1)
                    _t_s5_scan = time.perf_counter()
                    _gw_max, _chr_nulls = run_threshold_scans(
                        _null, cache,
                        phenotype=_y,
                        covariates=None,
                        kinship_decomp=kinship_decomp,
                        chrom=None,
                        sample_ids=_gids,
                    )
                    _scan_elapsed_s5 = time.perf_counter() - _t_s5_scan
                    chr_null_lods_s5 = _chr_nulls
                    perm_max_lods_s5 = _gw_max
                    perm_n_s5        = int(np.isfinite(_gw_max).sum())
                    _perms_per_hr_s5 = round(perm_n_s5 / _scan_elapsed_s5 * 3600, 1)

                    _gw_thr, _ = get_gev_thresholds(_gw_max, percentile=0.95)
                    log.info('  genome-wide GEV threshold=%.3f  (n_perms=%d)',
                             _gw_thr, perm_n_s5)
                    log.info('  scan loop: %.0fs for %d perms  (%.1f perms/hr)',
                             _scan_elapsed_s5, perm_n_s5, _perms_per_hr_s5)

                    # save genome-wide null distribution (single file)
                    pd.DataFrame({'max_lod': _gw_max}).to_csv(
                        py_dir / 's5_perm_max_lods.csv', index=False)

                    # save per-chromosome nulls for distribution plots
                    for _c5, _c5_null in _chr_nulls.items():
                        pd.DataFrame({'max_lod': _c5_null}).to_csv(
                            py_dir / f's5_perm_max_lods_chr{_c5}.csv', index=False)

                    # single-row threshold file with genome_wide sentinel
                    # scan_elapsed_s / perms_per_hr mirror R's thresholds_genome.csv
                    # for direct speed comparison
                    pd.DataFrame([{
                        'chr':            'genome_wide',
                        'threshold':      _gw_thr,
                        'perm_gene':      _perm_gene,
                        'h2_median':      _h2_med,
                        'n_perms':        perm_n_s5,
                        'scan_elapsed_s': round(_scan_elapsed_s5, 1),
                        'perms_per_hr':   _perms_per_hr_s5,
                    }]).to_csv(py_dir / 's5_threshold.csv', index=False)

                    # compare to R peak-chr permutation (approximate — different scope)
                    if r_perm_lod_path.exists():
                        r_max_lods = pd.read_csv(r_perm_lod_path)['max_lod'].values
                        r_thr, _   = get_gev_thresholds(r_max_lods, percentile=0.95)
                        log.info('  R chr%s GEV=%.3f  delta vs genome-wide=%.3f'
                                 '  (expected: gw > per-chr)', peak_chr, r_thr,
                                 _gw_thr - r_thr)

                    results['S5_perm_threshold'] = True
                    n_genes_s5_total = len(_eff_s5)

            _, _s5_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S5', 'permutation', n_genes_s5_total if 'n_genes_s5_total' in locals() else 0,
                          time.perf_counter() - _t_s5, _s5_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s5))
            _mark_done(py_dir, str(sec))

    # ── eGene calling (after S5 — genome-wide threshold from s5_threshold.csv)
    sec = 6
    eqtl_summary_path = py_dir / 'eqtl_summary.csv'
    if peak_lod_path.exists() and not _checkpoint(py_dir, str(sec), force):
        log.info('\n── eGene calling + GxE/PxG ──────────────────────────────────────')
        peak_py_loaded = pd.read_csv(peak_lod_path)
        py_thr_path = py_dir / 's5_threshold.csv'
        thr: 'float | None' = None
        if py_thr_path.exists():
            _thr_df  = pd.read_csv(py_thr_path)
            _thr_map = {str(r['chr']): float(r['threshold']) for _, r in _thr_df.iterrows()}
            # genome-wide sentinel: one threshold applied to all chromosomes
            _gw_thr_val = _thr_map.get('genome_wide')
            thr = _gw_thr_val if _gw_thr_val is not None else _thr_map.get(str(peak_chr))
            if thr is not None:
                eff_path = py_dir / 's4_allele_effects.csv'
                eff_tmp: 'pd.DataFrame | None' = None
                if eff_path.exists():
                    eff_tmp = pd.read_csv(
                        eff_path,
                        usecols=['gene', 'chr', 'peak_locus', 'peak_mb', 'peak_lod', 'h2'])
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
            if perm_max_lods_s5 is None:
                _pml_path = py_dir / 's5_perm_max_lods.csv'
                if _pml_path.exists():
                    perm_max_lods_s5 = pd.read_csv(_pml_path)['max_lod'].values
                    perm_n_s5 = len(perm_max_lods_s5)
                    # load all per-chromosome nulls saved during S5 run
                    chr_null_lods_s5 = {}
                    for _f in sorted(
                        py_dir.glob('s5_perm_max_lods_chr*.csv'),
                        key=lambda p: _chr_sort_key(
                            p.stem.replace('s5_perm_max_lods_chr', '')
                        ),
                    ):
                        _ch_key = _f.stem.replace('s5_perm_max_lods_chr', '')
                        chr_null_lods_s5[_ch_key] = pd.read_csv(_f)['max_lod'].values
                    if not chr_null_lods_s5:
                        chr_null_lods_s5 = {peak_chr: perm_max_lods_s5}

            # ── genome-wide FWER calling only (Storey q-value disabled) ─────────
            # Storey q-value requires each gene's p-value to be valid under its
            # own null. S5 derives the permutation null (and every per-chromosome
            # null) from ONE representative gene's variance components, shared
            # across all 43k genes — a fine simplification for a single global
            # FWER threshold, but invalid as a per-gene null for FDR: pi0_hat(λ)
            # climbed past 1.0 monotonically to 2.58 at λ=0.95 instead of
            # plateauing, and 12.9% of genes had p>0.95 vs the 5% a true uniform
            # null implies. Disabled until S5 produces a per-gene-calibrated null.
            # eGene calling above already uses the genome-wide GEV threshold
            # (`thr` from s5_threshold.csv) — that FWER gate is unaffected.
            if perm_max_lods_s5 is not None:
                egenes_df['perm_n']   = perm_n_s5
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
                _r_thr = pd.read_csv(r_thr_genome_path)
                # column may be threshold_gev or threshold
                _thr_col = next((c for c in ('threshold_gev', 'threshold')
                                 if c in _r_thr.columns), None)
                if _thr_col is not None:
                    _r_gev = float(_r_thr[_thr_col].iloc[0])
                    # peak_lod.csv column may be lod or peak_lod
                    _lod_col = 'peak_lod' if 'peak_lod' in _r_pk.columns else 'lod'
                    egenes_r = _r_pk[_r_pk[_lod_col] >= _r_gev][['gene', _lod_col]].rename(
                        columns={_lod_col: 'lod'})
                    log.info('[eGene R]  %d R eGenes at threshold=%.3f (from peak_lod.csv)',
                             len(egenes_r), _r_gev)
            if egenes_r is not None:
                eqtl_sum['in_r_egenes'] = eqtl_sum['gene'].isin(egenes_r['gene'])
                # ── R vs Python eGene LOD comparison ──────────────────────────
                _cmp = egenes_r.merge(
                    eqtl_sum[['gene', 'peak_lod']].rename(columns={'peak_lod': 'lod_py'}),
                    on='gene', how='inner',
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
                            _row['gene'], _row['lod_r'], _row['lod_py'],
                            _row['lod_r'] / _row['lod_py'],
                        )
            # add h2, start_mb from allele effects (computed during S4 scan)
            _eff_sum_path = py_dir / 's4_allele_effects.csv'
            if _eff_sum_path.exists():
                _eff_cols = ['gene', 'h2']
                _eff_peek = pd.read_csv(_eff_sum_path, nrows=0).columns.tolist()
                if 'start_mb' in _eff_peek:
                    _eff_cols.append('start_mb')
                elif 'tss_mb' in _eff_peek:
                    _eff_cols.append('tss_mb')
                _h2_sum = pd.read_csv(_eff_sum_path, usecols=_eff_cols)
                if 'tss_mb' in _h2_sum.columns:
                    _h2_sum = _h2_sum.rename(columns={'tss_mb': 'start_mb'})
                eqtl_sum = eqtl_sum.merge(_h2_sum, on='gene', how='left')

            # join near_tss_mb from s4_peak_lod (computed post-peak during S4 scan)
            if peak_lod_path.exists():
                _pk_cols_eq = ['gene']
                _pk_peek = pd.read_csv(peak_lod_path, nrows=0).columns.tolist()
                for _c in ('near_tss_mb', 'start_mb'):
                    if _c in _pk_peek and _c not in eqtl_sum.columns:
                        _pk_cols_eq.append(_c)
                if len(_pk_cols_eq) > 1:
                    _pk_tss = pd.read_csv(peak_lod_path, usecols=_pk_cols_eq)
                    eqtl_sum = eqtl_sum.merge(_pk_tss, on='gene', how='left')

            # cis_dist_mb = peak_mb − near_tss_mb; falls back to start_mb if NaN
            if 'near_tss_mb' in eqtl_sum.columns:
                eqtl_sum['cis_dist_mb'] = (
                    eqtl_sum['peak_mb'] - eqtl_sum['near_tss_mb']).round(6)
                if 'start_mb' in eqtl_sum.columns:
                    _nan_mask = eqtl_sum['near_tss_mb'].isna()
                    eqtl_sum.loc[_nan_mask, 'cis_dist_mb'] = (
                        eqtl_sum.loc[_nan_mask, 'peak_mb']
                        - eqtl_sum.loc[_nan_mask, 'start_mb']).round(6)
            elif 'start_mb' in eqtl_sum.columns:
                eqtl_sum['cis_dist_mb'] = (
                    eqtl_sum['peak_mb'] - eqtl_sum['start_mb']).round(6)

            # chi2_haplotype and p_haplotype derived from peak_lod (no extra scan)
            # df = J-1 = 7 for 8-founder additive model
            eqtl_sum['chi2_haplotype'] = 2.0 * np.log(10) * eqtl_sum['peak_lod']
            eqtl_sum['p_haplotype'] = sp_stats.chi2.sf(
                eqtl_sum['chi2_haplotype'].values, df=7)

            col_order = ['gene', 'chr', 'start_mb', 'near_tss_mb', 'peak_locus', 'peak_mb',
                         'cis_dist_mb', 'peak_lod', 'h2',
                         'perm_n', 'perm_chr',
                         'chi2_haplotype', 'p_haplotype',
                         'in_r_egenes']
            eqtl_sum = eqtl_sum[[c for c in col_order if c in eqtl_sum.columns]]
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
        _gm_cols = [c for c in ['gene', 'near_tss_mb', 'cis_dist_mb', 'peak_lod', 'peak_mb']
                    if c in egenes_df.columns]
        _gene_meta = (egenes_df[_gm_cols]
                      .drop_duplicates('gene')
                      .rename(columns={'peak_lod': 'peak_lod_lmm', 'peak_mb': 'peak_mb_lmm'})
                      .copy())
        _peak_mb_map = {
            str(r['gene']): float(r['peak_mb_lmm'])
            for _, r in _gene_meta.iterrows()
            if pd.notna(r.get('peak_mb_lmm'))
        }

    # ─────────────────────────────────────────────────────────────────────────
    # S7 — MI scan (significant genes on peak_chr)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 7
    r_mi_path = r_data / 'mi_lod_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s7 = _get_rss_mb()
            tracemalloc.start()
            _t_s7 = time.perf_counter()
            log.info('\nS7: MI scan...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S7_mi_scan'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _s7_ids = [r['gene']
                           for _, r in egenes_df.iterrows()]
                _s7_counts, _s7_loaded, _s7_sids = _load_expression(h5_path, _s7_ids)
                _s7_expr = {gid: _s7_counts[i].astype(float)
                            for i, gid in enumerate(_s7_loaded)}
                _s7_chr_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene']
                    _s7_chr_map[gid] = str(erow.get('chr', peak_chr))
                geno_ids_s7 = [meta_geno_map.get(s, s) for s in _s7_sids]
                cov_s7 = _build_covariates(meta, _s7_sids)
                _s7_path = py_dir / 's7_mi_lod.csv'
                log.info('  scanning %d eGenes via MI (streaming)...', len(_s7_expr))
                mi_df = scan_multi_gene_mi(
                    _s7_expr, _s7_chr_map, cache,
                    covariates=cov_s7,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_s7,
                    n_imp=11, seed=1,
                    peak_mb_map=_peak_mb_map,
                    output_path=_s7_path,
                    gene_meta=_gene_meta,
                )
                del _s7_expr, _s7_counts
                if not mi_df.empty:
                    log.info('  MI scan complete: %d genes', mi_df['gene'].nunique())
                    results['S7_mi_scan'] = True
                    if r_mi_path.exists():
                        r_mi = pd.read_csv(r_mi_path)
                        # peak_only=True → one row per gene; compare peak LOD per gene
                        r_mi_peak = (r_mi.groupby('gene')['lod'].max()
                                     .reset_index().rename(columns={'lod': 'r'}))
                        merged_mi = (mi_df[['gene', 'peak_lod']]
                                     .rename(columns={'peak_lod': 'py'})
                                     .merge(r_mi_peak, on='gene', how='inner'))
                        if len(merged_mi) >= 10:
                            r_mi_r, _ = sp_stats.pearsonr(merged_mi['py'],
                                                           merged_mi['r'])
                            merged_mi.to_csv(cmp_dir / 's7_mi_cmp.csv', index=False)
                            _check('MI LOD r', abs(r_mi_r), 0.75)
                            log.info('  vs R: r=%.4f  n=%d', r_mi_r, len(merged_mi))
                            fig, _ax = _cmp_scatter(
                                merged_mi, 'r', 'py',
                                'MI LOD (R)', 'MI LOD (Python)',
                                save_path=plt_dir / 's7_mi_scatter.png',
                            )
                            plt.close(fig)
                else:
                    results['S7_mi_scan'] = None
            _, _s7_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S7', 'mi_scan', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_s7, _s7_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s7))
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S8 — SNP scan (gated: allele_db required)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 8
    r_snp_path = r_data / 'snp_lod_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s8 = _get_rss_mb()
            tracemalloc.start()
            _t_s8 = time.perf_counter()
            log.info('\nS8: SNP scan...')
            if not ALLELE_H5.exists():
                log.info('  %s founder_alleles.h5 not found at %s — '
                         'build with build_founder_allele_db()', SKIP, ALLELE_H5)
                results['S8_snp_lod_r'] = None
            elif egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S8_snp_lod_r'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _s8_ids = [r['gene']
                           for _, r in egenes_df.iterrows()]
                _s8_counts, _s8_loaded, _s8_sids = _load_expression(h5_path, _s8_ids)
                _s8_expr = {gid: _s8_counts[i].astype(float)
                            for i, gid in enumerate(_s8_loaded)}
                _s8_chr_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene']
                    _s8_chr_map[gid] = str(erow.get('chr', peak_chr))
                geno_ids_s8 = [meta_geno_map.get(s, s) for s in _s8_sids]
                cov_s8 = _build_covariates(meta, _s8_sids)
                _s8_path = py_dir / 's8_snp_lod.csv'
                log.info('  scanning %d eGenes via SNP (streaming)...', len(_s8_expr))
                snp_df = snp_scan_multi_gene(
                    _s8_expr, _s8_chr_map, cache,
                    alleles_h5=str(ALLELE_H5),
                    covariates=cov_s8,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_s8,
                    founder_col_map=FOUNDER_CODE,
                    output_path=_s8_path,
                    gene_meta=_gene_meta,
                )
                del _s8_expr, _s8_counts
                if not snp_df.empty:
                    if r_snp_path.exists():
                        r_snp = pd.read_csv(r_snp_path)
                        # peak_only=True → one row per gene; compare peak SNP LOD per gene
                        r_snp_peak = (r_snp.groupby('gene')['lod'].max()
                                      .reset_index().rename(columns={'lod': 'r'}))
                        merged_snp = (snp_df[['gene', 'peak_lod']]
                                      .rename(columns={'peak_lod': 'py'})
                                      .merge(r_snp_peak, on='gene', how='inner'))
                        if len(merged_snp) >= 10:
                            r_snp_r, _ = sp_stats.pearsonr(merged_snp['py'],
                                                            merged_snp['r'])
                            slope_snp, *_ = sp_stats.linregress(
                                merged_snp['r'], merged_snp['py'])
                            merged_snp.to_csv(cmp_dir / 's8_snp_comparison.csv',
                                              index=False)
                            log.info('  r=%.4f  slope=%.3f  n=%d/%d',
                                     r_snp_r, slope_snp,
                                     len(merged_snp), len(r_snp))
                            fig, _ax = _cmp_scatter(
                                merged_snp, 'r', 'py',
                                'SNP LOD (R)', 'SNP LOD (Python)',
                                save_path=plt_dir / 's8_snp_scatter.png',
                            )
                            plt.close(fig)
                            results['S8_snp_lod_r'] = (abs(r_snp_r) >= 0.75 and
                                                        abs(slope_snp - 1.0) < 0.20)
                        else:
                            log.info('  %s too few matched SNPs', SKIP)
                            results['S8_snp_lod_r'] = None
                    else:
                        log.info('  Python SNP scan complete (%d genes) '
                                 '— no R reference yet (run gastroc_miqtl_scan.R S8)',
                                 snp_df['gene'].nunique())
                        results['S8_snp_lod_r'] = None
                else:
                    log.info('  %s SNP scan produced no results', SKIP)
                    results['S8_snp_lod_r'] = None
            _, _s8_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S8', 'snp_scan', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_s8, _s8_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s8))
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S9 — mediation scan
    # always runs on eGenes; compares to R mediation_genome.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 9
    r_med_path = r_data / 'mediation_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s9 = _get_rss_mb()
            tracemalloc.start()
            _t_s9 = time.perf_counter()
            log.info('\nS9: mediation scan...')
            if egenes_df is None or len(egenes_df) < 2:
                log.info('  %s need >=2 eGenes — run S4+S5 first', SKIP)
                results['S9_mediation'] = None
            elif 'chr' not in egenes_df.columns or 'peak_mb' not in egenes_df.columns:
                log.info('  %s egenes_df missing chr/peak_mb — run S4 first to build effects', SKIP)
                results['S9_mediation'] = None
            else:
                h5_path  = data_dir / 'counts_lognorm.h5'
                # restrict mediator pool to eGenes: a valid mediator must itself be
                # haplotype-regulated — non-eGenes have no haplotype effect to mediate
                egene_names_set = set(egenes_df['gene'].dropna())
                gene_info_egenes = gene_info[gene_info['gene_name'].isin(egene_names_set)]
                pairs    = select_cis_mediation_pairs(
                    egenes_df, gene_info_egenes,
                    window_mb=10.0,
                    info_gene_col='gene_name',
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
                _s9_all_names: set[str] = set(_pairs_by_outcome.keys())
                for _ml in _pairs_by_outcome.values():
                    _s9_all_names.update(_mn for _mn, _ in _ml)
                _s9_all_ids = list(dict.fromkeys(_s9_all_names))
                _s9_counts, _s9_loaded, _s9_sids = _load_expression(h5_path, _s9_all_ids)
                _s9_expr = {gid: _s9_counts[i].astype(float)
                            for i, gid in enumerate(_s9_loaded)}
                del _s9_counts
                log.info('  pre-loaded %d gene expressions (%d unique IDs) in 1 HDF5 call',
                         len(_s9_loaded), len(_s9_all_ids))

                geno_ids_s9 = [meta_geno_map.get(s, s) for s in _s9_sids]
                cov_s9 = _build_covariates(meta, _s9_sids)
                _med_path = py_dir / 's9_mediation.csv'
                log.info('  running mediation for %d outcomes (streaming)...',
                         len(_pairs_by_outcome))
                med_df = mediation_scan_multi_outcome(
                    _pairs_by_outcome, _s9_expr, cache,
                    covariates=cov_s9,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_s9,
                    output_path=_med_path,
                    gene_meta=_gene_meta,
                )
                del _s9_expr
                if not med_df.empty:
                    log.info('  mediation complete: %d genes', med_df['gene'].nunique())
                    results['S9_mediation'] = True
                    if r_med_path.exists():
                        r_med = pd.read_csv(r_med_path)
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
                            merged_med.to_csv(cmp_dir / 's9_mediation_cmp.csv',
                                              index=False)
                            _check('mediation LOD r', abs(r_m), 0.90)
                            log.info('  vs R: r=%.4f  n=%d', r_m, len(merged_med))
                            fig, _ax = _cmp_scatter(
                                merged_med, 'r', 'py',
                                'Mediated LOD (R)', 'Mediated LOD (Python)',
                                save_path=plt_dir / 's9_mediation_scatter.png',
                            )
                            plt.close(fig)
                else:
                    results['S9_mediation'] = None
            _, _s9_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S9', 'mediation', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_s9, _s9_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s9))
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S10 — bootstrap CI
    # always runs on eGenes; compares to R ci_summary_genome.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 10
    r_ci_path = r_data / 'ci_summary_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s10 = _get_rss_mb()
            tracemalloc.start()
            _t_s10 = time.perf_counter()
            log.info('\nS10: bootstrap CI...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S10_ci'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _s10_ids = [r['gene']
                            for _, r in egenes_df.iterrows()]
                _s10_counts, _s10_loaded, _s10_sids = _load_expression(h5_path, _s10_ids)
                _s10_expr = {gid: _s10_counts[i].astype(float)
                             for i, gid in enumerate(_s10_loaded)}
                del _s10_counts
                _s10_chr_map = {}
                _s10_peak_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene']
                    _s10_chr_map[gid] = str(erow.get('chr', peak_chr))
                    _s10_peak_map[gid] = float(erow.get('peak_mb', np.nan))
                geno_ids_s10 = [meta_geno_map.get(s, s) for s in _s10_sids]
                cov_s10 = _build_covariates(meta, _s10_sids)
                _s10_path = py_dir / 's10_ci.csv'
                log.info('  bootstrap CI for %d eGenes (streaming)...', len(_s10_expr))
                ci_df = ci_scan_multi_gene(
                    _s10_expr, _s10_chr_map, _s10_peak_map, cache,
                    covariates=cov_s10,
                    kinship_decomp=kinship_decomp,
                    sample_ids=geno_ids_s10,
                    n_boot=200, seed=1,
                    output_path=_s10_path,
                    gene_meta=_gene_meta,
                )
                del _s10_expr
                if not ci_df.empty:
                    log.info('  CI complete: %d genes  median width=%.2f Mb',
                             len(ci_df),
                             float(np.median(ci_df['ci_width_mb'])))
                    results['S10_ci'] = True
                    if r_ci_path.exists():
                        r_ci      = pd.read_csv(r_ci_path)
                        merged_ci = ci_df.merge(r_ci, on='gene', how='inner',
                                                suffixes=('_py', '_r'))
                        if len(merged_ci) > 0:
                            # Python cols: ci_lo, ci_hi; R cols: ci_lo_mb, ci_hi_mb
                            # No duplicate names → no suffix applied; reference directly
                            overlap = (
                                (merged_ci['ci_lo'] <= merged_ci['ci_hi_mb']) &
                                (merged_ci['ci_hi'] >= merged_ci['ci_lo_mb'])
                            ).mean()
                            merged_ci.to_csv(cmp_dir / 's10_ci_cmp.csv', index=False)
                            log.info('  vs R: CI overlap=%.2f  n=%d',
                                     overlap, len(merged_ci))
                else:
                    results['S10_ci'] = None
            _, _s10_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S10', 'bootstrap_ci', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_s10, _s10_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s10))
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S11 — instability scan
    # always runs on eGenes; compares to R instability_genome.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 11
    r_inst_path = r_data / 'instability_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s11 = _get_rss_mb()
            tracemalloc.start()
            _t_s11 = time.perf_counter()
            log.info('\nS11: instability scan...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S11_instability'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                _s11_ids = [r['gene']
                            for _, r in egenes_df.iterrows()]
                _s11_counts, _s11_loaded, _s11_sids = _load_expression(h5_path, _s11_ids)
                _s11_expr = {gid: _s11_counts[i].astype(float)
                             for i, gid in enumerate(_s11_loaded)}
                del _s11_counts
                _s11_chr_map = {}
                for _, erow in egenes_df.iterrows():
                    gid = erow['gene']
                    _s11_chr_map[gid] = str(erow.get('chr', peak_chr))
                geno_ids_s11 = [meta_geno_map.get(s, s) for s in _s11_sids]
                cov_s11 = _build_covariates(meta, _s11_sids)
                _s11_path = py_dir / 's11_instability.csv'
                log.info('  instability scan for %d eGenes (streaming)...', len(_s11_expr))
                inst_df = instability_scan_multi_gene(
                    _s11_expr, _s11_chr_map, cache,
                    covariates=cov_s11,
                    sample_ids=geno_ids_s11,
                    n_subsamples=100, seed=1,
                    peak_mb_map=_peak_mb_map,
                    output_path=_s11_path,
                    gene_meta=_gene_meta,
                )
                del _s11_expr
                if not inst_df.empty:
                    log.info('  instability complete: %d genes', inst_df['gene'].nunique())
                    results['S11_instability'] = True
                    if r_inst_path.exists():
                        r_inst = pd.read_csv(r_inst_path)
                        # peak_only=True → one row per gene; compare peak instability per gene
                        # R: neg_log10_p (median -log10 F-test p); Python: peak_lod
                        # Both monotone with instability signal — use Spearman (scale-invariant)
                        r_inst_peak = (r_inst.groupby('gene')['neg_log10_p'].max()
                                       .reset_index().rename(columns={'neg_log10_p': 'r'}))
                        merged_inst = (inst_df[['gene', 'peak_lod']]
                                       .rename(columns={'peak_lod': 'py'})
                                       .merge(r_inst_peak, on='gene', how='inner'))
                        if len(merged_inst) >= 10:
                            r_i, _ = sp_stats.spearmanr(merged_inst['py'],
                                                          merged_inst['r'])
                            merged_inst.to_csv(cmp_dir / 's11_instability_cmp.csv',
                                               index=False)
                            _check('instability rho', abs(r_i), 0.90)
                            log.info('  vs R (Spearman): rho=%.4f  n=%d'
                                     '  [py=peak_lod, r=peak neg_log10_p — rank correlation]',
                                     r_i, len(merged_inst))
                            fig, _ax = _cmp_scatter(
                                merged_inst, 'r', 'py',
                                'Instability −log₁₀p (R)', 'Instability LOD (Python)',
                                identity_line=False,
                                save_path=plt_dir / 's11_instability_scatter.png',
                            )
                            plt.close(fig)
                else:
                    results['S11_instability'] = None
            _, _s11_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S11', 'instability', len(egenes_df) if egenes_df is not None else 0,
                          time.perf_counter() - _t_s11, _s11_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s11))
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # gate summary + run_summary.csv
    # ─────────────────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - _t0
    log.info('\n── GATE SUMMARY ──────────────────────────────────────────────')

    gate_rows = []
    gate_labels = {
        'S1_cache':          'S1  cache characterization',
        'S2_haplotype':      'S2  haplotype probabilities',
        'S3_kinship':        'S3  LOCO kinship',
        'S4_lod_r2':         f'S4  {scan_type} peak LOD r²  (PRIMARY GATE)',
        'S5_perm_threshold': 'S5  permutation threshold',
        'S7_mi_scan':        'S7  MI scan',
        'S8_snp_lod_r':      'S8  SNP scan',
        'S9_mediation':      'S9  mediation scan',
        'S10_ci':            'S10 bootstrap CI',
        'S11_instability':   'S11 instability scan',
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

    pd.DataFrame(gate_rows).to_csv(py_dir / 'gate_summary.csv', index=False)

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
        's4_lod_r2':        results.get('S4_lod_r2'),
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
            plt_dir=plt_dir, py_dir=py_dir,
            cache=cache, kinship_decomp=kinship_decomp,
            founders=list(cache.founders), peak_chr=peak_chr,
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
            )
        except Exception as exc:
            log.warning('  comparison plots raised: %s', exc)

    # ── detailed scan summary ─────────────────────────────────────────────────
    n_loci_total = sum(
        len(cache.get_chromosome(c)[0])
        for c in chrs_to_scan
    ) if chrs_to_scan else 0
    r4_val  = results.get('S4_lod_r2')
    gpm_val = None
    if _scan_stats:
        s4_row = next((r for r in _scan_stats if r['section'] == 'S4'), None)
        if s4_row and s4_row['elapsed_s'] > 0:
            gpm_val = s4_row['n_items'] / s4_row['elapsed_s'] * 60
    log.info('\n── SCAN SUMMARY ──────────────────────────────────────────────')
    log.info('  genes scanned    : %d',  n_genes_scanned)
    log.info('  loci evaluated   : %d',  n_loci_total)
    log.info('  eGenes called    : %d',  n_egenes)
    log.info('  scan type        : %s',  scan_type)
    log.info('  peak chr         : %s',  peak_chr)
    log.info('  S4 LOD r²        : %s',  f'{r4_val}' if r4_val is not None else 'N/A')
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
        description='validate mosaiqtl vs miQTL on gastroc real data')
    ap.add_argument('--build-cache', action='store_true',
                    help='one-time: read HAPPY genomecache via rpy2 → save py_data/cache.h5, '
                         'then exit (requires rpy2 + miqtl R package)')
    ap.add_argument('--genomecache', default=None,
                    help='path to HAPPY genomecache dir for --build-cache '
                         f'(default: {GENOMECACHE})')
    ap.add_argument('--sections', default=None,
                    help='comma-separated section numbers (default: all)')
    ap.add_argument('--plots-only', action='store_true',
                    help='skip all scans, regenerate descriptive plots from existing py_data/')
    ap.add_argument('--force',    action='store_true',
                    help='ignore checkpoints and rerun all sections')
    ap.add_argument('--n_genes',  type=int, default=None,
                    help='limit cis genes per chr for S4 (testing)')
    ap.add_argument('--scan_type', default='C', choices=['C', 'G', 'L'],
                    help='C = cis (each gene vs its own chr, default); '
                         'G = genome-wide (all genes vs all chr); '
                         'L = local (cis genes, loci within ±window_mb of TSS)')
    ap.add_argument('--window_mb', type=float, default=1.0,
                    help='half-window in Mb for --scan_type L (default 1.0)')
    ap.add_argument('--chr', default='8',
                    help='chromosome(s) to scan: "all" or comma-separated like "8" or '
                         '"1,8,11" (default: 8)')
    ap.add_argument('--celltype', type=str, default=None,
                    help='cell type label (e.g. myonuclei); runs scan in '
                         'celltypes/{ct}/ instead of whole-tissue gastroc_data/')
    args = ap.parse_args()

    # ── --build-cache: one-time rpy2 step, exits when done ────────────────────
    if args.build_cache:
        gc_dir  = Path(args.genomecache) if args.genomecache else GENOMECACHE
        out_h5  = OUT_DIR / 'py_data' / 'cache.h5'
        if not gc_dir.exists():
            log.error('genomecache not found: %s', gc_dir)
            sys.exit(1)
        log.info('--build-cache  genomecache=%s  → %s', gc_dir, out_h5)
        _build_cache_from_happy(gc_dir, out_h5)
        log.info('done — run validate_gastroc.py without --build-cache to validate')
        sys.exit(0)

    if args.plots_only:
        sections: set[int] = set()
    else:
        sections = (set(int(s.strip()) for s in args.sections.split(','))
                    if args.sections else set(range(1, 12)))

    # parse scan_type (handle "L:2.5" format)
    scan_type_raw = args.scan_type.upper()
    window_mb     = args.window_mb
    if scan_type_raw.startswith('L'):
        parts = scan_type_raw.split(':')
        if len(parts) >= 2:
            window_mb = float(parts[1])
        scan_type = 'L'
    else:
        scan_type = scan_type_raw   # 'C' or 'G'

    # parse --chr
    if args.chr.lower() == 'all':
        scan_chr = None   # all chromosomes in kinship_decomp
    else:
        scan_chr = [c.strip() for c in args.chr.split(',')]

    # ── cell-type mode: redirect data and output dirs ─────────────────────────
    _ct_label = args.celltype
    if _ct_label is not None:
        import re as _re
        _ct = _re.sub(r'[^a-z0-9]+', '_', _ct_label.lower()).strip('_')
        _ct_dir     = GASTROC_DATA / 'celltypes' / _ct
        _plt_dir_ct = GASTROC_DATA / 'plots' / 'celltypes' / _ct
        _ct_exp_gi  = _ct_dir / f'exp_gene_info_{_ct}.csv'
        if not _ct_dir.exists():
            log.error('cell-type directory not found: %s\n'
                      'Run first: conda run -n bioinfo python validation/convert_h5ad.py'
                      ' --celltype %s', _ct_dir, _ct_label)
            sys.exit(1)
        if not _ct_exp_gi.exists():
            log.error('exp_gene_info_%s.csv not found — rerun convert_h5ad.py --celltype %s',
                      _ct, _ct_label)
            sys.exit(1)
        _run_data_dir = _ct_dir
        _run_out_dir  = _ct_dir
        _plt_override = _plt_dir_ct
    else:
        _run_data_dir = GASTROC_DATA
        _run_out_dir  = OUT_DIR
        _plt_override = None

    scan_desc = scan_type if scan_type != 'L' else f'L ±{window_mb:.1f} Mb'
    log.info('validate_gastroc.py  sections=%s  force=%s  n_genes=%s  '
             'scan_type=%s  chr=%s  celltype=%s',
             sorted(sections), args.force, args.n_genes, scan_desc, args.chr,
             _ct_label or 'whole_tissue')
    log.info('  data_dir = %s', _run_data_dir)
    log.info('  out_dir  = %s', _run_out_dir)

    run(_run_data_dir, _run_out_dir, sections, args.force, args.n_genes,
        scan_type=scan_type, window_mb=window_mb, scan_chr=scan_chr,
        celltype=_ct_label, plt_dir_override=_plt_override)


if __name__ == '__main__':
    main()

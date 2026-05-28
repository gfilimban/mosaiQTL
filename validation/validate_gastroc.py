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
  py_data/    scan CSVs, checkpoint .done files, run_summary.csv, egenes.csv
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

from mosaiqtl.genotype.founders import F1_COLORS, FOUNDER_CODE, FOUNDER_COLORS, FOUNDER_ORDER

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
)
from mosaiqtl.viz.haplotypes import (
    kinship_distribution as _viz_kinship_distribution,
    kinship_heatmap as _viz_kinship_heatmap,
    prob_heatmap as _viz_prob_heatmap,
    pxg_sex_facet,
)
from mosaiqtl.viz.effects import (
    effect_bars as _viz_effect_bars,
    haplotype_sex_profile,
)
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.scan import (
    grab_locus_from_scan,
    instability_lm_scan,
    scan_h2lmm,
    scan_h2lmm_mi,
)
from mosaiqtl.qtl.permutation import (
    generate_null_outcomes,
    get_gev_thresholds,
    run_positional_scans,
    run_threshold_scans,
)
from mosaiqtl.qtl.mediation import (
    mediation_scan_qr,
    mediation_scan_qr_batch,
    select_cis_mediation_pairs,
)
from mosaiqtl.qtl.snp_scan import imputed_snp_scan_from_db
from mosaiqtl.qtl.fdr import storey_qvalue
from mosaiqtl.qtl.effects import qtl_gxe_lrt
from mosaiqtl.io.covariates import encode_sex
from mosaiqtl.pipeline import call_egenes
from mosaiqtl.utils import build_z_matrix, rint

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_gastroc')

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

def make_names(genes: list[str]) -> list[str]:
    """approximate R's make.names(genes, unique=TRUE)."""
    seen: dict[str, int] = {}
    out = []
    for g in genes:
        n = re.sub(r'[^a-zA-Z0-9.]', '.', str(g))
        if n and (n[0].isdigit() or n[0] == '.'):
            n = 'X' + n
        if n in seen:
            seen[n] += 1
            n = f'{n}.{seen[n]}'
        else:
            seen[n] = 0
        out.append(n)
    return out


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
# canonical strain ordering
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_canonical_order(cache: GenomeCache) -> GenomeCache:
    """reorder cache sample axis to canonical: Founders → F1s → CC alphanumeric.

    Canonical order matches founders.py FOUNDER_ORDER + F1_COLORS key order +
    CC strains sorted alphanumerically.  No-op if already canonical.
    """
    current = list(cache.sample_ids)
    founders_set = set(FOUNDER_ORDER)
    f1_set       = set(F1_COLORS.keys())
    cc_strains   = sorted(s for s in current if s not in founders_set and s not in f1_set)

    canonical  = [s for s in FOUNDER_ORDER if s in founders_set.intersection(current)]
    canonical += [s for s in F1_COLORS.keys() if s in f1_set.intersection(current)]
    canonical += cc_strains

    if canonical == current:
        return cache

    cur_idx = {s: i for i, s in enumerate(current)}
    missing = [s for s in canonical if s not in cur_idx]
    if missing:
        log.warning('canonical order: %d strains missing from cache: %s',
                    len(missing), missing[:5])
        canonical = [s for s in canonical if s in cur_idx]

    perm = np.array([cur_idx[s] for s in canonical])

    new_data: dict = {}
    for chrom in cache.chromosomes:
        lids, pos, probs = cache.get_chromosome(chrom)
        new_data[chrom] = {
            'locus_ids': lids.copy(),
            'positions': pos.copy(),
            'probs':     probs[:, perm, :],   # (n_loci, n_canonical, J)
        }

    log.info('reordered cache to canonical strain order (%d strains)', len(canonical))
    return GenomeCache(
        sample_ids=np.array(canonical),
        founders=list(cache.founders),
        data=new_data,
    )


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
    cache = _ensure_canonical_order(cache)
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
# position-merge helper (same pattern as validate_phase3.py)
# ─────────────────────────────────────────────────────────────────────────────

def _pos_merge(py: pd.DataFrame, r: pd.DataFrame,
               py_val: str, r_val: str,
               round_dp: int = 3) -> pd.DataFrame:
    """inner-join on (chr, rounded pos_mb); deduplicate to avoid many-to-many."""
    py = py.copy()
    r  = r.copy()
    py['_chr'] = py['chr'].astype(str)
    r['_chr']  = r['chr'].astype(str)
    py['_pos'] = py['pos_mb'].round(round_dp)
    r['_pos']  = r['pos_mb'].round(round_dp)
    py_d = (py[['_chr', '_pos', py_val]]
            .sort_values(py_val, ascending=False)
            .drop_duplicates(['_chr', '_pos']))
    r_d  = (r[['_chr', '_pos', r_val]]
            .sort_values(r_val, ascending=False)
            .drop_duplicates(['_chr', '_pos']))
    return pd.merge(
        py_d.rename(columns={py_val: 'py'}),
        r_d.rename(columns={r_val: 'r'}),
        on=['_chr', '_pos'],
    ).dropna()


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
    f_colors = {f: FOUNDER_COLORS.get(f, '#888888') for f in founders}

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
            fig.savefig(plt_dir / 's1_loci_per_chr.png', dpi=300)
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
            fig.savefig(plt_dir / 's2_haplotype_heatmap.png', dpi=300)
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
                         bbox_inches='tight')
            plt.close(fig1)
            log.info('  s3_kinship_heatmap.png  (%d strains)', n)

            fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=300)
            _viz_kinship_distribution(K, ax=ax2, chrom=peak_chr)
            ax2.set_title(f'Kinship distribution  chr{peak_chr}', fontsize=9)
            fig2.tight_layout()
            fig2.savefig(plt_dir / 's3_kinship_distribution.png', dpi=300)
            plt.close(fig2)
            log.info('  s3_kinship_distribution.png')
        except Exception as exc:
            log.warning('  s3_kinship_heatmap: %s', exc)

    # ── S4: LOD profile — top gene on peak_chr ───────────────────────────────
    s4_lod_path = py_dir / 's4_lod.csv'
    if s4_lod_path.exists():
        try:
            lod_df = pd.read_csv(s4_lod_path)
            lod_df['chr'] = lod_df['chr'].astype(str)
            lod_pc = lod_df[lod_df['chr'] == str(peak_chr)]
            if len(lod_pc) > 0:
                top_gene = _pick_top_gene(lod_pc, peak_chr)
                lod_top  = lod_pc[lod_pc['gene'] == top_gene].sort_values('pos_mb')
                thr_path = py_dir / 's5_threshold.csv'
                thr = None
                if thr_path.exists():
                    _td = pd.read_csv(thr_path); _td['chr'] = _td['chr'].astype(str)
                    _tr = _td[_td['chr'] == str(peak_chr)]
                    thr = float(_tr['threshold'].iloc[0]) if len(_tr) > 0 else float(_td['threshold'].iloc[0])
                fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
                _viz_lod_profile(
                    lod_top['pos_mb'].values, lod_top['lod'].values,
                    ax=ax, chrom=peak_chr, threshold=thr, fill=True,
                )
                ax.set_title(f'LOD profile  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_lod_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s4_lod_profile.png  (gene=%s)', top_gene)
                # also plot Piezo1 if present and different from top gene
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in lod_pc['gene'].values:
                    lod_p1 = lod_pc[lod_pc['gene'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        lod_p1['pos_mb'].values, lod_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, threshold=thr, fill=True,
                    )
                    ax_p1.set_title(f'LOD profile  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(plt_dir / f's4_lod_profile_{_p1}.png', dpi=300)
                    plt.close(fig_p1)
                    log.info('  s4_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s4_lod_profile: %s', exc)

    # ── S4: founder coef bar at peak locus ───────────────────────────────────
    s4_eff_path = py_dir / 's4_allele_effects.csv'
    if s4_eff_path.exists():
        try:
            eff_df = pd.read_csv(s4_eff_path)
            eff_pc = eff_df[eff_df['chr'].astype(str) == str(peak_chr)]
            if len(eff_pc) > 0:
                top_row   = eff_pc.loc[eff_pc['peak_lod'].idxmax()]
                beta_cols = [c for c in eff_pc.columns if c.startswith('beta_')]
                betas     = np.array([top_row[c] for c in beta_cols])
                fnames    = _display_founders([c.replace('beta_', '') for c in beta_cols])
                fig, ax   = plt.subplots(figsize=(7, 3), dpi=300)
                _viz_effect_bars(betas, fnames, ax=ax)
                ax.set_title(
                    f'Founder effects  {top_row["gene"]}  '
                    f'{top_row["peak_locus"]}  LOD={top_row["peak_lod"]:.1f}',
                    fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_founder_effects.png', dpi=300)
                plt.close(fig)
                log.info('  s4_founder_effects.png  (gene=%s)', top_row['gene'])
                # also plot Piezo1 if present and different from top gene
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and top_row['gene'] != _p1 and _p1 in eff_pc['gene'].values:
                    p1_row   = eff_pc[eff_pc['gene'] == _p1].iloc[0]
                    betas_p1 = np.array([p1_row[c] for c in beta_cols])
                    fig_p1, ax_p1 = plt.subplots(figsize=(7, 3), dpi=300)
                    _viz_effect_bars(betas_p1, fnames, ax=ax_p1)  # fnames already display names
                    ax_p1.set_title(
                        f'Founder effects  {p1_row["gene"]}  '
                        f'{p1_row["peak_locus"]}  LOD={p1_row["peak_lod"]:.1f}',
                        fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(plt_dir / f's4_founder_effects_{_p1}.png', dpi=300)
                    plt.close(fig_p1)
                    log.info('  s4_founder_effects_%s.png', _p1)
        except Exception as exc:
            log.warning('  s4_founder_effects: %s', exc)

    # ── S4: Manhattan — peak LOD per gene across all scanned chr ──────────────
    peak_lod_path = py_dir / 's4_peak_lod.csv'
    if peak_lod_path.exists() and s4_lod_path.exists():
        try:
            pk_df   = pd.read_csv(peak_lod_path)
            lod_all = pd.read_csv(s4_lod_path)
            lod_all['chr'] = lod_all['chr'].astype(str)
            idx_pk   = lod_all.groupby('gene')['lod'].idxmax()
            peak_pos = lod_all.loc[idx_pk, ['gene', 'chr', 'pos_mb', 'lod']]
            peak_pos = peak_pos.merge(pk_df[['gene']], on='gene')

            def _chr_key(c):
                try:
                    return int(c)
                except ValueError:
                    return 100
            chrs_sorted = sorted(peak_pos['chr'].unique(), key=_chr_key)
            scans_dict  = {
                ch: (peak_pos[peak_pos['chr'] == ch]['pos_mb'].values,
                     peak_pos[peak_pos['chr'] == ch]['lod'].values)
                for ch in chrs_sorted
            }
            thr_path = py_dir / 's5_threshold.csv'
            thr = None
            if thr_path.exists():
                _td = pd.read_csv(thr_path); _td['chr'] = _td['chr'].astype(str)
                _tr = _td[_td['chr'] == str(peak_chr)]
                thr = float(_tr['threshold'].iloc[0]) if len(_tr) > 0 else float(_td['threshold'].iloc[0])
            fig = _viz_manhattan(scans_dict, threshold=thr)
            fig.axes[0].set_title(f'Manhattan  n={len(peak_pos)} genes', fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 's4_manhattan.png', dpi=300)
            plt.close(fig)
            log.info('  s4_manhattan.png')
        except Exception as exc:
            log.warning('  s4_manhattan: %s', exc)

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
            fig, ax  = plt.subplots(figsize=(7, 4), dpi=300)
            _viz_perm_distribution(max_lods, ax=ax, threshold=thr, chrom=peak_chr)
            ax.set_title(
                f'Permutation distribution  chr{peak_chr}  '
                f'n={np.isfinite(max_lods).sum()}',
                fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 's5_perm_distribution.png', dpi=300)
            plt.close(fig)
            log.info('  s5_perm_distribution.png')
        except Exception as exc:
            log.warning('  s5_perm_distribution: %s', exc)

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
            fig, ax    = plt.subplots(figsize=(7, 4), dpi=300)
            _viz_egene_lod_distribution(lods, is_egene, ax=ax,
                                        threshold=thr, chrom=peak_chr)
            ax.set_title(
                f'eGene calling  chr{peak_chr}  n={len(pk)} genes',
                fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 's6_egene_lod_distribution.png', dpi=300)
            plt.close(fig)
            log.info('  s6_egene_lod_distribution.png  (%d eGenes / %d genes)',
                     is_egene.sum(), len(pk))
        except Exception as exc:
            log.warning('  s6_egene_lod_distribution: %s', exc)

    # ── S8: SNP LOD profile for top eGene ────────────────────────────────────
    s8_path = py_dir / 's8_snp_lod.csv'
    if s8_path.exists():
        try:
            snp_df = pd.read_csv(s8_path)
            snp_df['chr'] = snp_df['chr'].astype(str)
            snp_pc = snp_df[snp_df['chr'] == str(peak_chr)]
            if len(snp_pc) > 0:
                top_gene = _pick_top_gene(snp_pc, peak_chr)
                snp_top  = snp_pc[snp_pc['gene'] == top_gene].sort_values('pos_mb')
                fig, ax  = plt.subplots(figsize=(10, 3), dpi=300)
                ax.scatter(snp_top['pos_mb'].values, snp_top['lod'].values,
                           color=FOUNDER_COLORS['129S1J'], s=4, alpha=0.7,
                           linewidths=0)
                ax.set_xlim(snp_top['pos_mb'].min(), snp_top['pos_mb'].max())
                ax.set_ylim(bottom=0)
                ax.set_xlabel(f'Chr {peak_chr} position (Mb)')
                ax.set_ylabel('LOD (SNP)')
                ax.set_title(f'SNP fine-mapping scan  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's8_snp_lod_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s8_snp_lod_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in snp_pc['gene'].values:
                    snp_p1 = snp_pc[snp_pc['gene'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    ax_p1.scatter(snp_p1['pos_mb'].values, snp_p1['lod'].values,
                                  color=FOUNDER_COLORS['129S1J'], s=4, alpha=0.7,
                                  linewidths=0)
                    ax_p1.set_xlim(snp_p1['pos_mb'].min(), snp_p1['pos_mb'].max())
                    ax_p1.set_ylim(bottom=0)
                    ax_p1.set_xlabel(f'Chr {peak_chr} position (Mb)')
                    ax_p1.set_ylabel('LOD (SNP)')
                    ax_p1.set_title(f'SNP fine-mapping scan  {_p1}  chr{peak_chr}',
                                    fontsize=9)
                    sns.despine(ax=ax_p1)
                    fig_p1.tight_layout()
                    fig_p1.savefig(plt_dir / f's8_snp_lod_profile_{_p1}.png', dpi=300)
                    plt.close(fig_p1)
                    log.info('  s8_snp_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s8_snp_lod_profile: %s', exc)

    # ── S10: CI strip per eGene ───────────────────────────────────────────────
    s10_path = py_dir / 's10_ci.csv'
    if s10_path.exists():
        try:
            ci_df = pd.read_csv(s10_path).sort_values('peak_mb')
            fig, ax = plt.subplots(
                figsize=(8, max(3, len(ci_df) * 0.35)), dpi=300)
            _viz_ci_strip(
                ci_df['gene'].tolist(),
                ci_df['ci_lo'].values,
                ci_df['ci_hi'].values,
                ci_df['peak_mb'].values,
                ax=ax, chrom=peak_chr,
            )
            ax.set_title(f'Bootstrap CI  {len(ci_df)} eGenes', fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 's10_ci_strip.png', dpi=300)
            plt.close(fig)
            log.info('  s10_ci_strip.png')
        except Exception as exc:
            log.warning('  s10_ci_strip: %s', exc)

    # ── S11: instability LOD profile for top gene ─────────────────────────────
    s11_path = py_dir / 's11_instability.csv'
    if s11_path.exists():
        try:
            inst_df = pd.read_csv(s11_path)
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
                ax.set_ylabel('Instability LOD')
                ax.set_title(f'Instability  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's11_instability_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s11_instability_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in inst_pc['gene'].values:
                    inst_p1 = inst_pc[inst_pc['gene'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        inst_p1['pos_mb'].values, inst_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, fill=True,
                        color=FOUNDER_COLORS['CASTJ'],
                    )
                    ax_p1.set_ylabel('Instability LOD')
                    ax_p1.set_title(f'Instability  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(plt_dir / f's11_instability_profile_{_p1}.png', dpi=300)
                    plt.close(fig_p1)
                    log.info('  s11_instability_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s11_instability_profile: %s', exc)

    # ── S7: MI scan LOD profile ───────────────────────────────────────────────
    s7_path = py_dir / 's7_mi_lod.csv'
    if s7_path.exists():
        try:
            mi_df = pd.read_csv(s7_path)
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
                ax.set_ylabel('LOD (MI)')
                ax.set_title(f'MI LOD profile  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's7_mi_lod_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s7_mi_lod_profile.png  (gene=%s)', top_gene)
                _p1 = PRIORITY_GENE.get(str(peak_chr))
                if _p1 and _p1 != top_gene and _p1 in mi_pc['gene'].values:
                    mi_p1 = mi_pc[mi_pc['gene'] == _p1].sort_values('pos_mb')
                    fig_p1, ax_p1 = plt.subplots(figsize=(10, 3), dpi=300)
                    _viz_lod_profile(
                        mi_p1['pos_mb'].values, mi_p1['lod'].values,
                        ax=ax_p1, chrom=peak_chr, fill=True,
                        color=FOUNDER_COLORS['WSBJ'],
                    )
                    ax_p1.set_ylabel('LOD (MI)')
                    ax_p1.set_title(f'MI LOD profile  {_p1}  chr{peak_chr}', fontsize=9)
                    fig_p1.tight_layout()
                    fig_p1.savefig(plt_dir / f's7_mi_lod_profile_{_p1}.png', dpi=300)
                    plt.close(fig_p1)
                    log.info('  s7_mi_lod_profile_%s.png', _p1)
        except Exception as exc:
            log.warning('  s7_mi_lod_profile: %s', exc)

    # ── S9: mediation LOD profile (base vs mediated) ──────────────────────────
    s9_path = py_dir / 's9_mediation.csv'
    if s9_path.exists():
        try:
            med_df = pd.read_csv(s9_path)
            med_df['chr'] = med_df['chr'].astype(str)
            med_pc = med_df[med_df['chr'] == str(peak_chr)]
            if len(med_pc) > 0:
                top_gene = _pick_top_gene(med_pc, peak_chr, lod_col='lod_base')
                med_top  = med_pc[med_pc['gene'] == top_gene].sort_values('pos_mb')
                fig, ax  = plt.subplots(figsize=(10, 3), dpi=300)
                _viz_mediation_profile(
                    med_top['pos_mb'].values,
                    med_top['lod_base'].values,
                    med_top['lod_mediated'].values,
                    ax=ax, chrom=peak_chr, gene=top_gene,
                )
                ax.set_title(f'Mediation  {top_gene}  chr{peak_chr}',
                             fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's9_mediation_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s9_mediation_profile.png  (gene=%s)', top_gene)
        except Exception as exc:
            log.warning('  s9_mediation_profile: %s', exc)

    # ── timing / memory summary ───────────────────────────────────────────────
    scan_stats_path = py_dir / 'scan_stats.csv'
    if scan_stats_path.exists():
        try:
            ss_raw = pd.read_csv(scan_stats_path)
            # keep only canonical S1–S11 rows (drop per-chr sub-rows like S4_chr8)
            ss_raw = ss_raw[ss_raw['section'].isin(_SECTION_ORDER)].copy()
            # fill missing sections (checkpointed/skipped) with 0
            existing = set(ss_raw['section'])
            fill_rows = [{'section': s, 'elapsed_s': 0.0,
                          'peak_heap_mb': 0.0, 'peak_rss_mb': 0.0}
                         for s in _SECTION_ORDER if s not in existing]
            if fill_rows:
                ss_raw = pd.concat(
                    [ss_raw, pd.DataFrame(fill_rows)], ignore_index=True)
            # enforce S1→S11 display order (bottom = S1, top = S11 on barh)
            ss_raw['_ord'] = ss_raw['section'].map(
                {s: i for i, s in enumerate(_SECTION_ORDER)})
            ss = ss_raw.sort_values('_ord').reset_index(drop=True)
            labels = [_SECTION_LABELS.get(s, s) for s in ss['section']]

            if len(ss) > 0:
                _mem_col = 'peak_rss_mb' if 'peak_rss_mb' in ss.columns else 'peak_heap_mb'
                _mem_label = 'Peak RSS (MB)' if _mem_col == 'peak_rss_mb' else 'Peak heap (MB)'
                has_mem = ss[_mem_col].gt(0).any() if _mem_col in ss.columns else False
                fig, axes = plt.subplots(1, 2 if has_mem else 1,
                                         figsize=(9 if has_mem else 5,
                                                  max(3, len(ss) * 0.45)))
                ax_t = axes[0] if has_mem else axes
                ax_t.barh(labels, ss['elapsed_s'], color='#4F6EAF')
                ax_t.invert_yaxis()
                ax_t.set_xlabel('Elapsed (s)')
                ax_t.set_title('Timing per step')
                sns.despine(ax=ax_t)
                if has_mem:
                    axes[1].barh(labels, ss[_mem_col].fillna(0), color='#52A5DB')
                    axes[1].invert_yaxis()
                    axes[1].set_xlabel(_mem_label)
                    axes[1].set_title('Memory per step')
                    sns.despine(ax=axes[1])
                fig.tight_layout()
                fig.savefig(plt_dir / 'timing_memory_summary.png', dpi=300)
                plt.close(fig)
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

                def _chr_sort_key(c: str) -> int:
                    try:
                        return int(c)
                    except ValueError:
                        return 100

                chr_rows = chr_rows.sort_values(
                    'chrom', key=lambda s: s.map(_chr_sort_key))
                chr_rows['gpm'] = (chr_rows['n_items']
                                   / chr_rows['elapsed_s'].replace(0, float('nan')) * 60)
                fig_tp, ax_tp = plt.subplots(figsize=(9, 3.5), dpi=300)
                ax_tp.bar(range(len(chr_rows)), chr_rows['gpm'].fillna(0),
                          color='#4F6EAF', edgecolor='none')
                ax_tp.set_xticks(range(len(chr_rows)))
                ax_tp.set_xticklabels(
                    [f'chr{c}' for c in chr_rows['chrom']], rotation=45,
                    ha='right', fontsize=7)
                ax_tp.set_ylabel('Genes / min', fontsize=9)
                ax_tp.set_title(
                    'Scan throughput per chromosome  (real gastroc data)',
                    fontsize=9)
                # median line
                med_gpm = chr_rows['gpm'].median()
                ax_tp.axhline(med_gpm, color='#D83026', linestyle='--',
                              linewidth=1, label=f'median {med_gpm:.0f} g/min')
                ax_tp.legend(fontsize=7)
                sns.despine(ax=ax_tp)
                fig_tp.tight_layout()
                fig_tp.savefig(plt_dir / 'throughput_by_chr.png',
                               dpi=300, bbox_inches='tight')
                plt.close(fig_tp)
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

    display_f = _display_founders(founders)  # full names for labels/colors
    fcolors = [FOUNDER_COLORS.get(f, '#888888') for f in display_f]

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
                r_corr, _ = sp_stats.pearsonr(py_v, r_v)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(r_v, py_v, alpha=0.15, s=4, linewidths=0, color='#4F6EAF')
                lim = max(r_v.max(), py_v.max())
                ax.plot([0, lim], [0, lim], 'r--', lw=1)
                ax.set_xlabel('R haplotype prob')
                ax.set_ylabel('Python haplotype prob')
                ax.set_title(f'Haplotype probs  r={r_corr:.4f}  n={len(py_v)}')
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's2_haplotype_scatter.png', dpi=300)
                plt.close(fig)
                log.info('  [plot] s2_haplotype_scatter.png  r=%.4f', r_corr)
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
                r_corr, _ = sp_stats.pearsonr(py_k, r_k)
                fig, ax = plt.subplots(figsize=(4, 4))
                ax.scatter(r_k, py_k, alpha=0.3, s=6, linewidths=0, color='#55AF5B')
                lim = max(max(r_k), max(py_k))
                ax.plot([0, lim], [0, lim], 'r--', lw=1)
                ax.set_xlabel('R kinship (LOCO chr8)')
                ax.set_ylabel('Python kinship (LOCO chr8)')
                ax.set_title(f'Kinship elements  r={r_corr:.4f}  n={len(py_k)}')
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's3_kinship_scatter.png', dpi=300)
                plt.close(fig)
                log.info('  [plot] s3_kinship_scatter.png  r=%.4f', r_corr)
        except Exception as exc:
            log.warning('  [plot] s3_kinship_scatter: %s', exc)

    # ── S4: LOD density (peak LOD distribution) ───────────────────────────────
    cmp_lod = cmp_dir / 's4_peak_lod_comparison.csv'
    if cmp_lod.exists():
        try:
            cdf = pd.read_csv(cmp_lod)
            fig, ax = plt.subplots(figsize=(5, 4))
            for col, label, color in [('peak_lod_py', 'Python', '#4F6EAF'),
                                       ('peak_lod_r',  'R',      '#D83026')]:
                if col in cdf.columns:
                    vals = cdf[col].dropna()
                    vals.plot.kde(ax=ax, label=label, color=color)
                    ax.axvline(vals.median(), color=color, ls='--', lw=1, alpha=0.7)
            ax.set_xlabel('peak LOD')
            ax.set_title(f'LMM peak LOD distribution  n={len(cdf)}')
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 's4_lod_density.png', dpi=300)
            plt.close(fig)
            log.info('  [plot] s4_lod_density.png')
        except Exception as exc:
            log.warning('  [plot] s4_lod_density: %s', exc)

    # ── S4: peak position scatter ──────────────────────────────────────────────
    eff_path = py_dir / 's4_allele_effects.csv'
    if eff_path.exists():
        try:
            eff_py = pd.read_csv(eff_path, usecols=['gene', 'peak_mb'])
            # collect R peak pos from chr*/lod.csv or lod_genome.csv
            r_peak_rows = []
            r_lod_gen = r_data / 'lod_genome.csv'
            if r_lod_gen.exists():
                r_all = pd.read_csv(r_lod_gen, usecols=['gene', 'pos_mb', 'lod'])
                r_pk  = (r_all.sort_values('lod', ascending=False)
                         .drop_duplicates('gene')[['gene', 'pos_mb']]
                         .rename(columns={'pos_mb': 'peak_mb_r'}))
                r_peak_rows.append(r_pk)
            else:
                for cand in sorted(r_data.glob('chr*/lod.csv')):
                    tmp = pd.read_csv(cand, usecols=['gene', 'pos_mb', 'lod'])
                    r_peak_rows.append(
                        tmp.sort_values('lod', ascending=False)
                           .drop_duplicates('gene')[['gene', 'pos_mb']]
                           .rename(columns={'pos_mb': 'peak_mb_r'}))
            if r_peak_rows:
                r_pos = pd.concat(r_peak_rows).groupby('gene')['peak_mb_r'].first().reset_index()
                merged_pos = eff_py.merge(r_pos, on='gene', how='inner')
                if len(merged_pos) >= 5:
                    r_corr, _ = sp_stats.pearsonr(merged_pos['peak_mb'],
                                                   merged_pos['peak_mb_r'])
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.scatter(merged_pos['peak_mb_r'], merged_pos['peak_mb'],
                               alpha=0.4, s=6, linewidths=0, color='#F4C245')
                    lim = max(merged_pos['peak_mb'].max(), merged_pos['peak_mb_r'].max())
                    ax.plot([0, lim], [0, lim], 'r--', lw=1)
                    ax.set_xlabel('R peak position (Mb)')
                    ax.set_ylabel('Python peak position (Mb)')
                    ax.set_title(f'Peak positions  r={r_corr:.4f}  n={len(merged_pos)}')
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's4_peak_pos_scatter.png', dpi=300)
                    plt.close(fig)
                    log.info('  [plot] s4_peak_pos_scatter.png  r=%.4f', r_corr)
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
                n_top = len(top_genes)
                r_lod_chr_path = r_data / f'chr{peak_chr}' / 'lod.csv'
                r_lod_chr = (pd.read_csv(r_lod_chr_path)
                             if r_lod_chr_path.exists() else None)
                fig, axes = plt.subplots(n_top, 1, figsize=(7, 2.5 * n_top), sharex=True)
                if n_top == 1:
                    axes = [axes]
                for ax, gene in zip(axes, top_genes):
                    py_g = lod_pc[lod_pc['gene'] == gene].sort_values('pos_mb')
                    ax.plot(py_g['pos_mb'], py_g['lod'], lw=1, color='#4F6EAF',
                            label='Python', zorder=3)
                    if r_lod_chr is not None and gene in r_lod_chr['gene'].values:
                        r_g = (r_lod_chr[r_lod_chr['gene'] == gene]
                               .sort_values('pos_mb'))
                        ax.plot(r_g['pos_mb'], r_g['lod'], lw=1, color='#D83026',
                                ls='--', label='R', alpha=0.8, zorder=2)
                    ax.set_ylabel('LOD', fontsize=7)
                    ax.set_title(gene, fontsize=8)
                    if ax is axes[0]:
                        ax.legend(fontsize=7, loc='upper right')
                    sns.despine(ax=ax)
                axes[-1].set_xlabel('position (Mb)')
                fig.suptitle(f'LOD profiles  chr{peak_chr}', fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_lod_profiles.png', dpi=300)
                plt.close(fig)
                log.info('  [plot] s4_lod_profiles.png  top %d genes on chr%s',
                         n_top, peak_chr)
        except Exception as exc:
            log.warning('  [plot] s4_lod_profiles: %s', exc)

    # ── S4: allele effects scatter (Python betas vs R betas, 2× scale note) ───
    r_eff_path = r_data / 'allele_effects_chr8.csv'
    if eff_path.exists() and r_eff_path.exists():
        try:
            py_eff = pd.read_csv(eff_path)
            r_eff  = pd.read_csv(r_eff_path)
            # R column naming: may have 'beta_AJ' or founder abbreviations
            # find founder beta columns by looking for 'beta_' prefix
            py_beta_cols = [f'beta_{f}' for f in founders if f'beta_{f}' in py_eff.columns]
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
                    r_corr, _ = sp_stats.pearsonr(py_b[mask], r_b[mask])
                    fig, ax = plt.subplots(figsize=(4, 4))
                    ax.scatter(r_b[mask], py_b[mask], alpha=0.3, s=5, linewidths=0)
                    lim = np.abs(np.concatenate([r_b[mask], py_b[mask]])).max()
                    ax.axhline(0, color='grey', lw=0.5)
                    ax.axvline(0, color='grey', lw=0.5)
                    ax.plot([-lim, lim], [-2*lim, 2*lim], 'r--', lw=1,
                            label='slope=2× expected')
                    ax.set_xlabel('R beta (per-allele scale)')
                    ax.set_ylabel('Python beta (per-haplotype scale)')
                    ax.set_title(f'Allele effects  r={r_corr:.4f}  n={mask.sum()}\n'
                                 'Python 2× R: dosage 0–1 vs 0–2')
                    ax.legend(fontsize=7)
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's4_allele_effects_scatter.png', dpi=300)
                    plt.close(fig)
                    log.info('  [plot] s4_allele_effects_scatter.png  r=%.4f', r_corr)
        except Exception as exc:
            log.warning('  [plot] s4_allele_effects_scatter: %s', exc)

    # ── S5: permutation distribution (py vs R, fitted GEV) ────────────────────
    py_perm_path = py_dir / 's5_perm_max_lods.csv'
    r_perm_path  = r_data / f'perm_max_lods_chr{peak_chr}.csv'
    if py_perm_path.exists() and r_perm_path.exists():
        try:
            py_perm = pd.read_csv(py_perm_path)['max_lod'].dropna().values
            r_perm  = pd.read_csv(r_perm_path)['max_lod'].dropna().values
            fig, ax = plt.subplots(figsize=(5, 4))
            for vals, label, color in [(py_perm, 'Python', '#4F6EAF'),
                                        (r_perm,  'R',      '#D83026')]:
                if len(vals) < 5:
                    continue
                ax.hist(vals, bins=30, alpha=0.35, color=color,
                        density=True, label=f'{label} n={len(vals)}')
                try:
                    c, loc, scale = gev_dist.fit(vals)
                    xg = np.linspace(vals.min(), vals.max(), 200)
                    ax.plot(xg, gev_dist.pdf(xg, c, loc, scale),
                            color=color, lw=1.5)
                    thr95 = gev_dist.ppf(0.95, c, loc, scale)
                    ax.axvline(thr95, color=color, ls='--', lw=1,
                               label=f'{label} p95={thr95:.2f}')
                except Exception:
                    pass
            ax.set_xlabel('max LOD')
            ax.set_ylabel('density')
            ax.set_title(f'Permutation max-LOD  chr{peak_chr}')
            ax.legend(fontsize=7)
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 's5_perm_distribution.png', dpi=300)
            plt.close(fig)
            log.info('  [plot] s5_perm_distribution.png')
        except Exception as exc:
            log.warning('  [plot] s5_perm_distribution: %s', exc)

    # ── S4: scan speed comparison (genes/min py vs R per chr) ─────────────────
    scan_stats_path = py_dir / 'scan_stats.csv'
    r_timing_path   = r_data / 'timing_chr.csv'
    if scan_stats_path.exists() and r_timing_path.exists():
        try:
            py_stats  = pd.read_csv(scan_stats_path)
            r_timing  = pd.read_csv(r_timing_path)
            # Python rows: section like 'S4_chr8' → extract chr
            py_s4 = py_stats[py_stats['section'].str.startswith('S4_chr')].copy()
            py_s4['chr'] = py_s4['section'].str.replace('S4_chr', '')
            py_s4['gpm'] = py_s4['n_items'] / py_s4['elapsed_s'] * 60
            # R timing: must have chr and genes_per_min or elapsed_s + n_genes
            if 'genes_per_min' in r_timing.columns:
                r_s4 = r_timing[['chr', 'genes_per_min']].copy()
                r_s4['chr'] = r_s4['chr'].astype(str)
            elif 'elapsed_s' in r_timing.columns and 'n_genes' in r_timing.columns:
                r_s4 = r_timing.copy()
                r_s4['chr'] = r_s4['chr'].astype(str)
                r_s4['genes_per_min'] = r_s4['n_genes'] / r_s4['elapsed_s'] * 60
            else:
                r_s4 = None
            if r_s4 is not None and len(py_s4) >= 2:
                merged_sp = py_s4.merge(r_s4, on='chr', how='inner')
                if len(merged_sp) >= 2:
                    fig, ax = plt.subplots(figsize=(5, 4))
                    x = np.arange(len(merged_sp))
                    w = 0.35
                    ax.bar(x - w/2, merged_sp['gpm'], w,
                           label='Python', color='#4F6EAF', alpha=0.8)
                    ax.bar(x + w/2, merged_sp['genes_per_min'], w,
                           label='R', color='#D83026', alpha=0.8)
                    ax.set_xticks(x)
                    ax.set_xticklabels([f'chr{c}' for c in merged_sp['chr']],
                                       rotation=45, ha='right', fontsize=7)
                    ax.set_ylabel('genes / min')
                    ax.set_title('LMM scan throughput: Python vs R')
                    ax.legend(fontsize=8)
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's4_scan_speed.png', dpi=300)
                    plt.close(fig)
                    log.info('  [plot] s4_scan_speed.png  %d chr', len(merged_sp))
        except Exception as exc:
            log.warning('  [plot] s4_scan_speed: %s', exc)


# ─────────────────────────────────────────────────────────────────────────────
# main validation run
# ─────────────────────────────────────────────────────────────────────────────

def run(data_dir: Path, out_dir: Path, sections: set[int],
        force: bool, n_genes: int | None,
        scan_type: str = 'C',
        window_mb: float = 1.0,
        scan_chr: list[str] | None = None) -> dict:
    """run gastroc characterization sections; return gate results dict.

    Parameters
    ----------
    data_dir  : root of gastroc_data/ — contains input files (metadata.csv,
                gene_info.csv, counts_lognorm.h5) and r_data/ subdirectory.
    scan_type : 'C' (cis), 'G' (genome-wide), or 'L' (local ±window_mb of TSS)
    window_mb : half-window size in Mb for L mode (ignored for C/G)
    scan_chr  : chromosomes to scan; None = all chromosomes in kinship_decomp
    """
    r_data  = data_dir / 'r_data'    # R reference outputs (optional)
    py_dir  = out_dir / 'py_data'
    cmp_dir = out_dir / 'compare'
    plt_dir = out_dir / 'plots'
    for d in (py_dir, cmp_dir, plt_dir):
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

    # ── load metadata ─────────────────────────────────────────────────────────
    meta_path = data_dir / 'metadata.csv'
    if not meta_path.exists():
        log.error('metadata.csv not found at %s — run convert_gastroc_h5ad.py', meta_path)
        sys.exit(1)
    meta = pd.read_csv(meta_path)
    sample_ids_meta = meta['sample_id'].tolist()
    # map sample-level IDs (individual mice) to genotype IDs (strain names in cache)
    meta_geno_map: dict[str, str] = dict(zip(meta['sample_id'], meta['genotype']))
    log.info('metadata: %d samples  %d genotypes  datasets: %s',
             len(meta), meta['genotype'].nunique(),
             sorted(meta['dataset'].unique()))

    # ── gene info (chr annotation + TSS for cis/local filter) ─────────────────
    gene_info_path = data_dir / 'gene_info.csv'
    gene_info = pd.read_csv(gene_info_path)
    gene_info['chr'] = gene_info['chr'].astype(str)
    gene_info = gene_info.copy()
    gene_info['gene_name'] = make_names(gene_info['gene_id'].tolist())
    gene_id_to_name = dict(zip(gene_info['gene_id'],   gene_info['gene_name']))
    gene_name_to_id = dict(zip(gene_info['gene_name'], gene_info['gene_id']))
    log.info('gene_info: %d genes annotated', len(gene_info))

    if scan_type == 'L' and 'start' not in gene_info.columns:
        log.warning('gene_info.csv has no "start" column — L scan mode will not filter loci')

    # ── load GenomeCache (must be pre-built as HDF5) ──────────────────────────
    hdf5_cache = py_dir / 'cache.h5'
    if hdf5_cache.exists():
        log.info('loading GenomeCache from HDF5: %s', hdf5_cache)
        cache = GenomeCache.from_hdf5(str(hdf5_cache))
        cache = _ensure_canonical_order(cache)  # fast no-op if already canonical
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
    kd_npz = py_dir / 'kinship_decomp.npz'
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

            h5_path = data_dir / 'counts_lognorm.h5'
            all_lod_rows:    list[pd.DataFrame] = []
            all_effect_rows: list[dict]          = []
            _rss_s4 = _get_rss_mb()
            tracemalloc.start()
            t_scan = time.perf_counter()
            n_genes_total = 0

            for ch in chrs_to_scan:
                # select genes for this chromosome
                if scan_type == 'G':
                    ch_genes = gene_info
                else:  # C or L — cis genes only
                    ch_genes = gene_info[gene_info['chr'] == str(ch)]

                if len(ch_genes) == 0:
                    continue

                gene_ids_ch   = ch_genes['gene_id'].tolist()
                gene_names_ch = ch_genes['gene_name'].tolist()

                if n_genes is not None:
                    gene_ids_ch   = gene_ids_ch[:n_genes]
                    gene_names_ch = gene_names_ch[:n_genes]

                counts_ch, loaded_ids, sids_ch = _load_expression(h5_path, gene_ids_ch)
                if counts_ch.shape[0] == 0:
                    log.info('  chr%s: no genes found in HDF5 — skipping', ch)
                    continue

                # map sample-level IDs → genotype IDs (strain names in cache)
                geno_ids_ch = [meta_geno_map.get(s, s) for s in sids_ch]

                # use gene_id_to_name for loaded genes (handles missing entries)
                loaded_names = [gene_id_to_name.get(gid, gid) for gid in loaded_ids]
                cov_ch = _build_covariates(meta, sids_ch)
                n_loci_ch = len(cache.get_chromosome(ch)[0])

                log.info('  chr%s: %d genes  %d loci (%s mode)',
                         ch, len(loaded_ids), n_loci_ch, scan_type)

                # for L mode: build TSS lookup
                tss_lookup: dict[str, float] = {}
                if scan_type == 'L' and 'start' in ch_genes.columns:
                    for _, grow in ch_genes.iterrows():
                        tss_lookup[grow['gene_name']] = float(grow['start']) / 1e6

                # map CLI scan_type to scan_window API
                _sw = {'C': 'cis', 'G': 'genome', 'L': 'local'}[scan_type]

                log.info('  running LMM scan on chr%s ...', ch)
                t_chr = time.perf_counter()
                sex_ch = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sids_ch])
                for gi, (gene_id, gene_name) in enumerate(zip(loaded_ids, loaded_names)):
                    if gi % 100 == 0 and gi > 0:
                        elapsed = time.perf_counter() - t_scan
                        log.info('    chr%s %d/%d  %.0fs elapsed',
                                 ch, gi + 1, len(loaded_ids), elapsed)
                    y = counts_ch[gi].astype(float)
                    tss_mb = tss_lookup.get(gene_name)
                    scan = scan_h2lmm(
                        y, cache,
                        covariates=cov_ch,
                        kinship_decomp=kinship_decomp,
                        sample_ids=geno_ids_ch,
                        use_rint=True,
                        phenotype_id=gene_name,
                        scan_window=_sw,
                        gene_chr=ch,
                        gene_pos_mb=tss_mb if _sw == 'local' else None,
                        window_mb=window_mb,
                    )
                    df = _scan_to_df(scan, gene_name)
                    all_lod_rows.append(df)

                    # collect peak-locus betas for allele effects comparison
                    if hasattr(scan, 'betas') and scan.betas is not None:
                        peak_idx = int(np.argmax(scan.lod))
                        eff: dict = {
                            'gene':        gene_name,
                            'gene_id':     gene_id,
                            'chr':         str(ch),
                            'peak_locus':  str(scan.locus_ids[peak_idx]),
                            'peak_mb':     float(scan.positions[peak_idx]),
                            'peak_lod':    float(scan.lod[peak_idx]),
                        }
                        for fi, fname in enumerate(cache.founders):
                            eff[f'beta_{fname}'] = float(scan.betas[peak_idx, fi])
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
                del counts_ch; gc.collect()

            elapsed_s = time.perf_counter() - t_scan
            if all_lod_rows:
                log.info('  writing scan results ...')
                lod_df = pd.concat(all_lod_rows, ignore_index=True)
                lod_df.to_csv(py_lod_path, index=False)
                genes_per_min = n_genes_total / elapsed_s * 60 if elapsed_s > 0 else 0
                log.info('  scan complete: %d genes  %d rows  %.1fs  %.1f genes/min',
                         n_genes_total, len(lod_df), elapsed_s, genes_per_min)
                _, _s4_peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                _record_stats('S4', 'lmm_scan', n_genes_total, elapsed_s,
                              _s4_peak / 1e6, max(0.0, _get_rss_mb() - _rss_s4))
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
                peak_py.to_csv(peak_lod_path, index=False)

                # ── compare to R reference (per-chr) ─────────────────────────
                r_peak_rows = []
                for ch in chrs_to_scan:
                    r_lod_path_ch = r_data / f'chr{ch}' / 'lod.csv'
                    if r_lod_path_ch.exists():
                        r_lod_ch = pd.read_csv(r_lod_path_ch, usecols=['gene', 'lod'])
                        pk_r = (r_lod_ch.groupby('gene')['lod'].max()
                                .reset_index().rename(columns={'lod': 'peak_lod_r'}))
                        r_peak_rows.append(pk_r)
                        del r_lod_ch; gc.collect()
                    else:
                        # fallback: lod_genome.csv for G mode
                        r_lod_genome = r_data / 'lod_genome.csv'
                        if r_lod_genome.exists() and not r_peak_rows:
                            r_lod_all = pd.read_csv(r_lod_genome, usecols=['gene', 'lod'])
                            pk_r = (r_lod_all.groupby('gene')['lod'].max()
                                    .reset_index().rename(columns={'lod': 'peak_lod_r'}))
                            r_peak_rows.append(pk_r)
                            del r_lod_all; gc.collect()
                            break

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

                        fig, ax = plt.subplots(figsize=(5, 5))
                        ax.scatter(cmp['peak_lod_r'], cmp['peak_lod_py'],
                                   alpha=0.4, s=8, linewidths=0)
                        lim = max(cmp['peak_lod_r'].max(), cmp['peak_lod_py'].max())
                        ax.plot([0, lim], [0, lim], 'r--', lw=1)
                        ax.set_xlabel('R peak LOD')
                        ax.set_ylabel('Python peak LOD')
                        ax.set_title(f'Peak LOD  r²={r2:.4f}  n={n_matched}')
                        sns.despine(ax=ax)
                        fig.tight_layout()
                        fig.savefig(plt_dir / 's4_peak_lod_scatter.png', dpi=300)
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

    # ── determine peak chromosome (from S4 results) ───────────────────────────
    # used as reference for S5–S11 downstream sections
    peak_chr = chrs_to_scan[0] if chrs_to_scan else '8'
    if peak_lod_path.exists():
        _pk = pd.read_csv(peak_lod_path)
        if len(_pk) > 0 and 'gene' in _pk.columns:
            # identify chr of peak gene from lod_df or from gene_info
            lod_for_peak = lod_df if 'lod_df' in dir() else None
            if lod_for_peak is not None and len(lod_for_peak) > 0:
                top_gene_name = _pk.iloc[_pk['peak_lod'].argmax()]['gene']
                top_rows = lod_for_peak[lod_for_peak['gene'] == top_gene_name]
                if len(top_rows) > 0:
                    peak_chr_row = top_rows.loc[top_rows['lod'].idxmax()]
                    peak_chr = str(peak_chr_row['chr'])

    log.info('[downstream sections] peak_chr=%s', peak_chr)

    egenes_df: 'pd.DataFrame | None' = None
    chr_null_lods_s5: 'dict | None' = None
    perm_max_lods_s5: 'np.ndarray | None' = None
    perm_n_s5: int = 500

    # ─────────────────────────────────────────────────────────────────────────
    # S5 — permutation threshold: one per chromosome, top-LOD gene per chr
    # Matches miQTL: descending-LOD order, first valid gene, no focal gene bias.
    # Piezo1 is reserved for additional plots only (via PRIORITY_GENE, never permutation).
    # ─────────────────────────────────────────────────────────────────────────
    sec = 5
    r_perm_lod_path = r_data / f'perm_max_lods_chr{peak_chr}.csv'
    if sec in sections:
        log.info('\nS5: permutation thresholds (all chromosomes, top-LOD gene per chr)...')
        if not _checkpoint(py_dir, str(sec), force):
            _rss_s5 = _get_rss_mb()
            tracemalloc.start()
            _t_s5 = time.perf_counter()
            if not peak_lod_path.exists():
                log.info('  %s s4_peak_lod.csv not found — run S4 first', SKIP)
                results['S5_perm_threshold'] = None
            else:
                peak_py_s5 = pd.read_csv(peak_lod_path)
                h5_path    = data_dir / 'counts_lognorm.h5'
                chr_null_lods_s5 = {}
                thr_rows_s5: list[dict] = []
                n_genes_s5_total = 0

                for _ch in chrs_to_scan:
                    _sub = peak_py_s5[peak_py_s5['chr'].astype(str) == str(_ch)]
                    if len(_sub) == 0:
                        log.info('  chr%s: no genes in s4_peak_lod — skipping', _ch)
                        continue
                    # top-LOD gene (miQTL native: descending LOD, no focal gene bias)
                    _perm_gene = _sub.loc[_sub['peak_lod'].idxmax(), 'gene']
                    _perm_id   = gene_name_to_id.get(_perm_gene, _perm_gene)
                    log.info('  chr%s  perm gene: %s', _ch, _perm_gene)

                    _cts, _lds, _sids = _load_expression(h5_path, [_perm_id])
                    if _cts.shape[0] == 0:
                        log.info('  chr%s  %s expression not found — skipping', _ch, _perm_id)
                        continue

                    _gids = [meta_geno_map.get(s, s) for s in _sids]
                    _y    = _cts[0].astype(float)
                    _cov  = _build_covariates(meta, _sids)
                    _null = generate_null_outcomes(_y, n_perms=250, seed=1)
                    _ch_max, _ch_nulls = run_threshold_scans(
                        _null, cache,
                        phenotype=_y,
                        covariates=_cov,
                        kinship_decomp={_ch: kinship_decomp[_ch]},
                        chrom=_ch,
                        sample_ids=_gids,
                    )
                    chr_null_lods_s5[_ch] = _ch_nulls.get(_ch, _ch_max)
                    _ch_thr, _ = get_gev_thresholds(_ch_max, percentile=0.95)
                    thr_rows_s5.append({
                        'chr': _ch, 'threshold': _ch_thr,
                        'perm_gene': _perm_gene, 'n_perms': 250,
                    })
                    log.info('  chr%s  GEV threshold=%.3f', _ch, _ch_thr)
                    n_genes_s5_total += len(_sub)

                    # save per-chromosome null (all chrs — used by checkpoint reload)
                    _ch_null_path = py_dir / f's5_perm_max_lods_chr{_ch}.csv'
                    pd.DataFrame({'max_lod': _ch_nulls.get(_ch, _ch_max)}).to_csv(
                        _ch_null_path, index=False)
                    # also save peak_chr null as the canonical file for R comparison
                    if str(_ch) == str(peak_chr):
                        perm_max_lods_s5 = _ch_max
                        pd.DataFrame({'max_lod': _ch_max}).to_csv(
                            py_dir / 's5_perm_max_lods.csv', index=False)
                        if r_perm_lod_path.exists():
                            r_max_lods = pd.read_csv(r_perm_lod_path)['max_lod'].values
                            r_thr, _ = get_gev_thresholds(r_max_lods, percentile=0.95)
                            delta = abs(_ch_thr - r_thr)
                            log.info('  vs R chr%s: R GEV=%.3f  delta=%.3f',
                                     _ch, r_thr, delta)
                            _check('GEV threshold delta', delta, 0.5, '<')

                if thr_rows_s5:
                    pd.DataFrame(thr_rows_s5).to_csv(
                        py_dir / 's5_threshold.csv', index=False)
                    log.info('  s5_threshold.csv: %d chromosomes', len(thr_rows_s5))
                    results['S5_perm_threshold'] = True
                else:
                    results['S5_perm_threshold'] = None

            _, _s5_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            _record_stats('S5', 'permutation', n_genes_s5_total if 'n_genes_s5_total' in locals() else 0,
                          time.perf_counter() - _t_s5, _s5_peak / 1e6,
                          max(0.0, _get_rss_mb() - _rss_s5))
            _mark_done(py_dir, str(sec))

    # ── eGene calling (after S5 — per-chromosome thresholds from s5_threshold.csv)
    if peak_lod_path.exists():
        log.info('\n── eGene calling + GxE/PxG ──────────────────────────────────────')
        peak_py_loaded = pd.read_csv(peak_lod_path)
        py_thr_path = py_dir / 's5_threshold.csv'
        thr: 'float | None' = None
        if py_thr_path.exists():
            _thr_df  = pd.read_csv(py_thr_path)
            _thr_map = {str(r['chr']): float(r['threshold']) for _, r in _thr_df.iterrows()}
            thr = _thr_map.get(str(peak_chr))  # scalar for logging / plots
            if _thr_map:
                eff_path = py_dir / 's4_allele_effects.csv'
                eff_tmp: 'pd.DataFrame | None' = None
                if eff_path.exists():
                    eff_tmp = pd.read_csv(
                        eff_path,
                        usecols=['gene', 'gene_id', 'chr', 'peak_locus', 'peak_mb', 'peak_lod'])
                _frames: 'list[pd.DataFrame]' = []
                _no_thr = 0
                for _ch, _ch_df in peak_py_loaded.groupby(
                        peak_py_loaded['chr'].astype(str)):
                    _ch_thr = _thr_map.get(str(_ch))
                    if _ch_thr is None:
                        _no_thr += len(_ch_df)
                        continue
                    _ch_eff = (eff_tmp[eff_tmp['chr'].astype(str) == str(_ch)]
                               if eff_tmp is not None else None)
                    # R uses strict `lod > thr`; pre-filter to match before call_egenes (>=)
                    _ch_df_strict = _ch_df[_ch_df['peak_lod'] > _ch_thr]
                    _frames.append(call_egenes(_ch_df_strict, _ch_thr, effects_df=_ch_eff))
                if _no_thr:
                    log.warning('[eGenes] %d genes on chromosomes with no threshold',
                                _no_thr)
                egenes_df = (pd.concat(_frames, ignore_index=True)
                             if _frames else pd.DataFrame())
                log.info('[eGenes] per-chr thresholds → %d / %d genes called',
                         len(egenes_df), len(peak_py_loaded))

            # ── reload perm nulls from CSV when running from checkpoint ─────────
            if perm_max_lods_s5 is None:
                _pml_path = py_dir / 's5_perm_max_lods.csv'
                if _pml_path.exists():
                    perm_max_lods_s5 = pd.read_csv(_pml_path)['max_lod'].values
                    perm_n_s5 = len(perm_max_lods_s5)
                    # load all per-chromosome nulls saved during S5 run
                    chr_null_lods_s5 = {}
                    for _f in sorted(py_dir.glob('s5_perm_max_lods_chr*.csv')):
                        _ch_key = _f.stem.replace('s5_perm_max_lods_chr', '')
                        chr_null_lods_s5[_ch_key] = pd.read_csv(_f)['max_lod'].values
                    if not chr_null_lods_s5:
                        chr_null_lods_s5 = {peak_chr: perm_max_lods_s5}

            # ── per-gene empirical p-values and Storey q-values ─────────────────
            if perm_max_lods_s5 is not None:
                p_chr_list: list[float] = []
                p_gnm_list: list[float] = []
                for _, grow in peak_py_loaded.iterrows():
                    lod_g    = float(grow['peak_lod'])
                    gene_ch  = str(grow.get('chr', peak_chr))
                    # use each gene's own chromosome null; fall back to peak_chr null
                    null_chr = chr_null_lods_s5.get(gene_ch, perm_max_lods_s5)
                    p_chr_list.append(float((null_chr         >= lod_g).sum()) / len(null_chr))
                    p_gnm_list.append(float((perm_max_lods_s5 >= lod_g).sum()) / len(perm_max_lods_s5))
                p_chr_arr = np.array(p_chr_list, dtype=float)
                p_gnm_arr = np.array(p_gnm_list, dtype=float)
                q_arr, pi0_val = storey_qvalue(p_chr_arr)
                genes_all = peak_py_loaded['gene'].values
                egenes_df['p_permtest_chr']  = egenes_df['gene'].map(dict(zip(genes_all, p_chr_arr)))
                egenes_df['q_fdr_storey']    = egenes_df['gene'].map(dict(zip(genes_all, q_arr)))
                egenes_df['pi0']               = pi0_val
                egenes_df['perm_n']            = perm_n_s5
                egenes_df['perm_chr']          = peak_chr
                log.info('[Storey] pi0=%.4f  eGenes at q<0.05: %d',
                         pi0_val, int((egenes_df['q_fdr_storey'] < 0.05).sum()))

            # ── LMM GxE (haplotype × sex) at peak locus via qtl_gxe_lrt ────────
            # kinship-corrected: qtl_gxe_lrt uses LOCO eigendecomp
            log.info('  GxE/PxG: computing for %d eGenes → pxg/ directory ...',
                     len(egenes_df))
            h5_path = data_dir / 'counts_lognorm.h5'
            gxe_rows: list[dict] = []
            pxg_dir = plt_dir / 'pxg'
            pxg_dir.mkdir(exist_ok=True)
            for _, erow in egenes_df.iterrows():
                gene_n     = erow['gene']
                gene_id_a  = gene_name_to_id.get(gene_n, gene_n)
                gene_chr_a = str(erow.get('chr', peak_chr))
                peak_loc_a = str(erow.get('peak_locus', ''))
                peak_mb_a  = float(erow.get('peak_mb', 0.0))
                lod_add_a  = float(erow.get('peak_lod', 0.0))
                if not h5_path.exists() or not peak_loc_a:
                    break
                counts_a, _, sids_a = _load_expression(h5_path, [gene_id_a])
                if counts_a.shape[0] == 0:
                    continue
                geno_ids_a = [meta_geno_map.get(s, s) for s in sids_a]
                y_a = counts_a[0].astype(float)
                try:
                    locus_ids_a, _, probs_a = cache.get_chromosome(gene_chr_a)
                    locus_list_a = list(locus_ids_a)
                    if peak_loc_a not in locus_list_a:
                        continue
                    hap_probs_all = probs_a[locus_list_a.index(peak_loc_a)]  # (n_strains, J)

                    # Z-matrix strain-mean reduction (correct for replicates)
                    z_mat_a, valid_sid_a = build_z_matrix(geno_ids_a,
                                                          list(cache.sample_ids))
                    valid_a = np.where(valid_sid_a)[0]
                    if not valid_a.size:
                        continue
                    z_sub_a   = z_mat_a[valid_a]
                    y_sub_a   = y_a[valid_a]
                    sids_v_a  = [sids_a[i] for i in valid_a]
                    cov_sub_a = _build_covariates(meta, sids_v_a)  # (n_valid, p) dataset dummies

                    r_cnt_a   = z_sub_a.sum(axis=0)
                    active_a  = r_cnt_a > 0
                    z_act_a   = z_sub_a[:, active_a]
                    r_act_a   = r_cnt_a[active_a]

                    y_str_a   = (z_act_a.T @ y_sub_a)  / r_act_a
                    cov_str_a = (z_act_a.T @ cov_sub_a) / r_act_a[:, None]

                    sex_sub_a = np.array(
                        [float(_sex_by_sid.get(s, 0.0)) for s in sids_v_a])
                    sex_str_a = (z_act_a.T @ sex_sub_a) / r_act_a

                    active_idx_a  = np.where(active_a)[0]
                    h_locus_a     = hap_probs_all[active_idx_a]  # (n_act, J)

                    # sex as main-effect covariate for GxE model
                    cov_gxe_a = np.column_stack([cov_str_a, sex_str_a[:, None]])

                    # subset LOCO kinship to active strains and re-eigendecomp (~40×40)
                    ev_f  = kinship_decomp[gene_chr_a]['values']
                    evc_f = kinship_decomp[gene_chr_a]['vectors']
                    K_f   = evc_f @ np.diag(ev_f) @ evc_f.T
                    K_a   = K_f[np.ix_(active_idx_a, active_idx_a)]
                    K_a   = (K_a + K_a.T) / 2
                    ev_a, evc_a = np.linalg.eigh(K_a)

                    gxe_res = qtl_gxe_lrt(
                        y_str_a, h_locus_a, sex_str_a, cov_gxe_a,
                        ev_a, evc_a, lod_add_a,
                    )
                    gxe_res['gene'] = gene_n
                    gxe_rows.append(gxe_res)

                    # PxG sex-faceted plot — sample level so M/F are binary per dot
                    # (strain-mean sex = 0.5 for 1M+1F strains rounds to 0=male via
                    # banker's rounding, hiding all female data; use per-sample instead)
                    sex_samp_a = np.array([int(round(float(
                        _sex_by_sid.get(s, 0.0)))) for s in sids_v_a])
                    h_samp_a   = (z_sub_a[:, active_a]
                                  @ hap_probs_all[active_idx_a])  # (n_valid, J)
                    y_pxg_a    = rint(y_sub_a.astype(float))
                    try:
                        fig_pxg = pxg_sex_facet(
                            y_pxg_a, h_samp_a, sex_samp_a,
                            founders=_display_founders(list(cache.founders)),
                            phenotype_label='Expression (RINT)',
                            chrom=gene_chr_a, position_mb=peak_mb_a,
                        )
                        fig_pxg.savefig(pxg_dir / f'pxg_{gene_n}.png',
                                        dpi=300, bbox_inches='tight')
                        log.debug('  [pxg] saved pxg_%s.png', gene_n)
                    except Exception as _pe:
                        log.warning('  [pxg] pxg_%s failed: %s', gene_n, _pe)
                    finally:
                        plt.close('all')

                    # Haplotype × sex profile with LMM significance annotation
                    # use sample-level data so M/F are binary per dot (not strain means)
                    fig_gxe, ax_gxe = plt.subplots(figsize=(6, 3.5), dpi=300)
                    try:
                        haplotype_sex_profile(
                            y_pxg_a, h_samp_a, sex_samp_a,
                            founders=_display_founders(list(cache.founders)),
                            ax=ax_gxe,
                            phenotype_label='Expression (RINT)',
                            chrom=gene_chr_a, position_mb=peak_mb_a,
                            gxe_result=gxe_res,
                        )
                        fig_gxe.tight_layout()
                        fig_gxe.savefig(pxg_dir / f'gxe_profile_{gene_n}.png',
                                        dpi=300, bbox_inches='tight')
                        log.debug('  [gxe] saved gxe_profile_%s.png', gene_n)
                    except Exception as _ge:
                        log.warning('  [gxe] gxe_profile_%s failed: %s', gene_n, _ge)
                    finally:
                        plt.close(fig_gxe)

                except Exception as exc:
                    log.warning('  [GxE/PxG] failed for %s: %s', gene_n, exc)

            if gxe_rows:
                gxe_df = pd.DataFrame(gxe_rows)[
                    ['gene', 'chi2_haplotype', 'p_haplotype',
                     'lod_gxe_sex', 'p_gxe_sex']]
                egenes_df = egenes_df.merge(gxe_df, on='gene', how='left')
                log.info('[GxE] LMM haplotype×sex computed for %d eGenes',
                         len(gxe_rows))

            # ── eqtl_summary CSV ───────────────────────────────────────────────
            eqtl_sum = egenes_df.copy()
            r_egenes_path = r_data / 'egenes.csv'
            if r_egenes_path.exists():
                egenes_r = pd.read_csv(r_egenes_path)
                eqtl_sum['in_r_egenes'] = eqtl_sum['gene'].isin(egenes_r['gene'])
            col_order = ['gene', 'chr', 'peak_locus', 'peak_mb', 'peak_lod',
                         'p_permtest_chr', 'q_fdr_storey',
                         'pi0', 'perm_n',
                         'chi2_haplotype', 'p_haplotype', 'lod_gxe_sex', 'p_gxe_sex',
                         'in_r_egenes']
            eqtl_sum = eqtl_sum[[c for c in col_order if c in eqtl_sum.columns]]
            eqtl_sum.to_csv(py_dir / 'eqtl_summary.csv', index=False)
            log.info('[eqtl_summary] %d eGenes → eqtl_summary.csv', len(eqtl_sum))
        else:
            log.info('[eGenes] no threshold available — run S5 first')

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
                # batch-load all eGene expressions once
                _s7_ids = [gene_name_to_id.get(r['gene'], r['gene'])
                           for _, r in egenes_df.iterrows()]
                _s7_counts, _s7_loaded, _s7_sids = _load_expression(h5_path, _s7_ids)
                _s7_cache = {gid: _s7_counts[i] for i, gid in enumerate(_s7_loaded)}
                _n_s7 = len(egenes_df)
                mi_rows = []
                for _i_s7, (_, erow) in enumerate(egenes_df.iterrows()):
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    gene_chr  = str(erow.get('chr', peak_chr))
                    if gene_id not in _s7_cache:
                        continue
                    sids_g     = _s7_sids
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = _s7_cache[gene_id].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    sex_g = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sids_g])
                    log.info('  [%d/%d] MI scan: %s  chr%s',
                             _i_s7 + 1, _n_s7, gene_name, gene_chr)
                    _t_s7g = time.perf_counter()
                    mi_scan = scan_h2lmm_mi(
                        y_g, cache,
                        covariates=cov_g,
                        kinship_decomp=kinship_decomp,
                        sample_ids=geno_ids_g,
                        n_imp=11, seed=1,
                        phenotype_id=gene_name,
                        scan_window='cis', gene_chr=gene_chr,
                    )
                    log.info('    done in %.1fs', time.perf_counter() - _t_s7g)
                    mi_rows.append(_scan_to_df(mi_scan, gene_name))

                if mi_rows:
                    mi_df = pd.concat(mi_rows, ignore_index=True)
                    mi_df.to_csv(py_dir / 's7_mi_lod.csv', index=False)
                    log.info('  MI scan complete: %d genes', len(mi_rows))
                    results['S7_mi_scan'] = True
                    # optional: compare to R
                    if r_mi_path.exists():
                        r_mi = pd.read_csv(r_mi_path)
                        merged_mi = _pos_merge(mi_df, r_mi, 'lod', 'lod')
                        if len(merged_mi) >= 10:
                            r_mi_r, _ = sp_stats.pearsonr(merged_mi['py'],
                                                           merged_mi['r'])
                            merged_mi.to_csv(cmp_dir / 's7_mi_cmp.csv', index=False)
                            _check('MI LOD r', abs(r_mi_r), 0.75)
                            log.info('  vs R: r=%.4f  n=%d', r_mi_r, len(merged_mi))
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
                # batch-load all eGene expressions once
                _s8_ids = [gene_name_to_id.get(r['gene'], r['gene'])
                           for _, r in egenes_df.iterrows()]
                _s8_counts, _s8_loaded, _s8_sids = _load_expression(h5_path, _s8_ids)
                _s8_cache = {gid: _s8_counts[i] for i, gid in enumerate(_s8_loaded)}
                _n_s8 = len(egenes_df)
                snp_rows = []
                for _i_s8, (_, erow) in enumerate(egenes_df.iterrows()):
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    gene_chr  = str(erow.get('chr', peak_chr))
                    if gene_id not in _s8_cache:
                        continue
                    sids_g     = _s8_sids
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = _s8_cache[gene_id].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    log.info('  [%d/%d] SNP scan: %s  chr%s',
                             _i_s8 + 1, _n_s8, gene_name, gene_chr)
                    _t_s8g = time.perf_counter()
                    try:
                        snp_scan = imputed_snp_scan_from_db(
                            y_g, cache,
                            alleles_h5=str(ALLELE_H5),
                            chrom=gene_chr,
                            covariates=cov_g,
                            kinship_decomp={gene_chr: kinship_decomp[gene_chr]},
                            sample_ids=geno_ids_g,
                            founder_col_map=FOUNDER_CODE,
                        )
                        log.info('    done in %.1fs', time.perf_counter() - _t_s8g)
                        if not snp_scan.empty:
                            snp_scan = snp_scan.copy()
                            snp_scan.insert(0, 'gene', gene_name)
                            snp_rows.append(snp_scan)
                    except Exception as exc:
                        log.warning('  SNP scan failed for %s: %s', gene_id, exc)

                if snp_rows:
                    snp_df = pd.concat(snp_rows, ignore_index=True)
                    snp_df.to_csv(py_dir / 's8_snp_lod.csv', index=False)
                    if r_snp_path.exists():
                        r_snp = pd.read_csv(r_snp_path)
                        merged_snp = _pos_merge(snp_df, r_snp, 'lod', 'lod')
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
                            results['S8_snp_lod_r'] = (abs(r_snp_r) >= 0.75 and
                                                        abs(slope_snp - 1.0) < 0.20)
                        else:
                            log.info('  %s too few matched SNPs', SKIP)
                            results['S8_snp_lod_r'] = None
                    else:
                        log.info('  Python SNP scan complete (%d SNPs across %d genes) '
                                 '— no R reference yet (run gastroc_miqtl_scan.R S8)',
                                 len(snp_df), snp_df['gene'].nunique())
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
                _s9_all_ids = list(dict.fromkeys(
                    gene_name_to_id.get(_nm, _nm) for _nm in _s9_all_names
                ))
                _s9_counts, _s9_loaded, _s9_sids = _load_expression(h5_path, _s9_all_ids)
                _s9_expr = {gid: _s9_counts[i] for i, gid in enumerate(_s9_loaded)}
                log.info('  pre-loaded %d gene expressions (%d unique IDs) in 1 HDF5 call',
                         len(_s9_loaded), len(_s9_all_ids))

                _n_s9 = len(_pairs_by_outcome)
                med_rows = []
                for _i_s9, (outcome_name, med_list) in enumerate(_pairs_by_outcome.items()):
                    outcome_id = gene_name_to_id.get(outcome_name, outcome_name)
                    if outcome_id not in _s9_expr:
                        continue
                    sids_g     = _s9_sids
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = _s9_expr[outcome_id].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    ch    = med_list[0][1]   # all mediators share same chr

                    med_vecs: dict[str, 'np.ndarray'] = {}
                    for med_name, _ in med_list:
                        med_id = gene_name_to_id.get(med_name, med_name)
                        if med_id in _s9_expr:
                            med_vecs[med_name] = _s9_expr[med_id].astype(float)

                    if not med_vecs:
                        continue
                    log.info('  [%d/%d] mediation: %s → %d mediator(s)',
                             _i_s9 + 1, _n_s9, outcome_name, len(med_vecs))
                    _t_med = time.perf_counter()
                    try:
                        batch_df = mediation_scan_qr_batch(
                            y_g, med_vecs, cache,
                            covariates=cov_g,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_g,
                            outcome_id=outcome_name,
                            scan_window='cis', gene_chr=ch,
                        )
                        log.info('    done in %.1fs', time.perf_counter() - _t_med)
                        if batch_df.empty:
                            continue
                        batch_df = batch_df.rename(columns={
                            'outcome_id': 'gene', 'mediator_id': 'mediator'})
                        batch_df = batch_df[
                            (batch_df['lod_base'] >= 1.5) &
                            (batch_df['lod_base'] - batch_df['lod_mediated'] >= 0.5)
                        ]
                        med_rows.append(
                            batch_df[['gene', 'mediator', 'chr', 'locus',
                                      'pos_mb', 'lod_base', 'lod_mediated']])
                    except Exception as exc:
                        log.warning('  mediation batch failed for %s: %s',
                                    outcome_name, exc)
                if med_rows:
                    log.info('  writing mediation results ...')
                    med_df = pd.concat(med_rows, ignore_index=True)
                    med_df.to_csv(py_dir / 's9_mediation.csv', index=False)
                    log.info('  mediation complete: %d genes', len(med_rows))
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
                        merged_med = _pos_merge(med_df, r_med, 'lod_mediated', 'lod')
                        if len(merged_med) >= 10:
                            r_m, _ = sp_stats.pearsonr(merged_med['py'],
                                                        merged_med['r'])
                            merged_med.to_csv(cmp_dir / 's9_mediation_cmp.csv',
                                              index=False)
                            _check('mediation LOD r', abs(r_m), 0.90)
                            log.info('  vs R: r=%.4f  n=%d', r_m, len(merged_med))
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
                # batch-load all eGene expressions once (avoids per-gene open/scan/close cycles)
                _s10_ids = [gene_name_to_id.get(r['gene'], r['gene'])
                            for _, r in egenes_df.iterrows()]
                _s10_counts, _s10_loaded, _s10_sids = _load_expression(h5_path, _s10_ids)
                _s10_cache = {gid: _s10_counts[i] for i, gid in enumerate(_s10_loaded)}
                _n_s10 = len(egenes_df)
                ci_rows = []
                for _i_s10, (_, erow) in enumerate(egenes_df.iterrows()):
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    gene_chr  = str(erow.get('chr', peak_chr))
                    if gene_id not in _s10_cache:
                        continue
                    sids_g     = _s10_sids
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = _s10_cache[gene_id].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    sex_g = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sids_g])
                    log.info('  [%d/%d] bootstrap CI: %s  chr%s',
                             _i_s10 + 1, _n_s10, gene_name, gene_chr)
                    _t_s10g = time.perf_counter()
                    try:
                        kd_pc    = {gene_chr: kinship_decomp[gene_chr]}
                        scan_g   = scan_h2lmm(
                            y_g, cache,
                            covariates=cov_g,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_g,
                            use_rint=True,
                            phenotype_id=gene_name,
                            scan_window='cis', gene_chr=gene_chr,
                        )
                        boot_out = generate_null_outcomes(
                            y_g, n_perms=200, seed=1, method='bootstrap')
                        ci_mbs, _ci_info = run_positional_scans(
                            boot_out, cache,
                            phenotype=y_g,
                            covariates=cov_g,
                            kinship_decomp=kd_pc,
                            peak_chrom=gene_chr,
                            sample_ids=geno_ids_g,
                        )
                        peak_mb = float(scan_g.positions[scan_g.lod.argmax()])
                        log.info('    done in %.1fs', time.perf_counter() - _t_s10g)
                        ci_rows.append({
                            'gene':    gene_name,
                            'chr':     gene_chr,
                            'peak_mb': peak_mb,
                            'ci_lo':   float(np.quantile(ci_mbs, 0.025)),
                            'ci_hi':   float(np.quantile(ci_mbs, 0.975)),
                            'ci_width_mb': float(np.quantile(ci_mbs, 0.975) -
                                                  np.quantile(ci_mbs, 0.025)),
                        })
                    except Exception as exc:
                        log.warning('  CI failed for %s: %s', gene_id, exc)
                if ci_rows:
                    ci_df = pd.DataFrame(ci_rows)
                    ci_df.to_csv(py_dir / 's10_ci.csv', index=False)
                    log.info('  CI complete: %d genes  median width=%.2f Mb',
                             len(ci_rows),
                             float(np.median([r['ci_width_mb'] for r in ci_rows])))
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
                h5_path   = data_dir / 'counts_lognorm.h5'
                # batch-load all eGene expressions once (avoids per-gene open/scan/close cycles)
                _s11_ids = [gene_name_to_id.get(r['gene'], r['gene'])
                            for _, r in egenes_df.iterrows()]
                _s11_counts, _s11_loaded, _s11_sids = _load_expression(h5_path, _s11_ids)
                _s11_cache = {gid: _s11_counts[i] for i, gid in enumerate(_s11_loaded)}
                _n_s11 = len(egenes_df)
                # instability_lm_scan calls fixef_eqtl_scan 100× per gene (per subsample),
                # each iterating all ~104k loci with just_these_loci set to filter one chr.
                # Time per gene scales with n_subsamples × total_loci_in_cache × per-locus cost.
                inst_rows = []
                for _i_s11, (_, erow) in enumerate(egenes_df.iterrows()):
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    gene_chr  = str(erow.get('chr', peak_chr))
                    if gene_id not in _s11_cache:
                        continue
                    sids_g     = _s11_sids
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = _s11_cache[gene_id].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    sex_g = np.array([float(_sex_by_sid.get(s, 0.0)) for s in sids_g])
                    log.info('  [%d/%d] instability: %s  chr%s',
                             _i_s11 + 1, _n_s11, gene_name, gene_chr)
                    _t_s11g = time.perf_counter()
                    try:
                        _lids_inst, _, _ = cache.get_chromosome(gene_chr)
                        inst = instability_lm_scan(
                            y_g, cache,
                            covariates=cov_g,
                            sample_ids=geno_ids_g,
                            n_subsamples=100, seed=1,
                            just_these_loci=set(_lids_inst),
                        )
                        log.info('    done in %.1fs', time.perf_counter() - _t_s11g)
                        inst_rows.append(_scan_to_df(inst, gene_name))
                    except Exception as exc:
                        log.warning('  instability failed for %s: %s', gene_id, exc)
                if inst_rows:
                    inst_df = pd.concat(inst_rows, ignore_index=True)
                    inst_df.to_csv(py_dir / 's11_instability.csv', index=False)
                    log.info('  instability complete: %d genes', len(inst_rows))
                    results['S11_instability'] = True
                    if r_inst_path.exists():
                        r_inst      = pd.read_csv(r_inst_path)
                        merged_inst = _pos_merge(inst_df, r_inst,
                                                  'lod', 'neg_log10_p')
                        if len(merged_inst) >= 10:
                            # R exports neg_log10_p (median -log10 F-test p across subsamples)
                            # Python exports lod (mean bootstrap LOD from fixef scan)
                            # Both are monotone with instability signal but on different scales
                            # — use Spearman rank correlation which is scale-invariant
                            r_i, _ = sp_stats.spearmanr(merged_inst['py'],
                                                          merged_inst['r'])
                            merged_inst.to_csv(cmp_dir / 's11_instability_cmp.csv',
                                               index=False)
                            _check('instability rho', abs(r_i), 0.90)
                            log.info('  vs R (Spearman): rho=%.4f  n=%d'
                                     '  [py=lod, r=neg_log10_p — rank correlation]',
                                     r_i, len(merged_inst))
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

    # ── comparison plots (only when R data present) ───────────────────────────
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

    scan_desc = scan_type if scan_type != 'L' else f'L ±{window_mb:.1f} Mb'
    log.info('validate_gastroc.py  sections=%s  force=%s  n_genes=%s  '
             'scan_type=%s  chr=%s',
             sorted(sections), args.force, args.n_genes, scan_desc, args.chr)
    log.info('  data_dir = %s', GASTROC_DATA)
    log.info('  r_data   = %s', GASTROC_DATA / 'r_data')
    log.info('  out_dir  = %s', OUT_DIR)

    run(GASTROC_DATA, OUT_DIR, sections, args.force, args.n_genes,
        scan_type=scan_type, window_mb=window_mb, scan_chr=scan_chr)


if __name__ == '__main__':
    main()

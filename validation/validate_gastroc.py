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

from mosaiqtl.genotype.founders import F1_COLORS, FOUNDER_ORDER
from mosaiqtl.genotype.genomecache import GenomeCache
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
from mosaiqtl.qtl.mediation import mediation_scan_qr
from mosaiqtl.qtl.snp_scan import imputed_snp_scan_from_db

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

ALLELE_H5    = Path('validation/gastroc_data/founder_alleles.h5')

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
    """build dataset one-hot covariate matrix aligned to sample_order.

    dataset has 5 levels: CC1, CC2, CC3, F1, Founder.
    CC1 is the reference (alphabetically first); dummies = CC2, CC3, F1, Founder.
    scan_h2lmm prepends the intercept internally, so we return 4 columns.
    """
    meta_idx = meta.set_index('sample_id')
    datasets = meta_idx.loc[sample_order, 'dataset'].values
    dummies  = pd.get_dummies(datasets, drop_first=True).values.astype(np.float64)
    return dummies   # (n_samples, 4)


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

FOUNDER_COLORS = {
    'A/J': '#F4C245', 'B6/J': '#C0BFBF', '129S1/J': '#DA9CC1',
    'NOD/J': '#4F6EAF', 'NZO/J': '#52A5DB', 'CAST/J': '#55AF5B',
    'PWK/J': '#D83026', 'WSB/J': '#683C91',
}


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
      s1_loci_per_chr.png       — loci per chromosome bar (qtl2: plot_scan1 x-axis)
      s2_haplotype_heatmap.png  — founder prob heatmap at sampled loci (qtl2: plot_genoprob)
      s3_kinship_heatmap.png    — LOCO kinship matrix + diagonal distribution
      s4_lod_profile.png        — LOD profile for top gene (qtl2: plot_scan1)
      s4_founder_effects.png    — founder coef bar at peak locus (qtl2: plot_coef)
      s4_manhattan.png          — peak LOD per gene across chr (qtl2-style Manhattan)
      s5_perm_distribution.png  — max-LOD histogram + GEV fit + threshold line
      s10_ci_strip.png          — CI strip per eGene (qtl2: plot_scan1 with interval)
      s11_instability_profile.png — instability LOD profile for top gene
    """
    sns.set_style('ticks')
    f_colors = {f: FOUNDER_COLORS.get(f, '#888888') for f in founders}

    # ── S1: loci per chromosome bar ───────────────────────────────────────────
    s1_path = py_dir / 's1_cache_info.csv'
    if s1_path.exists():
        try:
            ci = pd.read_csv(s1_path)
            ci['chr'] = ci['chr'].astype(str)
            # sort chromosomes naturally: autosomes then X
            def _chr_key(c):
                try:
                    return (0, int(c))
                except ValueError:
                    return (1, c)
            ci = ci.sort_values('chr', key=lambda s: [_chr_key(x) for x in s])
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.bar(range(len(ci)), ci['n_loci'], color='#4F6EAF', width=0.7)
            ax.set_xticks(range(len(ci)))
            ax.set_xticklabels(ci['chr'].tolist(), fontsize=7)
            ax.set_xlabel('Chromosome')
            ax.set_ylabel('Number of loci')
            ax.set_title(f'S1  Loci per chromosome  (total={ci["n_loci"].sum():,})')
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 's1_loci_per_chr.png', dpi=300)
            plt.close(fig)
            log.info('  s1_loci_per_chr.png')
        except Exception as exc:
            log.warning('  s1_loci_per_chr: %s', exc)

    # ── S2: haplotype prob heatmap (genotype × founder, canonical order) ───────
    s2_path = py_dir / 's2_haplotype_probs.csv'
    if s2_path.exists():
        try:
            from mosaiqtl.genotype.founders import FOUNDER_ORDER as _FO, F1_COLORS as _F1
            hp       = pd.read_csv(s2_path)
            locus0   = hp['locus'].iloc[0]
            sub      = hp[hp['locus'] == locus0].copy()
            # enforce canonical row order from cache.sample_ids
            strain_order = list(cache.sample_ids)
            sub      = sub.set_index('genotype').reindex(strain_order).reset_index()
            mat      = sub[founders].values  # (n_strains, 8)
            strains  = sub['genotype'].tolist()

            # group boundary indices
            fo_set  = set(_FO)
            f1_set  = set(_F1.keys())
            n_fo    = sum(1 for s in strains if s in fo_set)
            n_f1    = sum(1 for s in strains if s in f1_set)
            bounds  = [n_fo - 0.5, n_fo + n_f1 - 0.5]

            # abbreviated labels: show all if ≤60, else thin out CC strains
            max_labels = 60
            if len(strains) <= max_labels:
                yticks  = list(range(len(strains)))
                ylabels = [s.split('/')[0] for s in strains]
                yfsize  = max(4, 7 - len(strains) // 15)
            else:
                fo_f1_n = n_fo + n_f1
                step    = max(1, (len(strains) - fo_f1_n) // (max_labels - fo_f1_n))
                yticks  = list(range(fo_f1_n)) + list(range(fo_f1_n, len(strains), step))
                ylabels = [strains[i].split('/')[0] for i in yticks]
                yfsize  = 6

            fig_h = max(5, len(yticks) * 0.18)
            fig, ax = plt.subplots(figsize=(6, fig_h))
            im = ax.imshow(mat, aspect='auto', vmin=0, vmax=1,
                           cmap='YlOrRd', interpolation='nearest')
            ax.set_xticks(range(len(founders)))
            ax.set_xticklabels([f.split('/')[0] for f in founders],
                               fontsize=7, rotation=45, ha='right')
            ax.set_yticks(yticks)
            ax.set_yticklabels(ylabels, fontsize=yfsize)
            # separator lines between Founder / F1 / CC groups
            for b in bounds:
                if 0 < b < len(strains) - 1:
                    ax.axhline(b, color='white', linewidth=1.5, linestyle='--')
            # group labels on right margin
            group_mids = {
                'Founders': (0 + n_fo - 1) / 2 if n_fo > 0 else -1,
                'F1s':      (n_fo + n_fo + n_f1 - 1) / 2 if n_f1 > 0 else -1,
                'CC':       (n_fo + n_f1 + len(strains) - 1) / 2 if len(strains) > n_fo + n_f1 else -1,
            }
            for label, mid in group_mids.items():
                if mid >= 0:
                    ax.text(len(founders) + 0.1, mid, label, va='center',
                            fontsize=6, color='#444444',
                            transform=ax.get_yaxis_transform())
            ax.set_xlabel('Founder')
            ax.set_ylabel('Strain (canonical order)')
            ax.set_title(f'S2  Haplotype probs  locus={locus0}  '
                         f'(n={len(strains)} strains)')
            fig.colorbar(im, ax=ax, label='Probability', shrink=0.5,
                         pad=0.12)
            sns.despine(ax=ax, left=True, bottom=True)
            fig.tight_layout()
            fig.savefig(plt_dir / 's2_haplotype_heatmap.png', dpi=300)
            plt.close(fig)
            log.info('  s2_haplotype_heatmap.png  (%d strains, canonical order)',
                     len(strains))
        except Exception as exc:
            log.warning('  s2_haplotype_heatmap: %s', exc)

    # ── S3: kinship heatmap + diagonal distribution ───────────────────────────
    if peak_chr in kinship_decomp:
        try:
            from mosaiqtl.genotype.founders import FOUNDER_ORDER as _FO3, F1_COLORS as _F1_3
            kd  = kinship_decomp[peak_chr]
            K   = kd['vectors'] @ np.diag(kd['values']) @ kd['vectors'].T
            n   = K.shape[0]

            # canonical strain order from cache
            strains_k = list(cache.sample_ids)
            fo_set_k  = set(_FO3)
            f1_set_k  = set(_F1_3.keys())
            n_fo_k    = sum(1 for s in strains_k if s in fo_set_k)
            n_f1_k    = sum(1 for s in strains_k if s in f1_set_k)
            bounds_k  = [n_fo_k - 0.5, n_fo_k + n_f1_k - 0.5]

            # abbreviated tick labels (shared for both axes)
            max_labels = 60
            if len(strains_k) <= max_labels:
                kticks  = list(range(len(strains_k)))
                klabels = [s.split('/')[0] for s in strains_k]
                kfsize  = max(4, 7 - len(strains_k) // 15)
            else:
                fo_f1_n_k = n_fo_k + n_f1_k
                step_k    = max(1, (len(strains_k) - fo_f1_n_k) // (max_labels - fo_f1_n_k))
                kticks    = list(range(fo_f1_n_k)) + list(range(fo_f1_n_k, len(strains_k), step_k))
                klabels   = [strains_k[i].split('/')[0] for i in kticks]
                kfsize    = 6

            fig_h_k = max(5, len(kticks) * 0.18)
            fig, axes = plt.subplots(1, 2, figsize=(fig_h_k + 2, fig_h_k))
            # heatmap
            im = axes[0].imshow(K, aspect='auto', vmin=0, vmax=K.max(),
                                cmap='Blues', interpolation='nearest')
            axes[0].set_title(f'S3  LOCO kinship  chr{peak_chr}  (n={n})',
                              fontsize=9)
            axes[0].set_xlabel('Strain (canonical order)', fontsize=8)
            axes[0].set_ylabel('Strain (canonical order)', fontsize=8)
            axes[0].set_xticks(kticks)
            axes[0].set_xticklabels(klabels, fontsize=kfsize,
                                    rotation=90, ha='center')
            axes[0].set_yticks(kticks)
            axes[0].set_yticklabels(klabels, fontsize=kfsize)
            # separator lines between Founder / F1 / CC groups
            for b in bounds_k:
                if 0 < b < len(strains_k) - 1:
                    axes[0].axhline(b, color='white', linewidth=1.5, linestyle='--')
                    axes[0].axvline(b, color='white', linewidth=1.5, linestyle='--')
            fig.colorbar(im, ax=axes[0], shrink=0.8)
            sns.despine(ax=axes[0], left=True, bottom=True)
            # diagonal distribution
            diag = np.diag(K)
            tril = K[np.tril_indices(n, -1)]
            sns.kdeplot(diag, ax=axes[1], label='diagonal', color='#4F6EAF')
            sns.kdeplot(tril, ax=axes[1], label='off-diagonal', color='#DA9CC1')
            axes[1].set_xlabel('Kinship value')
            axes[1].set_title('S3  Kinship distribution')
            axes[1].legend(fontsize=8)
            sns.despine(ax=axes[1])
            fig.tight_layout()
            fig.savefig(plt_dir / 's3_kinship_heatmap.png', dpi=300)
            plt.close(fig)
            log.info('  s3_kinship_heatmap.png  (%d strains, canonical order)', n)
        except Exception as exc:
            log.warning('  s3_kinship_heatmap: %s', exc)

    # ── S4: LOD profile — top gene on peak_chr (qtl2: plot_scan1) ─────────────
    s4_lod_path = py_dir / 's4_lod.csv'
    if s4_lod_path.exists():
        try:
            lod_df = pd.read_csv(s4_lod_path)
            lod_df['chr'] = lod_df['chr'].astype(str)
            lod_pc  = lod_df[lod_df['chr'] == str(peak_chr)]
            if len(lod_pc) > 0:
                top_gene = (lod_pc.groupby('gene')['lod'].max()
                            .idxmax())
                lod_top  = lod_pc[lod_pc['gene'] == top_gene].sort_values('pos_mb')
                fig, ax  = plt.subplots(figsize=(10, 3))
                ax.plot(lod_top['pos_mb'], lod_top['lod'],
                        color='#4F6EAF', linewidth=1.2)
                ax.fill_between(lod_top['pos_mb'], 0, lod_top['lod'],
                                alpha=0.15, color='#4F6EAF')
                # threshold line if available
                thr_path = py_dir / 's5_threshold.csv'
                if thr_path.exists():
                    thr = float(pd.read_csv(thr_path)['threshold'].iloc[0])
                    ax.axhline(thr, color='#D83026', linewidth=0.8,
                               linestyle='--', label=f'GEV 95% = {thr:.2f}')
                    ax.legend(fontsize=8)
                ax.set_xlabel(f'Chr{peak_chr} position (Mb)')
                ax.set_ylabel('LOD')
                ax.set_title(f'S4  LOD profile  {top_gene}  chr{peak_chr}')
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_lod_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s4_lod_profile.png  (gene=%s)', top_gene)
        except Exception as exc:
            log.warning('  s4_lod_profile: %s', exc)

    # ── S4: founder coef bar at peak locus (qtl2: plot_coef) ──────────────────
    s4_eff_path = py_dir / 's4_allele_effects.csv'
    if s4_eff_path.exists():
        try:
            eff_df  = pd.read_csv(s4_eff_path)
            eff_pc  = eff_df[eff_df['chr'].astype(str) == str(peak_chr)]
            if len(eff_pc) > 0:
                top_row = eff_pc.loc[eff_pc['peak_lod'].idxmax()]
                beta_cols = [c for c in eff_pc.columns if c.startswith('beta_')]
                betas = [top_row[c] for c in beta_cols]
                fnames = [c.replace('beta_', '') for c in beta_cols]
                colors = [f_colors.get(f, '#888888') for f in founders
                          if f in [fn.replace('beta_', '') for fn in beta_cols]]
                colors = [f_colors.get(fn, '#888888') for fn in fnames]
                fig, ax = plt.subplots(figsize=(6, 3))
                bars = ax.bar(range(len(betas)), betas, color=colors, width=0.7)
                ax.axhline(0, color='black', linewidth=0.6)
                ax.set_xticks(range(len(fnames)))
                ax.set_xticklabels([f.split('/')[0] for f in fnames],
                                   fontsize=7, rotation=40, ha='right')
                ax.set_ylabel('Effect (beta)')
                ax.set_title(
                    f'S4  Founder effects  {top_row["gene"]}  '
                    f'{top_row["peak_locus"]}  LOD={top_row["peak_lod"]:.1f}')
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_founder_effects.png', dpi=300)
                plt.close(fig)
                log.info('  s4_founder_effects.png  (gene=%s)', top_row['gene'])
        except Exception as exc:
            log.warning('  s4_founder_effects: %s', exc)

    # ── S4: Manhattan — peak LOD per gene across all scanned chr ──────────────
    peak_lod_path = py_dir / 's4_peak_lod.csv'
    if peak_lod_path.exists() and s4_lod_path.exists():
        try:
            pk_df   = pd.read_csv(peak_lod_path)
            lod_all = pd.read_csv(s4_lod_path)
            lod_all['chr'] = lod_all['chr'].astype(str)
            # get peak position per gene
            idx_peak = lod_all.groupby('gene')['lod'].idxmax()
            peak_pos = lod_all.loc[idx_peak, ['gene', 'chr', 'pos_mb', 'lod']]
            peak_pos = peak_pos.merge(pk_df[['gene']], on='gene')

            def _chr_key(c):
                try:
                    return int(c)
                except ValueError:
                    return 100
            chrs_sorted = sorted(peak_pos['chr'].unique(), key=_chr_key)
            chr_offset  = {}
            off = 0.0
            for ch in chrs_sorted:
                chr_offset[ch] = off
                off += peak_pos[peak_pos['chr'] == ch]['pos_mb'].max() + 20
            peak_pos['x'] = peak_pos.apply(
                lambda r: r['pos_mb'] + chr_offset.get(r['chr'], 0), axis=1)

            fig, ax = plt.subplots(figsize=(12, 3))
            palette = ['#4F6EAF', '#52A5DB']
            for i, ch in enumerate(chrs_sorted):
                sub = peak_pos[peak_pos['chr'] == ch]
                ax.scatter(sub['x'], sub['lod'],
                           color=palette[i % 2], s=6, alpha=0.7, linewidths=0)
            thr_path = py_dir / 's5_threshold.csv'
            if thr_path.exists():
                thr = float(pd.read_csv(thr_path)['threshold'].iloc[0])
                ax.axhline(thr, color='#D83026', linewidth=0.8, linestyle='--',
                           label=f'GEV 95% = {thr:.2f}')
                ax.legend(fontsize=8)
            mid = {ch: chr_offset[ch] +
                   peak_pos[peak_pos['chr'] == ch]['pos_mb'].median()
                   for ch in chrs_sorted}
            ax.set_xticks(list(mid.values()))
            ax.set_xticklabels(list(mid.keys()), fontsize=7)
            ax.set_xlabel('Chromosome')
            ax.set_ylabel('Peak LOD')
            ax.set_title(f'S4  Manhattan  n={len(peak_pos)} genes')
            sns.despine(ax=ax)
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
            from scipy.stats import genextreme
            max_lods = pd.read_csv(s5_perm_path)['max_lod'].values
            max_lods = max_lods[np.isfinite(max_lods)]
            c, loc, scale = genextreme.fit(max_lods, f0=None)
            x_range = np.linspace(max_lods.min() - 0.5, max_lods.max() + 1, 200)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(max_lods, bins=20, density=True, color='#52A5DB',
                    alpha=0.6, label='Permutation max-LOD')
            ax.plot(x_range, genextreme.pdf(x_range, c, loc, scale),
                    color='#4F6EAF', linewidth=1.5, label='GEV fit')
            if s5_thr_path.exists():
                thr = float(pd.read_csv(s5_thr_path)['threshold'].iloc[0])
                ax.axvline(thr, color='#D83026', linewidth=1.2, linestyle='--',
                           label=f'95th pctile = {thr:.2f}')
            ax.set_xlabel('Max LOD')
            ax.set_ylabel('Density')
            ax.set_title(f'S5  Permutation distribution  chr{peak_chr}  n={len(max_lods)}')
            ax.legend(fontsize=8)
            sns.despine(ax=ax)
            fig.tight_layout()
            fig.savefig(plt_dir / 's5_perm_distribution.png', dpi=300)
            plt.close(fig)
            log.info('  s5_perm_distribution.png')
        except Exception as exc:
            log.warning('  s5_perm_distribution: %s', exc)

    # ── S10: CI strip per eGene (qtl2: LOD interval bar) ─────────────────────
    s10_path = py_dir / 's10_ci.csv'
    if s10_path.exists():
        try:
            ci_df = pd.read_csv(s10_path).sort_values('peak_mb')
            fig, ax = plt.subplots(figsize=(8, max(3, len(ci_df) * 0.35)))
            for i, row in enumerate(ci_df.itertuples()):
                ax.plot([row.ci_lo, row.ci_hi], [i, i],
                        color='#4F6EAF', linewidth=3, solid_capstyle='round')
                ax.scatter(row.peak_mb, i, color='#D83026', s=30, zorder=3)
            ax.set_yticks(range(len(ci_df)))
            ax.set_yticklabels(ci_df['gene'].tolist(), fontsize=7)
            ax.set_xlabel(f'Chr{peak_chr} position (Mb)')
            ax.set_title(f'S10  Bootstrap CI  {len(ci_df)} eGenes')
            sns.despine(ax=ax)
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
            inst_df  = pd.read_csv(s11_path)
            inst_df['chr'] = inst_df['chr'].astype(str)
            inst_pc  = inst_df[inst_df['chr'] == str(peak_chr)]
            if len(inst_pc) > 0:
                top_gene = (inst_pc.groupby('gene')['lod'].max().idxmax())
                inst_top = inst_pc[inst_pc['gene'] == top_gene].sort_values('pos_mb')
                fig, ax  = plt.subplots(figsize=(10, 3))
                ax.plot(inst_top['pos_mb'], inst_top['lod'],
                        color='#55AF5B', linewidth=1.2)
                ax.fill_between(inst_top['pos_mb'], 0, inst_top['lod'],
                                alpha=0.15, color='#55AF5B')
                ax.set_xlabel(f'Chr{peak_chr} position (Mb)')
                ax.set_ylabel('Instability LOD')
                ax.set_title(f'S11  Instability  {top_gene}  chr{peak_chr}')
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's11_instability_profile.png', dpi=300)
                plt.close(fig)
                log.info('  s11_instability_profile.png  (gene=%s)', top_gene)
        except Exception as exc:
            log.warning('  s11_instability_profile: %s', exc)


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

    FOUNDER_COLORS_LOCAL = {
        'A/J': '#F4C245', 'B6/J': '#C0BFBF', '129S1/J': '#DA9CC1',
        'NOD/J': '#4F6EAF', 'NZO/J': '#52A5DB', 'CAST/J': '#55AF5B',
        'PWK/J': '#D83026', 'WSB/J': '#683C91',
    }
    fcolors = [FOUNDER_COLORS_LOCAL.get(f, '#888888') for f in founders]

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
                ax.set_title(f'S2 haplotype probs  r={r_corr:.4f}  n={len(py_v)}')
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
                ax.set_title(f'S3 kinship elements  r={r_corr:.4f}  n={len(py_k)}')
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
            ax.set_title(f'S4 peak LOD distribution  n={len(cdf)}')
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
                    ax.set_title(f'S4 peak positions  r={r_corr:.4f}  n={len(merged_pos)}')
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
                fig.suptitle(f'S4 LOD profiles  chr{peak_chr}', fontsize=9)
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
                    ax.set_title(f'S4 allele effects  r={r_corr:.4f}  n={mask.sum()}\n'
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
            ax.set_title(f'S5 permutation max-LOD  chr{peak_chr}')
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
                    ax.set_title('S4 scan throughput: Python vs R')
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
                      elapsed_s: float, peak_heap_mb: float) -> None:
        _scan_stats.append({
            'section':       section,
            'component':     component,
            'n_items':       n_items,
            'elapsed_s':     round(elapsed_s, 3),
            'peak_heap_mb':  round(peak_heap_mb, 1),
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
    if hdf5_cache.exists() and not force:
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
    if kd_npz.exists() and not force:
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

    # ── build dataset covariates ──────────────────────────────────────────────
    covariates_meta = _build_covariates(meta, sample_ids_meta)
    log.info('covariates: shape %s  columns = %s',
             covariates_meta.shape,
             list(pd.get_dummies(meta['dataset'], drop_first=True).columns))

    # ─────────────────────────────────────────────────────────────────────────
    # S1 — cache characterization
    # always characterizes Python cache; compares to R cache_info.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 1
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS1: cache characterization...')
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
                          _s1_elapsed, _s1_peak / 1e6)
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
                          _s2_elapsed, _s2_peak / 1e6)
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
                          _s3_elapsed, _s3_peak / 1e6)
            log.info('  [timing] S3: %.2fs  heap_peak=%.0fMB',
                     _s3_elapsed, _s3_peak / 1e6)

            if _scan_stats:
                pd.DataFrame(_scan_stats).to_csv(
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

                t_chr = time.perf_counter()
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
                _record_stats(f'S4_chr{ch}', f'lmm_scan_chr{ch}',
                              len(loaded_ids), elapsed_chr, 0.0)
                log.info('    chr%s done: %d genes  %d loci  %.1fs  %.1f genes/min',
                         ch, len(loaded_ids), n_loci_ch, elapsed_chr, gpm_chr)

                n_genes_total += len(loaded_ids)
                del counts_ch; gc.collect()

            elapsed_s = time.perf_counter() - t_scan
            if all_lod_rows:
                lod_df = pd.concat(all_lod_rows, ignore_index=True)
                lod_df.to_csv(py_lod_path, index=False)
                genes_per_min = n_genes_total / elapsed_s * 60 if elapsed_s > 0 else 0
                log.info('  scan complete: %d genes  %d rows  %.1fs  %.1f genes/min',
                         n_genes_total, len(lod_df), elapsed_s, genes_per_min)
                _record_stats('S4', 'lmm_scan', n_genes_total, elapsed_s, 0.0)
            else:
                log.warning('  no LOD rows produced — check chr/gene_info alignment')
                lod_df = pd.DataFrame(columns=['gene', 'chr', 'locus', 'pos_mb', 'lod'])

            # save allele effects (peak-locus betas per gene)
            if all_effect_rows:
                eff_df = pd.DataFrame(all_effect_rows)
                eff_df.to_csv(py_dir / 's4_allele_effects.csv', index=False)
                log.info('  s4_allele_effects.csv: %d genes', len(eff_df))

            # peak LOD per gene
            if len(lod_df) > 0:
                peak_py = (lod_df.groupby('gene')['lod'].max()
                           .reset_index().rename(columns={'lod': 'peak_lod_py'}))
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
                        ax.set_title(f'S4 {scan_type} peak LOD  r²={r2:.4f}  n={n_matched}')
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
                top_gene_name = _pk.iloc[_pk['peak_lod_py'].argmax()]['gene']
                top_rows = lod_for_peak[lod_for_peak['gene'] == top_gene_name]
                if len(top_rows) > 0:
                    peak_chr_row = top_rows.loc[top_rows['lod'].idxmax()]
                    peak_chr = str(peak_chr_row['chr'])

    log.info('[downstream sections] peak_chr=%s', peak_chr)

    # ── eGene calling ─────────────────────────────────────────────────────────
    # prefer Python S5 threshold; fall back to R thresholds_chr.csv
    py_thr_path = py_dir / 's5_threshold.csv'
    r_thr_path  = r_data / 'thresholds_chr.csv'
    egenes_df: pd.DataFrame | None = None

    if peak_lod_path.exists():
        peak_py_loaded = pd.read_csv(peak_lod_path)
        thr: float | None = None

        if py_thr_path.exists():
            thr = float(pd.read_csv(py_thr_path)['threshold'].iloc[0])
            log.info('[eGenes] using Python S5 threshold=%.3f on chr%s', thr, peak_chr)
        elif r_thr_path.exists():
            thr_df  = pd.read_csv(r_thr_path)
            thr_map = {str(row['chr']): float(row['threshold'])
                       for _, row in thr_df.iterrows()}
            thr = thr_map.get(peak_chr)
            if thr is not None:
                log.info('[eGenes] using R threshold=%.3f on chr%s', thr, peak_chr)

        if thr is not None:
            egenes_df = peak_py_loaded[
                peak_py_loaded['peak_lod_py'] >= thr].copy()
            egenes_df.to_csv(py_dir / 'egenes.csv', index=False)
            log.info('[eGenes] threshold=%.3f  %d / %d genes called',
                     thr, len(egenes_df), len(peak_py_loaded))
        else:
            log.info('[eGenes] no threshold available — run S5 first')

    # ── eqtl_summary.csv (per-eGene table with peak position + R comparison) ──
    if egenes_df is not None and len(egenes_df) > 0:
        eff_path = py_dir / 's4_allele_effects.csv'
        if eff_path.exists():
            eff_tmp = pd.read_csv(eff_path,
                                  usecols=['gene', 'gene_id', 'chr',
                                           'peak_locus', 'peak_mb', 'peak_lod'])
            eqtl_sum = egenes_df.merge(eff_tmp, on='gene', how='left')
        else:
            eqtl_sum = egenes_df.copy()
        r_egenes_path = r_data / 'egenes.csv'
        if r_egenes_path.exists():
            egenes_r = pd.read_csv(r_egenes_path)
            eqtl_sum['in_r_egenes'] = eqtl_sum['gene'].isin(egenes_r['gene'])
        eqtl_sum.to_csv(py_dir / 'eqtl_summary.csv', index=False)
        log.info('[eqtl_summary] %d eGenes written to eqtl_summary.csv', len(eqtl_sum))

    # ─────────────────────────────────────────────────────────────────────────
    # S5 — permutation threshold (peak chromosome, top cis gene)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 5
    r_perm_lod_path = r_data / f'perm_max_lods_chr{peak_chr}.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS5: permutation threshold (chr%s)...', peak_chr)
            if not peak_lod_path.exists():
                log.info('  %s s4_peak_lod.csv not found — run S4 first', SKIP)
                results['S5_perm_threshold'] = None
            else:
                peak_py_s5    = pd.read_csv(peak_lod_path)
                top_gene_name = peak_py_s5.iloc[
                    peak_py_s5['peak_lod_py'].argmax()]['gene']
                top_gene_id   = gene_name_to_id.get(top_gene_name, top_gene_name)
                log.info('  top gene: %s', top_gene_id)

                h5_path = data_dir / 'counts_lognorm.h5'
                counts_g, loaded_g, sids_g = _load_expression(h5_path, [top_gene_id])
                if counts_g.shape[0] == 0:
                    log.info('  %s expression not found for %s', SKIP, top_gene_id)
                    results['S5_perm_threshold'] = None
                else:
                    geno_ids_s5 = [meta_geno_map.get(s, s) for s in sids_g]
                    y_top  = counts_g[0].astype(float)
                    cov_s5 = _build_covariates(meta, sids_g)

                    scan_top = scan_h2lmm(
                        y_top, cache,
                        covariates=cov_s5,
                        kinship_decomp=kinship_decomp,
                        sample_ids=geno_ids_s5,
                        use_rint=True,
                        phenotype_id=top_gene_name,
                        scan_window='cis', gene_chr=peak_chr,
                    )
                    peak_locus = grab_locus_from_scan(scan_top)
                    log.info('  peak locus: %s; running 100 permutations...', peak_locus)

                    kd_pc    = {peak_chr: kinship_decomp[peak_chr]}
                    null_out = generate_null_outcomes(
                        scan_top, n_perms=100, seed=1)
                    perm_out = run_threshold_scans(
                        null_out, cache, peak_chr, kd_pc, scan_top, n_perms=100)
                    py_max_lods  = perm_out['max_lods']
                    py_threshold = get_gev_thresholds(py_max_lods, percentile=0.95)

                    pd.DataFrame({'max_lod': py_max_lods}).to_csv(
                        py_dir / 's5_perm_max_lods.csv', index=False)
                    pd.DataFrame({'chr': [peak_chr], 'threshold': [py_threshold]}).to_csv(
                        py_dir / 's5_threshold.csv', index=False)
                    log.info('  py GEV threshold = %.3f', py_threshold)
                    results['S5_perm_threshold'] = True   # ran successfully

                    # optional: compare to R
                    if r_perm_lod_path.exists():
                        r_max_lods  = pd.read_csv(r_perm_lod_path)['max_lod'].values
                        r_threshold = get_gev_thresholds(r_max_lods, percentile=0.95)
                        delta = abs(py_threshold - r_threshold)
                        log.info('  vs R: R GEV=%.3f  delta=%.3f', r_threshold, delta)
                        _check('GEV threshold delta', delta, 0.5, '<')
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S7 — MI scan (significant genes on peak_chr)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 7
    r_mi_path = r_data / 'mi_lod_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS7: MI scan...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S7_mi_scan'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                mi_rows = []
                for _, erow in egenes_df.iterrows():
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    counts_g, loaded_g, sids_g = _load_expression(h5_path, [gene_id])
                    if counts_g.shape[0] == 0:
                        continue
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = counts_g[0].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    log.info('  MI scan: %s', gene_id)
                    mi_scan = scan_h2lmm_mi(
                        y_g, cache,
                        covariates=cov_g,
                        kinship_decomp=kinship_decomp,
                        sample_ids=geno_ids_g,
                        n_imp=11, seed=1,
                        phenotype_id=gene_name,
                        scan_window='cis', gene_chr=peak_chr,
                    )
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
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S8 — SNP scan (gated: allele_db required)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 8
    r_snp_path = r_data / 'snp_lod_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
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
                kd_pc   = {peak_chr: kinship_decomp[peak_chr]}
                snp_rows = []
                for _, erow in egenes_df.iterrows():
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    counts_g, loaded_g, sids_g = _load_expression(h5_path, [gene_id])
                    if counts_g.shape[0] == 0:
                        continue
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = counts_g[0].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    log.info('  SNP scan: %s', gene_id)
                    try:
                        snp_scan = imputed_snp_scan_from_db(
                            y_g, cache,
                            alleles_h5=str(ALLELE_H5),
                            chrom=peak_chr,
                            covariates=cov_g,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_g,
                        )
                        snp_rows.append(_scan_to_df(snp_scan, gene_name))
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
                        log.info('  %s snp_lod_genome.csv not found', SKIP)
                        results['S8_snp_lod_r'] = None
                else:
                    results['S8_snp_lod_r'] = None
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S9 — mediation scan
    # always runs on eGenes; compares to R mediation_genome.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 9
    r_med_path = r_data / 'mediation_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS9: mediation scan...')
            if egenes_df is None or len(egenes_df) < 2:
                log.info('  %s need >=2 eGenes — run S4+S5 first', SKIP)
                results['S9_mediation'] = None
            else:
                h5_path  = data_dir / 'counts_lognorm.h5'
                med_rows = []
                for _, erow in egenes_df.iterrows():
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    counts_g, _, sids_g = _load_expression(h5_path, [gene_id])
                    if counts_g.shape[0] == 0:
                        continue
                    others   = egenes_df[egenes_df['gene'] != gene_name]
                    if len(others) == 0:
                        continue
                    med_name = others.iloc[0]['gene']
                    med_id   = gene_name_to_id.get(med_name, med_name)
                    counts_m, _, _ = _load_expression(h5_path, [med_id])
                    if counts_m.shape[0] == 0:
                        continue
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = counts_g[0].astype(float)
                    y_m   = counts_m[0].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    log.info('  mediation: %s ~ %s', gene_id, med_id)
                    try:
                        med_scan = mediation_scan_qr(
                            y_g, y_m, cache,
                            covariates=cov_g,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_g,
                            scan_window='cis', gene_chr=peak_chr,
                        )
                        df_med = pd.DataFrame({
                            'gene':         gene_name,
                            'mediator':     med_name,
                            'chr':          med_scan.chromosomes,
                            'locus':        med_scan.locus_ids,
                            'pos_mb':       med_scan.positions,
                            'lod_base':     med_scan.lod_base,
                            'lod_mediated': med_scan.lod_mediated,
                        })
                        med_rows.append(df_med)
                    except Exception as exc:
                        log.warning('  mediation failed for %s: %s', gene_id, exc)
                if med_rows:
                    med_df = pd.concat(med_rows, ignore_index=True)
                    med_df.to_csv(py_dir / 's9_mediation.csv', index=False)
                    log.info('  mediation complete: %d genes', len(med_rows))
                    results['S9_mediation'] = True
                    if r_med_path.exists():
                        r_med      = pd.read_csv(r_med_path)
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
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S10 — bootstrap CI
    # always runs on eGenes; compares to R ci_summary_genome.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 10
    r_ci_path = r_data / 'ci_summary_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS10: bootstrap CI...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S10_ci'] = None
            else:
                h5_path = data_dir / 'counts_lognorm.h5'
                ci_rows = []
                for _, erow in egenes_df.iterrows():
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    counts_g, _, sids_g = _load_expression(h5_path, [gene_id])
                    if counts_g.shape[0] == 0:
                        continue
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = counts_g[0].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    log.info('  bootstrap CI: %s', gene_id)
                    try:
                        kd_pc    = {peak_chr: kinship_decomp[peak_chr]}
                        scan_g   = scan_h2lmm(
                            y_g, cache,
                            covariates=cov_g,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_g,
                            use_rint=True,
                            phenotype_id=gene_name,
                            scan_window='cis', gene_chr=peak_chr,
                        )
                        peak_loc = grab_locus_from_scan(scan_g)
                        boot_out = generate_null_outcomes(
                            scan_g, n_perms=200, seed=1,
                            model_type='alt', peak_locus=peak_loc)
                        ci_res  = run_positional_scans(
                            boot_out, cache, peak_chr, kd_pc, scan_g)
                        peak_mb = float(scan_g.positions[scan_g.lod.argmax()])
                        ci_mbs  = ci_res.get('peak_mbs', np.array([peak_mb]))
                        ci_rows.append({
                            'gene':    gene_name,
                            'chr':     peak_chr,
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
                            overlap = (
                                (merged_ci['ci_lo_py'] <= merged_ci['ci_hi_r']) &
                                (merged_ci['ci_hi_py'] >= merged_ci['ci_lo_r'])
                            ).mean()
                            merged_ci.to_csv(cmp_dir / 's10_ci_cmp.csv', index=False)
                            log.info('  vs R: CI overlap=%.2f  n=%d',
                                     overlap, len(merged_ci))
                else:
                    results['S10_ci'] = None
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S11 — instability scan
    # always runs on eGenes; compares to R instability_genome.csv if present
    # ─────────────────────────────────────────────────────────────────────────
    sec = 11
    r_inst_path = r_data / 'instability_genome.csv'
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS11: instability scan...')
            if egenes_df is None or len(egenes_df) == 0:
                log.info('  %s no eGenes — run S4+S5 first', SKIP)
                results['S11_instability'] = None
            else:
                h5_path   = data_dir / 'counts_lognorm.h5'
                inst_rows = []
                for _, erow in egenes_df.iterrows():
                    gene_name = erow['gene']
                    gene_id   = gene_name_to_id.get(gene_name, gene_name)
                    counts_g, _, sids_g = _load_expression(h5_path, [gene_id])
                    if counts_g.shape[0] == 0:
                        continue
                    geno_ids_g = [meta_geno_map.get(s, s) for s in sids_g]
                    y_g   = counts_g[0].astype(float)
                    cov_g = _build_covariates(meta, sids_g)
                    log.info('  instability: %s', gene_id)
                    try:
                        inst = instability_lm_scan(
                            y_g, cache,
                            covariates=cov_g,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_g,
                            n_samples=100, seed=1,
                            scan_window='cis', gene_chr=peak_chr,
                        )
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
                            r_i, _ = sp_stats.pearsonr(merged_inst['py'],
                                                        merged_inst['r'])
                            merged_inst.to_csv(cmp_dir / 's11_instability_cmp.csv',
                                               index=False)
                            _check('instability r', abs(r_i), 0.90)
                            log.info('  vs R: r=%.4f  n=%d', r_i, len(merged_inst))
                else:
                    results['S11_instability'] = None
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
        pd.DataFrame(_scan_stats).to_csv(py_dir / 'scan_stats.csv', index=False)

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

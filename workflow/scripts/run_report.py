"""Run-level summary stats + scan visualisations — called by Nextflow REPORT process.

Usage: python run_report.py <peak_lod.csv> <eqtl_summary.csv> <pval_perm.json>
           <perm_max_lods.csv> <gene_info.csv> <lod_genome.csv>
           <kinship_loco.h5> <genomecache.h5>
           <celltype> <scan_tag> <output_dir> [gtf_path|-] [tissue]

Per-chromosome perm files {tag}_perm_max_lods_chr*.csv must be staged alongside
<perm_max_lods.csv> in the same directory (Nextflow handles this automatically
when they appear as a path input in the calling process).

Writes:
  {tag}_eqtl_stats.csv                     — per-chr + genome-wide GEV params
  {tag}_genome_summary{_share}.png          — Manhattan + eGene count per chr
  {tag}_genome_diagnostic{_share}.png       — GEV thresholds + scan stats panel
  {tag}_loci_per_chr{_share}.png            — loci count per chromosome
  {tag}_lod_panel_grid{_share}.png          — top-gene LOD profile per chr (grid)
  {tag}_perm_distribution{_share}.png       — genome-wide perm max-LOD distribution
  {tag}_perm_panel_grid{_share}.png         — per-chr perm distributions (grid)
  {tag}_egene_lod_dist{_share}.png          — peak-LOD distribution by eGene status
  {tag}_cis_distance_panel{_share}.png      — full/zoom scatter + KDE (if tss_mb present)
  {tag}_cis_histogram{_share}.png           — upstream/downstream bar chart (if eGenes)
  {tag}_effects_grid{_share}.png            — founder effect bars per chr (if beta_* in summary)
  {tag}_kinship_heatmap{_share}.png         — LOCO kinship heatmap (first chr)
  {tag}_kinship_dist{_share}.png            — diagonal vs off-diagonal kinship KDE
  {tag}_lod_profile_chr{c}{_share}.png      — per-chr standalone LOD profile (top gene)
  {tag}_perm_dist_chr{c}{_share}.png        — per-chr perm null distribution
  {tag}_effects_chr{c}{_share}.png          — per-chr founder effects (if beta_* in summary)
  run_metadata.json
"""
import gc
import re
import sys, logging, datetime, warnings
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)

# eGenes drawn as fine-mapping locus panels per chromosome, ranked by peak LOD
FINEMAP_PANELS_PER_CHR = 3

# matplotlib reports font fallbacks (findfont) and per-glyph misses as one
# message per label per figure; they bury the run log without changing output.
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
warnings.filterwarnings('ignore', message=r'Glyph \d+ .* missing from font')
warnings.filterwarnings('ignore', message=r'.*findfont.*')


def _savefig(fig: 'plt.Figure', stem: str, out: Path) -> None:
    p = out / f'{stem}.png'
    fig.savefig(p, bbox_inches='tight', dpi=300, transparent=True)
    fig.savefig(p.with_name(f'{stem}_share.png'), bbox_inches='tight', dpi=300, facecolor='white')
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    log.info('wrote %s (+ _share)', p.name)


def _gev_row(chrom: str, max_lods: np.ndarray) -> dict:
    from mosaiqtl.io.results_io import build_gev_row
    return build_gev_row(chrom, max_lods)


def _load_lod_subset(
    path: Path, want_ids: set, chunksize: int = 500_000,
) -> 'tuple[pd.DataFrame, dict[str, int]]':
    """Stream the genome LOD table once, keeping only the rows of *want_ids*.

    Returns the retained rows and the number of distinct loci per chromosome
    (bare chromosome strings). Memory is bounded by one chunk plus the retained
    rows, independent of the number of genes or loci scanned.
    """
    from mosaiqtl.utils import strip_chr_prefix
    cols = list(pd.read_csv(path, nrows=0).columns)
    locus_col = 'locus' if 'locus' in cols else 'pos_mb'
    loci_by_chr: dict[str, set] = {}
    kept = []
    for chunk in pd.read_csv(
        path,
        dtype={'gene_id': str, 'chr': str, 'pos_mb': 'float32', 'lod': 'float32'},
        chunksize=chunksize,
    ):
        chunk['chr'] = strip_chr_prefix(chunk['chr'])
        for c, loci in chunk.groupby('chr')[locus_col]:
            loci_by_chr.setdefault(c, set()).update(loci.unique())
        sel = chunk[chunk['gene_id'].isin(want_ids)]
        if not sel.empty:
            kept.append(sel)
        del chunk, sel
    lod_df = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=cols)
    return lod_df, {c: len(s) for c, s in loci_by_chr.items()}


def _finemap_gene_ids(path: Path, chunksize: int = 1_000_000) -> set:
    """Distinct gene_ids in the fine-mapping table, read as a stream of chunks."""
    ids: set = set()
    for chunk in pd.read_csv(path, usecols=['gene_id'], dtype={'gene_id': str},
                             chunksize=chunksize):
        ids.update(chunk['gene_id'].unique())
    return ids


def _load_finemap_subset(
    path: Path, want_ids: set, chunksize: int = 1_000_000,
) -> pd.DataFrame:
    """Stream the fine-mapping table once, keeping only the rows of *want_ids*.

    Memory is bounded by one chunk plus the retained rows, independent of how
    many eGenes were fine-mapped.
    """
    from mosaiqtl.utils import strip_chr_prefix
    kept = []
    for chunk in pd.read_csv(path, dtype={'gene_id': str, 'chr': str}, chunksize=chunksize):
        sel = chunk[chunk['gene_id'].isin(want_ids)]
        if not sel.empty:
            kept.append(sel)
        del chunk, sel
    if not kept:
        return pd.DataFrame(columns=list(pd.read_csv(path, nrows=0).columns))
    fm_df = pd.concat(kept, ignore_index=True)
    fm_df['chr'] = strip_chr_prefix(fm_df['chr'])
    return fm_df


def main() -> None:
    peak_lod_csv   = Path(sys.argv[1])
    egene_csv      = Path(sys.argv[2])
    pval_perm_json = Path(sys.argv[3])
    perm_csv       = Path(sys.argv[4])
    gene_info_path = Path(sys.argv[5])
    lod_genome_csv = Path(sys.argv[6])
    kinship_path   = Path(sys.argv[7])
    cache_path     = Path(sys.argv[8])
    celltype       = sys.argv[9]
    stag           = sys.argv[10]
    out            = Path(sys.argv[11])
    gtf_arg        = sys.argv[12] if len(sys.argv) > 12 else '-'
    tissue         = sys.argv[13] if len(sys.argv) > 13 else ''
    finemap_arg    = sys.argv[14] if len(sys.argv) > 14 else '-'

    tag = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from mosaiqtl import __version__
    from mosaiqtl.genotype.cc import FOUNDER_ORDER, display_founders as _display_founders
    from mosaiqtl.io.atomic import atomic_write_json
    from mosaiqtl.io.results_io import write_eqtl_stats
    from mosaiqtl.utils import chr_sort_key, git_hash, strip_chr_prefix
    from mosaiqtl.viz.effects import effects_per_chr, genome_effects_grid
    from mosaiqtl.viz.genome import (
        cis_distance_histogram, cis_distance_panel, egene_lod_distribution,
        genome_diagnostic, genome_summary, loci_per_chr, lod_panel_grid,
        lod_profiles_per_chr, manhattan, perm_distribution,
        perm_distributions_per_chr, perm_panel_grid,
    )
    from mosaiqtl.viz.locus import locus_panel_grid, locus_panels_per_chr
    from mosaiqtl.viz.haplotypes import kinship_distribution, kinship_heatmap, prob_heatmap

    # ── load dataframes ────────────────────────────────────────────────────────
    peak_df   = pd.read_csv(peak_lod_csv,   dtype={'gene_id': str, 'chr': str})
    egenes_df = pd.read_csv(egene_csv,      dtype={'gene_id': str, 'chr': str})
    for _df in (peak_df, egenes_df):
        _df['chr'] = strip_chr_prefix(_df['chr'])

    if 'significance' in egenes_df.columns and egenes_df['significance'].notna().any():
        egenes_df = egenes_df[egenes_df['significance'] == 'T']

    # founder betas live in peak_lod.csv; the eGene summary schema carries none
    _beta_extra = [c for c in peak_df.columns
                   if c.startswith('beta_') and c not in egenes_df.columns]
    if _beta_extra:
        egenes_df = egenes_df.merge(peak_df[['gene_id'] + _beta_extra],
                                    on='gene_id', how='left')

    # Plots draw the top gene per chromosome (by peak LOD, over all genes and over
    # eGenes) and the fine-mapped eGenes; only those genes' LOD rows are loaded.
    want_ids = set()
    for _df in (peak_df, egenes_df):
        _ok = _df.dropna(subset=['peak_lod'])
        if not _ok.empty:
            want_ids.update(_ok.loc[_ok.groupby('chr')['peak_lod'].idxmax(), 'gene_id'])
    fm_panel_df = egenes_df.iloc[0:0]
    if finemap_arg != '-' and Path(finemap_arg).exists():
        _fm_ids = _finemap_gene_ids(Path(finemap_arg))
        _fm_eg = egenes_df[egenes_df['gene_id'].isin(_fm_ids)].dropna(subset=['peak_lod'])
        fm_panel_df = (_fm_eg.sort_values('peak_lod', ascending=False)
                       .groupby('chr', sort=False).head(FINEMAP_PANELS_PER_CHR))
        want_ids.update(fm_panel_df['gene_id'])
    lod_df, n_loci_by_chr = _load_lod_subset(lod_genome_csv, want_ids)
    # gene_symbol is a per-gene attribute; take it from peak_df so the LOD table
    # need not carry it
    lod_df['gene_symbol'] = lod_df['gene_id'].astype(str).map(
        dict(zip(peak_df['gene_id'], peak_df['gene_symbol'])))
    log.info('LOD rows kept: %d for %d genes', len(lod_df), len(want_ids))

    egene_ids = set(egenes_df['gene_id'])
    is_egene  = peak_df['gene_id'].isin(egene_ids).to_numpy()

    # ── permutation files ──────────────────────────────────────────────────────
    prefix    = f'{tag}_perm_max_lods_chr'
    chr_files = sorted(
        perm_csv.parent.glob(f'{prefix}*.csv'),
        key=lambda p: chr_sort_key(p.stem[len(prefix):]),
    )
    perm_by_chr:  dict[str, np.ndarray] = {}
    thr_by_chr:   dict[str, 'float | None'] = {}
    chrom_stats:  list[dict] = []

    for pf in chr_files:
        chrom = pf.stem[len(prefix):]
        ml    = pd.read_csv(pf)['max_lod'].to_numpy()
        perm_by_chr[chrom] = ml
        row = _gev_row(chrom, ml)
        row.update({
            'tissue': tissue, 'celltype': celltype,
            'n_genes_scanned': int((peak_df['chr'] == chrom).sum()),
            'n_egenes':        int((egenes_df['chr'] == chrom).sum()),
        })
        chrom_stats.append(row)
        thr_by_chr[chrom] = row['lod_threshold']

    genome_max_lods = pd.read_csv(perm_csv)['max_lod'].to_numpy()
    genome_row      = _gev_row('genome', genome_max_lods)
    genome_row.update({
        'tissue': tissue, 'celltype': celltype,
        'n_genes_scanned': int(len(peak_df)),
        'n_egenes':        int(len(egenes_df)),
        'git_hash':        git_hash(),
    })
    genome_thr: 'float | None' = genome_row['lod_threshold']

    # ── eqtl_stats.csv ────────────────────────────────────────────────────────
    stats_path = out / f'{tag}_eqtl_stats.csv'
    write_eqtl_stats(chrom_stats, genome_row, stats_path)
    log.info('wrote %s', stats_path.name)

    # ── kinship matrix (first chr available) ──────────────────────────────────
    K_chr: 'np.ndarray | None' = None
    K_chrom_label = ''
    try:
        with h5py.File(kinship_path, 'r') as f:
            chr_keys = sorted((k for k in f.keys() if k.startswith('chr')),
                               key=lambda k: chr_sort_key(k[3:]))
            if chr_keys:
                k0 = chr_keys[0]
                K_chrom_label = k0[3:]
                vecs = f[k0]['vectors'][:]
                vals = f[k0]['values'][:]
                K_chr = (vecs * vals) @ vecs.T
    except Exception as exc:
        log.warning('could not load kinship: %s', exc)

    # ── allele effects (optional — present when eqtl_summary carries beta_ cols) ──
    beta_cols   = sorted(c for c in egenes_df.columns if c.startswith('beta_'))
    has_effects = bool(beta_cols) and len(egenes_df) > 0

    # ── 1. genome_summary ─────────────────────────────────────────────────────
    fig = genome_summary(peak_df, is_egene, thr_by_chr=thr_by_chr or None)
    _savefig(fig, f'{tag}_genome_summary', out)

    # ── 2. genome_diagnostic ──────────────────────────────────────────────────
    thr_diag = dict(thr_by_chr)
    if genome_thr is not None:
        thr_diag['genome_wide'] = genome_thr
    cis_dist_mb: 'np.ndarray | None' = None
    if 'cis_dist_mb' in egenes_df.columns and len(egenes_df) > 0:
        cis_dist_mb = egenes_df['cis_dist_mb'].to_numpy(dtype=float)
    top_gene, peak_chr = '', ''
    if len(peak_df) > 0 and 'peak_lod' in peak_df.columns:
        top_row  = peak_df.loc[peak_df['peak_lod'].idxmax()]
        top_gene = str(top_row.get('gene_symbol', ''))
        peak_chr = str(top_row['chr'])
    diag_stats = {
        'n_genes': int(len(peak_df)), 'n_egenes': int(len(egenes_df)),
        'n_perms': int(len(genome_max_lods)),
        'top_gene': top_gene, 'peak_chr': peak_chr,
    }
    fig = genome_diagnostic(peak_df, thr_diag, cis_dist_mb=cis_dist_mb, scan_stats=diag_stats)
    _savefig(fig, f'{tag}_genome_diagnostic', out)

    # ── 3. loci_per_chr ───────────────────────────────────────────────────────
    chrs_ord = sorted(n_loci_by_chr, key=chr_sort_key)
    n_loci   = np.array([n_loci_by_chr[c] for c in chrs_ord], dtype=float)
    if n_loci.sum() > 0:
        ax  = loci_per_chr(chrs_ord, n_loci)
        fig = ax.figure
        _savefig(fig, f'{tag}_loci_per_chr', out)

    # ── 4. lod_panel_grid ─────────────────────────────────────────────────────
    if not lod_df.empty and 'gene_symbol' in lod_df.columns:
        fig = lod_panel_grid(lod_df, peak_df, thr_by_chr)
        _savefig(fig, f'{tag}_lod_panel_grid', out)

    # ── 5. perm_distribution (genome-wide) ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    perm_distribution(genome_max_lods, ax=ax, threshold=genome_thr)
    fig.tight_layout()
    _savefig(fig, f'{tag}_perm_distribution', out)

    # ── 6. perm_panel_grid ────────────────────────────────────────────────────
    if perm_by_chr:
        fig = perm_panel_grid(perm_by_chr, thr_by_chr)
        _savefig(fig, f'{tag}_perm_panel_grid', out)

    # ── 7. egene_lod_distribution ─────────────────────────────────────────────
    peak_lods = peak_df['peak_lod'].to_numpy(dtype=float) if 'peak_lod' in peak_df.columns else np.array([])
    if len(peak_lods) > 0:
        fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
        egene_lod_distribution(peak_lods, is_egene, ax=ax, threshold=genome_thr)
        fig.tight_layout()
        _savefig(fig, f'{tag}_egene_lod_dist', out)

    # ── 8. cis_distance_panel ─────────────────────────────────────────────────
    tss_col = next((c for c in ('tss_mb', 'start_mb') if c in peak_df.columns), None)
    if tss_col and 'peak_mb' in peak_df.columns and len(peak_df) > 0:
        tss_arr  = peak_df[tss_col].to_numpy(dtype=float)
        peak_arr = peak_df['peak_mb'].to_numpy(dtype=float)
        fig = cis_distance_panel(peak_lods, tss_arr, peak_arr, is_egene=is_egene, threshold=genome_thr)
        _savefig(fig, f'{tag}_cis_distance_panel', out)

        # ── 9. cis_histogram ──────────────────────────────────────────────────
        if len(egenes_df) > 0 and tss_col in egenes_df.columns and 'peak_mb' in egenes_df.columns:
            tss_eg  = egenes_df[tss_col].to_numpy(dtype=float)
            peak_eg = egenes_df['peak_mb'].to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(6, 3.8), dpi=300)
            cis_distance_histogram(tss_eg, peak_eg, ax=ax)
            fig.tight_layout()
            _savefig(fig, f'{tag}_cis_histogram', out)

    # ── 10. genome_effects_grid (if beta_ cols available in summary) ──────────
    if has_effects:
        fig = genome_effects_grid(egenes_df, FOUNDER_ORDER)
        _savefig(fig, f'{tag}_effects_grid', out)

    # ── 11 & 12. kinship_heatmap + kinship_distribution ──────────────────────
    if K_chr is not None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
        kinship_heatmap(K_chr, ax=ax, chrom=K_chrom_label)
        fig.tight_layout()
        _savefig(fig, f'{tag}_kinship_heatmap', out)

        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        kinship_distribution(K_chr, ax=ax, chrom=K_chrom_label)
        fig.tight_layout()
        _savefig(fig, f'{tag}_kinship_dist', out)

    del K_chr
    gc.collect()

    # ── 13. per-chr LOD profiles ──────────────────────────────────────────────
    if not lod_df.empty and 'gene_symbol' in lod_df.columns:
        per_chr_lods = lod_profiles_per_chr(lod_df, peak_df, thr_by_chr)
        for chrom, fig in per_chr_lods.items():
            _savefig(fig, f'{tag}_lod_profile_chr{chrom}', out)

    # ── 14. per-chr perm distributions ───────────────────────────────────────
    if perm_by_chr:
        per_chr_perms = perm_distributions_per_chr(perm_by_chr, thr_by_chr)
        for chrom, fig in per_chr_perms.items():
            _savefig(fig, f'{tag}_perm_dist_chr{chrom}', out)

    # ── 15. per-chr allele effects (if beta_ cols available) ─────────────────
    if has_effects:
        per_chr_effs = effects_per_chr(egenes_df, FOUNDER_ORDER)
        for chrom, fig in per_chr_effs.items():
            _savefig(fig, f'{tag}_effects_chr{chrom}', out)

    # ── 16. haplotype probability heatmap at peak locus ───────────────────────
    if peak_chr:
        try:
            peak_mb_val = float(top_row.get('peak_mb', 0)) if len(peak_df) > 0 else 0.0
            with h5py.File(cache_path, 'r') as _hf:
                _founders_raw = [s.decode() if isinstance(s, bytes) else s
                                 for s in _hf['founders'][:]]
                _grp = _hf[f'chr{peak_chr}']
                _pos = _grp['positions'][:]
                _idx = int(np.argmin(np.abs(_pos - peak_mb_val)))
                _probs = _grp['probs'][_idx]   # (n_subjects, n_founders)
            _n_s = _probs.shape[0]
            fig_h = max(3.5, _n_s * 0.18)
            fig, ax = plt.subplots(figsize=(6, fig_h), dpi=300)
            prob_heatmap(_probs, _display_founders(_founders_raw), ax=ax,
                         chrom=peak_chr, position_mb=peak_mb_val)
            fig.tight_layout()
            _savefig(fig, f'{tag}_haplotype_heatmap', out)
        except Exception as exc:
            log.warning('haplotype_heatmap: %s', exc)

    # ── 17. Manhattan (peak LOD per gene across all scanned chromosomes) ──────
    if len(peak_df) > 0 and 'peak_mb' in peak_df.columns and 'peak_lod' in peak_df.columns:
        try:
            _pk_m = peak_df.copy()
            _pk_m['chr'] = _pk_m['chr'].astype(str)
            _chrs_m = sorted(_pk_m['chr'].unique(), key=chr_sort_key)
            _scans_m = {
                ch: (_pk_m.loc[_pk_m['chr'] == ch, 'peak_mb'].values,
                     _pk_m.loc[_pk_m['chr'] == ch, 'peak_lod'].values)
                for ch in _chrs_m
            }
            fig = manhattan(_scans_m, threshold=genome_thr,
                            highlight_chrom=peak_chr if peak_chr else None)
            fig.tight_layout()
            _savefig(fig, f'{tag}_manhattan', out)
        except Exception as exc:
            log.warning('manhattan: %s', exc)

    # ── 18. locus_panel_grid (multi-gene LOD + effects grid) ──────────────────
    if not lod_df.empty and 'gene_symbol' in lod_df.columns:
        try:
            fig = locus_panel_grid(
                lod_df,
                egenes_df if has_effects else None,
                list(FOUNDER_ORDER) if has_effects else None,
                thr_by_chr,
            )
            _savefig(fig, f'{tag}_locus_panel_grid', out)
        except Exception as exc:
            log.warning('locus_panel_grid: %s', exc)

    # shared state for sections 19 and 20
    _lw_m    = re.match(r'^Lw([0-9.]+)$', stag)
    _zoom_mb = float(_lw_m.group(1)) if _lw_m else 5.0
    _gtf_path = gtf_arg if gtf_arg != '-' else None
    gi_full   = pd.read_csv(gene_info_path, dtype={'gene_id': str, 'chr': str})
    _genes_df = (gi_full if {'chr', 'start', 'gene_symbol'}.issubset(gi_full.columns)
                 else None)

    # ── 19. per-chr locus panels with optional gene track (GTF) ───────────────
    if not lod_df.empty and 'gene_symbol' in lod_df.columns:
        try:
            _tss_col = next((c for c in ('tss_mb', 'start_mb') if c in peak_df.columns), None)
            _tss_map = (dict(zip(peak_df['gene_symbol'], peak_df[_tss_col]))
                        if _tss_col and 'gene_symbol' in peak_df.columns else None)
            _near_tss_col = 'near_tss_mb'
            _near_tss_map = (
                {g: t for g, t in zip(egenes_df['gene_symbol'], egenes_df[_near_tss_col])
                 if pd.notna(t)}
                if has_effects and _near_tss_col in egenes_df.columns else None
            )
            lp_figs = locus_panels_per_chr(
                lod_df,
                egenes_df if has_effects else None,
                list(FOUNDER_ORDER) if has_effects else None,
                thr_by_chr,
                tss_mb_map=_tss_map,
                near_tss_mb_map=_near_tss_map,
                genes_df=_genes_df,
                zoom_window_mb=_zoom_mb,
                gtf_path=_gtf_path,
            )
            for chrom, fig in lp_figs.items():
                _savefig(fig, f'{tag}_locus_panel_chr{chrom}', out)
        except Exception as exc:
            log.warning('locus_panels_per_chr: %s', exc)

    # ── 20. finemap locus panels per eGene (requires merged eqtl_finemapping.csv) ──
    if finemap_arg != '-' and Path(finemap_arg).exists() and len(fm_panel_df) > 0:
        try:
            from mosaiqtl.viz.finemap import locus_finemap_panel
            fm_df = _load_finemap_subset(
                Path(finemap_arg), set(fm_panel_df['gene_id'].astype(str)))
            log.info('fine-mapping rows kept: %d for %d genes',
                     len(fm_df), len(fm_panel_df))
            lod_df_chr = lod_df
            for _, er in fm_panel_df.iterrows():
                gid   = str(er['gene_id'])
                chrom = str(er['chr'])
                gene_fm = fm_df[fm_df['gene_id'] == gid]
                if gene_fm.empty:
                    continue
                chr_lod = lod_df_chr[lod_df_chr['chr'] == chrom]
                gene_lod = chr_lod[chr_lod['gene_id'].astype(str) == gid]
                if gene_lod.empty:
                    gene_lod = chr_lod.drop_duplicates(subset=['pos_mb'])
                lod_pos = gene_lod['pos_mb'].to_numpy(dtype=float)
                lod_sc  = gene_lod['lod'].to_numpy(dtype=float)
                if len(lod_pos) == 0:
                    continue
                peak_mb = float(er['peak_mb']) if pd.notna(er.get('peak_mb')) else None
                sym     = str(er.get('gene_symbol', gid))
                thr     = thr_by_chr.get(chrom)
                fig = locus_finemap_panel(
                    gene_fm, gid, lod_pos, lod_sc,
                    chrom=chrom, peak_pos_mb=peak_mb,
                    genes_df=_genes_df, highlight_gene=sym,
                    window_mb=_zoom_mb, threshold=thr,
                )
                safe_sym = sym.replace('/', '_').replace(' ', '_')
                _savefig(fig, f'{tag}_finemap_panel_{safe_sym}_chr{chrom}', out)
        except Exception as exc:
            log.warning('finemap_panel: %s', exc)

    # ── run_metadata.json ─────────────────────────────────────────────────────
    metadata = {
        'tag':               tag,
        'celltype':          celltype,
        'tissue':            tissue,
        'scan_tag':          stag,
        'n_genes':           int(len(peak_df)),
        'n_egenes':          int(len(egenes_df)),
        'lod_threshold':     genome_thr,
        'mosaiqtl_version':  __version__,
        'git_hash':          genome_row['git_hash'],
        'generated_at':      datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_write_json(metadata, out / 'run_metadata.json')
    log.info('wrote run_metadata.json')


if __name__ == '__main__':
    main()

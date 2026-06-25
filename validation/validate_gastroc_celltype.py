"""validate_gastroc_celltype.py — cell-type-resolved eQTL scan on gastroc pseudobulk.

Parallel to validate_gastroc.py but runs on a single cell type extracted from
gastroc_all_pb.h5ad via convert_gastroc_h5ad.py --cell_type.

Shared inputs (from gastroc_data/ — same for all cell types):
    py_data/cache.h5          — GenomeCache HDF5 (built once by validate_gastroc.py --build-cache)
    py_data/kinship_decomp.npz — LOCO kinship eigendecomp (computed once)
    metadata.csv              — sample_id, genotype, sex, dataset
    gene_info.csv             — gene_id, chr, start, end

Cell-type-specific input:
    celltypes/{cell_type}/counts_lognorm.h5  — log2(CPM+1) for this cell type only

Sections:
    S4       — LMM cis scan (same machinery as validate_gastroc.py S4)
    S5       — permutation threshold (genome-wide GEV)
    S6       — eGene calling vs genome-wide GEV threshold
    Compare  — cell-type peak LOD vs whole-tissue peak LOD (new — the key scientific comparison)

Outputs (validation/gastroc_data/celltypes/{cell_type}/):
    py_data/   — s4_lod.csv, s4_peak_lod.csv, s4_allele_effects.csv,
                 s5_perm_max_lods.csv, eqtl_summary.csv
    plots/     — s4_manhattan.png, s4_lod_profile.png, s4_founder_effects.png,
                 s4_lod_histogram.png, s4_cis_distance.png,
                 s4_lod_panel_grid.png, s4_locus_panel_grid.png,
                 s4_locus_panel/{per-chr}.png,
                 s5_perm_distribution.png,
                 s6_egene_lod_distribution.png,
                 compare_lod_scatter.png

Usage:
    conda run -n bioinfo python validation/validate_gastroc_celltype.py
    conda run -n bioinfo python validation/validate_gastroc_celltype.py --cell_type myonuclei
    conda run -n bioinfo python validation/validate_gastroc_celltype.py --sections 4
    conda run -n bioinfo python validation/validate_gastroc_celltype.py --force
    conda run -n bioinfo python validation/validate_gastroc_celltype.py --n_genes 50
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

from mosaiqtl.genotype.founders import FOUNDER_CODE, FOUNDER_COLORS, FOUNDER_ORDER
from mosaiqtl.utils import chr_sort_key as _chr_sort_key
from mosaiqtl.viz.effects import effect_bars as _viz_effect_bars
from mosaiqtl.viz.genome import (
    cis_distance_histogram as _viz_cis_distance,
    egene_lod_distribution as _viz_egene_lod,
    lod_histogram as _viz_lod_histogram,
    lod_panel_grid as _viz_lod_panel_grid,
    locus_panel_grid as _viz_locus_panel_grid,
    locus_panels_per_chr as _viz_locus_panels,
    perm_distribution as _viz_perm_distribution,
)

_FOUNDER_CODE_INV: dict[str, str] = {v: k for k, v in FOUNDER_CODE.items()}


def _display_founders(cache_founders: list[str]) -> list[str]:
    """Map HAPPY cache codes (A–H) to FOUNDER_ORDER display names."""
    return [_FOUNDER_CODE_INV.get(f, _FOUNDER_CODE_INV.get(f.upper(), f))
            for f in cache_founders]


def _get_rss_mb() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return usage.ru_maxrss / 1024**2
    return usage.ru_maxrss / 1024


from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.genotype.founders import F1_COLORS, STRAIN_ORDER
from mosaiqtl.qtl.lmm import estimate_h2_null, precompute_chromosome, scan_chromosome
from mosaiqtl.qtl.scan import ScanResult, scan_h2lmm
from mosaiqtl.qtl.permutation import (
    generate_null_outcomes,
    get_gev_thresholds,
    run_threshold_scans,
)
from mosaiqtl.io.covariates import encode_sex
from mosaiqtl.pipeline import call_egenes
from mosaiqtl.utils import build_z_matrix, rint

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_gastroc_celltype')

PASS = '✅'
FAIL = '❌'
SKIP = '⚠️ '

_GASTROC_DATA = Path('validation/gastroc_data')
_NON_CC_ORDER: list[str] = list(STRAIN_ORDER)

PRIORITY_GENE: dict[str, str] = {'2': 'Cwc22'}


# ─────────────────────────────────────────────────────────────────────────────
# helpers (identical to validate_gastroc.py)
# ─────────────────────────────────────────────────────────────────────────────

def _ct_safe(cell_type: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', cell_type.lower()).strip('_')


def make_names(genes: list[str]) -> list[str]:
    """Approximate R's make.names(genes, unique=TRUE)."""
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


def _checkpoint(py_dir: Path, section: str, force: bool) -> bool:
    done = py_dir / f's{section}.done'
    if done.exists() and not force:
        log.info('  (checkpoint — skipping S%s; use --force to rerun)', section)
        return True
    return False


def _mark_done(py_dir: Path, section: str) -> None:
    (py_dir / f's{section}.done').write_text('done')


def _scan_to_df(scan: ScanResult, gene_id: str) -> pd.DataFrame:
    return pd.DataFrame({
        'gene':   gene_id,
        'chr':    scan.chromosomes,
        'locus':  scan.locus_ids,
        'pos_mb': scan.positions,
        'lod':    scan.lod,
    })


def _load_expression(h5_path: Path, genes_wanted: list[str],
                     ) -> tuple[np.ndarray, list[str], list[str]]:
    """Load only the requested genes from counts_lognorm.h5."""
    import h5py
    with h5py.File(h5_path, 'r') as f:
        all_genes   = np.array(f['genes']).astype(str).tolist()
        all_samples = np.array(f['samples']).astype(str).tolist()
        counts_ds   = f['counts']
        wanted_set  = set(genes_wanted)
        idx = [i for i, g in enumerate(all_genes) if g in wanted_set]
        if not idx:
            return np.empty((0, len(all_samples))), [], all_samples
        counts = np.array(counts_ds[idx, :], dtype=np.float32)
    genes_out = [all_genes[i] for i in idx]
    return counts, genes_out, all_samples


def _build_covariates(meta: pd.DataFrame, sample_order: list[str]) -> np.ndarray:
    """Build covariate matrix — mirrors R formula '~ sex + dataset'."""
    meta_idx = meta.set_index('sample_id')
    rows     = meta_idx.loc[sample_order]
    sex_vec  = encode_sex(rows['sex']).values.astype(np.float64)
    cat      = pd.Categorical(rows['dataset'].values,
                              categories=['Founder', 'CC1', 'CC2', 'CC3', 'F1'])
    dummies  = pd.get_dummies(cat, drop_first=True).values.astype(np.float64)
    return np.column_stack([sex_vec, dummies])


def _ensure_canonical_order(cache: GenomeCache) -> GenomeCache:
    """Reorder cache sample axis to canonical: Founders → F1s → CC alphanumeric."""
    current     = list(cache.sample_ids)
    founders_set = set(FOUNDER_ORDER)
    f1_set       = set(F1_COLORS.keys())
    cc_strains   = sorted(s for s in current if s not in founders_set and s not in f1_set)
    canonical    = [s for s in FOUNDER_ORDER if s in founders_set.intersection(current)]
    canonical   += [s for s in F1_COLORS.keys() if s in f1_set.intersection(current)]
    canonical   += cc_strains
    if canonical == current:
        return cache
    cur_idx = {s: i for i, s in enumerate(current)}
    canonical = [s for s in canonical if s in cur_idx]
    perm      = np.array([cur_idx[s] for s in canonical])
    new_data: dict = {}
    for chrom in cache.chromosomes:
        lids, pos, probs = cache.get_chromosome(chrom)
        new_data[chrom] = {
            'locus_ids': lids.copy(),
            'positions': pos.copy(),
            'probs':     probs[:, perm, :],
        }
    return GenomeCache(sample_ids=np.array(canonical), founders=list(cache.founders),
                       data=new_data)


# ─────────────────────────────────────────────────────────────────────────────
# run
# ─────────────────────────────────────────────────────────────────────────────

def run(cell_type: str, gastroc_data: Path, sections: set[int],
        force: bool, n_genes: int | None, scan_chr: list[str] | None) -> dict:
    """Run cell-type-specific eQTL scan and compare to whole-tissue.

    Parameters
    ----------
    cell_type    : cell type label (must match convert_gastroc_h5ad.py --cell_type output)
    gastroc_data : path to validation/gastroc_data/ (contains shared inputs)
    """
    ct = _ct_safe(cell_type)
    ct_dir  = gastroc_data / 'celltypes' / ct
    py_dir  = ct_dir / 'py_data'
    plt_dir = ct_dir / 'plots'
    for d in (py_dir, plt_dir):
        d.mkdir(parents=True, exist_ok=True)

    shared_py_dir = gastroc_data / 'py_data'    # cache.h5, kinship_decomp.npz live here
    h5_path = ct_dir / 'counts_lognorm.h5'

    results: dict[str, bool | None] = {}

    # ── verify prerequisites ──────────────────────────────────────────────────
    if not h5_path.exists():
        log.error('counts_lognorm.h5 not found at %s\n'
                  'Run first: conda run -n bioinfo python validation/convert_gastroc_h5ad.py'
                  ' --cell_type %s', h5_path, cell_type)
        sys.exit(1)

    # ── shared inputs ─────────────────────────────────────────────────────────
    meta = pd.read_csv(gastroc_data / 'metadata.csv')
    sample_ids_meta = meta['sample_id'].tolist()
    meta_geno_map: dict[str, str] = dict(zip(meta['sample_id'], meta['genotype']))
    log.info('metadata: %d samples  %d genotypes  datasets: %s',
             len(meta), meta['genotype'].nunique(), sorted(meta['dataset'].unique()))

    gene_info = pd.read_csv(gastroc_data / 'gene_info.csv')
    gene_info['chr'] = gene_info['chr'].astype(str)
    gene_info['gene_name'] = make_names(gene_info['gene_id'].tolist())
    gene_id_to_name = dict(zip(gene_info['gene_id'],   gene_info['gene_name']))
    log.info('gene_info: %d genes', len(gene_info))

    hdf5_cache = shared_py_dir / 'cache.h5'
    if not hdf5_cache.exists():
        log.error('cache.h5 not found at %s — run validate_gastroc.py --build-cache first',
                  hdf5_cache)
        sys.exit(1)
    log.info('loading GenomeCache from %s', hdf5_cache)
    cache = GenomeCache.from_hdf5(str(hdf5_cache))
    cache = _ensure_canonical_order(cache)
    log.info('cache: %d chromosomes  %d strains', len(cache.chromosomes), len(cache.sample_ids))

    kd_npz = shared_py_dir / 'kinship_decomp.npz'
    if kd_npz.exists():
        log.info('loading kinship decomp: %s', kd_npz)
        raw = np.load(kd_npz, allow_pickle=True)
        kinship_decomp = {k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
                          for k in cache.chromosomes if f'{k}_vals' in raw}
        del raw
    else:
        log.info('computing LOCO kinship (first run)...')
        kin_raw = calc_kinship(cache, method='loco')
        kinship_decomp = decomp_kinship(kin_raw)
        save_kd: dict[str, np.ndarray] = {}
        for ch, kd in kinship_decomp.items():
            save_kd[f'{ch}_vals'] = kd['values']
            save_kd[f'{ch}_vecs'] = kd['vectors']
        np.savez(kd_npz, **save_kd)
        del kin_raw, save_kd; gc.collect()

    chrs_to_scan = sorted([c for c in cache.chromosomes if c in kinship_decomp],
                           key=_chr_sort_key)
    if scan_chr is not None:
        chrs_to_scan = [c for c in chrs_to_scan if c in scan_chr]

    covariates_meta = _build_covariates(meta, sample_ids_meta)
    _sex_by_sid: dict[str, float] = (
        encode_sex(meta['sex']).set_axis(meta['sample_id']).to_dict()
        if 'sex' in meta.columns else {}
    )

    log.info('\ncell_type=%r  chromosomes to scan (%d): %s',
             cell_type, len(chrs_to_scan), chrs_to_scan)

    # ─────────────────────────────────────────────────────────────────────────
    # S4 — LMM cis scan
    # ─────────────────────────────────────────────────────────────────────────
    sec = 4
    py_lod_path  = py_dir / 's4_lod.csv'
    peak_lod_path = py_dir / 's4_peak_lod.csv'

    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS4: LMM cis scan  cell_type=%r  chrs=%s', cell_type, chrs_to_scan)

            all_effect_rows: list[dict] = []
            _s4_lod_header_written = False
            _rss_s4 = _get_rss_mb()
            tracemalloc.start()
            t_scan = time.perf_counter()
            n_genes_total = 0
            py_lod_path.write_text('')

            for ch in chrs_to_scan:
                ch_genes    = gene_info[gene_info['chr'] == str(ch)]
                if len(ch_genes) == 0:
                    continue

                gene_ids_ch   = ch_genes['gene_id'].tolist()
                gene_names_ch = ch_genes['gene_name'].tolist()
                if n_genes is not None:
                    gene_ids_ch   = gene_ids_ch[:n_genes]
                    gene_names_ch = gene_names_ch[:n_genes]

                counts_ch, loaded_ids, sids_ch = _load_expression(h5_path, gene_ids_ch)
                if counts_ch.shape[0] == 0:
                    log.info('  chr%s: no genes in HDF5 — skipping', ch)
                    continue

                geno_ids_ch  = [meta_geno_map.get(s, s) for s in sids_ch]
                loaded_names = [gene_id_to_name.get(gid, gid) for gid in loaded_ids]
                cov_ch       = _build_covariates(meta, sids_ch)

                tss_lookup: dict[str, float] = {}
                if 'start' in ch_genes.columns:
                    for _, grow in ch_genes.iterrows():
                        tss_lookup[grow['gene_name']] = float(grow['start']) / 1e6

                # precompute chromosome-constant quantities once for all genes
                _z_ch, _vsid_ch = build_z_matrix(geno_ids_ch, list(cache.sample_ids))
                _z_sub_ch    = _z_ch[_vsid_ch]
                _cov_valid   = cov_ch[_vsid_ch]
                _x_full_ch   = np.hstack([np.ones((_z_sub_ch.shape[0], 1)), _cov_valid])
                _ev_m        = kinship_decomp[ch]['values']
                _evc_m       = kinship_decomp[ch]['vectors']
                _K_m         = (_evc_m * _ev_m) @ _evc_m.T
                _K_n         = _z_sub_ch @ _K_m @ _z_sub_ch.T
                del _K_m
                _evals_ch, _evecs_ch = np.linalg.eigh(_K_n)
                del _K_n
                _chr_data_ch = precompute_chromosome(
                    cache, ch, _evals_ch, _evecs_ch, z_mat=_z_sub_ch)

                log.info('  chr%s: %d genes  %d loci  %d samples (precomputed)',
                         ch, len(loaded_ids), len(_chr_data_ch['locus_ids']),
                         _z_sub_ch.shape[0])
                t_chr = time.perf_counter()

                for gi, (gene_id, gene_name) in enumerate(zip(loaded_ids, loaded_names)):
                    if gi % 200 == 0 and gi > 0:
                        log.info('    chr%s %d/%d  %.0fs elapsed',
                                 ch, gi + 1, len(loaded_ids),
                                 time.perf_counter() - t_scan)
                    y_raw  = counts_ch[gi].astype(float)
                    tss_mb = tss_lookup.get(gene_name)

                    if not np.any(np.isnan(y_raw[_vsid_ch])):
                        y_sub  = rint(y_raw[_vsid_ch])
                        h2, _  = estimate_h2_null(
                            _evecs_ch.T @ y_sub, _evecs_ch.T @ _x_full_ch, _evals_ch)
                        chr_res = scan_chromosome(
                            y_sub, _chr_data_ch, _x_full_ch, _evals_ch, _evecs_ch, h2)
                        n_loci_res = len(chr_res['locus_ids'])
                        scan = ScanResult(
                            locus_ids    = chr_res['locus_ids'],
                            chromosomes  = np.full(n_loci_res, str(ch)),
                            positions    = chr_res['positions'],
                            lod          = chr_res['lod'],
                            p_value      = chr_res['p_value'],
                            betas        = chr_res['betas'],
                            h2_null      = {str(ch): h2},
                            phenotype_id = gene_name,
                            model        = 'additive',
                        )
                        _chr_precomputed = True
                    else:
                        scan = scan_h2lmm(
                            y_raw, cache,
                            covariates=cov_ch,
                            kinship_decomp=kinship_decomp,
                            sample_ids=geno_ids_ch,
                            use_rint=True,
                            phenotype_id=gene_name,
                            scan_window='cis',
                            gene_chr=ch,
                        )
                        _chr_precomputed = False

                    df = _scan_to_df(scan, gene_name)
                    df.to_csv(py_lod_path, mode='a',
                              header=not _s4_lod_header_written, index=False)
                    _s4_lod_header_written = True

                    if hasattr(scan, 'betas') and scan.betas is not None:
                        peak_idx = int(np.argmax(scan.lod))
                        _h2_val  = (list(scan.h2_null.values())[0]
                                    if scan.h2_null else float('nan'))
                        _disp_founders = _display_founders(list(cache.founders))
                        _rss0_g = float(chr_res.get('rss0', float('nan'))
                                        if _chr_precomputed else float('nan'))
                        _rss1_g = float(chr_res['rss1'][peak_idx]
                                        if _chr_precomputed and 'rss1' in chr_res
                                        else float('nan'))
                        _nv  = int(_z_sub_ch.shape[0]) if _chr_precomputed else 0
                        _p   = int(_x_full_ch.shape[1]) if _chr_precomputed else 0
                        _df  = max(_nv - _p, 1)
                        _ok  = _nv > _p and not np.isnan(_rss0_g) and _rss0_g > 0
                        _h2f = float(_h2_val) if not np.isnan(float(_h2_val)) else 0.0
                        eff: dict = {
                            'gene':       gene_name,
                            'chr':        str(ch),
                            'tss_mb':     round(float(tss_mb), 3) if tss_mb is not None else float('nan'),
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
                log.info('    chr%s done: %d genes  %.1fs  %.1f genes/min',
                         ch, len(loaded_ids), elapsed_chr,
                         len(loaded_ids) / elapsed_chr * 60 if elapsed_chr > 0 else 0)
                n_genes_total += len(loaded_ids)
                del counts_ch, _chr_data_ch, _evals_ch, _evecs_ch, _z_sub_ch, _cov_valid
                gc.collect()

            elapsed_s = time.perf_counter() - t_scan
            _, _s4_peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            if _s4_lod_header_written:
                log.info('  scan complete: %d genes  %.1fs  %.1f genes/min',
                         n_genes_total, elapsed_s,
                         n_genes_total / elapsed_s * 60 if elapsed_s > 0 else 0)
                lod_df = pd.read_csv(py_lod_path,
                                     usecols=['gene', 'chr', 'locus', 'pos_mb', 'lod'])
            else:
                log.warning('  no LOD rows produced — check chr/gene_info alignment')
                lod_df = pd.DataFrame(columns=['gene', 'chr', 'locus', 'pos_mb', 'lod'])

            if all_effect_rows:
                eff_df = pd.DataFrame(all_effect_rows)
                eff_df.to_csv(py_dir / 's4_allele_effects.csv', index=False)
                log.info('  s4_allele_effects.csv: %d genes', len(eff_df))

            if len(lod_df) > 0:
                peak_py = (lod_df.loc[lod_df.groupby('gene')['lod'].idxmax()]
                           [['gene', 'chr', 'locus', 'pos_mb', 'lod']]
                           .rename(columns={'lod': 'peak_lod',
                                            'locus': 'peak_locus',
                                            'pos_mb': 'peak_mb'})
                           .reset_index(drop=True))
                del lod_df; gc.collect()
                if all_effect_rows:
                    for col, field in [('h2', 'h2'), ('tss_mb', 'tss_mb'),
                                       ('n_valid', 'n_valid'), ('rss0', 'rss0'),
                                       ('rss1', 'rss1'), ('r2_peak', 'r2_peak'),
                                       ('sigma2_e', 'sigma2_e'), ('tau2', 'tau2')]:
                        m = {r['gene']: r[field] for r in all_effect_rows}
                        peak_py[col] = peak_py['gene'].map(m)
                    peak_py['cis_dist_mb'] = (
                        (peak_py['peak_mb'] - peak_py['tss_mb']).abs().round(3))
                _pk_cols = [
                    'gene', 'chr', 'tss_mb', 'peak_locus', 'peak_mb', 'cis_dist_mb',
                    'peak_lod', 'h2', 'n_valid', 'rss0', 'rss1', 'r2_peak', 'sigma2_e', 'tau2',
                ]
                peak_py = peak_py.reindex(
                    columns=[c for c in _pk_cols if c in peak_py.columns])
                peak_py.to_csv(peak_lod_path, index=False)
                log.info('  s4_peak_lod.csv: %d genes  peak LOD range %.2f – %.2f',
                         len(peak_py),
                         peak_py['peak_lod'].min(),
                         peak_py['peak_lod'].max())
                results['S4_lmm_scan'] = True
            else:
                results['S4_lmm_scan'] = None

            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S5 — permutation threshold (genome-wide GEV)
    # Uses the gene with the highest LOD as the representative for null draws.
    # ─────────────────────────────────────────────────────────────────────────
    sec = 5
    thr_path = py_dir / 's5_threshold.csv'
    perm_path = py_dir / 's5_perm_max_lods.csv'

    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS5: permutation threshold (median-h² gene, 1000 perms)...')
            if not peak_lod_path.exists():
                log.warning('  %s s4_peak_lod.csv missing — run S4 first', SKIP)
                results['S5_threshold'] = None
            else:
                peak_df_s5 = pd.read_csv(peak_lod_path)
                if len(peak_df_s5) == 0:
                    log.warning('  %s no genes in s4_peak_lod.csv', SKIP)
                    results['S5_threshold'] = None
                else:
                    peak_chr = str(peak_df_s5.iloc[peak_df_s5['peak_lod'].argmax()]['chr'])

                    # ── pick median-h² gene as representative (matches validate_gastroc.py) ──
                    # Using top-LOD gene inflates null LODs because its h2 is very high:
                    # sigma2_g >> sigma2_e → null draws dominated by kinship → permuted
                    # phenotype correlated with genotype → LODs ~20 instead of ~4-6.
                    # Median-h2 gives a calibrated threshold for the average gene.
                    _perm_gene: str
                    _h2_med: float
                    eff_path_s5 = py_dir / 's4_allele_effects.csv'
                    if eff_path_s5.exists():
                        _eff_s5 = pd.read_csv(eff_path_s5, usecols=['gene', 'h2'])
                        _h2_valid = _eff_s5['h2'].dropna()
                        _h2_med   = float(_h2_valid.median())
                        _best_row = _eff_s5.iloc[(_h2_valid - _h2_med).abs().values.argmin()]
                        _perm_gene = str(_best_row['gene'])
                        log.info('  median h²=%.3f  representative gene: %s',
                                 _h2_med, _perm_gene)
                    else:
                        # fallback: use highest-LOD gene on peak_chr
                        _sub = peak_df_s5[peak_df_s5['chr'].astype(str) == peak_chr]
                        _perm_gene = str(
                            _sub.loc[_sub['peak_lod'].idxmax(), 'gene'])
                        _h2_med = float('nan')
                        log.info('  no allele_effects.csv — fallback gene: %s', _perm_gene)

                    counts_s5, ids_s5, sids_s5 = _load_expression(h5_path, [_perm_gene])
                    if counts_s5.shape[0] == 0:
                        log.warning('  %s representative gene not in HDF5', SKIP)
                        results['S5_threshold'] = None
                    else:
                        _rss_s5 = _get_rss_mb()
                        tracemalloc.start()
                        t_s5 = time.perf_counter()

                        # build n-level Z-matrix (mirrors S4 per-chr precompute)
                        # so permutation LOD scale = (n/2)·log10(rss0/rss1)
                        # matches the S4 scan LOD scale — threshold is calibrated
                        _gids_s5  = [meta_geno_map.get(s, s) for s in sids_s5]
                        _z_s5, _vsid_s5 = build_z_matrix(
                            _gids_s5, list(cache.sample_ids))
                        _z_sub_s5   = _z_s5[_vsid_s5]              # (n_valid, m)
                        _cov_s5     = _build_covariates(meta, sids_s5)
                        _x_s5       = np.hstack([
                            np.ones((_z_sub_s5.shape[0], 1)),
                            _cov_s5[_vsid_s5]])
                        y_s5_raw    = counts_s5[0].astype(float)[_vsid_s5]
                        n_perms     = 1000
                        null_mat    = generate_null_outcomes(
                            y_s5_raw, n_perms=n_perms, seed=42)
                        log.info('  running %d permutations genome-wide'
                                 '  n=%d  m=%d ...',
                                 n_perms, _z_sub_s5.shape[0], _z_sub_s5.shape[1])
                        max_lods, _ = run_threshold_scans(
                            null_mat, cache,
                            phenotype=y_s5_raw,
                            covariates=_x_s5,
                            kinship_decomp=kinship_decomp,
                            chrom=None,
                            z_mat=_z_sub_s5,
                        )
                        pd.DataFrame({'max_lod': max_lods}).to_csv(perm_path, index=False)
                        gev_thr, _ = get_gev_thresholds(max_lods, percentile=0.95)
                        thr_df = pd.DataFrame([{
                            'cell_type':  cell_type,
                            'alpha':      0.05,
                            'threshold':  gev_thr,
                            'n_perms':    n_perms,
                            'ref_gene':   _perm_gene,
                            'ref_chr':    peak_chr,
                            'h2_median':  _h2_med,
                        }])
                        thr_df.to_csv(thr_path, index=False)
                        log.info('  GEV threshold (α=0.05): %.3f  (n_perms=%d)',
                                 gev_thr, n_perms)
                        results['S5_threshold'] = True

                        _, _s5_peak = tracemalloc.get_traced_memory()
                        tracemalloc.stop()
                        del null_mat, counts_s5
                        gc.collect()

            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S6 — eGene calling (peak LOD vs GEV threshold)
    # Output: eqtl_summary.csv — same schema as validate_gastroc.py
    # ─────────────────────────────────────────────────────────────────────────
    sec = 6
    eqtl_summary_path = py_dir / 'eqtl_summary.csv'
    egenes_df: pd.DataFrame | None = None

    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS6: eGene calling...')
            if not peak_lod_path.exists():
                log.warning('  %s s4_peak_lod.csv missing — run S4 first', SKIP)
                results['S6_egenes'] = None
            elif not thr_path.exists():
                log.warning('  %s s5_threshold.csv missing — run S5 first', SKIP)
                results['S6_egenes'] = None
            else:
                peak_df = pd.read_csv(peak_lod_path)
                thr_row = pd.read_csv(thr_path).iloc[0]
                thr     = float(thr_row['threshold'])
                perm_n  = int(thr_row.get('n_perms', 0))

                eff_path_s6 = py_dir / 's4_allele_effects.csv'
                eff_s6: pd.DataFrame | None = None
                if eff_path_s6.exists():
                    eff_s6 = pd.read_csv(
                        eff_path_s6, usecols=['gene', 'chr', 'peak_locus',
                                              'peak_mb', 'peak_lod', 'h2', 'tss_mb'])

                egenes_df = call_egenes(peak_df, thr, effects_df=eff_s6)

                # ── enrich with h2, tss_mb from allele effects ────────────────
                if eff_s6 is not None:
                    _h2_cols = ['gene', 'h2', 'tss_mb']
                    _h2_fill = eff_s6[[c for c in _h2_cols if c in eff_s6.columns]]
                    egenes_df = egenes_df.merge(_h2_fill, on='gene', how='left',
                                                suffixes=('', '_eff'))
                    for _c in ('h2', 'tss_mb'):
                        if f'{_c}_eff' in egenes_df.columns:
                            egenes_df[_c] = egenes_df[_c].combine_first(egenes_df.pop(f'{_c}_eff'))

                if 'tss_mb' in egenes_df.columns and 'peak_mb' in egenes_df.columns:
                    egenes_df['cis_dist_mb'] = (
                        (egenes_df['peak_mb'] - egenes_df['tss_mb']).abs().round(3))

                egenes_df['perm_n']           = perm_n
                egenes_df['chi2_haplotype']   = 2.0 * np.log(10) * egenes_df['peak_lod']
                egenes_df['p_haplotype']      = sp_stats.chi2.sf(
                    egenes_df['chi2_haplotype'].values, df=7)

                col_order = ['gene', 'chr', 'tss_mb', 'peak_locus', 'peak_mb',
                             'cis_dist_mb', 'peak_lod', 'h2',
                             'perm_n', 'chi2_haplotype', 'p_haplotype']
                egenes_df = egenes_df[[c for c in col_order if c in egenes_df.columns]]
                egenes_df.to_csv(eqtl_summary_path, index=False)

                n_eg  = len(egenes_df)
                n_tot = len(peak_df)
                log.info('  %s eGenes: %d / %d  (%.1f%%)  threshold=%.3f',
                         PASS if n_eg > 0 else SKIP,
                         n_eg, n_tot, 100 * n_eg / n_tot if n_tot > 0 else 0, thr)
                results['S6_egenes'] = n_eg > 0
            _mark_done(py_dir, str(sec))
    elif eqtl_summary_path.exists():
        egenes_df = pd.read_csv(eqtl_summary_path)

    # ─────────────────────────────────────────────────────────────────────────
    # Compare — cell-type LOD vs whole-tissue LOD
    # Tests the hypothesis: myonuclei eQTLs ≈ whole-tissue pseudobulk
    # ─────────────────────────────────────────────────────────────────────────
    sec_cmp = 99   # sentinel — not a standard section number
    wt_peak_path = shared_py_dir / 's4_peak_lod.csv'

    if 99 in sections or not sections.isdisjoint({4, 5, 6}):
        log.info('\nCompare: %s vs whole-tissue peak LOD', cell_type)
        if not peak_lod_path.exists():
            log.warning('  %s %s s4_peak_lod.csv not found — run S4 first', SKIP, cell_type)
        elif not wt_peak_path.exists():
            log.warning('  %s whole-tissue s4_peak_lod.csv not found at %s', SKIP, wt_peak_path)
        else:
            ct_peak = pd.read_csv(peak_lod_path)[['gene', 'peak_lod']].rename(
                columns={'peak_lod': f'lod_{ct}'})
            wt_peak = pd.read_csv(wt_peak_path)[['gene', 'peak_lod']].rename(
                columns={'peak_lod': 'lod_wt'})
            merged = ct_peak.merge(wt_peak, on='gene', how='inner')
            log.info('  genes compared: %d (inner join on gene name)', len(merged))

            if len(merged) >= 10:
                r_pearson, _ = sp_stats.pearsonr(merged[f'lod_{ct}'], merged['lod_wt'])
                r_spearman, _ = sp_stats.spearmanr(merged[f'lod_{ct}'], merged['lod_wt'])
                r2 = r_pearson ** 2
                log.info('  LOD r² (Pearson)  = %.4f', r2)
                log.info('  LOD r  (Spearman) = %.4f', r_spearman)
                results['Compare_r2_pearson']  = r2
                results['Compare_r_spearman']  = r_spearman

                merged.to_csv(py_dir / 'compare_peak_lod.csv', index=False)

                # ── scatter plot ──────────────────────────────────────────────
                sns.set_style('ticks')
                fig, ax = plt.subplots(figsize=(5, 5), dpi=300)
                ax.scatter(merged['lod_wt'], merged[f'lod_{ct}'],
                           s=8, alpha=0.4, linewidths=0, color='#4F6EAF')

                # highlight priority genes
                for _, row in merged[merged['gene'].isin(PRIORITY_GENE.values())].iterrows():
                    ax.annotate(str(row['gene']),
                                (row['lod_wt'], row[f'lod_{ct}']),
                                fontsize=7, color='#D83026',
                                xytext=(4, 4), textcoords='offset points')
                    ax.scatter([row['lod_wt']], [row[f'lod_{ct}']],
                               s=30, color='#D83026', zorder=5)

                _lim_max = max(merged['lod_wt'].max(), merged[f'lod_{ct}'].max()) * 1.05
                ax.set_xlim(0, _lim_max)
                ax.set_ylim(0, _lim_max)
                ax.plot([0, _lim_max], [0, _lim_max], 'k--', linewidth=0.7, alpha=0.5)
                ax.set_xlabel('Whole-tissue peak LOD', fontsize=10)
                ax.set_ylabel(f'{cell_type} peak LOD', fontsize=10)
                ax.set_title(
                    f'{cell_type} vs whole-tissue eQTL signal\n'
                    f'r²={r2:.4f}  r_sp={r_spearman:.4f}  n={len(merged):,}',
                    fontsize=9)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 'compare_lod_scatter.png', dpi=300,
                            bbox_inches='tight')
                plt.close(fig)
                log.info('  [plot] compare_lod_scatter.png')

    # ─────────────────────────────────────────────────────────────────────────
    # descriptive plots from S4 output
    # ─────────────────────────────────────────────────────────────────────────
    _draw_plots(plt_dir, py_dir, cache, cell_type)

    return results


def _draw_plots(plt_dir: Path, py_dir: Path,
                cache: GenomeCache, cell_type: str) -> None:
    """All descriptive plots from S4/S5/S6 outputs.

    Reads from py_dir CSVs and writes PNGs to plt_dir.  Safe to call whenever
    — each block gates on the relevant CSV existing.

    Outputs
    -------
    s4_manhattan.png            peak LOD per gene across all chromosomes
    s4_lod_profile.png          LOD profile for priority / top gene
    s4_founder_effects.png      founder effect bars for top gene
    s4_lod_histogram.png        peak LOD distribution (all genes)
    s4_cis_distance.png         peak − TSS distance histogram (eGenes)
    s4_lod_panel_grid.png       LOD-profile grid across chromosomes
    s4_locus_panel_grid.png     effects + LOD + zoom grid across chromosomes
    s4_locus_panel/             per-chromosome locus panels (effects + LOD + zoom)
    s5_perm_distribution.png    permutation max-LOD histogram + GEV fit
    s6_egene_lod_distribution.png  peak LOD histogram coloured by eGene status
    """
    sns.set_style('ticks')

    peak_lod_path = py_dir / 's4_peak_lod.csv'
    if not peak_lod_path.exists():
        return
    peak_df = pd.read_csv(peak_lod_path)
    if peak_df.empty:
        return

    founders = _display_founders(list(cache.founders))

    # ── Manhattan (peak LOD per gene, one dot per gene) ───────────────────────
    try:
        thr_df = (pd.read_csv(py_dir / 's5_threshold.csv')
                  if (py_dir / 's5_threshold.csv').exists() else None)
        thr = float(thr_df.iloc[0]['threshold']) if thr_df is not None else None

        peak_df['chr'] = peak_df['chr'].astype(str)
        chrs_sorted = sorted(peak_df['chr'].unique(), key=_chr_sort_key)
        chr_offset: dict[str, float] = {}
        offset = 0.0
        for c in chrs_sorted:
            chr_offset[c] = offset
            offset += peak_df[peak_df['chr'] == c]['peak_mb'].max() + 5.0

        fig, ax = plt.subplots(figsize=(14, 4), dpi=300)
        colors = ['#4F6EAF', '#8AB78D']
        for ci, ch in enumerate(chrs_sorted):
            sub = peak_df[peak_df['chr'] == ch]
            x   = sub['peak_mb'] + chr_offset[ch]
            ax.scatter(x, sub['peak_lod'], s=6, alpha=0.5, linewidths=0,
                       color=colors[ci % 2], label=None)
        if thr is not None:
            ax.axhline(thr, color='#D83026', linewidth=0.8, linestyle='--',
                       label=f'GEV thr={thr:.2f}')
            ax.legend(fontsize=8, frameon=False)
        ax.set_xlim(0, offset)
        ax.set_ylim(0, peak_df['peak_lod'].max() * 1.1)
        mid_x = {c: chr_offset[c] + peak_df[peak_df['chr'] == c]['peak_mb'].median()
                  for c in chrs_sorted}
        ax.set_xticks(list(mid_x.values()))
        ax.set_xticklabels([c for c in chrs_sorted], fontsize=7, rotation=45)
        ax.set_xlabel('Chromosome', fontsize=10)
        ax.set_ylabel('Peak LOD', fontsize=10)
        ax.set_title(f'{cell_type} — eQTL Manhattan ({len(peak_df):,} genes)', fontsize=10)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(plt_dir / 's4_manhattan.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  [plot] s4_manhattan.png')
    except Exception as exc:
        log.warning('  [plot] s4_manhattan: %s', exc)

    # ── stream-filter s4_lod.csv once for all downstream plots ──────────────────
    # s4_lod.csv can be 100M+ rows (43k genes × loci).  Determine needed genes
    # from the small fast CSVs, then stream in 2M-row chunks keeping only those
    # rows.  _lod_df is reused by every plot that needs full LOD curves.
    lod_csv  = py_dir / 's4_lod.csv'
    eff_csv  = py_dir / 's4_allele_effects.csv'
    _lod_df: 'pd.DataFrame | None' = None
    _eff_df: 'pd.DataFrame | None' = None
    _peak_df_str = peak_df.copy()
    _peak_df_str['chr'] = _peak_df_str['chr'].astype(str)

    if eff_csv.exists():
        try:
            _eff_df = pd.read_csv(eff_csv, dtype={'chr': str})
        except Exception as exc:
            log.warning('  s4_allele_effects.csv read failed: %s', exc)

    if lod_csv.exists():
        try:
            # top gene per chromosome from peak_df + priority genes + top per eff_df
            _needed: set[str] = set(PRIORITY_GENE.values())
            _needed.update(
                _peak_df_str.loc[
                    _peak_df_str.groupby('chr')['peak_lod'].idxmax(), 'gene'
                ].astype(str))
            if _eff_df is not None:
                _needed.update(
                    _eff_df.loc[
                        _eff_df.groupby('chr')['peak_lod'].idxmax(), 'gene'
                    ].astype(str))
            _lchunks: list[pd.DataFrame] = []
            for _lc in pd.read_csv(lod_csv,
                                    usecols=['gene', 'chr', 'pos_mb', 'lod'],
                                    dtype={'chr': str}, chunksize=2_000_000):
                _hit = _lc[_lc['gene'].isin(_needed)]
                if len(_hit):
                    _lchunks.append(_hit.copy())
            _lod_df = (pd.concat(_lchunks, ignore_index=True)
                       if _lchunks
                       else pd.DataFrame(columns=['gene', 'chr', 'pos_mb', 'lod']))
            del _lchunks; gc.collect()
            log.info('  s4_lod.csv filtered to %d genes (%d rows)',
                     len(_needed), len(_lod_df))
        except Exception as exc:
            log.warning('  s4_lod.csv read failed: %s', exc)

    # ── LOD profile for priority / top gene ──────────────────────────────────
    if _lod_df is not None:
        try:
            top_gene = next(
                (pg for pg in PRIORITY_GENE.values()
                 if not peak_df[peak_df['gene'] == pg].empty),
                None,
            ) or str(peak_df.loc[peak_df['peak_lod'].idxmax(), 'gene'])

            lod_gene = _lod_df[_lod_df['gene'] == top_gene]
            if not lod_gene.empty:
                fig, ax = plt.subplots(figsize=(10, 3), dpi=300)
                ax.fill_between(lod_gene['pos_mb'], lod_gene['lod'],
                                alpha=0.7, color='#4F6EAF', linewidth=0)
                ax.set_xlim(0, lod_gene['pos_mb'].max())
                ax.set_ylim(0, lod_gene['lod'].max() * 1.1)
                ax.set_xlabel(
                    f"chr{lod_gene['chr'].iloc[0]} position (Mb)", fontsize=10)
                ax.set_ylabel('LOD', fontsize=10)
                ax.set_title(
                    f'{cell_type} — {top_gene}  peak LOD {lod_gene["lod"].max():.2f}',
                    fontsize=10)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_lod_profile.png', dpi=300,
                            bbox_inches='tight')
                plt.close(fig)
                log.info('  [plot] s4_lod_profile.png  (gene=%s)', top_gene)
        except Exception as exc:
            log.warning('  [plot] s4_lod_profile: %s', exc)

    # ── S4: founder effect bars — one plot per chromosome ────────────────────
    if _eff_df is not None:
        try:
            beta_cols = [c for c in _eff_df.columns if c.startswith('beta_')]
            fnames    = _display_founders([b.replace('beta_', '') for b in beta_cols])
            _fe_dir   = plt_dir / 's4_founder_effects'
            _fe_dir.mkdir(exist_ok=True)
            _chrs_eff = sorted(_eff_df['chr'].unique(),
                                key=lambda c: (0 if c.isdigit() else 1,
                                               int(c) if c.isdigit() else c))
            for _ec in _chrs_eff:
                _eff_c = _eff_df[_eff_df['chr'] == _ec]
                if _eff_c.empty:
                    continue
                # priority gene on this chr, else top-LOD gene
                _top_e = PRIORITY_GENE.get(_ec)
                if _top_e is None or _top_e not in _eff_c['gene'].values:
                    _top_e = str(_eff_c.loc[_eff_c['peak_lod'].idxmax(), 'gene'])
                _row = _eff_c[_eff_c['gene'] == _top_e].iloc[0]
                betas = np.array([float(_row[b]) for b in beta_cols])
                fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
                _viz_effect_bars(betas, fnames, ax=ax)
                ax.set_title(
                    f'{cell_type} — {_top_e}  {_row["peak_locus"]}  '
                    f'LOD={_row["peak_lod"]:.1f}',
                    fontsize=9)
                fig.tight_layout()
                fig.savefig(_fe_dir / f's4_founder_effects_chr{_ec}.png',
                            dpi=300, bbox_inches='tight')
                plt.close(fig)
            log.info('  [plot] s4_founder_effects/  (%d chr)', len(_chrs_eff))
        except Exception as exc:
            log.warning('  [plot] s4_founder_effects: %s', exc)

    # ── S4: LOD histogram (all genes) ─────────────────────────────────────────
    try:
        thr_hist = None
        if (py_dir / 's5_threshold.csv').exists():
            _td = pd.read_csv(py_dir / 's5_threshold.csv')
            thr_hist = float(_td.iloc[0]['threshold'])
        lods_all = peak_df['peak_lod'].dropna().values
        fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
        _viz_lod_histogram(lods_all, ax=ax, threshold=thr_hist)
        ax.set_title(f'{cell_type} — peak LOD distribution  n={len(lods_all):,}',
                     fontsize=9)
        fig.tight_layout()
        fig.savefig(plt_dir / 's4_lod_histogram.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  [plot] s4_lod_histogram.png')
    except Exception as exc:
        log.warning('  [plot] s4_lod_histogram: %s', exc)

    # ── S4: cis-distance histogram (eGenes only) ──────────────────────────────
    _sum_path = py_dir / 'eqtl_summary.csv'
    if _sum_path.exists() and 'tss_mb' in peak_df.columns and 'peak_mb' in peak_df.columns:
        try:
            egenes_set = set(pd.read_csv(_sum_path)['gene'])
            eg_df = peak_df[peak_df['gene'].isin(egenes_set)].dropna(
                subset=['tss_mb', 'peak_mb'])
            if len(eg_df) > 0:
                fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
                _viz_cis_distance(eg_df['tss_mb'].values,
                                  eg_df['peak_mb'].values, ax=ax)
                ax.set_title(
                    f'{cell_type} — cis-distance  n={len(eg_df)} eGenes',
                    fontsize=9)
                fig.tight_layout()
                fig.savefig(plt_dir / 's4_cis_distance.png', dpi=300,
                            bbox_inches='tight')
                plt.close(fig)
                log.info('  [plot] s4_cis_distance.png  (%d eGenes)', len(eg_df))
        except Exception as exc:
            log.warning('  [plot] s4_cis_distance: %s', exc)

    # ── S4: locus panels — full LOD curves from _lod_df (already filtered) ──────
    # _lod_df was stream-filtered from s4_lod.csv above (top gene per chr +
    # priority genes).  _eff_df holds peak-locus betas for the same gene set.
    if _lod_df is not None and _eff_df is not None:
        try:
            _thr_by_chr: dict = {}
            if (py_dir / 's5_threshold.csv').exists():
                _td2 = pd.read_csv(py_dir / 's5_threshold.csv')
                if 'chr' in _td2.columns:
                    _td2['chr'] = _td2['chr'].astype(str)
                    _thr_by_chr = dict(zip(_td2['chr'], _td2['threshold']))
                elif len(_td2) == 1:
                    _thr_val = float(_td2.iloc[0]['threshold'])
                    _thr_by_chr = {c: _thr_val
                                   for c in _lod_df['chr'].unique()}

            _tss_map: dict[str, float] = {}
            if 'tss_mb' in _eff_df.columns:
                _tss_map = {str(g): float(t)
                            for g, t in zip(_eff_df['gene'], _eff_df['tss_mb'])
                            if pd.notna(t)}

            # per-chromosome locus panels (effects + LOD profile + zoom)
            _lp_dir = plt_dir / 's4_locus_panel'
            _lp_dir.mkdir(exist_ok=True)
            _lp_figs = _viz_locus_panels(
                _lod_df, _eff_df, founders,
                _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
                tss_mb_map=_tss_map or None,
            )
            for _lp_c, _lp_fig in _lp_figs.items():
                _lp_fig.savefig(_lp_dir / f's4_locus_panel_chr{_lp_c}.png', dpi=300)
                plt.close(_lp_fig)
            if _lp_figs:
                log.info('  [plot] s4_locus_panel/  (%d chr)', len(_lp_figs))

            # locus panel grid (effects + LOD)
            fig_lg = _viz_locus_panel_grid(
                _lod_df, _eff_df, founders,
                _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
            )
            fig_lg.savefig(plt_dir / 's4_locus_panel_grid.png', dpi=300,
                           bbox_inches='tight')
            plt.close(fig_lg)
            log.info('  [plot] s4_locus_panel_grid.png')

            # LOD-only panel grid — pk_df drives top-gene selection per chr
            _pk_df2 = _peak_df_str.copy()
            fig_lo = _viz_lod_panel_grid(
                _lod_df, _pk_df2, _thr_by_chr or None,
                priority_genes=PRIORITY_GENE,
            )
            fig_lo.savefig(plt_dir / 's4_lod_panel_grid.png', dpi=300,
                           bbox_inches='tight')
            plt.close(fig_lo)
            log.info('  [plot] s4_lod_panel_grid.png')
        except Exception as exc:
            log.warning('  [plot] s4_locus_panels: %s', exc)

    # ── S5: permutation max-LOD distribution + GEV fit ───────────────────────
    perm_csv = py_dir / 's5_perm_max_lods.csv'
    thr_csv  = py_dir / 's5_threshold.csv'
    if perm_csv.exists():
        try:
            max_lods = pd.read_csv(perm_csv)['max_lod'].values
            thr_s5 = None
            if thr_csv.exists():
                thr_s5 = float(pd.read_csv(thr_csv).iloc[0]['threshold'])
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
            _viz_perm_distribution(max_lods, ax=ax, threshold=thr_s5)
            ax.set_title(
                f'{cell_type} — permutation distribution  n={np.isfinite(max_lods).sum()}',
                fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 's5_perm_distribution.png', dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            log.info('  [plot] s5_perm_distribution.png')
        except Exception as exc:
            log.warning('  [plot] s5_perm_distribution: %s', exc)

    # ── S6: eGene LOD distribution + threshold ────────────────────────────────
    if _sum_path.exists():
        try:
            thr_s6 = None
            if thr_csv.exists():
                thr_s6 = float(pd.read_csv(thr_csv).iloc[0]['threshold'])
            egenes_s6 = set(pd.read_csv(_sum_path)['gene'])
            lods_s6   = peak_df['peak_lod'].dropna().values
            is_eg_s6  = peak_df.loc[peak_df['peak_lod'].notna(),
                                     'gene'].isin(egenes_s6).values
            fig, ax = plt.subplots(figsize=(7, 4), dpi=300)
            _viz_egene_lod(lods_s6, is_eg_s6, ax=ax, threshold=thr_s6)
            ax.set_title(
                f'{cell_type} — eGene calling  '
                f'{is_eg_s6.sum()} / {len(lods_s6)} genes',
                fontsize=9)
            fig.tight_layout()
            fig.savefig(plt_dir / 's6_egene_lod_distribution.png', dpi=300,
                        bbox_inches='tight')
            plt.close(fig)
            log.info('  [plot] s6_egene_lod_distribution.png  (%d eGenes)',
                     int(is_eg_s6.sum()))
        except Exception as exc:
            log.warning('  [plot] s6_egene_lod_distribution: %s', exc)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cell_type', type=str, default='myonuclei',
                        help='cell type to scan (default: myonuclei)')
    parser.add_argument('--gastroc_data', type=Path,
                        default=Path('validation/gastroc_data'),
                        help='path to gastroc_data/ root')
    parser.add_argument('--sections', type=str, default=None,
                        help='comma-separated sections to run, e.g. 4,5,6 '
                             '(default: 4,5,6,99 where 99=Compare)')
    parser.add_argument('--chr', type=str, default=None,
                        help='comma-separated chromosomes to scan, e.g. 8,19')
    parser.add_argument('--force', action='store_true',
                        help='rerun sections even if checkpoint exists')
    parser.add_argument('--n_genes', type=int, default=None,
                        help='limit genes per chromosome (fast test)')
    args = parser.parse_args()

    if args.sections is not None:
        sections = {int(s.strip()) for s in args.sections.split(',')}
    else:
        sections = {4, 5, 6, 99}

    scan_chr = None
    if args.chr is not None and args.chr.strip().lower() != 'all':
        scan_chr = [c.strip() for c in args.chr.split(',')]

    log.info('mosaiQTL cell-type validation  cell_type=%r  sections=%s',
             args.cell_type, sorted(sections))

    results = run(
        cell_type=args.cell_type,
        gastroc_data=args.gastroc_data,
        sections=sections,
        force=args.force,
        n_genes=args.n_genes,
        scan_chr=scan_chr,
    )

    log.info('\n── results ──────────────────────────────────────────────')
    for k, v in results.items():
        sym = PASS if v is True else (FAIL if v is False else SKIP)
        log.info('  %s  %s = %s', sym, k, v)


if __name__ == '__main__':
    main()

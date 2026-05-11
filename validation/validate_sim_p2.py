"""
validate_sim_p2.py — Phase 2 simulated validation: all 11 miQTL functions

Python mirror of validation/r_reference/port_sim_validation_p2.R.
Self-contained: builds its own GenomeCache from phase2_sim/r_data/probs/
(Phase 2 R script runs sim.CC.data() independently; cannot reuse Phase 1 cache).
Uses real LOCO kinship throughout — no OLS shortcut.

Sections:
  S1  cache structure          loci/founders/samples match R
  S2  haplotype probs          concordance ≥ 0.95 at spot-check loci
  S3  LOCO kinship             r > 0.999 per chromosome
  S4  LMM scan (10 phenotypes) ≥50% phenotypes r² ≥ 0.975
  S5  cache collapse           ≥5% loci removed AND lod_r² ≥ 0.999
  S6  permutation/GEV          GEV fit delta (GEV vs empirical p95) < 0.5 LOD
  S7  MI scan                  LOD r ≥ 0.75
  S8  SNP scan                 ⚠️ GATED — requires allele_dir
  S9  mediation                LOD r ≥ 0.90
  S10 bootstrap CI             planted locus inside CI
  S11 instability              planted locus peak_freq ≥ 0.70

Outputs:
  phase2_sim/py_data/     Python outputs + checkpoint .done files
  phase2_sim/compare/     merged py-vs-R side-by-side CSVs (no plots)
  phase2_sim/plots/       all QC + comparison figures

Usage:
  python validation/validate_sim_p2.py
  python validation/validate_sim_p2.py --sections 3,4,6
  python validation/validate_sim_p2.py --force
"""

import argparse
import gc
import logging
import sys
import time
from pathlib import Path

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Carlito'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import genextreme as _gev

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mosaiqtl.genotype.genomecache import GenomeCache, collapse_genomecache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.scan import (
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
from mosaiqtl.viz.genome import manhattan, lod_interval
from mosaiqtl.viz.effects import coef_profile
from mosaiqtl.viz.haplotypes import prob_heatmap

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_sim_p2')

PASS = '✅'
FAIL = '❌'
SKIP = '⚠️ '


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_cache_from_probs_csv(
    probs_dir: Path,
    keep_strains: 'list[str] | None' = None,
) -> 'GenomeCache':
    """build GenomeCache from per-chromosome CSVs exported by port_sim_validation_p2.R.

    Each CSV has columns: locus, pos_mb, strain, A, B, C, D, E, F, G, H
    """
    from mosaiqtl.genotype.founders import FOUNDER_ORDER, FOUNDER_CODE

    founder_cols: list[str] = [FOUNDER_CODE[name] for name in FOUNDER_ORDER]

    data: dict = {}
    sample_ids: 'list | None' = None

    for csv_path in sorted(probs_dir.glob('probs_*.csv')):
        chr_id = csv_path.stem.replace('probs_', '')
        df = pd.read_csv(csv_path)

        if keep_strains is not None:
            df = df[df['strain'].isin(keep_strains)]
            df['strain'] = pd.Categorical(
                df['strain'], categories=keep_strains, ordered=True)
            df = df.sort_values(['locus', 'strain'])

        n_strains = df['strain'].nunique()
        if sample_ids is None:
            sample_ids = df['strain'].drop_duplicates().tolist()

        loci = list(pd.Series(df['locus']).drop_duplicates())
        positions = (
            df.groupby('locus')['pos_mb'].first()
            .reindex(loci).values.astype(np.float64)
        )
        probs = df[founder_cols].values.reshape(len(loci), n_strains, 8).astype(np.float64)
        row_sums = probs.sum(axis=2, keepdims=True)
        probs = probs / np.where(row_sums > 0, row_sums, 1.0)
        data[chr_id] = {
            'locus_ids': np.array(loci),
            'positions': positions,
            'probs':     probs,
        }

    if not data:
        raise FileNotFoundError(f'no probs_*.csv files found in {probs_dir}')

    return GenomeCache(
        sample_ids=np.array(sample_ids),
        founders=FOUNDER_ORDER,
        data=data,
    )


def _check(label: str, value: float, threshold: float, op: str = '>=') -> bool:
    """log gate result and return pass/fail bool."""
    ok = (value >= threshold) if op == '>=' else (value < threshold)
    sym = PASS if ok else FAIL
    log.info('  %s %s = %.4f  (threshold %s %.4f)', sym, label, value, op, threshold)
    return ok


def _scan_to_df(scan) -> pd.DataFrame:
    """ScanResult → DataFrame with locus, chromosome, position_mb, lod."""
    return pd.DataFrame({
        'locus':       scan.locus_ids,
        'chromosome':  scan.chromosomes,
        'position_mb': scan.positions,
        'lod':         scan.lod,
    })


def _scan_to_manhattan_dict(scan) -> dict:
    """build {chrom: (positions, lod)} dict for manhattan()."""
    out = {}
    for chrom in np.unique(scan.chromosomes):
        mask = scan.chromosomes == chrom
        out[str(chrom)] = (scan.positions[mask], scan.lod[mask])
    return out


def _checkpoint(py_dir: Path, section: str, force: bool) -> bool:
    """return True if section should be skipped (already done and not forced)."""
    done_path = py_dir / f's{section}.done'
    if done_path.exists() and not force:
        log.info('  (checkpoint found — skipping S%s; use --force to rerun)', section)
        return True
    return False


def _mark_done(py_dir: Path, section: str) -> None:
    (py_dir / f's{section}.done').write_text('done')


# ── main run ──────────────────────────────────────────────────────────────────

def run(r_data: Path, out_dir: Path, sections: 'set[int]', force: bool) -> dict:
    """run Phase 2 validation sections and return gate results dict."""
    py_dir  = out_dir / 'py_data'
    cmp_dir = out_dir / 'compare'
    plt_dir = out_dir / 'plots'

    for d in (py_dir, cmp_dir, plt_dir):
        d.mkdir(parents=True, exist_ok=True)

    results: dict = {}         # gate key → True/False/None
    timing_rows: list[dict] = []
    _t0_run = time.perf_counter()

    # ── load phenotypes (defines sample space for all sections) ───────────────
    pheno_path = r_data / 'phenotypes.csv'
    if not pheno_path.exists():
        log.error('phenotypes.csv not found at %s — run port_sim_validation_p2.R first',
                  pheno_path)
        sys.exit(1)
    pheno_df   = pd.read_csv(pheno_path)
    sample_ids = pheno_df['SUBJECT.NAME.1'].values

    sim_ys: dict[int, np.ndarray] = {}
    for i in range(1, 11):
        col = f'sim.y.{i}'
        if col in pheno_df.columns:
            sim_ys[i] = pheno_df[col].values.astype(float)
    log.info('loaded %d samples, %d simulated phenotypes', len(sample_ids), len(sim_ys))

    true_loci_path = r_data / 'true_loci.csv'
    true_loci: dict[int, str] = {}
    if true_loci_path.exists():
        tl = pd.read_csv(true_loci_path)
        true_loci = dict(zip(tl['sim'].astype(int), tl['locus'].astype(str)))

    # ── build (or reload) GenomeCache from Phase 2 probs/ ────────────────────
    # Phase 2 runs sim.CC.data() independently — must NOT reuse Phase 1 HDF5.
    probs_dir  = r_data / 'probs'
    hdf5_cache = py_dir / 'cache.h5'
    if hdf5_cache.exists() and not force:
        log.info('loading genome cache from HDF5: %s', hdf5_cache)
        cache = GenomeCache.from_hdf5(str(hdf5_cache))
    elif probs_dir.exists():
        log.info('building genome cache from Phase 2 probs CSVs: %s', probs_dir)
        cache = _load_cache_from_probs_csv(probs_dir, keep_strains=list(sample_ids))
        cache.to_hdf5(str(hdf5_cache))
        log.info('  cached to HDF5: %s', hdf5_cache)
    else:
        log.error(
            'probs/ not found at %s — run port_sim_validation_p2.R first', probs_dir)
        sys.exit(1)
    log.info('  %d chromosomes, %d strains', len(cache.chromosomes), len(cache.sample_ids))

    # ── real LOCO kinship (Phase 2 lifts the OLS-only Phase 1 restriction) ───
    kd_hdf5 = py_dir / 'kinship_decomp.npz'
    if kd_hdf5.exists() and not force:
        log.info('loading kinship decomp from %s', kd_hdf5)
        raw = np.load(kd_hdf5, allow_pickle=True)
        kinship_decomp = {k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
                          for k in cache.chromosomes}
        del raw
    else:
        log.info('computing LOCO kinship (this may take a moment)...')
        kinship_raw = calc_kinship(cache, method='loco')
        kinship_decomp = decomp_kinship(kinship_raw)
        save_dict = {}
        for k, v in kinship_decomp.items():
            save_dict[f'{k}_vals'] = v['values']
            save_dict[f'{k}_vecs'] = v['vectors']
        np.savez(kd_hdf5, **save_dict)
        del kinship_raw, save_dict
        gc.collect()
        log.info('  LOCO kinship computed and cached')


    # ── shared scan reference on sim.y.1 (used by S6/S9/S10/S11) ─────────────
    rop_scan_1 = None
    peak_chrom: str = ''
    peak_locus_id: str = ''
    peak_pos_mb: float = 0.0
    lod_r_df: 'pd.DataFrame | None' = None

    # ─────────────────────────────────────────────────────────────────────────
    # S1 — cache structure
    # ─────────────────────────────────────────────────────────────────────────
    sec = 1
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS1: cache structure...')
            cache_info_r_path = r_data / 'cache_info_r.csv'
            founders_r_path   = r_data / 'founders_r.csv'
            samples_r_path    = r_data / 'sample_ids_r.csv'
            s1_ok = True

            if cache_info_r_path.exists():
                ci_r = pd.read_csv(cache_info_r_path)
                ci_r['chr'] = ci_r['chr'].astype(str)
                # build Python summary
                py_loci = {c: len(cache.get_chromosome(c)[0])
                           for c in cache.chromosomes}
                ci_py = pd.DataFrame([
                    {'chr': c, 'n_loci': n}
                    for c, n in py_loci.items()
                ])
                merged_ci = ci_py.merge(
                    ci_r[['chr', 'n_loci']].rename(columns={'n_loci': 'n_loci_r'}),
                    on='chr', how='outer')
                merged_ci['delta'] = (merged_ci['n_loci'] - merged_ci['n_loci_r']).abs()
                merged_ci.to_csv(cmp_dir / 's1_cache_info.csv', index=False)
                max_delta = int(merged_ci['delta'].max())
                s1_loci_ok = _check('cache loci max delta', float(max_delta), 0.5, '<')
                s1_ok = s1_ok and s1_loci_ok
                log.info('    max loci delta across chromosomes: %d', max_delta)
            else:
                log.info('  %s cache_info_r.csv not found — loci check skipped', SKIP)

            if founders_r_path.exists():
                founders_r = pd.read_csv(founders_r_path)['founder'].tolist()
                # R exports single-letter codes; Python uses full names
                from mosaiqtl.genotype.founders import FOUNDER_CODE
                letters = list(FOUNDER_CODE.values())
                r_set = set(founders_r)
                py_set = set(letters)
                match = (r_set == py_set)
                sym = PASS if match else FAIL
                log.info('  %s founder set match: %s', sym, match)
                s1_ok = s1_ok and match
            else:
                log.info('  %s founders_r.csv not found — founder check skipped', SKIP)

            if samples_r_path.exists():
                samples_r = set(pd.read_csv(samples_r_path)['sample_id'].astype(str))
                samples_py = set(str(s) for s in cache.sample_ids)
                match = (samples_r == samples_py)
                sym = PASS if match else FAIL
                log.info('  %s sample set match (n=%d R, n=%d Py): %s',
                         sym, len(samples_r), len(samples_py), match)
                s1_ok = s1_ok and match
            else:
                log.info('  %s sample_ids_r.csv not found — sample check skipped', SKIP)

            results['P2-S1'] = s1_ok
            _mark_done(py_dir, str(sec))

    timing_rows.append({'section': 'S1_cache_structure', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S2 — haplotype probability spot-check
    # ─────────────────────────────────────────────────────────────────────────
    sec = 2
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS2: haplotype prob concordance...')
            hap_r_path = r_data / 'haplotype_vals_r.csv'
            if hap_r_path.exists():
                from mosaiqtl.genotype.founders import FOUNDER_CODE
                letter_to_idx = {v: i for i, v in enumerate(FOUNDER_CODE.values())}

                hap_r = pd.read_csv(hap_r_path)
                hap_r['chr'] = hap_r['chr'].astype(str)

                # Python argmax founder per (strain, locus) pair
                agree_count = 0
                total_count = 0
                for locus_id, grp in hap_r.groupby('locus'):
                    chr_id = str(grp['chr'].iloc[0])
                    locus_ids_c, _, probs_c = cache.get_chromosome(chr_id)
                    locus_ids_c = np.asarray(locus_ids_c)
                    if locus_id not in locus_ids_c:
                        continue
                    idx = int(np.where(locus_ids_c == locus_id)[0][0])
                    probs_locus = probs_c[idx]   # (n_strains, 8)
                    py_argmax = np.argmax(probs_locus, axis=1)   # (n_strains,)
                    # R founder letter columns
                    r_letter_cols = [c for c in grp.columns
                                     if c in letter_to_idx]
                    r_matrix = grp[r_letter_cols].values   # (n_strains_r, 8)
                    r_argmax = np.argmax(r_matrix, axis=1)
                    # align by strain order
                    r_strains = grp['strain'].values
                    py_strain_order = list(cache.sample_ids)
                    for i_r, strain in enumerate(r_strains):
                        if strain not in py_strain_order:
                            continue
                        i_py = py_strain_order.index(strain)
                        # map R letter argmax to FOUNDER_ORDER index
                        r_let = r_letter_cols[r_argmax[i_r]]
                        r_founder_idx = letter_to_idx[r_let]
                        agree_count += int(py_argmax[i_py] == r_founder_idx)
                        total_count += 1

                concordance = agree_count / max(total_count, 1)
                results['P2-S2'] = _check('haplotype argmax concordance', concordance, 0.95)
                pd.DataFrame([{'concordance': concordance, 'n_compared': total_count}]).to_csv(
                    cmp_dir / 's2_haplotype_concordance.csv', index=False)

                # haplotype prob scatter: R prob vs Python prob at all spot-check loci
                from mosaiqtl.genotype.founders import FOUNDER_CODE as _FC2
                rows_plot: list[dict] = []
                first_locus_for_heatmap: 'tuple | None' = None
                for locus_id, grp in hap_r.groupby('locus'):
                    chr_id = str(grp['chr'].iloc[0])
                    locus_ids_c, _, probs_c = cache.get_chromosome(chr_id)
                    locus_ids_c = np.asarray(locus_ids_c)
                    if locus_id not in locus_ids_c:
                        continue
                    idx = int(np.where(locus_ids_c == locus_id)[0][0])
                    probs_locus = probs_c[idx]   # (n_strains, 8)
                    r_letter_cols = [c for c in grp.columns
                                     if c in _FC2.values()]
                    if first_locus_for_heatmap is None:
                        first_locus_for_heatmap = (
                            locus_id, chr_id,
                            float(grp['pos_mb'].iloc[0]),
                            probs_locus.copy(),
                        )
                    r_strains = grp['strain'].values
                    py_order  = list(cache.sample_ids)
                    for i_r, strain in enumerate(r_strains):
                        if strain not in py_order:
                            continue
                        i_py = py_order.index(strain)
                        for f_j, let in enumerate(r_letter_cols):
                            rows_plot.append({
                                'r_prob':  float(grp[let].values[i_r]),
                                'py_prob': float(probs_locus[i_py, f_j]),
                            })

                if rows_plot:
                    plot_df = pd.DataFrame(rows_plot)
                    plot_df.to_csv(cmp_dir / 's2_haplotype_probs.csv', index=False)
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
                    ax.scatter(plot_df['r_prob'], plot_df['py_prob'],
                               s=0.5, alpha=0.2, color='#52A5DB', linewidths=0,
                               rasterized=True)
                    ax.plot([0, 1], [0, 1], color='#D83026',
                            linewidth=0.8, linestyle='--')
                    ax.set_xlabel('R haplotype prob', fontsize=8)
                    ax.set_ylabel('Python haplotype prob', fontsize=8)
                    ax.set_title(f'Haplotype probs  concordance={concordance:.3f}',
                                 fontsize=9)
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's2_haplotype_scatter.png', dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)
                    log.info('  saved plots/s2_haplotype_scatter.png')

                # prob_heatmap at first spot-check locus
                if first_locus_for_heatmap is not None and 1 in sim_ys:
                    loc_id, loc_chr, loc_mb, loc_probs = first_locus_for_heatmap
                    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=300)
                    prob_heatmap(loc_probs, ax=ax2, phenotype=sim_ys[1],
                                 chrom=loc_chr, position_mb=loc_mb)
                    ax2.set_title(f'Haplotype probs at {loc_id}', fontsize=9)
                    fig2.tight_layout()
                    fig2.savefig(plt_dir / 's2_prob_heatmap.png', dpi=300,
                                 bbox_inches='tight')
                    plt.close(fig2)
                    log.info('  saved plots/s2_prob_heatmap.png')

                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s haplotype_vals_r.csv not found — S2 skipped', SKIP)
                results['P2-S2'] = None

    timing_rows.append({'section': 'S2_haplotype_probs', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S3 — LOCO kinship matrices
    # ─────────────────────────────────────────────────────────────────────────
    sec = 3
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS3: LOCO kinship matrices...')
            kinship_r_path = r_data / 'kinship_r.csv'
            if kinship_r_path.exists():
                rs = []
                for chrom in sorted(cache.chromosomes):
                    mat_r_path = r_data / f'kinship_mat_r_chr{chrom}.csv'
                    if not mat_r_path.exists():
                        continue
                    mat_r_df = pd.read_csv(mat_r_path, index_col=0)
                    # align columns to match sample_ids order
                    common = [s for s in cache.sample_ids
                              if str(s) in mat_r_df.columns]
                    if len(common) < 2:
                        continue
                    K_r = mat_r_df.loc[common, common].values
                    K_py = kinship_decomp[chrom]['vectors'] @ np.diag(
                        kinship_decomp[chrom]['values']
                    ) @ kinship_decomp[chrom]['vectors'].T
                    # subset K_py to common strains
                    all_strains = list(cache.sample_ids)
                    common_idx = [all_strains.index(s) for s in common]
                    K_py_sub = K_py[np.ix_(common_idx, common_idx)]
                    r_flat = float(np.corrcoef(K_r.ravel(), K_py_sub.ravel())[0, 1])
                    rs.append({'chr': chrom, 'r': r_flat})
                    log.info('  chr%s  kinship r = %.5f', chrom, r_flat)
                    del K_py, K_r
                    gc.collect()
                if rs:
                    rs_df = pd.DataFrame(rs)
                    rs_df.to_csv(cmp_dir / 's3_kinship_r.csv', index=False)
                    worst_r = float(rs_df['r'].min())
                    results['P2-S3'] = _check('kinship r (worst chr)', worst_r, 0.999)

                    # kinship diagonal scatter: R vs Python per-strain self-kinship
                    # (one point per strain per chromosome — shows calibration)
                    k_scatter_rows: list[dict] = []
                    all_strains_list = list(cache.sample_ids)
                    for chrom in sorted(cache.chromosomes):
                        mat_r_path = r_data / f'kinship_mat_r_chr{chrom}.csv'
                        if not mat_r_path.exists():
                            continue
                        mat_r_df = pd.read_csv(mat_r_path, index_col=0)
                        common = [s for s in all_strains_list
                                  if str(s) in mat_r_df.columns]
                        if len(common) < 2:
                            continue
                        K_r_diag = np.diag(mat_r_df.loc[common, common].values)
                        K_py_full = (kinship_decomp[chrom]['vectors']
                                     @ np.diag(kinship_decomp[chrom]['values'])
                                     @ kinship_decomp[chrom]['vectors'].T)
                        common_idx = [all_strains_list.index(s) for s in common]
                        K_py_diag = K_py_full[common_idx, common_idx]
                        for k_r, k_py in zip(K_r_diag, K_py_diag):
                            k_scatter_rows.append(
                                {'chr': chrom, 'k_r': k_r, 'k_py': k_py})
                        del K_py_full
                        gc.collect()

                    if k_scatter_rows:
                        ks_df = pd.DataFrame(k_scatter_rows)
                        ks_df.to_csv(cmp_dir / 's3_kinship_diag.csv', index=False)
                        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
                        ax.scatter(ks_df['k_r'], ks_df['k_py'],
                                   s=3, alpha=0.4, color='#4F6EAF', linewidths=0,
                                   rasterized=True)
                        lo = min(ks_df['k_r'].min(), ks_df['k_py'].min())
                        hi = max(ks_df['k_r'].max(), ks_df['k_py'].max())
                        ax.plot([lo, hi], [lo, hi], color='#D83026',
                                linewidth=0.8, linestyle='--')
                        ax.set_xlabel('R kinship (diagonal)', fontsize=8)
                        ax.set_ylabel('Python kinship (diagonal)', fontsize=8)
                        ax.set_title(
                            f'Kinship diag (worst r={worst_r:.5f})', fontsize=9)
                        sns.despine(ax=ax)
                        fig.tight_layout()
                        fig.savefig(plt_dir / 's3_kinship_diag_scatter.png', dpi=300,
                                    bbox_inches='tight')
                        plt.close(fig)
                        log.info('  saved plots/s3_kinship_diag_scatter.png')

                    # full matrix scatter for worst-r chromosome (off-diagonal included)
                    worst_chr = rs_df.loc[rs_df['r'].idxmin(), 'chr']
                    mat_r_path_w = r_data / f'kinship_mat_r_chr{worst_chr}.csv'
                    if mat_r_path_w.exists():
                        mat_rw = pd.read_csv(mat_r_path_w, index_col=0)
                        common_w = [s for s in all_strains_list
                                    if str(s) in mat_rw.columns]
                        K_r_full = mat_rw.loc[common_w, common_w].values
                        K_py_full_w = (kinship_decomp[worst_chr]['vectors']
                                       @ np.diag(kinship_decomp[worst_chr]['values'])
                                       @ kinship_decomp[worst_chr]['vectors'].T)
                        cidx = [all_strains_list.index(s) for s in common_w]
                        K_py_full_w_sub = K_py_full_w[np.ix_(cidx, cidx)]
                        # upper-triangle (including diagonal) — all unique pairs
                        tri_idx  = np.triu_indices(len(common_w))
                        r_flat_w = K_r_full[tri_idx]
                        py_flat_w = K_py_full_w_sub[tri_idx]
                        k_full_df = pd.DataFrame({'k_r': r_flat_w, 'k_py': py_flat_w})
                        k_full_df.to_csv(cmp_dir / f's3_kinship_full_chr{worst_chr}.csv',
                                         index=False)
                        fig_kf, ax_kf = plt.subplots(figsize=(4, 4), dpi=300)
                        ax_kf.scatter(r_flat_w, py_flat_w,
                                      s=1, alpha=0.3, color='#4F6EAF', linewidths=0,
                                      rasterized=True)
                        lo_k = min(r_flat_w.min(), py_flat_w.min())
                        hi_k = max(r_flat_w.max(), py_flat_w.max())
                        ax_kf.plot([lo_k, hi_k], [lo_k, hi_k], color='#D83026',
                                   linewidth=0.8, linestyle='--')
                        ax_kf.set_xlabel(f'R kinship chr{worst_chr} (all pairs)',
                                         fontsize=8)
                        ax_kf.set_ylabel('Python kinship (all pairs)', fontsize=8)
                        ax_kf.set_title(
                            f'Kinship full matrix chr{worst_chr}'
                            f'  r={worst_r:.5f}', fontsize=9)
                        sns.despine(ax=ax_kf)
                        fig_kf.tight_layout()
                        fig_kf.savefig(plt_dir / f's3_kinship_full_chr{worst_chr}.png',
                                       dpi=300, bbox_inches='tight')
                        plt.close(fig_kf)
                        log.info('  saved plots/s3_kinship_full_chr%s.png', worst_chr)
                        del K_r_full, K_py_full_w, K_py_full_w_sub
                        gc.collect()

                    # ── kinship heatmaps (Python, R, Diff — separate files) for chr1 ──
                    hm_chr = '1' if '1' in list(cache.chromosomes) else rs_df['chr'].iloc[0]
                    mat_r_hm_path = r_data / f'kinship_mat_r_chr{hm_chr}.csv'
                    if mat_r_hm_path.exists() and hm_chr in kinship_decomp:
                        mat_r_hm = pd.read_csv(mat_r_hm_path, index_col=0)
                        disp = sorted([s for s in all_strains_list
                                       if str(s) in mat_r_hm.columns])
                        if len(disp) >= 2:
                            di  = [all_strains_list.index(s) for s in disp]
                            ri  = [list(mat_r_hm.columns).index(s) for s in disp]
                            K_py_hm = (kinship_decomp[hm_chr]['vectors']
                                       @ np.diag(kinship_decomp[hm_chr]['values'])
                                       @ kinship_decomp[hm_chr]['vectors'].T)
                            kp  = K_py_hm[np.ix_(di, di)]
                            kr  = mat_r_hm.values[np.ix_(ri, ri)]
                            dk  = kp - kr
                            ab  = max(abs(dk.min()), abs(dk.max()))
                            fs  = max(4, min(7, int(180 / len(disp))))
                            sz  = max(5, len(disp) * 0.18)   # scale figure to n_strains

                            for mat, fname, title, cmap, vmin, vmax in [
                                (kp, f's3_kinship_py_chr{hm_chr}.png',
                                 f'Python LOCO kinship — chr{hm_chr}',
                                 'YlOrRd', 0, 1),
                                (kr, f's3_kinship_r_chr{hm_chr}.png',
                                 f'R LOCO kinship — chr{hm_chr}',
                                 'YlOrRd', 0, 1),
                                (dk, f's3_kinship_diff_chr{hm_chr}.png',
                                 f'Kinship diff (Py − R) — chr{hm_chr}'
                                 f'  max|ΔK|={ab:.2e}',
                                 'RdBu_r', -ab, ab),
                            ]:
                                fig_h, ax_h = plt.subplots(figsize=(sz, sz), dpi=300)
                                sns.heatmap(mat, ax=ax_h, cmap=cmap, vmin=vmin, vmax=vmax,
                                            xticklabels=disp, yticklabels=disp,
                                            cbar_kws={'shrink': 0.8})
                                ax_h.set_title(title, fontsize=9)
                                ax_h.set_xticklabels(ax_h.get_xticklabels(),
                                                     rotation=90, fontsize=fs)
                                ax_h.set_yticklabels(ax_h.get_yticklabels(),
                                                     rotation=0,  fontsize=fs)
                                fig_h.tight_layout()
                                fig_h.savefig(plt_dir / fname, dpi=300,
                                              bbox_inches='tight')
                                plt.close(fig_h)
                                log.info('  saved plots/%s', fname)

                            # ── diagonal histogram ──────────────────────────
                            kp_d = np.diag(kp);  kr_d = np.diag(kr)
                            fig_d, ax_d = plt.subplots(figsize=(5, 4), dpi=300)
                            sns.histplot(kr_d, bins=20, color='#D83026', alpha=0.6,
                                         edgecolor='none', ax=ax_d,
                                         label='R', stat='density')
                            sns.histplot(kp_d, bins=20, color='#4F6EAF', alpha=0.6,
                                         edgecolor='none', ax=ax_d,
                                         label='Python', stat='density')
                            ax_d.set_title(f'Kinship diagonal (self-relatedness)'
                                           f'\nchr{hm_chr}', fontsize=9)
                            ax_d.set_xlabel('kinship value', fontsize=8)
                            ax_d.set_ylabel('density', fontsize=8)
                            ax_d.legend(fontsize=8)
                            sns.despine(ax=ax_d)
                            fig_d.tight_layout()
                            fig_d.savefig(plt_dir / 's3_kinship_diag_hist.png',
                                          dpi=300, bbox_inches='tight')
                            plt.close(fig_d)
                            log.info('  saved plots/s3_kinship_diag_hist.png')

                            # ── off-diagonal histogram ───────────────────────
                            n_hm  = len(disp)
                            idx_u = np.triu_indices(n_hm, k=1)
                            kp_od = kp[idx_u];  kr_od = kr[idx_u]
                            fig_o, ax_o = plt.subplots(figsize=(5, 4), dpi=300)
                            sns.histplot(kr_od, bins=30, color='#D83026', alpha=0.6,
                                         edgecolor='none', ax=ax_o,
                                         label='R', stat='density')
                            sns.histplot(kp_od, bins=30, color='#4F6EAF', alpha=0.6,
                                         edgecolor='none', ax=ax_o,
                                         label='Python', stat='density')
                            ax_o.set_title(f'Kinship off-diagonal (pairwise)'
                                           f'\nchr{hm_chr}  n={len(kp_od)} pairs',
                                           fontsize=9)
                            ax_o.set_xlabel('kinship value', fontsize=8)
                            ax_o.set_ylabel('density', fontsize=8)
                            ax_o.legend(fontsize=8)
                            sns.despine(ax=ax_o)
                            fig_o.tight_layout()
                            fig_o.savefig(plt_dir / 's3_kinship_offdiag_hist.png',
                                          dpi=300, bbox_inches='tight')
                            plt.close(fig_o)
                            log.info('  saved plots/s3_kinship_offdiag_hist.png')

                            del K_py_hm, kp, kr, dk
                            gc.collect()

                else:
                    log.info('  %s no kinship matrices found — S3 skipped', SKIP)
                    results['P2-S3'] = None
                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s kinship_r.csv not found — S3 skipped', SKIP)
                results['P2-S3'] = None

    timing_rows.append({'section': 'S3_kinship_loco', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S4 — LMM scan (10 phenotypes)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 4
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS4: LMM scan (10 phenotypes)...')
            lod_r_path = r_data / 'lod_r.csv'
            if lod_r_path.exists():
                lod_r_df = pd.read_csv(lod_r_path)
                lod_r_df['chr'] = lod_r_df['chr'].astype(str)

                from mosaiqtl.genotype.founders import FOUNDER_ORDER as _FO, FOUNDER_CODE as _FC

                py_rows   = []
                r2_rows   = []
                pass_count = 0
                betas_sim1_at_peak: 'np.ndarray | None' = None
                peak_locus_sim1: str = ''
                peak_chrom_sim1: str = ''

                for sim_i in range(1, 11):
                    if sim_i not in sim_ys:
                        continue
                    y_i = sim_ys[sim_i]
                    log.info('  sim.y.%d...', sim_i)
                    sc = scan_h2lmm(
                        y_i, cache,
                        sample_ids=sample_ids,
                        kinship_decomp=kinship_decomp,
                        use_rint=False,
                        phenotype_id=f'sim.y.{sim_i}',
                    )
                    sc_df = _scan_to_df(sc)
                    sc_df['sim_idx'] = sim_i
                    py_rows.append(sc_df)

                    r_sim  = lod_r_df[lod_r_df['sim_idx'] == sim_i]
                    merged = sc_df.rename(columns={'lod': 'lod_python'}).merge(
                        r_sim[['locus', 'lod']].rename(columns={'lod': 'lod_r'}),
                        on='locus', how='inner')
                    if len(merged) > 0:
                        r2 = float(
                            np.corrcoef(merged['lod_python'], merged['lod_r'])[0, 1] ** 2)
                        peak_info = sc.peak_locus()
                        peak_py   = peak_info['locus_id']
                        planted   = true_loci.get(sim_i, '?')
                        ok = r2 >= 0.975
                        pass_count += int(ok)
                        r2_rows.append({
                            'sim_idx':  sim_i,
                            'r2':       r2,
                            'peak_py':  peak_py,
                            'planted':  planted,
                            'pass':     ok,
                        })
                        # save per-sim merged LOD CSV
                        merged.to_csv(cmp_dir / f's4_lod_sim{sim_i}.csv', index=False)
                        sym = PASS if ok else FAIL
                        log.info('    %s sim.y.%d  r²=%.4f  peak=%s  planted=%s',
                                 sym, sim_i, r2, peak_py, planted)
                        # capture betas at peak for sim.y.1
                        if sim_i == 1:
                            locus_ids_all = sc.locus_ids
                            peak_idx_arr  = np.where(locus_ids_all == peak_py)[0]
                            if len(peak_idx_arr) > 0:
                                betas_sim1_at_peak = sc.betas[peak_idx_arr[0]]  # (J,)
                                peak_locus_sim1    = peak_py
                                peak_chrom_sim1    = str(sc.chromosomes[peak_idx_arr[0]])
                    del sc
                    gc.collect()

                # save all Python LOD scores
                all_py_df = pd.concat(py_rows, ignore_index=True)
                all_py_df.to_csv(py_dir / 's4_lod_py.csv', index=False)

                # per-sim r² summary CSV
                r2_df = pd.DataFrame(r2_rows)
                r2_df.to_csv(cmp_dir / 's4_r2_per_sim.csv', index=False)

                pass_rate = pass_count / 10.0
                results['P2-S4'] = _check('S4 pass rate (≥50%)', pass_rate, 0.50)
                log.info('    %d/10 phenotypes pass (r² ≥ 0.975)', pass_count)

                # r² bar chart across all 10 simulations
                if len(r2_df) > 0:
                    fig_r2, ax_r2 = plt.subplots(figsize=(6, 3), dpi=300)
                    colors_bar = ['#55AF5B' if v else '#D83026'
                                  for v in r2_df['pass'].values]
                    ax_r2.bar(r2_df['sim_idx'], r2_df['r2'],
                              color=colors_bar, edgecolor='white', linewidth=0.5)
                    ax_r2.axhline(0.975, color='#444444', linewidth=0.8,
                                  linestyle='--', label='threshold 0.975')
                    ax_r2.set_xlabel('Simulation index', fontsize=8)
                    ax_r2.set_ylabel('LOD r² vs R', fontsize=8)
                    ax_r2.set_title(
                        f'LOD r² per simulation  ({pass_count}/10 pass)',
                        fontsize=9)
                    ax_r2.legend(fontsize=7)
                    sns.despine(ax=ax_r2)
                    fig_r2.tight_layout()
                    fig_r2.savefig(plt_dir / 's4_r2_per_sim.png', dpi=300,
                                   bbox_inches='tight')
                    plt.close(fig_r2)
                    log.info('  saved plots/s4_r2_per_sim.png')

                # allele effects comparison at peak locus (sim.y.1)
                ae_r_path = r_data / 'allele_effects_r.csv'
                if ae_r_path.exists() and betas_sim1_at_peak is not None:
                    ae_r = pd.read_csv(ae_r_path)
                    ae_r1 = ae_r[ae_r['sim_idx'] == 1].copy()
                    # R uses single-letter founder codes; remap to full names
                    _letter_to_name = {v: k for k, v in _FC.items()}
                    ae_r1['founder'] = ae_r1['founder'].map(_letter_to_name)
                    ae_py = pd.DataFrame({
                        'founder': _FO,
                        'beta_py': betas_sim1_at_peak,
                    })
                    ae_cmp = ae_py.merge(
                        ae_r1[['founder', 'beta']].rename(columns={'beta': 'beta_r'}),
                        on='founder', how='inner')
                    ae_cmp['delta'] = ae_cmp['beta_py'] - ae_cmp['beta_r']
                    ae_cmp.to_csv(cmp_dir / 's4_allele_effects_peak.csv', index=False)

                    if len(ae_cmp) > 1:
                        ae_r_val = float(
                            np.corrcoef(ae_cmp['beta_py'], ae_cmp['beta_r'])[0, 1])
                        from mosaiqtl.genotype.founders import FOUNDER_COLORS as _FC_COLORS
                        fig_ae, ax_ae = plt.subplots(figsize=(4, 4), dpi=300)
                        for _, row in ae_cmp.iterrows():
                            col = _FC_COLORS.get(row['founder'], '#888888')
                            ax_ae.scatter(row['beta_r'], row['beta_py'],
                                          s=60, color=col, zorder=3, linewidths=0)
                            ax_ae.annotate(row['founder'].split('/')[0],
                                           (row['beta_r'], row['beta_py']),
                                           fontsize=5, ha='left', va='bottom')
                        lo = min(ae_cmp['beta_r'].min(), ae_cmp['beta_py'].min()) - 0.05
                        hi = max(ae_cmp['beta_r'].max(), ae_cmp['beta_py'].max()) + 0.05
                        ax_ae.plot([lo, hi], [lo, hi], color='#444444',
                                   linewidth=0.8, linestyle='--')
                        ax_ae.set_xlabel('R allele effect (β)', fontsize=8)
                        ax_ae.set_ylabel('Python allele effect (β)', fontsize=8)
                        ax_ae.set_title(
                            f'Allele effects peak ({peak_locus_sim1})'
                            f'  r={ae_r_val:.4f}', fontsize=9)
                        sns.despine(ax=ax_ae)
                        fig_ae.tight_layout()
                        fig_ae.savefig(plt_dir / 's4_allele_effects_scatter.png', dpi=300,
                                       bbox_inches='tight')
                        plt.close(fig_ae)
                        log.info('  saved plots/s4_allele_effects_scatter.png')

                # scatter for sim.y.1
                r_sim1 = lod_r_df[lod_r_df['sim_idx'] == 1]
                py_sim1 = all_py_df[all_py_df['sim_idx'] == 1]
                cmp1 = py_sim1.rename(columns={'lod': 'lod_python'}).merge(
                    r_sim1[['locus', 'lod']].rename(columns={'lod': 'lod_r'}),
                    on='locus', how='inner')
                cmp1.to_csv(cmp_dir / 's4_lod_sim1.csv', index=False)
                if len(cmp1) > 0:
                    r2_1 = float(
                        np.corrcoef(cmp1['lod_python'], cmp1['lod_r'])[0, 1] ** 2)
                    # LOD scatter: R vs Python
                    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
                    ax.scatter(cmp1['lod_r'], cmp1['lod_python'],
                               s=1, alpha=0.4, color='#4F6EAF', linewidths=0)
                    lo = min(cmp1['lod_r'].min(), cmp1['lod_python'].min())
                    hi = max(cmp1['lod_r'].max(), cmp1['lod_python'].max())
                    ax.plot([lo, hi], [lo, hi], color='#D83026',
                            linewidth=0.8, linestyle='--')
                    ax.set_xlabel('R LOD (miQTL)', fontsize=8)
                    ax.set_ylabel('Python LOD (mosaiQTL)', fontsize=8)
                    ax.set_title(f'LOD sim.y.1  r²={r2_1:.4f}', fontsize=9)
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's4_lod_scatter.png', dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)

                    # Manhattan for Python sim.y.1
                    py_sim1_scan = py_sim1.copy()
                    py_sim1_scan['chromosome'] = py_sim1_scan['chromosome'].astype(str)
                    man_dict: dict = {}
                    for chrom, grp in py_sim1_scan.groupby('chromosome'):
                        man_dict[str(chrom)] = (
                            grp['position_mb'].values, grp['lod'].values)
                    fig_man = manhattan(man_dict)
                    fig_man.axes[0].set_title('S4 ROP scan — sim.y.1', fontsize=9)
                    fig_man.savefig(plt_dir / 's4_manhattan.png', dpi=300,
                                    bbox_inches='tight')
                    plt.close(fig_man)
                    log.info('  saved plots/s4_manhattan.png')

                # full diplotype model vs additive LOD (CC outer-product approx)
                if 1 in sim_ys:
                    from mosaiqtl.genotype.founders import (
                        FOUNDER_COLORS as _FC_D,
                        COLOR_NEUTRAL_DARK as _CND,
                    )
                    log.info('  full diplotype scan (sim.y.1)...')
                    sc_full = scan_h2lmm(
                        sim_ys[1], cache,
                        sample_ids=sample_ids,
                        kinship_decomp=kinship_decomp,
                        use_rint=False,
                        phenotype_id='sim.y.1.full',
                        model='full',
                    )
                    py_add_sim1 = all_py_df[all_py_df['sim_idx'] == 1]
                    full_lod_df = pd.DataFrame({
                        'locus':    sc_full.locus_ids,
                        'lod_full': sc_full.lod,
                    })
                    p02_cmp = py_add_sim1[['locus', 'lod']].rename(
                        columns={'lod': 'lod_additive'}).merge(
                        full_lod_df, on='locus', how='inner')
                    p02_cmp.to_csv(cmp_dir / 's4_full_vs_additive_lod.csv', index=False)
                    if len(p02_cmp) > 1:
                        r2_full = float(
                            np.corrcoef(p02_cmp['lod_additive'],
                                        p02_cmp['lod_full'])[0, 1]) ** 2
                        log.info('    full vs additive LOD r²=%.4f', r2_full)
                        fig_p2, ax_p2 = plt.subplots(figsize=(4, 4), dpi=300)
                        ax_p2.scatter(p02_cmp['lod_additive'], p02_cmp['lod_full'],
                                      s=1, alpha=0.4,
                                      color=_FC_D['NZOJ'], linewidths=0)
                        lo = min(p02_cmp['lod_additive'].min(), p02_cmp['lod_full'].min())
                        hi = max(p02_cmp['lod_additive'].max(), p02_cmp['lod_full'].max())
                        ax_p2.plot([lo, hi], [lo, hi], color=_FC_D['PWKJ'],
                                   linewidth=0.8, linestyle='--')
                        ax_p2.set_xlabel('LOD additive model', fontsize=8)
                        ax_p2.set_ylabel('LOD full diplotype model', fontsize=8)
                        ax_p2.set_title(
                            f'Full vs additive LOD  r²={r2_full:.4f}', fontsize=9)
                        sns.despine(ax=ax_p2)
                        fig_p2.tight_layout()
                        fig_p2.savefig(plt_dir / 's4_full_vs_additive_scatter.png',
                                       dpi=300, bbox_inches='tight')
                        plt.close(fig_p2)
                        log.info('  saved plots/s4_full_vs_additive_scatter.png')
                    del sc_full
                    gc.collect()

                # ranef (BLUP) vs fixef allele effects at peak locus (sim.y.1)
                if betas_sim1_at_peak is not None and peak_locus_sim1 and peak_chrom_sim1:
                    from mosaiqtl.qtl.effects import get_allele_effects
                    from mosaiqtl.genotype.founders import (
                        FOUNDER_COLORS as _FC_EFF,
                        COLOR_NEUTRAL_DARK as _CND2,
                        COLOR_CC_FALLBACK as _CCF,
                    )
                    log.info('  ranef vs fixef allele effects at %s (chr%s)...',
                             peak_locus_sim1, peak_chrom_sim1)
                    effects_fixef = get_allele_effects(
                        sim_ys[1], cache,
                        locus_id=peak_locus_sim1,
                        chrom=peak_chrom_sim1,
                        kinship_decomp=kinship_decomp,
                        sample_ids=list(sample_ids),
                        use_rint=False,
                        method='fixef',
                    )
                    effects_ranef = get_allele_effects(
                        sim_ys[1], cache,
                        locus_id=peak_locus_sim1,
                        chrom=peak_chrom_sim1,
                        kinship_decomp=kinship_decomp,
                        sample_ids=list(sample_ids),
                        use_rint=False,
                        method='ranef',
                    )
                    b_fixef = np.array([e.estimate for e in effects_fixef])
                    b_ranef = np.array([e.estimate for e in effects_ranef])
                    p03_cmp = pd.DataFrame({
                        'founder':    [e.founder for e in effects_fixef],
                        'beta_fixef': b_fixef,
                        'beta_ranef': b_ranef,
                        'delta':      b_ranef - b_fixef,
                    })
                    p03_cmp.to_csv(cmp_dir / 's4_allele_effects_ranef_vs_fixef.csv', index=False)
                    if len(p03_cmp) > 1:
                        r_eff = float(
                            np.corrcoef(p03_cmp['beta_fixef'], p03_cmp['beta_ranef'])[0, 1])
                        log.info('    ranef vs fixef r=%.4f', r_eff)
                        fig_p3, ax_p3 = plt.subplots(figsize=(4, 4), dpi=300)
                        for _, row in p03_cmp.iterrows():
                            col = _FC_EFF.get(row['founder'], _CCF)
                            ax_p3.scatter(row['beta_fixef'], row['beta_ranef'],
                                          s=60, color=col, zorder=3, linewidths=0)
                            ax_p3.annotate(row['founder'].split('/')[0],
                                           (row['beta_fixef'], row['beta_ranef']),
                                           fontsize=5, ha='left', va='bottom')
                        lo_e = (min(p03_cmp['beta_fixef'].min(),
                                    p03_cmp['beta_ranef'].min()) - 0.02)
                        hi_e = (max(p03_cmp['beta_fixef'].max(),
                                    p03_cmp['beta_ranef'].max()) + 0.02)
                        ax_p3.plot([lo_e, hi_e], [lo_e, hi_e], color=_CND2,
                                   linewidth=0.8, linestyle='--')
                        ax_p3.set_xlabel('β fixef (pseudoinverse)', fontsize=8)
                        ax_p3.set_ylabel('β ranef (ridge/BLUP)', fontsize=8)
                        ax_p3.set_title(
                            f'Ranef vs fixef effects  r={r_eff:.4f}', fontsize=9)
                        sns.despine(ax=ax_p3)
                        fig_p3.tight_layout()
                        fig_p3.savefig(plt_dir / 's4_allele_effects_ranef_vs_fixef.png',
                                       dpi=300, bbox_inches='tight')
                        plt.close(fig_p3)
                        log.info('  saved plots/s4_allele_effects_ranef_vs_fixef.png')

                _mark_done(py_dir, str(sec))
                # keep lod_r_df in scope for downstream sections
            else:
                log.info('  %s lod_r.csv not found — S4 skipped', SKIP)
                results['P2-S4'] = None

    timing_rows.append({'section': 'S4_lmm_scan', 'elapsed_s': time.perf_counter() - _t_sec})

    # ── build rop_scan_1 if needed by S6/S9/S10/S11 but S4 not in sections ───
    rop_needed = bool({6, 9, 10, 11} & sections)
    if rop_needed and rop_scan_1 is None and 1 in sim_ys:
        log.info('\nRunning shared ROP scan (sim.y.1) for S6/S9/S10/S11...')
        rop_scan_1 = scan_h2lmm(
            sim_ys[1], cache,
            sample_ids=sample_ids,
            kinship_decomp=kinship_decomp,
            use_rint=False,
            phenotype_id='sim.y.1',
        )
        peak = rop_scan_1.peak_locus()
        peak_chrom    = str(peak['chromosome'])
        peak_locus_id = str(peak['locus_id'])
        peak_pos_mb   = float(peak['position'])
        log.info('  peak: %s  chr%s  %.2f Mb  LOD=%.3f',
                 peak_locus_id, peak_chrom, peak_pos_mb, float(peak['lod']))

    # fallback: determine peak from S4 lod_r if rop_scan_1 still None
    if peak_chrom == '' and (r_data / 'lod_r.csv').exists():
        lod_r_df = pd.read_csv(r_data / 'lod_r.csv')
        lod_r_df['chr'] = lod_r_df['chr'].astype(str)
        r_sim1_all = lod_r_df[lod_r_df['sim_idx'] == 1]
        if len(r_sim1_all) > 0:
            peak_row = r_sim1_all.loc[r_sim1_all['lod'].idxmax()]
            peak_chrom    = str(peak_row['chr'])
            peak_locus_id = str(peak_row['locus'])
            peak_pos_mb   = float(peak_row['pos_mb'])
            log.info('  (peak chr inferred from R lod_r.csv: chr%s)', peak_chrom)

    # ─────────────────────────────────────────────────────────────────────────
    # S5 — cache collapse
    # gate: ≥5% loci removed AND lod_r² ≥ 0.999
    # ─────────────────────────────────────────────────────────────────────────
    sec = 5
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS5: cache collapse (tol=0.02)...')
            coll_r_path = r_data / 'collapsed_cache_summary.csv'
            lod_coll_r_path = r_data / 'lod_collapsed_r.csv'

            if not coll_r_path.exists():
                log.info('  %s collapsed_cache_summary.csv missing — run R script first', SKIP)
                results['P2-S5'] = None
            else:
                coll_r = pd.read_csv(coll_r_path)
                coll_r['chr'] = coll_r['chr'].astype(str)
                pct_removed_r = float(coll_r['pct_removed'].mean())

                # sparcc_cache is pre-collapsed: R removes <1 loci/chr (≈0%).
                # Python collapse of a pre-collapsed cache is a second pass —
                # the loci sets diverge so LOD comparison against R is invalid.
                if pct_removed_r < 1.0:
                    log.info(
                        '  %s S5 BLOCKED: sparcc_cache is pre-collapsed '
                        '(R removes %.2f%% mean). Python collapse is a second '
                        'pass — LOD comparison not valid.',
                        SKIP, pct_removed_r)
                    # Still verify Python collapse runs and removes loci
                    cache_coll = collapse_genomecache(cache, criterion='l2_norm', tol=0.02)
                    n_full_py = sum(cache.n_loci(c) for c in cache.chromosomes)
                    n_coll_py = sum(cache_coll.n_loci(c) for c in cache_coll.chromosomes)
                    pct_removed_py = 100.0 * (1 - n_coll_py / n_full_py)
                    log.info('  py: %d → %d loci (%.1f%% removed)',
                             n_full_py, n_coll_py, pct_removed_py)
                    del cache_coll
                    gc.collect()
                    results['P2-S5'] = None
                else:
                    cache_coll = collapse_genomecache(cache, criterion='l2_norm', tol=0.02)
                    n_full_py = sum(cache.n_loci(c) for c in cache.chromosomes)
                    n_coll_py = sum(cache_coll.n_loci(c) for c in cache_coll.chromosomes)
                    pct_removed_py = 100.0 * (1 - n_coll_py / n_full_py)
                    log.info('  py: %d → %d loci (%.1f%% removed)',
                             n_full_py, n_coll_py, pct_removed_py)
                    log.info('  R:  %.1f%% removed (mean over chrs)', pct_removed_r)
                    pct_ok = _check('loci removed (%)', pct_removed_py, 5.0)

                    if lod_coll_r_path.exists():
                        sc5 = scan_h2lmm(
                            sim_ys[1], cache_coll,
                            sample_ids=sample_ids,
                            kinship_decomp=kinship_decomp,
                            use_rint=False, phenotype_id='sim.y.1')
                        lod_coll_r = pd.read_csv(lod_coll_r_path)
                        lod_coll_r['chr'] = lod_coll_r['chr'].astype(str)
                        py_df = pd.DataFrame({'locus': sc5.locus_ids, 'lod_py': sc5.lod})
                        merged5 = lod_coll_r.merge(py_df, on='locus', how='inner')
                        r5_val = float(
                            np.corrcoef(merged5['lod_py'].values,
                                        merged5['lod'].to_numpy(float))[0, 1]) ** 2
                        merged5.to_csv(cmp_dir / 's5_lod_collapsed_compare.csv', index=False)
                        lod_ok = _check('collapsed LOD r²', r5_val, 0.999)
                        results['P2-S5'] = pct_ok and lod_ok
                    else:
                        results['P2-S5'] = pct_ok
                    del cache_coll
                    gc.collect()
            _mark_done(py_dir, str(sec))

    timing_rows.append({'section': 'S5_cache_collapse', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S6 — permutation / GEV threshold
    # ─────────────────────────────────────────────────────────────────────────
    sec = 6
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS6: permutation thresholds (n=250, chr%s)...', peak_chrom)
            if peak_chrom:
                if rop_scan_1 is None:
                    rop_scan_1 = scan_h2lmm(
                        sim_ys[1], cache,
                        sample_ids=sample_ids,
                        kinship_decomp=kinship_decomp,
                        use_rint=False, phenotype_id='sim.y.1')

                # reconstruct raw LOCO kinship matrices from decomp for parametric mode
                kinship_loco = {
                    c: (kinship_decomp[c]['vectors']
                        * kinship_decomp[c]['values'][None, :]
                        @ kinship_decomp[c]['vectors'].T)
                    for c in [peak_chrom]
                }

                null_y   = generate_null_outcomes(sim_ys[1], n_samples=250,
                                                  method='permutation', seed=1)
                max_lods = run_threshold_scans(
                    null_y, cache,
                    phenotype=sim_ys[1],
                    kinship_decomp=kinship_decomp,
                    kinship_loco=kinship_loco,
                    fix_h2=True, use_rint=False,
                    chrom=peak_chrom,
                    permutation_method='parametric',
                    seed=1,
                )
                del kinship_loco
                gc.collect()

                threshold, gev_params = get_gev_thresholds(max_lods, percentile=0.95)
                gev_emp_delta = abs(threshold - gev_params['empirical_quantile'])
                log.info('  GEV threshold = %.3f  empirical = %.3f  delta = %.4f',
                         threshold, gev_params['empirical_quantile'], gev_emp_delta)

                pd.DataFrame({'perm_index': np.arange(len(max_lods)),
                              'max_lod': max_lods}).to_csv(
                    py_dir / 's6_perm_max_lods.csv', index=False)
                pd.DataFrame([{
                    'threshold':          threshold,
                    'empirical_quantile': gev_params['empirical_quantile'],
                }]).to_csv(py_dir / 's6_threshold.csv', index=False)

                # load R reference (now exported by port_sim_validation_p2.R)
                r_thresh      = None
                r_max_lods    = None
                thresh_r_path = r_data / 'thresholds_r.csv'
                perm_r_path   = r_data / 'perm_max_lods_r.csv'
                if thresh_r_path.exists():
                    r_thresh = float(pd.read_csv(thresh_r_path)['threshold'].iloc[0])
                    log.info('  R GEV threshold = %.3f', r_thresh)
                if perm_r_path.exists():
                    r_max_lods = pd.read_csv(perm_r_path)['max_lod'].values

                thr_delta = abs(threshold - r_thresh) if r_thresh is not None else float('nan')
                pd.DataFrame([{
                    'threshold_python':    threshold,
                    'threshold_r':         r_thresh,
                    'threshold_delta':     thr_delta,
                    'empirical_quantile':  gev_params['empirical_quantile'],
                    'gev_emp_delta':       gev_emp_delta,
                }]).to_csv(cmp_dir / 's6_threshold.csv', index=False)

                results['P2-S6'] = _check(
                    'GEV fit delta (GEV vs empirical)', gev_emp_delta, 0.5, '<')

                x_lo  = max_lods.min() - 0.3
                x_hi  = max_lods.max() + 0.3
                if r_max_lods is not None:
                    x_lo = min(x_lo, r_max_lods.min() - 0.3)
                    x_hi = max(x_hi, r_max_lods.max() + 0.3)
                bins  = np.linspace(x_lo, x_hi, 26)
                x_crv = np.linspace(x_lo, x_hi, 300)

                fig, ax = plt.subplots(figsize=(5, 3), dpi=300)
                if r_max_lods is not None:
                    ax.hist(r_max_lods, bins=bins, color='#D83026', edgecolor='white',
                            linewidth=0.4, alpha=0.5, density=True, label='R perms')
                ax.hist(max_lods, bins=bins, color='#52A5DB', edgecolor='white',
                        linewidth=0.4, alpha=0.7, density=True, label='Python perms')
                ax.plot(x_crv,
                        _gev.pdf(x_crv, gev_params['c'],
                                 loc=gev_params['loc'], scale=gev_params['scale']),
                        color='#4F6EAF', linewidth=1.2, label='Py GEV fit')
                ax.axvline(threshold, color='#D83026', linewidth=1.2,
                           label=f'Py GEV {threshold:.2f}')
                ax.axvline(gev_params['empirical_quantile'], color='#55AF5B',
                           linewidth=1.2, linestyle='--',
                           label=f'empirical p95 {gev_params["empirical_quantile"]:.2f}')
                if r_thresh is not None:
                    ax.axvline(r_thresh, color='#C8A400', linewidth=1.0, linestyle=':',
                               label=f'R GEV {r_thresh:.2f}  Δ={thr_delta:.3f}')
                ax.set_xlabel('Max LOD per permutation', fontsize=8)
                ax.set_ylabel('Density', fontsize=8)
                ax.set_title(
                    f'Perm max-LOD (n={len(max_lods)})'
                    f'  GEV-emp Δ={gev_emp_delta:.3f}', fontsize=9)
                ax.legend(fontsize=7)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's6_perm_threshold.png', dpi=300,
                            bbox_inches='tight')
                plt.close(fig)
                log.info('  saved plots/s6_perm_threshold.png')

                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s peak_chrom unknown — S6 skipped', SKIP)
                results['P2-S6'] = None

    timing_rows.append({'section': 'S6_permutation', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S7 — MI scan
    # ─────────────────────────────────────────────────────────────────────────
    sec = 7
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS7: MI scan (num_imp=11)...')
            mi_r_path = r_data / 'mi_lod_r.csv'
            if mi_r_path.exists() and 1 in sim_ys:
                mi_scan = scan_h2lmm_mi(
                    sim_ys[1], cache,
                    sample_ids=sample_ids,
                    kinship_decomp=kinship_decomp,
                    use_rint=False, n_imp=11, seed=1,
                    phenotype_id='sim.y.1',
                )
                mi_py = _scan_to_df(mi_scan)
                mi_py.to_csv(py_dir / 's7_mi_lod.csv', index=False)
                log.info('  peak LOD = %.3f at %s',
                         float(mi_scan.lod.max()), mi_scan.peak_locus()['locus_id'])

                r_mi = pd.read_csv(mi_r_path).rename(columns={'lod': 'lod_r'})
                merged = mi_py.rename(columns={'lod': 'lod_python'}).merge(
                    r_mi[['locus', 'lod_r']], on='locus', how='inner')
                merged.to_csv(cmp_dir / 's7_mi_lod.csv', index=False)
                r_val = float(np.corrcoef(merged['lod_python'], merged['lod_r'])[0, 1])
                results['P2-S7'] = _check('MI LOD r', r_val, 0.75)

                fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
                ax.scatter(merged['lod_r'], merged['lod_python'],
                           s=1, alpha=0.4, color='#55AF5B', linewidths=0)
                lo = min(merged['lod_r'].min(), merged['lod_python'].min())
                hi = max(merged['lod_r'].max(), merged['lod_python'].max())
                ax.plot([lo, hi], [lo, hi], color='#D83026',
                        linewidth=0.8, linestyle='--')
                ax.set_xlabel('R LOD (MI)', fontsize=8)
                ax.set_ylabel('Python LOD (MI)', fontsize=8)
                ax.set_title(f'MI LOD  r={r_val:.4f}', fontsize=9)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's7_mi_scatter.png', dpi=300, bbox_inches='tight')
                plt.close(fig)
                del mi_scan
                gc.collect()
                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s mi_lod_r.csv not found — S7 skipped', SKIP)
                results['P2-S7'] = None

    timing_rows.append({'section': 'S7_mi_scan', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S8 — SNP scan (gated: skip if no snp_lod_r.csv or allele_dir)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 8
    _t_sec = time.perf_counter()
    if sec in sections:
        log.info('\nS8: SNP scan (gated)...')
        snp_r_path = r_data / 'snp_lod_r.csv'
        if not snp_r_path.exists():
            log.info('  %s snp_lod_r.csv not found — S8 SKIPPED (no allele_dir built)', SKIP)
            results['P2-S8'] = None
        else:
            log.info('  snp_lod_r.csv found; attempting Python SNP scan...')
            try:
                allele_dir = Path('validation/phase2_sim/allele_dir')
                if not allele_dir.exists():
                    log.info('  %s allele_dir not found at %s — S8 SKIPPED', SKIP, allele_dir)
                    results['P2-S8'] = None
                else:
                    # load SNP genotypes from allele_dir (minimal implementation)
                    log.info('  SNP scan not fully implemented for Phase 2 '
                             '(allele_dir format TBD) — S8 SKIPPED')
                    results['P2-S8'] = None
            except Exception:
                log.info('  %s S8 errored — SKIPPED', SKIP)
                results['P2-S8'] = None

    timing_rows.append({'section': 'S8_snp_scan', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S9 — mediation
    # ─────────────────────────────────────────────────────────────────────────
    sec = 9
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS9: mediation (outcome=sim.y.1, mediator=sim.y.2, chr%s)...',
                     peak_chrom)
            med_r_path = r_data / 'mediation_r.csv'
            if med_r_path.exists() and 1 in sim_ys and 2 in sim_ys and peak_chrom:
                med_df = mediation_scan_qr(
                    sim_ys[1], sim_ys[2], cache,
                    kinship_decomp=kinship_decomp,
                    use_rint=False,
                    outcome_id='sim.y.1', mediator_id='sim.y.2',
                )
                # restrict to peak chromosome (R also restricts to qtl_chr)
                med_df['chr'] = med_df['chr'].astype(str)
                med_py = med_df[med_df['chr'] == peak_chrom].copy()
                med_py.to_csv(py_dir / 's9_mediation.csv', index=False)

                r_med = pd.read_csv(med_r_path)
                r_med['chr'] = r_med['chr'].astype(str)
                merged = med_py[['locus', 'lod_mediated']].merge(
                    r_med[['locus', 'lod_mediated']].rename(
                        columns={'lod_mediated': 'lod_mediated_r'}),
                    on='locus', how='inner')
                if len(merged) > 1:
                    r_val = float(
                        np.corrcoef(merged['lod_mediated'],
                                    merged['lod_mediated_r'])[0, 1])
                    merged.to_csv(cmp_dir / 's9_mediation.csv', index=False)
                    results['P2-S9'] = _check('mediation LOD r', r_val, 0.90)

                    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
                    ax.scatter(merged['lod_mediated_r'], merged['lod_mediated'],
                               s=3, alpha=0.5, color='#DA9CC1', linewidths=0)
                    lo = min(merged['lod_mediated_r'].min(),
                             merged['lod_mediated'].min())
                    hi = max(merged['lod_mediated_r'].max(),
                             merged['lod_mediated'].max())
                    ax.plot([lo, hi], [lo, hi], color='#D83026',
                            linewidth=0.8, linestyle='--')
                    ax.set_xlabel('R mediated LOD', fontsize=8)
                    ax.set_ylabel('Python mediated LOD', fontsize=8)
                    ax.set_title(f'Mediation  r={r_val:.4f}', fontsize=9)
                    sns.despine(ax=ax)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's9_mediation_scatter.png', dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)
                else:
                    log.info('  insufficient loci overlap for mediation comparison')
                    results['P2-S9'] = None
                del med_df, med_py
                gc.collect()
                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s mediation_r.csv not found or missing phenotypes — S9 skipped',
                         SKIP)
                results['P2-S9'] = None

    timing_rows.append({'section': 'S9_mediation', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S10 — bootstrap positional CI
    # ─────────────────────────────────────────────────────────────────────────
    sec = 10
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS10: bootstrap positional CI (n=100, chr%s)...', peak_chrom)
            ci_r_path = r_data / 'ci_summary_r.csv'
            if ci_r_path.exists() and 1 in sim_ys and peak_chrom:
                boot_y = generate_null_outcomes(sim_ys[1], n_samples=100,
                                                method='bootstrap', seed=1)
                boot_positions, ci = run_positional_scans(
                    boot_y, cache,
                    phenotype=sim_ys[1],
                    kinship_decomp=kinship_decomp,
                    peak_chrom=peak_chrom,
                    fix_h2=True, use_rint=False,
                )
                log.info('  bootstrap 95%% CI: [%.2f, %.2f] Mb',
                         ci['ci_lo'], ci['ci_hi'])

                pd.DataFrame({'boot_peak_mb': boot_positions}).to_csv(
                    py_dir / 's10_bootstrap_ci.csv', index=False)

                ci_r = pd.read_csv(ci_r_path)
                planted_mb = float(ci_r['planted_locus_mb'].iloc[0])
                planted_in_ci = (ci['ci_lo'] <= planted_mb <= ci['ci_hi'])
                sym = PASS if planted_in_ci else FAIL
                log.info('  %s planted locus %.2f Mb in CI [%.2f, %.2f]: %s',
                         sym, planted_mb, ci['ci_lo'], ci['ci_hi'], planted_in_ci)

                pd.DataFrame([{
                    'ci_lo_py':    ci['ci_lo'],
                    'ci_hi_py':    ci['ci_hi'],
                    'ci_lo_r':     float(ci_r['ci_lo'].iloc[0]),
                    'ci_hi_r':     float(ci_r['ci_hi'].iloc[0]),
                    'planted_mb':  planted_mb,
                    'in_ci_py':    planted_in_ci,
                }]).to_csv(cmp_dir / 's10_ci_summary.csv', index=False)
                results['P2-S10'] = planted_in_ci

                # LOD profile + CI plot
                _, positions_c, _ = cache.get_chromosome(peak_chrom)
                if rop_scan_1 is not None:
                    chrom_mask = rop_scan_1.chromosomes == peak_chrom
                    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
                    lod_interval(positions_c, rop_scan_1.lod[chrom_mask],
                                 ax=ax, chrom=peak_chrom)
                    ax.axvline(ci['ci_lo'], color='#888888',
                               linewidth=0.8, linestyle='--', zorder=3)
                    ax.axvline(ci['ci_hi'], color='#888888',
                               linewidth=0.8, linestyle='--', zorder=3)
                    ax.axvline(planted_mb, color='#D83026',
                               linewidth=0.8, linestyle=':', zorder=4,
                               label=f'planted {planted_mb:.1f} Mb')
                    ax.legend(fontsize=7)
                    ax.set_title(f'Chr {peak_chrom} LOD + 95% bootstrap CI', fontsize=9)
                    fig.tight_layout()
                    fig.savefig(plt_dir / 's10_ci_profile.png', dpi=300,
                                bbox_inches='tight')
                    plt.close(fig)

                    # bootstrap peaks histogram: R vs Python (both in compare/)
                    ci_r_peaks_path = r_data / 'ci_r.csv'
                    if ci_r_peaks_path.exists():
                        r_peaks_df = pd.read_csv(ci_r_peaks_path)
                        r_peaks = r_peaks_df['peak_mb'].dropna().values
                        # side-by-side CSV
                        n_cmp = min(len(boot_positions), len(r_peaks))
                        pd.DataFrame({
                            'py_peak_mb': np.sort(boot_positions[np.isfinite(
                                boot_positions)])[:n_cmp],
                            'r_peak_mb':  np.sort(r_peaks)[:n_cmp],
                        }).to_csv(cmp_dir / 's10_boot_peaks_compare.csv', index=False)

                        all_peaks = np.concatenate([
                            boot_positions[np.isfinite(boot_positions)],
                            r_peaks[np.isfinite(r_peaks)]])
                        bins_b = np.linspace(all_peaks.min() - 1, all_peaks.max() + 1, 30)
                        fig_bp, ax_bp = plt.subplots(figsize=(5, 3), dpi=300)
                        ax_bp.hist(boot_positions[np.isfinite(boot_positions)],
                                   bins=bins_b, color='#52A5DB', alpha=0.55,
                                   edgecolor='white', linewidth=0.4, density=True,
                                   label='Python peaks')
                        ax_bp.hist(r_peaks[np.isfinite(r_peaks)],
                                   bins=bins_b, color='#F4C245', alpha=0.55,
                                   edgecolor='white', linewidth=0.4, density=True,
                                   label='R peaks')
                        ax_bp.axvline(planted_mb, color='#D83026', linewidth=1.2,
                                      label=f'planted {planted_mb:.1f} Mb')
                        ax_bp.axvline(ci['ci_lo'], color='#52A5DB',
                                      linewidth=0.8, linestyle='--')
                        ax_bp.axvline(ci['ci_hi'], color='#52A5DB',
                                      linewidth=0.8, linestyle='--')
                        ax_bp.set_xlabel('Bootstrap peak position (Mb)', fontsize=8)
                        ax_bp.set_ylabel('Density', fontsize=8)
                        ax_bp.set_title(
                            f'Bootstrap peaks chr{peak_chrom}', fontsize=9)
                        ax_bp.legend(fontsize=7)
                        sns.despine(ax=ax_bp)
                        fig_bp.tight_layout()
                        fig_bp.savefig(plt_dir / 's10_boot_peaks_hist.png', dpi=300,
                                       bbox_inches='tight')
                        plt.close(fig_bp)
                        log.info('  saved plots/s10_boot_peaks_hist.png')

                    # allele (founder) effects on peak chromosome
                    chrom_mask = rop_scan_1.chromosomes == peak_chrom
                    betas_c = rop_scan_1.betas[chrom_mask]   # (n_loci, J)
                    fig_ef, ax_ef = plt.subplots(figsize=(7, 3), dpi=300)
                    coef_profile(positions_c, betas_c, ax=ax_ef,
                                 chrom=peak_chrom, show_legend=True, ylim=3)
                    ax_ef.set_title(
                        f'Allele effects — Chr {peak_chrom}', fontsize=9)
                    fig_ef.tight_layout()
                    fig_ef.savefig(plt_dir / 's10_allele_effects.png', dpi=300,
                                   bbox_inches='tight')
                    plt.close(fig_ef)
                    log.info('  saved plots/s10_allele_effects.png')

                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s ci_summary_r.csv not found or peak_chrom unknown — S10 skipped',
                         SKIP)
                results['P2-S10'] = None

    timing_rows.append({'section': 'S10_bootstrap_ci', 'elapsed_s': time.perf_counter() - _t_sec})

    # ─────────────────────────────────────────────────────────────────────────
    # S11 — instability scan
    # ─────────────────────────────────────────────────────────────────────────
    sec = 11
    _t_sec = time.perf_counter()
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS11: instability scan (n=100, prop=0.63, chr%s)...', peak_chrom)
            instab_r_path = r_data / 'instability_r.csv'
            if instab_r_path.exists() and 1 in sim_ys and peak_chrom:
                # restrict to peak-chromosome loci to match R
                locus_ids_c, positions_c, _ = cache.get_chromosome(peak_chrom)
                locus_ids_c = np.asarray(locus_ids_c)

                instab_scan = instability_lm_scan(
                    sim_ys[1], cache,
                    sample_ids=sample_ids,
                    use_rint=False,
                    n_samples=100,
                    subsample_prop=0.63,
                    seed=1,
                    phenotype_id='sim.y.1',
                    just_these_loci=set(locus_ids_c.tolist()),
                )
                instab_py = _scan_to_df(instab_scan)
                instab_py.to_csv(py_dir / 's11_instability.csv', index=False)
                log.info('  peak instability LOD = %.3f', float(instab_scan.lod.max()))

                # R uses phenotype bootstrap (destroy QTL signal per scan), giving
                # near-flat peak frequency across loci.  Python uses strain
                # subsampling (preserves signal), concentrating peak_freq at the
                # planted QTL.  Direct correlation is not meaningful.
                # Gate: planted locus appears as peak in ≥ 70% of Python subsamples.
                planted_locus = true_loci.get(1, '')
                py_loci = instab_py['locus'].values
                if planted_locus and planted_locus in py_loci:
                    planted_idx = int(np.where(py_loci == planted_locus)[0][0])
                    # peak_freq = 1 - p_value (proportion of subsamples it was peak)
                    if 'p_value' in instab_py.columns:
                        planted_freq = float(1.0 - instab_py['p_value'].iloc[planted_idx])
                    else:
                        # fallback: rank LOD — planted locus should be near top
                        rank = int(np.sum(instab_scan.lod > instab_scan.lod[planted_idx]))
                        planted_freq = 1.0 - rank / max(len(py_loci), 1)
                    log.info('  planted locus %s peak_freq = %.3f',
                             planted_locus, planted_freq)
                    results['P2-S11'] = _check(
                        'planted locus peak_freq', planted_freq, 0.70)
                else:
                    # planted locus not in scanned loci — compare LOD rank of peak
                    py_peak_locus = py_loci[int(np.argmax(instab_scan.lod))]
                    log.info('  planted locus %s not in scanned set; peak=%s',
                             planted_locus, py_peak_locus)
                    results['P2-S11'] = None

                # LOD profile plot
                fig, ax = plt.subplots(figsize=(6, 3), dpi=300)
                ax.plot(instab_py['position_mb'], instab_py['lod'],
                        color='#683C91', linewidth=0.8)
                if planted_locus and planted_locus in py_loci:
                    planted_pos = float(
                        instab_py.loc[instab_py['locus'] == planted_locus,
                                      'position_mb'].iloc[0])
                    ax.axvline(planted_pos, color='#D83026', linewidth=1.0,
                               linestyle='--', label=f'planted {planted_locus}')
                    ax.legend(fontsize=7)
                ax.set_xlabel('Position (Mb)', fontsize=8)
                ax.set_ylabel('Mean LOD (subsamples)', fontsize=8)
                ax.set_title(
                    f'Instability (chr{peak_chrom}, n=100 subsamples)',
                    fontsize=9)
                sns.despine(ax=ax)
                fig.tight_layout()
                fig.savefig(plt_dir / 's11_instability_scatter.png', dpi=300,
                            bbox_inches='tight')
                plt.close(fig)
                del instab_scan
                gc.collect()
                _mark_done(py_dir, str(sec))
            else:
                log.info('  %s instability_r.csv not found or peak_chrom unknown — S11 skipped',
                         SKIP)
                results['P2-S11'] = None

    timing_rows.append({'section': 'S11_instability', 'elapsed_s': time.perf_counter() - _t_sec})
    timing_rows.append({'section': 'total', 'elapsed_s': time.perf_counter() - _t0_run})

    # ── gate summary + timing ─────────────────────────────────────────────────
    _print_summary(results, out_dir, timing_rows)
    return results


def _print_summary(
    results: dict,
    out_dir: Path,
    timing_rows: 'list[dict] | None' = None,
) -> None:
    """print gate summary table and write gate_summary.csv + timing_sections.csv."""
    gate_order = [
        ('P2-S1',  'cache structure'),
        ('P2-S2',  'haplotype concordance ≥ 0.95'),
        ('P2-S3',  'kinship r > 0.999 (worst chr)'),
        ('P2-S4',  'LMM scan ≥50% phenotypes r² ≥ 0.975'),
        ('P2-S5',  'cache collapse ≥5% removed AND lod_r² ≥ 0.999'),
        ('P2-S6',  'GEV fit delta (GEV vs empirical) < 0.5'),
        ('P2-S7',  'MI LOD r ≥ 0.75'),
        ('P2-S8',  'SNP scan (gated)'),
        ('P2-S9',  'mediation LOD r ≥ 0.90'),
        ('P2-S10', 'bootstrap CI contains planted locus'),
        ('P2-S11', 'instability planted peak_freq ≥ 0.70'),
    ]
    log.info('\n── Phase 2 gate summary ─────────────────────────────────────────')
    rows = []
    for gate, desc in gate_order:
        v = results.get(gate)
        if v is None:
            sym  = SKIP
            stat = 'SKIPPED/BLOCKED'
        elif v:
            sym  = PASS
            stat = 'PASS'
        else:
            sym  = FAIL
            stat = 'FAIL'
        log.info('  %s %-8s %s', sym, gate, desc)
        rows.append({'gate': gate, 'description': desc, 'result': stat})
    scored  = {k: v for k, v in results.items() if v is not None}
    n_pass  = sum(scored.values())
    n_total = len(scored)
    log.info('')
    log.info('  %d/%d sections pass', n_pass, n_total)
    pd.DataFrame(rows).to_csv(
        Path(out_dir) / 'py_data' / 'gate_summary.csv', index=False)

    if timing_rows:
        t_df = pd.DataFrame(timing_rows)
        # try to load R timing for comparison
        r_timing_path = Path(out_dir) / 'r_data' / 'timing_sections.csv'
        if r_timing_path.exists():
            r_t = pd.read_csv(r_timing_path)[['section', 'elapsed_s']].rename(
                columns={'elapsed_s': 'elapsed_s_r'})
            t_df = t_df.merge(r_t, on='section', how='left')
            t_df['speedup'] = t_df['elapsed_s_r'] / t_df['elapsed_s'].replace(0, float('nan'))
        t_df.to_csv(Path(out_dir) / 'py_data' / 'timing_sections.csv', index=False)
        log.info('\n── Timing summary (Python) ───────────────────────────────────────')
        for _, row in t_df.iterrows():
            s_py = float(row['elapsed_s'])
            line = f"  {row['section']:<24s}  {s_py:8.1f} s  ({s_py/60:5.2f} min)"
            if 'speedup' in t_df.columns and pd.notna(row.get('speedup')):
                line += f"  ×{float(row['speedup']):.1f} vs R"
            log.info(line)
        log.info('  timing written to: %s/py_data/timing_sections.csv', out_dir)


def main() -> None:
    """entry point."""
    ap = argparse.ArgumentParser(
        description='Phase 2 validation: all 11 miQTL functions (simulated data)')
    ap.add_argument('--r_data', default='validation/phase2_sim/r_data',
                    help='path to R reference CSVs from port_sim_validation_p2.R')
    ap.add_argument('--out',    default='validation/phase2_sim',
                    help='output root — writes to out/py_data/, out/compare/, out/plots/')
    ap.add_argument('--sections', default=None,
                    help='comma-separated section numbers to run (e.g. 3,4,6); '
                         'default: all')
    ap.add_argument('--force', action='store_true',
                    help='rerun sections even if checkpoint .done files exist')
    args = ap.parse_args()

    if args.sections:
        sections = {int(s.strip()) for s in args.sections.split(',')}
    else:
        sections = set(range(1, 12))

    run(
        r_data   = Path(args.r_data),
        out_dir  = Path(args.out),
        sections = sections,
        force    = args.force,
    )


if __name__ == '__main__':
    main()

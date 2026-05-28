"""
validate_phase3.py — Phase 3: speed + memory benchmarking (mosaiqtl vs miQTL and R/qtl2)

Builds its own GenomeCache from phase3/r_data/probs/ (exported by
port_sim_validation_p3.R) because sim.CC.data(seed=123) selects different
strains than Phase 2 (seed=42). The cache is saved to phase3/py_data/cache.h5
and reused on subsequent runs.
Loads Phase 3 phenotypes from phase3/r_data/phenotypes.csv (50 genes, seed=123,
independent from Phase 2).

Sections benchmarked: S1, S2, S3, S4, S6, S7, S8, S9, S10, S11 (no S5)
Each section is timed with perf_counter; S3 and S4 also tracked with
tracemalloc (Python-heap proxy — numpy C allocations not captured).

Outputs: validation/phase3/
  py_data/timing_sections.csv     wall-clock + peak tracemalloc MB per section
  py_data/benchmark_summary.csv   Python vs R speedup per section
  plots/s4_speedup_bar.png        speedup bar chart

Gates (logged, not hard-fail on first run — establishes baselines):
  SPEED-02  S4 scan speedup vs R  (target ≥ 5×)
  SPEED-06  S4 genes/min          (target > 100/min)
  MEM-02    S4 peak tracemalloc   (target < 4096 MB)

Usage:
  python validation/validate_phase3.py
  python validation/validate_phase3.py --sections 3,4
  python validation/validate_phase3.py --force
"""

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
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mosaiqtl.genotype.genomecache import GenomeCache
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

# reuse Phase 2's CSV→GenomeCache builder (same probs format)
sys.path.insert(0, str(Path(__file__).parent))
from validate_sim_p2 import _load_cache_from_probs_csv  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_phase3')

PASS = '✅'
FAIL = '❌'
SKIP = '⚠️ '
MEAS = '📏'


# ── timing + memory helpers ───────────────────────────────────────────────────

def _get_rss_mb() -> float:
    """Current process RSS in MB (macOS returns bytes; Linux returns kilobytes)."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    if sys.platform == 'darwin':
        return usage.ru_maxrss / 1024**2
    return usage.ru_maxrss / 1024


class _Timer:
    """context manager: records wall-clock elapsed, tracemalloc peak, and RSS delta."""

    def __init__(self) -> None:
        self.elapsed: float = 0.0
        self.peak_mb: float = 0.0
        self.peak_rss_mb: float = 0.0

    def __enter__(self) -> '_Timer':
        gc.collect()
        self._rss_start = _get_rss_mb()
        tracemalloc.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        tracemalloc.clear_traces()
        self.peak_mb = peak / 1024**2
        # ru_maxrss is a high-water mark; delta captures allocations made in this block
        self.peak_rss_mb = max(0.0, _get_rss_mb() - self._rss_start)


def _record(timing_rows: list, section: str, t: _Timer, note: str = '') -> None:
    timing_rows.append({
        'section':     section,
        'elapsed_s':   t.elapsed,
        'peak_mb':     t.peak_mb,
        'peak_rss_mb': t.peak_rss_mb,
        'note':        note,
    })
    log.info('  %-24s  %7.2f s  (%5.1f min)  RSS +%.1f MB  heap %.1f MB',
             section, t.elapsed, t.elapsed / 60, t.peak_rss_mb, t.peak_mb)


def _checkpoint(py_dir: Path, section: str, force: bool) -> bool:
    done = py_dir / f's{section}.done'
    if done.exists() and not force:
        log.info('  (checkpoint — skipping S%s; use --force to rerun)', section)
        return True
    return False


def _mark_done(py_dir: Path, section: str) -> None:
    (py_dir / f's{section}.done').write_text('done')


def _scan_to_df(scan) -> pd.DataFrame:
    """ScanResult → DataFrame with locus / chromosome / position_mb / lod.

    Matches Phase 2's _scan_to_df convention: column is 'locus' (same name as
    R's exported CSVs) so locus-name merges work without rename gymnastics.
    """
    return pd.DataFrame({
        'locus':       scan.locus_ids,
        'chromosome':  scan.chromosomes.astype(str),
        'position_mb': scan.positions,
        'lod':         scan.lod,
    })


# ── main ─────────────────────────────────────────────────────────────────────

def run(r_data: Path, out_dir: Path,
        sections: 'set[int]', force: bool) -> None:
    py_dir  = out_dir / 'py_data'
    plt_dir = out_dir / 'plots'
    cmp_dir = out_dir / 'compare'
    for d in (py_dir, plt_dir, cmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    timing_rows: list[dict] = []
    results: dict[str, object] = {}
    _t0_run = time.perf_counter()

    # ── load Phase 3 phenotypes (50 genes × 120 samples) ─────────────────────
    pheno_path = r_data / 'phenotypes.csv'
    if not pheno_path.exists():
        log.error('phenotypes.csv not found at %s — run port_sim_validation_p3.R first',
                  pheno_path)
        sys.exit(1)
    pheno_df   = pd.read_csv(pheno_path)
    sample_ids = pheno_df['SUBJECT.NAME.1'].values
    sim_ys: dict[int, np.ndarray] = {}
    for i in range(1, 51):
        col = f'sim.y.{i}'
        if col in pheno_df.columns:
            sim_ys[i] = pheno_df[col].values.astype(float)
    log.info('loaded %d samples, %d simulated phenotypes', len(sample_ids), len(sim_ys))

    true_loci: dict[int, str] = {}
    tl_path = r_data / 'true_loci.csv'
    if tl_path.exists():
        tl = pd.read_csv(tl_path)
        true_loci = dict(zip(tl['sim'].astype(int), tl['locus'].astype(str)))

    # ── build or load Phase 3 GenomeCache ────────────────────────────────────
    # Phase 3 sim.CC.data(seed=123) picks different strains than Phase 2 (seed=42)
    # so we build a Phase 3-specific cache from phase3/r_data/probs/.
    probs_dir = r_data / 'probs'
    hdf5_p3   = py_dir / 'cache.h5'
    if not force and hdf5_p3.exists():
        log.info('loading Phase 3 GenomeCache: %s', hdf5_p3)
        cache = GenomeCache.from_hdf5(str(hdf5_p3))
    elif probs_dir.exists():
        log.info('building Phase 3 GenomeCache from probs: %s', probs_dir)
        cache = _load_cache_from_probs_csv(probs_dir, keep_strains=list(sample_ids))
        cache.to_hdf5(str(hdf5_p3))
        log.info('  saved to %s', hdf5_p3)
    else:
        log.error(
            'probs/ not found at %s — run port_sim_validation_p3.R --force first',
            probs_dir)
        sys.exit(1)
    log.info('  %d chromosomes, %d strains', len(cache.chromosomes), len(cache.sample_ids))

    # ── compute LOCO kinship from scratch ────────────────────────────────────
    # No Phase 2 cached kinship used — full build here, timed as S3.
    kd_cache = py_dir / 'kinship_decomp_p3.npz'
    if not force and kd_cache.exists():
        log.info('loading cached Phase 3 kinship decomp: %s', kd_cache)
        raw = np.load(kd_cache, allow_pickle=True)
        kinship_decomp = {k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
                          for k in cache.chromosomes}
        del raw; gc.collect()
    else:
        log.info('computing LOCO kinship from scratch (will also be timed as S3)...')
        with _Timer() as _kd_timer:
            _kinship_raw = calc_kinship(cache, method='loco')
            kinship_decomp = decomp_kinship(_kinship_raw)
            del _kinship_raw; gc.collect()
        log.info('  kinship done in %.1f s', _kd_timer.elapsed)
        # cache for reruns where S3 is skipped
        arrays: dict[str, np.ndarray] = {}
        for chrom, kd in kinship_decomp.items():
            arrays[f'{chrom}_vals'] = kd['values']
            arrays[f'{chrom}_vecs'] = kd['vectors']
        np.savez(kd_cache, **arrays)
        del arrays; gc.collect()

    # peak chromosome for S6/S7/S9/S10/S11: from planted locus of sim.y.1
    planted_locus_1 = true_loci.get(1, '')
    peak_chrom: str = ''
    if planted_locus_1:
        for chrom in cache.chromosomes:
            lids, _, _ = cache.get_chromosome(chrom)
            if planted_locus_1 in np.asarray(lids):
                peak_chrom = chrom
                break
    if not peak_chrom:
        # fallback: most-represented chromosome
        peak_chrom = cache.chromosomes[len(cache.chromosomes) // 2]
        log.info('  planted locus not found; using fallback peak_chrom=%s', peak_chrom)
    else:
        log.info('  peak_chrom=%s  (planted locus %s)', peak_chrom, planted_locus_1)

    # ─────────────────────────────────────────────────────────────────────────
    # S1 — cache I/O: GenomeCache.from_hdf5 + spot-check locus reads
    # ─────────────────────────────────────────────────────────────────────────
    sec = 1
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS1: cache I/O (from_hdf5 + spot-check locus reads)...')
            with _Timer() as t:
                cache_bench = GenomeCache.from_hdf5(str(hdf5_p3))
                rng = np.random.default_rng(1)
                # read 50 random loci across all chromosomes
                all_lids = np.concatenate([
                    cache_bench.get_chromosome(c)[0] for c in cache_bench.chromosomes])
                sample_idx = rng.choice(len(all_lids), size=min(50, len(all_lids)),
                                        replace=False)
                for chr_id in cache_bench.chromosomes:
                    cache_bench.get_chromosome(chr_id)   # warm all chr blocks
                del cache_bench
            _record(timing_rows, 'S1_cache_info', t, '50 locus reads + full chr load')
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S2 — haplotype prob access (5 random loci per chromosome)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 2
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS2: haplotype prob access (spot-check)...')
            rng2 = np.random.default_rng(7)
            spot_loci: list[tuple[str, int]] = []
            for chrom in cache.chromosomes:
                lids, _, _ = cache.get_chromosome(chrom)
                n = len(lids)
                idxs = rng2.choice(n, size=min(5, n), replace=False)
                for idx in idxs:
                    spot_loci.append((chrom, int(idx)))

            with _Timer() as t:
                for chrom, idx in spot_loci:
                    _, _, probs = cache.get_chromosome(chrom)
                    _ = probs[idx]   # (n_strains, 8)
            _record(timing_rows, 'S2_haplotype_probs', t,
                    f'{len(spot_loci)} spot-check loci')
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S3 — LOCO kinship: full build from GenomeCache, result used in S4+
    # ─────────────────────────────────────────────────────────────────────────
    sec = 3
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS3: LOCO kinship build (all chromosomes)...')
            with _Timer() as t:
                _kinship_raw = calc_kinship(cache, method='loco')
                kinship_decomp = decomp_kinship(_kinship_raw)
                del _kinship_raw; gc.collect()
                # overwrite cache so S4+ uses the freshly-timed result
                arrays: dict[str, np.ndarray] = {}
                for chrom, kd in kinship_decomp.items():
                    arrays[f'{chrom}_vals'] = kd['values']
                    arrays[f'{chrom}_vecs'] = kd['vectors']
                np.savez(kd_cache, **arrays)
                del arrays; gc.collect()
            _record(timing_rows, 'S3_kinship_loco', t, 'full build; result used for S4+')
            results['MEM-02'] = max(float(results.get('MEM-02', 0)), t.peak_rss_mb)
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S4 — LMM genome-wide scan × 50 phenotypes
    # ─────────────────────────────────────────────────────────────────────────
    sec = 4
    rop_scan_1 = None   # built here, reused by S6/S9/S10/S11
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS4: LMM scan (50 phenotypes, genome-wide LOCO)...')
            with _Timer() as t:
                for sim_i in range(1, 51):
                    if sim_i not in sim_ys:
                        continue
                    sc = scan_h2lmm(
                        sim_ys[sim_i], cache,
                        sample_ids=sample_ids,
                        kinship_decomp=kinship_decomp,
                        use_rint=False,
                        phenotype_id=f'sim.y.{sim_i}',
                        scan_window='genome',
                    )
                    if sim_i == 1:
                        rop_scan_1 = sc
                    elif sim_i % 10 == 0:
                        peak = sc.peak_locus()
                        log.info('    sim.y.%d  peak LOD=%.2f  chr%s',
                                 sim_i, float(peak['lod']), peak['chromosome'])
                    if sim_i > 1:
                        del sc
                        gc.collect()

            n_genes = len(sim_ys)
            genes_min = n_genes / (t.elapsed / 60)
            log.info('  throughput: %.1f genes/min  (%.2f s/gene)',
                     genes_min, t.elapsed / n_genes)
            _record(timing_rows, 'S4_lmm_scan', t,
                    f'{n_genes} genes genome-wide')
            results['SPEED-06'] = genes_min
            results['MEM-02']   = t.peak_rss_mb
            if rop_scan_1 is not None:
                rop_scan_1.to_dataframe()[
                    ['locus_id', 'chromosome', 'position', 'lod']
                ].to_csv(py_dir / 's4_lod_py.csv', index=False)
                log.info('  saved s4_lod_py.csv')
            _mark_done(py_dir, str(sec))

    # build rop_scan_1 if S4 was skipped but downstream sections need it
    if rop_scan_1 is None and bool({6, 7, 9, 10, 11} & sections) and 1 in sim_ys:
        log.info('\nBuilding rop_scan_1 (sim.y.1) for downstream sections...')
        rop_scan_1 = scan_h2lmm(
            sim_ys[1], cache,
            sample_ids=sample_ids,
            kinship_decomp=kinship_decomp,
            use_rint=False,
            phenotype_id='sim.y.1',
            scan_window='genome',
        )
        peak = rop_scan_1.peak_locus()
        log.info('  peak: %s  chr%s  LOD=%.3f',
                 peak['locus_id'], peak['chromosome'], float(peak['lod']))

    # ─────────────────────────────────────────────────────────────────────────
    # S6 — permutation thresholds (n=250, genome-wide parametric)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 6
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS6: permutation thresholds (n=250, genome-wide)...')
            if rop_scan_1 is not None and 1 in sim_ys:
                kinship_loco_s6 = {
                    c: (kinship_decomp[c]['vectors']
                        * kinship_decomp[c]['values'][None, :]
                        @ kinship_decomp[c]['vectors'].T)
                    for c in kinship_decomp
                }
                with _Timer() as t:
                    null_y   = generate_null_outcomes(
                        sim_ys[1], n_perms=250, method='permutation', seed=1)
                    max_lods, _ = run_threshold_scans(
                        null_y, cache,
                        phenotype=sim_ys[1],
                        kinship_decomp=kinship_decomp,
                        kinship_loco=kinship_loco_s6,
                        fix_h2=True, use_rint=False,
                        permutation_method='parametric',
                        seed=1,
                    )
                    threshold, gev_params = get_gev_thresholds(max_lods, percentile=0.95)
                del kinship_loco_s6; gc.collect()
                log.info('  GEV threshold = %.3f  empirical = %.3f  delta = %.4f',
                         threshold, gev_params['empirical_quantile'],
                         abs(threshold - gev_params['empirical_quantile']))
                pd.DataFrame({'perm_index': range(len(max_lods)),
                              'max_lod': max_lods}).to_csv(
                    py_dir / 's6_perm_max_lods.csv', index=False)
                pd.DataFrame({'threshold': [threshold],
                              'empirical_quantile':
                                  [gev_params['empirical_quantile']]}).to_csv(
                    py_dir / 's6_threshold.csv', index=False)
                log.info('  saved s6_perm_max_lods.csv + s6_threshold.csv')
                _record(timing_rows, 'S6_permutation', t, 'n=250 genome-wide')
            else:
                log.info('  %s rop_scan_1 unavailable — S6 skipped', SKIP)
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S7 — MI scan (num_imp=11, peak chromosome)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 7
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS7: MI scan (num_imp=11, chr%s)...', peak_chrom)
            if 1 in sim_ys and peak_chrom:
                with _Timer() as t:
                    mi_scan = scan_h2lmm_mi(
                        sim_ys[1], cache,
                        sample_ids=sample_ids,
                        kinship_decomp=kinship_decomp,
                        n_imp=11,
                        use_rint=False,
                        seed=1,
                        scan_window='genome',
                    )
                log.info('  MI peak LOD = %.3f', float(mi_scan.lod.max()))
                mi_scan.to_dataframe()[
                    ['locus_id', 'chromosome', 'position', 'lod']
                ].to_csv(py_dir / 's7_mi_lod.csv', index=False)
                log.info('  saved s7_mi_lod.csv')
                del mi_scan; gc.collect()
                _record(timing_rows, 'S7_mi_scan', t, f'chr{peak_chrom} num_imp=11')
            else:
                log.info('  %s peak_chrom unknown — S7 skipped', SKIP)
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S8 — SNP scan (gated: GM_fo_mm39.csv required)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 8
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS8: SNP scan (gated)...')
            from mosaiqtl.genotype.snp_db import build_founder_allele_db
            from mosaiqtl.qtl.snp_scan import imputed_snp_scan_from_db
            gm_csv_path = Path('../input/GM_fo_mm39.csv')
            alleles_h5  = py_dir / 'founder_alleles.h5'
            if gm_csv_path.exists() and 1 in sim_ys and peak_chrom:
                if not alleles_h5.exists() or force:
                    _gm_fcols = ['AJ', 'B6J', '129S1J', 'NODJ',
                                 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
                    build_founder_allele_db(
                        gm_csv_path, cache, alleles_h5,
                        founder_cols=_gm_fcols, max_dist_mb=0.5)
                _fcol_map = {
                    'A': 'AJ', 'B': 'B6J', 'C': '129S1J',
                    'D': 'NODJ', 'E': 'NZOJ', 'F': 'CASTJ',
                    'G': 'PWKJ', 'H': 'WSBJ',
                }
                _s8_chrom = str(peak_chrom).lstrip('chr')
                with _Timer() as t:
                    snp_lod_py = imputed_snp_scan_from_db(
                        sim_ys[1], cache, alleles_h5,
                        kinship_decomp=kinship_decomp,
                        chrom=_s8_chrom,
                        founder_col_map=_fcol_map,
                        use_rint=False,
                        phenotype_id='sim.y.1',
                    )
                if not snp_lod_py.empty:
                    log.info('  SNP peak LOD = %.3f  n_snps=%d',
                             float(snp_lod_py['lod'].max()), len(snp_lod_py))
                    snp_lod_py.to_csv(py_dir / 's8_snp_lod.csv', index=False)
                del snp_lod_py; gc.collect()
                _record(timing_rows, 'S8_snp_scan', t, f'chr{peak_chrom}')
            else:
                log.info('  %s GM_fo_mm39.csv not found at %s — S8 skipped',
                         SKIP, gm_csv_path)
                timing_rows.append({'section': 'S8_snp_scan', 'elapsed_s': None,
                                    'peak_mb': None, 'peak_rss_mb': None,
                                    'note': 'SKIPPED'})
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S9 — mediation (outcome=sim.y.1, mediator=sim.y.2, peak chromosome)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 9
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS9: mediation (chr%s)...', peak_chrom)
            if 1 in sim_ys and 2 in sim_ys and peak_chrom:
                with _Timer() as t:
                    med_scan = mediation_scan_qr(
                        outcome=sim_ys[1],
                        mediator=sim_ys[2],
                        genomecache=cache,
                        kinship_decomp=kinship_decomp,
                    )
                log.info('  mediation done; %d loci', len(med_scan))
                med_scan.to_csv(py_dir / 's9_mediation.csv', index=False)
                log.info('  saved s9_mediation.csv')
                del med_scan; gc.collect()
                _record(timing_rows, 'S9_mediation', t, f'chr{peak_chrom}')
            else:
                log.info('  %s peak_chrom or phenotypes unavailable — S9 skipped', SKIP)
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S10 — bootstrap CI (n=100, peak chromosome)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 10
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS10: bootstrap CI (n=100, chr%s)...', peak_chrom)
            if rop_scan_1 is not None and peak_chrom and 1 in sim_ys:
                with _Timer() as t:
                    lids_c, pos_c, _ = cache.get_chromosome(peak_chrom)
                    null_boot = generate_null_outcomes(
                        sim_ys[1], n_perms=100, method='bootstrap', seed=1)
                    ci_peaks = run_positional_scans(
                        null_boot, cache,
                        phenotype=sim_ys[1],
                        kinship_decomp=kinship_decomp,
                        peak_chrom=peak_chrom,
                        sample_ids=sample_ids,
                    )
                log.info('  bootstrap CI done; %d peak positions', len(ci_peaks))
                del null_boot, ci_peaks; gc.collect()
                _record(timing_rows, 'S10_bootstrap_ci', t, 'n=100 chr-restricted')
            else:
                log.info('  %s rop_scan_1 or peak_chrom unavailable — S10 skipped', SKIP)
            _mark_done(py_dir, str(sec))

    # ─────────────────────────────────────────────────────────────────────────
    # S11 — instability scan (n=100 subsamples, peak chromosome)
    # ─────────────────────────────────────────────────────────────────────────
    sec = 11
    if sec in sections:
        if not _checkpoint(py_dir, str(sec), force):
            log.info('\nS11: instability scan (n=100, chr%s)...', peak_chrom)
            if 1 in sim_ys and peak_chrom:
                lids_c, pos_c, _ = cache.get_chromosome(peak_chrom)
                with _Timer() as t:
                    instab_scan = instability_lm_scan(
                        sim_ys[1], cache,
                        sample_ids=sample_ids,
                        n_subsamples=250,
                        seed=1,
                        just_these_loci=set(np.asarray(lids_c).tolist()),
                    )
                log.info('  instability done; %d loci', len(instab_scan.lod))
                instab_scan.to_dataframe()[
                    ['locus_id', 'chromosome', 'position', 'lod']
                ].to_csv(py_dir / 's11_instability.csv', index=False)
                log.info('  saved s11_instability.csv')
                del instab_scan; gc.collect()
                _record(timing_rows, 'S11_instability', t, 'n=250 chr-restricted')
            else:
                log.info('  %s peak_chrom unavailable — S11 skipped', SKIP)
            _mark_done(py_dir, str(sec))

    timing_rows.append({
        'section':     'total',
        'elapsed_s':   time.perf_counter() - _t0_run,
        'peak_mb':     None,
        'peak_rss_mb': None,
        'note':        '',
    })

    # ── compare vs R timing + produce summary ─────────────────────────────────
    _print_summary(timing_rows, results, r_data, out_dir, plt_dir)


def _print_summary(
    timing_rows: 'list[dict]',
    results: dict,
    r_data: Path,
    out_dir: Path,
    plt_dir: Path,
) -> None:
    py_dir  = out_dir / 'py_data'
    cmp_dir = out_dir / 'compare'
    cmp_dir.mkdir(parents=True, exist_ok=True)
    t_df   = pd.DataFrame(timing_rows)

    # pipeline_from_scratch = S1 + S3 + S4: full cost a user pays on a fresh run
    pipe_secs = ['S1_cache_info', 'S3_kinship_loco', 'S4_lmm_scan']
    pipe_rows = t_df[t_df['section'].isin(pipe_secs)]
    pipe_py = pipe_rows['elapsed_s'].sum() if len(pipe_rows) == len(pipe_secs) else float('nan')
    if pd.notna(pipe_py):
        t_df = pd.concat([t_df, pd.DataFrame([{
            'section':     'pipeline_from_scratch',
            'elapsed_s':   pipe_py,
            'peak_mb':     None,
            'peak_rss_mb': None,
            'note':        'S1+S3+S4',
        }])], ignore_index=True)

    r_timing_path = r_data / 'timing_sections.csv'
    if r_timing_path.exists():
        r_t = pd.read_csv(r_timing_path)[
            ['section', 'elapsed_s', 'peak_heap_mb']
        ].rename(columns={'elapsed_s': 'r_elapsed_s',
                          'peak_heap_mb': 'r_peak_heap_mb'})
        t_df = t_df.merge(r_t, on='section', how='left')
        t_df['speedup'] = (
            t_df['r_elapsed_s'] / t_df['elapsed_s'].replace(0.0, float('nan')))
    else:
        t_df['r_elapsed_s']    = float('nan')
        t_df['r_peak_heap_mb'] = float('nan')
        t_df['speedup']        = float('nan')
        log.info('\n  (R timing not found at %s — speedup column will be NaN)', r_timing_path)

    t_df.to_csv(py_dir / 'benchmark_summary.csv', index=False)

    log.info('\n── Phase 3 benchmark summary ───────────────────────────────────────────')
    hdr = (f"  {'Section':<24s}  {'Py (s)':>8s}  {'R (s)':>8s}  "
           f"{'Speedup':>8s}  {'RSS MB':>8s}  {'R MB':>8s}")
    log.info(hdr)
    log.info('  ' + '─' * 80)
    for _, row in t_df.iterrows():
        py_s    = row['elapsed_s']
        r_s     = row.get('r_elapsed_s', float('nan'))
        sp      = row.get('speedup', float('nan'))
        rss_mb  = row.get('peak_rss_mb')
        r_mb    = row.get('r_peak_heap_mb', float('nan'))
        py_str   = f'{py_s:8.1f}' if pd.notna(py_s)   else '    skip'
        r_str    = f'{r_s:8.1f}'  if pd.notna(r_s)    else '    n/a '
        sp_str   = f'{sp:7.1f}×'  if pd.notna(sp)     else '    n/a '
        rssmb_str = f'{rss_mb:8.1f}' if rss_mb is not None else '     n/a'
        rmb_str   = f'{r_mb:8.1f}'   if pd.notna(r_mb)     else '     n/a'
        log.info('  %-24s  %s  %s  %s  %s  %s',
                 row['section'], py_str, r_str, sp_str, rssmb_str, rmb_str)
    log.info('  ' + '─' * 80)

    # gates
    log.info('')
    n_genes = 50
    s4_rows = t_df[t_df['section'] == 'S4_lmm_scan']
    if len(s4_rows) > 0 and pd.notna(s4_rows.iloc[0]['elapsed_s']):
        s4_py   = float(s4_rows.iloc[0]['elapsed_s'])
        s4_r    = float(s4_rows.iloc[0].get('r_elapsed_s', float('nan')))
        s4_sp   = float(s4_rows.iloc[0].get('speedup', float('nan')))
        g_min   = n_genes / (s4_py / 60)

        ok_speed = s4_sp >= 5.0 if pd.notna(s4_sp) else None
        ok_gpm   = g_min > 100.0
        ok_mem   = float(results.get('MEM-02', 0)) < 4096.0

        for label, val, unit, gate, ok in [
            ('SPEED-02  S4 speedup', s4_sp, '×',      '≥ 5×',       ok_speed),
            ('SPEED-06  S4 throughput', g_min, 'genes/min', '> 100/min', ok_gpm),
            ('MEM-02    S4 peak RSS',  float(results.get('MEM-02', 0)), 'MB', '< 4096 MB', ok_mem),
        ]:
            if ok is None:
                sym = SKIP
            elif ok:
                sym = PASS
            else:
                sym = MEAS   # first run — log value, no hard fail
            log.info('  %s %-30s  %.1f %s  (gate %s)', sym, label, val, unit, gate)

    log.info('\n  benchmark_summary.csv written to: %s', py_dir / 'benchmark_summary.csv')

    # ── CSV correctness comparison (Py vs R per-section outputs) ─────────────
    _csv_pairs = [
        ('S4 LOD (sim.y.1)',   py_dir / 's4_lod_py.csv',    r_data / 'lod_r.csv',
         'lod',      'lod',      's4'),
        ('S7 MI LOD',          py_dir / 's7_mi_lod.csv',    r_data / 'mi_lod_r.csv',
         'lod',      'lod',      's7'),
        # S8: Python scan filtered to cache-locus SNPs — same scope as R's loci_by_chr.
        # R's allele dir only has entries for real GigaMUGA markers that are cache
        # loci; pseudomarker loci are silently skipped. Python mirrors this by
        # restricting just_these_snps to markers whose ID is in the cache locus set.
        ('S8 SNP LOD',         py_dir / 's8_snp_lod.csv',   r_data / 'snp_lod_r.csv',
         'lod',      'lod',      's8'),
        # S9: R exports lod = mediated scan (scan.h2lmm(formula=sim.y.1~sim.y.2)).
        # Python's lod_mediated is the equivalent column (scan with mediator covariate).
        ('S9 mediation LOD',   py_dir / 's9_mediation.csv', r_data / 'mediation_r.csv',
         'lod_mediated', 'lod',      's9'),
        # S11 excluded: R uses permutation (null distribution, mean -log10(p));
        # Python uses bootstrap subsampling (robustness, mean LOD). Different
        # algorithms and incompatible summary statistics — not comparable via r.
    ]
    log.info('\n── Phase 3 CSV correctness (Py vs R) ───────────────────────────────────')

    def _norm_chr(s: pd.Series) -> pd.Series:
        """Strip 'chr' prefix; keep as string; 'X'→'X'."""
        return s.astype(str).str.replace('^chr', '', regex=True).str.strip()

    def _pos_merge(py_df: pd.DataFrame, r_df: pd.DataFrame,
                   py_val: str, r_val_col: str) -> pd.DataFrame:
        """Merge on (chromosome, position_mb) rounded to 3 dp — locus-ID-agnostic.

        Handles both _scan_to_df output (position_mb) and to_dataframe() (position).
        R columns: chr / pos_mb.
        """
        py_chr = 'chromosome' if 'chromosome' in py_df.columns else 'chr'
        py_pos = ('position'    if 'position'    in py_df.columns else
                  'position_mb' if 'position_mb' in py_df.columns else 'pos_mb')
        r_chr  = 'chr'   if 'chr'   in r_df.columns else 'chromosome'
        r_pos  = 'pos_mb' if 'pos_mb' in r_df.columns else 'position'

        py = py_df[[py_chr, py_pos, py_val]].copy()
        r  = r_df[[r_chr,  r_pos,  r_val_col]].copy()

        py['_chr'] = _norm_chr(py[py_chr])
        py['_pos'] = pd.to_numeric(py[py_pos], errors='coerce').round(3)
        r['_chr']  = _norm_chr(r[r_chr])
        r['_pos']  = pd.to_numeric(r[r_pos],  errors='coerce').round(3)

        # Deduplicate by rounded position on each side before merging to prevent
        # many-to-many joins when two loci round to the same (chr, pos) key.
        # Keeps the highest-LOD row per position (fallback: first).
        py_dedup = (
            py[['_chr', '_pos', py_val]]
            .sort_values(py_val, ascending=False)
            .drop_duplicates(subset=['_chr', '_pos'], keep='first')
        )
        r_dedup = (
            r[['_chr', '_pos', r_val_col]]
            .sort_values(r_val_col, ascending=False)
            .drop_duplicates(subset=['_chr', '_pos'], keep='first')
        )

        return pd.merge(
            py_dedup.rename(columns={py_val: 'py'}),
            r_dedup.rename(columns={r_val_col: 'r'}),
            on=['_chr', '_pos'], how='inner',
        ).dropna()

    def _correctness_stats(py_arr: np.ndarray, r_arr: np.ndarray) -> dict:
        """Compute r, slope, intercept, max|Δ|, mean|Δ|.

        PASS requires all three: r ≥ 0.99, |slope−1| < 0.05, |intercept| < 0.1.
        Slope ≈ 1 and intercept ≈ 0 rule out systematic scale/offset bias that
        Pearson r alone cannot detect (e.g. Python LODs uniformly 2× R passes r=1).
        """
        r_pearson, _ = stats.pearsonr(py_arr, r_arr)
        slope, intercept, *_ = stats.linregress(r_arr, py_arr)
        abs_diff = np.abs(py_arr - r_arr)
        r_ok = r_pearson >= 0.99
        s_ok = abs(slope - 1.0) < 0.05
        i_ok = abs(intercept) < 0.1
        if r_ok and s_ok and i_ok:
            status = 'PASS'
        elif r_pearson >= 0.90:
            status = 'WARN'
        else:
            status = 'FAIL'
        return {
            'pearson_r':    float(r_pearson),
            'slope':        float(slope),
            'intercept':    float(intercept),
            'max_abs_diff': float(abs_diff.max()),
            'mean_abs_diff': float(abs_diff.mean()),
            'slope_ok':     s_ok,
            'intercept_ok': i_ok,
            'status':       status,
        }

    corr_rows: list[dict] = []
    for label, py_csv, r_csv, py_col, r_col, slug in _csv_pairs:
        if not py_csv.exists() or not r_csv.exists():
            missing = [s for s, p in (('py', py_csv), ('r', r_csv)) if not p.exists()]
            log.info('  %-22s  missing CSV(s): %s', label, ', '.join(missing))
            corr_rows.append({'section': label, 'pearson_r': float('nan'),
                              'n_matched': 0, 'n_py': 0, 'n_r': 0,
                              'match_rate': float('nan'), 'max_abs_diff': float('nan'),
                              'status': 'missing'})
            continue

        py_df = pd.read_csv(py_csv)
        r_df  = pd.read_csv(r_csv)
        n_py, n_r = len(py_df), len(r_df)

        if slug == 's8':
            # S8: merge on marker ID — R scans cache-locus SNPs (sparse haplotype
            # grid), Python scans all GigaMUGA SNPs (denser). Positions differ
            # because R reports locus positions, Python reports actual SNP positions.
            # Shared SNPs (~78% of R's set) are compared by marker ID.
            _py_snp_col = 'snp' if 'snp' in py_df.columns else py_df.columns[0]
            _r_snp_col  = 'snp' if 'snp' in r_df.columns  else r_df.columns[0]
            merged = (
                py_df[[_py_snp_col, py_col]]
                .rename(columns={_py_snp_col: '_snp', py_col: 'py'})
                .merge(
                    r_df[[_r_snp_col, r_col]]
                    .rename(columns={_r_snp_col: '_snp', r_col: 'r'}),
                    on='_snp', how='inner',
                )
                .dropna()
            )
        else:
            merged = _pos_merge(py_df, r_df, py_col, r_col)
        n_matched = len(merged)
        # denominator = n_r: fraction of R's reference loci verified in Python
        match_rate = n_matched / n_r if n_r > 0 else float('nan')

        if n_matched < 3:
            log.info('  %-22s  n_py=%d  n_r=%d  matched=%d/n_r (%.0f%%)  TOO FEW',
                     label, n_py, n_r, n_matched, match_rate * 100)
            corr_rows.append({'section': label, 'pearson_r': float('nan'),
                              'n_matched': n_matched, 'n_py': n_py, 'n_r': n_r,
                              'match_rate': match_rate, 'max_abs_diff': float('nan'),
                              'status': 'too_few'})
            continue

        if match_rate < 0.80:
            log.warning('  %-22s  LOW MATCH RATE %.0f%% (%d/%d R loci) — possible scope mismatch',
                        label, match_rate * 100, n_matched, n_r)

        cs   = _correctness_stats(merged['py'].values, merged['r'].values)
        status = cs['status']
        slope_flag    = '' if cs['slope_ok']     else '  ⚠ slope'
        intercept_flag = '' if cs['intercept_ok'] else '  ⚠ intercept'

        log.info('  %-22s  r=%.4f  slope=%.3f  intercept=%.3f  '
                 'max_Δ=%.3f  n=%d/%d (%.0f%%)  %s%s%s',
                 label, cs['pearson_r'], cs['slope'], cs['intercept'],
                 cs['max_abs_diff'], n_matched, n_r, match_rate * 100,
                 status, slope_flag, intercept_flag)

        corr_rows.append({
            'section': label, 'n_matched': n_matched, 'n_py': n_py, 'n_r': n_r,
            'match_rate': match_rate, **cs,
        })

        # save side-by-side comparison CSV to compare/ (Phase 2 pattern)
        merged.to_csv(cmp_dir / f'correctness_{slug}_merged.csv', index=False)

        # ── scatter plot: R LOD vs Python LOD ────────────────────────────────
        fig_s, ax_s = plt.subplots(figsize=(4.5, 4.5), dpi=300)
        ax_s.scatter(merged['r'].values, merged['py'].values,
                     s=6, alpha=0.4, color='#4F6EAF', linewidths=0, rasterized=True)
        _lo = min(float(merged['r'].min()), float(merged['py'].min())) - 0.2
        _hi = max(float(merged['r'].max()), float(merged['py'].max())) + 0.2
        _rx = np.array([_lo, _hi])
        # identity line (what perfect agreement looks like)
        ax_s.plot(_rx, _rx, color='#D83026', linewidth=1.0, linestyle='--',
                  label='identity (y = x)')
        # fitted regression line (reveals scale/offset bias)
        ax_s.plot(_rx, cs['slope'] * _rx + cs['intercept'],
                  color='#55AF5B', linewidth=1.0, linestyle='-',
                  label=f'fit  slope={cs["slope"]:.3f}  b={cs["intercept"]:.3f}')
        ax_s.set_xlim([_lo, _hi])
        ax_s.set_ylim([_lo, _hi])
        ax_s.set_xlabel('R / miQTL  LOD', fontsize=9)
        ax_s.set_ylabel('Python / mosaiqtl  LOD', fontsize=9)
        ax_s.set_title(
            f'Correctness: {label}\n'
            f'r={cs["pearson_r"]:.4f}  slope={cs["slope"]:.3f}  '
            f'intercept={cs["intercept"]:.3f}  max|Δ|={cs["max_abs_diff"]:.3f}'
            f'\nn={n_matched:,} / {n_r:,} R loci ({match_rate*100:.0f}%)  [{status}]',
            fontsize=7.5,
        )
        ax_s.legend(fontsize=6.5)
        sns.despine(ax=ax_s)
        fig_s.tight_layout()
        fig_s.savefig(plt_dir / f'correctness_{slug}_scatter.png', dpi=300,
                      bbox_inches='tight')
        plt.close(fig_s)
        log.info('  saved plots/correctness_%s_scatter.png', slug)

        # ── multi-line LOD profile (S4 only — full genome) ───────────────────
        if slug == 's4':
            def _sort_lod(df: pd.DataFrame, lod_col: str) -> pd.DataFrame:
                df = df.copy()
                _chr_src = ('chromosome' if 'chromosome' in df.columns else 'chr')
                _pos_src = ('position'    if 'position'    in df.columns else
                            'position_mb' if 'position_mb' in df.columns else 'pos_mb')
                df['_chr_int'] = pd.to_numeric(
                    df[_chr_src].astype(str)
                    .str.replace('chr', '', regex=False)
                    .str.replace('X', '99', regex=False),
                    errors='coerce').fillna(99).astype(int)
                df['_pos'] = pd.to_numeric(df[_pos_src], errors='coerce').fillna(0)
                return df.sort_values(['_chr_int', '_pos']).reset_index(drop=True)

            py_sorted = _sort_lod(py_df, py_col)
            r_sorted  = _sort_lod(r_df,  r_col)

            fig_l, ax_l = plt.subplots(figsize=(11, 3.5), dpi=300)
            ax_l.plot(np.arange(len(r_sorted)),  r_sorted[r_col].values,
                      color='#F4C245', linewidth=0.7, alpha=0.85, label='R / miQTL')
            ax_l.plot(np.arange(len(py_sorted)), py_sorted[py_col].values,
                      color='#52A5DB', linewidth=0.7, alpha=0.85, label='Python / mosaiqtl')

            # chromosome boundary ticks on x-axis
            chr_col_r = 'chromosome' if 'chromosome' in r_sorted.columns else 'chr'
            _chrs = r_sorted['_chr_int'].values
            _boundaries = np.where(np.diff(_chrs) != 0)[0] + 1
            _centers: list[int] = []
            _chr_labels: list[str] = []
            prev = 0
            for bnd in list(_boundaries) + [len(_chrs)]:
                _centers.append((prev + bnd) // 2)
                _chr_labels.append('X' if _chrs[prev] == 99 else str(_chrs[prev]))
                prev = bnd
            ax_l.set_xticks(_centers)
            ax_l.set_xticklabels(_chr_labels, fontsize=6)
            for bnd in _boundaries:
                ax_l.axvline(bnd, color='#E8E8E8', linewidth=0.5, zorder=0)

            ax_l.set_xlabel('Chromosome', fontsize=9)
            ax_l.set_ylabel('LOD', fontsize=9)
            ax_l.set_title(
                f'S4 genome-wide LOD profile: Python vs R (sim.y.1)\nr = {cs["pearson_r"]:.4f}',
                fontsize=9)
            ax_l.legend(fontsize=7, loc='upper right')
            sns.despine(ax=ax_l)
            fig_l.tight_layout()
            fig_l.savefig(plt_dir / 'correctness_s4_lod_profile.png',
                          dpi=300, bbox_inches='tight')
            plt.close(fig_l)
            log.info('  saved plots/correctness_s4_lod_profile.png')

        # ── SNP LOD profile (S8 only — peak chromosome) ──────────────────────
        if slug == 's8':
            py_s8 = py_df.copy()
            r_s8  = r_df.copy()

            # normalise chr and pos columns
            _pc = 'chromosome' if 'chromosome' in py_s8.columns else 'chr'
            _pp = ('position'    if 'position'    in py_s8.columns else
                   'position_mb' if 'position_mb' in py_s8.columns else 'pos_mb')
            _rc = 'chr' if 'chr' in r_s8.columns else 'chromosome'
            _rp = 'pos_mb' if 'pos_mb' in r_s8.columns else 'position'

            py_s8['_pos'] = pd.to_numeric(py_s8[_pp], errors='coerce')
            r_s8['_pos']  = pd.to_numeric(r_s8[_rp],  errors='coerce')
            py_s8 = py_s8.sort_values('_pos').reset_index(drop=True)
            r_s8  = r_s8.sort_values('_pos').reset_index(drop=True)

            fig_s8, ax_s8 = plt.subplots(figsize=(9, 3.5), dpi=300)
            ax_s8.plot(r_s8['_pos'].values, r_s8[r_col].values,
                       color='#F4C245', linewidth=0.7, alpha=0.85, label='R / miQTL')
            ax_s8.plot(py_s8['_pos'].values, py_s8[py_col].values,
                       color='#52A5DB', linewidth=0.7, alpha=0.85,
                       label='Python / mosaiqtl')

            ax_s8.set_xlabel('Position (Mb)', fontsize=9)
            ax_s8.set_ylabel('LOD', fontsize=9)
            _chr_label = str(py_s8[_pc].iloc[0]) if len(py_s8) else '?'
            ax_s8.set_title(
                f'S8 SNP LOD profile: Python vs R  chr{_chr_label}\n'
                f'r={cs["pearson_r"]:.4f}  '
                f'n_py={len(py_s8):,}  n_r={len(r_s8):,}  matched={n_matched:,}',
                fontsize=8,
            )
            ax_s8.legend(fontsize=7, loc='upper right')
            sns.despine(ax=ax_s8)
            fig_s8.tight_layout()
            fig_s8.savefig(plt_dir / 'correctness_s8_snp_profile.png',
                           dpi=300, bbox_inches='tight')
            plt.close(fig_s8)
            log.info('  saved plots/correctness_s8_snp_profile.png')

    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(py_dir / 'correctness_summary.csv', index=False)
        log.info('  correctness_summary.csv written')

    # ── plot configuration ───────────────────────────────────────────────────
    # pipeline_from_scratch excluded — R does not report it as a standalone section
    _SEC_ORDER = [
        'S1_cache_info', 'S2_haplotype_probs', 'S3_kinship_loco',
        'S4_lmm_scan', 'S6_permutation', 'S7_mi_scan', 'S8_snp_scan',
        'S9_mediation', 'S10_bootstrap_ci', 'S11_instability',
        'total',
    ]

    _SEC_LABELS: dict[str, str] = {
        'S1_cache_info':      'Cache read/write (S1)',
        'S2_haplotype_probs': 'Haplotype probabilities (S2)',
        'S3_kinship_loco':    'LOCO kinship build (S3)',
        'S4_lmm_scan':        'LMM genome scan — 50 genes (S4)',
        'S6_permutation':     'Permutation threshold — 250 perms (S6)',
        'S7_mi_scan':         'Multiple imputation scan (S7)',
        'S8_snp_scan':        'SNP scan (S8)',
        'S9_mediation':       'Mediation scan (S9)',
        'S10_bootstrap_ci':   'Bootstrap confidence interval (S10)',
        'S11_instability':    'Instability scan — 100 samples (S11)',
        'total':              'Total — all steps combined',
        'peak_overall':       'Peak memory — overall maximum',
    }

    def _label(sec: str) -> str:
        return _SEC_LABELS.get(sec, sec)

    def _ordered(df: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
        """Return df rows in order, dropping sections not in df."""
        ord_ = order if order is not None else _SEC_ORDER
        present = [s for s in ord_ if s in df['section'].values]
        return df.set_index('section').loc[present].reset_index()

    def _fmt_s(v: float) -> str:
        if v < 1.0:
            return f'{v * 1000:.0f} ms'
        if v < 60.0:
            return f'{v:.1f} s'
        return f'{v / 60:.1f} min'

    # ── compute total speedup from matched-pair section sums ─────────────────
    # 'total' row from the script's timer often includes Python-only overhead
    # (cache build, CSV loads) that R doesn't count — so use matched sections only.
    _core_secs = [s for s in _SEC_ORDER if s != 'total']
    if 'r_elapsed_s' in t_df.columns:
        _matched = t_df[
            t_df['section'].isin(_core_secs) &
            t_df['elapsed_s'].notna() &
            t_df['r_elapsed_s'].notna()
        ]
        if len(_matched) > 0 and _matched['elapsed_s'].sum() > 0:
            _total_spd = _matched['r_elapsed_s'].sum() / _matched['elapsed_s'].sum()
            if 'speedup' not in t_df.columns:
                t_df['speedup'] = float('nan')
            if 'total' in t_df['section'].values:
                t_df.loc[t_df['section'] == 'total', 'speedup'] = _total_spd
                t_df.loc[t_df['section'] == 'total', 'elapsed_s'] = (
                    _matched['elapsed_s'].sum())
                t_df.loc[t_df['section'] == 'total', 'r_elapsed_s'] = (
                    _matched['r_elapsed_s'].sum())
            else:
                _tot_row = pd.DataFrame([{
                    'section':     'total',
                    'elapsed_s':   _matched['elapsed_s'].sum(),
                    'r_elapsed_s': _matched['r_elapsed_s'].sum(),
                    'speedup':     _total_spd,
                    'peak_mb':     float('nan'),
                    'peak_rss_mb': float('nan'),
                }])
                t_df = pd.concat([t_df, _tot_row], ignore_index=True)

    # ── speedup grouped bar chart ─────────────────────────────────────────────
    spd_df = _ordered(t_df[t_df['speedup'].notna()])
    if len(spd_df) > 0:
        secs   = list(spd_df['section'])
        labels = [_label(s) for s in secs]
        vals   = list(spd_df['speedup'].astype(float))
        y      = np.arange(len(secs))
        colors = ['#C0BFBF' if s == 'total' else
                  '#55AF5B' if v >= 5.0 else
                  '#52A5DB' if v >= 1.0 else '#D83026'
                  for s, v in zip(secs, vals)]
        fig, ax = plt.subplots(figsize=(8, max(3, len(secs) * 0.65 + 1.5)), dpi=300)
        bars = ax.barh(y, vals, color=colors, height=0.55, edgecolor='none')
        ax.axvline(1.0, color='#444444', linewidth=0.8, linestyle='--', zorder=3)
        ax.text(1.02, len(secs) - 0.8, '1×', fontsize=6.5, color='#444444', va='top')
        for bar, v, c in zip(bars, vals, colors):
            ax.text(v + max(vals) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f'{v:.1f}×', va='center', ha='left', fontsize=6.5, color=c)
        # separator above total row
        if 'total' in secs:
            _sep = secs.index('total') - 0.5
            ax.axhline(_sep, color='#C0BFBF', linewidth=0.7, linestyle=':', zorder=2)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel('Speedup vs R / miQTL (×)', fontsize=9)
        ax.set_title(
            'mosaiqtl vs miQTL: speed improvement per pipeline step\n'
            '(50 phenotypes, 40 CC strains, seed=123)\n'
            'Note: total = weighted by R time — bottleneck steps dominate',
            fontsize=8,
        )
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(plt_dir / 'speedup_bar.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved plots/speedup_bar.png')

    # ── timing: two-panel grouped bar chart (Python seconds | R minutes) ─────
    _time_df = _ordered(t_df[t_df['elapsed_s'].notna()])
    if len(_time_df) > 0:
        secs    = list(_time_df['section'])
        labels  = [_label(s) for s in secs]
        py_vals = list(_time_df['elapsed_s'].astype(float))
        _r_col  = 'r_elapsed_s'
        r_vals  = ([float(v) if pd.notna(v) else float('nan')
                    for v in _time_df[_r_col]]
                   if _r_col in _time_df.columns
                   else [float('nan')] * len(_time_df))
        has_r = any(not np.isnan(v) for v in r_vals)
        y     = np.arange(len(secs))

        ncols = 2 if has_r else 1
        fig, axes = plt.subplots(
            1, ncols,
            figsize=(5.5 * ncols, max(3, len(secs) * 0.65 + 1.8)),
            dpi=300, sharey=True,
        )
        if ncols == 1:
            axes = [axes]

        # left panel: Python (seconds, linear)
        ax_py = axes[0]
        _py_finite = [v if not np.isnan(v) else 0.0 for v in py_vals]
        bars_py = ax_py.barh(y, _py_finite, color='#52A5DB', height=0.55, edgecolor='none')
        for bar, v in zip(bars_py, py_vals):
            if np.isnan(v):
                continue
            lbl = f'{v * 1000:.0f} ms' if v < 1.0 else f'{v:.2f} s'
            ax_py.text(v + max(_py_finite) * 0.01,
                       bar.get_y() + bar.get_height() / 2,
                       lbl, va='center', ha='left', fontsize=6.5, color='#52A5DB')
        ax_py.set_yticks(y)
        ax_py.set_yticklabels(labels, fontsize=8)
        ax_py.invert_yaxis()
        ax_py.set_xlabel('Elapsed time (seconds)', fontsize=9)
        ax_py.set_title('Python / mosaiqtl', fontsize=9, color='#52A5DB', fontweight='bold')
        sns.despine(ax=ax_py)

        # right panel: R (minutes, linear)
        if has_r:
            ax_r  = axes[1]
            r_mins = [v / 60.0 if not np.isnan(v) else 0.0 for v in r_vals]
            bars_r = ax_r.barh(y, r_mins, color='#F4C245', height=0.55, edgecolor='none')
            for bar, v_min, v_s in zip(bars_r, r_mins, r_vals):
                if np.isnan(v_s):
                    continue
                lbl = f'{v_s:.1f} s' if v_s < 60.0 else f'{v_min:.2f} min'
                ax_r.text(v_min + max(r_mins) * 0.01,
                          bar.get_y() + bar.get_height() / 2,
                          lbl, va='center', ha='left', fontsize=6.5, color='#C8960C')
            ax_r.set_xlabel('Elapsed time (minutes)', fontsize=9)
            ax_r.set_title('R / miQTL', fontsize=9, color='#C8960C', fontweight='bold')
            sns.despine(ax=ax_r)

        fig.suptitle(
            'mosaiqtl vs miQTL: wall-clock time per pipeline step\n'
            '(50 phenotypes, 40 CC strains, seed=123)',
            fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(plt_dir / 'timing_comparison.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved plots/timing_comparison.png')

    # ── memory grouped bar chart (Python RSS delta vs R gc max-used) ──────────
    _mem_r_avail = 'r_peak_heap_mb' in t_df.columns
    _mem_base    = t_df[~t_df['section'].isin(['total', 'peak_overall'])].copy()

    # synthetic peak_overall row: maximum across all sections
    _rss_col = 'peak_rss_mb'
    _max_py_mem = (_mem_base[_rss_col].max()
                   if _rss_col in _mem_base.columns else float('nan'))
    _max_r_mem  = (_mem_base['r_peak_heap_mb'].max()
                   if _mem_r_avail else float('nan'))
    _peak_row: dict = {'section': 'peak_overall', 'elapsed_s': float('nan'),
                       _rss_col: _max_py_mem, 'peak_mb': float('nan'),
                       'speedup': float('nan')}
    if _mem_r_avail:
        _peak_row['r_peak_heap_mb'] = _max_r_mem
    _mem_df_src = pd.concat(
        [_mem_base, pd.DataFrame([_peak_row])], ignore_index=True)

    _mem_sec_order = _SEC_ORDER[:-1] + ['peak_overall']
    mem_df = _ordered(
        _mem_df_src[_mem_df_src[_rss_col].notna()] if _rss_col in _mem_df_src.columns
        else _mem_df_src[_mem_df_src['peak_mb'].notna()],
        order=_mem_sec_order)
    if len(mem_df) == 0:
        mem_df = _ordered(_mem_df_src, order=_mem_sec_order)

    if len(mem_df) > 0:
        secs   = list(mem_df['section'])
        labels = [_label(s) for s in secs]
        # RSS delta is primary (captures numpy C allocations missed by tracemalloc)
        py_mem = ([float(v) if pd.notna(v) else 0.0
                   for v in mem_df[_rss_col]]
                  if _rss_col in mem_df.columns else [0.0] * len(mem_df))
        r_mem  = ([float(v) if pd.notna(v) else float('nan')
                   for v in mem_df['r_peak_heap_mb']]
                  if _mem_r_avail else [float('nan')] * len(mem_df))
        has_py_mem = any(v > 0 for v in py_mem)
        has_r_mem  = _mem_r_avail and any(not np.isnan(v) for v in r_mem)

        if has_py_mem or has_r_mem:
            gap  = 0.22
            bar_h = 0.38
            y    = np.arange(len(secs))
            fig2, ax2 = plt.subplots(
                figsize=(9, max(3, len(secs) * 0.75 + 1.5)), dpi=300)

            if has_py_mem:
                bars_py2 = ax2.barh(y + gap, py_mem, height=bar_h,
                                    color='#52A5DB', edgecolor='none',
                                    label='Python / mosaiqtl (RSS delta)')
                _py_max = max(v for v in py_mem if not np.isnan(v))
                for bar, v in zip(bars_py2, py_mem):
                    if v == 0:
                        continue
                    ax2.text(v + _py_max * 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{v:.0f} MB', va='center', ha='left',
                             fontsize=6.5, color='#52A5DB')

            if has_r_mem:
                _r_mem_plot = [v if not np.isnan(v) else 0.0 for v in r_mem]
                bars_r2 = ax2.barh(y - gap, _r_mem_plot, height=bar_h,
                                   color='#F4C245', edgecolor='none',
                                   label='R / miQTL (gc max-used)')
                _r_max = max(v for v in _r_mem_plot if v > 0) if any(v > 0 for v in _r_mem_plot) else 1.0
                for bar, v in zip(bars_r2, r_mem):
                    if np.isnan(v) or v == 0:
                        continue
                    ax2.text(v + _r_max * 0.01, bar.get_y() + bar.get_height() / 2,
                             f'{v:.0f} MB', va='center', ha='left',
                             fontsize=6.5, color='#C8960C')

            # separator above peak_overall
            if 'peak_overall' in secs:
                _sep_yi = secs.index('peak_overall') - 0.5
                ax2.axhline(_sep_yi, color='#C0BFBF', linewidth=0.7, linestyle=':')

            ax2.set_yticks(y)
            ax2.set_yticklabels(labels, fontsize=8)
            ax2.invert_yaxis()
            ax2.set_xlabel('Peak memory (MB)', fontsize=9)
            ax2.set_title(
                'mosaiqtl vs miQTL: peak memory per pipeline step\n'
                '(50 phenotypes, 40 CC strains, seed=123)',
                fontsize=9,
            )
            ax2.legend(fontsize=7)
            sns.despine(ax=ax2)
            fig2.tight_layout()
            fig2.savefig(plt_dir / 'memory_comparison.png', dpi=300, bbox_inches='tight')
            plt.close(fig2)
            log.info('  saved plots/memory_comparison.png')
        else:
            log.info('  memory comparison: no non-zero peak_rss_mb values recorded')


def main() -> None:
    """entry point."""
    ap = argparse.ArgumentParser(
        description='Phase 3 benchmark: mosaiqtl vs miQTL and R/qtl2 speed + memory')
    ap.add_argument('--r_data',    default='validation/phase3/r_data',
                    help='Phase 3 R reference data (phenotypes.csv, timing_sections.csv)')
    ap.add_argument('--out',       default='validation/phase3',
                    help='output root — writes to out/py_data/ and out/plots/')
    ap.add_argument('--sections',  default=None,
                    help='comma-separated section numbers (e.g. 3,4); default: all')
    ap.add_argument('--force', action='store_true',
                    help='rerun sections even if checkpoint .done files exist')
    args = ap.parse_args()

    sections = (
        {int(s.strip()) for s in args.sections.split(',')}
        if args.sections else set(range(1, 12))
    )
    run(
        r_data   = Path(args.r_data),
        out_dir  = Path(args.out),
        sections = sections,
        force    = args.force,
    )


if __name__ == '__main__':
    main()

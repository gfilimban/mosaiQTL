"""
validate_sim.py — Phase 1 validation: minimal miQTL workflow mirror

Python mirror of validation/r_reference/sim_validation.R.
If Python matches R here, the core logic base is correct.
Everything else is an expansion from this ground truth.

Step-by-step mirror of sim_validation.R:
  1. load sparcc_cache genome cache
  2. load sparc phenotypes (phenotypes.csv from R export)
  3. ROP scan  → Python output + py-vs-R comparison
  4. plot genome-wide LOD profile
  5. MI scan (num_imp=11)  → Python output + py-vs-R comparison
  6. plot
  7. permutation (n=500, seed=1, peak_chrom only) → GEV threshold → py-vs-R comparison → plot with threshold
  8. grab peak QTL locus
  9. prob_heatmap at QTL locus
  10. bootstrap positional CI (n=100) → single-chr LOD + CI plot
  11. founder effects on peak chr (genome.plotter.region equivalent)
  12. allele effects on peak chr (allele.plotter.region equivalent)

Outputs (three parallel sets):
  R reference    validation/phase1/r_data/   written by sim_validation.R
  Python output  validation/phase1/py_data/  written by this script
  Comparison     validation/phase1/compare/  merged py+R side-by-side

Pass criteria (all three must pass):
  P1-S1  ROP LOD r² vs R  ≥ 0.990
  P1-S2  MI  LOD r  vs R  ≥ 0.75
  P1-S3  GEV threshold delta vs R  < 0.5 LOD

Directory layout (validation/):
  sparcc_cache/   genome cache   (shared input, .claudeignored)
  sparcc_data/    phenotypes     (shared input: phenotypes.h5, .rds)
  phase1/r_data/  R outputs      (written by sim_validation.R)
  phase1/py_data/ Python outputs (written by this script)
  phase1/compare/ merged         (written by this script)
  phase1/plots/   figures        (written by this script)

Usage:
  python validation/validate_sim.py
  python validation/validate_sim.py --sparcc_cache validation/sparcc_cache \\
                                    --r_data       validation/phase1/r_data \\
                                    --out          validation/phase1
"""

import argparse
import logging
import sys
from pathlib import Path

logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Carlito'
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.qtl.scan import scan_h2lmm
from mosaiqtl.qtl.mi import scan_h2lmm_mi
from mosaiqtl.qtl.peaks import grab_locus_from_scan
from mosaiqtl.qtl.permutation import generate_null_outcomes, run_threshold_scans
from mosaiqtl.qtl.thresholds import get_gev_thresholds
from mosaiqtl.qtl.ci import lod_support_interval
from mosaiqtl.viz.genome import manhattan, lod_interval
from mosaiqtl.viz.effects import coef_profile
from mosaiqtl.viz.haplotypes import prob_heatmap

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_sim')


PASS = '✅'
FAIL = '❌'
SKIP = '⚠️ '


def _load_cache_from_probs_csv(
    probs_dir: Path,
    keep_strains: 'list[str] | None' = None,
) -> 'GenomeCache':
    """build GenomeCache from per-chromosome CSVs exported by sim_validation.R.

    Each CSV has columns: locus, pos_mb, strain, A, B, C, D, E, F, G, H
    Rows are grouped by locus (all strains for locus[0], then locus[1], ...).
    No rpy2 — the R script is the only thing that touches R objects.

    keep_strains : if given, subset to these strains in this order so that
        cache.sample_ids, kinship, and phenotype are all aligned from the start.
    """
    from mosaiqtl.genotype.cc import FOUNDER_ORDER, FOUNDER_CODE

    # R exports single-letter codes; derive column order from FOUNDER_ORDER via
    # FOUNDER_CODE so probs axis-2 aligns with FOUNDER_ORDER (not assumed alphabetical)
    founder_cols: list[str] = [FOUNDER_CODE[name] for name in FOUNDER_ORDER]
    # all three identity axes: order (index), full name, R letter code
    letter_to_founder: dict[str, str] = {v: k for k, v in FOUNDER_CODE.items()}  # noqa: F841
    founder_to_index: dict[str, int] = {n: i for i, n in enumerate(FOUNDER_ORDER)}  # noqa: F841
    letter_to_index: dict[str, int] = {FOUNDER_CODE[n]: i for i, n in enumerate(FOUNDER_ORDER)}  # noqa: F841

    data = {}
    sample_ids = None

    for csv_path in sorted(probs_dir.glob('probs_*.csv')):
        chr_id = csv_path.stem.replace('probs_', '')
        df = pd.read_csv(csv_path)

        if keep_strains is not None:
            df = df[df['strain'].isin(keep_strains)]
            df['strain'] = pd.Categorical(df['strain'], categories=keep_strains, ordered=True)
            df = df.sort_values(['locus', 'strain'])

        n_strains = df['strain'].nunique()
        if sample_ids is None:
            sample_ids = df['strain'].drop_duplicates().tolist()

        loci = list(pd.Series(df['locus']).drop_duplicates())
        positions = (
            df.groupby('locus')['pos_mb'].first().reindex(loci).values.astype(np.float64)
        )

        # rows arrive as (locus_0 × strains), (locus_1 × strains), ...
        # reshape directly: (n_loci * n_strains, 8) → (n_loci, n_strains, 8)
        probs = df[founder_cols].values.reshape(len(loci), n_strains, 8)

        data[chr_id] = {
            'locus_ids': np.array(loci),
            'positions': positions,
            'probs':     probs.astype(np.float64),
        }

    if not data:
        raise FileNotFoundError(f'no probs_*.csv files found in {probs_dir}')

    return GenomeCache(
        sample_ids=np.array(sample_ids),
        founders=FOUNDER_ORDER,
        data=data,
    )


def _check(label, value, threshold, op='>='):
    ok = (value >= threshold) if op == '>=' else (value < threshold)
    sym = PASS if ok else FAIL
    log.info('  %s %s = %.4f  (threshold %s %.4f)', sym, label, value, op, threshold)
    return ok


def _scan_to_manhattan_dict(scan):
    """build {chrom: (positions, lod)} dict for manhattan()."""
    out = {}
    for chrom in np.unique(scan.chromosomes):
        mask = scan.chromosomes == chrom
        out[str(chrom)] = (scan.positions[mask], scan.lod[mask])
    return out


def _scan_to_df(scan):
    """ScanResult → DataFrame with locus, chromosome, position_mb, lod."""
    return pd.DataFrame({
        'locus':        scan.locus_ids,
        'chromosome':   scan.chromosomes,
        'position_mb':  scan.positions,
        'lod':          scan.lod,
    })


def run(sparcc_cache, r_data, out_dir):
    """run Phase 1 validation and return dict of pass/fail per gate."""
    r_data = Path(r_data)
    out_dir     = Path(out_dir)

    py_dir  = out_dir / 'py_data'    # Python outputs
    cmp_dir = out_dir / 'compare'    # merged py-vs-R
    plt_dir = out_dir / 'plots'      # figures

    for d in (py_dir, cmp_dir, plt_dir):
        d.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── phenotypes first — defines the sample space ───────────────────────────
    pheno_path = r_data / 'phenotypes.csv'
    if not pheno_path.exists():
        log.error('phenotypes.csv not found at %s — run sim_validation.R first',
                  pheno_path)
        sys.exit(1)
    pheno_df   = pd.read_csv(pheno_path)
    sample_ids = pheno_df['SUBJECT.NAME.1'].values
    y          = pheno_df['sim.y.1'].values.astype(float)
    log.info('loaded %d samples, phenotype sim.y.1', len(y))

    # ── genome cache — filtered to phenotype strains from the start ───────────
    # cache, kinship, and scan all operate on the same 40-strain space;
    # no dimension mismatch between eigenvectors and phenotype mask
    probs_dir  = r_data / 'probs'
    hdf5_cache = py_dir / 'cache.h5'
    if hdf5_cache.exists():
        log.info('loading genome cache from HDF5: %s', hdf5_cache)
        cache = GenomeCache.from_hdf5(str(hdf5_cache))
    elif probs_dir.exists():
        log.info('loading genome cache from probs CSVs: %s', probs_dir)
        cache = _load_cache_from_probs_csv(probs_dir, keep_strains=list(sample_ids))
        cache.to_hdf5(str(hdf5_cache))
        log.info('  cached to HDF5 for subsequent runs: %s', hdf5_cache)
    else:
        log.error(
            'probs/ directory not found at %s — run sim_validation.R first',
            probs_dir,
        )
        sys.exit(1)
    log.info('  %d chromosomes, %d strains',
             len(cache.chromosomes), len(cache.sample_ids))

    # Phase 1 uses no kinship — R's scan.h2lmm is called without K → tau2=0 OLS.
    # Build an identity kinship decomp (eigvals=0, eigvecs=I) so scan functions
    # skip LOCO recomputation. With eigvals=0, w_sqrt=1 for any h2, making all
    # scans exactly OLS without touching function signatures.
    n_strains = len(cache.sample_ids)
    ols_kd = {
        c: {'values': np.zeros(n_strains), 'vectors': np.eye(n_strains)}
        for c in cache.chromosomes
    }
    ols_h2 = {c: 0.0 for c in cache.chromosomes}   # skip h2 estimation in ROP

    # ─────────────────────────────────────────────────────────────────────────
    # ROP scan
    # R: miqtl.rop.scan <- scan.h2lmm(..., use.multi.impute=FALSE,
    #                                       return.allele.effects=TRUE)
    # ─────────────────────────────────────────────────────────────────────────
    log.info('\nROP scan...')
    # use_rint=False: R collapses replicates → strain means BEFORE rank.normalize
    # (pheno.id=geno.id in scan.h2lmm triggers make.processed.data averaging).
    # Python sample-level RINT diverges from R post-averaging RINT; sparc
    # phenotypes are ~normal so skipping RINT here is the correct match.
    rop_scan = scan_h2lmm(y, cache, sample_ids=sample_ids,
                          kinship_decomp=ols_kd, fixed_h2_dict=ols_h2,
                          use_rint=False, phenotype_id='sim.y.1',
                          scan_window='genome')
    log.info('  peak LOD = %.3f at locus %s',
             float(rop_scan.lod.max()), rop_scan.peak_locus()['locus_id'])

    # Python output
    rop_py = _scan_to_df(rop_scan)
    rop_py.to_csv(py_dir / 'rop_lod.csv', index=False)

    # py-vs-R comparison
    rop_r_path = r_data / 'rop_lod.csv'
    if rop_r_path.exists():
        r_rop  = pd.read_csv(rop_r_path).rename(columns={'lod': 'lod_r'})
        merged = rop_py[['locus', 'chromosome', 'position_mb', 'lod']].rename(
            columns={'lod': 'lod_python'}).merge(r_rop[['locus', 'lod_r']], on='locus')
        merged['delta'] = merged['lod_python'] - merged['lod_r']
        merged.to_csv(cmp_dir / 'rop_lod.csv', index=False)
        common = merged.dropna()
        r2 = float(np.corrcoef(common['lod_python'], common['lod_r'])[0, 1] ** 2)
        results['P1-S1'] = _check('ROP LOD r²', r2, 0.990, '>=')

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.scatter(common['lod_r'], common['lod_python'], s=1, alpha=0.4,
                   color='#4F6EAF', linewidths=0)
        lo = min(common['lod_r'].min(), common['lod_python'].min())
        hi = max(common['lod_r'].max(), common['lod_python'].max())
        ax.plot([lo, hi], [lo, hi], color='#D83026', linewidth=0.8, linestyle='--')
        ax.set_xlabel('R LOD (miQTL ROP)', fontsize=8)
        ax.set_ylabel('Python LOD (mosaiQTL ROP)', fontsize=8)
        ax.set_title(f'P1-S1 ROP LOD  r²={r2:.4f}', fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(cmp_dir / 'p1_s1_rop_lod_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved compare/p1_s1_rop_lod_scatter.png')
    else:
        log.info('  %s rop_lod.csv not found — P1-S1 skipped', SKIP)
        results['P1-S1'] = None

    # R: genome.plotter.whole(scan.list=list(ROP=miqtl.rop.scan))
    fig = manhattan(_scan_to_manhattan_dict(rop_scan))
    fig.axes[0].set_title('ROP scan — sim.y.1', fontsize=9)
    fig.savefig(plt_dir / 'p1_rop_scan.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_rop_scan.png')

    # ─────────────────────────────────────────────────────────────────────────
    # MI scan
    # R: miqtl.mi11.scan <- scan.h2lmm(..., use.multi.impute=TRUE, num.imp=11)
    # ─────────────────────────────────────────────────────────────────────────
    log.info('\nMI scan (num_imp=11)...')
    mi_scan = scan_h2lmm_mi(y, cache, sample_ids=sample_ids,
                            kinship_decomp=ols_kd,
                            use_rint=False,  # same reason as ROP scan above
                            n_imp=11, seed=1,
                            phenotype_id='sim.y.1',
                            scan_window='genome')
    log.info('  peak LOD = %.3f at locus %s',
             float(mi_scan.lod.max()), mi_scan.peak_locus()['locus_id'])

    # Python output
    mi_py = _scan_to_df(mi_scan)
    mi_py.to_csv(py_dir / 'mi_lod.csv', index=False)

    # py-vs-R comparison
    mi_r_path = r_data / 'mi_lod.csv'
    if mi_r_path.exists():
        r_mi   = pd.read_csv(mi_r_path).rename(columns={'lod': 'lod_r'})
        merged = mi_py[['locus', 'chromosome', 'position_mb', 'lod']].rename(
            columns={'lod': 'lod_python'}).merge(r_mi[['locus', 'lod_r']], on='locus')
        merged['delta'] = merged['lod_python'] - merged['lod_r']
        merged.to_csv(cmp_dir / 'mi_lod.csv', index=False)
        common = merged.dropna()
        r_val  = float(np.corrcoef(common['lod_python'], common['lod_r'])[0, 1])
        results['P1-S2'] = _check('MI LOD r', r_val, 0.75, '>=')

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.scatter(common['lod_r'], common['lod_python'], s=1, alpha=0.4,
                   color='#55AF5B', linewidths=0)
        lo = min(common['lod_r'].min(), common['lod_python'].min())
        hi = max(common['lod_r'].max(), common['lod_python'].max())
        ax.plot([lo, hi], [lo, hi], color='#D83026', linewidth=0.8, linestyle='--')
        ax.set_xlabel('R LOD (miQTL MI, num.imp=11)', fontsize=8)
        ax.set_ylabel('Python LOD (mosaiQTL MI)', fontsize=8)
        ax.set_title(f'P1-S2 MI LOD  r={r_val:.4f}', fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(cmp_dir / 'p1_s2_mi_lod_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved compare/p1_s2_mi_lod_scatter.png')
    else:
        log.info('  %s mi_lod.csv not found — P1-S2 skipped', SKIP)
        results['P1-S2'] = None

    # R: genome.plotter.whole(scan.list=list(miqtl.mi11.scan))
    fig = manhattan(_scan_to_manhattan_dict(mi_scan))
    fig.axes[0].set_title('MI scan (num_imp=11) — sim.y.1', fontsize=9)
    fig.savefig(plt_dir / 'p1_mi_scan.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_mi_scan.png')

    # ─────────────────────────────────────────────────────────────────────────
    # permutation + GEV threshold
    # R: generate.sample.outcomes.matrix(..., method="permutation", num.samples=100)
    #    run.threshold.scans(..., use.multi.impute=FALSE, scan.seed=1)
    #    get.gev.thresholds(percentile=0.95)
    # R restricts permutation scans to the peak chromosome (encoded in the scan
    # object passed to generate.sample.outcomes.matrix); match that here.
    # ─────────────────────────────────────────────────────────────────────────
    peak_chrom_for_perm = rop_scan.peak_locus()['chromosome']
    log.info('\nPermutation (n=500, seed=1, chrom=%s)...', peak_chrom_for_perm)
    null_y   = generate_null_outcomes(y, n_perms=500, method='permutation', seed=1)
    max_lods, _ = run_threshold_scans(
        null_y, cache, phenotype=y,
        kinship_decomp=ols_kd,
        use_rint=False,
        permutation_method='plain',
        chrom=peak_chrom_for_perm,
    )
    threshold, gev_params = get_gev_thresholds(max_lods, percentile=0.95)
    if threshold is None:
        finite = max_lods[np.isfinite(max_lods)]
        threshold = float(np.quantile(finite, 0.95))
        gev_params = {
            'empirical_quantile': threshold,
            'c': float('nan'), 'loc': float('nan'), 'scale': float('nan'),
            'n_perms': len(finite),
        }
        log.warning('  GEV fit failed — using empirical 95th pct: %.3f LOD', threshold)
    else:
        log.info('  GEV threshold (95th pct) = %.3f LOD  (empirical = %.3f)',
                 threshold, gev_params['empirical_quantile'])

    # Python output
    pd.DataFrame({'perm_index': np.arange(len(max_lods)),
                  'max_lod': max_lods}).to_csv(py_dir / 'perm_max_lods.csv', index=False)
    pd.DataFrame([{
        'threshold':          threshold,
        'empirical_quantile': gev_params['empirical_quantile'],
        'gev_c':              gev_params['c'],
        'gev_loc':            gev_params['loc'],
        'gev_scale':          gev_params['scale'],
        'n_perms':            gev_params['n_perms'],
    }]).to_csv(py_dir / 'perm_threshold.csv', index=False)

    # py-vs-R comparison
    thresh_r_path = r_data / 'perm_threshold.csv'
    if thresh_r_path.exists():
        r_thresh = float(pd.read_csv(thresh_r_path)['threshold'].iloc[0])
        delta    = abs(threshold - r_thresh)
        cmp_thresh = pd.DataFrame([{
            'threshold_python':    threshold,
            'threshold_r':         r_thresh,
            'delta':               delta,
            'empirical_python':    gev_params['empirical_quantile'],
        }])
        cmp_thresh.to_csv(cmp_dir / 'perm_threshold.csv', index=False)
        results['P1-S3'] = _check('GEV threshold delta', delta, 0.5, '<')
        log.info('    (R threshold = %.3f)', r_thresh)

        from scipy.stats import genextreme as _gev

        r_max_lods_path = r_data / 'perm_max_lods.csv'
        r_max_lods_raw = (pd.read_csv(r_max_lods_path)['max_lod'].values
                          if r_max_lods_path.exists() else None)
        r_max_lods = (r_max_lods_raw[np.isfinite(r_max_lods_raw)]
                      if r_max_lods_raw is not None else None)

        all_vals = (max_lods if r_max_lods is None
                    else np.concatenate([max_lods, r_max_lods]))
        bins  = np.linspace(all_vals.min() - 0.3, all_vals.max() + 0.3, 26)
        x_crv = np.linspace(bins[0], bins[-1], 300)

        fig, ax = plt.subplots(figsize=(5, 3), dpi=300)

        ax.hist(max_lods, bins=bins, color='#52A5DB', edgecolor='white',
                linewidth=0.4, alpha=0.55, density=True, label='Python perms')
        ax.plot(x_crv,
                _gev.pdf(x_crv, gev_params['c'],
                         loc=gev_params['loc'], scale=gev_params['scale']),
                color='#4F6EAF', linewidth=1.2)

        if r_max_lods is not None:
            ax.hist(r_max_lods, bins=bins, color='#F4C245', edgecolor='white',
                    linewidth=0.4, alpha=0.55, density=True, label='R perms')
            r_c, r_loc, r_sc = _gev.fit(r_max_lods)
            ax.plot(x_crv, _gev.pdf(x_crv, r_c, loc=r_loc, scale=r_sc),
                    color='#C8A400', linewidth=1.2, linestyle='--')
            r_p95_emp = float(np.quantile(r_max_lods, 0.95))
            ax.axvline(r_p95_emp, color='#C8A400', linewidth=0.9, linestyle=':',
                       label=f'R emp {r_p95_emp:.2f}')

        py_emp = gev_params['empirical_quantile']
        ax.axvline(py_emp,    color='#52A5DB', linewidth=0.9, linestyle=':',
                   label=f'Py emp {py_emp:.2f}')
        ax.axvline(threshold, color='#D83026', linewidth=1.2,
                   label=f'Py GEV {threshold:.2f}')
        ax.axvline(r_thresh,  color='#F4C245', linewidth=1.2, linestyle='--',
                   label=f'R GEV {r_thresh:.2f}')

        ax.set_xlabel('Max LOD per permutation', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.set_title(
            f'P1-S3 perm max-LOD (n={len(max_lods)}, chr{peak_chrom_for_perm})'
            f'  Δ(GEV)={delta:.3f}',
            fontsize=9)
        ax.legend(fontsize=6, ncol=2)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(cmp_dir / 'p1_s3_perm_threshold.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved compare/p1_s3_perm_threshold.png')
    else:
        log.info('  %s perm_threshold.csv not found — P1-S3 skipped', SKIP)
        results['P1-S3'] = None

    # R: genome.plotter.whole(scan.list=list(ROP=miqtl.rop.scan),
    #                         hard.thresholds=permute_threshold)
    fig = manhattan(_scan_to_manhattan_dict(rop_scan), threshold=threshold)
    fig.axes[0].set_title('ROP scan with permutation threshold — sim.y.1', fontsize=9)
    fig.savefig(plt_dir / 'p1_rop_with_threshold.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_rop_with_threshold.png')

    # ─────────────────────────────────────────────────────────────────────────
    # peak QTL locus
    # R: qtl_locus <- grab.locus.from.scan(miqtl.rop.scan)
    # ─────────────────────────────────────────────────────────────────────────
    log.info('\nQTL locus characterization...')
    qtl_locus_id = grab_locus_from_scan(rop_scan)
    peak         = rop_scan.peak_locus()
    peak_chrom   = peak['chromosome']
    peak_pos_mb  = float(peak['position'])
    log.info('  peak locus: %s  chr %s  %.2f Mb  LOD=%.3f',
             qtl_locus_id, peak_chrom, peak_pos_mb, float(peak['lod']))

    # Python output
    qtl_locus_py = pd.DataFrame([{
        'locus':       qtl_locus_id,
        'chromosome':  peak_chrom,
        'position_mb': peak_pos_mb,
        'lod':         float(peak['lod']),
    }])
    qtl_locus_py.to_csv(py_dir / 'qtl_locus.csv', index=False)

    # py-vs-R comparison
    qtl_locus_r_path = r_data / 'qtl_locus.csv'
    if qtl_locus_r_path.exists():
        r_locus = pd.read_csv(qtl_locus_r_path)
        cmp_locus = qtl_locus_py.rename(columns={
            'position_mb': 'position_mb_python', 'lod': 'lod_python'
        }).assign(
            position_mb_r = r_locus['position_mb'].iloc[0],
            lod_r         = r_locus['lod'].iloc[0],
            locus_match   = qtl_locus_id == r_locus['locus'].iloc[0],
        )
        cmp_locus.to_csv(cmp_dir / 'qtl_locus.csv', index=False)
        locus_match = bool(cmp_locus['locus_match'].iloc[0])
        sym = PASS if locus_match else FAIL
        log.info('  %s peak locus  Py: %s  R: %s',
                 sym, qtl_locus_id, r_locus['locus'].iloc[0])
    else:
        log.info('  %s qtl_locus.csv not found — locus comparison skipped', SKIP)

    locus_ids_c, positions_c, probs_c = cache.get_chromosome(peak_chrom)
    locus_ids_c = np.asarray(locus_ids_c)
    qtl_idx     = int(np.where(locus_ids_c == qtl_locus_id)[0][0])
    hap_probs   = probs_c[qtl_idx]   # (n_strains, J)

    # R: prob.heatmap(marker=qtl_locus, ...)
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300)
    prob_heatmap(hap_probs, ax=ax, phenotype=y,
                 chrom=peak_chrom, position_mb=peak_pos_mb)
    ax.set_title(f'Haplotype probs at {qtl_locus_id}', fontsize=9)
    fig.tight_layout()
    fig.savefig(plt_dir / 'p1_prob_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_prob_heatmap.png')

    # allele effects at peak locus — betas are now (J,) sum ≈ 0 directly
    betas_peak = peak['betas']                          # (J,) sum ≈ 0
    from mosaiqtl.genotype.cc import FOUNDER_ORDER
    effects_py = pd.DataFrame({'founder': FOUNDER_ORDER, 'beta': betas_peak})
    effects_py.to_csv(py_dir / 'allele_effects_peak.csv', index=False)

    effects_r_path = r_data / 'allele_effects_peak.csv'
    if effects_r_path.exists():
        from mosaiqtl.genotype.cc import FOUNDER_CODE
        _letter_to_founder = {v: k for k, v in FOUNDER_CODE.items()}
        r_eff = pd.read_csv(effects_r_path)
        # R exports single-letter codes (A-H); remap to full names before merge
        r_eff['founder'] = r_eff['founder'].map(_letter_to_founder)
        merged = effects_py.rename(columns={'beta': 'beta_python'}).merge(
            r_eff.rename(columns={'beta': 'beta_r'}), on='founder')
        merged['delta'] = merged['beta_python'] - merged['beta_r']
        merged.to_csv(cmp_dir / 'allele_effects_peak.csv', index=False)
        r_eff_val = float(np.corrcoef(merged['beta_python'], merged['beta_r'])[0, 1])
        sym = PASS if r_eff_val >= 0.90 else FAIL
        log.info('  %s allele effects r = %.4f  (threshold >= 0.90)', sym, r_eff_val)

        from mosaiqtl.genotype.cc import FOUNDER_COLORS
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        for _, row in merged.iterrows():
            color = FOUNDER_COLORS.get(row['founder'], '#888888')
            ax.scatter(row['beta_r'], row['beta_python'], s=60, color=color,
                       zorder=3, linewidths=0)
            ax.annotate(row['founder'].split('/')[0], (row['beta_r'], row['beta_python']),
                        fontsize=5, ha='left', va='bottom')
        lo = min(merged['beta_r'].min(), merged['beta_python'].min()) - 0.05
        hi = max(merged['beta_r'].max(), merged['beta_python'].max()) + 0.05
        ax.plot([lo, hi], [lo, hi], color='#444444', linewidth=0.8, linestyle='--')
        ax.set_xlabel('R allele effect (β)', fontsize=8)
        ax.set_ylabel('Python allele effect (β)', fontsize=8)
        ax.set_title(f'Allele effects at peak locus  r={r_eff_val:.4f}', fontsize=9)
        sns.despine(ax=ax)
        fig.tight_layout()
        fig.savefig(cmp_dir / 'p1_allele_effects_scatter.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved compare/p1_allele_effects_scatter.png')
    else:
        log.info('  %s allele_effects_peak.csv not found — effects comparison skipped', SKIP)

    # ─────────────────────────────────────────────────────────────────────────
    # LOD support interval CI (1.5-LOD drop; Dupuis & Siegmund 1999)
    # R: single.chr.plotter.w.ci(...)
    # ─────────────────────────────────────────────────────────────────────────
    chrom_mask = rop_scan.chromosomes == peak_chrom
    ci_lo, ci_hi = lod_support_interval(positions_c, rop_scan.lod[chrom_mask])
    log.info('  LOD support CI: [%.2f, %.2f] Mb  (peak %.2f Mb)',
             ci_lo, ci_hi, peak_pos_mb)

    fig, ax    = plt.subplots(figsize=(7, 3), dpi=300)
    lod_interval(positions_c, rop_scan.lod[chrom_mask], ax=ax, chrom=peak_chrom)
    ax.axvline(ci_lo, color='#888888', linewidth=0.8, linestyle='--', zorder=3)
    ax.axvline(ci_hi, color='#888888', linewidth=0.8, linestyle='--', zorder=3)
    ax.set_title(f'Chr {peak_chrom} LOD + 1.5-LOD support CI', fontsize=9)
    fig.tight_layout()
    fig.savefig(plt_dir / 'p1_chr_lod_with_ci.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_chr_lod_with_ci.png')

    # ─────────────────────────────────────────────────────────────────────────
    # founder / allele effects on peak chr
    # R: genome.plotter.region(haplotype.association=list(miqtl.rop.scan), chr=14)
    #    allele.plotter.region(scan.object=miqtl.rop.scan, chr=14)
    # ─────────────────────────────────────────────────────────────────────────
    betas_c = rop_scan.betas[chrom_mask]   # (n_loci_on_chr, J)

    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
    coef_profile(positions_c, betas_c, ax=ax, chrom=peak_chrom, ylim=3)
    ax.set_title(f'Founder effects — Chr {peak_chrom}', fontsize=9)
    fig.tight_layout()
    fig.savefig(plt_dir / 'p1_founder_effects_region.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_founder_effects_region.png')

    fig, ax = plt.subplots(figsize=(7, 3), dpi=300)
    coef_profile(positions_c, betas_c, ax=ax, chrom=peak_chrom,
                 show_legend=True, ylim=3)
    ax.set_title(f'Allele effects — Chr {peak_chrom}', fontsize=9)
    fig.tight_layout()
    fig.savefig(plt_dir / 'p1_allele_effects_region.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    log.info('  saved plots/p1_allele_effects_region.png')

    # ── summary ───────────────────────────────────────────────────────────────
    log.info('\n── Phase 1 summary ──────────────────────────────────────────────')
    log.info('  outputs:  %s/py_data/    (Python LOD scores)', out_dir)
    log.info('            %s/compare/    (py vs R merged)', out_dir)
    log.info('            %s/plots/      (figures)', out_dir)
    log.info('')
    scored  = {k: v for k, v in results.items() if v is not None}
    n_pass  = sum(scored.values())
    n_total = len(scored)
    for gate, ok in results.items():
        sym = PASS if ok else (SKIP if ok is None else FAIL)
        log.info('  %s %s', sym, gate)
    if n_total == 0:
        log.info('\nNo R reference CSVs found. Run sim_validation.R first.')
        log.info('Python outputs saved to %s/py_data/', out_dir)
    elif n_pass == n_total:
        log.info('\nPhase 1 PASSED (%d/%d) ✅', n_pass, n_total)
    else:
        log.info('\nPhase 1 FAILED (%d/%d) ❌', n_pass, n_total)

    return results


def main():
    """entry point."""
    ap = argparse.ArgumentParser(
        description='Phase 1 validation: minimal miQTL workflow mirror')
    ap.add_argument('--sparcc_cache', default='validation/sparcc_cache',
                    help='path to sparcc HAPPY genome cache')
    ap.add_argument('--r_data',       default='validation/phase1/r_data',
                    help='path to R reference CSVs from sim_validation.R')
    ap.add_argument('--out',          default='validation/phase1',
                    help='output root — writes to out/py_data/, out/compare/, out/plots/')
    args = ap.parse_args()
    run(args.sparcc_cache, args.r_data, args.out)


if __name__ == '__main__':
    main()

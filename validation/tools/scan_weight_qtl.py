"""
scan_weight_qtl.py — per-gene weight QTL: for each gene, test whether the haplotype
at the gene's cis-locus explains body or tissue weight.

Approach mirrors eQTL (S4): one per-chromosome scan of the weight phenotype, shared
across all genes on that chromosome.  Per-gene LOD = peak weight LOD within ±window_mb
of the gene's TSS.  Output is a gene-level table like the eQTL peak_lod.csv.

Three models
------------
  body_weight       body_weight_g    ~ sex + dataset
  tissue_weight     tissue_weight_mg ~ sex + dataset
  tissue_weight_adj tissue_weight_mg ~ sex + dataset + body_weight_g (size-adjusted)

Prerequisites
-------------
  python validation/validate_gastroc.py --build-cache
  → builds gastroc_data/py_data/cache.h5 and kinship_decomp.npz once

Outputs
-------
  gastroc_data/py_data/weight_qtl/{trait}_per_gene.csv     per-gene weight LOD table
  gastroc_data/py_data/weight_qtl/{trait}_perm_lods.npy    null max-LOD distribution
  gastroc_data/py_data/weight_qtl/{trait}_threshold.csv    GEV permutation threshold
  gastroc_data/py_data/weight_qtl/{trait}_effects.csv      founder effects at top gene's locus
  gastroc_data/plots/weight_qtl/{trait}_manhattan.png      gene-level Manhattan
  gastroc_data/plots/weight_qtl/{trait}_effects.png        founder effect bars at top gene

Usage
-----
  conda run -n bioinfo python validation/scan_weight_qtl.py
  conda run -n bioinfo python validation/scan_weight_qtl.py --n_perms 100
  conda run -n bioinfo python validation/scan_weight_qtl.py --chr 8
  conda run -n bioinfo python validation/scan_weight_qtl.py --force
"""

from __future__ import annotations

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
from scipy.stats import chi2 as _chi2_dist

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from mosaiqtl.genotype.founders import FOUNDER_CODE
from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.io.covariates import encode_sex
from mosaiqtl.qtl.effects import get_allele_effects
from mosaiqtl.qtl.lmm import estimate_h2_null, precompute_chromosome, scan_chromosome
from mosaiqtl.qtl.thresholds import get_gev_thresholds
from mosaiqtl.utils import build_z_matrix, rint
from mosaiqtl.viz.effects import effect_bars as _viz_effect_bars
from mosaiqtl.viz.genome import manhattan as _viz_manhattan

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('scan_weight_qtl')

_FOUNDER_CODE_INV: dict[str, str] = {v: k for k, v in FOUNDER_CODE.items()}

GASTROC_DATA = Path('validation/gastroc_data')
PY_DIR       = GASTROC_DATA / 'py_data'
OUT_DIR      = GASTROC_DATA / 'py_data' / 'weight_qtl'
PLOT_DIR     = GASTROC_DATA / 'plots' / 'weight_qtl'


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def _display_founders(cache_founders: list[str]) -> list[str]:
    """Map HAPPY cache codes (A–H) to FOUNDER_ORDER display names (AJ, B6J, …)."""
    return [_FOUNDER_CODE_INV.get(f, _FOUNDER_CODE_INV.get(f.upper(), f))
            for f in cache_founders]


def _build_covariates(meta: pd.DataFrame,
                      sample_order: list[str],
                      extra_cols: list[str] | None = None) -> np.ndarray:
    """Build covariate matrix aligned to sample_order.

    Fixed effects: sex (0=Female ref, 1=Male) + dataset dummies (Founder=ref,
    CC1/CC2/CC3/F1 as indicators).  Optional continuous extras appended last.
    """
    idx  = meta.set_index('sample_id')
    rows = idx.loc[sample_order]

    sex     = encode_sex(rows['sex']).values.astype(np.float64)
    cat     = pd.Categorical(rows['dataset'].values,
                             categories=['Founder', 'CC1', 'CC2', 'CC3', 'F1'])
    dummies = pd.get_dummies(cat, drop_first=True).values.astype(np.float64)
    cols    = [sex, dummies]

    if extra_cols:
        for col in extra_cols:
            vals = rows[col].values.astype(np.float64)
            cols.append(vals.reshape(-1, 1) if vals.ndim == 1 else vals)

    return np.column_stack(cols)


def _build_sample_level_kn(
    geno_ids: list[str],
    cache_strain_ids: list[str],
    kinship_decomp_chr: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Expand LOCO K_m to sample-level K_n = Z K_m Z^T and eigendecompose.

    Returns (z_sub, vsid, evals, evecs) where z_sub is the (n_valid, m) Z-matrix
    restricted to valid samples and vsid is the boolean valid-sample mask.
    """
    z_ch, vsid_ch = build_z_matrix(geno_ids, cache_strain_ids)
    z_sub  = z_ch[vsid_ch]
    ev_m   = kinship_decomp_chr['values']
    evc_m  = kinship_decomp_chr['vectors']
    K_m    = (evc_m * ev_m) @ evc_m.T
    K_n    = z_sub @ K_m @ z_sub.T
    del K_m
    evals, evecs = np.linalg.eigh(K_n)
    del K_n
    return z_sub, vsid_ch, evals, evecs


# ─────────────────────────────────────────────────────────────────────────────
# per-gene weight scan
# ─────────────────────────────────────────────────────────────────────────────

def _scan_weight_per_gene(
    *,
    phenotype: np.ndarray,
    geno_ids: list[str],
    covariates: np.ndarray,
    cache: GenomeCache,
    kinship_decomp: dict,
    gene_info: pd.DataFrame,
    chrs_to_scan: list[str],
    window_mb: float,
) -> pd.DataFrame:
    """For each gene, extract peak weight LOD within ±window_mb of its TSS.

    Runs one scan per chromosome (weight phenotype is constant across all genes
    on that chromosome).  Per-gene LOD = max LOD in the cis window.

    Returns a DataFrame sorted by lod descending with columns:
    gene_id, gene, chr, tss_mb, peak_locus, peak_mb, lod, h2.
    """
    rows: list[dict] = []

    for ch in chrs_to_scan:
        if ch not in kinship_decomp:
            log.warning('  chr%s: no kinship decomp — skipping', ch)
            continue

        ch_genes = gene_info[gene_info['chr'].astype(str) == str(ch)]
        if len(ch_genes) == 0:
            log.info('  chr%s: no genes — skipping', ch)
            continue

        # expand K to sample level and eigendecompose
        z_sub, vsid_ch, evals_ch, evecs_ch = _build_sample_level_kn(
            geno_ids, list(cache.sample_ids), kinship_decomp[ch])
        if vsid_ch.sum() == 0:
            log.warning('  chr%s: no valid samples — skipping', ch)
            continue

        cov_valid = np.hstack(
            [np.ones((z_sub.shape[0], 1)), covariates[vsid_ch]])

        # precompute chromosome-constant rotation (H_t_raw, xt, xt_pinv)
        chr_data = precompute_chromosome(
            cache, ch, evals_ch, evecs_ch, z_mat=z_sub)
        positions = chr_data['positions']   # (n_loci,)
        locus_ids = chr_data['locus_ids']

        # weight phenotype for valid samples, RINT-transformed
        y_valid = rint(phenotype[vsid_ch])

        # estimate h² once (same weight phenotype for all genes on this chr)
        h2, _ = estimate_h2_null(
            evecs_ch.T @ y_valid,
            evecs_ch.T @ cov_valid,
            evals_ch,
        )

        # weight LOD across all loci on this chromosome — one scan, shared for all genes
        chr_res  = scan_chromosome(y_valid, chr_data, cov_valid, evals_ch, evecs_ch, h2)
        lods     = chr_res['lod']           # (n_loci,)
        res_lids = chr_res['locus_ids']

        log.info('  chr%s: %d loci  h²=%.3f  %d genes  weight LOD max=%.3f',
                 ch, len(locus_ids), h2, len(ch_genes),
                 float(np.nanmax(lods)) if len(lods) else 0.0)

        # annotate each gene with its cis-window peak
        for _, grow in ch_genes.iterrows():
            tss_mb = float(grow['start']) / 1e6
            mask   = np.abs(positions - tss_mb) <= window_mb
            if not mask.any():
                continue
            peak_i   = int(np.argmax(lods[mask]))
            _peak_mb = round(float(positions[mask][peak_i]), 3)
            rows.append({
                'gene':        grow.get('gene_name', grow['gene_id']),
                'chr':         str(ch),
                'tss_mb':      round(tss_mb, 3),
                'peak_locus':  str(res_lids[mask][peak_i]),
                'peak_mb':     _peak_mb,
                'cis_dist_mb': round(abs(_peak_mb - tss_mb), 3),
                'lod':         float(lods[mask][peak_i]),
                'h2':          round(float(h2), 4),
            })

        del evals_ch, evecs_ch, chr_data, lods, y_valid, z_sub
        gc.collect()

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('lod', ascending=False).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# permutation threshold
# ─────────────────────────────────────────────────────────────────────────────

def _perm_threshold(
    *,
    phenotype: np.ndarray,
    geno_ids: list[str],
    covariates: np.ndarray,
    cache: GenomeCache,
    kinship_decomp: dict,
    gene_info: pd.DataFrame,
    chrs_to_scan: list[str],
    window_mb: float,
    n_perms: int,
    seed: int,
    perm_path: Path,
    force: bool,
) -> np.ndarray:
    """Cis-window-calibrated permutation threshold for per-gene weight QTL.

    The test statistic for each gene is max LOD within ±window_mb of its TSS.
    The null must match: for each permutation, compute max per-gene cis-window
    LOD across all genes genome-wide.  Using a genome-wide max over all loci
    inflates the threshold by the ratio of total loci to cis-window loci.

    For each chromosome, H_perp and G_pinv are precomputed for cis-window loci
    only (union of all gene windows on that chr).  The batch of n_perms null
    phenotypes is then scored in one vectorised einsum pass, matching the
    computational pattern of scan_chromosome_batch but restricted to the same
    loci the actual scan tests.
    """
    if perm_path.exists() and not force:
        log.info('  loading cached perm null: %s', perm_path)
        return np.load(perm_path)

    # permutation indices generated once at full-sample level so all chromosomes
    # see the same null phenotype realisation (consistent genome-wide max)
    rng           = np.random.default_rng(seed)
    y_full_rint   = rint(phenotype)                          # (n_full,)
    n_full        = len(y_full_rint)
    perm_idx      = np.stack(
        [rng.permutation(n_full) for _ in range(n_perms)], axis=0)  # (n_perms, n_full)

    max_lods = np.zeros(n_perms)

    log.info('  cis-window permutation: %d perms  ±%.1f Mb  %d chromosomes',
             n_perms, window_mb, len(chrs_to_scan))

    for ch in chrs_to_scan:
        if ch not in kinship_decomp:
            continue

        ch_genes = gene_info[gene_info['chr'].astype(str) == str(ch)]
        if len(ch_genes) == 0:
            continue

        z_sub, vsid_ch, evals_ch, evecs_ch = _build_sample_level_kn(
            geno_ids, list(cache.sample_ids), kinship_decomp[ch])
        if vsid_ch.sum() == 0:
            continue

        y_ch    = y_full_rint[vsid_ch]
        cov_ch  = np.hstack([np.ones((z_sub.shape[0], 1)), covariates[vsid_ch]])
        n_valid = int(vsid_ch.sum())

        try:
            h2, _ = estimate_h2_null(evecs_ch.T @ y_ch, evecs_ch.T @ cov_ch, evals_ch)
        except Exception:
            log.warning('  chr%s: h² failed — skipping', ch)
            continue

        chr_data = precompute_chromosome(cache, ch, evals_ch, evecs_ch, z_mat=z_sub)
        positions = chr_data['positions']

        # union of all gene cis windows on this chromosome
        cis_mask = np.zeros(len(positions), dtype=bool)
        if 'start' in ch_genes.columns:
            for _, grow in ch_genes.iterrows():
                tss = float(grow['start']) / 1e6
                cis_mask |= np.abs(positions - tss) <= window_mb
        if not cis_mask.any():
            del chr_data; gc.collect()
            continue

        log.info('  chr%s: %d / %d loci in cis windows  h²=%.3f',
                 ch, int(cis_mask.sum()), len(positions), h2)

        # ── genotype-only precomputation (same for every permutation) ─────────
        w_sqrt  = 1.0 / np.sqrt(h2 * evals_ch + (1.0 - h2))        # (n,)
        xt      = (evecs_ch.T @ cov_ch) * w_sqrt[:, None]            # (n, p)
        xtwx    = xt.T @ xt
        xt_pinv = np.linalg.solve(xtwx, xt.T)                        # (p, n)

        H_t = chr_data['H_t_raw'][cis_mask] * w_sqrt[None, :, None]  # (L_cis, n, J)
        proj   = np.einsum('pn,lnj->lpj', xt_pinv, H_t, optimize=True)
        H_perp = H_t - np.einsum('np,lpj->lnj', xt, proj, optimize=True)
        del H_t, proj, chr_data

        G_all  = np.einsum('lnj,lnk->ljk', H_perp, H_perp, optimize=True)
        G_pinv = np.linalg.pinv(G_all, rcond=1e-10)
        del G_all; gc.collect()

        # ── batch permutation: same perm_idx applied to vsid_ch subset ────────
        # valid-sample indices so permutation maps into the same n_valid subset each chr
        vsid_idx = np.where(vsid_ch)[0]                              # int indices of valid samples
        null_mat_ch = y_full_rint[perm_idx[:, vsid_idx]].T          # (n_valid, n_perms)

        Y_rot = (evecs_ch.T @ null_mat_ch) * w_sqrt[:, None]        # (n, n_perms)
        e0    = Y_rot - xt @ (xt_pinv @ Y_rot)                      # (n, n_perms)
        rss0  = (e0 * e0).sum(axis=0)                               # (n_perms,)

        # two-step rss_red matches scan_chromosome_batch exactly
        c       = np.einsum('lnj,nb->ljb', H_perp, e0, optimize=True)      # (L_cis, J, n_perms)
        Gc      = np.einsum('ljk,lkb->ljb', G_pinv, c, optimize=True)      # (L_cis, J, n_perms)
        rss_red = np.einsum('ljb,ljb->lb',  c, Gc,    optimize=True)        # (L_cis, n_perms)
        rss1    = np.maximum(rss0[None, :] - rss_red, 1e-30)
        with np.errstate(divide='ignore', invalid='ignore'):
            lod = np.where(rss1 < rss0[None, :],
                           (n_valid / 2.0) * np.log10(rss0[None, :] / rss1),
                           0.0)                                       # (L_cis, n_perms)

        # sanity: loop-based rss_red for perm 0 must match batch
        c0       = np.einsum('lnj,n->lj',   H_perp, e0[:, 0], optimize=True)
        Gc0      = np.einsum('ljk,lk->lj',  G_pinv, c0,       optimize=True)
        rsd0     = np.einsum('lj,lj->l',    c0, Gc0,          optimize=True)
        lod0_loop = np.where(
            rss0[0] - rsd0 < rss0[0],
            (n_valid / 2.0) * np.log10(rss0[0] / np.maximum(rss0[0] - rsd0, 1e-30)),
            0.0)
        log.info(
            '  chr%s  n_valid=%d  h2=%.3f  rss0_mean=%.1f'
            '  rss_red_mean=%.4f  max_lod=%.3f'
            '  sanity_perm0 loop=%.4f batch=%.4f match=%s',
            ch, n_valid, h2,
            float(rss0.mean()), float(rss_red.mean()),
            float(lod.max()),
            float(lod0_loop.max()), float(lod[:, 0].max()),
            np.allclose(lod0_loop, lod[:, 0], rtol=1e-4),
        )
        del c0, Gc0, rsd0, lod0_loop
        max_lods = np.maximum(max_lods, lod.max(axis=0))

        del H_perp, G_pinv, c, rss_red, rss1, lod, null_mat_ch, Y_rot, e0
        gc.collect()

    np.save(perm_path, max_lods)
    log.info('  cis perm done  max_lod range [%.2f, %.2f]',
             float(max_lods.min()), float(max_lods.max()))
    return max_lods


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Entry point for per-gene weight QTL mapping."""
    parser = argparse.ArgumentParser(
        description='Per-gene weight QTL: for each gene, test whether its cis-locus '
                    'explains weight (mirrors eQTL approach)',
    )
    parser.add_argument('--n_perms', type=int, default=1000)
    parser.add_argument('--chr', dest='chrs', default=None,
                        help='comma-separated chromosomes (default: all)')
    parser.add_argument('--window_mb', type=float, default=1.0,
                        help='±Mb cis-window around gene TSS (default 1.0)')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    chrs_to_scan: list[str] | None = (
        [c.strip() for c in args.chrs.split(',')] if args.chrs else None
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── metadata ──────────────────────────────────────────────────────────────
    meta_path = GASTROC_DATA / 'metadata.csv'
    if not meta_path.exists():
        log.error('metadata.csv not found: %s', meta_path)
        sys.exit(1)
    meta = pd.read_csv(meta_path)

    for col in ('body_weight_g', 'tissue_weight_mg'):
        if col not in meta.columns:
            log.error('%s column missing from metadata.csv', col)
            sys.exit(1)
        if meta[col].isna().any():
            log.error('%s has %d NaN rows', col, meta[col].isna().sum())
            sys.exit(1)

    sample_order = meta['sample_id'].tolist()
    geno_ids     = meta['genotype'].tolist()   # genotype IDs for Z-matrix

    # ── gene info ─────────────────────────────────────────────────────────────
    gene_info_path = GASTROC_DATA / 'gene_info.csv'
    if not gene_info_path.exists():
        log.error('gene_info.csv not found: %s', gene_info_path)
        sys.exit(1)
    gene_info = pd.read_csv(gene_info_path)
    gene_info['chr'] = gene_info['chr'].astype(str)
    log.info('gene_info: %d genes', len(gene_info))

    # ── GenomeCache ───────────────────────────────────────────────────────────
    hdf5_cache = PY_DIR / 'cache.h5'
    if not hdf5_cache.exists():
        log.error('cache.h5 not found — build with:\n'
                  '  conda run -n bioinfo python validation/validate_gastroc.py --build-cache')
        sys.exit(1)
    cache = GenomeCache.from_hdf5(str(hdf5_cache))
    log.info('cache: %d chr  %d strains  founders: %s',
             len(cache.chromosomes), len(cache.sample_ids), list(cache.founders))

    # ── LOCO kinship decomp ───────────────────────────────────────────────────
    kd_npz = PY_DIR / 'kinship_decomp.npz'
    if kd_npz.exists():
        raw = np.load(kd_npz, allow_pickle=True)
        kinship_decomp = {
            k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
            for k in cache.chromosomes if f'{k}_vals' in raw
        }
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

    chrs_available = sorted(
        [c for c in cache.chromosomes if c in kinship_decomp],
        key=lambda c: (int(c) if c.isdigit() else 100, c),
    )
    if chrs_to_scan is not None:
        chrs_to_scan = [c for c in chrs_to_scan if c in kinship_decomp]
    else:
        chrs_to_scan = chrs_available
    log.info('chromosomes to scan: %s', chrs_to_scan)

    # ── covariates ────────────────────────────────────────────────────────────
    cov_base = _build_covariates(meta, sample_order)
    bw_raw   = (meta.set_index('sample_id')
                    .loc[sample_order, 'body_weight_g']
                    .values.astype(np.float64))
    bw_std   = (bw_raw - bw_raw.mean()) / bw_raw.std()
    cov_adj  = np.column_stack([cov_base, bw_std])

    # ── traits ────────────────────────────────────────────────────────────────
    meta_idx = meta.set_index('sample_id')
    traits = [
        {
            'label':      'body_weight',
            'phenotype':  meta_idx.loc[sample_order, 'body_weight_g'].values.astype(np.float64),
            'covariates': cov_base,
        },
        {
            'label':      'tissue_weight',
            'phenotype':  meta_idx.loc[sample_order, 'tissue_weight_mg'].values.astype(np.float64),
            'covariates': cov_base,
        },
        {
            'label':      'tissue_weight_adj',
            'phenotype':  meta_idx.loc[sample_order, 'tissue_weight_mg'].values.astype(np.float64),
            'covariates': cov_adj,
        },
    ]

    log.info('\n%s', '─' * 70)
    log.info('per-gene weight QTL  window_mb=%.1f  n_perms=%d  seed=%d',
             args.window_mb, args.n_perms, args.seed)
    log.info('%s\n', '─' * 70)

    display_names = _display_founders(list(cache.founders))
    summaries: list[dict] = []

    for t in traits:
        label      = t['label']
        phenotype  = t['phenotype']
        covariates = t['covariates']
        log.info('\n── %s ──────────────────────────────────────────', label)

        per_gene_path = OUT_DIR / f'{label}_per_gene.csv'
        if per_gene_path.exists() and not args.force:
            log.info('  loading cached per-gene LOD: %s', per_gene_path)
            per_gene = pd.read_csv(per_gene_path)
        else:
            t0 = time.perf_counter()
            per_gene = _scan_weight_per_gene(
                phenotype=phenotype,
                geno_ids=geno_ids,
                covariates=covariates,
                cache=cache,
                kinship_decomp=kinship_decomp,
                gene_info=gene_info,
                chrs_to_scan=chrs_to_scan,
                window_mb=args.window_mb,
            )
            elapsed = time.perf_counter() - t0
            per_gene.to_csv(per_gene_path, index=False)
            log.info('  scan done in %.1fs  %d genes  max LOD=%.3f',
                     elapsed, len(per_gene),
                     float(per_gene['lod'].max()) if not per_gene.empty else 0.0)

        if per_gene.empty:
            log.warning('  no genes in output — skipping %s', label)
            continue

        top = per_gene.iloc[0]
        log.info('  top gene: %s  chr%s  %.2f Mb  LOD=%.3f',
                 top['gene'], top['chr'], top['peak_mb'], top['lod'])

        # ── permutation threshold ─────────────────────────────────────────────
        perm_path = OUT_DIR / f'{label}_perm_lods.npy'
        thr_path  = OUT_DIR / f'{label}_threshold.csv'
        max_lods  = _perm_threshold(
            phenotype=phenotype,
            geno_ids=geno_ids,
            covariates=covariates,
            cache=cache,
            kinship_decomp=kinship_decomp,
            gene_info=gene_info,
            chrs_to_scan=chrs_to_scan,
            window_mb=args.window_mb,
            n_perms=args.n_perms,
            seed=args.seed,
            perm_path=perm_path,
            force=args.force,
        )
        thr_95 = float(get_gev_thresholds(max_lods, percentile=0.95)[0]) \
            if len(max_lods) > 0 else float('nan')
        log.info('  GEV 95th pct threshold=%.3f  top LOD=%.3f  %s',
                 thr_95, float(top['lod']),
                 'SIGNIFICANT' if float(top['lod']) > thr_95 else 'sub-threshold')
        pd.DataFrame([{
            'alpha': 0.95,
            'threshold': thr_95, 'n_perms': args.n_perms,
        }]).to_csv(thr_path, index=False)

        per_gene['significant']     = per_gene['lod'] > thr_95
        per_gene['chi2_haplotype']  = 2.0 * np.log(10) * per_gene['lod']
        per_gene['p_haplotype']     = _chi2_dist.sf(
            per_gene['chi2_haplotype'].values, df=7)
        per_gene.to_csv(per_gene_path, index=False)

        # ── founder effects at top gene's peak locus ──────────────────────────
        eff_path = OUT_DIR / f'{label}_effects.csv'
        kd_sub   = {c: kinship_decomp[c] for c in chrs_to_scan if c in kinship_decomp}
        effects: list = []
        try:
            effects = get_allele_effects(
                rint(phenotype),
                cache,
                locus_id=str(top['peak_locus']),
                chrom=str(top['chr']),
                kinship_decomp=kd_sub,
                covariates=covariates,
                sample_ids=geno_ids,
                use_rint=False,
                method='fixef',
            )
            pd.DataFrame([{
                'gene':     top['gene'],
                'locus':    top['peak_locus'],
                'founder':  display_names[i] if i < len(display_names) else e.founder,
                'estimate': e.estimate,
                'se':       e.se,
                'ci_lo':    e.ci_lo,
                'ci_hi':    e.ci_hi,
            } for i, e in enumerate(effects)]).to_csv(eff_path, index=False)
        except Exception as exc:
            log.warning('  founder effects failed: %s', exc)

        # ── gene-level Manhattan ──────────────────────────────────────────────
        scans_dict: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for ch, grp in per_gene.groupby('chr'):
            grp_s = grp.sort_values('tss_mb')
            scans_dict[str(ch)] = (grp_s['tss_mb'].values, grp_s['lod'].values)

        fig = _viz_manhattan(
            scans_dict,
            threshold=thr_95 if np.isfinite(thr_95) else None,
            highlight_chrom=str(top['chr']),
            figsize=(14, 4),
        )
        fig.suptitle(f'{label} — per-gene weight LOD Manhattan', fontsize=10, y=1.01)
        fig.savefig(PLOT_DIR / f'{label}_manhattan.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        log.info('  saved manhattan')

        if effects:
            betas = np.array([e.estimate for e in effects])
            se    = np.array([e.se       for e in effects])
            fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
            _viz_effect_bars(betas, display_names, ax=ax, se=se,
                             ylabel='Founder effect (RINT SD)')
            ax.set_title(
                f'{label}\n{top["gene"]}  chr{top["chr"]}:{top["peak_mb"]:.1f} Mb'
                f'  LOD={top["lod"]:.2f}',
                fontsize=9,
            )
            fig.tight_layout()
            fig.savefig(PLOT_DIR / f'{label}_effects.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            log.info('  saved effect bars')

        summaries.append({
            'trait':        label,
            'top_gene':     top['gene'],
            'chr':          top['chr'],
            'tss_mb':       top.get('tss_mb', float('nan')),
            'peak_mb':      top['peak_mb'],
            'cis_dist_mb':  top.get('cis_dist_mb', float('nan')),
            'peak_lod':     float(top['lod']),
            'h2':           top.get('h2', float('nan')),
            'threshold_95': thr_95,
            'n_perms':      args.n_perms,
            'significant':  float(top['lod']) > thr_95,
            'n_genes':      len(per_gene),
            'n_sig':        int(per_gene['significant'].sum()),
        })

    # ── summary ───────────────────────────────────────────────────────────────
    if summaries:
        summary = pd.DataFrame(summaries)
        summary.to_csv(OUT_DIR / 'weight_qtl_summary.csv', index=False)
        log.info('\n%s\nSUMMARY\n%s', '─' * 70, '─' * 70)
        _hdr = ('%-20s  %-18s  %-5s  %7s  %6s  %5s  %6s  %5s  %s',
                'trait', 'gene', 'chr', 'peak_mb', 'LOD', 'h2', 'thr', 'n_sig', 'status')
        log.info(_hdr[0], *_hdr[1:])
        log.info('  '.join(['-' * w for w in (20, 18, 5, 7, 6, 5, 6, 5, 13)]))
        for _, row in summary.iterrows():
            sig = 'SIGNIFICANT' if row['significant'] else 'sub-threshold'
            log.info('%-20s  %-18s  chr%-2s  %7.2f  %6.3f  %5.3f  %6.3f  %5d  %s',
                     row['trait'], row['top_gene'], row['chr'],
                     row['peak_mb'], row['peak_lod'], row.get('h2', float('nan')),
                     row['threshold_95'], row['n_sig'], sig)
    log.info('\nOutputs: %s', OUT_DIR)
    log.info('Plots:   %s', PLOT_DIR)


if __name__ == '__main__':
    main()

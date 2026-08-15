"""
check_mediation_h2_approx.py

Empirical check: does fixing h2 from the base model (Option B approximation)
give materially different LOD scores vs per-mediator h2 estimation (current code)?

Uses Phase 3 cache + phenotypes (50 sims, already built).
Tests several (outcome, mediator) pairs and reports h2 drift + LOD differences.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.qtl import lmm as lmm_module
from mosaiqtl.utils import build_z_matrix

PY_DATA  = ROOT / 'validation' / 'phase3' / 'py_data'
R_DATA   = ROOT / 'validation' / 'phase3' / 'r_data'
CACHE_H5 = PY_DATA / 'cache.h5'
KD_NPZ   = PY_DATA / 'kinship_decomp_p3.npz'
PHENO    = R_DATA  / 'phenotypes.csv'

# ── load data ──────────────────────────────────────────────────────────────────
print('loading cache...')
cache = GenomeCache.from_hdf5(str(CACHE_H5))
cache_ids = list(cache.sample_ids)

print('loading kinship decomp...')
raw = np.load(KD_NPZ, allow_pickle=True)
kinship_decomp = {k: {'values': raw[f'{k}_vals'], 'vectors': raw[f'{k}_vecs']}
                  for k in cache.chromosomes}
del raw

print('loading phenotypes...')
pheno_df   = pd.read_csv(PHENO)
sample_ids = pheno_df['SUBJECT.NAME.1'].values
sim_ys     = {i: pheno_df[f'sim.y.{i}'].values.astype(float)
              for i in range(1, 11) if f'sim.y.{i}' in pheno_df.columns}

# align sample_ids → cache_ids
z_mat, valid_sid = build_z_matrix(list(sample_ids), cache_ids)

# pick 5 outcome/mediator pairs spanning different sim indices
pairs = [(1, 2), (1, 3), (2, 3), (3, 4), (5, 6)]

# test on peak chromosome of sim.y.1 (to match S9 scope)
# find peak chrom from existing s4 output
s4 = pd.read_csv(PY_DATA / 's4_lod_py.csv')
peak_chrom = str(s4.loc[s4['lod'].idxmax(), 'chromosome'])
print(f'peak chromosome: {peak_chrom}')

chrom     = peak_chrom
ev        = kinship_decomp[chrom]['values']
evc       = kinship_decomp[chrom]['vectors']


def _reduce(y_raw):
    """valid mask + strain means via Z."""
    valid = (~np.isnan(y_raw)) & valid_sid
    y     = y_raw[valid]
    z_sub = z_mat[valid]
    r_counts = z_sub.sum(axis=0)
    return (z_sub.T @ y) / r_counts, valid


def _rot(vec, h2, ev_v):
    delta  = h2 / max(1 - h2, 1e-10)
    w_inv  = 1.0 / (delta * ev + 1.0)
    w_sqrt = np.sqrt(w_inv)
    return (ev_v.T @ vec) * w_sqrt, w_inv, w_sqrt


def _rss0_wls(y_rot, x_rot):
    xw   = x_rot                       # already weighted (w_sqrt applied)
    beta = np.linalg.lstsq(xw.T @ xw, xw.T @ y_rot, rcond=None)[0]
    e    = y_rot - xw @ beta
    return float(e @ e)


def _batch_lod_simple(y_rot, x_rot, H_rot, n, rss0):
    """same FWL batch as mediation.py but minimal for comparison."""
    xt_pinv = np.linalg.lstsq(x_rot.T @ x_rot, x_rot.T, rcond=None)[0]
    proj    = np.einsum('pn,lnj->lpj', xt_pinv, H_rot, optimize=True)
    H_perp  = H_rot - np.einsum('np,lpj->lnj', x_rot, proj, optimize=True)
    c       = np.einsum('lnj,n->lj', H_perp, y_rot, optimize=True)
    G       = np.einsum('lnj,lnk->ljk', H_perp, H_perp, optimize=True)
    betas   = (np.linalg.pinv(G, rcond=1e-10) @ c[..., None]).squeeze(-1)
    rss_red = np.einsum('lj,lj->l', c, betas)
    rss1    = np.maximum(rss0 - rss_red, 1e-30)
    return np.where(rss1 < rss0, (n / 2.0) * np.log10(rss0 / rss1), 0.0)


print(f'\n{"pair":<12} {"h2_base":>8} {"h2_med_true":>12} {"h2_drift":>10} '
      f'{"lod_r_exact":>12} {"lod_r_fixed":>12} {"max|Δlod|":>10}')
print('-' * 78)

lids, pos, probs = cache.get_chromosome(chrom)    # (L, n_strains, J)
n_loci = len(lids)

for out_i, med_i in pairs:
    y_raw = sim_ys[out_i]
    m_raw = sim_ys[med_i]

    y_strain, valid_y = _reduce(y_raw)
    m_strain, valid_m = _reduce(m_raw)
    n = len(y_strain)

    ev_v = evc   # already at strain level (kinship built on cache strains)

    x_base = np.ones((n, 1))
    m_std  = (m_strain - m_strain.mean()) / max(m_strain.std(), 1e-10)
    x_med  = np.hstack([m_std[:, None], x_base])

    # ── exact: h2 from mediated model ────────────────────────────────────────
    h2_base, _ = lmm_module.estimate_h2_null(ev_v.T @ y_strain, ev_v.T @ x_base, ev)
    h2_med_true, _ = lmm_module.estimate_h2_null(ev_v.T @ y_strain, ev_v.T @ x_med, ev)

    y_rot_exact, w_inv_e, w_sqrt_e = _rot(y_strain, h2_med_true, ev_v)
    x_rot_exact = (ev_v.T @ x_med) * w_sqrt_e[:, None]
    rss0_exact  = _rss0_wls(y_rot_exact, x_rot_exact)
    H_rot_exact = np.einsum('nk,lnj->lkj', ev_v, probs) * w_sqrt_e[None, :, None]
    lod_exact   = _batch_lod_simple(y_rot_exact, x_rot_exact, H_rot_exact, n, rss0_exact)

    # ── approx: h2 fixed from base model ─────────────────────────────────────
    y_rot_fixed, w_inv_f, w_sqrt_f = _rot(y_strain, h2_base, ev_v)
    x_rot_fixed = (ev_v.T @ x_med) * w_sqrt_f[:, None]
    rss0_fixed  = _rss0_wls(y_rot_fixed, x_rot_fixed)
    H_rot_fixed = np.einsum('nk,lnj->lkj', ev_v, probs) * w_sqrt_f[None, :, None]
    lod_fixed   = _batch_lod_simple(y_rot_fixed, x_rot_fixed, H_rot_fixed, n, rss0_fixed)

    r_exact = float(np.corrcoef(lod_exact, lod_fixed)[0, 1])
    r_fixed = r_exact   # same vector, just checking internal consistency below
    # Pearson r between exact and fixed LOD vectors
    r_ef    = float(np.corrcoef(lod_exact, lod_fixed)[0, 1])
    max_d   = float(np.max(np.abs(lod_exact - lod_fixed)))
    h2_drift = h2_med_true - h2_base

    print(f'({out_i},{med_i}){"":>6} {h2_base:>8.4f} {h2_med_true:>12.4f} '
          f'{h2_drift:>+10.4f} {"exact↔fixed r":>12} {r_ef:>12.6f} {max_d:>10.4f}')

print('\nInterpretation:')
print('  h2_drift ≈ 0       → fixing h2 is safe')
print('  r(exact,fixed) ≈ 1 → LOD ranking preserved (Spearman gate unaffected)')
print('  max|Δlod| < 0.1    → absolute LOD shift negligible')

# validate_nebula_ln.py — compare native NEBULA-LN against edgepython
#
# validates two independent correctness properties:
#   1. woodbury V^{-1} exactness vs direct inversion at small n (local, no edgepython)
#   2. native fit_null_sc_native vs edgepython glm_sc_fit on real data (HPC only)
#
# outputs to: validation/phase3/py_data/nebula_ln_comparison.csv
#             validation/phase3/plots/
#
# usage:
#   python validation/validate_nebula_ln.py               # property 1 only (no edgepython)
#   python validation/validate_nebula_ln.py --h5ad /path  # both properties (HPC)

import os, sys, argparse, logging
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from mosaiqtl.nb import (fit_null_nb_glmm_loco, _score_stats_loco,
                           fit_null_sc_native, _wb, _build_M)

PKG_DIR   = os.path.dirname(os.path.dirname(__file__))
PY_DATA   = os.path.join(PKG_DIR, 'validation', 'phase3', 'py_data')
PLOTS_DIR = os.path.join(PKG_DIR, 'validation', 'phase3', 'plots')
for d in (PY_DATA, PLOTS_DIR): os.makedirs(d, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger('validate_nebula_ln')

ap = argparse.ArgumentParser()
ap.add_argument('--h5ad',    default=None,  help='path to gastroc_all_processed2.h5ad (HPC)')
ap.add_argument('--cell_type', default='type2x')
ap.add_argument('--n_genes',   type=int, default=200)
ap.add_argument('--n_samp',    type=int, default=418)
args = ap.parse_args()

results = []

# ══════════════════════════════════════════════════════════════════════════════
# P1 — woodbury exactness (local, no edgepython)
# ══════════════════════════════════════════════════════════════════════════════
log.info('\n▶ P1 — Woodbury V^{-1} exactness')

np.random.seed(42)
n_wb, m_wb = 40, 8
rng = np.random.default_rng(0)
K_wb = np.eye(m_wb) + rng.random((m_wb, m_wb)) * 0.1
K_wb = (K_wb + K_wb.T) / 2; np.fill_diagonal(K_wb, np.diag(K_wb) + 0.5)
Z_wb = np.zeros((n_wb, m_wb))
for i in range(n_wb): Z_wb[i, i % m_wb] = 1.0
W_wb = rng.uniform(0.5, 3.0, n_wb)
tau  = 0.8

# explicit V_inv
K_n  = Z_wb @ K_wb @ Z_wb.T
V    = np.diag(1 / W_wb) + tau * K_n
V_inv_ex = np.linalg.inv(V)

# woodbury V_inv
M_wb    = _build_M(K_wb, tau, W_wb, Z_wb)
M_inv_w = np.linalg.inv(M_wb)

max_err = 0.0
for _ in range(100):
    v = rng.standard_normal(n_wb)
    ex  = V_inv_ex @ v
    wb  = _wb(v, W_wb, Z_wb, M_inv_w)
    max_err = max(max_err, float(np.abs(ex - wb).max()))

wb_pass = max_err < 1e-8
log.info(f'  max|V_inv_exact - V_inv_woodbury| = {max_err:.2e}  (threshold 1e-8)')
results.append({'section': 'woodbury_vinv', 'pass': wb_pass,
                'detail': f'max_err={max_err:.2e}'})

# LOD score exactness at n=40, L=50 loci
S_wb  = rng.negative_binomial(10, 0.5, n_wb).astype(float) + 1
x_wb  = np.ones((n_wb, 1))
off_wb = np.zeros(n_wb)
fit_w  = fit_null_nb_glmm_loco(S_wb, x_wb, off_wb, K_wb, Z_wb, phi=0.3)

H_wb = rng.dirichlet(np.ones(8), (50, n_wb))[:, :, :7]

# explicit P
K_n2  = Z_wb @ K_wb @ Z_wb.T
V2    = np.diag(1/fit_w['W']) + fit_w['tau'] * K_n2
Vi2   = np.linalg.inv(V2); ViX2 = Vi2 @ x_wb
P_ex  = Vi2 - ViX2 @ np.linalg.solve(x_wb.T @ ViX2, ViX2.T)
z_fin = (np.log(fit_w['mu']) - off_wb) + (S_wb - fit_w['mu']) / fit_w['mu']
r_ex  = P_ex @ z_fin
PH_ex = np.einsum('mn,lnj->lmj', P_ex, H_wb, optimize=True)
I_ex  = np.einsum('lnj,lnk->ljk', H_wb, PH_ex, optimize=True) + np.eye(7)[None]*1e-10
lod_ex = np.array([max(float(r_ex@H_wb[l]@np.linalg.solve(I_ex[l],H_wb[l].T@r_ex)),0)
                    for l in range(50)]) / (2*np.log(10))

kw   = dict(W=fit_w['W'], Z=fit_w['Z'], M_inv=fit_w['M_inv'],
            VinvX=fit_w['VinvX'], XtVinvX_inv=fit_w['XtVinvX_inv'])
lod_wb = _score_stats_loco(H_wb, fit_w['r_adj'], **kw) / (2*np.log(10))

lod_err = float(np.abs(lod_ex - lod_wb).max())
lod_pass = lod_err < 1e-6
log.info(f'  max|LOD_explicit - LOD_woodbury| = {lod_err:.2e}  (threshold 1e-6)')
results.append({'section': 'woodbury_lod', 'pass': lod_pass,
                'detail': f'max_err={lod_err:.2e}'})

# plot: LOD comparison
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].scatter(lod_ex, lod_wb, s=15, color='#4F6EAF', edgecolors='none', alpha=0.8)
lim = max(lod_ex.max(), lod_wb.max()) * 1.05
axes[0].plot([0, lim], [0, lim], '--', color='#D83026', lw=1)
axes[0].set(xlim=(0,lim), ylim=(0,lim),
            xlabel='LOD (explicit P)', ylabel='LOD (Woodbury)',
            title=f'LOD equivalence  max|Δ|={lod_err:.1e}')
sns.despine(ax=axes[0])
axes[1].scatter(range(50), lod_ex - lod_wb, s=10, color='#4F6EAF',
                edgecolors='none', alpha=0.8)
axes[1].axhline(0, color='#D83026', lw=1, ls='--')
axes[1].set(xlabel='locus index', ylabel='LOD_explicit - LOD_woodbury',
            title='residuals (should be ≈ 0)')
sns.despine(ax=axes[1])
plt.tight_layout()
fig.savefig(os.path.join(PLOTS_DIR, 'nebula_ln_woodbury_lod.png'), dpi=300)
plt.close()
log.info('  plot saved: nebula_ln_woodbury_lod.png')

# ══════════════════════════════════════════════════════════════════════════════
# P2 — native NEBULA-LN vs edgepython (HPC only)
# ══════════════════════════════════════════════════════════════════════════════
if args.h5ad is None:
    log.info('\n⚠ P2 — skipped (no --h5ad; HPC only)')
    results.append({'section': 'nebula_vs_edgepython', 'pass': None,
                    'detail': 'skipped — run on HPC with --h5ad'})
else:
    log.info(f'\n▶ P2 — native NEBULA-LN vs edgepython  ({args.cell_type}, {args.n_genes} genes)')
    import scanpy as sc
    adata = sc.read_h5ad(args.h5ad)
    ct    = adata[adata.obs['cell_type'] == args.cell_type]
    genes = ct.var_names[:args.n_genes].tolist()
    counts= ct[:, genes].X.toarray() if hasattr(ct[:, genes].X, 'toarray') else ct[:, genes].X
    counts= counts.T.astype(np.float64)         # (n_genes, n_cells)
    sids  = ct.obs['sample_id'].values

    # native
    log.info('  running native fit_null_sc_native...')
    out_nat = fit_null_sc_native(counts, sids, verbose=True)

    # edgepython
    try:
        import edgepython as ep
        from mosaiqtl.nb import fit_null_sc
        log.info('  running edgepython glm_sc_fit...')
        out_ep = fit_null_sc(counts, sids, use_native=False)
        ep_ok  = True
    except ImportError:
        log.info('  edgepython not available — skipping P2 comparison')
        ep_ok = False

    if ep_ok:
        mask_both = out_nat['gene_mask'] & out_ep['gene_mask']
        phi_nat   = out_nat['dispersion_use'][out_nat['gene_mask'][out_nat['gene_mask']]]
        phi_ep    = out_ep['dispersion_use'][out_ep['gene_mask'][mask_both[:out_ep['gene_mask'].sum()]]]
        # align by gene mask
        genes_nat = np.where(out_nat['gene_mask'])[0]
        genes_ep  = np.where(out_ep['gene_mask'])[0]
        shared    = np.intersect1d(genes_nat, genes_ep)
        idx_nat   = np.searchsorted(genes_nat, shared)
        idx_ep    = np.searchsorted(genes_ep,  shared)
        phi_n = out_nat['dispersion_use'][idx_nat]
        phi_e = out_ep['dispersion_use'][idx_ep]
        sig_n = out_nat['sigma_sample'][idx_nat]
        sig_e = out_ep['sigma_sample'][idx_ep]

        r_phi = float(np.corrcoef(phi_n, phi_e)[0,1]) if len(phi_n) > 3 else np.nan
        r_sig = float(np.corrcoef(sig_n, sig_e)[0,1]) if len(sig_n) > 3 else np.nan
        log.info(f'  n_shared={len(shared)}  phi_r={r_phi:.4f}  sigma2_r={r_sig:.4f}')

        comp_df = pd.DataFrame({'gene_idx': shared,
                                 'phi_native': phi_n, 'phi_ep': phi_e,
                                 'sigma2_native': sig_n, 'sigma2_ep': sig_e})
        comp_df.to_csv(os.path.join(PY_DATA, 'nebula_ln_comparison.csv'), index=False)

        # plots
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for ax, xv, yv, xlabel, title, r in [
            (axes[0], phi_e, phi_n, 'phi (edgepython)', f'cell-level dispersion  r={r_phi:.4f}', r_phi),
            (axes[1], sig_e, sig_n, 'sigma2 (edgepython)', f'subject overdispersion  r={r_sig:.4f}', r_sig),
        ]:
            ax.scatter(xv, yv, s=3, alpha=0.5, color='#4F6EAF', edgecolors='none')
            lim = max(xv.max(), yv.max()) * 1.05
            ax.plot([0,lim],[0,lim],'--',color='#D83026',lw=1)
            ax.set(xlim=(0,lim), ylim=(0,lim), xlabel=xlabel,
                   ylabel='native NEBULA-LN', title=title)
            sns.despine(ax=ax)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, 'nebula_ln_vs_edgepython.png'), dpi=300)
        plt.close()
        log.info('  plot saved: nebula_ln_vs_edgepython.png')

        nb_pass = (not np.isnan(r_phi)) and r_phi > 0.99
        results.append({'section': 'nebula_vs_edgepython', 'pass': nb_pass,
                        'detail': f'n_shared={len(shared)}  phi_r={r_phi:.4f}  sigma2_r={r_sig:.4f}'})

# ── summary ──────────────────────────────────────────────────────────────────
pd.DataFrame(results).to_csv(os.path.join(PY_DATA, 'nebula_ln_qc.csv'), index=False)
log.info('\n' + '='*55)
log.info(f'{"section":<30} {"status":<6} detail')
log.info('-'*55)
for r in results:
    st = 'PASS' if r['pass'] is True else ('SKIP' if r['pass'] is None else 'FAIL')
    log.info(f'{r["section"]:<30} {st:<6} {r["detail"]}')

# plot_phase2.py — plot helpers and per-section figure functions for phase 2 validation.
# call init(plots_dir, data_dir, logger) before any plot/save function.
# importable from validate_port.py and validate_phase3.py.

import numpy as np
import pandas as pd
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from scipy.stats import genextreme as _gev

sns.set_style('whitegrid')
plt.rcParams.update({'font.family': 'sans-serif',
                     'font.sans-serif': ['Carlito', 'DejaVu Sans'],
                     'font.size': 11})

# module-level state — set by init()
PLOTS   = None
PY_DATA = None
_log    = None

# canonical palettes (shared with validate_port.py via import)
FOUNDER_COLORS = {
    'AJ': '#F4C245', 'B6J': '#C0BFBF', '129S1J': '#DA9CC1', 'NODJ': '#4F6EAF',
    'NZOJ': '#52A5DB', 'CASTJ': '#55AF5B', 'PWKJ': '#D83026', 'WSBJ': '#683C91'}
CHR_PAL = ['#4F6EAF', '#888780']

_chr_sort = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper() == 'X' else 99)


def chr_colors(labels, chr_order):
    '''alternating CHR_PAL colors per chromosome label; used in manhattan-style plots.'''
    return [CHR_PAL[chr_order.index(c) % 2] if c in chr_order else '#aaa' for c in labels]


def init(plots_dir, data_dir, logger=None):
    '''set module-level output dirs and optional logger; call once before any plot/save.'''
    global PLOTS, PY_DATA, _log
    PLOTS, PY_DATA, _log = plots_dir, data_dir, logger


# ── core helpers ──────────────────────────────────────────────────────────────

def save_plot(fig, name):
    '''save fig to PLOTS at dpi=300 and close.'''
    fig.savefig(os.path.join(PLOTS, name), dpi=300, bbox_inches='tight')
    plt.close(fig)
    if _log: _log.debug(f'saved {name}')


def save_csv(df, name):
    '''save df to PY_DATA; return path.'''
    p = os.path.join(PY_DATA, name)
    df.to_csv(p, index=False)
    if _log: _log.debug(f'saved {name} ({len(df)} rows)')
    return p


def scatter_qc(ax, x, y, xlabel, ylabel, title):
    '''scatter with y=x reference line and pearson r annotation.'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    lim = max(float(np.nanmax(np.abs(np.concatenate([x, y])))), 0.1) * 1.05
    ax.scatter(x, y, s=3, alpha=1.0, color='#4F6EAF', edgecolors='none')
    ax.plot([0, lim], [0, lim], '--', color='#D83026', lw=1)
    r = float(np.corrcoef(x, y)[0, 1]) if len(x) > 3 else np.nan
    ax.text(0.05, 0.95, f'$r$={r:.4f}', transform=ax.transAxes, fontsize=9, va='top')
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=xlabel, ylabel=ylabel, title=title)
    sns.despine(ax=ax)
    return r


def ccc(x, y):
    '''concordance correlation coefficient (Lin 1989).
    equal to 1 only when both means and variances match; stricter than pearson r.'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y); x, y = x[ok], y[ok]
    if len(x) < 3: return np.nan
    mx, my = x.mean(), y.mean()
    return float(2 * np.cov(x, y)[0, 1] / (x.var() + y.var() + (mx - my) ** 2))


def bland_altman(ax, x, y, xlabel, title, color='#4F6EAF'):
    '''bland-altman method comparison (Bland & Altman 1986).
    reveals systematic additive bias invisible to pearson r.'''
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    mv = ((x + y) / 2)[ok]; dv = (y - x)[ok]
    md, sd = float(np.mean(dv)), float(np.std(dv))
    step = max(1, len(mv) // 5000)
    ax.scatter(mv[::step], dv[::step], s=2, alpha=0.4, color=color, edgecolors='none')
    ax.axhline(md,           color='#D83026', lw=1.2, ls='-',  label=f'bias={md:.4f}')
    ax.axhline(md + 1.96*sd, color='#D83026', lw=0.8, ls='--', label=f'ULoA={md+1.96*sd:.4f}')
    ax.axhline(md - 1.96*sd, color='#D83026', lw=0.8, ls='--', label=f'LLoA={md-1.96*sd:.4f}')
    ax.axhline(0, color='#888780', lw=0.5, ls=':')
    ax.set(xlabel=xlabel, ylabel='difference (py − R)', title=title)
    ax.legend(fontsize=7); sns.despine(ax=ax)
    return md, sd


def ecdf_plot(ax, data_list, labels, colors, xlabel, title):
    '''empirical cdf; more sensitive to tail differences than histograms (Phipson & Smyth 2010).'''
    for data, label, color in zip(data_list, labels, colors):
        d = np.sort(np.asarray(data, float)); d = d[np.isfinite(d)]
        ax.plot(d, np.linspace(0, 1, len(d)), color=color, lw=1.5, label=label)
    ax.set(xlabel=xlabel, ylabel='cumulative probability', title=title,
           xlim=(0, None), ylim=(0, 1))
    ax.legend(fontsize=8); sns.despine(ax=ax)


# ── per-section plot functions ────────────────────────────────────────────────

def plot_s1(info_df, py_founders, r_founders, py_sids, r_sids, fnd_match, sid_match):
    '''s1: loci-per-chr grouped bar + cache metadata table.'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]; x = np.arange(len(info_df)); w = 0.35
    ax.bar(x - w/2, info_df['n_loci_py'], w, color='#4F6EAF', edgecolor='none',
           label='mosaiQTL (Python)')
    ax.bar(x + w/2, info_df['n_loci_r'].fillna(0), w, color='#55AF5B',
           edgecolor='none', alpha=0.85, label='miQTL (R)')
    ax.set(xticks=x, xticklabels=info_df['chr'].tolist(),
           xlabel='chromosome', ylabel='loci', title='loci per chromosome')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); sns.despine(ax=ax)
    axes[1].axis('off')
    tbl = [['', 'Python', 'R'],
           ['n_founders', str(len(py_founders)), str(len(r_founders))],
           ['n_samples',  str(len(py_sids)),     str(len(r_sids))],
           ['founders match', '—', str(fnd_match)],
           ['samples match',  '—', str(sid_match)]]
    t = axes[1].table(tbl, loc='center', cellLoc='center')
    t.auto_set_font_size(False); t.set_fontsize(10)
    axes[1].set_title('cache metadata', fontsize=11)
    plt.tight_layout(); save_plot(fig, 's1_cache_info.png')


def plot_s2_pyonly(H_py0, fnames, disp_s, fs, rep_chr):
    '''s2 fallback: python-only heatmap when R haplotype_vals_r.csv is missing.'''
    fig_hm, ax_hm = plt.subplots(figsize=(7, 5))
    sns.heatmap(H_py0, ax=ax_hm, cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=fnames, yticklabels=disp_s, cbar_kws={'shrink': 0.7})
    ax_hm.set_title(f'Python haplotype probs  chr{rep_chr}', fontsize=9)
    ax_hm.set_xticklabels(ax_hm.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax_hm.set_yticklabels(ax_hm.get_yticklabels(), rotation=0, fontsize=fs)
    ax_hm.text(0.5, -0.12,
               'R comparison needs haplotype_vals_r.csv — run R S2 first',
               transform=ax_hm.transAxes, ha='center', fontsize=8, color='#D83026')
    plt.tight_layout(); save_plot(fig_hm, 's2_haplotype_heatmaps.png')


def plot_s2(hmap_data, chr_order, fnames, disp_s,
            all_r_probs, all_py_probs, val_r, val_mad, val_rows_all,
            fc_r=None, rep_chr=None):
    '''s2: haplotype prob heatmaps (3-panel + all-chr diff grid) + prob scatter
    + bland-altman + calibration reliability diagram.
    call plot_s2_pyonly separately when hmap_data is empty.'''
    _fs = max(4, min(8, int(160 / max(len(disp_s), 1))))

    # ── heatmaps (3-panel + all-chr diff grid) ────────────────────────────────
    if hmap_data:
        if rep_chr is None:
            rep_chr = '8' if '8' in hmap_data else list(hmap_data.keys())[0]
        H_py_r, H_r_r, rep_lid = hmap_data[rep_chr]
        diff_r = H_py_r - H_r_r
        ab_r   = max(abs(diff_r.min()), abs(diff_r.max())) + 1e-9
        fig_hm, axes_hm = plt.subplots(1, 3, figsize=(18, 5))
        for ax, mat, title, cmap, vmin, vmax in [
            (axes_hm[0], H_py_r, f'Python HMM  chr{rep_chr} / {rep_lid}', 'YlOrRd', 0, 1),
            (axes_hm[1], H_r_r,  f'R HAPPY  chr{rep_chr} / {rep_lid}',    'YlOrRd', 0, 1),
            (axes_hm[2], diff_r, f'Difference (Python-R)  max|delta|={ab_r:.3f}', 'RdBu_r', -ab_r, ab_r),
        ]:
            sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                        xticklabels=fnames[:mat.shape[1]], yticklabels=disp_s,
                        cbar_kws={'shrink': 0.7})
            ax.set_title(title, fontsize=9)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=_fs)
        plt.tight_layout(); save_plot(fig_hm, 's2_haplotype_heatmaps.png')

        chrs_d = [c for c in chr_order if c in hmap_data]
        ncols = 5; nrows = int(np.ceil(len(chrs_d) / ncols))
        fig_gd, axs = plt.subplots(nrows, ncols, figsize=(ncols*3.2, nrows*3.0))
        axs_flat = np.array(axs).flatten()
        for ai, chrom in enumerate(chrs_d):
            H_py_c, H_r_c, _ = hmap_data[chrom]
            diff_c = H_py_c - H_r_c
            ab_c = max(abs(diff_c.min()), abs(diff_c.max())) + 1e-9
            sns.heatmap(diff_c, ax=axs_flat[ai], cmap='RdBu_r', vmin=-ab_c, vmax=ab_c,
                        xticklabels=False, yticklabels=False, cbar_kws={'shrink': 0.5})
            axs_flat[ai].set_title(f'chr{chrom}  |d|={ab_c:.3f}', fontsize=8)
        for ai in range(len(chrs_d), len(axs_flat)):
            axs_flat[ai].set_visible(False)
        plt.suptitle('Haplotype prob difference (Python - R) — all chromosomes', fontsize=10)
        plt.tight_layout(); save_plot(fig_gd, 's2_haplotype_diff_grid.png')

    # ── probability scatter + bland-altman + calibration ─────────────────────
    if val_rows_all:
        rpa = np.array(all_r_probs); ppa = np.array(all_py_probs)
        ok  = np.isfinite(rpa) & np.isfinite(ppa)
        idx_s = np.random.default_rng(0).choice(np.where(ok)[0],
                                                  min(20000, ok.sum()), replace=False)
        vdf = pd.DataFrame(val_rows_all).dropna()
        _fc_r = fc_r if fc_r is not None else vdf['founder'].unique().tolist()
        fnd_r_vals = (vdf.groupby('founder')
                      .apply(lambda g: float(np.corrcoef(g['r_prob'], g['py_prob'])[0, 1])
                             if len(g) > 3 else np.nan)
                      .reindex(_fc_r).fillna(0))
        fig_sc, axes_sc = plt.subplots(1, 2, figsize=(11, 4))
        axes_sc[0].scatter(rpa[idx_s], ppa[idx_s], s=2, alpha=0.3,
                           color='#4F6EAF', edgecolors='none')
        axes_sc[0].plot([0, 1], [0, 1], '--', color='#D83026', lw=1)
        axes_sc[0].set(xlim=(0, 1.05), ylim=(0, 1.05),
                       xlabel='prob (R HAPPY, marginalised)', ylabel='prob (Python HMM)',
                       title=f'actual probability values  r={val_r:.4f}  mad={val_mad:.4f}')
        sns.despine(ax=axes_sc[0])
        fc_colors = [FOUNDER_COLORS.get(f, '#888') for f in fnd_r_vals.index]
        axes_sc[1].bar(range(len(fnd_r_vals)), fnd_r_vals.values,
                       color=fc_colors, edgecolor='none')
        axes_sc[1].axhline(0.90, color='#D83026', lw=1, ls='--', label='r=0.90')
        axes_sc[1].set(xticks=range(len(fnd_r_vals)),
                       xlabel='founder', ylabel='Pearson $r$ (R vs Python)',
                       title='per-founder probability correlation', ylim=(0, 1))
        axes_sc[1].set_xticklabels(list(fnd_r_vals.index), rotation=45, ha='right', fontsize=9)
        axes_sc[1].legend(fontsize=8); sns.despine(ax=axes_sc[1])
        plt.tight_layout(); save_plot(fig_sc, 's2_prob_comparison.png')

        # bland-altman: reveals systematic additive bias (Bland & Altman 1986)
        fig_ba, ax_ba = plt.subplots(figsize=(6, 4))
        bland_altman(ax_ba, rpa[ok], ppa[ok],
                     'mean probability  (R + Python) / 2',
                     'Bland-Altman: haplotype probs (all chr)')
        plt.tight_layout(); save_plot(fig_ba, 's2_bland_altman.png')

        # calibration reliability diagram (DeGroot & Fienberg 1983)
        fig_cal, ax_cal = plt.subplots(figsize=(5, 5))
        _bins = np.linspace(0, 1, 11)
        _bid  = np.clip(np.digitize(ppa[ok], _bins) - 1, 0, 9)
        _bp   = np.array([ppa[ok][_bid == b].mean() if (_bid == b).any() else np.nan for b in range(10)])
        _br   = np.array([rpa[ok][_bid == b].mean() if (_bid == b).any() else np.nan for b in range(10)])
        _bn   = np.array([(_bid == b).sum() for b in range(10)])
        ax_cal.scatter(_bp, _br, s=_bn / max(_bn) * 200 + 20,
                       color='#4F6EAF', edgecolors='none', alpha=0.85, label='decile bins')
        ax_cal.plot([0, 1], [0, 1], '--', color='#D83026', lw=1, label='perfect calibration')
        ax_cal.set(xlim=(0, 1), ylim=(0, 1),
                   xlabel='mean Python prob (bin)', ylabel='mean R prob (bin)',
                   title='probability calibration (reliability diagram)')
        ax_cal.legend(fontsize=8); sns.despine(ax_cal)
        plt.tight_layout(); save_plot(fig_cal, 's2_calibration.png')
    else:
        # placeholder when R data missing
        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
        ax_sc.text(0.5, 0.5,
                   'Run R S2 first:\nRscript validation/r_reference/port_validation.R all Inf 2',
                   ha='center', va='center', transform=ax_sc.transAxes,
                   fontsize=10, color='#D83026')
        ax_sc.set_title('actual prob comparison — R data missing'); sns.despine(ax=ax_sc)
        save_plot(fig_sc, 's2_prob_comparison.png')


def plot_s3(kp, kr, disp, chrom_k, ab):
    '''s3: 3-panel kinship heatmap + off-diagonal scatter + diagonal/off-diagonal histograms.
    kp/kr: numpy arrays (n_shared x n_shared) in canonical display order.'''
    n = len(disp); fs = max(5, min(8, int(200 / n)))
    # 3-panel heatmap
    dk = kp - kr
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title, cmap, vmin, vmax in [
        (axes[0], kp, f'Python LOCO\nchr{chrom_k}',          'YlOrRd', 0, 1),
        (axes[1], kr, f'R LOCO\nchr{chrom_k}',               'YlOrRd', 0, 1),
        (axes[2], dk, f'Difference\nmax|ΔK|={ab:.2e}',        'RdBu_r', -ab, ab),
    ]:
        sns.heatmap(mat, ax=ax, cmap=cmap, vmin=vmin, vmax=vmax,
                    xticklabels=disp, yticklabels=disp, cbar_kws={'shrink': 0.7})
        ax.set_title(title, fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=fs, fontfamily='Carlito')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=fs, fontfamily='Carlito')
    plt.tight_layout(); save_plot(fig, 's3_kinship.png')

    # off-diagonal scatter: tests covariance structure that determines LMM power
    # (Yang et al. 2010; Speed & Balding 2015)
    idx_u = np.triu_indices(n, k=1)
    kp_od = kp[idx_u]; kr_od = kr[idx_u]
    fig_od, ax_od = plt.subplots(figsize=(5, 5))
    scatter_qc(ax_od, kr_od, kp_od,
               'K (R, off-diagonal)', 'K (Python, off-diagonal)',
               f'kinship off-diagonal  n={len(kp_od)}\n'
               f'r={float(np.corrcoef(kr_od, kp_od)[0,1]):.4f}  CCC={ccc(kr_od, kp_od):.4f}')
    plt.tight_layout(); save_plot(fig_od, 's3_offdiag_scatter.png')

    # diagonal vs off-diagonal histograms: diagonal inflation → variance error;
    # off-diagonal inflation → covariance bias — distinct LMM consequences
    fig_rd, axes_rd = plt.subplots(1, 2, figsize=(10, 4))
    for ax_r, kp_v, kr_v, label in [
        (axes_rd[0], np.diag(kp), np.diag(kr), 'diagonal (self-relatedness)'),
        (axes_rd[1], kp_od,       kr_od,        'off-diagonal (pairwise)'),
    ]:
        sns.histplot(kr_v, bins=30, color='#D83026', alpha=0.6, edgecolor='none',
                     ax=ax_r, label='R', stat='density')
        sns.histplot(kp_v, bins=30, color='#4F6EAF', alpha=0.6, edgecolor='none',
                     ax=ax_r, label='Python', stat='density')
        ax_r.set(xlabel='kinship value', ylabel='density', title=label)
        ax_r.legend(fontsize=8); sns.despine(ax=ax_r)
    plt.tight_layout(); save_plot(fig_rd, 's3_relatedness_distributions.png')


def plot_s4(comp_df, stat_df, chr_colors_fn, py_betas, r_ae, n_s, n_l, n_f, J,
            ae_df=None):
    '''s4: lod scatter, residuals, gene tiers, gene summary, p-value QQ,
    founder effects, discrepancy manhattan, allele effect concordance.
    chr_colors_fn: callable list[str] -> list[color] using alternating CHR_PAL.'''
    tc = {'strict': '#4F6EAF', 'lenient': '#55AF5B', 'fail': '#D83026'}
    stat_df = stat_df.copy()

    # lod scatter
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    scatter_qc(ax1, comp_df['lod_r'], comp_df['lod_py'],
               'LOD (miQTL R)', 'LOD (mosaiQTL Python)', 'LOD comparison')
    plt.tight_layout(); save_plot(fig1, 's4_lod_scatter.png')

    # lod residuals along genome position
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    diff = comp_df['lod_py'] - comp_df['lod_r']
    xpos = comp_df['pos_mb'] if 'pos_mb' in comp_df.columns else np.arange(len(diff))
    ax2.scatter(xpos, diff, c=chr_colors_fn(comp_df['chr'].tolist()),
                s=2, alpha=1.0, edgecolors='none')
    ax2.axhline(0, color='#D83026', lw=0.8, ls='--')
    ax2.text(0.99, 0.97, f'mean={diff.mean():.3f}  sd={diff.std():.3f}',
             transform=ax2.transAxes, fontsize=8, va='top', ha='right')
    ax2.set(xlabel='position (Mb)', ylabel=r'$\Delta$LOD (py$-$R)', title='LOD residuals')
    sns.despine(ax=ax2); plt.tight_layout(); save_plot(fig2, 's4_lod_residuals.png')

    # gene tiers bar + fail rate by peak LOD bin
    fig3, axes3 = plt.subplots(1, 2, figsize=(11, 4))
    axes3[0].bar(['strict', 'lenient', 'fail'], [n_s, n_l, n_f],
                 color=[tc['strict'], tc['lenient'], tc['fail']], edgecolor='none')
    axes3[0].set(xlabel='tier', ylabel='genes',
                 title=f'gene tiers  strict={n_s}  lenient={n_l}  fail={n_f}')
    sns.despine(ax=axes3[0])
    bins = [0, 1, 3, 5, np.inf]; blabs = ['[0,1)', '[1,3)', '[3,5)', '[5,∞)']
    stat_df['lod_bin'] = pd.cut(stat_df['peak_lod_r'], bins=bins, labels=blabs, right=False)
    grp = stat_df.groupby('lod_bin', observed=False)['tier'].value_counts(
        normalize=True).unstack(fill_value=0)
    if 'fail' in grp.columns:
        axes3[1].bar(range(len(blabs)), grp.reindex(blabs)['fail'].fillna(0),
                     color='#D83026', edgecolor='none')
    axes3[1].set(xticks=range(len(blabs)), xticklabels=blabs,
                 xlabel='peak LOD (R)', ylabel='fail rate',
                 title='fail rate by LOD bin', ylim=(0, 1))
    sns.despine(ax=axes3[1]); plt.tight_layout(); save_plot(fig3, 's4_gene_tiers.png')

    # gene summary: h2 distribution + per-gene r + peak LOD scatter
    fig_gs, axes_gs = plt.subplots(1, 3, figsize=(14, 4))
    sns.histplot(stat_df['h2_null'].dropna(), bins=30, color='#4F6EAF',
                 edgecolor='none', ax=axes_gs[0])
    axes_gs[0].set(xlabel='$h^2$ (null)', ylabel='genes',
                   title='heritability distribution', xlim=(0, None), ylim=(0, None))
    sns.despine(ax=axes_gs[0])
    sns.histplot(stat_df['r'].dropna(), bins=30, color='#55AF5B',
                 edgecolor='none', ax=axes_gs[1])
    axes_gs[1].axvline(0.999, color='#D83026', lw=1, ls='--', label='r=0.999')
    axes_gs[1].set(xlabel='Pearson $r$ (LOD py vs R)', ylabel='genes',
                   title='per-gene LOD correlation', xlim=(0, None), ylim=(0, None))
    axes_gs[1].legend(fontsize=8); sns.despine(ax=axes_gs[1])
    for tier in ['strict', 'lenient', 'fail']:
        sub = stat_df[stat_df['tier'] == tier]
        if sub.empty: continue
        axes_gs[2].scatter(sub['peak_lod_r'], sub['peak_lod_py'],
                           s=4, alpha=0.7, edgecolors='none', color=tc[tier], label=tier)
    _lim = max(stat_df['peak_lod_r'].max(), stat_df['peak_lod_py'].max()) * 1.05
    axes_gs[2].plot([0, _lim], [0, _lim], '--', color='#888', lw=1)
    axes_gs[2].set(xlim=(0, _lim), ylim=(0, _lim),
                   xlabel='peak LOD (R)', ylabel='peak LOD (Python)', title='peak LOD per gene')
    axes_gs[2].legend(fontsize=8); sns.despine(ax=axes_gs[2])
    plt.tight_layout(); save_plot(fig_gs, 's4_gene_summary.png')

    # p-value QQ: lod -> chi2(J-1) -> p; lambda_GC flags LMM miscalibration
    _df_pq = J - 1
    _chi2o = np.sort(2 * np.log(10) * comp_df['lod_py'].dropna().values)[::-1]
    _n_pq  = len(_chi2o)
    _p_obs = sp_stats.chi2.sf(_chi2o, df=_df_pq)
    _p_exp = np.linspace(1 / _n_pq, 1.0, _n_pq)
    _step  = max(1, _n_pq // 5000)
    _lam   = float(np.nanmedian(_chi2o) / sp_stats.chi2.ppf(0.5, df=_df_pq))
    _qlim  = max(float(-np.log10(_p_obs[0])), 1.0) * 1.05
    fig_qq, ax_qq = plt.subplots(figsize=(5, 5))
    ax_qq.scatter(-np.log10(_p_exp[::_step]), -np.log10(_p_obs[::_step]),
                  s=2, alpha=0.5, color='#4F6EAF', edgecolors='none')
    ax_qq.plot([0, _qlim], [0, _qlim], '--', color='#D83026', lw=1, label='null')
    ax_qq.set(xlim=(0, _qlim), ylim=(0, None),
              xlabel=r'expected $-\log_{10}(p)$',
              ylabel=r'observed $-\log_{10}(p)$',
              title=f'LMM $p$-value QQ ($\\chi^2$ df={_df_pq})')
    ax_qq.text(0.05, 0.95, f'$\\lambda_{{GC}}$={_lam:.3f}',
               transform=ax_qq.transAxes, fontsize=9, va='top')
    ax_qq.legend(fontsize=8); sns.despine(ax=ax_qq)
    plt.tight_layout(); save_plot(fig_qq, 's4_pvalue_qq.png')

    # founder effects for highest-LOD gene with R reference
    if r_ae is not None and py_betas:
        best = next((g for g in
                     stat_df[stat_df['gene_id'].isin(r_ae['gene_id'].unique())]
                     .sort_values('peak_lod_r', ascending=False)['gene_id']
                     if g in py_betas), None)
        if best:
            chrom_b, peak_loc, f_names, py_eff = py_betas[best]
            r_sub = r_ae[(r_ae['gene_id'] == best) & (r_ae['locus'] == peak_loc)]
            fig, ax = plt.subplots(figsize=(9, 4)); x = np.arange(len(f_names)); w = 0.35
            cf = [FOUNDER_COLORS.get(f, '#888') for f in f_names]
            ax.bar(x - w/2, py_eff, w, color=cf, edgecolor='white', lw=0.5,
                   label='mosaiQTL (Python)')
            if not r_sub.empty:
                r_eff = [float(r_sub.loc[r_sub['founder'] == f, 'effect'].iloc[0])
                         if f in r_sub['founder'].values else 0.0 for f in f_names]
                ax.bar(x + w/2, r_eff, w, color=cf, hatch='///', edgecolor='#333',
                       lw=0.6, alpha=0.7, label='miQTL (R)')
                _r2 = (float(np.corrcoef(py_eff, r_eff)[0, 1])
                       if np.std(py_eff) > 0 and np.std(r_eff) > 0 else np.nan)
                ax.text(0.02, 0.97, f'r={_r2:.4f}' if np.isfinite(_r2) else 'r=n/a',
                        transform=ax.transAxes, fontsize=10, va='top')
            ax.axhline(0, color='#444', lw=0.5)
            ax.set(xticks=x, ylabel='effect size',
                   title=f'founder effects — {best}  chr{chrom_b}  {peak_loc}')
            ax.set_xticklabels(f_names, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=9); sns.despine(); plt.tight_layout()
            save_plot(fig, 's4_founder_effects.png')

    # discrepancy manhattan: |ΔLOD| per genomic position (Casale et al. 2015)
    _dm = comp_df.copy()
    _dm['abs_delta'] = (_dm['lod_py'] - _dm['lod_r']).abs()
    _dm['chr_str']   = _dm['chr'].astype(str)
    _sorted = sorted(_dm['chr_str'].unique(), key=_chr_sort)
    _off = {}; _cum = 0
    for _c in _sorted:
        _off[_c] = _cum; _cum += float(_dm[_dm['chr_str'] == _c]['pos_mb'].max()) + 10
    _dm['x_plot'] = _dm.apply(lambda r: _off[r['chr_str']] + r['pos_mb'], axis=1)
    _xt = [_off[c] + float(_dm[_dm['chr_str'] == c]['pos_mb'].median()) for c in _sorted]
    _cc = [CHR_PAL[i % 2] for i, c in enumerate(_sorted)
           for _ in range((_dm['chr_str'] == c).sum())]
    fig_dm, ax_dm = plt.subplots(figsize=(14, 3))
    ax_dm.scatter(_dm['x_plot'], _dm['abs_delta'], c=_cc, s=1, alpha=0.6, edgecolors='none')
    ax_dm.set(xticks=_xt, xticklabels=_sorted,
              xlabel='chromosome', ylabel='|ΔLOD| (py − R)',
              title='LMM discrepancy Manhattan — |ΔLOD| per locus')
    plt.setp(ax_dm.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    sns.despine(ax=ax_dm); plt.tight_layout(); save_plot(fig_dm, 's4_discrepancy_manhattan.png')

    # allele effect concordance distribution (Listgarten et al. 2012)
    if ae_df is not None and len(ae_df):
        fig_ae, axes_ae = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(ae_df['r_allele'].dropna(), bins=30, color='#4F6EAF',
                     edgecolor='none', ax=axes_ae[0])
        axes_ae[0].axvline(0, color='#D83026', lw=1, ls='--')
        axes_ae[0].set(xlabel='Pearson $r$ (allele effects py vs R)',
                       ylabel='genes', title='allele effect concordance distribution',
                       xlim=(-1, 1))
        sns.despine(ax=axes_ae[0])
        sns.histplot(ae_df['ccc_allele'].dropna(), bins=30, color='#55AF5B',
                     edgecolor='none', ax=axes_ae[1])
        axes_ae[1].axvline(0, color='#D83026', lw=1, ls='--')
        axes_ae[1].set(xlabel='CCC (Lin 1989) — allele effects py vs R',
                       ylabel='genes', title='allele effect CCC distribution', xlim=(-1, 1))
        sns.despine(ax=axes_ae[1])
        plt.tight_layout(); save_plot(fig_ae, 's4_allele_effect_concordance.png')


def plot_s5(coll_df, k_cors, mean_py, mean_r, mean_k_r,
            df_fc=None, r_fc=np.nan, cache_py=None, cache_coll=None, chr_order=None):
    '''s5: collapse reduction bar + kinship stability scatter + LOD preservation scatter
    + (optional) loci spacing histogram when both caches provided.'''
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    ax = axes[0]; x = np.arange(len(coll_df)); w = 0.35
    ax.bar(x - w/2, coll_df['pct_py'], w, color='#4F6EAF', edgecolor='none',
           label='mosaiQTL (Python)')
    ax.bar(x + w/2, coll_df['pct_r'].fillna(0), w, color='#55AF5B', edgecolor='none',
           alpha=0.85, label='miQTL (R)')
    ax.set(xticks=x, xticklabels=coll_df['chr'].tolist(),
           xlabel='chromosome', ylabel='loci reduced (%)',
           title=f'collapse l2.norm tol=0.02\npy={mean_py:.1f}%  r={mean_r:.1f}%')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); sns.despine(ax=ax)

    ax = axes[1]
    ax.scatter(range(len(k_cors)), k_cors, s=20, color='#4F6EAF', edgecolors='none')
    ax.axhline(0.99, color='#D83026', lw=1, ls='--', label='r=0.99')
    ax.set(xlabel='chr index', ylabel='kinship $r$ (full vs collapsed)',
           title=f'kinship stability\nmean r={mean_k_r:.4f}', ylim=(0.9, 1.005))
    ax.legend(fontsize=9); sns.despine(ax=ax)

    ax = axes[2]
    if df_fc is not None and not np.isnan(r_fc):
        scatter_qc(ax, df_fc['lod_full'], df_fc['lod_coll'],
                   'LOD (full)', 'LOD (collapsed)', f'LOD preservation\nr={r_fc:.4f}')
    else:
        ax.text(0.5, 0.5, 'LOD QC n/a', ha='center', transform=ax.transAxes)
    plt.tight_layout(); save_plot(fig, 's5_cache_collapse.png')

    # loci spacing: inter-locus gap distributions before vs after collapse
    # shows whether collapse is spatially uniform (Browning & Browning 2016)
    if cache_py is not None and cache_coll is not None and chr_order is not None:
        _rows = []
        for _c in chr_order:
            if _c not in cache_py._data or _c not in cache_coll._data: continue
            for _pos, _tag in [(cache_py._data[_c]['positions'], 'full'),
                               (cache_coll._data[_c]['positions'], 'collapsed')]:
                for g in np.diff(np.sort(_pos)):
                    _rows.append({'gap_mb': float(g), 'cache': _tag})
        if _rows:
            _sp = pd.DataFrame(_rows)
            _fg = _sp[_sp['cache'] == 'full']['gap_mb'].values
            _cg = _sp[_sp['cache'] == 'collapsed']['gap_mb'].values
            _xmax = float(np.percentile(np.concatenate([_fg, _cg]), 99)) * 1.1
            fig_sp, ax_sp = plt.subplots(figsize=(7, 4))
            sns.histplot(_fg[_fg <= _xmax], bins=60, color='#4F6EAF', alpha=0.65,
                         edgecolor='none', stat='density', ax=ax_sp, label='full')
            sns.histplot(_cg[_cg <= _xmax], bins=60, color='#D83026', alpha=0.65,
                         edgecolor='none', stat='density', ax=ax_sp, label='collapsed')
            ax_sp.set(xlabel='inter-locus distance (Mb)', ylabel='density',
                      title='loci spacing distribution (full vs collapsed)',
                      xlim=(0, _xmax), ylim=(0, None))
            ax_sp.legend(fontsize=8); sns.despine(ax=ax_sp)
            plt.tight_layout(); save_plot(fig_sp, 's5_loci_spacing.png')


def plot_s6(all_max_lods, gev_p, thresh_py, thresh_r_v, delta, r_perms, comp_df):
    '''s6: permutation null histogram + GEV fit + QQ scatter + per-chr delta bar
    + ECDF + threshold scatter.'''
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # pooled histogram + GEV fit
    ax = axes[0]
    sns.histplot(all_max_lods, bins=40, color='#4F6EAF', ax=ax,
                 edgecolor='none', stat='density', label='py (all chr)')
    if r_perms is not None:
        sns.histplot(r_perms['perm_lod'], bins=40, color='#55AF5B', ax=ax,
                     edgecolor='none', stat='density', alpha=0.5, label='R (all chr)')
    xf = np.linspace(all_max_lods.min(), all_max_lods.max() * 1.1, 200)
    ax.plot(xf, _gev.pdf(xf, gev_p['c'], loc=gev_p['loc'], scale=gev_p['scale']),
            color='#C0BFBF', lw=2, label='GEV fit (py pooled)')
    ax.axvline(thresh_py,  color='#4F6EAF', lw=1.5, ls='--', label=f'py 95%={thresh_py:.2f}')
    ax.axvline(thresh_r_v, color='#D83026', lw=1.5, ls=':',  label=f'R mean={thresh_r_v:.2f}')
    ax.set(xlabel='max LOD', ylabel='density',
           title=f'pooled permutation null  ({len(comp_df)} chr)')
    ax.legend(fontsize=8); sns.despine(ax=ax)

    # QQ: sorted pooled py vs sorted pooled R
    ax = axes[1]
    if r_perms is not None and len(r_perms) > 3:
        _n = min(len(r_perms), len(all_max_lods))
        scatter_qc(ax, np.sort(r_perms['perm_lod'].values)[:_n],
                   np.sort(all_max_lods)[:_n],
                   'R perm max-LOD (sorted)', 'Python perm max-LOD (sorted)',
                   f'QQ  mean Δthr={delta:.3f}')
    else:
        ax.text(0.5, 0.5, 'no R perm lods', ha='center', transform=ax.transAxes)

    # per-chr delta bar
    ax = axes[2]
    colors = ['#D83026' if d > 0.5 else '#4F6EAF' for d in comp_df['delta']]
    ax.bar(np.arange(len(comp_df)), comp_df['delta'], color=colors, edgecolor='none')
    ax.axhline(0.5, color='#D83026', lw=1, ls='--', label='pass threshold')
    ax.set(xticks=np.arange(len(comp_df)), xticklabels=comp_df['chr'].tolist(),
           xlabel='chromosome', ylabel='|Δ threshold| (py − R)',
           title='per-chr threshold agreement')
    ax.set_xticklabels(comp_df['chr'].tolist(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=8); sns.despine(ax=ax)
    plt.tight_layout(); save_plot(fig, 's6_permutation.png')

    # ECDF: more sensitive to tail differences than histogram (Phipson & Smyth 2010)
    fig_ec, ax_ec = plt.subplots(figsize=(6, 4))
    _d = [all_max_lods]; _lb = ['Python']; _cl = ['#4F6EAF']
    if r_perms is not None and 'perm_lod' in r_perms.columns:
        _d.append(r_perms['perm_lod'].dropna().values); _lb.append('R'); _cl.append('#D83026')
    ecdf_plot(ax_ec, _d, _lb, _cl, 'max LOD (permutation null)', 'permutation null ECDF')
    ax_ec.axvline(thresh_py,  color='#4F6EAF', lw=1, ls='--', label=f'py 95%={thresh_py:.2f}')
    ax_ec.axvline(thresh_r_v, color='#D83026', lw=1, ls=':',  label=f'R mean={thresh_r_v:.2f}')
    ax_ec.axhline(0.95, color='#888780', lw=0.6, ls=':')
    ax_ec.legend(fontsize=8); plt.tight_layout(); save_plot(fig_ec, 's6_ecdf.png')

    # threshold scatter: reveals whether over/under-estimation is systematic vs random
    if len(comp_df) > 1:
        fig_ts, ax_ts = plt.subplots(figsize=(5, 5))
        scatter_qc(ax_ts, comp_df['threshold_r'], comp_df['threshold_py'],
                   'threshold (R)', 'threshold (Python)',
                   f'per-chr threshold comparison  n={len(comp_df)}')
        plt.tight_layout(); save_plot(fig_ts, 's6_threshold_scatter.png')


def plot_s7(mi_comp, mi_diffs=None):
    '''s7: MI LOD scatter + profile for first gene + per-gene bias violin.'''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scatter_qc(axes[0], mi_comp['lod_r'], mi_comp['lod_py'],
               'LOD MI (miQTL R)', 'LOD MI (mosaiQTL Python)', 'MI LOD scatter')
    first_g = mi_comp['gene_id'].iloc[0]
    sub = mi_comp[mi_comp['gene_id'] == first_g]
    pos_x = sub['pos_mb'] if 'pos_mb' in sub.columns else np.arange(len(sub))
    axes[1].plot(pos_x.values, sub['lod_r'].values, color='#D83026', lw=1.2,
                 alpha=0.85, label='miQTL (R)')
    axes[1].plot(pos_x.values, sub['lod_py'].values, color='#4F6EAF', lw=1.2,
                 alpha=0.85, label='mosaiQTL (Python)')
    axes[1].set(xlabel='position (Mb)', ylabel='LOD', title=f'MI LOD profile — {first_g}')
    axes[1].legend(fontsize=9); sns.despine(ax=axes[1])
    plt.tight_layout(); save_plot(fig, 's7_mi_scan.png')

    # per-gene LOD diff violin: separates implementation error from MI sampling noise
    if mi_diffs is not None and len(mi_diffs) > 1:
        fig_vio, ax_vio = plt.subplots(figsize=(4, 5))
        ax_vio.violinplot([mi_diffs], positions=[0], showmedians=True, widths=0.6)
        ax_vio.scatter([0] * len(mi_diffs), mi_diffs, s=8, color='#4F6EAF',
                       alpha=0.6, edgecolors='none', zorder=3)
        ax_vio.axhline(0, color='#D83026', lw=1, ls='--', label='zero bias')
        ax_vio.set(xticks=[0], xticklabels=['MI LOD (py − R)'],
                   ylabel='mean LOD difference per gene', title='MI scan bias distribution')
        ax_vio.legend(fontsize=8); sns.despine(ax=ax_vio)
        plt.tight_layout(); save_plot(fig_vio, 's7_lod_diff_violin.png')


def plot_s8(snp_comp, snp_maf_df=None):
    '''s8: SNP LOD scatter + chr8 LOD profile + MAF concordance (if snp_maf_df provided).
    snp_maf_df must have columns: maf, abs_delta, pos_mb (sorted by maf externally).
    returns r_snp_all for record().'''
    r_snp_all = float(np.corrcoef(snp_comp['lod_r'], snp_comp['lod_py'])[0, 1])
    fig, ax = plt.subplots(figsize=(5, 5))
    scatter_qc(ax, snp_comp['lod_r'], snp_comp['lod_py'],
               'SNP LOD (miQTL R)', 'SNP LOD (mosaiQTL Python)',
               f'SNP scan  r={r_snp_all:.4f}')
    plt.tight_layout(); save_plot(fig, 's8_snp_scan.png')

    # chr8 profile: Piezo1 region; shows marginal vs diplotype LOD deviation
    _snp8 = snp_comp[snp_comp['chr'].astype(str) == '8'].copy()
    if not _snp8.empty and 'pos_mb' in _snp8.columns:
        _snp8 = _snp8.sort_values('pos_mb')
        fig_sp8, ax_sp8 = plt.subplots(figsize=(9, 4))
        ax_sp8.plot(_snp8['pos_mb'], _snp8['lod_r'],  color='#D83026', lw=0.8,
                    alpha=0.7, label='R (diplotype)')
        ax_sp8.plot(_snp8['pos_mb'], _snp8['lod_py'], color='#4F6EAF', lw=0.8,
                    alpha=0.7, label='Python (marginal)')
        ax_sp8.set(xlabel='position chr8 (Mb)', ylabel='SNP LOD',
                   title='SNP scan LOD profile — chr8', xlim=(0, None), ylim=(0, None))
        ax_sp8.legend(fontsize=8); sns.despine(ax=ax_sp8)
        plt.tight_layout(); save_plot(fig_sp8, 's8_lod_profile_chr8.png')

    # marginal approximation degrades at lower freq: |ΔLOD| vs MAF
    if snp_maf_df is not None and 'maf' in snp_maf_df.columns:
        _m = snp_maf_df.sort_values('maf')
        _roll = _m['abs_delta'].rolling(200, min_periods=20).median()
        fig_maf, ax_maf = plt.subplots(figsize=(6, 4))
        ax_maf.scatter(_m['maf'], _m['abs_delta'], s=2, alpha=0.4,
                       color='#4F6EAF', edgecolors='none')
        ax_maf.plot(_m['maf'], _roll, color='#D83026', lw=1.5, label='rolling median')
        ax_maf.set(xlabel='founder alt-allele frequency (MAF proxy)',
                   ylabel='|ΔLOD| (py − R)', title='SNP LOD discrepancy vs allele freq',
                   xlim=(0, 1), ylim=(0, None))
        ax_maf.legend(fontsize=8); sns.despine(ax=ax_maf)
        plt.tight_layout(); save_plot(fig_maf, 's8_maf_concordance.png')

    return r_snp_all


def plot_s9(med_comp, r_med_v, drop_r=None):
    '''s9: mediation LOD scatter + profile + LOD drop histogram (if drop_r provided).'''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    scatter_qc(axes[0], med_comp['lod_r'], med_comp['lod_py'],
               'mediation LOD (R)', 'mediation LOD (Python)',
               f'mediation  r={r_med_v:.4f}')
    _g0 = med_comp['gene_id'].iloc[0]; _c0 = med_comp['chr'].iloc[0]
    sub = med_comp[(med_comp['gene_id'] == _g0) & (med_comp['chr'] == _c0)]
    pos_x = sub['pos_mb'] if 'pos_mb' in sub.columns else np.arange(len(sub))
    axes[1].plot(pos_x.values, sub['lod_r'].values, color='#D83026', lw=1.2,
                 alpha=0.85, label='miQTL (R)')
    axes[1].plot(pos_x.values, sub['lod_py'].values, color='#4F6EAF', lw=1.2,
                 alpha=0.85, label='mosaiQTL (Python)')
    axes[1].set(xlabel='position (Mb)', ylabel='LOD (base)',
                title=f'mediation profile  {_g0} chr{_c0}')
    axes[1].legend(fontsize=9); sns.despine(ax=axes[1])
    plt.tight_layout(); save_plot(fig, 's9_mediation.png')

    # LOD drop distribution: concordance of amplification cases is a stringent test
    if drop_r is not None:
        _lim = abs(drop_r).max() * 1.1
        fig_ld, ax_ld = plt.subplots(figsize=(7, 4))
        sns.histplot(drop_r, bins=40, color='#D83026', alpha=0.6, edgecolor='none',
                     stat='density', ax=ax_ld, label='R lod_drop')
        ax_ld.axvline(0, color='#888780', lw=0.8, ls=':', label='zero (no mediation)')
        ax_ld.set(xlabel='LOD drop (base − mediated)', ylabel='density',
                  title='mediation LOD drop distribution (R)',
                  xlim=(-_lim, _lim), ylim=(0, None))
        ax_ld.legend(fontsize=8); sns.despine(ax=ax_ld)
        plt.tight_layout(); save_plot(fig_ld, 's9_lod_drop_distribution.png')


def plot_s10(ci_comp, boot_pos, r_boot, ci_gene, ci_chr, n_overlap):
    '''s10: bootstrap histogram + interval comparison + CI width scatter.'''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    vp = boot_pos[~np.isnan(boot_pos)] if boot_pos is not None and hasattr(boot_pos, '__len__') else []
    if len(vp):
        sns.histplot(vp, bins=30, color='#4F6EAF', ax=ax,
                     edgecolor='none', label='py bootstrap peaks')
    if r_boot is not None:
        _rb_c = r_boot[r_boot['chr'].astype(str) == ci_chr] if 'chr' in r_boot.columns else r_boot
        sns.histplot(_rb_c['boot_pos'].dropna(), bins=30, color='#55AF5B',
                     ax=ax, edgecolor='none', alpha=0.5, label='R bootstrap peaks')
    last = ci_comp.iloc[-1]
    ax.axvline(last['ci_lo_py'], color='#4F6EAF', lw=1.2, ls='--')
    ax.axvline(last['ci_hi_py'], color='#4F6EAF', lw=1.2, ls='--',
               label=f'py=[{last["ci_lo_py"]:.2f},{last["ci_hi_py"]:.2f}]')
    ax.axvline(last['ci_lo_r'],  color='#D83026', lw=1.2, ls=':')
    ax.axvline(last['ci_hi_r'],  color='#D83026', lw=1.2, ls=':',
               label=f'R=[{last["ci_lo_r"]:.2f},{last["ci_hi_r"]:.2f}]')
    ax.set(xlabel='peak position (Mb)', ylabel='bootstraps',
           title=f'positional CI — {ci_gene} chr{ci_chr}')
    ax.legend(fontsize=8); sns.despine(ax=ax)

    ax = axes[1]
    for ri, row in ci_comp.reset_index().iterrows():
        col = '#4F6EAF' if row['overlap'] else '#D83026'
        ax.hlines(ri,        row['ci_lo_py'], row['ci_hi_py'], colors=col, lw=4, alpha=0.7)
        ax.hlines(ri + 0.15, row['ci_lo_r'],  row['ci_hi_r'],  colors='#888', lw=4, alpha=0.5)
    ax.set(xlabel='position (Mb)', ylabel='gene index',
           title=f'CI comparison  overlap={n_overlap}/{len(ci_comp)}')
    sns.despine(ax=ax)
    plt.tight_layout(); save_plot(fig, 's10_positional_ci.png')

    # width scatter: 'overlap' is too coarse; width concordance is the stringent check
    # (Mangin et al. 1994)
    if len(ci_comp) > 1:
        ci_comp = ci_comp.copy()
        ci_comp['width_r']  = ci_comp['ci_hi_r']  - ci_comp['ci_lo_r']
        ci_comp['width_py'] = ci_comp['ci_hi_py'] - ci_comp['ci_lo_py']
        fig_ws, ax_ws = plt.subplots(figsize=(5, 5))
        scatter_qc(ax_ws, ci_comp['width_r'].values, ci_comp['width_py'].values,
                   'CI width (R, Mb)', 'CI width (Python, Mb)',
                   f'bootstrap CI width concordance  n={len(ci_comp)}')
        plt.tight_layout(); save_plot(fig_ws, 's10_width_scatter.png')


def plot_s11(mg_fx, r_fx_v, inst_py):
    '''s11: instability LOD scatter + profile + peak proportion scatter.'''
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if not np.isnan(r_fx_v):
        scatter_qc(axes[0], mg_fx['lod_r'], mg_fx['lod_py'],
                   'instability LOD (R)', 'instability LOD (Python)',
                   f'instability scan  r={r_fx_v:.4f}')
    _g0 = mg_fx['gene_id'].iloc[0]; _c0 = mg_fx['chr'].iloc[0]
    sub = mg_fx[(mg_fx['gene_id'] == _g0) & (mg_fx['chr'] == _c0)].sort_values('pos_mb_r')
    pos_x = sub['pos_mb_r'] if 'pos_mb_r' in sub.columns else np.arange(len(sub))
    axes[1].plot(pos_x.values, sub['lod_r'].values, color='#D83026', lw=1.2,
                 alpha=0.85, label='miQTL (R)')
    axes[1].plot(pos_x.values, sub['lod_py'].values, color='#4F6EAF', lw=1.2,
                 alpha=0.85, label='mosaiQTL (Python)')
    axes[1].set(xlabel='position (Mb)', ylabel='mean bootstrap LOD',
                title=f'instability profile — {_g0} chr{_c0}')
    axes[1].legend(fontsize=9); sns.despine(ax=axes[1])
    plt.tight_layout(); save_plot(fig, 's11_instability.png')

    # peak proportion vs R mean LOD; validates bootstrap sample size equivalence
    if hasattr(inst_py, 'p_value') and len(mg_fx) > 1:
        _pv_map = dict(zip(inst_py.locus_ids.tolist(), inst_py.p_value.tolist()))
        mg_fx = mg_fx.copy()
        mg_fx['pv_py'] = mg_fx['locus'].map(_pv_map)
        _pv = mg_fx.dropna(subset=['pv_py', 'lod_r'])
        if len(_pv) > 1:
            fig_pv, ax_pv = plt.subplots(figsize=(5, 4))
            ax_pv.scatter(_pv['lod_r'], 1 - _pv['pv_py'],
                          s=12, color='#4F6EAF', alpha=0.7, edgecolors='none')
            ax_pv.set(xlabel='mean bootstrap LOD (R)',
                      ylabel='peak proportion (Python)',
                      title='instability: R mean LOD vs Python peak proportion',
                      xlim=(0, None), ylim=(0, 1))
            sns.despine(ax=ax_pv)
            plt.tight_layout(); save_plot(fig_pv, 's11_peak_proportion.png')

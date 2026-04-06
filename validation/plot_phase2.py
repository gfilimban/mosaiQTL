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
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel=xlabel, ylabel=ylabel)
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
    ax.set(xlabel=xlabel, ylabel='difference (py 2212 R)')
    ax.legend(fontsize=7); sns.despine(ax=ax)
    return md, sd


def ecdf_plot(ax, data_list, labels, colors, xlabel, title):
    '''empirical cdf; more sensitive to tail differences than histograms (Phipson & Smyth 2010).'''
    for data, label, color in zip(data_list, labels, colors):
        d = np.sort(np.asarray(data, float)); d = d[np.isfinite(d)]
        ax.plot(d, np.linspace(0, 1, len(d)), color=color, lw=1.5, label=label)
    ax.set(xlabel=xlabel, ylabel='cumulative probability',
           xlim=(0, None), ylim=(0, 1))
    ax.legend(fontsize=8); sns.despine(ax=ax)


# ── per-section plot functions ────────────────────────────────────────────────

def plot_s1(info_df, py_founders, r_founders, py_sids, r_sids, fnd_match, sid_match):
    '''s1: loci-per-chr grouped bar + py-vs-R loci scatter (validation).'''
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # panel 1 — grouped bar per chromosome
    ax = axes[0]; x = np.arange(len(info_df)); w = 0.35
    ax.bar(x - w/2, info_df['n_loci_py'], w, color='#4F6EAF', edgecolor='none',
           label='mosaiQTL (Python)')
    ax.bar(x + w/2, info_df['n_loci_r'].fillna(0), w, color='#55AF5B',
           edgecolor='none', alpha=0.85, label='miQTL (R)')
    ax.set(xticks=x, xticklabels=info_df['chr'].tolist(),
           xlabel='chromosome', ylabel='loci')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); sns.despine(ax=ax)
    # panel 2 — scatter py vs R loci count per chr (validation of cache parity)
    ok = info_df['n_loci_r'].notna()
    scatter_qc(axes[1],
               info_df.loc[ok, 'n_loci_r'].values,
               info_df.loc[ok, 'n_loci_py'].values,
               'loci per chr (R)', 'loci per chr (Python)',
               f's1: loci concordance — founders={len(py_founders)}  '
               f'samples={len(py_sids)}\n'
               f'fnd_match={fnd_match}  sid_match={sid_match}')
    plt.tight_layout(); save_plot(fig, 's1_cache_info.png')


def plot_s2_pyonly(H_py0, fnames, disp_s, fs, rep_chr):
    '''s2 fallback: python-only heatmap when R haplotype_vals_r.csv is missing.'''
    fig_hm, ax_hm = plt.subplots(figsize=(7, 5))
    sns.heatmap(H_py0, ax=ax_hm, cmap='YlOrRd', vmin=0, vmax=1,
                xticklabels=fnames, yticklabels=disp_s, cbar_kws={'shrink': 0.7})

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
            _ylab = disp_s if ai % ncols == 0 else False   # y-labels only on leftmost column
            sns.heatmap(diff_c, ax=axs_flat[ai], cmap='RdBu_r', vmin=-ab_c, vmax=ab_c,
                        xticklabels=False, yticklabels=_ylab, cbar_kws={'shrink': 0.5})
            if ai % ncols == 0 and _ylab:
                axs_flat[ai].set_yticklabels(axs_flat[ai].get_yticklabels(),
                                              fontsize=max(3, min(6, int(120/max(len(disp_s),1)))),
                                              rotation=0)

        for ai in range(len(chrs_d), len(axs_flat)):
            axs_flat[ai].set_visible(False)
        plt.suptitle('s2: haplotype prob difference (Python − R) — all chromosomes', fontsize=10)
        plt.tight_layout(); save_plot(fig_gd, 's2_haplotype_diff_grid.png')

    # ── probability scatter + bland-altman + calibration ─────────────────────
    if val_rows_all:
        rpa = np.array(all_r_probs); ppa = np.array(all_py_probs)
        ok  = np.isfinite(rpa) & np.isfinite(ppa)
        idx_s = np.random.default_rng(0).choice(np.where(ok)[0],
                                                  min(20000, ok.sum()), replace=False)
        vdf = pd.DataFrame(val_rows_all).dropna()
        _fc_r = fc_r if fc_r is not None else vdf['founder'].unique().tolist()
        fnd_r_vals = (vdf.groupby('founder')[['r_prob', 'py_prob']]
                      .apply(lambda g: float(np.corrcoef(g['r_prob'], g['py_prob'])[0, 1])
                             if len(g) > 3 else np.nan, include_groups=False)
                      .reindex(_fc_r).fillna(0))
        fig_sc, axes_sc = plt.subplots(1, 2, figsize=(11, 4))
        axes_sc[0].scatter(rpa[idx_s], ppa[idx_s], s=2, alpha=0.3,
                           color='#4F6EAF', edgecolors='none')
        axes_sc[0].plot([0, 1], [0, 1], '--', color='#D83026', lw=1)
        axes_sc[0].set(xlim=(0, 1.05), ylim=(0, 1.05),
                       xlabel='prob (R HAPPY, marginalised)', ylabel='prob (Python HMM)')
        sns.despine(ax=axes_sc[0])
        fc_colors = [FOUNDER_COLORS.get(f, '#888') for f in fnd_r_vals.index]
        axes_sc[1].bar(range(len(fnd_r_vals)), fnd_r_vals.values,
                       color=fc_colors, edgecolor='none')
        axes_sc[1].axhline(0.90, color='#D83026', lw=1, ls='--', label='r=0.90')
        axes_sc[1].set(xticks=range(len(fnd_r_vals)),
                       xlabel='founder', ylabel='Pearson $r$ (R vs Python)', ylim=(0, 1))
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
                   xlabel='mean Python prob (bin)', ylabel='mean R prob (bin)')
        ax_cal.legend(fontsize=8); sns.despine(ax_cal)
        plt.tight_layout(); save_plot(fig_cal, 's2_calibration.png')
    else:
        # placeholder when R data missing
        fig_sc, ax_sc = plt.subplots(figsize=(6, 4))
        ax_sc.text(0.5, 0.5,
                   'Run R S2 first:\nRscript validation/r_reference/port_validation.R all Inf 2',
                   ha='center', va='center', transform=ax_sc.transAxes,
                   fontsize=10, color='#D83026')

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
        ax_r.set(xlabel='kinship value', ylabel='density')
        ax_r.legend(fontsize=8); sns.despine(ax=ax_r)
    plt.tight_layout(); save_plot(fig_rd, 's3_relatedness_distributions.png')


def plot_s4(comp_df, stat_df, chr_colors_fn, py_betas, r_ae, n_s, n_l, n_f, J,
            ae_df=None):
    '''s4: lod scatter, residuals (cumul. Mb), gene_summary×2, egene manhattan,
    p-value QQ (py+R), founder effects, discrepancy manhattan, allele concordance.'''
    tc = {'strict': '#4F6EAF', 'lenient': '#55AF5B', 'fail': '#D83026'}
    stat_df = stat_df.copy()

    # ── lod scatter ───────────────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    scatter_qc(ax1, comp_df['lod_r'], comp_df['lod_py'],
               'LOD (miQTL R)', 'LOD (mosaiQTL Python)', 's4: LOD comparison')
    plt.tight_layout(); save_plot(fig1, 's4_lod_scatter.png')

    # ── lod residuals — cumulative genomic Mb x-axis (Manhattan style) ────────
    _sorted_chrs = sorted(comp_df['chr'].astype(str).unique(), key=_chr_sort)
    _off = {}; _cum = 0
    for _c in _sorted_chrs:
        _off[_c] = _cum
        _cum += float(comp_df.loc[comp_df['chr'].astype(str) == _c, 'pos_mb'].max()) + 10
    _xplot = comp_df.apply(lambda r: _off[str(r['chr'])] + r['pos_mb'], axis=1)
    _xt = [_off[c] + float(comp_df.loc[comp_df['chr'].astype(str)==c,'pos_mb'].median())
           for c in _sorted_chrs]
    diff = comp_df['lod_py'] - comp_df['lod_r']
    fig2, ax2 = plt.subplots(figsize=(14, 3))
    ax2.scatter(_xplot, diff, c=chr_colors_fn(comp_df['chr'].tolist()),
                s=1, alpha=0.6, edgecolors='none')
    ax2.axhline(0, color='#D83026', lw=0.8, ls='--')
    ax2.text(0.99, 0.97, f'mean={diff.mean():.3f}  sd={diff.std():.3f}',
             transform=ax2.transAxes, fontsize=8, va='top', ha='right')
    ax2.set(xticks=_xt, xticklabels=_sorted_chrs,
            xlabel='position (cumulative Mb)', ylabel=r'$\Delta$LOD (py$-$R)')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    sns.despine(ax=ax2); plt.tight_layout(); save_plot(fig2, 's4_lod_residuals.png')

    # ── gene summary plot 1: LOD relative error histogram by tier ─────────────
    # mirrors the left panel of the reference figure (rel_error bins, tier colors)
    rel_bins  = [0, 0.05, 0.10, 0.20, 0.40, 0.80]
    rel_labs  = ['0–5%', '5–10%', '10–20%', '20–40%', '40–80%']
    stat_df['rel_bin'] = pd.cut(stat_df['rel_error'], bins=rel_bins,
                                 labels=rel_labs, right=False)
    fig_gs1, ax_gs1 = plt.subplots(figsize=(7, 5))
    x_pos = np.arange(len(rel_labs)); w = 0.25
    for i, (tier, color) in enumerate(tc.items()):
        sub = stat_df[stat_df['tier'] == tier]
        counts = sub['rel_bin'].value_counts().reindex(rel_labs, fill_value=0)
        ax_gs1.bar(x_pos + (i-1)*w, counts.values, w, color=color,
                   edgecolor='none', label=f'{tier} ({len(sub):,})')
    # tier boundary dashed lines at 10% and 20%
    ax_gs1.axvline(1.5, color='#888', lw=1, ls='--')   # between 5-10% and 10-20%
    ax_gs1.axvline(2.5, color='#55AF5B', lw=1, ls='--')  # between 10-20% and 20-40%
    ax_gs1.set(xticks=x_pos, xticklabels=rel_labs,
               xlabel=r'relative error ($\max|\Delta\mathrm{LOD}|$ / peak LOD R)',
               ylabel='genes')
    ax_gs1.legend(fontsize=9); sns.despine(ax=ax_gs1)
    plt.tight_layout(); save_plot(fig_gs1, 's4_gene_summary_error.png')

    # ── gene summary plot 2: peak locus concordance by tier ───────────────────
    # right panel of reference figure — peak_match vs mismatch stacked per tier
    fig_gs2, ax_gs2 = plt.subplots(figsize=(6, 5))
    tiers_ord = ['fail', 'lenient', 'strict']; x_pos2 = np.arange(len(tiers_ord)); w2 = 0.35
    for i, (tier, color) in enumerate(tc.items()):
        sub = stat_df[stat_df['tier'] == tier]
        n_match   = int(sub['peak_match'].sum())
        n_mismatch = len(sub) - n_match
        xi = tiers_ord.index(tier)
        ax_gs2.bar(xi - w2/2, n_mismatch, w2, color='#C0BFBF', edgecolor='none',
                   label='peak mismatch' if i == 0 else '_')
        ax_gs2.bar(xi + w2/2, n_match,    w2, color=color,    edgecolor='none',
                   label=f'peak match ({tier})' if i == 0 else tier)
    ax_gs2.set(xticks=np.arange(len(tiers_ord)), xticklabels=tiers_ord,
               ylabel='genes'
                     '(soft: R LOD at Python peak ≥ 99% of R peak)')
    ax_gs2.legend(fontsize=8); sns.despine(ax=ax_gs2)
    plt.tight_layout(); save_plot(fig_gs2, 's4_gene_summary_concordance.png')

    # ── genome-wide eGene distribution (peak LOD Manhattan per gene) ──────────
    # compute peak locus position per gene from comp_df
    _peak_rows = comp_df.loc[comp_df.groupby('gene_id')['lod_py'].idxmax(),
                              ['gene_id', 'chr', 'pos_mb']].copy()
    _peak_rows.columns = ['gene_id', 'chr', 'peak_pos_mb']
    _eg = stat_df.merge(_peak_rows, on='gene_id', how='left', suffixes=('', '_r'))
    _eg['chr_s'] = _eg['chr'].astype(str)
    _eg_sorted   = sorted(_eg['chr_s'].unique(), key=_chr_sort)
    _off2 = {}; _cum2 = 0
    for _c in _eg_sorted:
        _off2[_c] = _cum2
        _cum2 += float(_eg.loc[_eg['chr_s']==_c, 'peak_pos_mb'].max(skipna=True)) + 10
    _eg['x_eg'] = _eg.apply(lambda r: _off2.get(r['chr_s'], 0) + r['peak_pos_mb'], axis=1)
    _xt_eg = [_off2[c] + float(_eg.loc[_eg['chr_s']==c,'peak_pos_mb'].median(skipna=True))
              for c in _eg_sorted]
    _cc_eg = [CHR_PAL[_eg_sorted.index(c) % 2] for c in _eg['chr_s']]
    fig_em, ax_em = plt.subplots(figsize=(14, 3))
    # BH-FDR eGene count: LOD → chi2 p-value (df=J-1), then BH correction
    _df_em   = J - 1
    _chi2_eg = 2 * np.log(10) * _eg['peak_lod_py'].values
    _p_eg    = sp_stats.chi2.sf(np.maximum(_chi2_eg, 0), df=_df_em)
    _n_eg    = len(_p_eg)
    _p_rank  = np.argsort(_p_eg)                          # ascending p-value order
    _bh_thr  = (np.arange(1, _n_eg + 1) / _n_eg) * 0.05  # BH at FDR=5%
    _bh_mask = _p_eg[_p_rank] <= _bh_thr
    _n_egene = int(_bh_mask.sum())                        # count BH-significant genes
    # LOD threshold corresponding to BH cutoff (for horizontal reference line)
    if _n_egene > 0:
        _bh_lod_thr = float(_eg['peak_lod_py'].values[_p_rank[_bh_mask]].min())
    else:
        _bh_lod_thr = None

    for tier in ['fail', 'lenient', 'strict']:
        sub = _eg[_eg['tier'] == tier]
        ax_em.scatter(sub['x_eg'], sub['peak_lod_py'], c=tc[tier],
                      s=3, alpha=0.6, edgecolors='none', label=tier)
    # horizontal line + annotation for BH-FDR threshold
    if _bh_lod_thr is not None:
        ax_em.axhline(_bh_lod_thr, color='#683C91', lw=0.8, ls='--', alpha=0.8)
    ax_em.text(0.99, 0.97,
               f'eGenes (BH FDR 5%) = {_n_egene} / {_n_eg}',
               transform=ax_em.transAxes, fontsize=8,
               va='top', ha='right', color='#683C91')
    ax_em.set(xticks=_xt_eg, xticklabels=_eg_sorted,
              xlabel='chromosome', ylabel='peak LOD (Python)', ylim=(0, None))
    plt.setp(ax_em.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    ax_em.legend(fontsize=8, markerscale=3); sns.despine(ax=ax_em)
    plt.tight_layout(); save_plot(fig_em, 's4_egene_manhattan.png')

    # ── p-value QQ: both py and R, lambda_GC flags LMM miscalibration ─────────
    _df_pq = J - 1
    fig_qq, ax_qq = plt.subplots(figsize=(5, 5))
    for _lods, _label, _color in [
        (comp_df['lod_py'].dropna().values, 'Python', '#4F6EAF'),
        (comp_df['lod_r'].dropna().values,  'R',      '#D83026'),
    ]:
        _chi2o = np.sort(2 * np.log(10) * _lods)[::-1]
        _n     = len(_chi2o)
        _p_obs = sp_stats.chi2.sf(_chi2o, df=_df_pq)
        _p_exp = np.linspace(1 / _n, 1.0, _n)
        _step  = max(1, _n // 5000)
        _lam   = float(np.nanmedian(_chi2o) / sp_stats.chi2.ppf(0.5, df=_df_pq))
        _qlim  = max(float(-np.log10(_p_obs[0])), 1.0) * 1.05
        ax_qq.scatter(-np.log10(_p_exp[::_step]), -np.log10(_p_obs[::_step]),
                      s=2, alpha=0.5, color=_color, edgecolors='none',
                      label=f'{_label} $\\lambda_{{GC}}$={_lam:.3f}')
    ax_qq.plot([0, _qlim], [0, _qlim], '--', color='#888', lw=1)
    ax_qq.set(xlim=(0, _qlim), ylim=(0, None),
              xlabel=r'expected $-\log_{10}(p)$',
              ylabel=r'observed $-\log_{10}(p)$')
    ax_qq.legend(fontsize=8); sns.despine(ax=ax_qq)
    plt.tight_layout(); save_plot(fig_qq, 's4_pvalue_qq.png')

    # ── founder effects — highest-LOD gene with R reference ───────────────────
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
                # r_ae founder col may use single-letter HAPPY codes (A–H) or full names
                _code_map = {chr(65+i): fn for i, fn in enumerate(FOUNDER_COLORS.keys())}
                _r_founders = r_sub['founder'].values
                def _lookup(fn):
                    if fn in _r_founders: return float(r_sub.loc[r_sub['founder']==fn,'effect'].iloc[0])
                    code = chr(65 + list(FOUNDER_COLORS.keys()).index(fn)) if fn in FOUNDER_COLORS else None
                    if code and code in _r_founders: return float(r_sub.loc[r_sub['founder']==code,'effect'].iloc[0])
                    return 0.0
                r_eff = [_lookup(fn) for fn in f_names]
                ax.bar(x + w/2, r_eff, w, color=cf, hatch='///', edgecolor='#333',
                       lw=0.6, alpha=0.7, label='miQTL (R)')
                _r2 = (float(np.corrcoef(py_eff, r_eff)[0, 1])
                       if np.std(py_eff) > 0 and np.std(r_eff) > 0 else np.nan)
                ax.text(0.02, 0.97, f'r={_r2:.4f}' if np.isfinite(_r2) else 'r=n/a',
                        transform=ax.transAxes, fontsize=10, va='top')
            ax.axhline(0, color='#444', lw=0.5)
            ax.set(xticks=x, ylabel='effect size')
            ax.set_xticklabels(f_names, rotation=45, ha='right', fontsize=10)
            ax.legend(fontsize=9); sns.despine(ax=ax); plt.tight_layout()
            save_plot(fig, 's4_founder_effects.png')

    # ── discrepancy manhattan ─────────────────────────────────────────────────
    _dm = comp_df.copy()
    _dm['abs_delta'] = (_dm['lod_py'] - _dm['lod_r']).abs()
    _dm['chr_str']   = _dm['chr'].astype(str)
    _dm['x_plot']    = _dm.apply(lambda r: _off.get(r['chr_str'], 0) + r['pos_mb'], axis=1)
    _xt2 = [_off.get(c, 0) + float(_dm[_dm['chr_str']==c]['pos_mb'].median())
            for c in _sorted_chrs if c in _off]
    _cc2 = [CHR_PAL[_sorted_chrs.index(c) % 2] if c in _sorted_chrs else '#aaa'
            for c in _dm['chr_str']]
    fig_dm, ax_dm = plt.subplots(figsize=(14, 3))
    ax_dm.scatter(_dm['x_plot'], _dm['abs_delta'], c=_cc2, s=1, alpha=0.6, edgecolors='none')
    ax_dm.set(xticks=_xt2, xticklabels=[c for c in _sorted_chrs if c in _off],
              xlabel='chromosome', ylabel='|ΔLOD| (py − R)')
    plt.setp(ax_dm.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    sns.despine(ax=ax_dm); plt.tight_layout(); save_plot(fig_dm, 's4_discrepancy_manhattan.png')

    # ── allele effect concordance ─────────────────────────────────────────────
    if ae_df is not None and len(ae_df):
        fig_ae, axes_ae = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(ae_df['r_allele'].dropna(), bins=30, color='#4F6EAF',
                     edgecolor='none', ax=axes_ae[0])
        axes_ae[0].axvline(0, color='#D83026', lw=1, ls='--')
        axes_ae[0].set(xlabel='Pearson $r$ (allele effects py vs R)',
                       ylabel='genes', xlim=(-1, 1))
        sns.despine(ax=axes_ae[0])
        sns.histplot(ae_df['ccc_allele'].dropna(), bins=30, color='#55AF5B',
                     edgecolor='none', ax=axes_ae[1])
        axes_ae[1].axvline(0, color='#D83026', lw=1, ls='--')
        axes_ae[1].set(xlabel='CCC (Lin 1989) — allele effects py vs R',
                       ylabel='genes', xlim=(-1, 1))
        sns.despine(ax=axes_ae[1])
        plt.tight_layout(); save_plot(fig_ae, 's4_allele_effect_concordance.png')

    # ── per-gene peak LOD histogram: R vs Python ──────────────────────────────
    # derive per-gene peak for each side from comp_df
    _pk_r  = comp_df.groupby('gene_id')['lod_r'].max().reset_index().rename(
                columns={'lod_r': 'peak_lod'})
    _pk_py = comp_df.groupby('gene_id')['lod_py'].max().reset_index().rename(
                columns={'lod_py': 'peak_lod'})

    def _bh_egene_count(peak_lods, df):
        '''BH-FDR 5% eGene count from peak LODs via chi2(df).'''
        chi2v = 2 * np.log(10) * np.maximum(peak_lods, 0)
        pv    = sp_stats.chi2.sf(chi2v, df=df)
        n     = len(pv)
        order = np.argsort(pv)
        sig   = pv[order] <= (np.arange(1, n + 1) / n) * 0.05
        return int(sig.sum())

    _n_r_eg  = _bh_egene_count(_pk_r['peak_lod'].values,  J - 1)
    _n_py_eg = _bh_egene_count(_pk_py['peak_lod'].values, J - 1)
    _lod_max = max(_pk_r['peak_lod'].max(), _pk_py['peak_lod'].max())
    _bins    = np.linspace(0, min(_lod_max * 1.05, 40), 50)

    fig_lh, ax_lh = plt.subplots(figsize=(7, 4))
    ax_lh.hist(_pk_r['peak_lod'].values,  bins=_bins, color='#D83026',
               alpha=0.55, label=f'miQTL R  (n={_n_r_eg:,} eGenes)', edgecolor='none')
    ax_lh.hist(_pk_py['peak_lod'].values, bins=_bins, color='#4F6EAF',
               alpha=0.55, label=f'mosaiQTL Python  (n={_n_py_eg:,} eGenes)', edgecolor='none')
    ax_lh.set(xlabel='peak LOD score', ylabel='gene count',
              xlim=(0, None), ylim=(0, None))
    ax_lh.legend(fontsize=9); sns.despine(ax=ax_lh)
    plt.tight_layout(); save_plot(fig_lh, 's4_egene_lod_hist.png')

    # ── eGene count per chromosome: R vs Python ───────────────────────────────
    # BH significance flag per gene per side
    def _bh_flags(peak_df, lod_col, df):
        pv    = sp_stats.chi2.sf(
                    2 * np.log(10) * np.maximum(peak_df[lod_col].values, 0), df=df)
        n     = len(pv); order = np.argsort(pv)
        sig   = np.zeros(n, dtype=bool)
        sig[order] = pv[order] <= (np.arange(1, n + 1) / n) * 0.05
        peak_df = peak_df.copy(); peak_df['sig'] = sig
        return peak_df

    # merge chr back from comp_df peak locus
    _chr_map = comp_df.groupby('gene_id')[['lod_py', 'chr']].apply(
        lambda g: g.loc[g['lod_py'].idxmax(), 'chr'], include_groups=False).reset_index()
    _chr_map.columns = ['gene_id', 'chr']

    _pk_r2  = _bh_flags(_pk_r.merge(_chr_map, on='gene_id'),  'peak_lod', J - 1)
    _pk_py2 = _bh_flags(_pk_py.merge(_chr_map, on='gene_id'), 'peak_lod', J - 1)

    _eg_chrs = sorted(comp_df['chr'].astype(str).unique(), key=_chr_sort)
    _cnt_r   = _pk_r2[_pk_r2['sig']].groupby('chr').size().reindex(_eg_chrs, fill_value=0)
    _cnt_py  = _pk_py2[_pk_py2['sig']].groupby('chr').size().reindex(_eg_chrs, fill_value=0)
    _xb = np.arange(len(_eg_chrs)); _wb = 0.38

    fig_cb, ax_cb = plt.subplots(figsize=(14, 4))
    ax_cb.bar(_xb - _wb/2, _cnt_r.values,  _wb, color='#D83026', alpha=0.75,
              label='miQTL R',  edgecolor='none')
    ax_cb.bar(_xb + _wb/2, _cnt_py.values, _wb, color='#4F6EAF', alpha=0.75,
              label='mosaiQTL Python', edgecolor='none')
    ax_cb.set(xticks=_xb, xticklabels=_eg_chrs,
              xlabel='chromosome of lead locus', ylabel='eGene count',
              xlim=(-0.5, len(_eg_chrs) - 0.5), ylim=(0, None))
    ax_cb.legend(fontsize=9); sns.despine(ax=ax_cb)
    plt.tight_layout(); save_plot(fig_cb, 's4_egene_chr_bar.png')

    # ── scan summary table: R vs Python ───────────────────────────────────────
    def _summary_stats(peak_df, lod_col, comp_all, df, side_label):
        pv    = sp_stats.chi2.sf(
                    2 * np.log(10) * np.maximum(peak_df[lod_col].values, 0), df=df)
        n     = len(pv); order = np.argsort(pv)
        sig   = pv[order] <= (np.arange(1, n + 1) / n) * 0.05
        n_eg  = int(sig.sum())
        min_q = float(pv[order][sig].max()) if n_eg else np.nan   # largest p still passing = weakest
        # min q-value = min BH-adjusted p (rank * alpha / n)
        min_q_bh = float((np.arange(1, n + 1) / n * 0.05)[sig].min()) if n_eg else np.nan
        return {
            'side':         side_label,
            'n_genes':      n,
            'n_loci':       len(comp_all),
            'n_egene':      n_eg,
            'egene_rate':   n_eg / n if n else 0,
            'median_lod':   float(np.median(peak_df[lod_col])),
            'max_lod':      float(peak_df[lod_col].max()),
            'min_q':        min_q_bh,
            'n_chr':        int(comp_all['chr'].astype(str).nunique()),
        }

    _ss_r  = _summary_stats(_pk_r,  'peak_lod', comp_df, J - 1, 'miQTL R')
    _ss_py = _summary_stats(_pk_py, 'peak_lod', comp_df, J - 1, 'mosaiQTL Python')

    _metrics = [
        ('Genes surveyed',    f'{_ss_r["n_genes"]:,}',
                              f'{_ss_py["n_genes"]:,}'),
        ('Cis-loci tested',   f'{_ss_r["n_loci"]:,}',
                              f'{_ss_py["n_loci"]:,}'),
        ('eGenes (5% BH-FDR)',f'{_ss_r["n_egene"]:,}',
                              f'{_ss_py["n_egene"]:,}'),
        ('eGene rate',        f'{_ss_r["egene_rate"]:.1%}',
                              f'{_ss_py["egene_rate"]:.1%}'),
        ('Median peak LOD',   f'{_ss_r["median_lod"]:.1f}',
                              f'{_ss_py["median_lod"]:.1f}'),
        ('Max peak LOD',      f'{_ss_r["max_lod"]:.1f}',
                              f'{_ss_py["max_lod"]:.1f}'),
        ('Min q-value',       (f'{_ss_r["min_q"]:.2e}'  if np.isfinite(_ss_r["min_q"])  else 'n/a'),
                              (f'{_ss_py["min_q"]:.2e}' if np.isfinite(_ss_py["min_q"]) else 'n/a')),
        ('Chromosomes',       f'{_ss_r["n_chr"]} / 20',
                              f'{_ss_py["n_chr"]} / 20'),
        ('LMM df',            f'{J - 1}',
                              f'{J - 1}'),
    ]

    fig_tb, ax_tb = plt.subplots(figsize=(9, len(_metrics) * 0.52 + 0.6))
    ax_tb.axis('off')
    col_labels = ['METRIC', 'miQTL R', 'mosaiQTL Python']
    tbl = ax_tb.table(
        cellText=_metrics,
        colLabels=col_labels,
        loc='center', cellLoc='center',
        colWidths=[0.50, 0.25, 0.25],   # symmetric value columns
    )
    tbl.auto_set_font_size(False); tbl.set_fontsize(10)
    # uniform row height
    for (row, col), cell in tbl.get_celld().items():
        cell.set_height(0.1)
    # header row
    for col in range(3):
        tbl[(0, col)].set_facecolor('#2b2b2b')
        tbl[(0, col)].set_text_props(color='white', fontweight='bold')
    # data rows: alternating shading, left-align metric, colour values
    for row in range(1, len(_metrics) + 1):
        bg = '#f5f5f5' if row % 2 else 'white'
        tbl[(row, 0)].set_facecolor(bg)
        tbl[(row, 0)].set_text_props(ha='left')
        tbl[(row, 0)]._loc = 'left'
        for col in (1, 2):
            tbl[(row, col)].set_facecolor(bg)
        tbl[(row, 1)].set_text_props(color='#D83026', fontweight='bold')
        tbl[(row, 2)].set_text_props(color='#4F6EAF', fontweight='bold')
    plt.tight_layout(); save_plot(fig_tb, 's4_scan_summary.png')


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
           xlabel='chromosome', ylabel='loci reduced (%)'
                 f'py={mean_py:.1f}%  r={mean_r:.1f}%  '
                 f'(reduction = loci with redundant haplotype structure; '
                 f'kinship+LOD panels confirm no downstream distortion)')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    ax.legend(fontsize=9); sns.despine(ax=ax)

    ax = axes[1]
    ax.scatter(range(len(k_cors)), k_cors, s=20, color='#4F6EAF', edgecolors='none')
    ax.axhline(0.99, color='#D83026', lw=1, ls='--', label='r=0.99')
    ax.set(xlabel='chr index', ylabel='kinship $r$ (full vs collapsed)', ylim=(0.9, 1.005))
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
    ax.set(xlabel='max LOD', ylabel='density')
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
           xlabel='chromosome', ylabel='|Δ threshold| (py − R)')
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
    axes[1].set(xlabel='position (Mb)', ylabel='LOD')
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
                   ylabel='mean LOD difference per gene')
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
        ax_sp8.set(xlabel='position chr8 (Mb)', ylabel='SNP LOD', xlim=(0, None), ylim=(0, None))
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
                   ylabel='|ΔLOD| (py − R)',
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
    axes[1].set(xlabel='position (Mb)', ylabel='LOD (base)')
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
    ax.set(xlabel='peak position (Mb)', ylabel='bootstraps')
    ax.legend(fontsize=8); sns.despine(ax=ax)

    ax = axes[1]
    for ri, row in ci_comp.reset_index().iterrows():
        col = '#4F6EAF' if row['overlap'] else '#D83026'
        ax.hlines(ri,        row['ci_lo_py'], row['ci_hi_py'], colors=col, lw=4, alpha=0.7)
        ax.hlines(ri + 0.15, row['ci_lo_r'],  row['ci_hi_r'],  colors='#888', lw=4, alpha=0.5)
    ax.set(xlabel='position (Mb)', ylabel='gene index')
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
    axes[1].set(xlabel='position (Mb)', ylabel='mean bootstrap LOD')
    axes[1].legend(fontsize=9); sns.despine(ax=axes[1])
    plt.tight_layout(); save_plot(fig, 's11_instability.png')

    # peak proportion vs R mean LOD; validates bootstrap sample size equivalence
    if hasattr(inst_py, 'p_value') and inst_py.p_value is not None and len(mg_fx) > 1:
        _lids = list(inst_py.locus_ids) if hasattr(inst_py.locus_ids, '__iter__') else []
        _pvs  = list(inst_py.p_value)
        _pv_map = dict(zip(_lids, _pvs))
        mg_fx = mg_fx.copy()
        mg_fx['pv_py'] = mg_fx['locus'].map(_pv_map)
        _pv = mg_fx.dropna(subset=['pv_py', 'lod_r'])
        if len(_pv) > 1:
            fig_pv, ax_pv = plt.subplots(figsize=(5, 4))
            ax_pv.scatter(_pv['lod_r'], 1 - _pv['pv_py'],
                          s=12, color='#4F6EAF', alpha=0.7, edgecolors='none')
            ax_pv.set(xlabel='mean bootstrap LOD (R)',
                      ylabel='peak proportion (Python)',
                      xlim=(0, None), ylim=(0, 1))
            sns.despine(ax=ax_pv)
            plt.tight_layout(); save_plot(fig_pv, 's11_peak_proportion.png')


# ─────────────────────────────────────────────────────────────────────────────
# standalone replot: python validation/plot_phase2.py [--py_data ...] [--r_data ...] [--plots ...] [--sections 1,4,6]
# regenerates all plots from CSVs already written by validate_port.py.
# complex objects are reconstructed from their serialized CSVs:
#   hmap_data        ← haplotype_heatmap_py.csv
#   py_betas         ← peak_founder_effects_py.csv
#   boot_pos         ← boot_pos_py.csv
#   inst_py.p_value  ← instability_pvals_py.csv
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse, logging as _logging
    ap = argparse.ArgumentParser(description='replot phase2 from CSVs')
    ap.add_argument('--py_data',  default='validation/phase2/py_data')
    ap.add_argument('--r_data',   default='validation/phase2/r_data')
    ap.add_argument('--plots',    default='validation/phase2/plots')
    ap.add_argument('--sections', default=None, help='comma-separated, e.g. 1,4,6')
    _a = ap.parse_args()

    _log = _logging.getLogger('replot')
    init(_a.plots, _a.py_data, _log)
    _run = (set(int(s) for s in _a.sections.split(',')) if _a.sections else set(range(1, 12)))
    _cs  = lambda c: int(c) if str(c).isdigit() else (20 if str(c).upper()=='X' else 99)

    def _ld(name, side='py'):
        '''load CSV from py_data or r_data; return None if missing.'''
        p = os.path.join(_a.r_data if side == 'r' else _a.py_data, name)
        if not os.path.exists(p): return None
        df = pd.read_csv(p, low_memory=False)
        if 'chr' in df.columns: df['chr'] = df['chr'].astype(str)
        return df

    # S1 ─────────────────────────────────────────────────────────────────────
    if 1 in _run:
        _d = _ld('cache_info_py.csv')
        if _d is not None:
            _rf = _ld('founders_r.csv', 'r'); _rs = _ld('sample_ids_r.csv', 'r')
            _rfnd = _rf['founder'].tolist() if _rf is not None else []
            _rsid = _rs['sample_id'].tolist() if _rs is not None else []
            plot_s1(_d, _rfnd, _rfnd, _rsid, _rsid, None, None)

    # S2 — reconstruct hmap_data arrays from CSV if available; else prob-scatter only
    if 2 in _run:
        _hm_csv = _ld('haplotype_heatmap_py.csv')
        _d      = _ld('haplotype_val_comparison.csv')
        _hmap   = {}
        if _hm_csv is not None:
            for _hc, _grp in _hm_csv.groupby('chr'):
                _n_s = int(_grp['sample_idx'].max()) + 1
                _n_f = int(_grp['founder_idx'].max()) + 1
                _Hpy = np.zeros((_n_s, _n_f)); _Hr = np.zeros((_n_s, _n_f))
                for _, _row in _grp.iterrows():
                    _Hpy[int(_row['sample_idx']), int(_row['founder_idx'])] = _row['prob_py']
                    _Hr [int(_row['sample_idx']), int(_row['founder_idx'])] = _row['prob_r']
                _hmap[str(_hc)] = (_Hpy, _Hr, str(_grp['locus'].iloc[0]))
        if _d is not None:
            _rp = _d['r_prob'].values; _pp = _d['py_prob'].values
            _ok = np.isfinite(_rp) & np.isfinite(_pp)
            _vr  = float(np.corrcoef(_rp[_ok], _pp[_ok])[0,1]) if _ok.sum()>3 else np.nan
            _mad = float(np.abs(_rp[_ok]-_pp[_ok]).mean()) if _ok.sum()>0 else np.nan
            _fnames_s2 = sorted(_hmap.keys(), key=lambda c: int(c) if c.isdigit() else 99)
            plot_s2(_hmap, _fnames_s2, [], [],
                    _rp.tolist(), _pp.tolist(), _vr, _mad, _d.to_dict('records'))

    # S3 — load first chr matrix pair ────────────────────────────────────────
    if 3 in _run:
        _kc = _ld('kinship_comparison.csv')
        if _kc is not None:
            _chk = str(_kc['chr'].iloc[0])
            _kp_df = _ld(f'kinship_py_chr{_chk}.csv')
            _kr_df = _ld(f'kinship_mat_r_chr{_chk}.csv', 'r')
            if _kp_df is not None and _kr_df is not None:
                _sp = _kp_df['sample_id'].tolist(); _sr = _kr_df['sample_id'].tolist()
                _sh = [s for s in _sp if s in _sr]
                _pi = np.array([_sp.index(s) for s in _sh])
                _ri = np.array([_sr.index(s) for s in _sh])
                _kp = _kp_df.drop(columns='sample_id').values.astype(float)[np.ix_(_pi,_pi)]
                _kr = _kr_df.drop(columns='sample_id').values.astype(float)[np.ix_(_ri,_ri)]
                plot_s3(_kp, _kr, _sh, _chk, float(np.abs(_kp-_kr).max()))

    # S4 ─────────────────────────────────────────────────────────────────────
    if 4 in _run:
        _comp = _ld('lod_comparison.csv'); _stat = _ld('gene_summary.csv')
        _ae   = _ld('allele_effect_concordance.csv')
        _pfe  = _ld('peak_founder_effects_py.csv')
        if _comp is not None and _stat is not None:
            _co   = sorted(_comp['chr'].astype(str).unique(), key=_cs)
            _ccfn = lambda s: chr_colors(s, _co)
            _pybeta = {}
            if _pfe is not None:
                for _gid, _grp in _pfe.groupby('gene_id'):
                    _grp = _grp.sort_values('founder')
                    _pybeta[_gid] = (str(_grp['chr'].iloc[0]),
                                     str(_grp['locus'].iloc[0]),
                                     _grp['founder'].tolist(),
                                     _grp['effect_py'].tolist())
            plot_s4(_comp, _stat, _ccfn, _pybeta, None,
                    int((_stat['tier']=='strict').sum()),
                    int((_stat['tier']=='lenient').sum()),
                    int((_stat['tier']=='fail').sum()),
                    9, ae_df=_ae)

    # S5 ─────────────────────────────────────────────────────────────────────
    if 5 in _run:
        _cd = _ld('collapse_comparison.csv'); _kc = _ld('kinship_comparison.csv')
        if _cd is not None and _kc is not None:
            _kcors = _kc['r'].tolist()
            plot_s5(_cd, _kcors,
                    float(_cd['pct_py'].mean()),
                    float(_cd['pct_r'].mean(skipna=True)) if 'pct_r' in _cd else np.nan,
                    float(np.mean(_kcors)) if _kcors else 0.0)

    # S6 — refit GEV from pooled perm lods ───────────────────────────────────
    if 6 in _run:
        _pp = _ld('perm_lods_py.csv'); _tc = _ld('threshold_comparison.csv')
        _rp = _ld('perm_lods_r.csv', 'r')
        if _pp is not None and _tc is not None:
            _ml = _pp['perm_lod'].dropna().values
            _c, _loc, _sc = _gev.fit(_ml)
            _gp = {'c':_c,'loc':_loc,'scale':_sc}
            plot_s6(_ml, _gp,
                    float(_gev.ppf(0.95, _c, loc=_loc, scale=_sc)),
                    float(_tc['threshold_r'].mean()),
                    float(_tc['delta'].mean()), _rp, _tc)

    # S7 ─────────────────────────────────────────────────────────────────────
    if 7 in _run:
        _d = _ld('mi_lod_comparison.csv')
        if _d is not None:
            _diffs = _d.groupby('gene_id')[['lod_py', 'lod_r']].apply(
                lambda g: float((g['lod_py']-g['lod_r']).mean()), include_groups=False).values
            plot_s7(_d, mi_diffs=_diffs if len(_diffs)>1 else None)

    # S8 ─────────────────────────────────────────────────────────────────────
    if 8 in _run:
        _d = _ld('snp_lod_comparison.csv')
        if _d is not None: plot_s8(_d)

    # S9 ─────────────────────────────────────────────────────────────────────
    if 9 in _run:
        _d = _ld('mediation_comparison.csv'); _rf = _ld('mediation_r.csv', 'r')
        if _d is not None:
            _drop = _rf['lod_drop'].dropna().values if (_rf is not None and 'lod_drop' in _rf.columns) else None
            plot_s9(_d, float(np.corrcoef(_d['lod_r'],_d['lod_py'])[0,1]), drop_r=_drop)

    # S10 ────────────────────────────────────────────────────────────────────
    if 10 in _run:
        _d = _ld('ci_comparison.csv'); _rb = _ld('boot_pos_r.csv', 'r')
        _bp_df = _ld('boot_pos_py.csv')
        if _d is not None:
            _bp_arr = _bp_df['boot_pos'].values if _bp_df is not None else None
            plot_s10(_d, _bp_arr, _rb,
                     str(_d['gene_id'].iloc[-1]), str(_d['chr'].iloc[-1]),
                     int(_d['overlap'].sum()))

    # S11 ────────────────────────────────────────────────────────────────────
    if 11 in _run:
        _d = _ld('fixef_comparison.csv'); _ipv = _ld('instability_pvals_py.csv')
        if _d is not None:
            _r = float(np.corrcoef(_d['lod_r'],_d['lod_py'])[0,1]) if len(_d)>3 else np.nan
            class _InstProxy:  # lightweight stand-in for instability scan result object
                pass
            _iprx = _InstProxy()
            if _ipv is not None:
                _iprx.locus_ids = _ipv['locus'].tolist()
                _iprx.p_value   = _ipv['p_value'].tolist()
            else:
                _iprx.locus_ids = []
                _iprx.p_value   = None
            plot_s11(_d, _r, _iprx)

    print(f'plots written to {_a.plots}/')

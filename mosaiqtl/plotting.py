"""
mosaiqtl.plotting — qc and results visualization

all plots use seaborn over matplotlib where possible.
font: carlito. axes start from zero.
founder colors follow the cc palette convention.
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for headless use
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logger = logging.getLogger('mosaiqtl')

# cc founder color palette (user preference)
FOUNDER_COLORS = {
    'AJ':     '#f4c245',
    'B6J':    '#c0bfbf',
    '129S1J': '#da9cc1',
    'NODJ':   '#4f6eaf',
    'NZOJ':   '#52a5db',
    'CASTJ':  '#55af5b',
    'PWKJ':   '#d83026',
    'WSBJ':   '#683c91',
}

F1_COLORS = {
    'B6AF1J':     '#dac082',
    'B6129S1F1J': '#cdadc0',
    'B6CASTF1J':  '#8ab78d',
    'B6NODF1J':   '#8796b7',
    'B6NZOF1J':   '#89b2cd',
    'B6PWKF1J':   '#cc7772',
    'B6WSBF1J':   '#947da8',
}

# combine for general use
ALL_STRAIN_COLORS = {**FOUNDER_COLORS, **F1_COLORS}


def set_style():
    """set consistent plot style."""
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Carlito', 'DejaVu Sans', 'Arial'],
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
    })


def plot_lod_profile(scan_result, true_params=None, save_path=None):
    """genome-wide lod profile (manhattan-style).

    parameters
    ----------
    scan_result : ScanResult
        output of scan.scan_h2lmm()
    true_params : dict or None
        if provided, marks the true causal locus
    save_path : str or None
        save to file if provided
    """
    set_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    df = scan_result.to_dataframe()
    chromosomes = sorted(df['chromosome'].unique(), key=lambda x: int(x) if x.isdigit() else 99)

    # assign cumulative x positions per chromosome
    chr_offsets = {}
    offset = 0
    for chrom in chromosomes:
        chr_offsets[chrom] = offset
        chr_mask = df['chromosome'] == chrom
        if chr_mask.sum() > 0:
            offset += df.loc[chr_mask, 'position'].max() + 10  # gap between chr

    x_positions = np.array([
        chr_offsets[row['chromosome']] + row['position']
        for _, row in df.iterrows()
    ])

    # alternate chr colors
    chr_color_map = {}
    palette = ['#4f6eaf', '#c0bfbf']
    for i, chrom in enumerate(chromosomes):
        chr_color_map[chrom] = palette[i % 2]

    colors = [chr_color_map[c] for c in df['chromosome']]

    # scatter lod scores
    ax.scatter(x_positions, df['lod'], c=colors, s=12, alpha=0.7,
               edgecolors='none', rasterized=True)

    # mark peak
    peak = scan_result.peak_locus()
    peak_x = chr_offsets[peak['chromosome']] + peak['position']
    ax.axhline(y=peak['lod'], color='#d83026', linewidth=0.5,
               linestyle='--', alpha=0.5)
    ax.annotate(
        f'peak: chr{peak["chromosome"]}\nLOD={peak["lod"]:.1f}',
        xy=(peak_x, peak['lod']),
        xytext=(peak_x, peak['lod'] + 1),
        fontsize=9, color='#d83026', ha='center',
        arrowprops=dict(arrowstyle='->', color='#d83026', lw=0.8),
    )

    # mark true locus if provided
    if true_params is not None:
        true_chrom = true_params.get('causal_chrom', None)
        true_idx = true_params.get('causal_locus_idx', None)
        if true_chrom and true_idx is not None:
            true_mask = df['chromosome'] == true_chrom
            if true_mask.sum() > true_idx:
                true_pos = df.loc[true_mask, 'position'].iloc[true_idx]
                true_x = chr_offsets[true_chrom] + true_pos
                ax.axvline(x=true_x, color='#55af5b', linewidth=1,
                           linestyle=':', alpha=0.7, label='true causal locus')
                ax.legend(fontsize=9)

    # chr labels
    for chrom in chromosomes:
        chr_mask = df['chromosome'] == chrom
        if chr_mask.sum() > 0:
            mid_x = chr_offsets[chrom] + df.loc[chr_mask, 'position'].median()
            ax.text(mid_x, -0.5, f'chr{chrom}', ha='center', fontsize=8,
                    color='#444444')

    ax.set_ylim(bottom=0)
    ax.set_ylabel('LOD score')
    ax.set_xlabel('')
    ax.set_title('genome-wide LOD profile')
    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    sns.despine(bottom=True)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_kinship_heatmap(k_matrix, sample_ids=None, save_path=None):
    """kinship matrix heatmap.

    parameters
    ----------
    k_matrix : np.ndarray (n, n)
    sample_ids : list of str or None
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        k_matrix, ax=ax,
        cmap='YlOrRd', vmin=0, square=True,
        xticklabels=False, yticklabels=False,
        cbar_kws={'label': 'kinship', 'shrink': 0.7},
    )

    # annotate structure blocks if sample_ids available
    if sample_ids is not None:
        n = len(sample_ids)
        # count founders, f1s, cc strains
        n_founders = sum(1 for s in sample_ids if s in FOUNDER_COLORS)
        n_f1 = sum(1 for s in sample_ids if s in F1_COLORS)

        if n_founders > 0:
            ax.axhline(y=n_founders, color='white', linewidth=1.5)
            ax.axvline(x=n_founders, color='white', linewidth=1.5)
        if n_founders + n_f1 > 0 and n_f1 > 0:
            ax.axhline(y=n_founders + n_f1, color='white', linewidth=1)
            ax.axvline(x=n_founders + n_f1, color='white', linewidth=1)

    ax.set_title('LOCO kinship matrix')

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_founder_effects(scan_result, genome_cache, peak_chrom=None,
                         peak_idx=None, founder_names=None,
                         save_path=None):
    """founder allele effects at the peak locus.

    parameters
    ----------
    scan_result : ScanResult
    genome_cache : GenomeCache
    peak_chrom : str or None
        if none, uses the genome-wide peak
    peak_idx : int or None
    founder_names : list of str or None
    save_path : str or None
    """
    set_style()

    if peak_chrom is None:
        peak = scan_result.peak_locus()
        # find the index within that chromosome
        chr_mask = scan_result.chromosomes == peak['chromosome']
        chr_indices = np.where(chr_mask)[0]
        chr_lods = scan_result.lod[chr_mask]
        peak_idx = chr_indices[np.argmax(chr_lods)]
        peak_chrom = peak['chromosome']

    betas = scan_result.betas[peak_idx]
    j_minus_1 = len(betas)

    if founder_names is None:
        founder_names = genome_cache.founders[:j_minus_1]

    # add the dropped founder (effect = 0, reference)
    all_names = list(founder_names) + [genome_cache.founders[-1]]
    all_effects = np.append(betas, 0.0)

    # color by founder palette
    colors = [FOUNDER_COLORS.get(name, '#888888') for name in all_names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(range(len(all_names)), all_effects, color=colors,
                  edgecolor='white', linewidth=0.5)

    ax.axhline(y=0, color='#444444', linewidth=0.5)
    ax.set_xticks(range(len(all_names)))
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('founder effect ($\\hat{\\beta}_f$)')
    ax.set_title(f'founder haplotype effects at peak (chr{peak_chrom})')
    sns.despine()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_pvalue_qq(scan_result, save_path=None):
    """qq plot of p-values (uniform null expectation).

    parameters
    ----------
    scan_result : ScanResult
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    pvals = scan_result.p_value[scan_result.p_value > 0]
    pvals = np.sort(pvals)
    n = len(pvals)

    observed = -np.log10(pvals)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))

    ax.scatter(expected, observed, s=10, alpha=0.6, color='#4f6eaf',
               edgecolors='none', rasterized=True)
    ax.plot([0, max(expected)], [0, max(expected)], '--',
            color='#d83026', linewidth=1, label='$y = x$ (null)')

    # genomic inflation factor
    median_chi2 = np.median(observed) / 0.456  # chi2 df=1 median
    lambda_gc = np.median(pvals) / 0.5
    ax.text(0.05, 0.95, f'$\\lambda_{{GC}}$ = {1/lambda_gc:.2f}',
            transform=ax.transAxes, fontsize=10, va='top')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('expected $-\\log_{10}(p)$')
    ax.set_ylabel('observed $-\\log_{10}(p)$')
    ax.set_title('QQ plot')
    ax.legend(fontsize=9)
    sns.despine()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_dispersion_vs_truth(phi_estimated, phi_true, save_path=None):
    """scatter plot of estimated vs true per-subject dispersion.

    parameters
    ----------
    phi_estimated : np.ndarray
    phi_true : np.ndarray
    save_path : str or None
    """
    set_style()
    fig, ax = plt.subplots(figsize=(5, 5))

    valid = ~np.isnan(phi_estimated) & ~np.isnan(phi_true)
    phi_e = phi_estimated[valid]
    phi_t = phi_true[valid]

    ax.scatter(phi_t, phi_e, s=20, alpha=0.6, color='#683c91',
               edgecolors='white', linewidth=0.3)

    # diagonal
    lims = [0, max(phi_t.max(), phi_e.max()) * 1.1]
    ax.plot(lims, lims, '--', color='#d83026', linewidth=1, label='$y = x$')

    # correlation
    corr = np.corrcoef(phi_t, phi_e)[0, 1]
    ax.text(0.05, 0.95, f'$r$ = {corr:.3f}\n$n$ = {valid.sum()}',
            transform=ax.transAxes, fontsize=10, va='top')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('true $\\hat{\\phi}_i$')
    ax.set_ylabel('estimated $\\hat{\\phi}_i$')
    ax.set_title('per-subject dispersion estimation')
    ax.legend(fontsize=9)
    sns.despine()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


# ----------------------------------------------------------------
# cross-language comparison (mosaiqtl vs miqtl/r)
# ----------------------------------------------------------------

def plot_lod_correlation(py_scan_result, r_results_csv, save_path=None):
    """scatter of mosaiqtl lod vs miqtl lod at every shared locus.

    parameters
    ----------
    py_scan_result : ScanResult
        mosaiqtl scan output
    r_results_csv : str
        path to r validation scan_results.csv
    save_path : str or None
    """
    import pandas as pd
    set_style()

    py_df = py_scan_result.to_dataframe()
    r_df = pd.read_csv(r_results_csv)

    # merge on locus_id
    merged = py_df.merge(r_df, left_on='locus_id', right_on='locus',
                         suffixes=('_py', '_r'))

    if len(merged) == 0:
        logger.warning('no shared loci between python and r results')
        return None, None

    fig, ax = plt.subplots(figsize=(5, 5))

    ax.scatter(merged['lod_r'], merged['lod_py'], s=12, alpha=0.6,
               color='#4f6eaf', edgecolors='none', rasterized=True)

    # diagonal
    lims = [0, max(merged['lod_r'].max(), merged['lod_py'].max()) * 1.1]
    ax.plot(lims, lims, '--', color='#d83026', linewidth=1, label='$y = x$')

    # correlation
    corr = np.corrcoef(merged['lod_r'], merged['lod_py'])[0, 1]
    max_diff = np.max(np.abs(merged['lod_py'] - merged['lod_r']))
    ax.text(0.05, 0.95,
            f'$r$ = {corr:.6f}\nmax $|\\Delta|$ = {max_diff:.4f}\n$n$ = {len(merged)}',
            transform=ax.transAxes, fontsize=10, va='top')

    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('miQTL LOD (R)')
    ax.set_ylabel('mosaiQTL LOD (Python)')
    ax.set_title('cross-language LOD comparison')
    ax.legend(fontsize=9)
    sns.despine()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_lod_difference(py_scan_result, r_results_csv, save_path=None):
    """residual plot: lod difference (python - r) by genomic position.

    parameters
    ----------
    py_scan_result : ScanResult
    r_results_csv : str
    save_path : str or None
    """
    import pandas as pd
    set_style()

    py_df = py_scan_result.to_dataframe()
    r_df = pd.read_csv(r_results_csv)

    merged = py_df.merge(r_df, left_on='locus_id', right_on='locus',
                         suffixes=('_py', '_r'))

    if len(merged) == 0:
        return None, None

    fig, ax = plt.subplots(figsize=(12, 3))

    diff = merged['lod_py'] - merged['lod_r']

    # color by chromosome
    chromosomes = sorted(merged['chromosome'].unique(),
                         key=lambda x: int(x) if x.isdigit() else 99)
    palette = ['#4f6eaf', '#c0bfbf']
    colors = [palette[chromosomes.index(c) % 2] for c in merged['chromosome']]

    ax.scatter(range(len(diff)), diff, c=colors, s=8, alpha=0.6,
               edgecolors='none', rasterized=True)
    ax.axhline(y=0, color='#d83026', linewidth=0.8, linestyle='--')

    # annotate stats
    ax.text(0.98, 0.95,
            f'mean $\\Delta$ = {diff.mean():.4f}\n'
            f'sd $\\Delta$ = {diff.std():.4f}\n'
            f'max $|\\Delta|$ = {np.abs(diff).max():.4f}',
            transform=ax.transAxes, fontsize=9, va='top', ha='right')

    ax.set_ylim(bottom=min(-0.1, diff.min() * 1.2),
                top=max(0.1, diff.max() * 1.2))
    ax.set_xlabel('locus index')
    ax.set_ylabel('$\\Delta$ LOD (Python $-$ R)')
    ax.set_title('LOD residuals: mosaiQTL vs miQTL')
    sns.despine()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_founder_effects_comparison(py_scan_result, genome_cache,
                                     r_effects_csv, save_path=None):
    """side-by-side founder effects at peak: python vs r.

    parameters
    ----------
    py_scan_result : ScanResult
    genome_cache : GenomeCache
    r_effects_csv : str
        path to r allele_effects.csv
    save_path : str or None
    """
    import pandas as pd
    set_style()

    # python: peak locus effects
    peak = py_scan_result.peak_locus()
    chr_mask = py_scan_result.chromosomes == peak['chromosome']
    chr_indices = np.where(chr_mask)[0]
    peak_idx = chr_indices[np.argmax(py_scan_result.lod[chr_mask])]
    py_betas = py_scan_result.betas[peak_idx]

    j_minus_1 = len(py_betas)
    founder_names = genome_cache.founders[:j_minus_1]
    ref_founder = genome_cache.founders[-1]

    py_all = np.append(py_betas, 0.0)
    all_names = list(founder_names) + [ref_founder]

    # r: read effects for the same gene and peak locus
    r_effects = None
    if os.path.exists(r_effects_csv):
        r_df = pd.read_csv(r_effects_csv)
        # match the peak locus
        r_match = r_df[r_df['locus'] == peak['locus_id']]
        if len(r_match) > 0:
            r_effects = r_match.set_index('founder')['effect']

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(all_names))
    width = 0.35

    # python bars
    colors_py = [FOUNDER_COLORS.get(n, '#888888') for n in all_names]
    ax.bar(x - width / 2, py_all, width, color=colors_py,
           edgecolor='white', linewidth=0.5, label='mosaiQTL (Python)')

    # r bars (if available)
    if r_effects is not None:
        r_all = np.array([r_effects.get(n, 0.0) for n in all_names])
        ax.bar(x + width / 2, r_all, width, color=colors_py,
               edgecolor='#444444', linewidth=0.5, alpha=0.5,
               label='miQTL (R)')

    ax.axhline(y=0, color='#444444', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('founder effect ($\\hat{\\beta}_f$)')
    ax.set_title(f'founder effects comparison at peak (chr{peak["chromosome"]})')
    ax.legend(fontsize=9)
    sns.despine()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f'saved: {save_path}')
    return fig, ax


def plot_validation_panel(scan_result, k_matrix, genome_cache,
                          true_params=None, phi_estimated=None,
                          phi_true=None, save_dir='validation/qc_plots'):
    """generate full qc panel for validation.

    generates two sets of plots:
      1. internal qc (always): lod profile, kinship, founder effects,
         qq plot, dispersion scatter
      2. cross-language comparison (if r output exists): lod correlation,
         lod residuals, founder effects comparison

    also exports python scan results as csv for r comparison.

    parameters
    ----------
    scan_result : ScanResult
    k_matrix : np.ndarray
    genome_cache : GenomeCache
    true_params : dict or None
    phi_estimated : np.ndarray or None
    phi_true : np.ndarray or None
    save_dir : str
    """
    import pandas as pd
    os.makedirs(save_dir, exist_ok=True)

    # --- export python results for r comparison ---
    py_df = scan_result.to_dataframe()
    py_csv = os.path.join(save_dir, 'py_scan_results.csv')
    py_df.to_csv(py_csv, index=False)
    logger.info(f'exported python scan results: {py_csv}')

    # also export h2 per chromosome
    if hasattr(scan_result, 'h2_null') and scan_result.h2_null:
        h2_rows = [{'chromosome': c, 'h2_null': h}
                    for c, h in scan_result.h2_null.items()]
        pd.DataFrame(h2_rows).to_csv(
            os.path.join(save_dir, 'py_h2_null.csv'), index=False
        )

    # --- 1. internal qc plots (always generated) ---
    plot_lod_profile(
        scan_result, true_params=true_params,
        save_path=os.path.join(save_dir, 'lod_profile.png')
    )
    plot_kinship_heatmap(
        k_matrix, sample_ids=list(genome_cache.sample_ids),
        save_path=os.path.join(save_dir, 'kinship_heatmap.png')
    )
    plot_founder_effects(
        scan_result, genome_cache,
        save_path=os.path.join(save_dir, 'founder_effects.png')
    )
    plot_pvalue_qq(
        scan_result,
        save_path=os.path.join(save_dir, 'pvalue_qq.png')
    )
    if phi_estimated is not None and phi_true is not None:
        plot_dispersion_vs_truth(
            phi_estimated, phi_true,
            save_path=os.path.join(save_dir, 'dispersion_scatter.png')
        )

    # --- 2. cross-language comparison (if r output available) ---
    r_output_dir = os.path.join(os.path.dirname(save_dir), 'r_data', 'r_output')
    r_scan_csv = os.path.join(r_output_dir, 'scan_results.csv')
    r_effects_csv = os.path.join(r_output_dir, 'allele_effects.csv')

    if os.path.exists(r_scan_csv):
        logger.info('r output detected — generating cross-language comparison plots')

        plot_lod_correlation(
            scan_result, r_scan_csv,
            save_path=os.path.join(save_dir, 'cross_lod_correlation.png')
        )
        plot_lod_difference(
            scan_result, r_scan_csv,
            save_path=os.path.join(save_dir, 'cross_lod_residuals.png')
        )
        if os.path.exists(r_effects_csv):
            plot_founder_effects_comparison(
                scan_result, genome_cache, r_effects_csv,
                save_path=os.path.join(save_dir, 'cross_founder_effects.png')
            )
    else:
        logger.info(
            f'no r output at {r_scan_csv} — skipping cross-language plots. '
            f'run: Rscript validation/r_reference/run_validation.R validation/r_data/'
        )

    plt.close('all')
    logger.info(f'qc panel saved to {save_dir}/')

"""Standalone slide-figure generator: nova vs next, same celltypes.

Reads existing per-celltype scan outputs directly from CSV — no HDF5, no
scan re-run. All plotting goes through mosaiqtl.viz.compare.lod_scatter,
the same production function behind the celltypes/compare/*_lod.png
reference figures (density curves on each axis, viridis-by-LOD points,
dashed y=x diagonal, orange dotted threshold lines + per-quadrant eGene
counts, r²/r_sp/n stats box) — this script supplies data, the package
supplies style.

GENE-CALLING DISCREPANCY (must read before trusting any gene-level match):
nova's own ``gene_symbol`` column is not a gene symbol — it is the
versioned Ensembl ID (e.g. ``ENSMUSG00000034701.10``) for every single row
(confirmed 18,862/18,862 in glutamatergic_neuron/core/peak_lod.csv), while
next's ``gene_symbol`` is a real symbol for all but a genuine no-symbol
minority (1,126/14,950 fall back to the Ensembl ID). Root cause is upstream
of this script (nova's gene_info/annotation join), not addressed here.
Consequence: matching genes between generations by ``gene_symbol`` silently
drops or misjoins nearly every nova row. This script joins on the
unversioned ``gene_id`` (stripping any ``.NN`` version suffix defensively
on both sides) and always displays next's ``gene_symbol`` as the label,
never nova's.
"""
from __future__ import annotations

import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import pandas as pd

from mosaiqtl.viz._theme import INK
from mosaiqtl.viz.compare import lod_scatter

log = logging.getLogger('compare_generations')

_DEFAULT_CTS = (
    'astrocyte',
    'gabaergic_interneuron',
    'glutamatergic_neuron',
    'microglial_cell',
    'oligodendrocyte',
)


def _strip_version(gene_id: pd.Series) -> pd.Series:
    """Drop a trailing ``.NN`` Ensembl version suffix, if present."""
    return gene_id.astype(str).str.replace(r'\.\d+$', '', regex=True)


def _load_peak_lod(ct_dir: pathlib.Path) -> pd.DataFrame:
    """Load one generation's ``peak_lod.csv``, keyed on unversioned gene_id."""
    df = pd.read_csv(ct_dir / 'core' / 'peak_lod.csv')
    df['gene_id'] = _strip_version(df['gene_id'])
    return df


def _load_threshold(ct_dir: pathlib.Path) -> float:
    """Read the genome-wide FWER threshold from ``core/threshold.csv``."""
    thr = pd.read_csv(ct_dir / 'core' / 'threshold.csv')
    row = thr.loc[thr['chr'] == 'genome_wide', 'threshold']
    return float(row.iloc[0])


def compare_celltype(
    ct: str,
    next_dir: pathlib.Path,
    nova_dir: pathlib.Path,
    ref_gene: str,
    out_dir: pathlib.Path,
) -> dict[str, object]:
    """Compare one celltype's next vs nova scan and save a slide figure.

    Args:
        ct: Celltype label (used for titles/filenames only).
        next_dir: Path to ``.../next/celltypes/<ct>``.
        nova_dir: Path to ``.../nova/celltypes/<ct>``.
        ref_gene: Reference gene symbol (matched via next's gene_symbol,
            then carried across generations by gene_id) to highlight.
        out_dir: Directory to write the comparison PNG into.

    Returns:
        Summary dict (celltype, thresholds, eGene counts, ref-gene peak
        LOD in each generation) — one row for the printed/CSV summary
        table.
    """
    next_pk = _load_peak_lod(next_dir)
    nova_pk = _load_peak_lod(nova_dir)
    next_thr = _load_threshold(next_dir)
    nova_thr = _load_threshold(nova_dir)

    merged = next_pk[['gene_id', 'gene_symbol', 'peak_lod']].merge(
        nova_pk[['gene_id', 'peak_lod']], on='gene_id', suffixes=('_next', '_nova'), how='inner',
    )
    log.info('%s: %d next genes, %d nova genes, %d shared by gene_id',
             ct, len(next_pk), len(nova_pk), len(merged))

    ref_row = next_pk.loc[next_pk['gene_symbol'].str.lower() == ref_gene.lower()]
    ref_gene_id = ref_row['gene_id'].iloc[0] if len(ref_row) else None
    ref_next_lod = float(ref_row['peak_lod'].iloc[0]) if len(ref_row) else None
    ref_nova_lod = None
    if ref_gene_id is not None:
        nova_hit = nova_pk.loc[nova_pk['gene_id'] == ref_gene_id]
        if len(nova_hit):
            ref_nova_lod = float(nova_hit['peak_lod'].iloc[0])

    out_path = out_dir / f'{ct}_next_vs_nova_lod.png'
    lod_scatter(
        merged,
        x_col='peak_lod_next',
        y_col='peak_lod_nova',
        x_label='next peak LOD',
        y_label='nova peak LOD',
        priority_genes={ref_gene: INK} if ref_gene_id is not None else None,
        threshold_x=next_thr,
        threshold_y=nova_thr,
        save_path=out_path,
    )
    plt.close('all')
    log.info('%s: saved %s', ct, out_path)

    return {
        'celltype': ct,
        'next_n_genes': len(next_pk),
        'nova_n_genes': len(nova_pk),
        'shared_genes': len(merged),
        'next_threshold': next_thr,
        'nova_threshold': nova_thr,
        'next_egenes_gt_thr': int((next_pk['peak_lod'] >= next_thr).sum()),
        'nova_egenes_gt_thr': int((nova_pk['peak_lod'] >= nova_thr).sum()),
        f'{ref_gene}_next_lod': ref_next_lod,
        f'{ref_gene}_nova_lod': ref_nova_lod,
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=pathlib.Path,
                         default=pathlib.Path('validation/ctxhipp_data'),
                         help='tissue data root containing next/ and nova/ (default: '
                              'validation/ctxhipp_data)')
    parser.add_argument('--celltypes', nargs='+', default=list(_DEFAULT_CTS),
                         help='celltypes to compare (default: the 5 with both '
                              'next and nova core/ outputs)')
    parser.add_argument('--ref-gene', default='Neurod1',
                         help='reference gene symbol to highlight (default: Neurod1)')
    parser.add_argument('--out', type=pathlib.Path,
                         default=pathlib.Path('validation/ctxhipp_data/compare_generations'),
                         help='output directory for PNGs + summary.csv')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ct in args.celltypes:
        next_dir = args.data / 'next' / 'celltypes' / ct
        nova_dir = args.data / 'nova' / 'celltypes' / ct
        if not (next_dir / 'core' / 'peak_lod.csv').exists():
            log.warning('%s: missing next core output, skipping', ct)
            continue
        if not (nova_dir / 'core' / 'peak_lod.csv').exists():
            log.warning('%s: missing nova core output, skipping', ct)
            continue
        rows.append(compare_celltype(ct, next_dir, nova_dir, args.ref_gene, args.out))

    summary = pd.DataFrame(rows)
    summary_path = args.out / 'summary.csv'
    summary.to_csv(summary_path, index=False)
    log.info('wrote %s', summary_path)
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()

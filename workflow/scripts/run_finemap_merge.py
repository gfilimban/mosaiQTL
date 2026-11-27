"""Merge per-chromosome fine-mapping tables — called by Nextflow FINEMAP_MERGE process.

Usage: python run_finemap_merge.py <parts_dir> <tag> <output_dir> <haplotype_eqtl_summary.tsv>

Writes: {tag}_eqtl_finemapping.tsv.gz (gzip-compressed), {tag}_eqtl_finemapping_peak.tsv,
  {tag}_eqtl_finemapping_groups.tsv, {tag}_eqtl_ci.tsv
  Each is the header once followed by every per-chromosome part in natural
  chromosome order. A part that is empty, lacks a final newline or has a
  different header stops the merge; with no parts a header-only file is written.
  {tag}_haplotype_eqtl_summary.tsv is the input summary with each eGene's support
  interval (ci_lo, ci_hi, ci_width_mb, ci_truncated) added from {tag}_eqtl_ci.tsv.
"""
import sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    parts_arg, tag, out_arg, summary_arg = sys.argv[1:]
    parts_dir, out, summary_in = Path(parts_arg), Path(out_arg), Path(summary_arg)
    out.mkdir(parents=True, exist_ok=True)

    from natsort import natsorted

    from mosaiqtl.io.atomic import atomic_concat_table, atomic_write_table
    from mosaiqtl.io.tables import read_table
    from mosaiqtl.pipeline import (
        CI_COLS, EQTL_FINEMAPPING_COLS, EQTL_FINEMAPPING_GROUP_COLS, EQTL_FINEMAPPING_PEAK_COLS,
        attach_support_interval,
    )

    for suffix, ext, columns in (
        ('eqtl_finemapping', 'tsv.gz', EQTL_FINEMAPPING_COLS),
        ('eqtl_finemapping_peak', 'tsv', EQTL_FINEMAPPING_PEAK_COLS),
        ('eqtl_finemapping_groups', 'tsv', EQTL_FINEMAPPING_GROUP_COLS),
        ('eqtl_ci', 'tsv', CI_COLS),
    ):
        parts = natsorted(parts_dir.glob(f'*_{suffix}.tsv'), key=lambda p: p.name)
        dest = out / f'{tag}_{suffix}.{ext}'
        n_lines = atomic_concat_table([p for p in parts if p.resolve() != dest.resolve()], dest, columns)
        log.info('merged %d parts -> %s (%d rows)', len(parts), dest.name, n_lines)

    summary = read_table(summary_in, dtype={'gene_id': str})
    ci = read_table(out / f'{tag}_eqtl_ci.tsv', dtype={'gene_id': str})
    final = attach_support_interval(summary, ci)
    summary_out = out / f'{tag}_haplotype_eqtl_summary.tsv'
    atomic_write_table(final, summary_out)
    log.info('wrote %s (%d eGenes, %d with a support interval)', summary_out.name, len(final),
             int(final['ci_lo'].notna().sum()))


if __name__ == '__main__':
    main()

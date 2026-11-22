"""Merge per-chr LOD CSVs → genome-wide tables — called by Nextflow MERGE_SCAN process.

Usage: python run_merge.py <lod_csvs_dir> <gene_info.csv> <celltype>
           <scan_tag> <output_dir> <tissue> <gene_tss.csv|->
  tissue may be an empty string; gene_tss '-' means no transcript TSS table.

Writes: {tag}_lod_genome.csv, {tag}_peak_lod.csv, {tag}_scan_stats.csv
  where tag = {tissue}_{celltype}_{scan_tag}  (tissue omitted when empty)
  Note: {tag}_peak_lod.csv includes beta_{founder} columns from the LMM scan
  so run_finemap.py can enrich egenes_df without a separate allele_effects file.
"""
import sys, json, logging, time
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    stream=sys.stdout)
log = logging.getLogger(__name__)


def main() -> None:
    lod_arg, gene_info_arg, celltype, stag, out_arg, tissue, gene_tss_arg = sys.argv[1:]
    lod_dir, out = Path(lod_arg), Path(out_arg)

    tag = f'{tissue}_{celltype}_{stag}' if tissue else f'{celltype}_{stag}'
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    from mosaiqtl.io.atomic import atomic_write_csv
    from mosaiqtl.pipeline import PEAK_LOD_COLS, enrich_peak_lod, scanned_chromosomes
    from mosaiqtl.utils import add_chr_prefix

    chrs = scanned_chromosomes(lod_dir)
    log.info('START merge: tag=%s  %d chrs: %s', tag, len(chrs), ' '.join(chrs))

    def read_parts(stem: str) -> pd.DataFrame:
        return pd.concat([pd.read_csv(lod_dir / f'{stem}{c}.csv') for c in chrs], ignore_index=True)

    lod_genome = read_parts('lod_chr')
    lod_genome['chr'] = add_chr_prefix(lod_genome['chr'])
    atomic_write_csv(lod_genome, out / f'{tag}_lod_genome.csv')
    log.info('%s_lod_genome.csv: %d loci', tag, len(lod_genome))

    peak_lod = read_parts('peak_stats_chr')
    peak_lod['chr'] = peak_lod['chr'].astype(str)

    gene_tss = (
        None if gene_tss_arg == '-'
        else pd.read_csv(gene_tss_arg, usecols=['gene_id', 'start', 'end', 'strand'])
    )
    peak_lod = enrich_peak_lod(peak_lod, pd.read_csv(gene_info_arg), gene_tss)

    # beta_{founder} cols come through from run_lmm_scan_chr.py peak_stats write
    _base = [c for c in PEAK_LOD_COLS if c in peak_lod.columns]
    _beta = [c for c in peak_lod.columns if c.startswith('beta_')]
    peak_out = peak_lod[_base + _beta].copy()
    peak_out['chr'] = add_chr_prefix(peak_out['chr'])
    atomic_write_csv(peak_out, out / f'{tag}_peak_lod.csv')
    log.info('%s_peak_lod.csv: %d genes  %d beta cols', tag, len(peak_lod), len(_beta))

    timing = [json.loads((lod_dir / f'timing_chr{c}.json').read_text()) for c in chrs]
    atomic_write_csv(pd.DataFrame(timing), out / f'{tag}_scan_stats.csv')

    log.info('DONE merge: tag=%s  %.1fs', tag, time.perf_counter() - t0)


if __name__ == '__main__':
    main()

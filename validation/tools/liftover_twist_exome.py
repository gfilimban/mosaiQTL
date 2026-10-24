"""liftover_twist_exome.py — lift the Twist Mouse Exome target BED from mm10 to mm39.

The panel's physical probes do not change with the assembly; only the target
coordinates need lifting, so they can be intersected with mm39 annotations and
alignments.

Each interval is lifted by its first and last base through the UCSC mm10ToMm39
chain. An interval is kept only when both ends map, to one location each, on the
same chromosome and strand, and its length stays within a small tolerance of the
original. Anything else (an end in a deleted or unaligned stretch, an interval
split across chromosomes, a large resize) is written to the unmapped table with
the reason, never silently dropped or stretched.

Inputs:
    reference/protocols/exomecapture/Twist_Mouse_Exome_Target_Rev1_7APR20.bed
    analysis/input/mm10ToMm39.over.chain.gz

Outputs (analysis/input/):
    Twist_Mouse_Exome_Target_Rev1_7APR20_mm39.bed            BED4, name = source mm10 locus
    Twist_Mouse_Exome_Target_Rev1_7APR20_mm10_unmapped.tsv   source interval + reason

Usage:
    conda run -n bioinfo python validation/tools/liftover_twist_exome.py
"""
from __future__ import annotations

import gzip
import logging
import pathlib
import sys

import pandas as pd
from pyliftover import LiftOver

_REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / 'src'))

from mosaiqtl.utils import chr_sort_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

_ANALYSIS = _REPO.parents[1]
_INPUT = _ANALYSIS / 'input'
_SRC_BED = (_REPO.parents[5] / 'reference' / 'protocols' / 'exomecapture'
            / 'Twist_Mouse_Exome_Target_Rev1_7APR20.bed')
_CHAIN = _INPUT / 'mm10ToMm39.over.chain.gz'
_OUT_BED = _INPUT / 'Twist_Mouse_Exome_Target_Rev1_7APR20_mm39.bed'
_OUT_UNMAPPED = _INPUT / 'Twist_Mouse_Exome_Target_Rev1_7APR20_mm10_unmapped.tsv'

# a lifted interval may differ from its source length by at most
# max(_LEN_TOL_BP, _LEN_TOL_FRAC * source length); larger changes mean the chain
# has an indel inside the target, so the interval is reported instead of kept
_LEN_TOL_BP = 5
_LEN_TOL_FRAC = 0.10


def _mm39_sizes(chain_path: pathlib.Path) -> dict[str, int]:
    """Read mm39 contig sizes from the chain headers (the chain's query side)."""
    sizes: dict[str, int] = {}
    with gzip.open(chain_path, 'rt') as f:
        for line in f:
            if line.startswith('chain'):
                fields = line.split()
                sizes[fields[7]] = int(fields[8])
    return sizes


def _lift_base(lo: LiftOver, chrom: str, pos: int) -> tuple[str, int, str] | str:
    """Lift one 0-based base; returns (chrom, pos, strand) or a failure reason."""
    hits = lo.convert_coordinate(chrom, pos)
    if not hits:
        return 'unmapped'
    if len(hits) > 1:
        return 'multi_hit'
    new_chrom, new_pos, strand, _ = hits[0]
    return new_chrom, int(new_pos), strand


def _lift_interval(
    lo: LiftOver, chrom: str, start: int, end: int, sizes: dict[str, int],
) -> tuple[str, int, int] | str:
    """Lift a half-open interval by its first and last base.

    Returns (chrom, start, end) in mm39, or the reason the interval was rejected.
    """
    first = _lift_base(lo, chrom, start)
    last = _lift_base(lo, chrom, end - 1)
    for side, hit in (('start', first), ('end', last)):
        if isinstance(hit, str):
            return f'{side}_{hit}'
    assert not isinstance(first, str) and not isinstance(last, str)
    if first[0] != last[0]:
        return 'chrom_mismatch'
    if first[2] != last[2]:
        return 'strand_mismatch'
    new_start = min(first[1], last[1])
    new_end = max(first[1], last[1]) + 1
    old_len, new_len = end - start, new_end - new_start
    if abs(new_len - old_len) > max(_LEN_TOL_BP, _LEN_TOL_FRAC * old_len):
        return 'resized'
    if new_end > sizes.get(first[0], 0):
        return 'out_of_bounds'
    return first[0], new_start, new_end


def _sort_key(chrom: str) -> tuple[int, str]:
    return chr_sort_key(chrom.removeprefix('chr')), chrom


def main() -> None:
    src = pd.read_csv(_SRC_BED, sep='\t', header=None, names=['chrom', 'start', 'end'])
    log.info('read %d intervals  %.3f Mb  from %s', len(src),
             (src['end'] - src['start']).sum() / 1e6, _SRC_BED.name)

    lo = LiftOver(str(_CHAIN))
    sizes = _mm39_sizes(_CHAIN)

    kept: list[tuple[str, int, int, str]] = []
    rejected: list[tuple[str, int, int, str]] = []
    for chrom, start, end in src.itertuples(index=False, name=None):
        result = _lift_interval(lo, chrom, start, end, sizes)
        if isinstance(result, str):
            rejected.append((chrom, start, end, result))
        else:
            kept.append((result[0], result[1], result[2], f'{chrom}:{start}-{end}'))

    out = pd.DataFrame(kept, columns=['chrom', 'start', 'end', 'name'])
    out['_k'] = out['chrom'].map(_sort_key)
    out = out.sort_values(['_k', 'start', 'end']).drop(columns='_k')
    out.to_csv(_OUT_BED, sep='\t', header=False, index=False)

    unmapped = pd.DataFrame(rejected, columns=['chrom', 'start', 'end', 'reason'])
    unmapped.to_csv(_OUT_UNMAPPED, sep='\t', index=False)

    n_src_bp = int((src['end'] - src['start']).sum())
    n_out_bp = int((out['end'] - out['start']).sum())
    log.info('mapped %d / %d intervals (%.2f%%)', len(out), len(src), 100 * len(out) / len(src))
    log.info('total bp  mm10 %d  mm39 %d  (%+.3f%%)', n_src_bp, n_out_bp,
             100 * (n_out_bp - n_src_bp) / n_src_bp)
    if len(unmapped):
        main_src = ~unmapped['chrom'].str.contains('_')
        for reason, n in unmapped['reason'].value_counts().items():
            log.info('  unmapped %-16s %6d  (on main chromosomes: %d)', reason, n,
                     int(((unmapped['reason'] == reason) & main_src).sum()))

    dest_main = out['chrom'].str.removeprefix('chr').isin(
        [str(i) for i in range(1, 20)] + ['X', 'Y', 'M'])
    log.info('lifted intervals on non-primary mm39 contigs: %d', int((~dest_main).sum()))
    new_len = out['end'] - out['start']
    src_span = out['name'].str.extract(r':(\d+)-(\d+)$').astype(int)
    old_len = src_span[1] - src_span[0]
    log.info('length change: %d identical  %d changed', int((new_len == old_len).sum()),
             int((new_len != old_len).sum()))
    prev_end = out.groupby('chrom')['end'].cummax().groupby(out['chrom']).shift()
    log.info('intervals overlapping an earlier interval after lift: %d',
             int((out['start'] < prev_end).sum()))
    log.info('wrote %s', _OUT_BED)
    log.info('wrote %s', _OUT_UNMAPPED)


if __name__ == '__main__':
    main()

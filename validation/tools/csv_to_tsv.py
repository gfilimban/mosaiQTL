"""csv_to_tsv.py — convert CSV output tables to TSV, leaving every source file untouched.

Each ``X.csv`` becomes ``X.tsv`` and each ``X.csv.gz`` becomes ``X.tsv.gz`` in the same
directory. Tables are streamed row by row, so memory use does not depend on file size.
Every cell stays the exact text it was (nothing is parsed as a number or a boolean, so
full-precision floats and ``T``/``F`` flags are unchanged). Quoted CSV fields are
decoded and re-encoded: a cell that contains a tab, a newline or a double quote is
written quoted, so ``pandas.read_csv(path, sep='\\t')`` reads the table back unchanged.

Each converted file is checked before it is put in place: the TSV is read back and
compared with the CSV row by row, and a file whose rows do not all have the header's
width is refused. The destination is written to a temporary sibling, fsynced, then
renamed, so a killed run never leaves a truncated TSV. An existing TSV is skipped
unless ``--overwrite`` is given. The source CSV is never modified or removed.

Usage:
    conda run -n bioinfo python validation/tools/csv_to_tsv.py [-r] [--dry-run] [--overwrite] PATH [PATH ...]

    PATH   a .csv / .csv.gz file, or a directory holding them
    -r     descend into sub-directories
    --dry-run   list what would be converted and write nothing
    --overwrite replace a TSV that already exists

Exit status is 1 if any file fails, 0 otherwise.
"""

from __future__ import annotations

import csv
import gzip
import io
import logging
import os
import sys
import tempfile
from collections.abc import Callable, Iterator, Sequence
from itertools import zip_longest
from pathlib import Path
from typing import IO, NamedTuple

log = logging.getLogger('csv_to_tsv')

_CSV_SUFFIXES = ('.csv', '.csv.gz')
_FIELD_LIMIT = 2**31 - 1


class Converted(NamedTuple):
    """Outcome of converting one file: data rows written and cells that needed quoting."""

    rows: int
    quoted_cells: int


def is_csv(path: Path) -> bool:
    """True when ``path`` ends in ``.csv`` or ``.csv.gz``."""
    return path.name.endswith(_CSV_SUFFIXES)


def tsv_path(path: Path) -> Path:
    """Return the TSV destination of ``path``: ``X.csv`` → ``X.tsv``, ``X.csv.gz`` → ``X.tsv.gz``."""
    name = path.name
    if name.endswith('.csv.gz'):
        return path.with_name(name[: -len('.csv.gz')] + '.tsv.gz')
    if name.endswith('.csv'):
        return path.with_name(name[: -len('.csv')] + '.tsv')
    raise ValueError(f'not a CSV file: {path}')


def find_csvs(roots: Sequence[Path], recursive: bool) -> list[Path]:
    """List the CSV files named by ``roots`` (files taken as given, directories searched), sorted.

    Temporary files (names starting with a dot) are ignored.

    Raises
    ------
    ValueError
        If a root is a file that is not a ``.csv`` / ``.csv.gz``, or does not exist.
    """
    found: set[Path] = set()
    for root in roots:
        if root.is_dir():
            walk = root.rglob if recursive else root.glob
            found.update(p for suffix in _CSV_SUFFIXES for p in walk(f'*{suffix}') if p.is_file())
        elif root.is_file():
            if not is_csv(root):
                raise ValueError(f'not a .csv or .csv.gz file: {root}')
            found.add(root)
        else:
            raise ValueError(f'no such file or directory: {root}')
    return sorted(p for p in found if not p.name.startswith('.'))


def _open_text(path: Path, gz: bool) -> IO[str]:
    """Open ``path`` for reading as UTF-8 text with newlines left for the csv module to handle."""
    if gz:
        return gzip.open(path, 'rt', encoding='utf-8', newline='')
    return open(path, 'r', encoding='utf-8', newline='')


def _rows(handle: IO[str], delimiter: str) -> Iterator[list[str]]:
    """Yield the non-blank rows of a delimited table."""
    for row in csv.reader(handle, delimiter=delimiter):
        if row:
            yield row


def _copy_rows(src: Path, gz: bool, sink: IO[str], counts: list[int]) -> None:
    """Stream the rows of ``src`` to ``sink`` as TSV, recording ``[rows, quoted_cells]`` in ``counts``."""
    writer = csv.writer(sink, delimiter='\t', lineterminator='\n')
    width: 'int | None' = None
    with _open_text(src, gz) as handle:
        for row in _rows(handle, ','):
            if width is None:
                width = len(row)
            elif len(row) != width:
                raise ValueError(f'{src}: row {counts[0] + 1} has {len(row)} fields, the header has {width}')
            counts[1] += sum(1 for cell in row if any(ch in cell for ch in '\t\r\n"'))
            writer.writerow(row)
            counts[0] += 1


def _write_temp(fd: int, gz: bool, write: Callable[[IO[str]], None]) -> None:
    """Run ``write`` on the temporary file ``fd`` (gzip-compressed when ``gz``) and fsync it."""
    if gz:
        with os.fdopen(fd, 'wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as stream:
                with io.TextIOWrapper(stream, encoding='utf-8', newline='') as text:
                    write(text)
            raw.flush()
            os.fsync(raw.fileno())
    else:
        with os.fdopen(fd, 'w', encoding='utf-8', newline='') as text:
            write(text)
            text.flush()
            os.fsync(text.fileno())


def _verify(src: Path, tmp: Path, gz: bool) -> None:
    """Raise ValueError unless ``tmp`` (TSV) holds exactly the rows of ``src`` (CSV)."""
    with _open_text(src, gz) as csv_in, _open_text(tmp, gz) as tsv_in:
        pairs = zip_longest(_rows(csv_in, ','), _rows(tsv_in, '\t'))
        for n, (expected, actual) in enumerate(pairs, start=1):
            if expected != actual:
                raise ValueError(f'{src}: row {n} does not survive the conversion')


def convert_file(src: Path, dest: Path) -> Converted:
    """Convert one CSV (or ``.csv.gz``) to TSV at ``dest``, verified and atomic.

    Parameters
    ----------
    src:
        Source ``.csv`` / ``.csv.gz``; never modified.
    dest:
        Destination ``.tsv`` / ``.tsv.gz`` (compressed when it ends in ``.gz``); replaced if present.

    Returns
    -------
    Converted
        Data rows written (header included) and the number of cells written quoted.

    Raises
    ------
    ValueError
        If a row's width differs from the first row's, or the written TSV does not read
        back identical to the source. ``dest`` is left as it was.
    """
    csv.field_size_limit(_FIELD_LIMIT)
    gz = src.name.endswith('.gz')
    counts = [0, 0]
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f'.{dest.name}.', suffix='.tmp')
    try:
        _write_temp(fd, gz, lambda sink: _copy_rows(src, gz, sink, counts))
        _verify(src, Path(tmp_name), gz)
        os.replace(tmp_name, dest)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise
    return Converted(rows=counts[0], quoted_cells=counts[1])


def convert_all(roots: Sequence[Path], recursive: bool, dry_run: bool, overwrite: bool) -> int:
    """Convert every CSV under ``roots``; return the number of files that failed."""
    files = find_csvs(roots, recursive)
    if not files:
        log.warning('no .csv or .csv.gz files found')
        return 0
    failures = 0
    done = skipped = 0
    for src in files:
        dest = tsv_path(src)
        if dest.exists() and not overwrite:
            log.info('skip (exists)  %s', dest)
            skipped += 1
            continue
        if dry_run:
            log.info('would convert  %s -> %s', src, dest.name)
            continue
        try:
            result = convert_file(src, dest)
        except (ValueError, OSError, csv.Error, UnicodeDecodeError) as err:
            log.error('FAILED  %s: %s', src, err)
            failures += 1
            continue
        done += 1
        log.info('converted  %s -> %s  (%d rows, %d quoted cells)', src, dest.name, result.rows, result.quoted_cells)
    log.info('%d converted, %d skipped, %d failed', done, skipped, failures)
    return failures


def parse_args(argv: Sequence[str]) -> 'tuple[list[Path], bool, bool, bool]':
    """Split ``argv`` into ``(paths, recursive, dry_run, overwrite)``.

    Raises
    ------
    ValueError
        On an unknown option or when no path is given.
    """
    flags = {'-r': 'recursive', '--recursive': 'recursive', '--dry-run': 'dry_run', '--overwrite': 'overwrite'}
    chosen = {'recursive': False, 'dry_run': False, 'overwrite': False}
    paths: list[Path] = []
    for arg in argv:
        if arg in flags:
            chosen[flags[arg]] = True
        elif arg.startswith('-'):
            raise ValueError(f'unknown option: {arg}')
        else:
            paths.append(Path(arg))
    if not paths:
        raise ValueError('give at least one file or directory')
    return paths, chosen['recursive'], chosen['dry_run'], chosen['overwrite']


def main(argv: Sequence[str]) -> int:
    """Run the converter on ``argv`` (without the program name); return the exit status."""
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
    try:
        paths, recursive, dry_run, overwrite = parse_args(argv)
        failures = convert_all(paths, recursive, dry_run, overwrite)
    except ValueError as err:
        log.error('%s', err)
        log.error('usage: csv_to_tsv.py [-r] [--dry-run] [--overwrite] PATH [PATH ...]')
        return 2
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

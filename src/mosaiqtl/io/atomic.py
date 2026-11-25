"""Crash-safe file writers: a reader sees either the previous file or the complete new one.

Every table or JSON file on the Nextflow path is written to a temporary file in the
destination directory, flushed and fsynced, then renamed over the destination with
``os.replace``. The rename is atomic on a single POSIX filesystem, so a job that is
killed (out of memory, wall-time, node loss) never leaves a truncated file behind
for a downstream stage or a ``-resume`` to mistake for a finished output.
"""
from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO, Any

import numpy as np
import pandas as pd

__all__ = [
    'atomic_concat_csv', 'atomic_write_csv', 'atomic_write_json', 'atomic_write_text',
    'CSV_TRUE', 'CSV_FALSE',
]

_COPY_BLOCK = 1 << 20

CSV_TRUE = 'T'
CSV_FALSE = 'F'


def _is_boolean_column(col: pd.Series) -> bool:
    """True for a bool or nullable-boolean column, or an object column whose non-missing cells are all bool."""
    if pd.api.types.is_bool_dtype(col):
        return True
    if col.dtype != object:
        return False
    cells = col.dropna()
    return len(cells) > 0 and bool(cells.map(lambda v: isinstance(v, (bool, np.bool_))).all())


def _encode_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` with every boolean column as ``'T'``/``'F'`` strings (missing stays empty)."""
    cols = [c for c in df.columns if _is_boolean_column(df[c])]
    if not cols:
        return df
    out = df.copy(deep=False)
    for c in cols:
        out[c] = df[c].map({True: CSV_TRUE, False: CSV_FALSE})
    return out


def _atomic_write(path: 'Path | str', write: Callable[[IO[str]], None]) -> None:
    """Run ``write`` on a temporary sibling file, fsync it, then rename it to ``path``."""
    dest = Path(path)
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f'.{dest.name}.', suffix='.tmp')
    try:
        with os.fdopen(fd, 'w') as fh:
            write(fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, dest)
    except BaseException:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)
        raise


def atomic_write_csv(df: pd.DataFrame, path: 'Path | str') -> None:
    r"""Write ``df`` to ``path`` as CSV (header, no index) atomically.

    Floats are written at full precision (the shortest string that round-trips), so
    coordinates, probabilities and small p-values lose no digits. Boolean columns are
    written as ``'T'``/``'F'``; read them back with ``true_values=['T']`` and
    ``false_values=['F']``.

    Parameters
    ----------
    df:
        Table to write.
    path:
        Destination; the parent directory must exist.
    """
    encoded = _encode_booleans(df)
    _atomic_write(path, lambda fh: encoded.to_csv(fh, index=False))


def atomic_concat_csv(
    parts: Sequence['Path | str'],
    path: 'Path | str',
    columns: Sequence[str],
) -> int:
    r"""Concatenate CSV ``parts`` into ``path`` byte-for-byte, atomically.

    Each part must start with the same header and end with a newline; the header
    is written once, so no row is ever fused to its neighbour. Parts are copied in
    blocks, so memory use is independent of their size. With no parts the result is
    a header-only file built from ``columns``.

    Parameters
    ----------
    parts:
        Part files in output order.
    path:
        Destination; the parent directory must exist.
    columns:
        Header written when ``parts`` is empty.

    Returns
    -------
    int
        Number of data lines written.

    Raises
    ------
    ValueError
        If a part is empty, lacks a final newline, or its header differs from the
        first part's.
    """
    header = ','.join(columns) + '\n'
    n_lines = 0

    def write(fh: IO[str]) -> None:
        nonlocal n_lines
        expected: 'str | None' = None
        for part in parts:
            with open(part, 'r', newline='') as src:
                first = src.readline()
                if not first:
                    raise ValueError(f'{part}: empty file (no header)')
                if expected is None:
                    expected = first
                    fh.write(first if first.endswith('\n') else first + '\n')
                elif first != expected:
                    raise ValueError(f'{part}: header differs from {parts[0]}')
                last = first
                while block := src.read(_COPY_BLOCK):
                    fh.write(block)
                    n_lines += block.count('\n')
                    last = block
                if not last.endswith('\n'):
                    raise ValueError(f'{part}: does not end with a newline (truncated write)')
        if expected is None:
            fh.write(header)

    _atomic_write(path, write)
    return n_lines


def atomic_write_text(text: str, path: 'Path | str') -> None:
    """Write ``text`` to ``path`` atomically."""
    _atomic_write(path, lambda fh: fh.write(text))


def atomic_write_json(obj: Any, path: 'Path | str', indent: 'int | None' = 2) -> None:
    """Write ``obj`` to ``path`` as JSON atomically, with a trailing newline."""
    _atomic_write(path, lambda fh: (json.dump(obj, fh, indent=indent), fh.write('\n')))

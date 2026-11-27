"""Readers for the tab-separated tables the pipeline writes."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from mosaiqtl.io.atomic import TABLE_FALSE, TABLE_TRUE

__all__ = ['read_table']


def read_table(path: 'Path | str', **kwargs: Any) -> pd.DataFrame:
    r"""Read a tab-separated table (plain or ``.gz``) written by the pipeline.

    Cells ``'T'`` and ``'F'`` in a column of only those values load as booleans.
    Keyword arguments pass through to :func:`pandas.read_csv` (``usecols``,
    ``dtype``, ``nrows``, ``chunksize``, ...); ``sep`` is fixed to a tab.

    Parameters
    ----------
    path:
        Table to read.
    **kwargs:
        Further :func:`pandas.read_csv` options.

    Returns
    -------
    pandas.DataFrame
        The table, or an iterator of chunks when ``chunksize`` is given.
    """
    kwargs.setdefault('true_values', [TABLE_TRUE])
    kwargs.setdefault('false_values', [TABLE_FALSE])
    return pd.read_csv(path, sep='\t', **kwargs)

"""docs/output_schema.md documents every column the pipeline writes."""
import re
from pathlib import Path

import pytest

from mosaiqtl import pipeline
from mosaiqtl.qtl.ci import CI_COLS

_DOC = Path(__file__).resolve().parents[2] / 'docs' / 'output_schema.md'
_CALL_COLS = ['is_egene', 'pval_perm', 'beta_a', 'beta_b', 'perm_n', 'qval_storey', 'qval_bh', 'lfdr', 'pi0', 'fdr',
              'fdr_method', 'significance']
_FOUNDER_PREFIXES = ('beta_', 'effect_')


def _documented() -> set[str]:
    """Backticked names in the first cell of every column-reference row."""
    section = _DOC.read_text().split('## Column reference', 1)[1]
    cells = re.findall(r'^\| ([^|]+) \|', section, flags=re.MULTILINE)
    return {n for c in cells for n in re.findall(r'`([^`]+)`', c)}


def _written() -> dict[str, list[str]]:
    return {
        'LOD_COLS': pipeline.LOD_COLS,
        'PEAK_LOD_COLS': pipeline.PEAK_LOD_COLS,
        'call block': _CALL_COLS,
        'EGENE_SUMMARY_COLS': pipeline.EGENE_SUMMARY_COLS,
        'EQTL_FINEMAPPING_COLS': pipeline.EQTL_FINEMAPPING_COLS,
        'EQTL_FINEMAPPING_PEAK_COLS': pipeline.EQTL_FINEMAPPING_PEAK_COLS,
        'EQTL_FINEMAPPING_GROUP_COLS': pipeline.EQTL_FINEMAPPING_GROUP_COLS,
        'CI_COLS': CI_COLS,
    }


@pytest.mark.parametrize('name', list(_written()))
def test_written_columns_are_documented(name):
    documented = _documented()
    missing = [
        c for c in _written()[name]
        if c not in documented and not c.startswith(_FOUNDER_PREFIXES)
    ]
    assert not missing, f'{name}: undocumented columns {missing}'


def test_founder_column_patterns_documented():
    documented = _documented()
    assert {'beta_{founder}', 'effect_{founder}'} <= documented


def test_documented_columns_are_written():
    written = {c for cols in _written().values() for c in cols}
    stale = {
        c for c in _documented()
        if c not in written and '{' not in c
    }
    assert not stale, f'documented but never written: {sorted(stale)}'

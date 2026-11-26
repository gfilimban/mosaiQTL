"""Tests for validation/tools/csv_to_tsv.py."""
from __future__ import annotations

import gzip
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

_TOOL = Path(__file__).resolve().parents[2] / 'validation' / 'tools' / 'csv_to_tsv.py'
_spec = importlib.util.spec_from_file_location('csv_to_tsv', _TOOL)
assert _spec is not None and _spec.loader is not None
tool = importlib.util.module_from_spec(_spec)
sys.modules['csv_to_tsv'] = tool
_spec.loader.exec_module(tool)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding='utf-8', newline='')
    return path


class TestNaming:
    def test_plain_and_gzip_names(self) -> None:
        assert tool.tsv_path(Path('a/x.csv')) == Path('a/x.tsv')
        assert tool.tsv_path(Path('a/x.csv.gz')) == Path('a/x.tsv.gz')

    def test_non_csv_name_is_refused(self) -> None:
        with pytest.raises(ValueError):
            tool.tsv_path(Path('x.txt'))


class TestConvertFile:
    def test_cells_are_unchanged_text(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'gene_id,peak_lod,significance\nG1,0.1234567890123456,T\nG2,1e-300,F\n')
        dest = tmp_path / 'a.tsv'
        result = tool.convert_file(src, dest)
        assert dest.read_text() == 'gene_id\tpeak_lod\tsignificance\nG1\t0.1234567890123456\tT\nG2\t1e-300\tF\n'
        assert result.rows == 3 and result.quoted_cells == 0

    def test_comma_inside_quoted_field_is_not_split(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'id,desc\nG1,"kinase, serine/threonine"\n')
        dest = tmp_path / 'a.tsv'
        tool.convert_file(src, dest)
        assert dest.read_text() == 'id\tdesc\nG1\tkinase, serine/threonine\n'

    def test_tab_newline_and_quote_in_a_cell_survive_a_pandas_round_trip(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'id,note\nG1,"a\tb"\nG2,"line1\nline2"\nG3,"say ""hi"""\n')
        dest = tmp_path / 'a.tsv'
        result = tool.convert_file(src, dest)
        back = pd.read_csv(dest, sep='\t', dtype=str)
        assert back['note'].tolist() == ['a\tb', 'line1\nline2', 'say "hi"']
        assert result.quoted_cells == 3

    def test_missing_cells_stay_empty(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'a,b,c\n1,,3\n,,\n')
        dest = tmp_path / 'a.tsv'
        tool.convert_file(src, dest)
        assert dest.read_text() == 'a\tb\tc\n1\t\t3\n\t\t\n'

    def test_header_only_file(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'a,b,c\n')
        dest = tmp_path / 'a.tsv'
        assert tool.convert_file(src, dest).rows == 1
        assert dest.read_text() == 'a\tb\tc\n'

    def test_empty_file(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', '')
        dest = tmp_path / 'a.tsv'
        assert tool.convert_file(src, dest).rows == 0
        assert dest.read_text() == ''

    def test_gzip_in_gzip_out(self, tmp_path: Path) -> None:
        src = tmp_path / 'a.csv.gz'
        with gzip.open(src, 'wt', encoding='utf-8', newline='') as fh:
            fh.write('a,b\n1,"x,y"\n')
        dest = tmp_path / 'a.tsv.gz'
        tool.convert_file(src, dest)
        with gzip.open(dest, 'rt', encoding='utf-8', newline='') as fh:
            assert fh.read() == 'a\tb\n1\tx,y\n'

    def test_gzip_output_is_deterministic(self, tmp_path: Path) -> None:
        src = tmp_path / 'a.csv.gz'
        with gzip.open(src, 'wt', encoding='utf-8', newline='') as fh:
            fh.write('a,b\n1,2\n')
        first, second = tmp_path / 'one.tsv.gz', tmp_path / 'two.tsv.gz'
        tool.convert_file(src, first)
        tool.convert_file(src, second)
        assert first.read_bytes() == second.read_bytes()

    def test_ragged_row_is_refused_and_leaves_no_file(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'a,b\n1,2\n3\n')
        dest = tmp_path / 'a.tsv'
        with pytest.raises(ValueError, match='fields'):
            tool.convert_file(src, dest)
        assert not dest.exists()
        assert [p.name for p in tmp_path.iterdir()] == ['a.csv']

    def test_source_is_untouched(self, tmp_path: Path) -> None:
        text = 'a,b\n1,"x,y"\n'
        src = _write(tmp_path / 'a.csv', text)
        tool.convert_file(src, tmp_path / 'a.tsv')
        assert src.read_text() == text

    def test_failed_overwrite_keeps_the_previous_tsv(self, tmp_path: Path) -> None:
        src = _write(tmp_path / 'a.csv', 'a,b\n1,2\n3\n')
        dest = _write(tmp_path / 'a.tsv', 'old\n')
        with pytest.raises(ValueError):
            tool.convert_file(src, dest)
        assert dest.read_text() == 'old\n'


class TestFindCsvs:
    def test_directory_search_is_flat_unless_recursive(self, tmp_path: Path) -> None:
        _write(tmp_path / 'top.csv', 'a\n1\n')
        (tmp_path / 'sub').mkdir()
        _write(tmp_path / 'sub' / 'deep.csv', 'a\n1\n')
        _write(tmp_path / 'notes.txt', 'x')
        assert [p.name for p in tool.find_csvs([tmp_path], recursive=False)] == ['top.csv']
        assert [p.name for p in tool.find_csvs([tmp_path], recursive=True)] == ['deep.csv', 'top.csv']

    def test_temporary_files_are_ignored(self, tmp_path: Path) -> None:
        _write(tmp_path / '.a.csv.abc.tmp', 'x')
        _write(tmp_path / '.hidden.csv', 'a\n1\n')
        assert tool.find_csvs([tmp_path], recursive=False) == []

    def test_non_csv_file_and_missing_path_are_errors(self, tmp_path: Path) -> None:
        other = _write(tmp_path / 'a.txt', 'x')
        with pytest.raises(ValueError):
            tool.find_csvs([other], recursive=False)
        with pytest.raises(ValueError):
            tool.find_csvs([tmp_path / 'nope'], recursive=False)


class TestConvertAll:
    def test_existing_tsv_is_skipped_unless_overwrite(self, tmp_path: Path) -> None:
        _write(tmp_path / 'a.csv', 'a,b\n1,2\n')
        _write(tmp_path / 'a.tsv', 'keep\n')
        assert tool.convert_all([tmp_path], False, False, False) == 0
        assert (tmp_path / 'a.tsv').read_text() == 'keep\n'
        assert tool.convert_all([tmp_path], False, False, True) == 0
        assert (tmp_path / 'a.tsv').read_text() == 'a\tb\n1\t2\n'

    def test_dry_run_writes_nothing(self, tmp_path: Path) -> None:
        _write(tmp_path / 'a.csv', 'a,b\n1,2\n')
        assert tool.convert_all([tmp_path], False, True, False) == 0
        assert not (tmp_path / 'a.tsv').exists()

    def test_one_bad_file_does_not_stop_the_rest(self, tmp_path: Path) -> None:
        _write(tmp_path / 'bad.csv', 'a,b\n1\n')
        _write(tmp_path / 'good.csv', 'a,b\n1,2\n')
        assert tool.convert_all([tmp_path], False, False, False) == 1
        assert (tmp_path / 'good.tsv').exists() and not (tmp_path / 'bad.tsv').exists()


class TestMain:
    def test_exit_codes(self, tmp_path: Path) -> None:
        _write(tmp_path / 'a.csv', 'a,b\n1,2\n')
        assert tool.main([str(tmp_path)]) == 0
        assert tool.main([]) == 2
        assert tool.main(['--bogus', str(tmp_path)]) == 2
        _write(tmp_path / 'bad.csv', 'a,b\n1\n')
        assert tool.main([str(tmp_path), '--overwrite']) == 1

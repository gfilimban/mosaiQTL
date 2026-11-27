"""Tests for mosaiqtl.io.atomic: complete-or-absent output files."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import pandas as pd
import pytest

from mosaiqtl.io import atomic
from mosaiqtl.io.atomic import atomic_write_json, atomic_write_table, atomic_write_text
from mosaiqtl.io.tables import read_table


def _no_temp_files(directory: Path) -> bool:
    return not any(p.name.endswith('.tmp') for p in directory.iterdir())


class TestAtomicTable:
    def test_round_trip_and_trailing_newline(self, tmp_path: Path) -> None:
        df = pd.DataFrame({'gene_id': ['g1', 'g2'], 'lod': [3.25, 1.5]})
        dest = tmp_path / 'out.tsv'
        atomic_write_table(df, dest)
        assert dest.read_text().endswith('\n')
        assert dest.read_text().splitlines()[0] == 'gene_id\tlod'
        pd.testing.assert_frame_equal(read_table(dest), df)
        assert _no_temp_files(tmp_path)

    def test_floats_keep_every_digit(self, tmp_path: Path) -> None:
        dest = tmp_path / 'p.tsv'
        vals = [1.234567891234e-12, 123.456789012345, 0.1 + 0.2]
        atomic_write_table(pd.DataFrame({'p': vals}), dest)
        assert pd.read_csv(dest, sep='\t', float_precision='round_trip')['p'].tolist() == vals

    def test_booleans_written_as_t_f(self, tmp_path: Path) -> None:
        dest = tmp_path / 'b.tsv'
        df = pd.DataFrame({
            'a': [True, False],
            'b': pd.array([True, None], dtype='boolean'),
            'c': pd.Series([False, None], dtype=object),
        })
        atomic_write_table(df, dest)
        assert dest.read_text().splitlines() == ['a\tb\tc', 'T\tT\tF', 'F\t\t']
        back = read_table(dest)
        assert back['a'].tolist() == [True, False]
        assert df['a'].dtype == bool

    def test_embedded_commas_do_not_need_quoting(self, tmp_path: Path) -> None:
        dest = tmp_path / 'c.tsv'
        df = pd.DataFrame({'gene_id': ['g1'], 'note': ['a,b,c']})
        atomic_write_table(df, dest)
        assert dest.read_text().splitlines()[1] == 'g1\ta,b,c'
        pd.testing.assert_frame_equal(read_table(dest), df)

    def test_header_only_frame_is_parseable(self, tmp_path: Path) -> None:
        dest = tmp_path / 'empty.tsv'
        atomic_write_table(pd.DataFrame(columns=['a', 'b']), dest)
        assert list(read_table(dest).columns) == ['a', 'b']

    def test_failure_midway_leaves_no_partial_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dest = tmp_path / 'out.tsv'

        def explode(self: pd.DataFrame, fh, **kwargs) -> None:
            fh.write('gene_id\tlod\ng1\t')
            raise MemoryError('killed midway')

        monkeypatch.setattr(pd.DataFrame, 'to_csv', explode)
        with pytest.raises(MemoryError):
            atomic_write_table(pd.DataFrame({'gene_id': ['g1'], 'lod': [1.0]}), dest)
        assert not dest.exists()
        assert _no_temp_files(tmp_path)

    def test_failure_keeps_previous_complete_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dest = tmp_path / 'out.tsv'
        atomic_write_table(pd.DataFrame({'a': [1]}), dest)
        before = dest.read_text()

        def explode(self: pd.DataFrame, fh, **kwargs) -> None:
            fh.write('a\n9')
            raise KeyboardInterrupt

        monkeypatch.setattr(pd.DataFrame, 'to_csv', explode)
        with pytest.raises(KeyboardInterrupt):
            atomic_write_table(pd.DataFrame({'a': [2]}), dest)
        assert dest.read_text() == before
        assert _no_temp_files(tmp_path)

    def test_fsync_before_rename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []
        real_fsync, real_replace = atomic.os.fsync, atomic.os.replace
        monkeypatch.setattr(atomic.os, 'fsync', lambda fd: (order.append('fsync'), real_fsync(fd))[1])
        monkeypatch.setattr(atomic.os, 'replace', lambda a, b: (order.append('replace'), real_replace(a, b))[1])
        atomic_write_table(pd.DataFrame({'a': [1]}), tmp_path / 'o.tsv')
        assert order == ['fsync', 'replace']


class TestReadTable:
    def test_tab_separator_gives_real_columns(self, tmp_path: Path) -> None:
        path = tmp_path / 't.tsv'
        path.write_text('gene_id\tlod\ng1\t1.5\n')
        assert list(read_table(path).columns) == ['gene_id', 'lod']

    def test_t_f_parse_as_booleans(self, tmp_path: Path) -> None:
        path = tmp_path / 't.tsv'
        path.write_text('flag\nT\nF\nT\n')
        assert read_table(path)['flag'].tolist() == [True, False, True]

    def test_gzip_is_read_by_extension(self, tmp_path: Path) -> None:
        path = tmp_path / 't.tsv.gz'
        with gzip.open(path, 'wt', newline='') as fh:
            fh.write('gene_id\tlod\ng1\t1\n')
        assert read_table(path)['gene_id'].tolist() == ['g1']

    def test_usecols_passthrough(self, tmp_path: Path) -> None:
        path = tmp_path / 't.tsv'
        path.write_text('a\tb\tc\n1\t2\t3\n')
        assert list(read_table(path, usecols=['a', 'c']).columns) == ['a', 'c']


class TestAtomicConcatTable:
    def _part(self, directory: Path, name: str, text: str) -> Path:
        path = directory / name
        path.write_text(text)
        return path

    def test_header_once_rows_in_order(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\ng1\t1\ng2\t2\n')
        b = self._part(tmp_path, 'b.tsv', 'gene_id\tlod\ng3\t3\n')
        dest = tmp_path / 'merged.tsv'
        n_lines = atomic.atomic_concat_table([a, b], dest, ['gene_id', 'lod'])
        assert n_lines == 3
        assert dest.read_text() == 'gene_id\tlod\ng1\t1\ng2\t2\ng3\t3\n'

    def test_gz_destination_is_compressed_and_round_trips(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\ng1\t1\ng2\t2\n')
        b = self._part(tmp_path, 'b.tsv', 'gene_id\tlod\ng3\t3\n')
        dest = tmp_path / 'merged.tsv.gz'
        assert atomic.atomic_concat_table([a, b], dest, ['gene_id', 'lod']) == 3
        assert dest.read_bytes()[:2] == b'\x1f\x8b'
        with gzip.open(dest, 'rt', newline='') as fh:
            assert fh.read() == 'gene_id\tlod\ng1\t1\ng2\t2\ng3\t3\n'
        assert read_table(dest)['gene_id'].tolist() == ['g1', 'g2', 'g3']
        assert _no_temp_files(tmp_path)

    def test_gz_output_is_deterministic(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\ng1\t1\n')
        first, second = tmp_path / 'one.tsv.gz', tmp_path / 'two.tsv.gz'
        atomic.atomic_concat_table([a], first, ['gene_id', 'lod'])
        atomic.atomic_concat_table([a], second, ['gene_id', 'lod'])
        assert first.read_bytes() == second.read_bytes()

    def test_header_only_parts_give_header_only_file(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\n')
        dest = tmp_path / 'merged.tsv'
        assert atomic.atomic_concat_table([a], dest, ['gene_id', 'lod']) == 0
        assert dest.read_text() == 'gene_id\tlod\n'

    def test_no_parts_writes_fallback_header(self, tmp_path: Path) -> None:
        dest = tmp_path / 'merged.tsv'
        assert atomic.atomic_concat_table([], dest, ['gene_id', 'lod']) == 0
        assert list(read_table(dest).columns) == ['gene_id', 'lod']

    def test_part_without_final_newline_is_rejected_and_nothing_written(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\ng1\t1')
        b = self._part(tmp_path, 'b.tsv', 'gene_id\tlod\ng2\t2\n')
        dest = tmp_path / 'merged.tsv'
        with pytest.raises(ValueError, match='does not end with a newline'):
            atomic.atomic_concat_table([a, b], dest, ['gene_id', 'lod'])
        assert not dest.exists()
        assert _no_temp_files(tmp_path)

    def test_header_mismatch_is_rejected(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\ng1\t1\n')
        b = self._part(tmp_path, 'b.tsv', 'gene_id\tpip\ng2\t2\n')
        with pytest.raises(ValueError, match='header differs'):
            atomic.atomic_concat_table([a, b], tmp_path / 'merged.tsv', ['gene_id', 'lod'])

    def test_empty_part_is_rejected(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.tsv', '')
        with pytest.raises(ValueError, match='empty file'):
            atomic.atomic_concat_table([a], tmp_path / 'merged.tsv', ['gene_id'])

    def test_blocks_larger_than_copy_size_are_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(atomic, '_COPY_BLOCK', 16)
        rows = ''.join(f'g{i}\t{i}\n' for i in range(50))
        a = self._part(tmp_path, 'a.tsv', 'gene_id\tlod\n' + rows)
        dest = tmp_path / 'merged.tsv'
        assert atomic.atomic_concat_table([a], dest, ['gene_id', 'lod']) == 50
        assert dest.read_text() == 'gene_id\tlod\n' + rows


class TestAtomicJsonText:
    def test_json_round_trip_with_newline(self, tmp_path: Path) -> None:
        dest = tmp_path / 'p.json'
        atomic_write_json({'g1': 0.5, 'g2': 1e-9}, dest)
        assert dest.read_text().endswith('\n')
        assert json.loads(dest.read_text()) == {'g1': 0.5, 'g2': 1e-9}

    def test_text_overwrites_atomically(self, tmp_path: Path) -> None:
        dest = tmp_path / 't.txt'
        atomic_write_text('one\n', dest)
        atomic_write_text('two\n', dest)
        assert dest.read_text() == 'two\n'
        assert _no_temp_files(tmp_path)

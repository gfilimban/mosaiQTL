"""Tests for mosaiqtl.io.atomic: complete-or-absent output files."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from mosaiqtl.io import atomic
from mosaiqtl.io.atomic import atomic_write_csv, atomic_write_json, atomic_write_text


def _no_temp_files(directory: Path) -> bool:
    return not any(p.name.endswith('.tmp') for p in directory.iterdir())


class TestAtomicCsv:
    def test_round_trip_and_trailing_newline(self, tmp_path: Path) -> None:
        df = pd.DataFrame({'gene_id': ['g1', 'g2'], 'lod': [3.25, 1.5]})
        dest = tmp_path / 'out.csv'
        atomic_write_csv(df, dest)
        assert dest.read_text().endswith('\n')
        pd.testing.assert_frame_equal(pd.read_csv(dest), df)
        assert _no_temp_files(tmp_path)

    def test_float_format_keeps_small_values(self, tmp_path: Path) -> None:
        dest = tmp_path / 'p.csv'
        atomic_write_csv(pd.DataFrame({'p': [1.234567891234e-12]}), dest)
        assert pd.read_csv(dest)['p'][0] == pytest.approx(1.23456789e-12, rel=1e-8)

    def test_header_only_frame_is_parseable(self, tmp_path: Path) -> None:
        dest = tmp_path / 'empty.csv'
        atomic_write_csv(pd.DataFrame(columns=['a', 'b']), dest)
        assert list(pd.read_csv(dest).columns) == ['a', 'b']

    def test_failure_midway_leaves_no_partial_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dest = tmp_path / 'out.csv'

        def explode(self: pd.DataFrame, fh, **kwargs) -> None:
            fh.write('gene_id,lod\ng1,')
            raise MemoryError('killed midway')

        monkeypatch.setattr(pd.DataFrame, 'to_csv', explode)
        with pytest.raises(MemoryError):
            atomic_write_csv(pd.DataFrame({'gene_id': ['g1'], 'lod': [1.0]}), dest)
        assert not dest.exists()
        assert _no_temp_files(tmp_path)

    def test_failure_keeps_previous_complete_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dest = tmp_path / 'out.csv'
        atomic_write_csv(pd.DataFrame({'a': [1]}), dest)
        before = dest.read_text()

        def explode(self: pd.DataFrame, fh, **kwargs) -> None:
            fh.write('a\n9')
            raise KeyboardInterrupt

        monkeypatch.setattr(pd.DataFrame, 'to_csv', explode)
        with pytest.raises(KeyboardInterrupt):
            atomic_write_csv(pd.DataFrame({'a': [2]}), dest)
        assert dest.read_text() == before
        assert _no_temp_files(tmp_path)

    def test_fsync_before_rename(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        order: list[str] = []
        real_fsync, real_replace = atomic.os.fsync, atomic.os.replace
        monkeypatch.setattr(atomic.os, 'fsync', lambda fd: (order.append('fsync'), real_fsync(fd))[1])
        monkeypatch.setattr(atomic.os, 'replace', lambda a, b: (order.append('replace'), real_replace(a, b))[1])
        atomic_write_csv(pd.DataFrame({'a': [1]}), tmp_path / 'o.csv')
        assert order == ['fsync', 'replace']


class TestAtomicConcatCsv:
    def _part(self, directory: Path, name: str, text: str) -> Path:
        path = directory / name
        path.write_text(text)
        return path

    def test_header_once_rows_in_order(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.csv', 'gene_id,lod\ng1,1\ng2,2\n')
        b = self._part(tmp_path, 'b.csv', 'gene_id,lod\ng3,3\n')
        dest = tmp_path / 'merged.csv'
        n_lines = atomic.atomic_concat_csv([a, b], dest, ['gene_id', 'lod'])
        assert n_lines == 3
        assert dest.read_text() == 'gene_id,lod\ng1,1\ng2,2\ng3,3\n'

    def test_header_only_parts_give_header_only_file(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.csv', 'gene_id,lod\n')
        dest = tmp_path / 'merged.csv'
        assert atomic.atomic_concat_csv([a], dest, ['gene_id', 'lod']) == 0
        assert dest.read_text() == 'gene_id,lod\n'

    def test_no_parts_writes_fallback_header(self, tmp_path: Path) -> None:
        dest = tmp_path / 'merged.csv'
        assert atomic.atomic_concat_csv([], dest, ['gene_id', 'lod']) == 0
        assert list(pd.read_csv(dest).columns) == ['gene_id', 'lod']

    def test_part_without_final_newline_is_rejected_and_nothing_written(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.csv', 'gene_id,lod\ng1,1')
        b = self._part(tmp_path, 'b.csv', 'gene_id,lod\ng2,2\n')
        dest = tmp_path / 'merged.csv'
        with pytest.raises(ValueError, match='does not end with a newline'):
            atomic.atomic_concat_csv([a, b], dest, ['gene_id', 'lod'])
        assert not dest.exists()
        assert _no_temp_files(tmp_path)

    def test_header_mismatch_is_rejected(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.csv', 'gene_id,lod\ng1,1\n')
        b = self._part(tmp_path, 'b.csv', 'gene_id,pip\ng2,2\n')
        with pytest.raises(ValueError, match='header differs'):
            atomic.atomic_concat_csv([a, b], tmp_path / 'merged.csv', ['gene_id', 'lod'])

    def test_empty_part_is_rejected(self, tmp_path: Path) -> None:
        a = self._part(tmp_path, 'a.csv', '')
        with pytest.raises(ValueError, match='empty file'):
            atomic.atomic_concat_csv([a], tmp_path / 'merged.csv', ['gene_id'])

    def test_blocks_larger_than_copy_size_are_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(atomic, '_COPY_BLOCK', 16)
        rows = ''.join(f'g{i},{i}\n' for i in range(50))
        a = self._part(tmp_path, 'a.csv', 'gene_id,lod\n' + rows)
        dest = tmp_path / 'merged.csv'
        assert atomic.atomic_concat_csv([a], dest, ['gene_id', 'lod']) == 50
        assert dest.read_text() == 'gene_id,lod\n' + rows


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

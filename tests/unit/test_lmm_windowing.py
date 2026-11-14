import numpy as np

from mosaiqtl.qtl import lmm as lmm_module


def _chr_data() -> dict:
    return {
        'locus_ids': np.array(['a', 'b', 'c', 'd']),
        'positions': np.array([1.0, 2.0, 3.0, 9.0]),
        'H_t_raw': np.arange(4 * 3 * 2, dtype=float).reshape(4, 3, 2),
        'n_founders': 2,
    }


def test_window_keeps_only_loci_in_window() -> None:
    out = lmm_module.window_chromosome_data(_chr_data(), 2.0, 1.0)
    assert out is not None
    assert out['locus_ids'].tolist() == ['a', 'b', 'c']
    assert out['H_t_raw'].shape == (3, 3, 2)
    assert out['n_founders'] == 2


def test_window_with_no_loci_returns_none() -> None:
    assert lmm_module.window_chromosome_data(_chr_data(), 50.0, 1.0) is None

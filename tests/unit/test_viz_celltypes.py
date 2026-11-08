"""Smoke tests for viz.celltypes.ct_eqtl_summary."""
import numpy as np
import pandas as pd

from mosaiqtl.viz.celltypes import ct_eqtl_summary


def test_ct_eqtl_summary_writes_transparent_and_share(tmp_path):
    counts = pd.DataFrame(
        {
            'celltype': ['astrocyte', 'pericyte'],
            'n_tested': [100, 40],
            'n_eqtl': [60, 10],
        }
    )
    lods = np.random.default_rng(0).uniform(3, 9, 70)
    out = tmp_path / 'summary.png'
    ct_eqtl_summary(counts, lods, out, n_distinct_egenes=65)
    assert out.exists()
    assert (tmp_path / 'summary_share.png').exists()

"""
sim_sc — sparc phenotype → single-cell NB count simulation

converts sparc strain-level continuous phenotypes into NB count data
suitable for Phase 4 NB-GLMM validation.  high-phenotype strains get
higher NB mean, preserving the planted QTL signal.

usage (CLI):
    python validation/sim_sc.py \
        --sparc_h5 validation/sparcc_data/phenotypes.h5 \
        --sparcc_cache validation/sparcc_cache/ \
        --output tests/fixtures/data/sparc_cc40/ \
        --seed 42

programmatic:
    from validation.sim_sc import sparc_to_nb_counts
    counts, strain_ids, true_locus = sparc_to_nb_counts(
        phenotypes_h5='validation/sparcc_data/phenotypes.h5',
        sim_index=0, phi=0.3, n_cells_mean=200, seed=42)
"""

import argparse
import logging
import os
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger('mosaiqtl.sim_sc')


def sparc_to_nb_counts(
    phenotypes_h5: str,
    sim_index: int = 0,
    phi: float = 0.3,
    n_cells_mean: int = 200,
    baseline: float = 5.0,
    effect_scale: float = 1.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, str]:
    """convert sparc strain-level phenotype to NB single-cell counts.

    parameters
    ----------
    phenotypes_h5 : str
        path to HDF5 file written by TASK 0a R script
    sim_index : int
        which of the 10 sparc simulations to use (0-indexed)
    phi : float
        NB dispersion parameter (typical skeletal muscle ~0.3)
    n_cells_mean : int
        average cells per strain (Poisson-drawn per strain)
    baseline : float
        log-mean baseline expression added to all strains
    effect_scale : float
        scale factor applied to sparc phenotype before exp()
    seed : int
        random seed for reproducibility

    returns
    -------
    counts : np.ndarray (1, n_cells) int32
        raw UMI counts for one simulated gene
    strain_ids : np.ndarray (n_cells,) str
        strain label per cell
    true_locus : str
        planted QTL locus name from sparc
    """
    rng = np.random.default_rng(seed)

    with h5py.File(phenotypes_h5, 'r') as f:
        sim_y       = f['sim_y'][:]                      # (n_strains, n_sim)
        subject_ids = f['subject_ids'][:].astype(str)    # (n_strains, n_sim)
        true_loci   = f['true_locus'][:].astype(str)     # (n_sim,)

    y          = sim_y[sim_index, :]              # (n_strains,)
    strains    = subject_ids[sim_index, :]
    true_locus = true_loci[sim_index]

    n_strains = len(strains)

    # draw n_cells per strain ~ Poisson(n_cells_mean), min 1
    n_cells_per_strain = np.maximum(
        rng.poisson(n_cells_mean, size=n_strains), 1)

    # NB mean per strain: exp(baseline + effect_scale * phenotype)
    mu_strain = np.exp(baseline + effect_scale * y)   # (n_strains,)

    counts_list = []
    strain_id_list = []

    for i, (strain, mu_i, n_c) in enumerate(
            zip(strains, mu_strain, n_cells_per_strain)):
        # NB parameterisation: p = 1/(1 + phi*mu),  r = 1/phi
        r = 1.0 / phi
        p = 1.0 / (1.0 + phi * float(mu_i))
        c = rng.negative_binomial(r, p, size=int(n_c)).astype(np.int32)
        counts_list.append(c)
        strain_id_list.extend([strain] * int(n_c))

    counts    = np.concatenate(counts_list)[np.newaxis, :]  # (1, n_cells)
    strain_ids = np.array(strain_id_list)

    return counts, strain_ids, true_locus


def _save_fixtures(output_dir: str, seed: int = 42,
                   n_cells_mean: int = 200, phi: float = 0.3) -> None:
    """generate and save sparc_cc40 fixtures for all 10 simulations."""
    h5_path = 'validation/sparcc_data/phenotypes.h5'
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, 'r') as f:
        true_loci = f['true_locus'][:].astype(str)
        n_sim = len(true_loci)

    for i in range(n_sim):
        counts, strain_ids, true_locus = sparc_to_nb_counts(
            h5_path, sim_index=i, phi=phi,
            n_cells_mean=n_cells_mean, seed=seed + i)

        out_h5 = out / f'sim_{i:02d}.h5'
        with h5py.File(out_h5, 'w') as f:
            f.create_dataset('counts',     data=counts)
            f.create_dataset('strain_ids', data=strain_ids.astype('S'))
            f.attrs['true_locus']   = np.bytes_(true_locus)
            f.attrs['sim_index']    = i
            f.attrs['phi']          = phi
            f.attrs['seed']         = seed + i
            f.attrs['n_cells_mean'] = n_cells_mean

        logger.info('sim %02d → %s  (true_locus=%s)', i, out_h5, true_locus)

    logger.info('saved %d fixtures to %s', n_sim, output_dir)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    ap = argparse.ArgumentParser(
        description='convert sparc phenotypes to NB single-cell count fixtures')
    ap.add_argument('--sparc_h5',     default='validation/sparcc_data/phenotypes.h5')
    ap.add_argument('--sparcc_cache', default='validation/sparcc_cache/')
    ap.add_argument('--output',       default='tests/fixtures/data/sparc_cc40/')
    ap.add_argument('--seed',         type=int, default=42)
    ap.add_argument('--n_cells_mean', type=int, default=200)
    ap.add_argument('--phi',          type=float, default=0.3)
    args = ap.parse_args()

    _save_fixtures(args.output, seed=args.seed,
                   n_cells_mean=args.n_cells_mean, phi=args.phi)

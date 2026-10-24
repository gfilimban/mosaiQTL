"""
generate_sim_cc.py — Python-native CC phenotype and genome-cache simulator.

Produces a self-contained synthetic Collaborative Cross dataset from scratch —
no external genome cache download required.  Both outputs are drop-in
replacements for the sparcc-generated files used by validate_sim.py:

  cache.h5          GenomeCache HDF5 (haplotype probs for all chromosomes)
  phenotypes.h5     Planted-QTL phenotypes (format identical to sparcc_data/)

How it works
------------
1. Build a synthetic genetic map for the mouse genome (chr 1–19, X) using
   realistic physical lengths (mm39) and a uniform recombination rate.

2. For each of ``num_lines`` CC strains, simulate two haploid chromosomes along
   each chromosome via a Markov chain (Haldane recombination model).  Each
   haploid starts at a random founder; at each marker it switches founder with
   Haldane probability r(d) = (1 − exp(−2·cM·d)) / 2 where cM is the assumed
   rate (default 0.5 cM/Mb, matching average mouse genome).  The per-locus
   founder haplotype dosage is the mean of the two haploid indicators: values
   in {0, 0.5, 1.0} that sum to 1 per locus per strain — the haploid-scale
   convention used throughout the package.

3. Pick ``num_sim`` random loci (one per simulation), extract the founder
   haplotype dosage matrix H[strain, founder] at that locus, draw centred
   founder effects β, scale to achieve Var(H@β) = ``qtl_effect_size`` exactly
   (sparcc mode B), and add Gaussian noise for the residual variance.

4. Write both HDF5 files.

Usage
-----
python validation/tools/generate_sim_cc.py \\
    --output-dir  validation/sim_data \\
    --num-lines   40 \\
    --num-sim     10 \\
    --loci-per-mb 2 \\
    --qtl-effect-size 0.8 \\
    --seed        42

Programmatic
------------
from validation.tools.generate_sim_cc import sim_cc_data, write_outputs
result = sim_cc_data(num_lines=40, num_sim=10, qtl_effect_size=0.8, seed=42)
write_outputs(result, output_dir='validation/sim_data')
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np

log = logging.getLogger('mosaiqtl.generate_sim_cc')

# ── mouse chromosome physical lengths (Mb, mm39 rounded) ─────────────────────
_CHR_LEN_MB: dict[str, float] = {
    '1': 195.2, '2': 181.7, '3': 160.0, '4': 156.3, '5': 151.8,
    '6': 149.7, '7': 145.4, '8': 129.4, '9': 124.6, '10': 130.7,
    '11': 122.1, '12': 120.1, '13': 120.4, '14': 125.0, '15': 104.0,
    '16': 98.2,  '17': 94.9,  '18': 90.7,  '19': 61.4,  'X': 168.6,
}

# Average mouse recombination rate (cM per Mb); Morgan 2015 + Brunschwig 2012
_DEFAULT_RECOMB_CM_PER_MB: float = 0.5

# CC founders in canonical display-name order (matches genotype/founders.py)
_CC_FOUNDERS: list[str] = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']


# ── output data structure ─────────────────────────────────────────────────────
@dataclass
class SimCCResult:
    """Output of :func:`sim_cc_data`."""

    # haplotype probabilities: {chrom: (n_strains, n_loci, n_founders) float32}
    probs: dict[str, np.ndarray]
    # locus map: {chrom: {'locus_ids': (n_loci,) str, 'positions': (n_loci,) Mb float}}
    locus_map: dict[str, dict]
    # simulated phenotypes
    sim_y: np.ndarray            # (num_sim, num_lines) float64
    subject_ids: np.ndarray      # (num_sim, num_lines) str
    true_locus: np.ndarray       # (num_sim,) str  — planted locus ID per sim
    true_pos_mb: np.ndarray      # (num_sim,) float64
    true_chr: np.ndarray         # (num_sim,) str
    # metadata
    strain_ids: list[str]
    founders: list[str]
    num_lines: int
    qtl_effect_size: float
    seed: int
    loci_per_mb: float


# ── genetic map ───────────────────────────────────────────────────────────────
def _build_locus_map(
    loci_per_mb: float = 2.0,
    chromosomes: list[str] | None = None,
    chr_lengths: dict[str, float] | None = None,
) -> dict[str, dict]:
    """Return a locus map with evenly spaced markers per chromosome.

    Parameters
    ----------
    loci_per_mb:
        Marker density (default 2 → 1 locus every 0.5 Mb).
    chromosomes:
        Chromosomes to include; defaults to 1–19 + X.
    chr_lengths:
        Physical lengths in Mb; defaults to mm39 rounded values.

    Returns
    -------
    {chrom: {'locus_ids': (n_loci,) str, 'positions': (n_loci,) float64 Mb}}
    """
    chrs = chromosomes or list(_CHR_LEN_MB)
    lengths = chr_lengths or _CHR_LEN_MB
    locus_map: dict[str, dict] = {}
    for ch in chrs:
        length = lengths[ch]
        n_loci = max(2, round(length * loci_per_mb))
        pos = np.linspace(1.0, length - 1.0, n_loci)          # Mb, avoid telomeres
        ids = np.array([f'UNC{ch}_{i:05d}' for i in range(n_loci)])
        locus_map[ch] = {'locus_ids': ids, 'positions': pos.astype(np.float64)}
    return locus_map


# ── Markov-chain haplotype simulation ─────────────────────────────────────────
def _haldane_trans_prob(delta_mb: float, cm_per_mb: float) -> float:
    """Haldane recombination fraction for a physical distance.

    r(d) = (1 − exp(−2·cM·d)) / 2
    """
    cM = cm_per_mb * delta_mb
    return 0.5 * (1.0 - np.exp(-2.0 * cM / 100.0))


def _simulate_haploid_chrom(
    n_loci: int,
    pos_mb: np.ndarray,
    n_founders: int,
    cm_per_mb: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one haploid chromosome as a sequence of founder states.

    Returns
    -------
    state : (n_loci,) int8  — founder index [0, n_founders)
    """
    state = np.empty(n_loci, dtype=np.int8)
    state[0] = rng.integers(0, n_founders)
    for i in range(1, n_loci):
        d = pos_mb[i] - pos_mb[i - 1]
        r = _haldane_trans_prob(d, cm_per_mb)
        if rng.random() < r:
            # switch to a random *different* founder
            others = np.arange(n_founders, dtype=np.int8)
            others = others[others != state[i - 1]]
            state[i] = rng.choice(others)
        else:
            state[i] = state[i - 1]
    return state


def _simulate_probs_one_chr(
    pos_mb: np.ndarray,
    n_strains: int,
    n_founders: int,
    cm_per_mb: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate diploid haplotype probabilities for all strains on one chromosome.

    Each strain has two independent haploid chromosomes.  Diploid probability
    at locus l for founder f = (hap1[l]==f + hap2[l]==f) / 2 → values in
    {0, 0.5, 1.0}, rows sum to 1.

    Returns
    -------
    probs : (n_strains, n_loci, n_founders) float32
    """
    n_loci = len(pos_mb)
    probs = np.zeros((n_strains, n_loci, n_founders), dtype=np.float32)
    for s in range(n_strains):
        hap1 = _simulate_haploid_chrom(n_loci, pos_mb, n_founders, cm_per_mb, rng)
        hap2 = _simulate_haploid_chrom(n_loci, pos_mb, n_founders, cm_per_mb, rng)
        for f in range(n_founders):
            probs[s, :, f] = ((hap1 == f).astype(np.float32) +
                               (hap2 == f).astype(np.float32)) * 0.5
    return probs


# ── QTL phenotype simulation (sparcc mode B) ──────────────────────────────────
def _simulate_phenotype(
    H: np.ndarray,
    qtl_effect_size: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Simulate one phenotype from the founder haplotype dosage matrix H (mode-B scaling).

    Parameters
    ----------
    H : (n_strains, n_founders) float  — haplotype dosages, rows sum to 1
    qtl_effect_size : float            — target QTL heritability in [0, 1)

    Returns
    -------
    y : (n_strains,) float64  — unit total phenotypic variance
    """
    n_founders = H.shape[1]

    # draw centred founder effects (sum → 0, identifiable under full-rank H)
    beta_raw = rng.standard_normal(n_founders)
    beta = beta_raw - beta_raw.mean()

    y_qtl = H @ beta                                       # (n_strains,)
    var_qtl = float(np.var(y_qtl, ddof=0))
    if var_qtl < 1e-12:                                    # degenerate locus
        var_qtl = 1e-12
    # scale so empirical Var(y_qtl) = qtl_effect_size exactly
    y_qtl *= np.sqrt(qtl_effect_size / var_qtl)

    sigma_e = np.sqrt(max(0.0, 1.0 - qtl_effect_size))
    return (y_qtl + rng.standard_normal(H.shape[0]) * sigma_e).astype(np.float64)


# ── main simulation entry point ───────────────────────────────────────────────
def sim_cc_data(
    num_lines: int = 40,
    num_sim: int = 10,
    qtl_effect_size: float = 0.8,
    seed: int = 42,
    loci_per_mb: float = 2.0,
    recomb_cm_per_mb: float = _DEFAULT_RECOMB_CM_PER_MB,
    chromosomes: list[str] | None = None,
    founders: list[str] | None = None,
) -> SimCCResult:
    """Simulate a complete CC dataset without any external files.

    Parameters
    ----------
    num_lines:
        Number of CC strains (default 40, matching Phase 1 validation).
    num_sim:
        Number of independent phenotype simulations.
    qtl_effect_size:
        Proportion of phenotypic variance explained by the planted QTL.
    seed:
        Master random seed.  Genome and phenotype draws are fully reproducible.
    loci_per_mb:
        Marker density.  Default 2 → 1 marker every 0.5 Mb, giving ~1 600
        loci on a 100 Mb chromosome.
    recomb_cm_per_mb:
        Recombination rate assumed for the Markov chain (cM per Mb).
        Default 0.5 matches the average mouse genome (Morgan 2015).
    chromosomes:
        Restrict to these chromosomes.  ``None`` → chr 1–19 + X.
    founders:
        Founder order.  ``None`` → canonical CC founder order.

    Returns
    -------
    SimCCResult
    """
    import natsort

    rng = np.random.default_rng(seed)
    if founders is None:
        founders = _CC_FOUNDERS
    n_founders = len(founders)
    if chromosomes is None:
        chromosomes = list(_CHR_LEN_MB)

    strain_ids = [f'CC{i:03d}' for i in range(1, num_lines + 1)]
    log.info('simulating CC genome  n_strains=%d  n_founders=%d', num_lines, n_founders)

    # ── build locus map ──────────────────────────────────────────────────────
    locus_map = _build_locus_map(loci_per_mb=loci_per_mb, chromosomes=chromosomes)

    # ── simulate haplotype probs per chromosome ───────────────────────────────
    probs: dict[str, np.ndarray] = {}
    total_loci = 0
    for ch in natsort.natsorted(chromosomes):
        pos = locus_map[ch]['positions']
        log.info('  chr%s: %d loci  %.1f – %.1f Mb', ch, len(pos), pos[0], pos[-1])
        probs[ch] = _simulate_probs_one_chr(pos, num_lines, n_founders, recomb_cm_per_mb, rng)
        total_loci += len(pos)
    log.info('genome: %d chromosomes  %d total loci', len(chromosomes), total_loci)

    # ── plant QTL ─────────────────────────────────────────────────────────────
    # collect all (chrom, locus_idx) pairs
    all_loci: list[tuple[str, int]] = [
        (ch, i)
        for ch in natsort.natsorted(chromosomes)
        for i in range(len(locus_map[ch]['positions']))
    ]
    chosen_indices = rng.integers(0, len(all_loci), size=num_sim)

    sim_y = np.zeros((num_sim, num_lines), dtype=np.float64)
    true_locus = np.empty(num_sim, dtype=object)
    true_pos_mb = np.zeros(num_sim, dtype=np.float64)
    true_chr = np.empty(num_sim, dtype=object)

    for s_idx, idx in enumerate(chosen_indices):
        ch, l_idx = all_loci[idx]
        locus_id = str(locus_map[ch]['locus_ids'][l_idx])
        pos_mb   = float(locus_map[ch]['positions'][l_idx])
        H = probs[ch][:, l_idx, :]                        # (n_strains, n_founders)

        sim_y[s_idx]    = _simulate_phenotype(H, qtl_effect_size, rng)
        true_locus[s_idx] = locus_id
        true_pos_mb[s_idx] = pos_mb
        true_chr[s_idx]    = ch
        log.debug('sim %d/%d planted at %s chr%s %.2f Mb', s_idx + 1, num_sim,
                  locus_id, ch, pos_mb)

    subject_ids = np.array([strain_ids] * num_sim, dtype=object)
    log.info('planted %d QTL phenotypes  qtl_effect_size=%.2f', num_sim, qtl_effect_size)

    return SimCCResult(
        probs=probs,
        locus_map=locus_map,
        sim_y=sim_y,
        subject_ids=subject_ids,
        true_locus=true_locus.astype(str),
        true_pos_mb=true_pos_mb,
        true_chr=true_chr.astype(str),
        strain_ids=strain_ids,
        founders=founders,
        num_lines=num_lines,
        qtl_effect_size=qtl_effect_size,
        seed=seed,
        loci_per_mb=loci_per_mb,
    )


# ── I/O ───────────────────────────────────────────────────────────────────────
def write_cache_h5(result: SimCCResult, path: str | Path) -> None:
    """Write the simulated haplotype probs to a GenomeCache HDF5 file.

    The resulting file is readable by ``GenomeCache.from_hdf5()``.

    Layout (per chromosome group ``/chr{c}/``):
      founders    (root)   S bytes  — founder display names
      sample_ids           S bytes  — CC strain IDs
      locus_ids            S bytes  — synthetic marker names
      positions            float64  — physical positions (Mb)
      probs                float32  — (n_loci, n_strains, n_founders), gzip-4

    Note: axis order stored as (n_loci, n_strains, n_founders) to match the
    package's GenomeCache internal convention (written directly via h5py).
    """
    import natsort
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out, 'w') as f:
        f.create_dataset('founders', data=np.array(result.founders, dtype='S'))
        for ch in natsort.natsorted(result.probs):
            grp = f.create_group(f'chr{ch}')
            m = result.locus_map[ch]
            grp.create_dataset('sample_ids',
                               data=np.array(result.strain_ids, dtype='S'))
            grp.create_dataset('locus_ids',
                               data=np.array(m['locus_ids'], dtype='S'))
            grp.create_dataset('positions', data=m['positions'])
            # probs stored as (n_loci, n_strains, n_founders) — package convention
            p = np.transpose(result.probs[ch], (1, 0, 2)).astype(np.float32)
            grp.create_dataset('probs', data=p, compression='gzip', compression_opts=4)
    log.info('wrote cache.h5 → %s', out)


def write_phenotypes_h5(result: SimCCResult, path: str | Path) -> None:
    """Write simulated phenotypes to HDF5 (format matches sparcc_data/phenotypes.h5).

    Schema:
      sim_y            (num_sim, num_lines) float64
      subject_ids      (num_sim, num_lines) bytes — CC strain IDs
      true_locus       (num_sim,) bytes
      true_pos_mb      (num_sim,) float64
      num_lines        (1,) float64
      qtl_effect_size  (1,) float64
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    subj_bytes = np.array(
        [[s.encode() for s in row] for row in result.subject_ids], dtype='S20'
    )
    with h5py.File(out, 'w') as f:
        f.create_dataset('sim_y',           data=result.sim_y)
        f.create_dataset('subject_ids',     data=subj_bytes)
        f.create_dataset('true_locus',
                         data=np.array([l.encode() for l in result.true_locus], dtype='S64'))
        f.create_dataset('true_pos_mb',     data=result.true_pos_mb)
        f.create_dataset('num_lines',       data=np.array([result.num_lines], dtype=np.float64))
        f.create_dataset('qtl_effect_size',
                         data=np.array([result.qtl_effect_size], dtype=np.float64))
        f.attrs['seed']          = result.seed
        f.attrs['loci_per_mb']   = result.loci_per_mb
        f.attrs['num_founders']  = len(result.founders)
    log.info('wrote phenotypes.h5 → %s', out)


def write_outputs(result: SimCCResult, output_dir: str | Path) -> None:
    """Write both ``cache.h5`` and ``phenotypes.h5`` to ``output_dir``."""
    out = Path(output_dir)
    write_cache_h5(result,       out / 'cache.h5')
    write_phenotypes_h5(result,  out / 'phenotypes.h5')


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli() -> None:
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    ap = argparse.ArgumentParser(
        description='Generate synthetic CC genome cache + phenotypes (no external data needed).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument('--output-dir',       default='validation/sim_data',
                    help='directory for cache.h5 and phenotypes.h5')
    ap.add_argument('--num-lines',        type=int,   default=40)
    ap.add_argument('--num-sim',          type=int,   default=10)
    ap.add_argument('--qtl-effect-size',  type=float, default=0.8)
    ap.add_argument('--loci-per-mb',      type=float, default=2.0,
                    help='marker density (default 2 → one locus every 0.5 Mb)')
    ap.add_argument('--recomb-cm-per-mb', type=float,
                    default=_DEFAULT_RECOMB_CM_PER_MB,
                    help='recombination rate in cM/Mb (default 0.5 = average mouse)')
    ap.add_argument('--chromosomes',      default=None,
                    help='comma-separated chr list, e.g. 1,2,8 (default: all)')
    ap.add_argument('--seed',             type=int,   default=42)
    args = ap.parse_args()

    chrs = [c.strip() for c in args.chromosomes.split(',')] if args.chromosomes else None

    result = sim_cc_data(
        num_lines=args.num_lines,
        num_sim=args.num_sim,
        qtl_effect_size=args.qtl_effect_size,
        seed=args.seed,
        loci_per_mb=args.loci_per_mb,
        recomb_cm_per_mb=args.recomb_cm_per_mb,
        chromosomes=chrs,
    )
    write_outputs(result, args.output_dir)

    log.info('')
    log.info('done')
    log.info('  cache.h5       → %s/cache.h5', args.output_dir)
    log.info('  phenotypes.h5  → %s/phenotypes.h5', args.output_dir)
    log.info('  planted loci   : %s', ', '.join(
        f'chr{c} {p:.1f}Mb' for c, p in zip(result.true_chr[:3], result.true_pos_mb[:3])
    ) + ' …')
    log.info('  phenotype range: [%.3f, %.3f]',
             float(result.sim_y.min()), float(result.sim_y.max()))


if __name__ == '__main__':
    _cli()

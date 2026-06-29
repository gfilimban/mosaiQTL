"""
ase — allele-specific expression via kallisto|bustools (Layer 3c)

workflow per F1 cross:
  1. build two-founder transcriptome index (dam × sire) via kb ref
  2. run kallisto bus against the two-allele index
  3. bustools sort → bustools capture (t2g splits by allele prefix)
  4. bustools count → two count matrices (dam / sire)
  5. return per-gene allelic imbalance: AI = (dam − sire) / (dam + sire)

design notes:
  - only F1 heterozygous samples are processed (CC inbred strains skipped)
  - founder_fastas are MGP pseudogenomes (SNPs/indels applied to mm39 reference)
  - transcript names are prefixed with founder code to allow allele separation
    by bustools using a two-column t2g: transcript_allele → gene_id
  - pseudoalignment is fast and memory-frugal relative to STAR-based competitive
    mapping; k-mer discrimination is reliable for divergent founders (CAST, PWK)
    and may have lower precision for classical founders sharing high sequence identity

dependencies (external, must be in PATH):
  kallisto >= 0.50  bustools >= 0.43  kb-python >= 0.28
"""

import os
import subprocess
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger('mosaiqtl')

# ── founder code extraction from F1 cross name ──────────────────────────────
# convention: cross names encode dam × sire as 'B6{SIRE}F1J'
# e.g. 'B6CASTF1J' → dam='B6J', sire='CASTJ'
_CROSS_TO_FOUNDERS = {
    'B6AF1J':     ('B6J',    'AJ'),
    'B6129S1F1J': ('B6J',    '129S1J'),
    'B6NODF1J':   ('B6J',    'NODJ'),
    'B6NZOF1J':   ('B6J',    'NZOJ'),
    'B6CASTF1J':  ('B6J',    'CASTJ'),
    'B6PWKF1J':   ('B6J',    'PWKJ'),
    'B6WSBF1J':   ('B6J',    'WSBJ'),
}


def run_kallisto_ase(f1_fastq_dirs, founder_fastas, gtf_path,
                      output_dir=None, threads=8):
    """run kallisto|bustools ASE pipeline for all F1 crosses.

    parameters
    ----------
    f1_fastq_dirs  : dict {cross_name: [r1_path, r2_path]}
    founder_fastas : dict {founder_name: fasta_path}
    gtf_path       : str
    output_dir     : str or None  defaults to ./ase_output/
    threads        : int

    returns
    -------
    dict {cross_name: pd.DataFrame}
        columns: gene_id, dam_counts, sire_counts, n_cells, ai
    """
    output_dir = output_dir or os.path.join(os.getcwd(), 'ase_output')
    os.makedirs(output_dir, exist_ok=True)

    results = {}
    for cross, fastqs in f1_fastq_dirs.items():
        founders = _CROSS_TO_FOUNDERS.get(cross)
        if founders is None:
            logger.warning(f'ase: unknown cross {cross!r} — skipping')
            continue
        dam_name, sire_name = founders
        dam_fasta  = founder_fastas.get(dam_name)
        sire_fasta = founder_fastas.get(sire_name)
        if dam_fasta is None or sire_fasta is None:
            logger.warning(f'ase: missing FASTA for {dam_name} or {sire_name} — skipping {cross}')
            continue
        logger.info(f'ase: processing {cross} ({dam_name} × {sire_name})')
        cross_dir = os.path.join(output_dir, cross)
        os.makedirs(cross_dir, exist_ok=True)
        df = _kallisto_ase_one(cross, dam_name, dam_fasta,
                                sire_name, sire_fasta,
                                fastqs, gtf_path, cross_dir, threads)
        if df is not None:
            results[cross] = df
    return results


def _kallisto_ase_one(cross, dam_name, dam_fasta, sire_name, sire_fasta,
                       fastqs, gtf_path, out_dir, threads):
    """run kallisto ASE pipeline for one F1 cross.

    builds a concatenated two-allele transcriptome index where transcripts
    are prefixed with 'dam_' or 'sire_' to allow bustools allele separation.

    returns pd.DataFrame or None on failure.
    """
    index_path  = os.path.join(out_dir, 'two_allele.idx')
    t2g_path    = os.path.join(out_dir, 't2g.txt')
    bus_path    = os.path.join(out_dir, 'output.bus')
    sorted_path = os.path.join(out_dir, 'sorted.bus')
    dam_dir     = os.path.join(out_dir, 'dam')
    sire_dir    = os.path.join(out_dir, 'sire')
    for d in (dam_dir, sire_dir): os.makedirs(d, exist_ok=True)

    # step 1: build two-allele index with kb ref
    # prefixes transcripts so bustools can separate alleles via t2g
    _run(['kb', 'ref',
          '--workflow', 'nac',
          '-i', index_path,
          '-g', t2g_path,
          '-f1', os.path.join(out_dir, 'dam.fasta'),
          '-f2', os.path.join(out_dir, 'sire.fasta'),
          '--d-list', dam_fasta,    # dam as d-list (distinguishing k-mers)
          dam_fasta, gtf_path,
          sire_fasta, gtf_path], out_dir)

    # step 2: kallisto bus — pseudoalign to two-allele index
    _run(['kallisto', 'bus',
          '-i', index_path,
          '-o', out_dir,
          '-x', '10xv3',   # update if using different 10x chemistry
          '-t', str(threads)] + list(fastqs), out_dir)

    # step 3: bustools sort
    _run(['bustools', 'sort',
          '-t', str(threads),
          '-o', sorted_path,
          bus_path], out_dir)

    # step 4: bustools capture + count per allele
    # transcript names are prefixed 'dam|' and 'sire|' in two-allele index
    for allele, cap_dir in ((dam_name, dam_dir), (sire_name, sire_dir)):
        _run(['bustools', 'capture',
              '-s', sorted_path,
              '-o', os.path.join(cap_dir, 'captured.bus'),
              '-c', t2g_path,
              '--genemap', f'{allele}|'], out_dir)
        _run(['bustools', 'count',
              '-o', os.path.join(cap_dir, 'counts'),
              '-g', t2g_path,
              '--genecounts',
              os.path.join(cap_dir, 'captured.bus')], out_dir)

    # step 5: load and merge allele counts
    dam_df  = _load_counts(os.path.join(dam_dir, 'counts.mtx'),
                            os.path.join(out_dir, 'genes.txt'), dam_name)
    sire_df = _load_counts(os.path.join(sire_dir, 'counts.mtx'),
                            os.path.join(out_dir, 'genes.txt'), sire_name)
    if dam_df is None or sire_df is None:
        return None

    mg = dam_df.merge(sire_df, on='gene_id', how='outer').fillna(0)
    mg['n_cells']     = len(fastqs)   # placeholder; update with cell barcode count
    total             = mg['dam_counts'] + mg['sire_counts']
    mg['ai']          = np.where(total > 0,
                                  (mg['dam_counts'] - mg['sire_counts']) / total,
                                  np.nan)
    return mg[['gene_id', 'dam_counts', 'sire_counts', 'n_cells', 'ai']]


def _run(cmd, cwd):
    """run external command, log output, raise on failure."""
    logger.info(f'ase: {" ".join(cmd)}')
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f'ase command failed:\n{result.stderr}')
        raise RuntimeError(f'{cmd[0]} failed: {result.stderr[:200]}')


def _load_counts(mtx_path, genes_path, allele_name):
    """load bustools count output into a gene-level DataFrame."""
    if not os.path.exists(mtx_path):
        logger.warning(f'ase: count file missing: {mtx_path}')
        return None
    from scipy.io import mmread
    mat   = mmread(mtx_path).toarray()
    with open(genes_path) as _fh:
        genes = [ln.strip() for ln in _fh]
    counts = mat.sum(axis=0)   # sum across cells
    return pd.DataFrame({'gene_id': genes,
                          f'{allele_name.lower()}_counts': counts}).rename(
        columns={f'{allele_name.lower()}_counts': 'dam_counts'
                 if 'dam' in allele_name.lower() else 'sire_counts'})

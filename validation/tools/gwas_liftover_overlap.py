"""gwas_liftover_overlap.py — liftover + positional overlap for eGene GWAS matching.

For each unique eGene peak position (deduplicated across all cell types), lifts
the peak from mm39 to hg38 via the UCSC mm39ToHg38 chain file, then queries
the GWAS Catalog within a fixed ±window_mb window around the lifted peak.

Writes one row per (eGene × GWAS hit) — NOT per (eGene × celltype × GWAS hit):
the same gene expressed as an eGene in multiple cell types has one peak position
and therefore one set of GWAS overlaps.  Cell-type membership is downstream
information carried in the eqtl_summary files, not here.

Output:
    validation/ctxhipp_data/bioground/gwas_overlap/
        gwas_liftover_hits_{tag}.csv

bioground.py reads this file and builds trait → gene_symbol mappings for EFO
enrichment; it does not require a celltype column here.

Window choice
-------------
A fixed ±1 Mb window around the liftOver-projected peak matches standard
practice in comparative QTL papers and the CID window used elsewhere in this
analysis.

Usage
-----
    conda run -n bioinfo python validation/tools/gwas_liftover_overlap.py
    conda run -n bioinfo python validation/tools/gwas_liftover_overlap.py --tag C
    conda run -n bioinfo python validation/tools/gwas_liftover_overlap.py --window-mb 0.5
"""
from __future__ import annotations

import argparse
import logging
import pathlib
import sys
import zipfile

import pandas as pd
from pyliftover import LiftOver

# ── path wiring ───────────────────────────────────────────────────────────────
_TOOLS      = pathlib.Path(__file__).parent
_VALIDATION = _TOOLS.parent
_REPO       = _VALIDATION.parent
sys.path.insert(0, str(_REPO / 'src'))

logging.basicConfig(level=logging.INFO, format='%(levelname)s  %(message)s')
log = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
# Shared cross-tissue resources live in validation/gwas_overlap/ (not under
# any tissue-specific data dir).  Tissue-specific paths are passed via CLI.
_GWAS_ROOT  = _VALIDATION / 'gwas_overlap'
_CHAIN_GZ   = _GWAS_ROOT / 'mm39ToHg38.over.chain.gz'
_GWAS_ZIP   = _GWAS_ROOT / 'gwas_catalog_cache.zip'
_NOVA_DIR   = _VALIDATION / 'ctxhipp_data' / 'nova' / 'celltypes'   # overridden by --nova-dir
_OUT_DIR    = _GWAS_ROOT  # overridden by --out-dir; tissue subdir created at runtime

_WINDOW_MB_DEFAULT = 1.0   # ±1 Mb around lifted peak in hg38

# GWAS columns carried through to the hit table.
# DISEASE/TRAIT (free-text, redundant with MAPPED_TRAIT) and PUBMEDID
# (redundant with STUDY ACCESSION) are the only truly redundant columns.
# Everything else is retained for traceability and downstream analysis:
#   MAPPED_TRAIT_URI — stable EFO URI; gwas_overlap.py currently matches
#                      on MAPPED_TRAIT free-text, but the URI enables exact
#                      EFO lookup and future-proofs against label drift.
#   SNPS             — rsID for variant-level tracing back to source.
#   MAPPED_GENE      — GWAS Catalog's own gene mapping, useful for sanity check.
#   STUDY ACCESSION  — citation anchor.
_GWAS_KEEP = [
    'MAPPED_TRAIT', 'MAPPED_TRAIT_URI', 'P-VALUE', 'PVALUE_MLOG',
    'SNPS', 'MAPPED_GENE', 'STUDY ACCESSION',
]

# Genome-wide significance threshold.
# 5×10⁻⁸ = Bonferroni correction over ~10⁶ independent SNPs (0.05 / 10⁶).
# Established as the field standard for array-based GWAS (Risch & Merikangas
# 1996; Dudbridge & Gusnanto 2008 Genet Epidemiol).  Sub-threshold associations
# in the GWAS Catalog are not reliably replicated and inflate the overlap table.
_GWS_PVALUE = 5e-8


# ── helpers ───────────────────────────────────────────────────────────────────

def _lift_peak(
    lo: LiftOver,
    chrom: str,
    peak_mb: float,
    window_mb: float,
) -> tuple[str | None, int | None, int | None]:
    """Lift one mm39 peak position to hg38 and apply a fixed window.

    Parameters
    ----------
    lo : LiftOver
        Initialised pyliftover instance for mm39 → hg38.
    chrom : str
        Chromosome (with or without 'chr' prefix).
    peak_mb : float
        Peak QTL position in mm39, in Mb.
    window_mb : float
        Half-width of the query window in hg38, in Mb.

    Returns
    -------
    tuple[str | None, int | None, int | None]
        ``(hg38_chr, hg38_start, hg38_end)`` in bp, or ``(None, None, None)``
        if the peak position cannot be lifted.
    """
    chrom_name = f'chr{chrom}' if not str(chrom).startswith('chr') else str(chrom)
    peak_bp    = int(peak_mb * 1e6)
    result     = lo.convert_coordinate(chrom_name, peak_bp)
    if not result:
        return None, None, None

    hg38_chr   = result[0][0].removeprefix('chr')
    hg38_peak  = int(result[0][1])
    window_bp  = int(window_mb * 1e6)
    hg38_start = max(0, hg38_peak - window_bp)
    hg38_end   = hg38_peak + window_bp
    return hg38_chr, hg38_start, hg38_end


def _load_gwas_catalog(gwas_zip: pathlib.Path) -> pd.DataFrame:
    """Load GWAS Catalog from the local zip cache, pre-filtered to GWS.

    Only associations at P ≤ _GWS_PVALUE (5×10⁻⁸) are loaded.  Sub-threshold
    rows are excluded at parse time to keep memory and output size manageable.
    """
    log.info('loading GWAS Catalog from %s …', gwas_zip)
    with zipfile.ZipFile(gwas_zip) as z:
        fname = z.namelist()[0]
        with z.open(fname) as f:
            gwas = pd.read_csv(
                f, sep='\t', low_memory=False,
                usecols=['CHR_ID', 'CHR_POS', 'P-VALUE'] + _GWAS_KEEP,
            )
    gwas['CHR_POS'] = pd.to_numeric(gwas['CHR_POS'], errors='coerce')
    gwas['P-VALUE']  = pd.to_numeric(gwas['P-VALUE'],  errors='coerce')
    gwas = gwas.dropna(subset=['CHR_ID', 'CHR_POS', 'P-VALUE'])
    gwas['CHR_ID']  = gwas['CHR_ID'].astype(str).str.strip()
    gwas['CHR_POS'] = gwas['CHR_POS'].astype(int)
    log.info('  GWAS Catalog: %d rows with valid positions', len(gwas))
    gwas = gwas[gwas['P-VALUE'] <= _GWS_PVALUE].reset_index(drop=True)
    log.info('  after GWS filter (P ≤ %.0e): %d rows', _GWS_PVALUE, len(gwas))
    return gwas


def _positional_overlap(
    gwas: pd.DataFrame,
    hg38_chr: str,
    hg38_start: int,
    hg38_end: int,
) -> pd.DataFrame:
    """Return GWAS rows whose SNP falls within [hg38_start, hg38_end]."""
    mask = (
        (gwas['CHR_ID'] == hg38_chr) &
        (gwas['CHR_POS'] >= hg38_start) &
        (gwas['CHR_POS'] <= hg38_end)
    )
    return gwas[mask]


def _collect_unique_peaks(
    tag: str,
    nova_dir: pathlib.Path,
) -> pd.DataFrame:
    """Collect unique (gene_id, gene_symbol, chr, peak_mb) across all cell types.

    Deduplication strategy: one row per **``(gene_id, chr_mm39)``**.

    Rationale
    ---------
    A gene can be an eGene in multiple CTs on the **same** chromosome with
    slightly different peak positions (within a few kb).  Those peaks all land
    in the same ±1 Mb hg38 window, so lifting any one of them captures the
    identical GWAS SNPs — we keep only the max-LOD peak per (gene, chr) to
    avoid redundant liftovers with zero information gain.

    A gene can also have a cis peak on its own chromosome **and** a genuine
    trans-eQTL peak on a different chromosome in another CT.  Deduplicating
    on gene_id alone would silently drop the trans locus.  The per-(gene, chr)
    key ensures every distinct chromosomal locus is lifted and queried.

    Returns a DataFrame with columns:
        gene_id, gene_symbol, chr_mm39, peak_mb_mm39
    """
    frames: list[pd.DataFrame] = []
    ct_dirs = sorted(nova_dir.glob('*/'))

    for ct_path in ct_dirs:
        ct = ct_path.name
        egene_f = ct_path / tag / 'eqtl_summary.csv'
        if not egene_f.exists():
            log.debug('  %s: no eqtl_summary for tag=%s — skipping', ct, tag)
            continue

        df = pd.read_csv(egene_f, usecols=lambda c: c in {
            'gene_id', 'gene', 'gene_symbol', 'chr', 'peak_mb', 'peak_lod',
        })
        if 'gene' in df.columns and 'gene_symbol' not in df.columns:
            df = df.rename(columns={'gene': 'gene_symbol'})
        df['chr'] = df['chr'].astype(str)
        frames.append(df)
        log.debug('  %s: %d eGenes', ct, len(df))

    if not frames:
        log.error('no eqtl_summary files found for tag=%s under %s', tag, nova_dir)
        return pd.DataFrame(columns=['gene_id', 'gene_symbol', 'chr_mm39', 'peak_mb_mm39'])

    all_genes = pd.concat(frames, ignore_index=True)
    log.info('total eGene rows across all CTs: %d', len(all_genes))

    # Dedup on (gene_id, chr): keep the highest-LOD peak per gene per chromosome.
    # Within-chromosome position variation across CTs (a few kb) produces the
    # same ±1 Mb hg38 query window — lifting once is sufficient.
    # Cross-chromosome peaks (cis on gene's own chr + trans on another chr in a
    # different CT) are on different rows and are preserved — they must each be
    # lifted and queried independently to avoid dropping trans-eQTL GWAS hits.
    join_key = 'gene_id' if 'gene_id' in all_genes.columns else 'gene_symbol'
    all_genes = (
        all_genes
        .sort_values('peak_lod', ascending=False)
        .drop_duplicates(subset=[join_key, 'chr'])
        .rename(columns={'chr': 'chr_mm39', 'peak_mb': 'peak_mb_mm39'})
        [['gene_id', 'gene_symbol', 'chr_mm39', 'peak_mb_mm39']]
        .reset_index(drop=True)
    )
    log.info('unique (gene × chr) loci to lift: %d', len(all_genes))
    return all_genes


# ── main ──────────────────────────────────────────────────────────────────────

def run(
    tag: str,
    tissue: str,
    window_mb: float = _WINDOW_MB_DEFAULT,
    nova_dir: pathlib.Path = _NOVA_DIR,
    gwas_zip: pathlib.Path = _GWAS_ZIP,
    chain_gz: pathlib.Path = _CHAIN_GZ,
    out_dir: pathlib.Path = _OUT_DIR,
) -> pathlib.Path:
    """Compute liftover-based GWAS overlap for all eGenes (deduplicated across CTs).

    Parameters
    ----------
    tag : str
        Scan tag (e.g. ``'Lw1'`` or ``'C'``).
    tissue : str
        Tissue identifier (e.g. ``'ctxhipp'``).  Included in output filenames
        so that results from multiple tissues can coexist in the same directory.
    window_mb : float
        Half-width of the hg38 query window around the lifted peak (default 1.0 Mb).
    nova_dir : pathlib.Path
        Path to ``nova/celltypes/`` directory.
    gwas_zip : pathlib.Path
        Local GWAS Catalog cache zip.
    chain_gz : pathlib.Path
        mm39ToHg38.over.chain.gz liftover chain file.
    out_dir : pathlib.Path
        Output directory (default: ``validation/gwas_overlap/``).

    Returns
    -------
    pathlib.Path
        Path to the written ``gwas_liftover_hits_{tissue}_{tag}.csv``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── step 1: unique peaks across all CTs ──────────────────────────────────
    peaks = _collect_unique_peaks(tag, nova_dir)
    if peaks.empty:
        log.error('no eGenes found — aborting')
        return out_dir / f'gwas_liftover_hits_{tissue}_{tag}.csv'

    # ── step 2: liftover chain + GWAS catalog ────────────────────────────────
    log.info('loading liftover chain %s …', chain_gz)
    lo = LiftOver(str(chain_gz))

    gwas = _load_gwas_catalog(gwas_zip)
    log.info('hg38 query window: ±%.2f Mb around lifted peak', window_mb)

    # ── step 3: lift each unique peak, query GWAS ────────────────────────────
    hit_rows: list[dict] = []
    n_lifted = 0
    n_failed = 0

    for _, eg in peaks.iterrows():
        gene_id  = str(eg.get('gene_id', ''))
        gene_sym = str(eg.get('gene_symbol', ''))
        chrom    = str(eg['chr_mm39'])
        peak_mb  = float(eg['peak_mb_mm39'])

        hg38_chr, hg38_start, hg38_end = _lift_peak(lo, chrom, peak_mb, window_mb)
        if hg38_chr is None:
            n_failed += 1
            log.debug('    %s chr%s: liftover failed — skipping', gene_sym, chrom)
            continue

        n_lifted += 1
        hits = _positional_overlap(gwas, hg38_chr, hg38_start, hg38_end)
        if hits.empty:
            continue

        for _, h in hits.iterrows():
            row: dict = {
                'gene_id':      gene_id,
                'gene_symbol':  gene_sym,
                'chr_mm39':     chrom,
                'peak_mb_mm39': peak_mb,
                'hg38_chr':     hg38_chr,
                'hg38_start':   hg38_start,
                'hg38_end':     hg38_end,
                'CHR_ID':       h.get('CHR_ID', ''),
                'CHR_POS':      int(h['CHR_POS']),
            }
            for col in _GWAS_KEEP:
                row[col] = h.get(col, '')
            hit_rows.append(row)

    log.info('peaks lifted: %d  failed: %d', n_lifted, n_failed)

    hits_df = pd.DataFrame(hit_rows)
    out_path = out_dir / f'gwas_liftover_hits_{tissue}_{tag}.csv'

    if hits_df.empty:
        log.warning('no GWAS hits found — writing empty table to %s', out_path)
        hits_df.to_csv(out_path, index=False)
        return out_path

    # Deduplicate by (gene, position, trait): the same physical variant can be
    # associated with many distinct EFO traits — preserve all of them so that
    # gwas_overlap.py's trait→gene map captures the full trait diversity.
    # Within a (gene, position, trait) tuple, keep the most significant
    # association (lowest P-VALUE) to collapse redundant study-level rows for
    # the same variant × trait combination reported under different rs IDs.
    hits_df['P-VALUE'] = pd.to_numeric(hits_df['P-VALUE'], errors='coerce')
    hits_df = (
        hits_df
        .sort_values('P-VALUE', ascending=True, na_position='last')
        .drop_duplicates(subset=['gene_id', 'CHR_ID', 'CHR_POS', 'MAPPED_TRAIT'])
        .reset_index(drop=True)
    )

    n_genes = hits_df['gene_symbol'].nunique()
    log.info(
        'total: %d (gene × GWAS hit) rows | %d unique eGenes with ≥1 hit',
        len(hits_df), n_genes,
    )
    hits_df.to_csv(out_path, index=False)
    log.info('written → %s', out_path)

    # ── lightweight gene summary: one row per eGene ──────────────────────────
    # Columns:
    #   gene_id, gene_symbol
    #   n_gwas_hits  — distinct SNP positions overlapping the QTL window
    #   n_traits     — distinct EFO traits across all overlapping SNPs
    #   top_trait    — MAPPED_TRAIT of the single most-significant association
    #   best_pvalue  — lowest P-VALUE across all overlapping SNPs
    tag_str = f'{tissue}_{tag}'
    gene_list_path = out_path.with_name(f'gwas_hit_genes_{tag_str}.csv')

    # best-hit row per gene (already sorted ascending by P-VALUE from dedup step)
    best_hit = (
        hits_df[['gene_id', 'gene_symbol', 'P-VALUE', 'MAPPED_TRAIT']]
        .sort_values('P-VALUE', ascending=True, na_position='last')
        .drop_duplicates(subset=['gene_id'])
        .rename(columns={'MAPPED_TRAIT': 'top_trait', 'P-VALUE': 'best_pvalue'})
    )
    per_gene = (
        hits_df
        .groupby(['gene_id', 'gene_symbol'], sort=False)
        .agg(
            n_gwas_hits=('CHR_POS', 'nunique'),
            n_traits=('MAPPED_TRAIT', 'nunique'),
        )
        .reset_index()
    )
    gene_list = (
        per_gene
        .merge(best_hit[['gene_id', 'top_trait', 'best_pvalue']], on='gene_id', how='left')
        .sort_values('gene_symbol')
        .reset_index(drop=True)
    )
    gene_list.to_csv(gene_list_path, index=False)
    log.info('gene list → %s  (%d genes)', gene_list_path.name, n_genes)

    return out_path


def main() -> None:
    p = argparse.ArgumentParser(
        description='Liftover-based GWAS positional overlap for mosaiQTL eGenes.'
    )
    p.add_argument('--tissue',    default='ctxhipp',
                   help='Tissue identifier — sets nova-dir default and output filename '
                        '(default: ctxhipp)')
    p.add_argument('--tag',       default='Lw1',
                   help='Scan tag (default: Lw1)')
    p.add_argument('--window-mb', type=float, default=_WINDOW_MB_DEFAULT,
                   help='Half-width of hg38 query window around lifted peak in Mb '
                        f'(default: {_WINDOW_MB_DEFAULT})')
    p.add_argument('--nova-dir',  type=pathlib.Path, default=None,
                   help='Path to nova/celltypes/ directory '
                        '(default: validation/{tissue}_data/nova/celltypes/)')
    p.add_argument('--gwas-zip',  type=pathlib.Path, default=_GWAS_ZIP,
                   help='Local GWAS Catalog cache zip '
                        '(default: validation/gwas_overlap/gwas_catalog_cache.zip)')
    p.add_argument('--chain',     type=pathlib.Path, default=_CHAIN_GZ,
                   help='mm39→hg38 chain file '
                        '(default: validation/gwas_overlap/mm39ToHg38.over.chain.gz)')
    p.add_argument('--out-dir',   type=pathlib.Path, default=None,
                   help='Output directory '
                        '(default: validation/gwas_overlap/; '
                        'outputs named gwas_liftover_hits_{tissue}_{tag}.csv)')
    args = p.parse_args()

    tissue   = args.tissue
    nova_dir = args.nova_dir or (_VALIDATION / f'{tissue}_data' / 'nova' / 'celltypes')
    out_dir  = args.out_dir  or _GWAS_ROOT   # flat: gwas_liftover_hits_{tissue}_{tag}.csv

    run(
        tag=args.tag,
        tissue=tissue,
        window_mb=args.window_mb,
        nova_dir=nova_dir,
        gwas_zip=args.gwas_zip,
        chain_gz=args.chain,
        out_dir=out_dir,
    )


if __name__ == '__main__':
    main()

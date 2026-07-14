"""
snp_scan — haplotype-imputed SNP association scan

imputes biallelic snp dosages from founder haplotype probabilities,
then runs the haplotype lmm scan restricted to the imputed dosage.

  imputed.snp.scan.h2lmm()   →  imputed_snp_scan_h2lmm()
  build.allele.dir() + load  →  imputed_snp_scan_from_db()  (HDF5 path)
"""

import gc
import os
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..utils import chr_sort_key, _build_gm_idx, _enrich_scan_df

logger = logging.getLogger('mosaiqtl')

SNP_SCAN_COLS: list[str] = [
    'gene_name', 'snp', 'chr', 'pos_mb', 'lod', 'p_value',
    'f_stat', 'h2', 'n', 'rss0', 'rss1', 'beta_dosage',
]


def imputed_snp_scan_h2lmm(phenotype, genomecache, snp_genos,
                            covariates=None, kinship_decomp=None,
                            use_rint=True,
                            phenotype_id='unknown',
                            just_these_snps=None):
    """snp association scan using haplotype-imputed dosages.

    each snp dosage is estimated as the expected number of alt alleles:
      d_i = 2 * sum_f P(f_i) * I(founder f carries alt allele)

    founder haplotype probs used as weights over known founder genotypes.

    parameters
    ----------
    phenotype : np.ndarray (n,)
    genomecache : GenomeCache
    snp_genos : pd.DataFrame
        snp × founder genotype matrix, shape (n_snps, n_founders).
        values should be 0/1/2 (dosage) or 0/1 (biallelic allele count).
        index = snp ids, columns = founder names matching genomecache.founders.
        must include 'chr' and 'pos_mb' columns for locus lookup.
    covariates : np.ndarray (n, p) or None
    kinship_decomp : dict or None
    use_rint : bool  apply RINT to phenotype (default True)
    phenotype_id : str  stored in 'gene_name' column of output
    just_these_snps : list or None  restrict to these snp ids

    returns
    -------
    pd.DataFrame  columns: gene_name, snp, chr, pos_mb, lod, p_value, f_stat,
                           h2, n, rss0, rss1, beta_dosage
        NOT a ScanResult — SNP scan output is per-SNP dosage effects, not
        per-founder haplotype effects.  Do not pass to _scan_to_df or any
        ScanResult consumer.
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    _n_snps_requested = (len(just_these_snps) if just_these_snps is not None
                         else len(snp_genos))
    logger.info(
        'imputed_snp_scan_h2lmm: gene=%s  n_snps=%d',
        phenotype_id, _n_snps_requested,
    )

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method='loco')
        kinship_decomp = kin_module.decomp_kinship(kd)

    valid  = ~np.isnan(phenotype)
    y      = phenotype[valid].copy()
    if use_rint:
        y = _rint(y)
    n      = len(y)

    x = (covariates[valid].copy() if covariates is not None
         else np.ones((n, 1), dtype=float))
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    # align founder order between snp_genos columns and cache founders
    founders_cache = list(genomecache.founders)
    snp_founders   = [c for c in founders_cache if c in snp_genos.columns]
    if len(snp_founders) < 2:
        raise ValueError('snp_genos must have columns matching genomecache.founders')

    snps_to_scan = (just_these_snps if just_these_snps is not None
                    else snp_genos.index.tolist())

    results = []

    # normalize snp_genos chr col once: strip leading 'chr' to match cache format
    _snp_chr_norm = snp_genos['chr'].astype(str).str.replace(r'^chr', '', regex=True)

    # only iterate chromosomes with kinship available
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    for chrom in _chrom_list:
        _c = str(chrom).replace('chr', '')               # bare chr id e.g. '8'
        chr_snps = snp_genos[_snp_chr_norm == _c]
        chr_snps = chr_snps.loc[chr_snps.index.isin(snps_to_scan)]
        if chr_snps.empty:
            continue

        lids, pos, probs = genomecache.get_chromosome(chrom)   # probs: (L,n,J)
        eigvals = kinship_decomp[chrom]['values']
        eigvecs = kinship_decomp[chrom]['vectors']
        ev_v    = eigvecs[valid]
        y_rot   = ev_v.T @ y
        x_rot   = ev_v.T @ x
        h2, _   = lmm_module.estimate_h2_null(y_rot, x_rot, eigvals)

        for snp_id, row in chr_snps.iterrows():
            snp_pos = float(row.get('pos_mb', np.nan))
            if np.isnan(snp_pos):
                continue

            # find nearest locus for haplotype probs
            nearest_i = int(np.argmin(np.abs(pos - snp_pos)))
            h_locus   = probs[nearest_i][valid]   # (n_valid, J)

            # impute snp dosage: d = sum_f h_f * g_f  (g_f = alt allele count)
            def _to_float(v):
                """Coerce allele value to float; missing codes (N, -, etc.) become nan."""  # noqa: E501
                try: return float(v)
                except (TypeError, ValueError): return np.nan
            geno_f = np.array([_to_float(row.get(f, np.nan)) for f in founders_cache])
            if np.any(np.isnan(geno_f)):
                geno_f = np.where(np.isnan(geno_f), 0.0, geno_f)
            dosage = h_locus @ geno_f   # (n_valid,)

            # skip monomorphic snps
            if dosage.std() < 1e-8:
                continue

            # standardise dosage and fit lmm with dosage as scalar predictor
            d_std   = (dosage - dosage.mean()) / dosage.std()
            h_rot   = (ev_v.T @ d_std).reshape(-1, 1)

            res = lmm_module.fit_lmm_at_locus(
                y_rot, h_rot, x_rot, eigvals, h2)

            results.append({
                'gene_name':   phenotype_id,
                'snp':         snp_id,
                'chr':         chrom,
                'pos_mb':      snp_pos,
                'lod':         res.lod,
                'p_value':     res.p_value,
                'f_stat':      res.f_stat,
                'h2':          res.h2,
                'n':           n,
                'rss0':        res.rss_null,
                'rss1':        res.rss_alt,
                'beta_dosage': float(res.betas[0]),
            })

    return pd.DataFrame(results)


def imputed_snp_scan_from_db(
    phenotype: np.ndarray,
    genomecache,
    alleles_h5: 'str | os.PathLike',
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    chrom: 'str | None' = None,
    founder_col_map: 'dict[str, str] | None' = None,
    use_rint: bool = True,
    phenotype_id: str = 'unknown',
    just_these_snps: 'list | None' = None,
    sample_ids: 'list | None' = None,
) -> pd.DataFrame:
    """SNP scan loading founder allele calls from an HDF5 database.

    Wraps imputed_snp_scan_h2lmm: loads the founder allele HDF5 built by
    build_founder_allele_db(), converts nucleotide calls (A/T/C/G) to
    binary 0/1 dosage (alphabetically later allele = alt = 1), then passes
    the numeric snp_genos DataFrame to imputed_snp_scan_h2lmm.

    parameters
    ----------
    phenotype   : np.ndarray (n,) — sample-level if sample_ids provided, else strain-level
    genomecache : GenomeCache
    alleles_h5  : path to HDF5 built by build_founder_allele_db()
    covariates  : np.ndarray (n, p) or None
    kinship_decomp : dict or None
    chrom       : restrict scan to this chromosome (e.g. 'chr10')
    founder_col_map : dict mapping HDF5 founder column names to cache
                  founder names.  e.g. {'AJ': 'A', 'B6J': 'B', ...} when
                  GM_fo_mm39.csv was built with abbreviated founder names
                  but the cache uses single-letter codes.  None = no rename.
    use_rint    : bool
    phenotype_id : str
    just_these_snps : list or None  restrict to these SNP ids
    sample_ids  : list of genotype IDs (one per row of phenotype/covariates);
                  when provided, reduces to strain means via Z-matrix before
                  calling imputed_snp_scan_h2lmm.  Pass geno_ids (cache strain
                  names), not raw sample labels.

    returns
    -------
    pd.DataFrame  columns: gene_name, snp, chr, pos_mb, lod, p_value, f_stat,
                           h2, n, rss0, rss1, beta_dosage
        NOT a ScanResult — use the DataFrame directly; do not pass to
        _scan_to_df or any ScanResult consumer.
    """
    from ..genotype.snp_db import load_founder_alleles

    alleles_df = load_founder_alleles(alleles_h5, chrom=chrom)
    if alleles_df.empty:
        return pd.DataFrame(
            columns=['gene_name', 'snp', 'chr', 'pos_mb', 'lod', 'p_value',
                     'f_stat', 'h2', 'n', 'rss0', 'rss1', 'beta_dosage'])

    meta_cols = {'marker', 'chr', 'pos_mb', 'pos_bp', 'locus_id'}
    hdf5_fcols = [c for c in alleles_df.columns if c not in meta_cols]

    if founder_col_map is not None:
        alleles_df = alleles_df.rename(columns=founder_col_map)
        hdf5_fcols = [founder_col_map.get(c, c) for c in hdf5_fcols]

    cache_founders = list(genomecache.founders)
    overlap = [f for f in cache_founders if f in hdf5_fcols]
    if len(overlap) < 2:
        raise ValueError(
            f'founder columns in alleles_h5 {hdf5_fcols} do not overlap with '
            f'genomecache.founders {cache_founders}. Use founder_col_map to rename.')

    # convert nucleotide calls → 0/1 dosage (alt = alphabetically later allele)
    snp_ids, rows = [], []
    for _, row in alleles_df.iterrows():
        calls = {f: str(row[f]) for f in overlap}
        distinct = sorted(
            {v for v in calls.values() if v not in ('N', 'H', '', 'nan')})
        if len(distinct) < 2:
            continue  # monomorphic in the overlap founder set
        alt = distinct[-1]
        geno_row = {'chr': row['chr'], 'pos_mb': float(row['pos_mb'])}
        for f in overlap:
            geno_row[f] = 1.0 if calls[f] == alt else 0.0
        snp_ids.append(row['marker'])
        rows.append(geno_row)

    if not rows:
        return pd.DataFrame(
            columns=['gene_name', 'snp', 'chr', 'pos_mb', 'lod', 'p_value',
                     'f_stat', 'h2', 'n', 'rss0', 'rss1', 'beta_dosage'])

    snp_genos = pd.DataFrame(rows, index=snp_ids)
    snp_genos.index.name = None

    # reduce sample-level inputs to strain means when sample_ids are provided
    if sample_ids is not None:
        from ..utils import build_z_matrix
        z_mat, valid_mask = build_z_matrix(sample_ids, list(genomecache.sample_ids))
        valid_idx = np.where(valid_mask)[0]
        z_sub = z_mat[valid_idx]
        r_cnt = z_sub.sum(axis=0)
        active = r_cnt > 0
        z_act = z_sub[:, active]
        r_act = r_cnt[active]
        active_idx = np.where(active)[0]
        n_strains = len(genomecache.sample_ids)

        y_str = np.full(n_strains, np.nan)
        y_str[active_idx] = (z_act.T @ phenotype[valid_idx]) / r_act
        phenotype = y_str

        if covariates is not None:
            cov_str = np.full((n_strains, covariates.shape[1]), np.nan)
            cov_str[active_idx] = (z_act.T @ covariates[valid_idx]) / r_act[:, None]
            covariates = cov_str

    return imputed_snp_scan_h2lmm(
        phenotype, genomecache, snp_genos,
        covariates=covariates,
        kinship_decomp=kinship_decomp,
        use_rint=use_rint,
        phenotype_id=phenotype_id,
        just_these_snps=just_these_snps,
    )


def read_allele_dir(allele_dir, founders, chrom=None):
    """read flat allele_dir into a snp_genos DataFrame.

    allele_dir structure (HAPPY format):
      allele_dir/
        chr1/
          SNP1.RData    ← named vector of founder allele dosages (0/1/2)
          SNP2.RData
        chr8/
          ...

    parameters
    ----------
    allele_dir : str   path to allele directory
    founders   : list  founder names in cache order
    chrom      : str or None  if given, only load this chromosome

    returns
    -------
    pd.DataFrame  index=snp_id, columns=founders + ['chr','pos_mb']
    """
    import glob
    import os
    try:
        import rpy2.robjects as ro
    except ImportError:
        raise ImportError('rpy2 required for reading allele_dir .RData files')

    rows = []
    chr_dirs = sorted(
        glob.glob(os.path.join(allele_dir, 'chr*')),
        key=lambda p: chr_sort_key(os.path.basename(p).replace('chr', '')),
    )
    for chr_dir in chr_dirs:
        c = os.path.basename(chr_dir).replace('chr', '')
        if chrom is not None and c != str(chrom):
            continue
        rdata_files = sorted(glob.glob(os.path.join(chr_dir, '*.RData')))
        for rf in rdata_files:
            snp_id = os.path.splitext(os.path.basename(rf))[0]
            env    = ro.r['new.env']()
            ro.r['load'](rf, envir=env)
            obj    = ro.r['get'](ro.r['ls'](envir=env)[0], envir=env)
            arr    = np.array(obj, dtype=float).ravel()
            # arr length = n_founders; map to founder names
            row = {'snp_id': snp_id, 'chr': c, 'pos_mb': np.nan}
            for fi, f in enumerate(founders):
                row[f] = float(arr[fi]) if fi < len(arr) else np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['chr', 'pos_mb'] + list(founders))
    df = pd.DataFrame(rows).set_index('snp_id')
    logger.info(f'read_allele_dir: {len(df)} SNPs from {allele_dir}')
    return df


def build_allele_dir(snp_genos, allele_dir, pmap=None):
    """build a flat allele_dir from a snp_genos DataFrame.

    writes one .RData file per SNP using rpy2.


    parameters
    ----------
    snp_genos  : pd.DataFrame  index=snp_id, cols=founders + ['chr','pos_mb']
    allele_dir : str   output directory path
    pmap       : pd.DataFrame or None  optional; columns=['snp_id','chr','pos_mb']
                 used to fill missing chr/pos_mb in snp_genos

    returns
    -------
    str  path to allele_dir
    """
    import os
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri
        numpy2ri.activate()
    except ImportError:
        raise ImportError('rpy2 required for building allele_dir .RData files')

    os.makedirs(allele_dir, exist_ok=True)
    founder_cols = [c for c in snp_genos.columns if c not in ('chr', 'pos_mb')]
    n_written    = 0

    for snp_id, row in snp_genos.iterrows():
        c = str(row.get('chr', 'unknown'))
        chr_dir = os.path.join(allele_dir, f'chr{c}')
        os.makedirs(chr_dir, exist_ok=True)
        rdata_path = os.path.join(chr_dir, f'{snp_id}.RData')
        if os.path.exists(rdata_path):
            continue
        geno_vec = np.array([float(row.get(f, 0)) for f in founder_cols])
        rv       = ro.FloatVector(geno_vec)
        rv.names = ro.StrVector(founder_cols)
        env      = ro.r['new.env']()
        env[snp_id] = rv
        ro.r['save'](ro.StrVector([snp_id]), file=rdata_path, envir=env)
        n_written += 1

    logger.info(f'build_allele_dir: wrote {n_written} SNP files to {allele_dir}')
    return allele_dir


def read_alleles_flat(allele_dir, founders, chrom=None, pmap_dir=None):
    """read flat chrN.alleles files into a snp_genos DataFrame.

    format — 4 lines per SNP:
      line 1 : SNP_ID
      line 2 : ref_allele\tf1\t...\tf8   (0/1 per founder: 1=has ref)
      line 3 : alt_allele\tf1\t...\tf8   (0/1 per founder: 1=has alt)
      line 4 : NA\t0\t...\t0

    founder dosage returned is the alt allele indicator (line 3 values): 0.0
    for ref-homozygous founders, 1.0 for alt-homozygous founders.  consistent
    with imputed_snp_scan_h2lmm's dosage = h_locus @ geno_f.

    parameters
    ----------
    allele_dir : str   directory containing chrN.alleles flat files
    founders   : list  founder names in same order as .alleles file columns
    chrom      : str or None   restrict to this chromosome
    pmap_dir   : str or None   directory with GM_pmap*.csv files; adds pos_mb

    returns
    -------
    pd.DataFrame  index=snp_id, columns=founders + ['chr', 'pos_mb']
    """
    import glob as _glob

    rows, snp_ids = [], []
    for fpath in sorted(
        _glob.glob(os.path.join(allele_dir, 'chr*.alleles')),
        key=lambda p: chr_sort_key(
            os.path.basename(p).replace('chr', '').replace('.alleles', '')
        ),
    ):
        c = os.path.basename(fpath).replace('chr', '').replace('.alleles', '')
        if chrom is not None and c != str(chrom):
            continue
        with open(fpath) as fh:
            lines = [ln.rstrip('\n') for ln in fh]
        for i in range(0, len(lines) - 2, 4):    # 4 lines per SNP block
            snp_id    = lines[i].strip()
            alt_parts = lines[i + 2].split('\t')  # line 3: alt allele row
            alt_vals  = alt_parts[1:]              # drop allele label col
            row = {'chr': c}
            for fi, f in enumerate(founders):
                try:    row[f] = float(alt_vals[fi]) if fi < len(alt_vals) else 0.0
                except (ValueError, TypeError): row[f] = 0.0
            snp_ids.append(snp_id)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=['chr', 'pos_mb'] + list(founders))

    df = pd.DataFrame(rows, index=snp_ids)
    df.index.name = 'snp_id'
    df['pos_mb']  = np.nan

    # attach positions from GM_pmap*.csv if pmap_dir provided
    if pmap_dir is not None:
        pmap_files = sorted(_glob.glob(os.path.join(pmap_dir, 'GM_pmap*.csv')))
        frames = []
        for pf in pmap_files:
            try:
                pm = pd.read_csv(pf, comment='#')
                if 'marker' in pm.columns:
                    pm = pm.rename(columns={'marker': 'locus', 'pos': 'pos_mb'})
                frames.append(pm[['locus', 'pos_mb']])
            except Exception:
                pass
        if frames:
            pmap_all = pd.concat(frames).set_index('locus')
            shared   = df.index.intersection(pmap_all.index)
            df.loc[shared, 'pos_mb'] = pmap_all.loc[shared, 'pos_mb'].values

    logger.info(f'read_alleles_flat: {len(df)} SNPs  pos_filled={df["pos_mb"].notna().sum()}')
    return df


def build_snp_catalogue(scan_results, genomecache, mgp_vcf,
                          window_mb=10.0, founders=None):
    """Layer 3a — SNP catalogue from MGP VCF overlay (placeholder).

    for each gene with a QTL peak, queries the MGP VCF within window_mb
    of the peak, filters to SNPs that are private to founders with the
    largest haplotype effects, and ranks candidates by founder divergence.

    requires: pysam (for bgzipped + tabix-indexed VCF access).

    parameters
    ----------
    scan_results : dict {gene_id: ScanResult}
    genomecache : GenomeCache
    mgp_vcf      : str  path to Sanger MGP r9 VCF (bgzipped + tabix-indexed)
    window_mb    : float  window around QTL peak (Mb)
    founders     : list or None  subset of founders to query (default all)

    returns
    -------
    dict {gene_id: pd.DataFrame}
        columns: snp_id, chr, pos_mb, ref, alt, founder_genotypes...,
                 n_divergent_founders, effect_dir_match
        rows filtered to founder-divergent SNPs within the QTL window
    """
    try:
        import pysam
    except ImportError:
        raise ImportError('build_snp_catalogue requires pysam: pip install pysam')

    founders_use = founders or list(genomecache.founders)
    vcf = pysam.VariantFile(mgp_vcf)
    results = {}

    for gid, sr in scan_results.items():
        if not len(sr.lod):
            continue
        peak = sr.peak_locus()
        chrom, pos_mb = peak['chromosome'], peak['position']

        lo = max(0.0, pos_mb - window_mb) * 1e6
        hi = (pos_mb + window_mb) * 1e6
        chr_str = f'chr{chrom}' if not str(chrom).startswith('chr') else str(chrom)

        rows = []
        try:
            for rec in vcf.fetch(chr_str, int(lo), int(hi)):
                if len(rec.alts) != 1:  # biallelic only
                    continue
                gts = {}
                for f in founders_use:
                    s = rec.samples.get(f)
                    gts[f] = (int(s['GT'][0]) if s and s['GT'][0] is not None
                               else None)
                # count divergent founders (not all same genotype)
                called = [v for v in gts.values() if v is not None]
                n_div  = len(set(called)) - 1 if called else 0
                if n_div == 0:
                    continue
                row = {'snp_id': rec.id or f'{chrom}:{rec.pos}',
                        'chr': chrom, 'pos_mb': rec.pos / 1e6,
                        'ref': rec.ref, 'alt': rec.alts[0],
                        'n_divergent_founders': n_div}
                row.update({f: gts[f] for f in founders_use})
                rows.append(row)
        except (ValueError, KeyError):
            logger.warning(f'build_snp_catalogue: region fetch failed for {gid}')
            continue

        results[gid] = pd.DataFrame(rows) if rows else pd.DataFrame()

    vcf.close()
    return results


def snp_scan_multi_gene(
    expr_cache: 'dict[str, np.ndarray]',
    gene_chr_map: 'dict[str, str]',
    genomecache,
    alleles_h5: str,
    covariates: 'np.ndarray | None' = None,
    kinship_decomp: 'dict | None' = None,
    sample_ids: 'list[str] | None' = None,
    founder_col_map: 'dict[str, str] | None' = None,
    peak_only: bool = True,
    output_path: 'Path | None' = None,
    gene_meta: 'pd.DataFrame | None' = None,
) -> pd.DataFrame:
    """imputed SNP scan over a set of genes, streaming to disk.

    calls imputed_snp_scan_from_db per gene and streams each result immediately,
    keeping peak memory bounded to one gene at a time.

    parameters
    ----------
    expr_cache      : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map    : dict {gene_id: str}  chromosome for each gene
    genomecache     : GenomeCache
    alleles_h5      : str  path to founder_alleles.h5
    covariates      : np.ndarray (n, p) or None
    kinship_decomp  : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids      : list[str] or None  genotype-level strain IDs
    founder_col_map : dict {founder_name: allele_code} or None
    peak_only       : bool  when True (default) output one peak-SNP row per gene;
                     when False output all SNPs on the gene's chromosome
    output_path     : Path or None  stream CSV here; None accumulates in RAM
    gene_meta       : DataFrame keyed by 'gene_name' — extra columns joined into each row

    returns
    -------
    peak_only=True  — gene_name, chr, snp [+ gene_meta cols], peak_mb, peak_lod, beta_dosage
    peak_only=False — gene_name, chr [+ gene_meta cols], snp, pos_mb, lod, p_value,
                      f_stat, h2, n, rss0, rss1, beta_dosage
    """
    _gm_idx = _build_gm_idx(gene_meta)
    if peak_only:
        _id_cols   = ['gene_name', 'chr', 'snp']
        _data_cols = ['peak_mb', 'peak_lod', 'beta_dosage']
    else:
        _id_cols   = ['gene_name', 'chr']
        _data_cols = ['snp', 'pos_mb', 'lod', 'p_value', 'f_stat', 'h2', 'n',
                      'rss0', 'rss1', 'beta_dosage']
    _first = True
    _buf: 'list[pd.DataFrame]' = []
    _n_genes_snp = len(expr_cache)
    logger.info(
        'snp_scan_multi_gene: n_genes=%d  alleles_h5=%s',
        _n_genes_snp, Path(alleles_h5).name,
    )
    for _g_snp, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, '')
        if not gene_chr:
            logger.warning('snp_scan_multi_gene: no chr for %s — skipped', gene_id)
            continue
        logger.debug(
            'snp_scan_multi_gene: [%d/%d] %s', _g_snp + 1, _n_genes_snp, gene_id)
        _kd_g = ({gene_chr: kinship_decomp[gene_chr]}
                 if kinship_decomp is not None else None)
        try:
            _df = imputed_snp_scan_from_db(
                y, genomecache,
                alleles_h5=alleles_h5,
                chrom=gene_chr,
                covariates=covariates,
                kinship_decomp=_kd_g,
                sample_ids=sample_ids,
                founder_col_map=founder_col_map,
                phenotype_id=gene_id,
            )
            if not _df.empty:
                if peak_only:
                    _df = (_df.loc[[_df['lod'].idxmax()]]
                             .rename(columns={'pos_mb': 'peak_mb', 'lod': 'peak_lod'}))
                _df = _enrich_scan_df(_df, gene_id, _gm_idx, _id_cols, _data_cols)
                if output_path is not None:
                    _df.to_csv(output_path, mode='a', header=_first, index=False)
                    _first = False
                else:
                    _buf.append(_df)
            del _df
        except Exception as exc:
            logger.warning('snp_scan_multi_gene: %s failed — %s', gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (pd.read_csv(output_path)
                if output_path.exists() and not _first else pd.DataFrame())
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()

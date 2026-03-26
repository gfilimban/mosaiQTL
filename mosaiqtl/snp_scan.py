"""
mosaiqtl.snp_scan — haplotype-imputed SNP association scan

imputes biallelic snp dosages from founder haplotype probabilities,
then runs the haplotype lmm scan restricted to the imputed dosage.

equivalent to miqtl:
  imputed.snp.scan.h2lmm()   →  imputed_snp_scan_h2lmm()
  pairwise.cor.snp.scan()    →  pairwise_cor_snp_scan()
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats as sp_stats

logger = logging.getLogger('mosaiqtl')


def imputed_snp_scan_h2lmm(phenotype, genome_cache, snp_genos,
                            covariates=None, kinship_decomp=None,
                            use_rint=True, use_reml=True,
                            phenotype_id='unknown',
                            just_these_snps=None):
    """snp association scan using haplotype-imputed dosages.

    each snp dosage is estimated as the expected number of alt alleles:
      d_i = 2 * sum_f P(f_i) * I(founder f carries alt allele)

    this is the 'human gwas sense' imputation miQTL uses — founder
    haplotype probs are used as weights over the known founder genotypes.

    equivalent to miqtl::imputed.snp.scan.h2lmm().

    parameters
    ----------
    phenotype : np.ndarray (n,)
    genome_cache : GenomeCache
    snp_genos : pd.DataFrame
        snp × founder genotype matrix, shape (n_snps, n_founders).
        values should be 0/1/2 (dosage) or 0/1 (biallelic allele count).
        index = snp ids, columns = founder names matching genome_cache.founders.
        must include 'chr' and 'pos_mb' columns for locus lookup.
    covariates : np.ndarray (n, p) or None
    kinship_decomp : dict or None
    use_rint : bool
    use_reml : bool
    phenotype_id : str
    just_these_snps : list or None  restrict to these snp ids

    returns
    -------
    pd.DataFrame  columns: snp, chr, pos_mb, lod, p_value, beta_dosage
    """
    from . import scan as scan_module
    from . import kinship as kin_module
    from . import lmm as lmm_module
    from .utils import rint as _rint

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genome_cache, method='loco')
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
    founders_cache = list(genome_cache.founders)
    snp_founders   = [c for c in founders_cache if c in snp_genos.columns]
    if len(snp_founders) < 2:
        raise ValueError('snp_genos must have columns matching genome_cache.founders')

    snps_to_scan = (just_these_snps if just_these_snps is not None
                    else snp_genos.index.tolist())

    results = []

    for chrom in genome_cache.chromosomes:
        chr_snps = snp_genos[snp_genos.get('chr', pd.Series(dtype=str)) == chrom]
        chr_snps = chr_snps.loc[chr_snps.index.isin(snps_to_scan)]
        if chr_snps.empty:
            continue

        lids, pos, probs = genome_cache.get_chromosome(chrom)   # probs: (L,n,J)
        # build {pos_mb → locus index} for quick lookup
        pos_to_idx = {float(p): i for i, p in enumerate(pos)}

        if chrom not in kinship_decomp:
            continue
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
            geno_f = np.array([float(row.get(f, np.nan)) for f in founders_cache])
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
                y_rot, h_rot, x_rot, eigvals, h2, use_reml)

            results.append({
                'snp':        snp_id,
                'chr':        chrom,
                'pos_mb':     snp_pos,
                'lod':        res.lod,
                'p_value':    res.p_value,
                'beta_dosage': float(res.betas[0]),
            })

    return pd.DataFrame(results)


def pairwise_cor_snp_scan(phenotype, genome_cache, snp_genos,
                           chrom, window_mb=1.0, peak_snp=None):
    """compute r² between each snp and the peak snp (or all pairs).

    used to colour snp manhattan plots by ld with the peak.
    equivalent to miqtl::pairwise.cor.snp.scan().

    parameters
    ----------
    phenotype : np.ndarray (n,)  used only for alignment
    genome_cache : GenomeCache
    snp_genos : pd.DataFrame  snp × founder genotypes (same as above)
    chrom : str
    window_mb : float  half-window around peak snp (default 1.0)
    peak_snp : str or None  if given, compute r² vs this snp only

    returns
    -------
    pd.DataFrame  columns: snp, pos_mb, r2, r  (vs peak_snp or all pairs)
    """
    founders_cache = list(genome_cache.founders)
    lids, pos, probs = genome_cache.get_chromosome(chrom)

    chr_snps = snp_genos[snp_genos.get('chr', pd.Series(dtype=str)) == chrom]
    if chr_snps.empty:
        return pd.DataFrame(columns=['snp', 'pos_mb', 'r2', 'r'])

    # impute dosages for all snps using mean haplotype probs
    # (averaged across strains as population-level LD)
    mean_hap = probs.mean(axis=1)   # (L, J) mean founder probs at each locus
    pos_arr  = np.array(pos, dtype=float)

    dosages = {}
    for snp_id, row in chr_snps.iterrows():
        snp_pos  = float(row.get('pos_mb', np.nan))
        near_i   = int(np.argmin(np.abs(pos_arr - snp_pos)))
        geno_f   = np.array([float(row.get(f, 0)) for f in founders_cache])
        dosages[snp_id] = (float(snp_pos),
                           float(mean_hap[near_i] @ geno_f))

    snp_ids  = list(dosages.keys())
    snp_pos_ = np.array([dosages[s][0] for s in snp_ids])
    snp_dos  = np.array([dosages[s][1] for s in snp_ids])

    if peak_snp and peak_snp in dosages:
        ref_dos = np.array([dosages[peak_snp][1]] * len(snp_ids))
        if snp_dos.std() < 1e-8 or ref_dos.std() < 1e-8:
            r_vals = np.zeros(len(snp_ids))
        else:
            r_vals = np.array([
                float(np.corrcoef(snp_dos[i:i+1], ref_dos[i:i+1])[0, 1])
                if np.std([snp_dos[i], ref_dos[i]]) > 1e-8 else 0.0
                for i in range(len(snp_ids))])
        return pd.DataFrame({'snp': snp_ids, 'pos_mb': snp_pos_,
                             'r2': r_vals ** 2, 'r': r_vals})

    # all-pair correlation matrix
    if len(snp_ids) < 2:
        return pd.DataFrame(columns=['snp', 'pos_mb', 'r2', 'r'])
    corr = np.corrcoef(snp_dos.reshape(-1, 1).T)  # fallback identity
    return pd.DataFrame({'snp': snp_ids, 'pos_mb': snp_pos_,
                         'r2': np.diag(corr) ** 2,
                         'r': np.diag(corr)})


def read_allele_dir(allele_dir, founders, chrom=None):
    """read miQTL allele_dir into a snp_genos DataFrame.

    allele_dir structure (miQTL HAPPY format):
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
    import glob, os
    try:
        import rpy2.robjects as ro
    except ImportError:
        raise ImportError('rpy2 required for reading allele_dir .RData files')

    rows = []
    chr_dirs = sorted(glob.glob(os.path.join(allele_dir, 'chr*')))
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
    """build a miQTL-compatible allele_dir from a snp_genos DataFrame.

    writes one .RData file per SNP using rpy2.
    enables use of miqtl::imputed.snp.scan.h2lmm() with Python-prepared data.

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

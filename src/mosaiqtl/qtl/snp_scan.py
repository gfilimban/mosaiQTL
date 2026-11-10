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

from ..utils import _build_gm_idx, _enrich_scan_df

logger = logging.getLogger("mosaiqtl")

SNP_SCAN_COLS: list[str] = [
    "gene_id",
    "snp",
    "chr",
    "pos_mb",
    "lod",
    "p_value",
    "f_stat",
    "h2",
    "n",
    "rss0",
    "rss1",
    "beta_dosage",
]


def _reduce_to_strain_means(
    phenotype: np.ndarray,
    covariates: "np.ndarray | None",
    sample_ids: list,
    genomecache,
) -> "tuple[np.ndarray, np.ndarray | None]":
    """Average sample-level phenotype and covariates to cache strain order.

    Strains without any sample (or samples not in the cache) are NaN, which the
    scan's non-missing mask then drops.
    """
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

    cov_str = None
    if covariates is not None:
        cov_str = np.full((n_strains, covariates.shape[1]), np.nan)
        cov_str[active_idx] = (z_act.T @ covariates[valid_idx]) / r_act[:, None]
    return y_str, cov_str


def imputed_snp_scan_h2lmm(
    phenotype,
    genomecache,
    snp_genos,
    covariates=None,
    kinship_decomp=None,
    use_rint=True,
    phenotype_id="unknown",
    gene_symbol="",
    just_these_snps=None,
    sample_ids=None,
):
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
    phenotype_id : str  stored in 'gene_id' column of output
    gene_symbol : str  display name used only in log output; never written to output columns
    just_these_snps : list or None  restrict to these snp ids
    sample_ids : list or None  genotype ID per row of phenotype/covariates
        (cache strain names, one per sample).  When given, phenotype and
        covariates are reduced to strain means in cache order before the
        scan, so replicate-level inputs align with the strain-level
        haplotype probabilities and kinship.  None = inputs already in
        cache strain order.

    returns
    -------
    pd.DataFrame  columns: gene_id, snp, chr, pos_mb, lod, p_value, f_stat,
                           h2, n, rss0, rss1, beta_dosage
        NOT a ScanResult — SNP scan output is per-SNP dosage effects, not
        per-founder haplotype effects.  Do not pass to _scan_to_df or any
        ScanResult consumer.
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import rint as _rint

    _n_snps_requested = (
        len(just_these_snps) if just_these_snps is not None else len(snp_genos)
    )
    logger.info("imputed_snp_scan_h2lmm: n_snps=%d", _n_snps_requested)

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd)

    _rint_done = False
    if sample_ids is not None:
        pheno_arr = np.asarray(phenotype, dtype=float)
        if use_rint:
            # Apply RINT to sample-level phenotype before strain aggregation so
            # the transformation matches the haplotype scan, which RINTs y_raw
            # at sample level (rint(y_raw[vsid])) before scan_chromosome.
            _valid_ph = ~np.isnan(pheno_arr)
            pheno_arr[_valid_ph] = _rint(pheno_arr[_valid_ph])
            _rint_done = True
        phenotype, covariates = _reduce_to_strain_means(
            pheno_arr, covariates, sample_ids, genomecache
        )
    else:
        n_cache = len(list(genomecache.sample_ids))
        if len(phenotype) != n_cache:
            raise ValueError(
                f"imputed_snp_scan_h2lmm: phenotype length {len(phenotype)} != "
                f"cache n_strains {n_cache}; pass sample_ids to reduce "
                "replicate inputs to strain means before the scan"
            )

    valid = ~np.isnan(phenotype)
    y = phenotype[valid].copy()
    if use_rint and not _rint_done:
        y = _rint(y)
    n = len(y)

    x = (
        covariates[valid].copy()
        if covariates is not None
        else np.ones((n, 1), dtype=float)
    )
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])

    # align founder order between snp_genos columns and cache founders
    founders_cache = list(genomecache.founders)
    snp_founders = [c for c in founders_cache if c in snp_genos.columns]
    if len(snp_founders) < 2:
        raise ValueError("snp_genos must have columns matching genomecache.founders")

    snps_to_scan = (
        just_these_snps if just_these_snps is not None else snp_genos.index.tolist()
    )

    results: list = []
    _pat_best_dosage: dict = {}  # pattern → (dosage_vec, best_lod)

    # normalize snp_genos chr col once: strip leading 'chr' to match cache format
    _snp_chr_norm = snp_genos["chr"].astype(str).str.replace(r"^chr", "", regex=True)

    # only iterate chromosomes with kinship available
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    for chrom in _chrom_list:
        _c = str(chrom).replace("chr", "")  # bare chr id e.g. '8'
        chr_snps = snp_genos[_snp_chr_norm == _c]
        chr_snps = chr_snps.loc[chr_snps.index.isin(snps_to_scan)]
        if chr_snps.empty:
            continue

        lids, pos, probs = genomecache.get_chromosome(chrom)  # probs: (L,n,J)
        eigvals = kinship_decomp[chrom]["values"]
        eigvecs = kinship_decomp[chrom]["vectors"]
        ev_v = eigvecs[valid]
        y_rot = ev_v.T @ y
        x_rot = ev_v.T @ x
        h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, eigvals)

        for snp_id, row in chr_snps.iterrows():
            snp_pos = float(row.get("pos_mb", np.nan))
            if np.isnan(snp_pos):
                continue

            # find nearest locus for haplotype probs
            nearest_i = int(np.argmin(np.abs(pos - snp_pos)))
            h_locus = probs[nearest_i][valid]  # (n_valid, J)

            # impute snp dosage: d = sum_f h_f * g_f  (g_f = alt allele count)
            def _to_float(v):
                """Coerce allele value to float; missing codes (N, -, etc.) become nan."""  # noqa: E501
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return np.nan

            geno_f = np.array([_to_float(row.get(f, np.nan)) for f in founders_cache])
            if np.any(np.isnan(geno_f)):
                geno_f = np.where(np.isnan(geno_f), 0.0, geno_f)
            dosage = h_locus @ geno_f  # (n_valid,)

            # skip monomorphic snps
            if dosage.std() < 1e-8:
                continue

            # standardise dosage and fit lmm with dosage as scalar predictor
            d_std = (dosage - dosage.mean()) / dosage.std()
            h_rot = (ev_v.T @ d_std).reshape(-1, 1)

            res = lmm_module.fit_lmm_at_locus(y_rot, h_rot, x_rot, eigvals, h2)

            _pat = tuple(int(round(x)) for x in geno_f.tolist())
            results.append(
                {
                    "gene_id": phenotype_id,
                    "snp": snp_id,
                    "chr": chrom,
                    "pos_mb": snp_pos,
                    "lod": res.lod,
                    "p_value": res.p_value,
                    "f_stat": res.f_stat,
                    "h2": res.h2,
                    "n": n,
                    "rss0": res.rss_null,
                    "rss1": res.rss_alt,
                    "beta_dosage": float(res.betas[0]),
                    "dosage_mean": float(dosage.mean()),
                    "_founder_pattern": _pat,
                }
            )
            # track best dosage per unique founder pattern — one vector per
            # pattern (≤2^8=256 for 8 binary founder positions), not one per
            # SNP. memory stays O(n_patterns × n_valid) regardless of window.
            if _pat not in _pat_best_dosage or res.lod > _pat_best_dosage[_pat][1]:
                _pat_best_dosage[_pat] = (dosage, res.lod)

    df = pd.DataFrame(results)
    if df.empty:
        return df

    # compact LD representation: r² between pattern representatives.
    # callers extract this via df.attrs["_ld"] before any concat strips it.
    rep_patterns = list(_pat_best_dosage.keys())
    rep_dosages = np.column_stack(
        [_pat_best_dosage[p][0] for p in rep_patterns]
    )  # (n_valid, n_reps) — n_reps ≤ 256
    if rep_dosages.shape[1] > 1:
        corr = np.corrcoef(rep_dosages.T)
        r2_between_reps = corr ** 2
    else:
        r2_between_reps = np.array([[1.0]])
    df.attrs["_ld"] = {
        "rep_patterns": rep_patterns,   # list of 8-tuple
        "r2_matrix": r2_between_reps,   # (n_reps, n_reps) r² matrix
    }
    return df


def imputed_snp_scan_from_db(
    phenotype: np.ndarray,
    genomecache,
    alleles_h5: "str | os.PathLike",
    covariates: "np.ndarray | None" = None,
    kinship_decomp: "dict | None" = None,
    chrom: "str | None" = None,
    founder_col_map: "dict[str, str] | None" = None,
    use_rint: bool = True,
    phenotype_id: str = "unknown",
    just_these_snps: "list | None" = None,
    sample_ids: "list | None" = None,
) -> pd.DataFrame:
    """SNP scan loading founder allele calls from an HDF5 database.

    Wraps imputed_snp_scan_h2lmm: loads the founder allele HDF5 built by
    build_founder_allele_gm(), converts nucleotide calls (A/T/C/G) to
    binary 0/1 dosage (alphabetically later allele = alt = 1), then passes
    the numeric snp_genos DataFrame to imputed_snp_scan_h2lmm.

    parameters
    ----------
    phenotype   : np.ndarray (n,) — sample-level if sample_ids provided, else strain-level
    genomecache : GenomeCache
    alleles_h5  : path to HDF5 built by build_founder_allele_gm()
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
    pd.DataFrame  columns: gene_id, snp, chr, pos_mb, lod, p_value, f_stat,
                           h2, n, rss0, rss1, beta_dosage
        NOT a ScanResult — use the DataFrame directly; do not pass to
        _scan_to_df or any ScanResult consumer.
    """
    from ..genotype.snp_db import load_founder_alleles

    alleles_df = load_founder_alleles(alleles_h5, chrom=chrom)
    if alleles_df.empty:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "snp",
                "chr",
                "pos_mb",
                "lod",
                "p_value",
                "f_stat",
                "h2",
                "n",
                "rss0",
                "rss1",
                "beta_dosage",
            ]
        )

    meta_cols = {"marker", "chr", "pos_mb", "pos_bp", "locus_id"}
    hdf5_fcols = [c for c in alleles_df.columns if c not in meta_cols]

    if founder_col_map is not None:
        alleles_df = alleles_df.rename(columns=founder_col_map)
        hdf5_fcols = [founder_col_map.get(c, c) for c in hdf5_fcols]

    cache_founders = list(genomecache.founders)
    overlap = [f for f in cache_founders if f in hdf5_fcols]
    if len(overlap) < 2:
        raise ValueError(
            f"founder columns in alleles_h5 {hdf5_fcols} do not overlap with "
            f"genomecache.founders {cache_founders}. Use founder_col_map to rename."
        )

    # convert nucleotide calls → 0/1 dosage (alt = alphabetically later allele)
    snp_ids, rows = [], []
    for _, row in alleles_df.iterrows():
        calls = {f: str(row[f]) for f in overlap}
        distinct = sorted({v for v in calls.values() if v not in ("N", "H", "", "nan")})
        if len(distinct) < 2:
            continue  # monomorphic in the overlap founder set
        alt = distinct[-1]
        geno_row = {"chr": row["chr"], "pos_mb": float(row["pos_mb"])}
        for f in overlap:
            geno_row[f] = 1.0 if calls[f] == alt else 0.0
        snp_ids.append(row["marker"])
        rows.append(geno_row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "gene_id",
                "snp",
                "chr",
                "pos_mb",
                "lod",
                "p_value",
                "f_stat",
                "h2",
                "n",
                "rss0",
                "rss1",
                "beta_dosage",
            ]
        )

    snp_genos = pd.DataFrame(rows, index=snp_ids)
    snp_genos.index.name = None

    return imputed_snp_scan_h2lmm(
        phenotype,
        genomecache,
        snp_genos,
        covariates=covariates,
        kinship_decomp=kinship_decomp,
        use_rint=use_rint,
        phenotype_id=phenotype_id,
        just_these_snps=just_these_snps,
        sample_ids=sample_ids,
    )


def snp_scan_multi_gene(
    expr_cache: "dict[str, np.ndarray]",
    gene_chr_map: "dict[str, str]",
    genomecache,
    alleles_h5: str,
    covariates: "np.ndarray | None" = None,
    kinship_decomp: "dict | None" = None,
    sample_ids: "list[str] | None" = None,
    founder_col_map: "dict[str, str] | None" = None,
    peak_only: bool = True,
    output_path: "Path | None" = None,
    gene_meta: "pd.DataFrame | None" = None,
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
    gene_meta       : DataFrame keyed by 'gene_id' (the identity key; carries
                     gene_symbol and any other display columns) — extra
                     columns joined into each row

    returns
    -------
    peak_only=True  — gene_id, chr, snp [+ gene_meta cols], peak_mb, peak_lod, beta_dosage
    peak_only=False — gene_id, chr [+ gene_meta cols], snp, pos_mb, lod, p_value,
                      f_stat, h2, n, rss0, rss1, beta_dosage
    """
    _gm_idx = _build_gm_idx(gene_meta)
    if peak_only:
        _id_cols = ["gene_id", "chr", "snp"]
        _data_cols = ["peak_mb", "peak_lod", "beta_dosage"]
    else:
        _id_cols = ["gene_id", "chr"]
        _data_cols = [
            "snp",
            "pos_mb",
            "lod",
            "p_value",
            "f_stat",
            "h2",
            "n",
            "rss0",
            "rss1",
            "beta_dosage",
        ]
    _first = True
    _buf: "list[pd.DataFrame]" = []
    _n_genes_snp = len(expr_cache)
    logger.info(
        "snp_scan_multi_gene: n_genes=%d  alleles_h5=%s",
        _n_genes_snp,
        Path(alleles_h5).name,
    )
    for _g_snp, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, "")
        if not gene_chr:
            logger.warning("snp_scan_multi_gene: no chr for %s — skipped", gene_id)
            continue
        logger.debug(
            "snp_scan_multi_gene: [%d/%d] %s", _g_snp + 1, _n_genes_snp, gene_id
        )
        _kd_g = (
            {gene_chr: kinship_decomp[gene_chr]} if kinship_decomp is not None else None
        )
        try:
            _df = imputed_snp_scan_from_db(
                y,
                genomecache,
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
                    _df = _df.loc[[_df["lod"].idxmax()]].rename(
                        columns={"pos_mb": "peak_mb", "lod": "peak_lod"}
                    )
                _df = _enrich_scan_df(_df, gene_id, _gm_idx, _id_cols, _data_cols)
                if output_path is not None:
                    _df.to_csv(output_path, mode="a", header=_first, index=False)
                    _first = False
                else:
                    _buf.append(_df)
            del _df
        except Exception as exc:
            logger.warning("snp_scan_multi_gene: %s failed — %s", gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()

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

from ..genotype.interpolation import bracket_weights
from ..utils import _build_gm_idx, _enrich_scan_df

logger = logging.getLogger("mosaiqtl")

# Largest per-strain haplotype probability a founder with no allele call may carry
# before the SNP is dropped; below it the dosage is renormalised over called founders.
MAX_MISSING_HAP_MASS: float = 0.05

# Fewest strains that must carry the minor allele (summed expected dosage over the
# scanned strains, or its complement) for a SNP to be tested.
MIN_MINOR_STRAINS: int = 3

# SNPs processed together when interpolating probabilities and forming dosages;
# bounds the (chunk, strains, founders) temporary.
SNP_CHUNK: int = 512

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
    "se_dosage",
    "minor_strains",
]


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
        (cache strain names, one per sample).  The model is fit at sample
        level: each sample carries the dosage of its strain, and replicate
        samples share a genotype.  None = one sample per cache strain, in
        cache order.

    returns
    -------
    pd.DataFrame  columns: SNP_SCAN_COLS (gene_id, snp, chr, pos_mb, lod,
                           p_value, f_stat, h2, n, rss0, rss1, beta_dosage,
                           se_dosage, minor_strains) plus dosage_mean, the
                           mean dosage over the scanned samples.
                           minor_strains is the smaller of the summed strain
                           dosage and its complement over the strains present.
        NOT a ScanResult — SNP scan output is per-SNP dosage effects, not
        per-founder haplotype effects.  Do not pass to _scan_to_df or any
        ScanResult consumer.
    """
    from ..genotype import kinship as kin_module
    from . import lmm as lmm_module
    from ..utils import build_z_matrix, rint as _rint

    _n_snps_requested = (
        len(just_these_snps) if just_these_snps is not None else len(snp_genos)
    )
    logger.info("imputed_snp_scan_h2lmm: n_snps=%d", _n_snps_requested)

    if kinship_decomp is None:
        kd = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd)

    cache_ids = list(genomecache.sample_ids)
    pheno_arr = np.asarray(phenotype, dtype=float)
    if sample_ids is None:
        if len(pheno_arr) != len(cache_ids):
            raise ValueError(
                f"imputed_snp_scan_h2lmm: phenotype length {len(pheno_arr)} != "
                f"cache n_strains {len(cache_ids)}; pass sample_ids for replicate inputs"
            )
        z_mat, in_cache = np.eye(len(cache_ids)), np.ones(len(cache_ids), dtype=bool)
    else:
        z_mat, in_cache = build_z_matrix(sample_ids, cache_ids)

    # the model is fit at sample level, as the haplotype scan is, so the variant
    # and haplotype likelihoods are nested: same samples, same RINT, same
    # K_n = Z K_m Z^T, and the same ML h^2 for the gene
    valid = in_cache & ~np.isnan(pheno_arr)
    z_sub = z_mat[valid]
    y = pheno_arr[valid].copy()
    if use_rint:
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
    n_skipped_missing = 0

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
        ev_m, evc_m = kinship_decomp[chrom]["values"], kinship_decomp[chrom]["vectors"]
        eigvals, eigvecs = np.linalg.eigh(z_sub @ ((evc_m * ev_m) @ evc_m.T) @ z_sub.T)
        y_rot = eigvecs.T @ y
        x_rot = eigvecs.T @ x
        h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, eigvals, method="ml")

        # strains with at least one scanned sample; dosage is computed per strain
        # and expanded to samples by index
        used = np.flatnonzero(z_sub.any(axis=0))
        sample_to_used = np.searchsorted(used, z_sub.argmax(axis=1))
        probs_used = probs[:, used]  # (L, m, J)
        snp_ids = chr_snps.index.to_numpy()
        snp_mb = pd.to_numeric(chr_snps["pos_mb"], errors="coerce").to_numpy(float)
        # missing allele codes (N, -, ...) coerce to nan
        geno_all = (
            chr_snps.reindex(columns=founders_cache)
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(float)
        )  # (S, J)

        for a in range(0, len(snp_ids), SNP_CHUNK):
            sl = slice(a, a + SNP_CHUNK)
            g = geno_all[sl]
            known = ~np.isnan(g)
            g0 = np.where(known, g, 0.0)
            live = ~np.isnan(snp_mb[sl]) & known.any(axis=1)
            if not live.any():
                continue

            # haplotype probabilities at each SNP position, interpolated between
            # the two bracketing loci
            left, t = bracket_weights(pos, snp_mb[sl])
            t = t[:, None, None]
            p = (1.0 - t) * probs_used[left] + t * probs_used[left + 1]  # (s, m, J)

            # a founder without a call has an unknown allele: strains carrying
            # appreciable haplotype mass on it have no defined dosage
            miss = np.einsum("smj,sj->sm", p, (~known).astype(float))
            too_missing = live & (miss.max(axis=1) > MAX_MISSING_HAP_MASS)
            n_skipped_missing += int(too_missing.sum())
            live &= ~too_missing

            dos_strain = np.einsum("smj,sj->sm", p, g0) / np.maximum(
                1.0 - miss, 1e-12
            )  # (s, m)
            dosage = dos_strain[:, sample_to_used]  # (s, n)

            # skip monomorphic SNPs and SNPs whose minor allele is carried by too
            # few strains: replicates of a strain share one genotype, so the
            # strain count, not the sample count, bounds the independent evidence
            tot = dos_strain.sum(axis=1)
            minor_strains = np.minimum(tot, len(used) - tot)
            d_sd = dosage.std(axis=1)
            live &= (d_sd >= 1e-8) & (minor_strains >= MIN_MINOR_STRAINS)
            if not live.any():
                continue

            keep = np.flatnonzero(live)
            d_mean = dosage[keep].mean(axis=1)
            d_std = (dosage[keep] - d_mean[:, None]) / d_sd[keep, None]
            h_rot_all = eigvecs.T @ d_std.T  # (n, k): one rotation per chunk
            pats = np.where(known[keep], np.rint(g0[keep]), -1).astype(int)

            for j, i in enumerate(keep):
                res = lmm_module.fit_lmm_at_locus(
                    y_rot, h_rot_all[:, [j]], x_rot, eigvals, h2
                )
                _pat = tuple(pats[j].tolist())
                results.append(
                    {
                        "gene_id": phenotype_id,
                        "snp": snp_ids[a + i],
                        "chr": chrom,
                        "pos_mb": snp_mb[a + i],
                        "lod": res.lod,
                        "p_value": res.p_value,
                        "f_stat": res.f_stat,
                        "h2": res.h2,
                        "n": n,
                        "rss0": res.rss_null,
                        "rss1": res.rss_alt,
                        "beta_dosage": float(res.betas[0]),
                        "se_dosage": float(res.se_betas[0]),
                        "dosage_mean": float(d_mean[j]),
                        "minor_strains": float(minor_strains[i]),
                        "_founder_pattern": _pat,
                    }
                )
                # best dosage per unique founder pattern — one vector per
                # pattern (≤2^8=256 for 8 binary founder positions), not one per
                # SNP. memory stays O(n_patterns × n_valid) regardless of window.
                if _pat not in _pat_best_dosage or res.lod > _pat_best_dosage[_pat][1]:
                    _pat_best_dosage[_pat] = (dosage[i].copy(), res.lod)

    if n_skipped_missing:
        logger.debug(
            "imputed_snp_scan_h2lmm: skipped %d SNPs with missing founder alleles "
            "carrying haplotype mass > %.2f",
            n_skipped_missing,
            MAX_MISSING_HAP_MASS,
        )

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
                           h2, n, rss0, rss1, beta_dosage, se_dosage
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
                "se_dosage",
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
            if calls[f] in ("N", "H", "", "nan"):
                geno_row[f] = np.nan
            else:
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
                "se_dosage",
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
                      f_stat, h2, n, rss0, rss1, beta_dosage, se_dosage
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
            "se_dosage",
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

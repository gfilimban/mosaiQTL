"""
scan — genome-wide haplotype QTL scan

Runs the haplotype-based QTL scan chromosome by chromosome for one
phenotype, tying together the genome cache, the kinship matrices, and
the underlying linear mixed model engine.

Two identifier spaces are in play:
    pheno_id : one identifier per row of phenotype data (n rows total)
    geno_id  : one identifier per row of the genome cache (m rows, m <= n)

Some designs have multiple phenotype replicates per genotyped strain
(n > m). In that case a mapping matrix (Z, shape n x m) links each
phenotype sample to its strain's genotype, and both phenotype and
covariates are averaged down to one value per strain (m,) before
scanning. Heritability and the scan itself then run at the strain
level (m x m kinship), never expanded back out to n.

When there's one phenotype row per genotyped strain (n == m) with
matching IDs, this mapping is just the identity and results match
the simple one-sample-per-strain case exactly.
"""

import gc
import time
import numpy as np
import pandas as pd
import logging
from pathlib import Path

from ..genotype import kinship as kin_module
from . import lmm as lmm_module
from ..utils import rint, build_z_matrix, chr_sort_key, _build_gm_idx, _enrich_scan_df

logger = logging.getLogger("mosaiqtl")

__all__ = [
    "ScanResult",
    "scan_h2lmm",
    "scan_multi_gene",
    "scan_multi_gene_lmm",
]


class ScanResult:
    """holds the results of a genome-wide (or per-chromosome) QTL scan.

    attributes
    ----------
    locus_ids    : np.ndarray (L,) str   marker names
    chromosomes  : np.ndarray (L,) str   chromosome of each marker
    positions    : np.ndarray (L,) float physical position in Mb
    lod          : np.ndarray (L,) float LOD score at each marker
    p_value      : np.ndarray (L,) float F-test p-value at each marker
    betas        : np.ndarray (L, J)     estimated effect of each founder haplotype
                                          (roughly sums to 0 per marker)
    h2_null      : dict {chrom: float}   estimated heritability under the null model, per chromosome
    phenotype_id : str
    """

    def __init__(self, **kwargs):
        """store all scan result fields, given as keyword arguments."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def peak_locus(self):
        """return the locus with the highest LOD score, as a dict of its stats."""
        idx = np.argmax(self.lod)
        return {
            "locus_id": self.locus_ids[idx],
            "chromosome": self.chromosomes[idx],
            "position": self.positions[idx],
            "lod": self.lod[idx],
            "p_value": self.p_value[idx],
            "betas": self.betas[idx],
        }

    def to_dataframe(self):
        """convert the scan result into a pandas DataFrame, one row per locus."""
        df = pd.DataFrame(
            {
                "locus_id": self.locus_ids,
                "chromosome": self.chromosomes,
                "position": self.positions,
                "lod": self.lod,
                "p_value": self.p_value,
            }
        )
        for fi in range(self.betas.shape[1]):
            df[f"beta_f{fi}"] = self.betas[:, fi]
        return df


def _resolve_z(sample_ids, genomecache, pheno_id=None, geno_id=None):
    """build the sample-to-strain mapping matrix for a given sample set.

    if sample_ids already lines up exactly with the cache's strain order
    (same count, same ids), returns None for the mapping matrix — this
    tells callers no expansion is needed, they can treat samples and
    strains as the same thing. otherwise builds the full mapping.

    parameters
    ----------
    sample_ids   : array-like  phenotype-level ids (n)
    genomecache : GenomeCache
    pheno_id     : str or None  column name in sample_ids (if a DataFrame)
    geno_id      : str or None  column name for genotype lookup (if a DataFrame)

    returns
    -------
    z_mat       : np.ndarray (n, m) or None
    valid_mask  : np.ndarray (n,) bool
    """
    cache_ids = list(genomecache.sample_ids)
    sid_list = list(sample_ids)
    if sid_list == cache_ids:
        return None, np.ones(len(sid_list), dtype=bool)
    z_mat, valid = build_z_matrix(sid_list, cache_ids)
    return z_mat, valid


def scan_h2lmm(
    phenotype,
    genomecache,
    covariates=None,
    kinship_decomp=None,
    sample_ids=None,
    use_rint=True,
    phenotype_id="unknown",
    fixed_h2_dict=None,
    model="additive",
    scan_window="cis",
    gene_chr=None,
    gene_pos_mb=None,
    window_mb=10.0,
):
    """haplotype-based genome-wide QTL scan using a linear mixed model with
    leave-one-chromosome-out (LOCO) kinship.

    handles both a simple one-sample-per-strain design (n == m) and a
    replicated design where multiple phenotype samples share one genotyped
    strain (n > m).

    replicated design: when sample_ids differs from genomecache.sample_ids,
    the sample-to-strain mapping is built automatically and phenotype and
    covariates are averaged to strain means before scanning. heritability
    and the scan itself then run entirely at the strain level.

    parameters
    ----------
    phenotype       : np.ndarray (n,)    phenotype values; NaN excluded
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    kinship_decomp  : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids      : array-like (n,) or None
        phenotype-level sample identifiers. if None, assumed to match
        genomecache.sample_ids in order (one sample per strain).
    use_rint        : bool  apply rank-inverse normal transformation
    phenotype_id    : str
    model           : str  'additive' (default) or 'full'
        'additive' — one column per founder haplotype at each locus (J founders).
        'full'     — one column per possible founder pair at each locus,
                     built from the additive probabilities by multiplying
                     each pair's probabilities together; valid for CC inbred
                     data, where truly heterozygous animals are essentially absent.
    scan_window     : str  'cis' | 'genome' | 'local'  (default 'cis')
        'cis'    — scan only the chromosome carrying the gene (gene_chr required).
        'genome' — scan all chromosomes present in kinship_decomp.
        'local'  — cis scan restricted to loci within ±window_mb of the
                   gene's transcription start site (gene_chr and gene_pos_mb required).
    gene_chr        : str or None  chromosome of the gene (required for cis/local)
    gene_pos_mb     : float or None  TSS position in Mb (required for local)
    window_mb       : float  half-window in Mb for local mode (default 10.0)

    returns
    -------
    ScanResult  (betas shape (L, J) for additive; (L, J*(J+1)/2) for full)
    """
    if scan_window in ("cis", "local") and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")
    if scan_window == "local" and gene_pos_mb is None:
        raise ValueError("scan_window='local' requires gene_pos_mb")
    n = len(phenotype)
    logger.info(
        "scan_h2lmm: gene=%s  chr=%s  scan_window=%s  n_samples=%d  use_rint=%s",
        phenotype_id,
        gene_chr if gene_chr is not None else "genome",
        scan_window,
        n,
        use_rint,
    )
    if kinship_decomp is None:
        logger.warning(
            "scan_h2lmm: no kinship_decomp supplied for %s — computing from genomecache "
            "(slow; pass pre-computed kinship_decomp for multi-gene scans)",
            phenotype_id,
        )

    # build the sample-to-strain mapping
    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    z_mat, valid_sid = _resolve_z(sid, genomecache)

    # a sample is usable only if it has both a phenotype value and a genotype match
    valid_pheno = ~np.isnan(phenotype)
    valid = valid_pheno & valid_sid
    if valid.sum() < n:
        logger.info(
            "scan_h2lmm: %s  n_excluded=%d  (nan_pheno=%d  missing_geno=%d)",
            phenotype_id,
            n - valid.sum(),
            (~valid_pheno).sum(),
            (~valid_sid).sum(),
        )
    n_valid = int(valid.sum())
    if n_valid < 10:
        logger.warning(
            "scan_h2lmm: %s has only %d valid samples — LOD estimates unreliable",
            phenotype_id,
            n_valid,
        )

    y = phenotype[valid].copy()

    # covariates (add an intercept column if the caller didn't include one)
    if covariates is not None:
        x = covariates[valid].copy()
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((len(y), 1)), x])
    else:
        x = np.ones((len(y), 1))

    # keep only the mapping rows for samples that passed the validity check
    z_sub = z_mat[valid] if z_mat is not None else None

    # normalise the phenotype at the individual-sample level, before any
    # averaging down to strains — this keeps LOD scores on the same scale
    # as a per-sample scan, since the kinship is later expanded back out
    # to sample level (K_n = Z K_m Z^T) in the loop below.
    if use_rint:
        y = rint(y)

    # compute kinship if the caller didn't supply one already
    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []
    h2_null_dict = {}

    # only scan chromosomes we actually have kinship for — avoids noisy skip
    # messages when kinship_decomp was built for a single chromosome
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ("cis", "local"):
        _gene_chr = str(gene_chr)
        _chrom_list = [c for c in _chrom_list if c == _gene_chr]
    for chrom in _chrom_list:

        _ev = kinship_decomp[chrom]["values"]
        _evc = kinship_decomp[chrom]["vectors"]

        if z_sub is not None:
            # expand the strain-level kinship out to sample level and
            # re-decompose; this is a small matrix (n ≤ 416) so the cost
            # is negligible, and it's shared across every locus on this chromosome.
            _K_m = (_evc * _ev) @ _evc.T
            _K_n = z_sub @ _K_m @ z_sub.T
            evals, evecs = np.linalg.eigh(_K_n)
            del _K_m, _K_n
        elif _evc.shape[0] > len(y):
            # this path should only be reached if the caller passed a phenotype
            # with missing values while every sample otherwise matches the
            # genotype cache directly (no mapping matrix needed). rebuild the
            # kinship for just the valid samples and re-decompose it, rather
            # than naively dropping rows from the eigenvectors (which would not
            # give a valid eigendecomposition of the smaller matrix).
            logger.warning(
                "scan_h2lmm: %s — phenotype has %d NaN values; "
                "caller should pass a clean phenotype vector",
                phenotype_id,
                int((~valid).sum()),
            )
            _K_full = (_evc * _ev) @ _evc.T
            _K_sub = _K_full[np.ix_(valid, valid)]
            evals, evecs = np.linalg.eigh(_K_sub)
            del _K_full, _K_sub
        else:
            evals, evecs = _ev, _evc

        y_rot = evecs.T @ y
        x_rot = evecs.T @ x
        # reuse a pre-estimated heritability if one was supplied (e.g. from scan_h2lmm_mi)
        if fixed_h2_dict is not None and chrom in fixed_h2_dict:
            h2 = fixed_h2_dict[chrom]
        else:
            h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, evals, method="ml")
        h2_null_dict[chrom] = h2

        cd = lmm_module.precompute_chromosome(
            genomecache, chrom, evals, evecs, z_sub, model
        )
        chr_res = lmm_module.scan_chromosome(y, cd, x, evals, evecs, h2)

        if scan_window == "local":
            mask = np.abs(chr_res["positions"] - gene_pos_mb) <= window_mb
            if not mask.any():
                logger.debug(
                    "scan_h2lmm: %s  no loci within %.1fMb of %.3f on chr%s — skipped",
                    phenotype_id,
                    window_mb,
                    gene_pos_mb,
                    chrom,
                )
                continue
            chr_res = {
                k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
                for k, v in chr_res.items()
            }

        n_loci = len(chr_res["locus_ids"])
        all_lids.append(chr_res["locus_ids"])
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(chr_res["positions"])
        all_lod.append(chr_res["lod"])
        all_pval.append(chr_res["p_value"])
        all_betas.append(chr_res["betas"])

    if not all_lod:
        logger.warning(
            "scan_h2lmm: %s — no loci matched scan window; returning empty",
            phenotype_id,
        )
        return ScanResult(
            locus_ids=np.array([], dtype=str),
            chromosomes=np.array([], dtype=str),
            positions=np.array([], dtype=float),
            lod=np.array([], dtype=float),
            p_value=np.array([], dtype=float),
            betas=np.zeros((0, x.shape[1] if x.ndim > 1 else 1), dtype=float),
            peak_lod=0.0,
            peak_locus_id="",
            peak_position_mb=np.nan,
            peak_chr="",
            model="lmm",
        )
    _lod_cat = np.concatenate(all_lod)
    _peak_idx = int(np.argmax(_lod_cat))
    _peak_lids = np.concatenate(all_lids)
    logger.info(
        "scan_h2lmm: %s done — peak_lod=%.3f  peak_locus=%s",
        phenotype_id,
        float(_lod_cat[_peak_idx]),
        _peak_lids[_peak_idx],
    )
    return ScanResult(
        locus_ids=_peak_lids,
        chromosomes=np.concatenate(all_chrs),
        positions=np.concatenate(all_pos),
        lod=_lod_cat,
        p_value=np.concatenate(all_pval),
        betas=np.vstack(all_betas),
        h2_null=h2_null_dict,
        phenotype_id=phenotype_id,
        model=model,
    )


def scan_multi_gene(
    phenotype_matrix: np.ndarray,
    gene_ids: "list[str]",
    genomecache: "GenomeCache",  # type: ignore[name-defined]  # noqa: F821
    gene_chr_map: "dict[str, str] | None" = None,
    covariates: "np.ndarray | None" = None,
    kinship_decomp: "dict | None" = None,
    sample_ids: "list[str] | None" = None,
    use_rint: bool = True,
    scan_window: str = "genome",
    gene_pos_mb_map: "dict[str, float] | None" = None,
    window_mb: float = 10.0,
    output_dir: "Path | None" = None,
    model: str = "additive",
) -> "dict[str, ScanResult]":
    """scan many genes at once, sharing the same genome cache and kinship.

    the chromosome is the outer loop and genes are the inner loop: the
    per-chromosome setup (haplotype probability rotation, kinship
    eigendecomposition) is done once and reused for every gene on that
    chromosome, rather than repeating it per gene.

    when output_dir is set, a per-chromosome LOD file (lod_chr{ch}.csv) is
    written as soon as that chromosome finishes. this keeps peak memory
    bounded to one chromosome's worth of results rather than the whole genome.

    phenotype_matrix must not contain NaN values for any sample that is
    present in the genome cache. genes with a near-constant (degenerate)
    phenotype are handled gracefully by falling back to a safe default
    heritability estimate rather than failing.

    parameters
    ----------
    phenotype_matrix : np.ndarray (n, G)   columns = genes; no NaN
    gene_ids         : list[str]            length G
    genomecache      : GenomeCache
    gene_chr_map     : dict[str, str] | None
        maps gene_id → chromosome string.  required for scan_window='cis'
        and scan_window='local'.  for 'genome', pass None.
    covariates       : np.ndarray (n, p) or None
    kinship_decomp   : dict | None   {chrom: {'values':, 'vectors':}}
    sample_ids       : list[str] | None   phenotype-level sample ids
    use_rint         : bool   apply rank-inverse normal transformation
    scan_window      : str   'genome' | 'cis' | 'local'  (default 'genome')
    gene_pos_mb_map  : dict[str, float] | None   TSS in Mb; required for 'local'
    window_mb        : float   half-window for local mode (default 10.0)
    output_dir       : Path | None   write lod_chr{ch}.csv per chromosome
    model            : str   'additive' (default) or 'full'

    returns
    -------
    dict[str, ScanResult]   one entry per gene with loci from the scanned
    chromosomes.  empty when no loci pass the scan_window filter.
    """
    if scan_window in ("cis", "local") and gene_chr_map is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr_map")
    if scan_window == "local" and gene_pos_mb_map is None:
        raise ValueError("scan_window='local' requires gene_pos_mb_map")

    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # ── shared sample alignment ──────────────────────────────────────────────
    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    z_mat, valid_sid = _resolve_z(sid, genomecache)
    if not valid_sid.all():
        logger.info(
            "scan_multi_gene: %d samples excluded (missing geno)", (~valid_sid).sum()
        )

    y_mat = phenotype_matrix[valid_sid].astype(float)  # (n_valid, G)
    z_sub = z_mat[valid_sid] if z_mat is not None else None
    x = (
        covariates[valid_sid].copy()
        if covariates is not None
        else np.ones((y_mat.shape[0], 1))
    )
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((len(x), 1)), x])

    if use_rint:
        for gi in range(y_mat.shape[1]):
            y_mat[:, gi] = rint(y_mat[:, gi])

    gene_index = {gid: gi for gi, gid in enumerate(gene_ids)}

    # ── chromosome list ──────────────────────────────────────────────────────
    chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ("cis", "local"):
        chrs_needed = {  # type: ignore[union-attr]
            str(gene_chr_map.get(gid, "")) for gid in gene_ids
        }
        chrom_list = [c for c in chrom_list if str(c) in chrs_needed]

    # ── per-gene result accumulators ─────────────────────────────────────────
    acc: "dict[str, dict]" = {
        gid: {
            "lids": [],
            "chrs": [],
            "pos": [],
            "lod": [],
            "pval": [],
            "betas": [],
            "h2": {},
        }
        for gid in gene_ids
    }

    t_total = time.perf_counter()
    n_chrs = len(chrom_list)

    for ci, chrom in enumerate(chrom_list, 1):
        t_chr = time.perf_counter()
        ev = kinship_decomp[chrom]["values"]
        evc = kinship_decomp[chrom]["vectors"]

        # expand the strain-level kinship to sample level, once per chromosome
        if z_sub is not None:
            _K_m = (evc * ev) @ evc.T
            _K_n = z_sub @ _K_m @ z_sub.T
            evals, evecs = np.linalg.eigh(_K_n)
            del _K_m, _K_n
        else:
            evals, evecs = ev, evc

        # this per-chromosome setup is shared by every gene on this chromosome
        chr_data = lmm_module.precompute_chromosome(
            genomecache, chrom, evals, evecs, z_sub, model
        )

        # select genes assigned to this chromosome
        if scan_window in ("cis", "local") and gene_chr_map is not None:
            genes_this_chr = [
                gid for gid in gene_ids if str(gene_chr_map.get(gid, "")) == str(chrom)
            ]
        else:
            genes_this_chr = gene_ids

        # buffers for this chromosome's rows, if streaming to disk
        if output_dir is not None:
            _chr_gene: list = []
            _chr_locus: list = []
            _chr_pos: list = []
            _chr_lod: list = []

        for gid in genes_this_chr:
            gi = gene_index[gid]
            y_g = y_mat[:, gi]
            y_rot = evecs.T @ y_g
            x_rot = evecs.T @ x
            h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, evals, method="ml")
            acc[gid]["h2"][chrom] = h2

            chr_res = lmm_module.scan_chromosome(y_g, chr_data, x, evals, evecs, h2)

            if scan_window == "local":
                tss = float(gene_pos_mb_map[gid])  # type: ignore[index]
                mask = np.abs(chr_res["positions"] - tss) <= window_mb
                if not mask.any():
                    logger.debug(
                        "scan_multi_gene: %s  no loci within %.1fMb on chr%s",
                        gid,
                        window_mb,
                        chrom,
                    )
                    continue
                chr_res = {
                    k: (
                        v[mask]
                        if isinstance(v, np.ndarray) and len(v) == len(mask)
                        else v
                    )
                    for k, v in chr_res.items()
                }

            n_loci_g = len(chr_res["locus_ids"])
            acc[gid]["lids"].append(chr_res["locus_ids"])
            acc[gid]["chrs"].append(np.full(n_loci_g, chrom))
            acc[gid]["pos"].append(chr_res["positions"])
            acc[gid]["lod"].append(chr_res["lod"])
            acc[gid]["pval"].append(chr_res["p_value"])
            acc[gid]["betas"].append(chr_res["betas"])

            if output_dir is not None:
                _chr_gene.extend([gid] * n_loci_g)
                _chr_locus.extend(chr_res["locus_ids"])
                _chr_pos.extend(chr_res["positions"].tolist())
                _chr_lod.extend(chr_res["lod"].tolist())

        # write per-chr CSV and release row buffers immediately
        if output_dir is not None and _chr_gene:
            _lod_path = Path(output_dir) / f"lod_chr{chrom}.csv"
            pd.DataFrame(
                {
                    "gene_id": _chr_gene,
                    "chr": chrom,
                    "locus": _chr_locus,
                    "pos_mb": _chr_pos,
                    "lod": _chr_lod,
                }
            ).to_csv(_lod_path, index=False)
            logger.info(
                "scan_multi_gene: chr%s (%d/%d)  n_genes=%d  n_rows=%d  elapsed=%.1fs → %s",
                chrom,
                ci,
                n_chrs,
                len(genes_this_chr),
                len(_chr_gene),
                time.perf_counter() - t_chr,
                _lod_path.name,
            )
            del _chr_gene, _chr_locus, _chr_pos, _chr_lod
        else:
            logger.info(
                "scan_multi_gene: chr%s (%d/%d)  n_genes=%d  elapsed=%.1fs",
                chrom,
                ci,
                n_chrs,
                len(genes_this_chr),
                time.perf_counter() - t_chr,
            )

        del chr_data
        gc.collect()

    logger.info(
        "scan_multi_gene: %d genes  %d chromosomes  %.1fs total",
        len(gene_ids),
        n_chrs,
        time.perf_counter() - t_total,
    )

    # ── assemble ScanResult per gene ─────────────────────────────────────────
    results: "dict[str, ScanResult]" = {}
    for gid in gene_ids:
        a = acc[gid]
        if not a["lids"]:
            continue
        results[gid] = ScanResult(
            locus_ids=np.concatenate(a["lids"]),
            chromosomes=np.concatenate(a["chrs"]),
            positions=np.concatenate(a["pos"]),
            lod=np.concatenate(a["lod"]),
            p_value=np.concatenate(a["pval"]),
            betas=np.vstack(a["betas"]),
            h2_null=a["h2"],
            phenotype_id=gid,
            model=model,
        )
    return results


def scan_multi_gene_lmm(
    expr_cache: "dict[str, np.ndarray]",
    gene_chr_map: "dict[str, str]",
    genomecache: "GenomeCache",  # type: ignore[name-defined]  # noqa: F821
    covariates: "np.ndarray | None" = None,
    kinship_decomp: "dict | None" = None,
    sample_ids: "list[str] | None" = None,
    use_rint: bool = True,
    scan_window: str = "cis",
    gene_pos_mb_map: "dict[str, float] | None" = None,
    window_mb: float = 10.0,
    lod_path: "Path | None" = None,
    gene_meta: "pd.DataFrame | None" = None,
    founder_names: "list[str] | None" = None,
    model: str = "additive",
) -> pd.DataFrame:
    """streaming LMM scan over a set of genes; returns a peak-statistics DataFrame.

    same chromosome-outer, gene-inner strategy as scan_multi_gene, but takes
    a dict-based expression cache (matching scan_multi_gene_mi and
    instability_scan_multi_gene) and also records each gene's residual sum
    of squares before and after the haplotype term, for diagnostic use.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map   : dict {gene_id: str}  chromosome per gene; required for 'cis'/'local'
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None — intercept prepended automatically if absent
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids     : list[str] or None  genotype-level strain IDs matching genomecache
    use_rint       : bool
    scan_window    : str  'cis' (default) | 'genome' | 'local'
    gene_pos_mb_map: dict {gene_id: float} or None — TSS in Mb; required for 'local'
    window_mb      : float  half-window in Mb for 'local' (default 10.0)
    lod_path       : Path or None  append full per-locus rows here (gene, chr, locus, pos_mb, lod)
    gene_meta      : DataFrame keyed by 'gene_id' (the identity key; carries
                     gene_symbol and any other display columns) — extra
                     columns joined into output rows
    founder_names  : list[str] or None  display names for beta columns; None → genomecache.founders
    model          : str  'additive' (default) or 'full'

    returns
    -------
    DataFrame with one row per gene:
        gene_id, chr, peak_locus, peak_mb, peak_lod, h2, n_valid,
        rss0, rss1, r2_peak, sigma2_e, tau2, beta_{founder_names}
        plus any columns from gene_meta.
    """
    if scan_window in ("cis", "local") and not gene_chr_map:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr_map")
    if scan_window == "local" and gene_pos_mb_map is None:
        raise ValueError("scan_window='local' requires gene_pos_mb_map")

    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    z_mat, valid_sid = _resolve_z(sid, genomecache)

    x = (
        covariates[valid_sid].copy()
        if covariates is not None
        else np.ones((int(valid_sid.sum()), 1))
    )
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((len(x), 1)), x])

    n_valid = int(valid_sid.sum())
    p_cov = x.shape[1]
    z_sub = z_mat[valid_sid] if z_mat is not None else None

    _fcodes = list(genomecache.founders)
    if founder_names is not None:
        _fnames = list(founder_names)
    else:
        try:
            from ..genotype.founders import FOUNDER_CODE  # noqa: PLC0415

            _fc_inv = {v: k for k, v in FOUNDER_CODE.items()}
            _fnames = [_fc_inv.get(f, f) for f in _fcodes]
        except Exception:
            _fnames = _fcodes
    beta_cols = [f"beta_{fn}" for fn in _fnames]
    n_f = len(beta_cols)

    gene_ids = list(expr_cache.keys())
    chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ("cis", "local"):
        chrs_needed = {str(gene_chr_map.get(g, "")) for g in gene_ids}
        chrom_list = [c for c in chrom_list if str(c) in chrs_needed]

    peak_rows: "list[dict]" = []
    _lod_first = True
    t_total = time.perf_counter()

    for ci, chrom in enumerate(chrom_list, 1):
        t_chr = time.perf_counter()
        ev = kinship_decomp[chrom]["values"]
        evc = kinship_decomp[chrom]["vectors"]

        if z_sub is not None:
            _K_m = (evc * ev) @ evc.T
            _K_n = z_sub @ _K_m @ z_sub.T
            evals, evecs = np.linalg.eigh(_K_n)
            del _K_m, _K_n
        else:
            evals, evecs = ev, evc

        chr_data = lmm_module.precompute_chromosome(
            genomecache, chrom, evals, evecs, z_sub, model
        )

        if scan_window in ("cis", "local"):
            genes_this_chr = [
                g for g in gene_ids if str(gene_chr_map.get(g, "")) == str(chrom)
            ]
        else:
            genes_this_chr = gene_ids

        _lod_rows: "list[dict]" = []

        for gene_id in genes_this_chr:
            y_raw = expr_cache[gene_id]
            y_g = y_raw[valid_sid].astype(float)
            if use_rint:
                y_g = rint(y_g)

            y_rot = evecs.T @ y_g
            x_rot = evecs.T @ x
            h2, _ = lmm_module.estimate_h2_null(y_rot, x_rot, evals, method="ml")

            try:
                chr_res = lmm_module.scan_chromosome(y_g, chr_data, x, evals, evecs, h2)
            except Exception as exc:
                logger.warning(
                    "scan_multi_gene_lmm: %s on chr%s failed — %s",
                    gene_id,
                    chrom,
                    exc,
                )
                continue

            lod_arr = chr_res["lod"]
            pos_arr = chr_res["positions"]
            lid_arr = np.asarray(chr_res["locus_ids"])
            rss0 = chr_res.get("rss0", float("nan"))
            rss1_arr = chr_res.get("rss1", None)
            betas_all = chr_res.get("betas", None)

            if scan_window == "local" and gene_pos_mb_map is not None:
                tss = float(gene_pos_mb_map[gene_id])
                mask = np.abs(pos_arr - tss) <= window_mb
                if not mask.any():
                    logger.debug(
                        "scan_multi_gene_lmm: %s  no loci within %.1fMb on chr%s",
                        gene_id,
                        window_mb,
                        chrom,
                    )
                    continue
                lod_arr = lod_arr[mask]
                pos_arr = pos_arr[mask]
                lid_arr = lid_arr[mask]
                if rss1_arr is not None:
                    rss1_arr = rss1_arr[mask]
                if betas_all is not None:
                    betas_all = betas_all[mask]

            if len(lod_arr) == 0:
                continue

            peak_idx = int(np.argmax(lod_arr))
            rss1_peak = (
                float(rss1_arr[peak_idx]) if rss1_arr is not None else float("nan")
            )
            betas_pk = (
                betas_all[peak_idx]
                if betas_all is not None
                else np.full(n_f, float("nan"))
            )

            _ok = (
                n_valid > p_cov
                and np.isfinite(rss0)
                and rss0 > 0
                and np.isfinite(rss1_peak)
            )
            _df = max(n_valid - p_cov, 1)
            _h2f = float(h2) if np.isfinite(h2) else 0.0

            row: dict = {
                "gene_id": gene_id,
                "chr": str(chrom),
                "peak_locus": str(lid_arr[peak_idx]),
                "peak_mb": float(pos_arr[peak_idx]),
                "peak_lod": float(lod_arr[peak_idx]),
                "h2": float(h2),
                "n_valid": n_valid,
                "rss0": float(rss0),
                "rss1": rss1_peak,
                "r2_peak": round((rss0 - rss1_peak) / rss0, 6) if _ok else float("nan"),
                "sigma2_e": round(rss0 * (1 - _h2f) / _df, 6) if _ok else float("nan"),
                "tau2": round(rss0 * _h2f / _df, 6) if _ok else float("nan"),
            }
            for fi, bcol in enumerate(beta_cols):
                row[bcol] = float(betas_pk[fi]) if fi < len(betas_pk) else float("nan")
            peak_rows.append(row)

            if lod_path is not None:
                for _i in range(len(lod_arr)):
                    _lod_rows.append(
                        {
                            "gene_id": gene_id,
                            "chr": str(chrom),
                            "locus": str(lid_arr[_i]),
                            "pos_mb": float(pos_arr[_i]),
                            "lod": float(lod_arr[_i]),
                        }
                    )

        if lod_path is not None and _lod_rows:
            pd.DataFrame(_lod_rows).to_csv(
                lod_path, mode="a", header=_lod_first, index=False
            )
            _lod_first = False
        del _lod_rows

        logger.info(
            "scan_multi_gene_lmm: chr%s (%d)  n_genes=%d  elapsed=%.1fs",
            chrom,
            ci,
            len(genes_this_chr),
            time.perf_counter() - t_chr,
        )
        del chr_data
        gc.collect()

    logger.info(
        "scan_multi_gene_lmm: %d genes  %d chromosomes  %.1fs total",
        len(gene_ids),
        len(chrom_list),
        time.perf_counter() - t_total,
    )

    result = pd.DataFrame(peak_rows) if peak_rows else pd.DataFrame()
    if gene_meta is not None and not result.empty and not gene_meta.empty:
        _gm = gene_meta.drop_duplicates(subset="gene_id")
        result = result.merge(_gm, on="gene_id", how="left", suffixes=("", "_meta"))
    return result

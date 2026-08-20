"""
scan — genome-wide haplotype QTL scan

orchestrates the scan across chromosomes for a single phenotype,
combining genome cache, kinship, and the LMM engine.

global design — pheno_id / geno_id split:
    pheno_id : identifiers for each row of phenotype data (n)
    geno_id  : identifiers for rows of the genome cache (m ≤ n)

when n > m (replicated CC strains), a Z matrix (n×m) maps each
phenotype sample to its strain genotype. phenotype and covariates
are reduced to strain means (m,) before scanning. h² and the scan
both operate at strain level (m×m K_m eigendecomp); H is never
expanded to n.

when n == m and ids match, Z == I and all results are identical
to the 1:1 case.

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
    "scan_h2lmm_mi",
    "scan_multi_gene",
    "scan_multi_gene_lmm",
    "scan_multi_gene_mi",
    "instability_scan_multi_gene",
    "grab_locus_from_scan",
    "lod_drop_ci",
    "find_peaks",
    "fixef_eqtl_scan",
    "instability_lm_scan",
]


class ScanResult:
    """container for genome-wide scan output.

    attributes
    ----------
    locus_ids    : np.ndarray (L,) str   marker identifiers
    chromosomes  : np.ndarray (L,) str   chromosome per locus
    positions    : np.ndarray (L,) float physical position (Mb)
    lod          : np.ndarray (L,) float LOD scores
    p_value      : np.ndarray (L,) float F-test p-values
    betas        : np.ndarray (L, J)     founder haplotype effects (sum ≈ 0)
    h2_null      : dict {chrom: float}   null h² per LOCO chr
    phenotype_id : str
    """

    def __init__(self, **kwargs):
        """Set all scan result fields from keyword arguments."""
        for k, v in kwargs.items():
            setattr(self, k, v)

    def peak_locus(self):
        """return dict with peak locus info."""
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
        """convert to pandas DataFrame."""
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
    """build (Z, valid_mask, n_eff) for a given sample set.

    if sample_ids already matches the cache order exactly (n == m, same ids),
    returns Z=None (signals identity to callers — no expansion needed).
    otherwise builds the full Z matrix.

    parameters
    ----------
    sample_ids   : array-like  phenotype-level ids (n)
    genomecache : GenomeCache
    pheno_id     : str or None  column name in sample_ids (if DataFrame)
    geno_id      : str or None  column name for geno lookup (if DataFrame)

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
    """haplotype-based genome-wide scan with LOCO LMM.

    supports both 1:1 (n == m) and replicated (n > m) designs.

    replicated design: when sample_ids differs from genomecache.sample_ids,
    Z is built automatically and phenotype/covariates are reduced to strain
    means before scanning. h² is estimated at strain level (m×m K_m
    eigendecomp) and the scan operates entirely in the m-dimensional strain
    space.

    parameters
    ----------
    phenotype       : np.ndarray (n,)    phenotype values; NaN excluded
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    kinship_decomp  : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids      : array-like (n,) or None
        phenotype-level sample identifiers. if None, assumed to match
        genomecache.sample_ids in order (1:1 case).
    use_rint        : bool  apply rank-inverse normal transformation
    phenotype_id    : str
    model           : str  'additive' (default) or 'full'
        'additive' — 8 haplotype columns per locus (J founders).
        'full'     — J*(J+1)/2 diplotype columns constructed from additive
                     probs via outer product P(fi/fj)=P(fi)*P(fj); valid
                     for CC inbred data where heterozygous probs ≈ 0.
    scan_window     : str  'cis' | 'genome' | 'local'  (default 'cis')
        'cis'    — scan only the chromosome carrying the gene (gene_chr
                   required); equivalent to passing a single-chr kinship dict.
        'genome' — scan all chromosomes present in kinship_decomp.
        'local'  — cis scan restricted to loci within ±window_mb of the
                   gene TSS (gene_chr and gene_pos_mb required).
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

    # resolve Z mapping
    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    z_mat, valid_sid = _resolve_z(sid, genomecache)

    # combine sample validity with phenotype NaN mask
    valid_pheno = ~np.isnan(phenotype)
    valid = valid_pheno & valid_sid
    if valid.sum() < n:
        logger.info(
            f"{phenotype_id}: {n - valid.sum()} samples excluded "
            f"({(~valid_pheno).sum()} NaN pheno, "
            f"{(~valid_sid).sum()} missing geno)"
        )
    n_valid = int(valid.sum())
    if n_valid < 10:
        logger.warning(
            "scan_h2lmm: %s has only %d valid samples — LOD estimates unreliable",
            phenotype_id,
            n_valid,
        )

    y = phenotype[valid].copy()

    # covariates
    if covariates is not None:
        x = covariates[valid].copy()
        if not np.allclose(x[:, 0], 1.0):
            x = np.hstack([np.ones((len(y), 1)), x])
    else:
        x = np.ones((len(y), 1))

    # Z matrix for this valid subset
    z_sub = z_mat[valid] if z_mat is not None else None

    # RINT at sample level — matches miQTL's canonical replicated design: scan.h2lmm
    # applies RINT to all n samples before any reduction. z_sub is retained and passed
    # to scan_chromosome, which expands H from m→n via Z and fuses with eigvec rotation.
    # The kinship is expanded to K_n = Z K_m Z^T per chromosome in the loop below,
    # giving LODs on the same n-sample scale as miQTL's sample-level scan.
    if use_rint:
        y = rint(y)

    # kinship
    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []
    h2_null_dict = {}

    # restrict to chromosomes with kinship available; avoids noisy skip messages
    # when kinship_decomp is built for a single chromosome (e.g. --chr 8)
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ("cis", "local"):
        _gene_chr = str(gene_chr)
        _chrom_list = [c for c in _chrom_list if c == _gene_chr]
    for chrom in _chrom_list:

        _ev = kinship_decomp[chrom]["values"]
        _evc = kinship_decomp[chrom]["vectors"]

        if z_sub is not None:
            # expand LOCO kinship to sample level: K_n = Z K_m Z^T.
            # eigendecomp is O(n³) but n≤416 — negligible; shared across all loci.
            _K_m = (_evc * _ev) @ _evc.T  # broadcast avoids np.diag temp
            _K_n = z_sub @ _K_m @ z_sub.T
            evals, evecs = np.linalg.eigh(_K_n)
            del _K_m, _K_n
        elif _evc.shape[0] > len(y):
            # Defensive guard: in the z_sub=None path, valid_sid is all-True
            # (caller contract: all phenotype sample IDs are present in the
            # genomecache). This branch is therefore only reachable if the caller
            # passes a phenotype with NaN values, which violates the contract.
            # If triggered: the original code did `_evc[valid]` (row-subsetting
            # eigenvectors), which is mathematically invalid — a row-subset of
            # an eigenvector matrix is not the eigendecomp of the submatrix.
            # Correct path: reconstruct K_sub and re-decompose.
            logger.warning(
                'scan_h2lmm: %s — phenotype has %d NaN values; '
                'caller should pass a clean phenotype vector',
                phenotype_id, int((~valid).sum()),
            )
            _K_full = (_evc * _ev) @ _evc.T
            _K_sub = _K_full[np.ix_(valid, valid)]
            evals, evecs = np.linalg.eigh(_K_sub)
            del _K_full, _K_sub
        else:
            evals, evecs = _ev, _evc

        y_rot = evecs.T @ y
        x_rot = evecs.T @ x
        # use fixed h2 when provided (e.g. from scan_h2lmm_mi)
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
                    f"{phenotype_id}: no loci within {window_mb} Mb of "
                    f"{gene_pos_mb:.3f} on {chrom}; skipping"
                )
                continue
            chr_res = {
                k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
                for k, v in chr_res.items()
            }
            chr_res["locus_ids"] = np.asarray(chr_res["locus_ids"])[mask]

        n_loci = len(chr_res["locus_ids"])
        all_lids.append(chr_res["locus_ids"])
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(chr_res["positions"])
        all_lod.append(chr_res["lod"])
        all_pval.append(chr_res["p_value"])
        all_betas.append(chr_res["betas"])

    if not all_lod:
        logger.warning(
            'scan_h2lmm: %s — no loci matched scan window; returning empty', phenotype_id
        )
        return ScanResult(
            locus_ids=np.array([], dtype=str),
            chromosomes=np.array([], dtype=str),
            positions=np.array([], dtype=float),
            lod=np.array([], dtype=float),
            p_value=np.array([], dtype=float),
            betas=np.zeros((0, x.shape[1] if x.ndim > 1 else 1), dtype=float),
            peak_lod=0.0,
            peak_locus_id='',
            peak_position_mb=np.nan,
            peak_chr='',
            model='lmm',
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
    """scan multiple genes sharing the same genome cache and kinship.

    chromosome-outer, gene-inner loop: precompute_chromosome is called once
    per chromosome for all genes, amortising the O(n_loci·n·m·J) rotation
    across G genes.  K_n eigendecomposition is also computed once per
    chromosome.

    when output_dir is set, a per-chromosome LOD CSV (lod_chr{ch}.csv) is
    written immediately after each chromosome completes.  this bounds peak
    memory to one chromosome's worth of LOD rows rather than the full genome.

    phenotype_matrix must not contain NaN for samples that pass the common
    validity mask (sample present in genomecache).  genes with degenerate
    phenotypes are handled gracefully by estimate_h2_null's sentinel return.

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

        # expand LOCO kinship to sample level K_n = Z K_m Z^T (once per chr)
        if z_sub is not None:
            _K_m = (evc * ev) @ evc.T
            _K_n = z_sub @ _K_m @ z_sub.T
            evals, evecs = np.linalg.eigh(_K_n)
            del _K_m, _K_n
        else:
            evals, evecs = ev, evc

        # chromosome-constant H rotation — amortised across all genes on chr
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

        # per-chr streaming accumulators (arrays — no per-row dict overhead)
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
                        "%s: no loci within %.1f Mb on %s", gid, window_mb, chrom
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
                chr_res["locus_ids"] = np.asarray(chr_res["locus_ids"])[mask]

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
                    "gene_symbol": _chr_gene,
                    "chr": chrom,
                    "locus": _chr_locus,
                    "pos_mb": _chr_pos,
                    "lod": _chr_lod,
                }
            ).to_csv(_lod_path, index=False)
            logger.info(
                "[%d/%d] chr%s: %d genes  %d rows  %.1fs → %s",
                ci,
                n_chrs,
                chrom,
                len(genes_this_chr),
                len(_chr_gene),
                time.perf_counter() - t_chr,
                _lod_path.name,
            )
            del _chr_gene, _chr_locus, _chr_pos, _chr_lod
        else:
            logger.info(
                "[%d/%d] chr%s: %d genes  %.1fs",
                ci,
                n_chrs,
                chrom,
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


def grab_locus_from_scan(scan_result):
    """return the locus id with the highest LOD score."""
    return scan_result.locus_ids[np.argmax(scan_result.lod)]


def lod_drop_ci(scan_result, chrom, drop=1.5):
    """1.5-LOD-drop confidence interval on a single chromosome.

    returns (lo_mb, hi_mb, peak_mb).
    """
    mask = scan_result.chromosomes == chrom
    pos = scan_result.positions[mask]
    lod = scan_result.lod[mask]
    if len(lod) == 0:
        return np.nan, np.nan, np.nan
    pi = np.argmax(lod)
    thresh = lod[pi] - drop
    above = np.where(lod >= thresh)[0]
    return float(pos[above[0]]), float(pos[above[-1]]), float(pos[pi])


def find_peaks(scan_result, threshold, drop=1.5, allow_multiple=False, min_lod_gap=1.5):
    """find significant QTL peaks in a genome-wide scan.

    mirrors qtl2::find_peaks(scanone, map, threshold, drop, peakdrop).

    parameters
    ----------
    scan_result   : ScanResult
    threshold     : float  minimum LOD to report a peak
    drop          : float  LOD drop for CI bounds (default 1.5)
    allow_multiple: bool   if True, report secondary peaks per chromosome
        that are above threshold AND at least min_lod_gap LOD below the
        primary peak (i.e. separated local maxima).  default False returns
        only the highest-LOD peak per chromosome.
    min_lod_gap   : float  minimum LOD separation between multiple peaks
        on the same chromosome (only used when allow_multiple=True)

    returns
    -------
    pd.DataFrame  columns: phenotype_id, chr, locus_id, pos_mb, lod,
                            ci_lo_mb, ci_hi_mb
        one row per peak, sorted by chr then pos_mb.
        empty DataFrame (same columns) when no locus exceeds threshold.
    """
    _COLS = ("phenotype_id", "chr", "locus_id", "pos_mb", "lod", "ci_lo_mb", "ci_hi_mb")
    rows = []

    for chrom in sorted(np.unique(scan_result.chromosomes), key=chr_sort_key):
        mask = scan_result.chromosomes == chrom
        lod_c = scan_result.lod[mask]
        pos_c = scan_result.positions[mask]
        lids_c = scan_result.locus_ids[mask]

        if lod_c.max() < threshold:
            continue

        if not allow_multiple:
            # single best peak per chromosome
            pi = int(np.argmax(lod_c))
            lo, hi, pk = lod_drop_ci(scan_result, chrom, drop)
            rows.append(
                (
                    scan_result.phenotype_id,
                    chrom,
                    lids_c[pi],
                    float(pos_c[pi]),
                    float(lod_c[pi]),
                    lo,
                    hi,
                )
            )
        else:
            # all local maxima above threshold separated by >= min_lod_gap
            # a point is a local max if it is >= both neighbours
            n = len(lod_c)
            for i in range(n):
                if lod_c[i] < threshold:
                    continue
                left_ok = (i == 0) or (lod_c[i] >= lod_c[i - 1])
                right_ok = (i == n - 1) or (lod_c[i] >= lod_c[i + 1])
                if not (left_ok and right_ok):
                    continue
                # check separation from any already-accepted peak on this chr
                too_close = any(
                    abs(pos_c[i] - r[3]) < 1e-9  # same position
                    or (lod_c[i] < r[4] - min_lod_gap)  # not sep enough
                    for r in rows
                    if r[1] == chrom
                )
                if too_close:
                    continue
                # build a temporary ScanResult view for this chromosome to
                # get the CI (lod_drop_ci operates on a ScanResult)
                _sr_c = ScanResult(
                    locus_ids=lids_c,
                    chromosomes=np.array([chrom] * n),
                    positions=pos_c,
                    lod=lod_c,
                    p_value=scan_result.p_value[mask],
                    betas=scan_result.betas[mask],
                    h2_null={chrom: scan_result.h2_null.get(chrom, 0.0)},
                    phenotype_id=scan_result.phenotype_id,
                )
                lo, hi, pk = lod_drop_ci(_sr_c, chrom, drop)
                rows.append(
                    (
                        scan_result.phenotype_id,
                        chrom,
                        lids_c[i],
                        float(pos_c[i]),
                        float(lod_c[i]),
                        lo,
                        hi,
                    )
                )

    return pd.DataFrame(rows, columns=list(_COLS))


def fixef_eqtl_scan(
    phenotype,
    genomecache,
    covariates=None,
    sample_ids=None,
    use_rint=True,
    phenotype_id="unknown",
    just_these_loci=None,
):
    """fixed-effect eQTL scan — OLS, no kinship.


    parameters
    ----------
    phenotype       : np.ndarray (n,)
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    sample_ids      : array-like (n,) or None
    use_rint        : bool
    phenotype_id    : str
    just_these_loci : set or None

    returns
    -------
    ScanResult  (h2_null = {})
    """
    from scipy import stats as _stats

    sid = list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    z_mat, valid_sid = _resolve_z(sid, genomecache)
    valid = (~np.isnan(phenotype)) & valid_sid
    y = phenotype[valid].copy()
    if use_rint:
        y = rint(y)
    n = len(y)
    z_sub = z_mat[valid] if z_mat is not None else None

    x = (
        covariates[valid].copy()
        if covariates is not None
        else np.ones((n, 1), dtype=float)
    )
    if not np.allclose(x[:, 0], 1.0):
        x = np.hstack([np.ones((n, 1)), x])
    p = x.shape[1]
    e0 = y - x @ np.linalg.lstsq(x, y, rcond=None)[0]
    rss0 = float(e0 @ e0)

    all_lids, all_chrs, all_pos = [], [], []
    all_lod, all_pval, all_betas = [], [], []

    for chrom in genomecache.chromosomes:
        lids, pos, probs = genomecache.get_chromosome(chrom)
        nf = genomecache.n_founders  # J — all founders; lstsq handles rank J-1
        for i_l, lid in enumerate(lids):
            if just_these_loci and lid not in just_these_loci:
                continue
            h = probs[i_l]  # (n, J) — all founders
            if z_sub is not None:
                h = z_sub @ h
            z_ = np.hstack([h, x])
            try:
                beta = np.linalg.lstsq(z_, y, rcond=1e-10)[0]
                e1 = y - z_ @ beta
                rss1 = float(e1 @ e1)
            except np.linalg.LinAlgError:
                all_lod.append(0.0)
                all_pval.append(1.0)
                all_betas.append(np.zeros(nf))
                all_lids.append(lid)
                all_chrs.append(chrom)
                all_pos.append(pos[i_l])
                continue
            lod = max((n / 2.0) * np.log10(max(rss0, 1e-30) / max(rss1, 1e-30)), 0.0)
            # effective rank: rank([H, X]) - rank(X) = haplotype df
            q_eff = max(int(np.linalg.matrix_rank(z_, tol=1e-10)) - p, 0)
            df2 = n - p - q_eff
            if df2 > 0 and rss1 > 0 and q_eff > 0:
                f = ((rss0 - rss1) / q_eff) / (rss1 / df2)
                pv = 1.0 - _stats.f.cdf(max(f, 0), q_eff, df2)
            else:
                pv = 1.0
            all_lod.append(lod)
            all_pval.append(pv)
            all_betas.append(beta[:nf])
            all_lids.append(lid)
            all_chrs.append(chrom)
            all_pos.append(pos[i_l])

    nf_out = genomecache.n_founders
    return ScanResult(
        locus_ids=np.array(all_lids),
        chromosomes=np.array(all_chrs),
        positions=np.array(all_pos, dtype=float),
        lod=np.array(all_lod),
        p_value=np.array(all_pval),
        betas=(np.vstack(all_betas) if all_betas else np.empty((0, nf_out))),
        h2_null={},
        phenotype_id=phenotype_id,
    )


def instability_lm_scan(
    phenotype,
    genomecache,
    covariates=None,
    sample_ids=None,
    use_rint=True,
    n_subsamples=100,
    subsample_prop=0.63,
    seed=1,
    phenotype_id="unknown",
    just_these_loci=None,
):
    """bootstrap instability scan — fixed-effect only.

    subsamples strains b times, runs fixef_eqtl_scan() on each subsample,
    reports mean LOD and peak proportion across bootstraps.

        pre-generated sample matrix from generate.simple.sample.outcomes.matrix).
    this implementation handles sampling internally.

    parameters
    ----------
    phenotype       : np.ndarray (n,)
    genomecache    : GenomeCache
    covariates      : np.ndarray (n, p) or None
    sample_ids      : array-like (n,) or None
    use_rint        : bool
    n_subsamples    : int    number of bootstrap subsamples
    subsample_prop  : float  proportion of strains per subsample (0.63)
    seed            : int
    phenotype_id    : str
    just_these_loci : set or None

    returns
    -------
    ScanResult  lod = mean bootstrap LOD, p_value = 1 − peak proportion
    """
    rng = np.random.default_rng(seed)

    logger.info(
        "instability_lm_scan: gene=%s  n_subsamples=%d  subsample_prop=%.2f",
        phenotype_id,
        n_subsamples,
        subsample_prop,
    )

    # pre-reduce to strain means once so subsample indices are always in [0, n_strains)
    if sample_ids is not None:
        _z, _vs = _resolve_z(list(sample_ids), genomecache)
        _valid = (~np.isnan(phenotype)) & _vs
        if _z is not None:
            _z_v = _z[_valid]
            _r = _z_v.sum(axis=0)
            y_scan = (_z_v.T @ phenotype[_valid]) / _r
            cov_scan = (
                (_z_v.T @ covariates[_valid]) / _r[:, None]
                if covariates is not None
                else None
            )
        else:
            y_scan = phenotype[_valid]
            cov_scan = covariates[_valid] if covariates is not None else None
    else:
        y_scan = phenotype
        cov_scan = covariates

    n = len(y_scan)
    k = max(int(n * subsample_prop), 2)

    # derive which chromosomes contain the requested loci — avoids loading all
    # 20 chromosomes per subsample when just_these_loci restricts to one chr
    if just_these_loci:
        _loci_set = set(just_these_loci)
        _active_chroms = [
            c
            for c in genomecache.chromosomes
            if any(lid in _loci_set for lid in genomecache.get_chromosome(c)[0])
        ]
    else:
        _active_chroms = list(genomecache.chromosomes)

    class _SubCache:
        """Genome cache view restricted to a subset of strain indices."""

        def __init__(self, cache, idx):
            """Wrap cache, storing the index subset for subsetting probs."""
            self._c = cache
            self._idx = idx
            self.chromosomes = _active_chroms
            self.n_founders = cache.n_founders
            self.sample_ids = [cache.sample_ids[i] for i in idx]

        def get_chromosome(self, chrom):
            """Return lids, pos, and subset probs for chrom."""
            lids, pos, probs = self._c.get_chromosome(chrom)
            return lids, pos, probs[:, self._idx, :]

    # Load all loci probs once — avoids n_subsamples × n_loci HDF5 reads
    _loci_set = set(just_these_loci) if just_these_loci else None
    _lids_chunks, _chrs_chunks, _pos_chunks, _prob_chunks = [], [], [], []
    for chrom in _active_chroms:
        lids_c, pos_c, probs_c = genomecache.get_chromosome(chrom)
        if _loci_set:
            sel = [i for i, lid in enumerate(lids_c) if lid in _loci_set]
            if not sel:
                continue
            lids_c = np.asarray(lids_c)[sel]
            pos_c = pos_c[sel]
            probs_c = probs_c[sel]
        _lids_chunks.append(np.asarray(lids_c))
        _chrs_chunks.append(np.full(len(lids_c), chrom))
        _pos_chunks.append(pos_c)
        _prob_chunks.append(probs_c)

    probs_all = np.concatenate(_prob_chunks, axis=0)  # (L, n_strains, J)
    lids_all = np.concatenate(_lids_chunks)
    chrs_all = np.concatenate(_chrs_chunks)
    pos_all = np.concatenate(_pos_chunks)
    L = len(lids_all)
    J = genomecache.n_founders

    lod_sum = np.zeros(L)
    peak_cnt = np.zeros(L, dtype=int)
    betas_0 = None
    _t_inst = time.perf_counter()

    for _si in range(n_subsamples):
        idx = rng.choice(n, size=k, replace=False)
        y_b = y_scan[idx]
        if use_rint:
            y_b = rint(y_b)

        # build covariate matrix for this subsample
        if cov_scan is not None:
            x_b = cov_scan[idx].copy()
        else:
            x_b = np.ones((k, 1), dtype=float)
        if not np.allclose(x_b[:, 0], 1.0):
            x_b = np.hstack([np.ones((k, 1)), x_b])

        # FWL: residualise y against x_b once
        e0_b = y_b - x_b @ np.linalg.lstsq(x_b, y_b, rcond=None)[0]
        rss0_b = float(e0_b @ e0_b)

        # subsampled haplotype probs: (L, k, J)
        H_b = probs_all[:, idx, :]

        # project out covariates from H: H_perp = M_X H
        Xp_b = np.linalg.pinv(x_b)  # (p, k)
        proj = np.einsum("pn,lnj->lpj", Xp_b, H_b)  # (L, p, J)
        H_perp = H_b - np.einsum("np,lpj->lnj", x_b, proj)  # (L, k, J)
        del H_b, proj

        G = np.einsum("lni,lnj->lij", H_perp, H_perp)  # (L, J, J)
        c = np.einsum("lnj,n->lj", H_perp, e0_b)  # (L, J)
        G_pinv = np.linalg.pinv(G, rcond=1e-10)  # (L, J, J)
        rss_red = np.einsum("li,lij,lj->l", c, G_pinv, c)  # (L,)

        rss1 = np.maximum(rss0_b - rss_red, 1e-30)
        with np.errstate(divide="ignore", invalid="ignore"):
            lod_b = np.where(rss1 < rss0_b, (k / 2.0) * np.log10(rss0_b / rss1), 0.0)

        lod_sum += lod_b
        peak_cnt[np.argmax(lod_b)] += 1
        if betas_0 is None:
            betas_0 = np.einsum("lij,lj->li", G_pinv, c)  # (L, J)

        if (_si + 1) % 10 == 0 or _si == n_subsamples - 1:
            logger.debug(
                "%s instability: %d/%d subsamples  %.1fs elapsed",
                phenotype_id,
                _si + 1,
                n_subsamples,
                time.perf_counter() - _t_inst,
            )

    _inst_lod = lod_sum / n_subsamples
    _inst_peak_idx = int(np.argmax(peak_cnt))
    logger.info(
        "instability_lm_scan: %s done — peak_freq=%.3f at locus %s",
        phenotype_id,
        float(peak_cnt[_inst_peak_idx]) / n_subsamples,
        lids_all[_inst_peak_idx],
    )
    return ScanResult(
        locus_ids=lids_all,
        chromosomes=chrs_all,
        positions=pos_all,
        lod=_inst_lod,
        p_value=1.0 - peak_cnt / n_subsamples,
        betas=betas_0 if betas_0 is not None else np.empty((0, J)),
        h2_null={},
        phenotype_id=phenotype_id,
    )


def scan_h2lmm_mi(
    phenotype,
    genomecache,
    covariates=None,
    kinship_decomp=None,
    sample_ids=None,
    n_imp=11,
    seed=1,
    use_rint=True,
    phenotype_id="unknown",
    scan_window="cis",
    gene_chr=None,
    gene_pos_mb=None,
    window_mb=10.0,
):
    """multiple-imputation haplotype scan.

    samples n_imp diplotype states per locus and averages LOD scores.

    parameters
    ----------
    phenotype      : np.ndarray (n_samples,)  one value per mouse
    genomecache    : GenomeCache
    covariates     : np.ndarray (n_samples, p) or None
    kinship_decomp : dict or None
    sample_ids     : array-like (n_samples,) or None  genotype-level strain IDs
    n_imp          : int   number of imputations (default 11)
    seed           : int
    use_rint       : bool
    phenotype_id   : str
    scan_window    : str  'cis' | 'genome' | 'local'  (default 'cis')
    gene_chr       : str or None  chromosome of the gene (required for cis/local)
    gene_pos_mb    : float or None  TSS position in Mb (required for local)
    window_mb      : float  half-window for local mode (default 10.0)

    returns
    -------
    ScanResult  lod = mean over n_imp imputations
    """
    rng = np.random.default_rng(seed)

    if scan_window in ("cis", "local") and gene_chr is None:
        raise ValueError(f"scan_window='{scan_window}' requires gene_chr")
    if scan_window == "local" and gene_pos_mb is None:
        raise ValueError("scan_window='local' requires gene_pos_mb")

    logger.info(
        "scan_h2lmm_mi: gene=%s  chr=%s  n_imp=%d  n_samples=%d",
        phenotype_id,
        gene_chr if gene_chr is not None else "genome",
        n_imp,
        len(phenotype),
    )

    if kinship_decomp is None:
        kd_raw = kin_module.calc_kinship(genomecache, method="loco")
        kinship_decomp = kin_module.decomp_kinship(kd_raw)

    # only impute chromosomes that will actually be scanned
    _chrom_list = [c for c in genomecache.chromosomes if c in kinship_decomp]
    if scan_window in ("cis", "local"):
        _chrom_list = [c for c in _chrom_list if c == str(gene_chr)]

    # pre-estimate h² once from the REAL cache — held fixed across all imputations.
    # replicate-aware: same strain-mean path as scan_h2lmm.
    _sid_mi = (
        list(sample_ids) if sample_ids is not None else list(genomecache.sample_ids)
    )
    _z_mi, _vs = _resolve_z(_sid_mi, genomecache)
    valid_mi = (~np.isnan(phenotype)) & _vs
    y_base = phenotype[valid_mi].copy()
    x_base = (
        covariates[valid_mi].copy()
        if covariates is not None
        else np.ones((valid_mi.sum(), 1), dtype=float)
    )
    if not np.allclose(x_base[:, 0], 1.0):
        x_base = np.hstack([np.ones((len(x_base), 1)), x_base])
    _zsub_mi = _z_mi[valid_mi] if _z_mi is not None else None
    if use_rint:
        y_base = rint(y_base)
    if _zsub_mi is not None:
        _r_mi = _zsub_mi.sum(axis=0)
        y_base = (_zsub_mi.T @ y_base) / _r_mi
        x_base = (_zsub_mi.T @ x_base) / _r_mi[:, None]
    fixed_h2 = {}
    for _c in _chrom_list:
        _ev = kinship_decomp[_c]["values"]
        _evc = (
            kinship_decomp[_c]["vectors"]
            if _zsub_mi is not None
            else kinship_decomp[_c]["vectors"][valid_mi]
        )
        _yr = _evc.T @ y_base
        _xr = _evc.T @ x_base
        _h2, _ = lmm_module.estimate_h2_null(_yr, _xr, _ev, method="ml")
        fixed_h2[_c] = _h2

    # ── one pass per chromosome — all n_imp imputations vectorised ──────────
    # scan_chromosome_mi generates n_imp one-hot draws at once and computes
    # H_perp/G/G_pinv in a single batched einsum, eliminating:
    #   - n_imp × GenomeCache object constructions
    #   - n_imp × scan_chromosome calls per chromosome
    all_lids: list[np.ndarray] = []
    all_chrs: list[np.ndarray] = []
    all_pos: list[np.ndarray] = []
    all_lod: list[np.ndarray] = []
    # p_value and betas from soft-prob scan on first chromosome (reference only)
    all_pval: list[np.ndarray] = []
    all_betas: list[np.ndarray] = []
    h2_null_dict: dict[str, float] = {}

    for chrom in _chrom_list:
        ev = kinship_decomp[chrom]["values"]
        evc = kinship_decomp[chrom]["vectors"]

        mean_lod = lmm_module.scan_chromosome_mi(
            y_base,
            genomecache,
            chrom,
            x_base,
            ev,
            evc,
            fixed_h2[chrom],
            n_imp=n_imp,
            rng=rng,
        )
        # reference scan on soft probs — for p_value/betas fields only
        cd_ref = lmm_module.precompute_chromosome(genomecache, chrom, ev, evc)
        ref = lmm_module.scan_chromosome(
            y_base, cd_ref, x_base, ev, evc, fixed_h2[chrom]
        )
        n_loci = len(ref["locus_ids"])
        all_lids.append(ref["locus_ids"])
        all_chrs.append(np.full(n_loci, chrom))
        all_pos.append(ref["positions"])
        all_lod.append(mean_lod)
        all_pval.append(ref["p_value"])
        all_betas.append(ref["betas"])
        h2_null_dict[chrom] = fixed_h2[chrom]
        logger.debug(
            "%s MI chr%s done (%d loci)", phenotype_id, chrom, len(ref["locus_ids"])
        )

    _mi_lod = np.concatenate(all_lod)
    _mi_peak_idx = int(np.argmax(_mi_lod))
    logger.info(
        "scan_h2lmm_mi: %s done — peak_lod=%.3f (mean over %d imputations)",
        phenotype_id,
        float(_mi_lod[_mi_peak_idx]),
        n_imp,
    )
    return ScanResult(
        locus_ids=np.concatenate(all_lids),
        chromosomes=np.concatenate(all_chrs),
        positions=np.concatenate(all_pos),
        lod=_mi_lod,
        p_value=np.concatenate(all_pval),
        betas=np.vstack(all_betas),
        h2_null=h2_null_dict,
        phenotype_id=phenotype_id,
        model="additive",
    )


def scan_multi_gene_mi(
    expr_cache: "dict[str, np.ndarray]",
    gene_chr_map: "dict[str, str]",
    genomecache: "GenomeCache",  # type: ignore[name-defined]  # noqa: F821
    covariates: "np.ndarray | None" = None,
    kinship_decomp: "dict | None" = None,
    sample_ids: "list[str] | None" = None,
    n_imp: int = 11,
    seed: int = 1,
    use_rint: bool = True,
    peak_mb_map: "dict[str, float] | None" = None,
    window_mb: float = 10.0,
    peak_only: bool = True,
    output_path: "Path | None" = None,
    gene_meta: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """multiple-imputation scan over a set of genes, streaming to disk.

    calls scan_h2lmm_mi per gene and writes each result to output_path
    immediately, keeping peak memory proportional to one gene rather than
    all genes simultaneously.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}  pre-loaded expression vectors
    gene_chr_map   : dict {gene_id: str}  chromosome for cis scan
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None
    kinship_decomp : dict or None  {chrom: {'values':, 'vectors':}}
    sample_ids     : list[str] or None  genotype-level strain IDs
    n_imp          : int  imputations per gene (default 11)
    seed           : int  base seed; incremented per gene
    use_rint       : bool
    peak_mb_map    : dict {gene_id: float} or None — S4 peak positions; when provided
                     scan uses local window ± window_mb around each gene's peak
    window_mb      : float  half-window in Mb for local scan (default 10.0)
    peak_only      : bool   when True (default) output one peak-locus row per gene;
                     when False output all scanned loci (full per-locus profile)
    output_path    : Path or None  stream CSV here; None accumulates in RAM
    gene_meta      : DataFrame keyed by 'gene_symbol' — extra columns joined into each row

    returns
    -------
    peak_only=True  — gene_symbol, chr, locus [+ gene_meta cols], peak_mb, peak_lod
    peak_only=False — gene_symbol, chr [+ gene_meta cols], locus, pos_mb, lod
    """
    _gm_idx = _build_gm_idx(gene_meta)
    if peak_only:
        _id_cols = ["gene_symbol", "chr", "locus"]
        _data_cols = ["peak_mb", "peak_lod"]
    else:
        _id_cols = ["gene_symbol", "chr"]
        _data_cols = ["locus", "pos_mb", "lod"]
    _first = True
    _buf: "list[pd.DataFrame]" = []
    _n_genes_mi = len(expr_cache)
    logger.info(
        "scan_multi_gene_mi: n_genes=%d  n_imp=%d  scan_window=%s",
        _n_genes_mi,
        n_imp,
        "local" if peak_mb_map else "cis",
    )
    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, "")
        if not gene_chr:
            logger.warning("scan_multi_gene_mi: no chr for %s — skipped", gene_id)
            continue
        logger.debug("scan_multi_gene_mi: [%d/%d] %s", _g + 1, _n_genes_mi, gene_id)
        peak_mb = peak_mb_map.get(gene_id) if peak_mb_map else None
        scan_kw: "dict" = dict(
            scan_window="local" if peak_mb is not None else "cis",
            gene_chr=gene_chr,
        )
        if peak_mb is not None:
            scan_kw["gene_pos_mb"] = peak_mb
            scan_kw["window_mb"] = window_mb
        try:
            res = scan_h2lmm_mi(
                y,
                genomecache,
                covariates=covariates,
                kinship_decomp=kinship_decomp,
                sample_ids=sample_ids,
                n_imp=n_imp,
                seed=seed + _g,
                use_rint=use_rint,
                phenotype_id=gene_id,
                **scan_kw,
            )
            _df = pd.DataFrame(
                {
                    "gene_symbol": gene_id,
                    "chr": res.chromosomes,
                    "locus": res.locus_ids,
                    "pos_mb": res.positions,
                    "lod": res.lod,
                }
            )
            del res
            if peak_only and not _df.empty:
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
            logger.warning("scan_multi_gene_mi: %s failed — %s", gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()


def instability_scan_multi_gene(
    expr_cache: "dict[str, np.ndarray]",
    gene_chr_map: "dict[str, str]",
    genomecache: "GenomeCache",  # type: ignore[name-defined]  # noqa: F821
    covariates: "np.ndarray | None" = None,
    sample_ids: "list[str] | None" = None,
    n_subsamples: int = 100,
    subsample_prop: float = 0.63,
    seed: int = 1,
    peak_mb_map: "dict[str, float] | None" = None,
    window_mb: float = 10.0,
    peak_only: bool = True,
    output_path: "Path | None" = None,
    gene_meta: "pd.DataFrame | None" = None,
) -> pd.DataFrame:
    """instability scan over a set of genes, streaming to disk.

    calls instability_lm_scan per gene restricted to each gene's chromosome
    (or a local window when peak_mb_map is provided) and streams each result
    immediately, keeping peak memory bounded.

    parameters
    ----------
    expr_cache     : dict {gene_id: np.ndarray (n,)}
    gene_chr_map   : dict {gene_id: str}
    genomecache    : GenomeCache
    covariates     : np.ndarray (n, p) or None
    sample_ids     : list[str] or None  genotype-level strain IDs
    n_subsamples   : int  bootstrap iterations per gene (default 100)
    subsample_prop : float  proportion of strains per subsample (default 0.63)
    seed           : int  base seed; incremented per gene
    peak_mb_map    : dict {gene_id: float} or None — S4 peak positions; when provided
                     scan is restricted to loci within ± window_mb of each gene's peak
    window_mb      : float  half-window in Mb for local locus filter (default 10.0)
    peak_only      : bool   when True (default) output one peak-locus row per gene;
                     when False output all scanned loci
    output_path    : Path or None  stream CSV here; None accumulates in RAM
    gene_meta      : DataFrame keyed by 'gene_symbol' — extra columns joined into each row

    returns
    -------
    peak_only=True  — gene_symbol, chr, locus [+ gene_meta cols], peak_mb, peak_lod
    peak_only=False — gene_symbol, chr [+ gene_meta cols], locus, pos_mb, lod
    """
    _gm_idx = _build_gm_idx(gene_meta)
    if peak_only:
        _id_cols = ["gene_symbol", "chr", "locus"]
        _data_cols = ["peak_mb", "peak_lod"]
    else:
        _id_cols = ["gene_symbol", "chr"]
        _data_cols = ["locus", "pos_mb", "lod"]
    _first = True
    _buf: "list[pd.DataFrame]" = []
    _n_genes_inst = len(expr_cache)
    logger.info(
        "instability_scan_multi_gene: n_genes=%d  n_subsamples=%d",
        _n_genes_inst,
        n_subsamples,
    )
    for _g, (gene_id, y) in enumerate(expr_cache.items()):
        gene_chr = gene_chr_map.get(gene_id, "")
        if not gene_chr:
            logger.warning(
                "instability_scan_multi_gene: no chr for %s — skipped", gene_id
            )
            continue
        logger.debug(
            "instability_scan_multi_gene: [%d/%d] %s", _g + 1, _n_genes_inst, gene_id
        )
        try:
            lids_chr, _, pos_chr = genomecache.get_chromosome(gene_chr)
            peak_mb = peak_mb_map.get(gene_id) if peak_mb_map else None
            if peak_mb is not None:
                loci_set = {
                    lid
                    for lid, p in zip(lids_chr, pos_chr)
                    if abs(p - peak_mb) <= window_mb
                }
            else:
                loci_set = set(lids_chr)
            res = instability_lm_scan(
                y,
                genomecache,
                covariates=covariates,
                sample_ids=sample_ids,
                n_subsamples=n_subsamples,
                subsample_prop=subsample_prop,
                seed=seed + _g,
                phenotype_id=gene_id,
                just_these_loci=loci_set,
            )
            _df = pd.DataFrame(
                {
                    "gene_symbol": gene_id,
                    "chr": res.chromosomes,
                    "locus": res.locus_ids,
                    "pos_mb": res.positions,
                    "lod": res.lod,
                }
            )
            del res
            if peak_only and not _df.empty:
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
            logger.warning("instability_scan_multi_gene: %s failed — %s", gene_id, exc)
    gc.collect()
    if output_path is not None:
        return (
            pd.read_csv(output_path)
            if output_path.exists() and not _first
            else pd.DataFrame()
        )
    return pd.concat(_buf, ignore_index=True) if _buf else pd.DataFrame()


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
    """streaming LMM scan over a set of genes; returns peak statistics DataFrame.

    chromosome-outer, gene-inner loop with precomputed H rotation — mirrors
    scan_multi_gene but accepts a dict-based expression cache (consistent with
    scan_multi_gene_mi / instability_scan_multi_gene) and captures per-gene
    rss0/rss1 for diagnostic statistics.

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
    gene_meta      : DataFrame keyed by 'gene_symbol' — extra columns joined into output rows
    founder_names  : list[str] or None  display names for beta columns; None → genomecache.founders
    model          : str  'additive' (default) or 'full'

    returns
    -------
    DataFrame with one row per gene:
        gene_symbol, chr, peak_locus, peak_mb, peak_lod, h2, n_valid,
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
            betas_all = chr_res.get("betas", None)  # (n_loci, J)

            if scan_window == "local" and gene_pos_mb_map is not None:
                tss = float(gene_pos_mb_map[gene_id])
                mask = np.abs(pos_arr - tss) <= window_mb
                if not mask.any():
                    logger.debug(
                        "%s: no loci within %.1fMb on chr%s",
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
                "gene_symbol": gene_id,
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
                            "gene_symbol": gene_id,
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
            "[%d] scan_multi_gene_lmm chr%s: %d genes  %.1fs",
            ci,
            chrom,
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
        _gm = gene_meta.drop_duplicates(subset="gene_symbol")
        result = result.merge(_gm, on="gene_symbol", how="left", suffixes=("", "_meta"))
    return result

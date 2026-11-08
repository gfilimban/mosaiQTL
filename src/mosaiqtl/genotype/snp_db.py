"""
snp_db — founder allele database for haplotype-imputed SNP fine-mapping

reads a genotyping-array file listing which allele each founder strain
carries at each SNP, matches each SNP to the nearest marker already in the
genome cache (done once, up front, not on every scan), and saves the result
as an HDF5 file organized by chromosome.

works for any multi-founder population, not just the Collaborative Cross:
founder names and their order always come from the Population definition
rather than being hard-coded here.

at scan time, load_founder_alleles() reads this file back in and hands the
result to the SNP scan function.

see docs/REFERENCES.md, SNP fine-mapping section, for background.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

from ..utils import chr_sort_key

if TYPE_CHECKING:
    from .genomecache import GenomeCache
    from ..models.config import Population

logger = logging.getLogger("mosaiqtl")

__all__ = [
    "build_founder_allele_gm",
    "build_founder_allele_mgp",
    "CC_VCF_SAMPLE_MAP",
    "load_founder_alleles",
    "make_alt_mask",
    "read_allele_dir",
    "build_allele_dir",
    "read_alleles_flat",
    "build_snp_catalogue",
]

# GRCm39 (mm39) bare chromosome name → NCBI RefSeq accession.
# required to construct SPDI variant IDs: the seq_id field must be a recognised
# RefSeq accession, not a UCSC-style chr name.  source: NCBI assembly GCF_000001635.27.
_MM39_REFSEQ: dict[str, str] = {
    "1":  "NC_000067.7",
    "2":  "NC_000068.8",
    "3":  "NC_000069.7",
    "4":  "NC_000070.7",
    "5":  "NC_000071.7",
    "6":  "NC_000072.7",
    "7":  "NC_000073.7",
    "8":  "NC_000074.7",
    "9":  "NC_000075.7",
    "10": "NC_000076.7",
    "11": "NC_000077.7",
    "12": "NC_000078.7",
    "13": "NC_000079.7",
    "14": "NC_000080.7",
    "15": "NC_000081.7",
    "16": "NC_000082.7",
    "17": "NC_000083.7",
    "18": "NC_000084.7",
    "19": "NC_000085.7",
    "X":  "NC_000086.8",
    "Y":  "NC_000087.8",
    "MT": "NC_005089.1",
}

# maps MGP VCF sample names → canonical CC founder names in mosaiQTL
CC_VCF_SAMPLE_MAP: dict[str, str] = {
    "A_J": "AJ",
    "129S1_SvImJ": "129S1J",
    "NOD_ShiLtJ": "NODJ",
    "NZO_HlLtJ": "NZOJ",
    "CAST_EiJ": "CASTJ",
    "PWK_PhJ": "PWKJ",
    "WSB_EiJ": "WSBJ",
}


# ─────────────────────────────────────────────────────────────────────────────
# internal helpers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_founders(
    genomecache: "GenomeCache",
    population: "Population | None",
    founder_cols: list[str] | None,
) -> list[str]:
    """decide which founder names/order to use, in priority order.

    priority: an explicit list passed in, then the Population definition,
    then whatever the genome cache itself already has.
    """
    if founder_cols is not None:
        return list(founder_cols)
    if population is not None:
        return list(population.founders)
    return list(genomecache.founders)


_MISSING_BUILD: list[str] = ["N", "H", ""]


def _count_alleles_vectorised(founder_calls: np.ndarray) -> np.ndarray:
    """Count distinct non-missing alleles per SNP row without a Python loop.

    Parameters
    ----------
    founder_calls : np.ndarray of str, shape (n_snps, n_founders)

    Returns
    -------
    np.ndarray of int, shape (n_snps,)
    """
    calls = founder_calls.astype(str)
    # blank out missing codes so they sort together at the front
    calls = np.where(np.isin(calls, _MISSING_BUILD), "", calls)
    # sort row-wise so identical alleles are adjacent
    sorted_calls = np.sort(calls, axis=1)  # (n_snps, n_founders)
    # count transitions between adjacent non-empty values only
    left = sorted_calls[:, :-1]
    right = sorted_calls[:, 1:]
    both_nonempty = (left != "") & (right != "")
    transitions = ((left != right) & both_nonempty).sum(axis=1)
    has_data = (sorted_calls != "").any(axis=1)
    return np.where(has_data, transitions + 1, 0)


# ─────────────────────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────────────────────


def build_founder_allele_gm(
    gm_csv: str | Path,
    genomecache: "GenomeCache",
    out_h5: str | Path,
    population: "Population | None" = None,
    founder_cols: list[str] | None = None,
    max_dist_mb: float = 0.5,
) -> dict:
    """build a founder allele database from a genotyping-array file.

    each SNP is matched to the nearest genome cache marker by physical
    position, computed once here rather than at every scan.

    SNPs are dropped if:
      - every founder carries the same allele (no information to use)
      - more than 2 distinct alleles appear across founders
      - the nearest cache marker is farther than max_dist_mb away

    parameters
    ----------
    gm_csv       : path to the founder-genotype CSV.
                   required columns: marker, chr, pos (bp), plus one column
                   per founder strain.
    genomecache  : GenomeCache  supplies marker positions per chromosome.
    out_h5       : output HDF5 path (created or overwritten).
    population   : Population or None
                   when given, its founder list and order take precedence
                   over the cache's own founder list.
    founder_cols : list[str] or None
                   explicit override for founder column names, for cases
                   where they don't match either the Population or the
                   cache (e.g. a custom strain-naming scheme). takes
                   precedence over both.
    max_dist_mb  : SNPs farther than this from the nearest cache marker are
                   dropped (default 0.5 Mb).

    returns
    -------
    dict  summary counts:
        n_raw, n_monomorphic, n_multiallelic, n_too_far, n_written
        per_chr: {chr: n_snps}
    """
    gm_csv = Path(gm_csv)
    out_h5 = Path(out_h5)
    fcols = _resolve_founders(genomecache, population, founder_cols)

    logger.info("build_founder_allele_gm: %d founders, reading %s", len(fcols), gm_csv)
    raw = pd.read_csv(gm_csv, dtype={"chr": str})
    raw.columns = [c.strip() for c in raw.columns]

    missing = [c for c in (["marker", "chr", "pos"] + fcols) if c not in raw.columns]
    if missing:
        raise ValueError(
            f"CSV missing columns: {missing}\n" f"Available: {list(raw.columns)}"
        )

    n_raw = len(raw)
    raw["pos_mb"] = raw["pos"] / 1e6

    # drop SNPs that are uninformative (same allele in all founders) or not
    # biallelic (more than 2 distinct alleles)
    founder_calls = raw[fcols].to_numpy(dtype=str)
    n_alleles = _count_alleles_vectorised(founder_calls)
    mono_mask = n_alleles <= 1
    multi_mask = n_alleles > 2
    keep = ~mono_mask & ~multi_mask
    n_mono = int(mono_mask.sum())
    n_multi = int(multi_mask.sum())
    raw = raw[keep].reset_index(drop=True)
    logger.info(
        "  filtered %d monomorphic + %d multiallelic → %d remaining",
        n_mono,
        n_multi,
        len(raw),
    )

    # match each SNP to its nearest genome cache marker
    n_too_far = 0
    n_written = 0
    per_chr: dict[str, int] = {}

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()

    with h5py.File(out_h5, "w") as hf:
        meta = hf.require_group("metadata")
        meta.attrs["source_csv"] = str(gm_csv)
        meta.attrs["n_founders"] = len(fcols)
        meta.attrs["max_dist_mb"] = max_dist_mb
        meta.create_dataset(
            "founders", data=np.array(fcols, dtype=object), dtype=str_dt
        )

        for chrom in genomecache.chromosomes:
            # the CSV may label chromosomes '1' while the cache uses 'chr1'
            chr_key = chrom.lstrip("chr")
            chr_snps = raw[raw["chr"] == chr_key].copy()
            if len(chr_snps) == 0:
                continue

            lids_cache, pos_cache = genomecache.get_locus_metadata(chrom)
            snp_pos = chr_snps["pos_mb"].to_numpy()

            # get_chromosome() guarantees sorted ascending positions —
            # searchsorted directly, no argsort needed
            n_markers = len(pos_cache)
            idx_r = np.searchsorted(pos_cache, snp_pos).clip(0, n_markers - 1)
            idx_l = (idx_r - 1).clip(0, n_markers - 1)
            dist_r = np.abs(pos_cache[idx_r] - snp_pos)
            dist_l = np.abs(pos_cache[idx_l] - snp_pos)
            best = np.where(dist_l <= dist_r, idx_l, idx_r)
            best_dist = np.minimum(dist_l, dist_r)

            close = best_dist <= max_dist_mb
            n_far_chr = int((~close).sum())
            n_too_far += n_far_chr
            if n_far_chr:
                logger.debug(
                    "  %s: dropped %d SNPs > %.2f Mb from nearest locus",
                    chrom,
                    n_far_chr,
                    max_dist_mb,
                )

            chr_snps = chr_snps[close].reset_index(drop=True)
            best = best[close]
            if len(chr_snps) == 0:
                continue

            grp = hf.require_group(chrom)
            grp.create_dataset(
                "marker",
                data=np.array(chr_snps["marker"].to_numpy(), dtype=object),
                dtype=str_dt,
            )
            grp.create_dataset(
                "pos_mb", data=chr_snps["pos_mb"].to_numpy().astype(np.float32)
            )
            grp.create_dataset(
                "pos_bp", data=chr_snps["pos"].to_numpy().astype(np.int32)
            )
            grp.create_dataset(
                "locus_id",
                data=np.array(lids_cache[best].tolist(), dtype=object),
                dtype=str_dt,
            )
            alleles = chr_snps[fcols].to_numpy(dtype=object)
            grp.create_dataset("alleles", data=alleles, dtype=str_dt)

            n_chr = len(chr_snps)
            per_chr[chrom] = n_chr
            n_written += n_chr
            logger.info("  %s: %d SNPs written", chrom, n_chr)

        meta.attrs["n_snps_total"] = n_raw
        meta.attrs["n_snps_written"] = n_written

    summary = {
        "n_raw": n_raw,
        "n_monomorphic": n_mono,
        "n_multiallelic": n_multi,
        "n_too_far": n_too_far,
        "n_written": n_written,
        "per_chr": per_chr,
    }
    logger.info(
        "build_founder_allele_gm: done — %d/%d SNPs written to %s",
        n_written,
        n_raw,
        out_h5,
    )
    return summary


def build_founder_allele_mgp(
    vcf_path: str | Path,
    genomecache: "GenomeCache",
    out_h5: str | Path,
    population: "Population | None" = None,
    vcf_sample_map: dict[str, str] | None = None,
    ref_founder: str = "B6J",
    max_dist_mb: float = 0.5,
) -> dict:
    """Build a founder allele database from an MGP VCF file.

    Reads founder genotype calls from a pysam-accessible VCF/BCF, matches
    each SNP to the nearest genome cache marker, and writes the same HDF5
    format produced by :func:`build_founder_allele_gm`.

    Because B6J is the GRCm39 reference strain, it is absent from the MGP
    VCF; this function synthesises it as an all-REF column.

    auto-creates a tabix index for the VCF if none is found (requires that
    the VCF is bgzip-compressed).

    Parameters
    ----------
    vcf_path      : path to a bgzip-compressed VCF/BCF (optionally pre-indexed).
    genomecache   : GenomeCache — supplies chromosome list and marker positions.
    out_h5        : output HDF5 path (created or overwritten).
    population    : unused; reserved for future population-agnostic calls.
    vcf_sample_map: maps VCF sample names → canonical founder names.
                    pass ``CC_VCF_SAMPLE_MAP`` for the CC/DO panel.
                    ``None`` uses VCF sample names as-is.
    ref_founder   : name for the synthetic REF-only column.  Must not appear
                    in ``vcf_sample_map`` values or VCF sample names.
    max_dist_mb   : SNPs farther than this from the nearest cache marker are
                    dropped.

    Returns
    -------
    dict  summary counts: n_raw, n_monomorphic, n_multiallelic, n_too_far,
          n_written, per_chr.
    """
    try:
        import pysam  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("pysam is required: pip install pysam") from exc

    vcf_path = Path(vcf_path)
    out_h5 = Path(out_h5)

    tbi = Path(str(vcf_path) + ".tbi")
    csi = Path(str(vcf_path) + ".csi")
    if not tbi.exists() and not csi.exists():
        logger.info("build_founder_allele_mgp: creating tabix index for %s", vcf_path)
        pysam.tabix_index(str(vcf_path), preset="vcf", force=True)

    vcf = pysam.VariantFile(str(vcf_path))
    vcf_samples: list[str] = list(vcf.header.samples)

    if vcf_sample_map is not None:
        fcols_vcf = [s for s in vcf_samples if s in vcf_sample_map]
        fcols = [vcf_sample_map[s] for s in fcols_vcf]
    else:
        fcols_vcf = vcf_samples
        fcols = list(vcf_samples)

    if ref_founder in fcols:
        raise ValueError(
            f"ref_founder {ref_founder!r} already present in founder list — "
            "check vcf_sample_map"
        )
    # prepend: B6J is always REF at every position
    fcols = [ref_founder] + fcols

    logger.info(
        "build_founder_allele_mgp: %d founders (+ %s synthetic REF), reading %s",
        len(fcols),
        ref_founder,
        vcf_path,
    )

    n_raw = 0
    n_mono = 0
    n_multi = 0
    n_too_far = 0
    n_written = 0
    per_chr: dict[str, int] = {}

    out_h5.parent.mkdir(parents=True, exist_ok=True)
    str_dt = h5py.string_dtype()

    with h5py.File(out_h5, "w") as hf:
        meta = hf.require_group("metadata")
        meta.attrs["source_vcf"] = str(vcf_path)
        meta.attrs["n_founders"] = len(fcols)
        meta.attrs["max_dist_mb"] = max_dist_mb
        meta.create_dataset(
            "founders", data=np.array(fcols, dtype=object), dtype=str_dt
        )

        for chrom in genomecache.chromosomes:
            chr_key = chrom.lstrip("chr")  # VCF uses bare contig names: '1', '2', …

            markers_list: list[str] = []
            rsid_list: list[str] = []
            pos_bp_list: list[int] = []
            allele_rows: list[list[str]] = []

            seq_id = _MM39_REFSEQ.get(chr_key, f"chr{chr_key}")

            try:
                for rec in vcf.fetch(chr_key):
                    if len(rec.alts) != 1:
                        continue
                    ref_nuc: str = rec.ref
                    alt_nuc: str = rec.alts[0]
                    if len(ref_nuc) != 1 or len(alt_nuc) != 1:
                        continue

                    row: list[str] = [ref_nuc]  # B6J is GRCm39 reference; allele is always REF
                    for sname in fcols_vcf:
                        gt = rec.samples[sname]["GT"]
                        if gt == (0, 0):
                            row.append(ref_nuc)
                        elif gt == (1, 1):
                            row.append(alt_nuc)
                        else:
                            row.append("N")

                    # SPDI: {RefSeq_accession}:{pos_0based}:{ref}:{alt}
                    # rec.pos is already 0-based; seq_id from _MM39_REFSEQ
                    markers_list.append(f"{seq_id}:{rec.pos}:{ref_nuc}:{alt_nuc}")
                    rsid_list.append(rec.id if (rec.id and rec.id != ".") else "")
                    pos_bp_list.append(rec.pos + 1)
                    allele_rows.append(row)
                    n_raw += 1
            except ValueError:
                continue

            if not markers_list:
                continue

            founder_calls = np.array(allele_rows, dtype=object)
            n_alleles = _count_alleles_vectorised(founder_calls.astype(str))
            mono_mask = n_alleles <= 1
            multi_mask = n_alleles > 2
            keep = ~mono_mask & ~multi_mask
            n_mono += int(mono_mask.sum())
            n_multi += int(multi_mask.sum())

            if not keep.any():
                continue

            markers_arr = np.array(markers_list)[keep]
            rsid_arr = np.array(rsid_list)[keep]
            pos_bp_arr = np.array(pos_bp_list, dtype=np.int32)[keep]
            pos_mb_arr = pos_bp_arr.astype(np.float64) / 1e6
            founder_calls = founder_calls[keep]

            lids_cache, pos_cache = genomecache.get_locus_metadata(chrom)
            n_loci = len(pos_cache)
            idx_r = np.searchsorted(pos_cache, pos_mb_arr).clip(0, n_loci - 1)
            idx_l = (idx_r - 1).clip(0, n_loci - 1)
            dist_r = np.abs(pos_cache[idx_r] - pos_mb_arr)
            dist_l = np.abs(pos_cache[idx_l] - pos_mb_arr)
            best = np.where(dist_l <= dist_r, idx_l, idx_r)
            best_dist = np.minimum(dist_l, dist_r)

            close = best_dist <= max_dist_mb
            n_far_chr = int((~close).sum())
            n_too_far += n_far_chr
            if n_far_chr:
                logger.debug(
                    "  %s: dropped %d SNPs > %.2f Mb from nearest locus",
                    chrom,
                    n_far_chr,
                    max_dist_mb,
                )

            markers_arr = markers_arr[close]
            rsid_arr = rsid_arr[close]
            pos_bp_arr = pos_bp_arr[close]
            pos_mb_arr = pos_mb_arr[close]
            founder_calls = founder_calls[close]
            best = best[close]

            if len(markers_arr) == 0:
                continue

            grp = hf.require_group(chrom)
            grp.create_dataset(
                "marker",
                data=np.array(markers_arr, dtype=object),
                dtype=str_dt,
            )
            grp.create_dataset("pos_mb", data=pos_mb_arr.astype(np.float32))
            grp.create_dataset("pos_bp", data=pos_bp_arr)
            grp.create_dataset(
                "locus_id",
                data=np.array(lids_cache[best].tolist(), dtype=object),
                dtype=str_dt,
            )
            grp.create_dataset(
                "alleles", data=founder_calls.astype(object), dtype=str_dt
            )
            grp.create_dataset(
                "rsid",
                data=np.array(rsid_arr, dtype=object),
                dtype=str_dt,
            )

            n_chr = len(markers_arr)
            per_chr[chrom] = n_chr
            n_written += n_chr
            logger.info("  %s: %d SNPs written", chrom, n_chr)

        meta.attrs["n_snps_total"] = n_raw
        meta.attrs["n_snps_written"] = n_written

    vcf.close()

    summary = {
        "n_raw": n_raw,
        "n_monomorphic": n_mono,
        "n_multiallelic": n_multi,
        "n_too_far": n_too_far,
        "n_written": n_written,
        "per_chr": per_chr,
    }
    logger.info(
        "build_founder_allele_mgp: done — %d/%d SNPs written to %s",
        n_written,
        n_raw,
        out_h5,
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# load
# ─────────────────────────────────────────────────────────────────────────────


def load_founder_alleles(
    h5_path: str | Path,
    chrom: str | None = None,
) -> pd.DataFrame:
    """load founder allele database for one or all chromosomes.

    founder column names and order are read from the HDF5 metadata —
    no population-specific assumptions at load time.

    parameters
    ----------
    h5_path : path to HDF5 built by build_founder_allele_gm()
    chrom   : chromosome key (e.g. 'chr10').  if None, loads all chromosomes.

    returns
    -------
    pd.DataFrame  columns: marker, chr, pos_mb, pos_bp, locus_id, rsid,
                            <founder_0>, ..., <founder_J-1>
        founder column names match the population used at build time.
        rsid is the dbSNP rsID where annotated; NaN (empty string) otherwise.
    """
    h5_path = Path(h5_path)
    frames = []

    with h5py.File(h5_path, "r") as hf:
        # founder order from metadata — population-agnostic
        # .asstr() returns Python str directly; no per-element bytes-decode loop
        fcols = list(hf["metadata"]["founders"].asstr()[:])
        n_f = len(fcols)

        chroms = (
            [chrom] if chrom is not None else [k for k in hf.keys() if k != "metadata"]
        )

        for c in chroms:
            if c not in hf:
                raise KeyError(f"chromosome {c!r} not found in {h5_path}")
            grp = hf[c]
            n = grp["marker"].shape[0]
            # .asstr() vectorises the bytes→str decode for vlen string datasets
            alleles = grp["alleles"].asstr()[:].reshape(n, n_f)
            rsid_raw = grp["rsid"].asstr()[:]
            df = pd.DataFrame(
                {
                    "marker": grp["marker"].asstr()[:],
                    "chr": c,
                    "pos_mb": np.array(grp["pos_mb"]).astype(float),
                    "pos_bp": np.array(grp["pos_bp"]).astype(int),
                    "locus_id": grp["locus_id"].asstr()[:],
                    # empty string → NaN; keeps column dtype object (nullable string)
                    "rsid": np.where(rsid_raw == "", np.nan, rsid_raw),
                }
            )
            for fi, fname in enumerate(fcols):
                df[fname] = alleles[:, fi]
            frames.append(df)

    if not frames:
        empty_cols = ["marker", "chr", "pos_mb", "pos_bp", "locus_id"]
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(frames, ignore_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# biallelic mask
# ─────────────────────────────────────────────────────────────────────────────


def make_alt_mask(
    alleles_df: pd.DataFrame,
    founder_order: list[str],
    ref_founder: str = "B6J",
) -> np.ndarray:
    """Build a binary ALT-allele mask from founder nucleotide calls.

    For each SNP and each founder, the mask is 1 when the founder's call
    differs from the reference founder's call and is not missing, and 0
    otherwise.  The reference founder's own column is always 0.

    Intended for chip-format allele DataFrames returned by
    ``load_founder_alleles``, where each founder column holds a single
    nucleotide character.  Rows with a missing reference call are
    excluded by setting the entire row to 0; callers should filter those
    SNPs before passing to ``collapse_to_dosage``.

    Parameters
    ----------
    alleles_df : pd.DataFrame
        Output of :func:`load_founder_alleles`.  Must contain one column
        per founder in *founder_order*.
    founder_order : list[str]
        Canonical founder names whose order matches axis-2 of the
        haplotype probability tensor returned by
        ``GenomeCache.get_chromosome``.
    ref_founder : str, optional
        Name of the founder whose allele is treated as REF.  Must appear
        in both *founder_order* and *alleles_df*.  Defaults to ``'B6J'``,
        the C57BL/6J reference genome strain.

    Returns
    -------
    np.ndarray
        Float64 array of shape ``(n_snps, n_founders)`` with values in
        ``{0.0, 1.0}``.  Axis-1 order matches *founder_order*.

    Raises
    ------
    ValueError
        If *ref_founder* is not in *founder_order* or not in *alleles_df*.
    ValueError
        If any founder in *founder_order* is missing from *alleles_df*.
    """
    missing_cols = [f for f in founder_order if f not in alleles_df.columns]
    if missing_cols:
        raise ValueError(f"founders not found in alleles_df columns: {missing_cols}")
    if ref_founder not in founder_order:
        raise ValueError(f"ref_founder {ref_founder!r} not in founder_order")
    if ref_founder not in alleles_df.columns:
        raise ValueError(f"ref_founder {ref_founder!r} not in alleles_df columns")

    _MISSING_VALS: list[str] = ["N", "H", "", "nan", "-"]
    n_snps = len(alleles_df)
    ref_idx = founder_order.index(ref_founder)

    # extract all founder columns in one pass → (n_snps, n_founders)
    all_calls = alleles_df[founder_order].to_numpy(dtype=str)
    ref_calls = all_calls[:, ref_idx]  # (n_snps,)

    # missing and diff masks computed over the full matrix in single calls
    is_missing = np.isin(all_calls, _MISSING_VALS)  # (n_snps, n_founders)
    ref_missing = is_missing[:, ref_idx]  # (n_snps,)
    is_diff = all_calls != ref_calls[:, np.newaxis]  # (n_snps, n_founders)

    alt_mask = np.where(
        ref_missing[:, np.newaxis] | is_missing,
        0.0,
        is_diff.astype(np.float64),
    )
    alt_mask[:, ref_idx] = 0.0  # ref founder is always 0

    if ref_missing.any():
        logger.debug(
            "make_alt_mask: %d/%d SNPs have missing ref call — row set to 0",
            int(ref_missing.sum()),
            n_snps,
        )

    return alt_mask


# ── flat-file allele I/O ──────────────────────────────────────────────────────

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

    try:
        import rpy2.robjects as ro
    except ImportError:
        raise ImportError("rpy2 required for reading allele_dir .RData files")

    rows = []
    chr_dirs = sorted(
        glob.glob(os.path.join(allele_dir, "chr*")),
        key=lambda p: chr_sort_key(os.path.basename(p).replace("chr", "")),
    )
    for chr_dir in chr_dirs:
        c = os.path.basename(chr_dir).replace("chr", "")
        if chrom is not None and c != str(chrom):
            continue
        rdata_files = sorted(glob.glob(os.path.join(chr_dir, "*.RData")))
        for rf in rdata_files:
            snp_id = os.path.splitext(os.path.basename(rf))[0]
            env = ro.r["new.env"]()
            ro.r["load"](rf, envir=env)
            obj = ro.r["get"](ro.r["ls"](envir=env)[0], envir=env)
            arr = np.array(obj, dtype=float).ravel()
            # arr length = n_founders; map to founder names
            row = {"snp_id": snp_id, "chr": c, "pos_mb": np.nan}
            for fi, f in enumerate(founders):
                row[f] = float(arr[fi]) if fi < len(arr) else np.nan
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["chr", "pos_mb"] + list(founders))
    df = pd.DataFrame(rows).set_index("snp_id")
    logger.info(f"read_allele_dir: {len(df)} SNPs from {allele_dir}")
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
    try:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri

        numpy2ri.activate()
    except ImportError:
        raise ImportError("rpy2 required for building allele_dir .RData files")

    os.makedirs(allele_dir, exist_ok=True)
    founder_cols = [c for c in snp_genos.columns if c not in ("chr", "pos_mb")]
    n_written = 0

    for snp_id, row in snp_genos.iterrows():
        c = str(row.get("chr", "unknown"))
        chr_dir = os.path.join(allele_dir, f"chr{c}")
        os.makedirs(chr_dir, exist_ok=True)
        rdata_path = os.path.join(chr_dir, f"{snp_id}.RData")
        if os.path.exists(rdata_path):
            continue
        geno_vec = np.array([float(row.get(f, 0)) for f in founder_cols])
        rv = ro.FloatVector(geno_vec)
        rv.names = ro.StrVector(founder_cols)
        env = ro.r["new.env"]()
        env[snp_id] = rv
        ro.r["save"](ro.StrVector([snp_id]), file=rdata_path, envir=env)
        n_written += 1

    logger.info(f"build_allele_dir: wrote {n_written} SNP files to {allele_dir}")
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
        _glob.glob(os.path.join(allele_dir, "chr*.alleles")),
        key=lambda p: chr_sort_key(
            os.path.basename(p).replace("chr", "").replace(".alleles", "")
        ),
    ):
        c = os.path.basename(fpath).replace("chr", "").replace(".alleles", "")
        if chrom is not None and c != str(chrom):
            continue
        with open(fpath) as fh:
            lines = [ln.rstrip("\n") for ln in fh]
        for i in range(0, len(lines) - 2, 4):  # 4 lines per SNP block
            snp_id = lines[i].strip()
            alt_parts = lines[i + 2].split("\t")  # line 3: alt allele row
            alt_vals = alt_parts[1:]  # drop allele label col
            row = {"chr": c}
            for fi, f in enumerate(founders):
                try:
                    row[f] = float(alt_vals[fi]) if fi < len(alt_vals) else 0.0
                except (ValueError, TypeError):
                    row[f] = 0.0
            snp_ids.append(snp_id)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["chr", "pos_mb"] + list(founders))

    df = pd.DataFrame(rows, index=snp_ids)
    df.index.name = "snp_id"
    df["pos_mb"] = np.nan

    # attach positions from GM_pmap*.csv if pmap_dir provided
    if pmap_dir is not None:
        pmap_files = sorted(_glob.glob(os.path.join(pmap_dir, "GM_pmap*.csv")))
        frames = []
        for pf in pmap_files:
            try:
                pm = pd.read_csv(pf, comment="#")
                if "marker" in pm.columns:
                    pm = pm.rename(columns={"marker": "locus", "pos": "pos_mb"})
                frames.append(pm[["locus", "pos_mb"]])
            except Exception:
                pass
        if frames:
            pmap_all = pd.concat(frames).set_index("locus")
            shared = df.index.intersection(pmap_all.index)
            df.loc[shared, "pos_mb"] = pmap_all.loc[shared, "pos_mb"].values

    logger.info(
        f'read_alleles_flat: {len(df)} SNPs  pos_filled={df["pos_mb"].notna().sum()}'
    )
    return df


def build_snp_catalogue(
    scan_results, genomecache, mgp_vcf, window_mb=10.0, founders=None
):
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
        raise ImportError("build_snp_catalogue requires pysam: pip install pysam")

    founders_use = founders or list(genomecache.founders)
    vcf = pysam.VariantFile(mgp_vcf)
    results = {}

    for gid, sr in scan_results.items():
        if not len(sr.lod):
            continue
        peak = sr.peak_locus()
        chrom, pos_mb = peak["chromosome"], peak["position"]

        lo = max(0.0, pos_mb - window_mb) * 1e6
        hi = (pos_mb + window_mb) * 1e6
        chr_str = f"chr{chrom}" if not str(chrom).startswith("chr") else str(chrom)

        rows = []
        try:
            for rec in vcf.fetch(chr_str, int(lo), int(hi)):
                if len(rec.alts) != 1:  # biallelic only
                    continue
                gts = {}
                for f in founders_use:
                    s = rec.samples.get(f)
                    gts[f] = int(s["GT"][0]) if s and s["GT"][0] is not None else None
                # count divergent founders (not all same genotype)
                called = [v for v in gts.values() if v is not None]
                n_div = len(set(called)) - 1 if called else 0
                if n_div == 0:
                    continue
                row = {
                    "snp_id": rec.id or f"{chrom}:{rec.pos}",
                    "chr": chrom,
                    "pos_mb": rec.pos / 1e6,
                    "ref": rec.ref,
                    "alt": rec.alts[0],
                    "n_divergent_founders": n_div,
                }
                row.update({f: gts[f] for f in founders_use})
                rows.append(row)
        except (ValueError, KeyError):
            logger.warning(f"build_snp_catalogue: region fetch failed for {gid}")
            continue

        results[gid] = pd.DataFrame(rows) if rows else pd.DataFrame()

    vcf.close()
    return results

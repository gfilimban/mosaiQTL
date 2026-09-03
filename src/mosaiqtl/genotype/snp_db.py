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
from pathlib import Path
from typing import TYPE_CHECKING

import h5py
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .genomecache import GenomeCache
    from ..models.config import Population

logger = logging.getLogger("mosaiqtl")

__all__ = ["build_founder_allele_db", "load_founder_alleles"]


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


def _n_alleles(row: np.ndarray) -> int:
    """count how many distinct alleles a founder row carries (ignoring missing calls)."""
    return len({a for a in row if a not in ("N", "H", "")})


# ─────────────────────────────────────────────────────────────────────────────
# build
# ─────────────────────────────────────────────────────────────────────────────


def build_founder_allele_db(
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

    logger.info("build_founder_allele_db: %d founders, reading %s", len(fcols), gm_csv)
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
    founder_calls = raw[fcols].values
    n_alleles = np.array(
        [_n_alleles(founder_calls[i]) for i in range(len(founder_calls))]
    )
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

            lids_cache, pos_cache, _ = genomecache.get_chromosome(chrom)
            snp_pos = chr_snps["pos_mb"].to_numpy()

            # find the nearest cache marker for each SNP position; cache
            # positions aren't guaranteed sorted, so sort first and map back
            sort_ord = np.argsort(pos_cache)
            pos_sorted = pos_cache[sort_ord]

            idx_r = np.searchsorted(pos_sorted, snp_pos).clip(0, len(pos_sorted) - 1)
            idx_l = (idx_r - 1).clip(0, len(pos_sorted) - 1)
            dist_r = np.abs(pos_sorted[idx_r] - snp_pos)
            dist_l = np.abs(pos_sorted[idx_l] - snp_pos)
            best_sorted = np.where(dist_l <= dist_r, idx_l, idx_r)
            best_dist = np.minimum(dist_l, dist_r)
            # translate positions in the sorted order back to the cache's
            # original marker order
            best = sort_ord[best_sorted]

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
                data=np.array(chr_snps["marker"].tolist(), dtype=object),
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
            alleles = np.array(
                chr_snps[fcols].to_numpy().astype(str).tolist(), dtype=object
            )
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
        "build_founder_allele_db: done — %d/%d SNPs written to %s",
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
    h5_path : path to HDF5 built by build_founder_allele_db()
    chrom   : chromosome key (e.g. 'chr10').  if None, loads all chromosomes.

    returns
    -------
    pd.DataFrame  columns: marker, chr, pos_mb, pos_bp, locus_id,
                            <founder_0>, ..., <founder_J-1>
        founder column names match the population used at build time.
    """
    h5_path = Path(h5_path)
    frames = []

    with h5py.File(h5_path, "r") as hf:
        # founder order from metadata — population-agnostic
        fcols = [
            f.decode() if isinstance(f, bytes) else f
            for f in np.array(hf["metadata"]["founders"])
        ]
        n_f = len(fcols)

        def _decode(arr: np.ndarray) -> np.ndarray:
            """decode bytes elements to str, leaving str elements unchanged."""
            return np.array(
                [v.decode() if isinstance(v, bytes) else v for v in arr]
            ).astype(str)

        chroms = (
            [chrom] if chrom is not None else [k for k in hf.keys() if k != "metadata"]
        )

        for c in chroms:
            if c not in hf:
                raise KeyError(f"chromosome {c!r} not found in {h5_path}")
            grp = hf[c]
            n = grp["marker"].shape[0]
            alleles_raw = np.array(grp["alleles"]).reshape(n, n_f)
            # h5py may return vlen strings as bytes; decode to str
            alleles = np.array(
                [
                    [v.decode() if isinstance(v, bytes) else v for v in row]
                    for row in alleles_raw
                ],
                dtype=object,
            )
            df = pd.DataFrame(
                {
                    "marker": _decode(np.array(grp["marker"])),
                    "chr": c,
                    "pos_mb": np.array(grp["pos_mb"]).astype(float),
                    "pos_bp": np.array(grp["pos_bp"]).astype(int),
                    "locus_id": _decode(np.array(grp["locus_id"])),
                }
            )
            for fi, fname in enumerate(fcols):
                df[fname] = alleles[:, fi]
            frames.append(df)

    if not frames:
        empty_cols = ["marker", "chr", "pos_mb", "pos_bp", "locus_id"]
        return pd.DataFrame(columns=empty_cols)
    return pd.concat(frames, ignore_index=True)

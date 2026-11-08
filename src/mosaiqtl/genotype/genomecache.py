"""
genomecache — read and manage founder haplotype probabilities

supports two formats:
  1. happy genome cache (directory of per-locus .rds files) — read via rpy2
  2. native hdf5 cache (single file per chromosome) — fast python-native

the hdf5 format stores:
  /chr{c}/locus_ids    — string array of marker names
  /chr{c}/positions    — float array of physical positions (bp)
  /chr{c}/probs        — float array (n_loci, n_subjects, n_founders)
  /chr{c}/sample_ids   — string array of subject identifiers
  /founders            — string array of 8 founder strain names
"""

import numpy as np
import os
import glob
import logging

from ..utils import chr_sort_key

logger = logging.getLogger("mosaiqtl")

# default cc founder strain names (in standard order)
from .cc import FOUNDER_ORDER as _FOUNDER_ORDER, F1_COLORS as _F1_COLORS

CC_FOUNDERS: list[str] = list(_FOUNDER_ORDER)


def _encode_abh_vec(
    geno: np.ndarray,
    a1: np.ndarray,
    a2: np.ndarray,
    poly: np.ndarray,
    miss_codes: list,
) -> np.ndarray:
    """Vectorized A/B/H/- encoding over a (n_markers, n_col) string array.

    Parameters
    ----------
    geno : np.ndarray of dtype object, shape (n_m, n_col)
    a1, a2 : (n_m,) reference allele arrays (from founder-derived allele codes)
    poly : (n_m,) bool — markers with two distinct founder alleles
    miss_codes : list of strings treated as missing ('N', '-', 'nan', '')
    """
    miss = np.isin(geno, miss_codes)
    h = geno == "H"
    out = np.full(geno.shape, "-", dtype=object)
    a1e, a2e, pe = a1[:, None], a2[:, None], poly[:, None]
    valid = pe & ~miss
    out = np.where(valid & ~h & (geno == a1e), "A", out)
    out = np.where(valid & ~h & (geno == a2e), "B", out)
    out = np.where(valid & h, "H", out)
    return out


def _gigamuga_chr_worker(task: dict) -> dict:
    """ProcessPoolExecutor worker: run the diplotype HMM on one chromosome.

    Receives pre-encoded A/B/H/- genotype arrays and returns the
    per-chromosome probs dict ready for GenomeCache. Population-agnostic:
    n_gen and inbred come from the task dict set by from_gigamuga.
    """
    from .hmm import calc_genoprob_cc  # imported in worker process

    chrom = task["chrom"]
    geno_cc = task["geno_cc"]  # (n_m, n_samples) object
    geno_f = task["geno_f"]  # (n_m, J) object
    gmap_pos = task["gmap_pos"]  # (n_m,) cM
    pmap_pos = task["pmap_pos"]  # (n_m,) Mb
    locus_ids = task["locus_ids"]  # (n_m,) str
    n_founders = task["n_founders"]
    n_gen = task["n_gen"]
    inbred = task["inbred"]
    error_prob = task["error_prob"]
    cc_cols = task["cc_cols"]
    founder_cols = task["founder_cols"]
    f1_crosses = task["f1_crosses"]
    all_48 = task["all_48"]
    step_cM = task["step_cM"]

    result = calc_genoprob_cc(
        geno_cc,
        geno_f,
        gmap_pos,
        n_founders=n_founders,
        n_gen=n_gen,
        error_prob=error_prob,
        return_additive=True,
        cc_inbred=inbred,
        step_cM=step_cM,
    )

    if step_cM is not None:
        probs_cc, aug_gmap_cM, is_real = result
        aug_pmap_pos = np.interp(aug_gmap_cM, gmap_pos, pmap_pos)
        _digits = max(0, int(np.ceil(-np.log10(step_cM))))
        real_iter = iter(locus_ids)
        aug_locus_ids = np.array(
            [
                next(real_iter) if ir else f"c{chrom}.loc{aug_gmap_cM[k]:.{_digits}f}"
                for k, ir in enumerate(is_real)
            ]
        )
        n_aug = len(aug_gmap_cM)
    else:
        probs_cc = result
        aug_pmap_pos = pmap_pos
        aug_locus_ids = locus_ids
        n_aug = len(locus_ids)

    n_strains = len(all_48)
    probs_chr = np.zeros((n_strains, n_founders, n_aug), dtype=np.float32)

    for i, col in enumerate(cc_cols):
        probs_chr[all_48.index(col)] = probs_cc[i].astype(np.float32)
    for fi, fcol in enumerate(founder_cols):
        probs_chr[all_48.index(fcol), fi, :] = 1.0
    for sample_id, pi1, pi2 in f1_crosses:
        probs_chr[all_48.index(sample_id), pi1, :] = 0.5
        probs_chr[all_48.index(sample_id), pi2, :] = 0.5

    return {
        "chrom": chrom,
        "data": {
            "locus_ids": aug_locus_ids,
            "positions": aug_pmap_pos,
            # (n_loci, n_strains, J) C-contiguous
            "probs": np.ascontiguousarray(np.transpose(probs_chr, (2, 0, 1))),
        },
    }


class GenomeCache:
    """in-memory or lazy-loaded haplotype probability store.

    attributes
    ----------
    chromosomes : list of str
        available chromosome identifiers
    sample_ids : np.ndarray of str
        subject identifiers (consistent across all loci)
    founders : list of str
        founder strain names (length 8 for CC)
    n_founders : int
        number of founder strains
    """

    def __init__(self, sample_ids, founders, data=None, _h5_handle=None, _h5_path=None):
        """initialize genome cache.

        parameters
        ----------
        sample_ids : array-like of str
            subject identifiers
        founders : list of str
            founder strain names
        data : dict or None
            {chr_id: {'locus_ids': array, 'positions': array,
                       'probs': ndarray or h5py.Dataset (n_loci, n_subjects, n_founders)}}
        _h5_handle : h5py.File or None
            open HDF5 file kept alive for lazy probs reads; closed in __del__
        _h5_path : str or None
            path to the source HDF5 file; stored so thread workers can open
            independent handles via GenomeCache.from_hdf5(_h5_path)
        """
        self.sample_ids = np.asarray(sample_ids)
        self.founders = list(founders)
        self.n_founders = len(founders)
        self._data = data if data is not None else {}
        self._h5_handle = _h5_handle
        self._h5_path = str(_h5_path) if _h5_path is not None else None
        self.crossinfo = None  # DataFrame (strains × A-H) from cc_crossinfo.csv
        self.x_donor = {}  # strain_id -> founder_col of x-chr donor
        self.chromosomes = sorted(self._data.keys(), key=chr_sort_key)

    def __del__(self):
        """close open HDF5 handle on garbage collection."""
        h = getattr(self, "_h5_handle", None)
        if h is not None:
            try:
                h.close()
            except OSError:
                pass

    def get_locus(self, chrom, locus_idx):
        """retrieve haplotype probabilities for a single locus.

        parameters
        ----------
        chrom : str
            chromosome identifier
        locus_idx : int
            index into the chromosome's locus array

        returns
        -------
        np.ndarray (n_subjects, n_founders)
            founder haplotype probabilities at this locus
        """
        return np.asarray(self._data[chrom]["probs"][locus_idx])

    def get_locus_metadata(self, chrom: str) -> tuple:
        """Return locus_ids and positions without loading haplotype probabilities.

        Cheaper than :meth:`get_chromosome` when only marker positions are needed
        (e.g. building a locus-id → chromosome lookup).  ``locus_ids`` and
        ``positions`` are already numpy arrays in RAM from the HDF5 load; only
        ``probs`` is a lazy h5py Dataset that requires a disk read.

        Parameters
        ----------
        chrom : str
            Chromosome identifier (must be in :attr:`chromosomes`).

        Returns
        -------
        locus_ids : np.ndarray of str
        positions : np.ndarray of float
            Both guaranteed sorted ascending (same guarantee as
            :meth:`get_chromosome`).
        """
        d = self._data[chrom]
        lids = d["locus_ids"]
        pos = d["positions"]
        if "_sort_ord" in d:
            return lids[d["_sort_ord"]], pos[d["_sort_ord"]]
        if len(pos) > 1 and not np.all(pos[:-1] <= pos[1:]):
            ord_ = np.argsort(pos)
            lids = lids[ord_]
            pos = pos[ord_]
            d["locus_ids"] = lids
            d["positions"] = pos
            d["_sort_ord"] = ord_
        return lids, pos

    def get_chromosome(self, chrom):
        """retrieve all loci for a chromosome.

        loads probs into RAM at call time — one chromosome in RAM during scan,
        not all chromosomes (lazy h5py Dataset or eager numpy array both handled).

        positions are guaranteed sorted ascending on return.  HAPPY format
        stores loci in R's internal order (not genomic order); sorting is
        applied once on first access and cached so all subsequent callers
        (kinship, scan, snp_db, CI bootstrap) see consistent ordered data.

        parameters
        ----------
        chrom : str
            chromosome identifier

        returns
        -------
        locus_ids : np.ndarray of str
        positions : np.ndarray of float
        probs : np.ndarray (n_loci, n_subjects, n_founders)
        """
        d = self._data[chrom]
        lids = d["locus_ids"]
        pos = d["positions"]
        probs = np.asarray(d["probs"])  # h5py Dataset → numpy (one chr, then released)

        if len(pos) > 1 and not np.all(pos[:-1] <= pos[1:]):
            ord_ = np.argsort(pos)
            lids = lids[ord_]
            pos = pos[ord_]
            probs = probs[ord_]
            # cache sorted lids/pos so repeated callers see consistent order;
            # do NOT overwrite d['probs'] with the numpy array — keeping the
            # h5py Dataset ref means each call loads from disk into a temporary
            # buffer that is freed after this function returns, so only one
            # chromosome's probs live in RAM at a time regardless of how many
            # chromosomes have been accessed (e.g. during LOCO kinship).
            d["locus_ids"] = lids
            d["positions"] = pos
            d["_sort_ord"] = ord_  # apply on subsequent calls instead
        elif "_sort_ord" in d:
            probs = probs[d["_sort_ord"]]

        return lids, pos, probs

    def n_loci(self, chrom):
        """number of loci on a chromosome."""
        return len(self._data[chrom]["locus_ids"])

    @staticmethod
    def from_hdf5(filepath):
        """load genome cache from native hdf5 format.

        parameters
        ----------
        filepath : str
            path to .h5 file

        returns
        -------
        GenomeCache
        """
        import h5py

        # file kept open — probs Datasets are lazy; get_chromosome() loads one
        # chromosome at a time (n_loci × n_strains × J × 8 B) rather than the
        # full genome simultaneously
        f = h5py.File(filepath, "r")
        founders = [s.decode() for s in f["founders"][:]]
        sample_ids = None
        data = {}

        for key in f.keys():
            if not key.startswith("chr"):
                continue
            grp = f[key]
            chrom = key.replace("chr", "")

            if sample_ids is None:
                sample_ids = np.array([s.decode() for s in grp["sample_ids"][:]])

            data[chrom] = {
                "locus_ids": np.array([s.decode() for s in grp["locus_ids"][:]]),
                "positions": grp["positions"][:].astype(np.float64),
                "probs": grp["probs"],  # h5py Dataset — NOT loaded into RAM
            }

        return GenomeCache(sample_ids, founders, data, _h5_handle=f, _h5_path=filepath)

    def to_hdf5(self, filepath):
        """save genome cache to native hdf5 format.

        parameters
        ----------
        filepath : str
            output path for .h5 file
        """
        import h5py

        with h5py.File(filepath, "w") as f:
            # store founder names as root attribute
            f.create_dataset("founders", data=np.array(self.founders, dtype="S"))

            for chrom in self.chromosomes:
                grp = f.create_group(f"chr{chrom}")

                # store sample ids
                grp.create_dataset(
                    "sample_ids", data=np.array(self.sample_ids, dtype="S")
                )

                d = self._data[chrom]

                # store locus metadata
                grp.create_dataset(
                    "locus_ids", data=np.array(d["locus_ids"], dtype="S")
                )
                grp.create_dataset("positions", data=d["positions"])

                # store probability tensor with compression
                grp.create_dataset(
                    "probs", data=d["probs"], compression="gzip", compression_opts=4
                )

        logger.info(f"saved genome cache to {filepath}")

    @staticmethod
    def from_qtl2_probs(probs_dict, map_dict, sample_ids, founders=None):
        """construct genome cache from probability arrays in the alternate axis order.

        this is the bridge for haplotype probabilities already computed
        elsewhere and exported as numpy arrays.

        parameters
        ----------
        probs_dict : dict
            {chrom: np.ndarray (n_subjects, n_loci, n_founders)}
            note: subjects-loci-founders axis order, not this package's own
        map_dict : dict
            {chrom: {'locus_ids': array, 'positions': array}}
        sample_ids : array-like of str
        founders : list of str or None
            defaults to CC_FOUNDERS if None

        returns
        -------
        GenomeCache
        """
        if founders is None:
            founders = CC_FOUNDERS

        data = {}
        for chrom, probs_3d in probs_dict.items():
            # reorder axes to this package's convention (loci, subjects, founders)
            probs_transposed = np.transpose(probs_3d, (1, 0, 2))

            data[chrom] = {
                "locus_ids": np.asarray(map_dict[chrom]["locus_ids"]),
                "positions": np.asarray(map_dict[chrom]["positions"], dtype=np.float64),
                "probs": probs_transposed,
            }

        return GenomeCache(sample_ids, founders, data)

    @staticmethod
    def from_gigamuga(
        gigamuga_path,
        gmap_dir,
        pmap_dir,
        cc_cols,
        founder_cols,
        f1_crosses=None,
        n_founders=8,
        n_gen=1,
        error_prob=0.002,
        inbred=True,
        just_these_loci=None,
        qc=None,
        crossinfo_path=None,
        step_cM=None,
    ):
        """build genome cache from raw GigaMUGA genotype data.

        population-agnostic: works for CC (inbred=True, n_gen=1), DO
        (inbred=False, n_gen=<breeding_gen>), and custom MPPs. the caller
        supplies sample_cols, founder_cols, and optionally f1_crosses; the
        HMM parameters n_gen and inbred are passed through to calc_genoprob_cc.

        pipeline:
          gigamuga CSV → encode A/H/B (vectorised, J passes) → HMM (forward-backward,
          parallel per chromosome) → additive probs → deterministic founder arrays
          (homozygous, prob=1) → deterministic F1 arrays (prob=0.5 each parent)
          → GenomeCache(n_samples × n_founders × n_loci)

        this is the Python equivalent of founders_f1_genomecache.ipynb
        steps 1-7 without calling any R functions.

        parameters
        ----------
        gigamuga_path : str
            path to GigaMUGA consensus matrix CSV (markers × strains)
        gmap_path : str
            path to gmap.csv (marker, chr, pos_cM)
        pmap_path : str
            path to pmap.csv (marker, chr, pos_Mb)
        cc_cols : list of str
            CC strain column names in gigamuga_path
        founder_cols : list of str
            founder strain column names (length J, in founder order)
        f1_crosses : list of (sample_id, parent1_idx, parent2_idx) tuples
            e.g. [('B6AF1J', 1, 0), ...] where indices are into founder_cols
        n_founders : int
            number of founders J (8 for CC)
        n_gen : int
            transition scaling factor passed to calc_genoprob_cc. must be 1
            for CC with the GigaMUGA consensus map (map already encodes
            cumulative recombination history; r used directly). default 1.
        error_prob : float
            GigaMUGA genotyping error rate (0.002)
        just_these_loci : list of str or None
            restrict to this exact marker set (e.g. HAPPY cache markers for
            cross-language validation). applied after the 3-way intersection.
        qc : dict or None
            user-driven QC filters. keys: min_maf (float), max_missing (float).
            example: qc={'min_maf': 0.05, 'max_missing': 0.10}
        crossinfo_path : str or None
            path to cc_crossinfo.csv (columns A-H = funnel positions 1-8 per
            CC strain). if provided, cc_inbred=True is used for all CC strains
            in the crossinfo (homozygous-only HMM prior). if None, cc_inbred
            defaults to True for all CC strains anyway (safe default).
            also stored as cache.crossinfo for downstream use (X-chr analysis).

        returns
        -------
        GenomeCache  with 48 samples, 8 founders, all chromosomes
        """
        import glob
        import pandas as pd

        logger.info("building genome cache from raw GigaMUGA...")

        # crossinfo columns give each founder's position in the breeding
        # funnel per strain (for CC, columns A-H map to funnel positions
        # 1-8); used to confirm each strain is fully inbred and to identify
        # which founder donated the X chromosome
        _crossinfo = None
        _x_donor = {}  # strain_id -> founder_col of x-chr donor (funnel pos 8)
        if crossinfo_path is not None and os.path.exists(crossinfo_path):
            _crossinfo = pd.read_csv(crossinfo_path, index_col=0)
            for sid, row in _crossinfo.iterrows():
                # the founder at the last funnel position donated the X chromosome
                x_donor_code = row[row == 8].index[0]
                _x_donor[sid] = x_donor_code
            logger.info(
                f"crossinfo: loaded {len(_crossinfo)} strains; "
                f"{len(_x_donor)} x-chr donors identified"
            )
        # CC strains are treated as fully inbred (homozygous prior in the HMM);
        # founders and F1s are assigned directly without running the HMM at all

        gm_raw = pd.read_csv(gigamuga_path, low_memory=False)
        if "marker" in gm_raw.columns:
            gm_raw = gm_raw.set_index("marker")
        # drop chr/pos columns if present; the dedicated map files are used instead
        gm_raw = gm_raw.drop(columns=[c for c in ("chr", "pos") if c in gm_raw.columns])
        gm_raw = gm_raw.replace("N", "-")  # missing call code

        # accepts either one combined file (gmap.csv / pmap.csv, preferred) or
        # one file per chromosome (GM_gmap{chr}.csv, older layout)
        def _load_maps(map_dir, prefix, flat_name):
            """Load flat or per-chr map CSV; prefer flat_name, fall back to prefix*.csv."""
            flat = os.path.join(map_dir, flat_name)
            if os.path.exists(flat):
                return pd.read_csv(flat, comment="#")
            files = sorted(glob.glob(os.path.join(map_dir, f"{prefix}*.csv")))
            if not files:
                raise FileNotFoundError(f"no {flat_name} or {prefix}*.csv in {map_dir}")
            return pd.concat(
                [pd.read_csv(fp, comment="#") for fp in files], ignore_index=True
            )

        gmap = _load_maps(gmap_dir, "GM_gmap", "gmap.csv")
        pmap = _load_maps(pmap_dir, "GM_pmap", "pmap.csv")
        gmap.rename(
            columns={"marker": "locus", "pos": "pos_cM"}, inplace=True, errors="ignore"
        )
        pmap.rename(
            columns={"marker": "locus", "pos": "pos_mb"}, inplace=True, errors="ignore"
        )

        # chromosome must always be a string — comparing int 1 to string '1'
        # silently fails and would drop that chromosome's markers
        gmap["chr"] = gmap["chr"].astype(str)
        pmap["chr"] = pmap["chr"].astype(str)
        logger.info(f"gmap: {len(gmap)} loci | pmap: {len(pmap)} loci")

        # markers must end up in genomic order, not arbitrary set order: the
        # HMM computes recombination between adjacent markers, so if they're
        # out of order it scrambles haplotype blocks and corrupts every
        # downstream result (kinship, LOD)
        shared_set = set(gm_raw.index) & set(gmap["locus"]) & set(pmap["locus"])
        gmap_idx = gmap.set_index("locus")
        gmap_shared = gmap_idx.loc[list(shared_set)].copy()

        gmap_shared["_k"] = gmap_shared["chr"].map(chr_sort_key)
        gmap_shared = gmap_shared.sort_values(["_k", "pos_cM"]).drop(columns=["_k"])
        shared = gmap_shared.index.tolist()

        gm = gm_raw.loc[shared]
        gmap = gmap_shared
        pmap_idx = pmap.set_index("locus").loc[shared]
        logger.info(f"shared markers: {len(shared)}")

        if just_these_loci is not None:
            allow = set(just_these_loci)
            shared = [m for m in shared if m in allow]
            logger.info(f"just_these_loci filter: {len(shared)} markers retained")

        # markers that are identical across all founders (uninformative for
        # haplotype reconstruction) are already excluded upstream, so no
        # extra filtering is needed here unless the caller passes qc options
        if qc:
            keep = pd.Series(True, index=shared)
            if "max_missing" in qc:
                all_cols = [
                    c for c in list(founder_cols) + cc_cols if c in gm_raw.columns
                ]
                miss = (gm_raw.loc[shared, all_cols] == "-").mean(axis=1)
                keep &= miss <= qc["max_missing"]
            n_before = len(shared)
            shared = [m for m in shared if keep[m]]
            if n_before > len(shared):
                logger.info(
                    f"user qc: {n_before} -> {len(shared)} markers "
                    f"(dropped {n_before - len(shared)})"
                )

        # re-slice gm/gmap/pmap_idx to post-filter shared set
        gm = gm_raw.loc[shared]
        gmap = gmap.loc[shared]
        pmap_idx = pmap_idx.loc[shared]
        logger.info(f"post-filter markers: {len(shared)}")

        chrs = sorted(
            [c for c in gmap["chr"].unique() if c not in ("Y", "M", "MT", "nan")],
            key=chr_sort_key,
        )

        if f1_crosses is None:
            f1_crosses = []

        # encode every marker's genotype calls as A/B/H relative to the two
        # founder alleles, for founders and CC strains at once (J passes over
        # all markers instead of looping marker-by-marker)
        # axis order: founders → F1s → CC strains (alphanumeric) — inviolable
        all_48 = list(founder_cols) + [f[0] for f in f1_crosses] + cc_cols

        MISS_CODES = {"N", "-", "nan", ""}
        MISS_LIST = list(MISS_CODES | {"H"})  # excluded when finding a1/a2

        f_str = gm[list(founder_cols)].values.astype(str)
        cc_str = gm[cc_cols].values.astype(str)

        # loop once per founder rather than once per marker, so this scales
        # with founder count instead of with the (much larger) marker count
        valid_f = ~np.isin(f_str, MISS_LIST)
        a1_all = np.full(len(shared), "", dtype=object)
        a2_all = np.full(len(shared), "", dtype=object)
        for fi in range(n_founders):
            col = f_str[:, fi]
            need1 = valid_f[:, fi] & (a1_all == "")
            a1_all = np.where(need1, col, a1_all)
            need2 = valid_f[:, fi] & (a1_all != "") & (a2_all == "") & (col != a1_all)
            a2_all = np.where(need2, col, a2_all)
        poly_all = (a1_all != "") & (a2_all != "")

        miss_list = list(MISS_CODES)
        geno_f_all = _encode_abh_vec(f_str, a1_all, a2_all, poly_all, miss_list)
        geno_cc_all = _encode_abh_vec(cc_str, a1_all, a2_all, poly_all, miss_list)

        gmap_chr_col = gmap["chr"]
        gmap_pos_all = gmap["pos_cM"].values.astype(float)
        gmap_index_arr = np.array(gmap.index.tolist())

        tasks = []
        for chrom in chrs:
            mask = (gmap_chr_col == chrom).values
            if not mask.any():
                continue
            loci = gmap_index_arr[mask]
            tasks.append(
                {
                    "chrom": chrom,
                    "geno_cc": geno_cc_all[mask],
                    "geno_f": geno_f_all[mask],
                    "gmap_pos": gmap_pos_all[mask],
                    "pmap_pos": pmap_idx.loc[loci, "pos_mb"].values.astype(float),
                    "locus_ids": loci,
                    "n_founders": n_founders,
                    "n_gen": n_gen,
                    "inbred": inbred,
                    "error_prob": error_prob,
                    "cc_cols": cc_cols,
                    "founder_cols": list(founder_cols),
                    "f1_crosses": f1_crosses,
                    "all_48": all_48,
                    "step_cM": step_cM,
                }
            )

        # run the HMM for each chromosome in its own process, in parallel
        from concurrent.futures import ProcessPoolExecutor

        n_proc = min(len(tasks), os.cpu_count() or 4)
        logger.info(
            f"running HMM on {len(tasks)} chromosomes "
            f"(n_gen={n_gen}, inbred={inbred}, {n_proc} workers)..."
        )
        if n_proc > 1:
            with ProcessPoolExecutor(max_workers=n_proc) as pool:
                results = list(pool.map(_gigamuga_chr_worker, tasks))
        else:
            results = [_gigamuga_chr_worker(t) for t in tasks]

        data = {r["chrom"]: r["data"] for r in results}

        cache = GenomeCache(all_48, list(founder_cols), data)
        cache.crossinfo = _crossinfo  # None if crossinfo_path not provided
        cache.x_donor = _x_donor  # {} if crossinfo_path not provided
        return cache

    @staticmethod
    def from_rds(
        probs_cc_path, probs_founder_path, probs_f1_path, pmap_path, founders=None
    ):
        """build genome cache from raw HMM probability files (.rds format).

        reads probs_cc.rds, probs_founder.rds, probs_f1.rds and pmap.csv,
        merges all 48 genotypes (33 CC + 8 founders + 7 F1s), and returns a
        GenomeCache with the same haplotype probabilities as the standard
        conversion pipeline, computed here directly rather than read from
        an already-converted file.

        input shapes:
          - probs_cc:      HMM output, shape (n_cc, n_diplotype_states, n_loci)
          - probs_founder: deterministic single-sample arrays, one per founder
          - probs_f1:      deterministic single-sample arrays, one per F1

        the 36 possible diplotype states (pairs of 8 founder haplotypes) are
        collapsed down to 8 additive founder probabilities per sample, the
        same reduction used before computing kinship.

        parameters
        ----------
        probs_cc_path      : str  path to probs_cc.rds
        probs_founder_path : str  path to probs_founder.rds
        probs_f1_path      : str  path to probs_f1.rds
        pmap_path          : str  path to pmap.csv (marker, chr, pos in Mb)
        founders           : list of str or None  defaults to CC_FOUNDERS

        returns
        -------
        GenomeCache  with 48 samples, 8 founders, all chromosomes
        """
        import rpy2.robjects as ro
        import pandas as pd

        if founders is None:
            founders = CC_FOUNDERS

        logger.info("loading raw HMM probability arrays...")

        ro.r(f'probs_cc      <- readRDS("{probs_cc_path}")')
        ro.r(f'probs_founder <- readRDS("{probs_founder_path}")')
        ro.r(f'probs_f1      <- readRDS("{probs_f1_path}")')

        # samples must be merged into one combined array before collapsing
        # diplotype states to additive founder probabilities — doing it the
        # other way around (collapsing CC samples alone first) leaves a
        # shape mismatch (8 additive columns vs 36 diplotype states) between
        # the CC array and the founder/F1 arrays
        ro.r("library(qtl2)")
        ro.r("library(abind)")
        ro.r("""
            chrs <- names(probs_cc)

            # combine all 48 samples per chromosome while still in diplotype
            # form (36 possible founder pairs)
            probs_combined_list <- lapply(chrs, function(chr) {
                cc_arr      <- probs_cc[[chr]]
                founder_arr <- do.call(abind, c(
                    lapply(probs_founder, function(x) x[[chr]]), list(along=1)))
                f1_arr      <- do.call(abind, c(
                    lapply(probs_f1,      function(x) x[[chr]]), list(along=1)))
                abind(cc_arr, founder_arr, f1_arr, along=1)
            })
            names(probs_combined_list) <- chrs

            attributes(probs_combined_list) <- attributes(probs_cc)
            names(probs_combined_list)      <- chrs

            # collapse the 36 diplotype states down to 8 additive founder
            # probabilities per sample
            probs_merged <- genoprob_to_alleleprob(probs_combined_list)
        """)

        pmap = pd.read_csv(pmap_path)
        if "marker" in pmap.columns:
            pmap = pmap.rename(columns={"marker": "locus", "pos": "pos_mb"})
        pmap["chr"] = pmap["chr"].astype(str)

        chrs_r = list(ro.r("names(probs_merged)"))
        sample_ids = list(ro.r("rownames(probs_merged[[1]])"))

        data = {}
        for chrom in chrs_r:
            chr_pmap = pmap[pmap["chr"] == chrom]
            locus_ids = np.array(chr_pmap["locus"].tolist())
            positions = np.array(chr_pmap["pos_mb"].tolist(), dtype=float)

            # source array axis order is (n_strains, n_founders, n_loci);
            # reorder to this package's convention (n_loci, n_strains, n_founders)
            arr_r = np.array(ro.r(f'probs_merged[["{chrom}"]]'))
            if arr_r.ndim == 3:
                arr = np.transpose(arr_r, (2, 0, 1))
            else:
                logger.warning(f"chr{chrom}: unexpected array shape {arr_r.shape}")
                continue

            r_loci = list(ro.r(f'dimnames(probs_merged[["{chrom}"]])[[3]]'))
            if len(r_loci) == len(locus_ids):
                pass  # marker sets already match, nothing to subset
            else:
                # keep only markers present in both the map and the prob array
                locus_set = set(r_loci)
                mask = np.array([lid in locus_set for lid in locus_ids])
                locus_ids = locus_ids[mask]
                positions = positions[mask]
                loci_idx = {lid: i for i, lid in enumerate(r_loci)}
                idx = np.array([loci_idx[lid] for lid in locus_ids if lid in loci_idx])
                arr = arr[idx]

            data[chrom] = {
                "locus_ids": locus_ids,
                "positions": positions,
                "probs": arr,
            }
            logger.info(f"  chr{chrom}: {arr.shape[0]} loci × {arr.shape[1]} samples")

        return GenomeCache(sample_ids, founders, data)

    @staticmethod
    def from_happy_dir(cache_dir, model="additive"):
        """load genome cache from happy directory format.

        real structure (from founders_f1_genomecache.ipynb):
          cache_dir/additive/chr{c}/data/*.RData
        each .rdata file contains a 48 × 8 matrix saved via save().

        requires rpy2 for reading .rdata files.

        parameters
        ----------
        cache_dir : str
            path to happy genome cache root directory
        model : str
            'additive' (j founders) or 'full' (diplotypes)

        returns
        -------
        GenomeCache
        """
        try:
            import rpy2.robjects as ro
        except ImportError:
            raise ImportError(
                "rpy2 required for reading happy .rdata files. "
                "install with: pip install rpy2"
            )

        # expected layout: cache_dir/additive/chr{c}/data/*.RData
        model_dir = os.path.join(cache_dir, model)
        if not os.path.isdir(model_dir):
            # cache_dir may already point at the model directory itself
            model_dir = cache_dir

        chr_dirs = sorted(
            glob.glob(os.path.join(model_dir, "chr*")),
            key=lambda p: chr_sort_key(os.path.basename(p).replace("chr", "")),
        )
        if not chr_dirs:
            raise FileNotFoundError(f"no chr* directories found in {model_dir}")

        data = {}
        sample_ids = None
        founders = None

        for chr_dir in chr_dirs:
            chrom = os.path.basename(chr_dir).replace("chr", "")

            data_dir = os.path.join(chr_dir, "data")
            if not os.path.isdir(data_dir):
                # files may sit directly in the chromosome directory instead
                data_dir = chr_dir

            rdata_files = sorted(
                glob.glob(os.path.join(data_dir, "*.RData"))
                + glob.glob(os.path.join(data_dir, "*.rdata"))
            )

            if not rdata_files:
                logger.warning(f"no .rdata files found in {data_dir}")
                continue

            locus_ids = []
            positions = []
            prob_list = []

            for rdata_path in rdata_files:
                locus_name = os.path.splitext(os.path.basename(rdata_path))[0]
                locus_ids.append(locus_name)
                # load into its own namespace so object names don't collide
                # across files, then pull out whatever was saved there
                env = ro.r["new.env"]()
                ro.r["load"](rdata_path, envir=env)
                obj_names = list(ro.r["ls"](envir=env))

                if not obj_names:
                    logger.warning(f"empty .rdata file: {rdata_path}")
                    continue

                # each file holds one object: the haplotype probability matrix
                mat = np.array(ro.r["get"](obj_names[0], envir=env))

                if sample_ids is None:
                    r_obj = ro.r["get"](obj_names[0], envir=env)
                    rn = ro.r["rownames"](r_obj)
                    cn = ro.r["colnames"](r_obj)
                    if rn != ro.NULL:
                        sample_ids = list(rn)
                    if cn != ro.NULL:
                        founders = list(cn)

                prob_list.append(mat)
                positions.append(0.0)

            if not prob_list:
                continue

            probs_3d = np.stack(prob_list, axis=0)

            data[chrom] = {
                "locus_ids": np.array(locus_ids),
                "positions": np.array(positions),
                "probs": probs_3d,
            }

        if sample_ids is None:
            n_subjects = probs_3d.shape[1]
            sample_ids = [f"subject_{i}" for i in range(n_subjects)]

        if founders is None:
            founders = CC_FOUNDERS

        return GenomeCache(sample_ids, founders, data)

    def collapse(self, criterion="l2.norm", tol=0.02):
        """collapse adjacent redundant loci into a single representative locus.

        merges consecutive loci whose haplotype probability matrices are
        nearly identical, reducing cache size without information loss.

        algorithm:
          for each chromosome, scan adjacent locus pairs. if their
          euclidean distance (criterion='euclidean') or most-likely-allele
          identity (criterion='allele') is within tol, merge by averaging
          their probability matrices. the merged locus takes the id and
          position of the first locus in the run.

        parameters
        ----------
        criterion : str  'l2.norm' (default) or 'allele'.
        tol : float  merge threshold (euclidean l2 norm, or fraction of
                     strains with different most-likely allele).

        returns
        -------
        GenomeCache  new cache with fewer loci per chromosome
        """
        new_data = {}
        total_before = total_after = 0

        for chrom in self.chromosomes:
            lids, pos, probs = self.get_chromosome(chrom)
            n_loci = len(lids)
            total_before += n_loci

            keep_lids, keep_pos, keep_probs = [], [], []
            run_probs = [probs[0]]  # loci accumulated so far in the current run
            run_lid = lids[0]
            run_pos = pos[0]

            n_founders = probs.shape[2]
            for i in range(1, n_loci):
                if criterion in ("l2.norm", "euclidean"):
                    # per-strain distance between this locus and the last one
                    # kept in the run; merge only if every strain is close
                    diff = probs[i] - run_probs[-1]
                    l2 = np.linalg.norm(diff, axis=1) / np.sqrt(n_founders)
                    d = float(l2.max())
                else:
                    # fraction of strains whose most likely founder changed
                    d = float(
                        np.mean(probs[i].argmax(axis=1) != run_probs[-1].argmax(axis=1))
                    )

                if d <= tol:
                    run_probs.append(probs[i])
                else:
                    # close out the run: average its loci into one representative
                    # locus, keeping the first locus's id and position
                    keep_lids.append(run_lid)
                    keep_pos.append(run_pos)
                    keep_probs.append(np.mean(run_probs, axis=0))
                    run_probs = [probs[i]]
                    run_lid = lids[i]
                    run_pos = pos[i]

            keep_lids.append(run_lid)
            keep_pos.append(run_pos)
            keep_probs.append(np.mean(run_probs, axis=0))

            n_kept = len(keep_lids)
            total_after += n_kept
            logger.info(
                f"  chr{chrom}: {n_loci} → {n_kept} loci "
                f"({100*(n_loci-n_kept)/n_loci:.1f}% reduced)"
            )

            new_data[chrom] = {
                "locus_ids": np.array(keep_lids),
                "positions": np.array(keep_pos, dtype=float),
                "probs": np.stack(keep_probs),
            }

        logger.info(
            f"collapse: {total_before} → {total_after} total loci "
            f"({100*(total_before-total_after)/total_before:.1f}% reduced)"
        )
        return GenomeCache(list(self.sample_ids), list(self.founders), new_data)


# pheno_df — pseudobulk to per-strain phenotype matrix


def pheno_df(
    h5ad_path, strain_col="strain", layer=None, strains=None, min_cells=5, out_path=None
):
    """Average pseudobulked AnnData per strain to produce a phenotype matrix.

    Reads a pseudobulked `.h5ad` file where each observation is a
    sample × cell-type pseudobulk (or per-sample bulk), averages across
    observations belonging to the same strain, and returns a
    (n_strains × n_genes) DataFrame suitable for QTL mapping.

    This is comparison-only: the returned matrix is aligned to the 48
    genotyped strains in the genome cache, so it can be directly used
    as `phenotype_matrix` in `discover()` / `scan_h2lmm()`.

    Parameters
    ----------
    h5ad_path : str
        Path to pseudobulked AnnData (.h5ad). obs must contain `strain_col`.
    strain_col : str
        Column in `adata.obs` that identifies the strain (e.g. 'strain').
    layer : str or None
        AnnData layer to use. None → use `adata.X`.
    strains : list of str or None
        Ordered list of strain IDs to include and align. If None, uses all
        strains found in `adata.obs[strain_col]`.
    min_cells : int
        Minimum number of observations per strain to include (strains with
        fewer are set to NaN).
    out_path : str or None
        If provided, write the matrix to this CSV path.

    Returns
    -------
    pd.DataFrame  shape (n_strains, n_genes)
        Index = strain IDs, columns = gene names from `adata.var_names`.
    """
    import pandas as pd

    try:
        import anndata as ad
    except ImportError:
        raise ImportError("anndata required: pip install anndata")

    adata = ad.read_h5ad(h5ad_path)
    logger.info(
        f"pheno_df: loaded {adata.n_obs} obs × {adata.n_vars} genes "
        f"from {h5ad_path}"
    )

    if strain_col not in adata.obs.columns:
        raise ValueError(
            f"strain_col {strain_col!r} not found in adata.obs. "
            f"Available: {list(adata.obs.columns)[:10]}"
        )

    # use raw (unnormalized) counts — normalized values bias the NB model
    if layer is not None:
        X = adata.layers[layer]
    elif adata.raw is not None:
        X = adata.raw.X
    else:
        logger.warning(
            "adata.raw is None — falling back to adata.X. "
            "Ensure adata.X contains raw counts; normalized values will corrupt LOD scores."
        )
        X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    else:
        X = np.array(X)

    obs_strains = adata.obs[strain_col].astype(str).values
    gene_names = adata.var_names.tolist()

    if strains is None:
        strains = sorted(set(obs_strains))

    rows = {}
    for s in strains:
        mask = obs_strains == s
        n = mask.sum()
        if n < min_cells:
            logger.warning(f"  strain {s}: only {n} obs < min_cells={min_cells}, NaN")
            rows[s] = np.full(len(gene_names), np.nan)
        else:
            rows[s] = X[mask].mean(axis=0)

    df = pd.DataFrame(rows, index=gene_names).T
    df.index.name = "strain"

    n_valid = int((~df.isnull().all(axis=1)).sum())
    logger.info(
        f"pheno_df: {n_valid}/{len(strains)} strains with ≥{min_cells} cells, "
        f"{len(gene_names)} genes"
    )

    if out_path is not None:
        df.to_csv(out_path)
        logger.info(f"pheno_df: saved to {out_path}")

    return df


# module-level helpers for CLI build


if __name__ == '__main__':
    from .cache_builder import build as _build
    _build()

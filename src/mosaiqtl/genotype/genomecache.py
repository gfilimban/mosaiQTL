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

logger = logging.getLogger('mosaiqtl')

# default cc founder strain names (in standard order)
CC_FOUNDERS = ['aj', 'b6j', '129s1j', 'nodj', 'nzoj', 'castj', 'pwkj', 'wsbj']


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

    def __init__(self, sample_ids, founders, data=None):
        """initialize genome cache.

        parameters
        ----------
        sample_ids : array-like of str
            subject identifiers
        founders : list of str
            founder strain names
        data : dict or None
            {chr_id: {'locus_ids': array, 'positions': array,
                       'probs': 3d array (n_loci, n_subjects, n_founders)}}
        """
        self.sample_ids = np.asarray(sample_ids)
        self.founders = list(founders)
        self.n_founders = len(founders)
        self._data = data if data is not None else {}
        self.crossinfo = None   # DataFrame (strains × A-H) from cc_crossinfo.csv
        self.x_donor   = {}     # strain_id -> founder_col of x-chr donor
        self.chromosomes = sorted(self._data.keys())

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
        return self._data[chrom]['probs'][locus_idx]

    def get_chromosome(self, chrom):
        """retrieve all loci for a chromosome.

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
        return d['locus_ids'], d['positions'], d['probs']

    def n_loci(self, chrom):
        """number of loci on a chromosome."""
        return len(self._data[chrom]['locus_ids'])

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

        data = {}
        with h5py.File(filepath, 'r') as f:
            # read founder names
            founders = [s.decode() for s in f['founders'][:]]

            # read sample ids from first chromosome
            sample_ids = None

            for key in f.keys():
                if not key.startswith('chr'):
                    continue
                grp = f[key]
                chrom = key.replace('chr', '')

                # read sample ids (same across all chromosomes)
                if sample_ids is None:
                    sample_ids = np.array(
                        [s.decode() for s in grp['sample_ids'][:]]
                    )

                data[chrom] = {
                    'locus_ids': np.array(
                        [s.decode() for s in grp['locus_ids'][:]]
                    ),
                    'positions': grp['positions'][:].astype(np.float64),
                    'probs': grp['probs'][:].astype(np.float64),
                }

        return GenomeCache(sample_ids, founders, data)

    def to_hdf5(self, filepath):
        """save genome cache to native hdf5 format.

        parameters
        ----------
        filepath : str
            output path for .h5 file
        """
        import h5py

        with h5py.File(filepath, 'w') as f:
            # store founder names as root attribute
            f.create_dataset(
                'founders',
                data=np.array(self.founders, dtype='S')
            )

            for chrom in self.chromosomes:
                grp = f.create_group(f'chr{chrom}')

                # store sample ids
                grp.create_dataset(
                    'sample_ids',
                    data=np.array(self.sample_ids, dtype='S')
                )

                d = self._data[chrom]

                # store locus metadata
                grp.create_dataset(
                    'locus_ids',
                    data=np.array(d['locus_ids'], dtype='S')
                )
                grp.create_dataset(
                    'positions',
                    data=d['positions']
                )

                # store probability tensor with compression
                grp.create_dataset(
                    'probs',
                    data=d['probs'],
                    compression='gzip',
                    compression_opts=4
                )

        logger.info(f'saved genome cache to {filepath}')

    @staticmethod
    def from_qtl2_probs(probs_dict, map_dict, sample_ids, founders=None):
        """construct genome cache from qtl2-style probability arrays.

        this is the bridge for data already processed through qtl2
        in R and exported as numpy arrays (via rpy2 or saved as .npz).

        parameters
        ----------
        probs_dict : dict
            {chrom: np.ndarray (n_subjects, n_loci, n_founders)}
            note: qtl2 convention is subjects × loci × founders
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
            # transpose from qtl2 (subjects, loci, founders)
            # to our convention (loci, subjects, founders)
            probs_transposed = np.transpose(probs_3d, (1, 0, 2))

            data[chrom] = {
                'locus_ids': np.asarray(map_dict[chrom]['locus_ids']),
                'positions': np.asarray(
                    map_dict[chrom]['positions'], dtype=np.float64
                ),
                'probs': probs_transposed,
            }

        return GenomeCache(sample_ids, founders, data)

    @staticmethod
    def from_gigamuga(gigamuga_path, gmap_dir, pmap_dir,
                      cc_cols, founder_cols, f1_crosses,
                      n_founders=8, n_gen=1, error_prob=0.002,
                      just_these_loci=None, qc=None,
                      crossinfo_path=None, step_cM=None):
        """build genome cache from raw GigaMUGA genotype data.
        this is the fully Python-native path: reads the GigaMUGA consensus
        matrix CSV, runs the CC diplotype HMM (hmm.py), builds deterministic
        founder and F1 arrays, merges all 48 genotypes, and returns a
        GenomeCache with additive (8-founder) haplotype probabilities.

        pipeline:
          gigamuga CSV → encode A/H/B → CC HMM (forward-backward) → additive probs
                       → deterministic founder arrays (homozygous, prob=1)
                       → deterministic F1 arrays (heterozygous, prob=1 each parent 0.5)
                       → GenomeCache(48 strains × 8 founders × n_loci)

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
        from .hmm import calc_genoprob_cc

        logger.info('building genome cache from raw GigaMUGA (no R)...')

        # load per-strain crossinfo if provided
        # columns A-H = funnel position (1-8) per founder per CC strain
        # used to: (1) confirm cc_inbred=True for each strain (homozygous prior)
        #          (2) store x-chr donor per strain for downstream analysis
        _crossinfo = None
        _x_donor   = {}  # strain_id -> founder_col of x-chr donor (funnel pos 8)
        if crossinfo_path is not None and os.path.exists(crossinfo_path):
            _crossinfo = pd.read_csv(crossinfo_path, index_col=0)
            for sid, row in _crossinfo.iterrows():
                # founder with funnel position 8 = last dam = X-chr donor
                x_donor_code = row[row == 8].index[0]
                _x_donor[sid] = x_donor_code
            logger.info(f'crossinfo: loaded {len(_crossinfo)} strains; '
                        f'{len(_x_donor)} x-chr donors identified')
        # cc_inbred flag: True for all CC strains (homozygous-only HMM prior);
        # founders and F1s use uniform_init via the deterministic path (no HMM)

        # ── read GigaMUGA matrix ───────────────────────────────────────────────
        gm_raw = pd.read_csv(gigamuga_path, low_memory=False)
        if 'marker' in gm_raw.columns:
            gm_raw = gm_raw.set_index('marker')
        # drop chr/pos if present — use the dedicated map files instead
        gm_raw = gm_raw.drop(columns=[c for c in ('chr', 'pos')
                                       if c in gm_raw.columns])
        gm_raw = gm_raw.replace('N', '-')   # missing call code

        # ── load genetic + physical maps ──────────────────────────────────────
        # supports two layouts:
        #   (a) flat files: gmap_dir/gmap.csv + pmap_dir/pmap.csv
        #       written by build_genomecache.R from final_markers — preferred
        #   (b) per-chr files: gmap_dir/GM_gmap{chr}.csv (legacy layout)
        def _load_maps(map_dir, prefix, flat_name):
            flat = os.path.join(map_dir, flat_name)
            if os.path.exists(flat):
                return pd.read_csv(flat, comment='#')
            files = sorted(glob.glob(os.path.join(map_dir, f'{prefix}*.csv')))
            if not files:
                raise FileNotFoundError(f'no {flat_name} or {prefix}*.csv in {map_dir}')
            return pd.concat([pd.read_csv(fp, comment='#') for fp in files], ignore_index=True)

        gmap = _load_maps(gmap_dir, 'GM_gmap', 'gmap.csv')
        pmap = _load_maps(pmap_dir, 'GM_pmap', 'pmap.csv')
        # normalise column names — flat maps use 'marker'/'pos'; per-chr use same
        gmap.rename(columns={'marker': 'locus', 'pos': 'pos_cM'}, inplace=True, errors='ignore')
        pmap.rename(columns={'marker': 'locus', 'pos': 'pos_mb'}, inplace=True, errors='ignore')

        # cast chr to str immediately — mixed int/str types cause == comparisons
        # to silently fail (e.g. integer 1 != string '1')
        gmap['chr'] = gmap['chr'].astype(str)
        pmap['chr'] = pmap['chr'].astype(str)
        logger.info(f'gmap: {len(gmap)} loci | pmap: {len(pmap)} loci')

        # ── intersect markers and sort by position ────────────────────────────
        # CRITICAL: Python set() is unordered. If we use list(set(...)) the
        # marker order is random, meaning the HMM receives positions out of
        # genomic order and computes wrong inter-marker recombination fractions.
        # This scrambles haplotype blocks: wrong kinship, inflated/random LOD.
        # Fix: sort shared markers by (chr, pos_cM) after intersecting.
        shared_set  = set(gm_raw.index) & set(gmap['locus']) & set(pmap['locus'])
        gmap_idx    = gmap.set_index('locus')            # index by locus name
        gmap_shared = gmap_idx.loc[list(shared_set)].copy()
        # sort key: integer chrs by value, X=20, others=99

        def _chr_key(c):
            s = str(c)
            return int(s) if s.isdigit() else (20 if s == 'X' else 99)
        gmap_shared['_k'] = gmap_shared['chr'].map(_chr_key)
        gmap_shared = gmap_shared.sort_values(['_k', 'pos_cM']).drop(columns=['_k'])
        shared      = gmap_shared.index.tolist()          # position-sorted locus names

        gm       = gm_raw.loc[shared]
        gmap     = gmap_shared                            # already indexed and sorted
        pmap_idx = pmap.set_index('locus').loc[shared]
        logger.info(f'shared markers: {len(shared)}')

        # ── just_these_loci allowlist ─────────────────────────────────────────────
        if just_these_loci is not None:
            allow = set(just_these_loci)
            shared = [m for m in shared if m in allow]
            logger.info(f'just_these_loci filter: {len(shared)} markers retained')

        # ── additional user QC filters (optional) ────────────────────────────────
        # note: gmap/pmap written by build_genomecache.R from final_markers (post
        # find_unique_geno on founders), so gm∩gmap∩pmap already excludes markers
        # monomorphic across the 8 founders. no additional filter needed here.
        if qc:
            keep = pd.Series(True, index=shared)
            if 'max_missing' in qc:
                all_cols = [c for c in list(founder_cols) + cc_cols if c in gm_raw.columns]
                miss = (gm_raw.loc[shared, all_cols] == '-').mean(axis=1)
                keep &= miss <= qc['max_missing']
            n_before = len(shared)
            shared = [m for m in shared if keep[m]]
            if n_before > len(shared):
                logger.info(f'user qc: {n_before} -> {len(shared)} markers '
                            f'(dropped {n_before - len(shared)})')

        # re-slice gm/gmap/pmap_idx to post-filter shared set
        gm       = gm_raw.loc[shared]
        gmap     = gmap.loc[shared]
        pmap_idx = pmap_idx.loc[shared]
        logger.info(f'post-filter markers: {len(shared)}')

        # chr column already str — build sorted chromosome list
        chrs = sorted(
            [c for c in gmap['chr'].unique() if c not in ('Y', 'M', 'MT', 'nan')],
            key=lambda c: int(c) if c.isdigit() else (20 if c == 'X' else 99)
        )

        # ── process per chromosome ────────────────────────────────────────────
        data   = {}
        all_48 = cc_cols + list(founder_cols) + [f[0] for f in f1_crosses]

        for chrom in chrs:
            # gmap['chr'] is already str; chrom is str from chrs list
            chr_mask  = gmap['chr'] == chrom
            chr_loci   = gmap[chr_mask].index.tolist()
            if not chr_loci:
                continue

            gmap_pos = gmap.loc[chr_loci, 'pos_cM'].values.astype(float)
            pmap_pos = pmap_idx.loc[chr_loci, 'pos_mb'].values.astype(float)
            locus_ids = np.array(chr_loci)

            geno_cc       = gm.loc[chr_loci, cc_cols].values         # (n_m, n_cc) nucleotides
            geno_founders = gm.loc[chr_loci, list(founder_cols)].values  # (n_m, J) nucleotides

            # ── find_unique_geno + encode_geno equivalent (founder-first, qtl2 ground truth) ──
            # Canonical qtl2convert workflow (kbroman.org/qtl2/pages/prep_do_data.html):
            #   allele_codes  <- find_unique_geno(geno = geno_founders_clean)
            #   geno_founders_encoded <- encode_geno(geno_founders_clean, allele_codes)
            #   geno_cc_encoded       <- encode_geno(geno_cc_clean,       allele_codes)
            #
            # Allele codes (a1/a2) are derived from the 8 FOUNDER columns, not CC strains.
            # Markers where founders are monomorphic or all-missing have no haplotype-
            # discrimination power for the HAPPY HMM and are skipped (encoded as all '-').
            # CC strains are then encoded using the same founder-derived codes.
            n_m = len(chr_loci)
            MISS_CODES = {'N', '-', 'nan', ''}

            cc_str = geno_cc.astype(str)        # (n_m, n_cc)
            f_str  = geno_founders.astype(str)  # (n_m, J)

            geno_cc_abh = np.full(cc_str.shape, '-', dtype=object)
            geno_f_abh  = np.full(f_str.shape,  '-', dtype=object)

            for m in range(n_m):
                # find_unique_geno on founders: two alleles from 8 founder calls
                f_valid  = [g for g in f_str[m] if g not in MISS_CODES and g != 'H']
                unique_f = list(dict.fromkeys(f_valid))  # order of first appearance
                if len(unique_f) < 2:
                    # monomorphic or all-missing across 8 founders: zero haplotype-
                    # discrimination power; gmap/pmap exclude these via R step 3
                    continue
                a1, a2 = unique_f[0], unique_f[1]

                # encode_geno on founders using founder-derived allele codes
                for fi in range(geno_f_abh.shape[1]):
                    g = f_str[m, fi]
                    if g in MISS_CODES:    geno_f_abh[m, fi] = '-'
                    elif g == 'H':         geno_f_abh[m, fi] = 'H'
                    elif g == a1:          geno_f_abh[m, fi] = 'A'
                    elif g == a2:          geno_f_abh[m, fi] = 'B'

                # encode_geno on CC strains with the same founder-derived codes
                for si in range(geno_cc_abh.shape[1]):
                    g = cc_str[m, si]
                    if g in MISS_CODES:    geno_cc_abh[m, si] = '-'
                    elif g == 'H':         geno_cc_abh[m, si] = 'H'
                    elif g == a1:          geno_cc_abh[m, si] = 'A'
                    elif g == a2:          geno_cc_abh[m, si] = 'B'

            n_markers  = len(chr_loci)
            n_strains  = len(all_48)
            # CC strains: HMM forward-backward on A/H/B encoded calls
            logger.info(f'  chr{chrom}: running CC HMM ({len(cc_cols)} strains, ' +
                        f'{n_markers} markers)...')
            result = calc_genoprob_cc(
                geno_cc_abh, geno_f_abh, gmap_pos,
                n_founders=n_founders, n_gen=1,   # consensus map encodes recombination history
                error_prob=error_prob,
                return_additive=True,
                cc_inbred=True,   # homozygous-only prior for fully inbred CC strains
                step_cM=step_cM,
            )

            if step_cM is not None:
                # unpack augmented (real + pseudo) positions
                probs_cc, aug_gmap_cM, is_real = result
                n_aug = len(aug_gmap_cM)
                # interpolate physical positions at pseudomarker cM positions
                aug_pmap_pos  = np.interp(aug_gmap_cM, gmap_pos, pmap_pos)
                # real markers keep their IDs; pseudomarkers get generated IDs
                # qtl2 pseudomarker naming: c{chr}.loc{pos} with digits = ceil(-log10(step))
                _digits = max(0, int(np.ceil(-np.log10(step_cM))))
                real_iter = iter(locus_ids)
                aug_locus_ids = np.array([
                    next(real_iter) if is_r else f'c{chrom}.loc{aug_gmap_cM[k]:.{_digits}f}'
                    for k, is_r in enumerate(is_real)
                ])
                logger.info(f'  chr{chrom}: {n_markers} real + {n_aug - n_markers} '
                            f'pseudo = {n_aug} total loci')
            else:
                probs_cc      = result
                n_aug         = n_markers
                aug_pmap_pos  = pmap_pos
                aug_locus_ids = locus_ids

            probs_chr = np.zeros((n_strains, n_founders, n_aug), dtype=np.float64)

            for i, col in enumerate(cc_cols):
                si = all_48.index(col)
                probs_chr[si] = probs_cc[i]

            # founders: deterministic homozygous at all positions (real + pseudo)
            for fi, fcol in enumerate(founder_cols):
                si = all_48.index(fcol)
                probs_chr[si, fi, :] = 1.0

            # F1s: deterministic heterozygous at all positions (real + pseudo)
            for sample_id, pi1, pi2 in f1_crosses:
                si = all_48.index(sample_id)
                probs_chr[si, pi1, :] = 0.5
                probs_chr[si, pi2, :] = 0.5

            # store as (n_loci, n_samples, n_founders) — mosaiqtl convention
            data[str(chrom)] = {
                'locus_ids': aug_locus_ids,
                'positions': aug_pmap_pos,
                'probs':     np.transpose(probs_chr, (2, 0, 1)),
            }

        cache = GenomeCache(all_48, list(founder_cols), data)
        cache.crossinfo = _crossinfo   # None if crossinfo_path not provided
        cache.x_donor   = _x_donor     # {} if crossinfo_path not provided
        return cache

    @staticmethod
    def from_rds(probs_cc_path, probs_founder_path, probs_f1_path,
                 pmap_path, founders=None):
        """build genome cache from raw qtl2 rds prob files.

        reads probs_cc.rds, probs_founder.rds, probs_f1.rds and the
        pmap.csv produced by founders_f1_genomecache.ipynb, merges
        all 48 genotypes (33 CC + 8 founders + 7 F1s), and returns a
        GenomeCache — exactly replicating what R does inside
        convert.qtl2.to.HAPPY() but in Python.

        this is the "from scratch" path: Python builds its own cache
        from the raw prob arrays without reading any R-produced files
        other than the original rds outputs of calc_genoprob().

        the merging logic mirrors founders_f1_genomecache.ipynb steps 5-7:
          - probs_cc:      qtl2 HMM output, shape (n_cc, n_diplotypes, n_loci)
          - probs_founder: deterministic 1-sample arrays per founder
          - probs_f1:      deterministic 1-sample arrays per F1

        genoprob_to_alleleprob() is applied to collapse 36 diplotype
        states to 8 additive founder probs, matching what R does before
        calc_kinship() and convert.qtl2.to.HAPPY().

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

        logger.info('loading raw qtl2 prob arrays via rpy2...')

        # load all three rds files into R namespace
        ro.r(f'probs_cc      <- readRDS("{probs_cc_path}")')
        ro.r(f'probs_founder <- readRDS("{probs_founder_path}")')
        ro.r(f'probs_f1      <- readRDS("{probs_f1_path}")')

        # step 1: merge CC + founders + F1s as diplotype arrays (36 states)
        #         mirrors notebook step 5: abind along sample dim, copy attrs
        # step 2: apply genoprob_to_alleleprob() to the combined diplotype array
        #         mirrors notebook step 6: collapses 36 → 8 additive probs
        # important: merge first, then collapse — not the other way around.
        # founders/F1s are stored as lists-of-chr-arrays (diplotype 36 states);
        # applying genoprob_to_alleleprob to probs_cc alone before merging
        # causes a dimension mismatch (8 additive vs 36 diplotype).
        ro.r('library(qtl2)')
        ro.r('library(abind)')
        ro.r("""
            chrs <- names(probs_cc)

            # merge all 48 genotypes per chromosome in diplotype space (36 states)
            # same as notebook step 5: abind(cc_arr, founder_arr, f1_arr, along=1)
            probs_combined_list <- lapply(chrs, function(chr) {
                cc_arr      <- probs_cc[[chr]]
                founder_arr <- do.call(abind, c(
                    lapply(probs_founder, function(x) x[[chr]]), list(along=1)))
                f1_arr      <- do.call(abind, c(
                    lapply(probs_f1,      function(x) x[[chr]]), list(along=1)))
                abind(cc_arr, founder_arr, f1_arr, along=1)
            })
            names(probs_combined_list) <- chrs

            # copy qtl2 genoprob attributes from probs_cc so that
            # genoprob_to_alleleprob() recognises the object correctly
            attributes(probs_combined_list) <- attributes(probs_cc)
            names(probs_combined_list)      <- chrs

            # collapse 36 diplotype → 8 additive founder probs
            # mirrors notebook step 6 (used for kinship and HAPPY cache)
            probs_merged <- genoprob_to_alleleprob(probs_combined_list)
        """)

        # read physical map
        pmap = pd.read_csv(pmap_path)
        if 'marker' in pmap.columns:
            pmap = pmap.rename(columns={'marker': 'locus', 'pos': 'pos_mb'})
        pmap['chr'] = pmap['chr'].astype(str)

        # extract chromosomes and sample ids
        # use ro.r() calls (not ro.r[]) for R expressions with [[ ]] indexing
        chrs_r     = list(ro.r('names(probs_merged)'))
        sample_ids = list(ro.r('rownames(probs_merged[[1]])'))

        data = {}
        for chrom in chrs_r:
            chr_pmap  = pmap[pmap['chr'] == chrom]
            locus_ids = np.array(chr_pmap['locus'].tolist())
            positions = np.array(chr_pmap['pos_mb'].tolist(), dtype=float)

            # extract array from R — qtl2 stores as (n_samples, n_founders, n_loci)
            arr_r = np.array(ro.r(f'probs_merged[["{chrom}"]]'))
            if arr_r.ndim == 3:
                # transpose to mosaiqtl convention: (n_loci, n_samples, n_founders)
                arr = np.transpose(arr_r, (2, 0, 1))
            else:
                logger.warning(f'chr{chrom}: unexpected array shape {arr_r.shape}')
                continue

            # get locus ids from R dimnames (dim 3 = markers)
            r_loci = list(ro.r(f'dimnames(probs_merged[["{chrom}"]])[[3]]'))
            if len(r_loci) == len(locus_ids):
                pass   # assume aligned
            else:
                # subset to pmap loci that appear in the prob array
                locus_set = set(r_loci)
                mask = np.array([lid in locus_set for lid in locus_ids])
                locus_ids = locus_ids[mask]
                positions = positions[mask]
                loci_idx  = {lid: i for i, lid in enumerate(r_loci)}
                idx       = np.array([loci_idx[lid] for lid in locus_ids
                                      if lid in loci_idx])
                arr = arr[idx]

            data[chrom] = {
                'locus_ids': locus_ids,
                'positions': positions,
                'probs':     arr,
            }
            logger.info(f'  chr{chrom}: {arr.shape[0]} loci × {arr.shape[1]} samples')

        return GenomeCache(sample_ids, founders, data)

    @staticmethod
    def from_happy_dir(cache_dir, model='additive'):
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
                'rpy2 required for reading happy .rdata files. '
                'install with: pip install rpy2'
            )

        # real cache structure: cache_dir/additive/chr{c}/data/*.RData
        model_dir = os.path.join(cache_dir, model)
        if not os.path.isdir(model_dir):
            # fallback: cache_dir might already be the model directory
            model_dir = cache_dir

        # discover chromosome directories
        chr_dirs = sorted(glob.glob(os.path.join(model_dir, 'chr*')))
        if not chr_dirs:
            raise FileNotFoundError(
                f'no chr* directories found in {model_dir}'
            )

        data = {}
        sample_ids = None
        founders = None

        for chr_dir in chr_dirs:
            chrom = os.path.basename(chr_dir).replace('chr', '')

            # data subdirectory contains the .rdata files
            data_dir = os.path.join(chr_dir, 'data')
            if not os.path.isdir(data_dir):
                # fallback: .rdata files directly in chr dir
                data_dir = chr_dir

            # find .rdata files (case-insensitive)
            rdata_files = sorted(
                glob.glob(os.path.join(data_dir, '*.RData')) +
                glob.glob(os.path.join(data_dir, '*.rdata'))
            )

            if not rdata_files:
                logger.warning(f'no .rdata files found in {data_dir}')
                continue

            locus_ids = []
            positions = []
            prob_list = []

            for rdata_path in rdata_files:
                # extract locus name from filename
                locus_name = os.path.splitext(
                    os.path.basename(rdata_path)
                )[0]
                locus_ids.append(locus_name)
                # object saved via save() in r. load() returns the
                # object names; we extract the matrix from the env.
                env = ro.r['new.env']()
                ro.r['load'](rdata_path, envir=env)
                obj_names = list(ro.r['ls'](envir=env))

                if not obj_names:
                    logger.warning(f'empty .rdata file: {rdata_path}')
                    continue

                # first object should be the haplotype prob matrix
                mat = np.array(ro.r['get'](obj_names[0], envir=env))

                # extract sample ids and founder names from first file
                if sample_ids is None:
                    r_obj = ro.r['get'](obj_names[0], envir=env)
                    rn = ro.r['rownames'](r_obj)
                    cn = ro.r['colnames'](r_obj)
                    if rn != ro.NULL:
                        sample_ids = list(rn)
                    if cn != ro.NULL:
                        founders = list(cn)

                prob_list.append(mat)
                positions.append(0.0)

            if not prob_list:
                continue

            # stack into 3d array: (n_loci, n_subjects, n_founders)
            probs_3d = np.stack(prob_list, axis=0)

            data[chrom] = {
                'locus_ids': np.array(locus_ids),
                'positions': np.array(positions),
                'probs': probs_3d,
            }

        if sample_ids is None:
            n_subjects = probs_3d.shape[1]
            sample_ids = [f'subject_{i}' for i in range(n_subjects)]

        if founders is None:
            founders = CC_FOUNDERS

        return GenomeCache(sample_ids, founders, data)

    def collapse(self, criterion='l2.norm', tol=0.02):
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
            lids, pos, probs = self.get_chromosome(chrom)   # (L, n, J)
            n_loci = len(lids)
            total_before += n_loci

            keep_lids, keep_pos, keep_probs = [], [], []
            run_probs = [probs[0]]   # accumulate a run of similar loci
            run_lid   = lids[0]
            run_pos   = pos[0]

            n_founders = probs.shape[2]
            for i in range(1, n_loci):
                if criterion in ('l2.norm', 'euclidean'):
                    # l2-norm criterion:
                    #   l2_i = ||h_i,l - h_i,l-1||_2 / sqrt(J)  per strain
                    #   merge if max_i(l2_i) <= tol  (any > tol → keep separate)
                    diff   = probs[i] - run_probs[-1]           # (n, J)
                    l2     = (np.linalg.norm(diff, axis=1)       # (n,)
                              / np.sqrt(n_founders))
                    d = float(l2.max())
                else:
                    # fraction of strains where argmax differs
                    d = float(np.mean(
                        probs[i].argmax(axis=1) != run_probs[-1].argmax(axis=1)))

                if d <= tol:
                    # extend current run
                    run_probs.append(probs[i])
                else:
                    # flush run: average probs, keep first locus id/pos
                    keep_lids.append(run_lid)
                    keep_pos.append(run_pos)
                    keep_probs.append(np.mean(run_probs, axis=0))
                    # start new run
                    run_probs = [probs[i]]
                    run_lid   = lids[i]
                    run_pos   = pos[i]

            # flush final run
            keep_lids.append(run_lid)
            keep_pos.append(run_pos)
            keep_probs.append(np.mean(run_probs, axis=0))

            n_kept = len(keep_lids)
            total_after += n_kept
            logger.info(f'  chr{chrom}: {n_loci} → {n_kept} loci '
                        f'({100*(n_loci-n_kept)/n_loci:.1f}% reduced)')

            new_data[chrom] = {
                'locus_ids': np.array(keep_lids),
                'positions': np.array(keep_pos, dtype=float),
                'probs':     np.stack(keep_probs),  # (n_kept, n_s, J)
            }

        logger.info(f'collapse: {total_before} → {total_after} total loci '
                    f'({100*(total_before-total_after)/total_before:.1f}% reduced)')
        return GenomeCache(list(self.sample_ids), list(self.founders), new_data)


# ──────────────────────────────────────────────────────────────────────────────
# pheno_df — pseudobulk → per-strain phenotype matrix
# ──────────────────────────────────────────────────────────────────────────────

def pheno_df(h5ad_path, strain_col='strain', layer=None, strains=None,
             min_cells=5, out_path=None):
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
        raise ImportError('anndata required: pip install anndata')

    adata = ad.read_h5ad(h5ad_path)
    logger.info(f'pheno_df: loaded {adata.n_obs} obs × {adata.n_vars} genes '
                f'from {h5ad_path}')

    if strain_col not in adata.obs.columns:
        raise ValueError(f'strain_col {strain_col!r} not found in adata.obs. '
                         f'Available: {list(adata.obs.columns)[:10]}')

    # extract expression matrix as dense numpy
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X
    if hasattr(X, 'toarray'):
        X = X.toarray()
    else:
        X = np.array(X)

    obs_strains = adata.obs[strain_col].astype(str).values
    gene_names  = adata.var_names.tolist()

    if strains is None:
        strains = sorted(set(obs_strains))

    rows = {}
    for s in strains:
        mask = obs_strains == s
        n    = mask.sum()
        if n < min_cells:
            logger.warning(f'  strain {s}: only {n} obs < min_cells={min_cells}, NaN')
            rows[s] = np.full(len(gene_names), np.nan)
        else:
            rows[s] = X[mask].mean(axis=0)

    df = pd.DataFrame(rows, index=gene_names).T   # (n_strains, n_genes)
    df.index.name = 'strain'

    n_valid = int((~df.isnull().all(axis=1)).sum())
    logger.info(f'pheno_df: {n_valid}/{len(strains)} strains with ≥{min_cells} cells, '
                f'{len(gene_names)} genes')

    if out_path is not None:
        df.to_csv(out_path)
        logger.info(f'pheno_df: saved to {out_path}')

    return df


# ──────────────────────────────────────────────────────────────────────────────
# module-level helpers for CLI build
# ──────────────────────────────────────────────────────────────────────────────

def _save_group_probs(cache, sample_list, filepath):
    """Save per-group haplotype probabilities to HDF5 in qtl2 convention.

    Output format mirrors R's probs_cc.rds / probs_founder.rds / probs_f1.rds
    produced by qtl2::calc_genoprob().  This enables direct numerical comparison
    between Python and R outputs.

    HDF5 layout
    -----------
    /founders           str array  8 founder names
    /sample_ids         str array  sample names in this group
    /chr{c}/probs       float32    (n_samples, n_founders, n_loci) — qtl2 dim order
    /chr{c}/locus_ids   str array  marker names
    /chr{c}/positions   float64    physical positions (Mb)

    Parameters
    ----------
    cache       : GenomeCache  full combined cache (all 48 samples)
    sample_list : list of str  samples to extract (cc_cols, founder_cols, f1 ids)
    filepath    : str          output .h5 path
    """
    import h5py

    sid_to_idx = {sid: i for i, sid in enumerate(cache.sample_ids)}
    grp_idx  = [sid_to_idx[s] for s in sample_list if s in sid_to_idx]
    grp_sids = [s for s in sample_list if s in sid_to_idx]

    if not grp_idx:
        logger.warning(f'_save_group_probs: no samples found in cache for {filepath}')
        return

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('founders',   data=np.array(cache.founders,  dtype='S'))
        f.create_dataset('sample_ids', data=np.array(grp_sids,        dtype='S'))

        for chrom in cache.chromosomes:
            lids, pos, probs = cache.get_chromosome(chrom)
            # probs: (n_loci, n_all_samples, n_founders)
            # → subset group → transpose to qtl2: (n_grp, n_founders, n_loci)
            grp_probs = probs[:, grp_idx, :]            # (n_loci, n_grp, J)
            grp_probs = np.transpose(grp_probs, (1, 2, 0)).astype(np.float32)  # (n_grp, J, n_loci)

            grp = f.create_group(f'chr{chrom}')
            grp.create_dataset('locus_ids', data=np.array(lids, dtype='S'))
            grp.create_dataset('positions', data=pos)
            grp.create_dataset('probs', data=grp_probs,
                               compression='gzip', compression_opts=4)

    n_loci_total = sum(cache.n_loci(c) for c in cache.chromosomes)
    logger.info(f'saved {len(grp_sids)} samples × {len(cache.chromosomes)} chrs '
                f'× {n_loci_total} loci → {filepath}')


def _save_kinship(kinship_dict, sample_ids, filepath):
    """Save LOCO kinship matrices to HDF5.

    HDF5 layout
    -----------
    /sample_ids   str array   sample names
    /chr{c}       float64     (n_samples, n_samples) LOCO kinship matrix

    Parameters
    ----------
    kinship_dict : dict  {chrom: K}  from kinship.calc_kinship(method='loco')
    sample_ids   : array-like of str
    filepath     : str  output .h5 path
    """
    import h5py

    with h5py.File(filepath, 'w') as f:
        f.create_dataset('sample_ids', data=np.array(sample_ids, dtype='S'))
        for key, K in kinship_dict.items():
            f.create_dataset(f'chr{key}', data=K.astype(np.float64),
                             compression='gzip', compression_opts=4)

    logger.info(f'saved LOCO kinship ({len(kinship_dict)} chrs) → {filepath}')


def build(args=None):
    """CLI entry point: build genome cache from raw GigaMUGA data.

    Writes five output files to --out-dir:

        genomecache.h5      full combined cache (all 48 samples, all chrs)
        probs_cc.h5         CC haplotype probs   ← parallels probs_cc.rds
        probs_founders.h5   founder probs        ← parallels probs_founder.rds
        probs_f1.h5         F1 haplotype probs   ← parallels probs_f1.rds
        kinship_loco.h5     LOCO kinship matrices (48×48 per chr)

    Usage
    -----
    python -m mosaiqtl.genomecache [options]
    build-cache [options]       (after pip install -e .)

    Column lists (--cc-cols, --founder-cols) can be either a comma-separated
    string or a path to a JSON file containing a list of strings.

    F1 crosses (--f1-crosses) must be a JSON file: list of
    [sample_id, parent1_idx, parent2_idx] where indices are 0-based
    positions into --founder-cols.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(
        prog='build-cache',
        description='Build Python genome cache from raw GigaMUGA data (no R required)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--gigamuga',      required=True,
                        help='GigaMUGA consensus matrix CSV (markers × strains)')
    parser.add_argument('--gmap-dir',      required=True,
                        help='directory containing gmap.csv (marker, chr, pos_cM)')
    parser.add_argument('--pmap-dir',      required=True,
                        help='directory containing pmap.csv (marker, chr, pos_mb)')
    parser.add_argument('--cc-cols',       required=True,
                        help='comma-sep CC sample names, or path to JSON list')
    parser.add_argument('--founder-cols',  required=True,
                        help='comma-sep founder names in canonical order, or JSON')
    parser.add_argument('--f1-crosses',    default=None,
                        help='JSON file: [[sample_id, p1_idx, p2_idx], ...]')
    parser.add_argument('--crossinfo',     default=None,
                        help='cc_crossinfo.csv (funnel positions per CC strain)')
    parser.add_argument('--out-dir',       default='validation/phase2/py_data',
                        help='output directory')
    parser.add_argument('--step-cM',       type=float, default=0.2,
                        help='pseudomarker spacing in cM matching qtl2 insert_pseudomarkers() '
                             'default (0.2); pass 0 to disable. pseudomarkers are included '
                             'in probs_*.h5 but dropped from the HAPPY cache')
    parser.add_argument('--error-prob',    type=float, default=0.002,
                        help='GigaMUGA genotyping error probability')
    parser.add_argument('--n-gen',         type=int,   default=1,
                        help='HMM transition scaling (1 for consensus CC map)')
    parser.add_argument('--collapse',      action='store_true',
                        help='collapse redundant loci before saving')
    parser.add_argument('--verbose',       action='store_true',
                        help='enable DEBUG logging')

    args = parser.parse_args(args)

    import logging as _logging
    _logging.basicConfig(
        level=_logging.DEBUG if args.verbose else _logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )

    def _parse_list(val):
        if val.endswith('.json') and os.path.exists(val):
            with open(val) as fh:
                return json.load(fh)
        return [x.strip() for x in val.split(',')]

    cc_cols      = _parse_list(args.cc_cols)
    founder_cols = _parse_list(args.founder_cols)
    f1_crosses   = []
    if args.f1_crosses:
        with open(args.f1_crosses) as fh:
            f1_crosses = json.load(fh)   # [[sample_id, pi1, pi2], ...]

    os.makedirs(args.out_dir, exist_ok=True)

    # ── build genome cache (Python-native: no R) ──────────────────────────────
    logger.info('=== build-cache ===')
    logger.info(f'  gigamuga : {args.gigamuga}')
    logger.info(f'  gmap_dir : {args.gmap_dir}')
    logger.info(f'  pmap_dir : {args.pmap_dir}')
    logger.info(f'  CC strains ({len(cc_cols)}): {cc_cols[:4]}{"..." if len(cc_cols) > 4 else ""}')
    logger.info(f'  founders  ({len(founder_cols)}): {founder_cols}')
    logger.info(f'  F1 crosses: {len(f1_crosses)}')
    logger.info(f'  out_dir  : {args.out_dir}')

    step_cM = args.step_cM if args.step_cM and args.step_cM > 0 else None
    cache = GenomeCache.from_gigamuga(
        gigamuga_path  = args.gigamuga,
        gmap_dir       = args.gmap_dir,
        pmap_dir       = args.pmap_dir,
        cc_cols        = cc_cols,
        founder_cols   = founder_cols,
        f1_crosses     = f1_crosses,
        error_prob     = args.error_prob,
        n_gen          = args.n_gen,
        crossinfo_path = args.crossinfo,
        step_cM        = step_cM,
    )

    if args.collapse:
        logger.info('collapsing redundant loci...')
        cache = cache.collapse()

    # ── save full combined cache ───────────────────────────────────────────────
    full_path = os.path.join(args.out_dir, 'genomecache.h5')
    cache.to_hdf5(full_path)

    # ── save per-group probs paralleling R's RDS files ────────────────────────
    _save_group_probs(cache, cc_cols,
                      os.path.join(args.out_dir, 'probs_cc.h5'))
    _save_group_probs(cache, founder_cols,
                      os.path.join(args.out_dir, 'probs_founders.h5'))
    if f1_crosses:
        f1_ids = [row[0] for row in f1_crosses]
        _save_group_probs(cache, f1_ids,
                          os.path.join(args.out_dir, 'probs_f1.h5'))

    # ── compute and save LOCO kinship ─────────────────────────────────────────
    from .kinship import calc_kinship
    logger.info('computing LOCO kinship...')
    kinship_loco = calc_kinship(cache, method='loco')
    _save_kinship(kinship_loco, cache.sample_ids,
                  os.path.join(args.out_dir, 'kinship_loco.h5'))

    logger.info('=== build complete ===')
    logger.info(f'  outputs in {args.out_dir}:')
    for fname in ('genomecache.h5', 'probs_cc.h5', 'probs_founders.h5',
                  'probs_f1.h5', 'kinship_loco.h5'):
        fp = os.path.join(args.out_dir, fname)
        if os.path.exists(fp):
            size_mb = os.path.getsize(fp) / 1e6
            logger.info(f'    {fname:<22s} {size_mb:6.1f} MB')


if __name__ == '__main__':
    build()

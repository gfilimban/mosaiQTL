"""
mosaiqtl.genome_cache — read and manage founder haplotype probabilities

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
CC_FOUNDERS = ['aj', 'b6j', 'c129s1j', 'castj', 'nodj', 'nzoj', 'pwkj', 'wsbj']


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

        # collapse 36 diplotype probs → 8 additive founder probs
        # (genoprob_to_alleleprob mirrors what R does before the HAPPY cache)
        ro.r('library(qtl2)')
        ro.r('probs_cc_add <- genoprob_to_alleleprob(probs_cc)')

        # merge: CC + founders + F1s — same order as notebook step 5
        # founders and F1s are already 1-sample lists-of-chr-arrays
        # we merge per-chromosome using abind along dim 1 (sample axis)
        ro.r('library(abind)')
        ro.r("""
            chrs <- names(probs_cc_add)
            probs_merged <- lapply(chrs, function(chr) {
                cc_arr <- probs_cc_add[[chr]]
                founder_arr <- do.call(abind, c(
                    lapply(probs_founder, function(x) x[[chr]]), list(along=1)))
                f1_arr <- do.call(abind, c(
                    lapply(probs_f1, function(x) x[[chr]]), list(along=1)))
                abind(cc_arr, founder_arr, f1_arr, along=1)
            })
            names(probs_merged) <- chrs
        """)

        # read physical map
        pmap = pd.read_csv(pmap_path)
        if 'marker' in pmap.columns:
            pmap = pmap.rename(columns={'marker': 'locus', 'pos': 'pos_mb'})
        pmap['chr'] = pmap['chr'].astype(str)

        # extract chromosomes and sample ids
        chrs_r    = list(ro.r['chrs'])
        sample_ids = list(ro.r['rownames'](ro.r['probs_merged[[1]]']))

        data = {}
        for chrom in chrs_r:
            chr_pmap  = pmap[pmap['chr'] == chrom]
            locus_ids = np.array(chr_pmap['locus'].tolist())
            positions = np.array(chr_pmap['pos_mb'].tolist(), dtype=float)

            # extract (n_samples, n_loci, 8) array from R
            # R stores as (n_samples, n_founders, n_loci) — transpose needed
            arr_r = np.array(ro.r[f'probs_merged[["{chrom}"]]'])
            # arr_r shape: (n_samples, 8, n_loci) — R uses samples×founders×loci
            if arr_r.ndim == 3:
                # transpose to (n_loci, n_samples, 8)
                arr = np.transpose(arr_r, (2, 0, 1))
            else:
                logger.warning(f'chr{chrom}: unexpected array shape {arr_r.shape}')
                continue

            # align locus order to pmap
            r_loci = list(ro.r['dimnames'](ro.r[f'probs_merged[["{chrom}"]]'])[2])
            if len(r_loci) == len(locus_ids):
                pass   # assume aligned
            else:
                # subset to pmap loci that appear in the prob array
                locus_set = set(r_loci)
                mask = np.array([l in locus_set for l in locus_ids])
                locus_ids = locus_ids[mask]
                positions = positions[mask]
                loci_idx  = {l: i for i, l in enumerate(r_loci)}
                idx       = np.array([loci_idx[l] for l in locus_ids
                                      if l in loci_idx])
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
            (e.g., 'output/genomecache')
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
        # numpy2ri.activate() removed — deprecated in rpy2 >= 3.5
        # np.array(r_object) works directly without a converter

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

                # load() not readRDS() — .rdata files contain a named
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

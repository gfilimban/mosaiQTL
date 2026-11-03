"""Convert GM_ccf1fo_mm39.csv (in ref/) to igvf_cache.h5.

argv[1]  gigamuga.csv   path to GM_ccf1fo_mm39.csv
argv[2]  output_dir     directory to write igvf_cache.h5

gmap.csv and pmap.csv are expected alongside gigamuga.csv in the same directory.
cc_cols are auto-detected from the CSV header (columns starting with 'CC').
founder_cols are always FOUNDER_ORDER from mosaiqtl.genotype.cc.
"""

import logging
import os
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s,%(msecs)03d %(levelname)-5s %(message)s',
    datefmt='%H:%M:%S',
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

gigamuga_csv = sys.argv[1]
output_dir   = sys.argv[2]

out_path = os.path.join(output_dir, 'igvf_cache.h5')
if os.path.exists(out_path):
    logger.info('cache already exists — skipping: %s', out_path)
    sys.exit(0)

ref_dir = os.path.dirname(os.path.abspath(gigamuga_csv))

gmap_csv = os.path.join(ref_dir, 'gmap.csv')
pmap_csv = os.path.join(ref_dir, 'pmap.csv')

for p in (gigamuga_csv, gmap_csv, pmap_csv):
    if not os.path.exists(p):
        raise FileNotFoundError(f'required input not found: {p}')

from mosaiqtl.genotype.cc import FOUNDER_ORDER
from mosaiqtl.genotype.genomecache import GenomeCache

import pandas as pd
header = pd.read_csv(gigamuga_csv, nrows=0).columns.tolist()
cc_cols = [c for c in header if c.startswith('CC')]
if not cc_cols:
    raise ValueError(f'no CC* columns found in {gigamuga_csv}; check CSV format')

logger.info('converting GigaMUGA CSV → igvf_cache.h5')
logger.info('  gigamuga : %s', gigamuga_csv)
logger.info('  cc_cols  : %d strains', len(cc_cols))

cache = GenomeCache.from_gigamuga(
    gigamuga_path=gigamuga_csv,
    gmap_dir=ref_dir,
    pmap_dir=ref_dir,
    cc_cols=cc_cols,
    founder_cols=FOUNDER_ORDER,
)

logger.info('writing %s', out_path)
cache.to_hdf5(out_path)
logger.info('done')

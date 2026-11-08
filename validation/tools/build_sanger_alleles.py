"""Build ref/sanger_alleles.h5 from the MGP founders VCF.

Usage
-----
conda run -n bioinfo python validation/tools/build_sanger_alleles.py \\
    --vcf  /path/to/mgp_REL2021_snps_founders.rsID.vcf.gz \\
    --cache ref/igvf_cache.h5 \\
    --out   ref/sanger_alleles.h5

The VCF must carry rsID annotations (ID column populated).  Build the
annotated VCF first with bcftools annotate if needed:

    bcftools annotate \\
        -a /path/to/mgp_REL2021_snps.rsID.vcf.gz \\
        -c ID \\
        -o mgp_REL2021_snps_founders.rsID.vcf.gz \\
        -O z \\
        mgp_REL2021_snps_founders.vcf.gz
    bcftools index --tbi --force mgp_REL2021_snps_founders.rsID.vcf.gz

Founder name mapping applied automatically (CC_VCF_SAMPLE_MAP):
    A_J → AJ | 129S1_SvImJ → 129S1J | NOD_ShiLtJ → NODJ | ...
B6J is synthesised as the all-REF column (not in the MGP VCF).
"""
import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description='Build ref/sanger_alleles.h5 from MGP VCF.')
    ap.add_argument('--vcf',   required=True, help='MGP founders VCF (rsID-annotated, tabix-indexed)')
    ap.add_argument('--cache', required=True, help='Genome cache HDF5 (ref/igvf_cache.h5)')
    ap.add_argument('--out',   required=True, help='Output HDF5 path (ref/sanger_alleles.h5)')
    args = ap.parse_args()

    vcf_path   = Path(args.vcf)
    cache_path = Path(args.cache)
    out_path   = Path(args.out)

    if not vcf_path.exists():
        ap.error(f'VCF not found: {vcf_path}')
    if not cache_path.exists():
        ap.error(f'Genome cache not found: {cache_path}')

    from mosaiqtl.genotype.genomecache import GenomeCache
    from mosaiqtl.genotype.snp_db import CC_VCF_SAMPLE_MAP, build_founder_allele_mgp

    log.info('Loading genome cache: %s', cache_path)
    gc = GenomeCache.from_hdf5(cache_path)

    log.info('Building sanger_alleles.h5 from %s', vcf_path.name)
    log.info('VCF sample map: %s', CC_VCF_SAMPLE_MAP)
    stats = build_founder_allele_mgp(
        vcf_path, gc, out_path, vcf_sample_map=CC_VCF_SAMPLE_MAP,
    )
    log.info('Wrote %s', out_path)
    if stats:
        log.info('Stats: %s', stats)


if __name__ == '__main__':
    main()

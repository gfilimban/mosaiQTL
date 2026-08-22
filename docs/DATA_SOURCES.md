# docs/DATA_SOURCES.md — mosaiQTL Input Data Provenance
# All Layer-0 public sources and the derivation chain to GM_ccfo_mm39.csv
# Author: gfilimba@uci.edu | genome: mm39 | array: GigaMUGA | population: CC mice

---

## Overview

`GM_ccfo_mm39.csv` is the consolidated marker × sample consensus-genotype matrix that
drives `mosaiqtl build-cache --population cc`. It is an intermediate file — users do not
supply it manually; `mosaiqtl build-cache` produces it internally. Its exact shape
(marker count, strain set) is build-dependent.

Two versions exist — same 51-column schema (marker + chr + pos + 8 founders + 7 F1s + 33 CC strains):

| File | Markers | Notes |
|---|---|---|
| `old/GM_ccfo_mm39.csv` | 134,591 | Original build; legacy |
| `GM_ccfo_mm39_comprehensive.csv` | **137,280** | Rebuilt from Layer-0 public sources; **current reference** |

The comprehensive file was verified against the old file:
- **CC calls: 100% concordant** across all 4,357,419 shared marker × strain cells
- **Founders: zero true conflicts** — all 26,158 "discordant" cells are `old=N → new=call`,
  meaning the new file *recovers* 26,158 missing founder calls; no case where both had a
  non-N call that disagreed
- **5,237 new markers** added (mapped on GRCm39, in `gm_uwisc_v4.csv`, absent from old build)

---

## Layer-0 Public Sources

All files below are freely downloadable. The canonical URL for GigaMUGA and CC reference
data is the JAX Diversity Outbred Reference Data page:

> **JAX DO Reference Data**
> https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data

| File | Rows × Cols | Contents | Where used |
|---|---|---|---|
| `gm_uwisc_v4.csv` | 143,260 × 11 | GigaMUGA marker positions on GRCm39 (`bp_grcm39`) and mm10 (`bp_mm10`); Cox genetic map (`cM_cox`); strand, probe, QC flags | Marker position columns (chr, pos) in `GM_ccfo_mm39.csv` |
| `GigaMUGA_founder_consensus_genotypes.csv.gz` | 123,915 × 9 | Consensus ACGT calls for 8 CC/DO founders (A/J, C57BL/6J, 129S1/SvImJ, NOD/ShiLtJ, NZO/HlLtJ, CAST/EiJ, PWK/PhJ, WSB/EiJ); one column per founder | Founder columns (AJ, B6J, …, WSBJ) in `GM_ccfo_mm39.csv` |
| `GigaMUGA_founder_metadata.csv` | 170 × 5 | Sample × strain × sex × letter mapping for all GigaMUGA founder and F1 samples; letter codes AA–HH = homozygous founders, AB–HG = F1 crosses | Identifies which samples are the 8 founders and which F1 crosses to include |
| `GigaMUGA_sample_metadata.csv` | 279 × 4 | All GigaMUGA reference samples (non-CC); used for QC | Background reference |
| `GigaMUGA_genotypes.csv.gz` | ~124,000 × 280 | All GigaMUGA reference sample genotypes | Complete raw calls before consensus |
| `GigaMUGA_founder_consensus_genotypes_Mt.csv` | 20 × 6 | Mitochondrial haplogroup calls per founder | Organelle haplotype annotation |
| `GigaMUGA_founder_consensus_genotypes_Y.csv` | 13 × 7 | Y-chromosome haplogroup calls per founder | Organelle haplotype annotation |
| `CC_consensus_genotypes.csv` | ~133,948 × 77 | Consensus ACGT genotype calls for 76 CC strains (column headers `CC001/Unc`, `CC002/Unc`, …); one column per strain | CC strain columns in `GM_ccfo_mm39.csv` (subset to study strains) |
| `cc_sample_metadata.csv` | 76 × 5 | CC strain metadata: mitochondrial origin, Y-chromosome origin, number of founders, funnel order, breeding origin (UNC/TAU/GENIAD) | Cross-info for `cc_crossinfo.csv`; funnel codes drive HMM initialisation |
| `cc_variants_grcm39_ens104.sqlite` | ~70M SNPs | Sanger MGP REL2021 + Ensembl 104 annotated SNPs/indels for all 8 CC founders on GRCm39 | Fine-mapping (`snp_db.py`); not part of `GM_ccfo_mm39.csv` |

---

## Derivation Chain: Public Sources → GM_ccfo_mm39.csv

```
gm_uwisc_v4.csv              ──────────────────────────────────────────┐
  (marker, chr, bp_grcm39,                                             │
   cM_cox, strand, probe)                                              │
                                                                       │
GigaMUGA_founder_consensus_  ── consensus across replicate samples ──► │
  genotypes.csv.gz               (GigaMUGA_founder_metadata.csv        │
  (ACGT calls × 8 founders)       maps sample → founder letter)        │
  re-encode: A allele → 'A',                                           │
             B allele → 'B',                                           │
             missing  → '-'                                            │
  split per chr → GM_foundergeno{chr}.csv                             │
                                                                       ▼
CC_consensus_genotypes.csv   ── filter to study CC strains ──────► JOIN ──► GM_ccfo_mm39.csv
  (ACGT × 76 CC strains)          strip institution suffix            marker, chr, pos,
  CC001/Unc → CC001               (CC001/Unc → CC001)                [33 CC strains],
                                                                      [8 founders],
cc_sample_metadata.csv       ── funnel order (A–H per strain) ──►   [7 F1 crosses]
  (funnel, Mt origin, Y)          → cc_crossinfo.csv
  used by HMM initialisation      (HMM prior distribution)

F1 calls (7 B6J-as-dam)     ── inferred from founder haplotypes ──►  F1 columns
  B6J × AJ   → B6AF1J            heterozygous ACGT encoding           (appended last)
  B6J × 129  → B6129S1F1J        "T1/T2" at biallelic loci
  B6J × NOD  → B6NODF1J
  B6J × NZO  → B6NZOF1J
  B6J × CAST → B6CASTF1J
  B6J × PWK  → B6PWKF1J
  B6J × WSB  → B6WSBF1J
```

---

## Column Schema of GM_ccfo_mm39.csv

| Column group | Count | Encoding | Source |
|---|---|---|---|
| `marker` | 1 | string marker ID | `gm_uwisc_v4.csv` |
| `chr` | 1 | `'1'`–`'19'`, `'X'` (string) | `gm_uwisc_v4.csv` |
| `pos` | 1 | bp on GRCm39 | `gm_uwisc_v4.csv` `bp_grcm39` |
| CC strain columns | 33 (study-specific) | ACGT nucleotide call or `N` for missing | `CC_consensus_genotypes.csv` |
| Founder columns | 8 (`AJ`, `B6J`, `129S1J`, `NODJ`, `NZOJ`, `CASTJ`, `PWKJ`, `WSBJ`) | ACGT or `N` | `GigaMUGA_founder_consensus_genotypes.csv.gz` |
| F1 columns | 7 (B6J-as-dam crosses) | `'T1/T2'` heterozygous ACGT | Inferred from founder haplotypes |

**Axis order invariant (inviolable):** columns are always **Founders → F1s → CC strains** in
`GenomeCache` internal storage (see `genotype/genomecache.py::from_gigamuga`). The CSV column
order above (CC first, then founders, then F1s) is the raw intermediate — `from_gigamuga()`
reorders to the canonical axis order on load.

---

## Marker Filtering

Not all GigaMUGA markers appear in `GM_ccfo_mm39.csv`. Markers are retained only when:

1. Present in `gm_uwisc_v4.csv` with a valid `bp_grcm39` position (unmapped markers excluded)
2. On autosomes 1–19 or chrX (chrY and chrM handled separately via
   `GigaMUGA_founder_consensus_genotypes_Y.csv` / `_Mt.csv`)
3. `unique == TRUE` in `gm_uwisc_v4.csv` (single-position mapping on GRCm39)

The `loci_max_delta=1237` offset vs the R sparcc_cache is expected and correct — it reflects
pseudomarker exclusion in the miQTL HAPPY cache, not a data discrepancy.

---

## CC Strain Selection

The full JAX `CC_consensus_genotypes.csv` contains 76 CC strains; the mosaiQTL gastroc dataset
uses **33 strains** (CC001–CC074, non-contiguous). The study-specific subset is controlled by
the `--cc-cols` argument to `mosaiqtl build-cache`:

```bash
mosaiqtl build-cache --population cc \
    --data-dir /path/to/GM/ \
    --cc-cols cc_cols.txt \      # one CC strain ID per line, no institution suffix
    --output cache.h5
```

Any subset of the 76 CC strains in `CC_consensus_genotypes.csv` is valid.

---

## Genetic Map

The Cox composite genetic map (`cM_cox` column in `gm_uwisc_v4.csv`) is used for all
genetic-distance computations. This is the GRCm39/mm39 build of the map. The map was built
by the UWisc group (Broman lab); `gm_uwisc_v4.csv` is version 4 of the GigaMUGA position file.

Genetic positions are loaded into `GenomeCache` as `gmap` (cM); physical positions as `pmap`
(bp on GRCm39). Both are stored in HDF5 per chromosome.

---

## Files Currently in qtl/input/

| File | Size | Role |
|---|---|---|
| `CC_consensus_genotypes.csv` | 21 MB | Layer-0: 76 CC strain consensus genotypes |
| `cc_sample_metadata.csv` | 2.4 KB | Layer-0: CC strain metadata (Mt, Y, funnel, origin) |
| `cc_variants_grcm39_ens104.sqlite` | 3.9 GB | Fine-mapping: Sanger REL2021 + Ensembl 104 SNPs/indels |
| `gm_uwisc_v4.csv` | 17 MB | Layer-0: GigaMUGA marker positions + Cox map on GRCm39 |
| `GigaMUGA_founder_metadata.csv` | 8.8 KB | Layer-0: founder + F1 sample → strain letter mapping |
| `GigaMUGA_founder_consensus_genotypes_Mt.csv` | 616 B | Layer-0: mitochondrial haplogroups |
| `GigaMUGA_founder_consensus_genotypes_Y.csv` | 529 B | Layer-0: Y-chr haplogroups |
| `GigaMUGA_genotypes.csv.gz` | 8.5 MB | Layer-0: all reference sample genotypes (compressed) |
| `GigaMUGA_sample_metadata.csv` | 12 KB | Layer-0: all reference sample metadata |
| `gene_info.vm32.csv` / `gene_info.vm36.csv` | 2.8–4.1 MB | Gene annotations (two Vombatus marsupialis builds) |
| `gene_tss.vm32.csv` / `gene_tss.vm36.csv` | 12–24 MB | Gene TSS positions |
| `sample_metadata.csv` | 543 KB | Study sample metadata (sex, batch, tissue) |
| `old/GM_ccfo_mm39.csv` | 17 MB | Pre-assembled intermediate (do not edit directly) |
| `old/GM/` | — | Legacy per-chr gmap/pmap/foundergeno CSVs (now superseded by Layer-0 files above) |
| `old/cc_crossinfo.csv` | — | HMM funnel priors derived from `cc_sample_metadata.csv` |

---

## Reproducing GM_ccfo_mm39.csv from Scratch

All Layer-0 inputs are publicly available. The full rebuild (Layer-0 → HDF5 cache) requires:

```bash
# 1. Download Layer-0 files from JAX DO Reference Data page
#    https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data
#    Files needed: gm_uwisc_v4.csv, GigaMUGA_founder_consensus_genotypes.csv.gz,
#                  GigaMUGA_founder_metadata.csv, CC_consensus_genotypes.csv,
#                  cc_sample_metadata.csv

# 2. Prepare per-chr gmap/pmap (from gm_uwisc_v4.csv)
#    python tools/prep_gigamuga_maps.py \
#        --uwisc gm_uwisc_v4.csv \
#        --out-dir GM/gmap/ GM/pmap/

# 3. Prepare per-chr founder genotypes (from GigaMUGA_founder_consensus_genotypes.csv.gz)
#    python tools/prep_founder_genotypes.py \
#        --genotypes GigaMUGA_founder_consensus_genotypes.csv.gz \
#        --metadata GigaMUGA_founder_metadata.csv \
#        --out-dir GM/foundergeno/

# 4. Prepare cc_crossinfo.csv (from cc_sample_metadata.csv funnel codes)
#    python tools/prep_cc_crossinfo.py \
#        --metadata cc_sample_metadata.csv \
#        --out cc_crossinfo.csv

# 5. Run build-cache
mosaiqtl build-cache --population cc \
    --data-dir GM/ \
    --cc-cols cc_cols.txt \
    --output cache.h5
```

> **Note:** Steps 2–4 require utility scripts not yet shipped in mosaiQTL v0.5.
> `mosaiqtl build-cache --population cc` will wrap them as a single command in v0.6.
> Until then, `old/GM_ccfo_mm39.csv` serves as the pre-assembled intermediate.

---

## Data Citations

| Dataset | Citation |
|---|---|
| GigaMUGA marker positions (`gm_uwisc_v4.csv`) | Morgan AP et al. *G3* 2016. doi:10.1534/g3.116.037622 |
| Founder consensus genotypes | Morgan AP et al. *G3* 2016; JAX DO Reference Data page |
| CC consensus genotypes (`CC_consensus_genotypes.csv`) | Collaborative Cross Consortium. *Genetics* 2012. doi:10.1534/genetics.113.153452 |
| Sanger MGP variants (`cc_variants_grcm39_ens104.sqlite`) | Keane TM et al. *Nature* 2011. doi:10.1038/nature10413; Ensembl REL2021 |
| R/qtl2 support files | Broman KW et al. *Genetics* 2019. doi:10.1534/genetics.119.302286 |

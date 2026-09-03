# docs/DATA_SOURCES.md — mosaiQTL Input Data Provenance
# All Layer-0 public sources and the derivation chain to GM_ccf1fo_mm39.csv
# Author: gfilimba@uci.edu | genome: mm39 | array: GigaMUGA | population: CC mice

---

## Overview

`GM_ccf1fo_mm39.csv` (columns: CC strains, F1 crosses, founders — hence the name) is the
consolidated marker × sample consensus-genotype matrix that drives
`mosaiqtl build-cache --population cc`. It is an intermediate file — users do not supply
it manually; `mosaiqtl build-cache` produces it internally. Its exact shape (marker count,
strain set) is build-dependent.

**Provenance history (2026-08-23):** a prior build (`GM_ccfo_mm39_comprehensive.csv`,
137,280 markers) was found to source 3,417 markers' worth of founder genotype coverage
from `old/GM/foundergeno/GM_foundergeno{chr}.csv` — an uncited, undated-provenance legacy
file explicitly labeled "Legacy... superseded" elsewhere in this document, with no linked
public source. That file, `old/GM/`, was never part of the Layer-0 source list below.
`GM_ccfo_mm39_comprehensive.csv` and the older `old/GM_ccfo_mm39.csv` (134,591 markers,
same defect) have both been **deleted**. `GM_ccf1fo_mm39.csv` (**133,863 markers**) is the
replacement, rebuilt using *only* the Layer-0 public sources listed below — see
"Reproducing GM_ccf1fo_mm39.csv from Scratch" for the exact workflow and
`validation/tools/build_gm_ccf1fo.py` for the rebuild script.

Verified against `GM_ccfo_mm39_comprehensive.csv` on the 133,863 markers common to both:
- **CC calls: 100% concordant** — 0 mismatches across all shared marker × strain cells
- **Founders: 0 true conflicts** — 25,338 cells that were non-`N` in the old file are
  correctly `N` here (those calls came only from the excluded `foundergeno` file); no case
  where both files had a non-`N` call that disagreed
- **3,417 markers dropped** — present in the old file (via `foundergeno`) but absent from
  every remaining Layer-0 source for founder/CC genotypes

---

## Layer-0 Public Sources

All files below are freely downloadable. The canonical URL for GigaMUGA and CC reference
data is the JAX Diversity Outbred Reference Data page:

> **JAX DO Reference Data**
> https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data

| File | Rows × Cols | Contents | Where used |
|---|---|---|---|
| `gm_uwisc_v4.csv` | 143,259 × 11 | GigaMUGA marker positions on GRCm39 (`bp_grcm39`) and mm10 (`bp_mm10`); Cox genetic map (`cM_cox`); strand, probe, QC flags | Marker position columns (chr, pos) in `GM_ccf1fo_mm39.csv` |
| `GigaMUGA_founder_metadata.csv` | 170 × 5 | Sample × strain × sex × letter mapping for all GigaMUGA founder and F1 samples; letter codes AA–HH = homozygous founders, AB–HG = F1 crosses | Identifies which raw-genotype-file samples are replicates of each of the 8 founders |
| `GigaMUGA_sample_metadata.csv` | 279 × 4 | All GigaMUGA reference samples (non-CC); used for QC | Background reference |
| `GigaMUGA_genotypes.csv.gz` | ~124,000 × 280 | All GigaMUGA reference sample raw calls (A/C/G/T homozygous, `H` heterozygous, `N` no-call) | Founder consensus (majority vote across replicate samples sharing a letter code) → founder columns in `GM_ccf1fo_mm39.csv`. There is no separate pre-consensused founder file — consensus is computed by `build_gm_ccf1fo.py` from this file + `GigaMUGA_founder_metadata.csv` |
| `GigaMUGA_founder_consensus_genotypes_Mt.csv` | 20 × 6 | Mitochondrial haplogroup calls per founder | Organelle haplotype annotation |
| `GigaMUGA_founder_consensus_genotypes_Y.csv` | 13 × 7 | Y-chromosome haplogroup calls per founder | Organelle haplotype annotation |
| `CC_consensus_genotypes.csv` | ~133,948 × 77 | Consensus ACGT genotype calls for 76 CC strains (column headers `CC001/Unc`, `CC002/Unc`, …); one column per strain | CC strain columns in `GM_ccf1fo_mm39.csv` (subset to study strains); also gates which markers get a row (a marker must have CC data, else its position is dropped) |
| `cc_sample_metadata.csv` | 76 × 5 | CC strain metadata: mitochondrial origin, Y-chromosome origin, number of founders, funnel order, breeding origin (UNC/TAU/GENIAD) | Cross-info for `cc_crossinfo.csv`; funnel codes drive HMM initialisation |
| `mgp_REL2021_snps_founders.vcf.gz` | 33,511,156 SNPs genome-wide | Sanger Mouse Genomes Project REL2021 founder SNP calls, mm39-native from build (no liftover) | Fine-mapping (`snp_db.py`); not part of `GM_ccf1fo_mm39.csv` — see `docs/REFERENCES.md` for full grounding |

---

## Derivation Chain: Public Sources → GM_ccf1fo_mm39.csv

```
gm_uwisc_v4.csv ── filter (unique==TRUE, mapped, chr in 1-19/X) ──► marker positions
  (marker, chr, bp_grcm39,                                          (chr, pos)
   cM_cox, strand, probe, QC flags)                                       │
                                                                            │
GigaMUGA_genotypes.csv.gz ── restrict to gm_uwisc markers ──► raw calls    │
  (raw per-sample A/C/G/T/H/N,        (position filter above)   (n_markers │
   all 280 reference samples)                                    x 280)   │
         │                                                          │
         │  GigaMUGA_founder_metadata.csv (sample → letter, e.g. AA=A/J)  │
         ▼                                                                │
  majority-vote consensus per founder letter ──► founder_df               │
  (unanimous among non-N/non-H calls, else 'N')  (n_markers x 8 founders) │
         │                                                                │
         ▼                                                                │
  derive F1 = f'{B6J_call}/{other_founder_call}' ──► f1_df                │
  (from founder_df, not raw F1 samples)          (n_markers x 7 F1s)      │
         │                                                                │
         └──────────────────┬─────────────────────────────────────────────┘
                             │
CC_consensus_genotypes.csv  │ inner join on marker position (row set
  (ACGT x 76 CC strains)    │ requires real CC data); founder_df/f1_df
  CC001/Unc → CC001         │ LEFT-joined, fillna('N') for uncovered markers
  subset to study strains   │
         └───────────────────► JOIN ──► GM_ccf1fo_mm39.csv
                                          marker, chr, pos,
cc_sample_metadata.csv ── funnel order ──► cc_crossinfo.csv    [8 founders],
  (funnel, Mt origin, Y)     (HMM prior distribution,          [7 F1 crosses],
  used by HMM initialisation  not a column in this CSV)        [33 CC strains]
```

Rebuild script: `validation/tools/build_gm_ccf1fo.py` (6 steps, matches the diagram above).

---

## Column Schema of GM_ccf1fo_mm39.csv

| Column group | Count | Encoding | Source |
|---|---|---|---|
| `marker` | 1 | string marker ID | `gm_uwisc_v4.csv` |
| `chr` | 1 | `'1'`–`'19'`, `'X'` (string) | `gm_uwisc_v4.csv` |
| `pos` | 1 | bp on GRCm39 | `gm_uwisc_v4.csv` `bp_grcm39` |
| Founder columns | 8 (`AJ`, `B6J`, `129S1J`, `NODJ`, `NZOJ`, `CASTJ`, `PWKJ`, `WSBJ`) | ACGT or `N` | Majority-vote consensus over `GigaMUGA_genotypes.csv.gz` replicate samples, mapped via `GigaMUGA_founder_metadata.csv` |
| F1 columns | 7 (B6J-as-dam crosses) | `'{B6J_call}/{other_call}'`, or `N` if either side is `N` | Derived directly from the founder columns (not raw F1 samples) |
| CC strain columns | 33 (study-specific) | ACGT nucleotide call or `N` for missing | `CC_consensus_genotypes.csv` |

**Axis order invariant (inviolable):** columns are always **Founders → F1s → CC strains** in
`GenomeCache` internal storage (see `genotype/genomecache.py::from_gigamuga`). The CSV column
order above (CC first, then founders, then F1s) is the raw intermediate — `from_gigamuga()`
reorders to the canonical axis order on load.

---

## Marker Filtering

Not all GigaMUGA markers appear in `GM_ccf1fo_mm39.csv`. A marker's position is retained only
when, in `gm_uwisc_v4.csv`:

1. `unique == TRUE` (single-position mapping on GRCm39)
2. `unmapped == FALSE`
3. Has a valid `bp_grcm39` position
4. On autosomes 1–19 or chrX (chrY and chrM handled separately via
   `GigaMUGA_founder_consensus_genotypes_Y.csv` / `_Mt.csv`)

This yields 137,280 markers from `gm_uwisc_v4.csv` alone. The row set is then further
restricted to positions with real `CC_consensus_genotypes.csv` data (133,863 of the 137,280) —
founder/F1 coverage does not gate which markers get a row; a marker with CC data but no
founder-consensus call simply gets `N` in every founder/F1 column.

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

Genetic positions (`gmap`, cM) and physical positions (`pmap`, bp on GRCm39) are both derived
from `gm_uwisc_v4.csv` per marker; see `genotype/genomecache.py` for how `GenomeCache` actually
stores and accesses them.

---

## Files Currently in qtl/input/

| File | Size | Role |
|---|---|---|
| `CC_consensus_genotypes.csv` | 21 MB | Layer-0: 76 CC strain consensus genotypes |
| `cc_sample_metadata.csv` | 2.4 KB | Layer-0: CC strain metadata (Mt, Y, funnel, origin) |
| `gm_uwisc_v4.csv` | 17 MB | Layer-0: GigaMUGA marker positions + Cox map on GRCm39 |
| `GigaMUGA_founder_metadata.csv` | 8.8 KB | Layer-0: founder + F1 sample → strain letter mapping |
| `GigaMUGA_founder_consensus_genotypes_Mt.csv` | 616 B | Layer-0: mitochondrial haplogroups |
| `GigaMUGA_founder_consensus_genotypes_Y.csv` | 529 B | Layer-0: Y-chr haplogroups |
| `GigaMUGA_genotypes.csv.gz` | 8.5 MB | Layer-0: all reference sample genotypes (compressed) |
| `GigaMUGA_sample_metadata.csv` | 12 KB | Layer-0: all reference sample metadata |
| `gene_info.vm32.csv` / `gene_info.vm36.csv` | 2.8–4.1 MB | Gene annotations (two Vombatus marsupialis builds) |
| `gene_tss.vm32.csv` / `gene_tss.vm36.csv` | 12–24 MB | Gene TSS positions |
| `sample_metadata.csv` | 543 KB | Study sample metadata (sex, batch, tissue) |
| `GM_ccf1fo_mm39.csv` | 17.68 MB | Rebuilt intermediate — 133,863 markers × 51 cols; output of `build_gm_ccf1fo.py` |
| `old/cc_crossinfo.csv` | — | HMM funnel priors derived from `cc_sample_metadata.csv` |

---

## Reproducing GM_ccf1fo_mm39.csv from Scratch

All Layer-0 inputs are publicly available. The full rebuild (Layer-0 → HDF5 cache) requires:

```bash
# 1. Download Layer-0 files from JAX DO Reference Data page
#    https://www.jax.org/research-and-faculty/genetic-diversity-initiative/tools-data/diversity-outbred-reference-data
#    Files needed: gm_uwisc_v4.csv, GigaMUGA_genotypes.csv.gz,
#                  GigaMUGA_founder_metadata.csv, CC_consensus_genotypes.csv,
#                  cc_sample_metadata.csv

# 2. Rebuild GM_ccf1fo_mm39.csv (marker filter, founder consensus, F1 derivation, CC join)
conda run -n bioinfo python validation/tools/build_gm_ccf1fo.py

# 3. Prepare cc_crossinfo.csv (from cc_sample_metadata.csv funnel codes)
#    python tools/prep_cc_crossinfo.py \
#        --metadata cc_sample_metadata.csv \
#        --out cc_crossinfo.csv

# 4. Run the R HMM pipeline to build the HAPPY genomecache from GM_ccf1fo_mm39.csv
Rscript scripts/r_reference/build_genomecache.R [base_dir] [n_cores] [step_cM]

# 5. Convert the HAPPY genomecache to HDF5
conda run -n bioinfo python validation/validate_tissue.py --build-cache
```

Step 2 is the fully working, in-repo rebuild script (`validation/tools/build_gm_ccf1fo.py`) —
see "Derivation Chain" above for exactly what it does.

---

## Data Citations

| Dataset | Citation |
|---|---|
| GigaMUGA marker positions (`gm_uwisc_v4.csv`) | Morgan AP et al. *G3* 2016. doi:10.1534/g3.116.037622 |
| Founder consensus genotypes | Morgan AP et al. *G3* 2016; JAX DO Reference Data page |
| CC consensus genotypes (`CC_consensus_genotypes.csv`) | Collaborative Cross Consortium. *Genetics* 2012. doi:10.1534/genetics.113.153452 |
| Sanger MGP variants (`mgp_REL2021_snps_founders.vcf.gz`) | Keane TM et al. *Nature* 2011. doi:10.1038/nature10413; Ensembl REL2021 |
| R/qtl2 support files | Broman KW et al. *Genetics* 2019. doi:10.1534/genetics.119.302286 |

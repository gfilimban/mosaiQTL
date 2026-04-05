# mosaiQTL

Haplotype-based QTL mapping for multiparent populations — parametric on $J$ founders,
model-agnostic across expression data types.

Maps genetic effects to $J$ inbred founder haplotypes across any number of mosaic lines.
Designed for populations where each genome is a unique recombinant mosaic of segments
from $J$ founders: Collaborative Cross (CC, $J=8$), Diversity Outbred (DO), or any
user-defined MAGIC design.

---

## Three-layer API

```python
from mosaiqtl import discover, characterize, resolve

# Layer 1 — genome-wide QTL scan
disc = discover(counts, gene_ids, cache, covariates, sample_ids, model='nb')

# Layer 2 — characterize discovered QTLs
char = characterize(disc, covariates, modules=['allele_effects', 'vqtl'])

# Layer 3 — mechanistic resolution (pass only modules your data support)
res = resolve(disc, covariates,
              modules=['snp_catalogue', 'conditional_scan', 'ase', 'sqtl'],
              snp_catalogue_kwargs    = {'mgp_vcf': '/path/mgp.vcf.gz'},
              conditional_scan_kwargs = {'n_signals': 3},
              ase_kwargs              = {'f1_fastq_dirs': {...},
                                         'founder_fastas': {...},
                                         'gtf_path': '/path/gencode.gtf'},
              sqtl_kwargs             = {'psi_matrix': psi, 'junction_ids': jids})
```

---

## Models

| Model | Function | Use case |
|---|---|---|
| **NB-GLMM + LOCO** (primary) | `scan_nb()` | Raw pseudobulk counts — PQL null with LOCO kinship, score test |
| **NB GLM** (fallback) | `scan_nb(..., kinship_loco=None)` | No kinship correction — σ² inflation only |
| **Gaussian LMM** | `scan_h2lmm()` | Normalized expression — LOCO kinship, FWL vectorised |
| **Dispersion QTL** | `scan_vqtl()` | Transcriptional noise — scan log(φ̂) as phenotype |
| **Epistasis** | `scan_epistasis()` | Pairwise GxG interaction at haplotype resolution |

The NB-GLMM uses PQL (Breslow & Clayton 1993; Chen et al. 2016 AJHG) to fit a null
NB mixed model with LOCO kinship once per chromosome per gene, then a vectorised
kinship-corrected score test across all loci via `einsum`. The LMM uses
Frisch-Waugh-Lovell for equivalent vectorisation. Both are parametric on $J$.

---

## Module inventory

```
mosaiqtl/
  genome_cache.py   GenomeCache — from_gigamuga(), from_happy_dir(), HDF5 save/load, collapse()
  hmm.py            forward-backward HMM, Haldane map transitions, emission from GigaMUGA
  kinship.py        calc_kinship() LOCO via einsum, decomp_kinship() eigendecomposition
  lmm.py            estimate_h2_null() REML+grid, fit_lmm_at_locus(), scan_chromosome() FWL
  scan.py           scan_h2lmm(), scan_multi_gene(), scan_h2lmm_mi(), instability_lm_scan()
  nb.py             scan_nb(), scan_nb_multi_gene(), score test, LRT, dispersion EB pipeline
  vqtl.py           scan_vqtl(), estimate_dispersion_per_subject(), compare_mean_vs_dispersion()
  epistasis.py      scan_epistasis(), test_haplotype_combinations(),
                    scan_conditional(), scan_conditional_with_y()
  mediation.py      mediation_scan_qr(), fixef_mediation_scan() — FWL vectorised
  snp_scan.py       imputed_snp_scan_h2lmm(), build_snp_catalogue() (pysam+MGP VCF)
  permutation.py    generate_null_outcomes(), get_gev_thresholds(), run_positional_scans()
  pipeline.py       discover(), characterize(), resolve() — Layer 1/2/3 user API
  ase.py            run_kallisto_ase() — F1 ASE via kallisto|bustools two-allele index
  plotting.py       LOD profiles, kinship heatmap, founder effects, QQ, CC founder palette
  utils.py          rint(), align_samples(), build_z_matrix(), expand_kinship()
```

---

## Z-matrix design (n ≥ m)

mosaiQTL operates natively on $n$ samples mapping to $m \leq n$ genotyped strains.
The Z-matrix $Z \in \{0,1\}^{n \times m}$ ($Z_{ik} = 1$ if sample $i$ belongs to
strain $k$) expands haplotypes and kinship at scan time:

$$H_n = Z H_m \qquad K_n = Z K_m Z^\top$$

When $n = m$ and IDs match, $Z = I_m$ and all results are identical to the strain-mean
case. No hardcoded sample counts anywhere in the codebase.

---

## Quick start

```python
from mosaiqtl.genome_cache import GenomeCache
from mosaiqtl import scan_nb, scan_h2lmm

# build cache from GigaMUGA array data (R-free)
cache = GenomeCache.from_gigamuga(
    gigamuga_path = 'input/GM_ccfo_mm39.csv',
    gmap_dir      = 'input/GM/gmap/',
    pmap_dir      = 'input/GM/pmap/',
    cc_cols       = CC_COLS,
    founder_cols  = FOUNDER_COLS,
    f1_crosses    = F1_CROSSES,
)
cache.save('output/cache.h5')            # save once
cache = GenomeCache.load('output/cache.h5')  # fast reload

# NB GLM scan (primary model — raw counts)
result = scan_nb(counts, cache, covariates=X, sample_ids=sids,
                 lib_sizes=lib, just_these_loci=cis_set)

# Gaussian LMM scan (normalized expression)
result = scan_h2lmm(y_rint, cache, covariates=X, sample_ids=sids)

peak = result.peak_locus()   # {'locus_id', 'chromosome', 'position', 'lod'}
df   = result.to_dataframe() # pd.DataFrame all loci
```

---

## Validation

```bash
# Phase 2 — full port validation (all 11 sections, genome-wide)
# Run R reference first on HPC:
bash validation/run_validation_slurm.sh
# Then merge and validate:
Rscript validation/merge_chr_outputs.R
python validation/validate_port.py

# Rerun failed chromosomes:
bash validation/rerun_failed_chrs.sh 3,5

# Specific sections only:
python validation/validate_port.py --sections 4,6
```

Validation outputs: `validation/port_validation/py_data/qc_summary.csv`

---

## Development status

**Phase 1 — LMM port:** ✅ complete  
**Phase 2 — Full port validation (S1–S11):** ✅ complete (genome-wide, all 20 chr)  
**Phase 3 — Novel extensions validation:** 🔄 in progress

| Extension | Module | Status |
|---|---|---|
| NB GLM scan | `nb.py` | ✅ scaffolded — end-to-end test pending |
| Dispersion vQTL | `vqtl.py` | ✅ scaffolded — end-to-end test pending |
| Epistasis scan | `epistasis.py` | ✅ implemented |
| Conditional scan (BLUP) | `epistasis.py` | ✅ implemented |
| SNP catalogue (MGP VCF) | `snp_scan.py` | ✅ implemented |
| ASE validation (kallisto) | `ase.py` | ✅ scaffolded — F1 FASTQs required |
| sQTL integration | `pipeline.py` | ✅ scaffolded — PSI matrix required |

---

## Documentation

- `docs/PORT_VALIDATION_REPORT.md` — S1–S11 math, pass criteria, CSV column reference
- `docs/POC_REPORT.md` — Layer 1/2/3 design, model comparison (M1–M9), epistasis math
- `docs/ARCHITECTURE.md` — Z-matrix design, HMM details, FWL derivation

---

## Installation

```bash
pip install -e .
```

**Required:** `numpy scipy pandas h5py matplotlib seaborn edgepython`  
**Optional:** `rpy2` (HAPPY `.RData` readers), `pysam` (SNP catalogue), `kallisto bustools` (ASE)

---

## References

- Kang, H.M. *et al.* (2008). *Genetics* 178:1709 — eigendecomposition LMM
- Broman, K.W. *et al.* (2019). *Genetics* 211:495 — R/qtl2
- Mott, R. *et al.* (2000). *PNAS* 97:12649 — HAPPY HMM
- Churchill, G.A. *et al.* (2004). *Nat Genet* 36:1133 — Collaborative Cross
- Sullivan, D.K. *et al.* (2024). *Nat Protoc* — kallisto|bustools

## License

GPL-3.0

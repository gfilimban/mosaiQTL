# mosaiQTL

**Haplotype-based QTL mapping for multiparent mouse populations.** mosaiQTL tests whether a gene's expression
depends on which of the 8 founder haplotypes a strain carries near it, using a linear mixed model with
leave-one-chromosome-out (LOCO) kinship. It was built for single-nucleus pseudobulk data from the Collaborative
Cross (CC lines, founders and F1s; mm39) and calls eGenes, founder effects and fine-mapped variants.

> **Status — v0.7.2.** The Gaussian LMM is the validated production model, run by the Nextflow workflow and the
> CLI. The negative-binomial GLMM (`qtl/nb.py`) is experimental and available from the Python API only.

## How it works

| Step | What happens |
|---|---|
| 1. Haplotypes | An HMM turns GigaMUGA marker calls into founder probabilities per strain and locus (genome cache). |
| 2. Kinship | LOCO kinship from those probabilities, eigendecomposed once per chromosome. |
| 3. Scan | Per gene, $h^2$ is fitted once by maximum likelihood; every locus within ±1 Mb of the TSS is then tested in one batched solve (LOD). |
| 4. Significance | Each gene's peak is calibrated against its own simulated null (Beta fit), then Storey q-values are computed over all tested genes. |
| 5. Fine-mapping | For eGenes, Sanger SNPs in the 1.5-LOD support interval get Bayes factors; posterior probability (PIP) is shared within founder-allele patterns. |

The derivations are in [`docs/METHODS.typ`](docs/METHODS.typ) and [`docs/math/`](docs/math/).

## Install

```bash
git clone https://github.com/gfilimban/mosaiQTL.git && cd mosaiQTL
conda create -n mosaiqtl python=3.13 && conda activate mosaiqtl
pip install -e .            # add '.[dev]' for tests and linting
```
Optional: `pysam` (build the Sanger allele file from the MGP VCF), `rpy2` (convert R-format genome caches).

## Run the pipeline (Nextflow)

```bash
# check the DAG without running anything
nextflow run workflow/main.nf -profile local -stub-run -params-file config/default_params.yaml

# production run on SLURM
nextflow run workflow/main.nf -profile slurm -params-file config/gastroc_hpc_params.yaml -resume
```

Copy `config/default_params.yaml` and set at least:

| Parameter | Meaning |
|---|---|
| `celltypes_dir` | Folder with one subfolder per cell type, each holding `counts_lognorm.h5` |
| `metadata` | Sample sheet: `sample_id`, `genotype` (strain) and the covariate columns |
| `gene_info`, `gene_tss` | Gene coordinates and strand-aware transcript TSSs |
| `sanger_alleles` | Founder SNP alleles (HDF5); fine-mapping is skipped when absent |
| `scan_type`, `window_mb` | `L` (local, default) with a ±1 Mb window; `C` chromosome-wide; `G` genome-wide |
| `covariates` | Fixed effects, default `sex,dataset` |
| `call_rule`, `fdr_threshold` | `qvalue` (default), `bh` or `lfdr`; eGene cut, default 0.05 |

The genome cache and LOCO kinship are built on the first run (`ref/igvf_cache.h5`, `ref/igvf_kinship_loco.h5`)
and reused afterwards. Every run writes a Nextflow trace to `<output_dir>/logs/trace.txt`.

## Inputs

| Input | Format |
|---|---|
| Genotypes | GigaMUGA calls (CSV) for founders, F1s and CC lines; turned into the genome cache (HDF5) by `build-cache` or the first Nextflow run |
| Expression | One `counts_lognorm.h5` per cell type: `counts` (genes × samples, log-normalised), `genes`/`gene_ids`, `samples`/`sample_ids` |
| Sample metadata | CSV with `sample_id`, `genotype` (strain) and covariate columns (numeric used as is, text one-hot encoded) |
| Gene info | CSV with `gene_id`, `chr`, `start`; optional `end`, `strand` (strand-aware TSS), `gene_symbol` |
| Transcript TSS | Optional CSV of strand-aware TSSs, used for cis distances |
| Sanger alleles | Optional HDF5 of founder SNP alleles, built once from the MGP REL2021 VCF (`build_founder_allele_mgp`) |

## Outputs

Files are named `{tissue}_{celltype}_{scan_tag}_…` (for example `gastroc_type_ii_myonucleus_Lw1_peak_lod.csv`).
Every column is described in [`docs/output_schema.md`](docs/output_schema.md).

| File | Contents |
|---|---|
| `*_peak_lod.csv` | Per gene: peak locus, LOD, $h^2$, founder effects, cis distance, `is_egene` |
| `*_lod_genome.csv` | LOD at every scanned locus |
| `*_haplotype_eqtl_summary.csv` | eGene calls: calibrated p-value, q-values (Storey, BH), $\hat\pi_0$, significance |
| `*_perm_max_lods.csv`, `*_perm_gev_params.json` | Permutation null and GEV threshold (quality control) |
| `*_eqtl_ci.csv` | 1.5-LOD support interval per eGene |
| `*_eqtl_finemapping.csv` | Per-variant fine-mapping: PIP, credible set, Bayes factor, founder alleles |
| `*_eqtl_finemapping_groups.csv` | One row per founder-pattern group: group PIP, variants, span |
| `*_eqtl_finemapping_peak.csv` | Lead variant per eGene (lead PIP > 0.25) |
| `*_eqtl_stats.csv`, `*.png` | Run summary and figures |

## Command line and Python API

The `mosaiqtl` command runs the same steps one at a time (`build-kinship`, `scan-chr`, `merge-scan`, `permute`,
`calibrate`, `call-egenes`, `characterise`, `impute`, `finemap`, `report`, plus `scan-mi`, `mediate`,
`assess-stability`); `build-cache` is a separate command. Run `mosaiqtl <command> --help` for options. The
Nextflow workflow is the tested route; the CLI is useful for single steps and debugging. For the Python API see
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Known CLI limitations:** `mosaiqtl impute` fails (use Nextflow or `pipeline.finemap_egenes`), and
`calibrate`/`call-egenes` compute q-values only for genes passing a LOD gate, unlike Nextflow, which uses all
tested genes.

## Models

| Model | Entry point | Status |
|---|---|---|
| Gaussian LMM, LOCO kinship | `scan_h2lmm`, `scan_multi_gene_lmm` | Production (Nextflow, CLI) |
| Negative-binomial GLMM on raw counts | `scan_nb_gene`, `pipeline.discover(model='nb')` | Experimental, Python API only |
| LMM averaged over haplotype uncertainty | `scan_h2lmm_mi` (CLI `scan-mi`) | Follow-up |
| Dispersion (variance) QTL | `scan_vqtl` | Follow-up, Python API |
| Mediation | `mediation_scan_qr` (CLI `mediate`) | Follow-up |
| Founder × founder epistasis | `scan_epistasis` | Follow-up, Python API |

## Documentation

| | |
|---|---|
| [`docs/METHODS.typ`](docs/METHODS.typ), [`docs/math/`](docs/math/) | Methods and derivations (Typst; `make docs-math` builds PDFs) |
| [`docs/output_schema.md`](docs/output_schema.md) | Every output column, its meaning and unit |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Modules, data flow, Python API |
| [`config/`](config/) | Parameter files for the gastrocnemius and cortex/hippocampus runs |
| [`CHANGELOG.md`](CHANGELOG.md) | Changes by release |

## Development

```bash
pytest tests/unit -x -q     # fast tests
make lint                   # black, flake8, interrogate, mypy
make bench                  # performance benchmarks (benchmarks/README.md)
```
Work on `dev`; open pull requests to `main` using the PR template, with a `CHANGELOG.md` entry for any change to
`src/` or `workflow/`.

## License

GPL-3.0-or-later.

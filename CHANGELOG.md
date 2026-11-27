# Changelog
Format: [Keep a Changelog](https://keepachangelog.com). One line per user-visible change under Added, Changed,
Fixed or Removed, written in the PR that makes it. Releases follow pyproject.toml and are tagged vX.Y.Z.

## [Unreleased]
### Changed
- **Breaking:** every table the pipeline writes is tab-separated with a `.tsv` extension (`lod_chr*.tsv`,
  `{tag}_peak_lod.tsv`, `{tag}_peak_lod_all.tsv`, `{tag}_haplotype_peak_lod.tsv`, `{tag}_eqtl_summary.tsv`,
  `perm_max_lods.tsv`, `threshold_*.tsv`, `{tag}_eqtl_stats.tsv`, the fine-mapping tables, and the cell-type
  comparison tables); the merged fine-mapping table is `{tag}_eqtl_finemapping.tsv.gz`. Inputs under `ref/`
  (`gene_info`, metadata, `gene_tss`, GigaMUGA, maps) stay CSV. Run into a new output directory, or convert existing
  outputs with `validation/tools/csv_to_tsv.py`. A default `pd.read_csv` on a `.tsv` returns one column: use
  `mosaiqtl.io.tables.read_table` (tab separator, `T`/`F` booleans, gzip by extension).
- `atomic_write_csv` and `atomic_concat_csv` are renamed `atomic_write_table` and `atomic_concat_table`;
  `CSV_TRUE` and `CSV_FALSE` are renamed `TABLE_TRUE` and `TABLE_FALSE`.
### Added
- `validation/tools/convert_h5ad.py` writes `excluded_samples_{celltype}.tsv` per cell type (and
  `excluded_samples_all.tsv` with `--all_celltypes`): each sample dropped by the `min_counts`/`min_cells` gate with
  its pseudobulk counts, cells and reason, and whether the whole cell type was skipped for too few valid samples.
- `cis_dist_gene_tss_mb` in `{tag}_peak_lod.csv` and the eGene summary: signed distance from the peak to one fixed
  TSS per gene, the most upstream transcript TSS in `gene_tss` (`utils.gene_tss_anchor`). Unlike `cis_dist_mb`, whose
  anchor is the transcript TSS nearest the peak, it does not change with the peak and is comparable across cell types.
### Changed
- `FINEMAP_CHR` uses the `process_ultra` label (16 CPUs, 128 GB, 12 h).

## [0.7.3] — 2026-11-25
### Changed
- `calibrate_egenes_storey` is renamed `calibrate_egenes`: the call is no longer tied to the Storey q-value.
- eGene calling defaults to Benjamini-Hochberg (`call_rule='bh'`, `--call-rule bh`, `params.call_rule: 'bh'`) at
  `fdr_threshold` 0.05, and a gene is `significance` 'T' / `is_egene` only if its peak LOD also exceeds the
  permutation (GEV) LOD threshold (`lod_threshold` on `calibrate_egenes` and `call_significant_egenes`). The
  q-values are still computed over every tested gene; the LOD gate applies to the call only.
- `{tag}_haplotype_peak_lod.csv` keeps the tested genes with `qval_bh` <= `submission_fdr` (0.1), whatever the call
  rule; `{tag}_peak_lod_all.csv` carries `qval_bh`, `qval_storey` and `lfdr` for every tested gene.
- `atomic_write_csv` writes floats at full precision (it capped most tables at nine significant digits, and the
  eQTL peak, eQTL stats and LD-group tables at six; the `float_format` argument and `CSV_FLOAT_FORMAT` are removed)
  and writes every boolean column (`is_egene`, `peak_at_edge`, `h2_at_boundary`, `ci_truncated`, `in_cs`) as `T`/`F`
  instead of `True`/`False`. Read them with `true_values=['T'], false_values=['F']`.
- The merged fine-mapping table is written gzip-compressed as `{tag}_eqtl_finemapping.csv.gz` (per-chromosome parts
  stay plain CSV); `atomic_write_csv` and `atomic_concat_csv` compress any destination ending in `.gz`.

## [0.7.2] — 2026-11-22
### Added
- `fdr.bh`; `call_rule` (`'qvalue'` default, `'bh'`) on `calibrate_egenes_storey` and
  `call_significant_egenes`, CLI `call-egenes --call-rule`, Nextflow `params.call_rule`. `qval_storey`, `qval_bh`
  and `pi0` are written to the eGene summary, peak and fine-mapping tables; `fdr`/`significance` follow the rule.
- `fdr.lfdr` (local false discovery rate) and call rule `'lfdr'` on `calibrate_egenes_storey` and
  `call_significant_egenes`, CLI `--call-rule lfdr`, Nextflow `params.call_rule: 'lfdr'`; `lfdr` is written beside `qval_storey` and `qval_bh` (NaN below 10 tested genes).
- `fdr.qvalue` (renamed from `storey_qvalue`) warns when $\hat\pi_0 < 0.2$ (the call then passes every gene at thresholds above $\hat\pi_0$).
- `make bench` / `make bench-trace` (benchmarks/record.py): append-only benchmarks/history.csv and
  pipeline_history.csv; Nextflow trace written to `<output_dir>/logs/trace.txt`.
- `make docs-math`; CHANGELOG.md and a CI check that PRs changing src/ or workflow/ update it.
- `{tag}_eqtl_finemapping_groups.csv`: one row per LD group per eGene (`group_pip`, `patterns`, `n_variants`,
  `span_bp`, `best_lod_variant`, `in_cs`), written by `FINEMAP_CHR`, merged by `FINEMAP_MERGE`
  (`imputation.ld_group_summary`; docs/math/12 §LD-Group Summary). `group_pip` sums to 1 per gene.
- `log_abf` (natural log of the association-over-null approximate Bayes factor) in the fine-mapping tables;
  `pip` stays a linear probability.
- The fine-mapping peak file lists only eGenes whose lead variant has `pip` > 0.25 (`PEAK_PIP_THRESHOLD`); the
  log reports how many eGenes the filter left out. A 0.5 cut left most chromosomes with almost no peak rows
  because variants sharing a founder allele pattern have identical ABF and split the posterior. The per-variant and LD-group tables keep every eGene.
- `beta_a`, `beta_b`: the fitted Beta shape parameters behind `pval_perm`, written to the peak table so each
  p-value can be reproduced from its null; `_calibrate_pvalues_loop` returns them instead of discarding them.
- `peak_at_edge` in `{tag}_peak_lod.csv`: true when the scan maximum is the first or last locus of a local
  window. In S1 null simulations about 40% of peaks sit within 50 kb of a window edge at every $h^2$; p-values are
  unaffected, peak position and the support interval are truncated (`ci_truncated` already marks the interval).
- docs/output_schema.md: every written column with its meaning and unit, enforced by `tests/unit/test_output_schema.py`.
### Fixed
- HMM transition kernel (`hmm.py`): diplotype rows now sum to 1 exactly (the unordered-diplotype kernel sums the
  two ordered outcomes of a heterozygous target; the old code halved them and renormalised, giving row sums 0.996
  and 0.502 at $r = 0.005$ before renormalisation). The haplotype kernel is the exchangeable mixture
  $H = e^{-d} I + (1 - e^{-d})\mathbf{1}\mathbf{1}^\top/J$ and reaches the uniform law at $r = 0.5$.
  Fully inbred input (`cc_inbred=True`) runs an 8-state homozygous chain, about 20x cheaper, with no heterozygous
  mass at any marker; `cc_inbred=False` keeps the 36-state chain. **Cache posteriors change: rebuild
  `igvf_cache.h5`.** On chr19 (3578 loci, 33 strains) 5.4% of locus-strain cells move by more than 0.01 and 1.8%
  change their top founder, concentrated near breakpoints and in already ambiguous cells.
- Scan p-values (`lmm.py`) and haplotype/GxE p-values (`effects.py`) use the survival function instead of
  `1 − cdf`, which underflowed to exactly 0 for strong signals; values are otherwise unchanged.
- `GenomeCache.from_hdf5` raises when `sample_ids` differ between chromosomes or a `probs` shape disagrees with
  its loci, founders and samples (previously only the first chromosome's IDs were read).
- Categorical covariate encoding no longer crashes on mixed-type columns.
- `impute_egene`: alternative allele chosen by majority then lexicographic order (was set-iteration order, which
  varied between processes); `se_effect_size` taken from the fitted Wald SE (`se_dosage`) with the
  $|\beta|/|z|$ inversion only as fallback; an empty credible set no longer raises in `build_eqtl_finemapping_df`.
- `effects._locus_fwl` raises on a singular covariate design instead of returning zero effects.
- Fine-mapping row order was `pip` descending with ties in input order, so the lead variant among equal-`pip`
  LD-group members depended on input order. Ties now break on higher `lod`, then smaller distance to the
  strand-aware TSS (`fine_map_region(tss_mb=…)`); the lead is the first row.
- `estimate_h2_null` searched only $h^2 \ge 0.01$ or the floor; the grid now includes log-spaced points from
  $10^{-6}$, so an optimum in $(10^{-6}, 0.01)$ is reached (docs/math/03_lmm_scan.typ §2). Heritability estimates,
  LODs and p-values of genes whose optimum lay below 0.01 change.
- The permutation null, the support-interval scan and the imputed-SNP scan always applied the rank-inverse
  normal transform, whatever `use_rint` the haplotype scan used. `run_permute.py` and `run_finemap.py` now take
  `use_rint` (`permute.nf`, `finemap_chr.nf` pass `params.use_rint`); `impute_egene`, `impute_egenes` and
  `finemap_egenes` require a keyword `use_rint`; CLI `impute` requires `--rint/--no-rint`.
- `cis_dist_mb` in `{tag}_peak_lod.csv` was measured from the gene's leftmost coordinate, wrong for minus-strand
  genes. `enrich_peak_lod` now anchors on the strand-aware TSS: the nearest transcript TSS from `gene_tss` when
  given (`near_tss_mb`), else the gene_info TSS (`start_mb`). Genes join on the unversioned `gene_id`, so
  versioned and unversioned tables match. `MERGE_RESULTS` and `run_merge.py` take `gene_tss` (`-` when absent).
- claude-review.yml had a duplicated `jobs:` key; benchmark commands in benchmarks/README.md were invalid.
### Changed
- Version 0.7.2 across pyproject.toml, the Nextflow manifest and docs; README rewritten as a short user guide
  (install, Nextflow run, outputs, documentation map).
- docs/math/*.md and docs/METHODS.md → `.typ`: LaTeX math typeset by Typst via mitex (docs/math/_setup.typ).
- PR template: documentation and compute checks; Claude review workflow migrated to claude-code-action v1 inputs.
### Removed
- `modules/local/finemap.nf`: unused by `main.nf` and incompatible with `run_finemap.py`; `finemap_chr.nf` is the
  fine-mapping module.

## [0.7.0] — not yet tagged
35 commits after v0.6.7-alpha (8 feat, 20 fix, 2 refactor, 5 chore): `git log --oneline v0.6.7-alpha..` until v0.7.0 is tagged.

## [0.6.7-alpha] and earlier
See `git log --oneline v0.6.7-alpha`.

#import "_setup.typ": *
#show: mathdoc

= Haplotype-Compressed Approximate Bayes Factors for CC/DO Fine-Mapping
<haplotype-compressed-approximate-bayes-factors-for-ccdo-fine-mapping>
== Background
<background>
The Collaborative Cross (CC) and Diversity Outbred (DO) populations derive from eight
inbred founder strains. At any genomic position an animal carries at most two of the
eight founder haplotypes, so the genotype of each animal at any SNP is fully determined
by its 8-dimensional haplotype probability vector and the SNP allele pattern across the
eight founders. This gives rise to an important simplification: at most #mi(`2^8 = 256`);
distinct biallelic allele patterns can exist across all SNPs in any genomic window. In
practice, within the fine-mapping window (the eGene's 1.5-LOD support interval, capped
to #mi(`\pm 1`); Mb of the TSS in L mode), 20--100 distinct patterns are typical.

== Standard Single-Causal-Variant ABF (no LD)
<standard-single-causal-variant-abf-no-ld>
Given #mi(`n_{\text{snps}}`); association statistics (z-scores #mi(`z_j`); with standard errors #mi(`\mathrm{se}_j`);), the
Wakefield approximate Bayes factor for variant #mi(`j`); under the single-causal-variant model is

#mitex(`\log \mathrm{ABF}_j = \tfrac{1}{2}\log\frac{V_j}{V_j + W} + \frac{z_j^2\, W}{2(V_j + W)}`)

where #mi(`V_j = \mathrm{se}_j^2`);, #mi(`W`); is the prior effect variance (default 0.04 for cis-eQTL on
standardized phenotypes), and #mi(`z_j = \hat\beta_j / \mathrm{se}_j`); is the Wald z-score of the
LMM dosage effect (`beta_dosage / se_dosage`). SNPs with a non-finite or non-positive
#mi(`\mathrm{se}_j`); get a NaN ABF and near-zero posterior weight.

The posterior inclusion probability (PIP) under a uniform prior over #mi(`n_{\text{snps}}`); variants is

#mitex(`\mathrm{PIP}_j = \frac{\mathrm{ABF}_j}{\sum_{k=1}^{n} \mathrm{ABF}_k}`)

With #mi(`n_{\text{snps}}`); in the thousands (a 2 Mb window at #mi(`\sim 7 \times 10^{7}`); Sanger SNPs genome-wide gives #mi(`\sim 2000`);
SNPs per window), PIPs become small and uniformly distributed even when a true causal
signal is present --- the signal is diluted by the dense LD block it sits in.

== LD Structure in CC/DO Populations
<ld-structure-in-ccdo-populations>
In CC/DO data, linkage disequilibrium between SNPs arises entirely from shared founder
haplotypes, not from human-like population LD. Two SNPs that map to the same cache locus
are in perfect LD when they have identical founder allele patterns (same 8-bit vector of
#mi(`\{0,1\}`); values across the eight founders). More generally, LD between two SNPs #mi(`i`); and #mi(`j`); equals the squared
Pearson correlation between their imputed dosage vectors:

#mitex(`r^2(i,j) = \operatorname{Corr}(d_i, d_j)^2`)

where #mi(`d_k[s] = \sum_f H[s,f]\, g_k[f]`);, #mi(`H[s,f]`); is the haplotype probability of strain #mi(`s`);
for founder #mi(`f`); at the cache locus nearest SNP #mi(`k`);, and #mi(`g_k[f] \in \{0, 1\}`); is 1 when founder #mi(`f`);'s
call at SNP #mi(`k`); differs from the REF founder (`B6J`) call (`make_alt_mask`; missing calls are coded 0).

The key structural observation: two SNPs with the same 8-bit founder allele pattern that
map to the same nearest cache locus will have identical dosage vectors, giving
#mi(`r^2 = 1`);. Two SNPs with different patterns may still be highly correlated if their
patterns differ only for founders that are rare in the sample.

== Haplotype LD Groups
<haplotype-ld-groups>
A haplotype LD group is a set of SNPs whose founder allele patterns are linked by
pairwise correlations #mi(`r^2 \ge`); threshold. We form groups by:

+ Collecting the set of unique 8-bit founder allele patterns #mi(`P = \{p_1, \dots, p_K\}`);,
  #mi(`K \le 256`);. For each pattern #mi(`p_k`);, select the highest-LOD SNP carrying that pattern as
  the representative #mi(`r_k`);.

+ Computing the #mi(`K \times K`); pairwise #mi(`r^2`); matrix between representative dosage vectors. This
  matrix has size at most #mi(`256 \times 256`);, regardless of how many SNPs are in the window.

+ Applying a union-find merge: representatives #mi(`i`); and #mi(`j`); are merged into the same group
  when #mi(`r^2(r_i, r_j) \ge`); threshold (default 0.8). The canonical group id is the union-
  find root. Groups are the transitive closure of this rule (single linkage), so two
  representatives in one group can have #mi(`r^2`); below the threshold.

This produces #mi(`G \le K \le 256`); groups, each containing one or more patterns (and all SNPs
carrying those patterns).

== LD-Aware ABF
<ld-aware-abf>
Instead of placing a uniform prior over #mi(`n_{\text{snps}}`); variants, place it over #mi(`G`); LD groups.
The group-level ABF is the maximum ABF among group members:

#mitex(`\mathrm{ABF}_g = \max_{j \in g} \mathrm{ABF}_j`)

Known issue: the running maximum is seeded with the first member's ABF, so a NaN first
member leaves the whole group at NaN (near-zero PIP).

Group-level PIP:

#mitex(`\mathrm{PIP}_g = \frac{\mathrm{ABF}_g}{\sum_{g'=1}^{G} \mathrm{ABF}_{g'}}`)

Individual SNP PIP: the group\'s posterior mass is distributed equally among its members:

#mitex(`\mathrm{PIP}_j = \frac{\mathrm{PIP}_{g(j)}}{\lvert g(j) \rvert}`)

where #mi(`g(j)`); is the group containing #mi(`j`); and #mi(`\lvert g(j) \rvert`); is the number of members.

== Credible Set Construction
<credible-set-construction>
Grow the credible set group-by-group from highest to lowest group PIP until cumulative
mass #mi(`\ge`); coverage (default 0.95). All members of each included group enter the credible
set. There is no guarantee on internal #mi(`r^2`);: groups are single-linkage, and a credible
set that spans more than one group places no constraint on #mi(`r^2`); between groups.

== Why This Matters for CC/DO
<why-this-matters-for-ccdo>
+ #strong[Restores interpretable PIPs.] Dense LD blocks no longer dilute group PIPs to
  near zero. A true causal haplotype block receives #mi(`\mathrm{PIP} \approx 1/G`); where #mi(`G`); is the number of
  independent haplotype signals --- typically 1--5 in a 2 Mb cis window. The per-SNP
  `pip` column is #mi(`\mathrm{PIP}_g / \lvert g \rvert`); and stays small for large groups.

+ #strong[Compute-frugal.] The LD matrix is #mi(`K \times K`); (#mi(`\le 256 \times 256`);), not #mi(`n_{\text{snps}} \times n_{\text{snps}}`); (#mi(`\le 2000 \times 2000`);
  for a 2 Mb window). The dominant cost is one nearest-locus lookup + dosage
  computation per SNP, which the imputed scan already performs.

+ #strong[Biologically grounded.] In 8-founder populations, SNPs sharing a founder allele
  pattern are perfectly tagged by that pattern. The LD group is a natural unit of
  evidence: it represents the set of SNPs that are indistinguishable by the haplotype
  scan.

== Parameters
<parameters>
#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Parameter], [Default], [Description],),
    table.hline(),
    [`prior_var`], [0.04], [Prior effect variance #mi(`W`);; corresponds to prior SD #mi(`\approx 0.2`); for standardized cis-eQTL effect sizes],
    [`coverage`], [0.95], [Credible set coverage level],
    [`r2_threshold`], [0.8], [Minimum #mi(`r^2`); to merge two pattern representatives into the same LD group],
  )]
  , kind: table
  )

== Implementation
<implementation>
- `qtl/snp_scan.py`: per-SNP scan accumulates `_pat_best_dosage` dict (one representative
  dosage vector per unique founder allele pattern; #mi(`O(K \times n)`); storage where #mi(`K \le 256`);).
  Returns `df.attrs["_ld"]` carrying `rep_patterns` and `r2_matrix`, plus a per-row
  `_founder_pattern` column.

- `qtl/imputation.py::merge_ld_groups`: union-find merge of representatives; returns
  SNP-index → group-id mapping.

- `qtl/imputation.py::fine_map_region`: dispatches to LD-aware path when `ld_groups` is
  provided.

- `pipeline.py::impute_egene`: extracts `df.attrs["_ld"]` and the `_founder_pattern`
  column before slicing to `IMPUTED_SNP_COLS` and re-attaches them as attrs;
  `pipeline.py::impute_egenes` collects them into the `ld_by_gene` dict before
  `pd.concat` (which drops attrs) and returns `(df, ld_by_gene)`.

- `pipeline.py::finemap_imputed`: when `ld_by_gene` is given, calls `merge_ld_groups` and
  passes `ld_groups` to `fine_map_region`. Only `pipeline.py::finemap_egenes` (called by
  `workflow/scripts/run_finemap.py`) supplies `ld_by_gene`; the CLI `finemap` command
  reads the imputed-SNP CSV without it and uses the standard ABF.

== Relationship to Multi-Causal Models
<relationship-to-multi-causal-models>
The model above assumes at most one causal variant per region (single-causal-variant
assumption). This is appropriate for the 2 Mb cis window standard in eQTL analysis,
where most cis-eQTL signals are driven by a single regulatory variant or a few in tight
LD. Multi-causal models (#mi(`L > 1`); components) require additional machinery; the current
implementation does not attempt this. See `docs/REFERENCES.md` (SNP FINE-MAPPING section)
for references to multi-causal fine-mapping approaches.

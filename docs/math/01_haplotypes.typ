#import "_setup.typ": *
#show: mathdoc

= Haplotype Probability Inference
<haplotype-probability-inference>
#strong[Modules:] `genotype/hmm.py`, `genotype/genomecache.py` \ #strong[miQTL parallel:] `DiploprobReader$getLocusMatrix(locus, "additive")` + HAPPY cache format

#divider()

== 1. What the haplotype tensor represents
<1-what-the-haplotype-tensor-represents>
The core genotype object is:

#mitex(`H \in [0,1]^{m \times J \times L}, \quad \sum_{f=1}^{J} H_{i f \ell} = 1 \;\; \forall i, \ell`)

- #mi(`H_{if\ell}`); = posterior probability that strain #mi(`i`); at locus #mi(`\ell`); descended from founder #mi(`f`);
- Rows sum to 1 (probability simplex per strain per locus)
- Not the same as a SNP genotype --- never reduce to biallelic 0/1/2 for haplotype scans

The scan extracts one #mi(`m \times J`); slice per locus: #mi(`H_\ell = H[:, :, \ell]`);.

#divider()

== 2. Forward-backward HMM
<2-forward-backward-hmm>
#strong[Model.] Founder ancestry at each locus is modeled as a hidden Markov chain along
the chromosome (`hmm.py::calc_genoprob_cc`, run once per chromosome and per CC strain by
`GenomeCache.from_gigamuga`). The hidden state is an unordered #emph[diplotype]
#mi(`z_{i\ell} = (f_1, f_2)`);, #mi(`f_1 \leq f_2`);, giving #mi(`J(J+1)/2 = 36`); states for #mi(`J = 8`);.
Founder and F1 rows of #mi(`H`); are not inferred: founders are set to #mi(`e_f`); and each F1
listed in `f1_crosses` to #mi(`0.5\,e_{f_1} + 0.5\,e_{f_2}`); (`genomecache.py:118-122`).

#strong[Emission:] #mi(`P(\text{obs}_{i\ell} \mid z_{i\ell} = (f_1,f_2))`); --- probability of the
observed GigaMUGA call, encoded A/H/B relative to the two founder-derived alleles at the marker.
The expected call is the founders' common call if founders #mi(`f_1`); and #mi(`f_2`); agree,
otherwise H; with error rate #mi(`\varepsilon = 0.002`); (`build_emission_matrix`, `hmm.py:148-202`):

#mitex(`P(\text{obs} = o \mid (f_1,f_2)) = \begin{cases} 1 - \varepsilon & \quad o = \text{expected} \\ \varepsilon/2 & \quad \text{otherwise} \end{cases}`)

If either founder call is missing the emission is uniform (#mi(`1/3`);); a missing observation
contributes no emission factor.

#strong[Transition:] The recombination fraction between adjacent markers comes from the
Haldane map function applied to the genetic distance #mi(`d`); in Morgans (the cM map divided
by 100, `hmm.py:463-465`):

#mitex(`r = \tfrac{1}{2}\left(1 - e^{-2d}\right), \qquad r_{\text{eff}} = \min(n_{gen}\, r,\; 0.5)`)

A #mi(`J \times J`); haplotype kernel is built first (`transition_matrix`, `hmm.py:110-120`):

#mitex(`A_{gg'} = \begin{cases} 1 - r_{\text{eff}}\,(J-1)/J & \quad g = g' \\ r_{\text{eff}}/J & \quad g \neq g' \end{cases}`)

and the #mi(`36 \times 36`); diplotype kernel is assembled from it and row-normalised (`hmm.py:124-141`):

#mitex(`v_{(f_1 f_2)\to(g_1 g_2)} = c_{g_1 g_2} \times \begin{cases} A_{f_1 g_1} A_{f_2 g_2} + A_{f_2 g_1} A_{f_1 g_2} & \quad f_1 \neq f_2 \\ (1 + \mathbb{1}[g_1 \neq g_2])\, A_{f_1 g_1} A_{f_1 g_2} & \quad f_1 = f_2 \end{cases}, \qquad c_{g_1 g_2} = \begin{cases} 1/2 & \quad g_1 \neq g_2 \\ 1 & \quad g_1 = g_2 \end{cases}`)

#mitex(`T_{(f_1 f_2)\to(g_1 g_2)} = v_{(f_1 f_2)\to(g_1 g_2)} \Big/ \textstyle\sum_{(g_1' g_2')} v_{(f_1 f_2)\to(g_1' g_2')}`)

For the CC, #mi(`n_{gen}=1`);: the CC consensus genetic map already encodes the cumulative
recombination of the breeding programme, so #mi(`r`); is used directly; #mi(`n_{gen} > 1`); is
intended for outbred populations. With `cc_inbred=True` (default) the initial distribution puts
mass #mi(`1/J`); on each homozygous diplotype and 0 on heterozygous ones (`cc_inbred_init`);
otherwise it is uniform (#mi(`1/36`);).

Known issue: the #mi(`c_{g_1 g_2} = 1/2`); factor and the missing #mi(`g_1 \neq g_2`); guard for
heterozygous sources make #mi(`T`); deviate from the unordered-diplotype kernel (heterozygous
targets under-weighted; ledger HMM-01); `cc_inbred` only sets the prior at the first marker, so
heterozygous states regain mass along the chromosome (HMM-02); and because
#mi(`r_{\text{eff}} \leq 0.5`); the haplotype kernel never reaches the uniform distribution (HMM-03).

#strong[Forward variable:]

#mitex(`\alpha_{i s\ell} \propto P(\text{obs}_{i,1:\ell},\, z_{i\ell} = s)`)

computed left-to-right over diplotype states #mi(`s`);; renormalised to sum to 1 at every
marker to prevent underflow (the `log_scale` argument is unused).

#strong[Backward variable:]

#mitex(`\beta_{i s\ell} \propto P(\text{obs}_{i,\ell+1:L} \mid z_{i\ell} = s)`)

computed right-to-left, renormalised at every marker.

#strong[Posterior (forward-backward smoothing):]

#mitex(`\gamma_{i s\ell} = \frac{\alpha_{i s\ell} \cdot \beta_{i s\ell}}{\sum_{s'} \alpha_{i s'\ell} \cdot \beta_{i s'\ell}}`)

The 36 diplotype posteriors are collapsed to the 8 founder probabilities #mi(`H_{if\ell}`); as
described in Section 3 (`return_additive=True`, the default).

#divider()

== 3. Haploid vs diploid representation
<3-haploid-vs-diploid-representation>
#strong[miQTL / HAPPY format] estimates joint #emph[diplotype] probabilities:

#mitex(`P(f \times f' \mid i, \ell) \in \mathbb{R}^{J(J+1)/2}`)

For #mi(`J=8`);: 36 states (8 homozygous + 28 heterozygous diplotypes).

#strong[mosaiQTL] runs the same 36-state diplotype HMM (Section 2) and, by default
(`return_additive=True`; `from_gigamuga` always uses it, `genomecache.py:84-93`), collapses the
posteriors to #emph[additive] founder (allele) probabilities, weighting heterozygous diplotypes by
#mi(`1/2`); (`diplotype_to_allele_probs`, `hmm.py:49-73`):

#mitex(`H_{if\ell} = P(f \times f \mid i, \ell) + \tfrac{1}{2} \sum_{f' \neq f} P(f \times f' \mid i, \ell)`)

A cache built by `from_gigamuga` therefore stores only the #mi(`J`);-column additive probabilities;
diplotype probabilities from this HMM are available only from `calc_genoprob_cc(..., return_additive=False)`.

#strong[When this is exact:] For fully inbred CC strains, heterozygous diplotypes
#mi(`f \neq f'`); have near-zero probability (IBS #mi(`\approx`); IBD after many generations of
inbreeding). The marginal is therefore numerically equivalent to the diplotype diagonal.

#strong[When this matters (S8 SNP scan):] SNP imputation requires the diplotype-level
probabilities to model heterozygosity at each founder SNP. Using marginal probs
conflates homozygous and heterozygous states, producing the documented #mi(`r = 0.55`);
discrepancy between mosaiQTL and miQTL in the SNP scan (see `10_snp_scan.typ`).

#divider()

== 4. Cache formats: HAPPY vs HDF5
<4-cache-formats-happy-vs-hdf5>
#figure(
  align(center)[#table(
    columns: 3,
    align: (auto,auto,auto,),
    table.header([Property], [miQTL HAPPY (R)], [mosaiQTL HDF5 (Python)],),
    table.hline(),
    [Physical layout], [One `.RData` per locus per chromosome; e.g. `additive/chr8/data/rs12345.RData`], [Single `.h5` file; root `/founders`; one group per chromosome: `/chr8/probs` (gzip level 4), `/chr8/locus_ids`, `/chr8/positions` (Mb), `/chr8/sample_ids`],
    [Loading model], [`load()` called per locus; filesystem seek + R deserialise per call], [`h5py.File` kept open; `locus_ids`/`positions` read at load time; `probs` kept as `Dataset` refs, never eager-loaded],
    [Memory at startup], [Near zero (nothing loaded until scan)], [Locus ids and positions only (`probs` Dataset refs; `np.asarray()` called at `get_chromosome()`)],
    [Random access], [#mi(`O(1)`); seek but high overhead (\~1 ms/file on NFS)], [#mi(`O(\log L)`); HDF5 B-tree lookup; chunked, gzip-compressed read],
    [Per-chromosome load], [#mi(`L \times`); sequential file reads], [Single read of the chunked dataset: `np.asarray(d['probs'])`],
    [Probs shape], [`[n_strains, J]` per locus (read one at a time)], [`[n_loci, n_strains, J]` entire chromosome at once],
    [LOCO support], [No native support; workaround in `calc.kinship.from.genomecache.with.DOQTL`], [`calc_kinship(cache, method='loco')` native],
    [Memory per active chr], [Not applicable (streaming)], [#mi(`L \times m \times J \times 4`); bytes (float32, as written by `from_gigamuga`) #mi(`\approx 10`); MB for chr8 of `ref/igvf_cache.h5` (#mi(`L = 6591`);, #mi(`m = 48`);)],
  )]
  , kind: table
  )

#strong[Critical invariant:] `GenomeCache.from_hdf5` stores `h5py.Dataset` references for `probs` ---
never `grp['probs'][:]`. Full-genome eager loading would load \~206 MB for `ref/igvf_cache.h5`
(#mi(`133{,}863 \times 48 \times 8 \times 4`); bytes) at startup and violates the lazy-load architecture.
Known issue: `from_hdf5` takes `sample_ids` from the first chromosome group only and does not check
the other groups or the `probs` shape against them (ledger GC-01).

#divider()

== 5. Pseudomarkers and loci filtering
<5-pseudomarkers-and-loci-filtering>
miQTL inserts pseudomarkers at regular intervals between typed GigaMUGA markers to
improve QTL localisation. In mosaiQTL pseudomarkers are optional and decided at cache-build
time by `step_cM`; the scans do not filter loci and scan every locus in the cache.

- `from_gigamuga(step_cM=None)` (the default, used by the Nextflow `BUILD_CACHE` step via
  `workflow/scripts/build_cache.py`) keeps genotyped markers only; `ref/igvf_cache.h5` has
  0 pseudomarkers among its 133,863 loci.
- With `step_cM` set (the `build-cache` CLI defaults to `--step-cM 0.2`), `_augmented_positions`
  lays a grid from the first marker in steps of `step_cM` cM, drops grid points within
  `tol = 0.01` cM of a real marker, and `interpolate_posteriors` computes each pseudomarker's
  posterior from the flanking markers #mi(`j`); and #mi(`j+1`); (`hmm.py:278-375`):

#mitex(`\gamma_p \propto \left(\alpha_j\, T(r_1)\right) \odot \left(T(r_2)\,\big(E_{j+1,\,\text{obs}_{j+1}} \odot \beta_{j+1}\big)\right)`)

  where #mi(`r_1`);, #mi(`r_2`); are the Haldane recombination fractions from marker #mi(`j`); to the
  pseudomarker and from the pseudomarker to marker #mi(`j+1`);. Pseudomarker ids are
  `c{chr}.loc{cM}` and their physical positions are linearly interpolated from the map
  (`genomecache.py:95-107`).

History: an earlier S1 validation recorded a loci count delta of `loci_max_delta = 1237`
between mosaiQTL and miQTL.

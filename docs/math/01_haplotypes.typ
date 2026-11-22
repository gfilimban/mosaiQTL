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

#mitex(`r = \tfrac{1}{2}\left(1 - e^{-2d}\right)`)

where #mi(`d`); is the expected number of crossovers per meiosis (the single-meiosis map).

A #mi(`J \times J`); haplotype kernel #mi(`H`); gives the probability that the founder carried by one
chromosome at the next marker is #mi(`g`); given that it is #mi(`f`); at this one
(`haplotype_transition_matrix`). The kernel is that of a fixed line (recombinant inbred line, RIL)
founded by an eight-founder funnel and inbred by brother--sister mating, #mi(`J = 8`);. It is derived
by following the two loci of one fixed chromosome backwards through the pedigree.

#strong[Funnel.] Founders #mi(`A,\dots,H`); are inbred. The funnel pairs them as #mi(`A\times B`);,
#mi(`C\times D`);, #mi(`E\times F`);, #mi(`G\times H`); (two-way hybrids), then
#mi(`AB \times CD`); and #mi(`EF \times GH`); (four-way hybrids #mi(`Q_1, Q_2`);), then
#mi(`Q_1 \times Q_2`);. The offspring of that last cross are the first pair of siblings, generation
#mi(`0`);; they are mated to each other, their offspring likewise, and so on until every locus is fixed.

#strong[Backward chain through the sibling generations.] Fix two loci with recombination fraction
#mi(`r`);. Follow the lineage of each locus from a chromosome of generation #mi(`t`); back to
generation #mi(`t-1`);. At every generation the two lineages are in one of three states:

- #mi(`A`);: both loci on the same chromosome of one individual;
- #mi(`B`);: the loci on the two different chromosomes of one individual;
- #mi(`C`);: the loci in the two different siblings, one chromosome each.

Each chromosome of an individual is a gamete of one of its two parents, chosen with probability
#mi(`1/2`);, and the two parents of generation #mi(`t`); are the two siblings of generation
#mi(`t-1`);. Transitions:

- From #mi(`A`);: one gamete carries both loci. Without recombination (probability #mi(`1-r`);) both
  loci stay on one chromosome of the parent, state #mi(`A`);. With recombination (#mi(`r`);) the loci
  come from the two different chromosomes of the parent, state #mi(`B`);.
- From #mi(`B`);: the two chromosomes of the individual come one from each parent, so the loci
  descend from two different siblings of the previous generation, state #mi(`C`); with probability
  #mi(`1`);.
- From #mi(`C`);: each lineage independently picks one of the two parents (both siblings share the same
  two parents). With probability #mi(`1/2`); they pick different parents, state #mi(`C`);. With
  probability #mi(`1/2`); they pick the same parent, and each then picks one of its two chromosomes
  independently: the same chromosome (#mi(`1/4`); overall, state #mi(`A`);) or different chromosomes
  (#mi(`1/4`); overall, state #mi(`B`);).

#mitex(`M = \begin{pmatrix} 1-r & r & 0 \\ 0 & 0 & 1 \\ \tfrac14 & \tfrac14 & \tfrac12 \end{pmatrix}, \qquad (\text{rows: from } A, B, C)`)

The chain is irreducible for #mi(`r > 0`);. Fixation takes many generations, far more than its mixing
time, so the state reached at generation #mi(`0`); follows the stationary law #mi(`\pi = \pi M`);:

#mitex(`\pi_C = 4 r\,\pi_A, \qquad \pi_B = r\,\pi_A + \tfrac14 \pi_C = 2 r\,\pi_A, \qquad \pi = \frac{(1,\; 2r,\; 4r)}{1 + 6r}`)

The factor #mi(`1+6r`); is the normalisation of this law.

#strong[Funnel generations.] The founder identity at the two loci, given the state at generation
#mi(`0`);, is obtained by continuing the same backward argument through the funnel. A generation-#mi(`0`);
individual has one chromosome from #mi(`Q_1`); (founders #mi(`A,B,C,D`);) and one from #mi(`Q_2`);
(founders #mi(`E,F,G,H`);), each a gamete of its parent. Write #mi(`c_1`); for the chromosome #mi(`Q_1`);
inherited from the #mi(`AB`); hybrid and #mi(`c_2`); for the one from the #mi(`CD`); hybrid. A gamete of
the #mi(`AB`); hybrid carries the same founder at both loci with probability #mi(`1-r`);, and otherwise
carries #mi(`A`); at one locus and #mi(`B`); at the other. Hence for a single chromosome of
#mi(`c_1`);: #mi(`P(A,A) = P(B,B) = (1-r)/2`);, #mi(`P(A,B) = P(B,A) = r/2`);, and likewise for #mi(`c_2`); over
#mi(`C,D`);. Conditional on the generation-#mi(`0`); state, the probability #mi(`P_s(f,g)`); that the loci carry
#mi(`f`); and #mi(`g`); is (shown for #mi(`A`); in the first position, the other founders follow by symmetry):

- State #mi(`A`); (same chromosome of one individual, from #mi(`Q_1`); or #mi(`Q_2`);, each #mi(`1/2`);): the gamete of
  #mi(`Q_1`); has no recombination with probability #mi(`1-r`);, and is a copy of #mi(`c_1`); or #mi(`c_2`);. With
  recombination (#mi(`r`);) one locus comes from #mi(`c_1`); and the other from #mi(`c_2`);, giving
  independent founders from #mi(`\{A,B\}`); and #mi(`\{C,D\}`);. So
  #mi(`P_A(A,A) = (1-r)^2/8`);, #mi(`P_A(A,B) = r(1-r)/8`);, #mi(`P_A(A,C) = r/16`);, #mi(`P_A(A,E) = 0`);.
- State #mi(`B`); (different chromosomes of one individual, one from each of #mi(`Q_1`);, #mi(`Q_2`);):
  #mi(`P_B(A,A) = P_B(A,B) = P_B(A,C) = 0`);, #mi(`P_B(A,E) = 1/32`);.
- State #mi(`C`); (different siblings): the lineages choose parents and chromosomes as above. They carry
  #mi(`P_C(A,A) = (1-r)/32`);, #mi(`P_C(A,B) = r/32`);, #mi(`P_C(A,C) = 1/64`);, #mi(`P_C(A,E) = 1/64`);.

Combining, #mi(`P(f,g) = \pi_A P_A + \pi_B P_B + \pi_C P_C`);:

#mitex(`P(A,A) = \frac{\tfrac{(1-r)^2}{8} + 4r \cdot \tfrac{1-r}{32}}{1+6r} = \frac{1}{8}\,\frac{1-r}{1+6r}`)

#mitex(`P(A,B) = \frac{\tfrac{r(1-r)}{8} + 4r\cdot\tfrac{r}{32}}{1+6r} = \frac{r}{8(1+6r)}, \quad P(A,C) = \frac{\tfrac{r}{16} + 4r \cdot \tfrac{1}{64}}{1+6r} = \frac{r}{8(1+6r)}, \quad P(A,E) = \frac{2r\cdot\tfrac{1}{32} + 4r\cdot\tfrac{1}{64}}{1+6r} = \frac{r}{8(1+6r)}`)

The two loci of the fixed line carry two different founders with the same probability whether those
founders were paired in the funnel (#mi(`A,B`);), in the same four-way group (#mi(`A,C`);) or in different
groups (#mi(`A,E`);), so the result does not depend on the funnel order. The marginal law of the founder
at one locus is uniform, so dividing by #mi(`1/8`); gives the kernel:

#mitex(`H_{ff} = \frac{1 - r}{1 + 6r}, \qquad H_{fg} = \frac{r}{1 + 6r} \quad (f \neq g), \qquad k = \frac{1-2r}{1+6r}, \qquad H = k\, I + (1 - k)\tfrac{1}{J}\, \mathbf{1}\mathbf{1}^\top`)

Rows sum to 1, since #mi(`(1-r) + 7r = 1 + 6r`);.

#strong[Properties.] #mi(`H`); is symmetric with uniform stationary law; #mi(`H = I`); at #mi(`r = 0`); and
#mi(`H = \mathbf{1}\mathbf{1}^\top / J`); at #mi(`r = 0.5`);, so unlinked markers are independent. The
probability that the two loci carry different founders is #mi(`7r/(1+6r)`);, so the map of the line is
expanded by a factor of #mi(`7`); at small #mi(`r`);: seven founder breakpoints per Morgan of
single-meiosis map, not #mi(`(J-1)/J = 0.875`);. The kernel therefore takes the single-meiosis
#mi(`r`); and applies no further generation factor.

#strong[Check by simulation.] A forward simulation of the funnel and brother--sister mating with
Haldane gametes (400,000 independent lines per #mi(`r`);, 120 sibling generations, no use of the
kernel above) gives the probability that the two loci carry the same founder:

#table(
  columns: 4,
  [#mi(`r`);], [simulated], [#mi(`(1-r)/(1+6r)`);], [#mi(`\sqrt{1-2r}`); kernel],
  [0.001], [0.99316], [0.99304], [0.99912],
  [0.01], [0.93395], [0.93396], [0.99121],
  [0.05], [0.73111], [0.73077], [0.95510],
  [0.1], [0.56253], [0.56250], [0.90762],
  [0.25], [0.29903], [0.30000], [0.74372],
  [0.5], [0.12500], [0.12500], [0.12500],
)

For the 36-state diplotype space (unordered pairs) the two chromosomes move independently, and the
two ordered outcomes of a heterozygous target are summed, so every row sums to 1 without
renormalisation (`transition_matrix` raises when it does not):

#mitex(`T_{(f_1 f_2)\to(g_1 g_2)} = H_{f_1 g_1} H_{f_2 g_2} + \mathbb{1}[g_1 \neq g_2]\, H_{f_1 g_2} H_{f_2 g_1}`)

The kernel is specific to fixed lines founded by an eight-founder funnel and inbred by sibling mating.
It takes the single-meiosis recombination fraction directly, with no generation count. Outbred
populations (for example diversity outbred mice, heterogeneous stock and humans) are not fixed lines:
their chromosomes are heterozygous mosaics, so they need the diplotype chain with a kernel that depends
on the breeding scheme or, for humans, on the population's ancestry and haplotype structure.

With `cc_inbred=True` (default) the chain runs on the #mi(`J`); homozygous states only, with
emission #mi(`P(\text{obs} \mid (f,f))`);, uniform initial law #mi(`1/J`);, and kernel #mi(`H`);.
A residual heterozygous region would be forced to the best-fitting homozygous state, whose emission
for an `H` call is #mi(`\varepsilon/2`);; the current CC consensus input has no `H` calls, so
residual heterozygosity is not detectable from it. Populations that are not fully inbred use
`cc_inbred=False`, which runs the 36-state chain with the diplotype kernel above and a uniform
initial law, and returns the marginal haplotype probabilities (heterozygous states weighted 0.5).

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

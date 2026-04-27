# Haplotype Probability Inference

**Modules:** `genotype/hmm.py`, `genotype/genomecache.py`  
**miQTL parallel:** `DiploprobReader$getLocusMatrix(locus, "additive")` + HAPPY cache format

---

## 1. What the haplotype tensor represents

The core genotype object is:

$$H \in [0,1]^{m \times J \times L}, \quad \sum_{f=1}^{J} H_{i f \ell} = 1 \;\; \forall i, \ell$$

- $H_{if\ell}$ = posterior probability that strain $i$ at locus $\ell$ descended from founder $f$
- Rows sum to 1 (probability simplex per strain per locus)
- Not the same as a SNP genotype — never reduce to biallelic 0/1/2 for haplotype scans

The scan extracts one $m \times J$ slice per locus: $H_\ell = H[:, :, \ell]$.

---

## 2. Forward-backward HMM

**Model.** Founder ancestry at each locus is modeled as a hidden Markov chain along
the chromosome. Hidden state $z_{i\ell} \in \{1, \ldots, J\}$ = founder of origin.

**Emission:** $P(\text{obs}_{i\ell} \mid z_{i\ell} = f)$ — probability of the
observed GigaMUGA genotype call given descent from founder $f$ (based on founder
allele frequencies at each marker).

**Transition:** Recombination probability between adjacent loci $\ell$ and $\ell+1$
derived from the Haldane map function applied to the genetic distance in cM:

$$T_{ff',\ell \to \ell+1} = \begin{cases} e^{-2d} + (1 - e^{-2d})/J & f = f' \\ (1 - e^{-2d})/J & f \neq f' \end{cases}$$

where $d$ is the genetic distance in Morgans. For the CC, $n_{gen}=1$ (inbred lines
have resolved ancestry — no outbred recombination scaling needed).

**Forward variable:**
$$\alpha_{if\ell} = P(\text{obs}_{i,1:\ell} \mid z_{i\ell} = f)$$

computed left-to-right; scaled to prevent underflow.

**Backward variable:**
$$\beta_{if\ell} = P(\text{obs}_{i,\ell+1:L} \mid z_{i\ell} = f)$$

computed right-to-left.

**Posterior (Baum-Welch smoothing):**
$$H_{if\ell} = \frac{\alpha_{if\ell} \cdot \beta_{if\ell}}{\sum_{f'} \alpha_{if'\ell} \cdot \beta_{if'\ell}}$$

---

## 3. Haploid vs diploid representation

**miQTL / HAPPY format** estimates joint *diplotype* probabilities:

$$P(f \times f' \mid i, \ell) \in \mathbb{R}^{J(J+1)/2}$$

For $J=8$: 36 states (8 homozygous + 28 heterozygous diplotypes).

**mosaiQTL** estimates *marginal* haplotype probabilities by summing over the diplotype
partner:

$$H_{if\ell} = \sum_{f'} P(f \times f' \mid i, \ell)$$

**When this is exact:** For fully inbred CC strains, heterozygous diplotypes
$f \neq f'$ have near-zero probability (IBS $\approx$ IBD after many generations of
inbreeding). The marginal is therefore numerically equivalent to the diplotype diagonal.

**When this matters (S8 SNP scan):** SNP imputation requires the diplotype-level
probabilities to model heterozygosity at each founder SNP. Using marginal probs
conflates homozygous and heterozygous states, producing the documented $r = 0.55$
discrepancy between mosaiQTL and miQTL in the SNP scan (see `10_snp_scan.md`).

---

## 4. Cache formats: HAPPY vs HDF5

| Property | miQTL HAPPY (R) | mosaiQTL HDF5 (Python) |
|---|---|---|
| Physical layout | One `.RData` per locus per chromosome; e.g. `additive/chr8/data/rs12345.RData` | Single `.h5` file; one group per chromosome: `/chr8/probs`, `/chr8/locus_ids`, `/chr8/positions` |
| Loading model | `load()` called per locus; filesystem seek + R deserialise per call | `h5py.File` kept open; `Dataset` refs stored at load time; never eager-loaded |
| Memory at startup | Near zero (nothing loaded until scan) | Near zero (Dataset refs only; `np.asarray()` called at `get_chromosome()`) |
| Random access | $O(1)$ seek but high overhead (~1 ms/file on NFS) | $O(\log L)$ HDF5 B-tree lookup; contiguous chunk read |
| Per-chromosome load | $L \times$ sequential file reads | Single contiguous read: `np.asarray(grp['probs'])` |
| Probs shape | `[n_strains, J]` per locus (read one at a time) | `[n_loci, n_strains, J]` entire chromosome at once |
| LOCO support | No native support; workaround in `calc.kinship.from.genomecache.with.DOQTL` | `calc_kinship(cache, method='loco')` native |
| Memory per active chr | Not applicable (streaming) | $L \times m \times J \times 8$ bytes $\approx 40$ MB for CC chr8 |

**Critical invariant:** `GenomeCache.from_hdf5` stores `h5py.Dataset` references —
never `grp['probs'][:]`. Full-genome eager loading would load ~800 MB for CC at
startup and violates the lazy-load architecture.

---

## 5. Pseudomarkers and loci filtering

miQTL inserts pseudomarkers at regular intervals between typed GigaMUGA markers to
improve QTL localisation. mosaiQTL excludes pseudomarkers from scans — they carry
no additional genetic information (their probs are interpolated) and inflate L without
adding resolution. The expected loci count delta when comparing mosaiQTL vs miQTL is
`loci_max_delta = 1237` (confirmed in S1 validation).

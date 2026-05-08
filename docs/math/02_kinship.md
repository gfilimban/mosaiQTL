# Kinship Matrix Computation

**Modules:** `genotype/kinship.py`  
**miQTL parallel:** `calc.kinship.from.genomecache.with.DOQTL(genomecache, method='loco')`

---

## 1. Normalized haplotype dot-product kinship

The kinship between strains $i$ and $j$ is the average inner product of their
haplotype probability vectors across all loci:

$$K_{ij} = \frac{1}{L} \sum_{\ell=1}^{L} \sum_{f=1}^{J} H_{if\ell} \, H_{jf\ell}
         = \frac{1}{L} \sum_{\ell=1}^{L} h_{i\ell}^\top h_{j\ell}$$

where $h_{i\ell} \in \mathbb{R}^J$ is the haplotype probability vector (sums to 1).

**This is not van Raden (2008).** Van Raden's genomic relationship matrix centers
marker genotypes by allele frequency and normalizes by expected heterozygosity:
$K^{\text{vR}} = (M-P)(M-P)^\top \,/\, 2\sum_k p_k(1-p_k)$. The mosaiQTL/qtl2
formula uses raw haplotype probabilities with no allele-frequency centering.
The two formulas give different scales and diagonal values for outbred populations.

**Interpretation of diagonal entries:**

| Genotype type | $K_{ii}$ |
|---|---|
| Founder (homozygous, $h = e_f$) | $1.0$ |
| F1 hybrid ($h = 0.5 e_{f_1} + 0.5 e_{f_2}$) | $0.5$ |
| CC strain (partially inbred) | $\approx 0.2$–$0.35$ |

This matches `qtl2::calc_kinship(genoprob_to_alleleprob(probs), type='overall')` exactly.

**Implementation** (`kinship.py:50`):

```python
chr_num[chrom] = np.einsum('lni,lmi->nm', probs, probs)
# probs: (L, n, J)  →  (n, n)  via contraction over loci l and founders i
K = num_tot / nloci_tot
```

---

## 2. LOCO kinship

For the scan on chromosome $c$, the kinship must exclude all markers on $c$ to
prevent **proximal contamination** — inflated $\hat{h}^2$ caused by the QTL signal
on the test chromosome leaking into the random effect.

$$K^{(-c)}_{ij} = \frac{\displaystyle\sum_{\ell \notin c} h_{i\ell}^\top h_{j\ell}}{L - L_c}$$

**Implementation:** numerators and locus counts are accumulated per chromosome
during a single pass over the genome, then the target chromosome's contribution
is subtracted before normalizing:

```python
K_loco[c] = (num_tot - chr_num[c]) / (nloci_tot - chr_nloci[c])
```

This is $O(n^2 L)$ overall with $O(1)$ per-chromosome LOCO derivation — no
repeated passes over the genome.

---

## 3. Z-matrix kinship expansion for replicates

The genome cache stores probabilities at the strain level ($m$ unique strains).
When $n > m$ replicates are present, the strain-level kinship must be expanded
to sample space via the replicate-to-strain mapping matrix $Z \in \{0,1\}^{n \times m}$:

$$\tilde{K}_{ij} = K_{s(i),\, s(j)}, \quad \text{equivalently} \quad \tilde{K} = Z K Z^\top$$

where $s(i)$ maps sample $i$ to its strain index.

**When is this needed?** Only when phenotype and covariates are at sample level
($n = 384$ or $n = 418$). The scan itself always operates in strain space ($m \times m$)
after reducing replicate phenotypes to strain means via $Z$. See `04_scan_modes.md`.

---

## 4. Eigendecomposition and caching

The LMM scan requires solving a weighted linear system at each locus where the
weight matrix is $V = \delta K + I$ with $\delta = h^2 / (1-h^2)$. Direct
inversion of $V$ at every locus would be $O(n^3)$ per locus.

**Key identity:** if $K = U D U^\top$ (eigendecomposition), then:

$$V^{-1} = U \, \text{diag}\!\left(\frac{1}{\delta \lambda_k + 1}\right) U^\top$$

After rotating to the eigenbasis ($\tilde{y} = U^\top y$, $\tilde{X} = U^\top X$),
the covariance becomes diagonal:

$$\operatorname{Var}(\tilde{y}) = \sigma^2 \, \text{diag}(\delta \lambda_k + 1)$$

This replaces all $n \times n$ inversions with element-wise scalar divisions —
$O(n)$ per locus instead of $O(n^3)$.

**Implementation** (`kinship.py::decomp_kinship`):

```python
vals, vecs = np.linalg.eigh((K + K.T) / 2.0)   # symmetrize before eigh
order = np.argsort(vals)[::-1]                   # descending order
result[key] = {
    'values':  np.maximum(vals[order], 0.0),     # floor neg noise to 0
    'vectors': vecs[:, order],
}
```

`eigh` is used (not `eig`) because the symmetrized kinship is PSD; `eigh`
guarantees real eigenvalues and is $2\times$ faster than `eig` for symmetric
matrices. Negative eigenvalues from floating-point noise are floored to 0.

The decomposition is computed **once per chromosome** at startup and reused
across all permutations (when `fix_h2=True`) and all loci on that chromosome.

---

## 5. Per-chromosome contribution kinship $K^{(c)}$

LOCO kinship $K^{(-c)}$ excludes chromosome $c$. The complement —
the kinship contribution *from* chromosome $c$ alone — is:

$$K^{(c)}_{ij} = \frac{\displaystyle\sum_{\ell \in c} h_{i\ell}^\top h_{j\ell}}{L_c}
              = \frac{K_{\text{genome}} \cdot L - K^{(-c)} \cdot (L - L_c)}{L_c}$$

where $K_{\text{genome}} = \frac{1}{L}\sum_\ell h_{i\ell}^\top h_{j\ell}$
is the overall genome-wide kinship.

**Derivation:** The genome-wide numerator is the sum of all per-chromosome
numerators. A single pass accumulates $K^{(-c)}$ for all $c$; from those,
$K_{\text{genome}}$ is recovered as the weighted average:

$$K_{\text{genome}} = \frac{\displaystyle\sum_c K^{(-c)} \cdot (L - L_c)}{(n_{\text{chr}} - 1)\, L}$$

and then $K^{(c)}$ follows from the identity above.

**Use case:** $K^{(c)}$ shows the haplotype sharing structure attributable to
a single chromosome — useful for visualizing cis-genetic architecture and
identifying chromosomes with unusual sharing patterns (e.g., chrX sex dosage
effects). It is derived entirely from the LOCO matrices; no re-reading of the
genomecache is required.

**R implementation** (in `gastroc_miqtl_scan.R` S3):

```r
K_genome <- Reduce('+', lapply(chrs, function(ch) {
  kinship_loco[[ch]] * (L - cache_info$n_loci[cache_info$chr == ch])
})) / ((n_chr - 1) * L)

K_c <- (K_genome * L - kinship_loco[[chr]] * (L - L_c)) / L_c
```


#import "_setup.typ": *
#show: mathdoc

= Kinship Matrix Computation
<kinship-matrix-computation>
#strong[Modules:] `genotype/kinship.py` \ #strong[miQTL parallel:] `calc.kinship.from.genomecache.with.DOQTL(genomecache, method='loco')`

#divider()

== 1. Normalized haplotype dot-product kinship
<1-normalized-haplotype-dot-product-kinship>
The kinship between strains #mi(`i`); and #mi(`j`); is the average inner product of their
haplotype probability vectors across all loci:

#mitex(`K_{ij} = \frac{1}{L} \sum_{\ell=1}^{L} \sum_{f=1}^{J} H_{if\ell} \, H_{jf\ell}
         = \frac{1}{L} \sum_{\ell=1}^{L} h_{i\ell}^\top h_{j\ell}`)

where #mi(`h_{i\ell} \in \mathbb{R}^J`); is the haplotype probability vector (sums to 1).

`calc_kinship` then rescales every matrix it returns (overall, LOCO and per-chromosome)
so that its mean diagonal entry is 1:

#mitex(`K \leftarrow \frac{K}{\bar{d}}, \qquad \bar{d} = \frac{1}{m}\sum_{i=1}^{m} K_{ii}`)

(skipped if #mi(`\bar{d} \leq 0`);).

#strong[This is not van Raden (2008).] Van Raden\'s genomic relationship matrix centers
marker genotypes by allele frequency and normalizes by expected heterozygosity:
#mi(`K^{\text{vR}} = (M-P)(M-P)^\top \,/\, 2\sum_k p_k(1-p_k)`);. The mosaiQTL/qtl2
formula uses raw haplotype probabilities with no allele-frequency centering.
The two formulas give different scales and diagonal values for outbred populations.

#strong[Interpretation of diagonal entries] (raw matrix, before the mean-diagonal rescaling):

#figure(
  align(center)[#table(
    columns: 2,
    align: (auto,auto,),
    table.header([Genotype type], [#mi(`K_{ii}`);],),
    table.hline(),
    [Founder (homozygous, #mi(`h = e_f`);)], [#mi(`1.0`);],
    [F1 hybrid (#mi(`h = 0.5 e_{f_1} + 0.5 e_{f_2}`);)], [#mi(`0.5`);],
    [CC strain (inbred)], [#mi(`\approx 0.95`);--#mi(`0.99`); (measured on `ref/igvf_cache.h5`)],
  )]
  , kind: table
  )

In `ref/igvf_cache.h5` (8 founders, 7 F1s, 33 CC strains) the raw mean diagonal is
#mi(`\bar{d} \approx 0.91`);, so after rescaling founders sit at #mi(`\approx 1.10`);, F1s at
#mi(`\approx 0.55`); and CC strains at #mi(`\approx 1.05`);--#mi(`1.09`);.

The raw matrix is the quantity computed by `qtl2::calc_kinship(genoprob_to_alleleprob(probs), type='overall')`;
the matrices returned by `calc_kinship` equal it up to the scalar #mi(`1/\bar{d}`);.

#strong[Implementation] (`kinship.py:75`, `kinship.py:89-95`):

```python
chr_num[chrom] = np.einsum("lni,lmi->nm", probs, probs, optimize=True)
# probs: (L, n, J)  →  (n, n)  via contraction over loci l and founders i
K = _norm(num_tot / nloci_tot)   # _norm: divide by mean diagonal
```

#divider()

== 2. LOCO kinship
<2-loco-kinship>
For the scan on chromosome #mi(`c`);, the kinship must exclude all markers on #mi(`c`); to
prevent #strong[proximal contamination] --- inflated #mi(`\hat{h}^2`); caused by the QTL signal
on the test chromosome leaking into the random effect.

#mitex(`K^{(-c)}_{ij} = \frac{\displaystyle\sum_{\ell \notin c} h_{i\ell}^\top h_{j\ell}}{L - L_c}`)

#strong[Implementation:] numerators and locus counts are accumulated per chromosome
during a single pass over the genome, then the target chromosome\'s contribution
is subtracted before normalizing (`kinship.py:96-100`):

```python
K_loco[c] = _norm((num_tot - chr_num[c]) / (nloci_tot - chr_nloci[c]))
```

Each LOCO matrix is therefore also rescaled to mean diagonal 1.

This is #mi(`O(m^2 L J)`); overall with an #mi(`O(m^2)`); per-chromosome LOCO derivation --- no
repeated passes over the genome.

#divider()

== 3. Z-matrix kinship expansion for replicates
<3-z-matrix-kinship-expansion-for-replicates>
The genome cache stores probabilities at the strain level (#mi(`m`); unique strains).
When #mi(`n > m`); replicates are present, the strain-level kinship must be expanded
to sample space via the replicate-to-strain mapping matrix #mi(`Z \in \{0,1\}^{n \times m}`);:

#mitex(`\tilde{K}_{ij} = K_{s(i),\, s(j)}, \quad \text{equivalently} \quad \tilde{K} = Z K Z^\top`)

where #mi(`s(i)`); maps sample #mi(`i`); to its strain index.

#strong[When is this needed?] Whenever the phenotype sample ids differ from the cache's
strain ids (e.g. several pseudobulk samples per strain); `qtl/scan.py::_resolve_z` then builds
#mi(`Z`); with `utils.build_z_matrix`, otherwise #mi(`Z`); is the identity and is skipped. The
haplotype LMM scans (`scan_h2lmm`, `scan_multi_gene`, and the production `scan_multi_gene_lmm`)
do not average to strain means: once per chromosome they rebuild #mi(`K^{(-c)}`); from its
eigendecomposition, form #mi(`\tilde{K} = Z K^{(-c)} Z^\top`);, re-eigendecompose it
(`scan.py:267-268`, `485-486`, `728-729`), and project the haplotype probabilities through
#mi(`Z`); (#mi(`U^\top Z H_\ell`);, `qtl/lmm.py:275-277`), so #mi(`h^2`); and the scan run in sample
space (#mi(`n \times n`);). The MI scan (`qtl/mi.py`) and the SNP scan (`qtl/snp_scan.py::_reduce_to_strain_means`)
instead average phenotype and covariates to strain means via #mi(`Z`); and run in strain space.
`utils.expand_kinship` implements #mi(`Z K Z^\top`); but has no callers. See `04_scan_modes.typ`.

#divider()

== 4. Eigendecomposition and caching
<4-eigendecomposition-and-caching>
The LMM scan requires solving a weighted linear system at each locus where the
weight matrix is #mi(`V = h^2 K + (1-h^2) I`); (`qtl/lmm.py:71`). Direct
inversion of #mi(`V`); at every locus would be #mi(`O(n^3)`); per locus.

#strong[Key identity:] if #mi(`K = U D U^\top`); (eigendecomposition), then:

#mitex(`V^{-1} = U \, \text{diag}\!\left(\frac{1}{w_k}\right) U^\top, \qquad w_k = h^2 \lambda_k + (1 - h^2)`)

After rotating to the eigenbasis (#mi(`\tilde{y} = U^\top y`);, #mi(`\tilde{X} = U^\top X`);),
the covariance becomes diagonal:

#mitex(`\operatorname{Var}(\tilde{y}) = \sigma^2 \, \text{diag}(w_k)`)

This replaces all #mi(`n \times n`); inversions with element-wise scalar divisions ---
#mi(`O(n)`); per locus instead of #mi(`O(n^3)`);.

#strong[Implementation] (`kinship.py::decomp_kinship`):

```python
vals, vecs = np.linalg.eigh((K + K.T) / 2.0)   # symmetrize before eigh
order = np.argsort(vals)[::-1]                   # descending order
result[key] = {
    'values':  np.maximum(vals[order], 0.0),     # floor neg noise to 0
    'vectors': vecs[:, order],
}
```

`eigh` is used (not `eig`) because the symmetrized kinship is PSD; `eigh`
guarantees real eigenvalues and is #mi(`2\times`); faster than `eig` for symmetric
matrices. Negative eigenvalues from floating-point noise are floored to 0.

In production the decomposition is computed #strong[once per run]: the Nextflow
`KINSHIP_LOCO` process runs `workflow/scripts/build_kinship_loco.py`
(`calc_kinship(gc, method='loco')` → `decomp_kinship`) and writes one HDF5 file
(`chr{c}/values`, `chr{c}/vectors`, root `sample_ids`), skipped if `ref/igvf_kinship_loco.h5`
already exists. Each per-chromosome scan loads only its chromosome with
`load_kinship_decomp(path, chroms=[chrom])` and reuses it across all loci and genes on that
chromosome (and across permutations when `fix_h2=True`); when #mi(`Z`); is present the expanded
#mi(`\tilde{K}`); is re-decomposed once per chromosome (Section 3).

#divider()

== 5. Per-chromosome contribution kinship #mi(`K^{(c)}`);
<5-per-chromosome-contribution-kinship-zzi0043zz>
LOCO kinship #mi(`K^{(-c)}`); excludes chromosome #mi(`c`);. The complement ---
the kinship contribution #emph[from] chromosome #mi(`c`); alone --- is:

#mitex(`K^{(c)}_{ij} = \frac{\displaystyle\sum_{\ell \in c} h_{i\ell}^\top h_{j\ell}}{L_c}
              = \frac{K_{\text{genome}} \cdot L - K^{(-c)} \cdot (L - L_c)}{L_c}`)

where #mi(`K_{\text{genome}} = \frac{1}{L}\sum_\ell h_{i\ell}^\top h_{j\ell}`);
is the overall genome-wide kinship.

#strong[Python implementation:] `calc_kinship(cache, method='chr')` computes
#mi(`K^{(c)}`); directly from the per-chromosome numerators of the same single pass,
`_norm(chr_num[c] / chr_nloci[c])` (`kinship.py:101-102`), rescaled to mean diagonal 1 like the
other modes.

#strong[Derivation from LOCO (R reference):] The genome-wide numerator is the sum of all per-chromosome
numerators. A single pass accumulates #mi(`K^{(-c)}`); for all #mi(`c`);; from those,
#mi(`K_{\text{genome}}`); is recovered as the weighted average:

#mitex(`K_{\text{genome}} = \frac{\displaystyle\sum_c K^{(-c)} \cdot (L - L_c)}{(n_{\text{chr}} - 1)\, L}`)

and then #mi(`K^{(c)}`); follows from the identity above. These identities hold only for
un-rescaled matrices; they do not apply to the LOCO matrices returned by `calc_kinship`,
which are each divided by their own mean diagonal.

#strong[Use case:] #mi(`K^{(c)}`); shows the haplotype sharing structure attributable to
a single chromosome --- useful for visualizing cis-genetic architecture and
identifying chromosomes with unusual sharing patterns (e.g., chrX sex dosage
effects). In the R reference it is derived entirely from the LOCO matrices, with no
re-reading of the genomecache.

#strong[R implementation] (in `validation/r_reference/gastroc_miqtl_scan.R` S3):

```r
K_genome <- Reduce('+', lapply(chrs, function(ch) {
  kinship_loco[[ch]] * (L - cache_info$n_loci[cache_info$chr == ch])
})) / ((n_chr - 1) * L)

K_c <- (K_genome * L - kinship_loco[[chr]] * (L - L_c)) / L_c
```

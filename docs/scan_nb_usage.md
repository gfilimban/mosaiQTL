# `scan_nb` — NB GLM genome-wide scan

Primary mosaiQTL model for pseudobulk snRNA-seq eQTL mapping.

## API

```python
from mosaiqtl import scan_nb, scan_nb_multi_gene
from mosaiqtl.genome_cache import GenomeCache
```

### Single gene — score test (default, fast)

```python
result = scan_nb(
    counts       = pseudobulk_counts,   # np.ndarray (n,) raw integer counts
    genome_cache = cache,
    covariates   = covar_matrix,        # (n, p): intercept + batch + PCs
    sample_ids   = sample_list,         # list of str, length n
    lib_sizes    = total_counts,        # (n,) for log-offset; None = column sums
    dispersion   = None,                # float or None; None → estimated here
    use_lrt      = False,               # score test (O(L) per gene)
    just_these_loci = cis_loci_set,     # set of locus ids for cis-eQTL
    phenotype_id = 'Piezo1',
)

# result is a ScanResult with the same interface as scan_h2lmm
peak = result.peak_locus()        # {'locus_id', 'chromosome', 'position', 'lod', ...}
df   = result.to_dataframe()      # pd.DataFrame with all loci
```

### Single gene — LRT (exact betas, slow)

```python
result_lrt = scan_nb(
    counts, cache, covar_matrix, sample_ids,
    use_lrt   = True,               # fits alt model at every locus
    dispersion = phi,               # fix dispersion for speed
    just_these_loci = cis_set,
)
# result_lrt.betas contains log-scale founder effects
```

### Multiple genes

```python
# count_matrix: (n_samples, n_genes)
# dispersion_dict: pre-compute once with estimate_gene_dispersion()
results = scan_nb_multi_gene(
    count_matrix    = counts_mat,
    gene_ids        = gene_list,
    genome_cache    = cache,
    covariates      = covar_matrix,
    sample_ids      = sample_list,
    lib_sizes       = total_counts,
    dispersion_dict = {g: phi_g for g, phi_g in disp_map.items()},
    cis_loci_dict   = {g: cis_set_g for g in gene_list},
    n_cores         = 1,
)
# returns dict {gene_id: ScanResult}
```

## Mathematical formulation

**Null model** ($\ell = 0$, covariates only):

$$Y_i \sim \text{NB}(\mu_i^0, \phi) \quad \log \mu_i^0 = X_i \beta_0 + \log s_i$$

where $s_i$ is the library size offset.

**Alternative model** at locus $\ell$:

$$\log \mu_i^\ell = H_i^\ell \gamma + X_i \beta + \log s_i$$

where $H_i^\ell \in \mathbb{R}^{J-1}$ are founder haplotype dosages (last dropped).

**Score test statistic** (default):

$$Q_\ell = U_\ell^\top I_\ell^{-1} U_\ell \sim \chi^2(J-1)$$

$$U_\ell = H_\ell^\top r, \quad r_i = \frac{y_i - \hat\mu_i^0}{1 + \phi \hat\mu_i^0}$$

$$I_\ell = H_\ell^\top \text{diag}(w) H_\ell, \quad w_i = \frac{\hat\mu_i^0}{1 + \phi \hat\mu_i^0}$$

**LOD conversion**:

$$\text{LOD}_\ell = \frac{Q_\ell}{2 \ln 10}$$

For LRT: $\text{LOD}_\ell = (\ell_{\text{alt}} - \ell_{\text{null}}) / \ln 10$, exact.

**NB log-likelihood**:

$$\ell(y \mid \mu, \phi) = \sum_i \left[\ln\Gamma\!\left(y_i + \tfrac{1}{\phi}\right) - \ln\Gamma\!\left(\tfrac{1}{\phi}\right) + \tfrac{1}{\phi}\ln\!\frac{1/\phi}{1/\phi + \mu_i} + y_i \ln\!\frac{\mu_i}{1/\phi + \mu_i} - \ln\Gamma(y_i+1)\right]$$

## Design notes

- **No RINT**: RINT is for Gaussian LMM normality; NB handles discrete counts natively.
- **No kinship**: fixed-effect GLM; mixed NB (NEBULA-style) is a future extension.
- **Z-matrix**: when `sample_ids ≠ genome_cache.sample_ids`, $H_n = Z H_m$ is applied
  automatically — supports native $n=418$ operation without collapsing to strain means.
- **Dispersion fixed across loci**: estimated once from null model (edgeR EB pipeline),
  matching edgeR's `glmQLFit` convention.
- **Vectorised per chromosome**: score statistics for all loci computed with `einsum` —
  analogous to FWL vectorisation in `scan_chromosome`.

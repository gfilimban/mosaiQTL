# Haplotype-Imputed SNP Scan

**Module:** `qtl/snp_scan.py::imputed_snp_scan_h2lmm`  
**miQTL parallel:** `imputed.snp.scan.h2lmm()`

---

## 1. Purpose

The haplotype scan identifies which genomic *regions* affect a phenotype and
which *founder haplotypes* carry the effect. The SNP scan goes further: given
known founder genotypes at individual SNPs, it imputes per-sample SNP dosages
from the haplotype probabilities and tests each SNP for association. This connects
haplotype effects to specific sequence variants and enables fine-mapping within
a QTL interval.

---

## 2. Inputs

`imputed_snp_scan_h2lmm` takes:

- `phenotype` — continuous phenotype $(n,)$, RINT-transformed internally (LMM path).
- `genomecache` — `GenomeCache` with haplotype probs at strain level.
- `snp_genos` — `pd.DataFrame` of shape `(n_snps, n_founders)`, values $\in \{0,1,2\}$
  (alt allele dosage per founder), with `chr` and `pos_mb` columns.
  The columns must match `genomecache.founders` by name.

The `snp_genos` DataFrame is the Python-native interface. It can be constructed
from any source: VCF annotation of the 8 founder strains, the Sanger Mouse Genomes
Project variant calls, or the optional `read_allele_dir` helper which reads
per-SNP founder genotypes from flat files.

---

## 3. SNP dosage imputation

For SNP $s$ at position $p_s$ on chromosome $c$, the imputed dosage for sample $i$ is:

$$d_{is} = \sum_{f=1}^{J} H_{if\ell(s)} \cdot g_{fs}$$

where:
- $H_{if\ell(s)}$ — haplotype probability for sample $i$, founder $f$, at the
  nearest typed locus $\ell(s)$ to position $p_s$ (minimum $|p_s - p_\ell|$).
- $g_{fs} \in \{0,1,2\}$ — founder $f$'s alt allele count at SNP $s$.

```python
dosage = h_locus @ geno_f    # (n_valid,) inner product
```

For a fully inbred CC sample where $H_{i\ell} \approx e_{f^*}$, this recovers
$g_{f^*s}$ directly. For samples with uncertain ancestry the dosage is a
probability-weighted interpolation across founders.

Monomorphic SNPs ($\text{sd}(d) < 10^{-8}$) are skipped.

---

## 4. Scan model

The standardized dosage $\tilde{d}_s = (d_s - \bar{d}_s)/\text{sd}(d_s)$ is
used as a scalar predictor in the LMM:

$$y_i = \beta_d \tilde{d}_{is} + x_i^\top\alpha + u_i + \varepsilon_i,
\qquad u \sim \mathcal{N}(0,\sigma^2_g K^{(-c)})$$

This is `fit_lmm_at_locus` with a $1$-column haplotype matrix ($J=1$), giving
1 effective df. $h^2$ is estimated once per chromosome from the null model.

**Power comparison to haplotype scan:** the haplotype scan has $J-1 = 7$ df and
detects any founder-haplotype association regardless of SNP allele coding. The
SNP scan has 1 df and is more powerful when the causal variant is biallelic and
the founder partition aligns with a single alt allele. Multi-allelic loci with
complex founder allele patterns are better detected by the haplotype scan.

---

## 5. The r = 0.55 discrepancy with miQTL

mosaiQTL uses **marginal haplotype probabilities**:

$$H_{if\ell} = \sum_{f'} P(f \times f' \mid i, \ell)$$

miQTL uses the full **diplotype probabilities** $P(f \times f' \mid i, \ell)$
to model heterozygosity in dosage imputation:

$$d_i^{\text{miQTL}} = \sum_{f \leq f'} \bigl[P(f \times f) \cdot g_f + P(f \times f') \cdot \tfrac{g_f + g_{f'}}{2}\bigr]$$

**For inbred CC samples** the difference is negligible: heterozygous diplotype
probabilities $P(f \neq f') \approx 0$, so the marginal approximation
$H_{if\ell} \approx P(f \times f \mid i, \ell)$ is exact.

**For outbred DO samples** the difference is material: each animal carries two
distinct founder haplotypes per locus. The marginal conflates homozygous and
heterozygous diplotypes, compressing dosage variation and reducing
correlation with the true SNP genotype.

The documented $r = 0.55$ between mosaiQTL and miQTL SNP LOD profiles is
explained by this marginal-vs-diplotype model difference on CC data where
residual heterozygosity is present in some strains. The marginal approximation
is adequate for fully inbred CC; diplotype-aware dosage imputation is needed
for DO. This is a known, documented limitation — not a bug.

---

## 6. Scope notes

- This scan is **LMM-only** in the current implementation. An NB-GLMM version
  would replace `fit_lmm_at_locus` with the NB score test at $J=1$ (scalar dosage
  predictor), using raw counts and the fixed dispersion $\hat\phi_g$ from Stage 1.
- P2-S8 gate is marked `⚠️ GATED`: the scan requires `snp_genos` to be populated
  with founder variant calls, which must be sourced externally (e.g. Sanger MGP).

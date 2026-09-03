# Per-Gene Beta-Approximated Permutation p-values and Storey q-values

**Module:** `qtl/permutation.py` (per-gene calibration), `pipeline.py` (q-value / eGene calling)
**miQTL parallel:** none — miQTL uses BH-FDR on genome-wide max-LOD; this design follows the
FastQTL/tensorQTL/GTEx precedent instead (see §0).
**Replaces:** the BH-FDR placeholder in `pipeline.py::_SIG_COLS` (`fdr_method = 'BH'`), flagged
`SWE-09` since scaffolding.

---

## 0. Why this document exists

Two unrelated multiple-testing problems are stacked in every eQTL scan:

| Problem | Tested units | Current mechanism |
|---|---|---|
| Multiple **loci** within one gene's scan | $L$ correlated loci (LD/haplotype blocks) | Permutation + GEV threshold (`docs/math/06_permutation.md`) |
| Multiple **genes** scanned genome-wide | $G$ genes ($10^4$–$10^5$) | BH-FDR on `peak_lod` p-values (placeholder) |

The per-locus problem is solved correctly (§4–5 of `06_permutation.md`). The per-gene problem is
not: `pipeline.py` defines `pval_perm`/`fdr` fields in `EGENE_SUMMARY_COLS` but no function
populates them — `finalize_egene_summary()` never touches them, and `call_egenes()` applies a
single scalar LOD threshold (`get_gev_thresholds`) genome-wide instead. A prior attempt to wire
Storey q-values through this scalar-threshold p-value produced $\hat\pi_0 = 2.58$ — mathematically
impossible for a proportion — because the p-value passed to Storey was computed against **one
representative gene's** permutation null (`src/mosaiqtl/CLAUDE.md`: *"S5 null derived from one
representative gene — invalid per-gene FDR"*). Genes whose $\hat{h}^2$ (LMM) or $\hat\phi$
(NB-GLMM) differ from that one gene's value are scored against the wrong null, so their p-values
are not Uniform(0,1) under $H_0$ and Storey's flat-tail assumption (§3) is violated systemically.

The fix is **per-gene calibration**: every gene's p-value must come from a null built from that
gene's own fixed nuisance parameter. Full per-gene permutation (1000+ perms × $G$ genes) would
violate `SPEED-01` (<4h genome-wide, no GPU path yet — `SPEED-07`/`08` still 🔴). The standard
field solution — used by FastQTL (Ongen et al. 2016, *Bioinformatics* 10.1093/bioinformatics/btv722
— **citation not yet cross-checked against `docs/REFERENCES.md`; verify before commit**) and
adopted by tensorQTL and the GTEx pipeline — is to run a **small** number of per-gene permutations
and fit a **Beta distribution** to the resulting null: the Beta family absorbs most of the sampling
noise a raw empirical tail would carry at low $B$, so $B \approx 100$–$1000$ per gene suffices.

---

## 1. Per-gene max statistic and its null

For gene $g$ with cis-window loci $\ell = 1,\dots,L_g$, define the per-gene test statistic as the
best (minimum) nominal p-value across the window:

$$p_{\min,g} = \min_{\ell=1,\dots,L_g} p_{\ell,g}$$

where $p_{\ell,g}$ is the locus-level nominal p-value implied by the scan's LOD (LMM: $F$/$\chi^2$
approximation from the RSS-ratio LOD; NB-GLMM: $\chi^2(J-1)$ from the score statistic $Q_\ell$ —
both already computed per locus during a normal scan). $p_{\min,g}$ is the eQTL-calling analogue
of the genome-wide max-LOD in §4 of `06_permutation.md`, restricted to one gene's own window.

For each gene, generate $B$ permutations **using that gene's own fixed nuisance parameter**
($\hat{h}^2_g$ for LMM, $\hat\phi_g$ for NB-GLMM — both already estimated once from real data, no
extra cost) and record $\{p_{\min,g}^{(b)}\}_{b=1}^B$. This is exactly the `run_threshold_scans`
loop in §4 of `06_permutation.md`, called once per gene instead of once globally, with $B$ two to
three orders of magnitude smaller than the genome-wide $B=1000$.

---

## 2. Engine-agnosticism of §1 (LMM and NB-GLMM)

§1's null-generation loop consumes only a scalar max-statistic per permutation — it has no
dependence on which engine produced it, and both engines already support cheap per-gene
permutation via the same architectural pattern (`06_permutation.md` §1, §3; `qtl/nb.py`):

| | LMM | NB-GLMM |
|---|---|---|
| Fixed nuisance param (never refit per permutation) | $\hat h^2_g$ (REML on real $y$) | $\hat\phi_g$ (Stage 1, real cell-level data) |
| Per-permutation cost | `scan_chromosome` — null fit once, vectorized over all $L_g$ loci | `_score_stats_chromosome` / `_score_stats_loco` — same shape: null fit once, score stat vectorized over all loci |
| Statistic → LOD | RSS-ratio LOD | $Q_\ell\sim\chi^2(J{-}1) \Rightarrow \text{LOD}=Q_\ell/(2\ln10)$ |

Both engines already satisfy the project's core scan-design principle (`CLAUDE.md`: *"Score test
fits null once per gene, vectorises over all loci — no IRLS re-fit per marker"*) — the property
that makes per-gene (not just genome-wide) permutation tractable at all. NB-GLMM's $\hat\phi_g$ is
never re-estimated during permutation (`06_permutation.md` §3), so its marginal per-permutation
cost is score-stat evaluation only — cheaper than LMM's REML-based $\hat h^2_g$ path, not harder.
No engine-specific branch is needed anywhere downstream (§3–5) — only in this null-generation call,
which already branches by engine exactly as the genome-wide threshold does today.

---

## 3. Beta approximation to the per-gene null

Under $H_0$, order statistics of $L_g$ correlated p-values do not have a simple closed form, but
their minimum is well approximated by a $\mathrm{Beta}(a, b)$ distribution — the shape a minimum
of correlated Uniform-ish variables takes when $L_g$ is roughly stable across the window (Ongen et
al. 2016). Two parameters absorb the *effective* number of independent tests without computing it
explicitly.

**Method-of-moments starting point**, from the empirical mean $\hat\mu$ and variance $\hat\sigma^2$
of $\{p_{\min,g}^{(b)}\}$:

$$\hat{a}_0 = \hat\mu\left(\frac{\hat\mu(1-\hat\mu)}{\hat\sigma^2} - 1\right), \qquad
  \hat{b}_0 = (1-\hat\mu)\left(\frac{\hat\mu(1-\hat\mu)}{\hat\sigma^2} - 1\right)$$

**MLE refinement** (starting from $\hat{a}_0,\hat{b}_0$): maximize the Beta log-likelihood over
$\{p_{\min,g}^{(b)}\}$ via `scipy.optimize` — matches FastQTL's implementation, more accurate in
the tail than MoM alone, which is what makes a small $B$ sufficient.

**Beta-adjusted per-gene p-value**, applied to the real (unpermuted) minimum nominal p-value
$p_{\min,g}^{\text{obs}}$:

$$p_{\text{adj},g} = F_{\mathrm{Beta}(\hat{a},\hat{b})}\!\left(p_{\min,g}^{\text{obs}}\right)$$

$p_{\text{adj},g}$ is calibrated Uniform(0,1) under $H_0$ **for that gene specifically**, because
it was built from that gene's own $\hat{h}^2_g$/$\hat\phi_g$ null (§2 confirms this holds
identically for both engines). This value populates `pval_perm` in `EGENE_SUMMARY_COLS`.

---

## 4. Storey q-value across genes

With $\{p_{\text{adj},g}\}_{g=1}^G$ now genuinely Uniform(0,1) under each gene's own null, Storey's
construction (Storey & Tibshirani 2003, PNAS 10.1073/pnas.1530509100; Storey 2002, JRSS-B
10.1111/1467-9868.00346 — both already in `docs/REFERENCES.md`) applies validly, for genes scanned
under either engine — the q-value step operates purely on the p-value vector, with no engine
dependence at all.

**Step 1 — estimate $\hat\pi_0$** (proportion of true nulls) from the flat right tail of the
p-value histogram, over a range $\lambda \in \{0, 0.01, \dots, 0.95\}$:

$$\hat\pi_0(\lambda) = \frac{\#\{p_{\text{adj},g} > \lambda\}}{G(1-\lambda)}$$

Fit a natural cubic spline to $\hat\pi_0(\lambda)$ across $\lambda$ and evaluate at $\lambda \to 1$
for the final $\hat\pi_0$. By construction $\hat\pi_0 \in [0,1]$ under a valid Uniform(0,1) null —
the $\hat\pi_0 = 2.58$ failure mode from the mis-calibrated single-gene null (§0) is structurally
impossible once §1–3's per-gene calibration is in place.

**Step 2 — q-values**, from p-values sorted ascending $p_{(1)} \le \cdots \le p_{(G)}$:

$$\hat{q}(p_{(G)}) = \hat\pi_0 \, p_{(G)}, \qquad
  \hat{q}(p_{(i)}) = \min\!\left(\hat\pi_0 \frac{G \, p_{(i)}}{i},\; \hat{q}(p_{(i+1)})\right)
  \quad \text{for } i = G-1,\dots,1$$

the monotone (step-down) minimum enforces $\hat{q}(p_{(i)})$ non-decreasing in $i$. An eGene is
called at $\hat{q}(p_{\text{adj},g}) \le 0.05$ (`SWE-09`'s BIO-A5 gate target, unchanged).

**Versus BH-FDR:** BH is the $\hat\pi_0 = 1$ special case of the same formula — always
conservative when real signal exists among the $G$ genes. Storey recovers power BH wastes, but
*only* once its Uniform(0,1)-under-$H_0$ input assumption holds, which is exactly what §1–3
guarantees, identically for LMM- and NB-GLMM-scanned genes.

---

## 5. Two-stage design: what happens to the genome-wide GEV threshold

Per-gene Beta-calibration (§1–3) is what makes Storey valid, but running it for every gene in a
$10^4$–$10^5$-gene scan is still expensive relative to a single global permutation run. The
existing genome-wide/per-chromosome GEV threshold (`06_permutation.md` §4–5, `run_threshold_scans`
→ `get_gev_thresholds`) is **not discarded** — it is repurposed as an adaptive two-stage filter,
matching FastQTL's own adaptive-permutation strategy, and this repurposing is likewise identical
for both engines (the genome-wide threshold is already computed per engine today):

| Stage | Mechanism | Role |
|---|---|---|
| 1 — screen | Genome-wide/per-chromosome GEV threshold (existing, cheap: one global permutation run) | Coarse pre-filter — only genes with `peak_lod` clearing the chromosome-level GEV threshold are promoted to Stage 2. Also remains the Manhattan-plot significance line and keeps `gev_mu`/`gev_sigma`/`gev_xi`/`lod_threshold` in `EQTL_STATS_COLS` meaningful. |
| 2 — calibrate | Per-gene Beta-approximated permutation + Storey q-value (§1–4, new) | Determines actual eGene calling: `significance = 'T'` iff $\hat{q} \le 0.05$, not the raw Stage-1 LOD cutoff. |

`call_egenes(threshold=...)`'s raw scalar-threshold comparison stops being the final eGene-calling
criterion — its output becomes the Stage-1 promotion set. This preserves `SPEED-01`'s <4h budget
(Stage 2 only runs on the minority of genes clearing Stage 1) while giving every called eGene a
per-gene-calibrated q-value instead of a shared, generically miscalibrated one.

---

## 6. Summary of what changes downstream

- `qtl/permutation.py`: new per-gene entry point (name TBD at implementation time) wrapping the
  existing `run_threshold_scans` null-generation branches at $B_{\text{gene}} \ll B_{\text{genome}}$,
  dispatching on engine exactly as it does today, returning $(p_{\text{adj},g}, \hat{a}, \hat{b})$.
- `pipeline.py::_SIG_COLS`: `pval_perm` populated from §3's output; `fdr` from §4's q-value;
  `fdr_method` becomes `'storey'` (no longer hardcoded `'BH'`); `significance` gated on $\hat q \le 0.05$
  instead of the raw `peak_lod` threshold.
- `call_egenes()`: raw threshold comparison becomes the Stage-1 promotion gate (§5), not the final
  call.
- Validation: `BIO-A5` (observed FDR $\le 0.05$ on `sim_cc48_n200qtl`) must use this Storey path,
  not BH — per `GOALS.md` `SWE-09`. Both LMM- and NB-GLMM-scanned genes flow through the identical
  §3–4 machinery once their per-gene nulls exist (§1–2).

Implementation follows `DESIGN-03`/`DESIGN-05` order — this document precedes any code change to
`pipeline.py` or `qtl/permutation.py`.

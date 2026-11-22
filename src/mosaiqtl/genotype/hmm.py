"""Forward-backward HMM for CC/DO haplotype reconstruction."""

# hmm.py
# pure python CC/DO haplotype reconstruction HMM
#
# 8-founder diplotype HMM for CC inbred strain haplotype reconstruction:
#
#   hidden states : 36 unordered diplotypes {f1f2 : f1 <= f2, f1,f2 in 1..J}
#   observations  : A/H/B genotype calls at GigaMUGA markers
#   transitions   : haplotype substitution rates from genetic map (Haldane)
#   emissions     : P(obs | diplotype, founder genotypes)
#
# for CC mice (fully inbred after 20 generations), we use the additive
# (allele probability) representation directly — each CC strain has one
# dominant haplotype block per chromosomal region. the forward-backward
# algorithm gives posterior probabilities over diplotype states.
#
# references:
#   Broman et al. 2019 (qtl2) — https://doi.org/10.1534/genetics.118.301595
#   Churchill et al. 2004 (CC design) — https://doi.org/10.1038/ng1104
#   DOQTL: Gatti et al. 2014 — https://doi.org/10.1534/g3.114.013748

import numpy as np
import logging
from itertools import combinations_with_replacement

log = logging.getLogger(__name__)

# ── diplotype state index ─────────────────────────────────────────────────────


def build_diplotype_states(n_founders=8):
    """build ordered list of unordered diplotype state pairs.

    for J founders there are J*(J+1)/2 unordered diplotypes:
      homozygous: (0,0), (1,1), ..., (J-1,J-1)
      heterozygous: (0,1), (0,2), ..., (J-2,J-1)

    returns
    -------
    states : list of (f1, f2) tuples, f1 <= f2  (length J*(J+1)/2)
    state_to_idx : dict (f1,f2) → index
    """
    states = list(combinations_with_replacement(range(n_founders), 2))
    state_to_idx = {s: i for i, s in enumerate(states)}
    return states, state_to_idx


def diplotype_to_allele_probs(diplotype_probs, n_founders=8):
    """collapse diplotype probs (n_states,) → additive allele probs (n_founders,).

    for each founder f, allele prob = sum over all diplotype states
    containing f, weighted by 0.5 for heterozygous states.

    args:
        diplotype_probs : (n_states,) or (n_loci, n_strains, n_states)
    returns:
        allele_probs : same leading dims, last dim = n_founders
    """
    states, _ = build_diplotype_states(n_founders)
    n_states = len(states)

    # build (n_states, n_founders) contribution matrix
    contrib = np.zeros((n_states, n_founders))
    for si, (f1, f2) in enumerate(states):
        if f1 == f2:
            contrib[si, f1] = 1.0
        else:
            contrib[si, f1] = 0.5
            contrib[si, f2] = 0.5

    # matrix multiply last dim
    return diplotype_probs @ contrib


# ── transition probabilities ──────────────────────────────────────────────────


def haldane_recomb(d_morgan):
    """Haldane map function: recombination fraction from genetic distance (Morgans)."""
    return 0.5 * (1 - np.exp(-2 * np.abs(d_morgan)))


def haplotype_transition_matrix(r, n_founders=8):
    r"""founder-haplotype transition matrix of a fixed line at recombination fraction r.

    a line founded by an eight-founder funnel and inbred by sibling mating
    carries founder $g$ at one locus given founder $f$ at the other with
    $H_{ff} = (1 - r)/(1 + 6r)$ and $H_{fg} = r/(1 + 6r)$ for $f \neq g$,
    equivalently
    $H = k I + (1 - k)\mathbf{1}\mathbf{1}^\top / J$ with $k = (1 - 2r)/(1 + 6r)$.
    rows sum to 1, $H$ is symmetric (uniform stationary law), $H = I$ at
    $r = 0$ and $H = \mathbf{1}\mathbf{1}^\top / J$ at $r = 0.5$. the two loci
    differ with probability $7r/(1 + 6r)$, so the line's map is expanded
    sevenfold: $r$ is the single-meiosis recombination fraction.

    args:
        r         : single-meiosis recombination fraction between the loci (0 to 0.5)
        n_founders: J, must be 8 (the funnel size the kernel is derived for)
    returns:
        H : (n_founders, n_founders) matrix, rows sum to 1
    """
    if n_founders != 8:
        raise ValueError(
            f"the sibling-mating funnel kernel is derived for 8 founders, got {n_founders}"
        )
    J = n_founders
    k = (1.0 - 2.0 * r) / (1.0 + (J - 2) * r)
    H = np.full((J, J), (1.0 - k) / J)
    H[np.diag_indices(J)] += k
    return H


X_FUNNEL_FAVOURED = 2
X_FUNNEL_CARRIERS = (0, 1, 2, 4, 5)


def x_haplotype_prior(funnel):
    r"""founder law of the X chromosome of a fixed line from an eight-founder funnel.

    $\texttt{funnel}[p]$ is the founder at funnel position $p+1$. The X chromosome
    descends only from the founders at positions 1, 2, 3, 5 and 6; the founders at
    positions 4, 7 and 8 carry no X. The founder at position 3 has probability
    $1/3$ and each of the other four carriers $1/6$.

    args:
        funnel: (8,) int array, founder index at each funnel position
    returns:
        prior : (8,) float array summing to 1
    """
    funnel = np.asarray(funnel, dtype=int)
    prior = np.zeros(len(funnel))
    prior[funnel[list(X_FUNNEL_CARRIERS)]] = 1.0 / 6.0
    prior[funnel[X_FUNNEL_FAVOURED]] = 1.0 / 3.0
    return prior


def x_haplotype_transition_matrix(r, funnel):
    r"""founder-haplotype transition matrix of the X chromosome of a fixed line.

    With $c$ the founder at funnel position 3 and $S$ the five carrier founders,
    for $f, g \in S$: $H_{cc} = 1/(1 + 4r)$, $H_{ff} = (1 - r)/(1 + 4r)$ for
    $f \neq c$, $H_{fc} = 2r/(1 + 4r)$ for $f \neq c$, and $H_{fg} = r/(1 + 4r)$
    for $f \neq g$, both different from $c$. Founders outside $S$ never occur and
    keep identity rows. The matrix is reversible with respect to
    $\texttt{x\_haplotype\_prior}$ and equals that law at $r = 0.5$.

    args:
        r     : single-meiosis recombination fraction between the loci (0 to 0.5)
        funnel: (8,) int array, founder index at each funnel position
    returns:
        H : (8, 8) matrix, rows sum to 1
    """
    funnel = np.asarray(funnel, dtype=int)
    carriers = funnel[list(X_FUNNEL_CARRIERS)]
    favoured = funnel[X_FUNNEL_FAVOURED]
    d = 1.0 + 4.0 * r
    H = np.eye(len(funnel))
    for f in carriers:
        H[f, carriers] = r / d
        H[f, favoured] = 2.0 * r / d
        H[f, f] = 1.0 / d if f == favoured else (1.0 - r) / d
    return H


def transition_matrix(r, n_founders=8):
    """compute diplotype transition matrix at recombination fraction r.

    the two chromosomes of a diplotype evolve independently under the
    haplotype kernel, which is derived for fixed (inbred) lines.

    args:
        r         : single-meiosis recombination fraction (0 to 0.5)
        n_founders: J
    returns:
        T : (n_states, n_states) transition matrix, rows sum to 1
    """
    J = n_founders
    states, state_to_idx = build_diplotype_states(J)
    n_states = len(states)

    H = haplotype_transition_matrix(r, J)

    # the two haplotypes of a diplotype evolve independently, so an unordered
    # target {g1, g2} is reached through both ordered outcomes (g1, g2) and
    # (g2, g1); a homozygous target is a single ordered outcome:
    # P({f1,f2} -> {g1,g2}) = H[f1,g1] H[f2,g2] + 1[g1 != g2] H[f1,g2] H[f2,g1]
    # every row then sums to 1 exactly, with no renormalisation
    T = np.zeros((n_states, n_states))
    for i, (f1, f2) in enumerate(states):
        for j, (g1, g2) in enumerate(states):
            val = H[f1, g1] * H[f2, g2]
            if g1 != g2:
                val += H[f1, g2] * H[f2, g1]
            T[i, j] = val

    if not np.allclose(T.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("diplotype transition rows do not sum to 1")
    return T


# ── emission probabilities ────────────────────────────────────────────────────


def build_emission_matrix(founder_genos, allele_codes, error_prob=0.002, n_founders=8):
    """build emission matrix P(obs | diplotype state) per marker.

    args:
        founder_genos : (n_markers, J) array of founder genotype calls
                        values: 'A', 'H', 'B', '-' (missing)
        allele_codes  : (n_markers, 2) array of allele1/allele2 per marker
        error_prob    : genotyping error rate (GigaMUGA standard: 0.002)
        n_founders    : J

    returns:
        E : (n_markers, n_states, 3) emission matrix
            axis 2: [P(obs=A|state), P(obs=H|state), P(obs=B|state)]
    """
    states, _ = build_diplotype_states(n_founders)
    n_states = len(states)
    n_markers = founder_genos.shape[0]

    # obs_idx: A=0, H=1, B=2, missing=-1
    obs_map = {"A": 0, "H": 1, "B": 2, "-": -1}

    E = np.zeros((n_markers, n_states, 3))

    for m in range(n_markers):
        # founder alleles at this marker: 0=A, 1=H, 2=B, -1=missing
        f_alleles = np.array([obs_map.get(str(g), -1) for g in founder_genos[m]])

        for si, (f1, f2) in enumerate(states):
            a1 = f_alleles[f1]
            a2 = f_alleles[f2]

            # expected observation from diplotype (f1, f2)
            if a1 == -1 or a2 == -1:
                # missing founder data — uniform emission
                E[m, si, :] = 1 / 3
                continue

            # expected genotype from the two founder alleles
            # A=0, H=1, B=2; combine to get A/H/B
            # A+A=A, B+B=B, A+B=H, A+H=mixture, etc.
            # simplified: if same → homozygous, different → heterozygous
            if a1 == a2:
                expected = a1  # 0=A or 2=B
            else:
                expected = 1  # H

            # emit with error model
            e = error_prob / 2
            for obs in range(3):
                if obs == expected:
                    E[m, si, obs] = 1 - error_prob
                else:
                    E[m, si, obs] = e

    return E


# ── forward-backward HMM ──────────────────────────────────────────────────────


def forward_backward(obs_seq, E, T_list, init_probs, log_scale=True):
    """run forward-backward algorithm to get posterior diplotype probs.

    args:
        obs_seq    : (n_markers,) int array of observations (0=A, 1=H, 2=B, -1=missing)
        E          : (n_markers, n_states, 3) emission matrix
        T_list     : list of n_states×n_states transition matrices between markers
        init_probs : (n_states,) initial state distribution
        log_scale  : use log-space for numerical stability

    returns:
        gamma : (n_markers, n_states) posterior state probabilities
        alpha : (n_markers, n_states) scaled forward probabilities
        beta  : (n_markers, n_states) scaled backward probabilities

    alpha and beta are returned to allow pseudomarker interpolation via
    interpolate_posteriors() — posteriors at positions with no genotype
    observation are derived from flanking alpha/beta without re-running
    the full HMM (see interpolate_posteriors for pseudomarker interpolation).
    """
    n_markers, n_states = E.shape[0], E.shape[1]

    # forward pass
    alpha = np.zeros((n_markers, n_states))
    obs0 = obs_seq[0]
    if obs0 >= 0:
        alpha[0] = init_probs * E[0, :, obs0]
    else:
        alpha[0] = init_probs
    s = alpha[0].sum()
    alpha[0] /= max(s, 1e-300)

    for t in range(1, n_markers):
        T = T_list[t - 1]  # (n_states, n_states)
        obs_t = obs_seq[t]
        # alpha[t,j] = sum_i alpha[t-1,i] * T[i,j] * E[t,j,obs]
        pred = alpha[t - 1] @ T  # (n_states,)
        if obs_t >= 0:
            alpha[t] = pred * E[t, :, obs_t]
        else:
            alpha[t] = pred
        s = alpha[t].sum()
        alpha[t] /= max(s, 1e-300)

    # backward pass
    beta = np.zeros((n_markers, n_states))
    beta[-1] = 1.0

    for t in range(n_markers - 2, -1, -1):
        T = T_list[t]
        obs_t1 = obs_seq[t + 1]
        if obs_t1 >= 0:
            scaled = E[t + 1, :, obs_t1] * beta[t + 1]
        else:
            scaled = beta[t + 1]
        # beta[t,i] = sum_j T[i,j] * E[t+1,j,obs] * beta[t+1,j]
        beta[t] = T @ scaled
        s = beta[t].sum()
        beta[t] /= max(s, 1e-300)

    # posterior
    gamma = alpha * beta
    row_sums = gamma.sum(axis=1, keepdims=True)
    gamma /= np.maximum(row_sums, 1e-300)
    return gamma, alpha, beta


# ── pseudomarker interpolation ─────────────────────────────────────────────────


def _augmented_positions(gmap_cM, step_cM, tol=0.01):
    """chromosome-level pseudomarker grid with fixed step size.

    grid: seq(min(gmap), max(gmap), by=step_cM) anchored at the first real marker —
    NOT reset at each interval. pseudomarkers within tol cM of any real marker are
    dropped.

    args:
        gmap_cM  : (n_markers,) sorted genetic positions in cM
        step_cM  : pseudomarker spacing in cM (e.g. 0.2)
        tol      : drop grid points within this cM distance of a real marker (default 0.01)

    returns:
        all_pos_cM : (n_total,) sorted positions — real + pseudo
        is_real    : (n_total,) bool — True = real GigaMUGA marker
    """
    gmap_cM = np.asarray(gmap_cM, dtype=float)
    span = gmap_cM[-1] - gmap_cM[0]
    # ceil ensures we always generate enough steps; clip removes any floating-point overshoot
    n_steps = int(np.ceil(span / step_cM))
    grid = gmap_cM[0] + np.arange(n_steps + 1) * step_cM
    grid = grid[grid <= gmap_cM[-1] + step_cM * 1e-6]

    # for each grid point: distance to nearest real marker; drop if < tol
    dist_to_nearest = np.abs(grid[:, None] - gmap_cM[None, :]).min(axis=1)
    pseudo_pos = grid[dist_to_nearest >= tol]

    all_pos = np.concatenate([gmap_cM, pseudo_pos])
    is_real = np.concatenate(
        [np.ones(len(gmap_cM), dtype=bool), np.zeros(len(pseudo_pos), dtype=bool)]
    )
    order = np.argsort(all_pos, kind="stable")
    return all_pos[order], is_real[order]


def interpolate_posteriors(
    gamma, alpha, beta, E, obs_seq, gmap_cM, step_cM, n_founders, tol=0.01, kernel=None
):
    """interpolate HMM posteriors at pseudomarker positions on a fixed-step grid.

    positions are computed by _augmented_positions(): chromosome-level absolute
    grid seq(min(gmap), max(gmap), by=step_cM), dropping points within tol cM of
    any real marker.

    math at each pseudomarker p bracketed by real markers m and m+1:
      alpha_p(j) = Σ_i alpha[m](i) * T(m→p)[i,j]
      beta_p(j)  = Σ_k T(p→m+1)[j,k] * E[m+1,k,obs_{m+1}] * beta[m+1](k)

    note: beta[m+1] from forward_backward = P(obs[m+2..] | state at m+1) and does
    NOT include E[m+1]. beta_p must multiply E[m+1] back in explicitly.

    args:
        gamma    : (n_markers, n_states)
        alpha    : (n_markers, n_states)  scaled forward probs
        beta     : (n_markers, n_states)  scaled backward probs
        E        : (n_markers, n_states, 3)  emission matrix at real markers
        obs_seq  : (n_markers,) int  0/1/2 or -1=missing
        gmap_cM  : (n_markers,) sorted genetic positions in cM
        step_cM  : float  pseudomarker spacing in cM
        n_founders: int; gamma with n_founders columns is read as founder-haplotype
                    states, otherwise as diplotype states
        tol      : float  min cM distance from a real marker to keep a pseudomarker (default 0.01)
        kernel   : optional callable r -> (n_states, n_states) transition matrix replacing
                   the default kernel of the state space (used for the X chromosome)

    returns:
        gamma_all  : (n_total, n_states)
        all_pos_cM : (n_total,)
        is_real    : (n_total,) bool
    """
    aug_pos, is_real = _augmented_positions(gmap_cM, step_cM, tol)
    n_states = gamma.shape[1]
    # J states = founder-haplotype chain (inbred lines); J(J+1)/2 = diplotype chain
    haplotype_states = n_states == n_founders
    n_total = len(aug_pos)
    all_gamma = np.empty((n_total, n_states))

    # fill posteriors at real-marker positions
    for j, k in enumerate(np.where(is_real)[0]):
        all_gamma[k] = gamma[j]

    # fill posteriors at pseudomarker positions
    for k in np.where(~is_real)[0]:
        pos = aug_pos[k]
        # last real marker ≤ pos; clamp to valid interval
        j = int(np.searchsorted(gmap_cM, pos, side="right")) - 1
        j = min(max(j, 0), len(gmap_cM) - 2)

        r1 = haldane_recomb((pos - gmap_cM[j]) / 100.0)
        r2 = haldane_recomb((gmap_cM[j + 1] - pos) / 100.0)
        if kernel is not None:
            T1 = kernel(r1)
            T2 = kernel(r2)
        elif haplotype_states:
            T1 = haplotype_transition_matrix(r1, n_founders)
            T2 = haplotype_transition_matrix(r2, n_founders)
        else:
            T1 = transition_matrix(r1, n_founders)
            T2 = transition_matrix(r2, n_founders)
        alpha_p = alpha[j] @ T1
        # beta[j+1] = P(obs[j+2..] | state at j+1) — no E[j+1]; must re-include it
        obs_m1 = obs_seq[j + 1]
        scaled_m1 = E[j + 1, :, obs_m1] * beta[j + 1] if obs_m1 >= 0 else beta[j + 1]
        beta_p = T2 @ scaled_m1
        gp = alpha_p * beta_p
        s = gp.sum()
        gp /= max(s, 1e-300)
        all_gamma[k] = gp

    return all_gamma, aug_pos, is_real


# ── initial state distribution ────────────────────────────────────────────────


def uniform_init(n_founders=8):
    """uniform diplotype initial distribution (equal weight to all states).
    for non-CC / unknown breeding design.
    """
    states, _ = build_diplotype_states(n_founders)
    n = len(states)
    return np.full(n, 1.0 / n)


# ── top-level: compute haplotype probs for CC strains ────────────────────────


def calc_genoprob_cc(
    geno_matrix,
    founder_genos,
    gmap_positions,
    n_founders=8,
    error_prob=0.002,
    return_additive=True,
    cc_inbred=True,
    step_cM=None,
    x_funnel=None,
):
    """compute founder haplotype probabilities for CC inbred strains.

    runs the 8-founder diplotype HMM using forward-backward and returns
    either diplotype (36 states) or additive (8 founder) probabilities.

    args:
        geno_matrix   : (n_markers, n_cc) array of genotype calls (A/H/B/-)
                        columns = CC strains, rows = markers
        founder_genos : (n_markers, J) array of founder genotype calls
        gmap_positions: (n_markers,) single-meiosis genetic positions in cM
        n_founders    : J (8 for CC)
        error_prob   : GigaMUGA genotyping error rate (0.002)
        return_additive: if True return (n_cc, J, n_markers) additive probs
                         if False return (n_cc, n_states, n_markers) diplotype
        cc_inbred     : if True run the chain on the J homozygous states only
                        (fully inbred strains; uniform initial law 1/J,
                        haplotype kernel). if False run the 36-state chain
                        with the diplotype kernel and a uniform initial law
                        (outbred or partly inbred populations). default True.
        x_funnel      : (n_cc, J) int array, founder index at each funnel position of
                        each strain, for an X chromosome; requires cc_inbred. each
                        strain then runs on its own X prior and kernel
                        (x_haplotype_prior, x_haplotype_transition_matrix).

    returns:
        probs : (n_cc, J or n_states, n_markers) float64 array
                output dimensions: samples × states/founders × markers
    """
    n_markers, n_cc = geno_matrix.shape
    states, _ = build_diplotype_states(n_founders)
    n_states = len(states)

    obs_map = {"A": 0, "H": 1, "B": 2, "-": -1}

    log.info(f"building emission matrix ({n_markers} markers × {n_states} states)")
    E = build_emission_matrix(founder_genos, None, error_prob, n_founders)

    # a fully inbred line carries one founder haplotype at every locus, so its
    # chain lives on the J homozygous states; the 28 heterozygous states have no
    # support and would only leak mass along the chromosome
    if cc_inbred:
        _, state_to_idx = build_diplotype_states(n_founders)
        hom_idx = np.array([state_to_idx[(f, f)] for f in range(n_founders)])
        E = E[:, hom_idx, :]
        kernel = haplotype_transition_matrix
        init = np.full(n_founders, 1.0 / n_founders)
    else:
        kernel = transition_matrix
        init = uniform_init(n_founders)

    if x_funnel is not None:
        if not cc_inbred:
            raise ValueError("x_funnel requires cc_inbred=True")
        x_funnel = np.asarray(x_funnel, dtype=int)
        if x_funnel.shape != (n_cc, n_founders):
            raise ValueError(
                f"x_funnel must have shape {(n_cc, n_founders)}, got {x_funnel.shape}"
            )

    log.info("building transition matrices...")
    r_list = [
        haldane_recomb(abs(gmap_positions[t + 1] - gmap_positions[t]) / 100.0)
        for t in range(n_markers - 1)
    ]
    T_list = [kernel(r, n_founders) for r in r_list] if x_funnel is None else None

    n_out = n_founders if return_additive else n_states

    # determine n_out_loci from geometry only — no probability computation needed
    if step_cM is not None:
        _aug_cM_cache, _is_real_cache = _augmented_positions(gmap_positions, step_cM)
        n_out_loci = len(_aug_cM_cache)
    else:
        n_out_loci = n_markers

    probs = np.zeros((n_cc, n_out, n_out_loci), dtype=np.float64)

    for s in range(n_cc):
        if s % 5 == 0:
            log.debug(
                f"calc_genoprob_cc: strain {s+1}/{n_cc} ({100.0*(s+1)/n_cc:.1f}%)"
            )
        obs = np.array([obs_map.get(str(g), -1) for g in geno_matrix[:, s]])
        # (n_markers, n_states) posterior diplotype probs + forward/backward state
        if x_funnel is None:
            T_s, init_s, kernel_s = T_list, init, None
        else:
            funnel_s = x_funnel[s]
            T_s = [x_haplotype_transition_matrix(r, funnel_s) for r in r_list]
            init_s = x_haplotype_prior(funnel_s)

            def kernel_s(r, funnel_s=funnel_s):
                return x_haplotype_transition_matrix(r, funnel_s)

        gamma, alpha, beta = forward_backward(obs, E, T_s, init_s)
        if step_cM is not None:
            # interpolate posteriors at pseudomarker positions between real markers
            gamma, _, _ = interpolate_posteriors(
                gamma,
                alpha,
                beta,
                E,
                obs,
                gmap_positions,
                step_cM,
                n_founders,
                kernel=kernel_s,
            )
        if cc_inbred:
            if return_additive:
                probs[s] = gamma.T  # haplotype posteriors are the founder probs
            else:
                probs[s][hom_idx] = gamma.T
        elif return_additive:
            add = diplotype_to_allele_probs(gamma, n_founders)
            probs[s] = add.T  # (8, n_total)
        else:
            probs[s] = gamma.T  # (n_states, n_total)

    if step_cM is not None:
        return probs, _aug_cM_cache, _is_real_cache
    return probs

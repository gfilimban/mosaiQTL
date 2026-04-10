# hmm.py
# pure python CC/DO haplotype reconstruction HMM
#
# replaces R's qtl2::calc_genoprob() for CC inbred strains.
# implements the 8-founder diplotype HMM used by qtl2/DOQTL:
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
import pandas as pd
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
        diplotype_probs : (n_states,) or (n_loci, n_samples, n_states)
    returns:
        allele_probs : same leading dims, last dim = n_founders
    """
    states, _ = build_diplotype_states(n_founders)
    n_states   = len(states)

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


def transition_matrix(r, n_founders=8, n_gen=1):
    """compute diplotype transition matrix at recombination fraction r.

    for CC mice the genetic map (from the CC consensus map) already
    encodes the cumulative recombination from ~20 generations of
    intercrossing. the per-interval recombination fraction r derived
    from that map should therefore be used directly — do NOT multiply
    by n_gen (that would inflate r by 20×, collapsing all transitions
    toward the equilibrium distribution and destroying haplotype blocks).

    n_gen is kept as a parameter for non-CC populations (e.g. DO F1s)
    where the caller may need to adjust; for CC it should be left at 1.

    args:
        r         : per-interval recombination fraction (0 to 0.5)
        n_founders: J
        n_gen     : scaling factor (1 for CC with consensus map)
    returns:
        T : (n_states, n_states) transition matrix, rows sum to 1
    """
    J = n_founders
    states, state_to_idx = build_diplotype_states(J)
    n_states = len(states)

    # r is already the correct per-interval recombination fraction
    # n_gen=1 for CC with consensus map; capped at 0.5
    r_eff = min(r * n_gen, 0.5)

    # haplotype stay / switch probabilities
    p_stay   = 1 - r_eff * (J - 1) / J
    p_switch = r_eff / J

    # build haplotype transition matrix (J × J)
    H = np.full((J, J), p_switch)
    np.fill_diagonal(H, p_stay)
    # normalize rows
    H = H / H.sum(axis=1, keepdims=True)

    # diplotype transition = outer product of two haplotype transitions
    # P((f1,f2)→(g1,g2)) = H[f1,g1] * H[f2,g2]  (+ symmetric term)
    T = np.zeros((n_states, n_states))
    for i, (f1, f2) in enumerate(states):
        for j, (g1, g2) in enumerate(states):
            if f1 == f2:
                # homozygous source: both haplotypes are f1
                val = H[f1, g1] * H[f1, g2]
                if g1 != g2:
                    val += H[f1, g2] * H[f1, g1]   # symmetric diplotype
            else:
                # heterozygous source: haplotypes f1 and f2
                val = (H[f1, g1] * H[f2, g2] +
                       H[f2, g1] * H[f1, g2])
            if g1 != g2:
                val /= 2   # unordered diplotype: each ordered pair counted once
            T[i, j] = val

    # renormalize rows
    row_sums = T.sum(axis=1, keepdims=True)
    T = T / np.maximum(row_sums, 1e-300)
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
    n_states   = len(states)
    n_markers  = founder_genos.shape[0]

    # obs_idx: A=0, H=1, B=2, missing=-1
    obs_map = {'A': 0, 'H': 1, 'B': 2, '-': -1}

    E = np.zeros((n_markers, n_states, 3))

    for m in range(n_markers):
        # founder alleles at this marker: 0=A, 1=H, 2=B, -1=missing
        f_alleles = np.array([obs_map.get(str(g), -1)
                              for g in founder_genos[m]])

        for si, (f1, f2) in enumerate(states):
            a1 = f_alleles[f1]
            a2 = f_alleles[f2]

            # expected observation from diplotype (f1, f2)
            if a1 == -1 or a2 == -1:
                # missing founder data — uniform emission
                E[m, si, :] = 1/3
                continue

            # expected genotype from the two founder alleles
            # A=0, H=1, B=2; combine to get A/H/B
            # A+A=A, B+B=B, A+B=H, A+H=mixture, etc.
            # simplified: if same → homozygous, different → heterozygous
            if a1 == a2:
                expected = a1   # 0=A or 2=B
            else:
                expected = 1    # H

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
    the full HMM (see qtl2::insert_pseudomarkers equivalence).
    """
    n_markers, n_states = E.shape[0], E.shape[1]

    # forward pass
    alpha = np.zeros((n_markers, n_states))
    obs0  = obs_seq[0]
    if obs0 >= 0:
        alpha[0] = init_probs * E[0, :, obs0]
    else:
        alpha[0] = init_probs
    s = alpha[0].sum()
    alpha[0] /= max(s, 1e-300)

    for t in range(1, n_markers):
        T  = T_list[t - 1]                # (n_states, n_states)
        obs_t = obs_seq[t]
        # alpha[t,j] = sum_i alpha[t-1,i] * T[i,j] * E[t,j,obs]
        pred = alpha[t - 1] @ T           # (n_states,)
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
        T   = T_list[t]
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

def interpolate_posteriors(gamma, alpha, beta, E, obs_seq, gmap_cM, step_cM, n_founders):
    """interpolate HMM posteriors at pseudomarker positions between real markers.

    mathematically identical to qtl2's full HMM with pseudomarkers inserted
    as missing-observation (uniform emission) positions, PROVIDED the correct
    emission term is included in the backward message.

    derivation:
      alpha_p(j) = Σ_i alpha[m](i) * T1[i,j]
                 = forward prob at p (no emission; missing obs → uniform = 1)
      beta_p(j)  = Σ_k T2[j,k] * E[m+1, k, obs_{m+1}] * beta[m+1](k)
                 = backward message at p, including emission at the NEXT real marker

    note: beta[m+1] from forward_backward() encodes P(obs[m+2..] | state at m+1)
    and does NOT contain E[m+1]. the emission at m+1 is only in alpha[m+1].
    the backward message at p therefore must multiply E[m+1, :, obs_{m+1}]
    back in explicitly.

    this requires E and obs to be passed in — see calc_genoprob_cc(step_cM=...)
    for the augmented-HMM path that avoids this by running a single HMM over
    all positions (real + pseudo), which is equivalent but cleaner.

    args:
        gamma    : (n_markers, n_states)  posteriors at real markers
        alpha    : (n_markers, n_states)  scaled forward probs from forward_backward
        beta     : (n_markers, n_states)  scaled backward probs from forward_backward
        E        : (n_markers, n_states, 3)  emission matrix at real markers
        obs_seq  : (n_markers,) int  observations at real markers (0/1/2, -1=missing)
        gmap_cM  : (n_markers,)           genetic positions of real markers in cM
        step_cM  : float                  pseudomarker spacing in cM (e.g. 1.0)
        n_founders: int

    returns:
        gamma_all  : (n_total, n_states)  posteriors at real + pseudo positions
        all_pos_cM : (n_total,)           genetic positions (sorted)
        is_real    : (n_total,) bool      True = real GigaMUGA marker
    """
    n_markers = len(gmap_cM)
    n_states  = gamma.shape[1]

    all_gamma  = []
    all_pos_cM = []
    is_real    = []

    for m in range(n_markers - 1):
        # real marker m
        all_gamma.append(gamma[m])
        all_pos_cM.append(gmap_cM[m])
        is_real.append(True)

        # pseudomarkers in interval (gmap_cM[m], gmap_cM[m+1])
        d = gmap_cM[m + 1] - gmap_cM[m]
        if d > step_cM:
            n_pseudo = int(np.floor(d / step_cM))
            for k in range(1, n_pseudo + 1):
                p_cM = gmap_cM[m] + k * step_cM
                if p_cM >= gmap_cM[m + 1]:
                    break
                r1 = haldane_recomb((p_cM - gmap_cM[m]) / 100.0)
                r2 = haldane_recomb((gmap_cM[m + 1] - p_cM) / 100.0)
                T1 = transition_matrix(r1, n_founders)
                T2 = transition_matrix(r2, n_founders)
                alpha_p = alpha[m] @ T1              # (n_states,) forward to p
                # backward at p: must include emission at m+1 because
                # beta[m+1] = P(obs[m+2..] | state at m+1) — no E[m+1] in it
                obs_m1 = obs_seq[m + 1]
                if obs_m1 >= 0:
                    scaled_m1 = E[m + 1, :, obs_m1] * beta[m + 1]
                else:
                    scaled_m1 = beta[m + 1]
                beta_p  = T2 @ scaled_m1             # (n_states,)
                gp      = alpha_p * beta_p
                s       = gp.sum()
                gp     /= max(s, 1e-300)
                all_gamma.append(gp)
                all_pos_cM.append(p_cM)
                is_real.append(False)

    # last real marker
    all_gamma.append(gamma[-1])
    all_pos_cM.append(gmap_cM[-1])
    is_real.append(True)

    return (np.array(all_gamma),       # (n_total, n_states)
            np.array(all_pos_cM),      # (n_total,)
            np.array(is_real, dtype=bool))


# ── initial state distribution ────────────────────────────────────────────────

def uniform_init(n_founders=8):
    """uniform diplotype initial distribution (equal weight to all states).
    for non-CC / unknown breeding design.
    """
    states, _ = build_diplotype_states(n_founders)
    n = len(states)
    return np.full(n, 1.0 / n)


def cc_inbred_init(n_founders=8):
    """initial distribution for fully inbred CC strains.

    CC strains are homozygous throughout after ~20 generations of inbreeding.
    only the 8 homozygous diplotype states (AA, BB, ..., HH) have non-zero
    prior probability; the 28 heterozygous states get 0. this is the correct
    prior for a fully inbred line regardless of funnel order — the crossinfo
    funnel confirms the 8-way design but does not differentially weight founders
    on autosomes (all 8 contribute equally in expectation).
    """
    states, state_to_idx = build_diplotype_states(n_founders)
    init = np.zeros(len(states))
    for f in range(n_founders):
        init[state_to_idx[(f, f)]] = 1.0 / n_founders  # homozygous states only
    return init


# ── top-level: compute haplotype probs for CC strains ────────────────────────

def calc_genoprob_cc(geno_matrix, founder_genos, gmap_positions,
                     n_founders=8, n_gen=1, error_prob=0.002,
                     return_additive=True, cc_inbred=True):
    """compute founder haplotype probabilities for CC inbred strains.

    pure Python replacement for qtl2::calc_genoprob() on CC samples.
    runs the 8-founder diplotype HMM using forward-backward and returns
    either diplotype (36 states) or additive (8 founder) probabilities.

    args:
        geno_matrix   : (n_markers, n_cc) array of genotype calls (A/H/B/-)
                        columns = CC strains, rows = markers
        founder_genos : (n_markers, J) array of founder genotype calls
        gmap_positions: (n_markers,) genetic positions in Morgans (not cM)
        n_founders    : J (8 for CC)
        n_gen         : transition scaling factor. MUST be 1 for CC with the
                        consensus genetic map (JAX/UNC GigaMUGA map), because
                        that map already encodes the cumulative recombination
                        from ~20 generations of intercrossing — r is used
                        directly (r_eff = r * n_gen). use n_gen > 1 only for
                        outbred populations (e.g. DO) where the map does not
                        encode generational history.
        error_prob    : GigaMUGA genotyping error rate (0.002)
        return_additive: if True return (n_cc, J, n_markers) additive probs
                         if False return (n_cc, n_states, n_markers) diplotype
        cc_inbred     : if True use cc_inbred_init (homozygous-only prior,
                        correct for fully inbred CC strains). if False use
                        uniform_init (1/36 per state, for outbred/unknown).
                        default True.

    returns:
        probs : (n_cc, J or n_states, n_markers) float64 array
                qtl2 dimension convention: samples × states/founders × markers
    """
    n_markers, n_cc = geno_matrix.shape
    states, _       = build_diplotype_states(n_founders)
    n_states        = len(states)

    obs_map = {'A': 0, 'H': 1, 'B': 2, '-': -1}

    log.info(f'building emission matrix ({n_markers} markers × {n_states} states)')
    E = build_emission_matrix(founder_genos, None, error_prob, n_founders)

    log.info('building transition matrices...')
    T_list = []
    for t in range(n_markers - 1):
        d_cm     = abs(gmap_positions[t + 1] - gmap_positions[t])
        d_morgan = d_cm / 100.0
        r        = haldane_recomb(d_morgan)
        T_list.append(transition_matrix(r, n_founders, n_gen))

    init = cc_inbred_init(n_founders) if cc_inbred else uniform_init(n_founders)
    n_out = n_founders if return_additive else n_states
    probs = np.zeros((n_cc, n_out, n_markers), dtype=np.float64)

    for s in range(n_cc):
        if s % 5 == 0:
            log.debug(f'  strain {s+1}/{n_cc}')
        obs = np.array([obs_map.get(str(g), -1) for g in geno_matrix[:, s]])
        # (n_markers, n_states) posterior diplotype probs + forward/backward state
        gamma, alpha, beta = forward_backward(obs, E, T_list, init)
        if step_cM is not None:
            # interpolate at pseudomarker positions between real markers
            gamma, _, _ = interpolate_posteriors(gamma, alpha, beta,
                                                  gmap_positions, step_cM, n_founders)
        if return_additive:
            add = diplotype_to_allele_probs(gamma, n_founders)
            probs[s] = add.T      # (8, n_total)
        else:
            probs[s] = gamma.T    # (n_states, n_total)

    if step_cM is not None:
        # return augmented positions alongside probs so caller can update locus index
        _, all_gmap_cM, is_real = interpolate_posteriors(
            np.zeros_like(gamma), np.zeros_like(gamma), np.zeros_like(gamma),
            gmap_positions, step_cM, n_founders)
        return probs, all_gmap_cM, is_real
    return probs

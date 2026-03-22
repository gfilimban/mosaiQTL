"""
mosaiqtl validation — cross-language comparison against r miqtl/qtl2

simulates cc mouse data matching the real pipeline structure:
  - 48 unique genotypes (33 cc + 8 founders + 7 f1s)
  - ~8 replicates each → 384 samples (like the real 415)
  - happy genome cache: additive/chr{c}/data/*.rdata (48 × 8)
  - loco kinship: 48×48 matrices with strain labels as rownames
  - phenotype: rint'd log-cpm, subject.name, sex, batch, pc1..pc10
  - k expansion: 48→384 via strain label index replication

the r reference script uses scan.h2lmm() identically to scan_eqtl.R.

usage:
  python validation/validate_vs_r.py --module all
"""

import numpy as np
import pandas as pd
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mosaiqtl.genome_cache import GenomeCache, CC_FOUNDERS
from mosaiqtl import kinship as kin_module
from mosaiqtl import lmm as lmm_module
from mosaiqtl import scan as scan_module
from mosaiqtl import nb as nb_module
from mosaiqtl import vqtl as vqtl_module
from mosaiqtl.utils import rint

logging.basicConfig(level=logging.INFO, format='%(name)s | %(message)s')
logger = logging.getLogger('validation')


# ----------------------------------------------------------------
# strain labels matching real cc pipeline
# ----------------------------------------------------------------

# 8 founder strains (same order as genome cache columns)
FOUNDER_STRAINS = ['AJ', 'B6J', '129S1J', 'CASTJ', 'NODJ', 'NZOJ', 'PWKJ', 'WSBJ']

# 7 f1 hybrids (b6j as dam x each non-b6j founder)
F1_STRAINS = ['B6AF1J', 'B6129S1F1J', 'B6CASTF1J', 'B6NODF1J',
              'B6NZOF1J', 'B6PWKF1J', 'B6WSBF1J']

# 33 cc strains
CC_STRAINS = [f'CC{i:03d}' for i in range(1, 34)]

# all 48 unique genotypes — matches kinship rownames and cache rownames
ALL_48_STRAINS = FOUNDER_STRAINS + F1_STRAINS + CC_STRAINS

# replicates per strain (4m + 4f = 8)
REPS_PER_STRAIN = 8


# ----------------------------------------------------------------
# simulate genome cache in happy format (48 x 8 per locus)
# ----------------------------------------------------------------

def simulate_cc_haplotypes(n_loci_per_chr=50, n_chromosomes=3, seed=42):
    """simulate founder haplotype probabilities for 48 cc genotypes.

    founders: diagonal (probability 1 for self)
    f1s: 0.5 for b6j + 0.5 for other parent
    cc strains: mosaic of 8 founders with recombination breakpoints
    """
    rng = np.random.default_rng(seed)
    n_founders = 8
    n_subjects = len(ALL_48_STRAINS)  # 48

    data = {}
    for c in range(1, n_chromosomes + 1):
        chrom = str(c)
        # marker names mimic gigamuga format
        locus_ids = [f'UNC{c:02d}{m:06d}' for m in range(n_loci_per_chr)]
        # positions in mb (matching pmap.csv format from real pipeline)
        positions = np.linspace(3.0, 150.0, n_loci_per_chr)

        probs = np.zeros((n_loci_per_chr, n_subjects, n_founders))

        for l_idx in range(n_loci_per_chr):
            # founders: identity — each founder is 100% self at every locus
            for f_idx in range(8):
                probs[l_idx, f_idx, :] = 0.0
                probs[l_idx, f_idx, f_idx] = 1.0

            # f1s: 50% b6j (idx 1) + 50% other parent
            for f1_idx, other_parent_idx in enumerate([0, 2, 3, 4, 5, 6, 7]):
                s_idx = 8 + f1_idx
                probs[l_idx, s_idx, :] = 0.0
                probs[l_idx, s_idx, 1] = 0.5
                probs[l_idx, s_idx, other_parent_idx] = 0.5

            # cc strains: mosaic haplotype blocks
            for cc_idx in range(33):
                s_idx = 15 + cc_idx
                n_breaks = rng.poisson(3)
                break_loci = np.sort(rng.choice(
                    n_loci_per_chr,
                    size=min(n_breaks, n_loci_per_chr - 1),
                    replace=False
                ))
                seg_idx = np.searchsorted(break_loci, l_idx, side='right')
                dominant = (cc_idx * 7 + seg_idx + c) % n_founders

                probs[l_idx, s_idx, :] = 0.005
                probs[l_idx, s_idx, dominant] = 0.965
                probs[l_idx, s_idx, :] /= probs[l_idx, s_idx, :].sum()

        data[chrom] = {
            'locus_ids': np.array(locus_ids),
            'positions': positions,
            'probs': probs,
        }

    return GenomeCache(ALL_48_STRAINS, FOUNDER_STRAINS, data)


# ----------------------------------------------------------------
# simulate replicated phenotype (384 samples like the real 415)
# ----------------------------------------------------------------

def simulate_replicated_phenotype(genome_cache, causal_chrom='1',
                                   causal_locus_idx=25,
                                   causal_founder=3,  # castj
                                   effect_size=3.0,
                                   n_genes=5, seed=42):
    """simulate phenotype matching pheno_df structure from notebook cell 15.

    returns pheno_df with columns:
      sample_id, subject.name, sex, batch, pc1..pc10, gene1..geneN
    """
    rng = np.random.default_rng(seed)

    # build replicate sample table (4m + 4f per strain)
    rows = []
    for strain in ALL_48_STRAINS:
        for rep in range(REPS_PER_STRAIN):
            sex = 'M' if rep < 4 else 'F'
            batch = f'batch{(rep % 3) + 1}'
            sample_id = f'{strain}_{sex}_{rep:02d}'
            rows.append({
                'sample_id': sample_id,
                'subject.name': strain,
                'sex': sex,
                'batch': batch,
            })

    pheno_df = pd.DataFrame(rows)
    n_samples = len(pheno_df)

    # strain index per sample
    strain_to_idx = {s: i for i, s in enumerate(ALL_48_STRAINS)}
    sample_strain_idx = np.array([
        strain_to_idx[s] for s in pheno_df['subject.name']
    ])

    # causal haplotype dosage per strain
    h_causal = genome_cache.get_locus(causal_chrom, causal_locus_idx)

    # simulate gene expression columns
    gene_names = []
    for g in range(n_genes):
        gene_name = f'gene{g + 1}'
        gene_names.append(gene_name)

        if g == 0:
            # causal gene: strong founder effect
            strain_effect = h_causal[:, causal_founder] * effect_size
        else:
            # null genes: small random effects
            strain_effect = h_causal @ rng.normal(0, 0.1, size=8)

        # per-sample: strain + sex + batch + noise
        sex_effect = np.where(pheno_df['sex'] == 'M', 0.3, -0.3)
        batch_effect = np.array([
            {'batch1': 0.0, 'batch2': 0.2, 'batch3': -0.2}[b]
            for b in pheno_df['batch']
        ])
        y = (strain_effect[sample_strain_idx] +
             sex_effect + batch_effect +
             rng.normal(0, 0.8, n_samples))

        # rint transform (matching miqtl::rint)
        pheno_df[gene_name] = rint(y)

    # expression pcs (10 columns, matching notebook cell 16)
    for pc_idx in range(10):
        pheno_df[f'pc{pc_idx + 1}'] = rng.normal(0, 1, n_samples)

    true_params = {
        'causal_chrom': causal_chrom,
        'causal_locus_idx': causal_locus_idx,
        'causal_founder': causal_founder,
        'causal_founder_name': FOUNDER_STRAINS[causal_founder],
        'effect_size': effect_size,
        'causal_gene': 'gene1',
        'gene_names': gene_names,
    }

    return pheno_df, true_params


# ----------------------------------------------------------------
# simulate single-cell counts for vqtl
# ----------------------------------------------------------------

def simulate_sc_counts(genome_cache, pheno_df,
                       causal_chrom='1', causal_locus_idx=25,
                       causal_founder=3, mean_effect=1.5,
                       dispersion_effect=0.8, cells_per_sample=200,
                       base_phi=0.5, base_mu=10.0, seed=42):
    """simulate cell-level nb counts with mean + dispersion qtl."""
    rng = np.random.default_rng(seed)

    strain_to_idx = {s: i for i, s in enumerate(ALL_48_STRAINS)}
    h_causal = genome_cache.get_locus(causal_chrom, causal_locus_idx)

    all_counts, all_sids, all_libs = [], [], []

    for _, row in pheno_df.iterrows():
        s_idx = strain_to_idx[row['subject.name']]
        dosage = h_causal[s_idx, causal_founder]
        mu_s = base_mu * np.exp(mean_effect * dosage)
        phi_s = base_phi * np.exp(dispersion_effect * dosage)

        n_cells = max(rng.poisson(cells_per_sample), 10)
        r_s = 1.0 / phi_s
        counts_s = rng.negative_binomial(r_s, r_s / (r_s + mu_s), size=n_cells)
        lib_s = rng.lognormal(mean=np.log(10000), sigma=0.3, size=n_cells)

        all_counts.append(counts_s)
        # subject_ids = strain labels (genome cache uses strain, not sample_id)
        all_sids.append(np.full(n_cells, row['subject.name']))
        all_libs.append(lib_s)

    return (np.concatenate(all_counts),
            np.concatenate(all_sids),
            np.concatenate(all_libs),
            {'causal_chrom': causal_chrom,
             'causal_locus_idx': causal_locus_idx,
             'causal_founder': causal_founder})


# ----------------------------------------------------------------
# export for r validation
# ----------------------------------------------------------------

def export_for_r(genome_cache, pheno_df, output_dir):
    """export data in formats that the r reference script can read."""
    os.makedirs(output_dir, exist_ok=True)

    # phenotype table
    pheno_df.to_csv(os.path.join(output_dir, 'pheno_df.csv'), index=False)

    # physical map (marker, chr, pos in mb)
    pmap_rows = []
    for chrom in genome_cache.chromosomes:
        lids, positions, _ = genome_cache.get_chromosome(chrom)
        for lid, pos in zip(lids, positions):
            pmap_rows.append({'marker': lid, 'chr': chrom, 'pos': pos})
    pd.DataFrame(pmap_rows).to_csv(
        os.path.join(output_dir, 'pmap.csv'), index=False
    )

    # kinship (loco) — one csv per chromosome
    k_loco = kin_module.calc_kinship(genome_cache, method='loco')
    k_dir = os.path.join(output_dir, 'kinship')
    os.makedirs(k_dir, exist_ok=True)
    for chrom, k_mat in k_loco.items():
        k_df = pd.DataFrame(k_mat,
                            index=genome_cache.sample_ids,
                            columns=genome_cache.sample_ids)
        k_df.to_csv(os.path.join(k_dir, f'kinship_chr{chrom}.csv'))

    # haplotype probs — flat csv per chromosome
    hap_dir = os.path.join(output_dir, 'haplotypes')
    os.makedirs(hap_dir, exist_ok=True)
    for chrom in genome_cache.chromosomes:
        lids, positions, probs = genome_cache.get_chromosome(chrom)
        rows = []
        for l_idx in range(len(lids)):
            for s_idx in range(len(genome_cache.sample_ids)):
                row = {'locus': lids[l_idx], 'pos_mb': positions[l_idx],
                       'subject': genome_cache.sample_ids[s_idx]}
                for f_idx, f_name in enumerate(genome_cache.founders):
                    row[f_name] = probs[l_idx, s_idx, f_idx]
                rows.append(row)
        pd.DataFrame(rows).to_csv(
            os.path.join(hap_dir, f'chr{chrom}.csv'), index=False
        )

    logger.info(f'exported r validation data to {output_dir}')


# ----------------------------------------------------------------
# validation tests
# ----------------------------------------------------------------

def validate_kinship(genome_cache, tol=1e-6):
    """validate kinship: symmetry, psd, correct structure for 48 strains."""
    logger.info('=== validating kinship computation ===')

    k_loco = kin_module.calc_kinship(genome_cache, method='loco')

    for chrom, k_mat in k_loco.items():
        n = k_mat.shape[0]
        assert n == 48, f'kinship should be 48x48, got {n}x{n}'
        assert np.allclose(k_mat, k_mat.T, atol=tol), 'not symmetric'
        eigvals = np.linalg.eigvalsh(k_mat)
        assert np.all(eigvals >= -tol), 'negative eigenvalues'

        diag = np.diag(k_mat)
        off = k_mat[~np.eye(n, dtype=bool)]
        logger.info(
            f'  chr{chrom}: diag=[{diag.min():.4f}, {diag.max():.4f}], '
            f'off-diag=[{off.min():.4f}, {off.max():.4f}]'
        )

    # founder self-kinship should exceed cc self-kinship
    k_all = kin_module.calc_kinship(genome_cache, method='overall')['overall']
    assert np.diag(k_all)[:8].mean() > np.diag(k_all)[15:].mean(), \
        'founder self-kinship should be > cc self-kinship'

    logger.info('kinship validation passed ✓')
    return True


def validate_k_expansion(genome_cache, pheno_df):
    """validate 48→384 expansion matches r: k_chr[strain_idx, strain_idx]."""
    logger.info('=== validating k expansion (48 → n_samples) ===')

    k_loco = kin_module.calc_kinship(genome_cache, method='loco')
    k48 = k_loco['1']

    strain_to_idx = {s: i for i, s in enumerate(genome_cache.sample_ids)}
    idx = np.array([strain_to_idx[s] for s in pheno_df['subject.name']])
    k_exp = k48[np.ix_(idx, idx)]

    n = len(pheno_df)
    assert k_exp.shape == (n, n), f'wrong shape: {k_exp.shape}'
    assert np.allclose(k_exp, k_exp.T), 'not symmetric after expansion'

    # same-strain replicates must have identical rows
    cc001_idx = pheno_df[pheno_df['subject.name'] == 'CC001'].index[:2].tolist()
    if len(cc001_idx) == 2:
        assert np.allclose(k_exp[cc001_idx[0]], k_exp[cc001_idx[1]]), \
            'replicate rows differ'

    logger.info(f'  expanded: {n}x{n}, symmetric, replicate rows match')
    logger.info('k expansion validation passed ✓')
    return True


def validate_scan(genome_cache, pheno_df, true_params):
    """validate scan localizes causal qtl.

    path a: collapsed to 48 strain means (scan_eqtl.R approach —
            no sex covariate, 48x48 K, average rint per strain)
    """
    logger.info('=== validating genome scan (path a: 48 strain means) ===')

    gene = true_params['causal_gene']
    causal_chrom = true_params['causal_chrom']
    pc_cols = [f'pc{i}' for i in range(1, 11)]

    # collapse to 48 strain means (matching scan_eqtl.R aggregate() call)
    avg_cols = true_params['gene_names'] + pc_cols
    pheno_48 = pheno_df.groupby('subject.name')[avg_cols].mean()
    pheno_48 = pheno_48.loc[ALL_48_STRAINS]  # enforce genome cache order

    y_48 = pheno_48[gene].values
    x_48 = np.column_stack([np.ones(48), pheno_48[pc_cols].values])

    scan_result = scan_module.scan_h2lmm(
        y_48, genome_cache,
        covariates=x_48,
        use_rint=False,  # already rint'd
        phenotype_id=f'{gene}_path_a',
    )

    peak = scan_result.peak_locus()
    logger.info(
        f'  peak: chr{peak["chromosome"]} pos={peak["position"]:.1f} mb '
        f'lod={peak["lod"]:.2f} p={peak["p_value"]:.2e}'
    )
    logger.info(
        f'  true: chr{causal_chrom} locus_idx={true_params["causal_locus_idx"]} '
        f'founder={true_params["causal_founder_name"]}'
    )

    assert peak['chromosome'] == causal_chrom, \
        f'peak on chr{peak["chromosome"]}, expected chr{causal_chrom}'
    assert peak['lod'] > 2.0, f'peak lod too low: {peak["lod"]:.2f}'

    logger.info('scan validation passed ✓')
    return scan_result  # return result for qc plotting (truthy = passed)


def validate_dispersion(genome_cache, counts, subject_ids, lib_sizes):
    """validate per-subject φ̂ from cell-level counts."""
    logger.info('=== validating dispersion estimation ===')

    subjects, phi_hat = vqtl_module.estimate_dispersion_per_subject(
        counts, subject_ids, lib_sizes, method='moments', min_cells=20
    )

    valid = ~np.isnan(phi_hat)
    assert valid.sum() > 30, f'too few valid: {valid.sum()}'
    assert np.all(phi_hat[valid] > 0), 'negative φ values'

    logger.info(
        f'  {valid.sum()}/{len(phi_hat)} subjects, '
        f'median φ={np.nanmedian(phi_hat[valid]):.4f}'
    )
    logger.info('dispersion estimation validation passed ✓')
    return subjects, phi_hat  # return for qc plotting (truthy = passed)


def validate_vqtl(genome_cache, counts, subject_ids, lib_sizes, sc_params):
    """validate vqtl scan detects dispersion effect."""
    logger.info('=== validating vqtl scan ===')

    result = vqtl_module.scan_vqtl(
        counts, subject_ids, genome_cache,
        library_sizes=lib_sizes,
        phi_method='moments', min_cells=20,
        gene_id='test_gene', also_scan_mean=True
    )

    if result is None:
        logger.warning('vqtl scan returned none')
        return False

    vp = result.scan_result.peak_locus()
    mp = result.mean_scan_result.peak_locus()
    logger.info(f'  vqtl peak: chr{vp["chromosome"]} lod={vp["lod"]:.2f}')
    logger.info(f'  mean peak: chr{mp["chromosome"]} lod={mp["lod"]:.2f}')

    comp = vqtl_module.compare_mean_vs_dispersion(result)
    logger.info(f'  category: {comp["category"]}, colocalized: {comp["colocalized"]}')

    assert vp['lod'] > 1.5, f'vqtl lod too low: {vp["lod"]:.2f}'

    logger.info('vqtl scan validation passed ✓')
    return True


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

def run_all_validations():
    logger.info('generating simulated cc data (48 strains x 8 reps)...')

    gc = simulate_cc_haplotypes(n_loci_per_chr=50, n_chromosomes=3, seed=42)
    logger.info(
        f'genome cache: {len(gc.sample_ids)} strains, '
        f'{len(gc.chromosomes)} chr, {gc.n_loci("1")} loci/chr'
    )

    pheno_df, true_params = simulate_replicated_phenotype(
        gc, causal_chrom='1', causal_locus_idx=25,
        causal_founder=3, effect_size=3.0, n_genes=5, seed=42
    )
    logger.info(
        f'pheno_df: {len(pheno_df)} samples, causal={true_params["causal_gene"]} '
        f'chr{true_params["causal_chrom"]} ({true_params["causal_founder_name"]})'
    )

    counts, sids, libs, sc_params = simulate_sc_counts(
        gc, pheno_df, causal_chrom='1', causal_locus_idx=25,
        causal_founder=3, seed=42
    )
    logger.info(f'sc data: {len(counts)} cells, {len(np.unique(sids))} subjects')

    export_for_r(gc, pheno_df, 'validation/r_data')

    # run validations (some return objects for qc plotting)
    results = {}
    results['kinship'] = validate_kinship(gc)
    results['k_expansion'] = validate_k_expansion(gc, pheno_df)

    scan_result = validate_scan(gc, pheno_df, true_params)
    results['scan'] = bool(scan_result)

    disp_result = validate_dispersion(gc, counts, sids, libs)
    results['dispersion'] = bool(disp_result)

    results['vqtl'] = validate_vqtl(gc, counts, sids, libs, sc_params)

    # generate qc plots
    logger.info('\n=== generating qc plots ===')
    try:
        from mosaiqtl.plotting import plot_validation_panel
        from mosaiqtl import kinship as kin_module

        k_loco = kin_module.calc_kinship(gc, method='loco')
        k_chr1 = k_loco['1']

        # get true phi from simulation for dispersion scatter
        phi_true = None
        phi_estimated = None
        if disp_result:
            subjects_est, phi_estimated = disp_result
            # reconstruct true phi from sc_params
            h_causal = gc.get_locus(
                sc_params['causal_chrom'], sc_params['causal_locus_idx']
            )
            base_phi = 0.5
            disp_effect = 0.8
            phi_true_48 = base_phi * np.exp(
                disp_effect * h_causal[:, sc_params['causal_founder']]
            )
            # align to estimated subjects
            from mosaiqtl.utils import align_samples
            idx_est, idx_gc, _ = align_samples(subjects_est, gc.sample_ids)
            phi_true = phi_true_48[idx_gc]
            phi_estimated = phi_estimated[idx_est]

        plot_validation_panel(
            scan_result=scan_result,
            k_matrix=k_chr1,
            genome_cache=gc,
            true_params=true_params,
            phi_estimated=phi_estimated,
            phi_true=phi_true,
            save_dir='validation/qc_plots',
        )
    except Exception as e:
        logger.warning(f'qc plot generation failed: {e}')

    # summary
    logger.info('\n=== validation summary ===')
    all_ok = True
    for mod, ok in results.items():
        logger.info(f'  {mod}: {"✓ passed" if ok else "✗ failed"}')
        if not ok:
            all_ok = False
    return all_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='all')
    args = parser.parse_args()
    success = run_all_validations()
    sys.exit(0 if success else 1)

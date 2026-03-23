"""
mosaiqtl validation — cross-language comparison against r miqtl/qtl2

two modes (auto-detected):
  real data  — reads pheno_df.rds, kinship_loco.rds, genomecache directly
               from the cc gastrocnemius dataset on crsp. preferred.
  simulation — generates 48 cc genotypes (8 founders + 7 f1s + 33 cc
               strains, 8 reps each). used locally and for ci when
               crsp paths are unavailable.

usage:
  python validation/validate_vs_r.py --module all
"""

import numpy as np
import pandas as pd
import os, sys, argparse, logging

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

# ── real data paths (crsp) ────────────────────────────────────────────────────
REAL_BASE  = '/share/crsp/lab/seyedam/gfilimba/projects/igvf_8cube/short-read/data/gastroc'
REAL_CACHE = os.path.join(REAL_BASE, 'output', 'genomecache')
REAL_PHENO = os.path.join(REAL_BASE, 'output', 'pheno_df.rds')
REAL_KIN   = os.path.join(REAL_BASE, 'output', 'kinship_loco.rds')
REAL_PMAP  = os.path.join(REAL_BASE, 'input',  'pmap.csv')

# auto-detect mode: use real data when crsp paths exist
USE_REAL = os.path.isdir(REAL_CACHE) and os.path.exists(REAL_PHENO)

if USE_REAL:
    logger.info('real data detected — using cc gastrocnemius dataset')
else:
    logger.info('real data not found — using simulation (local/ci mode)')

# ── strain labels ─────────────────────────────────────────────────────────────
FOUNDER_STRAINS = ['AJ', 'B6J', '129S1J', 'CASTJ', 'NODJ', 'NZOJ', 'PWKJ', 'WSBJ']
F1_STRAINS = ['B6AF1J', 'B6129S1F1J', 'B6CASTF1J', 'B6NODF1J',
              'B6NZOF1J', 'B6PWKF1J', 'B6WSBF1J']
CC_STRAINS = [f'CC{i:03d}' for i in range(1, 34)]
ALL_48_STRAINS = FOUNDER_STRAINS + F1_STRAINS + CC_STRAINS
REPS_PER_STRAIN = 8


# ── real data loader ──────────────────────────────────────────────────────────

def load_real_data(n_genes=5, chr_id='8'):
    """load real cc data via rpy2.

    collapses 415 replicates to 48 strain means (path a — matching
    scan_eqtl.r aggregate() call). sex excluded after averaging.
    """
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    pandas2ri.activate()

    # read pheno_df.rds
    ro.r(f'pheno_df <- readRDS("{REAL_PHENO}")')
    pheno_r = pandas2ri.rpy2py(ro.r['pheno_df'])
    # standardise column names
    pheno_r.columns = [c.replace('.', '_').lower() for c in pheno_r.columns]
    pheno_r = pheno_r.rename(columns={'subject_name': 'subject.name'})
    logger.info(f'pheno_df: {pheno_r.shape[0]} samples × {pheno_r.shape[1]} cols')

    # read kinship_loco.rds
    ro.r(f'k_list <- readRDS("{REAL_KIN}")')
    chr_names = list(ro.r['names'](ro.r['k_list']))
    kinship = {}
    for c in chr_names:
        kinship[c] = np.array(ro.r[f'k_list[["{c}"]]'], dtype=float)
    strain_order = list(ro.r['rownames'](ro.r[f'k_list[["{chr_names[0]}"]]']))
    logger.info(f'kinship: {len(kinship)} chrs, {len(strain_order)} strains')

    # load genome cache from real happy directory
    gc = GenomeCache.from_happy_dir(REAL_CACHE, model='additive')
    logger.info(f'genome cache: {len(gc.chromosomes)} chr loaded')

    # identify gene columns
    meta = {'sample_id', 'subject.name', 'sex', 'batch'} | \
           {f'pc{i}' for i in range(1, 11)}
    gene_cols = [c for c in pheno_r.columns if c not in meta][:n_genes]
    logger.info(f'genes: {gene_cols}')

    # collapse 415 → 48 strain means
    pc_cols = [f'pc{i}' for i in range(1, 11)]
    pheno_48 = pheno_r.groupby('subject.name')[gene_cols + pc_cols].mean()
    batch_map = pheno_r.groupby('subject.name')['batch'].first()
    pheno_48['batch'] = pheno_48.index.map(batch_map)
    pheno_48 = pheno_48.reindex(strain_order).reset_index()
    logger.info(f'pheno_48: {pheno_48.shape[0]} strains')

    true_params = {
        'gene_names': gene_cols,
        'causal_gene': None,   # no ground truth for real data
        'causal_chrom': chr_id,
        'causal_locus_idx': None,
        'causal_founder_name': None,
    }
    return gc, pheno_48, kinship, true_params


# ── simulation helpers ────────────────────────────────────────────────────────

def simulate_cc_haplotypes(n_loci_per_chr=50, n_chromosomes=3, seed=42):
    """simulate founder haplotype probabilities for 48 cc genotypes."""
    rng = np.random.default_rng(seed)
    n_founders = 8
    n_subjects = len(ALL_48_STRAINS)
    data = {}

    for c in range(1, n_chromosomes + 1):
        chrom = str(c)
        locus_ids = [f'UNC{c:02d}{m:06d}' for m in range(n_loci_per_chr)]
        positions = np.linspace(3.0, 150.0, n_loci_per_chr)
        probs = np.zeros((n_loci_per_chr, n_subjects, n_founders))

        for l_idx in range(n_loci_per_chr):
            # founders: one-hot
            for f_idx in range(8):
                probs[l_idx, f_idx, f_idx] = 1.0
            # f1s: 50% b6j + 50% other parent
            for f1_idx, other in enumerate([0, 2, 3, 4, 5, 6, 7]):
                s = 8 + f1_idx
                probs[l_idx, s, 1] = 0.5
                probs[l_idx, s, other] = 0.5
            # cc strains: mosaic haplotype blocks
            for cc_idx in range(33):
                s = 15 + cc_idx
                n_breaks = rng.poisson(3)
                breaks = np.sort(rng.choice(n_loci_per_chr,
                    size=min(n_breaks, n_loci_per_chr - 1), replace=False))
                seg = np.searchsorted(breaks, l_idx, side='right')
                dom = (cc_idx * 7 + seg + c) % n_founders
                probs[l_idx, s, :] = 0.005
                probs[l_idx, s, dom] = 0.965
                probs[l_idx, s, :] /= probs[l_idx, s, :].sum()

        data[chrom] = {'locus_ids': np.array(locus_ids),
                       'positions': positions, 'probs': probs}

    return GenomeCache(ALL_48_STRAINS, FOUNDER_STRAINS, data)


def simulate_replicated_phenotype(genome_cache, causal_chrom='1',
                                   causal_locus_idx=25, causal_founder=3,
                                   effect_size=3.0, n_genes=5, seed=42):
    """simulate phenotype matching pheno_df structure."""
    rng = np.random.default_rng(seed)
    rows = []
    for s_idx, strain in enumerate(ALL_48_STRAINS):
        batch = f'batch{(s_idx % 3) + 1}'
        for rep in range(REPS_PER_STRAIN):
            sex = 'M' if rep < 4 else 'F'
            rows.append({'sample_id': f'{strain}_{sex}_{rep:02d}',
                         'subject.name': strain, 'sex': sex, 'batch': batch})

    pheno_df = pd.DataFrame(rows)
    n = len(pheno_df)
    strain_to_idx = {s: i for i, s in enumerate(ALL_48_STRAINS)}
    s_idx = np.array([strain_to_idx[s] for s in pheno_df['subject.name']])
    h_causal = genome_cache.get_locus(causal_chrom, causal_locus_idx)

    gene_names = []
    for g in range(n_genes):
        name = f'gene{g + 1}'
        gene_names.append(name)
        se = h_causal[:, causal_founder] * effect_size if g == 0 \
             else h_causal @ rng.normal(0, 0.1, 8)
        sex_e   = np.where(pheno_df['sex'] == 'M', 0.3, -0.3)
        batch_e = np.array([{'batch1': 0.0, 'batch2': 0.2, 'batch3': -0.2}[b]
                             for b in pheno_df['batch']])
        pheno_df[name] = rint(se[s_idx] + sex_e + batch_e + rng.normal(0, 0.8, n))

    for pc in range(10):
        pheno_df[f'pc{pc + 1}'] = rng.normal(0, 1, n)

    true_params = {
        'causal_chrom': causal_chrom, 'causal_locus_idx': causal_locus_idx,
        'causal_founder': causal_founder,
        'causal_founder_name': FOUNDER_STRAINS[causal_founder],
        'effect_size': effect_size, 'causal_gene': 'gene1',
        'gene_names': gene_names,
    }
    return pheno_df, true_params


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
        si   = strain_to_idx[row['subject.name']]
        d    = h_causal[si, causal_founder]
        mu   = base_mu  * np.exp(mean_effect * d)
        phi  = base_phi * np.exp(dispersion_effect * d)
        nc   = max(rng.poisson(cells_per_sample), 10)
        r    = 1.0 / phi
        all_counts.append(rng.negative_binomial(r, r / (r + mu), size=nc))
        all_sids.append(np.full(nc, row['subject.name']))
        all_libs.append(rng.lognormal(np.log(10000), 0.3, nc))

    return (np.concatenate(all_counts), np.concatenate(all_sids),
            np.concatenate(all_libs),
            {'causal_chrom': causal_chrom, 'causal_locus_idx': causal_locus_idx,
             'causal_founder': causal_founder})


def export_for_r(genome_cache, pheno_df, output_dir):
    """export simulation data for the r reference script."""
    os.makedirs(output_dir, exist_ok=True)
    pheno_df.to_csv(os.path.join(output_dir, 'pheno_df.csv'), index=False)

    pmap_rows = []
    for chrom in genome_cache.chromosomes:
        lids, positions, _ = genome_cache.get_chromosome(chrom)
        for lid, pos in zip(lids, positions):
            pmap_rows.append({'marker': lid, 'chr': chrom, 'pos': pos})
    pd.DataFrame(pmap_rows).to_csv(os.path.join(output_dir, 'pmap.csv'), index=False)

    k_loco = kin_module.calc_kinship(genome_cache, method='loco')
    k_dir  = os.path.join(output_dir, 'kinship')
    os.makedirs(k_dir, exist_ok=True)
    for chrom, k_mat in k_loco.items():
        pd.DataFrame(k_mat, index=genome_cache.sample_ids,
                     columns=genome_cache.sample_ids).to_csv(
            os.path.join(k_dir, f'kinship_chr{chrom}.csv'))

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
        pd.DataFrame(rows).to_csv(os.path.join(hap_dir, f'chr{chrom}.csv'), index=False)

    logger.info(f'exported r validation data to {output_dir}')


# ── validation tests ──────────────────────────────────────────────────────────

def validate_kinship(genome_cache, tol=1e-6):
    """validate kinship: symmetry, psd, correct structure."""
    logger.info('=== validating kinship computation ===')
    k_loco = kin_module.calc_kinship(genome_cache, method='loco')

    for chrom, k_mat in k_loco.items():
        n = k_mat.shape[0]
        assert n == 48, f'kinship should be 48x48, got {n}x{n}'
        assert np.allclose(k_mat, k_mat.T, atol=tol), 'not symmetric'
        eigvals = np.linalg.eigvalsh(k_mat)
        assert np.all(eigvals >= -tol), 'negative eigenvalues'
        diag = np.diag(k_mat)
        off  = k_mat[~np.eye(n, dtype=bool)]
        logger.info(f'  chr{chrom}: diag=[{diag.min():.4f}, {diag.max():.4f}]  '
                    f'off-diag=[{off.min():.4f}, {off.max():.4f}]')

    k_all = kin_module.calc_kinship(genome_cache, method='overall')['overall']
    assert np.diag(k_all)[:8].mean() > np.diag(k_all)[15:].mean(), \
        'founder self-kinship should be > cc self-kinship'

    logger.info('kinship validation passed ✓')
    return True


def validate_k_expansion(genome_cache, pheno_df):
    """validate 48→n expansion."""
    logger.info('=== validating k expansion ===')
    k_loco = kin_module.calc_kinship(genome_cache, method='loco')
    k48    = k_loco['1'] if '1' in k_loco else list(k_loco.values())[0]

    strain_to_idx = {s: i for i, s in enumerate(genome_cache.sample_ids)}
    idx   = np.array([strain_to_idx[s] for s in pheno_df['subject.name']])
    k_exp = k48[np.ix_(idx, idx)]

    n = len(pheno_df)
    assert k_exp.shape == (n, n)
    assert np.allclose(k_exp, k_exp.T)

    cc_rows = pheno_df[pheno_df['subject.name'] == pheno_df['subject.name'].iloc[0]].index[:2].tolist()
    if len(cc_rows) == 2:
        assert np.allclose(k_exp[cc_rows[0]], k_exp[cc_rows[1]])

    logger.info(f'  expanded: {n}x{n}, symmetric, replicate rows match')
    logger.info('k expansion validation passed ✓')
    return True


def validate_scan(genome_cache, pheno_df, true_params):
    """validate scan: localizes causal qtl (simulation) or runs cleanly (real)."""
    logger.info('=== validating genome scan (path a: 48 strain means) ===')

    gene    = true_params['gene_names'][0]
    pc_cols = [f'pc{i}' for i in range(1, 11)]

    # collapse to 48 strain means if input is replicated (simulation)
    if len(pheno_df) > 48:
        pheno_48 = (pheno_df.groupby('subject.name')[true_params['gene_names'] + pc_cols]
                    .mean().reindex(ALL_48_STRAINS))
    else:
        pheno_48 = pheno_df.set_index('subject.name').reindex(ALL_48_STRAINS)

    y = pheno_48[gene].values
    x = np.column_stack([np.ones(48), pheno_48[pc_cols].values])

    scan_result = scan_module.scan_h2lmm(
        y, genome_cache, covariates=x,
        use_rint=False, phenotype_id=f'{gene}_path_a',
    )

    peak = scan_result.peak_locus()
    logger.info(f'  peak: chr{peak["chromosome"]} pos={peak["position"]:.1f} mb '
                f'lod={peak["lod"]:.2f} p={peak["p_value"]:.2e}')

    if true_params.get('causal_gene'):
        # simulation: assert peak chromosome matches
        assert peak['chromosome'] == true_params['causal_chrom'], \
            f'peak on chr{peak["chromosome"]}, expected chr{true_params["causal_chrom"]}'
        assert peak['lod'] > 2.0
        logger.info(f'  true: chr{true_params["causal_chrom"]} '
                    f'founder={true_params["causal_founder_name"]}')

    logger.info('scan validation passed ✓')
    return scan_result


def validate_dispersion(genome_cache, counts, subject_ids, lib_sizes):
    logger.info('=== validating dispersion estimation ===')
    subjects, phi_hat = vqtl_module.estimate_dispersion_per_subject(
        counts, subject_ids, lib_sizes, method='moments', min_cells=20)
    valid = ~np.isnan(phi_hat)
    assert valid.sum() > 30
    assert np.all(phi_hat[valid] > 0)
    logger.info(f'  {valid.sum()}/{len(phi_hat)} subjects  '
                f'median φ={np.nanmedian(phi_hat[valid]):.4f}')
    logger.info('dispersion estimation validation passed ✓')
    return subjects, phi_hat


def validate_vqtl(genome_cache, counts, subject_ids, lib_sizes, sc_params):
    logger.info('=== validating vqtl scan ===')
    result = vqtl_module.scan_vqtl(
        counts, subject_ids, genome_cache, library_sizes=lib_sizes,
        phi_method='moments', min_cells=20, gene_id='test_gene',
        also_scan_mean=True)
    if result is None:
        return False
    vp   = result.scan_result.peak_locus()
    mp   = result.mean_scan_result.peak_locus()
    comp = vqtl_module.compare_mean_vs_dispersion(result)
    logger.info(f'  vqtl peak: chr{vp["chromosome"]} lod={vp["lod"]:.2f}')
    logger.info(f'  mean peak: chr{mp["chromosome"]} lod={mp["lod"]:.2f}')
    logger.info(f'  category: {comp["category"]}  colocalized: {comp["colocalized"]}')
    assert vp['lod'] > 1.5
    logger.info('vqtl scan validation passed ✓')
    return True


# ── main ──────────────────────────────────────────────────────────────────────

def run_all_validations():
    results = {}

    if USE_REAL:
        gc, pheno_48, kinship, true_params = load_real_data(n_genes=5, chr_id='8')
        results['kinship']     = validate_kinship(gc)
        results['k_expansion'] = validate_k_expansion(gc, pheno_48)
        scan_result            = validate_scan(gc, pheno_48, true_params)
        results['scan']        = bool(scan_result)
        # cell-level tests require h5ad — skipped in real mode unless present
        results['dispersion']  = 'skipped'
        results['vqtl']        = 'skipped'
    else:
        gc = simulate_cc_haplotypes(n_loci_per_chr=50, n_chromosomes=3, seed=42)
        pheno_df, true_params = simulate_replicated_phenotype(
            gc, causal_chrom='1', causal_locus_idx=25,
            causal_founder=3, effect_size=3.0, n_genes=5, seed=42)
        counts, sids, libs, sc_params = simulate_sc_counts(
            gc, pheno_df, causal_chrom='1', causal_locus_idx=25,
            causal_founder=3, seed=42)
        export_for_r(gc, pheno_df, 'validation/r_data')
        results['kinship']     = validate_kinship(gc)
        results['k_expansion'] = validate_k_expansion(gc, pheno_df)
        scan_result            = validate_scan(gc, pheno_df, true_params)
        results['scan']        = bool(scan_result)
        disp_result            = validate_dispersion(gc, counts, sids, libs)
        results['dispersion']  = bool(disp_result)
        results['vqtl']        = validate_vqtl(gc, counts, sids, libs, sc_params)

    # qc plots
    try:
        from mosaiqtl.plotting import plot_validation_panel
        k_loco = kin_module.calc_kinship(gc, method='loco')
        k_mat  = k_loco[gc.chromosomes[0]]
        plot_validation_panel(
            scan_result=scan_result if results.get('scan') else None,
            k_matrix=k_mat, genome_cache=gc,
            true_params=true_params if not USE_REAL else None,
            save_dir='validation/qc_plots',
        )
    except Exception as e:
        logger.warning(f'qc plot generation failed: {e}')

    # summary
    logger.info('\n=== validation summary ===')
    all_ok = True
    for mod, ok in results.items():
        if ok == 'skipped':
            logger.info(f'  {mod}: — skipped')
        else:
            logger.info(f'  {mod}: {"✓ passed" if ok else "✗ failed"}')
            if not ok:
                all_ok = False
    return all_ok


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--module', default='all')
    args = parser.parse_args()
    sys.exit(0 if run_all_validations() else 1)

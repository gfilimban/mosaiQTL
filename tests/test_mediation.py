# tests/test_mediation.py — unit tests for mosaiqtl.qtl.mediation
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from mosaiqtl.genotype.genomecache import GenomeCache
from mosaiqtl.genotype.kinship import calc_kinship, decomp_kinship
from mosaiqtl.qtl.mediation import fixef_mediation_scan, mediation_scan_qr

_REQUIRED_COLS = {'locus', 'chr', 'pos_mb', 'lod_base', 'lod_mediated',
                  'lod_drop', 'p_value', 'outcome_id', 'mediator_id'}


def _make_cache(n_samples=24, n_loci=10, n_founders=8, chroms=('1', '2')):
    rng  = np.random.default_rng(11)
    data = {}
    for c in chroms:
        probs = rng.dirichlet(np.ones(n_founders), size=(n_loci, n_samples))
        data[c] = {
            'locus_ids': np.array([f'{c}_L{i:02d}' for i in range(n_loci)]),
            'positions': np.linspace(1.0, 100.0, n_loci),
            'probs':     probs,
        }
    return GenomeCache(data=data,
                       sample_ids=[f's{i}' for i in range(n_samples)],
                       founders=[f'F{i}' for i in range(n_founders)])


def _planted(cache, chrom, locus_idx, noise=0.1, seed=42):
    """phenotype correlated with one locus in the cache."""
    _, _, probs = cache.get_chromosome(chrom)
    rng = np.random.default_rng(seed)
    return probs[locus_idx, :, 0] + noise * rng.standard_normal(probs.shape[1])


class TestFixefMediationScan:
    """OLS mediation scan — no kinship, fast, vectorised."""

    def setup_method(self):
        self.cache = _make_cache()

    def test_output_columns(self):
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n)
        m = rng.standard_normal(n)
        df = fixef_mediation_scan(y, m, self.cache)
        assert _REQUIRED_COLS.issubset(df.columns)

    def test_output_row_count(self):
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n)
        m = rng.standard_normal(n)
        df = fixef_mediation_scan(y, m, self.cache)
        total_loci = sum(self.cache.n_loci(c) for c in self.cache.chromosomes)
        assert len(df) == total_loci

    def test_lod_drop_nonneg(self):
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(1)
        y = rng.standard_normal(n)
        m = rng.standard_normal(n)
        df = fixef_mediation_scan(y, m, self.cache)
        assert (df['lod_drop'] >= 0).all()

    def test_true_mediator_positive_lod_drop(self):
        """when m is the true mediator (y ← m ← H), lod_drop > 0 at signal locus."""
        chrom, j = '1', 4
        m = _planted(self.cache, chrom, j, noise=0.05, seed=10)
        rng = np.random.default_rng(20)
        y = 3.0 * m + 0.2 * rng.standard_normal(len(m))
        df = fixef_mediation_scan(y, m, self.cache,
                                   outcome_id='y', mediator_id='m')
        signal_row = df[(df['chr'] == chrom) &
                        (df['locus'] == f'{chrom}_L{j:02d}')].iloc[0]
        assert signal_row['lod_drop'] > 0.5

    def test_unrelated_mediator_low_lod_drop(self):
        """pure noise mediator should not inflate lod_drop at signal locus."""
        chrom, j = '1', 4
        y = _planted(self.cache, chrom, j, noise=0.05, seed=30)
        rng = np.random.default_rng(31)
        m_noise = rng.standard_normal(len(y))
        df = fixef_mediation_scan(y, m_noise, self.cache)
        signal_row = df[(df['chr'] == chrom) &
                        (df['locus'] == f'{chrom}_L{j:02d}')].iloc[0]
        assert signal_row['lod_drop'] < 1.0

    def test_lod_base_positive_with_signal(self):
        """lod_base at signal locus must be clearly positive."""
        chrom, j = '1', 4
        y = _planted(self.cache, chrom, j, noise=0.1, seed=40)
        rng = np.random.default_rng(0)
        m = rng.standard_normal(len(y))
        df = fixef_mediation_scan(y, m, self.cache)
        peak_lod = df[df['chr'] == chrom]['lod_base'].max()
        assert peak_lod > 1.0

    def test_just_these_loci_filter(self):
        """just_these_loci restricts output to requested loci only."""
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n)
        m = rng.standard_normal(n)
        keep = {'1_L03', '1_L07'}
        df = fixef_mediation_scan(y, m, self.cache, just_these_loci=keep)
        assert set(df['locus'].unique()).issubset(keep)

    def test_outcome_mediator_ids_propagated(self):
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(0)
        df = fixef_mediation_scan(rng.standard_normal(n), rng.standard_normal(n),
                                   self.cache, outcome_id='geneA', mediator_id='geneB')
        assert (df['outcome_id'] == 'geneA').all()
        assert (df['mediator_id'] == 'geneB').all()


class TestMediationScanQr:
    """LMM-based mediation scan with LOCO kinship."""

    def setup_method(self):
        self.cache = _make_cache()
        kd = calc_kinship(self.cache, method='loco')
        self.kd_decomp = decomp_kinship(kd)

    def test_output_columns(self):
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(0)
        y = rng.standard_normal(n)
        m = rng.standard_normal(n)
        df = mediation_scan_qr(y, m, self.cache,
                                kinship_decomp=self.kd_decomp)
        assert _REQUIRED_COLS.issubset(df.columns)

    def test_lod_drop_nonneg(self):
        n = len(self.cache.sample_ids)
        rng = np.random.default_rng(1)
        y = rng.standard_normal(n)
        m = rng.standard_normal(n)
        df = mediation_scan_qr(y, m, self.cache,
                                kinship_decomp=self.kd_decomp)
        assert (df['lod_drop'] >= 0).all()

    def test_true_mediator_positive_lod_drop(self):
        """LMM mediation detects true mediator path."""
        chrom, j = '1', 4
        m = _planted(self.cache, chrom, j, noise=0.05, seed=50)
        rng = np.random.default_rng(51)
        y = 3.0 * m + 0.2 * rng.standard_normal(len(m))
        df = mediation_scan_qr(y, m, self.cache,
                                kinship_decomp=self.kd_decomp,
                                use_rint=False)
        signal_row = df[(df['chr'] == chrom) &
                        (df['locus'] == f'{chrom}_L{j:02d}')].iloc[0]
        assert signal_row['lod_drop'] > 0.5

    def test_lod_drop_consistent_with_fixef(self):
        """LMM and OLS lod_drop should be positively correlated at signal locus."""
        chrom, j = '1', 3
        m = _planted(self.cache, chrom, j, noise=0.05, seed=60)
        rng = np.random.default_rng(61)
        y = 2.0 * m + 0.3 * rng.standard_normal(len(m))
        df_lmm = mediation_scan_qr(y, m, self.cache,
                                    kinship_decomp=self.kd_decomp,
                                    use_rint=False)
        df_ols = fixef_mediation_scan(y, m, self.cache)
        lmm_peak = df_lmm[df_lmm['chr'] == chrom]['lod_drop'].max()
        ols_peak = df_ols[df_ols['chr'] == chrom]['lod_drop'].max()
        assert lmm_peak > 0 and ols_peak > 0

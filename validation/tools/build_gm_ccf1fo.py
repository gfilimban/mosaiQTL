"""Rebuild GM_ccf1fo_mm39.csv from Layer-0 public sources only.

Sources (all linked, cited in docs/DATA_SOURCES.md):
  gm_uwisc_v4.csv               - marker positions on GRCm39 + QC flags
  GigaMUGA_genotypes.csv.gz     - raw per-sample GigaMUGA calls (all reference samples)
  GigaMUGA_founder_metadata.csv - sample -> strain/letter mapping
  CC_consensus_genotypes.csv    - CC strain consensus calls

Explicitly EXCLUDED: old/GM/foundergeno/GM_foundergeno{chr}.csv (uncited, no
provenance, dated 2023-04-20, labeled "Legacy... superseded" in DATA_SOURCES.md).
Founder genotype cells that cannot be resolved from GigaMUGA_genotypes.csv.gz
consensus are left as 'N' rather than filled from that file.
"""
import sys
from collections import Counter

import pandas as pd

IN_DIR = '/Users/ghassan/Documents/journey/learning/graduate/lab/projects/igvf/analysis/qtl/input'

FOUNDER_LETTERS = {
    'AA': 'AJ', 'BB': 'B6J', 'CC': '129S1J', 'DD': 'NODJ',
    'EE': 'NZOJ', 'FF': 'CASTJ', 'GG': 'PWKJ', 'HH': 'WSBJ',
}
FOUNDER_ORDER = ['AJ', 'B6J', '129S1J', 'NODJ', 'NZOJ', 'CASTJ', 'PWKJ', 'WSBJ']
F1_ORDER = ['B6AF1J', 'B6129S1F1J', 'B6NODF1J', 'B6NZOF1J', 'B6CASTF1J', 'B6PWKF1J', 'B6WSBF1J']
F1_OTHER = {
    'B6AF1J': 'AJ', 'B6129S1F1J': '129S1J', 'B6NODF1J': 'NODJ', 'B6NZOF1J': 'NZOJ',
    'B6CASTF1J': 'CASTJ', 'B6PWKF1J': 'PWKJ', 'B6WSBF1J': 'WSBJ',
}
CC_STRAINS = [
    'CC001', 'CC002', 'CC003', 'CC004', 'CC005', 'CC006', 'CC007', 'CC008', 'CC009',
    'CC010', 'CC011', 'CC012', 'CC013', 'CC015', 'CC017', 'CC018', 'CC024', 'CC025',
    'CC028', 'CC029', 'CC030', 'CC032', 'CC036', 'CC037', 'CC038', 'CC041', 'CC043',
    'CC057', 'CC060', 'CC062', 'CC065', 'CC071', 'CC074',
]


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main() -> None:
    log('1/6 loading gm_uwisc_v4.csv (marker positions + QC flags)...')
    gm = pd.read_csv(f'{IN_DIR}/gm_uwisc_v4.csv', dtype=str)
    valid_chr = [str(i) for i in range(1, 20)] + ['X']
    unique_b = gm['unique'].str.upper() == 'TRUE'
    unmapped_b = gm['unmapped'].str.upper() == 'TRUE'
    gm_f = gm[unique_b & (~unmapped_b) & gm['bp_grcm39'].notna()
              & (gm['bp_grcm39'] != '') & gm['chr'].isin(valid_chr)].copy()
    gm_f = gm_f.set_index('marker')[['chr', 'bp_grcm39']]
    gm_f.columns = ['chr', 'pos']
    log(f'   {len(gm_f)} markers pass gm_uwisc_v4 filter (unique, mapped, autosome/X)')

    log('2/6 loading GigaMUGA_founder_metadata.csv (sample -> letter)...')
    meta = pd.read_csv(f'{IN_DIR}/GigaMUGA_founder_metadata.csv')
    letter_samples = {}
    for letter in FOUNDER_LETTERS:
        letter_samples[letter] = meta.loc[meta['letter'] == letter, 'sample'].tolist()
        log(f'   {letter} ({FOUNDER_LETTERS[letter]}): {len(letter_samples[letter])} replicate samples')

    log('3/6 loading GigaMUGA_genotypes.csv.gz (raw per-sample calls) — this takes a moment...')
    geno = pd.read_csv(f'{IN_DIR}/GigaMUGA_genotypes.csv.gz', dtype=str)
    geno = geno.set_index('marker')
    geno = geno[geno.index.isin(gm_f.index)]
    log(f'   raw genotype matrix restricted to gm_uwisc markers: {geno.shape}')

    log('4/6 computing founder consensus (majority vote across replicate samples, N on no-consensus)...')

    def consensus_col(samples):
        cols = [s for s in samples if s in geno.columns]
        missing = set(samples) - set(cols)
        if missing:
            log(f'   WARNING: samples not found in genotype matrix: {missing}')
        sub = geno[cols]

        def _vote(row):
            vals = [v for v in row if pd.notna(v) and v not in ('N', 'H')]
            if not vals:
                return 'N'
            c = Counter(vals)
            top, n = c.most_common(1)[0]
            # require unanimity among informative (non-N, non-H) calls
            if n == len(vals):
                return top
            return 'N'

        return sub.apply(_vote, axis=1)

    founder_cols = {}
    for letter, fname in FOUNDER_LETTERS.items():
        founder_cols[fname] = consensus_col(letter_samples[letter])
        n_resolved = (founder_cols[fname] != 'N').sum()
        log(f'   {fname}: {n_resolved}/{len(founder_cols[fname])} markers resolved (non-N)')

    founder_df = pd.DataFrame(founder_cols)

    log('5/6 deriving F1 columns (B6J / other founder, from founder consensus — not raw F1 samples)...')
    f1_cols = {}
    for f1name, other in F1_OTHER.items():
        b6 = founder_df['B6J']
        oth = founder_df[other]
        f1_cols[f1name] = [
            f'{a}/{b}' if a != 'N' and b != 'N' else 'N'
            for a, b in zip(b6, oth)
        ]
    f1_df = pd.DataFrame(f1_cols, index=founder_df.index)

    log('6/6 loading CC_consensus_genotypes.csv and joining...')
    cc = pd.read_csv(f'{IN_DIR}/CC_consensus_genotypes.csv', dtype=str)
    cc = cc.set_index('marker')
    cc.columns = [c.split('/')[0] for c in cc.columns]
    missing_cc = [s for s in CC_STRAINS if s not in cc.columns]
    if missing_cc:
        log(f'   WARNING: CC strains missing from CC_consensus_genotypes.csv: {missing_cc}')
    cc_sub = cc[[s for s in CC_STRAINS if s in cc.columns]]

    # Row set = positions with real CC genotype data (gm_uwisc position AND CC
    # consensus call). Founder/F1 columns are left-joined: a marker absent from
    # GigaMUGA_genotypes.csv.gz gets 'N' founders rather than being dropped —
    # dropping it would silently re-couple marker coverage to whichever legacy
    # file happens to have filled that gap.
    base = gm_f.join(cc_sub, how='inner')
    out = base.join(founder_df, how='left').join(f1_df, how='left')
    for col in FOUNDER_ORDER + F1_ORDER:
        out[col] = out[col].fillna('N')
    out = out.reset_index().rename(columns={'index': 'marker'})
    col_order = ['marker', 'chr', 'pos'] + FOUNDER_ORDER + F1_ORDER + CC_STRAINS
    out = out[col_order]

    out_path = f'{IN_DIR}/GM_ccf1fo_mm39.csv'
    out.to_csv(out_path, index=False)
    log(f'DONE: wrote {len(out)} markers x {len(col_order)} cols -> {out_path}')


if __name__ == '__main__':
    main()

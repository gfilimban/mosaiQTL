"""Build mosaiQTL intro slide deck (.pptx → import to Google Slides).

Visual placeholders are marked with [ VISUAL: ... ] — replace with actual plots/diagrams.
Presenter notes carry the full detail removed from slides.
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ── palette ────────────────────────────────────────────────────────────────
BLUE        = RGBColor(0x1A, 0x3A, 0x5C)   # deep navy — titles, headers
ACCENT      = RGBColor(0x4F, 0x6E, 0xAF)   # NOD/J blue
GOLD        = RGBColor(0xF4, 0xC2, 0x45)   # A/J gold
RED         = RGBColor(0xD8, 0x30, 0x26)   # PWK/J red
GREEN       = RGBColor(0x55, 0xAF, 0x5B)   # CAST/J green
PURPLE      = RGBColor(0x68, 0x3C, 0x91)   # WSB/J purple
TEAL        = RGBColor(0x52, 0xA5, 0xDB)   # NZO/J teal
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GREY  = RGBColor(0xF2, 0xF4, 0xF8)
MID_GREY    = RGBColor(0x88, 0x88, 0x88)
DARK_GREY   = RGBColor(0x44, 0x44, 0x44)
TITLE_BAND  = RGBColor(0x9A, 0xB7, 0xD2)   # muted steel blue — reference intro fill

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)
FONT    = 'Calibri'


# ── helpers ────────────────────────────────────────────────────────────────

def blank_slide(prs: Presentation) -> object:
    return prs.slides.add_slide(prs.slide_layouts[6])


def fill_bg(slide, color: RGBColor) -> None:
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def add_notes(slide, text: str) -> None:
    tf = slide.notes_slide.notes_text_frame
    tf.text = text


def add_rect(slide, left, top, width, height,
             fill: RGBColor | None = None,
             line: RGBColor | None = None,
             line_width: int = 0) -> object:
    shape = slide.shapes.add_shape(1, left, top, width, height)
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(line_width)
    else:
        shape.line.fill.background()
    return shape


def add_textbox(slide, text: str, left, top, width, height,
                size: float = 18, bold: bool = False, italic: bool = False,
                color: RGBColor = WHITE, align=PP_ALIGN.LEFT,
                wrap: bool = True) -> object:
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = wrap
    para = tf.paragraphs[0]
    para.alignment = align
    run = para.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    run.font.name = FONT
    return txBox


def add_bullets(slide, items: list[str], left, top, width, height,
                size: float = 14, color: RGBColor = DARK_GREY,
                bullet_color: RGBColor = ACCENT,
                line_spacing: float | None = None) -> object:
    from pptx.util import Pt as _Pt
    txBox = slide.shapes.add_textbox(left, top, width, height)
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = PP_ALIGN.LEFT
        if line_spacing:
            para.line_spacing = _Pt(line_spacing)
        br = para.add_run()
        br.text = '▸  '
        br.font.size = _Pt(size - 1)
        br.font.color.rgb = bullet_color
        br.font.name = FONT
        run = para.add_run()
        run.text = item
        run.font.size = _Pt(size)
        run.font.color.rgb = color
        run.font.name = FONT
    return txBox


def slide_header(slide, title: str, subtitle: str = '') -> None:
    """Clean title + thin accent divider — no dark band."""
    add_textbox(slide, title,
                Inches(0.55), Inches(0.15), Inches(12.2), Inches(0.65),
                size=26, bold=True, color=BLUE)
    if subtitle:
        add_textbox(slide, subtitle,
                    Inches(0.55), Inches(0.78), Inches(12.2), Inches(0.32),
                    size=12, italic=True, color=MID_GREY)
    add_rect(slide, Inches(0.55), Inches(1.15), Inches(12.23), Pt(2), fill=ACCENT)


def add_visual_placeholder(slide, label: str, left, top, width, height,
                            accent: RGBColor = ACCENT) -> None:
    """Dashed-style placeholder box — replace with actual figure."""
    add_rect(slide, left, top, width, height, fill=LIGHT_GREY)
    add_rect(slide, left, top, Pt(3), height, fill=accent)
    add_textbox(slide, f'[ VISUAL: {label} ]',
                left + Inches(0.15), top + (height - Pt(18)) / 2,
                width - Inches(0.2), Pt(24),
                size=11, italic=True, color=MID_GREY, align=PP_ALIGN.LEFT)


def card(slide, title: str, body: str, left, top, width, height,
         color: RGBColor = ACCENT) -> None:
    """Single info card — title + 1-2 line body."""
    add_rect(slide, left, top, width, height, fill=LIGHT_GREY)
    add_rect(slide, left, top, Pt(4), height, fill=color)
    add_textbox(slide, title,
                left + Inches(0.18), top + Inches(0.1), width - Inches(0.22), Inches(0.38),
                size=13, bold=True, color=BLUE)
    add_textbox(slide, body,
                left + Inches(0.18), top + Inches(0.46), width - Inches(0.22),
                height - Inches(0.5),
                size=11, color=DARK_GREY)


# ── slides ──────────────────────────────────────────────────────────────────

def slide_title(prs: Presentation) -> None:
    """Slide 1 — Title."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)

    # title band — matches reference #9AB7D2
    add_rect(s, Inches(0.31), Inches(1.05), Inches(9.37), Inches(1.85),
             fill=TITLE_BAND)

    add_textbox(s, 'mosaiQTL',
                Inches(0.5), Inches(1.12), Inches(9.0), Inches(1.0),
                size=48, bold=True, color=BLUE, align=PP_ALIGN.LEFT)
    add_textbox(s, 'Haplotype-Based QTL Mapping for Multiparent Populations',
                Inches(0.5), Inches(2.05), Inches(9.0), Inches(0.52),
                size=16, italic=True, color=BLUE, align=PP_ALIGN.LEFT)

    # author block
    add_textbox(s, 'Ghassan Filimban',
                Inches(0.55), Inches(3.2), Inches(5), Inches(0.42),
                size=15, bold=True, color=BLUE)
    add_textbox(s, 'Seyedam Lab  ·  IGVF  ·  UCI  ·  2026',
                Inches(0.55), Inches(3.62), Inches(5), Inches(0.38),
                size=12, color=MID_GREY)

    # 8-founder color dots
    founder_colors = [
        RGBColor(0xF4,0xC2,0x45), RGBColor(0xC0,0xBF,0xBF),
        RGBColor(0xDA,0x9C,0xC1), RGBColor(0x4F,0x6E,0xAF),
        RGBColor(0x52,0xA5,0xDB), RGBColor(0x55,0xAF,0x5B),
        RGBColor(0xD8,0x30,0x26), RGBColor(0x68,0x3C,0x91),
    ]
    for i, fc in enumerate(founder_colors):
        add_rect(s, Inches(0.55 + i * 0.42), Inches(4.3),
                 Inches(0.32), Inches(0.32), fill=fc)
    add_textbox(s, 'A/J  B6/J  129S1  NOD  NZO  CAST  PWK  WSB',
                Inches(0.55), Inches(4.68), Inches(4), Inches(0.3),
                size=9, color=MID_GREY)

    # right-side visual placeholder
    add_visual_placeholder(s, 'CC pedigree / LOD profile example',
                           Inches(10.0), Inches(1.2), Inches(3.0), Inches(5.5),
                           accent=TITLE_BAND)

    add_notes(s,
        'mosaiQTL is a Python port of miQTL extended for single-cell multiomics. '
        'It adds a two-stage NB-GLMM (NEBULA-LN + PQL), Z-matrix replicate awareness, '
        'and a Nextflow SLURM HPC pipeline. '
        'Phase 1 validation on sparcc simulated data: ROP LOD r²=0.9975, MI LOD r=0.9756, '
        'GEV Δ=0.154 — all gates green. '
        'The 8 CC founders are A/J, C57BL/6J, 129S1/SvImJ, NOD/ShiLtJ, NZO/HlLtJ, '
        'CAST/EiJ, PWK/PhJ, WSB/EiJ.')


def slide_qtl_concept(prs: Presentation) -> None:
    """Slide 2 — What is QTL mapping?"""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'What is QTL Mapping?',
                 'Linking genomic regions to phenotypic variation')

    add_bullets(s, [
        'Test every locus: do haplotypes here explain the trait?',
        'LOD = (n/2) · log₁₀(RSS₀ / RSS₁)',
        '8 founders → richer allelic series than biparental crosses',
        'Probabilities H[sample, locus, founder] from HMM on GigaMUGA data',
    ],
    Inches(0.55), Inches(1.3), Inches(6.0), Inches(4.5),
    size=16, color=DARK_GREY, bullet_color=ACCENT, line_spacing=28)

    add_visual_placeholder(s, 'LOD profile (chr with peak) + founder effect bars below',
                           Inches(6.8), Inches(1.3), Inches(6.2), Inches(5.8),
                           accent=ACCENT)

    add_notes(s,
        'A QTL is a chromosomal region statistically associated with variation in a '
        'measurable trait. The LOD score uses ML denominator n (not n-p) to match '
        'miQTL/qtl2 convention. Multiparent populations provide more haplotype diversity '
        'and mapping resolution than biparental F2 crosses. '
        'Haplotype probabilities come from a forward-backward HMM: '
        'H[i, ℓ, f] = P(state_f at locus ℓ | genotype data for sample i). '
        'Genome-wide threshold: permute expression, scan, GEV fit on max-LOD distribution.')


def slide_history(prs: Presentation) -> None:
    """Slide 3 — History."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'A Brief History of QTL Mapping',
                 'From biparental crosses to multiparent populations')

    timeline = [
        ('1989', 'Lander & Botstein', 'LOD score framework for inbred crosses'),
        ('2001', 'Sen & Churchill',   'Multiple imputation for genotype uncertainty'),
        ('2008', 'Kang (EMMA)',       'LMM + kinship — controls population structure'),
        ('2017', 'Broman (R/qtl2)',   'LOCO kinship; diplotype model; multiparent'),
        ('2019', 'Keele (miQTL)',     'CC/DO-specific; ROP + MI; replicate-aware LMM'),
        ('2021', 'He (NEBULA)',       'NB-GLMM for single-cell counts'),
        ('2026', 'mosaiQTL',         'Python · NB-GLMM · cell-type resolution · HPC'),
    ]

    col_x = [Inches(0.55), Inches(1.35), Inches(3.05)]
    y0, row_h = Inches(1.35), Inches(0.71)

    for i, (year, who, what) in enumerate(timeline):
        yp = y0 + i * row_h
        is_us = year == '2026'
        yc   = BLUE if is_us else DARK_GREY
        bc   = GOLD if is_us else ACCENT
        add_rect(s, Inches(0.45), yp + Inches(0.06), Inches(0.08), Inches(0.42), fill=bc)
        add_textbox(s, year, col_x[0], yp, Inches(0.75), row_h,
                    size=13, bold=True, color=bc)
        add_textbox(s, who,  col_x[1], yp, Inches(1.6),  row_h,
                    size=13, italic=True, color=ACCENT)
        add_textbox(s, what, col_x[2], yp, Inches(9.7),  row_h,
                    size=13, color=yc, bold=is_us)

    add_notes(s,
        '1993 Zeng: composite interval mapping. 2010s CC/DO launched: 8-founder '
        'Collaborative Cross and Diversity Outbred populations. '
        'mosaiQTL extends miQTL: NB-GLMM for raw UMI counts, Z-matrix replicates, '
        'HPC Nextflow pipeline, multiomics backbone (caQTL, sQTL, vQTL planned).')


def slide_key_terms(prs: Presentation) -> None:
    """Slide 4 — Key terms: HAPPY, LOCO, LOD, ROP."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Key Terms', 'Definitions used throughout this presentation')

    terms = [
        (ACCENT, 'HAPPY',
         'Haplotype Association Mapping by Probabilistic Inference — '
         'the per-locus .RData genome cache format used by miQTL'),
        (GREEN,  'LOCO  (Leave-One-Chromosome-Out)',
         'Kinship matrix recomputed excluding the chromosome under test — '
         'prevents proximal contamination of the polygenic random effect'),
        (GOLD,   'LOD  (Logarithm of the Odds)',
         'LOD = (n/2)·log₁₀(RSS₀/RSS₁) — likelihood ratio comparing null vs '
         'alternative haplotype model; threshold set by permutation'),
        (RED,    'ROP  (Regression on Probabilities)',
         'Default scan mode: haplotype probability matrix H enters as continuous '
         'covariates — no hard diplotype call required'),
    ]

    for i, (color, title, body) in enumerate(terms):
        row, col = i // 2, i % 2
        lx = Inches(0.45 + col * 6.5)
        ty = Inches(1.35 + row * 2.9)
        card(s, title, body, lx, ty, Inches(6.1), Inches(2.7), color=color)

    add_notes(s,
        'HAPPY: genome cache format used by miQTL; one .RData per locus in '
        'additive/chrN/data/. DiploprobReader is the R5 class reading it. '
        'LOCO: without LOCO, a cis-QTL on chr N is partially absorbed into K, '
        'deflating LOD (proximal contamination, Listgarten 2012). '
        'LOD uses ML denominator n to match miQTL/qtl2 convention. '
        'ROP contrasts with MI (Multiple Imputation) which draws hard J-dim calls '
        'from H and averages LOD over draws.')


def slide_miqtl_approach(prs: Presentation) -> None:
    """Slide 5 — miQTL scan.h2lmm approach."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'miQTL — scan.h2lmm',
                 'The R reference implementation mosaiQTL ports and validates against')

    # left column — core model
    add_textbox(s, 'Core model (ROP)',
                Inches(0.55), Inches(1.35), Inches(5.8), Inches(0.4),
                size=14, bold=True, color=BLUE)
    add_bullets(s, [
        'y = Hβ + Xγ + Zu + ε',
        'u ~ N(0, τ²K_LOCO)',
        'h² by profile REML (Brent + grid)',
        'LOD = (n/2) · log₁₀(RSS₀ / RSS₁)',
    ],
    Inches(0.55), Inches(1.78), Inches(5.8), Inches(3.0),
    size=14, color=DARK_GREY, bullet_color=ACCENT, line_spacing=24)

    add_rect(s, Inches(6.55), Inches(1.3), Pt(1.5), Inches(5.7), fill=LIGHT_GREY)

    # right column — scan modes
    add_textbox(s, 'Scan modes',
                Inches(6.75), Inches(1.35), Inches(6.0), Inches(0.4),
                size=14, bold=True, color=BLUE)
    add_bullets(s, [
        'ROP — H as continuous covariates',
        'MI — hard diplotype draws, averaged LOD',
        'Permutation — GEV fit on max-LOD null',
        'Positional CI — parametric bootstrap',
    ],
    Inches(6.75), Inches(1.78), Inches(6.0), Inches(3.0),
    size=14, color=DARK_GREY, bullet_color=GREEN, line_spacing=24)

    add_visual_placeholder(s, 'ROP schematic: H matrix → WLS → LOD profile',
                           Inches(0.55), Inches(4.85), Inches(12.23), Inches(2.35),
                           accent=ACCENT)

    add_notes(s,
        'H is n×J at each locus ℓ (haplotype probability matrix). '
        'u ~ N(0, τ²K_LOCO) is the polygenic random effect with LOCO kinship. '
        'h² estimated by profile REML: 1D Brent on grid ∈ [0,1]. '
        'LOD denominator n (ML, not n-p) matches miQTL/qtl2. '
        'MI: K hard-call imputations from H, LOD averaged over K. '
        'Permutation: expression permuted (not genotype); fix_h2=True; GEV on max-LOD. '
        'Positional CI: parametric bootstrap from fitted null model, peak position distribution.')


def slide_miqtl_replicates(prs: Presentation) -> None:
    """Slide 6 — miQTL replicate handling via lmmbygls."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Replicate Handling — lmmbygls',
                 'miQTL collapses replicates; mosaiQTL keeps all n samples via Z-matrix')

    add_bullets(s, [
        'miQTL default: one observation per strain (strain mean)',
        'lmmbygls.replicates(): expands K via Z @ K_strain @ Z′',
        'mosaiQTL: Z-matrix (n × m) explicit — all n samples retained',
        'kinship_samples = Z K Z′  ·  no strain averaging required',
    ],
    Inches(0.55), Inches(1.35), Inches(6.2), Inches(4.0),
    size=15, color=DARK_GREY, bullet_color=ACCENT, line_spacing=28)

    add_visual_placeholder(s, 'Z-matrix diagram: n=418 samples → m=48 strains mapping',
                           Inches(6.8), Inches(1.35), Inches(6.2), Inches(5.8),
                           accent=ACCENT)

    add_notes(s,
        'lmmbygls() is the core GLS/LMM solver inside scan.h2lmm. '
        'Replicates share strain genotype so K_{ij} = K_{s(i),s(j)}. '
        'Within-strain entries = 1, between-strain entries reflect IBD sharing. '
        'mosaiQTL: Z ∈ {0,1}^{n×m}, Z_{ik}=1 if sample i belongs to strain k. '
        'At scan time: H_n = Z H_m (expand probs), K_n = Z K_m Z^T. '
        'Validation comparisons use m=48 strain-level phenotypes for fair R vs Python comparison.')


def slide_miqtl_limits(prs: Presentation) -> None:
    """Slide 7 — miQTL limitations."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'miQTL Limitations',
                 'Gaps that motivate the mosaiQTL design')

    limits = [
        (RED,    'Gaussian only',        'Log-normalised counts → inflated FP for low-expr genes'),
        (RED,    'No count model',       'Cannot capture NB overdispersion in snRNA-seq UMI'),
        (RED,    'Bulk / strain-averaged','No native cell-type resolution from pseudobulk'),
        (RED,    'R-only, serial',       '~4 h genome-wide on n=48; no parallelisation'),
        (RED,    'No modality extensions','No caQTL, sQTL, rmodQTL, vQTL, epistasis'),
        (RED,    'No HPC pipeline',      'Manual SLURM chunking required'),
    ]

    for i, (color, title, body) in enumerate(limits):
        row, col = i // 2, i % 2
        lx = Inches(0.45 + col * 6.5)
        ty = Inches(1.35 + row * 1.9)
        card(s, title, body, lx, ty, Inches(6.1), Inches(1.75), color=color)

    add_notes(s,
        'Gaussian only: log-normalising inflates false positives for lowly-expressed genes '
        '(NB mean-variance relationship ignored). '
        'Bulk: miQTL takes one phenotype per strain — biological replication is discarded. '
        'R serial: ~4h per cell type for 500 genes on CC48. '
        'No modality extensions: mosaiQTL targets caQTL (ATAC), sQTL (splicing), '
        'rmodQTL (m6A), vQTL (dispersion), epistasis, mediation.')


def slide_mosaiqtl_novelty(prs: Presentation) -> None:
    """Slide 8 — mosaiQTL novelty (white background)."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'What mosaiQTL Adds',
                 'A Python port that extends miQTL to single-cell multiomics')

    novelties = [
        (GREEN,  'NB-GLMM primary model',    'Two-stage NEBULA-LN + PQL on raw UMI counts'),
        (ACCENT, 'Cell-type resolution',     'Per-cell-type pseudobulk; any aggregation level'),
        (GOLD,   'Z-matrix native design',   'n=418 samples; kinship_samples = Z K Z′'),
        (RED,    'Multiomics backbone',      'eQTL · caQTL · sQTL · vQTL · epistasis · mediation'),
        (TEAL,   'HPC + composable CLI',     'Nextflow DSL2 · fit-null / scan / permute / report'),
        (PURPLE, 'Validated against miQTL',  'Phase 1: r²=0.9975 · MI r=0.9756 · GEV Δ=0.154'),
    ]

    for i, (color, title, body) in enumerate(novelties):
        row, col = i // 2, i % 2
        lx = Inches(0.45 + col * 6.5)
        ty = Inches(1.35 + row * 1.9)
        card(s, title, body, lx, ty, Inches(6.1), Inches(1.75), color=color)

    add_notes(s,
        'NB-GLMM: Stage 1 NEBULA-LN cell null per gene → φ_shrunk, σ²; '
        'Stage 2 PQL NB-GLMM score scan with LOCO kinship via Woodbury identity (m×m ops). '
        'Z-matrix: n=418 samples / m=48 strains for gastroc CC data. '
        'Multiomics: new modality = new Stage 1 module only; Stage 2 scan unchanged. '
        'Composable CLI: each subcommand reads/writes HDF5 or CSV; no hidden state.')


def slide_architecture(prs: Presentation) -> None:
    """Slide 9 — Two-stage NB-GLMM architecture."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Architecture — Two-Stage NB-GLMM',
                 'Stage 1 fits null per gene; Stage 2 scans all loci with fixed dispersion')

    stages = [
        (ACCENT, 'Stage 1 — Null Fit  (nb.py)',
         ['Raw UMI counts → NEBULA-LN cell-level NB-GLMM',
          'Output: φ_shrunk, σ² per gene  (fixed for Stage 2)',
          'Empirical Bayes shrinkage of dispersion across genes']),
        (GREEN,  'Stage 2 — PQL Scan  (nb.py)',
         ['Score stat Q_ℓ ~ χ²(J−1) vectorised over all loci',
          'Woodbury identity: V = diag(1/W) + τ²ZK_LOCO Z′  →  ops at m×m',
          'Null fit once per gene per chr; scan O(1) per locus']),
    ]

    for i, (color, title, bullets) in enumerate(stages):
        ty = Inches(1.35 + i * 2.5)
        add_rect(s, Inches(0.45), ty, Inches(6.5), Inches(2.3), fill=LIGHT_GREY)
        add_rect(s, Inches(0.45), ty, Pt(4), Inches(2.3), fill=color)
        add_textbox(s, title,
                    Inches(0.65), ty + Inches(0.1), Inches(6.2), Inches(0.4),
                    size=13, bold=True, color=BLUE)
        add_bullets(s, bullets,
                    Inches(0.65), ty + Inches(0.52), Inches(6.1), Inches(1.65),
                    size=12, color=DARK_GREY, bullet_color=color, line_spacing=18)

    add_visual_placeholder(s, 'Two-stage pipeline flow: counts → Stage 1 → φ → Stage 2 → LOD table',
                           Inches(7.2), Inches(1.35), Inches(5.75), Inches(5.7),
                           accent=ACCENT)

    add_notes(s,
        'Stage 1: fit_null_sc_native() — NEBULA-LN cell null. '
        'φ_shrunk via empirical Bayes (WLEB) across genes. '
        'Stage 2: scan_nb_gene() — PQL iteration: '
        'W=μ/(1+φμ), z=(log μ−offset)+(S−μ)/μ → Gaussian LMM step → update μ. '
        'Woodbury: V=diag(1/W)+τ²ZK_LOCO Z^T — all ops at m×m (m=n_strains), never n×n. '
        'Null fit once per gene × chr; score scan is O(1) per locus → vectorisable.')


def slide_dir_before(prs: Presentation) -> None:
    """Slide 10 — Before the reorganization."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Before the Reorganization',
                 'Flat scripts → modular, testable, reproducible package')

    add_textbox(s, '✗  Before',
                Inches(0.55), Inches(1.35), Inches(5.8), Inches(0.42),
                size=14, bold=True, color=RED)
    add_bullets(s, [
        'Flat directory — no package boundary',
        'Modeling, I/O, and plotting mixed in same files',
        'Founder colors re-defined in every script',
        'Tests impossible — no importable module',
        '"Works on my machine" — hardcoded paths',
    ],
    Inches(0.55), Inches(1.82), Inches(5.7), Inches(4.2),
    size=13, color=DARK_GREY, bullet_color=RED, line_spacing=24)

    add_rect(s, Inches(6.55), Inches(1.3), Pt(1.5), Inches(5.7), fill=LIGHT_GREY)

    add_textbox(s, '✓  After',
                Inches(6.75), Inches(1.35), Inches(5.8), Inches(0.42),
                size=14, bold=True, color=GREEN)
    add_bullets(s, [
        'src/ layout — pip installable, editable',
        'Strict module boundaries per layer',
        'founders.py — one canonical source of truth',
        'pytest + CI — 90% coverage target',
        'Staged validation: phase1 → phase2 → phase3',
    ],
    Inches(6.75), Inches(1.82), Inches(5.7), Inches(4.2),
    size=13, color=DARK_GREY, bullet_color=GREEN, line_spacing=24)

    add_notes(s,
        'Before: no shared data contracts — function signatures drifted across scripts. '
        'R constants (founder colors, strain order) re-defined in every script. '
        'No version control on data, no reproducibility. '
        'After: typed dataclasses (ScanResult, VQTLResult, FounderEffect); '
        'composable CLI (fit-null / scan / permute / effects / report); '
        'CLAUDE.md at root + every subpackage for AI-assisted development context.')


def slide_dir_structure(prs: Presentation) -> None:
    """Slide 11 — Reorganized package structure."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Package Structure',
                 'src/ layout · three-layer module hierarchy · one CLAUDE.md per subpackage')

    cols = [
        (Inches(0.45), ACCENT, 'Genotype',
         ['genomecache.py — HDF5 B-tree',
          'hmm.py — forward-backward HMM',
          'kinship.py — LOCO, Woodbury',
          'founders.py — all constants',
          'models/records.py — ScanResult …',
          'models/config.py — Pydantic v2']),
        (Inches(4.6),  GREEN,  'QTL Engine',
         ['nb.py — NB-GLMM Stage 1+2',
          'lmm.py — Gaussian LMM port',
          'scan.py — Z-aware, MI, instability',
          'permutation.py — GEV thresholds',
          'effects.py — founder effects',
          'mediation · epistasis · vqtl · snp']),
        (Inches(8.75), GOLD,   'I/O · Viz · Infra',
         ['io/covariates.py',
          'io/anndata_io · qtl2_io · results_io',
          'viz/genome.py — LOD, Manhattan',
          'viz/effects · haplotypes · celltypes',
          'viz/validation.py — Phase 2 QC only',
          'pipeline.py · cli.py · utils.py']),
    ]

    for lx, color, title, items in cols:
        add_rect(s, lx, Inches(1.35), Inches(4.05), Inches(5.6), fill=LIGHT_GREY)
        add_rect(s, lx, Inches(1.35), Pt(4), Inches(5.6), fill=color)
        add_textbox(s, title,
                    lx + Inches(0.18), Inches(1.42), Inches(3.8), Inches(0.38),
                    size=14, bold=True, color=BLUE)
        add_bullets(s, items,
                    lx + Inches(0.18), Inches(1.84), Inches(3.8), Inches(4.95),
                    size=11, color=DARK_GREY, bullet_color=color, line_spacing=17)

    add_rect(s, Inches(0.45), Inches(7.1), Inches(12.45), Inches(0.28), fill=BLUE)
    add_textbox(s,
                'pip install -e ".[dev]"  ·  pytest tests/ -q  ·  make lint',
                Inches(0.65), Inches(7.12), Inches(12.1), Inches(0.24),
                size=11, color=WHITE, align=PP_ALIGN.CENTER)

    add_notes(s,
        'viz/validation.py is imported only from tests/ and validation/ scripts — '
        'never from the installed package. '
        'models/ contains shared dataclasses: ScanResult, VQTLResult, QTLResult, '
        'FounderEffect, ProvenanceRecord. '
        'PipelineConfig (Pydantic v2) validates all scan parameters at entry point. '
        'CLAUDE.md at root + genotype/ + qtl/ + io/ + viz/ gives AI tools '
        'module-level context without reading all source files.')


def slide_dir_principles(prs: Presentation) -> None:
    """Slide 12 — Design principles enforced by structure."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Design Principles',
                 'Structure enforces correctness — not just convention')

    principles = [
        (ACCENT, 'Single source of truth',   'founders.py — all colors, strain order, N_FOUNDERS'),
        (GREEN,  'Typed data contracts',     'scan_chromosome() always returns ScanResult'),
        (GOLD,   'Composable CLI',           'Each subcommand reads/writes files; no hidden state'),
        (RED,    'Staged validation',         'phase1 → phase2 → phase3; independent, reproducible'),
        (BLUE,   'viz/validation never ships','Phase 2 QC plots isolated from production code'),
        (PURPLE, 'Constant-memory streaming','Full LOD matrix never materialised — stream per-chr'),
    ]

    for i, (color, title, body) in enumerate(principles):
        row, col = i // 2, i % 2
        lx = Inches(0.45 + col * 6.5)
        ty = Inches(1.35 + row * 1.9)
        card(s, title, body, lx, ty, Inches(6.1), Inches(1.75), color=color)

    add_notes(s,
        'founders.py: FOUNDER_ORDER, FOUNDER_CODE, FOUNDER_COLORS, F1_COLORS, '
        'COLOR_NEUTRAL_DARK, COLOR_CC_FALLBACK, COLOR_NEUTRAL_LIGHT — never inline hex anywhere. '
        'ScanResult.chr is always str — root cause of S11 r=-0.54 was int chr column in merge. '
        'MEM-05 is a hard architectural constraint: if scan.py materialises the full LOD matrix '
        '(n_genes × n_loci), that is a design failure, not just a memory warning. '
        'viz/validation.py tests imports are gated by pytest.importorskip to avoid CI failures.')


def slide_genomecache(prs: Presentation) -> None:
    """Slide 13 — Genome cache: miQTL HAPPY vs mosaiQTL HDF5."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Genome Cache: miQTL vs mosaiQTL',
                 'Per-locus .RData files → single HDF5 file with lazy chromosome loading')

    # left panel — HAPPY
    lx_l = Inches(0.45)
    add_rect(s, lx_l, Inches(1.35), Inches(5.9), Inches(5.8), fill=LIGHT_GREY)
    add_rect(s, lx_l, Inches(1.35), Pt(4), Inches(5.8), fill=RED)
    add_textbox(s, 'miQTL — HAPPY (R)',
                lx_l + Inches(0.18), Inches(1.42), Inches(5.6), Inches(0.38),
                size=13, bold=True, color=BLUE)
    add_bullets(s, [
        '~855 .RData files per chromosome  (×20 chr = ~17k files)',
        'load() + deserialise per locus call',
        'Sequential: 855 file reads per phenotype per scan',
        'LOCO kinship: no native support → workaround needed',
        'DiploprobReader$getLocusMatrix(locus, "additive")',
    ],
    lx_l + Inches(0.18), Inches(1.85), Inches(5.6), Inches(4.9),
    size=12, color=DARK_GREY, bullet_color=RED, line_spacing=19)

    add_textbox(s, '→',
                Inches(6.5), Inches(4.0), Inches(0.5), Inches(0.5),
                size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)

    # right panel — HDF5
    lx_r = Inches(7.1)
    add_rect(s, lx_r, Inches(1.35), Inches(5.9), Inches(5.8), fill=LIGHT_GREY)
    add_rect(s, lx_r, Inches(1.35), Pt(4), Inches(5.8), fill=GREEN)
    add_textbox(s, 'mosaiQTL — HDF5 (Python)',
                lx_r + Inches(0.18), Inches(1.42), Inches(5.6), Inches(0.38),
                size=13, bold=True, color=BLUE)
    add_bullets(s, [
        '1 HDF5 file; 1 group per chr; probs shape (n_loci, n_strains, J)',
        'h5py.File kept open — Dataset refs stored, not loaded at startup',
        'get_chromosome(chr) → np.asarray(d["probs"]) — one slice per scan',
        'Batch einsum over all L loci — no Python loop',
        'calc_kinship(cache, method="loco") native — no workaround',
    ],
    lx_r + Inches(0.18), Inches(1.85), Inches(5.6), Inches(4.9),
    size=12, color=DARK_GREY, bullet_color=GREEN, line_spacing=19)

    add_rect(s, Inches(0.45), Inches(7.12), Inches(12.45), Inches(0.26), fill=BLUE)
    add_textbox(s,
                'Memory per active chr = n_loci × n_strains × J × 8 B  '
                '(CC: ~40 MB/chr  ·  MAGIC-19: ~450 MB/chr)',
                Inches(0.65), Inches(7.13), Inches(12.1), Inches(0.22),
                size=10, color=WHITE, align=PP_ALIGN.CENTER)

    add_notes(s,
        'HAPPY: per-locus .RData in additive/chrN/data/; markers.RData is the locus index. '
        'DiploprobReader (R5 class) reads one file per getLocusMatrix() call. '
        'mosaiQTL from_hdf5: keeps h5py.File open; stores Dataset refs '
        '(grp["probs"] — NOT grp["probs"][:]). '
        'get_chromosome() calls np.asarray(d["probs"]) to load one chr at a time. '
        'np.asarray() handles both h5py Dataset and numpy array transparently. '
        'FWL projection: M_X applied once via eigenrotation; '
        'then batch einsum/WLS over all L loci simultaneously — replaces R for-loop.')


def slide_validation(prs: Presentation) -> None:
    """Slide 14 — Validation progress — Phase 1."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'Validation — Phase 1',
                 'Minimal miQTL workflow mirror: sparcc_cache simulated data, 40 CC strains')

    rows = [
        ('P1-S1', 'ROP LOD r²',       '≥ 0.990', '0.9975', GREEN),
        ('P1-S2', 'MI LOD r',         '≥ 0.75',  '0.9756', GREEN),
        ('P1-S3', 'GEV threshold Δ',  '< 0.5',   '0.154',  GREEN),
    ]

    cols    = [Inches(0.55), Inches(1.35), Inches(3.1), Inches(5.5), Inches(7.5)]
    hdrs    = ['ID', 'Section', 'Metric', 'Threshold', 'Current']
    hwidths = [Inches(0.75), Inches(1.7), Inches(2.3), Inches(1.95), Inches(2.25)]

    ty = Inches(1.35)
    add_rect(s, Inches(0.45), ty, Inches(9.6), Inches(0.42), fill=BLUE)
    for hx, hw, ht in zip(cols, hwidths, hdrs):
        add_textbox(s, ht, hx, ty + Inches(0.04), hw, Inches(0.35),
                    size=12, bold=True, color=WHITE)

    for ri, (pid, section, thr, current, color) in enumerate(rows):
        ry = ty + Inches(0.47 + ri * 0.62)
        bg = LIGHT_GREY if ri % 2 == 0 else WHITE
        add_rect(s, Inches(0.45), ry, Inches(9.6), Inches(0.56), fill=bg)
        add_rect(s, Inches(0.45), ry, Pt(4), Inches(0.56), fill=color)
        for vx, vw, vv in zip(cols, hwidths,
                               [pid, section, thr, thr, current]):
            add_textbox(s, vv, vx, ry + Inches(0.08), vw, Inches(0.42),
                        size=12, color=DARK_GREY)

    # fixes box
    add_rect(s, Inches(0.45), Inches(3.55), Inches(9.6), Inches(1.65), fill=LIGHT_GREY)
    add_textbox(s, 'Key fixes',
                Inches(0.65), Inches(3.62), Inches(9.0), Inches(0.35),
                size=13, bold=True, color=BLUE)
    add_bullets(s, [
        'J-founder refactor — pinv on all J columns (was root cause of r²=0.9965)',
        'Permutation scope — peak chromosome only (R scans chr10, not all 20)',
        'R export fix — keep.full.scans=FALSE → correct n=500 max-LOD vector',
    ],
    Inches(0.65), Inches(4.0), Inches(9.1), Inches(1.1),
    size=11, color=DARK_GREY, bullet_color=ACCENT, line_spacing=16)

    add_visual_placeholder(s, 'LOD scatter R vs Python + GEV fit overlay (P1-S1, P1-S3)',
                           Inches(9.8), Inches(1.35), Inches(3.2), Inches(5.8),
                           accent=GREEN)

    add_notes(s,
        'P1-S1 ROP: r²=0.9975 on sparcc sim data (40 CC lines, 10 phenotypes, 2 replicates). '
        'J-founder fix: lmm.py was dropping WSB column always; R pivots to near-zero founder. '
        'P1-S3 GEV: restricting permutation to peak chromosome (chr10) matches R behavior. '
        'R export: keep.full.scans=FALSE in run.threshold.scans() returns numeric vector; '
        'TRUE returns a 2-row list object. '
        'Manhattan plot fix: ax.scatter() not ax.plot() (was drawing connected lines). '
        'Phase 2 (S1–S11) and Phase 3 (NB vs LMM) not yet run.')


def slide_next(prs: Presentation) -> None:
    """Slide 15 — Road ahead."""
    s = blank_slide(prs)
    fill_bg(s, WHITE)
    slide_header(s, 'What Comes Next', '')

    cols_data = [
        (GOLD,   'Phase 2 — Full miQTL Port',
         'S1–S11 · real LOCO kinship · sparcc sim data'),
        (GREEN,  'Phase 3 — NB vs LMM',
         'Piezo1 chr8 · vQTL · epistasis · simulated counts'),
        (ACCENT, 'SWE & HPC',
         '≥90% coverage · Nextflow stub-run · benchmarks'),
        (PURPLE, 'Production',
         'Gastroc snRNA-seq · cell-type eQTL tables · Streamlit browser'),
    ]

    for i, (color, title, body) in enumerate(cols_data):
        row, col = i // 2, i % 2
        lx = Inches(0.45 + col * 6.5)
        ty = Inches(1.35 + row * 2.65)
        card(s, title, body, lx, ty, Inches(6.1), Inches(2.5), color=color)

    add_notes(s,
        'Phase 2 sim: port_sim_validation_p2.R → validate_sim_p2.py. '
        'Uses real LOCO kinship (Phase 1 used identity/OLS shortcut). '
        'S5 collapse: sparcc_cache is pre-collapsed → minimal removal expected. '
        'S8 SNP scan: gated — requires allele_dir. '
        'Phase 3: sparc_to_nb_counts() generates NB count fixtures from sparc phenotypes. '
        'Production: UCI CRSP HPC, gastroc CC48 snRNA-seq, ~418 samples, all cell types. '
        'Open issues: S6 GEV instability (mean_delta=0.619 in gastroc track), '
        'Storey q-value not yet implemented (BH-FDR anti-conservative).')


# ── main ────────────────────────────────────────────────────────────────────

def main() -> None:
    prs = Presentation()
    prs.slide_width  = SLIDE_W
    prs.slide_height = SLIDE_H

    slide_title(prs)            # 1
    slide_qtl_concept(prs)      # 2  — visual: LOD profile + founder effects
    slide_history(prs)          # 3
    slide_key_terms(prs)        # 4  — HAPPY, LOCO, LOD, ROP
    slide_miqtl_approach(prs)   # 5  — visual: ROP schematic
    slide_miqtl_replicates(prs) # 6  — visual: Z-matrix diagram
    slide_miqtl_limits(prs)     # 7
    slide_mosaiqtl_novelty(prs) # 8
    slide_architecture(prs)     # 9  — visual: two-stage pipeline flow
    slide_dir_before(prs)       # 10 — before/after reorganization
    slide_dir_structure(prs)    # 11 — package structure
    slide_dir_principles(prs)   # 12 — design principles
    slide_genomecache(prs)      # 13 — HAPPY vs HDF5 deep dive
    slide_validation(prs)       # 14 — visual: LOD scatter + GEV fit
    slide_next(prs)             # 15

    out = Path(__file__).parent / 'mosaiqtl_deck.pptx'
    prs.save(out)
    print(f'Saved: {out}  ({len(prs.slides)} slides)')

    # ── recommended visuals (not yet implemented) ──────────────────────────
    print('\nRecommended visuals to add:')
    print('  Slide  1 — CC pedigree diagram or example LOD peak figure')
    print('  Slide  2 — LOD profile (1 chr, clear peak) + founder effect bars below')
    print('  Slide  5 — ROP schematic: H matrix → WLS per locus → LOD vector')
    print('  Slide  6 — Z-matrix diagram: n=418 rows → m=48 strain columns')
    print('  Slide  9 — Two-stage pipeline flow: counts → Stage1 → φ → Stage2 → LOD')
    print('  Slide 13 — HAPPY dir tree (left) vs HDF5 group structure (right) diagram')
    print('  Slide 14 — LOD scatter R vs Python (P1-S1) + GEV max-LOD overlay (P1-S3)')


if __name__ == '__main__':
    main()

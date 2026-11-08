"""
Static parity checks for the mosaiQTL Nextflow DSL2 pipeline.

Three check categories:
  1. argv_parity   — NF module script block passes exactly the expected number of
                     positional tokens; derived from argv table below.
  2. output_parity — each filename suffix listed in a Python script's "Writes:"
                     docstring section has a matching glob in the NF output: block.
  3. anatomy       — every NF module declares label, tag, stub, and publishDir
                     (publishDir excepted for LMM_SCAN, whose per-chr outputs are
                     collected by MERGE_RESULTS instead).

Run: pytest tests/workflow/ -v
"""
import fnmatch
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO        = Path(__file__).resolve().parents[2]
SCRIPTS     = REPO / 'workflow' / 'scripts'
MODULES     = REPO / 'modules' / 'local'
MAIN_NF     = REPO / 'workflow' / 'main.nf'
CONFIG      = REPO / 'config' / 'default_params.yaml'



# ── source parsers ────────────────────────────────────────────────────────────

def _nf_script_tokens(module_name: str) -> int:
    """Count positional arg tokens passed to the Python script in a module's script: block.

    Skips the leading 'python .../workflow/scripts/<name>.py' token and counts
    every remaining token across backslash-continued lines.  Single-quoted
    strings like '${celltype}' count as one token.  Returns -1 when no
    script: block is found.

    Handles Groovy 'def' lines before the triple-quote delimiter, e.g.:
        script:
        def sanger = params.x ?: ''
        \"\"\"
        python ... \\
        \"\"\"
    """
    text = (MODULES / module_name).read_text()
    # Find 'script:' keyword, then locate the first triple-quote shell block after it.
    m = re.search(r'\bscript:', text)
    if not m:
        return -1
    after = text[m.end():]
    tq_open = after.find('"""')
    if tq_open == -1:
        return -1
    tq_close = after.find('"""', tq_open + 3)
    if tq_close == -1:
        return -1
    block = after[tq_open + 3 : tq_close]

    tokens: list[str] = []
    past_cmd = False
    for raw_line in block.splitlines():
        stripped   = raw_line.strip()
        continued  = stripped.endswith('\\')
        content    = stripped.rstrip('\\').strip()

        if not content or content.startswith('#'):
            continue

        if re.search(r'\bpython\b.*workflow/scripts/', content):
            past_cmd = True
            continue

        if past_cmd:
            toks = re.findall(r"'[^']*'|\S+", content)
            tokens.extend(toks)
            if not continued:
                past_cmd = False

    return len(tokens)


def _nf_output_globs(module_name: str) -> list[str]:
    """Return all quoted path patterns from the output: block of an NF module.

    Handles both 'path "glob"' and 'path("glob")' forms, including optional
    outputs declared as 'path "glob", optional: true'.
    """
    text = (MODULES / module_name).read_text()
    m = re.search(r'\boutput:\s*\n(.*?)(?=\bscript:|\bstub:)', text, re.DOTALL)
    if not m:
        return []
    block = m.group(1)
    # Match both path 'glob' and path("glob") (single or double quoted)
    return re.findall(r"""path\s*(?:\()?\s*['"]([^'"]+)['"]""", block)


def _docstring_writes(script_name: str) -> list[str]:
    """Extract filenames/patterns from the 'Writes:' section of a script's module docstring."""
    text = (SCRIPTS / script_name).read_text()
    m = re.match(r'"""(.*?)"""', text, re.DOTALL)
    if not m:
        return []
    doc = m.group(1)

    in_writes = False
    patterns: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if re.match(r'writes\s*:', stripped, re.IGNORECASE):
            in_writes = True
            continue
        if not in_writes:
            continue
        # Stop at the first blank line followed by a non-indented word (next section)
        if not stripped:
            continue
        # Stop if we hit what looks like a new heading (no leading spaces, ends with ':')
        if stripped.endswith(':') and not stripped.startswith('{') and not '.' in stripped:
            break
        # Extract the first token that looks like a filename (contains a dot)
        fname = stripped.split()[0] if stripped.split() else ''
        if '.' in fname and not fname.startswith('(') and not fname.startswith('<'):
            patterns.append(fname)
    return patterns


def _nf_has_anatomy(module_name: str) -> dict[str, bool]:
    """Check that a module has the four required NF primitives."""
    text = (MODULES / module_name).read_text()
    return {
        'label':      bool(re.search(r'\blabel\s+', text)),
        'tag':        bool(re.search(r'\btag\s+', text)),
        'stub':       bool(re.search(r'\bstub:\s*\n', text)),
        'publishDir': bool(re.search(r'\bpublishDir\b', text)),
    }


# ── suffix-matching helpers ───────────────────────────────────────────────────

def _writes_canonical(pattern: str) -> list[str]:
    """Normalise a Writes: pattern to a glob-like suffix for comparison.

    Strips the leading {tag}_ prefix and replaces remaining {var} groups with *.
    Returns two variants when {_share} is present (with and without).
    """
    # Strip leading {tag}_ (covers {tag}, {tissue}_{ct}_, etc.)
    pat = re.sub(r'^\{[^}]+\}_?', '', pattern)
    # Produce variants for {_share}
    if '{_share}' in pat:
        variants = [pat.replace('{_share}', ''), pat.replace('{_share}', '_share')]
    else:
        variants = [pat]
    # Replace all remaining {var} with *
    return [re.sub(r'\{[^}]+\}', '*', v) for v in variants]


def _glob_suffix(nf_glob: str) -> str:
    """Strip leading * from an NF output glob to get the fixed suffix."""
    return nf_glob.lstrip('*')


def _suffix_covered(writes_pattern: str, nf_globs: list[str]) -> bool:
    """Return True when writes_pattern has a structurally matching NF output glob."""
    canonicals = _writes_canonical(writes_pattern)
    nf_suffixes = [_glob_suffix(g) for g in nf_globs]

    for cw in canonicals:
        # Strip leading underscore from the canonical form for comparison
        cw_plain = cw.lstrip('_')
        for ns, ng in zip(nf_suffixes, nf_globs):
            ns_plain = ns.lstrip('_')
            # Exact match after stripping leading wildcard/underscore
            if cw_plain == ns_plain:
                return True
            # fnmatch: canonical written pattern vs NF glob
            if fnmatch.fnmatch(cw, ng) or fnmatch.fnmatch(cw_plain, ns_plain):
                return True
            # Substring: the fixed part of the writes pattern appears in the NF glob
            # (handles lod_chr{c}.csv matching lod_chr${chrom}.csv)
            fixed_part = re.sub(r'\*', '', cw_plain)
            if fixed_part and fixed_part in ng:
                return True
    return False


# ── 1. argv count parity ──────────────────────────────────────────────────────

# Each tuple: (python_script, nf_module, expected_nf_token_count)
#
# expected_nf_token_count = mandatory_argc + optional_argc_always_passed_by_nf
#   build_cache.py         2 = 2 + 0
#   build_kinship_loco.py  2 = 2 + 0
#   run_lmm_scan_chr.py   10 = 10 + 0
#   run_merge.py           6 = 5 + 1 (tissue always appended)
#   run_permute.py        13 = 12 + 1 (tissue always appended)
#   run_call_egenes.py    15 = 14 + 1 (tissue always appended)
#   run_finemap.py        11 = 9 + 2 (${sanger} slot + tissue always appended)
#   run_report.py         13 = 11 + 2 (${gtf_arg} slot + tissue always appended)
_ARGC_TABLE = [
    ('build_cache.py',        'build_cache.nf',      2),
    ('build_kinship_loco.py', 'kinship_loco.nf',     2),
    ('run_lmm_scan_chr.py',   'lmm_scan.nf',        10),
    ('run_merge.py',          'merge_results.nf',     6),
    ('run_permute.py',        'permute.nf',          13),
    ('run_call_egenes.py',    'call_egenes.nf',      15),
    ('run_finemap.py',        'finemap.nf',          11),
    ('run_report.py',         'report.nf',           13),
]


@pytest.mark.parametrize('script, module, expected', _ARGC_TABLE,
                         ids=[r[0] for r in _ARGC_TABLE])
def test_argv_count_parity(script: str, module: str, expected: int) -> None:
    """NF script block passes exactly expected positional tokens to the Python script."""
    actual = _nf_script_tokens(module)
    assert actual != -1, f'no script: block found in {module}'
    assert actual == expected, (
        f'{script} <-> {module}: NF passes {actual} positional tokens, expected {expected}.\n'
        f'Update _ARGC_TABLE in this test if the signature changed intentionally.'
    )


# ── 2. output suffix parity ───────────────────────────────────────────────────

# lmm_scan has no "Writes:" section (outputs are per-chr intermediates), so it is
# verified separately via _LMM_EXPLICIT below instead of docstring parsing.
_OUTPUT_SCRIPTS = [
    ('build_cache.py',        'build_cache.nf'),
    ('build_kinship_loco.py', 'kinship_loco.nf'),
    ('run_merge.py',          'merge_results.nf'),
    ('run_permute.py',        'permute.nf'),
    ('run_call_egenes.py',    'call_egenes.nf'),
    ('run_finemap.py',        'finemap.nf'),
    ('run_report.py',         'report.nf'),
]

# Explicit output mapping for lmm_scan (no docstring "Writes:" section).
# Each (writes_pattern, nf_glob) pair must match structurally.
_LMM_EXPLICIT = [
    (r'lod_chr{chrom}.csv',          r'lod_chr${chrom}.csv'),
    (r'peak_stats_chr{chrom}.csv',   r'peak_stats_chr${chrom}.csv'),
    (r'timing_chr{chrom}.json',      r'timing_chr${chrom}.json'),
]


@pytest.mark.parametrize('script, module', _OUTPUT_SCRIPTS,
                         ids=[r[0] for r in _OUTPUT_SCRIPTS])
def test_output_suffix_parity(script: str, module: str) -> None:
    """Each 'Writes:' entry in the Python docstring has a matching NF output glob."""
    writes = _docstring_writes(script)
    if not writes:
        pytest.skip(f'{script} has no "Writes:" section in its module docstring')

    globs = _nf_output_globs(module)
    assert globs, f'{module} declares no quoted path() patterns in its output: block'

    missing = [w for w in writes if not _suffix_covered(w, globs)]
    assert not missing, (
        f'{script} -> {module}: these Writes: patterns have no matching NF output glob:\n'
        + '\n'.join(f'  {w!r}' for w in missing)
        + f'\nNF output globs: {globs}'
    )


@pytest.mark.parametrize('writes_pat, nf_glob', _LMM_EXPLICIT,
                         ids=[r[0] for r in _LMM_EXPLICIT])
def test_lmm_scan_output_parity(writes_pat: str, nf_glob: str) -> None:
    """run_lmm_scan_chr.py's per-chr outputs match lmm_scan.nf output declarations."""
    nf_globs = _nf_output_globs('lmm_scan.nf')
    assert nf_glob in nf_globs, (
        f'lmm_scan.nf output: block does not declare {nf_glob!r}.\n'
        f'Declared globs: {nf_globs}'
    )


# ── 3. module anatomy ─────────────────────────────────────────────────────────

# lmm_scan intentionally omits publishDir: per-chr outputs are transient
# intermediates collected by MERGE_RESULTS, which owns the publishDir.
_NO_PUBDIR = {'lmm_scan.nf'}

_ALL_MODULES = sorted(p.name for p in MODULES.glob('*.nf'))


@pytest.mark.parametrize('module', _ALL_MODULES)
def test_module_anatomy(module: str) -> None:
    """Every NF module declares label, tag, stub, and publishDir (except lmm_scan)."""
    checks = _nf_has_anatomy(module)
    required = set(checks.keys())
    if module in _NO_PUBDIR:
        required.discard('publishDir')

    missing = [k for k in required if not checks[k]]
    assert not missing, (
        f'{module} is missing required NF blocks: {missing}\n'
        f'All checks: {checks}'
    )


# ── 4. stub-run smoke test ────────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which('nextflow') is None, reason='nextflow not on PATH')
@pytest.mark.slow
def test_stub_run_exits_zero() -> None:
    """nextflow run main.nf -stub-run exercises the full DAG topology without executing scripts."""
    result = subprocess.run(
        [
            'nextflow', 'run', str(MAIN_NF),
            '-stub-run',
            '-params-file', str(CONFIG),
            '-profile', 'local',   # caps process_long memory to 8 GB (local profile in nextflow.config)
            '--gigamuga_csv', 'NO_GIGAMUGA',
            '--gene_info', 'NO_GENE_INFO',
            '--metadata', 'NO_METADATA',
            '--phenotypes', 'NO_PHENOTYPES',
            '--celltype', 'test_ct',
            '--output_dir', '/tmp/nf_parity_stub_run',
        ],
        capture_output=True, text=True, timeout=180,
        cwd=REPO,
    )
    if result.returncode != 0:
        pytest.fail(
            f'nextflow stub-run failed (exit {result.returncode}):\n'
            f'--- STDOUT (last 3000 chars) ---\n{result.stdout[-3000:]}\n'
            f'--- STDERR (last 3000 chars) ---\n{result.stderr[-3000:]}'
        )

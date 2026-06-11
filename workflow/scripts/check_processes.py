"""Validate that every .nf process file meets mosaiQTL submission requirements.

Checks each process block for:
  --require-label  : at least one `label` directive
  --require-tag    : at least one `tag` directive
  --require-stub   : a `stub:` block present

Usage:
    python workflow/scripts/check_processes.py \\
        --require-label --require-tag --require-stub
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(message)s')
log = logging.getLogger(__name__)

_PROCESS_RE = re.compile(r'process\s+\w+\s*\{', re.MULTILINE)
_LABEL_RE   = re.compile(r'^\s*label\s+', re.MULTILINE)
_TAG_RE     = re.compile(r'^\s*tag\s+', re.MULTILINE)
_STUB_RE    = re.compile(r'^\s*stub\s*:', re.MULTILINE)


def _extract_process_blocks(text: str) -> list[str]:
    """Return each process {...} body as a string."""
    blocks: list[str] = []
    for m in _PROCESS_RE.finditer(text):
        start = m.end()
        depth = 1
        i = start
        while i < len(text) and depth > 0:
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
            i += 1
        blocks.append(text[start:i - 1])
    return blocks


def _check_file(
    nf_path: Path,
    require_label: bool,
    require_tag: bool,
    require_stub: bool,
) -> list[str]:
    """Return list of violation messages for nf_path."""
    text = nf_path.read_text()
    violations: list[str] = []
    for i, block in enumerate(_extract_process_blocks(text), 1):
        prefix = f'{nf_path.name} process#{i}'
        if require_label and not _LABEL_RE.search(block):
            violations.append(f'{prefix}: missing label directive')
        if require_tag and not _TAG_RE.search(block):
            violations.append(f'{prefix}: missing tag directive')
        if require_stub and not _STUB_RE.search(block):
            violations.append(f'{prefix}: missing stub block')
    return violations


@click.command()
@click.option('--nf-dir', default='modules/local', show_default=True,
              type=click.Path(exists=True),
              help='Directory to scan for .nf files.')
@click.option('--require-label/--no-require-label', default=False)
@click.option('--require-tag/--no-require-tag',     default=False)
@click.option('--require-stub/--no-require-stub',   default=False)
def main(
    nf_dir: str,
    require_label: bool,
    require_tag: bool,
    require_stub: bool,
) -> None:
    """Check all .nf process files for required directives."""
    root = Path(nf_dir)
    nf_files = sorted(root.rglob('*.nf'))
    if not nf_files:
        log.warning('no .nf files found in %s', root)
        sys.exit(0)

    all_violations: list[str] = []
    for nf in nf_files:
        violations = _check_file(nf, require_label, require_tag, require_stub)
        for v in violations:
            log.error(v)
        all_violations.extend(violations)

    if all_violations:
        log.error('%d violation(s) found', len(all_violations))
        sys.exit(1)

    log.info('all %d .nf file(s) pass', len(nf_files))


if __name__ == '__main__':
    main()

# Changelog
Format: [Keep a Changelog](https://keepachangelog.com). One line per user-visible change under Added, Changed,
Fixed or Removed, written in the PR that makes it. Releases follow pyproject.toml and are tagged vX.Y.Z.

## [Unreleased]
### Added
- `make bench` / `make bench-trace` (benchmarks/record.py): append-only benchmarks/history.csv and
  pipeline_history.csv; Nextflow trace written to `<output_dir>/logs/trace.txt`.
- `make docs-math`; CHANGELOG.md and a CI check that PRs changing src/ or workflow/ update it.
### Changed
- docs/math/*.md and docs/METHODS.md → `.typ`: LaTeX math typeset by Typst via mitex (docs/math/_setup.typ).
- PR template: documentation and compute checks; Claude review workflow migrated to claude-code-action v1 inputs.
### Fixed
- claude-review.yml had a duplicated `jobs:` key; benchmark commands in benchmarks/README.md were invalid.

## [0.7.0] — not yet tagged
35 commits after v0.6.7-alpha (8 feat, 20 fix, 2 refactor, 5 chore): `git log --oneline v0.6.7-alpha..` until v0.7.0 is tagged.

## [0.6.7-alpha] and earlier
See `git log --oneline v0.6.7-alpha`.

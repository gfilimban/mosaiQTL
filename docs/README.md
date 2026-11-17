# docs/
Start with `ARCHITECTURE.md`. Update the matching file in the same commit as the code it describes.

| File | Contents | Update when |
|---|---|---|
| `ARCHITECTURE.md` | Modules, data flow, glossary | a module, entry point or data flow changes |
| `METHODS.typ` | Statistical methods (LMM, LOCO, permutation/GEV, FDR, NB-GLMM) | a method or default changes |
| `math/NN_*.typ` | Derivations, one file per algorithm (00–12) | before code that implements or changes a formula |
| `DATA_SOURCES.md` | Input data provenance | an input source or reference build changes |
| `REFERENCES.md` | Citations | a method or tool is added |
| `QTL_LITERATURE_OVERVIEW.md` | Background on MPP QTL mapping | as needed |

Math is written in LaTeX and typeset by Typst: `.typ` files import `math/_setup.typ` (mitex) and use
`` #mi(`…`); `` inline and `` #mitex(`…`) `` for display equations. `make docs-math` builds `docs/build/*.pdf`
(gitignored; needs the Typst CLI). GitHub shows `.typ` files as source, not rendered.

Release history: `../CHANGELOG.md`. Benchmarks: `../benchmarks/README.md`.
Local-only (gitignored): AGENTS.md (Claude Code subagents), ANALYSIS_NOTES.md, NEXT_STEPS.md, status and report
files.

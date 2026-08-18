# docs/

Project documentation. Start with `ARCHITECTURE.md`.

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | System architecture, module responsibilities, data flow, glossary |
| `METHODS.md` | Statistical methods: NB-GLMM, LOCO kinship, PQL, score test, GEV thresholds |
| `REFERENCES.md` | Citations for algorithms, tools, and datasets |
| `QTL_LITERATURE_OVERVIEW.md` | Background reading on MPP QTL mapping |
| `math/` | Per-algorithm mathematical derivations (prerequisite before implementation) |

## math/

LaTeX/Markdown derivations for every scan algorithm. A derivation must exist before
code is merged — see DESIGN-03 in `goals/GOALS.md`. Covers: FWL identity, PQL update,
Woodbury matrix lemma applications, LOCO kinship construction, GEV threshold fitting.

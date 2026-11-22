// Shared setup for mosaiQTL math documents: math is written in LaTeX and rendered by Typst via mitex.
// Inline: #mi(`...`);   Display: #mitex(`...`)   Build: make docs-math
#import "@preview/mitex:0.2.6" as mx
// mitex 0.2.6 maps these to symbol names that Typst 0.14 removed; substitute the glyphs before conversion.
#let _fix(it) = {
  let s = if type(it) == str { it } else { it.text }
  for (k, v) in (("\\otimes", "⊗"), ("\\odot", "⊙")) { s = s.replace(k, v) }
  s
}
#let mi(it) = mx.mi(_fix(it))
#let mitex(it) = mx.mitex(_fix(it))
#let mathdoc(body) = {
  set page(paper: "a4", margin: 2cm, numbering: "1")
  set text(size: 10.5pt)
  set par(justify: true)
  show raw.where(block: true): set text(size: 9pt)
  body
}

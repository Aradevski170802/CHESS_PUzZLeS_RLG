# Dissertation draft (LaTeX)

This folder holds the dissertation draft. Its structure follows the chapter plan in the
June 2026 portfolio: 8 chapters, the same subsections, and IEEE-style numeric citations.

## Compile on Overleaf

1. Upload `Dissertation_Overleaf.zip`: **New Project → Upload Project**.
2. Leave Overleaf's defaults as they are: pdfLaTeX compiler, BibTeX, `main.tex` as the main file.
3. Click **Recompile**. The bibliography appears after the second pass, which Overleaf runs automatically.

There was no TeX installation on the development machine, so the project has not been compiled
locally. It is checked statically instead:

```
python dissertation/check_latex.py
```

The checker looks for missing `\input` files, unbalanced braces and environments, citations or
labels that don't exist, missing figures, characters pdfLaTeX can't typeset, and unescaped `%`,
`_` or `#`. A negative test that plants one of each confirms it catches all seven.

## Before you submit

- Search for **`AUTHOR:`**. Each red marker is something only you can supply: supervisor name,
  ethics reference, AI-use statement, personal reflection, and word count.
- **Rewrite the draft in your own words.** It was produced with AI assistance. Declare that use as
  City College's policy requires; a draft statement is in `frontmatter/declaration.tex`.
- Check every number against `eval/research/*.json`. Regenerate them with the commands in
  Appendix A if anything changes.
- The "Honest status" boxes are deliberate. They say where the work deviates from the portfolio
  plan. Examiners respond better to stated deviations than to ones they discover themselves.

## Layout

```
main.tex                 preamble, macros (\authornote, honest/finding boxes), document order
frontmatter/             title, abstract, declaration (+ AI use), acknowledgements
chapters/01..08          Introduction ... Conclusion and Future Work
appendices/              A: reproduction and study protocol, B: categories and constants
references.bib           every entry checked against its publisher or primary source
figures/                 copied from eval/research/
check_latex.py           static checker
```

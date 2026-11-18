# Final report (PNAS template)

The paper for 02805, written in the official PNAS LaTeX template (`pnas-new.cls` v1.47).
Course limit: **6 pages including figures and references**.

## Where to write

| File | Section |
|---|---|
| `main.tex` | Title, authors, keywords, and the order of the sections. Rarely needs edits. |
| `sections/abstract.tex` | Abstract, about 150 words. Write it last. |
| `sections/significance.tex` | Significance statement, 50–120 words. |
| `sections/introduction.tex` | Introduction (no heading in PNAS). |
| `sections/results.tex` | Results, figures and tables. |
| `sections/discussion.tex` | Discussion and limitations. |
| `sections/methods.tex` | Materials and Methods. |
| `sections/contributions.tex` | Who did what (be specific). |
| `references.bib` | BibTeX entries. Cite with `\cite{key}`. |
| `figures/` | Figures used in the paper (PDF or PNG). |

Each section is its own file, so two people can edit different sections at the same time without merge conflicts.

## Build the PDF

### Option A: VS Code (recommended)

1. Install Tectonic, a small self-contained LaTeX engine: `brew install tectonic`.
   The first build downloads the packages it needs (takes about 2 minutes); later builds take a few seconds.
2. Install the **LaTeX Workshop** extension. VS Code suggests it when you open the repo.
3. Open any `.tex` file in `report/` and save. The PDF rebuilds on every save. Open it next to the editor with *View LaTeX PDF* (`⌥⌘V`). Clicking in the PDF jumps to the source line.

The repository's `.vscode/settings.json` already configures LaTeX Workshop to use Tectonic.

### Option B: terminal

```bash
cd report
make          # writes build/main.pdf
make open     # builds and opens the PDF
```

### Option C: Overleaf

Run `make zip`, then in Overleaf choose *New Project → Upload Project* and select `build/overleaf.zip`.
It compiles with the default pdfLaTeX. Overleaf does not sync with this repo automatically, so copy edits back before committing.

## Layout notes

- Line numbers in the margins are part of the course template, so they stay on. To preview the paper without them, set `\showlinenumbersfalse` near the top of `main.tex`, and set it back before handing in.
- On page 1, the text column is narrower because of the Significance box. `\Endparasplit` at the end of `sections/introduction.tex` marks where that narrow column ends. Move it once the introduction is final. Use `\Parasplit` if the page break falls in the middle of a paragraph.
- `\bibsplit[3]` in `main.tex` sets how many references go in the left column of the last page. Adjust it at the end.
- Column-wide figures are 8.7 cm wide (`figure`), and page-wide figures are 17.8 cm (`figure*`). Export figures from matplotlib at their final size so that the text in them is 6–12 pt.

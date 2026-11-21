# Network analysis text for the PNAS research-article template

This folder contains the network-analysis contribution to the group's paper. It is designed for the [official NAS/Overleaf PNAS research-article template (2025)](https://www.overleaf.com/latex/templates/template-for-preparing-your-research-report-submission-to-pnas-using-overleaf-2025/whbdryzwztnd), which uses `pnas-new` and `\templatetype{pnasresearcharticle}`. It is a section draft for the course paper, not a complete journal submission. The source is in English to match the repository and template.

Copy this entire `pnas_network` folder into the root of the group's PNAS Overleaf project. Then:

1. In the manuscript's Results section, add `\input{pnas_network/results}`. The file contains the two Results subsections and two wide figures.
2. In the manuscript's Discussion section, add `\input{pnas_network/discussion}`.
3. Paste the contents of `methods.tex` inside the existing `\matmethods{...}` block. The PNAS template usually has one shared Materials and Methods block for the whole manuscript.
4. Add `pnas_network/references` to the manuscript's bibliography command, or merge the `Majumder2019` entry into the group's `.bib` file. Avoid duplicate keys.
5. Adjust section names, figure numbering, and length after merging with the NLP contribution. Do not add another `\section*{Results}` or `\section*{Discussion}` if these headings already exist.

The two main figures are provided as vector PDFs and high-resolution PNGs. Their PDFs are referenced by `results.tex`; the PNGs are for quick review. The second main figure compares the cuisine-only and cuisine-by-dish-type ingredient randomizations. `pnas_recipe_null_figure.pdf` retains the earlier cuisine-only histograms, and `pnas_residual_sensitivity_figure.pdf` shows the exclusion checks; both are optional detailed figures. Quantitative results come from the executed full-data notebook in `notebooks/03_network_analysis_full.ipynb`, `data/processed/network_cuisine_pairs_full.csv`, and the randomizations in `scripts/recipe_ingredient_null.py` and `scripts/tagged_recipe_null.py`. The tag-based script requires the original Food.com `RAW_recipes.csv.zip` in ignored `data/raw/`; the archive is not included in this paper folder. Manual examples are documented in `docs/recipe_pair_audit.md`. Randomization results and limits are documented in `docs/recipe_ingredient_null.md` and `docs/tagged_recipe_null.md`; exclusion checks are in `docs/network_residual_sensitivity.md`.

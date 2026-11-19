# Ingredient normalization audit — Social Graphs

Source: `EstebanPerez99/cuisine-networks`, commit `ca8ff65`; processed Food.com recipes from notebook 02. The figures below use all 26,727 recipes and nine cuisine labels. This note documents the revised `03_network_analysis.ipynb` settings.

## Decisions

- Merge clear singular/plural and equivalent names, such as `shallots` → `shallot`, `bay leaves` → `bay leaf`, `lemon, juice of` → `lemon juice`, and `vegetable stock` → `vegetable broth`. The curated map has 72 entries. It reduces 7,296 raw names to 7,226 normalized names. A full list with frequencies is in `ingredient_alias_review.csv`.
- Keep distinctions that may matter: fresh versus dried herbs; ground spice versus seeds; red, white, and yellow onion; generic `red pepper` versus `red bell pepper`; and named oils such as olive, sesame, and peanut oil.
- Exclude 20 exact generic names from the recipe projection, including salt, generic pepper, water, unnamed oil, and compound entries such as `salt and pepper`. These appear in 18,629 recipes. Specific ingredients such as `white pepper`, `chili oil`, and `garlic salt` remain eligible.
- Keep ingredients appearing in 3 or more recipes and no more than 35% of recipes. The former removes unreliable rare names; the latter avoids a nearly complete projection while retaining olive oil and onion. The graph also requires at least two shared ingredients, Jaccard similarity ≥ 0.18, and each recipe's 15 strongest neighbors. Cuisine labels are not used to build edges.

## Effect of the choices

| Full-data variant | Within-cuisine edge weight | Isolated recipes | Cuisine/community AMI |
|---|---:|---:|---:|
| Initial aliases; 10% frequency cap; no explicit generic list | 64.0% | 875 | 0.363 |
| Revised aliases; 10% cap; no explicit generic list | 64.7% | 924 | 0.372 |
| Revised aliases and generic list; 10% cap | 66.8% | 1,245 | 0.368 |
| **Revised aliases and generic list; 35% cap (main analysis)** | **63.0%** | **345** | **0.367** |

The main analysis has 309,019 recipe edges. A fixed-graph shuffled-label null gives 17.1% mean within-cuisine edge weight; the one-sided permutation p-value is 0.005 with 199 shuffles (the resolution limit of this test). Varying the Jaccard cutoff to 0.12, 0.18, and 0.24 gives 62.7%, 63.0%, and 64.1% within-cuisine weight on the full data.

The *existence* of cuisine assortativity is stable across these choices. Its exact magnitude and which recipes rank as bridges depend on normalization and filtering. For example, French recipes have 34.8% within-cuisine edge weight under the main settings, versus an 8.2% random-label baseline based on cuisine size.

## Interpretation limits

Cuisine labels come from user tags and can be broad or wrong. The selected recipes omit mixed-cuisine labels, so the bridge ranking should be read as cross-tag ingredient similarity rather than proof of culinary fusion. This audit examined common variants and generic terms, not every low-frequency ingredient name. Review the mapping with the group before quoting individual recipe or ingredient examples in the paper.

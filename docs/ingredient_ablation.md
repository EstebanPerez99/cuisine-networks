# Which ingredients support the recipe network?

Exploratory sensitivity analysis inspired by Ahn et al., *Flavor network and the principles of food pairing* (2011, arXiv:1111.6074). Their flavor-compound question differs from ours: our links measure recipe similarity from shared ingredients. Here we ask whether a small number of ingredients materially affect the patterns already found in that recipe network.

## Method

We used the full Food.com project subset (26,727 recipes, nine cuisine labels). For each ingredient, we removed it from every recipe, then rebuilt the full binary recipe–ingredient matrix and Jaccard recipe network with the original rules: at least two shared retained ingredients, Jaccard similarity at least 0.18, and the union of each recipe's 15 best neighbors. The unmodified reconstruction reproduced all 309,019 previously exported edges and their weights exactly.

Four target ingredients were selected **after inspecting** the original network: vanilla extract and flour for the large mixed community 76, soy sauce for Chinese–Japanese links, and lemon juice for Greek–Middle Eastern links. For each target, we also removed five other eligible ingredients with the nearest recipe frequencies (within 15%). This is a small, descriptive comparison, not a hypothesis test. The original community-76 membership is held fixed to measure links incident to those same recipes after rebuilding.

The outcomes below are fractions of the **total cross-cuisine edge weight** in each rebuilt network. “Community 76” means an edge has at least one endpoint in the original community 76; cuisine pairs mean an edge connects the two named cuisine labels. Percentage-point changes can be positive because removing an ingredient changes Jaccard denominators and which neighbors enter the top-15 network.

## Results

| Removal | Recipes using ingredient | Outcome | Baseline | After removal | Change | Median matched-control change |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Vanilla extract | 385 | Cross-cuisine weight incident to community 76 | 24.93% | 25.21% | +0.29 pp | -0.01 pp |
| Flour | 2,828 | Cross-cuisine weight incident to community 76 | 24.93% | 23.90% | -1.03 pp | -0.18 pp |
| Soy sauce | 1,742 | Chinese–Japanese share | 4.38% | 3.51% | -0.87 pp | -0.02 pp |
| Lemon juice | 2,766 | Greek–Middle Eastern share | 6.85% | 6.14% | -0.71 pp | -0.02 pp |

Each target lies beyond the range of its five frequency-matched controls for its selected outcome. The controls' full ranges of changes are -0.02 to +0.12 pp for vanilla extract, -0.53 to +0.17 pp for flour, -0.06 to +0.02 pp for soy sauce, and -0.25 to +0.13 pp for lemon juice. Individual control ingredients and all network metrics are in `network_ingredient_ablation_full.csv`; the comparison summary is in `network_ingredient_ablation_comparison_full.csv`.

The contrast between vanilla extract and flour is useful. Vanilla extract appears in 377 recipes of community 76, making it a highly specific *marker* of part of that community. Removing it does not weaken the community-level cross-cuisine connection; its absolute incident cross-edge weight rises from 8,998.3 to 9,130.6. Flour appears in 1,000 community-76 recipes and its removal lowers the same weight to 8,470.8. Similarly, soy sauce and lemon juice are substantial contributors to the two selected cuisine-pair links in this particular network construction. The absolute Chinese–Japanese cross-edge weight falls from 1,581.8 to 1,273.5 after removing soy sauce; Greek–Middle Eastern weight falls from 2,471.2 to 2,204.1 after removing lemon juice.

## Interpretation limits

These targets and outcomes were chosen from the same data, and five controls per target are too few for calibrated p-values. Frequency matching does not equalize cuisine specificity or recipe type. Jaccard normalization and the top-15 cap mean removing an ingredient can create or strengthen other links; it is a network perturbation, not a causal estimate of an ingredient's culinary role. Community 76 is not redetected after each removal. The analysis therefore supports **sensitivity and mechanism hypotheses**. A subsequent recipe–ingredient null preserving recipe sizes and within-cuisine ingredient frequencies is reported in [`recipe_ingredient_null.md`](recipe_ingredient_null.md).

## Reproduce

Run `scripts/ingredient_ablation.py` from the repository environment after the full network notebook has exported `data/processed/network_edges_full.csv` and `network_recipe_metrics_full.csv`. It uses the notebook's alias map and stoplist, checks exact baseline reproduction, then writes the two tables and Figure 14.

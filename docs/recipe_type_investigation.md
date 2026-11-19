# Recipe type and cuisine structure: exploratory analysis

## Question and design

Does recipe type explain some of the cross-cuisine links in the ingredient-similarity network? We hold the full network fixed (26,727 recipes, 309,019 edges), assign a conservative type from explicit recipe-title words, and compare the observed within-cuisine edge-weight share with 999 cuisine-label shuffles. The type-preserving shuffle permutes cuisine labels only among recipes assigned the same title type, including the large `other/unclear` group. This preserves each type's cuisine composition. We also remove named sweet/drink recipes and the previously detected mixed community 76, then repeat the fixed-graph comparison on each induced subgraph. None of these operations changes the original ingredient-similarity threshold.

## Findings

| Measure | Result |
| --- | ---: |
| Recipes with an explicit drink, dessert/sweet, soup/stew, or salad cue | 4,201 / 26,727 (15.7%) |
| Recipes in mixed community 76 | 3,062 |
| Explicitly named dessert/sweet recipes in community 76 | 909 / 1,032 (88.1%) |
| Cross-cuisine edge weight incident to community 76 | 24.9% |
| Observed within-cuisine share of all edge weight | 63.0% |
| Mean under global cuisine-label shuffling | 17.1% |
| Mean under title-type-preserving cuisine-label shuffling | 17.3% |
| Observed share after excluding named sweet/drink recipes | 65.3% |
| Observed share after excluding community 76 | 67.7% |

The type-preserving permutation changes the null mean by only 0.14 percentage points because the title rules leave 84.3% of recipes as `other/unclear`. The result **does not establish that dish type is unimportant**. It establishes that the four *explicitly named* types, as currently captured, do not explain the very large cuisine-mixing signal. The observed 63.0% exceeds both null distributions (one-sided permutation p = 0.001, the minimum attainable with 999 shuffles).

Community 76 has a sweet-ingredient signature and contains most of the recipes explicitly named as desserts. Its cross-cuisine mixing lowers the network-wide within-cuisine share: the share rises from 63.0% to 67.7% when the entire community is excluded. It is responsible for about one quarter of observed cross-cuisine edge weight. The three strongest illustrative pair affinities remain, and increase, after that exclusion:

| Cuisine pair | Full graph lift | Without community 76 |
| --- | ---: | ---: |
| Chinese–Japanese | 8.54× | 10.02× |
| Greek–Middle Eastern | 4.79× | 5.96× |
| French–Italian | 2.73× | 3.15× |

The lift here is the pair's observed fraction of *cross-cuisine* edge weight divided by its mean fraction under a global cuisine-label shuffle on the corresponding fixed graph. It is a relative affinity measure, conditional on the graph and set of cuisines.

## Limits and next test

The title rules were designed for precision, not completeness; names such as “kheer” or “tres leches” may lack any rule word. They also make occasional errors (for example a savory “cake”). The audit sample contains 25 titles from each class for review. Community 76 was found using the same ingredient graph, so removing it is a sensitivity description, not an independent confirmation of a causal mechanism. This was the initial analysis before the original Food.com tags became available. The follow-up in [`tagged_recipe_null.md`](tagged_recipe_null.md) uses those source tags to define broader dish types and rebuilds the network under a cuisine-by-type ingredient randomization.

## Reproduce

Run `notebooks/03_network_analysis.ipynb` with `DATASET = 'full'` to export `data/processed/network_edges_full.csv` and `network_recipe_metrics_full.csv`. Then run `scripts/recipe_type_analysis.py` using the repository's Python environment. Its outputs are the type assignment, audit sample, null and sensitivity tables, pair tables, community 76 contribution table, and Figure 13 in `figures/`.

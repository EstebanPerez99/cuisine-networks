# Direct recipe–ingredient network check

The project proposal includes a bipartite recipe–ingredient network. Notebook 03 builds its binary adjacency matrix, but the first report draft focused on its recipe–recipe projection. We added a direct check of the bipartite graph and a descriptive ingredient co-occurrence projection. These use the same full dataset and the same ingredient normalization and filters as the existing network analysis.

## Bipartite graph

There are **26,727 recipe nodes**, **3,877 retained ingredient nodes**, and **227,965 recipe–ingredient edges**. An edge means that a recipe lists an ingredient. The median recipe has eight retained ingredients; the median ingredient occurs in ten recipes. Every recipe has at least one retained ingredient. We keep recipes with fewer than two retained ingredients in the bipartite graph even though they cannot form a recipe–recipe link under the main projection rule.

For each ingredient, we calculated the entropy of its recipe links across the nine cuisine labels and divided by `log(9)`. We then averaged ingredient entropies, weighting each ingredient by its number of recipe links. This asks whether a typical recipe–ingredient edge touches an ingredient used across cuisines or concentrated in a few. The observed value was **0.603**. In 199 shuffles of recipe cuisine labels on the fixed bipartite graph, the mean was **0.861** (central 95% reference interval **0.859–0.862**). None of the shuffled values was as low as observed, giving a lower-tail permutation value of **0.005** with a one-count correction.

The result shows that ingredient use is strongly associated with the cuisine labels. It does not establish that cuisines cause particular ingredients to be used. Cuisine labels come from user tags, and the dataset has unequal cuisine sizes. The label shuffle preserves those sizes but does not preserve cuisine-specific ingredient frequencies. The later Curveball randomization *does* preserve those frequencies, so it holds this entropy measure fixed while testing whether the particular combinations within recipes add structure to the recipe projection. This makes the two analyses complementary rather than independent confirmations of the same claim.

## Ingredient co-occurrence

Two ingredient nodes are linked in the ingredient projection if they occur together in at least one recipe. The full bipartite matrix produces **271,443 distinct co-occurring ingredient pairs**. Raw co-occurrence counts favor common ingredients, so for illustrative pairs we require at least 30 shared recipes and rank by ingredient Jaccard similarity. Examples are garlic paste–ginger paste (79 shared recipes, Jaccard 0.612), chili powder–cumin (831, 0.231), sesame oil–soy sauce (476, 0.227), and cilantro–lime juice (774, 0.213). These are descriptive patterns, not evidence of cultural influence or independent associations after controlling for ingredient frequencies.

## Reproduce

Set `DATASET = 'full'` in `notebooks/03_network_analysis.ipynb` and run through section 2a. The notebook writes `data/processed/network_bipartite_summary_full.csv` and `data/processed/network_ingredient_pairs_full.csv`. The generated tables are ignored by Git. The recipe projection rules and the stricter ingredient randomizations remain unchanged.

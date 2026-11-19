# How much cuisine structure remains after preserving ingredient frequencies?

## Question and reference model

The original fixed-network cuisine-label shuffle put only 17.1% of edge weight within cuisines, versus 63.0% in the observed recipe network. That comparison establishes an association, but it does not separate cuisine-specific ingredient popularity from the way ingredients are combined into recipes. Inspired by the ingredient-level null models in Ahn et al., *Flavor network and the principles of food pairing* (2011, arXiv:1111.6074), we tested a stricter reference for our different, recipe-similarity question.

We independently made 49 randomized recipe–ingredient matrices. In each, Curveball trades exchanged ingredients between pairs of recipes **within the same cuisine**. This preserves exactly (i) the number of retained ingredients in every recipe and (ii) the number of recipes using each retained ingredient within each cuisine. Each replicate used ten attempted trades per recipe, or 267,270 attempts in total. On average, 266,817 attempts involved two recipes with exchangeable ingredients; an eligible exchange can still randomly return the original allocation. We then rebuilt the binary Jaccard recipe network, requiring two shared ingredients and Jaccard similarity at least 0.18 and keeping the union of each recipe's 15 best neighbors. The original reconstruction reproduced all 309,019 exported edges and weights exactly. Recipe sizes and each cuisine-specific ingredient frequency were verified after every randomization.

Each randomized network changed effectively all ingredient sets and retained a median 10.8% of original recipe–ingredient incidences. A small mixing check with five runs at 20 attempted trades per recipe gave nearly the same within-cuisine median (59.26% versus 59.27% at ten trades); selected pair medians were also close. Two shorter five-trade runs had somewhat different pair shares, so ten trades was used for the main comparison. This check supports stability at the reported scale but is not a proof of perfect mixing.

## Results

| Outcome | Observed | Randomized median | Central 95% range of randomized networks | Observed minus median |
| --- | ---: | ---: | ---: | ---: |
| Within-cuisine share of **all** edge weight | 62.96% | 59.27% | 58.75–59.78% | +3.69 pp |
| Chinese–Japanese share of **cross-cuisine** edge weight | 4.38% | 5.00% | 4.77–5.18% | −0.62 pp |
| Greek–Middle Eastern share of **cross-cuisine** edge weight | 6.85% | 6.28% | 6.00–6.56% | +0.56 pp |
| French–Italian share of **cross-cuisine** edge weight | 14.56% | 14.11% | 13.64–14.73% | +0.45 pp |

None of the 49 randomized networks reached the observed within-cuisine share, giving a one-sided Monte Carlo $p=(0+1)/(49+1)=0.02$. The same upper-tail count applies to Greek–Middle Eastern share. All 49 randomized Chinese–Japanese shares were *above* the observed share; its upper-tail $p$ is 1.00, so the earlier 8.54-fold enrichment against a cuisine-label shuffle does not imply an excess beyond cuisine-specific ingredient frequencies. French–Italian lies within the randomized range (upper-tail $p=0.16$). The three pairs were selected after examining the original network; their tail counts are exploratory, not multiplicity-adjusted confirmation.

This changes the reading of the network. Most of the large gap between 63.0% observed and 17.1% under cuisine-label shuffling is compatible with the cuisines' different ingredient frequencies. A smaller, clear residual remains when those frequencies and recipe lengths are fixed: ingredient combinations carry additional cuisine structure. The Greek–Middle Eastern connection also exceeds this stricter reference in the sampled networks; Chinese–Japanese does not. These conditional pair shares can move when the total cross-cuisine edge weight changes, so they should be read as network composition, not direct counts of cultural exchange.

## Limits and reproducibility

This null preserves ingredient frequency within cuisine, but it breaks recipe types, ingredient compatibility, and near-duplicate recipe families. The residual can therefore reflect dish composition or dataset conventions as well as cuisine-specific combinations. Food.com cuisine tags are user supplied, and the nine-cuisine subset is not a representative sample of all food cultures. Forty-nine replicates give coarse Monte Carlo resolution, with minimum possible one-sided $p=0.02$; the central range is a reference distribution, not a confidence interval for a population effect. This procedure adapts the null-model idea from Ahn et al.; it does not reproduce their flavor-compound analysis.

A follow-up exclusion check in [`network_residual_sensitivity.md`](network_residual_sensitivity.md) shows that the residual persists after removing explicitly named sweet/drink recipes and very high-similarity edges. The stronger dish-type adjustment in [`tagged_recipe_null.md`](tagged_recipe_null.md) uses the original Food.com tags to restrict ingredient exchanges to recipes with both the same cuisine and primary type.

Run `scripts/recipe_ingredient_null.py --replicates 49 --trades-per-recipe 10 --tag full` after the full notebook has exported `data/processed/network_edges_full.csv`. The output is `data/processed/network_recipe_ingredient_null_full.csv` and Figure 15 in `figures/`. A five-replicate diagnostic at 20 trades per recipe is in `data/processed/network_recipe_ingredient_null_pilot20.csv`. Both tables contain the random seeds, network sizes, margin-preserving trade counts, and observed and randomized statistics.

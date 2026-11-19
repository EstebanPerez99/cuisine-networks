# Are the displayed cuisine pairs unusual combinations?

The recipe score in [`recipe_pair_distinctiveness.md`](recipe_pair_distinctiveness.md) uses a pair's prevalence in one cuisine minus its mean prevalence elsewhere. A high contrast can reflect the popularity of the two ingredients separately. We checked whether each of the nine pairs displayed in the report occurs together more or less often than expected when ingredient frequencies are kept fixed.

## Reference model

We selected the pair with the largest observed cross-cuisine prevalence difference in each cuisine among pairs occurring in at least 20 recipes there. This selection happened **before** the check, so the nine pairs are examples, not pre-registered tests. We generated 49 independent Curveball randomizations of the recipe–ingredient matrix. Trades occur only between recipes with the same cuisine and primary Food.com course type; every recipe retains its ingredient count, and each ingredient retains its frequency within each cuisine-by-type group. We verified those margins after every run. The primary types and their audit are documented in [`tagged_recipe_null.md`](tagged_recipe_null.md).

For each selected pair, we counted recipes in its named cuisine containing both ingredients. The middle 95% of the 49 randomized counts is a descriptive reference range. Because the pairs were selected as observed maxima, these ranges are **not selection-adjusted confidence intervals** and their tail counts should not be read as confirmatory p-values.

## Results

| Cuisine | Pair | Observed count | Null median | Middle 95% null range | Observed / median |
| --- | --- | ---: | ---: | ---: | ---: |
| Chinese | green onion + soy sauce | 465 | 431 | 413–448 | 1.08 |
| French | butter + flour | 308 | 222 | 208–240 | 1.39 |
| Greek | feta cheese + olive oil | 464 | 483 | 463–497 | 0.96 |
| Indian | onion + turmeric | 612 | 561 | 541–583 | 1.09 |
| Italian | olive oil + parmesan cheese | 1,277 | 1,437 | 1,393–1,465 | 0.89 |
| Japanese | soy sauce + sugar | 144 | 127 | 115–138 | 1.13 |
| Mexican | chili powder + cumin | 697 | 353 | 334–377 | 1.97 |
| Middle Eastern | lemon juice + olive oil | 317 | 263 | 243–281 | 1.21 |
| Thai | fish sauce + lime juice | 188 | 156 | 144–170 | 1.21 |

Seven observed counts exceed their individual central reference ranges. The Greek count lies inside its range. The Italian count falls below its range. Mexican chili powder + cumin appears almost twice as often as the null median, despite preserving the frequency of each ingredient within Mexican-tagged recipes and broad dish types.

As a mixing diagnostic, five additional randomizations with 20 rather than ten attempted trades per recipe gave similar medians for the two clearest contrasts: 349 for Mexican chili powder + cumin (main null: 353) and 1,436 for Italian olive oil + parmesan (main null: 1,437). All nine observed counts were on the same side of their diagnostic medians as in the main run. Five runs are too few to establish full mixing, but the direction of these examples did not depend on doubling the trade count.

This sharpens the interpretation of the main score. An ingredient pair can be *distinctive across cuisine labels* without being *unusually combined within its cuisine*. The Italian example is the clearest case. The recipe score remains a useful measure of cross-cuisine ingredient signatures, but it should not be presented as a pure interaction score. In the later review analysis, the ingredient-only score is a necessary baseline.

## Reproduce

Run `python scripts/recipe_pair_distinctiveness.py --dataset full` to recreate the selected-pair table, then `python scripts/pair_cooccurrence_null.py --replicates 49` with the user-supplied `data/raw/RAW_recipes.csv.zip` present. The latter writes `data/processed/network_selected_pair_null_full.csv`; `python scripts/plot_pair_null.py` creates Figure 19 and the report figure. The raw archive and generated CSV remain ignored by Git. The fixed seed is 20261119, and each replicate uses ten attempted trades per recipe. The code checks that observed counts match the exported pair table before randomizing. For the diagnostic, use `--replicates 5 --trades-per-recipe 20 --tag pilot20`. The null does not preserve exact dish families, preparation techniques, or ingredient compatibility.

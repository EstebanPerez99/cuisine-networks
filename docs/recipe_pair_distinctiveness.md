# Ingredient-network scores for the authenticity question

**Question for the later joint analysis:** Are recipes with ingredient combinations distinctive of their Food.com cuisine tag more likely to be described as authentic by reviewers?

## Network measure

We use the same binary recipe–ingredient graph and cleaning as notebook 03: 26,727 recipes, 3,877 retained ingredient names, and 227,965 recipe–ingredient links. The ingredient projection contains 271,443 observed pairs. Following the relative-prevalence idea in Ahn et al. (2011), for pair `(i,j)` and cuisine `c` we calculate the share of recipes in `c` containing both, minus the **equal-weight mean** of that share in the other eight cuisines. This is *pair distinctiveness*, not a cultural or historical definition of authenticity.

For each recipe, we average the contrasts of all its retained ingredient pairs. We exclude that recipe from its own cuisine's counts and denominator before scoring it; otherwise rare pairs would gain a spurious positive score merely because the scored recipe contains them. Averaging keeps recipes with longer ingredient lists from automatically scoring higher. The companion **ingredient-only score** applies the same calculation to individual ingredients. We will compare the two in the NLP analysis because pair scores can reflect the frequency of their individual ingredients.

The exact formula for recipe `r` with cuisine `c` and pair set `Q_r` is:

`S_pair(r) = mean_{(i,j) in Q_r} [ (n_cij - 1)/(N_c - 1) - mean_{d != c}(n_dij/N_d) ]`.

The subtraction of one applies to every pair in the scored recipe. `S_ingredient` replaces pairs with individual ingredients. A score is missing if a recipe has no retained pair. A negative score means its pairs are more prevalent, on average, in the comparison cuisines. Scores are proportions, so 0.01 corresponds to a one-percentage-point prevalence difference averaged over the recipe's pairs.

## Full-data checks

The score is available for **26,622 of 26,727 recipes**; 105 have fewer than two retained ingredients. The overall correlation between recipe pair and ingredient-only scores is **0.852** (within-cuisine correlations range from 0.757 to 0.901), so any later model should check whether pairs add information beyond individual ingredients. The pooled pair score has essentially no raw linear association with the number of retained ingredients (Pearson `r = 0.0002`), consistent with the average rather than a sum. Within-cuisine correlations with recipe length are sometimes larger, so length should also be checked in the review analysis.

Examples of supported pair signatures (at least 20 recipes in the named cuisine):

| Cuisine | Pair | Share in cuisine | Mean share elsewhere | Difference |
|---|---|---:|---:|---:|
| Chinese | green onion + soy sauce | 24.13% | 2.87% | 21.26 pp |
| Greek | feta cheese + olive oil | 21.26% | 0.41% | 20.85 pp |
| Indian | onion + turmeric | 23.03% | 0.95% | 22.08 pp |
| Italian | olive oil + parmesan cheese | 17.69% | 0.91% | 16.78 pp |
| Thai | fish sauce + lime juice | 16.04% | 0.04% | 16.00 pp |

These examples describe the Food.com sample and its tags. They were selected after inspecting the scores and do not establish culinary origin or cultural authenticity. Pair prevalence can be high because both ingredients are common in the cuisine, even without a special pairing preference. Our earlier ingredient-preserving randomizations found that ingredient frequencies explain much of the recipe network's cuisine separation.

## Reproduce and use

Run `python scripts/recipe_pair_distinctiveness.py --dataset full` from the repository root. Notebook 03 section 2b also computes the scores from its bipartite matrix. The checked-in `data/processed/network_recipe_distinctiveness_full.csv` has one row per recipe and can join to `reviews.csv` on `recipe_id`; `network_pair_signatures_full.csv` contains supported pair contrasts. The latter table uses full-sample contrasts for interpretation; the recipe scores use leave-one-out contrasts.

For the review comparison, a missing mention of “authentic” must remain **unlabeled**, rather than being called inauthentic. Affirming and denying mentions need separate text labels. Review volume, cuisine, dish type, recipe length, and the single-ingredient score should be accounted for. General sentiment and star ratings are related outcomes, not synonyms for perceived authenticity. The current exports contain no NLP result.

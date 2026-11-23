# Pairing score ("how")

**Question.** The ingredient and pair scores (Ahn et al.) measure *which* ingredients a recipe uses, and they are strongly correlated (r = 0.85). The pairing score asks something different: does the recipe combine its ingredients more often than their popularity in its cuisine predicts?

**Definition.** For cuisine *c* and pair (*i*, *j*), let *n* be the number of recipes in *c* that contain both ingredients. Let *E* be the mean of that count over 50 Curveball randomizations of the recipe–ingredient matrix within each cuisine. The randomizations keep every recipe's number of ingredients and every ingredient's frequency in the cuisine; this is the cuisine-only null of `scripts/recipe_ingredient_null.py`. The pair lift is log2((n + 1) / (E + 1)). A recipe's pairing score is the mean lift over its retained ingredient pairs, with leave-one-out: the observed count drops by one, and *E* is rescaled by (n_i − 1)(n_j − 1) / (n_i n_j).

Code: `pairing_scores()` in `src/cuisine_distinctiveness.py`, tested in `tests/`. Run with `python scripts/recipe_pairing_score.py` (about 3 minutes). It adds the column `pairing_score` to `data/processed/network_recipe_distinctiveness_full.csv`, and the columns `expected_in_cuisine` and `pairing_lift` to `data/processed/network_pair_signatures_full.csv`.

## Results (50 randomizations, seed 20261123)

- 26,622 of 26,727 recipes scored; the other 105 have fewer than two retained ingredients. Median 0.17; 79% of recipes are above 0. Recipes are built from recurring combinations, so most pairs co-occur somewhat above this null.
- **Almost independent of the other two scores:** Pearson r = −0.14 with the ingredient score and −0.02 with the pair score. The pairing score carries information that the Ahn-style scores do not.
- Showcase pairs, observed vs. expected: Mexican chili powder + cumin 697 vs 307 (×2.27); French butter + flour ×1.51; Middle Eastern lemon juice + olive oil ×1.39; Indian onion + turmeric ×1.30; Thai fish sauce + lime juice ×1.26; Chinese green onion + soy sauce ×1.12; Japanese soy sauce + sugar ×1.10; Greek feta + olive oil ×1.09; Italian olive oil + parmesan ×0.96. The directions agree with the stricter cuisine-by-dish-type null in `docs/pair_cooccurrence_null.md`, except Greek feta + olive oil, which falls just below that null (0.96).
- The strongest pairings are recipe templates: tiramisù (ladyfingers + mascarpone), margaritas (tequila + triple sec), mirepoix (carrot + celery), Indian garlic + ginger paste, Thai galangal + lemongrass, Japanese dashi + mirin. Some are generic or Americanized: baking powder + baking soda in several cuisines, and cream of chicken + cream of mushroom soup in Mexican-tagged casseroles.

## Limits

- The pairing score measures coherent combinations *within* a cuisine, not whether those combinations belong to that cuisine. A Mexican-tagged casserole with two canned soups scores high. The analysis with reviews should therefore test the pairing score alongside the ingredient score, and their interaction (characteristic ingredients that are also combined characteristically), not instead of it.
- The null keeps cuisine but not dish type, so baking pairs gain lift because desserts are a recognisable family. The stricter dish-type null needs the raw tag file (`scripts/tagged_recipe_null.py`).
- 50 randomizations estimate *E* well for common pairs. For rare pairs the +1 smoothing pulls the lift toward 0.

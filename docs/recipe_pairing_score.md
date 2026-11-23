# Pairing score ("how")

**Question.** The ingredient and pair scores (Ahn et al.) measure *which* ingredients a recipe uses, and they are strongly correlated (r = 0.85). The pairing score asks something different: does the recipe combine its ingredients more often than their popularity in its cuisine predicts?

**Definition.** For cuisine *c* and pair (*i*, *j*), let *n* be the number of other recipes in *c* that contain both ingredients (leave-one-out). Let *E* be the mean of that count over 50 Curveball randomizations of the recipe–ingredient matrix within each cuisine. The randomizations keep every recipe's number of ingredients and every ingredient's frequency in the cuisine; *E* is rescaled by (n_i − 1)(n_j − 1) / (n_i n_j) for leave-one-out. The pair lift is

    lift = log2((n + k) / (E + k)),   with pseudocount k = 5.

A recipe's pairing score is the mean lift over its retained ingredient pairs.

Code: `pairing_scores()` in `src/cuisine_distinctiveness.py`, tested in `tests/`. Run with `python scripts/recipe_pairing_score.py` (about 3 minutes). It writes these columns to `data/processed/network_recipe_distinctiveness_full.csv`:
- `pairing_score` (k = 5), the main score;
- `pairing_score_k2` and `pairing_score_k10`, robustness checks.

It also adds `expected_in_cuisine` and `pairing_lift` (k = 5) to `data/processed/network_pair_signatures_full.csv`.

## Why k = 5: rare pairs and split-half reliability

34% of all recipe pairs have less than 1 expected co-occurrence in their cuisine, and in a quarter of the recipes most pairs are that rare. For such pairs, whether the pair shows up in 0, 1 or 2 other recipes is luck. With k = 1, "1 recipe vs. 0 expected" gets lift +1.0, as large as "twice as often as chance" for a pair seen in hundreds of recipes. A larger k shrinks rare pairs toward 0 ("we don't know") and barely moves common pairs. This is the same idea as smoothing PMI for rare words (course week 6).

To choose k we measured **split-half reliability** with `scripts/pairing_score_reliability.py`. We split each cuisine's recipes at random into two halves and score every recipe twice: once with counts from half A only, once from half B only. A score that measures a real property of the recipe gives similar values in both halves; a score driven by small-count noise does not. Spearman–Brown, 2r / (1 + r), converts the half-data correlation into the reliability expected with all the data.

| k | split-half r | reliability with all data |
|---|---|---|
| 1 | 0.700 | 0.82 |
| 2 | 0.759 | 0.86 |
| **5** | **0.822** | **0.90** |
| 10 | 0.856 | 0.92 |

Results are in `docs/pairing_score_reliability.csv`; the paired half scores are in `docs/pairing_score_split_halves.csv`. Larger k is always more reliable, but it also shrinks real signal toward 0, so we take k = 5 and report k = 2 and k = 10 as robustness checks. The three versions correlate at r = 0.98 (k = 2) and r = 0.99 (k = 10) with k = 5.

Two alternatives were tested and not chosen:
- Averaging only pairs with E ≥ 5 (split-half r = 0.83) drops 1,689 recipes and needs an arbitrary cut. Any such filter must use *E*, never the observed count, which would select pairs that happened to be high.
- log2 of the summed observed over summed expected counts (r = 0.79) correlates more with the ingredient score (r = −0.31).

## Results (k = 5, 50 randomizations, seed 20261123)

- 26,622 of 26,727 recipes scored; the other 105 have fewer than two retained ingredients.
- **Independent of the other two scores:** Pearson r = −0.07 with the ingredient score and 0.06 with the pair score.
- Showcase pairs, observed vs. expected: Mexican chili powder + cumin 697 vs 307 (×2.27); French butter + flour ×1.51; Middle Eastern lemon juice + olive oil ×1.39; Indian onion + turmeric ×1.30; Thai fish sauce + lime juice ×1.26; Chinese green onion + soy sauce ×1.12; Japanese soy sauce + sugar ×1.10; Greek feta + olive oil ×1.09; Italian olive oil + parmesan ×0.96. These directions agree with the stricter cuisine-by-dish-type null in `docs/pair_cooccurrence_null.md`, except Greek feta + olive oil, which falls just below that null (0.96).
- The strongest pairings are recipe templates:
  - Italian carrot + celery (soffritto): 184 vs 19.
  - Indian garlic + ginger paste: 76 vs 4.0.
  - Mexican tequila + triple sec (margaritas): 45 vs 1.4.
  - French bay leaf + thyme: 39 vs 7.1.
  - Thai galangal + lemongrass: 24 vs 4.4.
- Some strong templates are generic or Americanized: garlic powder + onion powder in several cuisines, and canned cream soups in Mexican-tagged casseroles.

## Limits

- The pairing score measures coherent combinations *within* a cuisine, not whether those combinations belong to that cuisine. The comparison with reviews should test it alongside the ingredient score, and their interaction, not instead of it.
- The null keeps cuisine but not dish type, so baking pairs gain lift because desserts form a recognisable family.

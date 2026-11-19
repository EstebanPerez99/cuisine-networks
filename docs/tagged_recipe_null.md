# Food.com dish tags and the recipe–ingredient null

## Question and source

The previous recipe–ingredient null preserved each recipe's retained-ingredient count and each ingredient's frequency within a cuisine, but its exchanges could mix desserts, drinks, soups, and main dishes. We asked whether the remaining within-cuisine network signal was explained by this dish composition. The user supplied the original Food.com `RAW_recipes.csv.zip`; its `RAW_recipes.csv` contains a `tags` field absent from the four project datasets. We joined it by recipe ID and matched all 26,727 project recipes to unique raw records. The original archive is kept in ignored `data/raw/` and is not part of the Git repository.

## Dish-type assignment and audit

We parsed each tag list and assigned one **primary** dish type with this fixed precedence: beverages; desserts; soups/stews; salads; breads; breakfast or brunch; appetizers or snacks; side dishes; condiments; main dishes; otherwise `other/unclear`. This is a transparent partition for the null model, not a claim that each recipe has only one use. The priority resolves overlapping tags, such as a recipe tagged both `main-dish` and `appetizers`.

| Primary type | Recipes |
|---|---:|
| Main dish | 10,055 |
| Appetizer/snack | 3,019 |
| Side dish | 2,464 |
| Dessert | 1,971 |
| Salad | 1,778 |
| Other/unclear | 1,628 |
| Soup/stew | 1,539 |
| Breakfast/brunch | 1,526 |
| Condiment/sauce | 1,277 |
| Bread | 745 |
| Beverage | 725 |

Thus 25,099 recipes (93.9%) received at least one of these source course tags, while 1,628 (6.1%) remained unclear. A total of 5,679 recipes (21.2%) carried more than one course tag; priority can misrepresent some of them. For a limited consistency check, the assigned type agreed with the earlier explicit title-word classifier on 3,662 of 4,201 title-labeled recipes (87.2%). That classifier is itself noisy and incomplete, so agreement is not accuracy. The audit sample includes ten cases per primary type and thirty title/tag conflicts. It is a review set, not a random estimate of tag error.

The cuisine-by-type table has 99 nonempty strata. The smallest has two recipes, and only two recipes fall in strata with five or fewer members. Counts and per-recipe assignments are exported for inspection.

## Null model

We use the same 3,877 retained ingredient types and the same binary recipe–ingredient matrix as the main network. Within each **cuisine × primary-dish-type** stratum, Curveball exchanges ingredient identities between pairs of recipes, with ten attempted trades per recipe in each replicate. This exactly preserves every recipe's retained-ingredient count and each ingredient's occurrence count within every cuisine and dish-type stratum. We independently randomize from the original matrix, rebuild the Jaccard/top-15 network with unchanged thresholds, and recalculate the fraction of total edge weight joining equal cuisine labels and three previously chosen cross-cuisine pair shares. The script verifies the observed baseline edge for edge and checks all row and stratum ingredient-frequency margins after each replicate.

We compare 49 such randomizations with the earlier 49 cuisine-only randomizations. The two ensembles use different random seeds; the comparison is between their reference distributions, not paired realizations. The empirical one-sided upper-tail $p$ includes one pseudocount and has minimum attainable value 0.02. Central 95% randomization ranges describe the generated reference ensemble and are not population confidence intervals.

## Results

| Edge-weight share | Observed | Cuisine-only null median [central 95%] | Cuisine × type null median [central 95%] |
|---|---:|---:|---:|
| Within cuisine, of all edges | 62.96% | 59.27% [58.75–59.78] | 59.04% [58.70–59.41] |
| Chinese–Japanese, of cross-cuisine edges | 4.38% | 5.00% [4.77–5.18] | 4.62% [4.46–4.88] |
| Greek–Middle Eastern, of cross-cuisine edges | 6.85% | 6.28% [6.00–6.56] | 6.13% [5.88–6.36] |
| French–Italian, of cross-cuisine edges | 14.56% | 14.11% [13.64–14.73] | 14.37% [13.89–15.06] |

The observed within-cuisine share exceeded all 49 cuisine-by-type randomizations (one-sided Monte Carlo $p=0.02$). The type-stratified null median was close to the cuisine-only median, leaving a 3.92-percentage-point observed-minus-null gap. Preserving the coarse course categories therefore did **not** explain the residual in this network. This does not identify the source of the gap; particular ingredient combinations, preparation styles, recipe families, and remaining tagging errors are all possible contributors.

For the previously selected pairs, Chinese–Japanese share fell below all 49 type-stratified randomizations (one-sided lower-tail $p=0.02$), while Greek–Middle Eastern share exceeded all 49 (upper-tail $p=0.02$). French–Italian share fell inside its central randomization range (upper-tail $p=0.18$). These are conditional shares of total *cross-cuisine* edge weight, not the probability that two randomly chosen recipes are linked. The pairs were selected after inspecting the data, so these comparisons are exploratory rather than confirmatory discoveries. Figure 17 plots both null ensembles and the observed statistics.

Every type-stratified replicate changed more than 99.9% of recipe ingredient sets while retaining about 14.0% of the original ingredient incidences, compared with 10.8% in the cuisine-only randomizations. The randomized networks had a median of 313,216 edges (observed: 309,019); the same threshold and top-15 procedure was reapplied to every matrix.

Two small mixing and grouping checks gave similar reference medians. Five new randomizations with 20 attempted trades per recipe, rather than ten, had a 59.10% median within-cuisine share. Five randomizations grouping recipes by their *complete combination* of course tags, rather than the priority-based primary type, had a 59.13% median. The latter used 823 cuisine-by-tag-combination strata, including 274 singletons that could not trade; nevertheless, about 99.0% of recipes changed ingredient sets and only 16.5% of original ingredient incidences remained. These five-run checks are diagnostics, not precise tests or calibrated uncertainty estimates.

## Limits

Course tags are user supplied, overlapping, and sometimes ambiguous. A single priority-based label cannot capture multiple courses, preparation methods, or cultural variants. The stronger null still breaks recipe-specific ingredient compatibility and near-duplicate recipe families, and a residual therefore cannot be attributed uniquely to cuisine-specific culinary rules. We selected the three cuisine pairs after viewing the original network, so their comparisons are exploratory. The analysis concerns the Food.com sample and the chosen network construction, not all recipes or food cultures.

## Reproduction and outputs

After generating the full network exports, place the supplied `RAW_recipes.csv.zip` in `data/raw/` and run `scripts/tagged_recipe_null.py --replicates 49 --trades-per-recipe 10 --tag full`. The script writes `network_recipe_dish_types_full.csv`, `network_raw_tag_frequency_full.csv`, `network_dish_type_cuisine_counts_full.csv`, `network_dish_type_audit_sample_full.csv`, and `network_tag_stratified_null_full.csv` under `data/processed/`, plus Figure 17 under `figures/`. To reproduce the diagnostics, run the same script with `--replicates 5 --trades-per-recipe 20 --tag pilot20` and with `--replicates 5 --group-by all-tags --tag all_tags_pilot5`. It does not alter the raw archive.

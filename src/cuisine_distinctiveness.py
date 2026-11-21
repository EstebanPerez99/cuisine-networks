"""Cuisine-specific ingredient and ingredient-pair scores from a bipartite graph."""

from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd


def score_recipes(recipe_ids, labels, ingredient_rows, ingredient_names, min_pair_support=20):
    """Return recipe scores and supported pair signatures.

    The prevalence contrast follows Ahn et al. (2011): prevalence in a cuisine
    minus the equal-weight mean prevalence in the other cuisines. Each scored
    recipe is removed from its own cuisine's numerator and denominator. Pair
    scores are averaged over all retained pairs, so recipe length does not
    mechanically increase the score. Ingredient scores are a separate baseline.
    """
    recipe_ids = np.asarray(recipe_ids)
    labels = np.asarray(labels)
    cuisines = sorted(set(labels))
    if len(cuisines) < 2:
        raise ValueError("At least two cuisines are needed")
    if len(recipe_ids) != len(labels) or len(recipe_ids) != len(ingredient_rows):
        raise ValueError("Recipe IDs, labels, and ingredient rows must align")
    if len(set(recipe_ids)) != len(recipe_ids):
        raise ValueError("Recipe IDs must be unique")

    cuisine_index = {c: i for i, c in enumerate(cuisines)}
    y = np.array([cuisine_index[c] for c in labels], dtype=np.int16)
    n_c = np.bincount(y, minlength=len(cuisines))
    if np.any(n_c < 2):
        raise ValueError("Every cuisine needs at least two recipes for leave-one-out scoring")

    rows = [tuple(sorted(set(row))) for row in ingredient_rows]
    single_counts = [Counter() for _ in cuisines]
    pair_counts = [Counter() for _ in cuisines]
    for c, row in zip(y, rows):
        single_counts[c].update(row)
        pair_counts[c].update(combinations(row, 2))

    # Cache corpus-level contrasts. The scored recipe is subtracted below.
    single_contrasts = {}
    for ingredient in set().union(*(set(counts) for counts in single_counts)):
        prevalences = np.array([counts[ingredient] / n_c[c] for c, counts in enumerate(single_counts)])
        single_contrasts[ingredient] = prevalences
    pair_contrasts = {}
    for pair in set().union(*(set(counts) for counts in pair_counts)):
        pair_contrasts[pair] = np.array([counts[pair] / n_c[c] for c, counts in enumerate(pair_counts)])

    result = []
    for recipe_id, c, row in zip(recipe_ids, y, rows):
        singles = []
        for ingredient in row:
            prevalence = single_contrasts[ingredient]
            own_loo = (single_counts[c][ingredient] - 1) / (n_c[c] - 1)
            other = (prevalence.sum() - prevalence[c]) / (len(cuisines) - 1)
            singles.append(own_loo - other)

        pairs = []
        for pair in combinations(row, 2):
            prevalence = pair_contrasts[pair]
            own_loo = (pair_counts[c][pair] - 1) / (n_c[c] - 1)
            other = (prevalence.sum() - prevalence[c]) / (len(cuisines) - 1)
            pairs.append(own_loo - other)

        result.append({
            "recipe_id": recipe_id,
            "cuisine": cuisines[c],
            "retained_ingredients": len(row),
            "retained_pairs": len(pairs),
            "ingredient_distinctiveness": float(np.mean(singles)) if singles else np.nan,
            "pair_distinctiveness": float(np.mean(pairs)) if pairs else np.nan,
        })

    signatures = []
    for c, cuisine in enumerate(cuisines):
        for pair, count in pair_counts[c].items():
            if count < min_pair_support:
                continue
            prevalence = pair_contrasts[pair]
            contrast = prevalence[c] - (prevalence.sum() - prevalence[c]) / (len(cuisines) - 1)
            signatures.append({
                "cuisine": cuisine,
                "ingredient_a": ingredient_names[pair[0]],
                "ingredient_b": ingredient_names[pair[1]],
                "recipes_in_cuisine": int(count),
                "prevalence_in_cuisine": float(prevalence[c]),
                "mean_prevalence_elsewhere": float((prevalence.sum() - prevalence[c]) / (len(cuisines) - 1)),
                "pair_distinctiveness": float(contrast),
            })
    signatures = pd.DataFrame(signatures, columns=[
        "cuisine", "ingredient_a", "ingredient_b", "recipes_in_cuisine",
        "prevalence_in_cuisine", "mean_prevalence_elsewhere", "pair_distinctiveness",
    ]).sort_values(
        ["cuisine", "pair_distinctiveness", "recipes_in_cuisine"],
        ascending=[True, False, False],
    )
    return pd.DataFrame(result), signatures

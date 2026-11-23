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


def pairing_scores(recipe_ids, labels, ingredient_rows, randomized_rows, n_ingredients):
    """Score how much more often a recipe's ingredient pairs co-occur than chance.

    For each cuisine c and pair (i, j), the expected co-occurrence E is the mean
    count over `randomized_rows`: randomized versions of `ingredient_rows` that
    keep every recipe's size and every ingredient's frequency within its
    cuisine (e.g. Curveball). The pair lift is log2((n_ij + 1) / (E_ij + 1)),
    positive when the two ingredients are combined more often than their
    popularity predicts. A recipe's score is the mean lift over its pairs.

    Leave-one-out: the scored recipe is removed from the observed count
    (n_ij - 1), and E is rescaled by (n_i - 1)(n_j - 1) / (n_i n_j), because
    the expectation grows with the product of the two ingredient frequencies.

    Returns the per-recipe scores and a function `pair_expectation(c, i, j)`
    returning (observed, expected) for the full corpus.
    """
    from scipy import sparse

    labels = np.asarray(labels)
    n = len(labels)
    if len(recipe_ids) != n or len(ingredient_rows) != n:
        raise ValueError("Recipe IDs, labels, and ingredient rows must align")
    if not randomized_rows:
        raise ValueError("At least one randomized matrix is needed")

    def to_matrix(rows, index):
        r, c = [], []
        for k, i in enumerate(index):
            r.extend([k] * len(rows[i]))
            c.extend(rows[i])
        return sparse.csr_matrix((np.ones(len(r), dtype=np.float64), (r, c)),
                                 shape=(len(index), n_ingredients))

    scores = np.full(n, np.nan)
    observed, expected = {}, {}
    for cuisine in sorted(set(labels)):
        index = np.flatnonzero(labels == cuisine)
        obs = to_matrix(ingredient_rows, index)
        cooc = (obs.T @ obs).tocsr()
        exp = None
        for rows in randomized_rows:
            rnd = to_matrix(rows, index)
            if not np.array_equal(np.asarray(rnd.sum(axis=0)).ravel(), np.asarray(obs.sum(axis=0)).ravel()):
                raise ValueError("Randomized rows must keep ingredient frequencies within each cuisine")
            term = (rnd.T @ rnd).tocsr()
            exp = term if exp is None else exp + term
        exp = (exp / len(randomized_rows)).tocsr()
        freq = cooc.diagonal()
        observed[cuisine], expected[cuisine] = cooc, exp
        for i in index:
            row = sorted(set(ingredient_rows[i]))
            if len(row) < 2:
                continue
            a, b = np.array(list(combinations(row, 2))).T
            n_ab = np.asarray(cooc[a, b]).ravel() - 1
            e_ab = np.asarray(exp[a, b]).ravel()
            na, nb = freq[a], freq[b]
            e_ab = e_ab * (na - 1) * (nb - 1) / (na * nb)
            scores[i] = float(np.mean(np.log2((n_ab + 1) / (e_ab + 1))))

    def pair_expectation(cuisine, i, j):
        return float(observed[cuisine][i, j]), float(expected[cuisine][i, j])

    result = pd.DataFrame({"recipe_id": np.asarray(recipe_ids), "pairing_score": scores})
    return result, pair_expectation

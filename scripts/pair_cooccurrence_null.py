"""Check displayed cuisine pairs against an ingredient-preserving null model.

The nine pairs are selected from the observed data (one maximum prevalence
contrast per cuisine, with at least 20 supporting recipes). Null comparisons
are descriptive because this selection happened before the null was inspected.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from ingredient_ablation import MAX_DF_SHARE, MIN_DF, normalize, notebook_normalization
from recipe_ingredient_null import DATA, curveball, load_matrix, rows_to_matrix, verify_margins
from tagged_recipe_null import load_tags

ROOT = Path(__file__).resolve().parents[1]


def ingredient_index(recipes: pd.DataFrame, matrix) -> dict[str, int]:
    """Reconstruct and verify the sorted columns used by load_matrix."""
    aliases, generic = notebook_normalization()
    rows = [sorted({normalize(x, aliases) for x in json.loads(raw) if str(x).strip()})
            for raw in recipes.ingredients]
    counts = Counter(x for row in rows for x in row)
    kept = sorted(x for x, n in counts.items()
                  if MIN_DF <= n <= MAX_DF_SHARE * len(recipes) and x not in generic)
    assert len(kept) == matrix.shape[1]
    assert np.array_equal(np.asarray(matrix.sum(axis=0)).ravel(),
                          np.array([counts[x] for x in kept]))
    return {name: i for i, name in enumerate(kept)}


def pair_counts(rows: list[set[int]], groups: list[np.ndarray],
                pair_columns: list[tuple[int, int]]) -> np.ndarray:
    return np.array([
        sum(a in rows[int(i)] and b in rows[int(i)] for i in group)
        for group, (a, b) in zip(groups, pair_columns)
    ], dtype=np.int32)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data/raw/RAW_recipes.csv.zip")
    parser.add_argument("--replicates", type=int, default=49)
    parser.add_argument("--trades-per-recipe", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261119)
    parser.add_argument("--tag", default="full", help="Output filename suffix")
    args = parser.parse_args()
    if args.replicates < 1 or args.trades_per_recipe < 1:
        parser.error("replicates and trades-per-recipe must be positive")

    recipes, matrix = load_matrix()
    names = ingredient_index(recipes, matrix)
    types, _ = load_tags(args.archive, recipes)
    signatures = pd.read_csv(DATA / "network_pair_signatures_full.csv")
    selected = signatures.groupby("cuisine", sort=True).head(1).reset_index(drop=True)
    assert len(selected) == len(recipes.cuisine.unique()) == 9
    groups = [np.flatnonzero(recipes.cuisine.to_numpy() == c) for c in selected.cuisine]
    strata = [np.asarray(frame.index, dtype=np.int32)
              for _, frame in types.groupby(["cuisine", "dish_type"], sort=True)]
    pair_columns = [(names[a], names[b]) for a, b in
                    zip(selected.ingredient_a, selected.ingredient_b)]
    original_rows = [set(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]])
                     for i in range(matrix.shape[0])]
    observed = pair_counts(original_rows, groups, pair_columns)
    assert np.array_equal(observed, selected.recipes_in_cuisine.to_numpy()), \
        "The selected pair counts do not match the original bipartite matrix"

    null_counts = np.empty((args.replicates, len(selected)), dtype=np.int32)
    for rep in range(args.replicates):
        rows = [row.copy() for row in original_rows]
        curveball(rows, strata, args.trades_per_recipe,
                  np.random.default_rng(args.seed + rep))
        randomized = rows_to_matrix(rows, matrix.shape[1])
        verify_margins(matrix, randomized, strata)
        null_counts[rep] = pair_counts(rows, groups, pair_columns)
        print(f"{rep + 1}/{args.replicates}: " +
              ", ".join(f"{c}={n}" for c, n in
                        zip(selected.cuisine, null_counts[rep])), flush=True)

    out = selected[["cuisine", "ingredient_a", "ingredient_b",
                    "recipes_in_cuisine"]].copy()
    out = out.rename(columns={"recipes_in_cuisine": "observed_count"})
    out["null_median_count"] = np.median(null_counts, axis=0)
    out["null_025_count"] = np.quantile(null_counts, .025, axis=0)
    out["null_975_count"] = np.quantile(null_counts, .975, axis=0)
    out["observed_to_null_median"] = observed / out.null_median_count.replace(0, np.nan)
    out["null_replicates"] = args.replicates
    out.to_csv(DATA / f"network_selected_pair_null_{args.tag}.csv", index=False)
    print("\n" + out.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

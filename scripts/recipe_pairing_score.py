"""Add the pairing score ("how") to the recipe scores from the ingredient network.

For every cuisine and ingredient pair, the expected number of recipes that
contain both ingredients comes from Curveball randomizations of the
recipe-ingredient matrix within each cuisine (each recipe keeps its number of
ingredients; each ingredient keeps its frequency in that cuisine). A recipe's
pairing score is the mean log2 ratio between observed and expected
co-occurrence over its ingredient pairs, with leave-one-out. See
docs/recipe_pairing_score.md.

Run after scripts/recipe_pair_distinctiveness.py:
    python scripts/recipe_pairing_score.py            # 50 randomizations, ~3 min
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from recipe_ingredient_null import curveball, load_matrix  # noqa: E402
from src.cuisine_distinctiveness import pairing_scores  # noqa: E402

DATA = ROOT / "data" / "processed"
SHOWCASE = [  # the nine pairs of the report's pair figure
    ("chinese", "green onion", "soy sauce"), ("french", "butter", "flour"),
    ("greek", "feta cheese", "olive oil"), ("indian", "onion", "turmeric"),
    ("italian", "olive oil", "parmesan cheese"), ("japanese", "soy sauce", "sugar"),
    ("mexican", "chili powder", "cumin"), ("middle-eastern", "lemon juice", "olive oil"),
    ("thai", "fish sauce", "lime juice"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--trades-per-recipe", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261123)
    args = parser.parse_args()

    recipes, matrix = load_matrix()
    cuisine = recipes.cuisine.to_numpy()
    groups = [np.flatnonzero(cuisine == c) for c in sorted(set(cuisine))]
    base = [set(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]) for i in range(matrix.shape[0])]
    randomized = []
    for rep in range(args.replicates):
        rows = [row.copy() for row in base]
        curveball(rows, groups, args.trades_per_recipe, np.random.default_rng(args.seed + rep))
        randomized.append([sorted(r) for r in rows])
        print(f"randomization {rep + 1}/{args.replicates}", end="\r", flush=True)
    print()

    scores, expectation = pairing_scores(
        recipes.recipe_id, cuisine, [sorted(r) for r in base], randomized, matrix.shape[1])

    # Merge into the existing per-recipe scores.
    path = DATA / "network_recipe_distinctiveness_full.csv"
    existing = pd.read_csv(path).drop(columns=["pairing_score"], errors="ignore")
    merged = existing.merge(scores, on="recipe_id", how="left", validate="one_to_one")
    assert len(merged) == len(existing)
    merged.to_csv(path, index=False)

    # Add observed vs. expected co-occurrence to the pair-signature table.
    from collections import Counter
    import json
    from ingredient_ablation import MAX_DF_SHARE, MIN_DF, normalize, notebook_normalization
    aliases, generic = notebook_normalization()
    raw = pd.read_csv(DATA / "recipes.csv", usecols=["ingredients"]).ingredients
    freq = Counter(x for s in raw for x in {normalize(y, aliases) for y in json.loads(s) if str(y).strip()})
    kept = sorted(x for x, c in freq.items() if MIN_DF <= c <= MAX_DF_SHARE * len(raw) and x not in generic)
    index = {x: j for j, x in enumerate(kept)}
    assert len(kept) == matrix.shape[1]
    sig_path = DATA / "network_pair_signatures_full.csv"
    sig = pd.read_csv(sig_path).drop(columns=["expected_in_cuisine", "pairing_lift"], errors="ignore")
    exp_col, lift_col = [], []
    for c, a, b in zip(sig.cuisine, sig.ingredient_a, sig.ingredient_b):
        o, e = expectation(c, index[a], index[b])
        exp_col.append(e)
        lift_col.append(np.log2((o + 1) / (e + 1)))
    sig["expected_in_cuisine"] = exp_col
    sig["pairing_lift"] = lift_col
    sig.to_csv(sig_path, index=False)

    s = merged.dropna(subset=["pair_distinctiveness", "pairing_score"])
    print(f"Scored {merged.pairing_score.notna().sum():,} of {len(merged):,} recipes")
    print("Pearson r with the ingredient score:", round(s.pairing_score.corr(s.ingredient_distinctiveness), 3))
    print("Pearson r with the pair score:      ", round(s.pairing_score.corr(s.pair_distinctiveness), 3))
    print("\nShowcase pairs: observed vs. expected (cuisine-stratified Curveball mean)")
    for c, a, b in SHOWCASE:
        i, j = sorted((index[a], index[b]))
        o, e = expectation(c, i, j)
        print(f"  {c:15s} {a} + {b}: {o:.0f} vs {e:.0f}  (ratio {o / e:.2f})")
    print("\nMedian pairing score by cuisine:")
    print(merged.groupby("cuisine").pairing_score.median().round(3).to_string())
    print("\nStrongest pairings per cuisine (pairs in >= 20 recipes):")
    print(sig.sort_values("pairing_lift", ascending=False).groupby("cuisine").head(3)
          [["cuisine", "ingredient_a", "ingredient_b", "recipes_in_cuisine", "expected_in_cuisine", "pairing_lift"]]
          .sort_values("cuisine").round(2).to_string(index=False))


if __name__ == "__main__":
    main()

"""Export cuisine-specific recipe scores from the recipe–ingredient network."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cuisine_distinctiveness import score_recipes
from src.ingredient_preprocessing import GENERIC_INGREDIENTS, normalize_ingredient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["mini", "full"], default="full")
    args = parser.parse_args()
    recipes = pd.read_csv(ROOT / "data/processed" / f"recipes{'_mini' if args.dataset == 'mini' else ''}.csv")
    normalized = [
        sorted({normalize_ingredient(x) for x in json.loads(raw) if str(x).strip()})
        for raw in recipes.ingredients
    ]
    frequencies = Counter(x for row in normalized for x in row)
    n = len(recipes)
    kept = sorted(
        x for x, count in frequencies.items()
        if 3 <= count <= 0.35 * n and x not in GENERIC_INGREDIENTS
    )
    index = {name: i for i, name in enumerate(kept)}
    rows = [[index[x] for x in row if x in index] for row in normalized]
    scores, signatures = score_recipes(recipes.recipe_id, recipes.cuisine, rows, kept)
    score_path = ROOT / "data/processed" / f"network_recipe_distinctiveness_{args.dataset}.csv"
    signature_path = ROOT / "data/processed" / f"network_pair_signatures_{args.dataset}.csv"
    scores.to_csv(score_path, index=False)
    signatures.to_csv(signature_path, index=False)
    print(f"{n:,} recipes; {len(kept):,} retained ingredients; {sum(map(len, rows)):,} recipe–ingredient links")
    print(f"{scores.retained_pairs.eq(0).sum():,} recipes lack a retained ingredient pair")
    print(f"Saved {score_path.relative_to(ROOT)} and {signature_path.relative_to(ROOT)}")
    print(scores.groupby("cuisine").pair_distinctiveness.median().round(4).to_string())
    print(signatures.groupby("cuisine").head(3).to_string(index=False))


if __name__ == "__main__":
    main()

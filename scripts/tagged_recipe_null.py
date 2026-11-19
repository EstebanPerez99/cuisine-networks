"""Rebuild the recipe network under a cuisine-and-dish-type ingredient null.

Source tags are read from the user-supplied Food.com RAW_recipes.csv ZIP. A
transparent priority rule assigns one primary course; no title or ingredient
information is used for that assignment. Curveball trades occur only within
each cuisine × primary-course stratum, preserving the binary matrix's row
degrees and ingredient frequencies in every stratum exactly.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recipe_ingredient_null import (DATA, FIG, PAIRS, build_edges, curveball,
                                    edge_metrics, load_matrix, rows_to_matrix,
                                    verify_baseline, verify_margins)
from recipe_type_analysis import classify_title

ROOT = Path(__file__).resolve().parents[1]

COURSE_PRIORITY: tuple[tuple[str, frozenset[str]], ...] = (
    ("beverage", frozenset({"beverages"})),
    ("dessert", frozenset({"desserts"})),
    ("soup/stew", frozenset({"soups-stews"})),
    ("salad", frozenset({"salads"})),
    ("bread", frozenset({"breads"})),
    ("breakfast/brunch", frozenset({"breakfast", "brunch"})),
    ("appetizer/snack", frozenset({"appetizers", "snacks"})),
    ("side dish", frozenset({"side-dishes"})),
    ("condiment/sauce", frozenset({"condiments-etc"})),
    ("main dish", frozenset({"main-dish"})),
)
TITLE_TO_TAG = {"drink": "beverage", "dessert/sweet": "dessert",
                "soup/stew": "soup/stew", "salad": "salad"}


def primary_course(tags: set[str]) -> str:
    for name, match in COURSE_PRIORITY:
        if tags & match:
            return name
    return "other/unclear"


def load_tags(archive: Path, recipes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    with ZipFile(archive) as z:
        assert z.namelist() == ["RAW_recipes.csv"], "Expected RAW_recipes.csv in archive"
        with z.open("RAW_recipes.csv") as f:
            raw = pd.read_csv(f, usecols=["id", "tags"])
    assert raw.id.is_unique and raw.id.notna().all()
    joined = recipes[["recipe_id", "cuisine"]].merge(
        raw, left_on="recipe_id", right_on="id", how="left", validate="one_to_one")
    assert joined.tags.notna().all(), "Some project recipe IDs are absent from raw source"
    assert np.array_equal(joined.recipe_id.to_numpy(), recipes.recipe_id.to_numpy())
    parsed = []
    for value in joined.tags:
        tags = ast.literal_eval(value)
        assert isinstance(tags, list) and all(isinstance(t, str) for t in tags)
        parsed.append(set(tags))
    course_tags = set().union(*(match for _, match in COURSE_PRIORITY))
    names = pd.read_csv(DATA / "recipes.csv", usecols=["recipe_id", "name"])
    out = recipes[["recipe_id", "cuisine"]].merge(
        names, on="recipe_id", validate="one_to_one")
    assert np.array_equal(out.recipe_id.to_numpy(), recipes.recipe_id.to_numpy())
    out["dish_type"] = [primary_course(tags) for tags in parsed]
    out["course_tags"] = [json.dumps(sorted(tags & course_tags)) for tags in parsed]
    out["title_type"] = out.name.map(classify_title)
    out["title_tag_agree"] = [
        TITLE_TO_TAG.get(title) == course if title in TITLE_TO_TAG else np.nan
        for title, course in zip(out.title_type, out.dish_type)
    ]
    counts = Counter(t for tags in parsed for t in tags)
    frequency = pd.DataFrame(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])),
                             columns=["tag", "recipes"])
    return out, frequency


def audit_tables(types: pd.DataFrame, frequency: pd.DataFrame) -> None:
    types.to_csv(DATA / "network_recipe_dish_types_full.csv", index=False)
    frequency.to_csv(DATA / "network_raw_tag_frequency_full.csv", index=False)
    summary = (types.groupby(["cuisine", "dish_type"]).size()
               .rename("recipes").reset_index())
    summary.to_csv(DATA / "network_dish_type_cuisine_counts_full.csv", index=False)
    sample = pd.concat([
        group.sample(min(10, len(group)), random_state=20261118)
        for _, group in types.groupby("dish_type")
    ]).reset_index(drop=True)
    # Add a separate review set for disagreement with the title-only classifier.
    conflicts = types.loc[(types.title_type != "other/unclear") &
                          (types.title_tag_agree == False)]
    conflicts = conflicts.sample(min(30, len(conflicts)), random_state=20261118)
    sample = pd.concat([sample, conflicts]).drop_duplicates("recipe_id")
    sample.to_csv(DATA / "network_dish_type_audit_sample_full.csv", index=False)
    print("Dish type counts:\n" + types.dish_type.value_counts().to_string(), flush=True)
    title = types.loc[types.title_type != "other/unclear", "title_tag_agree"]
    print(f"Title-labeled agreement: {title.mean():.3f} among {len(title):,}; "
          f"strata: {len(summary)}, smallest: {summary.recipes.min()}", flush=True)


def plot_null(tagged: pd.DataFrame) -> None:
    cuisine_only = pd.read_csv(DATA / "network_recipe_ingredient_null_full.csv")
    metrics = ["within_cuisine_weight_share"] + [
        a.replace("-", "_") + "_" + b.replace("-", "_") + "_cross_share"
        for a, b in PAIRS]
    titles = ["A. Within-cuisine edge weight", "B. Chinese–Japanese cross links",
              "C. Greek–Middle Eastern cross links", "D. French–Italian cross links"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    for i, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
        observed = 100 * tagged.loc[0, metric]
        for y, (frame, color, label) in enumerate([
                (cuisine_only, "#7799ad", "Cuisine only"),
                (tagged, "#44766c", "Cuisine + dish type")]):
            values = 100 * frame.loc[1:, metric].to_numpy()
            lo, hi = np.quantile(values, [.025, .975])
            ax.hlines(y, lo, hi, color=color, lw=8, alpha=.65)
            ax.scatter(np.median(values), y, color=color, s=55, zorder=3,
                       label=label if i == 0 else None)
        ax.axvline(observed, color="#b76141", lw=2.1,
                   label="Observed" if i == 0 else None)
        ax.set_yticks([0, 1], ["Cuisine only", "Cuisine + dish type"])
        ax.set_ylim(1.5, -.5)
        ax.set(xlabel="Share of all edge weight (%)" if i == 0 else
               "Share of cross-cuisine edge weight (%)", title=title)
        ax.grid(axis="x", alpha=.18)
    axes.flat[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Ingredient-preserving nulls with and without dish type", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "17_tag_stratified_recipe_null_full.png", dpi=200)
    fig.savefig(FIG / "17_tag_stratified_recipe_null_full.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path,
                        default=ROOT / "data" / "raw" / "RAW_recipes.csv.zip")
    parser.add_argument("--replicates", type=int, default=49)
    parser.add_argument("--trades-per-recipe", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--tag", default="full")
    parser.add_argument("--group-by", choices=["primary", "all-tags"], default="primary",
                        help="Use primary dish type or the full combination of course tags")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    assert args.replicates >= 1 and args.trades_per_recipe >= 1
    recipes, matrix = load_matrix()
    types, frequency = load_tags(args.archive, recipes)
    audit_tables(types, frequency)
    if args.audit_only:
        return
    cuisine = recipes.cuisine.to_numpy()
    group_column = "dish_type" if args.group_by == "primary" else "course_tags"
    groups = [np.asarray(g.index, dtype=np.int32)
              for _, g in types.groupby(["cuisine", group_column], sort=True)]
    print(f"Grouping: {args.group_by}; strata: {len(groups)}; "
          f"singleton recipes: {sum(len(g) == 1 for g in groups)}", flush=True)
    sizes = np.diff(matrix.indptr)
    base_rows = [set(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]])
                 for i in range(matrix.shape[0])]
    baseline_edges = build_edges(matrix, sizes)
    verify_baseline(baseline_edges, recipes)
    records = [{"replicate": 0, "seed": args.seed, "attempts": 0, "eligible_trades": 0,
                "changed_recipe_share": 0.0, "retained_incidence_share": 1.0,
                **edge_metrics(baseline_edges, cuisine)}]
    for rep in range(1, args.replicates + 1):
        rng = np.random.default_rng(args.seed + rep)
        rows = [row.copy() for row in base_rows]
        eligible, attempts = curveball(rows, groups, args.trades_per_recipe, rng)
        shuffled = rows_to_matrix(rows, matrix.shape[1])
        verify_margins(matrix, shuffled, groups)
        changed = np.mean([row != original for row, original in zip(rows, base_rows)])
        retained = sum(len(row & original) for row, original in zip(rows, base_rows)) / matrix.nnz
        edges = build_edges(shuffled, sizes)
        record = {"replicate": rep, "seed": args.seed + rep, "attempts": attempts,
                  "eligible_trades": eligible, "changed_recipe_share": changed,
                  "retained_incidence_share": retained, **edge_metrics(edges, cuisine)}
        records.append(record)
        print(f"{rep:02}/{args.replicates}: within={record['within_cuisine_weight_share']:.4f}, "
              f"changed={changed:.3f}, retained={retained:.3f}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(DATA / f"network_tag_stratified_null_{args.tag}.csv", index=False)
    if args.tag == "full" and args.group_by == "primary":
        plot_null(results)
    print("\nObserved / null median / central 95% / upper-tail p:", flush=True)
    for metric in ["within_cuisine_weight_share"] + [
            a.replace("-", "_") + "_" + b.replace("-", "_") + "_cross_share"
            for a, b in PAIRS]:
        observed = results.loc[0, metric]
        null = results.loc[1:, metric].to_numpy()
        p = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        print(metric, f"{observed:.5f}", f"{np.median(null):.5f}",
              np.quantile(null, [.025, .975]), f"p={p:.4f}", flush=True)


if __name__ == "__main__":
    main()

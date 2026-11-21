"""Cuisine-stratified bipartite null for the full recipe similarity network.

Curveball trades ingredients between recipes of the same cuisine. Every trade
preserves each recipe's retained-ingredient count and each ingredient's count
within that cuisine. Rebuild the Jaccard/top-15 network after each randomized
ingredient matrix, using the exact projection implemented for ablation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse

from ingredient_ablation import (DATA, FIG, MAX_DF_SHARE, MIN_DF,
                                 build_edges, normalize, notebook_normalization)

PAIRS = (("chinese", "japanese"), ("greek", "middle-eastern"),
         ("french", "italian"))


def load_matrix() -> tuple[pd.DataFrame, sparse.csr_matrix]:
    recipes = pd.read_csv(DATA / "recipes.csv", usecols=["recipe_id", "cuisine", "ingredients"])
    assert recipes.recipe_id.is_unique
    aliases, generic = notebook_normalization()
    rows = [sorted({normalize(x, aliases) for x in json.loads(s) if str(x).strip()})
            for s in recipes.ingredients]
    frequency = Counter(x for row in rows for x in row)
    kept = sorted(x for x, count in frequency.items()
                  if MIN_DF <= count <= MAX_DF_SHARE * len(recipes) and x not in generic)
    index = {x: j for j, x in enumerate(kept)}
    row_idx, col_idx = [], []
    for i, row in enumerate(rows):
        for ingredient in row:
            if ingredient in index:
                row_idx.append(i)
                col_idx.append(index[ingredient])
    matrix = sparse.csr_matrix((np.ones(len(row_idx), dtype=np.int32),
                                (row_idx, col_idx)), shape=(len(rows), len(kept)))
    return recipes, matrix


def edge_metrics(edges: dict[tuple[int, int], float], cuisine: np.ndarray) -> dict:
    pairs = np.asarray(list(edges), dtype=np.int32)
    u, v = pairs[:, 0], pairs[:, 1]
    weights = np.fromiter(edges.values(), dtype=float, count=len(edges))
    same = cuisine[u] == cuisine[v]
    cross_weight = weights[~same].sum()
    degree = np.bincount(np.r_[u, v], minlength=len(cuisine))
    out = {
        "edges": len(edges),
        "isolates": int(np.count_nonzero(degree == 0)),
        "total_edge_weight": float(weights.sum()),
        "cross_edge_weight": float(cross_weight),
        "within_cuisine_weight_share": float(weights[same].sum() / weights.sum()),
    }
    for a, b in PAIRS:
        mask = (((cuisine[u] == a) & (cuisine[v] == b)) |
                ((cuisine[u] == b) & (cuisine[v] == a)))
        key = a.replace("-", "_") + "_" + b.replace("-", "_")
        out[key + "_cross_share"] = float(weights[mask].sum() / cross_weight)
        out[key + "_edge_weight"] = float(weights[mask].sum())
    return out


def curveball(rows: list[set[int]], groups: list[np.ndarray], attempts_per_row: int,
              rng: np.random.Generator) -> tuple[int, int]:
    """Trade exclusive items in two rows within a cuisine stratum."""
    eligible = 0
    attempts = 0
    for group in groups:
        if len(group) < 2:
            continue
        for _ in range(attempts_per_row * len(group)):
            attempts += 1
            i, j = rng.integers(len(group), size=2)
            if i == j:
                continue
            a, b = int(group[i]), int(group[j])
            common = rows[a] & rows[b]
            only_a = rows[a] - common
            only_b = rows[b] - common
            if not only_a or not only_b:
                continue
            pool = sorted(only_a | only_b)
            order = rng.permutation(len(pool))
            rows[a] = common | {pool[k] for k in order[:len(only_a)]}
            rows[b] = common | {pool[k] for k in order[len(only_a):]}
            eligible += 1
    return eligible, attempts


def rows_to_matrix(rows: list[set[int]], ncols: int) -> sparse.csr_matrix:
    indptr = np.empty(len(rows) + 1, dtype=np.int32)
    indptr[0] = 0
    indices = np.empty(sum(map(len, rows)), dtype=np.int32)
    cursor = 0
    for i, row in enumerate(rows):
        values = sorted(row)
        indices[cursor:cursor + len(values)] = values
        cursor += len(values)
        indptr[i + 1] = cursor
    return sparse.csr_matrix((np.ones(cursor, dtype=np.int32), indices, indptr),
                             shape=(len(rows), ncols))


def verify_margins(original: sparse.csr_matrix, randomized: sparse.csr_matrix,
                   groups: list[np.ndarray]) -> None:
    assert np.array_equal(original.indptr, randomized.indptr), "Recipe sizes changed"
    for group in groups:
        before = np.asarray(original[group].sum(axis=0)).ravel()
        after = np.asarray(randomized[group].sum(axis=0)).ravel()
        assert np.array_equal(before, after), "Within-cuisine ingredient frequency changed"


def verify_baseline(edges: dict[tuple[int, int], float], recipes: pd.DataFrame) -> None:
    original = pd.read_csv(DATA / "network_edges_full.csv")
    index = pd.Series(np.arange(len(recipes)), index=recipes.recipe_id)
    reference = {(int(index[a]), int(index[b])): float(weight)
                 for a, b, weight in original.itertuples(index=False, name=None)}
    assert edges.keys() == reference.keys(), "Baseline edge set differs from notebook export"
    assert max(abs(weight - reference[pair]) for pair, weight in edges.items()) < 1e-12


def plot_results(results: pd.DataFrame, stem: str) -> None:
    metrics = ["within_cuisine_weight_share",
               "chinese_japanese_cross_share",
               "greek_middle_eastern_cross_share",
               "french_italian_cross_share"]
    titles = ["A. Within-cuisine edge weight", "B. Chinese–Japanese cross links",
              "C. Greek–Middle Eastern cross links", "D. French–Italian cross links"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    baseline = results.iloc[0]
    random = results.iloc[1:]
    for i, (ax, metric, title) in enumerate(zip(axes.flat, metrics, titles)):
        values = 100 * random[metric].to_numpy()
        observed = 100 * baseline[metric]
        ax.hist(values, bins=min(16, max(5, len(values) // 3)),
                color="#7799ad", edgecolor="white")
        ax.axvline(observed, color="#b76141", linewidth=2.3, label="Observed")
        ax.axvline(np.median(values), color="#2d5770", linestyle="--",
                   linewidth=1.6, label="Null median")
        xlabel = ("Share of all edge weight (%)" if i == 0 else
                  "Share of cross-cuisine edge weight (%)")
        ax.set(xlabel=xlabel, ylabel="Randomized networks", title=title)
        ax.grid(axis="y", alpha=.15)
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Recipe–ingredient randomization within cuisines", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / f"{stem}.png", dpi=200)
    fig.savefig(FIG / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=49)
    parser.add_argument("--trades-per-recipe", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261118)
    parser.add_argument("--tag", default="full")
    args = parser.parse_args()
    assert args.replicates >= 1 and args.trades_per_recipe >= 1
    recipes, matrix = load_matrix()
    cuisine = recipes.cuisine.to_numpy()
    groups = [np.flatnonzero(cuisine == c) for c in sorted(set(cuisine))]
    sizes = np.diff(matrix.indptr)
    base_rows = [set(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]])
                 for i in range(matrix.shape[0])]
    baseline_edges = build_edges(matrix, sizes)
    verify_baseline(baseline_edges, recipes)
    print(f"Baseline matches notebook: {len(baseline_edges):,} edges; "
          f"{matrix.shape[0]:,} recipes, {matrix.shape[1]:,} ingredients", flush=True)
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
              f"edges={record['edges']:,}, changed={changed:.3f}, "
              f"retained={retained:.3f}", flush=True)
    results = pd.DataFrame(records)
    stem = f"network_recipe_ingredient_null_{args.tag}"
    results.to_csv(DATA / f"{stem}.csv", index=False)
    if args.tag == "full":
        plot_results(results, "15_recipe_ingredient_null_full")
    print("\nSummary (observed, null median, null 2.5–97.5%, upper-tail p):", flush=True)
    for metric in ["within_cuisine_weight_share"] + [
            a.replace("-", "_") + "_" + b.replace("-", "_") + "_cross_share"
            for a, b in PAIRS]:
        observed = results.loc[0, metric]
        null = results.loc[1:, metric].to_numpy()
        p = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        print(f"{metric}: {observed:.5f}, {np.median(null):.5f}, "
              f"{np.quantile(null, [.025, .975])}, p={p:.4f}", flush=True)


if __name__ == "__main__":
    main()

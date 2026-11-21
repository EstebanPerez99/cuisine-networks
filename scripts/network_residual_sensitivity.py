"""Check whether the ingredient-null residual survives recipe/edge exclusions.

The original community and title classifications are fixed across every
randomized network. This is a sensitivity analysis, not a new independent test.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from recipe_ingredient_null import (DATA, FIG, PAIRS, build_edges, curveball,
                                    load_matrix, rows_to_matrix, verify_baseline,
                                    verify_margins)


CONDITIONS = (
    "all",
    "exclude named sweet/drink",
    "exclude original community 76",
    "exclude Jaccard >= 0.75",
    "exclude community 76 and Jaccard >= 0.75",
)


def fixed_masks(recipes: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    communities = pd.read_csv(DATA / "network_recipe_metrics_full.csv",
                              usecols=["recipe_id", "community"])
    types = pd.read_csv(DATA / "network_recipe_types_full.csv",
                        usecols=["recipe_id", "recipe_type"])
    ids = recipes.recipe_id
    community = pd.Series(communities.community.to_numpy(), index=communities.recipe_id).loc[ids]
    recipe_type = pd.Series(types.recipe_type.to_numpy(), index=types.recipe_id).loc[ids]
    no_c76 = community.to_numpy() != 76
    no_sweet = ~recipe_type.isin(["dessert/sweet", "drink"]).to_numpy()
    return no_c76, no_sweet


def metrics(edges: dict[tuple[int, int], float], cuisine: np.ndarray,
            no_c76: np.ndarray, no_sweet: np.ndarray, replicate: int) -> list[dict]:
    pairs = np.asarray(list(edges), dtype=np.int32)
    u, v = pairs[:, 0], pairs[:, 1]
    w = np.fromiter(edges.values(), dtype=float, count=len(edges))
    keep = {
        CONDITIONS[0]: np.ones(len(w), dtype=bool),
        CONDITIONS[1]: no_sweet[u] & no_sweet[v],
        CONDITIONS[2]: no_c76[u] & no_c76[v],
        CONDITIONS[3]: w < .75,
        CONDITIONS[4]: no_c76[u] & no_c76[v] & (w < .75),
    }
    same = cuisine[u] == cuisine[v]
    pair_masks = {}
    for a, b in PAIRS:
        key = a.replace("-", "_") + "_" + b.replace("-", "_")
        pair_masks[key] = (((cuisine[u] == a) & (cuisine[v] == b)) |
                           ((cuisine[u] == b) & (cuisine[v] == a)))
    out = []
    for condition, mask in keep.items():
        selected = w[mask]
        cross = w[mask & ~same].sum()
        row = {"replicate": replicate, "condition": condition,
               "edges": int(mask.sum()),
               "edge_weight": float(selected.sum()),
               "within_cuisine_weight_share": float(w[mask & same].sum() / selected.sum())}
        for key, pair_mask in pair_masks.items():
            row[key + "_cross_share"] = float(w[mask & pair_mask].sum() / cross)
        out.append(row)
    return out


def plot(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.1))
    labels = ["All", "No named sweet/drink", "No original community 76",
              "No Jaccard ≥ 0.75", "No community 76 or ≥ 0.75"]
    for i, condition in enumerate(CONDITIONS):
        subset = results.loc[results.condition == condition]
        observed = 100 * float(subset.loc[subset.replicate == 0,
                                          "within_cuisine_weight_share"].iloc[0])
        null = 100 * subset.loc[subset.replicate != 0,
                                 "within_cuisine_weight_share"].to_numpy()
        ax.hlines(i, np.quantile(null, .025), np.quantile(null, .975),
                  color="#7799ad", lw=7, alpha=.65)
        ax.scatter(np.median(null), i, color="#2d5770", s=42, zorder=3)
        ax.scatter(observed, i, color="#b76141", marker="D", s=60, zorder=4)
    ax.set_yticks(range(len(labels)), labels)
    ax.invert_yaxis()
    ax.set(xlabel="Within-cuisine share of edge weight (%)",
           title="Cuisine structure after recipe and edge exclusions")
    ax.grid(axis="x", alpha=.2)
    ax.scatter([], [], color="#2d5770", label="Null median")
    ax.scatter([], [], color="#b76141", marker="D", label="Observed")
    ax.plot([], [], color="#7799ad", lw=7, alpha=.65, label="Central 95% null range")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3)
    fig.tight_layout()
    fig.savefig(FIG / "16_network_residual_sensitivity_full.png", dpi=200)
    fig.savefig(FIG / "16_network_residual_sensitivity_full.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", type=int, default=49)
    parser.add_argument("--trades-per-recipe", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    recipes, matrix = load_matrix()
    cuisine = recipes.cuisine.to_numpy()
    groups = [np.flatnonzero(cuisine == c) for c in sorted(set(cuisine))]
    no_c76, no_sweet = fixed_masks(recipes)
    base_rows = [set(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]])
                 for i in range(matrix.shape[0])]
    sizes = np.diff(matrix.indptr)
    edges = build_edges(matrix, sizes)
    verify_baseline(edges, recipes)
    records = metrics(edges, cuisine, no_c76, no_sweet, 0)
    reference = pd.read_csv(DATA / "network_recipe_ingredient_null_full.csv")
    assert args.replicates <= len(reference) - 1
    for rep in range(1, args.replicates + 1):
        rng = np.random.default_rng(args.seed + rep)
        rows = [row.copy() for row in base_rows]
        curveball(rows, groups, args.trades_per_recipe, rng)
        shuffled = rows_to_matrix(rows, matrix.shape[1])
        verify_margins(matrix, shuffled, groups)
        edges = build_edges(shuffled, sizes)
        batch = metrics(edges, cuisine, no_c76, no_sweet, rep)
        # Same seeds and algorithm must reproduce the previous full-null run.
        check = batch[0]["within_cuisine_weight_share"]
        assert abs(check - reference.loc[rep, "within_cuisine_weight_share"]) < 1e-12
        records.extend(batch)
        print(f"{rep:02}/{args.replicates}: full={batch[0]['within_cuisine_weight_share']:.4f}, "
              f"without-C76={batch[2]['within_cuisine_weight_share']:.4f}", flush=True)
    results = pd.DataFrame(records)
    results.to_csv(DATA / "network_residual_sensitivity_full.csv", index=False)
    plot(results)
    print("\nObserved / null median / central 95% / upper-tail p:", flush=True)
    for condition in CONDITIONS:
        subset = results.loc[results.condition == condition]
        observed = float(subset.loc[subset.replicate == 0,
                                    "within_cuisine_weight_share"].iloc[0])
        null = subset.loc[subset.replicate != 0, "within_cuisine_weight_share"].to_numpy()
        p = (1 + np.count_nonzero(null >= observed)) / (len(null) + 1)
        print(condition, f"{observed:.5f}", f"{np.median(null):.5f}",
              np.quantile(null, [.025, .975]), f"p={p:.4f}", flush=True)


if __name__ == "__main__":
    main()

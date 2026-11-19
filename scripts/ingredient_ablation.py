"""Rebuild the full recipe network after ingredient removals.

This exploratory sensitivity analysis reproduces the notebook's binary
recipe-ingredient projection (Jaccard >= 0.18, >=2 shared ingredients,
union of each recipe's 15 best neighbors). Candidate ingredients were chosen
after examining the original network; control comparisons are descriptive.
"""

from __future__ import annotations

import ast
import heapq
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
MIN_DF = 3
MAX_DF_SHARE = .35
MIN_SHARED = 2
MIN_JACCARD = .18
TOP_K = 15
TARGET_METRIC = {
    "vanilla extract": "c76_incident_share_of_cross_weight",
    "flour": "c76_incident_share_of_cross_weight",
    "soy sauce": "chinese_japanese_cross_share",
    "lemon juice": "greek_middle_eastern_cross_share",
}


def notebook_normalization() -> tuple[dict[str, str], set[str]]:
    # The notebook imports these shared constants; cell positions may change.
    code = (ROOT / "src" / "ingredient_preprocessing.py").read_text()
    values = {}
    for node in ast.parse(code).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"ALIASES", "GENERIC_INGREDIENTS"}:
                values[node.targets[0].id] = ast.literal_eval(node.value)
    return values["ALIASES"], values["GENERIC_INGREDIENTS"]


def normalize(raw: str, aliases: dict[str, str]) -> str:
    value = re.sub(r"\s+", " ", str(raw).strip().lower().replace("&", "and"))
    return aliases.get(value, value)


def build_edges(matrix: sparse.csr_matrix, sizes: np.ndarray) -> dict[tuple[int, int], float]:
    n = matrix.shape[0]
    heaps: list[list[tuple[float, int, int]]] = [[] for _ in range(n)]
    transpose = matrix.T.tocsr()
    for start in range(0, n, 512):
        block = (matrix[start:start + 512] @ transpose).tocoo()
        row = block.row + start
        col = block.col
        eligible = (row < col) & (block.data >= MIN_SHARED)
        row, col, count = row[eligible], col[eligible], block.data[eligible]
        similarity = count / (sizes[row] + sizes[col] - count)
        eligible = similarity >= MIN_JACCARD
        for a, b, weight in zip(row[eligible], col[eligible], similarity[eligible]):
            a, b, weight = int(a), int(b), float(weight)
            for source, target in ((a, b), (b, a)):
                heap = heaps[source]
                item = (weight, -target, target)
                if len(heap) < TOP_K:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    edges = {}
    for source, heap in enumerate(heaps):
        for weight, _, target in heap:
            edges[(min(source, target), max(source, target))] = weight
    return edges


def network_metrics(edges: dict[tuple[int, int], float], cuisine: np.ndarray,
                    c76: np.ndarray) -> dict[str, float | int]:
    pairs = np.array(list(edges), dtype=np.int32)
    u, v = pairs[:, 0], pairs[:, 1]
    w = np.fromiter(edges.values(), dtype=float, count=len(edges))
    different = cuisine[u] != cuisine[v]
    cross_weight = w[different].sum()
    # Vector comparisons avoid depending on arbitrary endpoint order.
    cross_u, cross_v, cross_w = u[different], v[different], w[different]
    cj = (((cuisine[cross_u] == "chinese") & (cuisine[cross_v] == "japanese")) |
          ((cuisine[cross_u] == "japanese") & (cuisine[cross_v] == "chinese")))
    gm = (((cuisine[cross_u] == "greek") & (cuisine[cross_v] == "middle-eastern")) |
          ((cuisine[cross_u] == "middle-eastern") & (cuisine[cross_v] == "greek")))
    degree = np.bincount(np.r_[u, v], minlength=len(cuisine))
    return {
        "edges": len(edges),
        "isolates": int(np.count_nonzero(degree == 0)),
        "within_cuisine_weight_share": float(w[~different].sum() / w.sum()),
        "cross_edge_weight": float(cross_weight),
        "c76_incident_cross_weight": float(
            cross_w[c76[cross_u] | c76[cross_v]].sum()),
        "c76_incident_share_of_cross_weight": float(
            cross_w[c76[cross_u] | c76[cross_v]].sum() / cross_weight),
        "chinese_japanese_edge_weight": float(cross_w[cj].sum()),
        "chinese_japanese_cross_share": float(cross_w[cj].sum() / cross_weight),
        "greek_middle_eastern_edge_weight": float(cross_w[gm].sum()),
        "greek_middle_eastern_cross_share": float(cross_w[gm].sum() / cross_weight),
    }


def main() -> None:
    recipes = pd.read_csv(DATA / "recipes.csv", usecols=["recipe_id", "cuisine", "ingredients"])
    assert recipes.recipe_id.is_unique
    n = len(recipes)
    aliases, generic = notebook_normalization()
    ingredient_rows = [sorted({normalize(x, aliases) for x in json.loads(s) if str(x).strip()})
                       for s in recipes.ingredients]
    frequency = Counter(x for row in ingredient_rows for x in row)
    kept = sorted(x for x, count in frequency.items()
                  if MIN_DF <= count <= MAX_DF_SHARE * n and x not in generic)
    ingredient_index = {x: j for j, x in enumerate(kept)}
    row_idx, col_idx = [], []
    for i, row in enumerate(ingredient_rows):
        for ingredient in row:
            if ingredient in ingredient_index:
                row_idx.append(i)
                col_idx.append(ingredient_index[ingredient])
    matrix = sparse.csr_matrix((np.ones(len(row_idx), dtype=np.int32), (row_idx, col_idx)),
                               shape=(n, len(kept)), dtype=np.int32)
    cuisine = recipes.cuisine.to_numpy()
    community = pd.read_csv(DATA / "network_recipe_metrics_full.csv",
                            usecols=["recipe_id", "community"])
    c76 = (pd.Series(community.community.to_numpy(), index=community.recipe_id)
           .loc[recipes.recipe_id].to_numpy() == 76)

    baseline_edges = build_edges(matrix, np.asarray(matrix.sum(axis=1)).ravel())
    original = pd.read_csv(DATA / "network_edges_full.csv")
    id_to_index = pd.Series(np.arange(n), index=recipes.recipe_id)
    reference = {(int(id_to_index[a]), int(id_to_index[b])): float(weight)
                 for a, b, weight in original.itertuples(index=False, name=None)}
    assert baseline_edges.keys() == reference.keys(), "Baseline edge set differs from notebook export"
    assert max(abs(weight - reference[pair]) for pair, weight in baseline_edges.items()) < 1e-12
    print(f"Baseline reproduced exactly: {len(baseline_edges):,} edges", flush=True)

    targets = list(TARGET_METRIC)
    controls_by_target = {}
    for target in targets:
        count = frequency[target]
        family = target.split()[0]
        options = [x for x in kept if x not in targets and family not in x
                   and .85 * count <= frequency[x] <= 1.15 * count]
        controls_by_target[target] = sorted(options, key=lambda x: (abs(frequency[x] - count), x))[:5]
        assert len(controls_by_target[target]) == 5
        print(f"{target} ({count} recipes) controls: " + ", ".join(
            f"{x} ({frequency[x]})" for x in controls_by_target[target]), flush=True)

    baseline = network_metrics(baseline_edges, cuisine, c76)
    records = [{"ingredient": "(none)", "role": "baseline", "target": "",
                "recipe_frequency": 0, **baseline}]
    treatments = list(dict.fromkeys(targets + [x for group in controls_by_target.values() for x in group]))
    for position, ingredient in enumerate(treatments, 1):
        column = ingredient_index[ingredient]
        retained_cols = np.ones(matrix.shape[1], dtype=bool)
        retained_cols[column] = False
        reduced = matrix[:, retained_cols].tocsr()
        edges = build_edges(reduced, np.asarray(reduced.sum(axis=1)).ravel())
        metrics = network_metrics(edges, cuisine, c76)
        role = "target" if ingredient in targets else "matched control"
        target = ingredient if role == "target" else next(
            t for t, controls in controls_by_target.items() if ingredient in controls)
        records.append({"ingredient": ingredient, "role": role, "target": target,
                        "recipe_frequency": frequency[ingredient], **metrics})
        print(f"{position:2}/{len(treatments)} {ingredient}: edges={metrics['edges']:,}, "
              f"within={metrics['within_cuisine_weight_share']:.4f}", flush=True)

    results = pd.DataFrame(records)
    for metric in ("within_cuisine_weight_share", "c76_incident_share_of_cross_weight",
                   "chinese_japanese_cross_share", "greek_middle_eastern_cross_share"):
        results[f"delta_{metric}_pp"] = 100 * (results[metric] - baseline[metric])
    results.to_csv(DATA / "network_ingredient_ablation_full.csv", index=False)

    comparisons = []
    for target, metric in TARGET_METRIC.items():
        target_row = results.loc[results.ingredient == target].iloc[0]
        controls = results.loc[results.ingredient.isin(controls_by_target[target])]
        col = f"delta_{metric}_pp"
        target_delta = float(target_row[col])
        control_delta = controls[col].to_numpy()
        comparisons.append({
            "target": target, "metric": metric,
            "target_recipe_frequency": frequency[target],
            "target_delta_percentage_points": target_delta,
            "control_median_delta_percentage_points": float(np.median(control_delta)),
            "control_min_delta_percentage_points": float(control_delta.min()),
            "control_max_delta_percentage_points": float(control_delta.max()),
            "target_minus_control_median_pp": target_delta - float(np.median(control_delta)),
            "matched_controls": "; ".join(controls_by_target[target]),
        })
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(DATA / "network_ingredient_ablation_comparison_full.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.3))
    titles = ["Mixed community 76", "Mixed community 76",
              "Chinese–Japanese links", "Greek–Middle Eastern links"]
    for ax, (target, metric), title in zip(axes.flat, TARGET_METRIC.items(), titles):
        values = results.loc[results.ingredient.isin(controls_by_target[target]),
                             f"delta_{metric}_pp"].to_numpy()
        point = float(results.loc[results.ingredient == target, f"delta_{metric}_pp"].iloc[0])
        ax.scatter(values, np.zeros(len(values)), color="#7799ad", s=35, label="Matched controls")
        ax.scatter([point], [0], color="#b76141", s=85, marker="D", label="Target ingredient")
        ax.axvline(0, color="#999999", lw=1)
        ax.set_yticks([])
        ax.set(xlabel="Change in cross-cuisine weight share (pp)", title=title)
        ax.text(.5, .85, f"Remove: {target}", transform=ax.transAxes,
                ha="center", va="center", fontsize=9)
        ax.grid(axis="x", alpha=.18)
    axes.flat[0].legend(frameon=False, loc="lower right", fontsize=8)
    fig.suptitle("Rebuilding the recipe network after one ingredient is removed", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "14_ingredient_ablation_full.png", dpi=200)
    fig.savefig(FIG / "14_ingredient_ablation_full.pdf")
    plt.close(fig)
    print("\n", comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()

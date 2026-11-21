"""Exploratory dish-type adjustment for the ingredient-similarity recipe network.

Run the full network notebook first; its edge-export cell writes
data/processed/network_edges_full.csv. This script keeps that fixed graph and
permutes cuisine labels globally or within conservative title-based types.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
SEED = 142
N_SHUFFLES = 999

# Order matters for titles mentioning more than one type. Patterns are
# deliberately narrow: "other/unclear" is not interpreted as a dish type.
TYPE_PATTERNS = {
    "drink": r"\b(?:cocktail|margarita|martini|mojito|sangria|smoothie|milkshake|lemonade|iced\s+(?:tea|coffee)|hot\s+chocolate|lassi|punch|mocktail|cosmopolitan|kir\s+royal|liqueur|shooter)\b",
    "dessert/sweet": r"\b(?:desserts?|cakes?|cupcakes?|cheesecakes?|cookies?|brownies?|blondies?|puddings?|custards?|ice\s+cream|sorbets?|gelato|fudge|mousse|tiramisu|baklava|panna\s+cotta|sweet\s+rolls?|parfaits?|cobblers?|fruit\s+tarts?|chocolate\s+tarts?|apple\s+pies?|pumpkin\s+pies?|pecan\s+pies?)\b",
    "soup/stew": r"\b(?:soup(?!\s+dumplings?)|stews?|chowder|bisque|gazpacho(?!\s+sauce)|pho|ramen|pozole|gumbo)\b",
    "salad": r"\b(?:salad|slaw)\b",
}


def classify_title(title: str) -> str:
    title = title.lower()
    # A few common food names contain drink words without describing a drink.
    drink_title = re.sub(r"\b(?:shrimp|prawn|ceviche)\s+cocktail\b", "", title)
    drink_title = re.sub(r"\bmargarita\s+(?:fajitas?|chicken|grill)\b", "", drink_title)
    if re.search(r"\b(?:smoothie|milkshake|lassi)\b", drink_title):
        return "drink"
    if re.search(r"\b(?:salmon|crab|fish|scallion|rice)\s+cakes?\b", title):
        dessert_title = re.sub(r"\b(?:salmon|crab|fish|scallion|rice)\s+cakes?\b", "", title)
    else:
        dessert_title = title
    if re.search(TYPE_PATTERNS["dessert/sweet"], dessert_title):
        return "dessert/sweet"
    soup_title = re.sub(r"\b(?:no\s+canned\s+soup|soup\s+(?:mix|base))\b", "", title)
    salad_title = re.sub(r"\bsalad\s+(?:dressing|sandwich)\b", "", title)
    for recipe_type, pattern in TYPE_PATTERNS.items():
        if recipe_type == "dessert/sweet":
            continue
        text = {"drink": drink_title, "soup/stew": soup_title,
                "salad": salad_title}.get(recipe_type, title)
        if re.search(pattern, text):
            return recipe_type
    return "other/unclear"


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    result = np.empty_like(p)
    result[order] = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1].clip(max=1)
    return result


def main() -> None:
    recipes = pd.read_csv(DATA / "recipes.csv", usecols=["recipe_id", "name", "cuisine"])
    edges = pd.read_csv(DATA / "network_edges_full.csv")
    assert recipes.recipe_id.is_unique
    assert {"source_recipe_id", "target_recipe_id", "weight"} <= set(edges.columns)
    id_to_index = pd.Series(np.arange(len(recipes)), index=recipes.recipe_id)
    u = id_to_index.loc[edges.source_recipe_id].to_numpy(dtype=np.int32)
    v = id_to_index.loc[edges.target_recipe_id].to_numpy(dtype=np.int32)
    w = edges.weight.to_numpy(dtype=float)
    assert len(edges) == 309019 and np.all(u < v)  # guards graph-order drift

    recipes["recipe_type"] = recipes.name.map(classify_title)
    type_order = list(TYPE_PATTERNS) + ["other/unclear"]
    types = recipes.recipe_type.to_numpy()
    cuisines = sorted(recipes.cuisine.unique())
    k = len(cuisines)
    cuisine_codes = pd.Categorical(recipes.cuisine, categories=cuisines).codes
    type_groups = [np.flatnonzero(types == t) for t in type_order]

    out = recipes[["recipe_id", "name", "cuisine", "recipe_type"]].copy()
    out.to_csv(DATA / "network_recipe_types_full.csv", index=False)

    # One-way title match rate and community 76 description are descriptive;
    # the network test below is the fixed-graph permutation comparison.
    community = pd.read_csv(DATA / "network_recipe_metrics_full.csv", usecols=["recipe_id", "community"])
    joined = out.merge(community, on="recipe_id", validate="one_to_one")
    community_ids = pd.Series(community.community.to_numpy(), index=community.recipe_id).loc[recipes.recipe_id].to_numpy()
    community_type = (joined.groupby(["community", "recipe_type"]).size()
                      .rename("recipes").reset_index())
    community_type.to_csv(DATA / "network_community_type_counts_full.csv", index=False)

    edge_same_cuisine = cuisine_codes[u] == cuisine_codes[v]
    observed = w[edge_same_cuisine].sum() / w.sum()
    same_type = types[u] == types[v]
    type_rows = []
    for t in type_order:
        node_mask = types == t
        edge_mask = same_type & (types[u] == t)
        ew = w[edge_mask].sum()
        type_rows.append({
            "recipe_type": t,
            "recipes": int(node_mask.sum()),
            "recipe_share": float(node_mask.mean()),
            "within_type_edge_weight": ew,
            "within_cuisine_share_within_type_edges": float(w[edge_mask & edge_same_cuisine].sum() / ew) if ew else np.nan,
        })
    type_table = pd.DataFrame(type_rows)
    type_table.to_csv(DATA / "network_type_summary_full.csv", index=False)

    rng = np.random.default_rng(SEED)
    global_null = np.empty(N_SHUFFLES)
    type_null = np.empty(N_SHUFFLES)
    pair_a, pair_b = np.triu_indices(k, 1)
    pair_ids = pair_a * k + pair_b
    pair_null = np.empty((N_SHUFFLES, len(pair_ids)))

    def pair_shares(labels: np.ndarray) -> np.ndarray:
        left = np.minimum(labels[u], labels[v])
        right = np.maximum(labels[u], labels[v])
        mass = np.bincount(left * k + right, weights=w, minlength=k * k)[pair_ids]
        return mass / mass.sum()

    observed_pair = pair_shares(cuisine_codes)
    for i in range(N_SHUFFLES):
        globally_shuffled = rng.permutation(cuisine_codes)
        global_null[i] = w[globally_shuffled[u] == globally_shuffled[v]].sum() / w.sum()

        within_type_shuffled = cuisine_codes.copy()
        for group in type_groups:
            within_type_shuffled[group] = rng.permutation(within_type_shuffled[group])
        type_null[i] = w[within_type_shuffled[u] == within_type_shuffled[v]].sum() / w.sum()
        pair_null[i] = pair_shares(within_type_shuffled)

    def test_row(name: str, values: np.ndarray) -> dict:
        return {
            "null": name,
            "observed_within_cuisine_weight_share": observed,
            "null_mean": float(values.mean()),
            "null_025": float(np.quantile(values, .025)),
            "null_975": float(np.quantile(values, .975)),
            "observed_minus_null_mean": float(observed - values.mean()),
            "one_sided_p": float((1 + np.count_nonzero(values >= observed)) / (N_SHUFFLES + 1)),
            "shuffles": N_SHUFFLES,
        }

    null_summary = pd.DataFrame([test_row("global cuisine shuffle", global_null),
                                 test_row("cuisine shuffle within title type", type_null)])
    null_summary.to_csv(DATA / "network_type_adjusted_null_full.csv", index=False)

    upper = (1 + np.count_nonzero(pair_null >= observed_pair, axis=0)) / (N_SHUFFLES + 1)
    lower = (1 + np.count_nonzero(pair_null <= observed_pair, axis=0)) / (N_SHUFFLES + 1)
    p_two = np.minimum(1, 2 * np.minimum(upper, lower))
    pair_table = pd.DataFrame({
        "cuisine_a": np.array(cuisines)[pair_a],
        "cuisine_b": np.array(cuisines)[pair_b],
        "observed_cross_share": observed_pair,
        "type_preserving_null_mean_cross_share": pair_null.mean(axis=0),
        "type_preserving_null_025": np.quantile(pair_null, .025, axis=0),
        "type_preserving_null_975": np.quantile(pair_null, .975, axis=0),
        "type_adjusted_lift": observed_pair / pair_null.mean(axis=0),
        "p_two_sided": p_two,
        "q_bh": bh_qvalues(p_two),
    })
    pair_table.sort_values("type_adjusted_lift", ascending=False).to_csv(
        DATA / "network_type_adjusted_pairs_full.csv", index=False)

    # Sensitivity: remove the named sweet/drink recipes, then the entire
    # ingredient-derived community 76. These subsets are descriptive checks;
    # community membership was learned from the same graph.
    subset_rows = []
    subset_pair_rows = []
    for subset_name, keep_node in [
        ("all recipes", np.ones(len(recipes), dtype=bool)),
        ("exclude named sweet/drink", ~np.isin(types, ["dessert/sweet", "drink"])),
        ("exclude community 76", community_ids != 76),
    ]:
        keep_edge = keep_node[u] & keep_node[v]
        su, sv, sw = u[keep_edge], v[keep_edge], w[keep_edge]
        labels_kept = cuisine_codes[keep_node]
        kept_indices = np.flatnonzero(keep_node)
        old_to_new = np.full(len(recipes), -1, dtype=np.int32)
        old_to_new[kept_indices] = np.arange(len(kept_indices))
        lu, lv = old_to_new[su], old_to_new[sv]
        pair_left = np.minimum(cuisine_codes[su], cuisine_codes[sv])
        pair_right = np.maximum(cuisine_codes[su], cuisine_codes[sv])
        cross_mass = np.bincount(pair_left * k + pair_right, weights=sw, minlength=k * k)[pair_ids]
        pair_share = cross_mass / cross_mass.sum()
        actual = sw[cuisine_codes[su] == cuisine_codes[sv]].sum() / sw.sum()
        shuffled_stat = np.empty(N_SHUFFLES)
        shuffled_pairs = np.empty((N_SHUFFLES, len(pair_ids)))
        for i in range(N_SHUFFLES):
            shuffled = rng.permutation(labels_kept)
            same = shuffled[lu] == shuffled[lv]
            shuffled_stat[i] = sw[same].sum() / sw.sum()
            left = np.minimum(shuffled[lu], shuffled[lv])
            right = np.maximum(shuffled[lu], shuffled[lv])
            mass = np.bincount(left * k + right, weights=sw, minlength=k * k)[pair_ids]
            shuffled_pairs[i] = mass / mass.sum()
        subset_rows.append({
            "subset": subset_name, "recipes": len(kept_indices), "edges": len(sw),
            "observed_within_cuisine_weight_share": actual,
            "global_shuffle_mean": shuffled_stat.mean(),
            "global_shuffle_025": np.quantile(shuffled_stat, .025),
            "global_shuffle_975": np.quantile(shuffled_stat, .975),
            "one_sided_p": (1 + np.count_nonzero(shuffled_stat >= actual)) / (N_SHUFFLES + 1),
        })
        for j, (a, b) in enumerate(zip(pair_a, pair_b)):
            subset_pair_rows.append({
                "subset": subset_name, "cuisine_a": cuisines[a], "cuisine_b": cuisines[b],
                "observed_cross_share": pair_share[j],
                "global_shuffle_mean_cross_share": shuffled_pairs[:, j].mean(),
                "lift": pair_share[j] / shuffled_pairs[:, j].mean(),
            })
    pd.DataFrame(subset_rows).to_csv(DATA / "network_type_sensitivity_full.csv", index=False)
    pd.DataFrame(subset_pair_rows).to_csv(DATA / "network_type_sensitivity_pairs_full.csv", index=False)

    # Portion of each observed cross-cuisine pair's weight incident to C76.
    c76_incident = (community_ids[u] == 76) | (community_ids[v] == 76)
    left = np.minimum(cuisine_codes[u], cuisine_codes[v])
    right = np.maximum(cuisine_codes[u], cuisine_codes[v])
    all_pair_mass = np.bincount(left * k + right, weights=w, minlength=k * k)[pair_ids]
    c76_pair_mass = np.bincount(left[c76_incident] * k + right[c76_incident],
                                weights=w[c76_incident], minlength=k * k)[pair_ids]
    pd.DataFrame({
        "cuisine_a": np.array(cuisines)[pair_a], "cuisine_b": np.array(cuisines)[pair_b],
        "all_cross_edge_weight": all_pair_mass,
        "c76_incident_cross_edge_weight": c76_pair_mass,
        "c76_incident_fraction": c76_pair_mass / all_pair_mass,
    }).to_csv(DATA / "network_community76_pair_contributions_full.csv", index=False)

    # Random audit sample: reviewers can assess title classification precision.
    audit = pd.concat([out[out.recipe_type == t].sample(min(25, (out.recipe_type == t).sum()),
                                                       random_state=SEED)
                       for t in type_order], ignore_index=True)
    audit.to_csv(DATA / "network_recipe_type_audit_sample.csv", index=False)

    sensitivity = pd.DataFrame(subset_rows)
    sensitivity_pairs = pd.DataFrame(subset_pair_rows)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11.2, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
    names = ["All recipes", "Without named\nsweet/drink", "Without mixed\ncommunity 76"]
    values = sensitivity.observed_within_cuisine_weight_share.to_numpy()
    positions = np.arange(3)
    ax0.barh(positions, values, color=["#446e87", "#648aa0", "#8cacc0"], height=.58)
    ax0.set_yticks(positions, labels=names)
    ax0.invert_yaxis()
    ax0.axvline(type_null.mean(), color="#b76944", ls="--", lw=1.8,
                label="Type-preserving shuffle")
    for y, value in zip(positions, values):
        ax0.text(value + .012, y, f"{value:.1%}", va="center", fontsize=9)
    ax0.set(xlim=(0, .78), xlabel="Within-cuisine share of edge weight",
            title="A. Cuisine clustering")
    ax0.legend(loc="lower right", frameon=False, fontsize=8)
    ax0.grid(axis="x", alpha=.15)

    pairs_to_show = [("chinese", "japanese"), ("greek", "middle-eastern"),
                     ("french", "italian")]
    pair_names = ["Chinese–Japanese", "Greek–Middle Eastern", "French–Italian"]
    all_lifts = []
    no_c76_lifts = []
    for a, b in pairs_to_show:
        row = sensitivity_pairs[(sensitivity_pairs.cuisine_a == a) &
                                (sensitivity_pairs.cuisine_b == b)]
        all_lifts.append(float(row.loc[row.subset == "all recipes", "lift"].iloc[0]))
        no_c76_lifts.append(float(row.loc[row.subset == "exclude community 76", "lift"].iloc[0]))
    x = np.arange(len(pair_names))
    ax1.bar(x - .18, all_lifts, width=.36, color="#446e87", label="All recipes")
    ax1.bar(x + .18, no_c76_lifts, width=.36, color="#b76944", label="Without community 76")
    ax1.set_xticks(x, labels=pair_names, rotation=18, ha="right")
    ax1.set(ylabel="Observed / shuffle-expected pair share",
            title="B. Cross-cuisine affinity remains")
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(axis="y", alpha=.15)
    covered_share = 1 - float(type_table.loc[type_table.recipe_type == "other/unclear", "recipe_share"].iloc[0])
    fig.suptitle(f"Recipe-type sensitivity: title rules identify {covered_share:.1%} of recipes", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "13_recipe_type_adjustment_full.png", dpi=200)
    fig.savefig(FIG / "13_recipe_type_adjustment_full.pdf")
    plt.close(fig)

    print(type_table.to_string(index=False))
    print("\n", null_summary.to_string(index=False))
    print("\nTop adjusted pairs:\n", pair_table.sort_values("type_adjusted_lift", ascending=False).head(8).to_string(index=False))
    print("\nCommunity 76 types:\n", community_type[community_type.community == 76].to_string(index=False))
    print("\nSensitivity:\n", pd.DataFrame(subset_rows).to_string(index=False))
    print("\nCommunity 76 cross-pair contribution:\n", pd.read_csv(DATA / "network_community76_pair_contributions_full.csv").sort_values("c76_incident_fraction", ascending=False).head(8).to_string(index=False))


if __name__ == "__main__":
    main()

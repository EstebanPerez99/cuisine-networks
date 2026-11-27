"""Plot Fig. 1: disparity-filter backbone of the ingredient co-occurrence network.

Builds the exact 26,727 x 3,877 binary recipe-ingredient matrix using
src/ingredient_preprocessing.py, projects onto ingredients (271,443 pairs),
applies the Serrano et al. (2009) disparity filter, and renders a clean
single-column PNAS figure (8.7 cm x 8.4 cm) with nodes colored by the cuisine
where each ingredient is most distinctive and sized by recipe count.
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ingredient_preprocessing import GENERIC_INGREDIENTS, normalize_ingredient

DATA = ROOT / "data/processed"
CM = 1 / 2.54
LAYOUT_K = 1.1

# Harmonious, colorblind-conscious 9-cuisine palette
CUISINE_COLORS = {
    "italian": "#c85a32",        # terracotta
    "mexican": "#d99b26",        # warm ochre
    "indian": "#a23b5c",         # deep crimson-rose
    "greek": "#2b7a9b",          # Aegean blue
    "french": "#6b5b95",         # muted plum
    "middle-eastern": "#5a8f5c", # olive sage
    "chinese": "#24476e",        # deep navy
    "thai": "#1b9e77",           # vibrant teal
    "japanese": "#e07a9a",       # soft sakura-rose
}
CUISINE_ORDER = [
    "italian", "mexican", "indian",
    "greek", "french", "middle-eastern",
    "chinese", "thai", "japanese",
]


def load_bipartite():
    df = pd.read_csv(DATA / "recipes.csv", usecols=["recipe_id", "cuisine", "ingredients"])
    rows_raw = []
    counts = Counter()
    for raw in df["ingredients"]:
        items = {normalize_ingredient(x) for x in ast.literal_eval(raw)}
        items = {x for x in items if x and x not in GENERIC_INGREDIENTS}
        rows_raw.append(items)
        counts.update(items)

    n_rec = len(df)
    max_count = int(0.35 * n_rec)
    vocab = sorted(ing for ing, c in counts.items() if 3 <= c <= max_count)
    ing_to_idx = {ing: i for i, ing in enumerate(vocab)}

    r_idx, c_idx = [], []
    for r, items in enumerate(rows_raw):
        for ing in items:
            idx = ing_to_idx.get(ing)
            if idx is not None:
                r_idx.append(r)
                c_idx.append(idx)

    X = sparse.csr_matrix(
        (np.ones(len(r_idx), dtype=np.float32), (r_idx, c_idx)),
        shape=(n_rec, len(vocab)),
    )
    return df["cuisine"].to_numpy(), vocab, X


def compute_distinctiveness(cuisines, vocab, X):
    uniq = sorted(set(cuisines))
    prev = np.zeros((len(uniq), len(vocab)), dtype=np.float64)
    for ci, c in enumerate(uniq):
        mask = cuisines == c
        prev[ci] = np.asarray(X[mask].sum(axis=0)).ravel() / mask.sum()

    # Ahn-style distinctiveness: own prevalence minus equal-weight mean of other 8
    total = prev.sum(axis=0)
    dist = np.zeros_like(prev)
    for ci in range(len(uniq)):
        other = (total - prev[ci]) / (len(uniq) - 1)
        dist[ci] = prev[ci] - other

    best_idx = np.argmax(dist, axis=0)
    best_cuisine = [uniq[i] for i in best_idx]
    best_dist = dist[best_idx, np.arange(len(vocab))]
    return best_cuisine, best_dist


def main() -> None:
    cuisines, vocab, X = load_bipartite()
    best_cuisine, best_dist = compute_distinctiveness(cuisines, vocab, X)
    rec_counts = np.asarray(X.sum(axis=0)).ravel().astype(int)

    # Co-occurrence projection
    W = (X.T @ X).tocsr()
    W.setdiag(0)
    W.eliminate_zeros()
    n_pairs = W.nnz // 2
    print(f"Bipartite: {X.shape[0]:,} recipes, {X.shape[1]:,} ingredients, {X.nnz:,} links, {n_pairs:,} pairs")

    # Serrano et al. (2009) disparity filter on the full ingredient projection
    strength = np.asarray(W.sum(axis=1)).ravel()
    degree = np.diff(W.indptr)
    W_coo = W.tocoo()
    upper = W_coo.row < W_coo.col
    u = W_coo.row[upper]
    v = W_coo.col[upper]
    w = W_coo.data[upper]

    p_uv = w / strength[u]
    p_vu = w / strength[v]
    alpha_uv = np.where(degree[u] > 1, np.power(np.clip(1.0 - p_uv, 0.0, 1.0), degree[u] - 1), 1.0)
    alpha_vu = np.where(degree[v] > 1, np.power(np.clip(1.0 - p_vu, 0.0, 1.0), degree[v] - 1), 1.0)
    alpha_min = np.minimum(alpha_uv, alpha_vu)

    # Full backbone stats at alpha < 0.01
    full_bb_edges = int((alpha_min < 0.01).sum())
    print(f"Full disparity backbone (alpha < 0.01): {full_bb_edges:,} edges ({full_bb_edges / n_pairs:.2%} of pairs)")

    # Select top distinctive ingredients per cuisine (plus top global hubs) for readable single-column display
    selected = set()
    for c in CUISINE_ORDER:
        c_idx = [i for i, bc in enumerate(best_cuisine) if bc == c and rec_counts[i] >= 90]
        c_idx.sort(key=lambda i: best_dist[i], reverse=True)
        selected.update(c_idx[:10])
    # Also ensure major backbone connectors are included
    top_freq = np.argsort(rec_counts)[::-1][:25]
    selected.update(int(i) for i in top_freq if best_dist[i] >= 0.03)

    # Build induced disparity-filtered backbone subgraph (alpha < 0.005, w >= 30)
    G = nx.Graph()
    for i in sorted(selected):
        G.add_node(i, name=vocab[i], cuisine=best_cuisine[i], count=int(rec_counts[i]), dist=float(best_dist[i]))

    for ui, vi, wi, ai in zip(u, v, w, alpha_min):
        if int(ui) in selected and int(vi) in selected and ai < 0.005 and wi >= 30:
            G.add_edge(int(ui), int(vi), weight=float(wi), alpha=float(ai))

    # Keep giant component
    giant = max(nx.connected_components(G), key=len)
    G = G.subgraph(giant).copy()
    print(f"Displayed backbone subgraph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Check for the caption: share of displayed links joining two ingredients of
    # the same colour, against 2,000 shuffles of the colours over the nodes.
    node_list = list(G.nodes())
    cz = np.array([G.nodes[n]["cuisine"] for n in node_list])
    ix = {n: k for k, n in enumerate(node_list)}
    eu = np.array([ix[a] for a, _ in G.edges()])
    ev = np.array([ix[b] for _, b in G.edges()])
    same_obs = float(np.mean(cz[eu] == cz[ev]))
    rng = np.random.default_rng(23)
    same_null = np.array([np.mean((p := rng.permutation(cz))[eu] == p[ev]) for _ in range(2000)])
    print(f"Same-cuisine share of displayed links: {same_obs:.1%}; shuffled colours mean {same_null.mean():.1%}, max {same_null.max():.1%}")

    # Layout: spring layout on link weights only. Cuisine plays no role in the
    # positions, so any grouping by colour comes from the links themselves.
    H = nx.Graph()
    for n in G.nodes():
        H.add_node(n)
    for ui, vi, d in G.edges(data=True):
        H.add_edge(ui, vi, weight=np.log1p(d["weight"]) / 5.0)

    pos = nx.spring_layout(H, k=LAYOUT_K, iterations=500, seed=23, weight="weight")

    # Normalize positions into [-1, 1] with comfortable padding
    xy = np.array([pos[n] for n in G.nodes()])
    xy -= np.median(xy, axis=0)
    # Radial stretch (monotone, keeps directions and order of distances from the
    # centre): opens up the dense core so single nodes can be told apart.
    rad = np.linalg.norm(xy, axis=1)
    rad_max = rad.max() if rad.max() > 0 else 1.0
    xy *= (np.power(rad / rad_max, 0.55) / np.where(rad > 0, rad, 1.0))[:, None]
    scale = np.max(np.abs(xy), axis=0)
    xy /= np.where(scale > 0, scale, 1.0)
    pos = {n: xy[k] for k, n in enumerate(G.nodes())}

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6.5,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(8.7 * CM, 8.6 * CM))
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.865])

    # Draw edges: intra-cuisine slightly darker than cross-cuisine
    edge_weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    w_norm = (np.log1p(edge_weights) - np.log1p(edge_weights.min())) / max(
        np.log1p(edge_weights.max()) - np.log1p(edge_weights.min()), 1e-6
    )
    for (ui, vi, d), wn in zip(G.edges(data=True), w_norm):
        x0, y0 = pos[ui]
        x1, y1 = pos[vi]
        same = G.nodes[ui]["cuisine"] == G.nodes[vi]["cuisine"]
        col = CUISINE_COLORS[G.nodes[ui]["cuisine"]] if same else "#aebec6"
        alpha = 0.30 + 0.36 * wn if same else 0.05 + 0.10 * wn
        lw = (0.35 + 1.05 * wn) if same else 0.3
        ax.plot([x0, x1], [y0, y1], color=col, alpha=alpha, linewidth=lw, zorder=1)

    # Draw nodes
    nodes = list(G.nodes())
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    counts_arr = np.array([G.nodes[n]["count"] for n in nodes], dtype=float)
    sizes = 8.0 + 80.0 * (np.sqrt(counts_arr) / np.sqrt(counts_arr.max()))
    colors = [CUISINE_COLORS[G.nodes[n]["cuisine"]] for n in nodes]
    ax.scatter(xs, ys, s=sizes, c=colors, edgecolors="white", linewidths=0.55, alpha=0.95, zorder=2)

    # Label top 2 most distinctive hubs per cuisine, with width-aware collision guard
    label_candidates = []
    for c in CUISINE_ORDER:
        cn = [n for n in nodes if G.nodes[n]["cuisine"] == c]
        cn.sort(key=lambda n: G.nodes[n]["dist"] * np.log1p(G.nodes[n]["count"]), reverse=True)
        label_candidates.extend(cn[:3])

    placed = []  # (x, y, half_width)
    halo = [pe.withStroke(linewidth=2.1, foreground="white", alpha=0.94)]
    for n in label_candidates:
        x0, y0 = pos[n]
        lbl = G.nodes[n]["name"]
        hw = 0.022 * len(lbl) + 0.04
        if any(abs(x0 - px) < (hw + phw) and abs(y0 - py) < 0.10 for px, py, phw in placed):
            continue
        placed.append((x0, y0, hw))
        ax.text(
            x0,
            y0 + 0.04,
            lbl,
            fontsize=5.6,
            fontweight="bold",
            color="#1f2933",
            ha="center",
            va="bottom",
            path_effects=halo,
            zorder=4,
        )

    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(-1.10, 1.14)
    ax.axis("off")

    # Compact 3x3 cuisine legend at bottom
    handles = [
        Line2D(
            [0], [0],
            marker="o",
            color="none",
            markerfacecolor=CUISINE_COLORS[c],
            markeredgecolor="white",
            markeredgewidth=0.4,
            markersize=4.8,
            label=c.replace("-", " ").title(),
        )
        for c in CUISINE_ORDER
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        fontsize=5.8,
        frameon=False,
        handletextpad=0.3,
        columnspacing=0.9,
        labelspacing=0.25,
    )

    for path in (
        ROOT / "figures/22_ingredient_backbone_full.pdf",
        ROOT / "figures/22_ingredient_backbone_full.png",
        ROOT / "report/figures/ingredient_backbone.pdf",
    ):
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

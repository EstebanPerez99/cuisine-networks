"""Split-half reliability of the pairing score for several pseudocounts.

Each cuisine's recipes are split at random into two halves, A and B. Every
recipe is then scored twice: once with pair counts taken only from half A,
once only from half B (the recipe itself is always left out). The expected
counts come from the same Curveball randomizations as the main score, scaled
to the size of the half. If the score measures a stable property of the
recipe, the two versions agree (high correlation); if it is driven by the
noise of small counts, they disagree. Spearman-Brown converts the half-data
correlation into the reliability expected with all the data: 2r / (1 + r).

    python scripts/pairing_score_reliability.py      # ~4 minutes
"""

from __future__ import annotations

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from recipe_ingredient_null import curveball, load_matrix  # noqa: E402

PSEUDOCOUNTS = (1, 2, 5, 10)


def to_matrix(rows, index, ncols):
    r, c = [], []
    for k, i in enumerate(index):
        r.extend([k] * len(rows[i]))
        c.extend(rows[i])
    return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(index), ncols))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replicates", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--split-seed", type=int, default=5)
    parser.add_argument("--out", default=str(ROOT / "docs" / "pairing_score_reliability.csv"))
    args = parser.parse_args()

    recipes, matrix = load_matrix()
    cuisine = recipes.cuisine.to_numpy()
    groups = {c: np.flatnonzero(cuisine == c) for c in sorted(set(cuisine))}
    base = [sorted(matrix.indices[matrix.indptr[i]:matrix.indptr[i + 1]]) for i in range(matrix.shape[0])]
    randomized = []
    for rep in range(args.replicates):
        rows = [set(r) for r in base]
        curveball(rows, list(groups.values()), 10, np.random.default_rng(args.seed + rep))
        randomized.append([sorted(r) for r in rows])

    half = np.random.default_rng(args.split_seed).integers(0, 2, len(base))
    ncols = matrix.shape[1]
    scores = {(h, k): {} for h in (0, 1) for k in PSEUDOCOUNTS}
    for c, index in groups.items():
        observed = to_matrix(base, index, ncols)
        freq = (observed.T @ observed).diagonal()
        expected = sum((to_matrix(r, index, ncols).T @ to_matrix(r, index, ncols)) for r in randomized) / len(randomized)
        expected = expected.tocsr()
        halves = {}
        for h in (0, 1):
            members = index[half[index] == h]
            m = to_matrix(base, members, ncols)
            halves[h] = ((m.T @ m).tocsr(), len(members) / (len(index) - 1))
        for i in index:
            row = base[i]
            if len(row) < 2:
                continue
            a, b = np.array(list(combinations(row, 2))).T
            e_full = np.asarray(expected[a, b]).ravel() * (freq[a] - 1) * (freq[b] - 1) / (freq[a] * freq[b])
            for h, (cooc, share) in halves.items():
                n = np.asarray(cooc[a, b]).ravel() - (1 if half[i] == h else 0)
                e = e_full * share
                for k in PSEUDOCOUNTS:
                    scores[(h, k)][i] = np.mean(np.log2((n + k) / (e + k)))

    table = []
    for k in PSEUDOCOUNTS:
        x = pd.Series(scores[(0, k)])
        y = pd.Series(scores[(1, k)]).loc[x.index]
        r = x.corr(y)
        table.append({"pseudocount": k, "recipes": len(x), "split_half_r": round(r, 3),
                      "spearman_brown_full_data": round(2 * r / (1 + r), 3)})
    table = pd.DataFrame(table)
    table.to_csv(args.out, index=False)
    print(table.to_string(index=False))
    # Paired half-A / half-B scores for plotting.
    pairs = pd.DataFrame({"recipe_id": recipes.recipe_id.to_numpy()[list(scores[(0, 1)])]})
    for k in PSEUDOCOUNTS:
        pairs[f"half_a_k{k}"] = list(scores[(0, k)].values())
        pairs[f"half_b_k{k}"] = [scores[(1, k)][i] for i in scores[(0, k)]]
    pairs.round(4).to_csv(Path(args.out).with_name("pairing_score_split_halves.csv"), index=False)


if __name__ == "__main__":
    main()

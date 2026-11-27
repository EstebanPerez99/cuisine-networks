"""Independent Week 7 semantic confirmation check using pure GTE-small embeddings
(384-dim vectors + Logistic Regression, ZERO KWIC rules).

Evaluates on train_100.csv / test_100.csv, scores all 7,281 Word2Vec-filtered
reviews, joins with network_recipe_distinctiveness_full.csv, and runs 2,000-
permutation shuffle tests against pairing_score.

Outputs:
  - data/processed/recipe_word2vec/pure_gte_recipe_scores.csv
  - data/processed/recipe_word2vec/pure_gte_shuffle_summary.csv
"""
from __future__ import annotations

from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import authenticity as au

DATA = ROOT / "data" / "processed"
W2V_DIR = DATA / "recipe_word2vec"
SHARE_DIR = DATA / "authenticity_shareable"


def main() -> None:
    filtered = pd.read_csv(W2V_DIR / "dictionary_matched_full_reviews.csv")
    recipes = pd.read_csv(DATA / "recipes.csv", usecols=["recipe_id", "name", "n_ingredients"])
    network = pd.read_csv(DATA / "network_recipe_distinctiveness_full.csv")
    train_df = pd.read_csv(SHARE_DIR / "train_100.csv")
    test_df = pd.read_csv(SHARE_DIR / "test_100.csv")

    with np.load(W2V_DIR / "gte_dictionary_review_vectors.npz") as cache:
        idx = pd.Index(cache["ids"]).get_indexer(filtered["review_id"])
        assert (idx >= 0).all()
        all_vecs = cache["vectors"][idx].astype(np.float32)

    rev_to_pos = pd.Series(np.arange(len(filtered)), index=filtered["review_id"])
    X_tr = all_vecs[rev_to_pos.loc[train_df["review_id"]].to_numpy()]
    X_te = all_vecs[rev_to_pos.loc[test_df["review_id"]].to_numpy()]
    y_tr = train_df["manual_label"].to_numpy(bool)
    y_te = test_df["manual_label"].to_numpy(bool)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=251)
    oof = np.zeros(len(y_tr))
    for tr_i, va_i in cv.split(X_tr, y_tr):
        clf_f = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X_tr[tr_i], y_tr[tr_i])
        oof[va_i] = clf_f.decision_function(X_tr[va_i])
    cutoff = au.choose_cutoff(y_tr, oof)

    clf = LogisticRegression(C=2.0, class_weight="balanced", max_iter=2000).fit(X_tr, y_tr)
    p_te = clf.decision_function(X_te) >= cutoff
    tp, fp, fn = int((y_te & p_te).sum()), int((~y_te & p_te).sum()), int((y_te & ~p_te).sum())
    pr = tp / (tp + fp) if (tp + fp) else 0.0
    rc = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0
    print(f"Pure GTE test_100: TP={tp}, FP={fp}, FN={fn} | P={pr:.3f}, R={rc:.3f}, F1={f1:.3f} (cutoff={cutoff:.3f})")

    scores = clf.decision_function(all_vecs)
    filtered["gte_score"] = scores
    filtered["gte_endorsed"] = scores >= cutoff

    rec = filtered.groupby("recipe_id").agg(
        matched_reviews=("review_id", "size"),
        gte_endorsed=("gte_endorsed", "any"),
        gte_max_score=("gte_score", "max"),
    ).reset_index()
    rec.to_csv(W2V_DIR / "pure_gte_recipe_scores.csv", index=False)

    m = (
        network.merge(rec, on="recipe_id", how="inner")
        .dropna(subset=["pairing_score"])
        .reset_index(drop=True)
    )
    m["y_gte"] = m["gte_endorsed"].astype(int)

    checks = [
        ("Pure GTE (binary) | Pairing above chance (pairing_score)", m, "y_gte", "pairing_score", ["cuisine"]),
        ("Pure GTE (continuous) | Pairing above chance (pairing_score)", m, "gte_max_score", "pairing_score", ["cuisine"]),
    ]
    summary = pd.DataFrame([{"test": lbl, **au.shuffle_test(sub, yc, xc, st, n_perm=2000, seed=42)} for lbl, sub, yc, xc, st in checks])
    summary.to_csv(W2V_DIR / "pure_gte_shuffle_summary.csv", index=False)
    print("\n=== Pure GTE Shuffle Summary ===")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

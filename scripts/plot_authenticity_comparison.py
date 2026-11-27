"""Plot the comparison between ingredient-network scores and perceived authenticity.

Generates Fig. 3 for the PNAS report (17.8 cm x 7.2 cm), matching the visual
style of Fig. 2 (scripts/plot_pairs_and_scores.py):
  Panel A: Endorsement rate across within-cuisine quartiles (Q1-Q4) of
           pairing_score (how) vs. ingredient_distinctiveness (what), with
           95% Wilson confidence intervals.
  Panel B: Per-cuisine Spearman rho with KWIC endorsements for pairing_score
           vs. ingredient_distinctiveness across all nine cuisines.
  Panel C: Stratified label-shuffle null tests (2,000 within-cuisine shuffles)
           comparing observed Spearman rho against the middle 95% of shuffles
           for all three scores, the 1-review control, and the independent
           Pure GTE embedding confirmation.
"""

from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import authenticity as au

DATA = ROOT / "data/processed"
W2V_DIR = DATA / "recipe_word2vec"
CM = 1 / 2.54
SMALL = 6.3

# Palette matching scripts/plot_pairs_and_scores.py
C_TERRA = "#bf6545"
C_SLATE = "#36566c"
C_NULL = "#aebec6"
C_MUTED = "#7b93a0"
C_GTE = "#2c6e8a"
C_GRID = "#e5e5e5"


def wilson_ci(k: np.ndarray, n: np.ndarray, z: float = 1.959964) -> tuple[np.ndarray, np.ndarray]:
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n)
    return center - half, center + half


def main() -> None:
    filtered = pd.read_csv(W2V_DIR / "dictionary_matched_full_reviews.csv")
    network = pd.read_csv(DATA / "network_recipe_distinctiveness_full.csv")
    gte_rec = pd.read_csv(W2V_DIR / "pure_gte_recipe_scores.csv")

    flags = au.kwic_flags(filtered["full_review"])
    filtered["kwic_endorsed"] = (flags[:, 0] == 1) | (flags[:, 3] == 1)

    recipe_preds = (
        filtered.groupby("recipe_id")
        .agg(
            matched_reviews=("review_id", "size"),
            kwic_endorsed=("kwic_endorsed", "any"),
        )
        .reset_index()
    )
    m = (
        network.merge(recipe_preds, on="recipe_id", how="inner")
        .dropna(subset=["pairing_score"])
        .reset_index(drop=True)
    )
    m["y_kwic"] = m["kwic_endorsed"].astype(int)
    one_rev = m[m["matched_reviews"] == 1].reset_index(drop=True)

    mg = (
        network.merge(gte_rec, on="recipe_id", how="inner")
        .dropna(subset=["pairing_score"])
        .reset_index(drop=True)
    )
    mg["y_gte"] = mg["gte_endorsed"].astype(int)

    # ---- Panel A data: within-cuisine quartiles ----
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    q_stats = {}
    for col in ("pairing_score", "ingredient_distinctiveness"):
        q = m.groupby("cuisine")[col].transform(
            lambda s: pd.qcut(s.rank(method="first"), 4, labels=q_labels)
        )
        g = m.groupby(q, observed=False)["y_kwic"].agg(["size", "sum", "mean"])
        lo, hi = wilson_ci(g["sum"].to_numpy(), g["size"].to_numpy())
        q_stats[col] = (g["mean"].to_numpy() * 100, lo * 100, hi * 100)

    # ---- Panel B data: per-cuisine Spearman rho ----
    c_rows = []
    for cuisine, g in m.groupby("cuisine"):
        r_pair, p_pair = spearmanr(g["y_kwic"], g["pairing_score"])
        r_ing, p_ing = spearmanr(g["y_kwic"], g["ingredient_distinctiveness"])
        c_rows.append(
            {
                "cuisine": cuisine.replace("-", " ").title(),
                "n": len(g),
                "r_pair": r_pair,
                "p_pair": p_pair,
                "r_ing": r_ing,
                "p_ing": p_ing,
            }
        )
    c_df = pd.DataFrame(c_rows).sort_values("r_pair").reset_index(drop=True)

    # ---- Panel C data: 2,000-permutation shuffle null tests ----
    shuffle_specs = [
        ("GTE score (pairing)", mg, "gte_max_score", "pairing_score", "gte"),
        ("GTE binary (pairing)", mg, "y_gte", "pairing_score", "gte"),
        ("Pairing (1-review only)", one_rev, "y_kwic", "pairing_score", "kwic"),
        ("Ingredient score (what)", m, "y_kwic", "ingredient_distinctiveness", "kwic"),
        ("Pair score (which pairs)", m, "y_kwic", "pair_distinctiveness", "kwic"),
        ("Pairing score (how)", m, "y_kwic", "pairing_score", "kwic"),
    ]
    shuf_rows = []
    for label, sub, yc, xc, kind in shuffle_specs:
        res = au.shuffle_test(sub, yc, xc, ["cuisine"], n_perm=2000, seed=42, return_null=True)
        lo, med, hi = np.percentile(res["null_dist"], [2.5, 50.0, 97.5])
        shuf_rows.append(
            {
                "label": label,
                "kind": kind,
                "obs": res["obs_rho"],
                "lo": lo,
                "med": med,
                "hi": hi,
                "z": res["z_score"],
                "p": res["empirical_p"],
            }
        )
    shuf_df = pd.DataFrame(shuf_rows)

    # ---- Figure layout ----
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": SMALL,
            "ytick.labelsize": SMALL,
            "pdf.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(17.8 * CM, 7.4 * CM))
    bottom, height = 0.27, 0.625
    ax1 = fig.add_axes([0.058, bottom, 0.205, height])
    ax2 = fig.add_axes([0.392, bottom, 0.208, height])
    ax3 = fig.add_axes([0.758, bottom, 0.228, height])

    # ---- Panel A: Quartile trend ----
    x = np.arange(4)
    baseline = m["y_kwic"].mean() * 100
    ax1.axhline(baseline, color=C_SLATE, linestyle=":", linewidth=0.7, zorder=0)
    ax1.text(3.38, baseline - 0.65, f"Mean {baseline:.0f}%", fontsize=5.8,
             color=C_SLATE, ha="right", va="top")

    ing_m, ing_lo, ing_hi = q_stats["ingredient_distinctiveness"]
    ax1.errorbar(
        x - 0.06,
        ing_m,
        yerr=[ing_m - ing_lo, ing_hi - ing_m],
        fmt="o--",
        color=C_MUTED,
        markersize=3.8,
        linewidth=1.1,
        elinewidth=0.9,
        capsize=2.0,
        zorder=2,
        label="Ingredient (what)",
    )

    pair_m, pair_lo, pair_hi = q_stats["pairing_score"]
    ax1.errorbar(
        x + 0.06,
        pair_m,
        yerr=[pair_m - pair_lo, pair_hi - pair_m],
        fmt="o-",
        color=C_TERRA,
        markersize=4.6,
        linewidth=1.5,
        elinewidth=1.0,
        capsize=2.0,
        zorder=3,
        label="Pairing (how)",
    )

    ax1.text(x[0] + 0.06, pair_lo[0] - 0.85, f"{pair_m[0]:.1f}%", ha="center", va="top",
             fontsize=SMALL, fontweight="bold", color=C_TERRA)
    ax1.text(x[3] + 0.06, pair_hi[3] + 0.75, f"{pair_m[3]:.1f}%", ha="center", va="bottom",
             fontsize=SMALL, fontweight="bold", color=C_TERRA)

    ax1.set_xticks(x, ["Q1\n(low)", "Q2", "Q3", "Q4\n(high)"])
    ax1.set_xlim(-0.38, 3.45)
    ax1.set_ylim(14.2, 39.2)
    ax1.set_yticks([15, 20, 25, 30, 35])
    ax1.set_ylabel("Endorsed recipes (%)")
    ax1.set_xlabel("Within-cuisine score quartile")
    ax1.grid(axis="y", color=C_GRID, linewidth=0.5)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    h1, l1 = ax1.get_legend_handles_labels()
    ax1.legend(
        h1[::-1],
        l1[::-1],
        loc="upper left",
        fontsize=SMALL,
        frameon=False,
        handlelength=1.5,
        handletextpad=0.45,
        labelspacing=0.3,
        borderaxespad=0.35,
    )

    # ---- Panel B: Per-cuisine Spearman rho ----
    yc = np.arange(len(c_df))
    ax2.axvline(0.0, color=C_SLATE, linewidth=0.6, zorder=0)
    for idx, row in c_df.iterrows():
        ax2.plot([row.r_ing, row.r_pair], [idx, idx], color="#cfd8de", linewidth=1.2, zorder=1)
        ax2.scatter(row.r_ing, idx, color=C_MUTED, s=16, zorder=2)
        if row.p_pair < 0.05:
            ax2.scatter(row.r_pair, idx, color=C_TERRA, s=28, zorder=3)
        else:
            ax2.scatter(row.r_pair, idx, facecolor="white", edgecolor=C_TERRA, linewidth=1.1, s=26, zorder=3)

    ax2.set_yticks(yc, c_df["cuisine"])
    ax2.set_xlim(-0.085, 0.285)
    ax2.set_xticks([0.0, 0.1, 0.2])
    ax2.set_xticklabels(["0", "0.1", "0.2"])
    ax2.set_ylim(-0.65, len(c_df) - 0.35)
    ax2.set_xlabel("Spearman $\\rho$ with endorsement")
    ax2.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right", "left"]].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    leg_b = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TERRA, markeredgecolor=C_TERRA,
               markersize=4.5, label="Pairing (p < 0.05)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor=C_TERRA,
               markeredgewidth=1.1, markersize=4.5, label="Pairing (n.s.)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_MUTED, markeredgecolor=C_MUTED,
               markersize=3.8, label="Ingredient"),
    ]
    fig.legend(
        handles=leg_b,
        loc="lower center",
        bbox_to_anchor=(0.455, 0.008),
        fontsize=SMALL,
        frameon=False,
        ncol=2,
        handlelength=0.9,
        handletextpad=0.4,
        columnspacing=0.9,
        labelspacing=0.25,
    )

    # ---- Panel C: Shuffle null tests ----
    # Place a slight gap between the 2 Pure GTE rows (bottom) and 4 KWIC rows (top)
    ys = np.array([0.0, 1.0, 2.35, 3.35, 4.35, 5.35])
    ax3.axvline(0.0, color=C_SLATE, linewidth=0.6, zorder=0)
    ax3.axhline(1.68, color="#d0d7dc", linestyle="--", linewidth=0.6, zorder=0)

    ax3.hlines(ys, shuf_df["lo"], shuf_df["hi"], color=C_NULL, linewidth=5.5, zorder=1)
    ax3.scatter(shuf_df["med"], ys, color=C_SLATE, s=9, zorder=2)

    x_text = 0.212
    for yi, row in zip(ys, shuf_df.itertuples()):
        if row.kind == "kwic":
            ax3.scatter(row.obs, yi, color=C_TERRA, s=28, zorder=3)
        else:
            ax3.scatter(row.obs, yi, color=C_GTE, marker="D", s=22, zorder=3)
        stars = "***" if row.p <= 0.001 else ("*" if row.p < 0.05 else "")
        ax3.text(x_text, yi, f"z = {row.z:.2f}{stars}", ha="right", va="center",
                 fontsize=SMALL, color="#333333")

    ax3.text(x_text, ys[-1] + 0.52, "Null z-score", ha="right", va="bottom",
             fontsize=SMALL, color="#333333", style="italic")

    ax3.set_yticks(ys, shuf_df["label"])
    ax3.set_xlim(-0.055, 0.215)
    ax3.set_xticks([-0.04, 0.0, 0.04, 0.08, 0.12])
    ax3.set_xticklabels(["-0.04", "0", "0.04", "0.08", "0.12"])
    ax3.set_ylim(-0.65, ys[-1] + 0.72)
    ax3.set_xlabel("Observed vs. shuffled $\\rho$")
    ax3.grid(axis="x", color=C_GRID, linewidth=0.5)
    ax3.set_axisbelow(True)
    ax3.spines[["top", "right", "left"]].set_visible(False)
    ax3.tick_params(axis="y", length=0)

    leg_c = [
        Line2D([0], [0], color=C_NULL, linewidth=4.5, label="Middle 95% of shuffles"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_SLATE, markeredgecolor=C_SLATE,
               markersize=3.2, label="Null median"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=C_TERRA, markeredgecolor=C_TERRA,
               markersize=4.5, label="KWIC"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=C_GTE, markeredgecolor=C_GTE,
               markersize=4.0, label="Pure GTE"),
    ]
    fig.legend(
        handles=leg_c,
        loc="lower right",
        bbox_to_anchor=(0.992, 0.008),
        fontsize=SMALL,
        frameon=False,
        ncol=2,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=0.9,
        labelspacing=0.25,
    )

    # Panel letters matching Fig. 2
    top = bottom + height + 0.03
    for letter, x_pos in (("A", 0.008), ("B", 0.295), ("C", 0.625)):
        fig.text(x_pos, top, letter, fontweight="bold", fontsize=9, va="bottom", ha="left")

    for path in (
        ROOT / "figures/21_authenticity_comparison_full.pdf",
        ROOT / "figures/21_authenticity_comparison_full.png",
        ROOT / "report/figures/authenticity_comparison.pdf",
    ):
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()

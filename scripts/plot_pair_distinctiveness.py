"""Plot cuisine pair signatures and the recipe-level ingredient baseline."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
scores = pd.read_csv(ROOT / "data/processed/network_recipe_distinctiveness_full.csv")
pairs = pd.read_csv(ROOT / "data/processed/network_pair_signatures_full.csv")
top = pairs.sort_values("pair_distinctiveness", ascending=False).drop_duplicates("cuisine")
top = top.sort_values("pair_distinctiveness")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.25), gridspec_kw={"width_ratios": [1.2, 1]})

labels = [f"{r.cuisine.title()}: {r.ingredient_a} + {r.ingredient_b}" for r in top.itertuples()]
ax1.barh(np.arange(len(top)), top.pair_distinctiveness * 100, color="#2f6f9f")
ax1.set_yticks(np.arange(len(top)), labels, fontsize=7)
ax1.set_xlabel("Pair prevalence difference (percentage points)")
ax1.set_title("A  Top ingredient pairs", loc="left", fontweight="bold", fontsize=8)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="x", color="#dddddd", linewidth=0.5)
ax1.set_axisbelow(True)

valid = scores.dropna(subset=["ingredient_distinctiveness", "pair_distinctiveness"])
ax2.hexbin(valid.ingredient_distinctiveness, valid.pair_distinctiveness, gridsize=45, mincnt=1, bins="log", cmap="Blues", linewidths=0)
correlation = valid[["ingredient_distinctiveness", "pair_distinctiveness"]].corr().iloc[0, 1]
ax2.text(0.04, 0.95, f"Pearson r = {correlation:.2f}\nn = {len(valid):,}", transform=ax2.transAxes, va="top", fontsize=7,
         bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9})
ax2.set_xlabel("Single-ingredient score")
ax2.set_ylabel("Ingredient-pair score")
ax2.set_title("B  Recipe scores", loc="left", fontweight="bold", fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)

fig.subplots_adjust(left=0.31, right=0.98, top=0.90, bottom=0.17, wspace=0.35)
for extension in ("pdf", "png"):
    fig.savefig(ROOT / "figures" / f"18_pair_distinctiveness_full.{extension}", dpi=220)
fig.savefig(ROOT / "report/figures/pair_distinctiveness.pdf", dpi=220)
plt.close(fig)

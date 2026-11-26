"""Plot the nine displayed pairs against their Curveball null and the recipe scores.

Panel A: observed co-occurrence of the top pair per cuisine relative to the
median of the cuisine-by-dish-type Curveball randomizations, annotated with the
pair's distinctiveness (prevalence difference, percentage points).
Panels B and C: recipe ingredient score against the pair score and against the
pairing score (pseudocount k = 5).

Needs data/processed/network_selected_pair_null_full.csv, written by
scripts/pair_cooccurrence_null.py (ignored by Git).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"
CM = 1 / 2.54
SMALL = 6.5

null = pd.read_csv(DATA / "network_selected_pair_null_full.csv")
signatures = pd.read_csv(DATA / "network_pair_signatures_full.csv")
scores = pd.read_csv(DATA / "network_recipe_distinctiveness_full.csv")

keys = ["cuisine", "ingredient_a", "ingredient_b"]
data = null.merge(signatures[keys + ["pair_distinctiveness"]], on=keys,
                  how="left", validate="one_to_one")
assert len(data) == 9 and data.pair_distinctiveness.notna().all()
data = data.sort_values("observed_to_null_median").reset_index(drop=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7,
                     "axes.labelsize": 7, "xtick.labelsize": SMALL,
                     "ytick.labelsize": SMALL, "pdf.fonttype": 42})
fig = plt.figure(figsize=(17.8 * CM, 6.8 * CM))
bottom, height = .245, .665
ax1 = fig.add_axes([.262, bottom, .205, height])
ax2 = fig.add_axes([.553, bottom, .172, height])
ax3 = fig.add_axes([.82, bottom, .172, height])

# A: observed co-occurrence relative to the randomized median
y = np.arange(len(data))
median = data.null_median_count.to_numpy()
ratios = data.observed_to_null_median.to_numpy()
ax1.hlines(y, data.null_025_count / median, data.null_975_count / median,
           color="#aebec6", linewidth=6,
           label="Middle 95% of randomized counts")
ax1.scatter(np.ones(len(data)), y, color="#36566c", s=9, zorder=3,
            label="Randomized median")
ax1.scatter(ratios, y, color="#bf6545", s=30, zorder=4, label="Observed")
ax1.axvline(1, color="#36566c", linewidth=.6, zorder=0)
labels = [f"{r.cuisine.replace('-', ' ').title()}: {r.ingredient_a} + {r.ingredient_b}"
          for r in data.itertuples()]
ax1.set_yticks(y, labels)
x_text = 2.7
for yi, value in zip(y, data.pair_distinctiveness * 100):
    ax1.text(x_text, yi, f"{value:+.1f} pp", ha="right", va="center",
             fontsize=SMALL, color="#333333")
ax1.text(x_text, len(data) - .28, "Distinctiveness", ha="right", va="bottom",
         fontsize=SMALL, color="#333333", style="italic")
ax1.set_xlabel("Observed / randomized median")
ax1.set_xlim(.77, 2.72)
ax1.set_xticks([1, 1.5, 2])
ax1.set_ylim(-.6, len(data) + .25)
ax1.grid(axis="x", color="#e5e5e5", linewidth=.5)
ax1.set_axisbelow(True)
ax1.spines[["top", "right", "left"]].set_visible(False)
ax1.tick_params(axis="y", length=0)
handles, names = ax1.get_legend_handles_labels()
fig.legend(handles, names, loc="lower left", bbox_to_anchor=(.005, -.01),
           fontsize=SMALL, frameon=False, ncol=3, handlelength=1.4,
           handletextpad=.4, columnspacing=1.0, borderaxespad=.2)


def hexbin_panel(ax, column, ylabel):
    valid = scores.dropna(subset=["ingredient_distinctiveness", column])
    ax.hexbin(valid.ingredient_distinctiveness, valid[column], gridsize=45,
              mincnt=1, bins="log", cmap="Blues", linewidths=0)
    r = valid[["ingredient_distinctiveness", column]].corr().iloc[0, 1]
    ax.text(.04, .97, f"Pearson r = {r:.2f}\nn = {len(valid):,}",
            transform=ax.transAxes, va="top", fontsize=SMALL,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9, "pad": 1.5})
    ax.set_ylabel(ylabel)
    ax.set_xticks([0, .1, .2, .3])
    ax.spines[["top", "right"]].set_visible(False)
    print(f"{column}: Pearson r = {r:.4f}, n = {len(valid):,}")


hexbin_panel(ax2, "pair_distinctiveness", "Pair distinctiveness score")
hexbin_panel(ax3, "pairing_score", "Pairing score\n(log2 observed / expected)")
fig.text((.553 + .82 + .172) / 2, bottom - .115, "Ingredient distinctiveness score",
         ha="center", va="baseline", fontsize=7)
ax3.axhline(0, color="#36566c", linewidth=.6, zorder=0)

top = bottom + height + .03
for ax, letter, x in ((ax1, "A", .005), (ax2, "B", .482), (ax3, "C", .74)):
    fig.text(x, top, letter, fontweight="bold", fontsize=9, va="bottom", ha="left")

print(data[keys + ["observed_to_null_median", "pair_distinctiveness"]]
      .assign(pair_distinctiveness=lambda d: d.pair_distinctiveness * 100)
      .round(2).to_string(index=False))
for path in (ROOT / "figures/20_pairs_and_scores_full.pdf",
             ROOT / "figures/20_pairs_and_scores_full.png",
             ROOT / "report/figures/pairs_and_scores.pdf"):
    fig.savefig(path, dpi=300)
plt.close(fig)

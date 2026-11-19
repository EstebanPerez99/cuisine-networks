"""Plot observed co-occurrence against cuisine-by-dish-type Curveball nulls."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
data = pd.read_csv(ROOT / "data/processed/network_selected_pair_null_full.csv")
data = data.sort_values("observed_to_null_median").reset_index(drop=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
fig, ax = plt.subplots(figsize=(7.0, 3.25))
y = np.arange(len(data))
null_median = data.null_median_count.to_numpy()
ax.hlines(y, data.null_025_count / null_median,
          data.null_975_count / null_median, color="#aebec6", linewidth=8,
          label="Middle 95% of randomized counts")
ax.scatter(np.ones(len(data)), y, color="#36566c", s=13, zorder=3,
           label="Randomized median")
ratios = data.observed_to_null_median.to_numpy()
ax.scatter(ratios, y, color="#bf6545", s=46, zorder=4, label="Observed")
ax.axvline(1, color="#36566c", linewidth=.6, zorder=0)
labels = [f"{r.cuisine.title()}: {r.ingredient_a} + {r.ingredient_b}"
          for r in data.itertuples()]
ax.set_yticks(y, labels, fontsize=7)
ax.set_xlabel("Co-occurrences / randomized median")
ax.set_xlim(.77, 2.07)
ax.set_ylim(-.6, len(data) - .4)
ax.grid(axis="x", color="#e5e5e5", linewidth=.5)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(.5, -.23),
          fontsize=7, frameon=False, ncol=3)
fig.subplots_adjust(left=.29, right=.98, top=.97, bottom=.25)
for path in (ROOT / "figures/19_pair_cooccurrence_null_full.pdf",
             ROOT / "figures/19_pair_cooccurrence_null_full.png",
             ROOT / "report/figures/pair_cooccurrence_null.pdf"):
    fig.savefig(path, dpi=220)
plt.close(fig)

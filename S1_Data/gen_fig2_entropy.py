import random
from paper_plot_style import *

data = rows("panel_pairs.csv")
rng = random.Random(20260914)
fig, ax = plt.subplots(figsize=(5.3, 3.7))
for i, cat in enumerate(CATEGORIES):
    vals = [float(row["vote_entropy"]) for row in data if row["category"] == cat]
    ax.scatter([i + rng.uniform(-0.19, 0.19) for _ in vals], vals,
               s=24, facecolors="white", edgecolors=COLORS[0], linewidths=0.8,
               alpha=0.8, zorder=2)
    mean = sum(vals) / len(vals)
    ax.plot([i+0.25, i+0.43], [mean, mean], color="black", linewidth=1.8, zorder=3)
    ax.text(i, 2.08, f"n={len(vals)}", ha="center", va="bottom", fontsize=9)
    zeros = sum(abs(v) < 1e-12 for v in vals)
    ax.text(i, 1.91, f"H=0: {zeros}", ha="center", va="bottom", fontsize=9)
ax.set_xticks(range(len(CATEGORIES)), [NAMES[c] for c in CATEGORIES])
ax.set_xlabel("Design category")
ax.set_ylabel("Within-pair response entropy (bits)")
ax.set_ylim(-0.07, 2.26)
ax.set_xlim(-0.55, len(CATEGORIES)-0.45)
fig.tight_layout()
save(fig, "Fig2")

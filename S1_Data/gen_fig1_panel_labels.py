from collections import Counter
from paper_plot_style import *

data = rows("panel_pairs.csv")
labels = ["L1", "L2", "L3", "L4", "L5", "L6", "TIE"]
legend = ["L1: equivalent", "L2: overlap", "L3: general–specific", "L4: same-label distinction",
          "L5: sibling", "L6: insufficient", "Unresolved"]
counts = {cat: Counter(row["majority_label"] for row in data if row["category"] == cat)
          for cat in CATEGORIES}
fig, ax = plt.subplots(figsize=(6.8, 3.8))
left = [0] * len(CATEGORIES)
for idx, label in enumerate(labels):
    values = [counts[cat][label] for cat in CATEGORIES]
    ax.barh(range(len(CATEGORIES)), values, left=left, height=0.65,
            color=COLORS[idx], edgecolor="white", linewidth=0.5, label=legend[idx],
            hatch=["", "//", "..", "xx", "\\\\", "++", "oo"][idx])
    for i, value in enumerate(values):
        if value:
            ax.text(left[i] + value / 2, i, str(value), ha="center", va="center",
                    fontsize=9, color="white" if idx in (0, 2, 5) else "black",
                    bbox={"facecolor":COLORS[idx], "edgecolor":"none", "pad":0.15})
    left = [x + y for x, y in zip(left, values)]
ax.set_yticks(range(len(CATEGORIES)), [f"{NAMES[c]} (n={sum(counts[c].values())})" for c in CATEGORIES])
ax.invert_yaxis()
ax.set_xlabel("Construct pairs")
ax.set_xlim(0, max(left) + 1)
ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.47, -0.24))
fig.subplots_adjust(bottom=0.32, left=0.18)
save(fig, "Fig1")

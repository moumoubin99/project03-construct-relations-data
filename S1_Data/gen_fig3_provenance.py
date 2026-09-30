from paper_plot_style import *

data = report()["S1_origin_counts"]["scored_per_category"]
fig, ax = plt.subplots(figsize=(6.6, 3.5))
bottom = [0] * len(CATEGORIES)
keys = ["no-backfill-record", "model-knowledge", "web-verified"]
names = ["N: No backfill record", "M: Backfilled, model-knowledge flag", "W: Backfilled, web-verified flag"]
for j, key in enumerate(keys):
    values = [data[c]["side_grounding_counts"].get(key, 0) for c in CATEGORIES]
    bars = ax.bar(range(len(CATEGORIES)), values, bottom=bottom, width=0.65,
                   color=COLORS[j], edgecolor="white", linewidth=0.5,
                   hatch=["", "//", ".."][j], label=names[j])
    for i, v in enumerate(values):
        if v >= 5:
            ax.text(i, bottom[i] + v/2, str(v), ha="center", va="center", fontsize=9,
                    color="white" if j == 0 else "black")
        elif v > 0:
            midpoint = bottom[i] + v/2
            if j == 2:
                target = (i, bottom[i] + v + 7)
                label = f"W: {v}"
            else:
                target = (i + 0.47, midpoint + 10)
                label = f"N: {v}"
            ax.annotate(label, xy=(i, midpoint), xytext=target,
                        ha="center", va="center", fontsize=9,
                        arrowprops={"arrowstyle": "-", "color": "black", "lw": 0.7},
                        bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none"})
    bottom = [b+v for b,v in zip(bottom, values)]
ax.set_xticks(range(len(CATEGORIES)), [f"{NAMES[c]}\n(n={data[c]['n_sides']})" for c in CATEGORIES])
ax.set_ylabel("Construct-side records")
ax.set_xlabel("Design category")
ax.set_ylim(0, max(bottom)*1.20)
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
fig.subplots_adjust(right=0.56, left=0.11, bottom=0.20)
save(fig, "Fig3")

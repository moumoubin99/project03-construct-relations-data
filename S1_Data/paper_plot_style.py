"""Shared PLOS-compatible numeric plotting style; no empirical values are hardcoded."""
from pathlib import Path
import csv
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
from PIL import Image

HERE = Path(__file__).resolve().parent
PACKAGED = (HERE / "reported_tables").is_dir()
ROOT = HERE if PACKAGED else HERE.parent
ANALYSIS = ROOT / "reported_tables" if PACKAGED else ROOT / "analysis"
OUT = ROOT / "figures" if PACKAGED else ROOT / "paper/figures_revised"
OUT.mkdir(parents=True, exist_ok=True)
findfont("Arial", fallback_to_default=False)
plt.rcParams.update({
    "font.family": "Arial", "font.size": 10, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "ps.fonttype": 42, "figure.dpi": 150,
    "savefig.dpi": 300, "axes.linewidth": 0.7,
})
CATEGORIES = ["A", "A_prime", "B", "C", "D"]
NAMES = {"A": "A", "A_prime": "A′", "B": "B", "C": "C", "D": "D"}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#999999"]

def rows(name):
    with (ANALYSIS / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))

def report():
    path = ROOT / "reported_reanalysis.json" if PACKAGED else ANALYSIS / "reanalysis.json"
    return json.loads(path.read_text(encoding="utf-8"))

def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(OUT / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.06,
                facecolor="white", transparent=False)
    with Image.open(OUT / f"{name}.png") as src:
        rgb = src.convert("RGB")
        rgb.save(OUT / f"{name}.tif", compression="tiff_lzw", dpi=(300, 300))
    plt.close(fig)
    print(OUT / f"{name}.tif")
